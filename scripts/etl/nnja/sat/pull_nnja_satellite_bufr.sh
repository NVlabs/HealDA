#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Mirror the NNJA microwave and infrared sounder BUFR archive from public NOAA
# S3 into a source-shaped local tree.
#
# Reads from the source-shaped NOAA route (<sensor>/<product>) but writes to a
# collapsed, sensor-keyed tree.
#
# Default destination (override with DST_DIR or --dst), which
# archive_runner.py reads as its default --raw-root:
#   $NNJA_ROOT/bufr/<sensor>/YYYY/MM/bufr/
#
# Examples:
#   bash scripts/etl/nnja/sat/pull_nnja_satellite_bufr.sh --dry-run
#   bash scripts/etl/nnja/sat/pull_nnja_satellite_bufr.sh --start-year 2018 --end-year 2025
#   bash scripts/etl/nnja/sat/pull_nnja_satellite_bufr.sh \
#     --sensors "atms iasi cris" --months "01 07"
#
# Environment overrides:
#   DST_DIR, START_YEAR, END_YEAR, MONTHS, SENSORS, NUM_WORKERS,
#   PRINT_COMMANDS, NNJA_COMPACT_LOG, PROGRESS_INTERVAL, S5CMD_BIN
# -----------------------------------------------------------------------------
set -euo pipefail

# The repo .env sets only variables the environment leaves unset, as python-dotenv does.
_healda_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
if [[ -f "${_healda_root}/.env" ]]; then
  _environment="$(export -p)"
  set -a; source "${_healda_root}/.env"; set +a
  eval "$_environment"
fi

usage() {
  cat <<'EOF'
Usage: pull_nnja_satellite_bufr.sh [options]

Options:
  --dry-run              Show the S3 operations without downloading data.
  --start-year YYYY      First requested year (default: 2000).
  --end-year YYYY        Last requested year (default: current UTC year).
  --months "MM ..."      Space- or comma-separated months (default: all).
  --sensors "NAME ..."  Space- or comma-separated sensors. Valid names:
                         amsua amsub atms mhs iasi cris airs
  --workers N            s5cmd global worker count (default: 512).
  --dst PATH             Local archive root (default: $NNJA_ROOT/bufr).
  -h, --help             Show this help.

The AIRS route is the NASA/Aqua airs_disc_final archive used by the
Earth2Studio NNJA AIRS datasource. It is not the overlapping GDAS airsev route.

The script uses s5cmd cp --if-size-differ. Re-running it resumes the mirror
and refreshes changed source objects without removing local files.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

DST_DIR="${DST_DIR:-}"
START_YEAR="${START_YEAR:-2000}"
END_YEAR="${END_YEAR:-$(date -u +%Y)}"
MONTHS="${MONTHS:-01 02 03 04 05 06 07 08 09 10 11 12}"
SENSORS="${SENSORS:-amsua amsub atms mhs iasi cris airs}"
NUM_WORKERS="${NUM_WORKERS:-512}"
PRINT_COMMANDS="${PRINT_COMMANDS:-1}"
NNJA_COMPACT_LOG="${NNJA_COMPACT_LOG:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
S5CMD_BIN="${S5CMD_BIN:-s5cmd}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --start-year)
      [[ $# -ge 2 ]] || die "--start-year requires a value"
      START_YEAR="$2"
      shift 2
      ;;
    --end-year)
      [[ $# -ge 2 ]] || die "--end-year requires a value"
      END_YEAR="$2"
      shift 2
      ;;
    --months)
      [[ $# -ge 2 ]] || die "--months requires a value"
      MONTHS="$2"
      shift 2
      ;;
    --sensors)
      [[ $# -ge 2 ]] || die "--sensors requires a value"
      SENSORS="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || die "--workers requires a value"
      NUM_WORKERS="$2"
      shift 2
      ;;
    --dst)
      [[ $# -ge 2 ]] || die "--dst requires a value"
      DST_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$START_YEAR" =~ ^[0-9]{4}$ ]] || die "invalid start year: $START_YEAR"
[[ "$END_YEAR" =~ ^[0-9]{4}$ ]] || die "invalid end year: $END_YEAR"
[[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "invalid worker count: $NUM_WORKERS"
[[ "$PROGRESS_INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "invalid progress interval: $PROGRESS_INTERVAL"
(( 10#$START_YEAR <= 10#$END_YEAR )) || die "start year exceeds end year"
if [[ -z "$DST_DIR" ]]; then
  [[ -n "${NNJA_ROOT:-}" ]] || die "set NNJA_ROOT or pass --dst"
  DST_DIR="${NNJA_ROOT}/bufr"
fi

MONTHS="${MONTHS//,/ }"
SENSORS="${SENSORS//,/ }"
read -r -a selected_months <<<"$MONTHS"
read -r -a selected_sensors <<<"$SENSORS"
[[ ${#selected_months[@]} -gt 0 ]] || die "no months selected"
[[ ${#selected_sensors[@]} -gt 0 ]] || die "no sensors selected"

for month in "${selected_months[@]}"; do
  [[ "$month" =~ ^(0[1-9]|1[0-2])$ ]] || die "invalid month: $month"
done

# Product paths match the public NNJA hierarchy, and coverage limits match the
# public bucket. Month-level clipping handles the partial first and final AIRS
# years.
product_path() {
  case "$1" in
    amsua) echo "amsua/1bamua" ;;
    amsub) echo "amsub/1bamub" ;;
    atms) echo "atms/atms" ;;
    mhs) echo "mhs/1bmhs" ;;
    iasi) echo "iasi/mtiasi" ;;
    cris) echo "cris/crisf4" ;;
    airs) echo "airs/nasa/aqua" ;;
    *) return 1 ;;
  esac
}

first_yearmonth() {
  case "$1" in
    amsua|amsub) echo "199801" ;;
    atms) echo "201202" ;;
    mhs) echo "200501" ;;
    iasi) echo "200801" ;;
    cris) echo "201801" ;;
    airs) echo "200208" ;;
    *) return 1 ;;
  esac
}

last_yearmonth() {
  case "$1" in
    amsub) echo "201406" ;;
    airs) echo "202301" ;;
    amsua|atms|mhs|iasi|cris) echo "999912" ;;
    *) return 1 ;;
  esac
}

for sensor in "${selected_sensors[@]}"; do
  product_path "$sensor" >/dev/null || die "unknown sensor: $sensor"
done

export PATH="$HOME/.local/bin:$PATH"
if [[ "$S5CMD_BIN" == */* ]]; then
  [[ -x "$S5CMD_BIN" ]] || die "s5cmd executable not found: $S5CMD_BIN"
else
  command -v "$S5CMD_BIN" >/dev/null 2>&1 || die "s5cmd not found in PATH"
fi

export S5CMD_STAT_PERIOD="${S5CMD_STAT_PERIOD:-10s}"
unset AWS_PROFILE AWS_DEFAULT_PROFILE
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"

BUCKET="noaa-reanalyses-pds"
PREFIX="observations/reanalysis"
CURRENT_YEARMONTH="$(date -u +%Y%m)"

RUNFILE="$(mktemp)"
trap 'rm -f "$RUNFILE"' EXIT

command_count=0
skipped_outside_product_range=0
for ((year = 10#$START_YEAR; year <= 10#$END_YEAR; year++)); do
  for month in "${selected_months[@]}"; do
    yearmonth="$(printf '%04d%s' "$year" "$month")"
    yearmonth_number=$((10#$yearmonth))

    # Do not emit future prefixes when END_YEAR is the current or a later year.
    if (( yearmonth_number > 10#$CURRENT_YEARMONTH )); then
      continue
    fi

    for sensor in "${selected_sensors[@]}"; do
      first_value="$(first_yearmonth "$sensor")"
      last_value="$(last_yearmonth "$sensor")"
      first_number=$((10#$first_value))
      last_number=$((10#$last_value))
      if (( yearmonth_number < first_number || yearmonth_number > last_number )); then
        ((skipped_outside_product_range += 1))
        continue
      fi

      product="$(product_path "$sensor")"
      source="s3://${BUCKET}/${PREFIX}/${product}/${year}/${month}/bufr/*"
      destination="${DST_DIR}/${sensor}/${year}/${month}/bufr/"
      if [[ "$NNJA_COMPACT_LOG" == "1" ]]; then
        printf 'cp --if-size-differ "%s" "%s"\n' "$source" "$destination" >>"$RUNFILE"
      else
        printf 'cp --sp --if-size-differ "%s" "%s"\n' "$source" "$destination" >>"$RUNFILE"
      fi
      ((command_count += 1))
    done
  done
done

echo ">> NNJA satellite BUFR mirror"
echo ">> source:      s3://${BUCKET}/${PREFIX}"
echo ">> destination: ${DST_DIR}"
echo ">> years:       ${START_YEAR}..${END_YEAR}"
echo ">> months:      ${selected_months[*]}"
echo ">> sensors:     ${selected_sensors[*]}"
echo ">> commands:    ${command_count}"
echo ">> range skips: ${skipped_outside_product_range}"

if (( command_count == 0 )); then
  echo ">> No source prefixes fall within the selected product ranges."
  exit 0
fi

if [[ "$PRINT_COMMANDS" == "1" ]]; then
  echo ">> Generated commands (${RUNFILE}):"
  cat "$RUNFILE"
  echo ">> End generated commands"
fi

if [[ "$DRY_RUN" == false ]]; then
  mkdir -p "$DST_DIR"
fi

s5cmd_args=(
  --no-sign-request
  --stat
  --numworkers "$NUM_WORKERS"
)
if [[ "$DRY_RUN" == true ]]; then
  s5cmd_args+=(--dry-run)
fi

compact_progress() {
  awk -v interval="$PROGRESS_INTERVAL" '
    BEGIN {
      start = systime()
      last = start
    }
    /^cp / {
      completed += 1
      now = systime()
      if (now - last >= interval) {
        elapsed = now - start
        rate = completed / (elapsed > 0 ? elapsed : 1)
        printf ">> progress: completed=%d elapsed=%02d:%02d:%02d avg=%.2f files/s\n",
          completed, int(elapsed / 3600), int(elapsed % 3600 / 60),
          elapsed % 60, rate
        fflush()
        last = now
      }
      next
    }
    {
      print
      fflush()
    }
    END {
      elapsed = systime() - start
      rate = completed / (elapsed > 0 ? elapsed : 1)
      printf ">> transfer summary: copied=%d elapsed=%02d:%02d:%02d avg=%.2f files/s\n",
        completed, int(elapsed / 3600), int(elapsed % 3600 / 60),
        elapsed % 60, rate
    }
  '
}

run_started="$(date +%s)"
echo ">> transfer started: $(date -u +%FT%TZ)"
if [[ "$NNJA_COMPACT_LOG" == "1" ]]; then
  set +e
  "$S5CMD_BIN" "${s5cmd_args[@]}" run "$RUNFILE" 2>&1 | compact_progress
  status=${PIPESTATUS[0]}
  set -e
elif "$S5CMD_BIN" "${s5cmd_args[@]}" run "$RUNFILE"; then
  status=0
else
  status=$?
fi

elapsed=$(( $(date +%s) - run_started ))
if (( status == 0 )); then
  printf '>> NNJA pull completed: sensors=%s elapsed=%02d:%02d:%02d end=%s\n' \
    "${selected_sensors[*]}" \
    $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 )) $(( elapsed % 60 )) \
    "$(date -u +%FT%TZ)"
else
  printf '>> NNJA pull failed: sensors=%s status=%d elapsed=%02d:%02d:%02d end=%s\n' \
    "${selected_sensors[*]}" "$status" \
    $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 )) $(( elapsed % 60 )) \
    "$(date -u +%FT%TZ)" >&2
  exit "$status"
fi
