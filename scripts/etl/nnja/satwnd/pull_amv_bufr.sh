#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Mirror satellite-wind (AMV) BUFR from public NOAA S3 into a local tree, one
# stream at a time.
#
# Source: s3://noaa-reanalyses-pds (public, --no-sign-request), under
#   observations/reanalysis/amv/{satwnd,merged}/<YYYY>/<MM>/bufr/
#   gdas.<YYYYMMDD>.t<HH>z.satwnd.tm00.bufr_d      (4 cycles/day: 00 06 12 18)
#
# Two streams with different coverage:
#   satwnd  NOAA/NCEP operational dump, continuing to the present.
#   merged  EUMETSAT AMV Climate Data Record, Release 2,
#           1979-2019 only; later years are clipped, not an error.
#
# Attribution: merged/ is licensed CC BY 4.0, DOI 10.15770/EUM_SEC_CLM_0020.
# Downstream use must carry the DOI and licence; the archive footer does so
# via DATA_ATTRIBUTION in satwnd/amv_writer.py.
#
# Idempotent: s5cmd cp --if-size-differ skips objects already present at the
# correct size. After each year, local .bufr_d files are counted against a live
# `s5cmd ls` of the bucket.
#
# Examples:
#   bash scripts/etl/nnja/satwnd/pull_amv_bufr.sh \
#     --stream merged --start-year 2000 --end-year 2019 --dry-run
#   bash scripts/etl/nnja/satwnd/pull_amv_bufr.sh \
#     --stream merged --start-year 2016 --end-year 2019
#   bash scripts/etl/nnja/satwnd/pull_amv_bufr.sh \
#     --stream satwnd --start-year 2020 --end-year 2025 --dst /some/writable/root
#
# Environment overrides:
#   DST_DIR, START_YEAR, END_YEAR, NUM_WORKERS, S5CMD_BIN, PRINT_COMMANDS
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
Usage: pull_amv_bufr.sh --stream satwnd|merged [options]

Required:
  --stream NAME           satwnd (NOAA operational) or merged (EUMETSAT AMV
                           Climate Data Record, Release 2; CC BY 4.0 -- see
                           the attribution note at the top of this script
                           before redistributing it).

Options:
  --dry-run                Show the S3 operations without downloading data.
  --start-year YYYY        First requested year (default: 2000).
  --end-year YYYY          Last requested year (default: current UTC year).
  --workers N               s5cmd global worker count (default: 512).
  --dst PATH                Local archive root (default: $NNJA_ROOT/bufr/<stream>,
                             where run_production_etl.py reads it).
  --skip-verify             Skip the post-year bucket-count check (faster,
                             use only for a quick exploratory pull).
  -h, --help                 Show this help.

The script uses s5cmd cp --if-size-differ (not sync --delete): re-running it
resumes/extends the mirror and refreshes changed source objects, never removes
local files.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

STREAM=""
START_YEAR="${START_YEAR:-2000}"
END_YEAR="${END_YEAR:-$(date -u +%Y)}"
NUM_WORKERS="${NUM_WORKERS:-512}"
DST_DIR="${DST_DIR:-}"
PRINT_COMMANDS="${PRINT_COMMANDS:-1}"
S5CMD_BIN="${S5CMD_BIN:-s5cmd}"
DRY_RUN=false
SKIP_VERIFY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stream)        [[ $# -ge 2 ]] || die "--stream requires a value"; STREAM="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=true; shift ;;
    --start-year)     [[ $# -ge 2 ]] || die "--start-year requires a value"; START_YEAR="$2"; shift 2 ;;
    --end-year)       [[ $# -ge 2 ]] || die "--end-year requires a value"; END_YEAR="$2"; shift 2 ;;
    --workers)        [[ $# -ge 2 ]] || die "--workers requires a value"; NUM_WORKERS="$2"; shift 2 ;;
    --dst)            [[ $# -ge 2 ]] || die "--dst requires a value"; DST_DIR="$2"; shift 2 ;;
    --skip-verify)    SKIP_VERIFY=true; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)                die "unknown argument: $1" ;;
  esac
done

[[ "$STREAM" == "satwnd" || "$STREAM" == "merged" ]] || die "--stream must be 'satwnd' or 'merged' (got: '$STREAM')"
[[ "$START_YEAR" =~ ^[0-9]{4}$ ]] || die "invalid start year: $START_YEAR"
[[ "$END_YEAR" =~ ^[0-9]{4}$ ]] || die "invalid end year: $END_YEAR"
[[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "invalid worker count: $NUM_WORKERS"
(( 10#$START_YEAR <= 10#$END_YEAR )) || die "start year exceeds end year"

# merged has no coverage after 2019; clip rather than error.
MERGED_LAST_YEAR=2019
if [[ "$STREAM" == "merged" && $(( 10#$START_YEAR )) -gt "$MERGED_LAST_YEAR" ]]; then
  echo ">> merged/ has no coverage after ${MERGED_LAST_YEAR}; requested range" \
       "${START_YEAR}-${END_YEAR} is entirely outside it. Nothing to do."
  exit 0
fi
EFFECTIVE_END_YEAR="$END_YEAR"
if [[ "$STREAM" == "merged" && $(( 10#$END_YEAR )) -gt "$MERGED_LAST_YEAR" ]]; then
  EFFECTIVE_END_YEAR="$MERGED_LAST_YEAR"
  echo ">> merged/ has no coverage after ${MERGED_LAST_YEAR}; clipping end year" \
       "${END_YEAR} -> ${EFFECTIVE_END_YEAR}"
fi

if [[ -z "$DST_DIR" ]]; then
  [[ -n "${NNJA_ROOT:-}" ]] || die "set NNJA_ROOT or pass --dst"
  DST_DIR="${NNJA_ROOT}/bufr/${STREAM}"
fi

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
PREFIX="observations/reanalysis/amv/${STREAM}"

echo ">> AMV BUFR pull: stream=${STREAM}"
echo ">> source:      s3://${BUCKET}/${PREFIX}"
echo ">> destination: ${DST_DIR}"
echo ">> years:       ${START_YEAR}..${EFFECTIVE_END_YEAR}"
if [[ "$STREAM" == "merged" ]]; then
  echo ">> LICENCE: merged/ is CC BY 4.0, DOI 10.15770/EUM_SEC_CLM_0020 (EUMETSAT" \
       "AMV Climate Data Record, Release 2). Attribution required on downstream" \
       "use, not just download -- see the header comment of this script."
fi

if [[ "$DRY_RUN" == false ]]; then
  mkdir -p "$DST_DIR"
fi

s5cmd_args=(--no-sign-request --stat --numworkers "$NUM_WORKERS")
[[ "$DRY_RUN" == true ]] && s5cmd_args+=(--dry-run)

overall_status=0
for ((year = 10#$START_YEAR; year <= 10#$EFFECTIVE_END_YEAR; year++)); do
  echo ">> ${year} start $(date -u +%FT%TZ)"
  source="s3://${BUCKET}/${PREFIX}/${year}/*"
  destination="${DST_DIR}/${year}/"

  if [[ "$PRINT_COMMANDS" == "1" ]]; then
    echo ">> cp --if-size-differ \"${source}\" \"${destination}\""
  fi

  year_status=0
  "$S5CMD_BIN" "${s5cmd_args[@]}" cp --if-size-differ "$source" "$destination" || year_status=$?
  if (( year_status != 0 )); then
    echo ">> ${year} S5CMD FAILED status=${year_status}" >&2
    overall_status=1
    continue
  fi

  if [[ "$DRY_RUN" == true ]]; then
    echo ">> ${year} dry-run: skipping verification"
    continue
  fi

  local_count=$(find "${DST_DIR}/${year}" -name '*.bufr_d' 2>/dev/null | wc -l)
  local_bytes=$(du -sb "${DST_DIR}/${year}" 2>/dev/null | cut -f1)

  if [[ "$SKIP_VERIFY" == true ]]; then
    echo ">> ${year} done files=${local_count} bytes=${local_bytes} (verification skipped)"
    continue
  fi

  # Live bucket count; the glob recurses through MM/bufr/ and expands the same
  # way for `ls` and `cp`.
  bucket_count=$("$S5CMD_BIN" --no-sign-request ls "s3://${BUCKET}/${PREFIX}/${year}/*" 2>/dev/null \
    | grep -c '\.bufr_d$' || true)

  status="OK"
  if [[ "$local_count" != "$bucket_count" ]]; then
    status="COUNT MISMATCH (local=${local_count} bucket=${bucket_count})"
    overall_status=1
  fi
  echo ">> ${year} done files=${local_count}/${bucket_count} bytes=${local_bytes} ${status} $(date -u +%FT%TZ)"
done

if (( overall_status == 0 )); then
  echo ">> AMV pull completed: stream=${STREAM} end=$(date -u +%FT%TZ)"
else
  echo ">> AMV pull finished WITH ERRORS: stream=${STREAM} end=$(date -u +%FT%TZ)" >&2
fi
exit "$overall_status"
