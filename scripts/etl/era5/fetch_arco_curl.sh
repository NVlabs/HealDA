#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Fetch an ARCO key list (make_arco_manifest.py output) over HTTPS with parallel curl.
#   ARCO_LOCAL=<dir> fetch_arco_curl.sh LIST [PARALLEL]
# Keys already present are skipped; each key is written to a .part file, then renamed.
set -euo pipefail
LIST=$1
PARALLEL=${2:-48}
export ARCO_LOCAL=${ARCO_LOCAL:?set ARCO_LOCAL to the local copy of the ARCO Zarr}
export ARCO_URL=${ARCO_URL:-https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3}
fetch_one() {
  local key=$1
  [ -f "$ARCO_LOCAL/$key" ] && return 0
  mkdir -p "$ARCO_LOCAL/$(dirname "$key")"
  local part="$ARCO_LOCAL/$key.part.$$"
  if curl -sf --retry 5 --retry-delay 2 -o "$part" "$ARCO_URL/$key"; then
    mv -f "$part" "$ARCO_LOCAL/$key"
  else
    rm -f "$part"
    echo "FAIL $key"
  fi
}
export -f fetch_one
xargs -a "$LIST" -P "$PARALLEL" -I{} bash -c 'fetch_one "$@"' _ {}
