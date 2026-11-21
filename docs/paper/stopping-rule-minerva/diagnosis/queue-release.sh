#!/bin/bash
# Post hoc: released 8-bit kernel timings for comparison with queue.sh.
set -u
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY_RELEASE_U8:?Python of a local v0.6.1 plus paired_oob build; path redacted for publication}  # released 8-bit kernel (v0.6.1 + paired_oob)
PKG=$D/../rustscenic-stopping-rule
ATLAS=$D/atlas-rebuilt
LOG=$D/queue-fits.tsv
[ -f "$LOG" ] || printf 'name\tmax_rss_bytes\n' > "$LOG"

fit() {  # cap mode seed data outname
  local out=$D/native-bins/$5
  if [ -f "$out/info.json" ]; then echo "skip $5"; return; fi
  local log=$D/native-bins/$5.log
  RUSTSCENIC_MAX_BINS=$1 /usr/bin/time -l "$PY" "$D/fit_native_bins.py" "$PKG" "$out" "$3" "$2" "$4" "$ATLAS" > "$log" 2>&1
  printf '%s\t%s\n' "$5" "$(awk '/maximum resident set size/ {print $1}' "$log")" >> "$LOG"
  head -1 "$log"
}

# Released 8-bit kernel against the exact-bin prototype, current rule, seed 777 (post hoc cost comparison).
fit 255 arboreto 777 bmmc release-u8-bmmc-arboreto-777
fit 255 arboreto 777 atlas100000 release-u8-atlas100000-arboreto-777-a
fit 255 arboreto 777 atlas100000 release-u8-atlas100000-arboreto-777-b
fit 255 arboreto 777 atlas500000 release-u8-atlas500000-arboreto-777
echo QUEUE DONE
