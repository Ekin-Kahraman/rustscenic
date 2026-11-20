#!/bin/bash
# Post hoc: full 1,306,127-cell atlas, current rule, seed 777. Released 8-bit kernel (v0.6.1 + paired_oob)
# against the finished exact-bin build (exp/finer-bins 095554b, default max_bins). Sequential, same machine.
set -u
D=$(cd "$(dirname "$0")" && pwd)
PKG=$D/../rustscenic-stopping-rule
FULL=$D/atlas-rebuilt-full
LOG=$D/queue-fits.tsv
if [ ! -f "$FULL/rna.h5" ]; then
  "${PY_EXACT_BINS:?Python of a local exp/finer-bins build; path redacted for publication}" -I "$D/rebuild_atlas.py" \
    "${FINAL_SCALING_PACKAGE:?final-scaling package directory; path redacted for publication}" \
    "${ATLAS_SOURCE_H5:?10x 1.3M mouse brain source h5; path redacted for publication}" "$FULL" 1306127 || exit 1
fi
run() {  # python outname
  local out=$D/native-bins/$2 log=$D/native-bins/$2.log
  if [ -f "$out/info.json" ]; then echo "skip $2"; return; fi
  /usr/bin/time -l "$1" "$D/fit_native_bins.py" "$PKG" "$out" 777 arboreto atlas1306127 "$FULL" > "$log" 2>&1
  printf '%s\t%s\n' "$2" "$(awk '/maximum resident set size/ {print $1}' "$log")" >> "$LOG"
  head -1 "$log"
}
run "${PY_RELEASE_U8:?Python of a local v0.6.1 plus paired_oob build; path redacted for publication}" release-u8-atlas1306127-arboreto-777
run "${PY_EXACT_BINS:?Python of a local exp/finer-bins build; path redacted for publication}" exact-atlas1306127-arboreto-777
echo QUEUE DONE
