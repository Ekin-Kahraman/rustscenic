#!/bin/bash
# Release-candidate timing: v0.7.0 RC (exact bins, optimised split search) against the released
# v0.6.1 255-bin kernel, current rule, seed 777, sequential on an idle 10-core laptop.
set -u
D=$(cd "$(dirname "$0")" && pwd)
PKG=$D/../rustscenic-stopping-rule
FULL=$D/atlas-rebuilt-full
LOG=$D/final-timings.tsv
RC=${PY_RC070:?Python of a local v0.7.0 release-candidate build; path redacted for publication}
U8=${PY_RELEASE_U8:?Python of a local v0.6.1 plus paired_oob build; path redacted for publication}
[ -f "$LOG" ] || printf 'name\tmax_rss_bytes\n' > "$LOG"
run() {  # python outname data [prepared]
  local out=$D/native-bins/$2 log=$D/native-bins/$2.log
  if [ -f "$out/info.json" ]; then echo "skip $2"; return; fi
  /usr/bin/time -l "$1" "$D/fit_native_bins.py" "$PKG" "$out" 777 arboreto "$3" ${4:-} > "$log" 2>&1
  printf '%s\t%s\n' "$2" "$(awk '/maximum resident set size/ {print $1}' "$log")" >> "$LOG"
  head -1 "$log"
}
run "$RC" rc070-bmmc-arboreto-777-a bmmc
run "$U8" release-u8-bmmc-arboreto-777-b bmmc
run "$RC" rc070-bmmc-arboreto-777-b bmmc
run "$U8" release-u8-bmmc-arboreto-777-c bmmc
run "$RC" rc070-atlas1306127-arboreto-777 atlas1306127 "$FULL"
echo TIMING DONE
