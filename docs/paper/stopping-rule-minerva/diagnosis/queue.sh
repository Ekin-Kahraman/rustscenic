#!/bin/bash
# Post hoc queue, run sequentially so timings are comparable: atlas fidelity with exact bins,
# then cost of exact bins at scale, then smaller bin budgets on bone marrow.
# Each fit: RUSTSCENIC_MAX_BINS=<cap> fit_native_bins.py; peak RSS from /usr/bin/time -l.
set -u
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY_EXACT_BINS:?Python of a local exp/finer-bins build; path redacted for publication}
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

# 1. Atlas 20,000: confirm the rebuild (255 bins must reproduce Minerva bit for bit), then exact bins.
fit 255 arboreto 777 atlas20000 atlas20000-cap255-arboreto-777
for s in 777 778 779; do fit 65535 arboreto $s atlas20000 atlas20000-cap65535-arboreto-$s; done
for s in 777 778 779; do fit 65535 paired_oob $s atlas20000 atlas20000-cap65535-paired_oob-$s; done
# Calibration: local 255-bin fits on the rebuilt input (it matches Minerva's Pearson values to 3e-8 but
# not bit for bit, so exact bins are compared with local 255-bin fits as well as with the returned ones).
for s in 778 779; do fit 255 arboreto $s atlas20000 atlas20000-cap255-arboreto-$s; done
for s in 777 778 779; do fit 255 paired_oob $s atlas20000 atlas20000-cap255-paired_oob-$s; done
# 2. Cost at scale, current rule, seed 777 (A B B A at 100,000; A B at 500,000).
fit 255 arboreto 777 atlas100000 atlas100000-cap255-arboreto-777-a
fit 65535 arboreto 777 atlas100000 atlas100000-cap65535-arboreto-777-a
fit 65535 arboreto 777 atlas100000 atlas100000-cap65535-arboreto-777-b
fit 255 arboreto 777 atlas100000 atlas100000-cap255-arboreto-777-b
fit 255 arboreto 777 atlas500000 atlas500000-cap255-arboreto-777
fit 65535 arboreto 777 atlas500000 atlas500000-cap65535-arboreto-777
# 3. Bone marrow: smaller budgets (current rule, three seeds), then an idle-machine cost pair.
for cap in 1023 4095; do for s in 777 778 779; do fit $cap arboreto $s bmmc cap$cap-arboreto-$s; done; done
fit 255 arboreto 777 bmmc bmmc-cost-cap255-arboreto-777
fit 65535 arboreto 777 bmmc bmmc-cost-cap65535-arboreto-777
echo QUEUE DONE
