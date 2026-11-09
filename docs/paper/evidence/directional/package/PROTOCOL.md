# Directional-activity confirmation, 8 November 2026

Confirm the implementation and costs of one already selected representation.
No inference, threshold optimisation, new data selection or biological claim
selection is permitted within this confirmation.

External biology: frozen BMMC networks trained on 5,000 and 20,000 cells,
seeds 777/778/779, both RustScenic and SCENIC+. Score existing target sets on
the same 12,315 Kang cells using the corrected native scorer, top fraction
0.05, seed 20261106 and original >=80% gene coverage. Keep the same common
TF roster, two directions per TF, zero-fill missing directions. Score once;
derive the original positive-minus-negative contrast from those same values.
The frozen utility script fits standardisation and balanced logistic regression
C=1 within each donor-training fold. Report all 192 donor folds, convergence,
confusions, effective feature counts and paired donor-bootstrap intervals.
No new BMMC development or holdout evaluation is needed.

The included expected arrays are numerical controls only. They do not select
models or labels. Verify complete ordered cell/feature axes and numerical
agreement within 1e-7. Recompute both classification endpoints on HPC; slight
cross-platform prediction differences are allowed. Biological inferiority,
superiority or equivalence is not defined by an after-the-fact cutoff.

Atlas cost: use the exact cached 817,740-cell HDMA RNA matrix and three frozen
native networks. On one 16-CPU allocation run seeded TF-union and directional
activity in fresh processes, order U/D, D/U, U/D. Both use float32 output and
4096-row batches. Union scores come from union target sets, not summed AUCs.
Report per-pair and median scoring time, read/write-inclusive stage time,
whole-process and analysis-stage sampled process-tree PSS, output columns and
bytes. A 10% cost increase triggers review, not an automatic scientific failure.
Preserve every pair. No comparison across different nodes is summarised.

Check signed sets and frozen network hashes, finite outputs and exact axes.
Derive contrast on the same 10,000 sampled atlas cells and compare against
the existing corrected HPC scores, avoiding another reference/contrast pass.
Local real paired gene/region replay already covers the public workflow option;
these atlas timings concern gene activity on RNA only.

Resume only exact package/input provenance and completed checksummed tasks.
Stop expensive work on numerical-integrity failures; collect logs on success
or failure. A successful run confirms reproducibility of the representation
fix on reused cohorts, not untouched biological validation or new full-workflow
speedups. Earlier endpoints and benchmark claims remain unchanged.
