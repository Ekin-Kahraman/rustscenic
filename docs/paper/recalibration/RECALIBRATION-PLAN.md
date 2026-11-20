# Frozen plan for post-hoc evaluation repair

12 November 2026. Written before SCEPTRE calibration or corrected method comparisons. The original Replogle outcomes have already been inspected; this is a post-hoc re-analysis, not a fresh confirmatory experiment. No production changes, model fitting, sending or publishing are authorised by this plan.

## Scope and separation

Use the exact held-out evaluation cells and 5,000-gene panels from the sent package. Recover integral raw counts from the pinned source H5AD files; verify barcodes, original library totals and aggregate counts against the frozen staged inputs. Preserve the existing training/evaluation split and all saved networks for three seeds. The original 156 K562 and 117 RPE1 evaluable TFs form the primary comparison roster for continuity; retain all other targeted TFs in coverage and report supplementary results where estimable. Never discard missing network outputs.

## Replacement statistical analysis

Pin official SCEPTRE source commit 20c33349c2c5a0df66589e37f67238e67c53bcb0 (version 0.99.1) in an isolated local R library. Use its low-MOI, two-sided, negative-control-cell comparison with permutation resampling and its default skew-normal approximation. Adjust for log full-source library size and capture batch. Encoded guide-pair constructs are the existing assigned perturbation units, not newly measured guide UMI counts: import a binary assignment matrix and use maximum assignment to preserve exactly one recorded construct per cell. Combine constructs targeting the same gene using union. Use original prefiltered cells without a new outcome-dependent cell filter; require at least seven nonzero treated and seven nonzero control cells per target-response pair. No inference is made for failed pairwise QC.

Calibrate before discovery. Each cohort gets two independent RNG runs of 10,000 negative-control pairs at a control-construct group size approximating the median targeted-regulator cell count divided by the median non-targeting-construct cell count. Additional 3,000-pair checks use one construct and a group size approximating the 90th-percentile targeted-regulator cell count. Clamp group sizes to 1..floor(number of control constructs/2). Record actual sample-size distributions. Primary seeds: 2026111201 and 2026111202; auxiliary seeds: 2026111203 and 2026111204.

Operational calibration gate, stated before results: among finite QC-passing negative pairs, fractions with p <= .05/.01/.001 must not exceed .065/.015/.0025, respectively; no more than 10 BH(.05) discoveries per run; absolute median estimated log2 fold-change <= .1. Require at least 1,000 finite QC-passing pairs per main check. Check gross over-conservatism separately using the p-value distribution and the target's own-gene knockdown positive controls. These empirical diagnostics are not a proof of FDR control or independent biological replication. If a check fails, preserve it and diagnose; do not inspect corrected network comparisons until a justified replacement has passed controls. Any changed specification is a documented amendment, not silent tuning.

After calibration, test all available targeted-TF/panel-gene pairs excluding self. Use BH correction across the tested pairs within each cohort, nominal q=.05, plus absolute estimated log2 fold-change >= .5 for primary response labels; retain .25 and 1.0 effect sensitivities. Self-target pairs form a separate power check, never ranking positives. Report both significance-only and effect-filtered counts. Untested/failed/nonfinite pairs are unavailable, not true negatives. All non-significant tested genes remain benchmark nonresponders, acknowledging incomplete power and indirect responses. Do not call these labels direct causal-edge truth. A corrected pooling-before-log effect calculation is retained only as a separate mechanism/regression diagnostic.

## Saved-network scoring

Use unchanged original and experimental native regulons, full reference, raw importance, full Pearson and uniform baselines. Equal-TF macro AP, averaging seeds within TF; report all-TF and originally eligible-TF rosters separately, evaluable gene counts, target recovery, emitted top-50 precision/counts, matched-budget comparisons and motif availability. Conditional 5,000-resample TF-bootstrap intervals (seed 20261112) do not establish donor-level equivalence. Gene-universe differences from the old heuristic must be disclosed. No new recipe selection or promotion on this inspected cohort.

Reproduce the separate saved activity diagnostic from group means, retaining its low support and descriptive scope. Its computations do not depend on the invalid gene-response labels; do not replace it with a favourable activity endpoint. Confirm original prediction hashes again after analysis.

## Verification and claims

Add regression tests demonstrating the small-pool log bias, the corrected pooling order, BH/QC/missingness handling, ranking-tie behaviour and source/axis integrity. Calibrate with real negative-control constructs; check own-target knockdown for power; reproduce final summary tables from per-pair/per-TF outputs. Preserve all failures, amendments and version/hash receipts. Scientific figures are generated from saved tables.

The publication decision must separate measured computational performance, descriptive biological utility, post-hoc perturbation-response recovery, and unproven causal accuracy or superiority. Preserve prior verified scaling evidence with its exact software/version and feature-panel scope. Do not promise journal acceptance. No new large HPC scaling run is part of this repair.

Primary method documentation: https://katsevich-lab.github.io/sceptre/ and Barry et al., Genome Biology (2024), DOI 10.1186/s13059-024-03254-2.
