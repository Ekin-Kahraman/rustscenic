# Final scoring-only confirmation handoff

Prepared 8 November 2026 for Huangfu. Returned and audited 9 November 2026.

All tasks completed. See [verified return](../../docs/paper/evidence/directional/AUDIT.txt)
for the confirmed F1 improvement, recall tradeoff and small atlas scoring cost.
The protocol below records what was sent; no further run is pending.

Package: `RustScenic-Directional-Confirmation-Huangfu-20261108.zip`

SHA-256: `1f0962563b0da372a15e66a6a70282cd1a8b03bb01350efe06342ab2aa2d40cd`

Local artifacts: `/Users/ekin/Desktop/RustScenic/RustScenic-Directional-Confirmation-20261108`.
The ZIP contains a pinned Linux wheel, current source snapshot, protocol,
small frozen networks and external RNA, numerical controls, launcher, collector
and tests. It reuses the recovery environment and cached HDMA RNA on HPC.

## Minimum remaining execution

- External confirmation: both methods' six frozen BMMC networks, 12,315 Kang
  cells and all eight donors. Calculate each directional activity matrix once,
  derive the contrast and evaluate both with the frozen donor-aware classifier.
  Retain all 192 folds, per-class confusions and coverage. Compare numerical
  values with existing controls; do not require exact cross-platform F1.
- Atlas cost: three frozen networks, union and directional modes, six fresh
  scoring processes on the same 817,740 RNA cells and node. Report scoring and
  I/O-inclusive times, process-tree PSS, output features and storage. Compare
  derived contrasts with previously confirmed atlas sample scores.
- No network fitting, ATAC imputation, motif processing, BMMC development
  reruns or legacy-engine comparisons. Existing paired gene/region replay
  covers those unchanged paths; this is gene-activity scoring confirmation.

## Run and return

Extract beside `rustscenic-recovery`, enter `rustscenic-directional-confirmation`
and run from the login node:

```bash
bash start.sh acc_DiseaseGeneCell
```

Return `work/SEND_BACK.tar.gz` and `work/SEND_BACK.sha256`, including failures.
Full activity arrays remain on HPC with hashes recorded in the return; preserve
them until archived. A collector runs after success or failure. Exact completed
tasks resume; a host change repeats only the six short timing arms to preserve
matched comparisons, keeping earlier diagnostics.

## Audit

Independent Codex implementation audit passed. The Linux tests cover scoring,
memory-bounded batches, resource requests, numerical integrity, resumption,
node changes, failure collection and duplicate submission. The exact shell
entry point was exercised against a fake scheduler, never the real HPC.
The archive's 71 manifested files match the reviewed source and current Python
wheel contents. Full real-data Linux QA receipts are stored alongside the ZIP.
All 16 Linux tests and 12 real-data numerical comparisons passed with zero
score difference. Local QA consolidated 192 donor-fold results across two
attempts: 176 completed fits reused and 16 fitted after an emulator timeout.
All scores were recomputed. Six scoring passes on a 500-cell real atlas sample,
collection, return-file hashes and resume checks passed. These emulator runs
are compatibility checks, not atlas performance measurements. The timeout,
QA-only resume and independent review records are retained beside the ZIP.

Treat a successful return as independent execution on already-inspected cohorts,
not fresh biological validation or a new full-workflow speed benchmark. The
collector reports execution, numerical agreement, convergence and memory
completeness separately. All favourable and adverse endpoints are retained.
