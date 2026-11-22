# RustScenic paper evidence

Paper evidence directory, updated 22 November 2026.

**Positioning:** a scalable Rust implementation of core SCENIC-style analysis
stages, with measured workflow gains and an explicit biological trade-off.

- [Final Minerva audit](minerva-final/README.md): public-release scaling,
  independently frozen K562 perturbation results, controls and local replay.
- [Corrected perturbation evaluation](recalibration/REPORT.md): calibrated
  Replogle K562/RPE1 response tests, unchanged saved networks, full coverage,
  independent verification and the resulting publication claim limits. This
  supersedes the flawed Replogle response labels, not the earlier runtime data.
- [Stopping-rule findings](stopping-rule/FINDINGS.md): why the biological
  results were weak, the scikit-learn 1.3 change that truncates GRNBoost2, the
  restored rule, its pre-specified confirmation on four untouched datasets, and
  the perturbation-benchmark and lineage diagnoses. Not rebuilt by
  `validation/paper/build.py`.
- [Stopping-rule replication and binning fix](stopping-rule-minerva/README.md):
  the pre-specified HPC replication at 20,000 cells (8 of 12 checks met), the
  post hoc diagnosis of 255-bin TF quantisation, exact-binning refits, cost and
  Figure 3 (`stopping-rule-minerva/make_figure.py`).
- [Post hoc checks](post-hoc-checks/README.md): checks added after an internal
  review of the preprint draft: atlas-scale fidelity of the benchmarked 0.5
  engine, tree-count correlations and regulon concordance between stopping rules.
- [Verified results](RESULTS.md): complete performance and biological tables.
- [Methods and captions](METHODS.md): comparison scope, endpoint history and figures.
- [Audited researcher workflow](../../site_docs/validated-workflow.md): local
  one-command RNA and paired recipes, corrected scoring and regulator coverage.
- [Biological case study](case-study/README.md): donor-level lineage activity,
  random-target controls and the filtering stage excluding three regulators.
- [Directional activity diagnostic](../../validation/f1_gap/README.md): closes
  the average F1 gap on reused cohorts by retaining both activity channels;
  includes local cost measurements and an explicit researcher-facing option.
- [Independent directional confirmation](evidence/directional/AUDIT.txt): verified
  external F1, remaining recall differences and atlas scoring/storage costs.
- [Evidence manifest](evidence/manifest.json): 212 checksummed source files.
- [Build verification](build-verification.json): reproduced statistics and output hashes.

Rebuild the original scaling/donor tables and the three scientific figures from the repository root:

```bash
python validation/paper/build.py
```

Tested with Python 3.13.9, NumPy 2.5.4, pandas 3.1.1 and matplotlib 3.11.2.
This performs no inference or classifier fitting. The new final-Minerva
subset has its own source manifest and replay receipt under `minerva-final/`;
the older figure builder does not regenerate that subset. It verifies every source
hash, recomputes the seven workflow summaries and all 96 corrected and 192 directional donor-fold
metrics, and produces CSV tables plus SVG, PDF and PNG figures.

The corrected Replogle analysis also has a separate [reproduction guide](recalibration/REPRODUCE.md),
verification receipt, source manifest and two figures. The original three-figure
builder does not regenerate it. The compact copy does not include the full raw
counts, per-pair results or local R resume objects; their hashes and locations
are recorded in the analysis manifest.

The small evidence directory contains audited summaries, per-run measurements,
confusion matrices, protocols and exact key source files. It is not a complete
raw-data archive or a standalone HPC runner. Original return archives and some
full matrices are not included.
Historical audits describe the knowledge available on their dates. The 9 November
directional confirmation supersedes earlier requests for further scoring tests.

The fix is explicit: `rustscenic.aucell.score(expression, regulons, seed=20261106)`.
Default scoring remains unchanged. The signed contrast helper is an assessment
adapter in `validation/biological_fidelity`, not a silent change to pipeline output.

Existing community outputs and earlier release benchmarks remain separate
evidence. The 704 CELLxGENE-derived outputs demonstrate a distributed resource,
not 704 independent biological validations or this paper's benchmark archive.
