# RustScenic benchmark and biological evidence

Verified measurements, methods and reproducible figures, updated 9 November 2026.

**Positioning:** a scalable Rust implementation of core SCENIC-style analysis
stages, with measured workflow gains and an explicit biological trade-off.

- [Verified results](RESULTS.md): complete performance and biological tables.
- [Methods and captions](METHODS.md): comparison scope, endpoint history and figures.
- [Audited researcher workflow](../../site_docs/validated-workflow.md):
  one-command RNA and paired recipes, corrected scoring and regulator coverage.
- [Biological case study](case-study/README.md): donor-level lineage activity,
  random-target controls and the filtering stage excluding three regulators.
- [Directional activity diagnostic](../../validation/f1_gap/README.md): closes
  the average F1 gap on reused cohorts by retaining both activity channels;
  includes local cost measurements and an explicit researcher-facing option.
- [Independent directional confirmation](evidence/directional/AUDIT.txt): verified
  external F1, remaining recall differences and atlas scoring/storage costs.
- [Evidence manifest](evidence/manifest.json): 213 checksummed source files.
- [Build verification](build-verification.json): reproduced statistics and output hashes.

Rebuild all tables and the three scientific figures from the repository root:

```bash
python validation/paper/build.py
```

Tested with Python 3.13.9, NumPy 2.5.4, pandas 3.1.1 and matplotlib 3.11.2.
This performs no inference or classifier fitting. It verifies every source
hash, recomputes the seven workflow summaries and all 96 corrected and 192 directional donor-fold
metrics, and produces CSV tables plus SVG, PDF and PNG figures.

The small evidence directory contains audited summaries, per-run measurements,
confusion matrices, protocols and exact key source files. It is not a complete
raw-data archive or a standalone HPC runner. Original return archives and some
full matrices still require a durable deposit before paper submission.
Historical audits describe the knowledge available on their dates. The 9 November
directional confirmation supersedes earlier requests for further scoring tests.

The fix is explicit: `rustscenic.aucell.score(expression, regulons, seed=20261106)`.
Default scoring remains unchanged. The signed contrast helper is an assessment
adapter in `validation/biological_fidelity`, not a silent change to pipeline output.

Existing community outputs and earlier release benchmarks remain separate
evidence. The 704 CELLxGENE-derived outputs demonstrate a distributed resource,
not 704 independent biological validations or this paper's benchmark archive.
