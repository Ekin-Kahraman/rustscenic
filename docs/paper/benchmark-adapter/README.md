# Benchmark adapter as run

`workflows.py` is the orchestration adapter used in the replicated workflow
comparisons (RustScenic 0.5.0 engine, pySCENIC 0.12.1, SCENIC+ 1.0a2). It is
copied unchanged from the recovery run package; its SHA-256,
`2b982675241c53c78b5b539a334d640a6fe780350fa605a5495e7a43f60489c4`, is recorded
in that package's `RECOVERY-SHA256SUMS`. It calls the memory-bounded batching
code in `../evidence/recovery/returned/package/bounded.py` (`peak_batch=256`,
`cell_batch=2048`). `../evidence/workflow/workflows.py` is an earlier version
kept for provenance.
