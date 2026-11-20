# Confirmation run log

- 21:01 UTC protocol hashed (CONFIRMATION-FREEZE.json). Inputs staged afterwards.
- native_cross, native_same: all 24 fits completed first time.
- reference_cross: all 12 fits completed first time.
- reference_same: 4 of 12 fits completed, then the run was stopped because the
  laptop was oversubscribed (load average above 100: several jobs sharing ten
  cores, BLAS threads not capped). Restarted with OMP_NUM_THREADS=1,
  OPENBLAS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1. Completed fits are kept and
  skipped; the interrupted fit restarts from scratch with the same seed. No
  setting, seed, dataset or metric changed. No confirmation score had been
  computed at that point.
