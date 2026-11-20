# Implementation notes

The initial SCEPTRE input adapter used the maximum-assignment default minimum of five guide counts on binary assignment indicators. QC rejected all cells before calibration or discovery. Set min_grna_n_umis_threshold=1 explicitly and verify one assigned construct per cell, as required by the frozen plan. The source already supplies guide assignments; these indicators are not measured guide UMI counts. The rejected attempt is preserved in k562-calibration-initial-adapter-failure.log. This is an adapter correction, not an outcome-driven statistical change.

The installed data.table lacks fwrite gzip support. Outputs use plain CSV instead; no package or statistical settings changed. The failed write occurred before calibration.

The public assignment getter omits column names. The adapter instead verifies the imported barcode alignment before import, the number/order of assignment columns and exact assigned construct in every cell. A rejected assertion on the intentionally absent getter names is preserved. No calibration result had been produced.

The first parallel calibration stalled in a libgomp barrier after fork, before any result. The process sample and log are retained. OpenMP and data.table are limited to one thread per worker to prevent inherited thread-pool deadlock; the two SCEPTRE worker processes and statistical settings remain unchanged.
