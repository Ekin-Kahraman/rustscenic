# Calibration sampling amendment

Recorded before any corrected network comparison or discovery result was inspected. All initial control checks passed their frozen criteria.

The pinned SCEPTRE C++ samplers hard-code seed 4. Consequently external R seeds 2026111201 and 2026111202 produced byte-identical median-a and median-b files in both cohorts. These are a deterministic reproduction, not independent calibration samples, and must not be counted twice. No upstream source is changed.

For each cohort, run the official calibration function on 20,000 median-size pairs. Retain the entire result, remove pair identities already present in median-a, and select 10,000 additional distinct pair identities using R sample() with seed 2026111202, without using p-values/effects for selection. Require at least 10,000 new identities. Apply the original operational gate unchanged to this validation set. Report its overlap policy and recognise shared cells/genes/control constructs, so it is not independent biological replication. Preserve the duplicate attempt and original checks. Discovery remains blocked until this distinct-pair validation passes. The calibration family and discovery method are unchanged.
