# Retaining directional activity closes the average classification gap

Local diagnostic, 8 November 2026. No new network fitting, HPC request or
publication. All original benchmark endpoints remain intact.

## Cause and change

The signed contrast reduces positive-target AUC and negative-target AUC to
one number per TF. RustScenic retained both directions for 28 of 55 common
TF/seed combinations at 20,000 training cells; SCENIC+ retained only one
direction for every corresponding TF. Subtraction therefore discarded an
extra directional feature only in RustScenic. The seeded scorer itself was
correct: recomputed external contrast matrices match Huangfu's corrected
matrices bit for bit.

Keep the two channels separate with `rustscenic validated --activity-mode
directional`. RNA already retains directions. Paired union defaults remain
unchanged. Networks, motif thresholds and native kernels are unchanged, and
only the selected activity representation is computed. Regulator coverage
recognises the signed names. The suffixes describe expression association,
not a proven causal effect; region channels inherit their target context.

## Results

Donor-averaged macro-F1, with three network seeds averaged within each donor.
Higher is better. Network-training cells are distinct from evaluation cells.

| Evaluation | Network-training cells | Rust contrast | Rust separate channels | SCENIC+ separate channels |
| --- | ---: | ---: | ---: | ---: |
| BMMC development, 7 donors | 5,000 | 0.392550 | 0.434780 | 0.386355 |
| BMMC development, 7 donors | 20,000 | 0.489839 | 0.512792 | 0.489602 |
| BMMC test, 10,000 cells / 3 donors | 5,000 | 0.405922 | 0.447488 | 0.426799 |
| BMMC test, 10,000 cells / 3 donors | 20,000 | 0.511929 | 0.533793 | 0.512542 |
| Kang external, 12,315 cells / 8 donors | 5,000 | 0.606962 | 0.663669 | 0.662094 |
| Kang external, 12,315 cells / 8 donors | 20,000 | 0.692892 | 0.712225 | 0.711258 |

Rust improved in every size/seed/cohort combination. External Rust-minus-
SCENIC+ differences are +0.001575 at 5,000 training cells (paired donor-bootstrap
95% interval −0.021691 to +0.026655) and +0.000967 at 20,000 (−0.003210 to
+0.004221). These intervals do not establish a prospectively defined
equivalence margin. Some individual seeds and cell classes still favour the
reference. All 432 classifier folds converged.

The fixed comparison uses the same original common TF roster and two slots
per TF, zero-filling absent directions for both methods. Effective nonconstant
features differ: on BMMC, Rust has 15/11/19 versus reference 9/6/12 at 5,000,
and 22/31/30 versus 14/21/20 at 20,000. External 5,000-cell networks have
15/11/18 Rust features because one BCL11A direction fails the frozen 80%
gene-coverage rule; other counts are unchanged. This is a test of retaining useful existing
information, not a feature-count-matched proof of better inferred edges.

The [protocol](PROTOCOL.md) fixed this single alternative before evaluations.
Classifier, donor splits, 80% external gene coverage, seed and AUCell settings
were unchanged. Development donors contributed to network fitting. BMMC test
and Kang results are retrospective because both cohorts were already examined.
This diagnostic supports downstream utility; it does not establish causal
network accuracy or fresh independent biological equivalence.

Small downstream classifier differences occur between local ARM/macOS and
HPC x86/Linux despite identical input scores and package versions. At 5,000,
the old HPC contrast mean was 0.606447, versus 0.606962 locally. Use the paired
local baseline above for this diagnostic, retaining historical HPC endpoints.
The reference was also rescored natively: float32 values match the shared
reference after rounding (maximum original float64 difference 1.49e-8).
Its 20,000-cell mean is 0.711258 locally versus 0.711326 in the original report.

## Cost and software checks

Six fresh processes replayed all downstream paired stages on 5,000 real BMMC
cells, 5,000 genes and 20,000 peaks, reusing only fitted topics and GRN. Three
runs per mode, alternating order, same four-thread limit and installed wheel:

| Measurement | Historical union | Separate channels |
| --- | ---: | ---: |
| Median post-fit time | 9.6466 s | 9.6459 s |
| Median sampled process RSS peak | 3.986 GB | 3.579 GB |
| Gene + region activity arrays | 8.68 MB | 9.96 MB |
| Signatures per activity array | 217 | 249 |

There was no observed time or peak-memory regression in this replay. Starting
RSS varied, so do not claim a percentage memory improvement. Extra columns
are a real output-storage cost. These are local downstream measurements, not
new atlas PSS measurements or updated full-workflow speedups.

All six replays preserved edges exactly. Directional gene and region sets
match the signed archived edges exactly; directional gene scores match direct
native scoring. Default union activity remains bit-exact to the previous
candidate. Batching and float32 output are preserved. A positive two-direction
regression test covers the selected representation, edges and native scores.
The complete candidate suite passed 761 tests with one optional live-network
test skipped in the current-dependency regression environment. The diagnostic
environment's broader suite also exposed unrelated missing optional Scanpy,
legacy pandas merge-order expectations and a pre-existing sub-millisecond
benchmark-rounding failure; those logs are retained. The focused workflow
tests and real-data diagnostic passed with the original pinned dependencies.

## Reproduce

Install the local candidate identified by its wheel hash. The biological
diagnostic uses the frozen NumPy 1.23.5, pandas 1.5.3, SciPy 1.12.0 and
scikit-learn 1.3.2 environment. Preserve input and feature order.

```bash
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  validation/f1_gap/assess.py --artifacts /path/to/RustScenic \
  --output /path/to/new-external-diagnostic --cohort external
```

Use `--cohort development` or `bmmc_test` for the other endpoints. Each result
records source hashes, all donor confusions, feature coverage and timings.
`check_workflow.py --help` describes cached recipe replay and cost measurement.
It checks against the archived outputs and refuses to overwrite an audit.

Full local evidence, including independent audit, wheel, logs and activity
matrices: `/Users/ekin/Desktop/RustScenic/RustScenic-F1-Diagnosis-20261108`.
The candidate reuses version metadata 0.5.1; identify it by SHA-256 until an
explicitly versioned release is built. Nothing has been sent or published.
