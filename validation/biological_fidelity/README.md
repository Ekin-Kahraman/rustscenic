# Preserve direction in biological activity comparisons

The November BMMC benchmark merged each TF's positively and negatively
correlated targets before AUCell scoring. RustScenic already preserves these
groups during eRegulon assembly. The benchmark export discarded that information.

`polarity.py` extracts the signed groups from the returned edge tables. Score
each group with the same AUCell implementation for both methods, then calculate:

```text
TF contrast = AUC(positively correlated targets) - AUC(negatively correlated targets)
```

This gives one feature per TF, using the original common-TF roster. It can be
negative and is not an ordinary AUCell AUC. Correlation sign does not establish
functional activation or repression. Unknown or ambiguous contexts fail explicitly.

```python
from polarity import signed_gene_sets, signed_contrast
from rustscenic.aucell import score

# edges: archived tf/target/context table; expression: AnnData or (matrix, genes)
signatures = signed_gene_sets(edges, method="rustscenic")
directional_activity = score(expression, signatures, seed=20261106)
tf_contrast = signed_contrast(directional_activity)
```

Use `method="reference"` for the archived SCENIC+ context format. For comparisons,
use the same scorer, gene axis and scoring seed for both methods. Retain the historical unsigned endpoint
and label this additional endpoint explicitly; do not silently replace results.

## Local diagnostic, 7 November 2026

Real paired BMMC data, 5,000 genes and 20,000 regions. Networks were fit on seven
training donors. Classification uses the original fixed classifier, three test
donors and three inference seeds. These test results had already been inspected:
the following is a post hoc diagnostic, not independent validation or an
equivalence test. Seeds are not independent biological replicates.

| Network training cells | RustScenic, original union macro-F1 | RustScenic, signed contrast | SCENIC+, signed contrast |
| ---: | ---: | ---: | ---: |
| 5,000 | 0.368849 | 0.405922 | 0.426781 |
| 20,000 | 0.481983 | 0.512074 | 0.512593 |

The corrected endpoint improved RustScenic in all six size/seed settings during
training-donor development. On test donors, seed-averaged improvement was positive
for all three donors at both sizes. The 20,000-cell mean gap almost disappeared;
the 5,000-cell gap remains. This does not show improved TF-target edge accuracy.

At 20,000 cells, 28 of 55 common TF/seed combinations mixed positive and negative
targets in RustScenic. None mixed both signs in the reference. Naive B-cell F1
rose from 0.446 to 0.588 and CD8 T-cell F1 from 0.266 to 0.350, still below the
reference's 0.644 and 0.405. SPI1 and GATA1 lineage AUROCs remain above 0.99.
PAX5, EBF1 and TCF7 remain missing. No network edges were added or removed.
Some classes worsened: proerythroblast F1 fell 0.019, lymphoid progenitors 0.018,
HSCs 0.017 and plasma cells 0.013. Mean improvement is not universal improvement.

A training-only control adds the two separately normalised AUCs instead of
subtracting them. At 20,000 cells its macro-F1 was 0.475, versus 0.446 for the
original union and 0.490 for signed contrast. At 5,000 cells the corresponding
values were 0.332, 0.340 and 0.393. The correction changes both direction and
relative group weighting; its benefit cannot be attributed to a sign flip alone.

Trimming weaker GRN targets was inconsistent in development, so no pruning change
was adopted. GRN inference, motifs, topics, enhancer links and public API defaults
remain unchanged. This diagnostic uses a cached-output assessment adapter.

## Verification and cost

- Original shared pySCENIC scores reproduced to a maximum absolute error of
  `4.72e-16`; all 12 corrected matrices reproduced exactly after the allocation change.
- Same common-TF columns, cell identities, training/test donor split and classifier.
- 384 classification folds checked against their confusion matrices; all converged.
- The unchanged RustScenic 0.5.0 scorer also improved mean macro-F1 from 0.481359
  to 0.511851 on the same 20,000-cell networks and test donors.
- Local scoring check: 30,000 cells, 5,000 genes, 157 TFs, 191 signed signatures,
  16 Rust threads. Warm mean scoring/export time was about 0.091 s unsigned and
  0.140 s signed. Whole-process peak RSS was 440 MB versus 503 MB on macOS.
  These are small local scoring measurements, not new atlas runtime or PSS claims.
- The helper allocates the final contrast matrix once. Twelve focused regression
  tests and 23 existing eRegulon tests passed against the benchmark's 0.5.0 engine.

## Returned external validation and native tie fix

Huangfu's independent return exposed a scorer difference on the external Kang
cohort: the native scorer resolved equal expression values by gene index, while
the shared pySCENIC scorer used a seeded gene permutation. The new optional
`score(..., seed=...)` matches that ordering in Rust, without shuffling or
densifying the expression matrix. Omitting the seed preserves existing results.

Reusing all 12 returned comparisons (12,315 cells, 4,741 genes, two training sizes,
three inference seeds, union and signed endpoints), seeded native scores agreed
with the shared scorer within `3.31e-8`. All legacy matrices remained bit-exact.
Of 96 donor-held-out classifier confusion matrices, 93 matched exactly; three
had small differences consistent with native float32 rounding.

For the 20,000-cell networks, native signed macro-F1 rose from 0.682918 to
0.692892. The same RustScenic networks scored by the shared reference gave
0.692902; SCENIC+ networks gave 0.711326. The scoring discrepancy is resolved,
but the remaining network-level biological gap is not. These reused results are
a compatibility check, not a new untouched biological validation cohort.

The 30,000-cell local cost check used three fresh processes per mode and seven
warm repetitions per process. Median warm scoring plus contrast time was
0.1123 s without the seed and 0.1185 s with it; median whole-process peak RSS was
506.2 MB and 506.6 MB. These macOS measurements include input-loading memory;
they do not establish new atlas-scale runtime or PSS results. Original HPC
performance measurements remain attached to their original code and settings.

## Independent HPC confirmation, 8 November 2026

All 18 confirmation tasks completed and all 195 returned file hashes matched.
The corrected native scorer matched the shared reference within `3.31e-8` on
all 12 external matrices (12,315 cells each). It also agreed within `3.13e-8`
on 10,000 sampled atlas cells per frozen network. All three full scoring runs
used 817,740 real cells; reference agreement was inspected on the returned
samples, not every full-atlas row. Legacy outputs retained identical hashes.

Against the same build with legacy tie ordering, seeded scoring plus signed
contrast added 0.42% time and 3.48% peak sampled process-tree PSS (median paired
ratios across three seeds). This confirms no material scoring-stage regression
in the tested configuration; it is not a new whole-workflow speedup.

The corrected native macro-F1 is 0.606447 at 5,000 training cells and 0.692892
at 20,000, versus 0.662094 and 0.711326 for the SCENIC+ networks. At 20,000,
the paired donor interval for the difference is `[-0.03448, -0.00296]`.
The remaining network-level gap must be reported. Of 96 HPC confusion matrices,
92 match the shared scorer exactly; the earlier 93/96 count above describes
the local run. Native score matrices themselves reproduce exactly across these
runs, with small downstream classifier differences.

The consolidated [paper evidence](../../docs/paper/README.md) preserves the
original union endpoint, signed diagnostics, controls and source provenance.
