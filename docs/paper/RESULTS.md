# Consolidated results

Generated from checksummed evidence by `validation/paper/build.py`. All values
are scoped to their original experiment; different measurements are not pooled.

## Configured workflows

21 complete same-allocation pairs, seven sizes, three seeds each, 16 CPUs.
Times include prepared-input loading, analysis and standard export. Shared
preparation and raw-data processing are excluded. Native inference, pruning,
topics and eRegulon algorithms differ. These are workflow gains, not a pure
Rust language speedup. Headline ratios divide method medians; paired-ratio
sensitivities and all 42 runs are retained in the CSV tables.

| dataset | cells | rust_minutes | reference_minutes | speedup | rust_pss_gb | reference_pss_gb | memory_reduction_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| atlas | 100000 | 1.874 | 6.822 | 3.641 | 0.537 | 10.584 | 94.926 |
| atlas | 1306127 | 42.314 | 168.382 | 3.979 | 4.563 | 134.877 | 96.617 |
| bmmc | 10000 | 2.186 | 20.362 | 9.315 | 4.055 | 13.389 | 69.716 |
| bmmc | 50000 | 12.494 | 79.601 | 6.371 | 10.142 | 33.998 | 70.169 |
| hdma | 100000 | 12.128 | 150.592 | 12.417 | 18.228 | 44.760 | 59.276 |
| hdma | 500000 | 57.908 | 747.044 | 12.901 | 91.531 | 214.947 | 57.417 |
| hdma | 817740 | 102.547 | 1474.232 | 14.376 | 149.878 | 349.054 | 57.062 |

## Biological utility

BMMC original common-TF union endpoint, 10,000 cells from three unseen donors.
Algorithm seeds are averaged within each donor before averaging donors.
The expression baseline uses 50 training-fitted principal components.

| n | rust_common | reference_common | rust_native | reference_native | expression_pca50 |
| --- | --- | --- | --- | --- | --- |
| 5000 | 0.369 | 0.427 | 0.623 | 0.427 | 0.741 |
| 20000 | 0.482 | 0.512 | 0.638 | 0.514 | 0.763 |

The historical columns `rust_native`/`reference_native` mean all TF signatures
with the common scorer, not native scoring. They have unequal feature counts
and cannot establish better networks. The signed BMMC diagnostic is post hoc
and is separately retained in `bmmc-signed-diagnostic.csv`.

Kang external cohort: 12,315 cells, eight donors, frozen BMMC networks. Classifiers
leave one external donor out; networks never fit those donors. Corrected native
signed scores are shown below. The tie fix reuses this inspected cohort and is
a numerical repair, not a fresh biological confirmation. Intervals are paired
donor bootstrap intervals; no noninferiority margin was prespecified.

| training_cells | rust_native_signed | rust_reference_signed | reference_signed | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| 5000 | 0.606 | 0.607 | 0.662 | -0.056 | -0.078 | -0.030 |
| 20000 | 0.693 | 0.693 | 0.711 | -0.018 | -0.034 | -0.003 |

The 20,000-cell old native signed endpoint was 0.682918; the corrected endpoint
is 0.692892. Its observed improvement interval includes zero. The remaining
gap to SCENIC+ networks for this contrast endpoint is negative at both training
sizes. The separate-direction confirmation below retains more activity
information. Neither metric is the fraction of correct regulatory edges.

## Directional activity: independent HPC confirmation, 9 November

All 192 donor-fold metrics and paired donor-bootstrap intervals reproduce.
The fixed representation retains positive and negative target activity as
separate features. Networks and cohorts are reused; this is independent
execution, not untouched biological validation. Both network sources use the
same native scorer, common TF roster and donor-aware classifier.

| training_cells | comparison | metric | rust_directional | comparator | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | directional_vs_contrast | macro_f1 | 0.663828 | 0.606447 | 0.057381 | 0.042065 | 0.076293 |
| 5000 | directional_vs_contrast | balanced_accuracy | 0.749546 | 0.711445 | 0.038100 | 0.019146 | 0.057584 |
| 5000 | directional_vs_scenicplus | macro_f1 | 0.663828 | 0.662094 | 0.001733 | -0.021513 | 0.026811 |
| 5000 | directional_vs_scenicplus | balanced_accuracy | 0.749546 | 0.758360 | -0.008814 | -0.043321 | 0.033228 |
| 20000 | directional_vs_contrast | macro_f1 | 0.712332 | 0.692892 | 0.019441 | 0.004866 | 0.034706 |
| 20000 | directional_vs_contrast | balanced_accuracy | 0.782637 | 0.778246 | 0.004391 | -0.019417 | 0.024923 |
| 20000 | directional_vs_scenicplus | macro_f1 | 0.712332 | 0.711258 | 0.001074 | -0.003019 | 0.004277 |
| 20000 | directional_vs_scenicplus | balanced_accuracy | 0.782637 | 0.797620 | -0.014983 | -0.023889 | -0.004481 |

At 20,000 network-training cells, mean macro-F1 is 0.712332 versus 0.711258
for SCENIC+ networks. The average macro-F1 deficit is closed; the interval
does not establish superiority or formal equivalence. Mean cell-type recall
remains 0.782637 versus 0.797620. All original endpoints and adverse class
results are retained, including weak experimental-reference recovery below.

On 817,740 cells, median paired overhead versus union activity is 2.62% for
scoring, 0.29% including I/O, and 4.52% for peak process-tree PSS. One seed has
15.94% higher PSS. Raw activity storage rises 34.33%; full HDF5 files rise
about 27.05%. This is scoring-stage cost evidence, not a new full-workflow
speedup. Full arrays remain on HPC with hashes; returned samples match prior
numerical controls exactly. See the [confirmation audit](evidence/directional/AUDIT.txt).

## RNA reference recovery

Mean of three algorithm seeds. Unlisted edges are unlabelled, not established
negatives. Missing motif-pruned edges keep zero scores in the same candidate
universe. ChIP binding and perturbation response do not prove direct causality.

| reference | method | average_precision | reference_prevalence | retained_reference_recall |
| --- | --- | --- | --- | --- |
| hESC-ChIP-seq-network.tsv | reference | 0.148 | 0.147 | 0.005 |
| hESC-ChIP-seq-network.tsv | rustscenic | 0.148 | 0.147 | 0.006 |
| mDC-ChIP-seq-network.tsv | reference | 0.048 | 0.046 | 0.010 |
| mDC-ChIP-seq-network.tsv | rustscenic | 0.048 | 0.046 | 0.011 |
| mESC-ChIP-seq-network.tsv | reference | 0.293 | 0.291 | 0.007 |
| mESC-ChIP-seq-network.tsv | rustscenic | 0.293 | 0.291 | 0.006 |
| mESC-lofgof-network.tsv | reference | 0.118 | 0.110 | 0.018 |
| mESC-lofgof-network.tsv | rustscenic | 0.117 | 0.110 | 0.016 |

Absolute Pearson exceeds both original GRN methods on all four references.
Pruned recovery remains weak in absolute terms; retain coverage and controls.

## Scoring confirmation

18 completed tasks; 195 verified return hashes; 96 recomputed donor folds.
Corrected native/reference maximum absolute error: 3.301e-8 on all 12 external
matrices and 3.122e-8 on the returned 10,000-cell sample per atlas network.
Full execution used 817,740 cells, but reference numerical inspection there
is sampled. Complete outputs remain on HPC with recorded hashes.

Seeded correction versus the same candidate build without the seed: median
paired scoring-time ratio 1.004175, PSS ratio 1.034812. Against v0.5.0, those
ratios are 0.928426 and 0.908883. Small three-run differences are descriptive;
they do not revise the full-workflow performance figures above.

See METHODS.md for endpoint chronology, controls, parameter differences and
figure captions. Full retained HPC output arrays still require durable archival.
