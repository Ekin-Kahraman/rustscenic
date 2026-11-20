# RustScenic Minerva validation package

This package combines the published-release atlas checks, the existing
GSE274113/GATA1 transfer analyses below, and the new independent K562 study
specified in HOLDOUT-PROTOCOL.md. The K562 comparison fits control cells only;
its perturbation-response performance remains unexamined at handoff. Large
atlas inference and the K562 experiment are new runs; the existing biology
arm reproduces already-inspected analyses and is not a new holdout.

## Transfer to GSE274113

This protocol compares the six frozen 20,000-cell BMMC networks from the 6 November benchmark against an external primary human erythroid CRISPR screen. It also evaluates two low-cost rerankings using the correlations already exported by the workflows. This is a first external test for the frozen networks. Choosing a reranking based on this screen would make that choice exploratory until checked on another dataset.

## Data and contrasts

- Use the public GSE274113 annotated metadata and the 14 rep-numbered 10x feature-barcode matrices. Match barcodes and reconcile per-cell RNA and ATAC depths against the metadata. Retain all guide-assigned cells with the recorded study QC. Use each library's AAVS1 guides as cutting controls and all six NT-labelled guides as a sensitivity analysis.
- Compare each targeting guide with control cells within each library and day. Aggregate counts across cells, then compute log2(CPM + 1) differences with full modality library-size denominators. Require 20 cells per guide and 20 control cells in each contributing library. Split the sorted library numbers within each day into alternating halves. Average library contrasts with equal weights within each half. A response requires the same two or more guides to exceed an absolute 0.25 log2 difference in the same direction in both halves. Require detection in at least 1% of control cells in each half. Exclude the directly perturbed gene from retrieval. The halves provide within-screen technical replication. They do not constitute independent donors.
- Match uniquely named RNA genes to the frozen 5,000-gene panel. Count response genes outside the panel before restricting to common genes. Analyse ATAC secondarily on the fixed 20,000-peak BMMC panel. Align peaks per library by a unique reciprocal overlap of at least 50%, dropping ambiguous matches on either side. Require control detection and a measurable guide contrast in both halves. The study's per-library peak calls may differ in their interval boundaries.

## Endpoints

- Primary comparison uses the AAVS1 response sets, with all targeted regulators and days included whenever both positive and unlabelled eligible genes exist. Report average precision and tie-aware expected hits in the top 50. Missing edges receive score zero. Also report emitted predictions and hits at the same cap using positive scores only. For macro means, average algorithm seeds and days within each TF, then give TFs equal weight. Report the subgroup where both final workflows output the TF on the same seed and day.
- Compare raw importance, final membership ranked by raw importance, absolute Pearson across all eligible BMMC genes, and a uniform rank baseline. Pearson uses the exact shared 20,000 training cells. Use each final output's number of eligible edges per TF as a second fixed budget for comparing ranking methods. Report raw candidates, the overlap with candidates supported by a motif-associated region and a region-to-gene link, and final membership. A saved target without a positive raw importance gets a tied positive score below every positive raw importance in the final list.
- Evaluate two rerankings offline. Rank existing positive raw candidates by absolute saved Pearson, and rank the final list by the same statistic. Compare both to their corresponding importance ranking. Test these on the frozen screen only, with no parameter search. Timings cover construction of scores from cached results. A matched end-to-end run is needed to quantify production runtime and memory after any code change.
- For final edges with one direction in the archived assembly context, report agreement between that sign and the inverse sign of the knockout response. Test ATAC final membership and motif NES as secondary prioritisation results. The RNA and ATAC contrasts can include downstream responses and do not establish direct enhancer-to-gene links.
- Export per-regulator and day tables, all seeds, response counts, coverage, gene/peak lists and source hashes. Report null or unfavourable comparisons with the rest.

Study DOI 10.1126/science.ads7951. GEO accession GSE274113.

## Additional GATA1 assay

Evaluate the same frozen GATA1 RNA predictions against the published
24-hour GATA1 versus AAVS1 PRO-seq differential-expression table in
GSE274110. Use unique gene symbols with a finite adjusted p-value and
exclude GATA1 itself. A response has an adjusted p-value at most 0.05
and absolute log2 fold change at least 0.25. Use the same ranking
metrics on the shared panel. This is a different assay from the
single-cell RNA and ATAC screen in the same paper. Report it separately
from the 19-regulator result and retain the different response prevalence.

## Scaling of the published release

- Reuse the original prepared matrix checksums and run the published
  v0.6.1 wheel on the 1,306,127-cell RNA atlas and the 817,740-cell paired
  atlas at seed 777 and 16 threads. Use the public validated workflow,
  with directional scoring for the paired run. Return stage times,
  input load time, process-tree PSS samples, coverage, manifests, compact
  predictions and hashes of the full outputs. These are one-seed release
  measurements. The earlier paired baselines used the archived v0.5.0
  workflow, so retain their version labels.
- On the same node allocation as the RNA run, repeat the GRN and
  correlation stages with 6,000 genes and the same cells, normalisation,
  256 TFs, seed and hyperparameters. Keep the original 3,000 genes and
  add the highest-variance expressed genes from the first 10,000 source
  barcodes, breaking ties by symbol. Use the benchmark's original cell
  permutation. Verify all shared expression values and the complete
  barcode order against the cached 3,000-gene matrix before fitting.
  Compare named stage times and stage-specific sampled PSS. Keep input
  preparation time separate.
  The 6,000-gene arm covers inference and correlation. Whole-process PSS
  measurements describe each arm's scope.
- For each produced GRN, alternate five repeats of top-50 selection
  by raw importance and absolute saved Pearson correlation. Hold the
  positive-importance candidate set fixed and break ties by target gene.
  Save both lists and the time to create them in memory. The measurements
  describe the optional ranking step and do not include a modified
  regulon assembly.

Release DOI 10.5281/zenodo.24093808. Paired dataset DOI
10.1038/s41586-026-10326-9. The two archived input manifests pin the
exact matrices and resources.
