# Independent K562 functional validation

Protocol fixed on 11 November 2026 before fitting or evaluating RustScenic or
the comparator on this dataset. Metadata and control-cell expression may be
inspected to establish feasibility. No perturbation-response performance is
used to select genes, settings, thresholds or methods. This is a prospective
analysis plan within this project, not an external preregistration.

## Question and source

Do networks fitted to unperturbed K562 cells prioritise expression responses
to transcription-factor activation in the same cell context? Use Norman et
al., Science 2019, DOI 10.1126/science.aax4438, GSE133344. Pin the scPerturb
file NormanWeissman2019_filtered.h5ad in Zenodo record 7041849, original MD5
c870e6967d91c017d9da827bab183cd6. Cite the harmonisation paper, DOI
10.1038/s41592-023-02144-y. This separate study has not previously been used
to assess or tune this project's networks. It is CRISPR activation, not
knockout, and tests RNA response prioritisation, not direct binding or
individual enhancer-to-gene links.

## Separation and inference

- Keep the study's filtered cells. Use the harmonised single-target and
  control assignments; exclude combinations. Record all counts and exclusions.
- Within each of the eight gemgroups, order control barcodes by SHA-256 of
  `20261111:<barcode>`. Use the first floor(0.8*n) controls for fitting and
  the remainder solely as evaluation controls. These sets must be disjoint.
- Fit on log1p(10,000*counts/full-cell-UMI-total), using only training controls
  to select features. Retain genes detected in at least 1% of these cells
  with positive variance. Include all eligible genes in the fixed human TF
  list, then add the most variable remaining genes to 5,000, ties by symbol.
  Record all single-target TFs, including those absent from the fitted panel.
- Compare the published RustScenic 0.6.1 RNA recipe with the existing
  arboreto 0.1.6 / pySCENIC 0.12.1 RNA recipe, seeds 777, 778, 779. Both use
  the exact same cells, genes, TF predictors, original human motif ranks and
  annotations. Preserve the original 27,090-gene motif rank universe.
  Use the existing 5,000-tree ceiling, learning rate .01, max_features .1,
  subsample .9, max_depth 3 and early-stop window 25. Algorithms differ.
- Save raw networks and motif-filtered memberships before reading the
  evaluation count aggregates. Compare original importance, the previously
  specified absolute-Pearson reranking of positive raw candidates, final
  membership ranked by importance, and final Pearson reranking. Include
  full Pearson and uniform-rank baselines. No parameter search or new
  default is selected on this screen.

## Responses and endpoints

- Aggregate raw counts by gemgroup and guide construct. Treat multiple
  guide configurations as constructs, not independent guide-sequence or
  biological replicates. Within each gemgroup, pool the held-out negative
  controls and single-target cells separately. A usable contrast requires
  at least 10 targeting cells and 20 evaluation controls.
- Split sorted gemgroups into alternating halves (1,3,5,7 and 2,4,6,8).
  Require at least two usable gemgroups and at least 50 targeting cells
  in each half. Average per-gemgroup log2(CPM+1) contrasts with equal weights.
  A response exceeds an absolute .25 difference with the same sign in both
  halves. Report .5 and 1.0 thresholds as prespecified sensitivities, never
  as replacement primary endpoints. Require control detection in at least
  1% of evaluation controls in both halves. Exclude the perturbed TF itself.
- These are reproducible effect-size response labels, not significance tests
  or a complete causal ground truth. Record null control contrasts by
  comparing each negative-control construct with the remaining evaluation
  controls using the identical rules. Retain null and adverse results.
- Primary endpoint: equal-TF macro average precision at threshold .25 for
  final importance, with seeds averaged within TF. Compare RustScenic with
  pySCENIC, full Pearson, and the uniform baseline on identical gene axes.
  Tied absent edges get score zero. Secondary: raw-network AP, rerankings,
  tie-aware expected hits among 50, emitted hits and number of predictions,
  coverage, shared-final-TF subgroup, and CRISPRa direction concordance.
  Count eligible and responsive genes outside the selected panel separately.
- Return all per-TF, per-seed and sensitivity rows. Compute paired differences
  after averaging seeds within TF and 2,000 TF-resampling percentile intervals
  with fixed seed 20261111. These describe variation across assayed TFs
  conditional on this screen. Cells, gemgroups, constructs and algorithm
  seeds are not independent donors. Do not claim donor-level uncertainty.
- Save raw predictions and their hashes even if evaluation is null or fails.
  Do not relabel a partial run as complete. Report both coverage and ranking;
  distinguish a smaller output from a more accurate output.

## Interpretation and stopping

This adds a context-matched experiment to the earlier cross-context transfer
test. It cannot isolate context mismatch as the sole cause of a difference:
the studies also differ in intervention, assay and cell system. A positive
result supports usefulness in this K562 setting; a null result narrows the
claims. Either result is retained. Do not commission another sweep simply to
obtain a favourable outcome. No blanket accuracy percentage, noninferiority,
causal-edge superiority or production reranking change follows from this run.

Sources: https://doi.org/10.1126/science.aax4438 ;
https://doi.org/10.1038/s41592-023-02144-y ;
https://doi.org/10.5281/zenodo.7041849 .
