# Methods and figure captions

## Study design and evidence hierarchy

| Evidence | Purpose | Scope and interpretation |
| --- | --- | --- |
| September release and 20k-cell GRN benchmarks | Historical feasibility and controlled comparisons | Retained in existing release documentation; not pooled with later measurements |
| October 26 arboreto atlas comparison | Isolate core GRN cost | 1,306,127 real mouse-brain cells; 2,095 genes, 256 TFs; three seeds; loading excluded |
| October 28 RNA references and 3,009-cell multiome pilot | Initial biological comparison | Four RNA reference panels; permissive multiome pilot, with no significant GSEA modules; not a full-workflow headline |
| October 29 to November 6 completed scaling grid | Configured workflow performance | 21 complete pairs, seven sizes; loading through standard export; selected feature panels |
| November 6 BMMC transfer and RNA refinement | Prespecified biological utility | Seven training and three unseen BMMC donors; reference retrieval and controls |
| November 7 signed BMMC diagnostic | Explain activity-representation loss | Post hoc; same inspected test donors; retains original union endpoint |
| November 7 Kang cohort return | External transfer of frozen networks | Eight separate donors; prospective signed endpoint at first evaluation |
| November 8 native scoring confirmation | Numerical compatibility and cost | Reuses inspected Kang data and frozen atlas networks; not a fresh biological validation |
| November 9 directional confirmation | Reproduce the selected representation and atlas costs | 192 donor folds; six scoring arms on 817,740 cells; same inspected cohorts and frozen networks |

## Data, features and hardware

The mouse-brain RNA matrix is the public 10x 1M-neuron count matrix. The complete
workflow uses 3,000 selected genes and 256 TFs at 100,000 and 1,306,127 cells.
The earlier core-GRN comparison instead used 2,095 genes; its timings cannot be
subtracted from the later workflow to estimate downstream cost.

BMMC paired RNA/ATAC data are from GSE194122. Performance uses 10,000 and 50,000
cells. The human developmental multiome atlas (HDMA, source Zenodo record
15053816) provides 100,000, 500,000 and 817,740 paired cells. Each paired
performance analysis uses 5,000 selected genes, 20,000 selected regions and
256 TFs. These are real measured cells, not replicated or simulated rows.
Dataset URLs and source checksums are in `evidence/recovery/returned/package/sources.json`.

Each matched pair used the same prepared input, host, allocation and 16 CPU
cores. Seeds were 777, 778 and 779. CPU models vary across allocations, so
curves across cell counts are descriptive, not a fixed-hardware complexity fit.
The final grid retains the latest completed pair together; it never pairs an
old Rust time with a rerun reference time. Original failed attempts remain
documented, including resource failures. Their absence from completed timing
medians is not evidence that failures did not occur.

## Workflow comparison and algorithm differences

RustScenic 0.5.0, pySCENIC 0.12.1 and SCENIC+ 1.0a2 were used for full workflows.
The published Rust native binary was unchanged. The benchmark adds an explicitly
archived orchestration adapter with 256-peak linking batches and 2,048-cell
scoring batches. This is not a benchmark of an unconfigured, one-call default
pipeline. Prepared input and feature selection are shared; several calculations
are intentionally method-specific.

| Stage | RustScenic benchmark | Reference benchmark |
| --- | --- | --- |
| RNA GRN | Native gradient-boosted trees; arboreto-style early stopping | arboreto GRNBoost2 or SCENIC+ published GBM API |
| GRN settings | 5,000 tree ceiling, learning rate 0.01, feature fraction 0.1, row fraction 0.9, depth 3, early-stop window 25 | Same explicit RNA recipe; SCENIC+ API does not expose fitted-tree counts |
| RNA modules | Signed top-50 targets; at least 10 genes; correlation threshold 0.03 | Same top-50/size/sign recipe through pySCENIC's module API |
| RNA motif refinement | Native enrichment/pruning; recovery fraction 0.05, NES at least 3, minimum 5 genes | pySCENIC prune2df; recovery fraction 0.05, NES at least 3, rank threshold 1,500, annotation filtering |
| Topics | Native Gibbs implementation | pycisTopic/MALLET |
| Topic settings | 30 topics, 500 iterations, alpha 50/K, eta 0.1, top 1,000 regions per topic | Matching topic count/iterations/prior recipe, different sampler implementation |
| Region motifs | Native cisTarget enrichment, recovery fraction 0.005, NES at least 3 | pycisTarget enrichment, matching recovery fraction/NES, method-specific pruning |
| Region-to-gene | Pearson links within 150 kb; absolute correlation at least 0.03 | GBM importance plus Pearson direction in the same search region |
| eRegulons | Native assembly with at least 5 target genes and motif support | Signed GSEA with 1,000 permutations; final adjusted p below 0.05, at least 5 genes |
| Activity | Native AUCell | pySCENIC/SCENIC+ AUCell |

SCENIC+ initially returns GSEA candidates under a permissive threshold so all
statistics can be saved, then the final grid explicitly retains adjusted
p < 0.05 before merging. The earlier small pilot did not use this final strict
output selection and stays separate. Species-matched RNA motif databases were
used; the historical mouse-gene filename alias also names human data in the
refinement harness, so the manifest, not the alias, establishes species.

These differences affect both cost and predictions. In particular, Pearson
linking is not an accelerated execution of the same GBM linking calculation.
The paper must not attribute the complete workflow ratios solely to language.
Exact source is retained under `evidence/workflow` and `evidence/recovery`.

## Timing, memory and summaries

The primary clock covers prepared-matrix loading, analysis and standard export.
Shared preparation, alignment, peak calling, motif-database construction,
process startup and final output hashing are excluded. Whole-process elapsed
values remain in the per-run table. Memory is the peak sampled sum of live
process-tree proportional set size, in decimal GB. Sampling was nominally one
second; observed intervals and process-read checks were audited. PSS is neither
requested RAM nor summed RSS, and sampling can miss brief peaks.

Headline speedup divides the reference median time by the Rust median time.
Memory reduction is one minus the ratio of median PSS. Both use three seeds.
Median within-pair ratios are also tabulated as a sensitivity; these estimators
must not be interchanged. Seed repeats describe computational variability and
are not independent biological samples. Stage costs are saved for all 42 runs;
unmatched stage labels should not be treated as identical calculations.

## Biological assessment

The prospective BMMC experiment used seven donors to fit networks and three
unseen donors for assessment (10,000 test cells). Features were selected using
the 20,000 training-cell pool: 5,000 genes, including 1,176 eligible TFs, and
20,000 regions. The 5,000-cell sensitivity reused that training vocabulary.
No test donor entered network fitting. Every method/size used three seeds.

The prespecified primary endpoint used the same TF-name intersection, a common
AUCell scorer and a fixed classifier. Network target sets can differ. Macro-F1
averages over cell types present in each donor; balanced accuracy is also
retained. Seeds are averaged within donors, then donors receive equal weight.
Three test donors are too few for precise population-level uncertainty.

Controls include training-fitted expression PCA50, common-TF expression, ten
within-donor label permutations and five signature-size-matched random gene
sets at the fixed primary seed. Secondary endpoints include all TF signatures,
ATAC activity, removal of TF self-targets, predeclared lineage markers and seed
stability. All-signature comparisons have unequal feature counts. The historical
`native` label in BMMC tables means all signatures with the common scorer, not
native activity calculation. Missing TFs are retained as coverage failures.

Post hoc inspection found that benchmark export merged positive and negative
TF-target programmes. The signed diagnostic scores each separately and takes
positive AUC minus negative AUC. This changes direction and group weighting;
it is not ordinary AUCell AUC and is not proof of activation or repression.
It leaves the inferred networks unchanged. Original union results are retained.

Frozen BMMC networks were then assessed on 12,315 unstimulated Kang2018 PBMCs
from eight separate donors, using 4,741 intersecting genes. One external donor
is held out for each classifier fold; scaling and classification use the other
seven. The classifier is StandardScaler plus balanced L2 logistic regression,
C=1, lbfgs, max_iter=2,000. Neither labels nor outcomes alter the networks.
Target coverage below 80% yields zero scores under both scorers without dropping
the TF column. Matched TF counts are 14, 21 and 20 for the 20,000-cell networks.

Intervals resample paired donor effects after seed averaging. No equivalence
margin was prespecified. The external cohort was independent of development
at its first signed-endpoint evaluation. Subsequent native tie correction and
confirmation reused inspected data and are reported as diagnostic/compatibility
checks, not as a second untouched biological validation.

Experimental RNA references are BEELINE mESC and hESC ChIP, mDC ChIP and mESC
loss/gain-of-function references. Original and motif-pruned networks use the
same candidate universe; missing edges score zero. Report average precision,
prevalence, retained-edge support and recall, and the absolute-Pearson baseline.
Unlisted edges are unlabelled. ChIP binding can be nonfunctional and perturbation
effects indirect. No functional enhancer-to-gene validation was performed.

## Native scoring correction

Legacy native AUCell breaks ties by gene-column index; pySCENIC's seeded path
first permutes columns. Optional `seed=` creates one NumPy RandomState gene
permutation. Dense and sparse Rust kernels use its priorities without shuffling
or densifying the expression matrix. Default callers retain legacy behaviour.
Gene axis, signatures, coverage, threshold and seed must also match.

The independently returned candidate source matches the integrated four core
files byte for byte. The Linux candidate identifies as development 0.5.1; the
build/native hash, not that reused version string, identifies the experiment.
It must receive a distinct release identity before public deposition. Public
0.5.1 is not evidence that this fix is already released.

All 12 external matrices reproduce, with maximum reference discrepancy below
3.31e-8. Atlas cost was assessed on 817,740 cells and three frozen networks,
with identical 4,096-cell scoring blocks. Numerical reference checking covers
10,000 returned cells per network. Full arrays remain on HPC with hashes.
The same-build seeded/unseeded control isolates tie ordering; candidate/v0.5.0
also includes intervening changes and is not an isolated tie-fix speedup.

## Directional activity confirmation

The fixed 8 November protocol was independently executed on HPC and returned
on 9 November. Each of six frozen BMMC networks per method was scored on the
same 12,315 Kang cells. Two columns per common TF retain positive and negative
target activity; absent directions are zero-filled. The contrast endpoint is
derived from these same arrays. All eight leave-one-donor-out folds, three
network seeds, both methods and two representations yield 192 fits. The
classifier and coverage threshold remain unchanged. Report macro-F1 and
balanced accuracy, averaging seeds within donors before paired bootstrap.
No noninferiority threshold or new biological model selection was introduced.

Six fresh scoring processes compare union and directional activity on the
same cached 817,740-cell RNA matrix and three frozen networks, on one node
with 16 allocated CPUs. Order alternates U/D, D/U, U/D. Both use 4,096-cell
batches and float32 outputs. Report scoring-only, I/O-inclusive, whole-process
and sampled PSS measures, all pairs and storage costs. A 10% aggregate cost
flag is operational, not a statistical test; retain the seed with 15.94% PSS
increase even though the median is below the flag. No network fitting, ATAC
imputation, motif analysis or full-workflow comparison was repeated.

All 96 return-manifest hashes and 192 confusion-derived fold metrics were
verified locally. Numerical controls are exact on all twelve external matrices
and three 10,000-cell atlas contrast samples. The full atlas score arrays
remain hash-listed on HPC. The correction is supported on reused cohorts;
independent execution does not make them untouched biological validation.

## Figure captions

**Figure 1. Configured workflow scaling.** Dots are individual seed runs;
lines connect medians of three seeds. Top: prepared-input loading through
analysis and export; bottom: peak sampled process-tree PSS. RNA compares with
pySCENIC; paired data compare with SCENIC+. All pairs use 16 CPUs and shared
prepared inputs. Hardware can differ between points. Algorithm differences
and selected feature panels are detailed above. No biological equivalence is implied.

**Figure 2. Biological utility under matched TF coverage.** Each point is one
donor's macro-F1 after averaging three inference seeds; lines pair methods for
the same donor. Bars mark the donor mean. Top: prespecified BMMC union endpoint,
three unseen donors. Middle: corrected native signed Rust scores versus
reference-scored SCENIC+ networks on eight Kang donors. Bottom: the independently
confirmed separate-direction representation on the
same Kang donors. Both later rows reuse the inspected cohort. Row endpoints and cohorts differ;
vertical comparisons do not measure an improvement from the fix. The signed
BMMC diagnostic and controls remain in supplementary tables.

**Figure 3. Experimental-reference recovery.** Bars show mean average precision
over three algorithm seeds; dotted lines show reference prevalence. Original
GRN, absolute-Pearson and motif-pruned results use the same candidate universe
within each reference. Weak absolute performance, unlabelled non-reference
edges and low retained-edge recall limit biological interpretation.
