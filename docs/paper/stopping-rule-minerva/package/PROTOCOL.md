# Protocol: GRNBoost2 stopping rule, replication at scale on Minerva

13 November 2026. Fixed before this package ran on Minerva. `SHA256SUMS`
freezes this file with the code and inputs, and every arm records its hash.

A first version was written earlier the same day and never sent. This version
adds the two headline-scale runs, the reference-recovery endpoint, the Pearson
baseline and the return of complete networks. The twelve confirmatory checks
are unchanged. No Minerva result existed at either point.

## Question

scikit-learn 1.3 changed `oob_improvement_` in gradient boosting. Up to 1.2.2
entry *i* was the loss before round *i* minus the loss after it, both on round
*i*'s out-of-bag cells (same mask). From 1.3 it is the previous round's
out-of-bag loss, measured on a different random 10% of cells, minus this
round's (cross mask). arboreto stops a target when the mean of the last 25
entries is negative. Under the cross-mask definition that mean compares two
unrelated subsets, and on every dataset examined GRNBoost2 stopped at a median
of 25 to 30 trees per target, against a ceiling of 5,000. pySCENIC is affected
on any current installation, and RustScenic 0.5.0 onwards reproduces the
current behaviour by design.

On 12 November this was tested locally on three development datasets and four
previously unused BEELINE datasets (about 400 to 1,100 cells). Restoring the
same-mask rule trained 6 to 8 times longer in both tools and raised the
seed-to-seed overlap of each TF's top 50 targets 1.5 to 2.1 times. Under both
rules RustScenic agreed with pySCENIC at 94 to 99% of pySCENIC's own
seed-to-seed agreement. Recovery of cell-type ChIP-seq links did not change
detectably. Recovery of non-specific ChIP-seq and STRING links rose in all
eight dataset-reference pairs of the confirmation, a secondary endpoint
reported without intervals.

This package asks whether that holds:

1. at realistic scale (20,000 to 1.3 million cells);
2. on HPC, executed by someone other than the author;
3. against pySCENIC on a real scikit-learn 1.2.2, where the 12 November work
   used a patched monitor on 1.3.2;
4. for recovery of broad reference links, now with intervals and rules fixed
   in advance;

and what the original rule costs in time and memory, including on the two
benchmarks the manuscript leads with.

## Data

**Bone marrow** (GSE194122, bundled in `inputs/bmmc`). 20,000 training cells
from seven donors, 10,000 evaluation cells from three other donors, 5,000 genes
(all 1,069 expressed listed TFs, then the highest-variance genes). The split
and panel are those staged on 12 November by `stage_bmmc.py` and are unchanged.
This is development data: a 5,000-cell subset of these training cells was
fitted under both rules on 12 November, and the 20,000 cells were fitted under
the current rule once. No 20,000-cell network under the original rule existed
when this protocol was fixed, apart from the resource pilot below.

**Mouse brain atlas** (10x E18, 1,306,127 cells, 3,000 genes, 256 TFs; on
Minerva at `recovery/work/prepared/atlas`, verified against
`resources/expected-atlas.json`). It has been used only for speed and memory
under the current rule. No network from it has been fitted under the original
rule or scored against a reference, and the author holds no copy of the
expression data. Subsets are the first 20,000, 100,000 and 500,000 cells of the
fixed seeded cell order, plus all cells, so they are nested and count as one
data source.

**HDMA RNA** (Human Development Multiomic Atlas, 817,740 cells, 5,000 genes,
256 TFs; on Minerva at `recovery/work/prepared/hdma`, verified against
`resources/expected-hdma.json`). This is the RNA matrix of the paired benchmark
run on 11 November. Used here only to time the gene-network stage.

**Reference networks** (`resources/references`, from the checksum-pinned
BEELINE archive used since 28 October). Non-specific ChIP-seq and STRING, human
and mouse, upper-cased with self-links and duplicates removed. Nothing was
filtered by any expression value or result. Coverage, computed from gene names
alone: bone marrow 463 TFs and 38,664 links (ChIP-seq), 483 TFs and 16,696
links (STRING); atlas 128 TFs and 2,144 links, 111 TFs and 794 links. STRING is
a functional-association network, so these are broad references and not
cell-type-specific ground truth.

Genes that are constant within a subset are removed for every arm alike and
counted in each arm's summary.

## Arms

Seeds 777, 778 and 779. Every fitted arm uses a 5,000-tree ceiling, learning
rate 0.01, max_features 0.1, subsample 0.9, depth 3 and a 25-round window. One
arm per process, 16 CPU slots.

| Arm | Implementation | Stopping definition |
| --- | --- | --- |
| `native_cross` | bundled RustScenic build, `early_stop_mode="arboreto"` | cross mask (current) |
| `native_same` | bundled RustScenic build, `early_stop_mode="paired_oob"` | same mask (original) |
| `reference_cross` | arboreto 0.1.6 on the installed scikit-learn 1.3.2 | cross mask (current) |
| `reference_original` | arboreto 0.1.6 on the bundled scikit-learn 1.2.2 | same mask (original) |
| `pearson` | absolute Pearson correlation of every TF-target pair | none (baseline) |

| Setting | Arms | Seeds |
| --- | --- | --- |
| Bone marrow, 20,000 cells | all four, then regulons and lineage scores | 3 |
| Atlas, 20,000 cells | all four | 3 |
| Atlas, 100,000 cells | RustScenic both rules; pySCENIC both rules for seed 777 | 3 |
| Atlas, 500,000 cells | RustScenic both rules | 1 |
| Atlas, 1,306,127 cells | RustScenic both rules | 1 |
| HDMA RNA, 817,740 cells | RustScenic both rules | 1 |

The Pearson baseline is computed once per setting, in the seed-777 job.

The bundled RustScenic build is v0.6.1 plus `resources/KERNEL-PATCH.diff`. Its
default is unchanged. `resources/BUILD.json` records how it was built.

For bone marrow each network is then turned into motif-supported regulons with
the unchanged 12 November code: RustScenic networks with `recipe.py`
(`original`, and `diverse_importance` whose module families mirror pySCENIC's),
pySCENIC networks with pySCENIC's own module and pruning functions
(`reference.py`).

Importances are stored, ranked and scored in single precision, so every
reported number can be recomputed exactly from the returned files.

## Confirmatory checks

Evaluated separately for bone marrow and for the 20,000-cell atlas subset, the
two settings where all four arms run with three seeds. Pass marks are those
fixed on 12 November for the BEELINE confirmation. Checks 1 and 2 are applied
here to both tools.

1. **Training length.** Mean fitted trees per target, original rule over
   current rule. Pass: at least 3, for RustScenic and for pySCENIC.
2. **Seed stability.** For each TF, the Jaccard overlap of its top 50 targets
   by positive importance between two seeds, averaged over TFs and over the
   three seed pairs. Ratio of original rule over current rule. Pass: at least
   1.5, for RustScenic and for pySCENIC.
3. **Fidelity.** The same overlap between a RustScenic network and a pySCENIC
   network (nine seed pairs), divided by the overlap between pySCENIC seeds
   (three pairs). Pass: at least 0.90 under the current rule and under the
   original rule.

That is twelve checks: two settings, times two tools for checks 1 and 2 and two
rules for check 3. Each is reported as met, not met, or not evaluable. A check
is not evaluable when an arm it needs did not complete, and that is never
counted as a pass.

## Reference recovery

Metric: the manuscript's pooled, tie-aware average precision on a fixed
candidate universe (`recovery.py`, which reproduces the 12 November tables to
rounding error). For bone marrow and the 20,000-cell atlas subset, and for each
of the two references, the quantity is RustScenic's average precision under the
original rule minus under the current rule, seeds averaged, with a 95% interval
from 2,000 resamples of reference TFs (seed 20261113).

Two statements are allowed, each decided by a rule fixed here:

- **"No loss of recovery at scale"** if none of the four intervals lies
  entirely below zero.
- **"The original rule improves recovery of broad reference links at scale"**
  only if all four intervals lie entirely above zero.

If neither holds, that is what is reported. The same differences for pySCENIC,
each tool against the Pearson baseline, RustScenic against pySCENIC, and point
estimates at the larger sizes are reported in full without a rule.

## Descriptive endpoints

Reported in full, with no pass mark.

- **Scale.** Trees, time and memory for RustScenic under both rules at 20,000,
  100,000, 500,000 and 1,306,127 atlas cells and at 817,740 HDMA cells, and
  seed stability at 100,000. At 1,306,127 cells the current rule averaged 48.5
  trees (median 27) on 11 November, so both the truncation and the benefit of
  the original rule may change with cell number. Whatever is measured is
  reported.
- **The two headline benchmarks.** On 11 November gene-network inference took
  2,422 of the 2,441 seconds of the RNA workflow at 1,306,127 cells, and 2,368
  of the 6,205 seconds of the paired workflow at 817,740 cells. No other stage
  takes the stopping rule as a setting. The two single-seed runs here
  therefore give the cost of each workflow under the original rule as the
  measured gene-network time plus the remaining stages as timed on
  11 November. They are reported as that sum, not as a repeat of the full
  workflows.
- **Speed against pySCENIC under the original rule.** Same-node inference time
  and peak memory at 20,000 cells (two datasets, three seeds) and at 100,000
  cells (one seed). If pySCENIC does not finish 100,000 cells within its
  72-hour limit, that is the result.
- **Regulon stability** (bone marrow). Regulons per seed, regulons found in all
  three seeds, and the overlap of target sets for regulons shared by two seeds.
- **Lineage regulators** (bone marrow). For the five primary regulators (SPI1,
  GATA1, PAX5, EBF1, TCF7) and 29 extended ones in `lineage_regulators.json`,
  written on 12 November: whether an activating regulon exists in each seed,
  and the held-out-donor AUROC of its activity for the expected lineage,
  alongside the regulator's own expression and size-matched random gene sets.
  Expectation, stated in advance and not a pass mark: under the original rule
  RustScenic yields PAX5 and EBF1 activating regulons in at least two of three
  seeds. Development at 5,000 cells, current rule then original rule, seeds
  with an activating regulon out of three: PAX5 0 and 3, EBF1 3 and 3, GATA1 3
  and 3, SPI1 3 and 2, TCF7 1 and 1.

## What the results can and cannot support

If the twelve checks are met: both tools are truncated at scale on a current
installation; the original rule restores longer training and more reproducible
networks; and RustScenic agrees with pySCENIC on a real scikit-learn 1.2.2 at
90% or more of pySCENIC's own seed-to-seed agreement.

The reference-recovery statements are limited to broad references in two
datasets. Nothing here measures accuracy against cell-type-specific
experimental ground truth at scale. The lineage endpoint is descriptive and
comes from one tissue and one study. Seeds are not biological replicates. The
two confirmatory settings are two datasets, not a sample of datasets.

Any check that is not met is reported as not met. Arms, data, settings, pass
marks and the two recovery rules are not changed after results are seen.
Additional analyses are labelled as post hoc.

## Fallbacks and limits fixed in advance

- **scikit-learn 1.2.2 cannot be loaded on Minerva.** The launcher tests this
  on the login node. If it fails, `reference_original` runs arboreto on
  scikit-learn 1.3.2 with the same-mask monitor in `reference_paired.py`, and
  every output and the verdict say so. On 13 November that monitor was
  compared with a real scikit-learn 1.2.2 on seven BEELINE datasets and three
  seeds each. All 21 pairs of networks were identical: the same tree count for
  every target and no difference in any importance.
- **Time limits.** Each arm has its own limit (hours, by job: see `PLAN` in
  `orchestrate.py`). An arm that exceeds it is recorded as a timeout, the
  remaining arms still run, and checks that need it become not evaluable.
- **Failures after the fit.** The network is written as soon as inference
  ends. Correlation, recovery scoring, regulons and lineage scores are each
  guarded: an error there is recorded and the fitted network is kept and
  returned, so the stage can be repeated from it without refitting.
- **Interrupted jobs.** `retry.sh` resubmits only jobs with an arm that never
  finished fitting. Finished arms are kept. Seeds and settings do not change.
- **Atlas or HDMA inputs missing or changed.** Jobs that need them stop with
  that message. The others are unaffected.

## What was looked at before this was fixed

- The 12 November local results summarised above.
- The load test in `preflight.py` (600 synthetic cells).
- A resource pilot to set time limits. 32 bone-marrow targets chosen at random
  (seed 20261113) were fitted at 20,000 cells with arboreto on scikit-learn
  1.2.2, recording only trees and seconds per target: mean 326 trees, 90th
  percentile 481, maximum 672. No stability, fidelity, recovery, regulon or
  lineage quantity was computed. From this and the 12 November Minerva
  timings, the pySCENIC original-rule arm on bone marrow is expected to take
  roughly 12 hours on 16 slots; its limit is 35 hours.
- Reference coverage, from gene names alone (see Data).
- Rehearsals of the whole package on reduced inputs (1,500 bone-marrow cells
  and 400 genes; synthetic 60-gene matrices in place of the atlas and HDMA).
  Their numbers are not results and are not reported.
- A run of the save, recovery, regulon, lineage and bootstrap code on the one
  existing 20,000-cell bone-marrow network (RustScenic, current rule, seed 777,
  fitted on 12 November), to confirm those stages complete at full size
  (`stage_test_fullsize.py`, kept outside the package). Only run times, row
  counts and file sizes were printed. Its recovery values were not printed or
  saved. Its regulon and lineage results were already known from 12 November.

## Returned files

`SEND_BACK.tar.gz` holds, for every arm: the complete network with importance
and correlation, the summary, resource monitor, fitted trees per target, the
top 100 targets per TF, recovery scores, regulons and lineage scores (bone
marrow), logs, and the tables and `VERDICT.json` computed by `summarise.py`.
Every reported number can be recomputed from it, and any further analysis of
the networks can be done without returning to Minerva.
