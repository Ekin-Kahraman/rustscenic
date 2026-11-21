# Frozen confirmation protocol: GRNBoost2 stopping rule

12 November 2026. Written and hashed before any expression matrix, network or
score from the four confirmation datasets was computed. Development used only
hESC, mDC and mESC (BEELINE), BMMC, and Replogle K562/RPE1, all previously
inspected. Nothing here authorises sending, publishing or changing a default.

## Question

scikit-learn 1.3 changed `oob_improvement_` from a same-mask quantity (loss
before minus loss after a boosting round, on that round's out-of-bag rows) to a
cross-mask one (previous round's out-of-bag loss, on a different random mask,
minus this round's). arboreto stops when the mean of the last 25 entries is
negative. Development showed that under the cross-mask definition both pySCENIC
and RustScenic stop at a median of 25 to 29 trees, and that restoring the
same-mask definition trains about eight times longer, doubles seed stability
and improves recovery on three of four experimental references.

Does that hold on data not used to find it?

## Data

BEELINE scRNA-seq datasets never used in any RustScenic analysis: hHep, mHSC-E,
mHSC-GM, mHSC-L, from the checksum-pinned archives already in
`RustScenic-Biological-Validation-20261028/source-cache` (Zenodo 19009603,
Pratapa et al. 2020). Staging uses the 28 October `prepare_inputs.py` unchanged:
Bonferroni-significant varying genes, all eligible TFs plus up to 500 non-TFs by
variance, no reference links used in selection, source expression unchanged.

References: cell-type-specific ChIP-seq (HepG2 for hHep; mHSC for the three mHSC
lineages) is primary. Non-specific ChIP-seq and STRING are secondary. The three
mHSC lineages share one study and one reference, so they count as one family:
there are two independent families (hHep, mHSC).

## Arms

Seeds 777, 778, 779. 5,000-tree ceiling, learning rate 0.01, max_features 0.1,
subsample 0.9, depth 3, window 25, for every arm.

| Arm | Implementation | Stopping definition |
| --- | --- | --- |
| native_cross | RustScenic 0.6.1 kernel, `early_stop_mode="arboreto"` | cross-mask |
| native_same | same kernel plus `early_stop_mode="paired_oob"` (branch `fix/paired-oob-early-stop`) | same-mask |
| reference_cross | arboreto 0.1.6, scikit-learn 1.3.2 | cross-mask |
| reference_same | arboreto 0.1.6, scikit-learn 1.3.2, same-mask monitor (`reference_paired.py`) | same-mask |
| pearson | absolute Pearson correlation on the same matrix | none |

`reference_same` recomputes the same-mask improvement from the fitted stage and
applies arboreto's unchanged window rule. It emulates scikit-learn < 1.3; it is
not a run on an old scikit-learn installation, and is reported as such.

## Primary endpoints and decision rules

Metrics are those already in the manuscript (`reference_metrics.py`, 6 November
package): fixed candidate universe, pooled average precision, tie-aware top-50.

1. **Training length.** Mean fitted trees per target, same-mask over cross-mask.
   Confirmed if the ratio is at least 3 in all four datasets, for both
   implementations.
2. **Seed stability.** Mean per-TF Jaccard of the top-50 positive targets
   between seeds (three pairs). Confirmed if native_same exceeds native_cross in
   all four datasets and the mean ratio is at least 1.5.
3. **Implementation fidelity.** Top-50 agreement between RustScenic and arboreto
   (nine seed pairs) divided by agreement between arboreto seeds (three pairs),
   within each stopping definition. Confirmed if the ratio is at least 0.90 in
   all four datasets for both definitions.
4. **Reference recovery.** Pooled AP against the cell-type-specific ChIP-seq
   reference, mean of three seeds, native_same minus native_cross, with a 95%
   interval from 2,000 resamples of reference TFs (seed 20261112). Reported per
   dataset. "Improved" is claimed for a dataset only if its interval excludes
   zero. An overall improvement is claimed only if the difference is positive in
   both families (mHSC averaged over its three lineages). Otherwise the result
   is reported as no consistent change.

## Secondary, reported in full

Non-specific ChIP-seq and STRING recovery; top-50 fold over uniform; every arm
against Pearson; reference_same minus reference_cross; fitted-tree distributions;
wall time per fit; motif-supported regulon counts and regulon seed stability
through one common downstream recipe (`beeline_regulons.py`).

## Rules

No parameter, threshold, gene panel, metric or dataset changes after the hash.
All four datasets, all arms, all seeds and all adverse results are reported.
Failures to run are reported as failures, not dropped. Development and
confirmation results stay in separate tables. These are small differentiation
datasets: a confirmed result supports the stopping-rule finding at this scale
and does not by itself establish the effect at atlas scale or on perturbation
responses.
