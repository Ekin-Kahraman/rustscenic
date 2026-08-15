# Validation

RustScenic validation tracks implementation agreement, runtime, memory and
real-data usability against established SCENIC ecosystem outputs where a fair
comparison is possible.

The standard is publication-minded: every serious claim should point to a
dataset, command, version, hardware context, runtime, memory measurement and
output sanity check.

## Credibility Snapshot

| Signal | Evidence |
| --- | --- |
| Released package | `v0.4.7` is the current GitHub release and PyPI package. |
| Controlled benchmark path | `validation/head_to_head/head_to_head_summary.json` records machine, seed, Python versions, runtime, peak RSS and output signatures. |
| Lab validation | Huang Lab collaborator artefacts include Kamath dopaminergic neurons and 10x human brain GEM-X multiome runs. |
| Full monolith real-data run | Human brain GEM-X `v0.4.6` artefact completed GRN, regulons, cisTarget, enhancer links and eRegulons on 8,215 post-QC cells and 123,089 peaks. |
| Biological sanity check | The full monolith human brain run recovered 16 of 17 expected brain TFs. |
| CI coverage | Audit, docs, release and nightly real-data validation workflows keep the public evidence path checked. |

## Headline Results

| Test | Result |
| --- | --- |
| AUCell vs pySCENIC on Ziegler 2021 airway atlas | Mean per-cell Pearson `0.984`; `91.7%` cells above `0.95`. |
| Canonical airway TF benchmark | RustScenic and pySCENIC-unit both recover `8/14`; same miss set. |
| cisTarget AUC kernel vs `ctxcore.recovery.aucs` | Pearson `1.0000`; mean absolute difference about `2.4e-5`. |
| PBMC3k GRN early stopping | 362,608 fitted trees versus arboreto's 363,178 (0.16% difference); median 26 and p95 30 match. On the fixed 337,414-edge three-way common universe, importance Spearman increased from 0.6146 in legacy mode to 0.6363. |
| PBMC3k activator/repressor split | 2,000 seeded sampled edges matched SciPy polarity classifications exactly; maximum absolute rho difference `5.05e-08`. |
| Human brain GEM-X SCENIC+ comparison | Region-to-gene edge-set Jaccard `1.000`; region AUCell mean Pearson `0.823`; gene AUCell and eRegulon-edge parity remain weaker. |
| Real multiome pipeline runs | PBMC 3k, mouse brain E18 5k, PBMC granulocyte 10k. |
| Local unit/integration suite | 223 tests passed, 1 skipped in the 2026-05-24 audit. |

## External Validation

External reports are useful adoption evidence, but they are not used for the
headline speedup claim unless they include the same benchmark controls:
hardware, command, version, runtime, memory and output signatures.

| Tier | Dataset | Source | Evidence | Caveat |
| --- | --- | --- | --- | --- |
| Committed collaborator adoption artefact | Kamath et al. 2022 midbrain dopaminergic neurons | [issue #68](https://github.com/Ekin-Kahraman/rustscenic/issues/68), [PR #71](https://github.com/Ekin-Kahraman/rustscenic/pull/71), [JSON](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/community/kamath_da_grn.json) | RustScenic `0.4.0` on Google Colab completed GRN plus cisTarget: 266,805 GRN edges, 9 regulons, 174,019 cisTarget rows, 9 of 9 expected DA-neuron TFs recovered. | Not a full multiome E2E run. AUCell, enhancer links and eRegulons were out of scope; 3 of 9 regulons had low expression-matrix gene overlap. |
| Committed collaborator adoption artefact | 10x Multiome GEM-X 10k human brain, full monolith run | [issue #80](https://github.com/Ekin-Kahraman/rustscenic/issues/80), [JSON](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/community/human_brain_10k_v0.4.6.json) | RustScenic `0.4.6` completed GRN, regulons, cisTarget, enhancer links and eRegulons on 8,215 post-QC cells and 123,089 peaks: 4,314,539 GRN edges, 108,736 cisTarget rows, 927,002 enhancer links, 16 of 17 expected brain TFs recovered, peak RSS 24.99 GB, total pipeline runtime 54.9 min. | Collaborator real-data run, not a SCENIC+ head-to-head row. Used preprocessed ATAC `.h5ad`; `fragments_to_matrix` was skipped. Microglial cells were filtered before analysis. |
| Committed collaborator adoption artefact | 10x Multiome GEM-X 10k human brain | [issue #70](https://github.com/Ekin-Kahraman/rustscenic/issues/70), [PR #74](https://github.com/Ekin-Kahraman/rustscenic/pull/74), [JSON](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/community/human_brain_10k_v0.4.1.json) | RustScenic `0.4.1` completed GRN, AUCell and topics on 8,215 post-QC cells and 123,089 peaks: 4,293,902 GRN edges, 1,748 regulons, peak RSS 9.08 GB. | cisTarget, enhancer links and eRegulons were not run. Biological sanity is a top-regulon signal after immune-cell subsetting, not full cell-type enrichment. |
| Issue-linked report | 10x lymphoma 14k | [issue #69](https://github.com/Ekin-Kahraman/rustscenic/issues/69) | RustScenic `0.4.1` completed GRN, AUCell and topics on 14,039 post-QC cells; review notes 1,663 regulons and B-cell regulators including `POU2F2`, `PAX5`, `MEF2B`, `SPIB`, `EBF1` and `BCL11A`. | JSON is attached to the issue but not committed in-repo. Low ARI is treated as expected for a mostly homogeneous sample, so this is adoption evidence only. |
| Issue-linked report | Human cortex 44k | [issue #85](https://github.com/Ekin-Kahraman/rustscenic/issues/85) | RustScenic `0.4.7` completed the reported pipeline on 44,222 RNA/ATAC cells in 6,651 s with 16.8 GB maximum RSS. | Not a scaling baseline: one non-MPI process was allocated across six hosts and exceeded the per-host thread allocation. Zero eRegulons is incomplete biological output and needs a correctly packed rerun. |
| Issue-linked report | Brain and breast preprocessing | [issue #94](https://github.com/Ekin-Kahraman/rustscenic/issues/94) | RustScenic `0.4.7` completed two 2-CPU preprocessing runs in 227.87 s and 598.75 s at 10.0 GB peak RSS. | Execution evidence only. The brain matrix uses raw observed barcodes and the 1.06-million-peak breast result lacks FRiP, TSS-enrichment or reference-overlap checks. |

These rows show the package running outside the maintainer benchmark path. The
controlled head-to-head scripts and saved validation artefacts remain the source
for public performance claims.

## Validation Notes

- GRN edge rankings are not expected to be bit-identical to arboreto because the
  implementation uses an independent histogram-GBM path.
- Downstream cell-level AUCell agreement is stronger than fine-grained GRN edge
  agreement.
- Some real-data biological checks currently use expected TF recovery by name;
  cell-type enrichment checks are part of the next validation tier.
- The next benchmark tier adds more real multiome datasets, repeated runs and a
  second machine.

## Where To Look

- [Benchmarks](benchmarks.md)
- [validation/VALIDATION_SUMMARY.md](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/VALIDATION_SUMMARY.md)
- [validation/ziegler_headtohead_2026-04-19.md](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/ziegler_headtohead_2026-04-19.md)
- [Issue #95 PBMC3k early-stop record](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/parity_v0310/ISSUE_95_EARLY_STOP_2026-08-14.md)
- [Issue #96 PBMC3k polarity record](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/parity_v0310/ISSUE_96_POLARITY_2026-08-14.md)
- [validation/community/](https://github.com/Ekin-Kahraman/rustscenic/tree/main/validation/community)
- [Issue #94/#85/#69 evidence record](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/community/ISSUE_EVIDENCE_94_85_69_2026-08-14.md)
- [validation/scaling/](https://github.com/Ekin-Kahraman/rustscenic/tree/main/validation/scaling)
- [docs/v0.4.x-benchmark-plan.md](https://github.com/Ekin-Kahraman/rustscenic/blob/main/docs/v0.4.x-benchmark-plan.md)
