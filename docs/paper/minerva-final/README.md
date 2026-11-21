# Final Minerva return audit, 12 November 2026

**The planned run completed successfully. The evidence supports an atlas-scale software paper with explicit biological limitations; it does not establish high causal-edge accuracy, biological equivalence or superior networks. No broad HPC rerun is indicated by this return.**

## Integrity and reproducibility

- The collaborator returned the files on 12 November 2026. The collector finished at 17:34 UTC on 11 November.
- `SEND_BACK.tar.gz`: 61,152,003 bytes; SHA-256 `ac9be5f73e3843617a5bc58ad9a0c6fa1f70b89efb42886d329c1af6e2a3f643`. It matches the checksum the collaborator reported with the return. The 197 tar entries were checked for unsafe paths and links before extraction.
- The returned package manifest and source/protocol files match the exact sent package. All six worker jobs and all 13 monitored subprocesses completed with exit code zero; all six LSF worker logs confirm completion.
- The official Linux v0.6.1 native extension hash matches the returned environments. All six holdout prediction receipts match their seed, training matrix, gene panel, protocol and prediction files.
- Local replay using the unchanged sent analysis code reproduced 4,248 K562 ranking rows, the macro summaries and TF-bootstrap intervals, and the earlier erythroid RNA/ATAC and GATA1 analyses. In total, 28 regenerated data/summary files matched (numeric tolerance 1e-10 relative/1e-12 absolute; label arrays exact). Machine name and measured wall times are excluded from the biological replay comparison.
- 38 output hash checks passed. The HPC audit verified all 42 previously retained activity arrays. This is evidence that they were intact on HPC at audit time, not that they are publicly deposited or copied locally.
- Sixteen full-output entries are intentionally absent from this compact return. Their sizes and hashes are preserved in `audit-receipt.json`; they still require durable retention or a documented regeneration route.

## Release scaling

All three runs used seed 777 and 16 CPU slots. GB means decimal GB of sampled process-tree PSS. These are release confirmations, not a new replicated head-to-head speed benchmark.

| v0.6.1 run | Scope | Time | Peak PSS |
| --- | --- | ---: | ---: |
| RNA, 1,306,127 cells × 3,000 genes, 256 TFs | Prepared-input loading through normal workflow export | 40.78 min | 4.66 GB |
| Paired, 817,740 cells × 5,000 genes × 20,000 regions, 256 TFs | Prepared-input loading through directional workflow export | 104.35 min | 44.00 GB |
| RNA, same cells × 6,000 genes, same 256 TFs | GRN inference plus correlation only | 119.55 min | 44.70 GB whole process |

The wide run's whole process, including preparation and diagnostic exports, took 123.07 min. GRN inference alone increased from 40.37 to 118.46 min, a 2.935-fold increase; its sampled stage peak rose from 4.42 to 40.94 GB. The wider feature set therefore has a substantial memory cost. It does not establish unrestricted whole-transcriptome scaling. The paired release uses disk-backed imputation; its 44 GB figure is not a direct, controlled improvement estimate against the old 149.88 GB median. The historical v0.5.0 comparator grid is labelled separately.

## Independent K562 functional assessment

The frozen protocol trained both methods only on 9,480 negative-control cells, using 5,000 genes and 1,204 TF predictors, with three seeds. Evaluation used held-out controls and single-TF CRISPRa cells from the Norman2019 study, obtained from the pinned [scPerturb deposit](https://zenodo.org/records/7041849). Of the 41 targeted TFs, 40 met the capture-half cell requirements; HES7 did not. Seven evaluable TFs were outside the predictor panel and retained as zero-prediction cases.

The primary endpoint is equal-TF macro average precision (AP) at the frozen 0.25 log2(CPM+1) effect threshold, with seeds averaged within TF. **AP is a ranking measure, not the percentage of biologically correct edges.** Response labels indicate concordant changes across technical capture halves, including indirect responses. They are neither significance-tested effects nor complete causal ground truth.

| Ranking | Macro AP | Expected response hits in top 50 |
| --- | ---: | ---: |
| RustScenic final regulons, primary | 0.149289 | 7.711 |
| Reference final regulons | 0.149576 | 7.972 |
| Uninformative tied-score baseline | 0.148976 | 7.449 |
| Full absolute Pearson | 0.164303 | 10.599 |
| RustScenic raw importance | 0.152788 | 9.224 |
| RustScenic raw candidates reranked by absolute Pearson | 0.153795 | 10.340 |

RustScenic's final AP difference from the reference is -0.000286 (conditional TF-bootstrap 95% interval -0.000874 to +0.000074). Against the uninformative baseline it is +0.000313 (+0.000085 to +0.000575), a very small absolute gain. Against full Pearson it is -0.015014 (-0.029341 to -0.004820). These intervals describe variation among assayed TFs in this one cell-line screen, not donor uncertainty. A small difference from the reference does not establish equivalence.

Coverage is a central limitation: raw predictions cover 33/40 evaluable TFs in both methods, but final eligible targets survive for only 7–8/40 RustScenic TFs and 8–11/40 reference TFs per seed. RustScenic emits a mean 2.475 positive predictions and 0.883 response hits within the top-50 budget; most of the reported 7.711 expected hits arise from fractional credit among tied, unpredicted genes. Expected top-50 hits must not be presented as 7.711 actual nominated hits.

Negative-control construct contrasts label 549–821 of approximately 12,000 eligible full-universe genes at the primary threshold (4.59–6.87%). This shows appreciable control/technical variation and limits mechanistic interpretation; it is not a calibrated false-discovery rate. At stricter thresholds, control response counts fall, but the final-network AP remains below full Pearson. All thresholds, seeds, per-TF results, coverage and null contrasts are retained. The previously defined shared-final-TF subset is secondary and varies in membership across seeds.

The earlier erythroid transfer and GATA1 PRO-seq findings also reproduced. Taken together, the data support useful lineage activity and weak recovery of perturbation-responsive targets by the final networks. They do not justify claiming an accuracy improvement or changing a default on this held-out test. Raw Pearson reranking remains a secondary result, not a validated new default.

## Interpretation

Both favourable and adverse endpoints are reported, and release confirmation is kept separate from the older replicated comparator grid. The hash-listed full outputs remained on Minerva scratch storage at audit time and are not part of this compact subset; the software DOI does not archive this benchmark.

More cells alone will not solve final-network coverage or target ranking. If improved target prediction becomes a separate objective, changes would need to be developed on designated development data and then tested on a fresh untouched cohort. This K562 outcome has now been inspected and cannot serve as an untouched test for subsequent tuning. No acceptance or publication-standard guarantee follows from a completed benchmark.

## Audit artefacts

The original return, the safely extracted files, `audit.py` and the replayed outputs are kept in the author's local audit directory and are not published. This directory holds a copy of `audit-receipt.json` and a hash-listed compact evidence subset (`source-manifest.json`). The replay ran `audit.py` in the local Python environment that carried the reference packages of the final Minerva package; no model fitting is repeated. The exact inputs and analysis code of the sent package are not included here. The LSF submission record (`returned/submission.json`) is omitted from this public copy because it names the HPC allocation account; its SHA-256 remains listed in `source-manifest.json`.
