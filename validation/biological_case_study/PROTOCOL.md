# Existing-output biological case study, 8 November 2026

This is a post hoc diagnostic and illustrative case study, not a new untouched
validation cohort or a replacement for the paper's original endpoints.

1. Trace all five previously specified regulators (SPI1, GATA1, PAX5, EBF1,
   TCF7), both methods, both training sizes and all three seeds. Reconstruct
   the exact top-1,000 region sets from saved topic weights. Verify every
   decoded ranking block; preserve original ranks, the full motif universe,
   AUC fraction 0.005 and NES threshold 3. Recompute enrichment independently
   and compare with the returned scores. Retain absent regulators as failures
   of coverage. Do not relax thresholds to recover selected markers.
2. Reuse saved activity matrices and cell labels. Assess the original five
   lineage programmes in the three BMMC test donors, with no fitting on those
   donors. Report gene and observed-region activity separately, direction,
   effect sizes and donor variation. Reuse external Kang scores for the
   applicable programmes; erythroid GATA1 cannot be evaluated in that cohort.
   Show all seeds, feature availability and failures. Aggregate seeds within
   donors; cells and algorithm seeds are not independent donor replicates.
3. Include expression and size-matched random-target controls where available
   or cheaply reconstructable from the fixed signatures. No classifier,
   network or topic model is refitted. Biological interpretation is supported
   by independently cited literature, not by treating SCENIC+ as ground truth.
   A lineage result does not validate individual causal TF/region/gene links
   or a disease response.
4. Preserve the original benchmarks and inputs. Fix only verified defects;
   measure any runtime changes on identical inputs and check output parity.
   Do not move computation to Rust solely because its wrapper is Python.
   Finish with an independent agent audit before delivery. No messages,
   public repository updates or HPC jobs are authorised by this analysis.
