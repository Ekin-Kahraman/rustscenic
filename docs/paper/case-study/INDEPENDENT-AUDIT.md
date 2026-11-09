# Independent Codex audit, 8 November 2026

**Verdict: passed for the stated post hoc lineage case study. No outstanding
implementation or numerical blocker was found.** This does not establish
general biological equivalence, causal-edge validity or paper acceptance.

The audit independently inspected the analysis code, source matrices, original
rank transport, donor metadata, figure and manuscript changes. It did not fit
networks or classifiers, run HPC jobs, change the engine or contact collaborators.

## Independent verification

- Verified all **150 unique source-file SHA256 hashes** in the final trace and
  case receipts, and all **40 recorded core-source hashes**. There is no engine
  diff from the already verified local integration.
- Recalculated **218 assessed donor AUROCs** from the original BMMC gene/region
  matrices and corrected external matrices using rank sums, independently of
  the analysis function. Verified **910 explicit missing-regulon rows** rather
  than treating missing predictions as zero-valued successful outputs.
- Independently decoded all **331 ranking blocks** with a separate bit-shift
  decoder and checked their original hashes. Recalculation over the full motif
  universe reproduced **all 60 regulator/run maximum NES values**. PAX5, EBF1
  and TCF7 peak at 2.587161, 2.760509 and 2.130392, below the fixed cutoff of 3.
- Independently recalculated all **1,800 training-topic/lineage AUROCs**. These
  use training cells only; best-of-30 selection is explicitly post hoc and is
  not presented as held-out performance or proof of a particular TF programme.
- Independently reproduced **270 random-target and self-removal donor AUROCs**
  after the correction below. Maximum discrepancy: **2.22e-16**.
- All **five focused tests pass**, including rank-cache corruption, missing
  outputs, donor alignment, rank cutoffs and exact AUC tie preservation.

The primary numerical findings reproduce: BMMC SPI1/GATA1 gene AUROCs
0.995057/0.995897, observed-region AUROCs 0.945154/0.954507 and external SPI1
union AUROC 0.982360. Corrected size-matched random-target means are
0.479593/0.509962. Removing the TF itself preserves the lineage signal.

## Issue found and fixed

The initial control scorer summed fractional per-target weights. Floating-point
roundoff could split theoretical AUC ties, changing a control AUROC by at most
0.0001072 in the independent comparison. The scorer now sums integer rank
contributions before dividing once per signature. An exact-tie regression test
was added, and all affected controls were recomputed. The primary saved activity
matrices and engine were unaffected. The biological interpretation is unchanged.

## Scientific interpretation and scope

Donors, rather than cells or algorithm seeds, are the biological replication
unit. Seeds are averaged within donors. Three BMMC donors are disjoint from
the seven network-training donors. External networks remain frozen. All five
predefined regulator checks and both training sizes remain available.

The report correctly distinguishes lineage discrimination from edge accuracy,
RNA-derived labels from independent functional validation, and size-matched
random controls from expression-matched controls. SCENIC+ means are explicitly
conditional on incomplete marker coverage across seeds. Missing markers remain
visible in the figure and tables. The known common-TF macro-F1 deficit and weak
experimental-edge recovery are retained. Shared missing regulators do not
explain that between-method deficit.

The trace establishes the filtering stage and numerical reason for exclusion;
it does not establish which upstream region-selection change would improve
biology. No thresholds were relaxed or favourable networks selected. The
training-topic diagnostic makes absence of the relevant cell populations an
unlikely simple explanation, without proving motif or causal-regulatory truth.

The figure is readable, uses the stated source tables and separates biological
utility from coverage failures. Its donor ranges are descriptive, not confidence
intervals. Full reanalysis requires the documented existing archive layout;
this small case-study directory is not a complete raw-data release. Dependency
pins and source hashes accompany the analysis. Permanent data archiving remains
a submission task already identified in the main paper readiness document.

## Performance decision

The measured dominant stages already run in Rust: approximately 99% of the
largest RNA workflow time is network inference, and 92% of the largest paired
workflow is network inference, topic fitting and region-to-gene linking.
Keeping these verified kernels unchanged is justified. This case study adds
no whole-workflow speed or memory claim, and the numerical control fix affects
only local analysis. No speculative Python-to-Rust port is warranted by this
audit.
