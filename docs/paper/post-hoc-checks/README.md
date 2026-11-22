# Post hoc checks added after internal review (22 November 2026)

These checks were added after every result in the manuscript was known, in response
to an internal review of the draft. They are descriptive. Each reads only saved
networks and was written separately from the analysis code that produced the
reported results. They are reported in the Supplementary results ("Post hoc checks
added after internal review", Supplementary Table S15) and cited in the main text.

| Script | Input | Output | Reported as |
| --- | --- | --- | --- |
| `grn_only_fidelity.py` | the 26 October GRN-only runs (RustScenic 0.5.0 and arboreto 0.1.6, 1,306,127 cells, 2,095 genes, 256 TFs, seeds 777 to 779) | `grn_only_fidelity.out` | fidelity 0.878 of the benchmarked engine at atlas scale |
| `rules_concordance.py` | the returned 20,000-cell test archive (`jobs/`), BMMC and atlas subset, both tools, both rules, three seeds | `rules_concordance.out` | tree-count correlations; regulon concordance between rules; lineage AUROC by rule |

The bone-marrow exact-bin fidelity values (0.966 and 0.985) that Figure 3 shows are
saved separately in
[`../stopping-rule-minerva/diagnosis/fidelity-bmmc-exact.txt`](../stopping-rule-minerva/diagnosis/fidelity-bmmc-exact.txt),
and the original-rule check of the published 0.7.0 wheel is in
[`../stopping-rule-minerva/diagnosis/published-wheel-check/`](../stopping-rule-minerva/diagnosis/published-wheel-check/README.md).

## Commands

```bash
python3 -I grn_only_fidelity.py <GRN-only results>/runs > grn_only_fidelity.out
python3 -I rules_concordance.py <returned 20,000-cell archive>/jobs > rules_concordance.out
```

The fidelity metric matches the protocol's: the per-TF Jaccard index of the 50
highest positive-importance targets (self-edges removed, importances cast to
float32, ties broken by target name), averaged over the nine between-tool seed
pairs and divided by the mean over the three within-reference seed pairs.

Tree-count correlations are Spearman correlations: between seeds of the same arm,
and, for seed 777, between a target's fitted tree count and its largest absolute
Pearson correlation with any TF in the Pearson arm. Regulon consensus is presence in
all three seeds; identity overlap is the Jaccard index of regulon names, and target
overlap the mean Jaccard index of target sets over regulons present in both networks.
Lineage AUROCs are the returned `lineage.csv` values, paired by regulator and seed
where both rules produced the regulon.
