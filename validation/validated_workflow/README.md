# Audited workflow verification

The public `rustscenic validated` command connects the frozen native RNA and
paired recipes to the verified seeded scorer. The generic pipeline keeps its
defaults; its seed can now be set explicitly, and unsupported fallback is labelled.

## Verified on existing real data

- Nine RNA replays: hESC, mESC and mDC, seeds 777/778/779. Reused the returned
  GRNs; motif refinement reproduced all final signatures exactly. Corrected
  scores match direct seeded scoring exactly after aligning signature names.
- Fresh RNA CLI: 758 hESC cells, 910 genes, 410 requested TFs, 55 supported
  regulons. All inference and downstream stages ran from input files.
- Paired replay: 5,000 BMMC cells. All 217 gene and region signatures,
  13,213 eRegulon edges, 22,513 peak-to-gene links, 1,035,908 motif-support rows
  and 3,560 motif scores match the returned run. Only topics and GRN were reused.
- Fresh paired CLI: the first 256 real BMMC training cells, 1,000 genes and
  5,000 peaks, with 121 selected TFs. All native stages ran without cached
  fits, retaining 25 TF signatures with finite, nonzero gene and region activity.
- Bounded imputation: 20,000 real cell-topic rows and 1,000 real topic-peak
  columns reproduce the original integer accessibility values exactly across
  the 8,192-row batch boundary.

These are compatibility and usability checks. The fresh subsets are not an
additional atlas benchmark, and replay is not an independent biological cohort.
The existing biological results and macro-F1 limitation retain their scope.

## Reproduce the RNA audit

Install the candidate wheel or source distribution in a clean environment.
Use the existing paper-validation input package and Huangfu's returned results:

```bash
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python \
  validation/validated_workflow/check_rna.py \
  --package /path/to/rustscenic-paper-validation \
  --returned /path/to/returned/results \
  --output /path/to/new-audit-directory
```

`receipt.json` records input hashes, imported module location, replay checks and
the fresh CLI command. The broader suite covers invalid inputs, rank-universe
metadata, motif failure, aliases, signed assembly, seeded batching and scratch
cleanup. Independent review is recorded with the local candidate handoff.
