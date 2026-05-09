# Evaluation Metrics

This page gives collaborators a shared way to report clustering metrics from
rustscenic outputs.

## ARI

Adjusted Rand Index (ARI) compares two cluster label vectors over the same
cells. In rustscenic validation reports, state exactly which two labels were
compared. Common examples:

- AUCell/regulon-activity Leiden clusters vs manual cell-type labels.
- AUCell/regulon-activity Leiden clusters vs ATAC topic Leiden clusters.
- Topic assignments vs external cell-type labels.

The number is only interpretable with that comparator. For example, an ARI
drop after subsetting can mean the removed cells carried a strong cluster
boundary, or it can mean the remaining subset has less internal structure.

```python
from sklearn.metrics import adjusted_rand_score

# Both Series must be indexed by the same cell barcodes.
left = aucell_clusters.loc[shared_cells]
right = reference_labels.loc[shared_cells]

ari = adjusted_rand_score(left, right)
print(f"ARI: {ari:.3f}")
```

Report the comparator in plain text, for example:

```text
ARI was computed between AUCell Leiden clusters and ATAC Leiden clusters
on the 8,215 cells retained after immune subsetting.
```
