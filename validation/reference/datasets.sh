#!/usr/bin/env bash
# Fetch reference datasets used by the audit.
set -euo pipefail

mkdir -p /data

# PBMC 3k via scanpy
python -c "
import scanpy as sc
adata = sc.datasets.pbmc3k()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.write_h5ad('/data/pbmc3k.h5ad')
print('pbmc3k:', adata.shape)
"

# aertslab TF list
curl -sL https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt -o /data/allTFs_hg38.txt
echo "TFs: $(wc -l < /data/allTFs_hg38.txt)"
