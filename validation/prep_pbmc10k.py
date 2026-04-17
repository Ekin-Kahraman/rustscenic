"""Preprocess PBMC-10k for GRN inference."""
from pathlib import Path
import scanpy as sc

src = Path("/Users/ekin/rustscenic/validation/reference/data/pbmc10k/filtered_feature_bc_matrix")
adata = sc.read_10x_mtx(src, var_names="gene_symbols", make_unique=True)
print(f"raw: {adata.shape}")

# standard QC + norm
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"post-QC: {adata.shape}")

out = Path("/Users/ekin/rustscenic/validation/reference/data/pbmc10k.h5ad")
adata.write_h5ad(out)
print(f"wrote {out}: {out.stat().st_size/1e6:.1f} MB")
