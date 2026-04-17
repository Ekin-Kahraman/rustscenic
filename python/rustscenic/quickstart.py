"""Quickstart: load PBMC-3k, run rustscenic.grn.infer, print top regulators."""
import anndata as ad
import rustscenic
import rustscenic.grn

adata = ad.read_h5ad("/Users/ekin/rustscenic/validation/reference/data/pbmc3k.h5ad")
tfs = [t for t in rustscenic.grn.load_tfs(
    "/Users/ekin/rustscenic/validation/reference/data/allTFs_hg38.txt"
) if t in set(adata.var_names)]
print(f"data: {adata.shape}   tfs: {len(tfs)}")

# Small run for quickstart
grn = rustscenic.grn.infer(adata, tfs, seed=777, n_estimators=50)
print(f"top-5 edges:\n{grn.nlargest(5, 'importance').to_string(index=False)}")
