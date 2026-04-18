"""Quickstart: load PBMC-3k, run rustscenic.grn.infer, print top regulators.

Works out-of-the-box — uses scanpy's bundled PBMC-3k (no network path needed).
Run with: `python -m rustscenic.quickstart`
"""
import sys


def main() -> int:
    try:
        import anndata as ad  # noqa: F401
        import scanpy as sc
        import rustscenic.grn
    except ImportError as e:
        print(f"error: missing dependency: {e.name}. "
              f"Install with: pip install {e.name}", file=sys.stderr)
        return 1

    print("rustscenic quickstart: loading scanpy-bundled PBMC-3k...")
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"  preprocessed: {adata.shape}")

    # A few canonical PBMC TFs — for a real workflow, load from
    # https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt
    tfs = ["SPI1", "CEBPD", "MAFB", "CEBPB", "KLF4", "IRF8",
           "PAX5", "EBF1", "TCF7", "LEF1", "TBX21"]
    tfs = [t for t in tfs if t in adata.var_names]
    print(f"  TFs in data: {len(tfs)}")

    print("running rustscenic.grn.infer (n_estimators=50, small demo)...")
    grn = rustscenic.grn.infer(adata, tfs, seed=777, n_estimators=50)
    print(f"  {len(grn)} edges\n")
    print("top-10 TF -> target edges by importance:")
    print(grn.nlargest(10, "importance").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
