"""Quickstart: load PBMC-3k, run rustscenic.grn.infer, print top regulators.

Tries scanpy's PBMC-3k (network download to scanpy's cache); if the
upstream dataset server is unreachable, falls back to a small synthetic
fixture so the demo still runs. Run with:

    python -m rustscenic.quickstart
"""
import sys
import urllib.error


def _load_pbmc3k_with_retry(sc, attempts: int = 3):
    """Try scanpy's PBMC-3k download, retrying on transient network error."""
    last_err: BaseException | None = None
    for i in range(attempts):
        try:
            return sc.datasets.pbmc3k(), "pbmc3k"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            print(
                f"  attempt {i + 1}/{attempts} failed: {e}; retrying...",
                file=sys.stderr,
            )
    raise RuntimeError(f"PBMC-3k download failed after {attempts} attempts") from last_err


def _synthetic_fixture():
    """Fallback dataset when the network download is unavailable."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n_cells, n_genes = 500, 300
    # Include the TF symbols the demo asks for so the GRN has something to fit.
    tf_genes = ["SPI1", "CEBPD", "MAFB", "CEBPB", "KLF4",
                "IRF8", "PAX5", "EBF1", "TCF7", "LEF1", "TBX21"]
    gene_names = tf_genes + [f"GENE{i:04d}" for i in range(n_genes - len(tf_genes))]
    X = rng.poisson(lam=1.5, size=(n_cells, n_genes)).astype("float32")
    libsize = X.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    X = np.log1p(X / libsize * 1e4).astype("float32")
    return (
        ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
            var=pd.DataFrame(index=gene_names),
        ),
        "synthetic",
    )


def main() -> int:
    try:
        import anndata as ad  # noqa: F401
        import scanpy as sc
        import rustscenic.grn
    except ImportError as e:
        print(
            f"error: missing dependency: {e.name}. Install with: pip install {e.name}",
            file=sys.stderr,
        )
        return 1

    print("rustscenic quickstart: loading PBMC-3k...")
    try:
        adata, source = _load_pbmc3k_with_retry(sc)
    except RuntimeError as e:
        print(
            f"  upstream unreachable ({e}); falling back to a synthetic fixture "
            f"so the demo still runs.",
            file=sys.stderr,
        )
        adata, source = _synthetic_fixture()

    if source == "pbmc3k":
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    print(f"  source={source}, preprocessed: {adata.shape}")

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
