"""Real-life multi-dataset rustscenic test.

Runs the same rustscenic workflow (auto-resolve var_names → GRN → AUCell)
across four different real datasets, measuring time + correctness for
each. Datasets span cell-count tiers, species, conventions:

    Tier S    1,248 cells   Mouse     cellxgene ENSEMBL   ovary endothelial
    Tier RNA  2,711 cells   Human     10x symbol-id       PBMC 3k Multiome
    Tier M   13,691 cells   Human     cellxgene ENSEMBL   Kamath OPC
    Tier L   30,084 cells   Human     cellxgene ENSEMBL   Tabula Sapiens LI

For each: log size, GRN runtime, AUCell runtime, AUC non-zero fraction,
mean regulon coverage, peak RSS. Any silent-zero / convention-mismatch
issue surfaces here, not in unit tests.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent


def rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024**3 if sys.platform == "darwin" else 1024**2)


def normalise(adata: ad.AnnData) -> ad.AnnData:
    """log1p(normalize_total) — standard scanpy preproc, idempotent if
    already log-scaled (warning fires upstream if values look raw)."""
    counts = np.asarray(adata.X.sum(axis=1)).flatten()
    counts[counts == 0] = 1.0
    if sp.issparse(adata.X):
        x = adata.X.multiply(1e4 / counts[:, None]).tocsr()
        x.data = np.log1p(x.data)
        adata.X = x.astype(np.float32)
    else:
        x = adata.X * (1e4 / counts[:, None])
        adata.X = np.log1p(x).astype(np.float32)
    return adata


def load_10x_h5(path: Path) -> ad.AnnData:
    """Load 10x Multiome RNA from filtered_feature_bc_matrix.h5."""
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = m["shape"][:]
        bcs = m["barcodes"][:].astype(str)
        feat = m["features"]
        is_rna = feat["feature_type"][:].astype(str) == "Gene Expression"
        feat_id = feat["id"][:].astype(str)
        feat_name = feat["name"][:].astype(str)
    full = sp.csc_matrix((data, indices, indptr), shape=tuple(shape)).T
    return ad.AnnData(
        X=full[:, is_rna].tocsr().astype(np.float32),
        obs=pd.DataFrame(index=bcs),
        var=pd.DataFrame({"feature_name": feat_name[is_rna]}, index=feat_id[is_rna]),
    )


DATASETS = [
    {
        "label": "S_mouse_ovary",
        "tier": "S (1.2k mouse cellxgene)",
        "loader": lambda: ad.read_h5ad(HERE / "mouse_ovary.h5ad"),
        "tfs": ("mouse", ["Pax5", "Spi1", "Tcf7", "Gata1", "Sox2", "Foxj1"]),
    },
    {
        "label": "RNA_pbmc_multiome",
        "tier": "RNA 2.7k human 10x Multiome",
        "loader": lambda: load_10x_h5(Path("validation/real_multiome/rna.h5")),
        "tfs": ("human", ["SPI1", "PAX5", "TCF7", "CEBPB", "GATA3", "TBX21", "IRF8"]),
    },
    {
        "label": "M_kamath_opc",
        "tier": "M 13.7k human cellxgene",
        "loader": lambda: ad.read_h5ad(Path("validation/kamath/kamath_opc.h5ad")),
        "tfs": ("human", ["MALAT1", "MT-RNR2", "LINC00486", "MT-CO2", "FOS", "JUN"]),
    },
    {
        "label": "L_tabsap_intestine",
        "tier": "L 30.1k human cellxgene",
        "loader": lambda: ad.read_h5ad(HERE / "tabsap_intestine.h5ad"),
        "tfs": ("human", ["SPI1", "PAX5", "TCF7", "CEBPB", "GATA3", "FOXJ1"]),
    },
]


def run_one(spec: dict) -> dict:
    import warnings
    import rustscenic.aucell
    import rustscenic.grn

    label = spec["label"]
    print(f"\n{'='*72}\n  {label}  —  {spec['tier']}\n{'='*72}")
    t0 = time.monotonic()
    adata = spec["loader"]()
    print(f"  loaded: {adata.shape} in {time.monotonic()-t0:.1f}s, "
          f"X dtype {adata.X.dtype}, var_names sample {list(adata.var_names[:3])}")

    # Some cellxgene datasets store log-norm in .X already; some store raw.
    # Detect via max value and normalise only if needed.
    X = adata.X
    max_val = float(X.max() if not sp.issparse(X) else X.max())
    if max_val > 50:
        print(f"  X looks raw (max={max_val:.1f}), normalising...")
        adata = normalise(adata)
    else:
        print(f"  X looks pre-normalised (max={max_val:.2f})")

    species, candidate_tfs = spec["tfs"]
    out: dict = {"label": label, "tier": spec["tier"]}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # GRN
        t0 = time.monotonic()
        grn = rustscenic.grn.infer(
            adata, tf_names=candidate_tfs, n_estimators=30, seed=0, verbose=False,
        )
        out["grn_s"] = time.monotonic() - t0
        out["grn_edges"] = len(grn)
        out["tfs_recovered"] = sorted(set(grn["TF"].unique())) if not grn.empty else []

        # Build top-30 regulons; AUCell
        regulons = [
            (tf, grn[grn["TF"] == tf].nlargest(30, "importance")["target"].tolist())
            for tf in out["tfs_recovered"]
        ]
        regulons = [(n, g) for n, g in regulons if len(g) >= 5]
        if regulons:
            t0 = time.monotonic()
            auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
            out["aucell_s"] = time.monotonic() - t0
            out["auc_shape"] = list(auc.shape)
            out["auc_nonzero_frac"] = float((auc.values > 0).mean())
            out["auc_max"] = float(auc.values.max())
            cov = auc.attrs.get("regulon_coverage", {})
            covs = [m / t for m, t in cov.values() if t > 0]
            out["mean_coverage"] = float(np.mean(covs)) if covs else None
        else:
            out["aucell_s"] = None
            out["mean_coverage"] = 0

    # Capture warning categories that fired (proxy for what guards triggered)
    warning_msgs = [str(w.message) for w in caught if w.category is UserWarning]
    out["warnings"] = []
    for msg in warning_msgs:
        if "ENSEMBL" in msg:
            out["warnings"].append("ensembl_swap")
        elif "duplicate" in msg.lower():
            out["warnings"].append("duplicate_swap")
        elif "Titlecase" in msg or "UPPERCASE" in msg:
            out["warnings"].append("species_case")
        elif "no genes overlap" in msg or "regulons dropped" in msg:
            out["warnings"].append("regulon_drop")
        elif "log-normalised input" in msg:
            out["warnings"].append("not_lognorm")

    out["peak_rss_gb"] = rss_gb()
    print(f"  GRN: {out['grn_s']:.1f}s, edges={out['grn_edges']:,}, TFs recovered={len(out['tfs_recovered'])}/{len(candidate_tfs)}")
    if out.get("aucell_s") is not None:
        print(f"  AUCell: {out['aucell_s']:.1f}s, shape={out['auc_shape']}, "
              f"non-zero {out['auc_nonzero_frac']*100:.0f}%, "
              f"mean coverage {out['mean_coverage']*100:.0f}% , max AUC {out['auc_max']:.3f}")
    else:
        print(f"  AUCell skipped (no regulons recovered)")
    print(f"  warnings fired: {out['warnings']}")
    print(f"  peak RSS: {out['peak_rss_gb']:.2f} GB")
    return out


def main() -> int:
    results = []
    for spec in DATASETS:
        try:
            results.append(run_one(spec))
        except Exception as e:
            print(f"  FAIL {spec['label']}: {type(e).__name__}: {e}")
            results.append({"label": spec["label"], "tier": spec["tier"], "error": f"{type(e).__name__}: {e}"})

    print("\n\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["label"], r["tier"], "ERROR", "-", "-", r["error"][:40]])
        else:
            rows.append([
                r["label"],
                r["tier"],
                f'{r["grn_s"]:.1f}s',
                f'{r["aucell_s"]:.1f}s' if r.get("aucell_s") else "-",
                f'{r.get("mean_coverage", 0)*100:.0f}%',
                ",".join(r.get("warnings", [])) or "none",
            ])
    print(f"{'label':<20s} {'tier':<32s} {'GRN':<8s} {'AUC':<8s} {'cov':<6s} warnings")
    for row in rows:
        print(f"{row[0]:<20s} {row[1]:<32s} {row[2]:<8s} {row[3]:<8s} {row[4]:<6s} {row[5]}")

    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults → {HERE / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
