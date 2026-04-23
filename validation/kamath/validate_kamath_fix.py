"""End-to-end validation of the cellxgene silent-zero fix on Kamath 2022.

This is the test that directly validates Fuaad's scenario, on the actual
dataset class he reported the bug against.

Cellxgene's convention stores ENSEMBL IDs in `var_names` and gene
symbols in `var["feature_name"]`. Before PR #18 our AUCell scored all
zeros because regulon gene names (HGNC symbols) couldn't match
ENSEMBL IDs. The fix auto-detects the convention and swaps.

Assertions:
  1. The loaded AnnData really does have ENSEMBL var_names (Fuaad's
     convention, not our synthetic reshape).
  2. rustscenic.aucell emits the ENSEMBL-detected warning (proves the
     fix fired, not just happened to work by coincidence).
  3. AUCell output is NOT all-zero (the bug symptom).
  4. Regulon coverage is ≥ 50% (enough regulons resolved).

Dataset: Kamath et al. 2022 "Single-cell genomic profiling of human
dopamine neurons...", OPC cells subset (13,691 cells, 106 MB).
Cellxgene dataset id f25a8375-1db5-49a0-9c85-b72dbe5e2a92.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
H5AD = HERE / "kamath_opc.h5ad"


def main() -> int:
    print(f"Loading {H5AD} ...")
    adata = ad.read_h5ad(H5AD)
    print(f"  shape: {adata.shape}  (cells × genes)")

    # --- 1. Confirm this IS the cellxgene ENSEMBL-in-var_names scenario
    sample_var_names = list(adata.var_names[:5])
    print(f"  var_names sample: {sample_var_names}")
    assert any(v.startswith("ENSG") for v in sample_var_names), (
        "this dataset doesn't have ENSEMBL var_names — we need one that "
        "reproduces Fuaad's convention"
    )
    print("  ✓ var_names are ENSEMBL IDs (cellxgene convention)")
    assert "feature_name" in adata.var.columns, (
        "dataset is missing var['feature_name'] — not a standard cellxgene "
        "release?"
    )
    print(f"  feature_name sample: {list(adata.var['feature_name'][:5])}")
    print("  ✓ gene symbols live in var['feature_name']")

    # Subset to speed up: 2k cells, top 5k highly-variable genes by variance
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    if adata.n_obs > 2000:
        sel_cells = rng.choice(adata.n_obs, size=2000, replace=False)
        adata = adata[sel_cells].copy()

    X = adata.X
    if sp.issparse(X):
        variances = np.asarray(X.power(2).mean(axis=0)).flatten() - (
            np.asarray(X.mean(axis=0)).flatten()
        ) ** 2
    else:
        variances = X.var(axis=0)
    top_genes = np.argsort(variances)[::-1][:5000]
    adata = adata[:, top_genes].copy()
    print(f"  subsetted to {adata.shape}")

    # --- 2. Build a small regulon set with HGNC-symbol gene names
    symbol_pool = adata.var["feature_name"].astype(str).tolist()
    symbol_pool = [s for s in symbol_pool if s and not s.startswith("ENSG")]
    print(f"  gene symbols available: {len(symbol_pool)} (e.g. {symbol_pool[:5]})")

    # Build some realistic regulons by picking symbols
    regulons = []
    for i in range(5):
        tf_name = f"TF_{i}"
        targets = list(rng.choice(symbol_pool, size=50, replace=False))
        regulons.append((tf_name, targets))
    print(f"  built {len(regulons)} synthetic regulons over HGNC symbols")

    # --- 3. Run AUCell; check that the ENSEMBL-warning fires and output is non-zero
    import rustscenic.aucell

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)

    fix_fired = [
        w for w in caught
        if "ENSEMBL" in str(w.message) or "feature_name" in str(w.message)
    ]
    print(f"\nAUCell output shape: {auc.shape}")
    print(f"  non-zero cells: {(auc.values > 0).sum(axis=0)}")
    print(f"  mean AUC per regulon: {auc.mean(axis=0).values.round(4).tolist()}")

    # Regulon coverage diagnostic (PR #18 round-trips this via .attrs)
    coverage = auc.attrs.get("regulon_coverage")
    print(f"  regulon_coverage: {coverage}")

    # --- Validation assertions ---
    failures = []
    if not fix_fired:
        failures.append(
            "ENSEMBL auto-swap warning did NOT fire — the fix isn't running"
        )
    if not (auc.values > 0).any():
        failures.append(
            "AUCell output is all-zero — the bug is NOT fixed"
        )
    if coverage is not None:
        low = [
            tf for tf, (matched, total) in coverage.items()
            if total > 0 and matched / total < 0.5
        ]
        if low:
            failures.append(
                f"regulons with < 50% coverage: {low[:3]}... (may mask gene-name conv mismatch)"
            )

    # --- 3.5. Duplicate-symbol auto-dedupe on the FULL gene set ---
    # Kamath has 11 duplicate symbols after ENSEMBL→HGNC swap
    # (e.g. SPATA13 mapped by 2 ENSG IDs). Before PR #28 this raised
    # a cryptic ValueError. Now it sums columns with a warning.
    print("\nDup-safety check: AUCell on full 33k-gene matrix")
    full_adata = ad.read_h5ad(H5AD)[:500, :].copy()
    with warnings.catch_warnings(record=True) as full_caught:
        warnings.simplefilter("always")
        import rustscenic.aucell
        full_auc = rustscenic.aucell.score(
            full_adata,
            [("R_full", ["SPATA13", "MALAT1"])],
            top_frac=0.05,
        )
    dup_warned = [
        w for w in full_caught if "duplicate gene name" in str(w.message)
    ]
    print(f"  full-matrix AUC shape: {full_auc.shape}")
    print(f"  dup-dedupe warning fired: {bool(dup_warned)}")
    print(f"  coverage: {full_auc.attrs['regulon_coverage']}")
    if not dup_warned:
        failures.append(
            "full-gene Kamath run didn't fire duplicate-symbol warning — "
            "silent dedupe is back"
        )
    if (full_auc.values > 0).sum() == 0:
        failures.append(
            "full-gene Kamath run scored all-zero — dedupe broke the signal"
        )

    # --- 4. GRN on ENSEMBL-keyed data must also resolve via feature_name ---
    import rustscenic.grn
    real_tfs = [s for s in symbol_pool[:100] if len(s) < 10][:3]
    print(f"\nGRN test with TFs (HGNC symbols): {real_tfs}")
    with warnings.catch_warnings(record=True) as grn_caught:
        warnings.simplefilter("always")
        grn_df = rustscenic.grn.infer(
            adata, tf_names=real_tfs, n_estimators=20, seed=0, verbose=False,
        )
    grn_fix_fired = [
        w for w in grn_caught
        if "ENSEMBL" in str(w.message) or "feature_name" in str(w.message)
    ]
    print(f"  GRN rows emitted: {len(grn_df)}")
    print(f"  GRN TFs recovered: {sorted(grn_df['TF'].unique().tolist()) if not grn_df.empty else 'NONE'}")
    if grn_df.empty:
        failures.append("GRN returned empty DataFrame on ENSEMBL-keyed data")
    elif not set(grn_df["TF"].unique()) & set(real_tfs):
        failures.append("GRN TFs didn't match any requested HGNC symbols")

    if failures:
        print("\n=== VALIDATION FAILED ===")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("\n=== VALIDATION PASSED ===")
    print(f"  ✓ ENSEMBL var_names detected ({sum(1 for v in sample_var_names if v.startswith('ENSG'))}/5)")
    print(f"  ✓ Auto-swap warning fired: {fix_fired[0].message}")
    print(f"  ✓ AUCell output is non-zero across {(auc.values > 0).any(axis=0).sum()}/{auc.shape[1]} regulons")
    print(f"  ✓ Mean AUC > 0 on {(auc.mean(axis=0) > 0).sum()}/{auc.shape[1]} regulons")
    print(f"  ✓ GRN recovered {len(set(grn_df['TF'].unique()) & set(real_tfs))}/{len(real_tfs)} requested TFs")
    print(f"  ✓ GRN auto-swap warning fired: {'yes' if grn_fix_fired else 'no (but output non-empty)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
