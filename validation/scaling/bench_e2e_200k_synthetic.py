"""200k synthetic full 7-stage E2E — pushes the v0.3.2 atlas proof
from 100k cells to 200k.

Memory budget on a 32 GB laptop is tight at 200k. We use a slightly
narrower vocabulary (8,000 genes, 30,000 peaks) to fit the same dense
RNA + sparse ATAC pattern within the OS+Python+rustscenic working set.

The point of this bench is to demonstrate that all 7 stages connect
at 200k cells, not to claim biological correctness on synthetic data.
For real-data quality numbers see the PBMC head-to-head benches.

Reproducible:
  python validation/scaling/bench_e2e_200k_synthetic.py
"""
from __future__ import annotations

import json
import resource
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _build_synth_multiome(n_cells, n_genes, n_peaks, n_programmes, seed):
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_programmes, size=n_cells, dtype=np.int32)

    # ATAC: sparse, 70% block + 30% noise
    nnz_per_cell = 5_000  # smaller than 100k bench's 8000 to fit memory
    rows, cols = [], []
    for c in range(n_cells):
        prog = int(cluster[c])
        block_start = prog * (n_peaks // n_programmes)
        block_size = n_peaks // n_programmes
        n_block = int(0.7 * nnz_per_cell)
        n_other = nnz_per_cell - n_block
        block_peaks = rng.integers(block_start, block_start + block_size, size=n_block)
        other_peaks = rng.integers(0, n_peaks, size=n_other)
        peaks = np.unique(np.concatenate([block_peaks, other_peaks]))
        rows.extend([c] * peaks.size)
        cols.extend(peaks.tolist())
    atac_X = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_peaks),
    )

    # RNA: dense, allocated once with noise + programme overlay
    rna = (0.1 * rng.normal(size=(n_cells, n_genes))).astype(np.float32)
    for c in range(n_cells):
        prog = int(cluster[c])
        rna[c, prog * 25:prog * 25 + 25] += (
            2.0 + 0.5 * rng.normal(size=25).astype(np.float32)
        )
    np.clip(rna, 0.0, None, out=rna)
    rna += 0.05

    cell_names = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"GENE_{i:05d}" for i in range(n_genes)]
    peak_names = [f"chr1:{i*5000}-{i*5000+500}" for i in range(n_peaks)]

    rna_adata = ad.AnnData(
        X=rna,
        obs=pd.DataFrame({"cluster": cluster}, index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )
    atac_adata = ad.AnnData(
        X=atac_X,
        obs=pd.DataFrame({"cluster": cluster}, index=cell_names),
        var=pd.DataFrame(index=peak_names),
    )

    gene_tss = np.zeros(n_genes, dtype=np.int64)
    for g in range(n_genes):
        if g < n_programmes * 25:
            prog = g // 25
            peak_for_gene = prog * (n_peaks // n_programmes) + (g % 25) * 2
            gene_tss[g] = peak_for_gene * 5000 + 250
        else:
            gene_tss[g] = 5_000_000_000 + g
    gene_coords = pd.DataFrame(
        {"gene": gene_names, "chrom": ["chr1"] * n_genes, "tss": gene_tss}
    )

    motif_names = [f"M_PROG_{p}" for p in range(n_programmes)]
    rankings = np.full((n_programmes, n_genes), n_genes - 1, dtype=np.int32)
    for p in range(n_programmes):
        for rank, g in enumerate(range(p * 25, (p + 1) * 25)):
            rankings[p, g] = rank
    motif_rankings = pd.DataFrame(rankings, index=motif_names, columns=gene_names)
    tf_names = [f"GENE_{p * 25:05d}" for p in range(n_programmes)]
    return rna_adata, atac_adata, gene_coords, tf_names, motif_rankings


def main() -> int:
    n_cells = 200_000
    n_genes = 8_000
    n_peaks = 30_000
    K = 30

    print(f"Building synthetic multiome: {n_cells:,} cells × "
          f"{n_genes:,} genes / {n_peaks:,} peaks, K={K}", flush=True)
    t0 = time.monotonic()
    rna, atac, gene_coords, tfs, motif_rankings = _build_synth_multiome(
        n_cells=n_cells, n_genes=n_genes, n_peaks=n_peaks,
        n_programmes=K, seed=42,
    )
    build_t = time.monotonic() - t0
    print(f"  built in {build_t:.1f}s, ATAC nnz={atac.X.nnz:,}", flush=True)

    elapsed: dict = {"build": round(build_t, 1)}
    rss_marks: list = []

    def mark(label: str):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)
        rss_marks.append({"label": label, "rss_gb": round(rss_gb, 2)})
        print(f"  [{label}] peak RSS: {rss_gb:.2f} GB", flush=True)

    mark("after_build")

    print("\n[1/7] topics — Gibbs, 8-thread AD-LDA, 30 iters", flush=True)
    import rustscenic.topics
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topics_result = rustscenic.topics.fit_gibbs(
            atac, n_topics=K, n_iters=30, seed=42,
            n_threads=8, verbose=False,
        )
    elapsed["topics"] = round(time.monotonic() - t0, 1)
    unique = int(np.unique(topics_result.cell_topic.values.argmax(axis=1)).size)
    print(f"  → fit in {elapsed['topics']}s, {unique}/{K} unique", flush=True)
    mark("after_topics")

    print("\n[2/7] GRN — 30 TFs, n_est=15", flush=True)
    import rustscenic.grn
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grn = rustscenic.grn.infer(
            rna, tf_names=tfs, n_estimators=15, seed=42, verbose=False,
        )
    elapsed["grn"] = round(time.monotonic() - t0, 1)
    print(f"  → {len(grn):,} edges in {elapsed['grn']}s", flush=True)
    mark("after_grn")

    print("\n[3/7] regulons", flush=True)
    t0 = time.monotonic()
    regulons = []
    for tf in tfs:
        top = grn[grn["TF"] == tf].nlargest(20, "importance")["target"].tolist()
        if len(top) >= 5:
            regulons.append((f"{tf}_regulon", top))
    elapsed["regulons"] = round(time.monotonic() - t0, 2)
    print(f"  → {len(regulons)} regulons", flush=True)

    print("\n[4/7] cistarget", flush=True)
    import rustscenic.cistarget
    t0 = time.monotonic()
    ct = rustscenic.cistarget.enrich(
        motif_rankings, regulons, top_frac=0.05, auc_threshold=0.0,
    )
    elapsed["cistarget"] = round(time.monotonic() - t0, 2)
    print(f"  → {len(ct):,} hits in {elapsed['cistarget']}s", flush=True)

    print("\n[5/7] enhancer→gene linking", flush=True)
    import rustscenic.enhancer
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        links = rustscenic.enhancer.link_peaks_to_genes(
            rna, atac, gene_coords,
            max_distance=500_000, min_abs_corr=0.1,
        )
    elapsed["enhancer"] = round(time.monotonic() - t0, 1)
    print(f"  → {len(links):,} links in {elapsed['enhancer']}s", flush=True)
    mark("after_enhancer")

    print("\n[6/7] eRegulon assembly", flush=True)
    import rustscenic.eregulon
    ct_peak = []
    for tf in tfs:
        prog = int(tf.split("_")[1]) // 25
        block_start = prog * (n_peaks // K)
        block_end = block_start + (n_peaks // K)
        for peak_idx in range(block_start, block_end):
            ct_peak.append({
                "regulon": f"{tf}_regulon",
                "peak_id": atac.var_names[peak_idx],
                "motif": f"M_PROG_{prog}",
                "tf": tf,
                "auc": 0.5,
            })
    ct_peak_df = pd.DataFrame(ct_peak)
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eregs = rustscenic.eregulon.build_eregulons(
            grn, ct_peak_df, links,
            min_target_genes=5, min_enhancer_links=2,
        )
    elapsed["eregulon"] = round(time.monotonic() - t0, 2)
    print(f"  → {len(eregs)} eRegulons", flush=True)

    print("\n[7/7] AUCell", flush=True)
    import rustscenic.aucell
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg_for_aucell = (
            [(f"{er.tf}_eregulon", er.target_genes) for er in eregs]
            if eregs else regulons
        )
        auc = rustscenic.aucell.score(rna, reg_for_aucell, top_frac=0.05)
    elapsed["aucell"] = round(time.monotonic() - t0, 1)
    print(f"  → {auc.shape} in {elapsed['aucell']}s", flush=True)
    mark("after_aucell")

    total_pipe = sum(v for k, v in elapsed.items() if k != "build")
    elapsed["TOTAL"] = round(total_pipe, 1)

    print("\n" + "=" * 60, flush=True)
    print(f"200k synthetic E2E — wall-clock summary", flush=True)
    print("=" * 60, flush=True)
    for stage in ["build", "topics", "grn", "regulons", "cistarget",
                  "enhancer", "eregulon", "aucell", "TOTAL"]:
        print(f"  {stage:12s} {elapsed[stage]:>8.1f}s", flush=True)
    print(f"\n  Final peak RSS: {rss_marks[-1]['rss_gb']} GB", flush=True)
    print(f"  Topics:    {unique}/{K} unique", flush=True)
    print(f"  GRN edges: {len(grn):,}", flush=True)
    print(f"  Links:     {len(links):,}", flush=True)
    print(f"  eRegulons: {len(eregs)}", flush=True)
    print(f"  AUCell:    {auc.shape}", flush=True)

    record = {
        "n_cells": n_cells, "n_genes": n_genes, "n_peaks": n_peaks, "K": K,
        "elapsed": elapsed,
        "rss_marks": rss_marks,
        "unique_topics": unique,
        "n_grn_edges": int(len(grn)),
        "n_enhancer_links": int(len(links)),
        "n_eregulons": int(len(eregs)),
        "aucell_shape": list(auc.shape),
    }
    out = Path(__file__).parent / "e2e_200k_synthetic.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nrecord → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
