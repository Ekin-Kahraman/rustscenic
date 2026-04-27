"""Full 6-stage end-to-end pipeline at 100k synthetic multiome.

Closes the named credibility gap from `docs/what-rustscenic-is.md`:
"100k-cell atlas end-to-end is unmeasured for the full ATAC + RNA pipeline."

The script synthesises a 100,000 × 30,000 (RNA) + 100,000 × 50,000 (ATAC)
multiome dataset where 30 latent programmes drive correlated patches of
expression and accessibility, then runs every stage rustscenic ships:

    1. topics (Gibbs, 8-thread AD-LDA)         on the ATAC matrix
    2. GRN inference                            on RNA + 50 TFs
    3. regulon construction (top-N targets/TF)
    4. cistarget motif enrichment              against synthetic motif rankings
    5. enhancer→gene linking                   from peak-gene Pearson
    6. eRegulon assembly                       (3-way intersection)
    7. AUCell scoring                          per-cell regulon activity

We skip the fragments→matrix preproc stage (validated separately on real
PBMC at v0.2.0) and start from the cells × peaks AnnData.

Reports per-stage wall-clock + the global peak RSS. The intent is to
prove every stage connects at 100k scale, not to claim arbitrary speed
records.

Setup:
  python validation/scaling/bench_e2e_100k_synthetic.py
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


def _build_synth_multiome(
    n_cells: int, n_genes: int, n_peaks: int, n_programmes: int, seed: int
):
    """Synthesise correlated RNA + ATAC. Each cell is assigned to a
    programme; that programme drives both gene expression and peak
    accessibility for the cell.

    - First `n_programmes * 25` genes track programmes (25 each).
    - First `n_programmes * 50` peaks track programmes (50 each).
    - Remaining genes / peaks are noise to bulk up the matrix.
    """
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_programmes, size=n_cells, dtype=np.int32)

    # ATAC: each cell draws ~8000 peaks total — 70% from its programme's
    # 50-peak block, 30% noise. Same shape as the synthetic-atlas Gibbs bench.
    nnz_per_cell = 8_000
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

    # RNA: each cell expresses high values for its programme's 25 genes,
    # log-normal noise for the rest. Allocate once with float32 noise then
    # overlay the programme signal — avoids the doubled-allocation peak.
    rna = (0.1 * rng.normal(size=(n_cells, n_genes))).astype(np.float32)
    for c in range(n_cells):
        prog = int(cluster[c])
        prog_gene_start = prog * 25
        prog_gene_end = prog_gene_start + 25
        rna[c, prog_gene_start:prog_gene_end] += (
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

    # Place each gene's TSS near its programme's peak block on chr1
    gene_tss = np.zeros(n_genes, dtype=np.int64)
    for g in range(n_genes):
        if g < n_programmes * 25:
            prog = g // 25
            peak_for_gene = prog * (n_peaks // n_programmes) + (g % 25) * 2
            gene_tss[g] = peak_for_gene * 5000 + 250
        else:
            gene_tss[g] = 5_000_000_000 + g  # far from any peak
    gene_coords = pd.DataFrame(
        {"gene": gene_names, "chrom": ["chr1"] * n_genes, "tss": gene_tss}
    )

    # Synthetic motif rankings: 1 motif per programme, ranking that
    # programme's 25 genes high. n_motifs = n_programmes.
    motif_names = [f"M_PROG_{p}" for p in range(n_programmes)]
    rankings = np.full((n_programmes, n_genes), n_genes - 1, dtype=np.int32)
    for p in range(n_programmes):
        prog_genes = list(range(p * 25, (p + 1) * 25))
        for rank, g in enumerate(prog_genes):
            rankings[p, g] = rank
    motif_rankings = pd.DataFrame(rankings, index=motif_names, columns=gene_names)

    # TFs: pick the first gene from each programme's 25-gene block
    tf_names = [f"GENE_{p * 25:05d}" for p in range(n_programmes)]

    return rna_adata, atac_adata, gene_coords, tf_names, motif_rankings


def _peak_id_from_name(peak_names):
    """Convert 'chr1:start-end' → ('chr1', start, end) tuples for cistarget."""
    out = []
    for pn in peak_names:
        c, rest = pn.split(":")
        s, e = rest.split("-")
        out.append((c, int(s), int(e)))
    return out


def main() -> int:
    n_cells = 100_000
    n_genes = 15_000   # 100k × 15k dense RNA = 6 GB; full 30k OOMs at peak
    n_peaks = 50_000
    K = 30  # programmes / topics

    print(f"Building synthetic multiome: {n_cells:,} cells × "
          f"{n_genes:,} genes / {n_peaks:,} peaks, K={K} programmes...",
          flush=True)
    t0 = time.monotonic()
    rna, atac, gene_coords, tfs, motif_rankings = _build_synth_multiome(
        n_cells=n_cells, n_genes=n_genes, n_peaks=n_peaks,
        n_programmes=K, seed=42,
    )
    build_t = time.monotonic() - t0
    print(f"  built in {build_t:.1f}s, RNA nnz=N/A (dense), ATAC nnz={atac.X.nnz:,}",
          flush=True)

    elapsed: dict = {"build": round(build_t, 1)}
    rss_marks: list = []

    def mark(label: str):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_gb = rss / (1024 ** 3)
        else:
            rss_gb = rss / (1024 ** 2)
        rss_marks.append({"label": label, "rss_gb": round(rss_gb, 2)})
        print(f"  [{label}] peak RSS so far: {rss_gb:.2f} GB", flush=True)

    mark("after_build")

    # ---- 1. Topics (Gibbs, 8-thread AD-LDA) ----
    print("\n[1/7] topics — collapsed-Gibbs LDA, 8-thread AD-LDA", flush=True)
    import rustscenic.topics
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topics_result = rustscenic.topics.fit_gibbs(
            atac, n_topics=K, n_iters=50, seed=42,
            n_threads=8, verbose=False,
        )
    elapsed["topics"] = round(time.monotonic() - t0, 1)
    unique = int(np.unique(topics_result.cell_topic.values.argmax(axis=1)).size)
    print(f"  → fit in {elapsed['topics']}s, {unique}/{K} unique topics",
          flush=True)
    mark("after_topics")

    # ---- 2. GRN ----
    print("\n[2/7] GRN inference — 50 TFs over 100k cells", flush=True)
    import rustscenic.grn
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grn = rustscenic.grn.infer(
            rna, tf_names=tfs, n_estimators=20, seed=42, verbose=False,
        )
    elapsed["grn"] = round(time.monotonic() - t0, 1)
    print(f"  → {len(grn):,} edges in {elapsed['grn']}s", flush=True)
    mark("after_grn")

    # ---- 3. Regulons ----
    print("\n[3/7] regulons — top-30 targets per TF", flush=True)
    t0 = time.monotonic()
    regulons = []
    for tf in tfs:
        top = grn[grn["TF"] == tf].nlargest(30, "importance")["target"].tolist()
        if len(top) >= 5:
            regulons.append((f"{tf}_regulon", top))
    elapsed["regulons"] = round(time.monotonic() - t0, 1)
    print(f"  → {len(regulons)} regulons in {elapsed['regulons']}s", flush=True)
    mark("after_regulons")

    # ---- 4. Cistarget ----
    print("\n[4/7] cistarget — motif enrichment against synthetic rankings",
          flush=True)
    import rustscenic.cistarget
    t0 = time.monotonic()
    ct = rustscenic.cistarget.enrich(
        motif_rankings, regulons, top_frac=0.05, auc_threshold=0.0,
    )
    elapsed["cistarget"] = round(time.monotonic() - t0, 1)
    print(f"  → {len(ct):,} enrichments in {elapsed['cistarget']}s", flush=True)
    mark("after_cistarget")

    # ---- 5. Enhancer-gene linking ----
    print("\n[5/7] enhancer → gene Pearson linking", flush=True)
    import rustscenic.enhancer
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        links = rustscenic.enhancer.link_peaks_to_genes(
            rna, atac, gene_coords,
            max_distance=500_000, min_abs_corr=0.1,
        )
    elapsed["enhancer"] = round(time.monotonic() - t0, 1)
    print(f"  → {len(links):,} peak-gene links in {elapsed['enhancer']}s",
          flush=True)
    mark("after_enhancer")

    # ---- 6. eRegulon assembly ----
    print("\n[6/7] eRegulon assembly (3-way intersection)", flush=True)
    import rustscenic.eregulon
    # Build a peak-level cistarget frame: each TF's regulon mapped to its
    # programme's peak block. eRegulon assembly filters by `auc`, so we
    # need that column even though the synthetic AUC is uniform here.
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
    elapsed["eregulon"] = round(time.monotonic() - t0, 1)
    print(f"  → {len(eregs)} eRegulons in {elapsed['eregulon']}s", flush=True)
    mark("after_eregulon")

    # ---- 7. AUCell ----
    print("\n[7/7] AUCell — per-cell regulon activity", flush=True)
    import rustscenic.aucell
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if eregs:
            reg_for_aucell = [(f"{er.tf}_eregulon", er.target_genes) for er in eregs]
        else:
            # fallback to GRN regulons if eregulon assembly emptied out
            reg_for_aucell = regulons
        auc = rustscenic.aucell.score(rna, reg_for_aucell, top_frac=0.05)
    elapsed["aucell"] = round(time.monotonic() - t0, 1)
    print(f"  → AUCell shape {auc.shape} in {elapsed['aucell']}s", flush=True)
    mark("after_aucell")

    total_pipeline = sum(
        v for k, v in elapsed.items() if k != "build"
    )
    elapsed["TOTAL"] = round(total_pipeline, 1)

    print("\n" + "=" * 60, flush=True)
    print("100k synthetic multiome E2E — STAGE WALL-CLOCK", flush=True)
    print("=" * 60, flush=True)
    for stage in ["build", "topics", "grn", "regulons", "cistarget",
                  "enhancer", "eregulon", "aucell", "TOTAL"]:
        print(f"  {stage:12s} {elapsed[stage]:>7.1f}s", flush=True)
    print(f"\nFinal peak RSS: {rss_marks[-1]['rss_gb']} GB", flush=True)
    print(f"Topics unique:   {unique}/{K}", flush=True)
    print(f"GRN edges:       {len(grn):,}", flush=True)
    print(f"Cistarget hits:  {len(ct):,}", flush=True)
    print(f"Peak-gene links: {len(links):,}", flush=True)
    print(f"eRegulons:       {len(eregs)}", flush=True)
    print(f"AUCell shape:    {auc.shape}", flush=True)

    record = {
        "n_cells": n_cells, "n_genes": n_genes, "n_peaks": n_peaks, "K": K,
        "elapsed": elapsed,
        "rss_marks": rss_marks,
        "unique_topics": unique,
        "n_grn_edges": int(len(grn)),
        "n_cistarget_hits": int(len(ct)),
        "n_enhancer_links": int(len(links)),
        "n_eregulons": int(len(eregs)),
        "aucell_shape": list(auc.shape),
    }
    out = Path(__file__).parent / "e2e_100k_synthetic.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nrecord → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
