"""Seven-stage analysis scale check on a 100k-cell synthetic multiome.

Closes the named credibility gap from `docs/what-rustscenic-is.md`:
"100k-cell atlas end-to-end is unmeasured for the full ATAC + RNA pipeline."

The script synthesises a 100,000 × 15,000 (RNA) + 100,000 × 50,000 (ATAC)
multiome dataset where 30 latent programmes drive correlated patches of
expression and accessibility, then runs every stage rustscenic ships:

    1. topics (Gibbs, 8-thread AD-LDA)         on the ATAC matrix
    2. GRN inference                            on RNA + 30 TFs
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

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _build_synth_multiome(
    n_cells: int,
    n_genes: int,
    n_peaks: int,
    n_programmes: int,
    seed: int,
    nnz_per_cell: int = 8_000,
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
    nnz_per_cell = min(nnz_per_cell, n_peaks)
    # Preallocate CSR indices directly.  The historical implementation built
    # two Python ``list[int]`` objects with up to 800 million entries at the
    # 100k default, making the generator itself the dominant and unbounded
    # memory risk.  Direct CSR construction keeps generation memory O(nnz)
    # with four-byte indices and no per-integer Python objects.
    capacity = n_cells * nnz_per_cell
    indices = np.empty(capacity, dtype=np.int32)
    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    cursor = 0
    for c in range(n_cells):
        prog = int(cluster[c])
        block_start = prog * (n_peaks // n_programmes)
        block_size = n_peaks // n_programmes
        n_block = int(0.7 * nnz_per_cell)
        n_other = nnz_per_cell - n_block
        block_peaks = rng.integers(block_start, block_start + block_size, size=n_block)
        other_peaks = rng.integers(0, n_peaks, size=n_other)
        peaks = np.unique(np.concatenate([block_peaks, other_peaks]))
        next_cursor = cursor + peaks.size
        indices[cursor:next_cursor] = peaks
        cursor = next_cursor
        indptr[c + 1] = cursor
    atac_X = sp.csr_matrix(
        (np.ones(cursor, dtype=np.float32), indices[:cursor], indptr),
        shape=(n_cells, n_peaks),
        copy=False,
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


def _source_sha() -> str | None:
    explicit = os.environ.get("RUSTSCENIC_SOURCE_SHA")
    if explicit:
        return explicit
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _harness_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _topic_normalisation_checks(
    cell_topic: np.ndarray,
    topic_peak: np.ndarray,
) -> tuple[dict[str, bool], float, float]:
    cell_error = float(np.max(np.abs(cell_topic.sum(axis=1) - 1.0)))
    peak_error = float(np.max(np.abs(topic_peak.sum(axis=1) - 1.0)))
    return (
        {
            "topic_cell_rows_normalised": cell_error < 1e-5,
            # topic_peak rows can contain 50k+ float32 values; sequential
            # float32 accumulation introduces roughly 1e-3 row-sum error even
            # though the Rust kernel normalises posterior counts correctly.
            "topic_peak_rows_normalised": peak_error < 2e-3,
        },
        cell_error,
        peak_error,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=100_000)
    parser.add_argument("--n-genes", type=int, default=15_000)
    parser.add_argument("--n-peaks", type=int, default=50_000)
    parser.add_argument("--n-programmes", type=int, default=30)
    parser.add_argument("--nnz-per-cell", type=int, default=8_000)
    parser.add_argument("--topics-iters", type=int, default=50)
    parser.add_argument("--topics-threads", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "e2e_100k_synthetic.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    n_cells = args.n_cells
    n_genes = args.n_genes
    n_peaks = args.n_peaks
    K = args.n_programmes
    if (
        n_cells < 1
        or n_genes < K * 25
        or n_peaks < K * 50
        or args.topics_iters < 1
        or args.topics_threads < 1
        or args.nnz_per_cell < 1
    ):
        raise ValueError(
            "n_cells must be positive, n_genes >= n_programmes*25 and "
            "n_peaks >= n_programmes*50; topic iterations, threads and "
            "nnz-per-cell must be positive"
        )

    print(f"Building synthetic multiome: {n_cells:,} cells × "
          f"{n_genes:,} genes / {n_peaks:,} peaks, K={K} programmes...",
          flush=True)
    t0 = time.monotonic()
    rna, atac, gene_coords, tfs, motif_rankings = _build_synth_multiome(
        n_cells=n_cells, n_genes=n_genes, n_peaks=n_peaks,
        n_programmes=K, seed=42, nnz_per_cell=args.nnz_per_cell,
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
    print(
        "\n[1/7] topics — collapsed-Gibbs LDA, "
        f"{args.topics_threads}-thread AD-LDA, {args.topics_iters} sweeps",
        flush=True,
    )
    import rustscenic.topics
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topics_result = rustscenic.topics.fit_gibbs(
            atac, n_topics=K, n_iters=args.topics_iters, seed=42,
            n_threads=args.topics_threads, verbose=False,
        )
    elapsed["topics"] = round(time.monotonic() - t0, 1)
    unique = int(np.unique(topics_result.cell_topic.values.argmax(axis=1)).size)
    print(f"  → fit in {elapsed['topics']}s, {unique}/{K} unique topics",
          flush=True)
    mark("after_topics")

    # ---- 2. GRN ----
    print(
        f"\n[2/7] GRN inference — {len(tfs)} TFs over {n_cells:,} cells",
        flush=True,
    )
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
        eregs = rustscenic.eregulon.build_eregulons_dataframe(
            grn, ct_peak_df, links,
            min_target_genes=5, min_enhancer_links=2,
        )
    elapsed["eregulon"] = round(time.monotonic() - t0, 1)
    n_eregulons = int(eregs.attrs.get("n_eregulons", 0))
    print(f"  → {n_eregulons} eRegulons in {elapsed['eregulon']}s", flush=True)
    mark("after_eregulon")

    # ---- 7. AUCell ----
    print("\n[7/7] AUCell — per-cell regulon activity", flush=True)
    import rustscenic.aucell
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n_eregulons:
            reg_for_aucell = rustscenic.eregulon.regulons_from_dataframe(eregs)
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
    print(
        f"{n_cells:,}-cell synthetic multiome E2E — STAGE WALL-CLOCK",
        flush=True,
    )
    print("=" * 60, flush=True)
    for stage in ["build", "topics", "grn", "regulons", "cistarget",
                  "enhancer", "eregulon", "aucell", "TOTAL"]:
        print(f"  {stage:12s} {elapsed[stage]:>7.1f}s", flush=True)
    print(f"\nFinal peak RSS: {rss_marks[-1]['rss_gb']} GB", flush=True)
    print(f"Topics unique:   {unique}/{K}", flush=True)
    print(f"GRN edges:       {len(grn):,}", flush=True)
    print(f"Cistarget hits:  {len(ct):,}", flush=True)
    print(f"Peak-gene links: {len(links):,}", flush=True)
    print(f"eRegulons:       {n_eregulons}", flush=True)
    print(f"eRegulon rows:   {len(eregs):,}", flush=True)
    print(f"AUCell shape:    {auc.shape}", flush=True)

    topic_cell_values = topics_result.cell_topic.to_numpy(dtype=np.float32, copy=False)
    topic_peak_values = topics_result.topic_peak.to_numpy(dtype=np.float32, copy=False)
    normalisation_checks, topic_cell_row_sum_max_abs_error, topic_peak_row_sum_max_abs_error = (
        _topic_normalisation_checks(topic_cell_values, topic_peak_values)
    )
    checks = {
        "topics_finite": bool(
            np.isfinite(topic_cell_values).all() and np.isfinite(topic_peak_values).all()
        ),
        **normalisation_checks,
        "grn_non_empty": bool(len(grn)),
        "regulons_non_empty": bool(regulons),
        "cistarget_non_empty": bool(len(ct)),
        "enhancer_links_non_empty": bool(len(links)),
        "eregulons_non_empty": bool(n_eregulons),
        "aucell_shape_valid": list(auc.shape) == [n_cells, len(reg_for_aucell)],
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise AssertionError(f"100k E2E correctness checks failed: {failed_checks}")

    record = {
        "benchmark_kind": "synthetic_scale_check",
        "claim_scope": (
            "Seven-stage execution-scale evidence; not a default-parameter, "
            "full-TF, raw-fragment-preprocessing or reference-memory comparison."
        ),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "rustscenic_version": version("rustscenic"),
        "rustscenic_sha": _source_sha(),
        "harness_sha256": _harness_sha256(),
        "command": (
            "python validation/scaling/bench_e2e_100k_synthetic.py "
            f"--n-cells {n_cells} --n-genes {n_genes} --n-peaks {n_peaks} "
            f"--n-programmes {K} --nnz-per-cell {args.nnz_per_cell} "
            f"--topics-iters {args.topics_iters} "
            f"--topics-threads {args.topics_threads} --out <external-output>"
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "thread_env": {
                key: os.environ.get(key)
                for key in (
                    "RAYON_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "repetitions": 1,
        "warmup": "none; fresh synthetic construction and one measured run",
        "n_cells": n_cells, "n_genes": n_genes, "n_peaks": n_peaks, "K": K,
        "nnz_per_cell_requested": args.nnz_per_cell,
        "seed": 42,
        "topics_iters": args.topics_iters,
        "topics_threads": args.topics_threads,
        "n_grn_estimators": 20,
        "raw_fragment_preprocessing_included": False,
        "elapsed": elapsed,
        "rss_marks": rss_marks,
        "unique_topics": unique,
        "topic_cell_row_sum_max_abs_error": topic_cell_row_sum_max_abs_error,
        "topic_peak_row_sum_max_abs_error": topic_peak_row_sum_max_abs_error,
        "n_grn_edges": int(len(grn)),
        "n_cistarget_hits": int(len(ct)),
        "n_enhancer_links": int(len(links)),
        "n_eregulons": n_eregulons,
        "n_eregulon_rows": int(len(eregs)),
        "aucell_shape": list(auc.shape),
        "correctness_checks": checks,
        "path_policy": "portable",
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nrecord → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
