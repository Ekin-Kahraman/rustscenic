"""Full-pipeline integration test — preproc → grn → cistarget → enhancer → eregulon → aucell.

Simulates a minimal multiome workflow end-to-end and asserts every
stage connects to the next without silent breakage. This is the test
that would have caught the cellxgene ``var_names`` bug before Fuaad
did — it runs on a shape that includes a cellxgene-style ENSEMBL
AnnData in addition to the scanpy-native one.

The simulated biology:
  - 150 cells, 80 genes, 30 peaks
  - Three latent programmes drive correlated patches of expression
    AND accessibility
  - One TF in each programme is the intended "cause"
  - Correlated peaks carry the TF's synthetic motif in the cistarget
    ranking matrix

Every stage must survive at least one surviving regulon / eRegulon
for the test to pass — every downstream stage empties out if the
upstream one broke.
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import rustscenic.aucell
import rustscenic.cistarget
import rustscenic.enhancer
import rustscenic.eregulon
import rustscenic.grn


N_CELLS = 150
N_GENES = 80
N_PEAKS = 30
N_MOTIFS = 12
SEED = 0


def _simulate_multiome():
    """Generate matched RNA + ATAC AnnDatas driven by 3 latent programmes.

    Returns (rna_adata, atac_adata, gene_coords, tf_names, motif_rankings).
    """
    rng = np.random.default_rng(SEED)
    latents = rng.normal(size=(3, N_CELLS))  # three programmes

    # Genes 0..25 track programme 0; 26..50 track programme 1; 51..75 programme 2
    # Remaining are noise
    gene_programme = np.full(N_GENES, -1, dtype=int)
    gene_programme[0:25] = 0
    gene_programme[25:50] = 1
    gene_programme[50:75] = 2

    rna = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
    for g in range(N_GENES):
        p = gene_programme[g]
        if p >= 0:
            rna[:, g] = 0.8 * latents[p] + 0.2 * rng.normal(size=N_CELLS)
        else:
            rna[:, g] = rng.normal(size=N_CELLS)

    # Peaks 0..9 track programme 0; 10..19 programme 1; 20..29 programme 2
    peak_programme = np.full(N_PEAKS, -1, dtype=int)
    peak_programme[0:10] = 0
    peak_programme[10:20] = 1
    peak_programme[20:30] = 2

    atac = np.zeros((N_CELLS, N_PEAKS), dtype=np.float32)
    for pk in range(N_PEAKS):
        p = peak_programme[pk]
        atac[:, pk] = 0.8 * latents[p] + 0.2 * rng.normal(size=N_CELLS)

    cell_names = [f"cell{i}" for i in range(N_CELLS)]
    gene_names = [f"GENE_{i:03d}" for i in range(N_GENES)]
    peak_names = [f"chr1:{i*10_000}-{i*10_000+500}" for i in range(N_PEAKS)]
    tf_names = ["GENE_000", "GENE_025", "GENE_050"]  # one TF per programme

    rna_adata = ad.AnnData(
        X=rna,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )
    atac_adata = ad.AnnData(
        X=atac,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=peak_names),
    )

    # Gene TSS coordinates — programme-0 genes all live within 500 kb of
    # programme-0 peaks on chr1 so enhancer linking will find them.
    gene_tss = np.zeros(N_GENES, dtype=np.int64)
    for g in range(N_GENES):
        p = gene_programme[g]
        if p >= 0:
            # Place each gene near one of its programme's peaks
            peak_for_gene = (p * 10) + (g % 10)
            gene_tss[g] = peak_for_gene * 10_000 + 250
        else:
            gene_tss[g] = 5_000_000 + g * 1000  # far from any peak
    gene_coords = pd.DataFrame(
        {"gene": gene_names, "chrom": ["chr1"] * N_GENES, "tss": gene_tss}
    )

    # Synthetic motif ranking matrix: one motif per TF, ranking the
    # correct target genes high.
    motif_names = [f"MOTIF_{i}" for i in range(N_MOTIFS)]
    rankings = np.full((N_MOTIFS, N_GENES), N_GENES - 1, dtype=np.int32)
    # First 3 motifs rank each TF's programme genes high
    for tf_idx, tf_name in enumerate(tf_names):
        programme = tf_idx  # TF 0 → programme 0, etc.
        programme_genes = [i for i, gp in enumerate(gene_programme) if gp == programme]
        for rank, g in enumerate(programme_genes):
            rankings[tf_idx, g] = rank
    motif_rankings = pd.DataFrame(
        rankings,
        index=motif_names,
        columns=gene_names,
    )
    return rna_adata, atac_adata, gene_coords, tf_names, motif_rankings


def test_end_to_end_multiome_pipeline():
    """Every stage must produce non-empty output on synthetic data where
    the biology is known — three programmes with matched TFs, genes,
    and peaks.

    If any stage breaks silently (empties out), the assertion on the
    next stage's output will fail loudly. This is the guard against
    the exact class of bug Fuaad hit."""
    rna, atac, gene_coords, tf_names, motif_rankings = _simulate_multiome()

    # ---- GRN ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grn = rustscenic.grn.infer(
            rna, tf_names=tf_names, n_estimators=50, seed=SEED, verbose=False,
        )
    assert not grn.empty, "GRN produced zero edges"
    assert set(grn["TF"].unique()) == set(tf_names), \
        f"GRN dropped TFs: expected {set(tf_names)}, got {set(grn['TF'].unique())}"

    # ---- Build regulons from GRN, score cistarget ----
    regulons = []
    for tf in tf_names:
        top = grn[grn["TF"] == tf].nlargest(15, "importance")["target"].tolist()
        if len(top) >= 3:
            regulons.append((f"{tf}_regulon", top))
    assert regulons, "No regulons built from GRN"

    ct = rustscenic.cistarget.enrich(
        motif_rankings, regulons, top_frac=0.2, auc_threshold=0.0,
    )
    assert not ct.empty, "Cistarget produced zero enrichments"
    # Each of our three TF regulons should hit at least one motif
    assert len(set(ct["regulon"].unique())) >= 3, \
        f"Expected ≥3 regulons enriched, got {ct['regulon'].unique()}"

    # ---- Enhancer → gene ----
    links = rustscenic.enhancer.link_peaks_to_genes(
        rna, atac, gene_coords, max_distance=500_000, min_abs_corr=0.3,
    )
    assert not links.empty, "No enhancer-gene links survived"
    # At least one link per programme should survive
    assert len(links) >= 3

    # ---- eRegulon assembly ----
    # Synthesise a cistarget-compatible frame with explicit peak_id
    # mapping each TF to its programme's peaks
    ct_for_eregulon = _make_peak_level_cistarget(ct, tf_names, atac)
    eregs = rustscenic.eregulon.build_eregulons(
        grn, ct_for_eregulon, links,
        min_target_genes=3, min_enhancer_links=2,
    )
    assert len(eregs) >= 1, "No eRegulons survived assembly"

    # ---- AUCell scoring on the assembled regulons ----
    reg_for_aucell = [(f"{er.tf}_eregulon", er.target_genes) for er in eregs]
    auc = rustscenic.aucell.score(rna, reg_for_aucell, top_frac=0.1)
    assert auc.shape[0] == rna.n_obs
    assert auc.shape[1] == len(reg_for_aucell)
    assert (auc.values > 0).any(), "AUCell output is entirely zero"


def test_end_to_end_on_cellxgene_shaped_rna():
    """Same pipeline, but with RNA AnnData reshaped to cellxgene
    convention (ENSEMBL in var_names, symbols in feature_name). Would
    have failed silently on the whole chain before PR #18."""
    rna, atac, gene_coords, tf_names, motif_rankings = _simulate_multiome()

    # Reshape to cellxgene shape
    symbols = list(rna.var_names)
    ensembl_ids = [f"ENSG0000011{i:04d}" for i in range(rna.n_vars)]
    new_var = pd.DataFrame({"feature_name": symbols}, index=ensembl_ids)
    rna_cx = ad.AnnData(X=rna.X, obs=rna.obs, var=new_var)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # GRN must still find the TFs via resolve_gene_names
        grn = rustscenic.grn.infer(
            rna_cx, tf_names=tf_names, n_estimators=50, seed=SEED, verbose=False,
        )
    assert not grn.empty, "GRN empty on cellxgene-shape RNA — resolver regressed"
    assert set(grn["TF"].unique()) == set(tf_names)

    # AUCell — uses the same resolve path
    regulons = [
        (f"{tf}_regulon", grn[grn["TF"] == tf].nlargest(15, "importance")["target"].tolist())
        for tf in tf_names
    ]
    regulons = [(n, g) for n, g in regulons if len(g) >= 3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = rustscenic.aucell.score(rna_cx, regulons, top_frac=0.2)
    assert (auc.values > 0).any(), "AUCell empty on cellxgene-shape RNA"


def _make_peak_level_cistarget(ct, tf_names, atac):
    """Build a cistarget-style DataFrame enriched with a peak_id column
    for eRegulon assembly. Maps each TF's enriched motif to its
    programme's peaks."""
    peak_names = list(atac.var_names)
    # Programme assignment mirrors the simulator
    peak_programme = {p: i // 10 for i, p in enumerate(peak_names)}
    rows = []
    for tf in tf_names:
        programme = tf_names.index(tf)
        for p, pg in peak_programme.items():
            if pg == programme:
                rows.append({
                    "regulon": f"{tf}_regulon",
                    "motif": f"MOTIF_{programme}",
                    "peak_id": p,
                    "auc": 0.2,
                })
    return pd.DataFrame(rows)


# ---- Full SCENIC+ orchestration: enhancer + eRegulon stages -----------


def test_pipeline_run_with_atac_and_gene_coords_emits_eregulons(tmp_path):
    """The orchestrator must run all 8 stages when fragments + peaks +
    gene_coords + motif_rankings are all supplied. Closes the audit gap
    that pipeline.run stopped at AUCell."""
    import gzip, os, anndata as ad, numpy as np, pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(0)

    # 200 cells split across 3 programmes; expression + fragment density
    # both driven by the same activity vector so peak↔gene correlation
    # is real signal, not a chance artefact.
    n_cells = 200
    cluster = np.array([i * 3 // n_cells for i in range(n_cells)], dtype=np.uint32)
    activity = np.zeros((3, n_cells), dtype=np.float32)
    for p in range(3):
        activity[p] = (cluster == p).astype(np.float32) + 0.1 * rng.normal(size=n_cells)

    rna_genes = [f"G{i:03d}" for i in range(30)]
    X = np.zeros((n_cells, 30), dtype=np.float32)
    for i in range(15):
        X[:, i] = activity[i // 5] + 0.2 * rng.normal(size=n_cells)
    for i in range(15, 30):
        X[:, i] = rng.normal(size=n_cells).astype(np.float32)
    X = np.clip(X, 0, None) + 0.1
    cells = [f"cell{i}" for i in range(n_cells)]
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cluster": cluster}, index=cells),
        var=pd.DataFrame(index=rna_genes),
    )

    # Fragments — dense per programme region, plus noise
    frag_lines = []
    for p in range(3):
        for ci in np.where(cluster == p)[0]:
            for _ in range(15):
                start = 10_000 + p * 100_000 + int(rng.integers(0, 5_000))
                frag_lines.append(f"chr1\t{start}\t{start+150}\t{cells[ci]}\t1")
        for ci in np.where(cluster == p)[0]:
            for _ in range(3):
                start = int(rng.integers(0, 2_000_000))
                frag_lines.append(f"chr1\t{start}\t{start+120}\t{cells[ci]}\t1")
    frag_path = tmp_path / "fragments.tsv.gz"
    with gzip.open(frag_path, "wt") as fh:
        fh.write("\n".join(frag_lines) + "\n")

    # Peaks BED covering each programme region
    peaks_path = tmp_path / "peaks.bed"
    with open(peaks_path, "w") as fh:
        for p in range(3):
            for j in range(3):
                start = 10_000 + p * 100_000 + j * 5_000
                fh.write(f"chr1\t{start}\t{start + 500}\tpeak_{p}_{j}\n")

    # Gene coords near each programme's peaks
    gene_coords = pd.DataFrame(
        [
            (f"G{i:03d}", "chr1", 10_000 + (i // 5) * 100_000 + 250)
            for i in range(15)
        ],
        columns=["gene", "chrom", "tss"],
    )

    # Synthetic motif rankings — each TF ranks its programme's genes high.
    motif_names = ["M_G000", "M_G005", "M_G010"]
    n_genes = len(rna_genes)
    rank_matrix = np.full((len(motif_names), n_genes), n_genes - 1, dtype=np.int32)
    for tf_idx, motif in enumerate(motif_names):
        prog_idx = tf_idx
        for rank, gene_idx in enumerate(
            [i for i in range(n_genes) if (i // 5 == prog_idx) and (i < 15)]
        ):
            rank_matrix[tf_idx, gene_idx] = rank
    motif_rankings = pd.DataFrame(rank_matrix, index=motif_names, columns=rna_genes)

    out = tmp_path / "pipeline_out"
    result = rustscenic.pipeline.run(
        rna,
        out,
        fragments=str(frag_path),
        peaks=str(peaks_path),
        tfs=["G000", "G005", "G010"],
        motif_rankings=motif_rankings,
        gene_coords=gene_coords,
        grn_n_estimators=15,
        grn_top_targets=10,
        topics_n_topics=5,
        topics_n_passes=2,
        cistarget_top_frac=0.2,
        cistarget_auc_threshold=0.0,
        enhancer_min_abs_corr=0.15,
        eregulon_min_target_genes=2,
        eregulon_min_enhancer_links=1,
        seed=0,
        verbose=False,
    )

    # Every stage emitted an artifact
    assert result.atac_matrix_path.exists()
    assert result.grn_path.exists()
    assert result.aucell_path.exists()
    assert result.cistarget_path.exists()
    assert result.enhancer_links_path.exists()
    # eregulons file exists; n_eregulons may be 0 on synthetic data
    assert result.eregulons_path is not None
    assert result.eregulons_path.exists()
    assert result.n_eregulons is not None


def test_pipeline_run_with_pre_built_adata_atac_skips_fragments_to_matrix(tmp_path):
    """When ``adata_atac`` is supplied, pipeline.run uses it directly and
    does not call fragments_to_matrix. Closes the v0.4 gate item: real-data
    workflows pre-subset ATAC to QC'd cells before topics, so the orchestrator
    must accept a pre-built matrix instead of always rebuilding from raw 10x.

    Also: every downstream stage (topics, enhancer, eRegulon) must still
    fire, identical to the fragments+peaks path.
    """
    import anndata as ad, numpy as np, pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(0)
    n_cells = 200
    cluster = np.array([i * 3 // n_cells for i in range(n_cells)], dtype=np.uint32)
    activity = np.zeros((3, n_cells), dtype=np.float32)
    for p in range(3):
        activity[p] = (cluster == p).astype(np.float32) + 0.1 * rng.normal(size=n_cells)

    rna_genes = [f"G{i:03d}" for i in range(30)]
    X = np.zeros((n_cells, 30), dtype=np.float32)
    for i in range(15):
        X[:, i] = activity[i // 5] + 0.2 * rng.normal(size=n_cells)
    for i in range(15, 30):
        X[:, i] = rng.normal(size=n_cells).astype(np.float32)
    X = np.clip(X, 0, None) + 0.1
    cells = [f"cell{i}" for i in range(n_cells)]
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cluster": cluster}, index=cells),
        var=pd.DataFrame(index=rna_genes),
    )

    # Build a cells × peaks matrix directly (this is what a user with a
    # pre-subset ATAC AnnData would have).
    n_peaks = 9
    peak_names = [f"chr1:{10000 + (j // 3) * 100000 + (j % 3) * 5000}-{10500 + (j // 3) * 100000 + (j % 3) * 5000}" for j in range(n_peaks)]
    peak_X = np.zeros((n_cells, n_peaks), dtype=np.float32)
    for j in range(n_peaks):
        prog = j // 3
        peak_X[:, j] = activity[prog] + 0.1 * rng.normal(size=n_cells)
    peak_X = np.clip(peak_X, 0, None)
    var = pd.DataFrame({
        "chrom": ["chr1"] * n_peaks,
        "start": [10000 + (j // 3) * 100000 + (j % 3) * 5000 for j in range(n_peaks)],
        "end": [10500 + (j // 3) * 100000 + (j % 3) * 5000 for j in range(n_peaks)],
    }, index=peak_names)
    adata_atac = ad.AnnData(
        X=peak_X,
        obs=pd.DataFrame(index=cells),
        var=var,
    )

    gene_coords = pd.DataFrame(
        [(f"G{i:03d}", "chr1", 10_000 + (i // 5) * 100_000 + 250) for i in range(15)],
        columns=["gene", "chrom", "tss"],
    )

    motif_names = ["M_G000", "M_G005", "M_G010"]
    n_genes = len(rna_genes)
    rank_matrix = np.full((len(motif_names), n_genes), n_genes - 1, dtype=np.int32)
    for tf_idx in range(3):
        for rank, gene_idx in enumerate([i for i in range(n_genes) if (i // 5 == tf_idx) and (i < 15)]):
            rank_matrix[tf_idx, gene_idx] = rank
    motif_rankings = pd.DataFrame(rank_matrix, index=motif_names, columns=rna_genes)

    out = tmp_path / "pipeline_out"
    result = rustscenic.pipeline.run(
        rna,
        out,
        adata_atac=adata_atac,
        tfs=["G000", "G005", "G010"],
        motif_rankings=motif_rankings,
        gene_coords=gene_coords,
        grn_n_estimators=15,
        grn_top_targets=10,
        topics_n_topics=5,
        topics_n_passes=2,
        cistarget_top_frac=0.2,
        cistarget_auc_threshold=0.0,
        enhancer_min_abs_corr=0.15,
        eregulon_min_target_genes=2,
        eregulon_min_enhancer_links=1,
        seed=0,
        verbose=False,
    )

    # ATAC path was honoured (fragments_to_matrix not called)
    assert result.atac_matrix_path is not None
    assert result.atac_matrix_path.exists()
    # All downstream stages still fired
    assert result.grn_path.exists()
    assert result.aucell_path.exists()
    assert result.cistarget_path.exists()
    assert result.enhancer_links_path.exists()
    assert result.eregulons_path is not None
    assert result.eregulons_path.exists()


def test_pipeline_run_topics_method_gibbs(tmp_path):
    """When ``topics_method='gibbs'`` (with ``topics_n_threads > 1``)
    the orchestrator runs the parallel collapsed-Gibbs sampler instead
    of online VB. Verifies that the alternative path runs end-to-end
    and the topics artifact is present."""
    import gzip, anndata as ad, numpy as np, pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(0)
    n_cells = 60
    cluster = np.array([i * 3 // n_cells for i in range(n_cells)], dtype=np.uint32)
    rna_genes = [f"G{i:03d}" for i in range(20)]
    X = np.zeros((n_cells, 20), dtype=np.float32)
    for i in range(20):
        X[:, i] = (cluster == (i % 3)).astype(np.float32) + 0.1 * rng.normal(size=n_cells)
    X = np.clip(X, 0, None) + 0.1
    cells = [f"cell{i}" for i in range(n_cells)]
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cluster": cluster}, index=cells),
        var=pd.DataFrame(index=rna_genes),
    )

    # Sparse fragments file — enough for the topics fit to have signal.
    frag_lines = []
    for p in range(3):
        for ci in np.where(cluster == p)[0]:
            for _ in range(20):
                start = 10_000 + p * 100_000 + int(rng.integers(0, 5_000))
                frag_lines.append(f"chr1\t{start}\t{start+150}\t{cells[ci]}\t1")
    frag_path = tmp_path / "fragments.tsv.gz"
    with gzip.open(frag_path, "wt") as fh:
        fh.write("\n".join(frag_lines) + "\n")

    peaks_path = tmp_path / "peaks.bed"
    with open(peaks_path, "w") as fh:
        for p in range(3):
            for j in range(3):
                start = 10_000 + p * 100_000 + j * 5_000
                fh.write(f"chr1\t{start}\t{start + 500}\tpeak_{p}_{j}\n")

    out = tmp_path / "pipeline_out"
    result = rustscenic.pipeline.run(
        rna, out,
        fragments=str(frag_path), peaks=str(peaks_path),
        tfs=["G000", "G005", "G010"],
        grn_n_estimators=10, grn_top_targets=5,
        topics_n_topics=4, topics_n_passes=2,
        topics_method="gibbs", topics_n_iters=20, topics_n_threads=2,
        seed=0, verbose=False,
    )

    # The orchestrator wrote the ATAC matrix and a topics directory
    assert result.atac_matrix_path.exists()
    assert (out / "topics" / "cell_topic.npy").exists()
    assert (out / "topics" / "topic_peak.npy").exists()


def test_attribute_peaks_to_cistarget_at_scale():
    """The gene-only cistarget→peak bridge stalled at real-PBMC scale
    (35k cistarget × 30 targets × 5 peaks ≈ 5M Python row dicts via
    iterrows). The vectorised merge-based replacement must:
      1. Produce the same conceptual output (one row per
         (regulon, motif, peak_id) where the TF's GRN target is
         linked to that peak via enhancer correlation).
      2. Complete in seconds at 5k+ cistarget rows.
    """
    import time
    import pandas as pd
    from rustscenic.pipeline import _attribute_peaks_to_cistarget

    rng = np.random.default_rng(0)

    # 30 TFs × 50 targets each
    n_tfs = 30
    n_targets_per_tf = 50
    grn_rows = []
    for t in range(n_tfs):
        for tg in range(n_targets_per_tf):
            grn_rows.append({
                "TF": f"TF{t}",
                "target": f"GENE_{t}_{tg}",
                "importance": float(rng.uniform()),
            })
    grn = pd.DataFrame(grn_rows)

    # ~10 enhancer links per gene
    link_rows = []
    for t in range(n_tfs):
        for tg in range(n_targets_per_tf):
            for p in range(10):
                link_rows.append({
                    "peak_id": f"chr1:{t}_{tg}_{p}",
                    "gene": f"GENE_{t}_{tg}",
                    "correlation": 0.6,
                })
    enhancer_links = pd.DataFrame(link_rows)

    # 5,010 cistarget enrichments (~167 motifs / TF × 30 TFs)
    enriched_rows = []
    for t in range(n_tfs):
        for m in range(167):
            enriched_rows.append({
                "regulon": f"TF{t}_regulon",
                "motif": f"motif_{t}_{m}",
                "auc": 0.5,
            })
    enriched = pd.DataFrame(enriched_rows)

    t0 = time.monotonic()
    out = _attribute_peaks_to_cistarget(enriched, grn, enhancer_links)
    elapsed = time.monotonic() - t0

    # Pre-fix this stalled indefinitely on real PBMC; lock the regression
    assert elapsed < 30, f"bridge took {elapsed:.1f}s, regression"
    assert set(out.columns) == {"regulon", "motif", "peak_id", "auc"}
    # Magnitude check: 30 TFs × 167 motifs × 50 targets × 10 peaks per gene
    # = 2.5M rows. Allow a 50% lower bound for any drift.
    assert len(out) >= n_tfs * 167 * 50 * 10 // 2, (
        f"unexpectedly small output: {len(out)} rows"
    )
    assert set(out["regulon"].unique()) == set(enriched["regulon"].unique())


def test_attribute_peaks_to_cistarget_handles_empty():
    """Empty cistarget → empty output frame with the right schema."""
    import pandas as pd
    from rustscenic.pipeline import _attribute_peaks_to_cistarget

    enriched = pd.DataFrame(columns=["regulon", "motif", "auc"])
    grn = pd.DataFrame({"TF": [], "target": [], "importance": []})
    links = pd.DataFrame(columns=["peak_id", "gene", "correlation"])
    out = _attribute_peaks_to_cistarget(enriched, grn, links)
    assert list(out.columns) == ["regulon", "motif", "peak_id", "auc"]
    assert out.empty


def test_pipeline_run_topics_method_invalid(tmp_path):
    """Unknown topics_method raises a clear ValueError."""
    import gzip, anndata as ad, numpy as np, pandas as pd
    import rustscenic.pipeline
    import pytest

    rng = np.random.default_rng(0)
    rna = ad.AnnData(
        X=np.abs(rng.normal(size=(10, 5)).astype(np.float32)) + 0.1,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    frag_path = tmp_path / "fragments.tsv.gz"
    with gzip.open(frag_path, "wt") as fh:
        fh.write("chr1\t100\t200\tc0\t1\n")
    peaks_path = tmp_path / "peaks.bed"
    peaks_path.write_text("chr1\t100\t200\tpeak0\n")

    with pytest.raises(ValueError, match="topics_method"):
        rustscenic.pipeline.run(
            rna, tmp_path / "out",
            fragments=str(frag_path), peaks=str(peaks_path),
            tfs=["g0"], topics_n_topics=2,
            topics_method="not_a_method",
            verbose=False,
        )


def test_pipeline_run_uses_region_cistarget_when_supplied(tmp_path):
    """When `region_motif_rankings` is supplied, pipeline.run runs
    region-based cistarget against the linked peaks (exact path) instead
    of bridging via GRN ∩ enhancer (approximate path). Verifies the
    new region path is taken end-to-end."""
    import gzip, os, anndata as ad, numpy as np, pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(0)
    n_cells = 200
    cluster = np.array([i * 3 // n_cells for i in range(n_cells)], dtype=np.uint32)
    activity = np.zeros((3, n_cells), dtype=np.float32)
    for p in range(3):
        activity[p] = (cluster == p).astype(np.float32) + 0.1 * rng.normal(size=n_cells)

    rna_genes = [f"G{i:03d}" for i in range(30)]
    X = np.zeros((n_cells, 30), dtype=np.float32)
    for i in range(15):
        X[:, i] = activity[i // 5] + 0.2 * rng.normal(size=n_cells)
    for i in range(15, 30):
        X[:, i] = rng.normal(size=n_cells).astype(np.float32)
    X = np.clip(X, 0, None) + 0.1
    cells = [f"cell{i}" for i in range(n_cells)]
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cluster": cluster}, index=cells),
        var=pd.DataFrame(index=rna_genes),
    )
    frag_lines = []
    for p in range(3):
        for ci in np.where(cluster == p)[0]:
            for _ in range(15):
                start = 10_000 + p * 100_000 + int(rng.integers(0, 5_000))
                frag_lines.append(f"chr1\t{start}\t{start+150}\t{cells[ci]}\t1")
        for ci in np.where(cluster == p)[0]:
            for _ in range(3):
                start = int(rng.integers(0, 2_000_000))
                frag_lines.append(f"chr1\t{start}\t{start+120}\t{cells[ci]}\t1")
    frag_path = tmp_path / "fragments.tsv.gz"
    with gzip.open(frag_path, "wt") as fh:
        fh.write("\n".join(frag_lines) + "\n")

    peaks_path = tmp_path / "peaks.bed"
    peak_names = []
    with open(peaks_path, "w") as fh:
        for p in range(3):
            for j in range(3):
                start = 10_000 + p * 100_000 + j * 5_000
                name = f"peak_{p}_{j}"
                peak_names.append(name)
                fh.write(f"chr1\t{start}\t{start + 500}\t{name}\n")

    gene_coords = pd.DataFrame(
        [(f"G{i:03d}", "chr1", 10_000 + (i // 5) * 100_000 + 250) for i in range(15)],
        columns=["gene", "chrom", "tss"],
    )

    # Synthetic gene rankings (used for the gene-cistarget step)
    motif_names = ["M_G000", "M_G005", "M_G010"]
    n_genes = len(rna_genes)
    rank_matrix = np.full((len(motif_names), n_genes), n_genes - 1, dtype=np.int32)
    for tf_idx, motif in enumerate(motif_names):
        for rank, gene_idx in enumerate(
            [i for i in range(n_genes) if (i // 5 == tf_idx) and (i < 15)]
        ):
            rank_matrix[tf_idx, gene_idx] = rank
    motif_rankings = pd.DataFrame(rank_matrix, index=motif_names, columns=rna_genes)

    # Synthetic REGION rankings — same kernel, different feature set
    n_peaks = len(peak_names)
    region_rank = np.full((len(motif_names), n_peaks), n_peaks - 1, dtype=np.int32)
    for tf_idx in range(3):
        # programme tf_idx peaks: peak_{tf_idx}_*
        for rank, j in enumerate(
            [i for i, n in enumerate(peak_names) if n.startswith(f"peak_{tf_idx}_")]
        ):
            region_rank[tf_idx, j] = rank
    region_rankings = pd.DataFrame(region_rank, index=motif_names, columns=peak_names)

    out = tmp_path / "pipeline_out"
    result = rustscenic.pipeline.run(
        rna,
        out,
        fragments=str(frag_path),
        peaks=str(peaks_path),
        tfs=["G000", "G005", "G010"],
        motif_rankings=motif_rankings,
        region_motif_rankings=region_rankings,
        gene_coords=gene_coords,
        grn_n_estimators=15,
        grn_top_targets=10,
        topics_n_topics=5,
        topics_n_passes=2,
        cistarget_top_frac=0.3,
        cistarget_auc_threshold=0.0,
        enhancer_min_abs_corr=0.15,
        eregulon_min_target_genes=2,
        eregulon_min_enhancer_links=1,
        seed=0,
        verbose=False,
    )
    # All artifacts emitted including eregulons via region path
    assert result.atac_matrix_path.exists()
    assert result.cistarget_path.exists()
    assert result.enhancer_links_path.exists()
    assert result.eregulons_path.exists()
    assert result.n_eregulons is not None

    # Region-only should also work. The exact region-cistarget path must
    # not accidentally depend on gene-based motif rankings having run
    # first; real SCENIC+ users may bring only region ranking DBs for
    # eRegulon assembly.
    out_region_only = tmp_path / "pipeline_region_only"
    region_only = rustscenic.pipeline.run(
        rna,
        out_region_only,
        fragments=str(frag_path),
        peaks=str(peaks_path),
        tfs=["G000", "G005", "G010"],
        motif_rankings=None,
        region_motif_rankings=region_rankings,
        gene_coords=gene_coords,
        grn_n_estimators=15,
        grn_top_targets=10,
        topics_n_topics=5,
        topics_n_passes=2,
        cistarget_top_frac=0.3,
        cistarget_auc_threshold=0.0,
        enhancer_min_abs_corr=0.15,
        eregulon_min_target_genes=2,
        eregulon_min_enhancer_links=1,
        seed=0,
        verbose=False,
    )
    assert region_only.cistarget_path is not None
    assert region_only.cistarget_path.exists()
    assert region_only.eregulons_path is not None
    assert region_only.eregulons_path.exists()
    assert region_only.n_eregulons is not None


def test_coerce_rankings_accepts_aertslab_feather_path(tmp_path):
    """Aertslab motif-ranking feathers store motif IDs in a `motifs`
    column. Passing that file path directly to pipeline.run must work;
    the real PBMC benchmark used to hide this by pre-loading the file.
    """
    from rustscenic.pipeline import _coerce_rankings

    path = tmp_path / "genes_vs_motifs.rankings.feather"
    pd.DataFrame({
        "GATA1": [1, 2],
        "SPI1": [2, 1],
        "motifs": ["MOTIF_A", "MOTIF_B"],
    }).to_feather(path)

    rankings = _coerce_rankings(path)

    assert list(rankings.index) == ["MOTIF_A", "MOTIF_B"]
    assert list(rankings.columns) == ["GATA1", "SPI1"]


def test_coerce_rankings_accepts_first_column_motif_export(tmp_path):
    """Ad hoc parquet/CSV conversions often name the motif column
    something other than `motifs`. If the first column is strings and
    the remaining columns are numeric ranks, use it as the motif index.
    """
    from rustscenic.pipeline import _coerce_rankings

    path = tmp_path / "rankings.parquet"
    pd.DataFrame({
        "motif_id": ["M1", "M2"],
        "GENE1": [1, 2],
        "GENE2": [2, 1],
    }).to_parquet(path, index=False)

    rankings = _coerce_rankings(path)

    assert list(rankings.index) == ["M1", "M2"]
    assert list(rankings.columns) == ["GENE1", "GENE2"]
