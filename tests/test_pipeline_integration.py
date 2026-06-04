"""Full-pipeline integration test - preproc → grn → cistarget → enhancer → eregulon → aucell.

Simulates a minimal multiome workflow end-to-end and asserts every
stage connects to the next without silent breakage. This is the test
that would have caught the cellxgene ``var_names`` bug before Fuaad
did - it runs on a shape that includes a cellxgene-style ENSEMBL
AnnData in addition to the scanpy-native one.

The simulated biology:
  - 150 cells, 80 genes, 30 peaks
  - Three latent programmes drive correlated patches of expression
    AND accessibility
  - One TF in each programme is the intended "cause"
  - Correlated peaks carry the TF's synthetic motif in the cistarget
    ranking matrix

Every stage must survive at least one surviving regulon / eRegulon
for the test to pass - every downstream stage empties out if the
upstream one broke.
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
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

    # Gene TSS coordinates - programme-0 genes all live within 500 kb of
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
    the biology is known - three programmes with matched TFs, genes,
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
    assert not grn.empty, "GRN empty on cellxgene-shape RNA - resolver regressed"
    assert set(grn["TF"].unique()) == set(tf_names)

    # AUCell - uses the same resolve path
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
    import gzip, anndata as ad, numpy as np, pandas as pd
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

    # Fragments - dense per programme region, plus noise
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
    # Raw 10x fragments also contain observed non-cell barcodes. The
    # pipeline must drop these before topics, otherwise a full raw fragments
    # file can blow up memory and wall time.
    for j in range(5):
        start = 1_500_000 + j * 200
        frag_lines.append(f"chr1\t{start}\t{start+120}\tempty_bc_{j}\t1")
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

    # Synthetic motif rankings - each TF ranks its programme's genes high.
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
    atac_written = ad.read_h5ad(result.atac_matrix_path)
    assert atac_written.n_obs == n_cells
    assert set(atac_written.obs_names) == set(cells)
    assert result.grn_path.exists()
    assert result.aucell_path.exists()
    assert result.cistarget_path.exists()
    assert result.enhancer_links_path.exists()
    # eregulons file exists; n_eregulons may be 0 on synthetic data
    assert result.eregulons_path is not None
    assert result.eregulons_path.exists()
    assert result.n_eregulons is not None
    assert result.n_grn_edges is not None and result.n_grn_edges > 0
    assert result.n_cistarget_rows is not None and result.n_cistarget_rows > 0
    assert result.n_enhancer_links is not None and result.n_enhancer_links > 0
    assert result.n_eregulon_rows is not None
    assert result.aucell_shape == [n_cells, result.n_regulons]
    assert "grn" in result.memory
    assert "aucell" in result.memory
    import json
    manifest = json.loads((out / "manifest.json").read_text())
    assert "memory" in manifest
    assert "integrated_adata" in manifest["memory"]
    assert result.backend_execution["grn"]["symbols"] == [
        "gene_duplicate_summary",
        "grn_infer",
    ]
    assert result.backend_execution["preproc"]["symbols"] == [
        "preproc_fragments_to_matrix"
    ]
    assert result.backend_execution["topics"]["symbols"] == ["topics_fit"]
    assert result.backend_execution["cistarget"]["symbols"] == [
        "cistarget_enrichment_from_rankings_i32"
    ]
    assert result.backend_execution["enhancer"]["symbols"] == [
        "enhancer_align_cell_indices",
        "preproc_peak_coords_for_names",
        "enhancer_match_gene_coords_to_rna",
        "enhancer_normalise_chrom_codes",
        "enhancer_prepare_gene_order",
        "enhancer_link_pearson"
    ]
    assert result.backend_execution["eregulon_peak_attribution"]["symbols"] == [
        "pipeline_attribute_peaks_to_cistarget_rows_f32"
    ]
    assert result.backend_execution["eregulons"]["symbols"] == ["eregulon_assemble_f32"]
    assert result.backend_execution["aucell"]["symbols"] == [
        "gene_duplicate_summary",
        "stage_prepare_regulon_indices_with_coverage",
        "aucell_score",
    ]
    assert manifest["backend_execution"]["enhancer"]["symbols"] == [
        "enhancer_align_cell_indices",
        "preproc_peak_coords_for_names",
        "enhancer_match_gene_coords_to_rna",
        "enhancer_normalise_chrom_codes",
        "enhancer_prepare_gene_order",
        "enhancer_link_pearson"
    ]
    assert manifest["n_grn_edges"] == result.n_grn_edges
    assert manifest["aucell_shape"] == result.aucell_shape


def test_attach_aucell_to_obs_avoids_join_for_aligned_cells(monkeypatch):
    import rustscenic.pipeline

    cells = ["c0", "c1", "c2"]
    adata = ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"batch": ["a", "b", "c"], "TF1_regulon": [9.0, 9.0, 9.0]},
            index=cells,
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    auc = pd.DataFrame(
        {
            "TF1_regulon": [0.1, 0.2, 0.3],
            "TF2_regulon": [0.4, 0.5, 0.6],
        },
        index=cells,
    )

    def fail_join(*_args, **_kwargs):
        raise AssertionError("AUCell integration should not materialise obs.join")

    monkeypatch.setattr(pd.DataFrame, "join", fail_join)

    rustscenic.pipeline._attach_aucell_to_obs(adata, auc)

    assert list(adata.obs.columns) == ["batch", "TF1_regulon", "TF2_regulon"]
    np.testing.assert_allclose(adata.obs["TF1_regulon"].to_numpy(), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(adata.obs["TF2_regulon"].to_numpy(), [0.4, 0.5, 0.6])


def test_attach_aucell_to_obs_avoids_fragmented_column_inserts():
    import warnings

    import rustscenic.pipeline

    cells = ["c0", "c1", "c2"]
    adata = ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame({"batch": ["a", "b", "c"]}, index=cells),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    auc = pd.DataFrame(
        np.arange(3 * 500, dtype=np.float32).reshape(3, 500),
        index=cells,
        columns=[f"TF{i}_regulon" for i in range(500)],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        rustscenic.pipeline._attach_aucell_to_obs(adata, auc)

    assert not [
        warning for warning in caught
        if issubclass(warning.category, pd.errors.PerformanceWarning)
    ]
    assert adata.obs.shape == (3, 501)
    assert list(adata.obs.columns[:2]) == ["batch", "TF0_regulon"]
    np.testing.assert_allclose(adata.obs["TF499_regulon"].to_numpy(), auc["TF499_regulon"].to_numpy())


def test_subset_atac_to_rna_cells_uses_rust_indices_and_preserves_atac_order():
    import rustscenic.pipeline

    rna = ad.AnnData(
        X=np.zeros((2, 3), dtype=np.float32),
        obs=pd.DataFrame(index=["c0", "c2"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    atac = ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(4, 3),
        obs=pd.DataFrame(index=["empty", "c2", "c0", "other"]),
        var=pd.DataFrame(index=["p0", "p1", "p2"]),
    )
    log_lines: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        subset = rustscenic.pipeline._subset_atac_to_rna_cells(
            rna,
            atac,
            log=log_lines.append,
        )

    assert list(subset.obs_names) == ["c2", "c0"]
    np.testing.assert_array_equal(subset.X, atac.X[[1, 2]])
    assert any("subsetting ATAC from 4 barcodes to 2" in str(w.message) for w in caught)
    assert log_lines == ["      ATAC subset to RNA cells: (2, 3)"]


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


def test_pipeline_run_with_motif_annotations_scores_pruned_regulons(tmp_path):
    """When motif annotations are supplied, active regulons must be the
    annotation-pruned set rather than the raw GRN top-target candidates.
    """
    import json
    import anndata as ad
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(3)
    genes = ["TF_A", "TF_B"] + [f"G{i:02d}" for i in range(24)]
    X = rng.lognormal(mean=0.2, sigma=0.4, size=(90, len(genes))).astype("float32")
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=genes),
    )

    rankings = pd.DataFrame(
        np.tile(np.arange(len(genes), dtype=np.int32), (2, 1)),
        index=["M_TF_A", "M_TF_B"],
        columns=genes,
    )
    annotations = pd.DataFrame(
        {
            "motif": ["M_TF_A"],
            "TF": ["TF_A"],
        }
    )

    result = rustscenic.pipeline.run(
        rna,
        tmp_path,
        tfs=["TF_A", "TF_B"],
        motif_rankings=rankings,
        motif_annotations=annotations,
        grn_n_estimators=10,
        grn_top_targets=10,
        cistarget_top_frac=1.0,
        cistarget_auc_threshold=0.0,
        verbose=False,
    )

    candidates = json.loads(result.candidate_regulons_path.read_text())
    active = json.loads(result.regulons_path.read_text())
    pruned = json.loads(result.pruned_regulons_path.read_text())
    auc = pd.read_parquet(result.aucell_path)

    assert set(candidates) == {"TF_A_regulon", "TF_B_regulon"}
    assert set(active) == {"TF_A_regulon"}
    assert active == pruned
    assert list(auc.columns) == ["TF_A_regulon"]
    assert result.regulon_source == "motif_annotation_pruned"
    assert result.n_candidate_regulons == 2
    assert result.n_pruned_regulons == 1
    assert result.backend_execution["cistarget_pruning"]["symbols"] == [
        "cistarget_motif_annotation_prune_standard_rows_f32",
        "cistarget_prune_regulon_targets_i32",
    ]


def test_pipeline_run_without_motif_annotations_keeps_candidate_regulons(tmp_path):
    """Adding optional motif-annotation pruning must not change the
    historical cistarget path when no annotations are supplied."""
    import json
    import anndata as ad
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(5)
    genes = ["TF_A", "TF_B"] + [f"G{i:02d}" for i in range(24)]
    X = rng.lognormal(mean=0.2, sigma=0.4, size=(90, len(genes))).astype("float32")
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=genes),
    )
    rankings = pd.DataFrame(
        np.tile(np.arange(len(genes), dtype=np.int32), (2, 1)),
        index=["M_TF_A", "M_TF_B"],
        columns=genes,
    )

    result = rustscenic.pipeline.run(
        rna,
        tmp_path,
        tfs=["TF_A", "TF_B"],
        motif_rankings=rankings,
        motif_annotations=None,
        grn_n_estimators=10,
        grn_top_targets=10,
        cistarget_top_frac=1.0,
        cistarget_auc_threshold=0.0,
        verbose=False,
    )

    candidates = json.loads(result.candidate_regulons_path.read_text())
    active = json.loads(result.regulons_path.read_text())
    auc = pd.read_parquet(result.aucell_path)

    assert active == candidates
    assert result.regulon_source == "candidate_grn_top_targets"
    assert result.pruned_regulons_path is None
    assert result.n_pruned_regulons is None
    assert list(auc.columns) == list(active)


def test_attribute_peaks_normalises_compound_regulon_names():
    from rustscenic.pipeline import _attribute_peaks_to_cistarget

    enriched = pd.DataFrame(
        [{"regulon": "PAX5_regulon(+)", "motif": "m1", "auc": 0.2}]
    )
    enhancer_links = pd.DataFrame(
        [{"peak_id": "peak_1", "gene": "GENE_A"}]
    )
    regulons = {"PAX5_regulon(+)": ["GENE_A"]}

    out = _attribute_peaks_to_cistarget(
        enriched,
        enhancer_links,
        regulons=regulons,
    )

    assert out[["regulon", "motif", "peak_id"]].to_dict("records") == [
        {"regulon": "PAX5_regulon(+)", "motif": "m1", "peak_id": "peak_1"}
    ]


def test_pipeline_run_warns_when_motif_annotations_supplied_without_rankings(tmp_path):
    """``motif_annotations`` without ``motif_rankings`` is a silent-fail
    trap: pruning needs both, so the annotations would be ignored and the
    user thinks they got pruned regulons. Pipeline must warn loudly and
    keep the candidate regulon path."""
    import warnings as _warnings
    import anndata as ad
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(7)
    genes = ["TF_A", "TF_B"] + [f"G{i:02d}" for i in range(24)]
    X = rng.lognormal(0.2, 0.4, size=(90, len(genes))).astype("float32")
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(90)]),
        var=pd.DataFrame(index=genes),
    )
    annotations = tmp_path / "ignored_missing_annotations.tsv"

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = rustscenic.pipeline.run(
            rna,
            tmp_path,
            tfs=["TF_A", "TF_B"],
            motif_rankings=None,
            motif_annotations=annotations,
            grn_n_estimators=10,
            grn_top_targets=10,
            verbose=False,
        )

    relevant = [w for w in caught if "motif_annotations" in str(w.message)]
    assert relevant, "expected UserWarning about motif_annotations + missing motif_rankings"
    assert result.regulon_source == "candidate_grn_top_targets"
    assert result.pruned_regulons_path is None
    assert result.n_pruned_regulons is None


def test_pipeline_run_warns_and_falls_back_when_pruning_removes_all_regulons(tmp_path):
    """When motif annotations don't match any candidate TF, pruning removes
    every regulon. Pipeline must warn loudly and fall back to the candidate
    regulon set so AUCell isn't silently scored on zero columns."""
    import warnings as _warnings
    import anndata as ad
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(11)
    genes = ["TF_A", "TF_B"] + [f"G{i:02d}" for i in range(24)]
    X = rng.lognormal(0.2, 0.4, size=(90, len(genes))).astype("float32")
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(90)]),
        var=pd.DataFrame(index=genes),
    )
    rankings = pd.DataFrame(
        np.tile(np.arange(len(genes), dtype=np.int32), (1, 1)),
        index=["M_TF_A"],
        columns=genes,
    )
    # Annotation maps the only motif to a TF that doesn't appear in the GRN
    # candidate set, so prune_regulons returns {}.
    bogus_annotations = pd.DataFrame({"motif": ["M_TF_A"], "TF": ["UNKNOWN_TF"]})

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = rustscenic.pipeline.run(
            rna,
            tmp_path,
            tfs=["TF_A", "TF_B"],
            motif_rankings=rankings,
            motif_annotations=bogus_annotations,
            grn_n_estimators=10,
            grn_top_targets=10,
            cistarget_top_frac=1.0,
            cistarget_auc_threshold=0.0,
            verbose=False,
        )

    relevant = [w for w in caught if "removed all" in str(w.message)]
    assert relevant, "expected UserWarning that pruning removed all regulons"
    auc = pd.read_parquet(result.aucell_path)
    assert auc.shape[1] > 0, "AUCell must not be empty after pruning fallback"
    assert result.regulon_source == "candidate_grn_top_targets_after_failed_pruning"
    assert result.n_pruned_regulons == 0
    assert result.n_regulons == result.n_candidate_regulons
    assert result.pruned_regulons_path is None, (
        "pruned_regulons_path must be None on the fallback path; a non-None "
        "path here would mislead callers into reading an empty-dict JSON as "
        "a successful pruning result"
    )
    assert not (tmp_path / "pruned_regulons.json").exists(), (
        "no pruned_regulons.json should be written when pruning removed all "
        "candidates"
    )


def test_pipeline_run_fallback_removes_stale_pruned_regulons_json(tmp_path):
    """Re-running ``pipeline.run`` on the same output_dir, first with
    annotations that produce a non-empty pruned set and then with
    annotations that fall back, must remove the stale ``pruned_regulons.json``
    from the first run. The PipelineResult and manifest already report None
    on the fallback path; leaving the stale file behind misleads any caller
    that probes the filesystem directly.
    """
    import warnings as _warnings
    import anndata as ad
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(13)
    genes = ["TF_A", "TF_B"] + [f"G{i:02d}" for i in range(24)]
    X = rng.lognormal(0.2, 0.4, size=(90, len(genes))).astype("float32")
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(90)]),
        var=pd.DataFrame(index=genes),
    )
    rankings = pd.DataFrame(
        np.tile(np.arange(len(genes), dtype=np.int32), (1, 1)),
        index=["M_TF_A"],
        columns=genes,
    )
    matching_annotations = pd.DataFrame(
        {"motif": ["M_TF_A"], "TF": ["TF_A"]}
    )
    bogus_annotations = pd.DataFrame(
        {"motif": ["M_TF_A"], "TF": ["UNKNOWN_TF"]}
    )

    common = {
        "rna": rna,
        "output_dir": tmp_path,
        "tfs": ["TF_A", "TF_B"],
        "motif_rankings": rankings,
        "grn_n_estimators": 10,
        "grn_top_targets": 10,
        "cistarget_top_frac": 1.0,
        "cistarget_auc_threshold": 0.0,
        "verbose": False,
    }

    # Run 1: matching annotations -> pruned_regulons.json is written
    result1 = rustscenic.pipeline.run(motif_annotations=matching_annotations, **common)
    pruned_path = tmp_path / "pruned_regulons.json"
    assert result1.pruned_regulons_path is not None, (
        "matching annotations should produce a non-None pruned_regulons_path; "
        "if this fails the fixture itself is wrong (no pruned regulons survived)"
    )
    assert pruned_path.exists(), (
        "first run should write pruned_regulons.json when annotations match"
    )

    # Run 2: bogus annotations -> fallback, stale file must be removed
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result2 = rustscenic.pipeline.run(motif_annotations=bogus_annotations, **common)

    assert any("removed all" in str(w.message) for w in caught), (
        "expected the fallback warning on run 2"
    )
    assert result2.regulon_source == "candidate_grn_top_targets_after_failed_pruning"
    assert result2.pruned_regulons_path is None
    assert not pruned_path.exists(), (
        "stale pruned_regulons.json from run 1 must be removed when run 2 "
        "takes the fallback path; otherwise filesystem probes return data "
        "that contradicts the current run's regulon_source"
    )


def test_pipeline_run_cistarget_nes_threshold_filters_enriched(tmp_path):
    """Pipeline cistarget_nes_threshold should reach cistarget.enrich and
    filter enriched rows by NES. Setting an unrealistically high threshold
    should drop most rows; the run should still complete without errors and
    the cistarget_enriched.parquet should reflect the filter."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline

    rng = np.random.default_rng(17)
    genes = ["TF_A", "TF_B"] + [f"G{i:02d}" for i in range(28)]
    X = rng.lognormal(0.2, 0.4, size=(90, len(genes))).astype("float32")
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(90)]),
        var=pd.DataFrame(index=genes),
    )
    # 60 motifs is comfortably above the 30-motif floor so NES is computed,
    # not NaN. Motif 0 placed regulon genes (TF_A targets at top) so NES is
    # high; the other 59 motifs are random permutations.
    n_motifs = 60
    rank_rows = []
    for i in range(n_motifs):
        if i == 0:
            perm = ["TF_A", "TF_B"] + [f"G{j:02d}" for j in range(28)]
        else:
            perm = list(rng.permutation(genes))
        rank_rows.append([perm.index(g) for g in genes])
    rankings = pd.DataFrame(
        np.asarray(rank_rows, dtype=np.int32),
        index=[f"m{i}" for i in range(n_motifs)],
        columns=genes,
    )

    base_kwargs = {
        "rna": rna,
        "output_dir": tmp_path,
        "tfs": ["TF_A", "TF_B"],
        "motif_rankings": rankings,
        "grn_n_estimators": 10,
        "grn_top_targets": 10,
        "cistarget_top_frac": 0.2,
        "cistarget_auc_threshold": 0.0,
        "verbose": False,
    }

    # Run with NES filter at 3.0: should keep only motifs where the regulon
    # has a strong z-score lift.
    result_filtered = rustscenic.pipeline.run(
        **base_kwargs, cistarget_nes_threshold=3.0,
    )
    enriched_filtered = pd.read_parquet(result_filtered.cistarget_path)
    assert "nes" in enriched_filtered.columns, (
        "v0.4.4 cistarget_enriched.parquet must include the NES column"
    )
    if len(enriched_filtered) > 0:
        assert (enriched_filtered["nes"] >= 3.0).all(), (
            "every enriched row must satisfy the NES filter when "
            "cistarget_nes_threshold=3.0"
        )

    # Re-run with no filter; row count must be greater than or equal to
    # the filtered row count.
    import shutil
    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    result_unfiltered = rustscenic.pipeline.run(**base_kwargs)
    enriched_unfiltered = pd.read_parquet(result_unfiltered.cistarget_path)
    assert "nes" in enriched_unfiltered.columns
    assert len(enriched_unfiltered) >= len(enriched_filtered), (
        "unfiltered cistarget output must contain at least as many rows as "
        "the NES-filtered output on the same input"
    )


def test_region_cistarget_helper_honors_nes_threshold():
    """The region-cistarget path inside pipeline.run must honour
    ``cistarget_nes_threshold`` too. Caught post-merge by codex on PR #76:
    v0.4.4 wired NES through the gene path but the region helper
    ``_region_cistarget_with_peak_ids`` only took ``top_frac`` and
    ``auc_threshold``, so any caller supplying ``cistarget_nes_threshold``
    plus ``region_motif_rankings`` silently got an unfiltered region
    enrich frame, contradicting the v0.4.4 release claim. Red-green test.
    """
    import numpy as np
    import pandas as pd
    from rustscenic.pipeline import _region_cistarget_with_peak_ids

    rng = np.random.default_rng(23)
    n_motifs = 60
    n_regions = 100
    region_names = [f"p{i:03d}" for i in range(n_regions)]
    motif_names = [f"m{i}" for i in range(n_motifs)]
    # Motif 0 ranks the regulon's peaks (p000..p019) at the top, so its
    # AUC is high. Every other motif gets a random permutation.
    rank_rows = []
    for i in range(n_motifs):
        if i == 0:
            perm = [f"p{j:03d}" for j in range(20)] + [
                f"p{j:03d}" for j in range(20, n_regions)
            ]
        else:
            perm = [region_names[k] for k in rng.permutation(n_regions)]
        rank_rows.append([perm.index(r) for r in region_names])
    region_rankings = pd.DataFrame(
        np.asarray(rank_rows, dtype=np.int32),
        index=motif_names,
        columns=region_names,
    )
    peak_regulons = [("TF_A_regulon", [f"p{i:03d}" for i in range(20)])]

    # No NES threshold: all enriched rows survive (60 motifs minus those
    # below auc_threshold 0.0 which is none).
    region_enrich_no_nes, _ = _region_cistarget_with_peak_ids(
        region_rankings, peak_regulons, top_frac=0.2, auc_threshold=0.0,
    )
    assert len(region_enrich_no_nes) == n_motifs
    # NES column must be present even without filtering.
    assert "nes" in region_enrich_no_nes.columns

    # NES threshold 3.0: motif 0 is the only one with z-score >> 3 on this
    # synthetic fixture (verified separately by cistarget.enrich tests).
    region_enrich_with_nes, _ = _region_cistarget_with_peak_ids(
        region_rankings, peak_regulons,
        top_frac=0.2, auc_threshold=0.0, nes_threshold=3.0,
    )
    assert len(region_enrich_with_nes) < len(region_enrich_no_nes), (
        f"NES filter must reduce region cistarget rows; got "
        f"{len(region_enrich_with_nes)} with NES vs "
        f"{len(region_enrich_no_nes)} without"
    )
    assert (region_enrich_with_nes["nes"] >= 3.0).all(), (
        "every surviving row must satisfy the NES floor"
    )
    assert "m0" in set(region_enrich_with_nes["motif"]), (
        "the true-positive motif m0 should survive the NES filter on this "
        "fixture; if it does not, the synthetic design or the NES wiring "
        "has regressed"
    )


def test_region_cistarget_peak_attribution_matches_pandas_reference():
    """Rust region attribution must match the previous pandas melt/merge path."""
    import numpy as np
    import pandas as pd
    from rustscenic.pipeline import _region_cistarget_with_peak_ids

    base = np.array(
        [
            [0, 1, 4, 5, 7],
            [4, 0, 1, 6, 7],
            [5, 6, 0, 1, 7],
            [7, 6, 5, 4, 0],
        ],
        dtype=np.int32,
    )
    peak_regulons = [
        ("TF_A_regulon", ["p0", "p1", "missing_peak"]),
        ("TF_B_regulon", ["p2", "p3"]),
    ]

    for values in (base, base.astype(np.int16), base.astype(np.float64)):
        region_rankings = pd.DataFrame(
            values,
            index=["motif_a", "motif_b", "motif_c", "motif_noise"],
            columns=["p0", "p1", "p2", "p3", "unused"],
        )
        region_enrich, got = _region_cistarget_with_peak_ids(
            region_rankings,
            peak_regulons,
            top_frac=0.4,
            auc_threshold=0.0,
        )
        expected = _reference_region_peak_attribution(
            region_rankings,
            peak_regulons,
            region_enrich,
            top_frac=0.4,
        )
        pd.testing.assert_frame_equal(
            _normalise_region_attribution(got),
            _normalise_region_attribution(expected),
            check_dtype=False,
        )


def test_region_cistarget_with_peak_ids_preserves_float32_auc_dtype():
    import numpy as np
    import pandas as pd
    from rustscenic.pipeline import _region_cistarget_with_peak_ids

    region_rankings = pd.DataFrame(
        np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32),
        index=["motif_a", "motif_b"],
        columns=["p0", "p1", "p2"],
    )
    peak_regulons = [("TF_A_regulon", ["p0", "p1"])]

    region_enrich, attributed = _region_cistarget_with_peak_ids(
        region_rankings,
        peak_regulons,
        top_frac=0.67,
        auc_threshold=0.0,
    )

    assert region_enrich["auc"].dtype == np.float32
    assert attributed["auc"].dtype == np.float32
    assert attributed.attrs["rust_backend"] == {
        "engine": "rust",
        "symbols": [
            "cistarget_enrichment_from_rankings_i32",
            "cistarget_region_attribution_peak_values_i32",
            "pipeline_expand_region_cistarget_rows_f32",
        ],
    }


def test_region_rankings_helper_keeps_strided_integer_buffers():
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline as pipeline
    from rustscenic.pipeline import _region_rankings_kernel_arg

    values = np.asfortranarray(
        np.array(
            [
                [0, 1, 4, 5],
                [4, 0, 1, 5],
                [5, 4, 0, 1],
            ],
            dtype=np.int16,
        )
    )
    rankings = pd.DataFrame(
        values,
        index=["motif_a", "motif_b", "motif_c"],
        columns=["p0", "p1", "p2", "p3"],
    )
    rankings_arg, kernel, cutoff = _region_rankings_kernel_arg(rankings, rank_cutoff=2)

    assert np.shares_memory(rankings_arg, values)
    assert rankings_arg.flags.f_contiguous
    assert not rankings_arg.flags.c_contiguous
    assert kernel is pipeline._cistarget_region_attribution_i16
    assert cutoff == 2

    rankings_arg, kernel, cutoff = _region_rankings_kernel_arg(rankings, rank_cutoff=50_000)

    assert np.shares_memory(rankings_arg, values)
    assert kernel is pipeline._cistarget_region_attribution_i16
    assert cutoff == np.iinfo(np.int16).max


def test_region_rankings_helper_keeps_int32_buffers_without_downcast():
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline as pipeline
    from rustscenic.pipeline import _region_rankings_kernel_arg

    values = np.asfortranarray(
        np.array(
            [
                [0, 1, 4, 5],
                [4, 0, 1, 5],
                [5, 4, 0, 1],
            ],
            dtype=np.int32,
        )
    )
    rankings = pd.DataFrame(
        values,
        index=["motif_a", "motif_b", "motif_c"],
        columns=["p0", "p1", "p2", "p3"],
    )
    rankings_arg, kernel, cutoff = _region_rankings_kernel_arg(rankings, rank_cutoff=2)

    assert np.shares_memory(rankings_arg, values)
    assert rankings_arg.flags.f_contiguous
    assert not rankings_arg.flags.c_contiguous
    assert kernel is pipeline._cistarget_region_attribution_i32
    assert cutoff == 2


def test_region_rankings_helper_keeps_int64_buffers_without_downcast():
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline as pipeline
    from rustscenic.pipeline import _region_rankings_kernel_arg

    values = np.asfortranarray(
        np.array(
            [
                [0, 1, 4, 5],
                [4, 0, 1, 5],
                [5, 4, 0, 1],
            ],
            dtype=np.int64,
        )
    )
    rankings = pd.DataFrame(
        values,
        index=["motif_a", "motif_b", "motif_c"],
        columns=["p0", "p1", "p2", "p3"],
    )
    rankings_arg, kernel, cutoff = _region_rankings_kernel_arg(rankings, rank_cutoff=2)

    assert np.shares_memory(rankings_arg, values)
    assert rankings_arg.flags.f_contiguous
    assert not rankings_arg.flags.c_contiguous
    assert kernel is pipeline._cistarget_region_attribution_i64
    assert cutoff == 2


def test_region_rankings_helper_converts_float_rankings_in_rust(monkeypatch):
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline as pipeline
    from rustscenic.pipeline import _region_rankings_kernel_arg

    values = np.asfortranarray(
        np.array(
            [
                [0, 1, 4, 5],
                [4, 0, 1, 5],
                [5, 4, 0, 1],
            ],
            dtype=np.float64,
        )
    )
    rankings = pd.DataFrame(
        values,
        index=["motif_a", "motif_b", "motif_c"],
        columns=["p0", "p1", "p2", "p3"],
    )

    def fake_to_i32(values_arg):
        assert np.shares_memory(values_arg, values)
        assert values_arg.flags.f_contiguous
        assert not values_arg.flags.c_contiguous
        return values.astype(np.int32)

    monkeypatch.setattr(pipeline, "_rankings_to_i32_f64", fake_to_i32)

    rankings_arg, kernel, cutoff = _region_rankings_kernel_arg(rankings, rank_cutoff=2)

    assert rankings_arg.dtype == np.int32
    assert kernel is pipeline._cistarget_region_attribution_i32
    assert cutoff == 2


def test_region_cistarget_attribution_passes_full_rankings_to_rust(monkeypatch):
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline as pipeline

    values = np.array(
        [
            [0, 4, 1],
            [3, 0, 2],
        ],
        dtype=np.int32,
    )
    region_rankings = pd.DataFrame(
        values,
        index=["motif_a", "motif_b"],
        columns=["p0", "unused_peak", "p1"],
    )
    captured = {}

    def fake_region_kernel(
        rankings,
        motif_names,
        peak_names,
        peak_regulon_names,
        peak_regulon_peaks,
        enriched_regulons,
        enriched_motifs,
        rank_cutoff,
    ):
        captured["shape"] = rankings.shape
        captured["shares_memory"] = np.shares_memory(rankings, values)
        captured["peak_names"] = peak_names
        return np.array([], dtype=np.uint64), []

    monkeypatch.setattr(
        pipeline,
        "_cistarget_region_peak_values_i32",
        fake_region_kernel,
    )

    region_enrich, attributed = pipeline._region_cistarget_with_peak_ids(
        region_rankings,
        [("TF_A_regulon", ["p0"])],
        top_frac=0.5,
        auc_threshold=0.0,
    )

    assert not region_enrich.empty
    assert attributed.empty
    assert captured["shape"] == values.shape
    assert captured["shares_memory"]
    assert captured["peak_names"] == ["p0", "unused_peak", "p1"]


def test_region_cistarget_uses_rust_row_expander(monkeypatch):
    import numpy as np
    import pandas as pd
    import rustscenic.pipeline as pipeline

    region_rankings = pd.DataFrame(
        np.array([[0, 1], [1, 0]], dtype=np.int32),
        index=["motif_a", "motif_b"],
        columns=["p0", "p1"],
    )
    captured = {}

    def fake_region_kernel(*args):
        return np.array([1, 0], dtype=np.uint64), ["p1", "p0"]

    def fake_expand_rows(row_indices, peak_ids, enriched_regulons, enriched_motifs, enriched_aucs):
        captured["row_indices"] = row_indices
        captured["peak_ids"] = peak_ids
        captured["regulons"] = enriched_regulons
        captured["motifs"] = enriched_motifs
        captured["auc_dtype"] = enriched_aucs.dtype
        return (
            ["TF_A_regulon", "TF_A_regulon"],
            ["motif_b", "motif_a"],
            ["p1", "p0"],
            np.array([0.75, 0.5], dtype=np.float32),
        )

    monkeypatch.setattr(pipeline, "_cistarget_region_peak_values_i32", fake_region_kernel)
    monkeypatch.setattr(pipeline, "_pipeline_expand_region_rows_f32", fake_expand_rows)

    _, attributed = pipeline._region_cistarget_with_peak_ids(
        region_rankings,
        [("TF_A_regulon", ["p0", "p1"])],
        top_frac=1.0,
        auc_threshold=0.0,
    )

    np.testing.assert_array_equal(captured["row_indices"], np.array([1, 0], dtype=np.uint64))
    assert captured["peak_ids"] == ["p1", "p0"]
    assert captured["regulons"] == ["TF_A_regulon", "TF_A_regulon"]
    assert captured["motifs"] == ["motif_a", "motif_b"]
    assert captured["auc_dtype"] == np.float32
    assert attributed.to_dict("list") == {
        "regulon": ["TF_A_regulon", "TF_A_regulon"],
        "motif": ["motif_b", "motif_a"],
        "peak_id": ["p1", "p0"],
        "auc": [np.float32(0.75), np.float32(0.5)],
    }
    assert attributed.attrs["rust_backend"]["symbols"] == [
        "cistarget_enrichment_from_rankings_i32",
        "cistarget_region_attribution_peak_values_i32",
        "pipeline_expand_region_cistarget_rows_f32",
    ]


def _reference_region_peak_attribution(
    region_rankings: pd.DataFrame,
    peak_regulons: list[tuple[str, list[str]]],
    region_enrich: pd.DataFrame,
    *,
    top_frac: float,
) -> pd.DataFrame:
    n_regions = region_rankings.shape[1]
    rank_cutoff = max(1, int(np.ceil(top_frac * n_regions)))
    peak_long = pd.DataFrame(
        [(name, p) for name, peaks in peak_regulons for p in peaks],
        columns=["regulon", "peak_id"],
    )
    if peak_long.empty:
        return pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])

    needed_peaks = set(peak_long["peak_id"].astype(str))
    rank_cols = [p for p in region_rankings.columns if str(p) in needed_peaks]
    if not rank_cols:
        return pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])

    rank_long = (
        region_rankings[rank_cols]
        .reset_index()
        .melt(
            id_vars=region_rankings.index.name or "index",
            var_name="peak_id",
            value_name="rank",
        )
    )
    rank_long.columns = ["motif", "peak_id", "rank"]
    rank_long = rank_long[rank_long["rank"].astype(float) <= rank_cutoff]

    enriched = (
        region_enrich.merge(peak_long, on="regulon", how="inner")
        .merge(rank_long[["motif", "peak_id"]], on=["motif", "peak_id"], how="inner")
    )
    if enriched.empty:
        return pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])
    return enriched[["regulon", "motif", "peak_id", "auc"]].copy()


def _normalise_region_attribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["regulon", "motif", "peak_id", "auc"])
        .reset_index(drop=True)
    )


def test_filter_cistarget_peak_rows_matches_pandas_merge_reference():
    from rustscenic.pipeline import _filter_cistarget_peak_rows

    enriched_with_peaks = pd.DataFrame(
        {
            "regulon": ["TF1_regulon", "TF2_regulon", "TF1_regulon", "TF3_regulon"],
            "motif": ["m1", "m2", "m3", "m4"],
            "peak_id": ["p1", "p2", "p3", "p4"],
            "auc": np.asarray([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        }
    )
    prior_symbols = [
        "cistarget_enrichment_from_rankings_i32",
        "cistarget_region_attribution_peak_values_i32",
        "pipeline_expand_region_cistarget_rows_f32",
    ]
    enriched_with_peaks.attrs["rust_backend"] = {
        "engine": "rust",
        "symbols": prior_symbols,
    }
    keep = pd.DataFrame(
        {
            "regulon": ["TF1_regulon", "TF1_regulon", "TF2_regulon"],
            "motif": ["m3", "m3", "m2"],
        }
    )

    expected = enriched_with_peaks.merge(
        keep.drop_duplicates(),
        on=["regulon", "motif"],
        how="inner",
    )
    got = _filter_cistarget_peak_rows(enriched_with_peaks, keep)

    pd.testing.assert_frame_equal(got, expected)
    assert got.attrs["rust_backend"]["symbols"] == prior_symbols + [
        "pipeline_filter_cistarget_peak_rows_f32"
    ]


def test_filter_cistarget_peak_rows_uses_rust_row_helper(monkeypatch):
    import rustscenic.pipeline as pipeline

    enriched_with_peaks = pd.DataFrame(
        {
            "regulon": ["TF1_regulon", "TF2_regulon"],
            "motif": ["m1", "m2"],
            "peak_id": ["p1", "p2"],
            "auc": np.asarray([0.8, 0.7], dtype=np.float32),
        }
    )
    keep = pd.DataFrame({"regulon": ["TF2_regulon"], "motif": ["m2"]})

    def fake_filter_rows(row_regulons, row_motifs, row_peaks, row_aucs, keep_regulons, keep_motifs):
        assert row_regulons == ["TF1_regulon", "TF2_regulon"]
        assert row_motifs == ["m1", "m2"]
        assert row_peaks == ["p1", "p2"]
        assert row_aucs.dtype == np.float32
        assert keep_regulons == ["TF2_regulon"]
        assert keep_motifs == ["m2"]
        return (
            ["TF2_regulon"],
            ["m2"],
            ["p2"],
            np.asarray([0.7], dtype=np.float32),
        )

    monkeypatch.setattr(
        pipeline,
        "_pipeline_filter_cistarget_peak_rows_f32",
        fake_filter_rows,
    )

    got = pipeline._filter_cistarget_peak_rows(enriched_with_peaks, keep)

    assert got.to_dict("list") == {
        "regulon": ["TF2_regulon"],
        "motif": ["m2"],
        "peak_id": ["p2"],
        "auc": [np.float32(0.7)],
    }
    assert got["auc"].dtype == np.float32
    assert got.attrs["rust_backend"]["symbols"] == [
        "pipeline_filter_cistarget_peak_rows_f32"
    ]


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

    # Sparse fragments file - enough for the topics fit to have signal.
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
    assert (out / "topics" / "cell_topic.parquet").exists()
    assert (out / "topics" / "topic_peak.parquet").exists()


def test_pipeline_grn_top_targets_below_ten_still_builds_candidates(tmp_path, monkeypatch):
    import json
    import rustscenic.pipeline

    cells = [f"cell{i}" for i in range(8)]
    genes = ["TF1"] + [f"G{i}" for i in range(5)]
    rna = ad.AnnData(
        X=np.ones((len(cells), len(genes)), dtype=np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )

    def fake_infer(*_args, **_kwargs):
        return pd.DataFrame(
            {
                "TF": ["TF1"] * 5,
                "target": [f"G{i}" for i in range(5)],
                "importance": [5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )

    def fake_score(expression, regulons, *, top_frac):
        assert regulons == [("TF1_regulon", [f"G{i}" for i in range(5)])]
        return pd.DataFrame({"TF1_regulon": np.zeros(expression.n_obs)}, index=expression.obs_names)

    monkeypatch.setattr(rustscenic.grn, "infer", fake_infer)
    monkeypatch.setattr(rustscenic.aucell, "score", fake_score)

    result = rustscenic.pipeline.run(
        rna,
        tmp_path,
        tfs=["TF1"],
        grn_top_targets=5,
        grn_n_estimators=2,
        verbose=False,
    )

    candidates = json.loads(result.candidate_regulons_path.read_text())
    assert candidates == {"TF1_regulon": [f"G{i}" for i in range(5)]}
    assert result.n_candidate_regulons == 1
    assert result.n_regulons == 1


def test_candidate_regulons_from_grn_keeps_top_targets_without_per_tf_scan():
    import rustscenic.pipeline

    grn = pd.DataFrame(
        {
            "TF": ["TF1", "TF1", "TF2", "TF1", "TF2", "TF2", "TF3"],
            "target": ["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
            "importance": [0.2, 0.9, 0.1, 0.5, 0.8, 0.7, 1.0],
        }
    )

    out = rustscenic.pipeline._candidate_regulons_from_grn(
        grn,
        top_targets=2,
        min_targets=2,
    )

    assert out == {
        "TF1_regulon": ["G2", "G4"],
        "TF2_regulon": ["G5", "G6"],
    }


def _reference_candidate_regulons_from_grn_pandas(
    grn: pd.DataFrame,
    *,
    top_targets: int,
    min_targets: int,
) -> dict[str, list[str]]:
    if grn.empty:
        return {}
    top = (
        grn.sort_values("importance", ascending=False, kind="mergesort")
        .groupby("TF", sort=False, group_keys=False)
        .head(top_targets)
    )
    return {
        f"{tf}_regulon": group["target"].tolist()
        for tf, group in top.groupby("TF", sort=False)
        if len(group) >= min_targets
    }


def test_candidate_regulons_from_grn_matches_pandas_reference_ties_and_nan():
    import rustscenic.pipeline

    grn = pd.DataFrame(
        {
            "TF": ["TF1", "TF2", "TF1", "TF1", "TF2", "TF3", "TF3"],
            "target": ["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
            "importance": [0.5, 0.9, 0.5, np.nan, 0.2, np.nan, 0.1],
        }
    )

    got = rustscenic.pipeline._candidate_regulons_from_grn(
        grn,
        top_targets=2,
        min_targets=2,
    )
    expected = _reference_candidate_regulons_from_grn_pandas(
        grn,
        top_targets=2,
        min_targets=2,
    )

    assert got == expected


def test_peak_regulons_from_edges_uses_rust_bridge_without_groupby_sets():
    import rustscenic.pipeline

    grn = pd.DataFrame(
        {
            "TF": ["TF1", "TF1", "TF2", "TF1", "TF3"],
            "target": ["G1", "G2", "G2", "G1", "missing"],
            "importance": [1.0, 0.9, 0.8, 0.7, 0.1],
        }
    )
    enhancer_links = pd.DataFrame(
        {
            "gene": ["G1", "G1", "G2", "G2", "other"],
            "peak_id": ["p1", "p1", "p2", "p3", "p4"],
        }
    )

    assert rustscenic.pipeline._peak_regulons_from_edges(grn, enhancer_links) == [
        ("TF1_regulon", ["p1", "p2", "p3"]),
        ("TF2_regulon", ["p2", "p3"]),
    ]


def test_peak_regulons_and_projection_features_uses_rust_unique_sort():
    import rustscenic.pipeline

    grn = pd.DataFrame(
        {
            "TF": ["TF1", "TF1", "TF2", "TF2"],
            "target": ["G1", "G2", "G2", "G3"],
            "importance": [1.0, 0.9, 0.8, 0.7],
        }
    )
    enhancer_links = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G2", "G3"],
            "peak_id": ["pB", "pA", "pC", "pB"],
        }
    )

    peak_regulons, features = rustscenic.pipeline._peak_regulons_and_projection_features(
        grn, enhancer_links
    )

    assert peak_regulons == [
        ("TF1_regulon", ["pB", "pA", "pC"]),
        ("TF2_regulon", ["pA", "pC", "pB"]),
    ]
    assert features == ["pA", "pB", "pC"]


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

    # 30 TFs × 50 targets each
    n_tfs = 30
    n_targets_per_tf = 50
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
    regulons = {
        f"TF{t}_regulon": [f"GENE_{t}_{tg}" for tg in range(n_targets_per_tf)]
        for t in range(n_tfs)
    }

    t0 = time.monotonic()
    out = _attribute_peaks_to_cistarget(enriched, enhancer_links, regulons)
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


def _reference_attribute_peaks_to_cistarget_pandas(
    enriched: pd.DataFrame,
    enhancer_links: pd.DataFrame,
    regulons,
) -> pd.DataFrame:
    from rustscenic._stage_utils import iter_regulon_pairs, tf_from_regulon_name

    if enriched.empty:
        return pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])
    tf_target_rows = []
    for regulon_name, targets in iter_regulon_pairs(regulons):
        tf = tf_from_regulon_name(str(regulon_name))
        for g in targets:
            tf_target_rows.append((tf, str(g)))
    tf_target = pd.DataFrame(tf_target_rows, columns=["tf", "gene"])

    gene_peak = pd.DataFrame({
        "gene": [str(v) for v in enhancer_links["gene"].to_numpy(copy=False)],
        "peak_id": [str(v) for v in enhancer_links["peak_id"].to_numpy(copy=False)],
    }).drop_duplicates()
    tf_peak = tf_target.merge(gene_peak, on="gene", how="inner")[["tf", "peak_id"]]
    tf_peak = tf_peak.drop_duplicates()

    ct_cols = {
        "regulon": enriched["regulon"].to_numpy(copy=False),
        "tf": [tf_from_regulon_name(str(v)) for v in enriched["regulon"].to_numpy(copy=False)],
        "auc": enriched["auc"].to_numpy(copy=False),
    }
    cols = ["regulon", "tf", "auc"]
    if "motif" in enriched.columns:
        ct_cols["motif"] = enriched["motif"].to_numpy(copy=False)
        cols.insert(2, "motif")
    ct = pd.DataFrame(ct_cols, columns=cols)

    out = ct.merge(tf_peak, on="tf", how="inner").drop(columns=["tf"])
    if "motif" not in out.columns:
        out["motif"] = None
    return out[["regulon", "motif", "peak_id", "auc"]].reset_index(drop=True)


def test_attribute_peaks_to_cistarget_matches_pandas_reference_with_regulons():
    from rustscenic.pipeline import _attribute_peaks_to_cistarget

    enriched = pd.DataFrame(
        {
            "regulon": [
                "TF_A_extended_regulon",
                "TF_B_regulon",
                "TF_A_extended_regulon",
                "TF_C_regulon",
            ],
            "motif": ["m_a1", "m_b", "m_a2", "m_c"],
            "auc": [0.8, 0.5, 0.9, 0.1],
        }
    )
    regulons = {
        "TF_A_extended_regulon": ["G2", "G1", "G1"],
        "TF_B_regulon": ["G3"],
        "TF_D_regulon": ["G9"],
    }
    enhancer_links = pd.DataFrame(
        {
            "gene": ["G1", "G1", "G2", "G1", "G3", "G9"],
            "peak_id": ["p1", "p1", "p2", "p3", "p4", "p9"],
            "correlation": [0.7, 0.7, 0.6, 0.5, 0.8, 0.2],
        }
    )

    got = _attribute_peaks_to_cistarget(enriched, enhancer_links, regulons=regulons)
    expected = _reference_attribute_peaks_to_cistarget_pandas(
        enriched, enhancer_links, regulons=regulons,
    )
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_attribute_peaks_to_cistarget_preserves_float32_auc_dtype():
    from rustscenic.pipeline import _attribute_peaks_to_cistarget

    auc_values = np.array([0.8, 0.5], dtype=np.float32)
    enriched = pd.DataFrame(
        {
            "regulon": ["TF_A_regulon", "TF_B_regulon"],
            "motif": ["m_a", "m_b"],
            "auc": auc_values,
        }
    )
    enhancer_links = pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "peak_id": ["p1", "p2"],
            "correlation": [0.7, 0.6],
        }
    )
    regulons = {"TF_A_regulon": ["G1"], "TF_B_regulon": ["G2"]}

    got = _attribute_peaks_to_cistarget(enriched, enhancer_links, regulons)

    assert got["auc"].dtype == np.float32
    assert got.attrs["rust_backend"]["symbols"] == [
        "pipeline_attribute_peaks_to_cistarget_rows_f32"
    ]


def test_attribute_peaks_to_cistarget_uses_rust_row_helper(monkeypatch):
    import rustscenic.pipeline as pipeline

    enriched = pd.DataFrame(
        {
            "regulon": ["TF_A_regulon", "TF_B_regulon"],
            "motif": ["m_a", "m_b"],
            "auc": np.array([0.8, 0.5], dtype=np.float32),
        }
    )
    enhancer_links = pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "peak_id": ["p1", "p2"],
            "correlation": [0.7, 0.6],
        }
    )
    regulons = {"TF_A_regulon": ["G1"], "TF_B_regulon": ["G2"]}

    def fake_attribute_peak_rows(
        enriched_regulons,
        enriched_motifs,
        enriched_aucs,
        regulon_names,
        regulon_targets,
        enhancer_genes,
        enhancer_peaks,
    ):
        assert enriched_regulons == ["TF_A_regulon", "TF_B_regulon"]
        assert enriched_motifs == ["m_a", "m_b"]
        assert enriched_aucs.dtype == np.float32
        np.testing.assert_array_equal(enriched_aucs, np.array([0.8, 0.5], dtype=np.float32))
        assert regulon_names == ["TF_A_regulon", "TF_B_regulon"]
        assert regulon_targets == [["G1"], ["G2"]]
        assert enhancer_genes == ["G1", "G2"]
        assert enhancer_peaks == ["p1", "p2"]
        return (
            ["TF_B_regulon", "TF_A_regulon"],
            ["m_b", "m_a"],
            ["p2", "p1"],
            np.array([0.5, 0.8], dtype=np.float32),
        )

    monkeypatch.setattr(
        pipeline,
        "_pipeline_attribute_peak_rows_f32",
        fake_attribute_peak_rows,
    )

    got = pipeline._attribute_peaks_to_cistarget(enriched, enhancer_links, regulons)

    assert got.to_dict("list") == {
        "regulon": ["TF_B_regulon", "TF_A_regulon"],
        "motif": ["m_b", "m_a"],
        "peak_id": ["p2", "p1"],
        "auc": [np.float32(0.5), np.float32(0.8)],
    }


def test_attribute_peaks_to_cistarget_matches_pandas_reference_without_motif():
    from rustscenic.pipeline import _attribute_peaks_to_cistarget

    enriched = pd.DataFrame({"regulon": ["TF1_regulon", "TF2_regulon"], "auc": [0.4, 0.9]})
    enhancer_links = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G1", "other"],
            "peak_id": ["p1", "p2", "p1", "p3"],
            "correlation": [0.5, 0.6, 0.5, 0.1],
        }
    )
    regulons = {"TF1_regulon": ["G1", "G1", "G2"], "TF2_regulon": ["missing"]}

    got = _attribute_peaks_to_cistarget(enriched, enhancer_links, regulons)
    expected = _reference_attribute_peaks_to_cistarget_pandas(
        enriched, enhancer_links, regulons
    )
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_attribute_peaks_to_cistarget_handles_empty():
    """Empty cistarget → empty output frame with the right schema."""
    import pandas as pd
    from rustscenic.pipeline import _attribute_peaks_to_cistarget

    enriched = pd.DataFrame(columns=["regulon", "motif", "auc"])
    links = pd.DataFrame(columns=["peak_id", "gene", "correlation"])
    out = _attribute_peaks_to_cistarget(enriched, links, {})
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


def test_peak_coords_from_bed_uses_rust_parser_and_preserves_atac_order(tmp_path):
    from rustscenic.pipeline import _peak_coords_from_bed

    peaks_path = tmp_path / "peaks.bed"
    peaks_path.write_text(
        "chr1\t100\t200\tp1\n"
        "chr2\t300\t450\tp2\n"
        "chr3\t500\t700\tp3\n"
    )

    got = _peak_coords_from_bed(peaks_path, ["p3", "missing", "p1"])

    assert got.index.tolist() == ["p3", "p1"]
    assert got["chrom"].tolist() == ["chr3", "chr1"]
    assert got["start"].tolist() == [500, 100]
    assert got["end"].tolist() == [700, 200]
    assert got.attrs["rust_backend"] == {
        "engine": "rust",
        "symbols": ["preproc_peak_coords_for_names"],
    }


def test_pipeline_run_uses_region_cistarget_when_supplied(tmp_path, monkeypatch):
    """When `region_motif_rankings` is supplied, pipeline.run runs
    region-based cistarget against the linked peaks (exact path) instead
    of bridging via GRN ∩ enhancer (approximate path). Verifies the
    new region path is taken end-to-end."""
    import gzip, json, anndata as ad, numpy as np, pandas as pd
    from pathlib import Path
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

    # Synthetic REGION rankings - same kernel, different feature set
    n_peaks = len(peak_names)
    region_rank = np.full((len(motif_names), n_peaks), n_peaks - 1, dtype=np.int32)
    for tf_idx in range(3):
        # programme tf_idx peaks: peak_{tf_idx}_*
        for rank, j in enumerate(
            [i for i, n in enumerate(peak_names) if n.startswith(f"peak_{tf_idx}_")]
        ):
            region_rank[tf_idx, j] = rank
    region_rankings = pd.DataFrame(region_rank, index=motif_names, columns=peak_names)
    region_export = region_rankings.copy()
    region_export["unused_peak_not_in_run"] = np.arange(len(region_export), dtype=np.int32)
    region_export.insert(0, "motifs", region_export.index)
    region_rankings_path = tmp_path / "regions_vs_motifs.rankings.feather"
    region_export.reset_index(drop=True).to_feather(region_rankings_path)

    read_feather_calls = []
    real_read_feather = pd.read_feather

    def recording_read_feather(path, *args, **kwargs):
        if Path(path) == region_rankings_path:
            read_feather_calls.append(kwargs.get("columns"))
        return real_read_feather(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_feather", recording_read_feather)

    out = tmp_path / "pipeline_out"
    result = rustscenic.pipeline.run(
        rna,
        out,
        fragments=str(frag_path),
        peaks=str(peaks_path),
        tfs=["G000", "G005", "G010"],
        motif_rankings=motif_rankings,
        region_motif_rankings=region_rankings_path,
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
    assert read_feather_calls
    assert all(cols is not None for cols in read_feather_calls)
    assert all("motifs" in cols for cols in read_feather_calls)
    assert all("unused_peak_not_in_run" not in cols for cols in read_feather_calls)
    region_symbols = [
        "cistarget_enrichment_from_rankings_i32",
        "cistarget_region_attribution_peak_values_i32",
        "pipeline_expand_region_cistarget_rows_f32",
    ]
    assert result.backend_execution["eregulon_peak_attribution"]["symbols"] == region_symbols
    manifest = json.loads((out / "manifest.json").read_text())
    assert (
        manifest["backend_execution"]["eregulon_peak_attribution"]["symbols"]
        == region_symbols
    )

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
        region_motif_rankings=region_rankings_path,
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
    assert (
        region_only.backend_execution["eregulon_peak_attribution"]["symbols"]
        == region_symbols
    )


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


def test_coerce_rankings_projects_aertslab_feather_columns(tmp_path):
    """Large region-ranking feathers must support column projection.

    A full aertslab region DB can be tens of GB. pipeline.run should read
    only the current run's peaks for region cistarget instead of materialising
    the full file.
    """
    from rustscenic.pipeline import _coerce_rankings

    path = tmp_path / "regions_vs_motifs.rankings.feather"
    pd.DataFrame({
        "motifs": ["MOTIF_A", "MOTIF_B"],
        "chr1:100-200": [1, 3],
        "chr1:300-400": [2, 1],
        "chr1:500-600": [3, 2],
    }).to_feather(path)

    rankings = _coerce_rankings(
        path,
        feature_names=["chr1:300-400", "chr1:does-not-exist"],
    )

    assert list(rankings.index) == ["MOTIF_A", "MOTIF_B"]
    assert list(rankings.columns) == ["chr1:300-400"]


def test_coerce_rankings_projects_dataframe_without_losing_motif_index():
    from rustscenic.pipeline import _coerce_rankings

    rankings = _coerce_rankings(
        pd.DataFrame({
            "motifs": ["MOTIF_A", "MOTIF_B"],
            "peak_a": [1, 2],
            "peak_b": [2, 1],
        }),
        feature_names=["peak_b"],
    )

    assert list(rankings.index) == ["MOTIF_A", "MOTIF_B"]
    assert list(rankings.columns) == ["peak_b"]


def test_ranking_column_projection_preserves_schema_order_and_duplicates():
    from rustscenic.pipeline import _ranking_column_projection

    keep, examples = _ranking_column_projection(
        ["motifs", "peak_b", "peak_a", "peak_b", "other"],
        ["peak_a", "missing", "peak_b"],
        motif_col="motifs",
    )

    assert keep == ["motifs", "peak_b", "peak_a", "peak_b"]
    assert examples == ["missing", "peak_a", "peak_b"]


def test_ranking_column_projection_uses_rust_name_projection(monkeypatch):
    import rustscenic.pipeline as pipeline

    seen = {}

    def fake_projection(columns, requested_features, motif_col):
        seen["columns"] = columns
        seen["requested_features"] = requested_features
        seen["motif_col"] = motif_col
        return ["motifs", "peak_a"], ["missing", "peak_a"]

    monkeypatch.setattr(
        pipeline,
        "_pipeline_project_ranking_columns",
        fake_projection,
    )

    keep, examples = pipeline._ranking_column_projection(
        ["motifs", "peak_a", "other"],
        ["peak_a", "missing"],
        motif_col="motifs",
    )

    assert seen == {
        "columns": ["motifs", "peak_a", "other"],
        "requested_features": ["peak_a", "missing"],
        "motif_col": "motifs",
    }
    assert keep == ["motifs", "peak_a"]
    assert examples == ["missing", "peak_a"]


def test_projected_ranking_columns_reports_sorted_requested_examples(tmp_path):
    import pytest

    from rustscenic.pipeline import _projected_ranking_columns

    path = tmp_path / "regions_vs_motifs.rankings.feather"
    pd.DataFrame({
        "motifs": ["MOTIF_A"],
        "peak_present": [1],
    }).to_feather(path)

    with pytest.raises(ValueError, match="First requested peaks"):
        _projected_ranking_columns(
            path,
            ["z_peak", "a_peak", "m_peak"],
            kind="feather",
        )


def test_region_ranking_projection_keeps_dataframe_inputs_unsliced():
    from rustscenic.pipeline import _ranking_projection_features

    df = pd.DataFrame({"motifs": ["MOTIF_A"], "peak_a": [1], "peak_b": [2]})

    assert _ranking_projection_features(df, ["peak_a"]) is None
    assert list(_ranking_projection_features("rankings.feather", ["peak_a"])) == ["peak_a"]


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


def test_rankings_with_motif_index_checks_dtypes_without_drop(monkeypatch):
    """Large ranking frames must not clone/drop value columns for dtype checks."""
    from pathlib import Path

    from rustscenic.pipeline import _rankings_with_motif_index

    df = pd.DataFrame({
        "motif_id": ["M1", "M2"],
        "GENE1": [1, 2],
        "GENE2": [2, 1],
    })

    def fail_drop(*_args, **_kwargs):
        raise AssertionError("dtype detection should not call DataFrame.drop")

    monkeypatch.setattr(df, "drop", fail_drop)

    rankings = _rankings_with_motif_index(df, Path("rankings.parquet"))

    assert list(rankings.index) == ["M1", "M2"]
    assert list(rankings.columns) == ["GENE1", "GENE2"]
