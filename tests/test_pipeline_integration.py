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
    grn = pd.DataFrame(
        [{"TF": "PAX5", "target": "GENE_A", "importance": 1.0}]
    )
    enhancer_links = pd.DataFrame(
        [{"peak_id": "peak_1", "gene": "GENE_A"}]
    )
    regulons = {"PAX5_regulon(+)": ["GENE_A"]}

    out = _attribute_peaks_to_cistarget(
        enriched,
        grn,
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

    common = dict(
        rna=rna,
        output_dir=tmp_path,
        tfs=["TF_A", "TF_B"],
        motif_rankings=rankings,
        grn_n_estimators=10,
        grn_top_targets=10,
        cistarget_top_frac=1.0,
        cistarget_auc_threshold=0.0,
        verbose=False,
    )

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

    base_kwargs = dict(
        rna=rna,
        output_dir=tmp_path,
        tfs=["TF_A", "TF_B"],
        motif_rankings=rankings,
        grn_n_estimators=10,
        grn_top_targets=10,
        cistarget_top_frac=0.2,
        cistarget_auc_threshold=0.0,
        verbose=False,
    )

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


def test_pipeline_run_uses_region_cistarget_when_supplied(tmp_path, monkeypatch):
    """When `region_motif_rankings` is supplied, pipeline.run runs
    region-based cistarget against the linked peaks (exact path) instead
    of bridging via GRN ∩ enhancer (approximate path). Verifies the
    new region path is taken end-to-end."""
    import gzip, os, anndata as ad, numpy as np, pandas as pd
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
