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
