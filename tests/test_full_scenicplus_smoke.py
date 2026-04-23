"""Full SCENIC+ pipeline smoke test — every new module, one run.

Walks the complete rustscenic public API shipped today, in sequence,
on synthetic multiome data with known structure:

    fragments.tsv.gz + cluster labels
       │
       ├─▶ rustscenic.preproc.qc.insert_size_stats
       ├─▶ rustscenic.preproc.call_peaks  ──▶ peaks.bed
       ├─▶ rustscenic.preproc.qc.frip (needs peaks)
       ├─▶ rustscenic.preproc.qc.tss_enrichment (needs TSS)
       └─▶ rustscenic.preproc.fragments_to_matrix  ──▶ cells×peaks AnnData

    cells×peaks AnnData  +  matched RNA AnnData
       │
       ├─▶ rustscenic.grn.infer                    ──▶ TF→target edges
       ├─▶ rustscenic.cistarget.enrich             ──▶ motif enrichment
       ├─▶ rustscenic.enhancer.link_peaks_to_genes ──▶ peak↔gene links
       ├─▶ rustscenic.eregulon.build_eregulons     ──▶ TF × enhancers × genes
       └─▶ rustscenic.aucell.score                 ──▶ per-cell activity

If any one module silently broke under realistic interplay with the
others, this catches it. The per-module tests we already have can
pass while the interfaces between modules quietly mismatch — this is
the interface-layer guard.
"""
from __future__ import annotations

import gzip
import os
import tempfile
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest


N_CELLS = 120


def _simulate_multiome_with_fragments():
    """Generate matched RNA + fragment-file + clusters. Three simulated
    regulatory programmes, each with:
      - 5 genes whose expression tracks its programme membership
      - 10 peak regions where that programme's cells have dense fragments
      - 1 TSS near each peak for TSS enrichment to register

    Expression and fragment density are driven by the **same** programme
    membership vector so peak↔gene correlation is a real signal, not a
    chance artefact of two independent random draws.
    """
    rng = np.random.default_rng(0)
    n_cells = N_CELLS

    # 3 clusters, evenly split
    cluster_per_cell = np.array([i * 3 // n_cells for i in range(n_cells)], dtype=np.uint32)
    cell_names = [f"cell{i}" for i in range(n_cells)]

    # Programme activity vector — binary cluster membership + mild noise.
    # Peaks and expression both draw from this so their correlation is real.
    activity = np.zeros((3, n_cells), dtype=np.float32)
    for p in range(3):
        activity[p] = (cluster_per_cell == p).astype(np.float32)
        activity[p] += 0.1 * rng.normal(size=n_cells).astype(np.float32)

    # RNA: 30 genes, first 5 track programme 0, next 5 track programme 1, etc.
    rna_genes = [f"G{i:03d}" for i in range(30)]
    rna_X = np.zeros((n_cells, 30), dtype=np.float32)
    for i in range(30):
        programme = i // 5 if i < 15 else -1
        if programme >= 0:
            rna_X[:, i] = activity[programme] + 0.2 * rng.normal(size=n_cells)
        else:
            rna_X[:, i] = rng.normal(size=n_cells)

    # TFs: G000, G005, G010 (one per programme)
    tf_names = ["G000", "G005", "G010"]

    # Build fragments file. Per programme, a dense block of fragments at a
    # specific genomic position, only in cells belonging to that programme.
    frag_lines = []
    for programme in range(3):
        peak_region_starts = [10_000 + programme * 100_000 + j * 5_000 for j in range(10)]
        # For each cell in this programme, drop ~20 fragments randomly in
        # the programme's peak regions.
        cells_in_prog = np.where(cluster_per_cell == programme)[0]
        for cell_idx in cells_in_prog:
            barcode = cell_names[cell_idx]
            for _ in range(20):
                peak_start = rng.choice(peak_region_starts)
                pos = peak_start + int(rng.integers(0, 500))
                frag_size = int(rng.integers(50, 400))
                frag_lines.append(
                    f"chr1\t{pos}\t{pos + frag_size}\t{barcode}\t1"
                )
        # Plus a handful of noise fragments scattered everywhere in this
        # programme's cells
        for cell_idx in cells_in_prog:
            barcode = cell_names[cell_idx]
            for _ in range(5):
                pos = int(rng.integers(0, 2_000_000))
                frag_size = int(rng.integers(50, 200))
                frag_lines.append(
                    f"chr1\t{pos}\t{pos + frag_size}\t{barcode}\t1"
                )

    # ATAC per-cell intensity matrix matching fragment density per programme
    # (used later as the .X for enhancer linking; doesn't have to come from
    # fragments — real pipelines also use the preproc fragments_to_matrix).
    # We'll build the real ATAC AnnData from fragments + called peaks below.

    td = tempfile.TemporaryDirectory()
    frag_path = os.path.join(td.name, "fragments.tsv.gz")
    with gzip.open(frag_path, "wt") as fh:
        fh.write("\n".join(frag_lines) + "\n")

    rna = ad.AnnData(
        X=rna_X,
        obs=pd.DataFrame(
            {"cluster": cluster_per_cell.astype(int)},
            index=cell_names,
        ),
        var=pd.DataFrame(index=rna_genes),
    )

    # Gene TSS coords: place each programme's genes near the first peak region
    gene_tss_rows = []
    for i, g in enumerate(rna_genes):
        programme = i // 5 if i < 15 else -1
        if programme >= 0:
            tss_pos = 10_000 + programme * 100_000 + 250
        else:
            tss_pos = 5_000_000 + i * 1_000
        gene_tss_rows.append((g, "chr1", tss_pos))
    gene_coords = pd.DataFrame(gene_tss_rows, columns=["gene", "chrom", "tss"])

    return rna, frag_path, td, cluster_per_cell, tf_names, gene_coords


def test_full_scenicplus_smoke():
    """Every new API walked in one test, with biology-informed checks."""
    import rustscenic.aucell
    import rustscenic.cistarget
    import rustscenic.enhancer
    import rustscenic.eregulon
    import rustscenic.grn
    import rustscenic.preproc

    rna, frag_path, td, cluster_per_cell, tf_names, gene_coords = (
        _simulate_multiome_with_fragments()
    )

    with td:
        # --- 1. Per-barcode insert-size stats ---
        is_stats = rustscenic.preproc.qc.insert_size_stats(frag_path)
        assert isinstance(is_stats, pd.DataFrame)
        assert not is_stats.empty
        assert (is_stats["n_fragments"] > 0).any()

        # --- 2. Call peaks from pseudobulked fragments ---
        # Barcodes in fragments appear in insertion order; build a cluster
        # array parallel to that ordering.
        barcode_order = is_stats.index.tolist()
        cell_idx_for_bc = {cell: i for i, cell in enumerate(
            [f"cell{i}" for i in range(N_CELLS)]
        )}
        cluster_per_bc = [
            int(cluster_per_cell[cell_idx_for_bc[bc]]) for bc in barcode_order
        ]
        peaks_df = rustscenic.preproc.call_peaks(
            frag_path,
            cluster_per_barcode=cluster_per_bc,
            n_clusters=3,
        )
        assert not peaks_df.empty, "peak calling produced no peaks"
        # We should recover at least one peak per programme's region
        for programme in range(3):
            prog_start = 10_000 + programme * 100_000
            hits = (
                (peaks_df["chrom"] == "chr1")
                & (peaks_df["start"] < prog_start + 60_000)
                & (peaks_df["end"] > prog_start - 1_000)
            ).sum()
            assert hits >= 1, (
                f"no peaks called in programme {programme}'s region "
                f"(near chr1:{prog_start})"
            )

        # --- 3. FRiP using the peaks we just called ---
        # Write peaks to a BED file for FRiP + fragments_to_matrix
        peaks_bed = os.path.join(
            os.path.dirname(frag_path), "called_peaks.bed"
        )
        peaks_df.to_csv(peaks_bed, sep="\t", header=False, index=False)
        frip_scores = rustscenic.preproc.qc.frip(frag_path, peaks_bed)
        assert isinstance(frip_scores, pd.Series)
        assert (frip_scores > 0.01).any(), (
            "FRiP is zero everywhere — the peaks we just called don't "
            "overlap the fragments they were called from (self-consistency "
            "broken)."
        )

        # --- 4. TSS enrichment ---
        tss_df = pd.DataFrame(
            {
                "chrom": ["chr1"] * 3,
                "position": [10_250, 110_250, 210_250],  # programme TSS centres
            }
        )
        tss_scores = rustscenic.preproc.qc.tss_enrichment(frag_path, tss_df)
        assert (tss_scores > 0).any(), "TSS enrichment is zero for every cell"

        # --- 5. Fragments → cells × peaks ATAC matrix ---
        atac = rustscenic.preproc.fragments_to_matrix(frag_path, peaks_bed)
        assert atac.n_obs > 0 and atac.n_vars > 0
        import scipy.sparse as sp
        X_atac = atac.X.toarray() if sp.issparse(atac.X) else atac.X
        assert X_atac.sum() > 0, "fragments_to_matrix returned all zeros"

        # --- 6. GRN on RNA ---
        grn = rustscenic.grn.infer(
            rna, tf_names=tf_names, n_estimators=50, seed=0, verbose=False,
        )
        assert not grn.empty
        assert set(grn["TF"].unique()) == set(tf_names)

        # --- 7. Cistarget (synthetic rankings) ---
        # Rankings for each TF rank its programme genes high; others low.
        motif_names = [f"M_{tf}" for tf in tf_names]
        n_genes = rna.n_vars
        rank_matrix = np.full((len(motif_names), n_genes), n_genes - 1, dtype=np.int32)
        for tf_idx, tf in enumerate(tf_names):
            programme_idx = tf_idx  # G000→0, G005→1, G010→2
            programme_genes = [
                i for i in range(n_genes)
                if (i // 5 == programme_idx) and (i < 15)
            ]
            for rank, gene_idx in enumerate(programme_genes):
                rank_matrix[tf_idx, gene_idx] = rank
        rankings = pd.DataFrame(
            rank_matrix,
            index=motif_names,
            columns=list(rna.var_names),
        )
        regulons = []
        for tf in tf_names:
            top = grn[grn["TF"] == tf].nlargest(10, "importance")["target"].tolist()
            if len(top) >= 3:
                regulons.append((f"{tf}_regulon", top))
        assert regulons
        ct = rustscenic.cistarget.enrich(
            rankings, regulons, top_frac=0.2, auc_threshold=0.0,
        )
        assert not ct.empty

        # --- 8. Enhancer → gene linking ---
        # Build matched ATAC AnnData on RNA's cell ordering
        atac_for_enhancer = atac[rna.obs_names.intersection(atac.obs_names)].copy()
        rna_for_enhancer = rna[atac_for_enhancer.obs_names].copy()
        links = rustscenic.enhancer.link_peaks_to_genes(
            rna_for_enhancer,
            atac_for_enhancer,
            gene_coords,
            max_distance=500_000,
            min_abs_corr=0.2,
        )
        assert not links.empty, "no peak-gene links survived correlation filter"

        # --- 9. eRegulon assembly ---
        # Simulate cistarget output: each TF's motif is enriched in every
        # enhancer-linked peak. In a real run, cistarget returns its own
        # ranking-derived peak set; here we just make sure the three
        # inputs (GRN, cistarget, enhancer links) share peak identities.
        linked_peaks = links["peak_id"].unique().tolist()
        assert linked_peaks, "no linked peaks to feed cistarget fixture"
        ct_with_peaks = pd.DataFrame(
            [
                {
                    "regulon": f"{tf}_regulon",
                    "motif": f"M_{tf}",
                    "peak_id": peak,
                    "auc": 0.3,
                }
                for tf in tf_names
                for peak in linked_peaks
            ]
        )
        eregs = rustscenic.eregulon.build_eregulons(
            grn, ct_with_peaks, links,
            min_target_genes=2, min_enhancer_links=1,
            use_grn_intersection=True,
        )
        # At least one eRegulon should survive — we built the simulation so
        # that programme TFs have both GRN targets and correlated peaks.
        assert len(eregs) >= 1, "eRegulon assembler dropped every TF"

        # --- 10. AUCell on the assembled regulons ---
        reg_list = [(f"{er.tf}_eregulon", er.target_genes) for er in eregs]
        auc = rustscenic.aucell.score(rna_for_enhancer, reg_list, top_frac=0.3)
        assert auc.shape[0] > 0 and auc.shape[1] == len(eregs)
        assert (auc.values > 0).any(), "AUCell output is all-zero"
        # Coverage diagnostic must round-trip from PR #18
        assert "regulon_coverage" in auc.attrs


def test_eregulon_output_matches_expected_schema():
    """Format guard: eregulons_to_dataframe must emit exactly the columns
    downstream consumers rely on. Catches silent schema drift."""
    from rustscenic.eregulon import ERegulon, eregulons_to_dataframe

    eregs = [
        ERegulon(
            tf="TF_A",
            enhancers=["peak1", "peak2"],
            target_genes=["G1", "G2"],
            n_enhancer_links=3,
            motif_auc=0.2,
        )
    ]
    df = eregulons_to_dataframe(eregs)
    assert set(df.columns) == {
        "tf", "enhancer", "target_gene", "n_enhancer_links", "motif_auc",
    }
    # One row per (enhancer × gene) pair
    assert len(df) == 2 * 2
