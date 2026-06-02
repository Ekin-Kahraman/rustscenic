"""End-to-end rustscenic stage orchestrator.

Public API:
    rustscenic.pipeline.run(rna, output_dir, *, fragments=None, peaks=None,
                            tfs=None, motif_rankings=None, ...) -> PipelineResult

One call runs every rustscenic stage the user provides input for:

    1. preproc  (fragments + peaks)      → cells × peaks AnnData
    2. topics   (cells × peaks AnnData)  → cell-topic + topic-peak matrices
    3. grn      (RNA expression + TFs)   → TF-target importances
    4. regulons (grn)                    → top-N targets per TF
    5. cistarget (regulons + motif DB)   → motif-enriched regulons [optional]
    6. enhancer (RNA + ATAC + TSS)       → peak-gene links [optional]
    7. eRegulon (GRN + motifs + links)   → TF-enhancer-gene modules [optional]
    8. aucell   (RNA + regulons)         → per-cell regulon activity

Outputs are written to ``output_dir`` as parquet / json / h5ad files so
downstream notebooks can pick up where the pipeline left off.

No new Python dependencies. Uses only numpy, pandas, pyarrow, scipy,
plus the rustscenic Rust backend.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    cistarget_region_attribution_i16 as _cistarget_region_attribution_i16,
    cistarget_region_attribution_i32 as _cistarget_region_attribution_i32,
    cistarget_region_attribution_i64 as _cistarget_region_attribution_i64,
    cistarget_region_attribution_peak_values_i16 as _cistarget_region_peak_values_i16,
    cistarget_region_attribution_peak_values_i32 as _cistarget_region_peak_values_i32,
    cistarget_region_attribution_peak_values_i64 as _cistarget_region_peak_values_i64,
    cistarget_rankings_to_i32_f32 as _rankings_to_i32_f32,
    cistarget_rankings_to_i32_f64 as _rankings_to_i32_f64,
    pipeline_attribute_peaks_to_cistarget_rows_f32 as _pipeline_attribute_peak_rows_f32,
    pipeline_attribute_peaks_to_cistarget_rows_f64 as _pipeline_attribute_peak_rows_f64,
    pipeline_candidate_regulons_from_grn as _pipeline_candidate_regulons_from_grn,
    pipeline_expand_region_cistarget_rows_f32 as _pipeline_expand_region_rows_f32,
    pipeline_expand_region_cistarget_rows_f64 as _pipeline_expand_region_rows_f64,
    pipeline_filter_cistarget_peak_rows_f32 as _pipeline_filter_cistarget_peak_rows_f32,
    pipeline_filter_cistarget_peak_rows_f64 as _pipeline_filter_cistarget_peak_rows_f64,
    pipeline_match_atac_cell_indices as _pipeline_match_atac_cell_indices,
    pipeline_peak_regulons_and_features_from_edges as _pipeline_peak_regulons_and_features,
    pipeline_project_ranking_columns as _pipeline_project_ranking_columns,
    preproc_peak_coords_for_names as _preproc_peak_coords_for_names,
)
from rustscenic._stage_utils import (
    as_float32_contiguous,
    iter_regulon_pairs,
    tf_from_regulon_name,
)


@dataclass
class PipelineResult:
    """Artifacts and metadata from a pipeline run.

    All file paths point inside ``output_dir``. Stages that were skipped
    because inputs weren't provided have ``None`` for their result path.
    """

    output_dir: Path
    atac_matrix_path: Path | None = None
    grn_path: Path | None = None
    regulons_path: Path | None = None
    candidate_regulons_path: Path | None = None
    pruned_regulons_path: Path | None = None
    aucell_path: Path | None = None
    topics_dir: Path | None = None
    cistarget_path: Path | None = None
    enhancer_links_path: Path | None = None
    eregulons_path: Path | None = None
    integrated_adata_path: Path | None = None
    elapsed: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    n_cells: int | None = None
    n_grn_edges: int | None = None
    n_regulons: int | None = None
    n_candidate_regulons: int | None = None
    n_pruned_regulons: int | None = None
    n_cistarget_rows: int | None = None
    n_enhancer_links: int | None = None
    n_eregulon_rows: int | None = None
    n_eregulons: int | None = None
    aucell_shape: list[int] | None = None
    regulon_source: str = "candidate_grn_top_targets"
    backend_execution: dict = field(default_factory=dict)

    def manifest(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


def run(
    rna: str | Path | Any,
    output_dir: str | Path,
    *,
    adata_atac: Any | None = None,
    fragments: str | Path | None = None,
    peaks: str | Path | None = None,
    tfs: str | Path | Iterable[str] | None = None,
    motif_rankings: str | Path | pd.DataFrame | None = None,
    motif_annotations: str | Path | pd.DataFrame | None = None,
    region_motif_rankings: str | Path | pd.DataFrame | None = None,
    gene_coords: str | Path | pd.DataFrame | None = None,
    grn_n_estimators: int = 500,
    grn_max_features: float = 0.1,
    grn_target_block_size: int | None = None,
    grn_top_targets: int = 50,
    aucell_top_frac: float = 0.05,
    topics_n_topics: int = 30,
    topics_n_passes: int = 3,
    topics_method: str = "vb",
    topics_n_iters: int = 200,
    topics_n_threads: int = 1,
    cistarget_top_frac: float = 0.05,
    cistarget_auc_threshold: float = 0.05,
    cistarget_nes_threshold: float | None = None,
    enhancer_max_distance: int = 500_000,
    enhancer_min_abs_corr: float = 0.1,
    eregulon_min_target_genes: int = 5,
    eregulon_min_enhancer_links: int = 2,
    seed: int = 777,
    verbose: bool = True,
) -> PipelineResult:
    """Run the available rustscenic stages end-to-end.

    The workflow runs only the stages the user supplies inputs for. At
    minimum, ``rna`` is required (for GRN + AUCell). Providing
    ``fragments`` and ``peaks`` enables preproc + topics. Providing
    ``motif_rankings`` enables cistarget.

    Parameters
    ----------
    rna
        An AnnData, a path to an ``.h5ad``, or a pandas DataFrame
        (cells × genes).
    output_dir
        Directory where all artifacts are written. Created if missing.
    adata_atac
        Pre-built cells × peaks ``AnnData``, or a path to one on disk.
        Use this when you already have a cleaned/subset ATAC matrix
        (e.g. cell-called barcodes only, post-QC). Mutually exclusive
        with ``fragments`` + ``peaks``: if ``adata_atac`` is provided,
        the fragments + peaks path is skipped. This avoids carrying
        the full raw 10x barcode set (~450k empty droplets typical)
        through topics, which can stall downstream stages on consumer
        hardware.
    fragments, peaks
        Paths to a 10x-style ``fragments.tsv[.gz]`` and peak BED. When
        both are provided AND ``adata_atac`` is not, rustscenic.preproc
        builds the cells × peaks AnnData and topics fits on it.
    tfs
        Candidate transcription factor names. Path to a newline-separated
        file, an iterable of strings, or ``None`` to use the bundled
        human TF list.
    motif_rankings
        Motif ranking DataFrame, or a path to a parquet / feather file
        with motifs as rows and genes as columns. If provided, cistarget
        runs to score candidate regulons for motif enrichment.
    motif_annotations
        Optional motif-to-TF annotation table. When provided alongside
        ``motif_rankings``, the active regulons are pruned to enriched
        motifs annotated back to the source TF, and target genes are
        restricted to the motif ranking recovery window.
    region_motif_rankings
        Optional region-based motif ranking DataFrame, or path, with motifs
        as rows and peak / region IDs as columns. When supplied alongside
        ATAC inputs and gene coordinates, eRegulon assembly uses this exact
        region-cistarget path instead of the gene-cistarget bridge. File-backed
        parquet / feather rankings are projected to the peaks used by the
        current run so large region-ranking databases do not need to be loaded
        in full.
    gene_coords
        DataFrame with columns ``['gene', 'chrom', 'tss']``, or a path
        to a parquet/csv file with the same shape. When supplied
        alongside ``fragments`` + ``peaks``, the orchestrator runs
        ``rustscenic.enhancer.link_peaks_to_genes`` and, when either
        gene- or region-based motif rankings are supplied,
        ``rustscenic.eregulon.build_eregulons``.
    topics_method
        ``"vb"`` (default) - online VB LDA, fast at small K (≤ 10).
        ``"gibbs"`` - collapsed-Gibbs LDA (Mallet-class), slower per
        sweep but recovers ~10× more distinct topics on sparse scATAC
        at K ≥ 30. Pair with ``topics_n_threads > 1`` for AD-LDA
        parallel speedup at atlas scale.
    topics_n_iters
        Gibbs sweeps (only used when ``topics_method='gibbs'``). 200
        is a reasonable default; bump to 500–1000 for higher-quality
        posterior estimates.
    topics_n_threads
        Threads for the Gibbs sampler (only used when
        ``topics_method='gibbs'``). 1 = bit-deterministic serial
        path. > 1 = AD-LDA parallel path.
    grn_max_features
        Fraction of candidate TFs sampled per split in GRN boosting.
        ``0.1`` matches arboreto/SCENIC defaults. Lower values can be
        much faster on high-TF datasets, but change edge rankings and
        should be treated as a speed/quality tradeoff.
    grn_target_block_size
        Optional target block width passed through to
        ``rustscenic.grn.infer``. ``None`` uses the adaptive default,
        which shrinks the block at high cell counts to reduce
        memory-bandwidth pressure.

    Returns
    -------
    PipelineResult - dataclass with paths to every artifact written.
    """
    import warnings as _warnings

    import anndata as ad
    import rustscenic.aucell
    import rustscenic.grn

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _Logger(verbose)
    elapsed: dict = {}
    memory: dict = {}
    backend_execution: dict = {}
    n_cistarget_rows: int | None = None
    n_enhancer_links: int | None = None
    n_eregulon_rows: int | None = None

    def mark_memory(stage: str) -> None:
        memory[stage] = _peak_rss_gb()

    if motif_annotations is not None and motif_rankings is None:
        # Pruning needs both. Without rankings we have no enriched motifs to
        # filter, so the annotations are dead weight; warn instead of silently
        # routing the user into an un-pruned run they didn't ask for.
        _warnings.warn(
            "motif_annotations supplied without motif_rankings; pruning "
            "requires both, so the annotations will be ignored and the "
            "active regulon set will be the GRN top-target candidates. "
            "Pass motif_rankings to enable annotation-based pruning.",
            UserWarning,
            stacklevel=2,
        )

    # ---- 1. load / normalise RNA ----
    log("[1/8] loading RNA expression")
    adata_rna = _coerce_adata(rna)
    n_cells = adata_rna.n_obs
    log(f"      RNA shape: {adata_rna.shape}")
    mark_memory("load_rna")

    # ---- 2. preproc + topics (only if ATAC inputs provided) ----
    # Two paths into the cells × peaks ATAC matrix:
    #   (a) `adata_atac` - caller passed an already-built (and typically
    #       cell-QC-subset) AnnData. Skip preproc entirely.
    #   (b) `fragments` + `peaks` - read raw 10x outputs and call
    #       `rustscenic.preproc.fragments_to_matrix`. Note this returns
    #       ALL observed barcodes (including empty droplets); on raw 10x
    #       this can be ~10–100× larger than the QC-passed cell count
    #       and stall downstream stages. Prefer (a) for real workflows.
    atac_matrix_path = None
    topics_dir = None
    have_atac_input = adata_atac is not None or (fragments is not None and peaks is not None)
    if have_atac_input:
        if adata_atac is not None:
            if isinstance(adata_atac, (str, Path)):
                log("[2/8] preproc: loading pre-built ATAC AnnData from disk")
                adata_atac = ad.read_h5ad(adata_atac)
            else:
                log("[2/8] preproc: using caller-provided ATAC AnnData (skipping fragments_to_matrix)")
            elapsed["preproc"] = 0.0
            backend_execution["preproc"] = _skipped_execution("caller provided pre-built ATAC AnnData")
            log(f"      ATAC shape: {adata_atac.shape}")
        else:
            import rustscenic.preproc
            log("[2/8] preproc: fragments + peaks → cells × peaks")
            t0 = time.perf_counter()
            adata_atac = rustscenic.preproc.fragments_to_matrix(fragments, peaks)
            elapsed["preproc"] = time.perf_counter() - t0
            backend_execution["preproc"] = _rust_execution_from_attrs(
                adata_atac,
                "preproc_fragments_to_matrix",
            )
            log(f"      ATAC shape: {adata_atac.shape}, took {elapsed['preproc']:.1f}s")
        mark_memory("preproc")

        adata_atac = _subset_atac_to_rna_cells(adata_rna, adata_atac, log=log)

        # Persist the artefact first; only mark have_atac=True (via
        # atac_matrix_path) once the file is on disk. If write fails (disk
        # full, unserializable obs), downstream stages must skip rather than
        # raise FileNotFoundError reading a path that was never written.
        _atac_artefact = output_dir / "atac_cells_by_peaks.h5ad"
        adata_atac.write_h5ad(_atac_artefact)
        atac_matrix_path = _atac_artefact

        # Topics on the sparse ATAC matrix
        import rustscenic.topics
        if topics_method not in ("vb", "gibbs"):
            raise ValueError(
                f"topics_method must be 'vb' or 'gibbs', got {topics_method!r}"
            )
        log(f"[3/8] topics: fitting LDA K={topics_n_topics} via {topics_method}")
        t0 = time.perf_counter()
        if topics_method == "vb":
            topics_result = rustscenic.topics.fit(
                adata_atac,
                n_topics=topics_n_topics,
                n_passes=topics_n_passes,
                seed=seed,
            )
            backend_execution["topics"] = _rust_execution_from_attrs(
                topics_result.cell_topic,
                "topics_fit",
            )
        else:
            topics_result = rustscenic.topics.fit_gibbs(
                adata_atac,
                n_topics=topics_n_topics,
                n_iters=topics_n_iters,
                n_threads=topics_n_threads,
                seed=seed,
            )
            backend_execution["topics"] = _rust_execution_from_attrs(
                topics_result.cell_topic,
                "topics_fit_gibbs",
            )
        elapsed["topics"] = time.perf_counter() - t0
        log(f"      fit in {elapsed['topics']:.1f}s")
        mark_memory("topics")

        topics_dir = output_dir / "topics"
        topics_dir.mkdir(exist_ok=True)
        # topics_result is typically a (cell_topic, topic_peak) pair
        if hasattr(topics_result, "cell_topic"):
            np.save(topics_dir / "cell_topic.npy", topics_result.cell_topic)
            np.save(topics_dir / "topic_peak.npy", topics_result.topic_peak)
            topics_result.cell_topic.to_parquet(topics_dir / "cell_topic.parquet")
            topics_result.topic_peak.to_parquet(topics_dir / "topic_peak.parquet")
    else:
        log("[2/8] preproc + topics: skipped (no fragments / peaks)")
        log("[3/8] topics: skipped")

    # ---- 3. GRN ----
    log("[4/8] GRN inference on RNA")
    tf_list = _load_tfs(tfs)
    log(f"      {len(tf_list)} candidate TFs")
    t0 = time.perf_counter()
    grn = rustscenic.grn.infer(
        adata_rna,
        tf_names=tf_list,
        n_estimators=grn_n_estimators,
        max_features=grn_max_features,
        target_block_size=grn_target_block_size,
        seed=seed,
        verbose=False,
    )
    elapsed["grn"] = time.perf_counter() - t0
    backend_execution["grn"] = _rust_execution_from_attrs(
        grn,
        "grn_infer",
        "grn_infer_sparse_csc",
    )
    n_grn_edges = int(len(grn))
    grn_path = output_dir / "grn.parquet"
    grn.to_parquet(grn_path, index=False)
    log(f"      {n_grn_edges:,} edges in {elapsed['grn']:.1f}s → {grn_path.name}")
    mark_memory("grn")

    # ---- 4. build candidate regulons ----
    if grn_top_targets < 1:
        raise ValueError(f"grn_top_targets must be >= 1, got {grn_top_targets}")
    log(f"[5/8] candidate regulons: top-{grn_top_targets} targets per TF")
    min_targets_for_candidate = min(10, grn_top_targets)
    candidate_regulons = _candidate_regulons_from_grn(
        grn,
        top_targets=grn_top_targets,
        min_targets=min_targets_for_candidate,
    )
    backend_execution["candidate_regulons"] = _rust_execution(
        "pipeline_candidate_regulons_from_grn"
    )
    candidate_regulons_path = output_dir / "candidate_regulons.json"
    candidate_regulons_path.write_text(json.dumps(candidate_regulons, indent=2))
    candidate_regulon_pairs = list(iter_regulon_pairs(candidate_regulons))
    regulons = dict(candidate_regulons)
    regulon_source = "candidate_grn_top_targets"
    pruned_regulons_path = None
    n_pruned_regulons: int | None = None
    log(
        f"      {len(candidate_regulons)} candidate regulons "
        f"(≥{min_targets_for_candidate} targets) → {candidate_regulons_path.name}"
    )
    mark_memory("candidate_regulons")

    # ---- 4b. cistarget (optional) ----
    cistarget_path = None
    enriched: pd.DataFrame | None = None
    enriched_for_eregulons: pd.DataFrame | None = None
    motif_annotations_df = (
        _coerce_motif_annotations(motif_annotations)
        if motif_annotations is not None and motif_rankings is not None else None
    )
    if motif_rankings is not None:
        import rustscenic.cistarget
        rankings_df = _coerce_rankings(motif_rankings)
        log(f"[6/8] cistarget: {len(rankings_df):,} motifs × {rankings_df.shape[1]:,} genes")
        t0 = time.perf_counter()
        enriched = rustscenic.cistarget.enrich(
            rankings_df,
            candidate_regulon_pairs,
            top_frac=cistarget_top_frac,
            auc_threshold=cistarget_auc_threshold,
            nes_threshold=cistarget_nes_threshold,
        )
        backend_execution["cistarget"] = _rust_execution_from_attrs(
            enriched,
            "cistarget_enrichment_from_rankings_i32",
        )
        enriched_for_eregulons = enriched
        elapsed["cistarget"] = time.perf_counter() - t0
        n_cistarget_rows = int(len(enriched))
        cistarget_path = output_dir / "cistarget_enriched.parquet"
        enriched.to_parquet(cistarget_path, index=False)
        log(
            f"      {len(enriched):,} enriched pairs in {elapsed['cistarget']:.1f}s"
            + (f" (nes_threshold={cistarget_nes_threshold})" if cistarget_nes_threshold is not None else "")
        )
        mark_memory("cistarget")

        if motif_annotations_df is not None:
            log("      pruning enriched motifs with motif annotations")
            # nes_threshold is applied inside enrich(), so `enriched` already
            # reflects it. Avoid passing it again to prune_* to keep the
            # filter site singular and the data flow auditable.
            pruned_enriched = rustscenic.cistarget.prune_enriched_motifs(
                enriched,
                motif_annotations_df,
                auc_threshold=cistarget_auc_threshold,
            )
            pruned_regulons = rustscenic.cistarget.prune_regulons(
                enriched,
                candidate_regulon_pairs,
                motif_annotations_df,
                rankings=rankings_df,
                top_frac=cistarget_top_frac,
                auc_threshold=cistarget_auc_threshold,
                min_genes=1,
            )
            pruning_symbols = _rust_backend_symbols(pruned_enriched)
            if not pruned_enriched.empty:
                pruning_symbols.extend(
                    rustscenic.cistarget._prune_regulons_backend_symbols(rankings_df)
                )
            backend_execution["cistarget_pruning"] = (
                _rust_execution(*pruning_symbols)
                if pruning_symbols
                else _skipped_execution("no enriched cistarget rows to prune")
            )
            n_pruned_regulons = len(pruned_regulons)
            if pruned_regulons:
                pruned_enriched_path = output_dir / "cistarget_pruned_enriched.parquet"
                pruned_enriched.to_parquet(pruned_enriched_path, index=False)
                pruned_regulons_path = output_dir / "pruned_regulons.json"
                pruned_regulons_path.write_text(json.dumps(pruned_regulons, indent=2))
                regulons = pruned_regulons
                enriched_for_eregulons = pruned_enriched
                regulon_source = "motif_annotation_pruned"
                log(
                    f"      {len(pruned_regulons)} pruned regulons from "
                    f"{len(pruned_enriched)} annotation-supported motif rows"
                )
            else:
                example_tf = tf_from_regulon_name(
                    next(iter(candidate_regulons), "TF_X_regulon")
                )
                # All candidate regulons were pruned away. Most common cause
                # is a TF-symbol convention mismatch between the GRN (regulon
                # names like ``PAX5_regulon``) and the annotation table
                # (column ``TF``). Falling back to candidates so the run
                # isn't silently empty downstream (AUCell would otherwise
                # score 0 cells x 0 regulons without complaint).
                _warnings.warn(
                    f"motif-annotation pruning removed all "
                    f"{len(candidate_regulons)} candidate regulons. Likely "
                    f"cause: motif_annotations TF symbols don't match the "
                    f"GRN regulon TF symbols (case / convention mismatch). "
                    f"Compare the regulon TF (e.g. "
                    f"`{example_tf}`) "
                    f"with the annotation table's TF column. Falling back "
                    f"to candidate GRN top-target regulons for AUCell.",
                    UserWarning,
                    stacklevel=2,
                )
                regulon_source = "candidate_grn_top_targets_after_failed_pruning"
                # Remove any pruned_regulons.json left over from a prior
                # successful-pruning run on the same output_dir. The
                # PipelineResult and manifest already report None for the
                # fallback path; if we leave the stale file in place, any
                # caller probing the filesystem directly would load a
                # previous run's pruned set and silently mismatch the
                # current run's regulon_source field. `missing_ok=True`
                # plus the OSError guard turns the unlikely cases (the
                # path is a directory, or owned by another user) into a
                # warning rather than aborting the pipeline after GRN,
                # topics, and cistarget have already written outputs.
                _stale = output_dir / "pruned_regulons.json"
                try:
                    _stale.unlink(missing_ok=True)
                except OSError as exc:
                    _warnings.warn(
                        f"could not remove stale {_stale}: {exc}. "
                        f"PipelineResult.pruned_regulons_path is None and "
                        f"manifest.json records the fallback, but any caller "
                        f"probing the filesystem directly may load the prior "
                        f"run's data. Remove the file by hand if downstream "
                        f"tooling reads it.",
                        UserWarning,
                        stacklevel=2,
                    )
                log(
                    f"      pruning removed all regulons; falling back to "
                    f"{len(candidate_regulons)} candidate regulons"
                )

    regulons_path = output_dir / "regulons.json"
    regulons_path.write_text(json.dumps(regulons, indent=2))
    log(f"      active regulons: {len(regulons)} ({regulon_source}) → {regulons_path.name}")

    # ---- 4c. enhancer → gene linking (optional, requires multiome + gene_coords) ----
    enhancer_links_path: Path | None = None
    enhancer_links: pd.DataFrame | None = None
    have_atac = atac_matrix_path is not None
    coords_df = _coerce_gene_coords(gene_coords) if gene_coords is not None else None
    if have_atac and coords_df is not None:
        import rustscenic.enhancer
        log(f"[7/8] enhancer: linking peaks → genes ({len(coords_df):,} TSS records)")
        t0 = time.perf_counter()
        # adata_atac is still in scope from the preproc/topics block above.
        # Use it directly rather than round-tripping through h5ad - saves the
        # disk read on big matrices and avoids dropping non-serialisable
        # obs/varm/uns the caller may have attached.
        adata_atac_for_link = adata_atac
        shared_atac_idx = _pipeline_match_atac_cell_indices(
            [str(cell) for cell in adata_rna.obs_names],
            [str(cell) for cell in adata_atac_for_link.obs_names],
        )
        if len(shared_atac_idx) == 0:
            log("      skipped - no shared barcodes between RNA and ATAC")
        else:
            # Two paths to peak coords:
            #   (a) `peaks` BED supplied - read coords from it (handles the
            #       case where var_names came from the BED name column and
            #       aren't `chr:start-end`-formatted).
            #   (b) `adata_atac` was passed pre-built - caller is expected
            #       to have either coord-formatted var_names OR `chrom`/
            #       `start`/`end` columns in `var`. enhancer.link_peaks_to_genes
            #       handles both via `peak_coords=None`.
            if peaks is not None:
                peak_coords = _peak_coords_from_bed(peaks, adata_atac_for_link.var_names)
            else:
                peak_coords = None
            enhancer_links = rustscenic.enhancer.link_peaks_to_genes(
                adata_rna,
                adata_atac_for_link,
                coords_df,
                peak_coords=peak_coords,
                max_distance=enhancer_max_distance,
                min_abs_corr=enhancer_min_abs_corr,
            )
            backend_execution["enhancer"] = _rust_execution_from_attrs(
                enhancer_links,
                "enhancer_link_pearson",
                "enhancer_link_pearson_sparse_rna",
            )
            elapsed["enhancer"] = time.perf_counter() - t0
            n_enhancer_links = int(len(enhancer_links))
            enhancer_links_path = output_dir / "enhancer_links.parquet"
            enhancer_links.to_parquet(enhancer_links_path, index=False)
            log(
                f"      {n_enhancer_links:,} peak-gene links in "
                f"{elapsed['enhancer']:.1f}s"
            )
            mark_memory("enhancer")
    elif have_atac and gene_coords is None:
        log("[7/8] enhancer: skipped (no gene_coords supplied)")
    else:
        log("[7/8] enhancer: skipped (no ATAC inputs)")

    # ---- 4d. eRegulon assembly (optional, needs grn + cistarget + enhancer) ----
    eregulons_path: Path | None = None
    n_eregulons: int | None = None
    if enhancer_links is not None and (enriched_for_eregulons is not None or region_motif_rankings is not None):
        import rustscenic.eregulon
        log("[7b/8] eRegulons: assembling TF × enhancer × target intersection")
        t0 = time.perf_counter()
        # Two paths to (TF → peaks) associations:
        # 1. EXACT: if region_motif_rankings supplied, run cistarget on
        #    the linked peaks against region rankings - true motif
        #    enrichment per peak per TF (matches scenicplus semantics).
        # 2. APPROXIMATE: gene-only path - attribute peaks via
        #    GRN targets ∩ enhancer links. Used when region rankings
        #    aren't available.
        if region_motif_rankings is not None:
            import rustscenic.cistarget
            log("      using region-based cistarget for exact peak attribution")
            peak_regulons, needed_peaks = _peak_regulons_and_projection_features(
                grn, enhancer_links
            )
            backend_execution["eregulon_peak_regulons"] = _rust_execution(
                "pipeline_peak_regulons_and_features_from_edges"
            )
            if peak_regulons:
                region_rankings_df = _coerce_rankings(
                    region_motif_rankings,
                    feature_names=_ranking_projection_features(
                        region_motif_rankings,
                        needed_peaks,
                    ),
                )
                region_enrich, enriched_with_peaks = _region_cistarget_with_peak_ids(
                    region_rankings_df,
                    peak_regulons,
                    top_frac=cistarget_top_frac,
                    auc_threshold=cistarget_auc_threshold,
                    nes_threshold=cistarget_nes_threshold,
                )
                backend_execution["eregulon_peak_attribution"] = _rust_execution_from_attrs(
                    enriched_with_peaks,
                    "cistarget_region_attribution_i16",
                    "cistarget_region_attribution_i32",
                    "cistarget_region_attribution_i64",
                    "pipeline_expand_region_cistarget_rows_f32",
                    "pipeline_expand_region_cistarget_rows_f64",
                )
                if motif_annotations_df is not None and not region_enrich.empty:
                    region_enrich = rustscenic.cistarget.prune_enriched_motifs(
                        region_enrich,
                        motif_annotations_df,
                        auc_threshold=cistarget_auc_threshold,
                    )
                    enriched_with_peaks = _filter_cistarget_peak_rows(
                        enriched_with_peaks,
                        region_enrich,
                )
                    backend_execution["eregulon_peak_filter"] = _rust_execution_from_attrs(
                        enriched_with_peaks,
                        "pipeline_filter_cistarget_peak_rows_f32",
                        "pipeline_filter_cistarget_peak_rows_f64",
                    )
                if cistarget_path is None:
                    cistarget_path = output_dir / "region_cistarget_enriched.parquet"
                    n_cistarget_rows = int(len(region_enrich))
                    region_enrich.to_parquet(cistarget_path, index=False)
                    log(
                        f"      {len(region_enrich):,} region-enriched pairs → "
                        f"{cistarget_path.name}"
                    )
                else:
                    region_enrich.to_parquet(
                        output_dir / "region_cistarget_enriched.parquet",
                        index=False,
                    )
            else:
                enriched_with_peaks = pd.DataFrame(
                    columns=["regulon", "motif", "peak_id", "auc"]
                )
        else:
            log("      gene-only - bridging via active regulon targets")
            enriched_with_peaks = _attribute_peaks_to_cistarget(
                enriched_for_eregulons, enhancer_links, regulons=regulons,
            )
            backend_execution["eregulon_peak_attribution"] = _rust_execution_from_attrs(
                enriched_with_peaks,
                "pipeline_attribute_peaks_to_cistarget_rows_f32",
                "pipeline_attribute_peaks_to_cistarget_rows_f64",
            )
        eregulons_df = rustscenic.eregulon._build_eregulons_dataframe(
            grn,
            enriched_with_peaks,
            enhancer_links,
            min_target_genes=eregulon_min_target_genes,
            min_enhancer_links=eregulon_min_enhancer_links,
        )
        backend_execution["eregulons"] = _rust_execution_from_attrs(
            eregulons_df,
            "eregulon_assemble",
            "eregulon_assemble_f32",
        )
        elapsed["eregulons"] = time.perf_counter() - t0
        n_eregulon_rows = int(len(eregulons_df))
        eregulons_path = output_dir / "eregulons.parquet"
        eregulons_df.to_parquet(eregulons_path, index=False)
        n_eregulons = int(eregulons_df.attrs.get("n_eregulons", 0))
        log(
            f"      {n_eregulons} eRegulons assembled in "
            f"{elapsed['eregulons']:.1f}s"
        )
        mark_memory("eregulons")
    elif gene_coords is not None and motif_rankings is not None and not have_atac:
        log("[7b/8] eRegulons: skipped (need ATAC for enhancer linking)")
    elif enriched_for_eregulons is None or enhancer_links is None:
        log("[7b/8] eRegulons: skipped (need motif rankings + enhancer links)")

    # ---- 5. AUCell ----
    log("[8/8] AUCell: per-cell regulon activity")
    t0 = time.perf_counter()
    active_regulon_pairs = list(iter_regulon_pairs(regulons))
    auc = rustscenic.aucell.score(
        adata_rna,
        active_regulon_pairs,
        top_frac=aucell_top_frac,
    )
    backend_execution["aucell"] = _rust_execution_from_attrs(
        auc,
        "aucell_score",
        "aucell_score_sparse_csr",
    )
    elapsed["aucell"] = time.perf_counter() - t0
    aucell_shape = [int(auc.shape[0]), int(auc.shape[1])]
    aucell_path = output_dir / "aucell.parquet"
    auc.to_parquet(aucell_path)
    log(f"      {auc.shape[0]:,} cells × {auc.shape[1]} regulons in {elapsed['aucell']:.1f}s")
    mark_memory("aucell")

    # ---- 6. integrate into AnnData ----
    _attach_aucell_to_obs(adata_rna, auc)
    backend_execution["integrated_adata"] = _python_io_execution("AnnData obs attachment and h5ad write")
    integrated_path = output_dir / "rna_with_regulons.h5ad"
    adata_rna.write_h5ad(integrated_path)
    log(f"      integrated → {integrated_path.name}")
    mark_memory("integrated_adata")

    result = PipelineResult(
        output_dir=output_dir,
        atac_matrix_path=atac_matrix_path,
        grn_path=grn_path,
        regulons_path=regulons_path,
        candidate_regulons_path=candidate_regulons_path,
        pruned_regulons_path=pruned_regulons_path,
        aucell_path=aucell_path,
        topics_dir=topics_dir,
        cistarget_path=cistarget_path,
        enhancer_links_path=enhancer_links_path,
        eregulons_path=eregulons_path,
        integrated_adata_path=integrated_path,
        elapsed=elapsed,
        memory=memory,
        n_cells=n_cells,
        n_grn_edges=n_grn_edges,
        n_regulons=len(regulons),
        n_candidate_regulons=len(candidate_regulons),
        n_pruned_regulons=n_pruned_regulons,
        n_cistarget_rows=n_cistarget_rows,
        n_enhancer_links=n_enhancer_links,
        n_eregulon_rows=n_eregulon_rows,
        n_eregulons=n_eregulons,
        aucell_shape=aucell_shape,
        regulon_source=regulon_source,
        backend_execution=backend_execution,
    )
    # Manifest is the single source of truth for "what did this run produce"
    (output_dir / "manifest.json").write_text(json.dumps(result.manifest(), indent=2))
    log(f"done. total: {sum(elapsed.values()):.1f}s. manifest → manifest.json")
    return result


def _rust_execution(*symbols: str) -> dict:
    return {"engine": "rust", "symbols": list(symbols)}


def _rust_execution_from_attrs(obj, *fallback_symbols: str) -> dict:
    backend = getattr(obj, "attrs", {}).get("rust_backend")
    if not isinstance(backend, dict):
        backend = getattr(obj, "uns", {}).get("rust_backend")
    if (
        isinstance(backend, dict)
        and backend.get("engine") == "rust"
        and isinstance(backend.get("symbols"), list)
        and all(isinstance(symbol, str) and symbol for symbol in backend["symbols"])
    ):
        return {"engine": "rust", "symbols": list(backend["symbols"])}
    return _rust_execution(*fallback_symbols)


def _skipped_execution(reason: str) -> dict:
    return {"engine": "skipped", "reason": reason}


def _python_io_execution(reason: str) -> dict:
    return {"engine": "python_io", "reason": reason}


def _candidate_regulons_from_grn(
    grn: pd.DataFrame,
    *,
    top_targets: int,
    min_targets: int,
) -> dict[str, list[str]]:
    """Build top-target candidate regulons in Rust."""
    if grn.empty:
        return {}
    names, target_lists = _pipeline_candidate_regulons_from_grn(
        grn["TF"].astype(str).tolist(),
        grn["target"].astype(str).tolist(),
        grn["importance"].to_numpy(dtype=np.float64, copy=False),
        int(top_targets),
        int(min_targets),
    )
    return dict(zip(names, target_lists, strict=True))


def _peak_regulons_from_edges(
    grn: pd.DataFrame,
    enhancer_links: pd.DataFrame,
) -> list[tuple[str, list[str]]]:
    peak_regulons, _ = _peak_regulons_and_projection_features(grn, enhancer_links)
    return peak_regulons


def _peak_regulons_and_projection_features(
    grn: pd.DataFrame,
    enhancer_links: pd.DataFrame,
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    names, peaks, features = _pipeline_peak_regulons_and_features(
        grn["TF"].astype(str).tolist(),
        grn["target"].astype(str).tolist(),
        enhancer_links["gene"].astype(str).tolist(),
        enhancer_links["peak_id"].astype(str).tolist(),
    )
    return list(zip(names, peaks, strict=True)), list(features)


def _coerce_adata(rna):
    """Accept AnnData, h5ad path, or (cells × genes) DataFrame."""
    import anndata as ad

    if isinstance(rna, ad.AnnData):
        return rna
    if isinstance(rna, (str, Path)):
        return ad.read_h5ad(rna)
    if isinstance(rna, pd.DataFrame):
        return ad.AnnData(X=as_float32_contiguous(rna.values), obs=pd.DataFrame(index=rna.index), var=pd.DataFrame(index=rna.columns))
    raise TypeError(f"rna: expected AnnData / path / DataFrame, got {type(rna).__name__}")


def _subset_atac_to_rna_cells(adata_rna, adata_atac, *, log):
    """Keep ATAC barcodes that exist in RNA, preserving ATAC row order."""
    matched_atac_idx = _pipeline_match_atac_cell_indices(
        [str(cell) for cell in adata_rna.obs_names],
        [str(cell) for cell in adata_atac.obs_names],
    )
    matched_atac_idx = np.asarray(matched_atac_idx, dtype=np.intp)
    if matched_atac_idx.size == 0:
        raise ValueError(
            "ATAC input shares no cell barcodes with RNA input. Raw 10x "
            "fragments include many non-cell barcodes, but at least one "
            "called cell barcode must match rna.obs_names."
        )
    if matched_atac_idx.size == adata_atac.n_obs:
        return adata_atac
    import warnings as _warnings
    _warnings.warn(
        f"subsetting ATAC from {adata_atac.n_obs:,} barcodes to "
        f"{matched_atac_idx.size:,} RNA-matched cells before topics. "
        "This avoids carrying raw 10x empty droplets through the "
        "pipeline; pass a pre-filtered adata_atac to control this "
        "step explicitly.",
        UserWarning,
        stacklevel=2,
    )
    adata_atac = adata_atac[matched_atac_idx].copy()
    log(f"      ATAC subset to RNA cells: {adata_atac.shape}")
    return adata_atac


def _load_tfs(tfs):
    if tfs is None:
        # Default: the bundled aertslab HGNC human TF list. Safe zero-config
        # starting point for the common hg38 workflow; override for mouse or
        # custom lists.
        from . import data
        return data.tfs(species="hs")
    # Species shortcut. Accept the same set of aliases ``data.tfs()`` accepts
    # (single source of truth in ``rustscenic.data._TF_ALIASES``), case-
    # insensitively, and route to the bundled list. Without this branch the
    # ``isinstance(str, Path)`` check below treats ``"hs"`` (or ``Path("hs")``)
    # as a relative path and crashes with ``FileNotFoundError: 'hs'``
    # (regression in v0.4.0).
    if isinstance(tfs, (str, Path)):
        from . import data
        if str(tfs).lower() in data._TF_ALIASES:
            return data.tfs(species=str(tfs))
        path = Path(tfs)
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return list(tfs)


def _coerce_rankings(rankings, *, feature_names: Iterable[str] | None = None):
    if isinstance(rankings, pd.DataFrame):
        df = _rankings_with_motif_index(rankings, Path("rankings"))
        if feature_names is not None:
            features = _feature_name_list(feature_names)
            if features:
                keep, _ = _ranking_column_projection(
                    list(df.columns),
                    features,
                    motif_col=None,
                )
                if not keep:
                    raise ValueError(
                        "none of the requested ranking features were present "
                        "in the rankings DataFrame. Check that ATAC peak IDs "
                        "match the motif-ranking column names."
                    )
                df = df.loc[:, keep]
        return df
    path = Path(rankings)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".feather"):
        return _rankings_with_motif_index(
            _read_rankings_file(path, feature_names=feature_names),
            path,
        )
    raise ValueError(f"unsupported motif-ranking format: {suffix}")


def _ranking_projection_features(
    rankings,
    feature_names: Iterable[str],
) -> Iterable[str] | None:
    """Project ranking files on read, but avoid copying DataFrame inputs."""
    if isinstance(rankings, pd.DataFrame):
        return None
    return feature_names


def _feature_name_list(feature_names: Iterable[str] | None) -> list[str]:
    return [] if feature_names is None else [str(x) for x in feature_names]


def _read_rankings_file(
    path: Path, *, feature_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read a motif ranking file, optionally projecting to needed features.

    Large aertslab region-ranking feathers can be tens of GB wide. When
    ``feature_names`` is supplied, read only the motif ID column plus columns
    for peaks used by the current run instead of materialising the full DB.
    """
    suffix = path.suffix.lower()
    features = _feature_name_list(feature_names)
    if not features:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".feather":
            return pd.read_feather(path)

    if suffix == ".feather":
        cols = _projected_ranking_columns(path, features, kind="feather")
        return pd.read_feather(path, columns=cols)
    if suffix == ".parquet":
        cols = _projected_ranking_columns(path, features, kind="parquet")
        return pd.read_parquet(path, columns=cols)
    raise ValueError(f"unsupported motif-ranking format: {suffix}")


def _projected_ranking_columns(path: Path, features: list[str], *, kind: str) -> list[str]:
    columns = _ranking_file_columns(path, kind=kind)
    motif_col = _detect_motif_column(columns, path)
    keep, examples = _ranking_column_projection(columns, features, motif_col=motif_col)
    feature_count = len(keep) - (1 if motif_col is not None and keep and keep[0] == motif_col else 0)
    if feature_count == 0:
        raise ValueError(
            "none of the current run's peak IDs were present in the "
            f"motif-ranking columns for {path.name}. First requested peaks: "
            f"{examples}. Check that the BED peak IDs match the ranking DB "
            "region IDs, or rename/subset the BED to the ranking convention."
        )
    return keep


def _ranking_column_projection(
    columns: list,
    features: list[str],
    *,
    motif_col: str | None,
) -> tuple[list, list[str]]:
    keep, examples = _pipeline_project_ranking_columns(
        [str(c) for c in columns],
        features,
        motif_col,
    )
    return keep, list(examples)


def _ranking_file_columns(path: Path, *, kind: str) -> list[str]:
    if kind == "feather":
        import pyarrow.ipc as ipc

        with ipc.open_file(str(path)) as reader:
            return list(reader.schema.names)
    if kind == "parquet":
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    raise ValueError(f"unsupported ranking file kind: {kind}")


def _detect_motif_column(columns: list[str], path: Path) -> str | None:
    if "motifs" in columns:
        return "motifs"
    if path.stem in columns:
        return path.stem
    for candidate in ("motif", "motif_id", "features", "feature"):
        if candidate in columns:
            return candidate
    first_col = columns[0] if columns else None
    if first_col is not None and str(first_col).lower().startswith("motif"):
        return first_col
    return None


def _coerce_motif_annotations(annotations):
    if isinstance(annotations, pd.DataFrame):
        return annotations
    path = Path(annotations)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix in (".csv", ".tsv", ".txt"):
        return pd.read_csv(path, sep="\t" if suffix in (".tsv", ".txt") else ",")
    raise ValueError(f"unsupported motif_annotations format: {suffix}")



def _rankings_with_motif_index(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Normalise aertslab-style ranking files to motifs as the index.

    Public aertslab feather files usually store motif IDs in a ``motifs``
    column; ad hoc parquet exports often preserve that as the first string
    column. Accept both shapes so users can pass file paths directly to
    ``pipeline.run`` instead of hand-loading rankings first.
    """
    if df.index.name is not None and not isinstance(df.index, pd.RangeIndex):
        return df
    if "motifs" in df.columns:
        return df.set_index("motifs")
    if path.stem in df.columns:
        return df.set_index(path.stem)
    first_col = df.columns[0] if len(df.columns) else None
    if first_col is not None and (
        pd.api.types.is_string_dtype(df[first_col])
        or pd.api.types.is_object_dtype(df[first_col])
    ):
        if _all_non_index_columns_numeric(df):
            return df.set_index(first_col)
    return df


def _all_non_index_columns_numeric(df: pd.DataFrame) -> bool:
    """Check rank-value dtypes without materialising a column-dropped frame."""
    return all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes.iloc[1:])


def _attribute_peaks_to_cistarget(
    enriched: pd.DataFrame,
    enhancer_links: pd.DataFrame,
    regulons: dict,
) -> pd.DataFrame:
    """Bridge gene-based cistarget output to peak-aware eRegulon input.

    Cistarget on a gene-based motif ranking emits ``(regulon, motif, auc)``
    rows but no peak column - the eRegulon assembler requires one. Until
    region-based cistarget ships, attribute each enriched TF's peaks via
    the TF's regulon-target list ∩ enhancer-link peak set: a peak is
    associated with TF X if it links to a gene that's in X's regulon.

    The Rust helper below replaces the pandas merge bridge and preserves the
    same first-seen de-duplication/order semantics: gene→peak, then TF→peak,
    then enriched cistarget row expansion. Passing ``regulons`` keeps this on
    the same top-N target set that was scored by cistarget.
    """
    if enriched.empty:
        return pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])

    regulon_pairs = [
        (str(regulon_name), [str(g) for g in targets])
        for regulon_name, targets in iter_regulon_pairs(regulons)
    ]
    enriched_regulons = enriched["regulon"].astype(str).tolist()
    enriched_motifs = (
        enriched["motif"].astype(str).tolist()
        if "motif" in enriched.columns else None
    )
    enriched_aucs = _auc_column_arg(enriched["auc"], name="enriched['auc']")
    enhancer_genes = [str(v) for v in enhancer_links["gene"].to_numpy(copy=False)]
    enhancer_peaks = [str(v) for v in enhancer_links["peak_id"].to_numpy(copy=False)]
    kernel = (
        _pipeline_attribute_peak_rows_f32
        if enriched_aucs.dtype == np.float32
        else _pipeline_attribute_peak_rows_f64
    )
    backend_symbol = (
        "pipeline_attribute_peaks_to_cistarget_rows_f32"
        if enriched_aucs.dtype == np.float32
        else "pipeline_attribute_peaks_to_cistarget_rows_f64"
    )
    regulon_values, motif_values, peak_values, auc_values = kernel(
        enriched_regulons,
        enriched_motifs,
        enriched_aucs,
        [name for name, _ in regulon_pairs],
        [targets for _, targets in regulon_pairs],
        enhancer_genes,
        enhancer_peaks,
    )
    if len(regulon_values) == 0:
        out = pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])
        out.attrs["rust_backend"] = {"engine": "rust", "symbols": [backend_symbol]}
        return out

    out = pd.DataFrame(
        {
            "regulon": regulon_values,
            "motif": motif_values,
            "peak_id": peak_values,
            "auc": auc_values,
        },
        columns=["regulon", "motif", "peak_id", "auc"],
    ).reset_index(drop=True)
    out.attrs["rust_backend"] = {"engine": "rust", "symbols": [backend_symbol]}
    return out


def _region_cistarget_with_peak_ids(
    region_rankings: pd.DataFrame,
    peak_regulons: list[tuple[str, list[str]]],
    *,
    top_frac: float,
    auc_threshold: float,
    nes_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run region cistarget and retain the motif-supported peak IDs.

    ``cistarget.enrich`` answers whether a motif is enriched for a peak
    set, but eRegulon assembly also needs peak identifiers. Keep only
    peaks from the source peak set that lie inside the motif's top-ranked
    region window, instead of attributing every linked peak to every
    enriched motif. ``nes_threshold`` is passed through to ``enrich`` so
    the region path honours the same canonical pyscenic / pycistarget
    cutoff as the gene path.
    """
    import rustscenic.cistarget

    region_enrich = rustscenic.cistarget.enrich(
        region_rankings,
        peak_regulons,
        top_frac=top_frac,
        auc_threshold=auc_threshold,
        nes_threshold=nes_threshold,
    )
    if region_enrich.empty:
        empty = pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])
        empty.attrs["rust_backend"] = _region_attribution_backend(region_enrich)
        return region_enrich, empty

    n_regions = region_rankings.shape[1]
    rank_cutoff = max(1, int(np.ceil(top_frac * n_regions)))
    peak_regulon_names = [str(name) for name, _ in peak_regulons]
    peak_regulon_peaks = [[str(p) for p in peaks] for _, peaks in peak_regulons]
    if not any(peak_regulon_peaks):
        empty = pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])
        empty.attrs["rust_backend"] = _region_attribution_backend(region_enrich)
        return region_enrich, empty

    ranking_values, kernel, rank_cutoff_arg = _region_peak_values_kernel_arg(
        region_rankings,
        rank_cutoff,
    )
    peak_values_symbol = _region_peak_values_backend_symbol(kernel)
    motif_names = [str(m) for m in region_rankings.index]
    peak_names = [str(p) for p in region_rankings.columns]
    common_args = (
        motif_names,
        peak_names,
        peak_regulon_names,
        peak_regulon_peaks,
        region_enrich["regulon"].astype(str).tolist(),
        region_enrich["motif"].astype(str).tolist(),
    )
    row_idx, peak_ids = kernel(
        ranking_values,
        *common_args,
        rank_cutoff_arg,
    )
    if len(row_idx) == 0:
        empty = pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])
        empty.attrs["rust_backend"] = _region_attribution_backend(
            region_enrich,
            peak_values_symbol,
        )
        return region_enrich, empty
    region_regulons = region_enrich["regulon"].astype(str).tolist()
    region_motifs = region_enrich["motif"].astype(str).tolist()
    region_aucs = _auc_column_arg(region_enrich["auc"], name="region_enrich['auc']")
    expand_kernel = (
        _pipeline_expand_region_rows_f32
        if region_aucs.dtype == np.float32
        else _pipeline_expand_region_rows_f64
    )
    expand_symbol = (
        "pipeline_expand_region_cistarget_rows_f32"
        if region_aucs.dtype == np.float32
        else "pipeline_expand_region_cistarget_rows_f64"
    )
    regulon_values, motif_values, peak_values, auc_values = expand_kernel(
        row_idx,
        peak_ids,
        region_regulons,
        region_motifs,
        region_aucs,
    )
    enriched = pd.DataFrame(
        {
            "regulon": regulon_values,
            "motif": motif_values,
            "peak_id": peak_values,
            "auc": auc_values,
        },
        columns=["regulon", "motif", "peak_id", "auc"],
    )
    enriched.attrs["rust_backend"] = _region_attribution_backend(
        region_enrich,
        peak_values_symbol,
        expand_symbol,
    )
    return region_enrich, enriched


def _auc_column_arg(values: pd.Series, *, name: str) -> np.ndarray:
    arr = values.to_numpy(copy=False)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    return arr.astype(np.float64, copy=False)


def _region_peak_values_kernel_arg(rankings: pd.DataFrame, rank_cutoff: int):
    values, kernel, rank_cutoff_arg = _region_rankings_kernel_arg(rankings, rank_cutoff)
    if kernel is _cistarget_region_attribution_i16:
        return values, _cistarget_region_peak_values_i16, rank_cutoff_arg
    if kernel is _cistarget_region_attribution_i32:
        return values, _cistarget_region_peak_values_i32, rank_cutoff_arg
    if kernel is _cistarget_region_attribution_i64:
        return values, _cistarget_region_peak_values_i64, rank_cutoff_arg
    raise RuntimeError("unknown region cistarget attribution kernel")


def _region_peak_values_backend_symbol(kernel) -> str:
    if kernel is _cistarget_region_peak_values_i16:
        return "cistarget_region_attribution_peak_values_i16"
    if kernel is _cistarget_region_peak_values_i32:
        return "cistarget_region_attribution_peak_values_i32"
    if kernel is _cistarget_region_peak_values_i64:
        return "cistarget_region_attribution_peak_values_i64"
    raise RuntimeError("unknown region cistarget peak-value kernel")


def _region_attribution_backend(region_enrich: pd.DataFrame, *symbols: str) -> dict:
    return {
        "engine": "rust",
        "symbols": _rust_backend_symbols(region_enrich)
        + [symbol for symbol in symbols if symbol],
    }


def _filter_cistarget_peak_rows(
    enriched_with_peaks: pd.DataFrame,
    keep: pd.DataFrame,
) -> pd.DataFrame:
    if enriched_with_peaks.empty or keep.empty:
        out = enriched_with_peaks.iloc[[]].copy().reset_index(drop=True)
        out.attrs["rust_backend"] = {
            "engine": "rust",
            "symbols": _rust_backend_symbols(enriched_with_peaks),
        }
        return out
    auc_values = _auc_column_arg(enriched_with_peaks["auc"], name="enriched_with_peaks['auc']")
    kernel = (
        _pipeline_filter_cistarget_peak_rows_f32
        if auc_values.dtype == np.float32
        else _pipeline_filter_cistarget_peak_rows_f64
    )
    filter_symbol = (
        "pipeline_filter_cistarget_peak_rows_f32"
        if auc_values.dtype == np.float32
        else "pipeline_filter_cistarget_peak_rows_f64"
    )
    regulon_values, motif_values, peak_values, auc_values = kernel(
        enriched_with_peaks["regulon"].astype(str).tolist(),
        enriched_with_peaks["motif"].astype(str).tolist(),
        enriched_with_peaks["peak_id"].astype(str).tolist(),
        auc_values,
        keep["regulon"].astype(str).tolist(),
        keep["motif"].astype(str).tolist(),
    )
    out = pd.DataFrame(
        {
            "regulon": regulon_values,
            "motif": motif_values,
            "peak_id": peak_values,
            "auc": auc_values,
        },
        columns=["regulon", "motif", "peak_id", "auc"],
    ).reset_index(drop=True)
    out.attrs["rust_backend"] = {
        "engine": "rust",
        "symbols": _rust_backend_symbols(enriched_with_peaks) + [filter_symbol],
    }
    return out


def _rust_backend_symbols(obj) -> list[str]:
    backend = getattr(obj, "attrs", {}).get("rust_backend")
    if (
        isinstance(backend, dict)
        and backend.get("engine") == "rust"
        and isinstance(backend.get("symbols"), list)
    ):
        return [symbol for symbol in backend["symbols"] if isinstance(symbol, str) and symbol]
    return []


def _region_rankings_kernel_arg(rankings: pd.DataFrame, rank_cutoff: int):
    values = rankings.to_numpy(copy=False)
    if values.dtype == object:
        raise TypeError("region rankings DataFrame has dtype=object")
    if values.dtype == np.int16:
        return (
            values,
            _cistarget_region_attribution_i16,
            min(int(rank_cutoff), int(np.iinfo(np.int16).max)),
        )
    if values.dtype == np.int32:
        return (
            values,
            _cistarget_region_attribution_i32,
            int(rank_cutoff),
        )
    if values.dtype == np.int64:
        return (
            values,
            _cistarget_region_attribution_i64,
            int(rank_cutoff),
        )
    if values.dtype == np.float32:
        return (
            _rankings_to_i32_f32(values),
            _cistarget_region_attribution_i32,
            int(rank_cutoff),
        )
    if values.dtype == np.float64:
        return (
            _rankings_to_i32_f64(values),
            _cistarget_region_attribution_i32,
            int(rank_cutoff),
        )
    raise TypeError("region rankings must contain integer rank values")


def _peak_coords_from_bed(bed_path, atac_var_names):
    """Build a per-peak chrom/start/end DataFrame indexed by ATAC var_names.

    The orchestrator hands `link_peaks_to_genes` an explicit `peak_coords`
    rather than relying on `chr:start-end` parsing of var_names - that
    parser only works when no name column was present in the BED.
    """
    peak_ids, chroms, starts, ends = _preproc_peak_coords_for_names(
        str(Path(bed_path)),
        [str(name) for name in atac_var_names],
    )
    return pd.DataFrame(
        {
            "chrom": list(chroms),
            "start": np.asarray(starts, dtype=np.uint32),
            "end": np.asarray(ends, dtype=np.uint32),
        },
        index=pd.Index(list(peak_ids), name="name"),
    )


def _coerce_gene_coords(coords):
    if isinstance(coords, pd.DataFrame):
        df = coords
    else:
        path = Path(coords)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in (".csv", ".tsv"):
            df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        else:
            raise ValueError(f"unsupported gene_coords format: {suffix}")
    required = {"gene", "chrom", "tss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"gene_coords missing columns: {sorted(missing)}. "
            f"Required: gene, chrom, tss."
        )
    return df


def _attach_aucell_to_obs(adata_rna, auc: pd.DataFrame) -> None:
    """Attach AUCell columns without per-regulon pandas inserts.

    The AUCell result is produced in RNA cell order in normal pipeline runs.
    If a caller supplies an externally aligned AUCell frame in future, reindex
    only the AUCell frame, then rebuild obs once. Assigning thousands of
    regulon columns one by one fragments pandas' block manager and inflates the
    final integrated-output step.
    """
    obs = adata_rna.obs
    overlap = [col for col in auc.columns if col in obs.columns]
    if overlap:
        obs = obs.drop(columns=overlap)

    if not pd.Index(auc.index).equals(pd.Index(adata_rna.obs_names)):
        auc = auc.reindex(adata_rna.obs_names)

    data = {col: obs[col] for col in obs.columns}
    data.update(
        {
            col: pd.Series(
                auc[col].to_numpy(copy=False),
                index=obs.index,
                name=col,
            )
            for col in auc.columns
        }
    )
    adata_rna.obs = pd.DataFrame(data, index=obs.index)


def _peak_rss_gb() -> float:
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(float(rss) / (1024**3), 6)
    return round(float(rss) / (1024**2), 6)


class _Logger:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __call__(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


__all__ = ["run", "PipelineResult"]
