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
    6. aucell   (RNA + regulons)         → per-cell regulon activity

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
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class PipelineResult:
    """Artifacts and metadata from a pipeline run.

    All file paths point inside ``output_dir``. Stages that were skipped
    because inputs weren't provided have ``None`` for their result path.
    """

    output_dir: Path
    rna_path: Optional[Path] = None
    atac_matrix_path: Optional[Path] = None
    grn_path: Optional[Path] = None
    regulons_path: Optional[Path] = None
    aucell_path: Optional[Path] = None
    topics_dir: Optional[Path] = None
    cistarget_path: Optional[Path] = None
    enhancer_links_path: Optional[Path] = None
    eregulons_path: Optional[Path] = None
    integrated_adata_path: Optional[Path] = None
    elapsed: dict = field(default_factory=dict)
    n_cells: Optional[int] = None
    n_regulons: Optional[int] = None
    n_eregulons: Optional[int] = None

    def manifest(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


def run(
    rna: Union[str, Path, Any],
    output_dir: Union[str, Path],
    *,
    fragments: Union[str, Path, None] = None,
    peaks: Union[str, Path, None] = None,
    tfs: Union[str, Path, Iterable[str], None] = None,
    motif_rankings: Union[str, Path, pd.DataFrame, None] = None,
    gene_coords: Union[str, Path, pd.DataFrame, None] = None,
    grn_n_estimators: int = 500,
    grn_top_targets: int = 50,
    aucell_top_frac: float = 0.05,
    topics_n_topics: int = 30,
    topics_n_passes: int = 3,
    cistarget_top_frac: float = 0.05,
    cistarget_auc_threshold: float = 0.05,
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
    fragments, peaks
        Paths to a 10x-style ``fragments.tsv[.gz]`` and peak BED. When
        both are provided, rustscenic.preproc builds the cells × peaks
        AnnData and topics fits on it.
    tfs
        Candidate transcription factor names. Path to a newline-separated
        file, an iterable of strings, or ``None`` (in which case the
        caller must provide them; see `rustscenic.data.tfs()` for
        bundled lists once shipped).
    motif_rankings
        Motif ranking DataFrame, or a path to a parquet / feather file
        with motifs as rows and genes as columns. If provided, cistarget
        runs to filter regulons to motif-enriched TFs.
    gene_coords
        DataFrame with columns ``['gene', 'chrom', 'tss']``, or a path
        to a parquet/csv file with the same shape. When supplied
        alongside ``fragments`` + ``peaks``, the orchestrator runs
        ``rustscenic.enhancer.link_peaks_to_genes`` and (if cistarget
        also ran) ``rustscenic.eregulon.build_eregulons``. Current
        eRegulons use a gene-based cistarget bridge onto linked peaks;
        strict scenicplus region-cistrome parity is still pending.

    Returns
    -------
    PipelineResult — dataclass with paths to every artifact written.
    """
    import anndata as ad
    import rustscenic.aucell
    import rustscenic.grn

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _Logger(verbose)
    elapsed: dict = {}

    # ---- 1. load / normalise RNA ----
    log("[1/6] loading RNA expression")
    adata_rna = _coerce_adata(rna)
    n_cells = adata_rna.n_obs
    log(f"      RNA shape: {adata_rna.shape}")

    # ---- 2. preproc + topics (only if ATAC inputs provided) ----
    atac_matrix_path = None
    topics_dir = None
    if fragments is not None and peaks is not None:
        import rustscenic.preproc
        log("[2/6] preproc: fragments + peaks → cells × peaks")
        t0 = time.perf_counter()
        adata_atac = rustscenic.preproc.fragments_to_matrix(fragments, peaks)
        elapsed["preproc"] = time.perf_counter() - t0
        log(f"      ATAC shape: {adata_atac.shape}, took {elapsed['preproc']:.1f}s")

        atac_matrix_path = output_dir / "atac_cells_by_peaks.h5ad"
        adata_atac.write_h5ad(atac_matrix_path)

        # Topics on the sparse ATAC matrix
        import rustscenic.topics
        log(f"[3/6] topics: fitting LDA K={topics_n_topics}")
        t0 = time.perf_counter()
        topics_result = rustscenic.topics.fit(
            adata_atac,
            n_topics=topics_n_topics,
            n_passes=topics_n_passes,
            seed=seed,
        )
        elapsed["topics"] = time.perf_counter() - t0
        log(f"      fit in {elapsed['topics']:.1f}s")

        topics_dir = output_dir / "topics"
        topics_dir.mkdir(exist_ok=True)
        # topics_result is typically a (cell_topic, topic_peak) pair
        if hasattr(topics_result, "cell_topic"):
            np.save(topics_dir / "cell_topic.npy", topics_result.cell_topic)
            np.save(topics_dir / "topic_peak.npy", topics_result.topic_peak)
    else:
        log("[2/6] preproc + topics: skipped (no fragments / peaks)")
        log("[3/6] topics: skipped")

    # ---- 3. GRN ----
    log("[4/6] GRN inference on RNA")
    tf_list = _load_tfs(tfs)
    log(f"      {len(tf_list)} candidate TFs")
    t0 = time.perf_counter()
    grn = rustscenic.grn.infer(
        adata_rna,
        tf_names=tf_list,
        n_estimators=grn_n_estimators,
        seed=seed,
        verbose=False,
    )
    elapsed["grn"] = time.perf_counter() - t0
    grn_path = output_dir / "grn.parquet"
    grn.to_parquet(grn_path, index=False)
    log(f"      {len(grn):,} edges in {elapsed['grn']:.1f}s → {grn_path.name}")

    # ---- 4. build regulons ----
    log(f"[5/6] regulons: top-{grn_top_targets} targets per TF")
    regulons = {}
    for tf in grn["TF"].unique():
        top = grn[grn["TF"] == tf].nlargest(grn_top_targets, "importance")["target"].tolist()
        if len(top) >= 10:
            regulons[f"{tf}_regulon"] = top
    regulons_path = output_dir / "regulons.json"
    regulons_path.write_text(json.dumps(regulons, indent=2))
    log(f"      {len(regulons)} regulons (≥10 targets) → {regulons_path.name}")

    # ---- 4b. cistarget (optional) ----
    cistarget_path = None
    enriched: Optional[pd.DataFrame] = None
    if motif_rankings is not None:
        import rustscenic.cistarget
        rankings_df = _coerce_rankings(motif_rankings)
        log(f"[5b/8] cistarget: {len(rankings_df):,} motifs × {rankings_df.shape[1]:,} genes")
        t0 = time.perf_counter()
        enriched = rustscenic.cistarget.enrich(
            rankings_df,
            [(n, g) for n, g in regulons.items()],
            top_frac=cistarget_top_frac,
            auc_threshold=cistarget_auc_threshold,
        )
        elapsed["cistarget"] = time.perf_counter() - t0
        cistarget_path = output_dir / "cistarget_enriched.parquet"
        enriched.to_parquet(cistarget_path, index=False)
        log(f"      {len(enriched):,} enriched pairs in {elapsed['cistarget']:.1f}s")

    # ---- 4c. enhancer → gene linking (optional, requires multiome + gene_coords) ----
    enhancer_links_path: Optional[Path] = None
    enhancer_links: Optional[pd.DataFrame] = None
    have_atac = atac_matrix_path is not None
    coords_df = _coerce_gene_coords(gene_coords) if gene_coords is not None else None
    if have_atac and coords_df is not None:
        import rustscenic.enhancer
        log(f"[5c/8] enhancer: linking peaks → genes ({len(coords_df):,} TSS records)")
        t0 = time.perf_counter()
        adata_atac_for_link = ad.read_h5ad(atac_matrix_path)
        common = adata_rna.obs_names.intersection(adata_atac_for_link.obs_names)
        if len(common) == 0:
            log("      skipped — no shared barcodes between RNA and ATAC")
        else:
            # If var_names came from the peak BED's name column they may
            # not be coord-formatted (`chr1:100-200`). Read coords from
            # the BED file directly and align to var_names so the linker
            # always has chrom/start/end.
            peak_coords = _peak_coords_from_bed(peaks, adata_atac_for_link.var_names)
            enhancer_links = rustscenic.enhancer.link_peaks_to_genes(
                adata_rna[common].copy(),
                adata_atac_for_link[common].copy(),
                coords_df,
                peak_coords=peak_coords,
                max_distance=enhancer_max_distance,
                min_abs_corr=enhancer_min_abs_corr,
            )
            elapsed["enhancer"] = time.perf_counter() - t0
            enhancer_links_path = output_dir / "enhancer_links.parquet"
            enhancer_links.to_parquet(enhancer_links_path, index=False)
            log(
                f"      {len(enhancer_links):,} peak-gene links in "
                f"{elapsed['enhancer']:.1f}s"
            )
    elif have_atac and gene_coords is None:
        log("[5c/8] enhancer: skipped (no gene_coords supplied)")
    else:
        log("[5c/8] enhancer: skipped (no ATAC inputs)")

    # ---- 4d. eRegulon assembly (optional, needs grn + cistarget + enhancer) ----
    eregulons_path: Optional[Path] = None
    n_eregulons: Optional[int] = None
    if enriched is not None and enhancer_links is not None:
        import rustscenic.eregulon
        log("[5d/8] eRegulons: assembling TF × enhancer × target intersection")
        t0 = time.perf_counter()
        # Cistarget here is gene-based — its output identifies which TFs
        # are motif-enriched in each regulon, but doesn't carry a peak
        # column. The eRegulon assembler needs (TF → peaks) associations,
        # so we bridge: each enriched TF gets attributed the peaks linked
        # to its GRN targets via the enhancer DataFrame. Approximate but
        # correct in spirit until region-based cistarget ships in v0.3.
        enriched_with_peaks = _attribute_peaks_to_cistarget(
            enriched, grn, enhancer_links
        )
        eregs = rustscenic.eregulon.build_eregulons(
            grn,
            enriched_with_peaks,
            enhancer_links,
            min_target_genes=eregulon_min_target_genes,
            min_enhancer_links=eregulon_min_enhancer_links,
        )
        elapsed["eregulons"] = time.perf_counter() - t0
        eregulons_path = output_dir / "eregulons.parquet"
        rustscenic.eregulon.eregulons_to_dataframe(eregs).to_parquet(
            eregulons_path, index=False
        )
        n_eregulons = len(eregs)
        log(
            f"      {n_eregulons} eRegulons assembled in "
            f"{elapsed['eregulons']:.1f}s"
        )
    elif gene_coords is not None and motif_rankings is not None and not have_atac:
        log("[5d/8] eRegulons: skipped (need ATAC for enhancer linking)")
    elif enriched is None or enhancer_links is None:
        log("[5d/8] eRegulons: skipped (need cistarget + enhancer links)")

    # ---- 5. AUCell ----
    log("[6/6] AUCell: per-cell regulon activity")
    t0 = time.perf_counter()
    auc = rustscenic.aucell.score(
        adata_rna,
        [(n, g) for n, g in regulons.items()],
        top_frac=aucell_top_frac,
    )
    elapsed["aucell"] = time.perf_counter() - t0
    aucell_path = output_dir / "aucell.parquet"
    auc.to_parquet(aucell_path)
    log(f"      {auc.shape[0]:,} cells × {auc.shape[1]} regulons in {elapsed['aucell']:.1f}s")

    # ---- 6. integrate into AnnData ----
    adata_rna.obs = adata_rna.obs.join(auc, how="left")
    integrated_path = output_dir / "rna_with_regulons.h5ad"
    adata_rna.write_h5ad(integrated_path)
    log(f"      integrated → {integrated_path.name}")

    result = PipelineResult(
        output_dir=output_dir,
        atac_matrix_path=atac_matrix_path,
        grn_path=grn_path,
        regulons_path=regulons_path,
        aucell_path=aucell_path,
        topics_dir=topics_dir,
        cistarget_path=cistarget_path,
        enhancer_links_path=enhancer_links_path,
        eregulons_path=eregulons_path,
        integrated_adata_path=integrated_path,
        elapsed=elapsed,
        n_cells=n_cells,
        n_regulons=len(regulons),
        n_eregulons=n_eregulons,
    )
    # Manifest is the single source of truth for "what did this run produce"
    (output_dir / "manifest.json").write_text(json.dumps(result.manifest(), indent=2))
    log(f"done. total: {sum(elapsed.values()):.1f}s. manifest → manifest.json")
    return result


def _coerce_adata(rna):
    """Accept AnnData, h5ad path, or (cells × genes) DataFrame."""
    import anndata as ad

    if isinstance(rna, ad.AnnData):
        return rna
    if isinstance(rna, (str, Path)):
        return ad.read_h5ad(rna)
    if isinstance(rna, pd.DataFrame):
        return ad.AnnData(X=rna.values.astype(np.float32), obs=pd.DataFrame(index=rna.index), var=pd.DataFrame(index=rna.columns))
    raise TypeError(f"rna: expected AnnData / path / DataFrame, got {type(rna).__name__}")


def _load_tfs(tfs):
    if tfs is None:
        # Default: the bundled aertslab HGNC human TF list. Safe zero-config
        # starting point for the common hg38 workflow; override for mouse or
        # custom lists.
        from . import data
        return data.tfs(species="hs")
    if isinstance(tfs, (str, Path)):
        path = Path(tfs)
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        return lines
    return list(tfs)


def _coerce_rankings(rankings):
    if isinstance(rankings, pd.DataFrame):
        return rankings
    path = Path(rankings)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path).set_index(path.stem)
    raise ValueError(f"unsupported motif-ranking format: {suffix}")


def _attribute_peaks_to_cistarget(
    enriched: pd.DataFrame,
    grn: pd.DataFrame,
    enhancer_links: pd.DataFrame,
) -> pd.DataFrame:
    """Bridge gene-based cistarget output to peak-aware eRegulon input.

    Cistarget on a gene-based motif ranking emits ``(regulon, motif, auc)``
    rows but no peak column — the eRegulon assembler requires one. Until
    region-based cistarget ships, attribute each enriched TF's peaks via
    the GRN's predicted targets ∩ enhancer-link peak set: a peak is
    associated with TF X if it links to a gene that GRN predicts X
    regulates.
    """
    grn_targets_by_tf: dict[str, set[str]] = (
        grn.groupby("TF")["target"].apply(set).to_dict()
    )
    peaks_by_target: dict[str, set[str]] = (
        enhancer_links.groupby("gene")["peak_id"].apply(set).to_dict()
    )
    rows = []
    for _, ct_row in enriched.iterrows():
        tf = str(ct_row["regulon"]).replace("_regulon", "")
        targets = grn_targets_by_tf.get(tf, set())
        peaks_for_tf: set[str] = set()
        for tg in targets:
            peaks_for_tf.update(peaks_by_target.get(tg, set()))
        for peak in peaks_for_tf:
            rows.append({
                "regulon": ct_row["regulon"],
                "motif": ct_row.get("motif"),
                "peak_id": peak,
                "auc": ct_row["auc"],
            })
    if not rows:
        # Preserve schema so downstream eRegulon validation has columns.
        return pd.DataFrame(columns=["regulon", "motif", "peak_id", "auc"])
    return pd.DataFrame(rows)


def _peak_coords_from_bed(bed_path, atac_var_names):
    """Build a per-peak chrom/start/end DataFrame indexed by ATAC var_names.

    The orchestrator hands `link_peaks_to_genes` an explicit `peak_coords`
    rather than relying on `chr:start-end` parsing of var_names — that
    parser only works when no name column was present in the BED.
    """
    import gzip as _gzip
    bed_path = Path(bed_path)
    opener = _gzip.open if str(bed_path).endswith(".gz") else open
    rows = []
    with opener(bed_path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            name = parts[3] if len(parts) >= 4 else f"{chrom}:{start}-{end}"
            rows.append((name, chrom, start, end))
    bed_df = pd.DataFrame(rows, columns=["name", "chrom", "start", "end"]).set_index("name")
    # Reindex to match the ATAC AnnData var_names; missing rows fall through
    # silently here, the linker will warn separately if alignment is poor.
    aligned = bed_df.reindex(list(atac_var_names))
    return aligned[["chrom", "start", "end"]].dropna()


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


class _Logger:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __call__(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


__all__ = ["run", "PipelineResult"]
