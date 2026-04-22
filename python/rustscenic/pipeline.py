"""End-to-end SCENIC+ orchestrator.

Public API:
    rustscenic.pipeline.run(rna, output_dir, *, fragments=None, peaks=None,
                            tfs=None, motif_rankings=None, ...) -> PipelineResult

One call runs every SCENIC+ stage the user provides input for:

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
    integrated_adata_path: Optional[Path] = None
    elapsed: dict = field(default_factory=dict)
    n_cells: Optional[int] = None
    n_regulons: Optional[int] = None

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
    grn_n_estimators: int = 500,
    grn_top_targets: int = 50,
    aucell_top_frac: float = 0.05,
    topics_n_topics: int = 30,
    topics_n_passes: int = 3,
    cistarget_top_frac: float = 0.05,
    cistarget_auc_threshold: float = 0.05,
    seed: int = 777,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full SCENIC+ pipeline end-to-end.

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
    if motif_rankings is not None:
        import rustscenic.cistarget
        rankings_df = _coerce_rankings(motif_rankings)
        log(f"[5b/6] cistarget: {len(rankings_df):,} motifs × {rankings_df.shape[1]:,} genes")
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
        integrated_adata_path=integrated_path,
        elapsed=elapsed,
        n_cells=n_cells,
        n_regulons=len(regulons),
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


class _Logger:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __call__(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


__all__ = ["run", "PipelineResult"]
