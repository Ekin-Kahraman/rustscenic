"""Audited native recipes with explicit motif support and seeded AUCell.

These recipes reproduce the native analysis stages used in the paper, with
the subsequently verified scoring correction. They do not guarantee accuracy
on a new dataset. No candidate-network fallback or reference-tool dependency.
"""
from __future__ import annotations

import json
import hashlib
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import aucell, cistarget, grn
from .pipeline import (
    _coerce_adata, _coerce_gene_coords, _coerce_motif_annotations,
    _coerce_rankings_with_metadata, _matrix_profile, _load_tfs,
)
from ._stage_utils import tf_from_regulon_name

GRN_SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1,
                    subsample=.9, max_depth=3, early_stop_window=25,
                    early_stop_mode="arboreto")
SCORING_SEED = 20261106


def load_rankings(source, features):
    """Project on read while preserving all motifs and original rank positions."""
    frame, metadata = _coerce_rankings_with_metadata(source, feature_names=features)
    declared = None
    if isinstance(source, pd.DataFrame):
        declared = source.attrs.get("rank_universe_size")
        original_width = len(source.columns) - int("motifs" in source.columns)
    else:
        sidecar = Path(source).with_suffix(".json")
        if sidecar.exists():
            info = json.loads(sidecar.read_text())
            declared = info.get("original_rank_universe", info.get("rank_universe_size"))
        original_width = metadata.get("rank_universe_size") or frame.shape[1]
    universe = declared if declared is not None else original_width
    if isinstance(universe, bool) or int(universe) != universe or universe < frame.shape[1]:
        raise ValueError("rank_universe_size must be an integer at least as large as the loaded feature axis")
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("rankings must have non-empty, unique motif and feature axes")
    # Bounded rank validation catches lost projection metadata without copying
    # an entire wide motif database or renumbering the authentic ranks.
    fingerprint = hashlib.sha256(str(int(universe)).encode())
    rank_batch = max(1, (8 * 1024 ** 2) // (max(1, frame.shape[1]) * 8))
    for start in range(0, len(frame), rank_batch):
        block = frame.iloc[start:start + rank_batch].to_numpy(copy=False)
        if (not np.issubdtype(block.dtype, np.number) or not np.isfinite(block).all()
                or (block < 0).any() or (block >= universe).any()
                or not np.equal(block, np.floor(block)).all()):
            raise ValueError("rankings must be integer ranks in [0, original rank universe). "
                             "Projected files need an original_rank_universe JSON sidecar "
                             "or DataFrame attrs['rank_universe_size']; do not renumber ranks")
        fingerprint.update(np.ascontiguousarray(block))
    for axis in (frame.index, frame.columns):
        for name in axis:
            fingerprint.update(str(name).encode() + b"\0")
    frame.attrs["rank_universe_size"] = int(universe)
    frame.attrs["content_sha256"] = fingerprint.hexdigest()
    return frame


def load_annotations(source, *, paired=False):
    """Accept standard aertslab tables; preserve the audited annotation policy."""
    frame = _coerce_motif_annotations(source)
    frame = frame.rename(columns={"#motif_id": "motif", "gene_name": "TF"})
    if not {"motif", "TF"}.issubset(frame.columns):
        raise ValueError("motif annotations need motif/TF or #motif_id/gene_name columns")
    if paired:
        if "description" not in frame:
            raise ValueError("paired annotations need description to select direct/orthologous evidence")
    elif bool("motif_similarity_qvalue" in frame) != bool("orthologous_identity" in frame):
        raise ValueError("annotations need both motif_similarity_qvalue and orthologous_identity, or neither for a prefiltered motif/TF table")
    elif {"motif_similarity_qvalue", "orthologous_identity"}.issubset(frame.columns):
        frame = frame.loc[(frame.motif_similarity_qvalue <= .001) & (frame.orthologous_identity >= 0)]
    return frame.dropna(subset=["motif", "TF"]).drop_duplicates()


def _check_matrix(adata, name):
    if adata.n_obs == 0 or adata.n_vars < 40:
        raise ValueError(f"{name} needs cells and at least 40 features for the fixed 5% AUCell cutoff")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f"{name} cell and feature names must be unique; resolve duplicates before running")
    values = adata.X.data if sparse.issparse(adata.X) else adata.X
    if values is None or getattr(values, "ndim", 0) not in (1, 2):
        raise ValueError(f"{name} needs an in-memory dense or sparse matrix in X")
    # Scan bounded slices, avoiding a full-sized boolean temporary for an atlas.
    step = max(1, (8 * 1024 ** 2) // (values.dtype.itemsize * (values.shape[1] if values.ndim == 2 else 1)))
    digest = hashlib.sha256()
    digest.update(str((adata.shape, str(values.dtype))).encode())
    for start in range(0, len(values), step):
        block = values[start:start + step]
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError(f"{name} must contain finite, non-negative values")
        digest.update(np.ascontiguousarray(block))
    if sparse.issparse(adata.X):
        if adata.X.format not in {"csr", "csc"}:
            raise ValueError(f"{name} sparse matrix must use CSR or CSC storage")
        digest.update(adata.X.format.encode())
        for axis in (adata.X.indices, adata.X.indptr):
            digest.update(np.ascontiguousarray(axis))
    for axis in (adata.obs_names, adata.var_names):
        for value in axis:
            digest.update(str(value).encode() + b"\0")
    return digest.hexdigest()


def _rna(rna, tfs, rankings, annotations, seed, aucell_seed):
    stages = {}
    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        stages[name] = time.perf_counter() - start
        return result
    network = timed("gene_network", lambda: grn.infer(rna, tf_names=tfs, seed=seed, **GRN_SETTINGS))
    network = timed("gene_correlation", lambda: grn.add_correlation(network, rna, rho_threshold=.03))
    candidates = timed("candidate_modules", lambda: grn.build_regulons(
        network, top_targets_per_tf=50, min_targets=10, include_repressors=True))
    universe = rankings.attrs["rank_universe_size"]
    enriched = timed("motif_enrichment", lambda: cistarget.enrich(
        rankings, candidates, top_frac=.05, auc_threshold=0., nes_threshold=3.,
        rank_universe_size=universe))
    supported = timed("motif_pruning", lambda: cistarget.prune_regulons(
        enriched, candidates, annotations, rankings=rankings, top_frac=.05,
        auc_threshold=0., nes_threshold=3., min_genes=5, rank_universe_size=universe))
    supported = {str(k): sorted(set(supported[k])) for k in sorted(supported)}
    scores = timed("gene_activity", lambda: aucell.score(rna, supported, top_frac=.05, seed=aucell_seed)
                   if supported else pd.DataFrame(index=rna.obs_names))
    return dict(grn=network, candidate_regulons=candidates, regulons=supported,
                gene_activity=scores, motif_evidence=enriched, stages=stages)


def _coverage(tfs, rna, results):
    """One row per requested TF, including absent/unsupported regulators."""
    genes = set(rna.var_names)
    candidates = results.get("candidate_regulons", {})
    is_rna = "candidate_regulons" in results
    candidate_tfs = {tf_from_regulon_name(k) for k in candidates}
    signatures = results["regulons"]
    retained = {}
    for name, targets in signatures.items():
        tf = tf_from_regulon_name(name)
        retained.setdefault(tf, set()).update(targets)
    rows = []
    for tf in tfs:
        status = ("not_in_expression" if tf not in genes else
                  "retained" if tf in retained else
                  "no_candidate_module" if is_rna and tf not in candidate_tfs else
                  "not_retained_after_motif_and_size_filters")
        rows.append(dict(TF=tf, status=status, retained_target_genes=len(retained.get(tf, ()))))
    return pd.DataFrame(rows)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _resource_identity(source):
    if isinstance(source, pd.DataFrame):
        # Resource tables only, never a cell-by-feature matrix.
        values = pd.util.hash_pandas_object(source, index=True).values
        return dict(source="dataframe", shape=list(source.shape),
                    table_hash=hashlib.sha256(values).hexdigest())
    path = Path(source)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(source=str(path.resolve()), sha256=digest.hexdigest())


def run(rna, output_dir, *, tfs, motif_annotations, motif_rankings=None,
        atac=None, region_motif_rankings=None, region_mapping=None, gene_coords=None,
        seed=777, aucell_seed=SCORING_SEED, threads=16, activity_mode=None):
    """Run the fixed audited RNA or paired recipe and return its manifest.

    RNA must be prepared non-negative expression (normally log-normalised),
    with unique gene-symbol and cell axes. ATAC must contain paired peak counts.
    Rankings are motifs-by-features with authentic integer ranks. Projected
    ranking files need a JSON sidecar declaring ``original_rank_universe``;
    DataFrames use ``attrs['rank_universe_size']``. Paired runs require a
    ``Target`` (input peak) / ``Query`` (database region) overlap mapping and
    gene/chrom/tss coordinates from the same genome build.

    The output directory must be new or empty, preventing stale results from
    being mistaken for current evidence. Empty supported networks are valid
    negative results, explicitly reported, and never replaced by candidates.
    ``threads`` controls topic fitting; set RAYON_NUM_THREADS before Python
    starts to control the native GRN pool (the CLI sets this for you).
    ``activity_mode='directional'`` keeps positively and negatively associated
    target sets separate. Defaults preserve signed RNA and paired TF unions.
    """
    started = time.perf_counter()
    paired = atac is not None
    if activity_mode is None:
        activity_mode = "union" if paired else "directional"
    if activity_mode not in {"union", "directional"} or (not paired and activity_mode != "directional"):
        raise ValueError("activity_mode must be directional for RNA, or union/directional for paired data")
    required = (region_motif_rankings, region_mapping, gene_coords) if paired else (motif_rankings,)
    if any(x is None for x in required):
        raise ValueError("paired runs require region rankings, region mapping and gene coordinates" if paired
                         else "RNA runs require gene motif rankings and motif annotations")
    if motif_annotations is None:
        raise ValueError("motif annotations are required; unfiltered fallback is disabled")
    if (paired and motif_rankings is not None) or (not paired and any(
            x is not None for x in (region_motif_rankings, region_mapping, gene_coords))):
        raise ValueError("provide gene rankings for RNA, or region rankings/mapping/coordinates with ATAC")
    for name, value in (("seed", seed), ("aucell_seed", aucell_seed), ("threads", threads)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (1 if name == "threads" else 0):
            raise ValueError(f"{name} must be a {'positive' if name == 'threads' else 'non-negative'} integer")
    if aucell_seed > 2**32 - 1:
        raise ValueError("aucell_seed must fit a NumPy RandomState seed (0..2**32-1)")
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output directory must be new or empty; existing results are preserved")
    rna = _coerce_adata(rna)
    input_hashes = {"rna": _check_matrix(rna, "RNA")}
    factors = list(dict.fromkeys(_load_tfs(tfs)))
    if not set(factors).intersection(rna.var_names):
        raise ValueError("no TFs match RNA gene names; use the same gene-symbol convention")
    annotations = load_annotations(motif_annotations, paired=paired)
    if not set(annotations.TF).intersection(factors):
        raise ValueError("no motif annotation TFs match the requested TF names")
    if paired:
        from . import _audited_multiome
        atac = _coerce_adata(atac)
        input_hashes["atac"] = _check_matrix(atac, "ATAC")
        mapping = region_mapping if isinstance(region_mapping, pd.DataFrame) else pd.read_csv(region_mapping, sep="\t")
        if not {"Target", "Query"}.issubset(mapping):
            raise ValueError("region mapping needs Target and Query columns")
        rankings = load_rankings(region_motif_rankings, mapping.Query.unique().tolist())
        coords = _coerce_gene_coords(gene_coords)
        # The helper checks paired cells before fitting or creating its scratch data.
        results = _audited_multiome.run(rna, atac, factors, gene_coords=coords,
            region_rankings=rankings, region_mapping=mapping, motif_annotations=annotations,
            output_dir=out, seed=seed, aucell_seed=aucell_seed, threads=threads,
            activity_mode=activity_mode)
    else:
        rankings = load_rankings(motif_rankings, rna.var_names.tolist())
        out.mkdir(parents=True, exist_ok=True)
        results = _rna(rna, factors, rankings, annotations, seed, aucell_seed)
    out.mkdir(parents=True, exist_ok=True)
    results["regulator_coverage"] = _coverage(factors, rna, results)
    artifacts = {}
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            path = out / f"{name}.parquet"
            value.to_parquet(path)
            artifacts[name] = path.name
        elif name.endswith("regulons"):
            path = out / f"{name}.json"
            path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")
            artifacts[name] = path.name
    manifest = dict(recipe="audited-paired-v1" if paired else "audited-rna-v1",
        status="complete" if results["regulons"] else "no_supported_regulons",
        scoring="seeded native AUCell; unweighted; top 5%; no candidate fallback",
        activity_mode=activity_mode,
        scored_signatures=results["metadata"]["scored_signatures"] if paired else "motif-pruned signed RNA regulons",
        seed=int(seed), aucell_seed=int(aucell_seed), topic_threads=int(threads) if paired else None,
        rayon_num_threads=os.environ.get("RAYON_NUM_THREADS", "automatic"),
        grn_settings=GRN_SETTINGS, motif_nes_threshold=3., min_target_genes=5,
        rank_universe_size=rankings.attrs["rank_universe_size"],
        loaded_ranking_features=rankings.shape[1], n_motifs=rankings.shape[0],
        matrix_inputs={"rna": _matrix_profile(rna), **({"atac": _matrix_profile(atac)} if paired else {})},
        n_requested_tfs=len(factors), requested_tfs=factors, input_fingerprints=input_hashes,
        input_fingerprints_scope="matrix values, storage and ordered cell/feature axes",
        resources=dict(rankings_source="dataframe" if isinstance(region_motif_rankings if paired else motif_rankings, pd.DataFrame)
                       else str(Path(region_motif_rankings if paired else motif_rankings).resolve()),
                       loaded_ranking_sha256=rankings.attrs["content_sha256"],
                       annotations=_resource_identity(motif_annotations),
                       **({"region_mapping": _resource_identity(region_mapping),
                           "gene_coords": _resource_identity(gene_coords)} if paired else {})),
        n_regulons=len(results["regulons"]),
        gene_activity_shape=list(results["gene_activity"].shape),
        scoring_coverage=results["gene_activity"].attrs.get("regulon_coverage", {}),
        stages=results.get("stages", results.get("elapsed", {})),
        grn_fit=results.get("grn_fit", results["grn"].attrs.get("grn_fit", {})),
        elapsed_seconds=time.perf_counter() - started,
        details=results.get("metadata", {}), artifacts=artifacts,
        interpretation="Motif-supported associations and per-cell activity, not proof of causal regulation. Missing TFs are not evidence of inactivity.")
    from . import __version__
    manifest["rustscenic_version"] = __version__
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    if not results["regulons"]:
        warnings.warn("No regulons passed the fixed motif and size filters. Empty activity output written; see regulator_coverage.parquet.", UserWarning, stacklevel=2)
    return manifest
