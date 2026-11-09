"""Native paired recipe used in the paper, with the verified seeded scorer.

This is deliberately separate from the flexible historical pipeline. Topic
regions, motif support, imputed accessibility and final eRegulon signatures
must stay connected. Cached stages are an internal audit/replay hook only.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import mmap
import shutil
import tempfile
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from rustscenic import aucell, cistarget, enhancer, eregulon, grn, topics

GRN_SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1,
                    subsample=.9, max_depth=3, early_stop_window=25,
                    early_stop_mode="arboreto")
_SUPPORT_COLUMNS = ["TF", "peak_id", "motif", "auc", "nes"]
_SCORE_COLUMNS = ["region_set", "motif", "auc", "nes"]


def _ann(matrix, cells, features):
    return ad.AnnData(matrix, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=features))


def _check_axes(rna, atac):
    if not rna.obs_names.is_unique or not atac.obs_names.is_unique:
        raise ValueError("RNA and ATAC cell names must be unique")
    if not rna.var_names.is_unique or not atac.var_names.is_unique:
        raise ValueError("RNA gene and ATAC region names must be unique")
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("Paired RNA and ATAC must contain the same cells in the same order")
    if min(rna.shape + atac.shape) < 1:
        raise ValueError("Paired RNA and ATAC matrices must not be empty")


def _topics(atac, seed, threads, cached):
    binary = sparse.csr_matrix(atac.X, dtype=np.float32, copy=True)
    binary.sum_duplicates()
    binary.eliminate_zeros()
    if (not np.isfinite(binary.data).all() or np.any(binary.data < 0)
            or np.any(binary.data != np.rint(binary.data))):
        raise ValueError("ATAC must contain finite, non-negative integer counts")
    binary.data[:] = 1
    active = np.asarray(binary.getnnz(axis=0)).ravel() > 0
    feature_names = atac.var_names[active]
    if not len(feature_names):
        raise ValueError("ATAC contains no accessible regions")
    if ("cell_topic" in cached) != ("topic_peak" in cached):
        raise ValueError("Replay requires both cell_topic and topic_peak")
    if "cell_topic" in cached:
        cell_topic, topic_peak = cached["cell_topic"], cached["topic_peak"]
        if not cell_topic.index.equals(atac.obs_names):
            raise ValueError("Cached topics have different cells or cell order")
        if not topic_peak.columns.equals(feature_names):
            raise ValueError("Cached topics have different active regions or region order")
        if not cell_topic.columns.equals(topic_peak.index) or cell_topic.shape[1] != 30:
            raise ValueError("Cached topic axes must match the 30-topic recipe")
    else:
        fitted = topics.fit_gibbs(_ann(binary[:, active], atac.obs_names, feature_names),
            n_topics=30, alpha=50. / 30, eta=.1, n_iters=500,
            seed=seed, n_threads=threads, verbose=False)
        cell_topic, topic_peak = fitted.cell_topic, fitted.topic_peak
        names = [f"Topic{i + 1}" for i in range(30)]
        cell_topic.index = atac.obs_names
        cell_topic.columns = names
        topic_peak.index = names
    for frame in (cell_topic, topic_peak):
        values = frame.to_numpy(copy=False)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Topic probabilities must be finite and non-negative")
    region_sets = topics.TopicsResult(cell_topic, topic_peak, 30).top_peaks_per_topic(1000)
    return cell_topic, topic_peak, {k: sorted(v) for k, v in region_sets.items()}


def _region_support(region_sets, tfs, ranks, mapping, annotation):
    if not {"Target", "Query"}.issubset(mapping.columns):
        raise ValueError("Region mapping requires Target (input peak) and Query (database region)")
    motif_col = "#motif_id" if "#motif_id" in annotation else "motif"
    tf_col = "gene_name" if "gene_name" in annotation else "TF"
    if not {motif_col, tf_col, "description"}.issubset(annotation.columns):
        raise ValueError("Paired motif annotations require motif, TF and description columns")
    if mapping[["Target", "Query"]].isna().any().any():
        raise ValueError("Region mapping contains missing names")
    queries_by_peak, peaks_by_query = {}, {}
    rank_columns = set(ranks.columns)
    for peak, query in zip(mapping.Target, mapping.Query):
        if query not in rank_columns:
            raise ValueError(f"Mapped database region is absent from rankings: {query}")
        queries_by_peak.setdefault(peak, set()).add(query)
        peaks_by_query.setdefault(query, set()).add(peak)
    annotation_map = {}
    allowed_tfs = set(tfs)
    for motif, tf, description in zip(annotation[motif_col], annotation[tf_col], annotation.description):
        direct = description == "gene is directly annotated"
        orthology = isinstance(description, str) and "orthologous" in description and "similar" not in description
        if tf in allowed_tfs and (direct or orthology):
            annotation_map.setdefault(motif, set()).add(tf)
    query_sets = {name: sorted({q for p in peaks for q in queries_by_peak.get(p, ())})
                  for name, peaks in region_sets.items()}
    query_sets = {k: v for k, v in query_sets.items() if v}
    if not query_sets:
        raise ValueError("No fitted topic regions overlap the motif database mapping")
    universe = int(ranks.attrs.get("rank_universe_size", ranks.shape[1]))
    enriched = cistarget.enrich(ranks, query_sets, top_frac=.005, auc_threshold=0.,
                               nes_threshold=3., rank_universe_size=universe)
    scores = enriched.rename(columns={"regulon": "region_set"})[_SCORE_COLUMNS]
    motif_ix = {v: i for i, v in enumerate(ranks.index)}
    query_ix = {v: i for i, v in enumerate(ranks.columns)}
    values = ranks.to_numpy(copy=False)
    rows = set()
    for region_set, motif, auc, nes in zip(enriched.regulon, enriched.motif, enriched.auc, enriched.nes):
        factors = annotation_map.get(motif, ())
        if not factors:
            continue
        queries = query_sets[region_set]
        indices = [query_ix[q] for q in queries]
        supported = [q for q, rank in zip(queries, values[motif_ix[motif], indices])
                     if rank < round(.005 * universe)]
        topic_peaks = set(region_sets[region_set])
        peaks = {p for q in supported for p in peaks_by_query[q] if p in topic_peaks}
        rows.update((tf, peak, motif, float(auc), float(nes)) for tf in factors for peak in peaks)
    support = pd.DataFrame(sorted(rows), columns=_SUPPORT_COLUMNS)
    support = support.astype({"auc": np.float64, "nes": np.float64})
    return support, scores


def _release_pages(values):
    # Flush before dropping pages: later stages must read the same int32 matrix.
    values.flush()
    if hasattr(values._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        values._mmap.madvise(mmap.MADV_DONTNEED)


@contextmanager
def _imputed(cell_topic, topic_peak, output_dir):
    """Posterior product with bounded scratch space; never a dense atlas copy."""
    shape = (len(cell_topic), topic_peak.shape[1])
    required = int(shape[0]) * int(shape[1]) * 4
    if shutil.disk_usage(output_dir).free < required + 64 * 1024 ** 2:
        raise ValueError(f"Imputation needs {required / 1024 ** 3:.2f} GiB of free scratch disk in {output_dir}")
    with tempfile.TemporaryDirectory(prefix=".rustscenic-imputed-", dir=output_dir) as temp:
        values = np.memmap(Path(temp) / "accessibility.i32", mode="w+", dtype=np.int32,
                           shape=shape, order="F")
        left = cell_topic.to_numpy(dtype=np.float32)
        right = topic_peak.to_numpy(dtype=np.float32)
        keep = np.zeros(shape[1], dtype=bool)
        try:
            for start in range(0, shape[1], 2000):
                end = min(start + 2000, shape[1])
                for row in range(0, shape[0], 8192):
                    stop = min(row + 8192, shape[0])
                    block = (left[row:stop] @ right[:, start:end] * np.float32(1000000)).astype(np.int32)
                    values[row:stop, start:end] = block
                    keep[start:end] |= np.any(block != 0, axis=0)
                _release_pages(values)
            yield values, np.flatnonzero(keep)
        finally:
            values._mmap.close()


def _link_imputed(rna, values, kept, peak_names, coords):
    # Same independent Pearson kernels as the audited HPC memory adapter.
    matrix = rna.X.tocsc().astype(np.float32, copy=False) if sparse.issparse(rna.X) else rna.X
    prepared = ad.AnnData(matrix, obs=rna.obs.copy(), var=rna.var.copy())
    pieces = []
    for start in range(0, len(kept), 256):
        columns = kept[start:start + 256]
        block = _ann(np.asarray(values[:, columns]), rna.obs_names, peak_names[columns])
        pieces.append(enhancer.link_peaks_to_genes(prepared, block, coords,
                                                 max_distance=150000, min_abs_corr=.03))
        del block
        _release_pages(values)
    if not pieces:
        return pd.DataFrame(columns=enhancer._LINK_COLUMNS)
    return pd.concat(pieces, ignore_index=True)


def _score_batched(expression, regulons, seed):
    if not regulons:
        return pd.DataFrame(index=expression.obs_names)
    pieces = [aucell.score(expression[start:start + 2048, :].copy(), regulons,
                           top_frac=.05, seed=seed)
              for start in range(0, expression.n_obs, 2048)]
    result = pd.concat(pieces)
    result.attrs = pieces[0].attrs.copy()
    return result


def _score_imputed(values, kept, cell_names, peak_names, regulons, seed):
    if not regulons:
        return pd.DataFrame(index=cell_names)
    pieces = []
    for start in range(0, len(cell_names), 2048):
        stop = min(start + 2048, len(cell_names))
        block = _ann(np.asarray(values[start:stop, :])[:, kept], cell_names[start:stop], peak_names[kept])
        with warnings.catch_warnings():
            # Imputed accessibility is deliberately scaled; AUCell uses ranks.
            warnings.filterwarnings("ignore", message="expression matrix max value is .*", category=UserWarning)
            pieces.append(aucell.score(block, regulons, top_frac=.05, seed=seed))
        del block
        _release_pages(values)
    result = pd.concat(pieces)
    result.attrs = pieces[0].attrs.copy()
    return result


def run(rna, atac, tfs, *, gene_coords, region_rankings, region_mapping,
        motif_annotations, output_dir, seed=777, aucell_seed=20261106,
        threads=1, activity_mode="union", cached_stages=None):
    """Run the audited paired recipe and return tables for the public writer.

    Inputs must already be harmonised, with RNA expression in ``X`` and raw
    ATAC counts in ``X``. ``cached_stages`` is only for internal parity audits;
    callers must verify cache provenance separately. It is never an automatic
    resume mechanism or part of the researcher-facing API.
    """
    _check_axes(rna, atac)
    if activity_mode not in {"union", "directional"}:
        raise ValueError("activity_mode must be union or directional")
    if threads < 1:
        raise ValueError("threads must be positive")
    cached = {} if cached_stages is None else dict(cached_stages)
    if set(cached) - {"cell_topic", "topic_peak", "grn"}:
        raise ValueError("Only topic matrices and GRN can be replayed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    elapsed = {}

    def stage(name, fn):
        start = time.perf_counter()
        result = fn()
        elapsed[name] = time.perf_counter() - start
        return result

    cell_topic, topic_peak, region_sets = stage("topic_modelling", lambda: _topics(atac, seed, threads, cached))
    support, motif_scores = stage("region_motif_enrichment", lambda: _region_support(
        region_sets, tfs, region_rankings, region_mapping, motif_annotations))
    network = stage("gene_network", lambda: cached["grn"].copy() if "grn" in cached else
                    grn.infer(rna, tf_names=tfs, **GRN_SETTINGS, seed=seed))
    network = stage("gene_correlation", lambda: grn.add_correlation(network, rna, rho_threshold=.03))
    imputation_start = time.perf_counter()
    with _imputed(cell_topic, topic_peak, output_dir) as (values, kept):
        elapsed["accessibility_imputation"] = time.perf_counter() - imputation_start
        peak_names = topic_peak.columns
        support = support[support.peak_id.isin(peak_names[kept])]
        links = stage("region_to_gene", lambda: _link_imputed(rna, values, kept, peak_names, gene_coords))
        motif_rows = support.rename(columns={"TF": "regulon"})[["regulon", "peak_id", "motif", "auc"]]
        signed = []
        for suffix in ("_activator", "_repressor"):
            frame = motif_rows.copy()
            frame["regulon"] = frame.regulon + suffix
            signed.append(frame)
        assembled = stage("regulon_assembly", lambda: eregulon.build_eregulons_dataframe(network,
            pd.concat(signed, ignore_index=True), links, min_target_genes=5,
            min_enhancer_links=1, cistarget_auc_threshold=0.))
        # Preserve the historical union option. Directional activity retains
        # both expression-associated programmes instead of merging them.
        union = assembled.copy(deep=False)
        union["tf"] = assembled.tf.astype(str).str.replace(r"_(activator|repressor)$", "", regex=True)
        signatures = assembled if activity_mode == "directional" else union
        genes = {name.removesuffix("_eregulon"): features
                 for name, features in eregulon.regulons_from_dataframe(signatures)}
        regions = {name.removesuffix("_eregulon"): features
                   for name, features in eregulon.regulons_from_dataframe(signatures, feature_col="enhancer")}
        genes = {key: sorted(genes[key]) for key in sorted(genes)}
        regions = {key: sorted(regions[key]) for key in sorted(regions)}
        gene_activity = stage("gene_activity", lambda: _score_batched(rna, genes, aucell_seed))
        region_activity = stage("region_activity", lambda: _score_imputed(
            values, kept, rna.obs_names, peak_names, regions, aucell_seed))
    edges = union.rename(columns={"enhancer": "region", "target_gene": "target"})[["tf", "region", "target"]].copy()
    edges["context"] = assembled.tf.astype(str) + "; positive region-gene links"
    return dict(grn=network, regulons=genes, region_regulons=regions,
        gene_activity=gene_activity, region_activity=region_activity, eregulon_edges=edges,
        region_to_gene=links.rename(columns={"peak_id": "region", "gene": "target", "correlation": "rho"}),
        cell_topic=cell_topic, topic_peak=topic_peak, motif_evidence=support,
        elapsed=elapsed, grn_fit=network.attrs.get("grn_fit", {}),
        region_motif_scores=motif_scores, n_eregulons=assembled.attrs.get("n_eregulons", 0),
        metadata={"recipe": "paired-native-paper-v1-seeded-aucell", "elapsed_seconds": elapsed,
                  "replayed_stages": sorted(cached), "aucell_seed": int(aucell_seed),
                  "activity_mode": activity_mode,
                  "scored_signatures": ("separate directional eRegulon gene and region sets" if activity_mode == "directional"
                                        else "TF unions of motif-supported signed eRegulon genes and regions"),
                  "accessibility": "posterior product * 1000000 cast int32; zero columns removed",
                  "imputation_scratch_bytes": int(len(cell_topic) * topic_peak.shape[1] * 4),
                  "active_imputed_regions": int(len(kept)), "topic_threads": int(threads),
                  "status": "motif_supported_eregulons" if genes else "no_supported_eregulons"})
