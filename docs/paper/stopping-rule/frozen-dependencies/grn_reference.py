"""Published arboreto kernel with an audited sparse fork scheduler."""
import time
import numpy as np
import pandas as pd
_STATE = None

def target_network(index):
    from arboreto.core import infer_partial_network
    matrix, tf_matrix, tf_names, genes, parameters, seed = _STATE
    kwargs = {k: v for k, v in parameters.items() if k != "early_stop_window"}
    links, meta = infer_partial_network(
        "GBM", kwargs, tf_matrix, tf_names, genes[index], matrix[:, index],
        include_meta=True, early_stop_window_length=parameters["early_stop_window"], seed=seed)
    if len(meta) != 1 or str(meta.iloc[0].target) != genes[index]:
        raise RuntimeError(f"arboreto failed or skipped target {genes[index]}")
    return links, meta

def infer_multiprocessing(matrix, genes, tfs, parameters, seed, threads):
    global _STATE
    from arboreto.core import to_tf_matrix
    from multiprocessing import get_context
    started = time.perf_counter()
    csc = matrix.tocsc()
    tf_matrix, tf_names = to_tf_matrix(csc, genes, tfs)
    conversion_s = time.perf_counter() - started
    # No whole-matrix toarray(): CSC is shared read-only through fork.
    _STATE = (csc, tf_matrix, tf_names, genes, parameters, seed)
    started = time.perf_counter()
    try:
        if threads == 1:
            results = list(map(target_network, range(len(genes))))
        else:
            with get_context("fork").Pool(threads) as pool:
                results = list(pool.imap_unordered(target_network, range(len(genes)), chunksize=1))
        links = pd.concat([r[0] for r in results], ignore_index=True)
        meta = pd.concat([r[1] for r in results], ignore_index=True)
    finally:
        _STATE = None
    execution_s = time.perf_counter() - started
    if set(meta.target) != set(genes) or len(meta) != len(genes):
        raise RuntimeError("arboreto target completion mismatch")
    counts = meta.n_estimators.to_numpy()
    fit = {"target_count": len(meta), "fitted_trees_min": int(counts.min()),
           "fitted_trees_max": int(counts.max()), "fitted_trees_mean": float(counts.mean()),
           "fitted_trees_median": float(np.median(counts)), "metadata_source": "arboreto target fit metadata"}
    return links, meta, fit, conversion_s, execution_s
