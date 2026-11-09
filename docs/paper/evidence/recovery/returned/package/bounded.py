"""Memory-bounded orchestration of unchanged RustScenic 0.5.0 kernels.

Peak batches are independent Pearson calculations. AUCell batches contain
whole cells and all features, preserving deterministic within-cell rankings.
No thinning, resampling, feature removal or threshold change is performed.
"""
import gc
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad


def link_peaks(rna, atac, coords, *, max_distance, min_abs_corr, peak_batch=256):
    from rustscenic.enhancer import link_peaks_to_genes
    if peak_batch < 1:
        raise ValueError('peak_batch must be positive')
    # Convert RNA once. The public linker then reuses this CSC representation.
    x = rna.X.tocsc().astype(np.float32, copy=False) if sparse.issparse(rna.X) else rna.X
    prepared_rna = ad.AnnData(x, obs=rna.obs.copy(), var=rna.var.copy())
    pieces = []
    for start in range(0, atac.n_vars, peak_batch):
        # Copy only this peak window, never the full imputed dense atlas.
        block = atac[:, start:start + peak_batch].copy()
        result = link_peaks_to_genes(prepared_rna, block, coords,
                                    max_distance=max_distance, min_abs_corr=min_abs_corr)
        pieces.append(result)
        del block
    del prepared_rna, x
    gc.collect()
    if not pieces:
        raise ValueError('No ATAC regions')
    result = pd.concat(pieces, ignore_index=True)
    # Match the public native API's global ordering, including exact ties:
    # descending |rho|, original peak index, then TSS-sorted gene index.
    # For a given peak all candidate genes share its chromosome.
    result['_abs_rho'] = result.correlation.abs()
    result['_peak_order'] = result.peak_id.map({v:i for i,v in enumerate(atac.var_names)})
    result['_coord_order'] = result.gene.map({v:i for i,v in enumerate(coords.gene)})
    result = result.sort_values(['_abs_rho','_peak_order','gene_tss','_coord_order'],
                                ascending=[False,True,True,True],kind='mergesort')
    result = result.drop(columns=['_abs_rho','_peak_order','_coord_order']).reset_index(drop=True)
    result.attrs['memory_adapter'] = {'engine': 'unchanged RustScenic 0.5.0',
                                    'peak_batch': peak_batch, 'all_cells_used': True}
    return result


def aucell(adata, regulons, *, top_frac=.05, cell_batch=2048):
    from rustscenic.aucell import score
    if cell_batch < 1:
        raise ValueError('cell_batch must be positive')
    pieces = []
    for start in range(0, adata.n_obs, cell_batch):
        block = adata[start:start + cell_batch, :].copy()
        pieces.append(score(block, regulons, top_frac=top_frac))
        del block
    result = pd.concat(pieces)
    if result.index.tolist() != adata.obs_names.tolist():
        raise ValueError('Batched AUCell changed cell order')
    return result
