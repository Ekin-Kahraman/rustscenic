"""Fit one arm for one dataset and seed. One arm per process, so nothing leaks between arms.

Arms (the four fitted arms share SETTINGS; only the stopping definition differs)
  native_cross        RustScenic, early_stop_mode="arboreto"    current rule
  native_same         RustScenic, early_stop_mode="paired_oob"  original rule
  reference_cross     arboreto 0.1.6 on the installed scikit-learn 1.3.2
  reference_original  arboreto 0.1.6 on the bundled scikit-learn 1.2.2
                      (--emulate: same-mask monitor on 1.3.2, used only if 1.2.2 cannot load)
  pearson             absolute Pearson correlation for every TF-target pair (baseline, no fitting)

The network is written as soon as inference ends. Every later stage is guarded:
a failure there is recorded in SUMMARY.json and never discards the fitted network.
"""
import argparse
import hashlib
import importlib.metadata
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse

from common import ROOT, digest, read_json, write_json

SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3, early_stop_window=25)
NATIVE = {'native_cross': 'arboreto', 'native_same': 'paired_oob'}
REFERENCE = ('reference_cross', 'reference_original')
ARMS = (*NATIVE, *REFERENCE, 'pearson')
PREPARED = ('atlas', 'hdma')  # recovery/work/prepared/<name>/rna.h5 and tfs.txt, verified on 11 November
TOP = 100


def phase(out, name):
    write_json(out / 'phase.json', {'phase': name})


def load(dataset, recovery):
    """Training matrix, genes and TFs. Atlas and HDMA subsets are the first N cells of the seeded cell order."""
    if dataset == 'bmmc':
        folder = ROOT / 'inputs/bmmc'
        panel = read_json(folder / 'panel.json')
        x, genes, tfs = sparse.load_npz(folder / 'training.npz').tocsr(), panel['genes'], panel['tfs']
        cells = pd.read_csv(folder / 'training-cells.tsv', sep='\t', dtype=str).barcode.tolist()
    else:
        from data_io import load_matrix
        kind = next(k for k in PREPARED if dataset.startswith(k))
        folder = recovery / 'work/prepared' / kind
        x, genes, cells = load_matrix(folder / 'rna.h5', int(dataset.removeprefix(kind)))
        tfs = (folder / 'tfs.txt').read_text().split()
    x = x.tocsc()
    mean = np.asarray(x.mean(axis=0)).ravel()
    varying = np.asarray(x.power(2).mean(axis=0)).ravel() - mean ** 2 > 0
    genes = [g for g, keep in zip(genes, varying) if keep]
    x = x[:, np.flatnonzero(varying)].tocsr()
    kept = set(genes)
    info = {'cells': x.shape[0], 'genes': len(genes), 'constant_genes_removed': int((~varying).sum()),
            'cells_sha256': hashlib.sha256('\n'.join(cells).encode()).hexdigest(),
            'genes_sha256': hashlib.sha256('\n'.join(genes).encode()).hexdigest()}
    return x, genes, [t for t in tfs if t in kept], info


def top_table(network):
    """Top targets per TF, ordered exactly as the stability metric orders them."""
    edges = network.loc[network.importance.gt(0) & network.TF.ne(network.target)]
    return (edges.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True], kind='stable')
            .groupby('TF', sort=False).head(TOP).reset_index(drop=True))


def save_network(network, out, keep_rho):
    """Complete network and its top-100 view. Importance is stored, ranked and scored in single precision.

    Correlation depends only on the gene pair, so the complete table of it is kept once, in the Pearson arm.
    """
    columns = [c for c in ('TF', 'target', 'importance', 'rho') if c in network]
    (network[[c for c in columns if keep_rho or c != 'rho']].astype({'TF': 'category', 'target': 'category'})
     .to_parquet(out / 'network.parquet', index=False, compression='zstd'))
    top_table(network)[columns].to_parquet(out / 'top100.parquet', index=False)


def infer_native(x, genes, tfs, seed, arm):
    from rustscenic import grn
    network = grn.infer((x, genes), tf_names=tfs, seed=seed, verbose=False, early_stop_mode=NATIVE[arm], **SETTINGS)
    trees = network.groupby('target').importance.sum().round().astype(int)
    return network[['TF', 'target', 'importance']], trees, {'grn_fit': network.attrs.get('grn_fit', {})}


def infer_reference(x, genes, tfs, seed, arm, threads, emulate):
    import sklearn
    if importlib.metadata.version('arboreto') != '0.1.6':
        raise ValueError('arboreto 0.1.6 required')
    wanted = '1.3.2' if arm == 'reference_cross' or emulate else '1.2.2'
    if sklearn.__version__ != wanted:
        raise ValueError(f'{arm} needs scikit-learn {wanted}, found {sklearn.__version__} at {sklearn.__file__}')
    if emulate:
        import reference_paired
        reference_paired.install()
    from grn_reference import infer_multiprocessing
    network, meta, fit_info, _, _ = infer_multiprocessing(x, genes, tfs, SETTINGS, seed, threads)
    trees = meta.set_index('target').n_estimators.astype(int)
    return network[['TF', 'target', 'importance']], trees, {
        'scikit_learn': sklearn.__version__, 'scikit_learn_path': sklearn.__file__, 'same_mask_monitor_emulated': bool(emulate),
        'arboreto_fit': fit_info}


def all_pairs(genes, tfs):
    tf = np.repeat(np.array(tfs, dtype=object), len(genes))
    target = np.tile(np.array(genes, dtype=object), len(tfs))
    return pd.DataFrame({'TF': tf, 'target': target, 'importance': 1.}).loc[tf != target].reset_index(drop=True)


def refine(arm, network, x, genes, out, threads):
    """Motif-supported regulons with the unchanged 12 November recipes."""
    folder = ROOT / 'inputs/bmmc'
    annotations = ROOT / 'resources/human-gene-annotations.tbl'
    regulons, info = {}, {}
    if arm in NATIVE:
        from rankings import load_projection, native_annotations
        from recipe import refine as native_refine
        rankings, universe = load_projection(folder / 'motifs.feather')
        table = native_annotations(annotations)
        for variant in ('original', 'diverse_importance'):
            regulons[variant], _, info[variant], _ = native_refine(network, rankings, table, universe, variant)
    else:
        from rankings import ProjectedRankingDatabase
        from reference import refine_reference
        expression = pd.DataFrame(x.toarray(), columns=genes)
        regulons['reference'], info['reference'] = refine_reference(
            network, expression, ProjectedRankingDatabase(folder / 'motifs.feather', 'human-v10'), annotations, threads)
    for name, members in regulons.items():
        write_json(out / f'regulons-{name}.json', members)
    return regulons, info


def fit(dataset, seed, arm, out, recovery, threads, emulate=False):
    from rustscenic import __version__, grn
    from monitoring import environment
    if __version__ != '0.6.1':
        raise ValueError('The bundled 0.6.1 build is required')
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    write_json(out / 'environment.json', environment())
    phase(out, 'input')
    x, genes, tfs, info = load(dataset, recovery)
    write_json(out / 'panel.json', {'genes': genes, 'tfs': tfs})
    errors, detail, trees, inference_s = {}, {}, None, 0.

    def guarded(stage, function):
        phase(out, stage)
        try:
            return function()
        except Exception:
            errors[stage] = traceback.format_exc()[-4000:]
            print(errors[stage], flush=True)

    if arm == 'pearson':
        network = all_pairs(genes, tfs)
    else:
        phase(out, 'grn_inference')
        t = perf_counter()
        if arm in NATIVE:
            network, trees, detail = infer_native(x, genes, tfs, seed, arm)
        else:
            network, trees, detail = infer_reference(x, genes, tfs, seed, arm, threads, emulate)
        inference_s = perf_counter() - t
        network = network.assign(importance=network.importance.astype(np.float32))
        save_network(network, out, False)
        trees = trees.reindex(genes).fillna(0).astype(int)
        trees.rename_axis('target').rename('trees').to_csv(out / 'trees.csv')

    t = perf_counter()
    correlated = guarded('grn_correlation', lambda: grn.add_correlation(network, (x, genes), rho_threshold=.03))
    correlation_s = perf_counter() - t
    if correlated is not None:
        network = correlated.assign(rho=correlated.rho.astype(np.float32))
        if arm == 'pearson':
            network = network.assign(importance=network.rho.abs().fillna(0.).astype(np.float32))
        save_network(network, out, arm == 'pearson')
    def recovery_table():
        from recovery import score_network
        score_network(network, genes, tfs, dataset).to_csv(out / 'recovery.csv', index=False)

    def lineage_table(regulons):
        from lineage import evaluate
        evaluate(ROOT / 'inputs/bmmc', regulons).to_csv(out / 'lineage.csv', index=False)

    if 'rho' in network or arm != 'pearson':
        guarded('reference_recovery', recovery_table)
    refinement = {}
    if dataset == 'bmmc' and arm != 'pearson' and 'rho' in network:
        made = guarded('refinement', lambda: refine(arm, network, x, genes, out, threads))
        if made is not None:
            refinement = made[1]
            guarded('lineage', lambda: lineage_table(made[0]))
    files = sorted(p.name for p in out.iterdir() if p.suffix in ('.parquet', '.csv') or p.name.startswith('regulons-'))
    numbers = {} if trees is None else {
        'trees_mean': float(trees.mean()), 'trees_median': float(trees.median()), 'trees_p90': float(trees.quantile(.9)),
        'trees_max': int(trees.max()), 'share_at_ceiling': float((trees >= SETTINGS['n_estimators']).mean())}
    write_json(out / 'SUMMARY.json', {
        'status': 'complete', 'stage_errors': errors, 'dataset': dataset, 'seed': seed, 'arm': arm, **info, 'TFs': len(tfs),
        'settings': SETTINGS, 'stopping_rule': NATIVE.get(arm, 'scikit-learn ' + detail['scikit_learn'] if arm in REFERENCE else 'none'),
        **detail, 'inference_s': inference_s, 'correlation_s': correlation_s, 'wall_s': perf_counter() - start, 'threads': threads,
        'edges': len(network), **numbers, 'refinement': refinement, 'protocol_sha256': digest(ROOT / 'PROTOCOL.md'),
        'outputs': {name: digest(out / name) for name in files}})
    phase(out, 'complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('seed', type=int)
    p.add_argument('arm', choices=ARMS)
    p.add_argument('out', type=Path)
    p.add_argument('--recovery', type=Path)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--emulate', action='store_true')
    a = p.parse_args()
    fit(a.dataset, a.seed, a.arm, a.out, a.recovery, a.threads, a.emulate)
