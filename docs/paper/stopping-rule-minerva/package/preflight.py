"""Seconds-long load test of the bundled builds on a small synthetic matrix.

native     the patched RustScenic wheel imports, accepts both stopping rules and
           the original rule trains longer on predictable targets than the
           current rule, while unpredictable targets still stop early.
reference  arboreto on whichever scikit-learn the interpreter resolves: reports
           the version and the trees fitted for one predictable target.
Ten targets are sums of two regulators plus noise; ten are pure noise. Inputs
are small integers scaled by a power of two, so they are identical everywhere.
"""
import json
import sys
import types

import numpy as np

SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3, early_stop_window=25)
TFS, TARGETS, CELLS = 20, 20, 600


def toy():
    rng = np.random.default_rng(20261113)
    x = rng.integers(0, 40, size=(CELLS, TFS + TARGETS)).astype(np.float32)
    for j in range(TFS, TFS + TARGETS // 2):
        a, b = j - TFS, (j - TFS + 1) % TFS
        x[:, j] = 2 * x[:, a] + x[:, b] + rng.integers(0, 20, size=CELLS)
    genes = [f'g{i:02d}' for i in range(x.shape[1])]
    return x / 16, genes, genes[:TFS], genes[TFS:TFS + TARGETS // 2], genes[TFS + TARGETS // 2:]


def native(stub=None):
    if stub:  # bare interpreter: load the extension and grn.py without the package's optional imports
        package = types.ModuleType('rustscenic')
        package.__path__ = [stub + '/rustscenic']
        sparse = types.ModuleType('scipy.sparse')
        sparse.issparse = lambda value: False
        sys.modules.update({'rustscenic': package, 'scipy': types.ModuleType('scipy'), 'scipy.sparse': sparse})
    from rustscenic import grn
    from rustscenic._rustscenic import __version__
    x, genes, tfs, signal, noise = toy()
    result = {'rustscenic': __version__, 'extension': grn._grn_infer.__module__}
    for mode in ('arboreto', 'paired_oob'):
        net = grn.infer((x, genes), tf_names=tfs, seed=777, verbose=False, early_stop_mode=mode, **SETTINGS)
        trees = net.groupby('target').importance.sum()
        top = net.sort_values(['importance', 'TF', 'target'], ascending=[False, True, True]).head(5)
        result[mode] = {'signal_trees': float(trees.reindex(signal).mean()), 'noise_trees': float(trees.reindex(noise).mean()),
                        'edges': len(net), 'importance_total': float(net.importance.sum()),
                        'top': [[a, b, round(float(c), 3)] for a, b, c in top.itertuples(index=False, name=None)]}
    cross, same = result['arboreto'], result['paired_oob']
    result['ok'] = bool(__version__ == '0.6.1' and same['signal_trees'] >= 3 * cross['signal_trees']
                        and cross['noise_trees'] <= 60 and same['noise_trees'] <= 60)
    return result


def reference():
    import sklearn
    from arboreto.core import infer_partial_network
    x, genes, tfs, signal, noise = toy()
    index = {g: i for i, g in enumerate(genes)}
    kwargs = {k: v for k, v in SETTINGS.items() if k != 'early_stop_window'}
    trees = {}
    for label, target in (('signal', signal[0]), ('noise', noise[0])):
        _, meta = infer_partial_network('GBM', kwargs, x[:, :TFS].astype(np.float64), tfs, target, x[:, index[target]].astype(np.float64),
                                        include_meta=True, early_stop_window_length=SETTINGS['early_stop_window'], seed=777)
        trees[label] = int(meta.n_estimators.iloc[0])
    return {'scikit_learn': sklearn.__version__, 'sklearn_path': sklearn.__file__, 'trees': trees, 'ok': True}


if __name__ == '__main__':
    kind = sys.argv[1]
    outcome = native(*sys.argv[2:3]) if kind == 'native' else reference()
    print(json.dumps(outcome))
    sys.exit(0 if outcome['ok'] else 1)
