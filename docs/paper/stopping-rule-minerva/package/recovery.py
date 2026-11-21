"""Recovery of broad reference links on a fixed candidate universe (the manuscript's metric).

For one reference, the universe is every (TF, target) pair with the TF among
the reference TFs that are panel TFs with at least one link to a panel gene, and
the target any panel gene other than the TF. Pairs absent from a network score
zero. Average precision is pooled over the whole universe and is tie-aware.
Symbols are compared in upper case; genes whose upper-case names collide are
left out. Intervals resample reference TFs with replacement.
"""
import numpy as np
import pandas as pd

from common import ROOT

REFERENCES = {'human': ('human-nonspecific-chip', 'human-string'), 'mouse': ('mouse-nonspecific-chip', 'mouse-string')}
COMPARISONS = (('native_same', 'native_cross'), ('reference_original', 'reference_cross'), ('native_same', 'pearson'),
               ('native_cross', 'pearson'), ('native_same', 'reference_original'), ('native_cross', 'reference_cross'))
DRAWS, SEED = 2000, 20261113


def species(dataset):
    return 'mouse' if dataset.startswith('atlas') else 'human'


def load_reference(name):
    return pd.read_csv(ROOT / 'resources/references' / f'{name}.tsv.gz', sep='\t', dtype=str)


class Universe:
    def __init__(self, genes, tfs, reference):
        upper = pd.Series([g.upper() for g in genes])
        self.columns = upper[~upper.duplicated(keep=False)].tolist()
        self.column = {g: j for j, g in enumerate(self.columns)}
        panel_tfs = {t.upper() for t in tfs} & set(self.columns)
        links = reference.loc[reference.TF.isin(panel_tfs) & reference.target.isin(self.column) & reference.TF.ne(reference.target)]
        self.tfs = sorted(links.TF.unique())
        self.row = {t: i for i, t in enumerate(self.tfs)}
        self.labels = np.zeros((len(self.tfs), len(self.columns)), dtype=bool)
        self.labels[links.TF.map(self.row).to_numpy(), links.target.map(self.column).to_numpy()] = True
        self.eligible = np.ones_like(self.labels)
        self.eligible[np.arange(len(self.tfs)), [self.column[t] for t in self.tfs]] = False

    def scores(self, network):
        tf, target = _codes(network.TF, self.row), _codes(network.target, self.column)
        keep = (tf >= 0) & (target >= 0)
        values = np.zeros(self.labels.shape)
        values[tf[keep], target[keep]] = network.importance.to_numpy(float)[keep]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite score')
        return values


def _codes(series, index):
    """Row or column number of each symbol (-1 if outside the universe); categorical columns are mapped once per category."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        lookup = np.array([index.get(str(c).upper(), -1) for c in series.cat.categories] + [-1], dtype=np.int64)
        return lookup[series.cat.codes.to_numpy()]
    return series.astype(str).str.upper().map(index).fillna(-1).astype(np.int64).to_numpy()


def average_precision(labels, scores, weights=None):
    """Tie-aware step integral of precision over recall; identical to scikit-learn's average_precision_score."""
    order = np.argsort(-scores, kind='stable')
    return _ap_sorted(labels[order], scores[order], None if weights is None else weights[order])


def _ap_sorted(labels, scores, weights):
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    if weights is None:
        hits, seen = np.cumsum(labels)[last], last + 1.
    else:
        hits, seen = np.cumsum(weights * labels)[last], np.cumsum(weights)[last]
    precision = np.divide(hits, seen, out=np.zeros(len(last)), where=seen > 0)
    return float(np.sum(np.diff(np.r_[0., hits / hits[-1]]) * precision))


def expected_hits_at_k(labels, scores, k):
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, tied = scores > threshold, scores == threshold
    return float(labels[above].sum() + (k - above.sum()) * labels[tied].mean())


def metrics(universe, scores):
    y, s = universe.labels[universe.eligible], scores[universe.eligible]
    ap, prevalence = average_precision(y, s), float(y.mean())
    hits = uniform = 0.
    for i in range(len(universe.tfs)):
        row_y, row_s = universe.labels[i, universe.eligible[i]], scores[i, universe.eligible[i]]
        k = min(50, len(row_y))
        hits += expected_hits_at_k(row_y, row_s, k)
        uniform += k * float(row_y.mean())
    return {'AP': ap, 'prevalence': prevalence, 'AP_over_prevalence': ap / prevalence, 'top50_fold_over_uniform': hits / uniform,
            'reference_TFs': len(universe.tfs), 'eligible_links': int(y.sum()), 'candidate_pairs': int(len(y))}


def score_network(network, genes, tfs, dataset):
    """Point estimates for every reference of the dataset's species. A reference with no eligible link is skipped."""
    rows = []
    for name in REFERENCES[species(dataset)]:
        universe = Universe(genes, tfs, load_reference(name))
        if len(universe.tfs) >= 2 and universe.labels.sum() >= 20:
            rows.append({'reference': name, **metrics(universe, universe.scores(network))})
    return pd.DataFrame(rows)


def intervals(networks, genes, tfs, dataset, draws=DRAWS, seed=SEED):
    """AP differences between arms (seeds averaged) with TF-bootstrap intervals.

    networks: {(arm, seed): frame, or a function returning one so that only one network is in memory at a time}.
    """
    rows = []
    for name in REFERENCES[species(dataset)]:
        universe = Universe(genes, tfs, load_reference(name))
        if len(universe.tfs) < 2 or universe.labels.sum() < 20:
            continue
        flat_labels = universe.labels[universe.eligible]
        flat_tf = np.broadcast_to(np.arange(len(universe.tfs))[:, None], universe.labels.shape)[universe.eligible]
        prepared = {}
        for key, network in networks.items():
            scores = universe.scores(network() if callable(network) else network)[universe.eligible]
            order = np.argsort(-scores, kind='stable')
            prepared[key] = (flat_labels[order].astype(float), scores[order], flat_tf[order])
        point = {key: _ap_sorted(y, s, None) for key, (y, s, _) in prepared.items()}
        rng = np.random.default_rng(seed)
        table = np.empty((draws, len(prepared)))
        for d in range(draws):
            counts = np.bincount(rng.integers(0, len(universe.tfs), size=len(universe.tfs)), minlength=len(universe.tfs)).astype(float)
            for j, (y, s, tf) in enumerate(prepared.values()):
                table[d, j] = _ap_sorted(y, s, counts[tf])
        column = {key: j for j, key in enumerate(prepared)}
        for a, b in COMPARISONS:
            first, second = ([k for k in prepared if k[0] == arm] for arm in (a, b))
            if not first or not second:
                continue
            boot = table[:, [column[k] for k in first]].mean(1) - table[:, [column[k] for k in second]].mean(1)
            low, high = np.quantile(boot, [.025, .975])
            rows.append({'dataset': dataset, 'reference': name, 'comparison': f'{a} minus {b}',
                         'AP_difference': float(np.mean([point[k] for k in first]) - np.mean([point[k] for k in second])),
                         'low': float(low), 'high': float(high), 'seeds_first': len(first), 'seeds_second': len(second),
                         'reference_TFs': len(universe.tfs), 'draws': draws})
    return pd.DataFrame(rows)
