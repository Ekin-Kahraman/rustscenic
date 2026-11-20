"""Invariant checks for the scoring code. Run: python test_package.py (or pytest)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import orchestrate
import summarise
from common import write_json
from fit import top_table
from lineage import auroc


def fake_arm(work, dataset, arm, seed, tops, trees, regulons=None, emulated=False):
    folder = work / f'jobs/{dataset}-{seed}/{arm}'
    folder.mkdir(parents=True)
    rows = [(tf, target, 100. - rank) for tf, targets in tops.items() for rank, target in enumerate(targets)]
    pd.DataFrame(rows, columns=['TF', 'target', 'importance']).to_parquet(folder / 'top100.parquet', index=False)
    write_json(folder / 'SUMMARY.json', {
        'status': 'complete', 'cells': 10, 'genes': 10, 'TFs': 2, 'stopping_rule': arm, 'same_mask_monitor_emulated': emulated,
        'inference_s': 1., 'wall_s': 2., 'trees_mean': trees, 'trees_median': trees, 'trees_p90': trees, 'trees_max': int(trees),
        'share_at_ceiling': 0., 'edges': len(rows)})
    for name, members in (regulons or {}).items():
        write_json(folder / f'regulons-{name}.json', members)


def targets(prefix, n=50):
    return [f'{prefix}{i:02d}' for i in range(n)]


def test_summary_values_and_verdicts():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shared, other = targets('s'), targets('o')
        for dataset in ('bmmc', 'atlas20000'):
            for seed in (777, 778, 779):
                # current rule: every seed and tool disagrees on TF A; original rule: identical everywhere
                fake_arm(work, dataset, 'native_cross', seed, {'A': targets(f'n{seed}'), 'B': shared}, 25.)
                fake_arm(work, dataset, 'reference_cross', seed, {'A': targets(f'r{seed}'), 'B': shared}, 26.)
                fake_arm(work, dataset, 'native_same', seed, {'A': other, 'B': shared}, 250.,
                         {'diverse_importance': {'A_activator': ['x', 'y', str(seed)]}})
                fake_arm(work, dataset, 'reference_original', seed, {'A': other, 'B': shared}, 260.)
        summarise.summarise(work)
        stability = pd.read_csv(work / 'summary/stability.csv').set_index(['dataset', 'arm']).top50_jaccard
        assert stability['bmmc', 'native_cross'] == .5 and stability['bmmc', 'native_same'] == 1.
        fidelity = pd.read_csv(work / 'summary/fidelity.csv').set_index(['dataset', 'rule']).ratio
        assert fidelity['bmmc', 'current'] == 1. and fidelity['atlas20000', 'original'] == 1.
        regulons = pd.read_csv(work / 'summary/regulon-stability.csv').iloc[0]
        assert regulons.regulons_in_all_seeds == 1 and regulons.target_jaccard_shared == .5
        verdict = json.loads((work / 'summary/VERDICT.json').read_text())
        confirmatory = [c for c in verdict['checks'] if c['confirmatory']]
        assert len(confirmatory) == 12 and verdict['status'] == 'complete' and verdict['all_confirmatory_checks_met']
        assert verdict['reference_original_implementation'] == 'scikit-learn 1.2.2'
        trees = [c['observed'] for c in confirmatory if c['endpoint'].startswith('1') and c['subject'] == 'pySCENIC']
        assert trees == [10., 10.]


def test_missing_arm_is_not_a_pass():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fake_arm(work, 'bmmc', 'native_cross', 777, {'A': targets('a')}, 25.)
        summarise.summarise(work)
        verdict = json.loads((work / 'summary/VERDICT.json').read_text())
        assert verdict['status'] == 'partial' and verdict['all_confirmatory_checks_met'] is None
        assert all(c['met'] is None for c in verdict['checks'])
        assert verdict['reference_recovery']['status'] == 'partial' and verdict['reference_recovery']['improves'] is None


def test_auroc_matches_scikit_learn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    for _ in range(50):
        score = rng.integers(0, 6, size=200).astype(float)
        positive = rng.random(200) < .3
        assert abs(auroc(score, positive) - roc_auc_score(positive, score)) < 1e-12


def test_top_table_order_and_filters():
    network = pd.DataFrame({'TF': ['A', 'A', 'A', 'A', 'B'], 'target': ['z', 'y', 'A', 'x', 'q'],
                            'importance': [2., 2., 9., 0., 1.]})
    top = top_table(network)
    assert top[['TF', 'target']].values.tolist() == [['A', 'y'], ['A', 'z'], ['B', 'q']]


def test_plan_fits_inside_wall_time():
    assert len(orchestrate.JOBS) == 12
    assert sum(len(orchestrate.arms_of(name)) for name in orchestrate.JOBS) == 44
    for name, spec in orchestrate.PLAN.items():
        arms = orchestrate.arms_of(name)
        hours = int(spec['wall'].split(':')[0])
        assert set(arms) <= set(spec['limit']) and sum(spec['limit'].values()) < hours <= 144, name
        assert arms[0] == 'pearson' if name.endswith('-777') else 'pearson' not in arms
    assert orchestrate.arms_of('bmmc-777')[1:3] == ('native_cross', 'native_same')
    assert orchestrate.arms_of('bmmc-778')[:2] == ('native_same', 'native_cross')


def test_recovery_matches_scikit_learn_and_brute_force_bootstrap():
    import recovery
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    genes = [f'g{i}' for i in range(60)] + ['Dup', 'DUP']  # mixed case; one upper-case collision is dropped
    tfs = genes[:8]
    reference = pd.DataFrame([(t.upper(), g.upper()) for t in tfs for g in rng.choice(genes[:60], 10, replace=False) if g != t],
                             columns=['TF', 'target'])
    original, recovery.load_reference = recovery.load_reference, lambda name: reference
    try:
        networks = {(arm, 1): pd.DataFrame([(t, g, float(rng.integers(0, 5))) for t in tfs for g in genes if g != t],
                                           columns=['TF', 'target', 'importance']) for arm in ('native_same', 'native_cross')}
        universe = recovery.Universe(genes, tfs, reference)
        assert len(universe.columns) == 60
        for network in networks.values():
            scores = universe.scores(network)
            assert np.array_equal(scores, universe.scores(network.astype({'TF': 'category', 'target': 'category'})))
            got = recovery.metrics(universe, scores)['AP']
            assert abs(got - average_precision_score(universe.labels[universe.eligible], scores[universe.eligible])) < 1e-12
        row = recovery.intervals(networks, genes, tfs, 'bmmc', draws=200, seed=5).iloc[0]
        draw_rng, differences = np.random.default_rng(5), []
        for _ in range(200):
            draw = draw_rng.integers(0, len(universe.tfs), size=len(universe.tfs))
            ap = [average_precision_score(np.concatenate([universe.labels[i, universe.eligible[i]] for i in draw]),
                                          np.concatenate([universe.scores(n)[i, universe.eligible[i]] for i in draw]))
                  for n in networks.values()]
            differences.append(ap[0] - ap[1])
        assert np.allclose([row.low, row.high], np.quantile(differences, [.025, .975]), atol=1e-12)
    finally:
        recovery.load_reference = original


def test_same_mask_monitor_differs_from_cross_mask_after_round_zero():
    import sklearn
    from sklearn.ensemble import GradientBoostingRegressor
    from reference_paired import PairedEarlyStopMonitor
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5))
    y = x[:, 0] + rng.normal(size=300)
    monitor = PairedEarlyStopMonitor(10 ** 6)  # never stops; round zero asserts equality with scikit-learn
    model = GradientBoostingRegressor(n_estimators=30, subsample=.9, learning_rate=.01, random_state=1).fit(x, y, monitor=monitor)
    same_definition = np.allclose(monitor.history, model.oob_improvement_, rtol=1e-6, atol=1e-9)
    assert same_definition == (tuple(map(int, sklearn.__version__.split('.')[:2])) < (1, 3))


if __name__ == '__main__':
    for name, function in sorted(globals().items()):
        if name.startswith('test_'):
            function()
            print('passed', name)
