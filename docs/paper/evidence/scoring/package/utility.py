"""Fixed, donor-aware biological utility checks. No model selection on test labels."""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20261106


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def evaluate(x, meta, name, pca_components=None, permutation=None, test_donors=None, train_donors=None):
    """Equal donor weights; labels unseen in training retained as errors, never dropped."""
    if not x.index.is_unique or not meta.index.is_unique:
        raise ValueError('Duplicate cell IDs')
    meta = meta.loc[x.index]
    if meta[['cell_type', 'donor']].isna().any().any():
        raise ValueError('Missing labels or donors')
    values = x.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite features')
    y = meta.cell_type.astype(str).to_numpy()
    donors = meta.donor.astype(str).to_numpy()
    train_y = y.copy()
    if permutation is not None:
        rng = np.random.default_rng(SEED + permutation)
        for donor in sorted(set(donors)):
            ix = np.flatnonzero(donors == donor)
            train_y[ix] = rng.permutation(y[ix])
    rows = []
    for donor in sorted(set(donors)) if test_donors is None else test_donors:
        test = donors == donor
        train = ~test if train_donors is None else np.isin(donors, train_donors)
        if np.any(train & test):
            raise ValueError('Training and test donors overlap')
        if test.sum() < 50 or len(set(y[train])) < 2:
            rows.append({'donor': str(donor), 'status': 'not_assessed', 'reason': 'insufficient cells/classes'})
            continue
        steps = [StandardScaler()]
        if pca_components:
            steps.append(PCA(n_components=min(pca_components, values.shape[1], train.sum() - 1),
                             svd_solver='randomized', random_state=SEED))
            steps.append(StandardScaler())
        steps.append(LogisticRegression(C=1., class_weight='balanced', max_iter=2000,
                                        solver='lbfgs', random_state=SEED))
        model = make_pipeline(*steps)
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(values[train], train_y[train])
            predicted = model.predict(values[test])
        labels = sorted(set(y))
        rows.append({'donor': str(donor), 'status': 'complete', 'test_cells': int(test.sum()),
                     'train_cells': int(train.sum()), 'features': values.shape[1],
                     'balanced_accuracy': float(balanced_accuracy_score(y[test], predicted)),
                     'macro_f1': float(f1_score(y[test], predicted, labels=sorted(set(y[test])),
                                              average='macro', zero_division=0)),
                     'unseen_test_labels': sorted(set(y[test]) - set(y[train])),
                     'converged': not any(issubclass(w.category, ConvergenceWarning) for w in captured),
                     'labels': labels, 'confusion_matrix': confusion_matrix(y[test], predicted, labels=labels).tolist()})
    return {'name': name, 'permutation': permutation, 'pca_components': pca_components,
            'signature_names': x.columns.tolist(), 'folds': rows}


def paired_summary(left, right, bootstrap=10000):
    """Average algorithm seeds within donors before bootstrapping biological replicates."""
    def by_donor(results, metric):
        vals = {}
        for result in results:
            for row in result['folds']:
                if row['status'] == 'complete':
                    vals.setdefault(row['donor'], []).append(row[metric])
        return {d: float(np.mean(v)) for d, v in vals.items()}
    out = {}
    for metric in ['balanced_accuracy', 'macro_f1']:
        a, b = by_donor(left, metric), by_donor(right, metric)
        donors = sorted(a.keys() & b.keys())
        if len(donors) < 2:
            out[metric] = {'status': 'insufficient_donors'}
            continue
        delta = np.array([a[d] - b[d] for d in donors])
        draw = np.random.default_rng(SEED).integers(0, len(delta), size=(bootstrap, len(delta)))
        ci = np.quantile(delta[draw].mean(axis=1), [.025, .975])
        out[metric] = {'left_mean': float(np.mean([a[d] for d in donors])),
                       'right_mean': float(np.mean([b[d] for d in donors])),
                       'difference': float(delta.mean()), 'paired_donor_bootstrap_ci95': ci.tolist(),
                       'donors': len(donors), 'donor_differences': dict(zip(donors, delta.tolist()))}
    return out


def pair_features(rust, reference):
    if set(rust.index) != set(reference.index):
        raise ValueError('Unmatched cell identities')
    reference = reference.loc[rust.index]
    common = sorted(set(rust.columns) & set(reference.columns))
    if not common:
        raise ValueError('No common TF signatures')
    return {'rust_native': rust, 'reference_native': reference,
            'rust_common': rust[common], 'reference_common': reference[common]}


def marker_checks(frame, meta, expected):
    from sklearn.metrics import roc_auc_score
    meta = meta.loc[frame.index]
    rows = []
    for tf, labels in expected.items():
        cols = [c for c in [tf, tf + '_activator'] if c in frame.columns]
        if not cols:
            rows.append({'TF': tf, 'status': 'missing_regulon', 'expected': labels})
            continue
        for donor in sorted(meta.donor.unique()):
            take = meta.donor.eq(donor)
            target = meta.loc[take, 'cell_type'].isin(labels).to_numpy()
            if target.sum() < 20 or (~target).sum() < 20:
                continue
            rows.append({'TF': tf, 'donor': str(donor), 'status': 'assessed',
                         'target_cells': int(target.sum()), 'other_cells': int((~target).sum()),
                         'auroc': float(roc_auc_score(target, frame.loc[take, cols[0]])),
                         'expected': labels})
    return rows
