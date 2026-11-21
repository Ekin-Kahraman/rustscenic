"""Diagnosis 1: what do the calibrated Replogle response labels actually reward?

Reads only saved evidence (SCEPTRE pair results, frozen networks, Pearson). No refits.
Development analysis on already inspected K562/RPE1 data; not confirmation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
RECAL = ROOT / 'RustScenic-Biology-Recalibration-20261112'
PACKAGE = ROOT / 'RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm'
RETURNED = ROOT / 'RustScenic-Biology-Confirm-Results-Audit-20261112/returned'
OUT = Path(__file__).resolve().parent / 'diagnosis'
OUT.mkdir(exist_ok=True)
SEEDS = (777, 778, 779)


def macro_ap(labels, eligible, scores):
    """Equal-TF macro AP; rows = TFs, columns = genes; NaN-free scores."""
    values = []
    for i in range(labels.shape[0]):
        sel = eligible[i]
        y = labels[i, sel]
        if 0 < y.sum() < len(y):
            values.append(average_precision_score(y, scores[i, sel]))
        else:
            values.append(np.nan)
    return np.array(values)


rows = []
for dataset in ('k562', 'rpe1'):
    panel = json.loads((PACKAGE / f'inputs/{dataset}/panel.json').read_text())
    genes = np.array(panel['genes'])
    primary = (RECAL / f'inputs/{dataset}/primary-tfs.txt').read_text().split()
    pairs = pd.read_parquet(RECAL / f'results/{dataset}/corrected-pair-results.parquet')
    pairs = pairs[pairs.grna_target.isin(primary)]
    tfs = np.array(sorted(pairs.grna_target.unique()))
    wide = lambda column, fill: (pairs.pivot(index='grna_target', columns='response_id', values=column)
                                 .reindex(index=tfs, columns=genes).fillna(fill).to_numpy())
    eligible = wide('eligible', False).astype(bool)
    response = wide('response', False).astype(bool) & eligible
    effect = wide('log_2_fold_change', 0.0)
    n_tf = len(tfs)

    # Leave-one-TF-out responder frequency: how often a gene responds to OTHER knockdowns.
    total_resp = response.sum(0, keepdims=True)
    total_elig = eligible.sum(0, keepdims=True)
    loo_freq = (total_resp - response) / np.maximum(1, total_elig - eligible)
    gene_freq = (total_resp / np.maximum(1, total_elig)).ravel()

    scores = {'uniform': np.zeros(response.shape), 'generic_responder_LOO': loo_freq}
    training = None
    from scipy import sparse
    training = sparse.load_npz(PACKAGE / f'inputs/{dataset}/training.npz')
    mean_expr = np.asarray(training.mean(0)).ravel()
    detect = np.asarray((training > 0).mean(0)).ravel()
    scores['control_mean_expression'] = np.tile(mean_expr, (n_tf, 1))
    scores['control_detection_rate'] = np.tile(detect, (n_tf, 1))

    per_seed = {name: [] for name in ('full_pearson', 'native_raw', 'reference_raw')}
    for seed in SEEDS:
        job = RETURNED / f'jobs/{dataset}-{seed}'
        with np.load(job / 'native/pearson.npz') as z:
            rho = pd.DataFrame(z['rho'], index=z['TFs'], columns=z['genes']).reindex(index=tfs, columns=genes).to_numpy()
        per_seed['full_pearson'].append(np.abs(rho))
        for name, path in (('native_raw', 'native/network.parquet'), ('reference_raw', 'reference/network.parquet')):
            net = pd.read_parquet(job / path)
            mat = (net[net.TF.isin(tfs)].pivot(index='TF', columns='target', values='importance')
                   .reindex(index=tfs, columns=genes).fillna(0).to_numpy())
            per_seed[name].append(mat)
    signed_rho = rho

    result = {name: macro_ap(response, eligible, s) for name, s in scores.items()}
    for name, mats in per_seed.items():
        result[name] = np.nanmean([macro_ap(response, eligible, m) for m in mats], axis=0)

    # Specific responses: remove genes that respond to many other knockdowns.
    for cap in (0.10, 0.05, 0.02):
        specific = response & (loo_freq <= cap)
        elig_specific = eligible & ~(response & ~specific)  # drop generic positives from the universe
        result[f'specific<={cap}|uniform'] = np.array([
            specific[i, elig_specific[i]].mean() if 0 < specific[i, elig_specific[i]].sum() else np.nan for i in range(n_tf)])
        for name in ('full_pearson', 'native_raw', 'generic_responder_LOO'):
            mats = per_seed.get(name) or [scores[name]]
            result[f'specific<={cap}|{name}'] = np.nanmean([macro_ap(specific, elig_specific, m) for m in mats], axis=0)

    frame = pd.DataFrame(result, index=tfs)
    frame.to_csv(OUT / f'{dataset}-label-diagnosis-per-TF.csv')
    prevalence = np.array([response[i, eligible[i]].mean() for i in range(n_tf)])
    summary = frame.mean().to_dict()
    summary = {'dataset': dataset, 'TFs': n_tf, 'prevalence_macro': float(np.nanmean(prevalence)), **summary}

    # How concentrated are the positives?
    order = np.argsort(-gene_freq)
    cum = np.cumsum(total_resp.ravel()[order]) / total_resp.sum()
    summary['genes_holding_half_of_all_positives'] = int(np.searchsorted(cum, 0.5) + 1)
    summary['genes_responding_to_>=10%_of_knockdowns'] = int((gene_freq >= .10).sum())
    summary['positives_in_those_genes_fraction'] = float(total_resp.ravel()[gene_freq >= .10].sum() / total_resp.sum())
    # Per-TF concentration: share of positives contributed by the 20 strongest knockdowns.
    per_tf = response.sum(1)
    summary['share_of_positives_from_top20_knockdowns'] = float(np.sort(per_tf)[::-1][:20].sum() / per_tf.sum())
    summary['median_responders_per_TF'] = float(np.median(per_tf))
    # Does |rho| in control cells simply track generic responsiveness / expression?
    from scipy.stats import spearmanr
    mean_abs_rho = np.abs(signed_rho).mean(0)
    summary['spearman_gene_meanabsrho_vs_responder_freq'] = float(spearmanr(mean_abs_rho, gene_freq).statistic)
    summary['spearman_gene_expression_vs_responder_freq'] = float(spearmanr(mean_expr, gene_freq).statistic)
    summary['control_training_cells'] = int(training.shape[0])
    summary['rho_sd_observed'] = float(signed_rho.std())
    summary['rho_sd_expected_from_sampling_noise'] = float(1 / np.sqrt(training.shape[0]))
    rows.append(summary)

out = pd.DataFrame(rows).set_index('dataset').T
out.to_csv(OUT / 'label-diagnosis-summary.csv')
pd.set_option('display.width', 200, 'display.max_rows', 200)
print(out.to_string(float_format=lambda v: f'{v:.6g}'))
