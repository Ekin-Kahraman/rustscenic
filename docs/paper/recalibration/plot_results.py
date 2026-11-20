"""Publication figure from the frozen corrected-score tables, with source data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

B = Path(__file__).resolve().parent
rank = pd.read_csv(B/'summary/rankings-per-TF.csv')
rank = rank[rank.primary_roster & rank.threshold.eq(.5)]
coverage = pd.read_csv(B/'summary/coverage-macro.csv')
methods = ['original', 'diverse_importance', 'reference', 'native_raw', 'reference_raw', 'full_pearson']
names = ['RustScenic original', 'RustScenic experimental', 'pySCENIC final',
         'RustScenic raw GRN', 'Arboreto raw GRN', 'Absolute Pearson']
colours = ['#3c75a9', '#dd9851', '#709c79', '#3c75a9', '#709c79', '#7d6c9f']
rows = []
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'svg.fonttype':'none'})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), gridspec_kw={'height_ratios':[1.3, 1]})
for col, dataset in enumerate(['k562', 'rpe1']):
    a = rank[rank.dataset.eq(dataset)].groupby(['TF', 'method']).average_precision.mean().unstack()
    ax = axes[0, col]
    for i, method in enumerate(methods):
        delta = (a[method] - a['uniform']).to_numpy()
        rng = np.random.default_rng(20261112)
        bootstrap = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(1)
        lo, hi = np.quantile(bootstrap, [.025, .975])
        mean = delta.mean()
        ax.errorbar(mean, i, xerr=[[max(0., mean-lo)], [max(0., hi-mean)]],
                    fmt='o', capsize=3, color=colours[i], markersize=6)
        rows.append(dict(dataset=dataset, method=method, TFs=len(delta),
                         mean_AP_difference=mean, low=lo, high=hi))
    ax.axvline(0, color='#777777', linewidth=1, linestyle='--')
    ax.set_yticks(range(len(methods)), names)
    ax.invert_yaxis()
    ax.set_xlabel('Average precision minus uniform baseline')
    ax.set_title(f'{dataset.upper()}: response ranking\n{len(a)} evaluable regulators', loc='left', pad=15, fontsize=11)
    ax.grid(axis='x', alpha=.15)

    ax = axes[1, col]
    c = coverage[coverage.dataset.eq(dataset)].set_index('method').loc[methods[:3]]
    x = np.arange(3)
    bars1=ax.bar(x-.18, c.TFs_with_predictions/c.roster_TFs, .34, color='#3c75a9', label='Regulators with predictions')
    bars2=ax.bar(x+.18, c.all_roster_pooled_precision, .34, color='#dd9851', label='Detected responses among emitted predictions')
    ax.bar_label(bars1, labels=[f'{v:.1%}' for v in c.TFs_with_predictions/c.roster_TFs], padding=3, fontsize=8)
    ax.bar_label(bars2, labels=[f'{v:.1%}' for v in c.all_roster_pooled_precision], padding=3, fontsize=8)
    ax.set_xticks(x, ['Original', 'Experimental', 'Reference'])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, .2)
    ax.set_title(f'{dataset.upper()}: full primary roster\n{int(c.roster_TFs.iloc[0])} regulators', loc='left', pad=15, fontsize=11)
    ax.set_ylabel('Fraction (different denominators)')
    ax.grid(axis='y', alpha=.15)
    if col == 0:
        ax.legend(loc='upper left', fontsize=8, frameon=False)

fig.suptitle('Corrected perturbation-response recovery and prediction coverage', fontsize=16, y=.98)
fig.text(.5, .025,
         'Top: seeds averaged within regulator; conditional 95% regulator-bootstrap intervals.\n'
         'Bottom: prediction coverage across the full primary roster; precision among emitted tested edges.\n'
         'Post-hoc analysis of two cell contexts from one study; response recovery is not direct-edge accuracy.',
         ha='center', fontsize=9, color='#555555')
fig.tight_layout(rect=[0, .10, 1, .95], h_pad=3, w_pad=3)
for suffix in ['png', 'pdf', 'svg']:
    fig.savefig(B/f'figures/response-recovery.{suffix}', dpi=180)
pd.DataFrame(rows).to_csv(B/'summary/figure-ap-differences.csv', index=False)
print('Wrote response-recovery figure and source table')
