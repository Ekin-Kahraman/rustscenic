"""Stopping-rule figure from the saved development and confirmation tables.

Colour = stopping rule (two validated categorical slots). Marker = implementation.
Reads only CSVs written by beeline_score.py; fits nothing.
Text is sized to print at 7 pt or more at the 174 mm (6.85 in) text width (20 November 2026).
Restyled 21 November 2026: data lines and connectors print at 1.5 pt or more, marker edges at 1.2 pt
or more, stronger versions of the same colours, and a taller figure so the larger markers stay apart.
22 November 2026 (layout audit): white ground, bold panel letters, direct rule labels in panel A, panel D titled
as in the text.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/figures'
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, MUTED, GRID = '#ffffff', '#0b0b0b', '#52514e', '#e4e3df'
# original rule, current rule, Pearson: stronger versions (same OKLCH hue) of '#2a78d6', '#eb6834', '#8a8983'
SAME, CROSS, NEUTRAL = '#0b69cc', '#de5310', '#6d6c66'
CONNECT = '#b2b1ad'  # dumbbell connectors; was GRID
GROUPS = [('Development', 'beeline-dev', ['hESC', 'mDC', 'mESC']),
          ('Confirmation', 'beeline-confirmation', ['hHep', 'mHSC-E', 'mHSC-GM', 'mHSC-L'])]
PRIMARY = {'hESC': 'hESC-ChIP-seq', 'mDC': 'mDC-ChIP-seq', 'mESC': 'mESC-ChIP-seq', 'hHep': 'HepG2-ChIP-seq',
           'mHSC-E': 'mHSC-ChIP-seq', 'mHSC-GM': 'mHSC-ChIP-seq', 'mHSC-L': 'mHSC-ChIP-seq'}

PRINT = 11.2 / 6.85  # set size = printed size x PRINT once the 11.2 in figure is placed at 6.85 in
# Printed line and marker sizes (pt): data lines/connectors 1.6, Pearson 1.8, reference lines 1.0,
# marker edges 1.25, circles 4.15, diamonds 3.75 (a diamond's diagonal is 1.41 x its size)
LW, PEARSON_LW, REF_LW, MEW = 1.6 * PRINT, 1.8 * PRINT, 1.0 * PRINT, 1.25 * PRINT
MS_O, MS_D = 4.15 * PRINT, 3.75 * PRINT
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.3 * PRINT, 'axes.edgecolor': GRID, 'axes.linewidth': .8,
                     'xtick.color': MUTED, 'ytick.color': INK, 'text.color': INK, 'axes.labelcolor': MUTED,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})

rows, y = [], 0
positions, separators = {}, []
for label, folder, datasets in GROUPS:
    for d in datasets:
        positions[d] = y
        rows.append((d, folder))
        y += 1
    separators.append((label, y - len(datasets), y - 1))
    y += .9
tables = {folder: {name: pd.read_csv(HERE / 'results' / folder / f'{name}.csv') for name in ('cost', 'stability', 'recovery', 'fidelity')}
          for _, folder, _ in GROUPS}

H = 6.0  # was 5.3 in; the extra 0.7 in goes to the panels so the circle and diamond rows do not touch
fig, axes = plt.subplots(1, 4, figsize=(11.2, H), sharey=True, facecolor=SURFACE)
fig.subplots_adjust(left=.105, right=.985, top=1 - .848 / H, bottom=1.3515 / H, wspace=.12)  # same title and legend space
OFFSET = .17  # RustScenic above, pySCENIC below


def dumbbell(ax, yv, cross, same, marker, filled):
    ax.plot([cross, same], [yv, yv], color=CONNECT, lw=LW, zorder=1, solid_capstyle='round')
    for value, colour in ((cross, CROSS), (same, SAME)):
        ax.plot(value, yv, marker=marker, ms=MS_O if marker == 'o' else MS_D, mfc=colour if filled else SURFACE,
                mec=colour, mew=MEW, zorder=3, ls='none')


for d, folder in rows:
    t = tables[folder]
    yv = positions[d]
    cost = t['cost'].set_index(['dataset', 'arm']).trees_mean
    stab = t['stability'].set_index(['dataset', 'arm']).top50_jaccard
    for prefix, dy, marker, filled in (('native', -OFFSET, 'o', True), ('reference', OFFSET, 'D', False)):
        dumbbell(axes[0], yv + dy, cost[(d, f'{prefix}_cross')], cost[(d, f'{prefix}_same')], marker, filled)
        dumbbell(axes[1], yv + dy, stab[(d, f'{prefix}_cross')], stab[(d, f'{prefix}_same')], marker, filled)
    rec = t['recovery']
    rec = rec[(rec.dataset == d) & (rec.reference == PRIMARY[d])].set_index('arm').AP_over_prevalence
    for prefix, dy, marker, filled in (('native', -OFFSET, 'o', True), ('reference', OFFSET, 'D', False)):
        dumbbell(axes[2], yv + dy, rec[f'{prefix}_cross'], rec[f'{prefix}_same'], marker, filled)
    axes[2].plot([rec['pearson']] * 2, [yv - .32, yv + .32], color=NEUTRAL, lw=PEARSON_LW, zorder=2, solid_capstyle='round')
    fid = t['fidelity'].set_index(['dataset', 'pair']).ratio
    for arm, colour, dy in (('native_cross vs reference_cross', CROSS, -OFFSET), ('native_same vs reference_same', SAME, OFFSET)):
        axes[3].plot(fid[(d, arm)], yv + dy, marker='o', ms=MS_O, mfc=colour, mec=colour, mew=MEW, ls='none', zorder=3)

titles = [r'$\bf{A}$  Mean trees fitted' + '\n     per gene', r'$\bf{B}$  Seed-to-seed overlap' + '\n     of top-50 targets',
          r'$\bf{C}$  Cell-type ChIP recovery' + '\n     (AP ÷ chance)', r'$\bf{D}$  Fidelity ratio' + '\n     (RustScenic vs arboreto)']
for ax, title in zip(axes, titles):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=7.6 * PRINT, color=INK, loc='left', pad=7)
    ax.grid(axis='x', color=GRID, lw=.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=3, color=GRID)
axes[0].set_xlim(0, 290)
_c = tables['beeline-dev']['cost'].set_index(['dataset', 'arm']).trees_mean
axes[0].text(_c[('hESC', 'native_cross')], -0.42, 'current\nrule', color=CROSS, fontsize=7 * PRINT, ha='center', va='bottom')
axes[0].text(_c[('hESC', 'native_same')], -0.42, 'original\nrule', color=SAME, fontsize=7 * PRINT, ha='center', va='bottom')
axes[1].set_xlim(0, .75)
axes[2].set_xlim(.93, 1.24)
axes[2].axvline(1, color=MUTED, lw=REF_LW, zorder=1)
axes[2].text(1.004, -0.75, 'chance', fontsize=7 * PRINT, color=MUTED, va='center')
axes[3].set_xlim(.86, 1.03)
axes[3].axvline(.90, color=MUTED, lw=REF_LW, ls=(0, (4, 3)), zorder=1)
axes[3].text(.903, -0.75, 'criterion 0.90', fontsize=7 * PRINT, color=MUTED, va='center',
             bbox=dict(fc=SURFACE, ec='none', pad=1))
axes[3].axvline(1, color=GRID, lw=1.2, zorder=1)
axes[0].set_yticks([positions[d] for d, _ in rows])
axes[0].set_yticklabels([d for d, _ in rows])
axes[0].set_ylim(y - .9 - .45, -1.2)
for label, top, bottom in separators:
    fig.text(.012, axes[0].transData.transform((0, (top + bottom) / 2))[1] / fig.bbox.height, label, rotation=90,
             va='center', ha='center', fontsize=7.3 * PRINT, color=MUTED)

rules = [Line2D([], [], marker='o', ls='none', ms=MS_O, mfc=CROSS, mec=CROSS, mew=MEW, label='Current rule (scikit-learn ≥ 1.3): improvement compared across different cells'),
         Line2D([], [], marker='o', ls='none', ms=MS_O, mfc=SAME, mec=SAME, mew=MEW, label='Original rule: improvement measured on the same held-out cells')]
tools = [Line2D([], [], marker='o', ls='none', ms=MS_O, mfc=MUTED, mec=MUTED, mew=MEW, label='RustScenic'),
         Line2D([], [], marker='D', ls='none', ms=MS_D, mfc=SURFACE, mec=MUTED, mew=MEW, label='arboreto'),
         Line2D([], [], color=NEUTRAL, lw=PEARSON_LW, label='Absolute Pearson')]
style = dict(frameon=False, fontsize=7.3 * PRINT, handletextpad=.4, columnspacing=1.6, labelcolor=MUTED)
fig.legend(handles=rules, loc='lower left', ncol=1, bbox_to_anchor=(.095, .318 / H), **style)
fig.legend(handles=tools, loc='lower left', ncol=3, bbox_to_anchor=(.095, .0), **style)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'stopping-rule.{ext}', dpi=220, facecolor=SURFACE)
print('written', OUT / 'stopping-rule.png')
