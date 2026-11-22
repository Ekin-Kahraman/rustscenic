"""Figure 3: TF binning and fidelity to pySCENIC, from the saved audit and diagnosis tables.

A: protocol fidelity with RustScenic 0.6's 255 bins (returned, confirmatory) and exact bins
   (local refits, post hoc). B: bone-marrow fidelity against the bin budget (current rule).
C: GRN time of the 0.7.0 release candidate (exact bins) relative to a v0.6.1 build (255 bins).
Fill and colour both encode binning, so identity is not colour alone. Fits nothing.
Text is sized to print at 7 pt or more at the 174 mm (6.85 in) text width (20 November 2026).
Restyled 21 November 2026: connectors print at 1.6 pt, marker edges at 1.25 pt, and the grey and green
are stronger (the grey darker, so the pair stays apart under red-green colour-vision deficiency).
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)

SURFACE, INK, MUTED, GRID = '#ffffff', '#0b0b0b', '#52514e', '#e4e3df'
OLD, NEW = '#494842', '#098064'  # 255 bins (hollow), exact bins (filled); were '#8a8983', '#0f8a6c'
CONNECT = '#b2b1ad'  # connectors; was GRID
PRINT = 11.2 / 6.85  # set size = printed size x PRINT once the 11.2 in figure is placed at 6.85 in
# Printed sizes (pt): connectors 1.6, reference lines 1.0, markers 4.4 across with 1.25 pt edges
LW, REF_LW, MS, MEW = 1.6 * PRINT, 1.0 * PRINT, 4.4 * PRINT, 1.25 * PRINT
SMALL, BODY, HEAD = 7 * PRINT, 7.3 * PRINT, 7.6 * PRINT
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': BODY, 'axes.edgecolor': GRID, 'axes.linewidth': .8,
                     'xtick.color': MUTED, 'ytick.color': INK, 'text.color': INK, 'axes.labelcolor': MUTED,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})

checks = pd.read_csv(HERE / 'audit/checks-recomputed.csv')
returned = checks[checks.check == 'fidelity'].set_index(['dataset', 'subject']).observed
pattern = re.compile(r'^\s*(\S+)\s+(\S+)\s+(current|original)\s+(all TFs)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
local = {}
for line in ((HERE / 'diagnosis/fidelity-queue.txt').read_text() + (HERE / 'diagnosis/fidelity-bmmc-exact.txt').read_text()).splitlines():
    m = pattern.match(line)
    if m:
        local[m.group(2)] = float(m.group(8))
exact_bmmc = {'current': local['cap65535-arboreto-{seed}'], 'original': local['cap65535-paired_oob-{seed}']}  # fidelity-bmmc-exact.txt

rows = [('BMMC, current rule', returned[('bmmc', 'current rule')], exact_bmmc['current']),
        ('BMMC, original rule', returned[('bmmc', 'original rule')], exact_bmmc['original']),
        ('Atlas 20,000, current rule', returned[('atlas20000', 'current rule')], local['atlas20000-cap65535-arboreto-{seed}']),
        ('Atlas 20,000, original rule', returned[('atlas20000', 'original rule')], local['atlas20000-cap65535-paired_oob-{seed}'])]
budget = [('255', returned[('bmmc', 'current rule')]), ('1,023', local['cap1023-arboreto-{seed}']),
          ('4,095', local['cap4095-arboreto-{seed}']), ('Exact', exact_bmmc['current'])]

timing = pd.read_csv(HERE / 'diagnosis/release-timings.csv')  # final-timing.sh runs, laptop, seed 777, current rule
ratios = []
for label, inp in (('BMMC\n20,000\ncells\n2 runs/build', 'BMMC 20,000'), ('Atlas\n1,306,127\ncells\n1 run/build', 'Atlas 1,306,127')):
    t = timing[timing.input == inp]
    exact = t[t.max_bins == 65535].inference_s
    binned = t[t.max_bins == 255].inference_s
    ratios.append((label, exact.mean() / binned.mean()))

fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.4), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.35, 1, 1]})
fig.subplots_adjust(left=.12, right=.985, top=.795, bottom=.215, wspace=.38)

ax = axes[0]
for i, (label, old, new) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([old, new], [y, y], color=CONNECT, lw=LW, zorder=1, solid_capstyle='round')
    ax.plot(old, y, 'o', ms=MS, mfc=SURFACE, mec=OLD, mew=MEW, zorder=3)
    ax.plot(new, y, 'o', ms=MS, mfc=NEW, mec=NEW, mew=MEW, zorder=3)
    ax.text(new + .011, y, f'{new:.3f}', va='center', fontsize=SMALL, color=INK, bbox=dict(fc=SURFACE, ec='none', pad=1), zorder=4)
    ax.text(old - .011, y, f'{old:.3f}', va='center', ha='right', fontsize=SMALL, color=MUTED, bbox=dict(fc=SURFACE, ec='none', pad=1), zorder=4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0].replace(', ', ',\n', 1) for r in rows][::-1])  # two lines: row names on one line no longer fit
ax.set_xlim(.68, 1.04)
ax.set_title(r'$\bf{A}$  Fidelity to pySCENIC (protocol metric)', fontsize=HEAD, loc='left', pad=7)

ax = axes[1]
xs = range(len(budget))
ax.plot(list(xs), [v for _, v in budget], color=CONNECT, lw=LW, zorder=1)
for x, (label, v) in zip(xs, budget):
    exact = label == 'Exact'
    ax.plot(x, v, 'o', ms=MS, mfc=NEW if label != '255' else SURFACE, mec=NEW if label != '255' else OLD, mew=MEW, zorder=3)
    ax.text(x + (.12 if label == '255' else 0), v + (-.03 if label == '255' else .012), f'{v:.3f}', ha='left' if label == '255' else 'center', fontsize=SMALL, color=INK if label != '255' else MUTED)
ax.set_xticks(list(xs))
ax.set_xticklabels([b for b, _ in budget])
ax.set_xlabel('Bins per TF (BMMC, current rule)')
ax.set_ylim(.70, 1.03)
ax.set_title(r'$\bf{B}$  Fidelity against bin budget', fontsize=HEAD, loc='left', pad=7)

ax = axes[2]
ax.axhline(1, color=MUTED, lw=REF_LW, zorder=1)
for x, (label, r) in enumerate(ratios):
    ax.bar(x, r - 1, bottom=1, width=.55, color=NEW, zorder=2)
    ax.text(x, r + (.006 if r >= 1 else -.006), f'{(r - 1) * 100:+.1f}%', ha='center', va='bottom' if r >= 1 else 'top',
            fontsize=SMALL, color=INK)
ax.set_xticks(range(len(ratios)))
ax.set_xticklabels([label for label, _ in ratios])
ax.set_xlim(-.6, len(ratios) - .4)
ax.set_ylim(.95, 1.12)
ax.set_yticks([.96, .98, 1, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12])  # the ticks the smaller text gave
ax.set_ylabel('GRN time, 0.7.0 ÷ 0.6.1')
ax.set_title(r'$\bf{C}$  Cost of exact binning', fontsize=HEAD, loc='left', pad=7)

for ax in axes[:2]:
    # zorder .9: above the gridline at 0.90 (zorder .5), which otherwise splits the dashes; below the data
    ax.axvline(.9, color=MUTED, lw=REF_LW, ls=(0, (3, 3)), zorder=.9) if ax is axes[0] else ax.axhline(.9, color=MUTED, lw=REF_LW, ls=(0, (3, 3)), zorder=.9)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(axis='x' if ax is axes[0] else 'y', color=GRID, lw=.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, color=GRID)
axes[0].tick_params(axis='y', length=0, pad=10)
axes[0].spines['left'].set_visible(False)

handles = [plt.Line2D([], [], marker='o', ls='none', ms=MS, mfc=SURFACE, mec=OLD, mew=MEW, label='255 bins (RustScenic 0.6; HPC, pre-specified)'),
           plt.Line2D([], [], marker='o', ls='none', ms=MS, mfc=NEW, mec=NEW, mew=MEW, label='Finer or exact bins (local refits, post hoc)'),
           plt.Line2D([], [], color=MUTED, lw=REF_LW, ls=(0, (3, 3)), label='Pre-specified criterion 0.90')]
fig.legend(handles=[handles[0], handles[2], handles[1]], loc='upper left', bbox_to_anchor=(.01, .99), ncol=2,
           frameon=False, fontsize=BODY)  # two rows: 255 bins | finer bins, then the criterion
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'05-binning-fidelity.{ext}', dpi=300 if ext == 'png' else None, facecolor=SURFACE)
print('ratios', [(l.replace(chr(10), ' '), round(r, 3)) for l, r in ratios])
