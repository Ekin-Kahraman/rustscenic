"""Print-legible Figures 1, 4 and 5 for the preprint (20 November 2026).

python3 -I figures_fix.py

Reads the same checksummed evidence as validation/paper/build.py and repeats its
figure code with larger text (every label prints at 7 pt or more when the figure
is placed at the 174 mm text width) and reader-facing labels. Restyled 21 November 2026
after co-author feedback: data lines and connectors print at 1.5 pt or more, marker edges at
1.2 pt or more, and the data colours are stronger versions of the same hues. Writes only
docs/paper/figures/0{1,2,3}-*.{svg,pdf,png}; no tables, RESULTS.md or
build-verification.json. The plotted values are those of build.py.
22 November 2026 (layout audit): bold panel letters, no titles inside the images (captions carry them),
dataset names as in the text, Figure 4 shortened with y axes from 0.2, a thinner halo in Figure 5, 300 dpi.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / 'docs/paper'
E = PAPER / 'evidence'
FIG = PAPER / 'figures'
load = lambda name: json.loads((E / name).read_text())
csv = lambda name: pd.read_csv(E / name)
manifest = load('manifest.json')
for f in manifest['files']:
    data = (E / f['path']).read_bytes()
    assert len(data) == f['bytes'] and hashlib.sha256(data).hexdigest() == f['sha256'], f['path']

# ---- inputs, as derived in validation/paper/build.py (checks kept, writes removed) ----
runs = csv('scaling/consolidated-runs.csv')
assert len(runs) == 42 and runs.status.eq('complete').all()

folds = csv('biology/all-utility-folds.csv')
primary = folds[folds.group.eq('primary')]
donors = primary.groupby(['n', 'name', 'donor']).macro_f1.mean().reset_index()
original = donors.groupby(['n', 'name']).macro_f1.mean().unstack()
reported_bio = load('biology/returned/biological-summary.json')['holdout_contrasts']
for n in [5000, 20000]:
    for name, side in [('rust_common', 'left_mean'), ('reference_common', 'right_mean')]:
        assert np.isclose(original.loc[n, name], reported_bio[str(n)]['common']['macro_f1'][side], atol=1e-12)

verification = load('scoring/returned/biology/seeded/verification.json')
corrected = []
for record in verification['records']:
    for f in record['result']['folds']:
        m = np.asarray(f['confusion_matrix'], dtype=float)
        truth = m.sum(1)
        present = truth > 0
        denom = m.sum(0) + truth
        f1 = np.divide(2*np.diag(m), denom, out=np.zeros(len(m)), where=denom > 0)[present].mean()
        assert f['converged'] and np.isclose(f1, f['macro_f1'], atol=1e-12)
        corrected.append(dict(n=record['training_cells'], seed=record['inference_seed'],
                              mode=record['mode'], donor=f['donor'], macro_f1=f1))
corrected = pd.DataFrame(corrected)
assert len(corrected) == 96
external = csv('external/returned/biology-folds.csv')

directional_rows = []
for record in load('directional/returned/biology/verification.json')['records']:
    assert record['passed'] and record['max_absolute_error'] == 0
    for mode, result in record['results'].items():
        for fold in result['folds']:
            m = np.asarray(fold['confusion_matrix'], dtype=float)
            support, predicted = m.sum(1), m.sum(0)
            present = support > 0
            f1 = np.divide(2*m.diagonal(), support+predicted,
                           out=np.zeros(len(m)), where=support+predicted > 0)[present].mean()
            assert fold['converged'] and fold['status'] == 'complete'
            np.testing.assert_allclose(f1, fold['macro_f1'], rtol=0, atol=1e-12)
            directional_rows.append(dict(training_cells=record['training_cells'], seed=record['seed'],
                                         method=record['method'], mode=mode, donor=fold['donor'], macro_f1=f1))
directional = pd.DataFrame(directional_rows)
assert len(directional) == 192

ref = csv('biology/returned/rna-reference-recovery.csv')
rna = ref.groupby(['reference', 'method', 'stage'])[['average_precision', 'reference_prevalence',
    'retained_reference_recall', 'retained_hits_over_uniform']].mean().reset_index()
infer = csv('rna-inference/analysis/reference-comparison.csv').set_index('reference')

# ---- figures: build.py styling, with text sized for a 6.85 in (174 mm) column ----
PRINT_WIDTH_IN = 6.85
# Data colours: stronger versions of the earlier '#D46F40' orange, '#57616C' grey and '#9DA7B3'
# baseline grey (same OKLCH hue, more chroma or lower lightness). Axes keep the earlier grey.
ORANGE, GREY, BASELINE, AXIS = '#d3580d', '#445160', '#808e9e', '#57616C'
CONNECTOR = '#b3c0ca'  # paired-donor lines; was '#D8DFE5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','svg.hashsalt':'rustscenic-paper',
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
    'text.color':'#24292F','axes.labelcolor':'#24292F','axes.titlecolor':'#24292F',
    'xtick.color':'#24292F','ytick.color':'#24292F','axes.edgecolor':AXIS,
    'axes.linewidth':.8,'grid.color':'#D8DFE5','grid.linewidth':.8,'axes.axisbelow':True})


def text_size(fig, printed_pt):
    """Font size that prints at printed_pt once the figure is scaled to the text width."""
    return printed_pt * fig.get_figwidth() / PRINT_WIDTH_IN


pt = text_size  # the same scaling gives line widths and marker sizes in printed points


def save(fig, name):
    for extension in ['svg','pdf','png']:
        metadata = {'Date':None} if extension=='svg' else (
            {'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
        fig.savefig(FIG/f'{name}.{extension}', dpi=300, facecolor='white',
                    metadata=metadata)
        if extension == 'svg':
            path = FIG/f'{name}.{extension}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


# Figure 1: workflow scaling (12.5 in wide; text 7.2 to 8.8 pt printed)
fig, axes = plt.subplots(2,3,figsize=(12.5,7.9),layout='constrained')
body, head = text_size(fig, 7.2), text_size(fig, 7.8)
# printed: lines 2.0 pt (RustScenic) and 1.6 pt (reference), markers 3.8 pt with 1.25 pt edges
style = {'rustscenic': dict(ls='-', lw=pt(fig, 2.0)), 'reference': dict(ls='--', lw=pt(fig, 1.6))}
marker_style = dict(ms=pt(fig, 3.8), mew=pt(fig, 1.25))
for col, (dataset,title) in enumerate([('atlas','Mouse brain atlas (RNA)'),('bmmc','BMMC (RNA + ATAC)'),('hdma','HDMA (RNA + ATAC)')]):
    subset = runs[runs.dataset.eq(dataset)]
    for row,(metric,label,scale) in enumerate([('input_to_export_s','Analysis + export (min)',60),('peak_pss_gb','Peak sampled PSS (GB)',1)]):
        ax=axes[row,col]
        for method,color,marker in [('rustscenic',ORANGE,'o'),('reference',GREY,'s')]:
            s=subset[subset.method.eq(method)]
            med=s.groupby('cells')[metric].median()/scale
            ax.plot(med.index/1000,med.values,color=color,marker=marker,**style[method],**marker_style)
            ax.scatter(s.cells/1000,s[metric]/scale,color=color,s=pt(fig, 3.3)**2,alpha=.75,linewidths=0)
        ax.set_ylim(bottom=0);ax.set_xlabel('Cells (thousands)',fontsize=body);ax.set_ylabel(label,fontsize=body)
        ax.tick_params(labelsize=body)
        ax.grid(axis='y')
        if row==0:ax.set_title(title,pad=12,fontsize=head)
        ax.set_title('ABCDEF'[row*3+col],loc='left',pad=12,fontsize=head,fontweight='bold')
fig.legend(handles=[Line2D([0],[0],color=ORANGE,marker='o',**style['rustscenic'],**marker_style,label='RustScenic'),
                    Line2D([0],[0],color=GREY,marker='s',**style['reference'],**marker_style,
                           label='pySCENIC (RNA) / SCENIC+ (paired)')],
           loc='outside lower center',ncol=2,frameon=False,fontsize=body)
save(fig,'01-workflow-scaling')

# Figure 4: biological utility (10 in wide; text 7.5 to 9.6 pt printed). Rows become
# reader-facing headers; panels, values and order are unchanged.
fig=plt.figure(figsize=(10,6.4),layout='constrained')
body, head = text_size(fig, 7.5), text_size(fig, 8.2)
rows=['BMMC held-out donors (pre-specified endpoint)','Kang donors, signed activity',
      'Kang donors, separate directions (post hoc)']
subfigs=fig.subfigures(3,1,hspace=.04)
axes=np.array([sub.subplots(1,2) for sub in subfigs])
for sub,letter,label in zip(subfigs,'ABC',rows):
    sub.suptitle(f'{letter}   {label}',x=.01,ha='left',fontsize=head,fontweight='bold')
for col,n in enumerate([5000,20000]):
    for row in [0,1,2]:
        ax=axes[row,col]
        if row==0:
            left=donors[(donors.n==n)&(donors.name=='rust_common')].set_index('donor').macro_f1
            right=donors[(donors.n==n)&(donors.name=='reference_common')].set_index('donor').macro_f1
        elif row==1:
            left=corrected[(corrected.n==n)&(corrected['mode']=='signed')].groupby('donor').macro_f1.mean()
            right=external[(external.training_cells==n)&(external.feature_set=='reference-signed-common')].groupby('donor').macro_f1.mean()
        else:
            selected=directional[directional.training_cells.eq(n)&directional['mode'].eq('directional')]
            left=selected[selected.method.eq('rustscenic')].groupby('donor').macro_f1.mean()
            right=selected[selected.method.eq('reference')].groupby('donor').macro_f1.mean()
        assert set(left.index)==set(right.index)
        for donor in left.index:
            ax.plot([0,1],[left[donor],right[donor]],color=CONNECTOR,lw=pt(fig, 1.6),zorder=1)
        # printed: connectors 1.6 pt, donor points 3.6 pt across with a 1.25 pt edge, mean bars 2.5 pt
        donor_style = dict(s=pt(fig, 3.6)**2, linewidths=pt(fig, 1.25), zorder=2)
        ax.scatter(np.zeros(len(left)),left,color=ORANGE,**donor_style)
        ax.scatter(np.ones(len(right)),right,color=GREY,**donor_style)
        for x,values,color in [(0,left,ORANGE),(1,right,GREY)]:
            ax.plot([x-.10,x+.10],[values.mean()]*2,color=color,lw=pt(fig, 2.5))
            ax.text(x, .90, f'Mean {values.mean():.3f}',ha='center',fontsize=body,color=color,transform=ax.get_xaxis_transform())
        ax.set(xticks=[0,1],xticklabels=['RustScenic','SCENIC+ networks'],ylim=(.2,.9),xlim=(-.35,1.35))
        ax.set_ylabel('Macro-F1',fontsize=body)
        ax.tick_params(labelsize=body)
        ax.set_title(f'{n:,}-cell training networks',pad=10,fontsize=body,fontweight='normal')
        ax.grid(axis='y')
save(fig,'02-biological-utility')

# Figure 5: experimental-reference recovery (11 in wide; text 7.0 to 9.4 pt printed)
labels={'mESC-ChIP-seq-network.tsv':'Mouse ESC: ChIP','mESC-lofgof-network.tsv':'Mouse ESC: perturbation',
        'hESC-ChIP-seq-network.tsv':'Human ESC: ChIP','mDC-ChIP-seq-network.tsv':'Mouse dendritic: ChIP'}
fig,axes=plt.subplots(2,2,figsize=(11,5.6),layout='constrained')
fig.get_layout_engine().set(hspace=.06)
body, head = text_size(fig, 7.0), text_size(fig, 7.8)
for i,(ax,(reference,title)) in enumerate(zip(axes.flat,labels.items())):
    sub=rna[rna.reference.eq(reference)].set_index(['method','stage'])
    values=[infer.loc[reference,'rustscenic'],infer.loc[reference,'arboreto_mp_sparse'],
        infer.loc[reference,'absolute_pearson'],sub.loc[('rustscenic','motif_pruned'),'average_precision'],
        sub.loc[('reference','motif_pruned'),'average_precision']]
    ax.bar(range(5),values,color=[ORANGE,GREY,BASELINE,ORANGE,GREY])
    # printed 1.6 pt, with a thin white halo so the dotted line reads across the darker bars
    ax.axhline(sub.reference_prevalence.iloc[0],color='#24292F',ls=':',lw=pt(fig, 1.6),label='Reference prevalence',
               path_effects=[pe.Stroke(linewidth=pt(fig, 2.0), foreground='white'), pe.Normal()])
    ax.set(xticks=range(5),xticklabels=['RustScenic\nGRN','arboreto\nGRN','Absolute\nPearson','RustScenic\npruned','pySCENIC\npruned'])
    if i%2==0:ax.set_ylabel('Average precision',fontsize=body)
    ax.tick_params(labelsize=body)
    ax.set_title(title,pad=10,fontsize=head);ax.set_ylim(0,max(values)*1.22);ax.grid(axis='y')
    ax.set_title('ABCD'[i],loc='left',pad=10,fontsize=head,fontweight='bold')
fig.legend(handles=[Line2D([0],[0],color='#24292F',ls=':',lw=pt(fig, 1.6),label='Reference prevalence')],
           loc='outside lower center',frameon=False,fontsize=body)
save(fig,'03-rna-reference-recovery')
print('written', ', '.join(f'{n}.{{svg,pdf,png}}' for n in
      ['01-workflow-scaling','02-biological-utility','03-rna-reference-recovery']), 'in', FIG)
