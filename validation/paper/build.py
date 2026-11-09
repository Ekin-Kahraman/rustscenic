"""Rebuild paper tables and figures from audited, immutable evidence. No model fits.

Run from any directory: python /path/to/validation/paper/build.py
Requires NumPy, pandas and matplotlib. Source hashes and numerical summaries
are checked before outputs are written. Raw HPC audit remains a separate tier.
"""
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'docs/paper'
E = PAPER / 'evidence'
OUT = PAPER / 'tables'
FIG = PAPER / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
load = lambda name: json.loads((E / name).read_text())
csv = lambda name: pd.read_csv(E / name)
manifest = load('manifest.json')
for f in manifest['files']:
    data = (E / f['path']).read_bytes()
    assert len(data) == f['bytes'] and hashlib.sha256(data).hexdigest() == f['sha256'], f['path']

# Keep complete method pairs from the same allocation. Reproduce the existing
# ratio-of-medians headline and add paired ratios as an explicit sensitivity.
runs = csv('scaling/consolidated-runs.csv')
assert len(runs) == 42 and runs.status.eq('complete').all()
perf = []
for (dataset, cells), group in runs.groupby(['dataset', 'cells'], sort=True):
    a = group[group.method.eq('rustscenic')].set_index('seed').sort_index()
    b = group[group.method.eq('reference')].set_index('seed').sort_index()
    assert a.index.tolist() == b.index.tolist() == [777, 778, 779]
    for key in ['host', 'job_id', 'source', 'threads']:
        assert a[key].equals(b[key]), (dataset, cells, key)
    assert a.threads.eq(16).all()
    rt, bt = a.input_to_export_s.median()/60, b.input_to_export_s.median()/60
    rm, bm = a.peak_pss_gb.median(), b.peak_pss_gb.median()
    perf.append(dict(dataset=dataset, cells=cells,
        comparator='pySCENIC' if dataset == 'atlas' else 'SCENIC+',
        rust_minutes=rt, reference_minutes=bt, speedup=bt/rt,
        rust_pss_gb=rm, reference_pss_gb=bm, memory_reduction_percent=100*(1-rm/bm),
        paired_median_speedup=(b.input_to_export_s/a.input_to_export_s).median(),
        paired_median_memory_reduction_percent=100*(1-a.peak_pss_gb/b.peak_pss_gb).median()))
perf = pd.DataFrame(perf)
reported = csv('scaling/consolidated-performance.csv')
np.testing.assert_allclose(perf.speedup, reported.speedup, rtol=1e-12)
np.testing.assert_allclose(perf.memory_reduction_percent, reported.memory_reduction_percent, rtol=1e-12)
perf.to_csv(OUT/'workflow-performance.csv', index=False)
runs.to_csv(OUT/'workflow-runs.csv', index=False)
csv('scaling/stage-timings.csv').to_csv(OUT/'workflow-stage-timings.csv', index=False)

# Retain prespecified BMMC union endpoints and the post hoc signed diagnostic.
folds = csv('biology/all-utility-folds.csv')
primary = folds[folds.group.eq('primary')]
donors = primary.groupby(['n', 'name', 'donor']).macro_f1.mean().reset_index()
original = donors.groupby(['n', 'name']).macro_f1.mean().unstack()
reported_bio = load('biology/returned/biological-summary.json')['holdout_contrasts']
for n in [5000, 20000]:
    for name, side in [('rust_common', 'left_mean'), ('reference_common', 'right_mean')]:
        assert np.isclose(original.loc[n,name], reported_bio[str(n)]['common']['macro_f1'][side], atol=1e-12)
original.to_csv(OUT/'bmmc-original-utility.csv')
diagnostic = csv('signed-diagnostic/paired-common-tf-comparison.csv')
signed = diagnostic.groupby(['training_cells','method','donor'])[['union_f1','signed_f1']].mean()
signed.groupby(['training_cells','method']).mean().to_csv(OUT/'bmmc-signed-diagnostic.csv')

# Recover every corrected donor metric from the actual confusion matrices.
verification = load('scoring/returned/biology/seeded/verification.json')
corrected = []
for record in verification['records']:
    for f in record['result']['folds']:
        m = np.asarray(f['confusion_matrix'], dtype=float)
        truth = m.sum(1)
        present = truth > 0
        denom = m.sum(0) + truth
        f1 = np.divide(2*np.diag(m), denom, out=np.zeros(len(m)), where=denom>0)[present].mean()
        assert f['converged'] and np.isclose(f1, f['macro_f1'], atol=1e-12)
        corrected.append(dict(n=record['training_cells'], seed=record['inference_seed'],
                              mode=record['mode'], donor=f['donor'], macro_f1=f1))
corrected = pd.DataFrame(corrected)
assert len(corrected) == 96
corrected.to_csv(OUT/'kang-corrected-folds.csv', index=False)
external = csv('external/returned/biology-folds.csv')
external_means = external.groupby(['training_cells','feature_set','donor']).macro_f1.mean().groupby(level=[0,1]).mean().unstack()
external_means.to_csv(OUT/'kang-original-endpoints-and-controls.csv')
final = load('scoring/verified-results.json')
bio = []
for n in [5000,20000]:
    mean = corrected[corrected.n.eq(n)&corrected['mode'].eq('signed')].groupby('donor').macro_f1.mean().mean()
    result = final['biology'][str(n)]['reference_network']['macro_f1']
    assert np.isclose(mean, result['left_mean'], atol=1e-12)
    bio.append(dict(training_cells=n, rust_native_signed=mean,
        rust_reference_signed=external_means.loc[n,'rustscenic-signed-common'],
        reference_signed=result['right_mean'], difference=result['difference'],
        ci_low=result['paired_donor_bootstrap_ci95'][0],ci_high=result['paired_donor_bootstrap_ci95'][1]))
bio = pd.DataFrame(bio)
bio.to_csv(OUT/'kang-final-comparison.csv', index=False)

ref = csv('biology/returned/rna-reference-recovery.csv')
rna = ref.groupby(['reference','method','stage'])[['average_precision','reference_prevalence',
    'retained_reference_recall','retained_hits_over_uniform']].mean().reset_index()
rna.to_csv(OUT/'rna-reference-recovery.csv', index=False)
scoring = csv('scoring/returned/scoring-performance.csv')
scoring.to_csv(OUT/'scoring-confirmation-runs.csv', index=False)

# Independent directional confirmation; preserve all earlier endpoints above.
directional_records = load('directional/returned/biology/verification.json')['records']
directional_rows = []
for record in directional_records:
    assert record['passed'] and record['max_absolute_error'] == 0
    for mode, result in record['results'].items():
        for fold in result['folds']:
            m = np.asarray(fold['confusion_matrix'], dtype=float)
            support, predicted = m.sum(1), m.sum(0)
            present = support > 0
            f1 = np.divide(2*m.diagonal(), support+predicted,
                           out=np.zeros(len(m)), where=support+predicted>0)[present].mean()
            recall = np.divide(m.diagonal(), support, out=np.zeros(len(m)), where=present)[present].mean()
            assert fold['converged'] and fold['status'] == 'complete'
            np.testing.assert_allclose([f1, recall], [fold['macro_f1'], fold['balanced_accuracy']], rtol=0, atol=1e-12)
            directional_rows.append(dict(training_cells=record['training_cells'], seed=record['seed'],
                method=record['method'], mode=mode, donor=fold['donor'], macro_f1=f1, balanced_accuracy=recall))
directional = pd.DataFrame(directional_rows)
assert len(directional) == 192
directional.to_csv(OUT/'directional-confirmation-folds.csv', index=False)
directional_summary = load('directional/returned/biological-summary.json')
directional_bio = []
for n in [5000, 20000]:
    means = directional[directional.training_cells.eq(n)].groupby(['method','mode','donor'])[['macro_f1','balanced_accuracy']].mean()
    for comparison, method, mode in [('directional_vs_contrast','rustscenic','contrast'), ('directional_vs_scenicplus','reference','directional')]:
        for metric in ['macro_f1','balanced_accuracy']:
            left = means.loc[('rustscenic','directional'),metric].sort_index()
            right = means.loc[(method,mode),metric].sort_index()
            assert left.index.equals(right.index) and len(left) == 8
            delta = (left-right).to_numpy()
            draws = np.random.default_rng(20261106).integers(0,8,size=(10000,8))
            interval = np.quantile(delta[draws].mean(axis=1),[.025,.975])
            expected = directional_summary[str(n)][comparison][metric]
            np.testing.assert_allclose([left.mean(),right.mean(),delta.mean(),*interval],
                [expected['left_mean'],expected['right_mean'],expected['difference'],*expected['paired_donor_bootstrap_ci95']],rtol=0,atol=1e-12)
            directional_bio.append(dict(training_cells=n,comparison=comparison,metric=metric,
                rust_directional=left.mean(),comparator=right.mean(),difference=delta.mean(),ci_low=interval[0],ci_high=interval[1]))
directional_bio = pd.DataFrame(directional_bio)
directional_bio.to_csv(OUT/'directional-confirmation-comparisons.csv',index=False)
directional_perf = csv('directional/returned/scoring-performance.csv')
assert len(directional_perf)==6 and directional_perf.host.nunique()==1
assert directional_perf.cells.eq(817740).all() and directional_perf.genes.eq(5000).all()
directional_cost = load('directional/returned/SUMMARY.json')
for key, expected in directional_cost['median_ratios'].items():
    values = directional_perf.pivot(index='seed',columns='arm',values=key.removesuffix('_ratio'))
    assert np.isclose((values.directional/values.union).median(), expected, rtol=0, atol=1e-12)
directional_perf.to_csv(OUT/'directional-confirmation-performance.csv',index=False)

ORANGE, GREY = '#D46F40', '#57616C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','svg.hashsalt':'rustscenic-paper',
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
    'text.color':'#24292F','axes.labelcolor':'#24292F','axes.titlecolor':'#24292F',
    'xtick.color':'#24292F','ytick.color':'#24292F','axes.edgecolor':GREY,
    'axes.linewidth':.8,'grid.color':'#D8DFE5','grid.linewidth':.8,'axes.axisbelow':True})
def save(fig, name):
    for extension in ['svg','pdf','png']:
        metadata = {'Date':None} if extension=='svg' else (
            {'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
        fig.savefig(FIG/f'{name}.{extension}', dpi=180, facecolor='white',
                    metadata=metadata)
        if extension == 'svg':
            path = FIG/f'{name}.{extension}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)

fig, axes = plt.subplots(2,3,figsize=(12.5,7.5),layout='constrained')
for col, (dataset,title) in enumerate([('atlas','Mouse-brain RNA'),('bmmc','Bone-marrow RNA + ATAC'),('hdma','Developmental RNA + ATAC')]):
    subset = runs[runs.dataset.eq(dataset)]
    for row,(metric,label,scale) in enumerate([('input_to_export_s','Analysis + export (min)',60),('peak_pss_gb','Peak sampled PSS (GB)',1)]):
        ax=axes[row,col]
        for method,color,marker in [('rustscenic',ORANGE,'o'),('reference',GREY,'s')]:
            s=subset[subset.method.eq(method)]
            med=s.groupby('cells')[metric].median()/scale
            ax.plot(med.index/1000,med.values,color=color,marker=marker,
                    ls='-' if method=='rustscenic' else '--',lw=2.8 if method=='rustscenic' else 1.8)
            ax.scatter(s.cells/1000,s[metric]/scale,color=color,s=15,alpha=.5)
        ax.set_ylim(bottom=0);ax.set_xlabel('Cells (thousands)');ax.set_ylabel(label)
        ax.grid(axis='y')
        if row==0:ax.set_title(title,pad=12)
fig.suptitle('Configured workflow scaling on real cells',fontsize=16)
fig.legend(handles=[Line2D([0],[0],color=ORANGE,marker='o',label='RustScenic'),
                    Line2D([0],[0],color=GREY,marker='s',ls='--',label='pySCENIC (RNA) / SCENIC+ (paired)')],
           loc='outside lower center',ncol=2,frameon=False)
save(fig,'01-workflow-scaling')

fig,axes=plt.subplots(3,2,figsize=(10,10),layout='constrained')
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
            ax.plot([0,1],[left[donor],right[donor]],color='#D8DFE5',lw=1,zorder=1)
        ax.scatter(np.zeros(len(left)),left,color=ORANGE,s=22,zorder=2)
        ax.scatter(np.ones(len(right)),right,color=GREY,s=22,zorder=2)
        for x,values,color in [(0,left,ORANGE),(1,right,GREY)]:
            ax.plot([x-.10,x+.10],[values.mean()]*2,color=color,lw=3)
            ax.text(x, .94, f'Mean {values.mean():.3f}',ha='center',fontsize=10,color=color)
        ax.set(xticks=[0,1],xticklabels=['RustScenic','SCENIC+ networks'],ylim=(0,1),xlim=(-.35,1.35),ylabel='Cell-type macro-F1')
        ax.set_title(f'{n:,}-cell training networks',pad=10)
        ax.grid(axis='y')
axes[0,0].set_ylabel('BMMC: original union\nCell-type macro-F1')
axes[1,0].set_ylabel('Kang: corrected signed\nCell-type macro-F1')
axes[2,0].set_ylabel('Kang: separate directions\nCell-type macro-F1')
fig.suptitle('Cell-type prediction from matched TF signatures',fontsize=16)
save(fig,'02-biological-utility')

infer=csv('rna-inference/analysis/reference-comparison.csv').set_index('reference')
labels={'mESC-ChIP-seq-network.tsv':'Mouse ESC: ChIP','mESC-lofgof-network.tsv':'Mouse ESC: perturbation',
        'hESC-ChIP-seq-network.tsv':'Human ESC: ChIP','mDC-ChIP-seq-network.tsv':'Mouse dendritic: ChIP'}
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,(reference,title) in zip(axes.flat,labels.items()):
    sub=rna[rna.reference.eq(reference)].set_index(['method','stage'])
    values=[infer.loc[reference,'rustscenic'],infer.loc[reference,'arboreto_mp_sparse'],
        infer.loc[reference,'absolute_pearson'],sub.loc[('rustscenic','motif_pruned'),'average_precision'],
        sub.loc[('reference','motif_pruned'),'average_precision']]
    ax.bar(range(5),values,color=[ORANGE,GREY,'#9DA7B3',ORANGE,GREY])
    ax.axhline(sub.reference_prevalence.iloc[0],color='#24292F',ls=':',lw=1.4,label='Reference prevalence')
    ax.set(xticks=range(5),xticklabels=['Rust\nGRN','arboreto\nGRN','Absolute\nPearson','Rust\npruned','pySCENIC\npruned'],ylabel='Average precision')
    ax.set_title(title,pad=10);ax.set_ylim(0,max(values)*1.22);ax.grid(axis='y')
    ax.legend(frameon=False,fontsize=8,loc='upper right')
fig.suptitle('Experimental-reference recovery, including simple baselines',fontsize=16)
save(fig,'03-rna-reference-recovery')

def table(frame, decimals=3):
    def fmt(v):
        return f'{v:.{decimals}f}' if isinstance(v,(float,np.floating)) else str(v)
    rows=[list(frame.columns)]+[[fmt(v) for v in row] for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+
                     ['| '+' | '.join(row)+' |' for row in rows[1:]])

content='''# Consolidated results

Generated from checksummed evidence by `validation/paper/build.py`. All values
are scoped to their original experiment; different measurements are not pooled.

## Configured workflows

21 complete same-allocation pairs, seven sizes, three seeds each, 16 CPUs.
Times include prepared-input loading, analysis and standard export. Shared
preparation and raw-data processing are excluded. Native inference, pruning,
topics and eRegulon algorithms differ. These are workflow gains, not a pure
Rust language speedup. Headline ratios divide method medians; paired-ratio
sensitivities and all 42 runs are retained in the CSV tables.

'''+table(perf[['dataset','cells','rust_minutes','reference_minutes','speedup','rust_pss_gb','reference_pss_gb','memory_reduction_percent']])+'''

## Biological utility

BMMC original common-TF union endpoint, 10,000 cells from three unseen donors.
Algorithm seeds are averaged within each donor before averaging donors.
The expression baseline uses 50 training-fitted principal components.

'''+table(original[['rust_common','reference_common','rust_native','reference_native','expression_pca50']].reset_index())+'''

The historical columns `rust_native`/`reference_native` mean all TF signatures
with the common scorer, not native scoring. They have unequal feature counts
and cannot establish better networks. The signed BMMC diagnostic is post hoc
and is separately retained in `bmmc-signed-diagnostic.csv`.

Kang external cohort: 12,315 cells, eight donors, frozen BMMC networks. Classifiers
leave one external donor out; networks never fit those donors. Corrected native
signed scores are shown below. The tie fix reuses this inspected cohort and is
a numerical repair, not a fresh biological confirmation. Intervals are paired
donor bootstrap intervals; no noninferiority margin was prespecified.

'''+table(bio)+'''

The 20,000-cell old native signed endpoint was 0.682918; the corrected endpoint
is 0.692892. Its observed improvement interval includes zero. The remaining
gap to SCENIC+ networks for this contrast endpoint is negative at both training
sizes. The separate-direction confirmation below retains more activity
information. Neither metric is the fraction of correct regulatory edges.

## Directional activity: independent HPC confirmation, 9 November

All 192 donor-fold metrics and paired donor-bootstrap intervals reproduce.
The fixed representation retains positive and negative target activity as
separate features. Networks and cohorts are reused; this is independent
execution, not untouched biological validation. Both network sources use the
same native scorer, common TF roster and donor-aware classifier.

'''+table(directional_bio,6)+'''

At 20,000 network-training cells, mean macro-F1 is 0.712332 versus 0.711258
for SCENIC+ networks. The average macro-F1 deficit is closed; the interval
does not establish superiority or formal equivalence. Mean cell-type recall
remains 0.782637 versus 0.797620. All original endpoints and adverse class
results are retained, including weak experimental-reference recovery below.

On 817,740 cells, median paired overhead versus union activity is 2.62% for
scoring, 0.29% including I/O, and 4.52% for peak process-tree PSS. One seed has
15.94% higher PSS. Raw activity storage rises 34.33%; full HDF5 files rise
about 27.05%. This is scoring-stage cost evidence, not a new full-workflow
speedup. Full arrays remain on HPC with hashes; returned samples match prior
numerical controls exactly. See the [confirmation audit](evidence/directional/AUDIT.txt).

## RNA reference recovery

Mean of three algorithm seeds. Unlisted edges are unlabelled, not established
negatives. Missing motif-pruned edges keep zero scores in the same candidate
universe. ChIP binding and perturbation response do not prove direct causality.

'''+table(rna[rna.stage.eq('motif_pruned')][['reference','method','average_precision','reference_prevalence','retained_reference_recall']])+'''

Absolute Pearson exceeds both original GRN methods on all four references.
Pruned recovery remains weak in absolute terms; retain coverage and controls.

## Scoring confirmation

18 completed tasks; 195 verified return hashes; 96 recomputed donor folds.
Corrected native/reference maximum absolute error: 3.301e-8 on all 12 external
matrices and 3.122e-8 on the returned 10,000-cell sample per atlas network.
Full execution used 817,740 cells, but reference numerical inspection there
is sampled. Complete outputs remain on HPC with recorded hashes.

Seeded correction versus the same candidate build without the seed: median
paired scoring-time ratio 1.004175, PSS ratio 1.034812. Against v0.5.0, those
ratios are 0.928426 and 0.908883. Small three-run differences are descriptive;
they do not revise the full-workflow performance figures above.

See METHODS.md for endpoint chronology, controls, parameter differences and
figure captions. Full retained HPC output arrays still require durable archival.
'''
(PAPER/'RESULTS.md').write_text(content)
output_files=sorted(list(OUT.glob('*.csv'))+list(FIG.glob('*'))+[PAPER/'RESULTS.md'])
report={'status':'passed','source_files_verified':len(manifest['files']),
        'selected_runs':len(runs),'matched_pairs':21,'corrected_confusion_matrices_verified':len(corrected),
        'directional_confusion_matrices_verified':len(directional),'directional_intervals_reproduced':True,
        'source_statistics_recomputed':True,'model_fits_performed':0,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,
                       'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
        'outputs':[{'path':str(p.relative_to(PAPER)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in output_files]}
(PAPER/'build-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='outputs'},indent=2))
