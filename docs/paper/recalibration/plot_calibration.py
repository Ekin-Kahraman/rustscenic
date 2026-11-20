from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;out=B/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(10,7.4));stats=[]
for row,dataset in enumerate(['k562','rpe1']):
 r=B/'results'/dataset;assert json.loads((r/'CALIBRATION-STATUS.json').read_text())['status']=='passed'
 frames=[]
 for run in ['median-a','validation','single','upper']:
  f=pd.read_csv(r/f'calibration-{run}.csv');f['run']=run;frames.append(f)
 allf=pd.concat(frames,ignore_index=True);assert not allf[['grna_target','response_id']].duplicated().any();n=len(allf)
 for run,color in [('median-a','#3267A5'),('validation','#C16A32')]:
  p=np.sort(allf.loc[allf.run.eq(run),'p_value'].to_numpy());m=len(p);i=np.arange(1,m+1);expected=(i-.5)/m
  idx=np.unique(np.r_[np.arange(min(100,m)),np.geomspace(1,m,800).astype(int)-1]);axes[row,0].scatter(-np.log10(expected[idx]),-np.log10(p[idx]),s=7,alpha=.65,label='Initial 10,000' if run=='median-a' else 'Distinct 10,000',color=color,rasterized=True)
 lim=4.4;axes[row,0].plot([0,lim],[0,lim],color='#777777',linestyle='--',linewidth=1);axes[row,0].set(xlim=(0,lim),ylim=(0,lim),xlabel='Expected −log₁₀(p)',ylabel='Observed −log₁₀(p)',title=f'{dataset.upper()} · control p-values');axes[row,0].legend(frameon=False,fontsize=8,loc='upper left')
 effects=allf.log_2_fold_change.to_numpy();axes[row,1].hist(effects,bins=np.linspace(-1,1,61),color='#3267A5',alpha=.8);axes[row,1].axvline(0,color='#444444',linestyle='--',linewidth=1);axes[row,1].set(xlabel='Estimated log₂ fold change',ylabel='Control pairs',title=f'{dataset.upper()} · {n:,} distinct control pairs');axes[row,1].text(.97,.95,f'{int(allf.significant.sum())} discoveries after BH at 5%\n'+f'Median effect {np.median(effects):+.3f}',transform=axes[row,1].transAxes,ha='right',va='top',fontsize=9)
 axes[row,1].text(.97,.79,f'{int((abs(effects)>1).sum())} effects outside plotted range',transform=axes[row,1].transAxes,ha='right',va='top',fontsize=7,color='#555555')
 stats.append({'dataset':dataset,'distinct_pairs':n,'raw_p05_fraction':float(np.mean(allf.p_value<=.05)),'median_effect':float(np.median(effects)),'effects_outside_histogram':int((abs(effects)>1).sum())})
fig.suptitle('Corrected evaluation passes the prespecified negative-control checks',fontsize=13,y=.98)
fig.text(.5,.02,'Post-hoc evaluation repair · shared cells and constructs · empirical calibration, not independent biological replication',ha='center',fontsize=8,color='#555555')
fig.tight_layout(rect=(0,.045,1,.945))
for ext in ['png','pdf','svg']:fig.savefig(out/f'calibration.{ext}',dpi=200)
(B/'CALIBRATION-FIGURE-RECEIPT.json').write_text(json.dumps(stats,indent=2)+'\n')
print(stats)
