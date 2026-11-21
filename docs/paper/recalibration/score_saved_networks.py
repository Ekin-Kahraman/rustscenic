"""Score frozen networks after, and only after, the replacement calibration gate."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
from corrected_evaluation import calibrated_labels,bh_adjust
from analysis_paths import BASE as B,PACKAGE as P,RETURNED as R
sys.path.insert(0,str(P));from metrics import score_rank,expected_hits
from evaluate import verify_predictions
rows=[];coverage=[];matched=[];label_summary=[];power_summary=[]
annotations=pd.read_csv(P/'resources/human-gene-annotations.tbl',sep='\t')
annotated=set(annotations.loc[(annotations.motif_similarity_qvalue<=.001)&(annotations.orthologous_identity>=0),'gene_name'])
for dataset in ['k562','rpe1']:
 results=B/'results'/dataset;assert json.loads((results/'CALIBRATION-STATUS.json').read_text())['status']=='passed';assert json.loads((results/'DISCOVERY-STATUS.json').read_text())['status']=='complete'
 d=pd.read_csv(results/'discovery.csv');panel=json.loads((P/f'inputs/{dataset}/panel.json').read_text());genes=np.array(panel['genes']);targets=panel['target_tfs'];primary=set((B/f'inputs/{dataset}/primary-tfs.txt').read_text().splitlines())
 assert set(d.grna_target)==set(targets);assert set(d.response_id)<=set(genes);assert len(d)==len(targets)*len(genes)-len(set(targets)&set(genes))
 power=pd.read_csv(results/'positive-controls.csv');valid=power.pass_qc & np.isfinite(power.p_value);power['q_value']=np.nan;power.loc[valid,'q_value']=bh_adjust(power.loc[valid,'p_value']);power.to_csv(results/'positive-controls-with-q.csv',index=False)
 kd=set(power.loc[valid & power.q_value.le(.05)&power.log_2_fold_change.le(-.25),'grna_target'])
 power_summary.append(dict(dataset=dataset,assayed_targets=len(targets),self_in_panel=len(power),qc_passing_finite=int(valid.sum()),significant_down=int((valid&power.q_value.le(.05)&power.log_2_fold_change.lt(0)).sum()),adequate_knockdown=len(kd),median_log2_effect=float(power.loc[valid,'log_2_fold_change'].median())))
 labels={threshold:calibrated_labels(d,threshold) for threshold in [.25,.5,1.]}
 labels[.5].to_parquet(results/'corrected-pair-results.parquet',index=False)
 for threshold,l in labels.items():
  for roster in ['primary','all']:
   a=l[l.grna_target.isin(primary)] if roster=='primary' else l
   bytf=a.groupby('grna_target').agg(eligible=('eligible','sum'),responders=('response','sum'))
   label_summary.append(dict(dataset=dataset,roster=roster,threshold=threshold,targets=len(bytf),planned_pairs=len(a),qc_passing_pairs=int(a.pass_qc.sum()),finite_tested_pairs=int((a.pass_qc&np.isfinite(a.p_value)).sum()),eligible_pairs=int(a.eligible.sum()),significance_only_pairs=int(a.q_value.le(.05).sum()),response_pairs=int(a.response.sum()),response_up=int(a.response_direction.gt(0).sum()),response_down=int(a.response_direction.lt(0).sum()),TFs_with_response=int(bytf.responders.gt(0).sum()),TFs_rank_evaluable=int(((bytf.responders>0)&(bytf.responders<bytf.eligible)).sum()),TFs_without_tested_pairs=int(bytf.eligible.eq(0).sum()),median_eligible_genes=float(bytf.eligible.median())))
 for seed in [777,778,779]:
  job=R/f'jobs/{dataset}-{seed}';verify_predictions(P/'inputs'/dataset,job,dataset,seed)
  native=pd.read_parquet(job/'native/network.parquet');ref=pd.read_parquet(job/'reference/network.parquet')
  for frame in [native,ref]:
   assert not frame[['TF','target']].duplicated().any() and np.isfinite(frame.importance).all() and frame.importance.ge(0).all()
  raw={name:{tf:g.set_index('target').importance.reindex(genes,fill_value=0).to_numpy() for tf,g in frame.groupby('TF')} for name,frame in [('native_raw',native),('reference_raw',ref)]}
  regs={name:json.loads((job/path).read_text()) for name,path in [('original','native/original.json'),('diverse_importance','native/diverse_importance.json'),('reference','reference/reference.json')]}
  with np.load(job/'native/pearson.npz') as pearson:
   assert np.array_equal(pearson['genes'],genes) and np.array_equal(pearson['TFs'],targets) and np.isfinite(pearson['rho']).all();rhos={tf:rho for tf,rho in zip(pearson['TFs'],pearson['rho'])}
  for threshold,l in labels.items():
   for tf,g in l.groupby('grna_target',sort=True):
    g=g.set_index('response_id').reindex(genes);sel=g.eligible.fillna(False).to_numpy(bool);y=g.response.fillna(False).to_numpy(bool)[sel]
    scores={};degrees={};base={'dataset':dataset,'seed':seed,'TF':tf,'threshold':threshold,'primary_roster':tf in primary,'adequate_knockdown':tf in kd,'in_training_predictors':tf in panel['tfs'],'motif_annotation_present':tf in annotated,'eligible_genes':int(sel.sum()),'response_genes':int(y.sum())}
    for method in regs:
     r=raw['reference_raw' if method=='reference' else 'native_raw'].get(tf,np.zeros(len(genes)))[sel]
     members=set(regs[method].get(tf+'_activator',[]))|set(regs[method].get(tf+'_repressor',[]));f=np.isin(genes[sel],list(members-{tf}));low=r[r>0].min()/2 if (r>0).any() else 1.
     scores[method]=np.where(f,np.where(r>0,r,low),0);degrees[method]=int(f.sum())
     emitted=scores[method]>0
     coverage.append({**base,'method':method,'edges':int(f.sum()),'emitted_predictions50':min(50,int(emitted.sum())),'emitted_hits50':expected_hits(y[emitted],scores[method][emitted],50),'status':'evaluable' if 0<y.sum()<len(y) else 'no_tested_pairs' if not len(y) else 'no_label_variation'})
    scores.update({method:raw[method].get(tf,np.zeros(len(genes)))[sel] for method in raw});scores['full_pearson']=abs(rhos[tf][sel]);scores['uniform']=np.zeros(sel.sum())
    for method,score in scores.items():
     m=score_rank(y,score,degrees.get(method,0))
     if m:rows.append({**base,'method':method,**m})
    k=min(50,degrees['original'],degrees['diverse_importance'])
    if k and 0<y.sum()<len(y):
     a=scores['original'];b=scores['diverse_importance'];matched.append({**base,'budget':k,'original_hits':expected_hits(y[a>0],a[a>0],k),'diverse_hits':expected_hits(y[b>0],b[b>0],k)})
  print('scored',dataset,seed,flush=True)
summary=B/'summary';summary.mkdir(exist_ok=True)
f=pd.DataFrame(rows);f.to_csv(summary/'rankings-per-TF.csv',index=False);pd.DataFrame(coverage).to_csv(summary/'coverage-per-TF.csv',index=False);pd.DataFrame(matched).to_csv(summary/'matched-budget-per-TF.csv',index=False);pd.DataFrame(label_summary).to_csv(summary/'label-coverage.csv',index=False);pd.DataFrame(power_summary).to_csv(summary/'positive-control-summary.csv',index=False)
macro=[];deltas=[]
for roster in ['primary','all','primary_adequate_knockdown']:
 a=f if roster=='all' else f[f.primary_roster]
 if roster.endswith('adequate_knockdown'):a=a[a.adequate_knockdown]
 for (dataset,threshold,method),g in a.groupby(['dataset','threshold','method']):
  t=g.groupby('TF').mean(numeric_only=True)
  macro.append(dict(dataset=dataset,roster=roster,threshold=threshold,method=method,TFs=len(t),average_precision=t.average_precision.mean(),response_prevalence=t.response_prevalence.mean(),expected_hits50=t.expected_hits_at_50.mean(),emitted_hits50=t.emitted_hits_at_50.mean(),predictions50=t.predictions_at_50.mean(),pooled_emitted_precision=g.emitted_hits_at_50.sum()/g.predictions_at_50.sum() if g.predictions_at_50.sum() else np.nan))
 for (dataset,threshold),g in a.groupby(['dataset','threshold']):
  wide=g.groupby(['TF','method']).average_precision.mean().unstack('method')
  for method,baseline in [('diverse_importance','original'),('diverse_importance','reference'),('diverse_importance','uniform'),('diverse_importance','full_pearson'),('original','reference'),('original','uniform'),('original','full_pearson'),('native_raw','reference_raw')]:
   delta=(wide[method]-wide[baseline]).dropna().to_numpy();rng=np.random.default_rng(20261112);boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
   deltas.append(dict(dataset=dataset,roster=roster,threshold=threshold,comparison=f'{method} minus {baseline}',TFs=len(delta),mean_AP_difference=delta.mean(),TF_bootstrap_low=lo,TF_bootstrap_high=hi,scope='Post-hoc re-analysis; conditional assayed-TF variation, not donor replication'))
pd.DataFrame(macro).to_csv(summary/'macro-metrics.csv',index=False);pd.DataFrame(deltas).to_csv(summary/'paired-differences.csv',index=False)
(B/'SCORING-STATUS.json').write_text(json.dumps({'status':'complete','ranking_rows':len(rows),'coverage_rows':len(coverage),'network_refits':0,'recipe_changes':0,'scope':'Post-hoc calibrated response re-analysis; labels include indirect responses and incomplete power'},indent=2)+'\n')
print('SAVED NETWORK SCORING COMPLETE',flush=True)
