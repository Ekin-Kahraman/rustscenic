"""Export exact frozen evaluation cells as raw counts for the pinned R evaluator."""
from pathlib import Path
import gzip,json,hashlib
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from analysis_paths import BASE as B,PACKAGE as P,RETURNED as R
S=P.parent/'source-cache'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(8*1024**2),b''):h.update(v)
 return h.hexdigest()
def write_bin(path,values):
 with gzip.open(path,'wb',compresslevel=1) as f:f.write(values.tobytes())
for dataset in ['k562','rpe1']:
 inputs=P/'inputs'/dataset;out=B/'inputs'/dataset;out.mkdir(parents=True,exist_ok=True)
 prov=json.loads((inputs/'provenance.json').read_text());source=S/prov['source_name'];assert sha(source)==prov['source_sha256']
 panel=json.loads((inputs/'panel.json').read_text());cells=pd.read_csv(inputs/'evaluation-cells.tsv.gz',sep='\t');groups=pd.read_csv(inputs/'evaluation-groups.tsv',sep='\t')
 data=ad.read_h5ad(source,backed='r');idx=data.obs_names.get_indexer(cells.barcode);cols=data.var_names.get_indexer(panel['genes']);assert (idx>=0).all() and (cols>=0).all()
 obs=data.obs.iloc[idx];assert np.array_equal(obs.batch.astype(int),cells.batch);assert np.array_equal(obs.guide_id.astype(str),cells.construct);assert np.array_equal(obs.perturbation.astype(str),cells.target)
 assert not set(cells.barcode)&set(pd.read_csv(inputs/'training-cells.tsv',sep='\t').barcode)
 aggregate=np.zeros((len(groups),len(data.var_names)),dtype=np.uint32);blocks=[];totals=[];detects=[]
 for start in range(0,len(cells),512):
  subset=idx[start:start+512];order=np.argsort(subset);raw=np.asarray(data.X[subset[order]])[np.argsort(order)]
  assert np.isfinite(raw).all() and (raw>=0).all() and (raw==np.floor(raw)).all()
  raw=raw.astype(np.uint32);total=raw.sum(1,dtype=np.uint64);assert np.array_equal(total,obs.iloc[start:start+len(raw)].ncounts)
  np.add.at(aggregate,cells.group.iloc[start:start+len(raw)].to_numpy(),raw)
  blocks.append(sparse.csr_matrix(raw[:,cols]));totals.extend(total);detects.extend((raw>0).sum(1))
 with np.load(inputs/'evaluation-counts.npz') as a:assert np.array_equal(aggregate,a['counts'])
 del aggregate
 x=sparse.vstack(blocks,format='csr');del blocks
 # CSR cell x gene arrays are CSC gene x cell arrays without transposition work.
 write_bin(out/'counts.x.bin.gz',x.data.astype('<i4'));write_bin(out/'counts.i.bin.gz',x.indices.astype('<i4'));write_bin(out/'counts.p.bin.gz',x.indptr.astype('<i4'))
 cells['full_ncounts']=totals;cells['full_ndetected']=detects;cells.to_csv(out/'cells.tsv.gz',sep='\t',index=False)
 (out/'genes.txt').write_text('\n'.join(panel['genes'])+'\n')
 coverage=pd.read_csv(R/f'jobs/{dataset}-777/evaluation/coverage.csv');roster=sorted(coverage.loc[coverage.status.eq('evaluable'),'TF'].unique())
 (out/'primary-tfs.txt').write_text('\n'.join(roster)+'\n')
 nt=cells[cells.target.eq('control')].groupby('construct').size();trt=cells[~cells.target.eq('control')].groupby('target').size();sizes=[1,int(np.rint(trt.median()/nt.median())),int(np.ceil(trt.quantile(.9)/nt.median()))];sizes=[min(max(1,n),len(nt)//2) for n in sizes]
 meta={'dataset':dataset,'cells':x.shape[0],'genes':x.shape[1],'nnz':len(x.data),'source_sha256':prov['source_sha256'],'source_aggregate_exact':True,'training_cells_disjoint':True,'median_nt_construct_cells':float(nt.median()),'median_target_cells':float(trt.median()),'q90_target_cells':float(trt.quantile(.9)),'calibration_group_sizes':sizes,'primary_TFs':len(roster),'all_target_TFs':len(trt),'plan_sha256':sha(B/'RECALIBRATION-PLAN.md'),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='metadata.json'}}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');data.file.close();del x;print(meta,flush=True)
