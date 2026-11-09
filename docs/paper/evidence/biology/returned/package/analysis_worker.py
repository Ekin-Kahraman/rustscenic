"""New validation only; frozen benchmark code and old outputs remain unchanged."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'baseline'))
import workflows
from workflows import ann,Clock
from common import digest,write_json,read_json
from data_io import load_matrix
from worker import save_activity,export_table
from reference_metrics import reference_metrics,retained_support


def export(out,result,cells):
    for key,value in result.items():
        if isinstance(value,pd.DataFrame) and key not in ['gene_activity','region_activity']:
            export_table(out,key,value)
        elif key in ['regulons','region_regulons']:
            write_json(out/(key.replace('_','-')+'.json'),value)
    for kind in ['gene','region']:
        if kind+'_activity' in result:save_activity(out,kind,result[kind+'_activity'],cells)
    write_json(out/'cells.json',cells)


def run(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(ROOT/'baseline/config.json');cfg['threads']=args.threads
    clock=Clock();started=time.perf_counter()
    if args.kind=='rna':
        folder=ROOT/'rna-inputs'/args.dataset;info=read_json(folder/'input.json')
        x=sparse.load_npz(folder/'matrix.npz');cells=[f'{args.dataset}:{i}' for i in range(x.shape[0])]
        rna=ann(x,info['genes'],cells)
        old_method='rustscenic' if args.method=='rustscenic' else 'arboreto_mp_sparse'
        previous=ROOT/f'reused-networks/{args.dataset}-seed{args.seed}-{old_method}/network.parquet'
        grn=pd.read_parquet(previous)
        # Reuse the exact completed inference output, avoiding 18 needless inference fits.
        def frozen_inference(*unused,**kwargs):
            clock.fit_summary={'reused_network_sha256':digest(previous)}
            return grn.copy()
        workflows.infer_rna=frozen_inference
        result=workflows.rna_workflow(args.method,rna,info['tfs'],ROOT/'rna-motifs'/info['species'],cfg,args.seed,clock)
        export(out,result,cells)
        memberships={(name.removesuffix('_activator').removesuffix('_repressor'),gene)
                     for name,genes in result['regulons'].items() for gene in genes}
        pruned=grn.loc[[(a,b) in memberships for a,b in grn[['TF','target']].itertuples(index=False,name=None)]].copy()
        pruned.to_parquet(out/'pruned-network.parquet',index=False)
        scores=[]
        for ref in info['references']:
            truth=pd.read_csv(folder/ref['file'],sep='\t')
            for stage,network in [('inference',grn),('motif_pruned',pruned)]:
                scores.append({'reference':ref['file'],'stage':stage,**reference_metrics(network,truth,info['genes'],info['tfs'])})
                if stage=='motif_pruned':scores[-1].update(retained_support(pruned,grn,truth,info['genes'],info['tfs']))
        write_json(out/'reference-recovery.json',scores)
        source={'dataset':args.dataset,'input':digest(folder/'input.json'),'reused_network':digest(previous),
                'motif_metadata':read_json(ROOT/'rna-motifs'/info['species']/'mouse-gene.json'),
                'inference_repeated':False}
    else:
        folder=args.inputs or ROOT/'holdout-inputs';info=read_json(folder/'input.json')
        x,genes,cells=load_matrix(folder/'rna.h5',args.cells)
        y,peaks,other=load_matrix(folder/'atac.h5',args.cells)
        assert cells==other
        meta=pd.read_csv(folder/'cell-metadata.tsv',sep='\t').set_index('cell').loc[cells]
        if not meta['split'].eq('train').all():raise ValueError('Test cells entered network fitting')
        tfs=(folder/'tfs.txt').read_text().splitlines()
        if args.smoke:
            cfg['grn']['n_estimators']=50;cfg['topics']['iterations']=20;cfg['topics']['n_topics']=5
        original_support=workflows.region_motif_support
        def typed_support(*positional,**keywords):
            support,scores=original_support(*positional,**keywords)
            # A genuine empty motif result has object dtype in pandas by default.
            # Keep it empty, but satisfy the public native API's numeric schema.
            if support.empty:
                for column in ['auc','nes']:
                    if column in support:support[column]=support[column].astype(float)
            return support,scores
        workflows.region_motif_support=typed_support
        result=workflows.multiome_workflow(args.method,ann(x,genes,cells),ann(y,peaks,cells),None,tfs,
            folder,folder/'motifs',cfg,args.seed,out,args.mallet,clock)
        export(out,result,cells)
        source={'dataset':'BMMC donor-disjoint validation','input':digest(folder/'input.json'),
                'train_donors':info['train_donors'],'test_donors':info['test_donors'],
                'fitted_cells':len(cells),'gene_features':len(genes),'tfs':len(tfs),'peaks':len(peaks),
                'empty_motif_schema_adapter':'Numeric dtype for empty auc/nes columns only; no scores or thresholds changed.'}
    import importlib.metadata as md
    summary={'status':'complete','kind':args.kind,'method':args.method,'seed':args.seed,'smoke':args.smoke,
             'elapsed_s':time.perf_counter()-started,'stages':clock.stages,'regulons':len(result['regulons']),
             'source':source,'settings':cfg,'versions':{n:md.version(n) for n in ['rustscenic','pyscenic','scenicplus','numpy','pandas']},
             'outputs':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.parquet','.json','.npy'] and p.name!='complete.json'}}
    write_json(out/'complete.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','method','regulons','elapsed_s']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['rna','holdout'],required=True)
    p.add_argument('--dataset');p.add_argument('--method',choices=['rustscenic','reference'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--cells',type=int,default=20000)
    p.add_argument('--threads',type=int,default=16);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mallet',type=Path);p.add_argument('--inputs',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if not os.environ.get('LSB_JOBID') and not a.smoke:raise RuntimeError('Run through start.sh on HPC; --smoke is only for local validation')
    run(a)
