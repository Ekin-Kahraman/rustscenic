"""Fixed signed representation with bounded shared/native AUCell scoring."""
import json,time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from common import write_json,digest
from matrix_io import strings,csr_slice,validate_matrix
from polarity import signed_gene_sets,signed_contrast

SEED=20261106


def signatures(edges,method,mode):
    signed=signed_gene_sets(edges,method)
    if mode=='signed':return signed
    if mode!='union':raise ValueError('Unknown score mode')
    return edges.groupby('tf').target.agg(lambda v:sorted(set(v))).to_dict()


def coverage(sets,genes):
    available=set(genes)
    return {k:{'total':len(v),'present':len(set(v)&available),
               'fraction':len(set(v)&available)/len(v) if v else 0.} for k,v in sets.items()}


def score_block(x,genes,cells,sets,scorer,threads):
    cov=coverage(sets,genes)
    eligible={k:v for k,v in sorted(sets.items()) if cov[k]['fraction']>=.8}
    if not eligible:return pd.DataFrame(0.,index=cells,columns=sorted(sets))
    if scorer=='shared':
        from pyscenic.aucell import create_rankings,aucell4r
        from ctxcore.genesig import GeneSignature
        from process_guard import guard_aucell_workers
        guard_aucell_workers()
        ranked=create_rankings(pd.DataFrame(x.toarray(),index=cells,columns=genes),seed=SEED)
        frame=aucell4r(ranked,[GeneSignature(k,v) for k,v in eligible.items()],auc_threshold=.05,
                       noweights=True,normalize=False,num_workers=threads).reindex(cells)
    elif scorer in ('native','seeded'):
        from rustscenic.aucell import score
        kwargs={'seed':SEED} if scorer=='seeded' else {}
        frame=score((x,genes),eligible,**kwargs);frame.index=cells
    else:raise ValueError('Unknown scorer')
    frame=frame.reindex(columns=sorted(sets),fill_value=0.)
    if not np.isfinite(frame.to_numpy()).all():raise ValueError('Nonfinite scores')
    return frame


def score_file(matrix,edges,method,mode,scorer,out,threads=16,block_size=4096):
    """Write every real cell once, returning small samples and separate timings."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    genes,cells=validate_matrix(matrix)
    sets=signatures(edges,method,mode)
    if not sets:raise ValueError('No signatures; retain failure instead of a vacuous comparison')
    tf_names=sorted(set(edges.tf));cov=coverage(sets,genes)
    write_json(out/'signatures.json',sets);write_json(out/'coverage.json',cov)
    times={k:0. for k in ['read_seconds','score_seconds','combine_seconds','write_seconds']}
    sample_rows=np.sort(np.random.default_rng(20261028).choice(len(cells),size=min(10000,len(cells)),replace=False))
    samples=[];nonzero=np.zeros(len(tf_names),np.int64);sums=np.zeros(len(tf_names));squares=sums.copy()
    final=out/'full-scores.h5';temporary=out/'full-scores.h5.tmp'
    start_all=time.perf_counter();write_json(out/'phase.json',{'phase':'analysis'})
    with h5py.File(matrix) as src,h5py.File(temporary,'w') as dst:
        target=dst.create_dataset('scores',shape=(len(cells),len(tf_names)),dtype='f8',
                                  chunks=(min(block_size,len(cells)),len(tf_names)))
        for key,values in [('cells',cells),('features',tf_names)]:dst.create_dataset(key,data=np.asarray(values,dtype=object),dtype=h5py.string_dtype())
        for start in range(0,len(cells),block_size):
            end=min(start+block_size,len(cells));t=time.perf_counter();x=csr_slice(src,start,end)
            times['read_seconds']+=time.perf_counter()-t
            t=time.perf_counter();frame=score_block(x,genes,cells[start:end],sets,scorer,threads)
            times['score_seconds']+=time.perf_counter()-t
            t=time.perf_counter();frame=signed_contrast(frame) if mode=='signed' else frame
            frame=frame[tf_names];values=frame.to_numpy(dtype=np.float64)
            times['combine_seconds']+=time.perf_counter()-t
            t=time.perf_counter();target[start:end]=values
            times['write_seconds']+=time.perf_counter()-t
            nonzero+=(values!=0).sum(0);sums+=values.sum(0);squares+=(values**2).sum(0)
            take=sample_rows[(sample_rows>=start)&(sample_rows<end)]-start
            if len(take):samples.append(frame.iloc[take])
    temporary.replace(final);elapsed=time.perf_counter()-start_all
    write_json(out/'phase.json',{'phase':'export'})
    pd.concat(samples).to_parquet(out/'sampled-scores.parquet')
    write_json(out/'score-summary.json',{'method':method,'mode':mode,'scorer':scorer,'cells':len(cells),
        'genes':len(genes),'tf_features':len(tf_names),'input_signatures':len(sets),'block_size':block_size,
        'stage_elapsed_seconds':elapsed,'timings':times,
        'per_tf':{tf:{'nonzero':int(nonzero[i]),'mean':float(sums[i]/len(cells)),
                       'std':float(np.sqrt(max(0,squares[i]/len(cells)-(sums[i]/len(cells))**2)))} for i,tf in enumerate(tf_names)},
        'low_coverage_signatures':[k for k,v in cov.items() if v['fraction']<.8],
        'full_scores_sha256':digest(final),'full_scores_bytes':final.stat().st_size})
    return final


def read_scores(path):
    with h5py.File(path) as h:return pd.DataFrame(h['scores'][:],index=strings(h['cells'][:]),columns=strings(h['features'][:]))
