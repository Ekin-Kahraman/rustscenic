"""Fresh-process original/native wrapper comparison; fitted stages reused.

Run each arm in a separate process with identical thread environment variables.
Uses the frozen native workflow plus bounded.py; reference packages are not used.
All input paths are explicit. Compare output hashes before interpreting costs.
"""
import hashlib,importlib.util,json,resource,sys,time,types,warnings,gc,threading
import pyarrow as pa
from pathlib import Path
import h5py,numpy as np,pandas as pd,pyarrow.feather as feather
from scipy import sparse
from rustscenic import _audited_multiome as w,aucell,topics
from check_paired import arguments, load_case, digest
parser=arguments()
parser.description=__doc__
parser.add_argument('--mode',choices=['original','updated'],required=True)
parser.add_argument('--repeat',default='1')
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--sample-rss',action='store_true',help='Sample analysis RSS every10ms; requires psutil')
args=parser.parse_args()
mode,label=args.mode,args.repeat
root,inputs,baseline,saved=args.output,args.inputs,args.baseline,args.saved
root.mkdir(parents=True,exist_ok=True)
case=load_case(args,use_threads=False)
ranks,rna,atac,cached,frames,tfs=(case[k] for k in ('ranks','rna','atac','cached','frames','tfs'))
if mode=='original':
 # Supply only native resource loader bindings; reference classes are not used.
 common=types.ModuleType('common');common.read_json=lambda p:json.loads(Path(p).read_text());sys.modules['common']=common
 ranking=types.ModuleType('rankings');ranking.load_projection=lambda p:(ranks,ranks.attrs['rank_universe_size'])
 ranking.native_annotations=lambda p:None;ranking.ProjectedRankingDatabase=None;sys.modules['rankings']=ranking
 for name in ['bounded','original_workflows']:
  file=baseline/('workflows.py' if name=='original_workflows' else 'bounded.py')
  spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 wf=sys.modules['original_workflows'];bounded=sys.modules['bounded']
 # Both arms use the same subsequently verified scorer. The historical baseline
 # batching and original dense imputation are otherwise left intact.
 native_score=aucell.score
 def seeded_score(*args,**kwargs):
  kwargs['seed']=20261106
  return native_score(*args,**kwargs)
 aucell.score=seeded_score
 topics.fit_gibbs=lambda *args,**kwargs:topics.TopicsResult(cached['cell_topic'].copy(),cached['topic_peak'].copy(),30)
 wf.infer_rna=lambda *args,**kwargs:cached['grn'].copy()
 pd.read_csv=lambda p,*args,**kwargs:frames[str(p)].copy(deep=False)
 clock=wf.Clock()
else:clock=None
gc.collect()
pa.default_memory_pool().release_unused()
gc.collect()
memory={"resource_loading":"serial Feather; garbage collection and Arrow release_unused before analysis"}
samples=[]
if args.sample_rss:
 import psutil
 process=psutil.Process()
 memory['analysis_start_rss_bytes']=process.memory_info().rss
 done=threading.Event()
 def sample_memory():
  while not done.wait(.01):samples.append(process.memory_info().rss)
 monitor=threading.Thread(target=sample_memory)
 monitor.start()
start=time.perf_counter()
with warnings.catch_warnings():
 warnings.filterwarnings('ignore',message='expression matrix max value is .*')
 if mode=='original':
  result=wf.multiome_workflow('rustscenic',rna,atac,None,tfs,inputs,inputs/'motifs',
      {'threads':4,'topics':{'n_topics':30,'iterations':500},'grn':w.GRN_SETTINGS},777,root,None,clock)
 else:
  result=w.run(rna,atac,tfs,gene_coords=frames[str(inputs/'gene-coordinates.tsv')],
      region_rankings=ranks,region_mapping=frames[str(inputs/'motifs/screen-overlap-mapping.tsv')],
      motif_annotations=frames[str(inputs/'motifs/human-gene-annotations.tbl')],output_dir=root,
      threads=4,cached_stages=cached)
elapsed=time.perf_counter()-start
if args.sample_rss:
 done.set();monitor.join()
 memory['analysis_end_rss_bytes']=process.memory_info().rss
 memory['analysis_sampled_peak_rss_bytes']=max(samples,default=memory['analysis_end_rss_bytes'])
hashes={}
for key in ['regulons','region_regulons']:
 value={k:sorted(v) for k,v in result[key].items()}
 hashes[key]=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
for key in ['gene_activity','region_activity']:
 value=result[key].sort_index(axis=1)
 hashes[key]=hashlib.sha256(np.ascontiguousarray(value.to_numpy(dtype=np.float32))).hexdigest()
for key,columns in [('eregulon_edges',['tf','region','target','context']),('region_to_gene',['region','target','rho']),('region_motif_scores',['region_set','motif','auc','nes'])]:
 value=result[key][columns].sort_values(columns).reset_index(drop=True)
 numeric=value.select_dtypes(include='number').columns
 value=value.astype({column:np.float64 for column in numeric})
 hashes[key]=hashlib.sha256(value.to_csv(index=False).encode()).hexdigest()
receipt={'mode':mode,'repeat':label,'elapsed_seconds':elapsed,'memory':memory,
 'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024),
 'hashes':hashes,'module_sha256':digest(w.__file__),
 'frozen_workflow_sha256':digest(baseline/'workflows.py'),
 'frozen_bounded_sha256':digest(baseline/'bounded.py'),
 'scope':f'Real BMMC cells: {args.cells}. Preloaded identical resources and fitted topics/GRN reused. Original audited native downstream recipe plus corrected seeded scorer vs new public helper. Independent fresh-process peak RSS includes imports/resources, not PSS or a full-inference benchmark.',
 'stages':clock.stages if clock else result['elapsed']}
(root/f'performance-{mode}-{label}.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
