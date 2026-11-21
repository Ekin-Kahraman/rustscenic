"""Published pySCENIC refinement, with the full set of standard module families."""
from time import perf_counter
import numpy as np
import pandas as pd

def refine_reference(network, expression, database, annotations, threads=4):
    if 'object' not in np.__dict__:np.object=object
    from pyscenic.utils import modules_from_adjacencies
    from pyscenic.prune import prune2df
    from pyscenic.transform import df2regulons
    start=perf_counter()
    modules=list(modules_from_adjacencies(network, expression, thresholds=(.75,.90),
        top_n_targets=(50,),top_n_regulators=(5,10,50),min_genes=20,
        keep_only_activating=False,rho_dichotomize=True,rho_threshold=.03,rho_mask_dropouts=False))
    stage=perf_counter()
    enriched=prune2df([database],modules,str(annotations),rank_threshold=1500,
        auc_threshold=.05,nes_threshold=3.,weighted_recovery=False,
        client_or_address='custom_multiprocessing',num_workers=threads,filter_for_annotation=True) if modules else pd.DataFrame()
    result={}
    for reg in df2regulons(enriched) if not enriched.empty else []:
        if len(reg.genes)>=5:
            result[str(reg.name).replace('(+)','_activator').replace('(-)','_repressor')]=sorted(reg.genes)
    return result,{'module_s':stage-start,'pruning_s':perf_counter()-stage,'wall_s':perf_counter()-start,'candidate_modules':len(modules),'regulons':len(result)}
