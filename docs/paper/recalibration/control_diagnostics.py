"""Summarise the actual control sample sizes and positive-control coverage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from corrected_evaluation import bh_adjust

B = Path(__file__).resolve().parent
summaries, sizes, power_rows = [], [], []
for dataset in ['k562', 'rpe1']:
    folder = B/'results'/dataset
    cells = pd.read_csv(B/'inputs'/dataset/'cells.tsv.gz', sep='\t')
    mapping = pd.read_csv(folder/'construct-map.csv').set_index('grna_id')
    constructs = cells.groupby('construct').size()
    nt = cells.target.eq('control')
    targets = cells[~nt].groupby('target').size()
    for run in ['median-a', 'validation', 'single', 'upper']:
        a = pd.read_csv(folder/f'calibration-{run}.csv')
        counted = []
        for group in a.grna_target:
            ids = group.split('&')
            assert mapping.loc[ids, 'grna_target'].eq('non-targeting').all()
            counted.append(int(constructs.loc[mapping.loc[ids, 'original_construct']].sum()))
        counted = np.asarray(counted)
        assert (a.n_nonzero_trt <= counted).all()
        assert (a.n_nonzero_cntrl <= nt.sum()-counted).all()
        valid = a.pass_qc & np.isfinite(a.p_value)
        q = bh_adjust(a.loc[valid, 'p_value'])
        summaries.append(dict(dataset=dataset, run=run, tests=len(a), finite_tests=int(valid.sum()),
                              nominal_p05=float(a.loc[valid, 'p_value'].le(.05).mean()),
                              BH_discoveries=int((q <= .05).sum()),
                              median_log2_effect=float(a.loc[valid, 'log_2_fold_change'].median())))
        sizes.append(dict(dataset=dataset, group=run, distinct_construct_groups=a.grna_target.nunique(),
                          treatment_min=int(counted.min()), treatment_q05=float(np.quantile(counted, .05)),
                          treatment_median=float(np.median(counted)), treatment_q95=float(np.quantile(counted, .95)),
                          treatment_max=int(counted.max()), control_median=float(np.median(nt.sum()-counted))))
    sizes.append(dict(dataset=dataset, group='discovery-targets', distinct_construct_groups=len(targets),
                      treatment_min=int(targets.min()), treatment_q05=float(targets.quantile(.05)),
                      treatment_median=float(targets.median()), treatment_q95=float(targets.quantile(.95)),
                      treatment_max=int(targets.max()), control_median=int(nt.sum())))
    a = pd.read_csv(folder/'positive-controls.csv')
    valid = a.pass_qc & np.isfinite(a.p_value)
    a['q_value'] = np.nan
    a.loc[valid, 'q_value'] = bh_adjust(a.loc[valid, 'p_value'])
    significant_down = valid & a.q_value.le(.05) & a.log_2_fold_change.lt(0)
    a.to_csv(folder/'positive-controls-with-q.csv', index=False)
    power_rows.append(dict(dataset=dataset, evaluation_cells=len(cells), control_cells=int(nt.sum()),
                           captures=cells.batch.nunique(), total_targeted_TFs=len(targets),
                           self_gene_in_panel=len(a), self_QC_pass=int(valid.sum()),
                           significant_down=int(significant_down.sum()),
                           adequate_KD=int((valid & a.q_value.le(.05) & a.log_2_fold_change.le(-.25)).sum()),
                           fraction_down_among_QC_pass=float(significant_down.sum()/valid.sum()),
                           self_below_treatment_detection_QC=int(a.n_nonzero_trt.lt(7).sum()),
                           self_below_control_detection_QC=int(a.n_nonzero_cntrl.lt(7).sum()),
                           median_eligible_log2_effect=float(a.loc[valid, 'log_2_fold_change'].median())))

(B/'summary').mkdir(exist_ok=True)
pd.DataFrame(summaries).to_csv(B/'summary/control-calibration.csv', index=False)
pd.DataFrame(sizes).to_csv(B/'summary/control-sample-sizes.csv', index=False)
pd.DataFrame(power_rows).to_csv(B/'summary/positive-control-coverage.csv', index=False)
print(pd.DataFrame(sizes).to_string(index=False))
print(pd.DataFrame(power_rows).to_string(index=False))
