"""Corrected response processing. Saved network recipes are never changed here."""
import numpy as np

def bh_adjust(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=1 or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('BH requires finite probabilities in [0,1]')
    if not len(p):return p.copy()
    order=np.argsort(p,kind='stable');ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    ranked=np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1]);out=np.empty_like(p);out[order]=ranked
    return out

def calibrated_labels(frame,threshold=.5,alpha=.05):
    f=frame.copy()
    if f[['grna_target','response_id']].duplicated().any():raise ValueError('Duplicate response pair')
    if f.grna_target.eq(f.response_id).any():raise ValueError('Self-target pairs belong to the separate power check')
    p=f.p_value.to_numpy(float);qc=f.pass_qc.to_numpy(bool)
    if np.any(np.isfinite(p)&((p<0)|(p>1))):raise ValueError('Invalid p-value')
    tested=qc & np.isfinite(p)
    q=np.full(len(f),np.nan);q[tested]=bh_adjust(p[tested]);effect=f.log_2_fold_change.to_numpy(float)
    eligible=tested&np.isfinite(effect)
    response=eligible&(q<=alpha)&(np.abs(effect)>=threshold)
    f['q_value']=q;f['eligible']=eligible;f['response']=response
    f['response_direction']=np.where(response,np.sign(effect),0).astype(np.int8)
    return f

def pooled_log_effect(treated_cpm,control_cpm,weights):
    a=np.asarray(treated_cpm,float);b=np.asarray(control_cpm,float);w=np.asarray(weights,float)
    if a.shape!=b.shape or len(a)!=len(w) or np.any(w<0) or not w.sum():raise ValueError('Invalid pooled comparison')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<0) or np.any(b<0):raise ValueError('Invalid CPM')
    return np.log2(1+np.average(a,axis=0,weights=w))-np.log2(1+np.average(b,axis=0,weights=w))
