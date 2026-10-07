#!/usr/bin/env python3
"""Independent arithmetic/readback audit and current-only CPM figure."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def holm(p):
    p=np.array(p);order=np.argsort(p);adj=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)));out=np.empty_like(p);out[order]=adj;return out

def main():
    tab=pd.read_csv(HERE/'table_cpm_resolved.csv');pred=pd.read_csv(HERE/'cpm_prediction_data.csv');dist=np.load(HERE/'cpm_distributions.npz');exe=json.loads((HERE/'execution_verification.json').read_text())
    assert len(tab)==6 and len(pred)==33 and pred.subject.nunique()==33 and exe['status']=='completed'
    perm=dist['permutation_r'];assert perm.shape==(10000,6) and np.isfinite(perm).all()
    names=['DefaultA','A-B','A-C','DefaultB','B-C','DefaultC'];r=[]
    for name in names:
        v=pred[f'predicted_{name}_ms'].to_numpy();y=pred.observed_fold_centered_ms.to_numpy()
        r.append(stats.pearsonr(v,y).statistic if np.std(v)>1e-14 else 0)
    r=np.array(r);assert np.allclose(r,tab.r_loocv,atol=1e-12) and np.allclose(r,dist['observed_r'],atol=1e-12)
    upper=((perm>=r).sum(axis=0)+1)/10001;lower=((perm<=r).sum(axis=0)+1)/10001
    equal=np.minimum(1,2*np.minimum(upper,lower));absolute=((np.abs(perm)>=np.abs(r)).sum(axis=0)+1)/10001
    assert np.allclose(equal,tab.p_raw_equal_tail,atol=1e-14) and np.allclose(absolute,tab.p_raw_absolute_sensitivity,atol=1e-14)
    assert np.allclose(holm(equal),tab.p_Holm6_equal_tail,atol=1e-14) and np.allclose(holm(absolute),tab.p_Holm6_absolute_sensitivity,atol=1e-14)
    cross=pd.read_csv(HERE/'physical_yeo7_to_17_crosswalk.csv');assert len(cross)==399 and cross.yeo7_id.nunique()==399 and cross.yeo17_id.nunique()==399 and set(cross.yeo7_id)==set(range(1,401))-{323}
    assert (cross.network7=='Default').sum()==91 and cross.network17.str.startswith('Default').sum()==79
    folds=pd.read_csv(HERE/'observed_fold_selection_counts.csv');assert len(folds)==198 and folds.held_out_subject.nunique()==33
    for p,h in json.loads((HERE/'input_fingerprints.json').read_text()).items():
        with Path(p).open('rb') as f:
            q=hashlib.file_digest(f,'sha256').hexdigest()
        assert q==h,p
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(11.5,7),constrained_layout=True)
    palette={'SS':'#D98C5F','DS':'#5B4B8A'}
    for f,(name,ax) in enumerate(zip(names,axes.flat)):
        x=pred.observed_fold_centered_ms.to_numpy();y=pred[f'predicted_{name}_ms'].to_numpy()
        for g in ['SS','DS']:
            ix=(pred.group==g).to_numpy();ax.scatter(x[ix],y[ix],s=35,color=palette[g],edgecolors='white',linewidths=.5,label=g)
        lo,hi=np.min(np.r_[x,y]),np.max(np.r_[x,y]);pad=max(1,(hi-lo)*.08);xx=np.array([lo-pad,hi+pad])
        ax.plot(xx,xx,'--',color='#999999',lw=.8)
        if np.std(y)>1e-14:
            coef=np.polyfit(x,y,1);ax.plot(xx,np.polyval(coef,xx),color='#333333',lw=1)
        ax.set_title(name.replace('-','–'));ax.set_xlabel('Observed capture (fold-centered, ms)');ax.set_ylabel('Held-out prediction (ms)')
        ax.text(.04,.96,f"r = {r[f]:+.2f}\nPermutation p = {equal[f]:.3f}\nHolm p = {holm(equal)[f]:.3f}",transform=ax.transAxes,ha='left',va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'#cccccc','alpha':.9})
        if f==0:ax.legend(frameon=False,loc='lower right')
    fig.savefig(HERE/'Figure_CPM_resolved.png',dpi=300);fig.savefig(HERE/'Figure_CPM_resolved.pdf');plt.close(fig)
    report={'status':'all_checks_passed','subjects':33,'cells':6,'fold_selection_rows':198,'permutations':10000,'max_r_readback_error':float(np.max(np.abs(r-tab.r_loocv))),'input_hashes_unchanged':True,'Holm_families_recomputed':True,'physical_Default7_91_Default17_79_verified':True}
    (HERE/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
