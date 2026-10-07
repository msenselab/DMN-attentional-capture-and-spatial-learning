#!/usr/bin/env python3
"""Seven intact physical Yeo7 CPM models; all outputs confined to this directory."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
from pathlib import Path
import json,hashlib,time,argparse,platform
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
CPM=HERE.parent
ROUND=CPM.parent.parent
REV=ROUND.parent.parent
RES=ROUND/'resolved_reanalysis'
NAMES=['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']
LABELS=['Visual','Somatomotor','Dorsal attention','Ventral attention / salience','Limbic','Frontoparietal control','Default mode']
PARCEL_COUNTS=[61,77,46,47,25,52,91]
EDGES=[1830,2926,1035,1081,300,1326,4095]
NPERM,NBOOT=10000,5000


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p,kind='stable');out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return out

def hash_map(paths):return {str(p.resolve()):sha(p) for p in paths}

def verify_hashes(values):
    for p,h in values.items():
        if sha(p)!=h:raise RuntimeError('Hash changed: '+p)

def evaluate(Xs,Y,W,groups,details=False):
    """Batched multiplicity-weighted, identity-excluded LOOCV. Y,W are N x B."""
    n,b=Y.shape
    pred=np.zeros((n,b,7));target=np.empty((n,b))
    counts=np.zeros((n,b,7,3),dtype=np.int16) if details else None
    # Cache raw second moments; weighted normalization must still be refit each draw/fold.
    squares=[X*X for X in Xs]
    for test in range(n):
        trainW=W.copy();trainW[test]=0
        nt=trainW.sum(axis=0)
        yt=Y.copy()
        for g in ['DS','SS']:
            ix=groups==g;sw=trainW[ix].sum(axis=0)
            if np.any(sw<=0):raise RuntimeError('Group has no training identities')
            mu=np.sum(trainW[ix]*Y[ix],axis=0)/sw
            yt[ix]-=mu
            if groups[test]==g:target[test]=Y[test]-mu
        mean_y=np.sum(trainW*yt,axis=0)/nt
        centered_y=yt-mean_y
        yvar=np.sum(trainW*centered_y**2,axis=0)
        if np.any(yvar<=0):raise RuntimeError('Training target is constant')
        weighted_y=trainW*centered_y
        tc=stats.t.isf(.005,nt-2);rc=tc/np.sqrt(tc*tc+nt-2)
        for f,(X,Xsq) in enumerate(zip(Xs,squares)):
            sumx=X.T@trainW
            ssx=np.maximum(0,Xsq.T@trainW-sumx*sumx/nt)
            cov=X.T@weighted_y
            den=np.sqrt(ssx*yvar)
            corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-14)
            signs=np.where(np.abs(corr)>rc,np.sign(corr),0.)
            strengths=X@signs
            sm=np.sum(trainW*strengths,axis=0)/nt
            centered_strength=strengths-sm
            ss=np.sum(trainW*centered_strength**2,axis=0)
            slope=np.divide(np.sum(weighted_y*centered_strength,axis=0),ss,out=np.zeros(b),where=ss>1e-14)
            pred[test,:,f]=mean_y+slope*(strengths[test]-sm)
            pred[test,ss<=1e-14,f]=0.
            if details:
                counts[test,:,f,0]=np.sum(signs>0,axis=0)
                counts[test,:,f,1]=np.sum(signs<0,axis=0)
                counts[test,:,f,2]=ss<=1e-14
    weight_total=W.sum(axis=0)
    yc=target-np.sum(W*target,axis=0)/weight_total
    pc=pred-np.sum(W[:,:,None]*pred,axis=0)/weight_total[:,None]
    denom=np.sqrt(np.sum(W*yc*yc,axis=0)[:,None]*np.sum(W[:,:,None]*pc*pc,axis=0))
    r=np.divide(np.sum(W[:,:,None]*yc[:,:,None]*pc,axis=0),denom,out=np.zeros((b,7)),where=denom>1e-14)
    return r,pred,target,counts

def self_test():
    rng=np.random.default_rng(413)
    groups=np.array(['DS']*5+['SS']*5)
    Xs=[rng.normal(size=(10,9)) for _ in range(7)]
    y=rng.normal(size=10)
    w=np.array([2,1,0,1,1,1,2,1,0,1],float)
    rb,pb,tb,cb=evaluate(Xs,y[:,None],w[:,None],groups,True)
    pp=np.zeros((10,7));tt=np.empty(10)
    for test in range(10):
        copies=np.repeat(np.arange(10),w.astype(int));train=copies[copies!=test]
        yy=y[train].copy()
        for g in ['DS','SS']:
            mu=yy[groups[train]==g].mean();yy[groups[train]==g]-=mu
            if groups[test]==g:tt[test]=y[test]-mu
        for f,X in enumerate(Xs):
            xx=X[train];rr=stats.pearsonr(xx,yy[:,None],axis=0)
            sg=np.where(rr.pvalue<.01,np.sign(rr.statistic),0.)
            ss=xx@sg
            if np.sum((ss-ss.mean())**2)>1e-14:
                beta=np.linalg.lstsq(np.c_[np.ones(len(train)),ss],yy,rcond=None)[0]
                pp[test,f]=beta[0]+beta[1]*(X[test]@sg)
    assert np.allclose(pp,pb[:,0,:],atol=1e-10,rtol=0)
    assert np.allclose(tt,tb[:,0],atol=1e-10,rtol=0)
    return {'status':'PASS_SYNTHETIC_NOT_EVIDENCE','replicated_rows_identity_exclusion_max_prediction_error':float(np.max(np.abs(pp-pb[:,0,:])))}

def load():
    inputs=[HERE/'ANALYSIS_SPEC.md',HERE/'run_yeo7_cpm.py',CPM.parent/'input_freeze.json',CPM.parent/'ANALYSIS_SPEC.md',CPM.parent/'learning_target_scores.csv',CPM.parent/'learning_target_run_counts.csv',
      RES/'extraction_common_coverage/primary_participant_matrices.npy',RES/'extraction_common_coverage/primary_matrix_subjects.csv',RES/'common399_analysis_ready/physical_yeo7_common399_parcels.csv',
      REV/'attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv',ROUND/'spatial_learning_dmn_v1/participant_analysis/participant_learning_scores.csv']
    freeze=json.loads((CPM.parent/'input_freeze.json').read_text())
    verify_hashes(freeze['input_hashes'])
    assert sha(CPM.parent/'learning_target_scores.csv')==freeze['target_hash']
    protected=[p for p in CPM.rglob('*') if p.is_file() and HERE not in p.parents]
    manifest={'immutable_inputs':hash_map(inputs),'protected_old_cpm_paths':hash_map(protected),'models':NAMES,'permutation_seed':20261004,'bootstrap_seed':20261006,'n_permutations':NPERM,'n_bootstrap':NBOOT,'numpy':np.__version__}
    mp=HERE/'input_manifest.json'
    if mp.exists():
        prior=json.loads(mp.read_text());assert manifest==prior,'Manifest/config changed; refuse overwrite'
    else:mp.write_text(json.dumps(manifest,indent=2)+'\n')
    ids=pd.read_csv(inputs[7],dtype={'subject':str});target=pd.read_csv(inputs[4],dtype={'subject':str})
    assert ids[['subject','group']].equals(target[['subject','group']]) and len(ids)==33 and ids.subject.nunique()==33
    assert ids.group.value_counts().to_dict()=={'DS':17,'SS':16}
    trials=pd.read_csv(inputs[9],dtype={'subject':str})
    assert len(trials)==10995 and trials.correct.eq(1).all() and np.isfinite(trials.RT_ms).all() and trials.RT_ms.gt(200).all() and trials.RT_ms.le(3000).all()
    assert set(trials.subject)==set(ids.subject)
    b=trials[trials.category.eq('biased')];assert len(b)==6721
    rt=b.groupby(['subject','DisRegion']).RT_ms.mean().unstack().reindex(ids.subject)
    y=(rt.Rare-rt.Frequent).to_numpy(float)
    assert np.allclose(y,target.learning_rare_minus_frequent_ms,atol=1e-10,rtol=0)
    old=pd.read_csv(inputs[10]).set_index('subject').loc[ids.subject]
    assert np.allclose(y,old.learning_rare_minus_frequent_ms,atol=1e-10,rtol=0)
    parcels=pd.read_csv(inputs[8]).sort_values('matrix_index')
    assert np.array_equal(parcels.matrix_index,np.arange(399)) and set(parcels.native_label)==set(range(1,401))-{323}
    assert np.array_equal(parcels.yeo7_network,parcels.physical_label.str.split('_').str[2])
    conn=np.load(inputs[6],mmap_mode='r');assert conn.shape==(33,399,399) and np.isfinite(conn).all()
    assert np.allclose(conn,conn.swapaxes(1,2),atol=1e-12,rtol=0)
    Xs=[];rows=[]
    for name,nparcel,nedge in zip(NAMES,PARCEL_COUNTS,EDGES):
        ix=np.flatnonzero(parcels.yeo7_network.to_numpy()==name);assert len(ix)==nparcel
        a,c=np.triu_indices(len(ix),1);assert len(a)==nedge
        Xs.append(np.array(conn[:,ix[a],ix[c]],float))
        rows.append({'network':name,'parcels':nparcel,'within_edges':nedge})
    pd.DataFrame(rows).to_csv(HERE/'physical_network_geometry.csv',index=False)
    return manifest,ids,y,Xs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--batch-size',type=int,default=100);args=ap.parse_args()
    test=self_test();(HERE/'synthetic_algorithm_test.json').write_text(json.dumps(test,indent=2)+'\n')
    if args.self_test:print(test);return
    started=time.monotonic();manifest,ids,y,Xs=load();g=ids.group.to_numpy(str)
    obs,pred,fold,counts=evaluate(Xs,y[:,None],np.ones((33,1)),g,True)
    obs,pred,fold,counts=obs[0],pred[:,0,:],fold[:,0],counts[:,0,:,:]
    df=ids[['subject','group']].copy();df['raw_learning_ms']=y;df['fold_target_ms']=fold
    for f,name in enumerate(NAMES):df[f'pred_{name}_ms']=pred[:,f]
    df.to_csv(HERE/'observed_predictions.csv',index=False)
    rows=[]
    for test in range(33):
        for f,name in enumerate(NAMES):rows.append({'held_out_index':test,'subject':ids.subject.iloc[test],'network':name,'positive_edges':int(counts[test,f,0]),'negative_edges':int(counts[test,f,1]),'zero_strength':int(counts[test,f,2])})
    pd.DataFrame(rows).to_csv(HERE/'observed_selection_counts.csv',index=False)
    ds,ss=np.flatnonzero(g=='DS'),np.flatnonzero(g=='SS')
    rng=np.random.default_rng(20261004);maps=np.empty((NPERM,33),np.int16)
    for d in range(NPERM):
        maps[d,ds]=rng.permutation(ds);maps[d,ss]=rng.permutation(ss)
    permstate=rng.bit_generator.state
    rng=np.random.default_rng(20261006);weights=np.zeros((NBOOT,33),np.int16)
    for d in range(NBOOT):
        ix=np.r_[rng.choice(ds,len(ds),replace=True),rng.choice(ss,len(ss),replace=True)]
        weights[d]=np.bincount(ix,minlength=33)
        assert np.count_nonzero(weights[d,ds])>=2 and np.count_nonzero(weights[d,ss])>=2
    bootstate=rng.bit_generator.state
    np.savez_compressed(HERE/'source_maps_and_weights.npz',permutation_maps=maps,bootstrap_weights=weights)
    perm=np.empty((NPERM,7));boot=np.empty((NBOOT,7))
    for kind,total,dest in [('permutation',NPERM,perm),('bootstrap',NBOOT,boot)]:
        for low in range(0,total,args.batch_size):
            high=min(total,low+args.batch_size)
            if kind=='permutation':Y=y[maps[low:high]].T;W=np.ones_like(Y)
            else:Y=np.broadcast_to(y[:,None],(33,high-low)).copy();W=weights[low:high].T.astype(float)
            dest[low:high]=evaluate(Xs,Y,W,g)[0]
            if high%1000==0:print(f'{kind} {high}/{total}',flush=True)
        np.save(HERE/(kind+'_r_checkpoint.npy'),dest)
    np.savez_compressed(HERE/'archive.npz',permutation_maps=maps,permutation_r=perm,bootstrap_weights=weights,bootstrap_r=boot)
    up=(perm>=obs).sum(axis=0);down=(perm<=obs).sum(axis=0)
    peq=np.minimum(1,2*np.minimum((up+1)/(NPERM+1),(down+1)/(NPERM+1)))
    pab=(np.sum(np.abs(perm)>=np.abs(obs),axis=0)+1)/(NPERM+1)
    heq,hab=holm(peq),holm(pab)
    ranking=stats.rankdata(-obs,method='min').astype(int)
    rows=[]
    for f,name in enumerate(NAMES):
        rows.append({'network':name,'network_label':LABELS[f],'n':33,'parcels':PARCEL_COUNTS[f],'possible_edges':EDGES[f],'r_loocv':obs[f],'descriptive_signed_r_rank':ranking[f],
          'p_raw_equal_tail':peq[f],'p_Holm7_equal_tail':heq[f],'p_raw_absolute_sensitivity':pab[f],'p_Holm7_absolute_sensitivity':hab[f],
          'upper_exceedances':int(up[f]),'lower_exceedances':int(down[f]),'rmse_ms_descriptive':float(np.sqrt(np.mean((pred[:,f]-fold)**2))),
          'zero_prediction_baseline_rmse_ms':float(np.sqrt(np.mean(fold**2))),'zero_strength_folds':int(counts[:,f,2].sum()),
          'bootstrap_r_ci_low':np.quantile(boot[:,f],.025),'bootstrap_r_ci_high':np.quantile(boot[:,f],.975)})
    tab=pd.DataFrame(rows);tab.to_csv(HERE/'network_results.csv',index=False)
    delta=obs[-1]-obs[:-1];dboot=boot[:,-1,None]-boot[:,:-1]
    pcomp=(np.sum(np.abs(dboot-delta)>=np.abs(delta),axis=0)+1)/(NBOOT+1)
    hcomp=holm(pcomp);rows=[]
    for f,name in enumerate(NAMES[:-1]):
        rows.append({'other_network':name,'r_Default':obs[-1],'r_other':obs[f],'delta_r_Default_minus_other':delta[f],
           'bootstrap_delta_ci_low':np.quantile(dboot[:,f],.025),'bootstrap_delta_ci_high':np.quantile(dboot[:,f],.975),
           'bootstrap_delta_simultaneous_low':np.quantile(dboot[:,f],.05/12),'bootstrap_delta_simultaneous_high':np.quantile(dboot[:,f],1-.05/12),
           'p_raw_centered_bootstrap_approximate':pcomp[f],'p_Holm6_centered_bootstrap_approximate':hcomp[f],
           'corrected_positive_advantage':bool(delta[f]>0 and hcomp[f]<.05)})
    comp=pd.DataFrame(rows);comp.to_csv(HERE/'dmn_comparisons.csv',index=False)
    archive_files=['archive.npz','source_maps_and_weights.npz','permutation_r_checkpoint.npy','bootstrap_r_checkpoint.npy']
    archive_manifest={'status':'complete','permutations':NPERM,'bootstraps':NBOOT,'permutation_rng_final_state':permstate,'bootstrap_rng_final_state':bootstate,
       'archive_sha256':{p:sha(HERE/p) for p in archive_files},'immutable_inputs':manifest['immutable_inputs']}
    (HERE/'archive_manifest.json').write_text(json.dumps(archive_manifest,indent=2)+'\n')
    verify_hashes(manifest['immutable_inputs']);verify_hashes(manifest['protected_old_cpm_paths'])
    execution={'status':'completed_numerical_verification_pending','seconds':time.monotonic()-started,'models':NAMES,'n':33,'permutations':NPERM,'bootstraps':NBOOT,
      'inputs_unchanged':True,'old_cpm_files_unchanged':True,'old_cpm_file_count':len(manifest['protected_old_cpm_paths']),'dmn_rank_descriptive':int(ranking[-1]),
      'dmn_corrected_positive_prediction':bool(obs[-1]>0 and heq[-1]<.05),'dmn_advantages_pass_all_six_approximate_corrected_tests':bool(comp.corrected_positive_advantage.all()),
      'python':platform.python_version(),'numpy':np.__version__}
    (HERE/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    plot(tab,comp,pred,fold,ids)
    print(tab.to_string(index=False));print(comp.to_string(index=False));print(json.dumps(execution,indent=2))

def plot(tab,comp,pred,fold,ids):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.2),layout='constrained')
    ax=axes[0];cols=['#888888']*6+['#5B4B8A'];x=np.arange(7)
    low=tab.bootstrap_r_ci_low.to_numpy();high=tab.bootstrap_r_ci_high.to_numpy();r=tab.r_loocv.to_numpy()
    for f in range(7):
        ax.plot([f,f],[low[f],high[f]],color=cols[f],lw=2)
        ax.scatter(f,r[f],color=cols[f],s=55,zorder=3)
        ax.annotate(f'{r[f]:+.2f}',(f,r[f]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(0,color='#bbbbbb',lw=1);ax.set_xticks(x,['Visual','Somato-\nmotor','Dorsal\nattention','Ventral\nattention','Limbic','Control','DMN'])
    ax.set_ylabel('Held-out predictive correlation (signed r)');ax.set_title('A   Seven intact Yeo7 networks',loc='left',fontweight='bold')
    ax.text(0,-.23,'Intervals: 95% full-pipeline stratified participant bootstrap\nPrimary inference: nested permutations, Holm across seven models',transform=ax.transAxes,fontsize=9)
    ax=axes[1];names=['Visual','Somatomotor','Dorsal attention','Ventral attention / salience','Limbic','Control'];positions=np.arange(6)
    for f in positions:
        ax.plot([comp.bootstrap_delta_ci_low.iloc[f],comp.bootstrap_delta_ci_high.iloc[f]],[f,f],color='#5B4B8A',lw=2)
        ax.scatter(comp.delta_r_Default_minus_other.iloc[f],f,color='#5B4B8A',s=45,zorder=3)
    ax.axvline(0,color='#bbbbbb',lw=1);ax.set_yticks(positions,names);ax.invert_yaxis()
    ax.set_xlabel('DMN minus comparator predictive correlation (Δr)');ax.set_title('B   Paired DMN comparisons',loc='left',fontweight='bold')
    ax.text(0,-.23,'Intervals: paired 95% full-pipeline bootstrap\nSuperiority tests: approximate centered bootstrap, Holm across six',transform=ax.transAxes,fontsize=9)
    fig.savefig(HERE/'Figure_Yeo7_CPM_learning.png',dpi=240);fig.savefig(HERE/'Figure_Yeo7_CPM_learning.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7.3),layout='constrained')
    for f,ax in enumerate(axes.flat):
        if f==7:ax.axis('off');continue
        for group,c in [('SS','#D98C5F'),('DS','#5B4B8A')]:
            ix=ids.group.eq(group).to_numpy();ax.scatter(fold[ix],pred[ix,f],color=c,s=28,label=group,edgecolor='white',lw=.4)
        slope,intercept,*_=stats.linregress(fold,pred[:,f]);line=np.array([fold.min(),fold.max()]);ax.plot(line,intercept+slope*line,color='#555555',lw=1)
        ax.set_title(LABELS[f],fontsize=10);ax.set_xlabel('Observed learning\n(fold-centered ms)');ax.set_ylabel('Held-out prediction (ms)')
        ax.text(.03,.97,f'r={tab.r_loocv.iloc[f]:+.2f}\nHolm p={tab.p_Holm7_equal_tail.iloc[f]:.3f}',transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','edgecolor':'#dddddd','alpha':.9})
        if f==0:ax.legend(frameon=False,loc='lower right',fontsize=8)
    fig.savefig(HERE/'Figure_Yeo7_CPM_scatter.png',dpi=240);fig.savefig(HERE/'Figure_Yeo7_CPM_scatter.pdf');plt.close(fig)

if __name__=='__main__':main()
