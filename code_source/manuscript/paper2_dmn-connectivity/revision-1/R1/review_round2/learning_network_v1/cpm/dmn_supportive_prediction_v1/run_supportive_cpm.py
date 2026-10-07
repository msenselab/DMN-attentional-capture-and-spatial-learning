#!/usr/bin/env python3
"""Six intact-DMN supportive CPMs; all writes confined to this directory."""
import os,sys
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,hashlib,time,argparse
import numpy as np,pandas as pd
from scipy import stats
P=Path(__file__).resolve().parent;R=P.parents[2];RES=R/'resolved_reanalysis'
CONTEXTS=[(o,s) for o in ['capture50','learning'] for s in ['Pooled','SS','DS']]
INPUTS=[RES/'extraction_common_coverage/primary_participant_matrices.npy',RES/'extraction_common_coverage/primary_matrix_subjects.csv',RES/'common399_analysis_ready/physical_yeo7_common399_parcels.csv',R.parents[1]/'attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv',R/'location_standardized_capture_v1/participant_companion/participant_50_50_scores.csv',R/'learning_network_v1/learning_target_scores.csv',P.parent/'yeo7_within_v1/archive.npz',P.parent/'yeo7_within_v1/observed_predictions.csv',P/'ANALYSIS_SPEC.md',P/'run_supportive_cpm.py']
BPERM,BBOOT=10000,5000

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def hashes(ps):return {str(p.resolve()):sha(p) for p in ps}
def audit(vals):
    for p,h in vals.items():assert sha(p)==h,('hash',p)
def holm(ps):
    ps=np.asarray(ps);ix=np.argsort(ps);z=np.empty_like(ps);z[ix]=np.minimum(1,np.maximum.accumulate(ps[ix]*np.arange(len(ps),0,-1)));return z

def fit(X,Y,W,groups,details=False):
    """Batched identity-excluded weighted LOOCV; Y,W both N x draws."""
    n,b=Y.shape;pred=np.zeros((n,b));target=np.empty((n,b));counts=np.zeros((n,b,3),np.int16) if details else None;X2=X*X
    for test in range(n):
        tw=W.copy();tw[test]=0;nt=tw.sum(0);yc=Y.copy()
        for g in np.unique(groups):
            ids=groups==g;wg=tw[ids].sum(0);assert (wg>0).all()
            mu=(tw[ids]*Y[ids]).sum(0)/wg;yc[ids]-=mu
            if groups[test]==g:target[test]=Y[test]-mu
        ym=(tw*yc).sum(0)/nt;cy=yc-ym;vy=(tw*cy*cy).sum(0);assert (vy>0).all()
        wx=X.T@tw;xx=np.maximum(0,X2.T@tw-wx*wx/nt);cov=X.T@(tw*cy);den=np.sqrt(xx*vy)
        rr=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-14)
        tcrit=stats.t.isf(.005,nt-2);rcrit=tcrit/np.sqrt(tcrit*tcrit+nt-2)
        sg=np.where(np.abs(rr)>rcrit,np.sign(rr),0.);strength=X@sg;sm=(tw*strength).sum(0)/nt;cs=strength-sm;ss=(tw*cs*cs).sum(0)
        slope=np.divide((tw*cy*cs).sum(0),ss,out=np.zeros(b),where=ss>1e-14)
        pred[test]=ym+slope*(strength[test]-sm);pred[test,ss<=1e-14]=0.
        if details:counts[test,:,0]=(sg>0).sum(0);counts[test,:,1]=(sg<0).sum(0);counts[test,:,2]=ss<=1e-14
    total=W.sum(0);tc=target-(W*target).sum(0)/total;pc=pred-(W*pred).sum(0)/total;den=np.sqrt((W*tc*tc).sum(0)*(W*pc*pc).sum(0))
    r=np.divide((W*tc*pc).sum(0),den,out=np.zeros(b),where=den>1e-14)
    return r,pred,target,counts

def synthetic():
    rng=np.random.default_rng(32);X=rng.normal(size=(12,20));y=rng.normal(size=12);g=np.array(['DS']*6+['SS']*6);w=np.array([2,1,0,1,1,1,1,2,1,0,1,1])
    for ids in [np.arange(12),np.arange(6),np.arange(6,12)]:
        x=X[ids];z=y[ids];gg=g[ids];ww=w[ids];_,p,t,c=fit(x,z[:,None],ww[:,None],gg,True);expected=[]
        for test in range(len(ids)):
            copies=np.repeat(np.arange(len(ids)),ww);train=copies[copies!=test];yy=z[train].copy()
            for label in np.unique(gg):yy[gg[train]==label]-=yy[gg[train]==label].mean()
            result=stats.pearsonr(x[train],yy[:,None],axis=0);signs=np.where(result.pvalue<.01,np.sign(result.statistic),0);s=x[train]@signs
            if np.sum((s-s.mean())**2)<=1e-14:expected.append(0.)
            else:beta=np.linalg.lstsq(np.c_[np.ones(len(train)),s],yy,rcond=None)[0];expected.append(beta@np.array([1.,x[test]@signs]))
        assert np.allclose(expected,p[:,0],atol=1e-10,rtol=0)
    return {'status':'PASS_SYNTHETIC_NOT_EMPIRICAL_EVIDENCE','explicit_replicated_rows_match_weighted_all_identity_exclusion':True,'pooled_and_both_single_group_fits_exercised':True}

def load():
    ids=pd.read_csv(INPUTS[1])[['subject','group']];assert ids.subject.is_unique and ids.group.value_counts().to_dict()=={'DS':17,'SS':16}
    meta=pd.read_csv(INPUTS[2]).sort_values('matrix_index');assert meta.matrix_index.to_list()==list(range(399)) and set(meta.native_label)==set(range(1,401))-{323}
    assert np.array_equal(meta.yeo7_network,meta.physical_label.str.split('_').str[2]);loc=np.flatnonzero(meta.yeo7_network=='Default');assert len(loc)==91
    mats=np.load(INPUTS[0],mmap_mode='r');assert mats.shape==(33,399,399) and np.isfinite(mats).all() and np.allclose(mats,mats.swapaxes(1,2),atol=1e-12,rtol=0)
    a,b=np.triu_indices(91,1);X=np.array(mats[:,loc[a],loc[b]]);assert X.shape==(33,4095)
    tr=pd.read_csv(INPUTS[3]);assert len(tr)==10995 and tr.correct.eq(1).all() and tr.RT_ms.gt(200).all() and tr.RT_ms.le(3000).all()
    idx=pd.MultiIndex.from_frame(ids);biased=tr[tr.category=='biased'];assert len(biased)==6721
    cells=biased.groupby(['subject','group','DisRegion']).RT_ms.mean().unstack().reindex(idx);absent=tr[tr.category=='absent'].groupby(['subject','group']).RT_ms.mean().reindex(idx)
    Y=np.c_[.5*(cells.Frequent+cells.Rare)-absent,cells.Rare-cells.Frequent]
    cap=pd.read_csv(INPUTS[4]).set_index(['subject','group']).loc[idx];learning=pd.read_csv(INPUTS[5]).set_index(['subject','group']).loc[idx]
    assert np.allclose(Y[:,0],cap.B50_ms,atol=1e-10,rtol=0) and np.allclose(Y[:,1],learning.learning_rare_minus_frequent_ms,atol=1e-10,rtol=0) and np.allclose(X.mean(1),cap.dmn_conn_primary,atol=1e-12,rtol=0)
    out=ids.copy();out['dmn_mean_Fisher_z']=X.mean(1);out['capture50_ms']=Y[:,0];out['learning_ms']=Y[:,1];out.to_csv(P/'participant_targets.csv',index=False)
    pd.DataFrame({'edge_index':np.arange(4095),'matrix_index_a':loc[a],'matrix_index_b':loc[b],'native_label_a':meta.native_label.iloc[loc[a]].to_numpy(),'native_label_b':meta.native_label.iloc[loc[b]].to_numpy()}).to_csv(P/'candidate_edges.csv',index=False)
    return ids,X,Y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--batch-size',type=int,default=100);args=ap.parse_args()
    if args.self_test:
        result=synthetic();(P/'synthetic_algorithm_test.json').write_text(json.dumps(result,indent=2));print(result);return
    started=time.monotonic();frozen=hashes(INPUTS);old=hashes(f for f in P.parent.rglob('*') if f.is_file() and not f.is_relative_to(P));manifest={'inputs':frozen,'protected_old_cpm':old,'contexts':CONTEXTS,'permutations':BPERM,'bootstrap':BBOOT,'permutation_seed':20261004,'bootstrap_seed':20261006,'screen_p':.01,'batch_size':args.batch_size,'numpy':np.__version__}
    (P/'input_manifest.json').write_text(json.dumps(manifest,indent=2));ids,X,Y=load();g=ids.group.to_numpy();ds=np.flatnonzero(g=='DS');ss=np.flatnonzero(g=='SS');observed=np.zeros(6);rows=[];features=[]
    for ci,(outcome,scope) in enumerate(CONTEXTS):
        ix=np.arange(33) if scope=='Pooled' else np.flatnonzero(g==scope);y=Y[ix,ci//3];r,pred,target,count=fit(X[ix],y[:,None],np.ones((len(ix),1)),g[ix],True);observed[ci]=r[0]
        for k,i in enumerate(ix):
            rows.append({'outcome':outcome,'scope':scope,'subject':ids.subject.iloc[i],'group':g[i],'raw_target_ms':y[k],'fold_target_ms':target[k,0],'predicted_centered_ms':pred[k,0],'predicted_raw_ms':pred[k,0]+y[k]-target[k,0]})
            features.append({'outcome':outcome,'scope':scope,'subject':ids.subject.iloc[i],'positive_edges':int(count[k,0,0]),'negative_edges':int(count[k,0,1]),'zero_strength':int(count[k,0,2]),'training_N':len(ix)-1})
    predictions=pd.DataFrame(rows);selections=pd.DataFrame(features);predictions.to_csv(P/'observed_predictions.csv',index=False);selections.to_csv(P/'observed_selection_counts.csv',index=False)
    rng=np.random.default_rng(20261004);maps=np.empty((BPERM,33),np.int16)
    for k in range(BPERM):maps[k,ds]=rng.permutation(ds);maps[k,ss]=rng.permutation(ss)
    permstate=rng.bit_generator.state;rng=np.random.default_rng(20261006);weights=np.zeros((BBOOT,33),np.int16)
    for k in range(BBOOT):
        samples=np.r_[rng.choice(ds,len(ds),replace=True),rng.choice(ss,len(ss),replace=True)];weights[k]=np.bincount(samples,minlength=33);assert np.count_nonzero(weights[k,ds])>=2 and np.count_nonzero(weights[k,ss])>=2
    bootstate=rng.bit_generator.state
    np.savez_compressed(P/'sources.npz',permutation_maps=maps,bootstrap_weights=weights)
    null=np.empty((BPERM,6));boot=np.empty((BBOOT,6))
    for ci,(outcome,scope) in enumerate(CONTEXTS):
        ix=np.arange(33) if scope=='Pooled' else np.flatnonzero(g==scope);y=Y[:,ci//3]
        for label,total,dest in [('permutation',BPERM,null),('bootstrap',BBOOT,boot)]:
            for start in range(0,total,args.batch_size):
                stop=min(total,start+args.batch_size)
                if label=='permutation':yy=y[maps[start:stop][:,ix]].T;w=np.ones_like(yy)
                else:yy=np.broadcast_to(y[ix,None],(len(ix),stop-start));w=weights[start:stop][:,ix].T.astype(float)
                dest[start:stop,ci]=fit(X[ix],yy,w,g[ix])[0]
                if stop%2000==0:print(outcome,scope,label,stop,'/',total,flush=True)
    assert np.isfinite(null).all() and np.isfinite(boot).all()
    oldar=np.load(INPUTS[6]);assert np.array_equal(maps,oldar['permutation_maps']) and np.array_equal(weights,oldar['bootstrap_weights'])
    assert np.allclose(null[:,3],oldar['permutation_r'][:,-1],atol=1e-11,rtol=0) and np.allclose(boot[:,3],oldar['bootstrap_r'][:,-1],atol=1e-11,rtol=0)
    oldp=pd.read_csv(INPUTS[7]);sub=predictions[(predictions.outcome=='learning')&(predictions.scope=='Pooled')];assert sub.subject.to_list()==oldp.subject.to_list() and np.allclose(sub.predicted_centered_ms,oldp.pred_Default_ms,atol=1e-10,rtol=0) and np.allclose(sub.fold_target_ms,oldp.fold_target_ms,atol=1e-10,rtol=0)
    np.savez_compressed(P/'archive.npz',observed_r=observed,permutation_maps=maps,bootstrap_weights=weights,permutation_r=null,bootstrap_r=boot)
    up=(null>=observed).sum(0);down=(null<=observed).sum(0);peq=np.minimum(1,2*np.minimum((up+1)/(BPERM+1),(down+1)/(BPERM+1)));pabs=(1+(abs(null)>=abs(observed)).sum(0))/(BPERM+1);heq=holm(peq);habs=holm(pabs);results=[]
    for ci,(outcome,scope) in enumerate(CONTEXTS):
        z=predictions[(predictions.outcome==outcome)&(predictions.scope==scope)];sel=selections[(selections.outcome==outcome)&(selections.scope==scope)];constant=bool(z.predicted_centered_ms.var()==0)
        results.append({'outcome':outcome,'scope':scope,'N':len(z),'candidate_edges':4095,'r_loocv':observed[ci],'constant_prediction_r_zero_fallback':constant,'upper_exceedances':int(up[ci]),'lower_exceedances':int(down[ci]),'p_equal_tail_raw':peq[ci],'p_equal_tail_Holm6':heq[ci],'p_absolute_raw_sensitivity':pabs[ci],'p_absolute_Holm6_sensitivity':habs[ci],'bootstrap_r_ci_low':np.quantile(boot[:,ci],.025),'bootstrap_r_ci_high':np.quantile(boot[:,ci],.975),'rmse_ms':np.sqrt(np.mean((z.predicted_centered_ms-z.fold_target_ms)**2)),'zero_baseline_rmse_ms':np.sqrt(np.mean(z.fold_target_ms**2)),'zero_strength_folds':int(sel.zero_strength.sum()),'positive_edges_median':float(sel.positive_edges.median()),'negative_edges_median':float(sel.negative_edges.median()),'corrected_positive_prediction':bool(observed[ci]>0 and heq[ci]<.05),'repeat_of_previous_pooled_model':bool(outcome=='learning' and scope=='Pooled')})
    tab=pd.DataFrame(results);tab.to_csv(P/'prediction_results.csv',index=False)
    delta=np.array([observed[1]-observed[2],observed[4]-observed[5]]);bd=np.c_[boot[:,1]-boot[:,2],boot[:,4]-boot[:,5]];pg=(1+(abs(bd-delta)>=abs(delta)).sum(0))/(BBOOT+1);hg=holm(pg)
    group=[]
    for k,outcome in enumerate(['capture50','learning']):group.append({'outcome':outcome,'delta_r_SS_minus_DS':delta[k],'bootstrap_ci_low':np.quantile(bd[:,k],.025),'bootstrap_ci_high':np.quantile(bd[:,k],.975),'p_raw_centered_bootstrap_approx':pg[k],'p_Holm2_centered_bootstrap_approx':hg[k]})
    pd.DataFrame(group).to_csv(P/'group_performance_comparisons.csv',index=False)
    cov=np.cov(bd,rowvar=False);inverse=np.linalg.pinv(cov);T=delta@inverse@delta;cb=bd-delta;ts=np.einsum('bi,ij,bj->b',cb,inverse,cb);rank=np.linalg.matrix_rank(cov)
    (P/'group_omnibus.json').write_text(json.dumps({'method':'approximate centered-bootstrap Wald across two predictive-r group differences','rank':int(rank),'T':float(T),'p_raw':float((1+(ts>=T).sum())/(BBOOT+1)),'p_chi2_sensitivity':float(stats.chi2.sf(T,rank)),'covariance':cov.tolist()},indent=2))
    np.savez_compressed(P/'group_archive.npz',observed_delta=delta,bootstrap_delta=bd)
    (P/'archive_manifest.json').write_text(json.dumps({'files':hashes([P/'archive.npz',P/'sources.npz',P/'group_archive.npz']),'permutation_rng_end_state':permstate,'bootstrap_rng_end_state':bootstate,'Bperm':BPERM,'Bboot':BBOOT},indent=2))
    audit(frozen);audit(old)
    (P/'execution.json').write_text(json.dumps({'status':'COMPLETED_INDEPENDENT_VERIFICATION_PENDING','seconds':time.monotonic()-started,'models':6,'permutations':BPERM,'bootstrap':BBOOT,'inputs_unchanged':True,'protected_old_cpm_files':len(old),'old_cpm_unchanged':True,'old_pooled_learning_observed_and_every_null_and_bootstrap_replayed':True,'python':sys.version,'numpy':np.__version__},indent=2))
    print(tab.to_string(index=False));print(pd.DataFrame(group).to_string(index=False))
if __name__=='__main__':main()
