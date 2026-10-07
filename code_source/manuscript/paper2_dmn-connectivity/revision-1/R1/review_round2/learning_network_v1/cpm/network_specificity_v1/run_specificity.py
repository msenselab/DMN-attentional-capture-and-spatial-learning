#!/usr/bin/env python3
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy import stats
P=Path(__file__).resolve().parent; ROOT=P.parents[2]; RES=ROOT/'resolved_reanalysis'
NETWORKS=['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']
CONTEXTS=[(o,s) for o in ['capture50','learning','capture_original'] for s in ['Pooled','SS','DS']]
INPUTS=[RES/'extraction_common_coverage/primary_participant_matrices.npy', RES/'extraction_common_coverage/primary_matrix_subjects.csv',RES/'common399_analysis_ready/physical_yeo7_common399_parcels.csv',ROOT.parents[1]/'attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv',ROOT/'location_standardized_capture_v1/participant_companion/participant_50_50_scores.csv',ROOT/'learning_network_v1/learning_target_scores.csv',RES/'network_diagnostics/results_common399_n33/yeo7_network_summary_scores_n33.csv',P/'ANALYSIS_SPEC.md',Path(__file__)]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def holm(ps):
    ps=np.asarray(ps); order=np.argsort(ps); z=np.minimum(1,np.maximum.accumulate(ps[order]*np.arange(len(ps),0,-1))); out=np.empty(len(ps));out[order]=z;return out

def load():
    order=pd.read_csv(INPUTS[1])[['subject','group']]; groups=order.group.to_numpy(); assert order.subject.is_unique and order.group.value_counts().to_dict()=={'DS':17,'SS':16}
    meta=pd.read_csv(INPUTS[2]).sort_values('matrix_index'); assert meta.matrix_index.tolist()==list(range(399)) and set(meta.native_label)==set(range(1,401))-{323}
    mats=np.load(INPUTS[0]); assert mats.shape==(33,399,399) and np.isfinite(mats).all() and np.allclose(mats,mats.transpose(0,2,1))
    i,j=np.triu_indices(399,1); labels=meta.yeo7_network.to_numpy(); edgecounts=[]; X=[]
    for net in NETWORKS:
        mask=(labels[i]==net)&(labels[j]==net); edgecounts.append(int(mask.sum()));X.append(mats[:,i[mask],j[mask]].mean(1))
    assert edgecounts==[1830,2926,1035,1081,300,1326,4095];X=np.array(X).T
    saved=pd.read_csv(INPUTS[6]).set_index(['subject','group']).loc[pd.MultiIndex.from_frame(order)]
    assert np.allclose(X,saved[[n+'--'+n for n in NETWORKS]],atol=1e-12,rtol=0)
    trials=pd.read_csv(INPUTS[3]);assert len(trials)==10995 and (trials.correct==1).all() and trials.RT_ms.gt(200).all() and trials.RT_ms.le(3000).all()
    idx=pd.MultiIndex.from_frame(order); b=trials[trials.category=='biased'];assert len(b)==6721
    cells=b.groupby(['subject','group','DisRegion']).RT_ms.mean().unstack().reindex(idx)
    absent=trials[trials.category=='absent'].groupby(['subject','group']).RT_ms.mean().reindex(idx)
    by=b.groupby(['subject','group']).RT_ms.mean().reindex(idx)
    Y=np.array([(.5*(cells.Frequent+cells.Rare)-absent).to_numpy(),(cells.Rare-cells.Frequent).to_numpy(),(by-absent).to_numpy()]).T
    cap=pd.read_csv(INPUTS[4]).set_index(['subject','group']).loc[idx];learn=pd.read_csv(INPUTS[5]).set_index(['subject','group']).loc[idx]
    assert np.allclose(Y[:,0],cap.B50_ms,atol=1e-10,rtol=0) and np.allclose(Y[:,1],learn.learning_rare_minus_frequent_ms,atol=1e-10,rtol=0) and np.allclose(X[:,-1],cap.dmn_conn_primary,atol=1e-12,rtol=0)
    df=order.copy()
    for k,n in enumerate(NETWORKS):df[n]=X[:,k]
    for k,o in enumerate(['capture50','learning','capture_original']):df[o]=Y[:,k]
    df.to_csv(P/'participant_scores.csv',index=False)
    return X,Y,groups,edgecounts

def correlations(x,y,scope,groups):
    # Explicit resampled-row centering, broadcasting arbitrary leading draw axis.
    a=x.copy();b=y.copy()
    if scope=='Pooled':
        for g in ['DS','SS']:
            ids=np.flatnonzero(groups==g);a[...,ids,:]-=a[...,ids,:].mean(axis=-2,keepdims=True);b[...,ids,:]-=b[...,ids,:].mean(axis=-2,keepdims=True)
    else:
        ids=np.flatnonzero(groups==scope);a=a[...,ids,:];b=b[...,ids,:];a-=a.mean(axis=-2,keepdims=True);b-=b.mean(axis=-2,keepdims=True)
    return np.einsum('...nk,...nm->...mk',a,b)/np.sqrt(np.einsum('...nk,...nk->...k',a,a)[...,None,:]*np.einsum('...nm,...nm->...m',b,b)[...,:,None])

def all_r(x,y,groups):
    rs=[correlations(x,y,s,groups) for s in ['Pooled','SS','DS']]
    return np.stack([rs[s][...,o,:] for o in range(3) for s in range(3)],axis=-2)

def williams(a,b,c,n):
    det=1-a*a-b*b-c*c+2*a*b*c
    denom=2*det*(n-1)/(n-3)+.25*(a+b)**2*(1-c)**3
    assert denom>0
    t=(a-b)*np.sqrt((n-1)*(1+c)/denom)
    return t,2*stats.t.sf(abs(t),n-3)

def main():
    old={str(f):digest(f) for f in P.parent.rglob('*') if f.is_file() and not f.is_relative_to(P)}
    frozen={str(f):digest(f) for f in INPUTS};(P/'input_manifest.json').write_text(json.dumps({'inputs':frozen,'protected_old_cpm':old,'networks':NETWORKS,'contexts':CONTEXTS,'n_bootstrap':20000,'n_permutation':10000,'bootstrap_seed':20261007,'permutation_seed':20261008},indent=2))
    X,Y,groups,counts=load(); observed=all_r(X,Y,groups); boot=[];maps=[];rng=np.random.default_rng(20261007)
    for start in range(0,20000,1000):
        source=np.tile(np.arange(33),(1000,1))
        for g in ['DS','SS']:
            ids=np.flatnonzero(groups==g);source[:,ids]=rng.choice(ids,size=(1000,len(ids)),replace=True)
        boot.append(all_r(X[source],Y[source],groups));maps.append(source)
    boot=np.concatenate(boot);bmap=np.concatenate(maps);prng=np.random.default_rng(20261008);pmap=np.tile(np.arange(33),(10000,1))
    for k in range(10000):
        for g in ['DS','SS']:
            ids=np.flatnonzero(groups==g);pmap[k,ids]=prng.permutation(ids)
    null=all_r(np.broadcast_to(X,(10000,33,7)),Y[pmap],groups)
    assert np.isfinite(boot).all() and np.isfinite(null).all()
    bd=boot[:,:,-1,None]-boot[:,:,:6];od=observed[:,-1,None]-observed[:,:6]
    moderation=np.stack([od[o*3+1]-od[o*3+2] for o in range(3)])
    bm=np.stack([bd[:,o*3+1]-bd[:,o*3+2] for o in range(3)],axis=1)
    np.savez_compressed(P/'draw_archive.npz',bootstrap_sources=bmap,permutation_sources=pmap,bootstrap_r=boot,permutation_r=null,observed_r=observed,bootstrap_delta=bd,observed_delta=od,bootstrap_moderation=bm,observed_moderation=moderation)
    assoc=[];comp=[]
    for ci,(o,s) in enumerate(CONTEXTS):
        n=33 if s=='Pooled' else int((groups==s).sum()); ne=n-1 if s=='Pooled' else n; df=ne-2
        ids=np.arange(33) if s=='Pooled' else np.flatnonzero(groups==s); z=X[ids].copy()
        if s=='Pooled':
            for g in ['DS','SS']:z[groups==g]-=z[groups==g].mean(0)
        c=np.corrcoef(z.T)[-1,:6]
        for k,net in enumerate(NETWORKS):
            r=observed[ci,k];t=r*np.sqrt(df/(1-r*r));p=2*stats.t.sf(abs(t),df);lo,hi=np.tanh(np.arctanh(r)+np.array([-1,1])*stats.norm.ppf(.975)/np.sqrt(ne-3));bl,bh=np.quantile(boot[:,ci,k],[.025,.975])
            assoc.append(dict(outcome=o,scope=s,network=net,N=n,r=r,t=t,df=df,p_raw=p,permutation_p_raw=(1+np.sum(abs(null[:,ci,k])>=abs(r)))/10001,ci_low=lo,ci_high=hi,bootstrap_ci_low=bl,bootstrap_ci_high=bh))
        for k,net in enumerate(NETWORKS[:6]):
            t,p=williams(observed[ci,-1],observed[ci,k],c[k],ne);bl,bh=np.quantile(bd[:,ci,k],[.025,.975]);bp=(1+np.sum(abs(bd[:,ci,k]-od[ci,k])>=abs(od[ci,k])))/20001
            comp.append(dict(outcome=o,scope=s,comparator=net,N=n,r_DMN=observed[ci,-1],r_other=observed[ci,k],r_DMN_other=c[k],delta_r=od[ci,k],t_Williams=t,df=ne-3,p_raw=p,bootstrap_ci_low=bl,bootstrap_ci_high=bh,bootstrap_p_raw=bp))
    a=pd.DataFrame(assoc);c=pd.DataFrame(comp)
    for tab,pc in [(a,['p_raw','permutation_p_raw']),(c,['p_raw','bootstrap_p_raw'])]:
        for primary in [True,False]:
            mask=(tab.outcome!='capture_original')==primary
            for col in pc:tab.loc[mask,col.replace('_raw','_Holm')]=holm(tab.loc[mask,col])
            tab.loc[mask,'family']=('primary' if primary else 'sensitivity')+' Holm-'+str(mask.sum())
    a.to_csv(P/'network_associations.csv',index=False);c.to_csv(P/'dmn_comparisons.csv',index=False)
    mr=[];om=[];covs=[]
    for o,outcome in enumerate(['capture50','learning','capture_original']):
        cov=np.cov(bm[:,o,:],rowvar=False);covs.append(cov);pinv=np.linalg.pinv(cov);gamma=moderation[o];rank=np.linalg.matrix_rank(cov);T=gamma@pinv@gamma;center=bm[:,o,:]-gamma;Ts=np.einsum('bi,ij,bj->b',center,pinv,center)
        om.append(dict(outcome=outcome,rank=int(rank),T=T,p_raw=(1+np.sum(Ts>=T))/20001,p_chi2_sensitivity=stats.chi2.sf(T,rank)))
        for k,n in enumerate(NETWORKS[:6]):
            lo,hi=np.quantile(bm[:,o,k],[.025,.975]);p=(1+np.sum(abs(bm[:,o,k]-gamma[k])>=abs(gamma[k])))/20001
            mr.append(dict(outcome=outcome,comparator=n,delta_r_SS=od[o*3+1,k],delta_r_DS=od[o*3+2,k],gamma_SS_minus_DS=gamma[k],bootstrap_ci_low=lo,bootstrap_ci_high=hi,p_raw=p))
    m=pd.DataFrame(mr);v=pd.DataFrame(om)
    for tab in [m,v]:
        for primary in [True,False]:
            mask=(tab.outcome!='capture_original')==primary;tab.loc[mask,'p_Holm']=holm(tab.loc[mask,'p_raw']);tab.loc[mask,'family']=('primary' if primary else 'sensitivity')+' Holm-'+str(mask.sum())
    m.to_csv(P/'group_moderation.csv',index=False);v.to_csv(P/'group_omnibus.csv',index=False);np.save(P/'moderation_covariances.npy',np.array(covs))
    assert all(digest(Path(f))==h for f,h in frozen.items()) and all(digest(Path(f))==h for f,h in old.items())
    manifest={f.name:digest(f) for f in P.iterdir() if f.is_file() and f.suffix in ['.csv','.npz','.npy']};(P/'output_manifest.json').write_text(json.dumps(manifest,indent=2))
    (P/'execution.json').write_text(json.dumps({'status':'PASS','N':33,'SS':16,'DS':17,'eligible_trials':10995,'biased_trials':6721,'all_inputs_unchanged':True,'protected_cpm_files_unchanged':len(old),'bootstrap_draws':20000,'permutation_draws':10000,'primary_associations':42,'primary_direct_comparisons':36,'primary_moderation_tests':12,'primary_omnibus_tests':2,'edges':counts},indent=2))
    print(v.to_string(index=False));print(c[c.outcome!='capture_original'].to_string(index=False))
if __name__=='__main__':main()
