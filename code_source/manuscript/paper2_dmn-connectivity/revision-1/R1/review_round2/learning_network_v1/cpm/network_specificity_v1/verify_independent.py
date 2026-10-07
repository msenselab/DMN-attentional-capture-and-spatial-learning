#!/usr/bin/env python3
"""Independent weighted-moment replay; never imports the analysis runner."""
import os,sys
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.stats import t as tdist, norm, chi2
P=Path(__file__).resolve().parent; R=P.parents[2];RES=R/'resolved_reanalysis'
nets=['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']; outs=['capture50','learning','capture_original']; scopes=['Pooled','SS','DS']
errors={}
def eq(key,a,b,tol=2e-10):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(key,a.shape,b.shape)
    diff=float(np.max(np.abs(a-b))) if a.size else 0.;errors[key]=diff;assert np.isfinite(a).all() and np.isfinite(b).all() and diff<=tol,(key,diff)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def adjust(p):
    p=list(p); seq=sorted(range(len(p)),key=lambda k:p[k]);answer=[0.]*len(p);largest=0.
    for rank,k in enumerate(seq):largest=max(largest,(len(p)-rank)*p[k]);answer[k]=min(1.,largest)
    return np.array(answer)
def moments(X,Y,g,W):
    # Distinct-subject multiplicities: no replicated-row centering code.
    groupstats={}
    for label in ['DS','SS']:
        ids=np.flatnonzero(g==label); w=W[:,ids]; x=X[ids]-X[ids].mean(0);y=Y[ids]-Y[ids].mean(0); n=w.sum(1)
        sx=w@x;sy=w@y
        xx=w@(x*x)-sx*sx/n[:,None]; yy=w@(y*y)-sy*sy/n[:,None]
        xy=(w@np.einsum('nk,nm->nmk',x,y).reshape(len(ids),21)).reshape(len(W),3,7)-sy[:,:,None]*sx[:,None,:]/n[:,None,None]
        groupstats[label]=(xx,yy,xy)
    res={}
    for s in scopes:
        if s=='Pooled':parts=[groupstats['DS'],groupstats['SS']];xx=sum(z[0] for z in parts);yy=sum(z[1] for z in parts);xy=sum(z[2] for z in parts)
        else:xx,yy,xy=groupstats[s]
        res[s]=xy/np.sqrt(yy[:,:,None]*xx[:,None,:])
    return np.stack([res[s][:,o] for o in range(3) for s in scopes],axis=1)
def main():
    man=json.loads((P/'input_manifest.json').read_text())
    for name,h in man['inputs'].items():assert sha(name)==h,('input',name)
    for name,h in man['protected_old_cpm'].items():assert sha(name)==h,('old cpm',name)
    for name,h in json.loads((P/'output_manifest.json').read_text()).items():assert sha(P/name)==h,('output',name)
    subjects=pd.read_csv(RES/'extraction_common_coverage/primary_matrix_subjects.csv')[['subject','group']];g=subjects.group.to_numpy();assert list(subjects.group.value_counts().sort_index())==[17,16]
    meta=pd.read_csv(RES/'common399_analysis_ready/physical_yeo7_common399_parcels.csv').sort_values('matrix_index');assert len(meta)==399 and 323 not in meta.native_label.to_list()
    matrices=np.load(RES/'extraction_common_coverage/primary_participant_matrices.npy');X=np.empty((33,7));counts=[]
    for k,n in enumerate(nets):
        loc=np.flatnonzero(meta.yeo7_network.to_numpy()==n);size=len(loc);counts.append(size*(size-1)//2)
        block=matrices[:,loc[:,None],loc];X[:,k]=(block.sum(axis=(1,2))-np.trace(block,axis1=1,axis2=2))/(size*(size-1))
    assert counts==[1830,2926,1035,1081,300,1326,4095]
    trials=pd.read_csv(R.parents[1]/'attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv');assert len(trials)==10995
    assert np.all((trials.correct==1)&(trials.RT_ms>200)&(trials.RT_ms<=3000))
    values=defaultdict(list)
    for row in trials.itertuples():
        if row.category=='absent':values[(row.subject,'A')].append(row.RT_ms)
        elif row.category=='biased':values[(row.subject,'F' if row.DisRegion=='Frequent' else 'R')].append(row.RT_ms)
    Y=[]
    for sid in subjects.subject:
        f=values[(sid,'F')];r=values[(sid,'R')];a=np.mean(values[(sid,'A')]);Y.append([.5*(np.mean(f)+np.mean(r))-a,np.mean(r)-np.mean(f),np.mean(f+r)-a])
    Y=np.asarray(Y);saved=pd.read_csv(P/'participant_scores.csv');assert saved.subject.tolist()==subjects.subject.tolist();eq('network_reconstruction',X,saved[nets]);eq('target_reconstruction',Y,saved[outs]);assert sum(len(v) for (s,c),v in values.items() if c!='A')==6721
    ar=np.load(P/'draw_archive.npz'); rng=np.random.default_rng(20261007);generated=[]
    for start in range(0,20000,1000):
        z=np.broadcast_to(np.arange(33),(1000,33)).copy()
        for label in ['DS','SS']:
            ids=np.flatnonzero(g==label);z[:,ids]=rng.choice(ids,(1000,len(ids)),replace=True)
        generated.append(z)
    sources=np.vstack(generated);assert np.array_equal(sources,ar['bootstrap_sources'])
    ob=moments(X,Y,g,np.ones((1,33)))[0];eq('observed_r',ob,ar['observed_r']);br=[]
    for start in range(0,20000,500):
        z=sources[start:start+500];weights=np.zeros((len(z),33));np.add.at(weights,(np.repeat(np.arange(len(z)),33),z.ravel()),1);br.append(moments(X,Y,g,weights))
    br=np.concatenate(br);eq('all_bootstrap_r',br,ar['bootstrap_r'])
    rng=np.random.default_rng(20261008);perm=[]
    for draw in range(10000):
        z=np.arange(33)
        for label in ['DS','SS']:
            ids=np.flatnonzero(g==label);z[ids]=rng.permutation(ids)
        perm.append(z)
    perm=np.asarray(perm);assert np.array_equal(perm,ar['permutation_sources']);nr=[]
    for start in range(0,10000,500):
        py=Y[perm[start:start+500]]; grouped={}
        for scope in scopes:
            ids=np.arange(33) if scope=='Pooled' else np.flatnonzero(g==scope);x=X[ids].copy();y=py[:,ids].copy()
            for label in (['DS','SS'] if scope=='Pooled' else [scope]):
                mask=g[ids]==label;x[mask]-=x[mask].mean(0);y[:,mask]-=y[:,mask].mean(1,keepdims=True)
            cross=np.einsum('nk,bnm->bmk',x,y);den=np.sqrt((y*y).sum(1)[:,:,None]*(x*x).sum(0)[None,None,:]);grouped[scope]=cross/den
        nr.append(np.stack([grouped[s][:,o] for o in range(3) for s in scopes],axis=1))
    nr=np.concatenate(nr);eq('all_permutation_r',nr,ar['permutation_r'])
    bd=br[:,:,6:7]-br[:,:,:6];od=ob[:,6:7]-ob[:,:6];eq('all_bootstrap_delta',bd,ar['bootstrap_delta']);eq('observed_delta',od,ar['observed_delta'])
    associations=pd.read_csv(P/'network_associations.csv');comparisons=pd.read_csv(P/'dmn_comparisons.csv');assert len(associations)==63 and len(comparisons)==54
    for ci,(outcome,scope) in enumerate([(o,s) for o in outs for s in scopes]):
        n=32 if scope=='Pooled' else (16 if scope=='SS' else 17);df=n-2
        for k,net in enumerate(nets):
            row=associations.iloc[ci*7+k];assert (row.outcome,row.scope,row.network)==(outcome,scope,net)
            r=ob[ci,k];test=r*np.sqrt(df)/np.sqrt(1-r*r);ci_bounds=np.tanh(norm.interval(.95,loc=np.arctanh(r),scale=1/np.sqrt(n-3)))
            eq('association_'+str(ci)+'_'+net,[row.r,row.t,row.df,row.p_raw,row.permutation_p_raw,row.ci_low,row.ci_high,row.bootstrap_ci_low,row.bootstrap_ci_high],[r,test,df,tdist.sf(abs(test),df)*2,(1+(abs(nr[:,ci,k])>=abs(r)).sum())/10001,*ci_bounds,*np.percentile(br[:,ci,k],[2.5,97.5])])
        ids=np.arange(33) if scope=='Pooled' else np.flatnonzero(g==scope);x=X[ids].copy()
        if scope=='Pooled':
            for label in ['DS','SS']:x[g==label]-=np.mean(x[g==label],0)
        cross=np.corrcoef(x.T)
        for k,net in enumerate(nets[:6]):
            row=comparisons.iloc[ci*6+k];a=ob[ci,6];b=ob[ci,k];c=cross[6,k];det=np.linalg.det([[1,a,b],[a,1,c],[b,c,1]])
            denominator=np.sqrt(2*det*(n-1)/(n-3)+(a+b)**2*(1-c)**3/4);stat=(a-b)*np.sqrt((n-1)*(1+c))/denominator
            bp=(1+(abs(bd[:,ci,k]-od[ci,k])>=abs(od[ci,k])).sum())/20001
            eq('comparison_'+str(ci)+'_'+net,[row.r_DMN,row.r_other,row.r_DMN_other,row.delta_r,row.t_Williams,row.df,row.p_raw,row.bootstrap_ci_low,row.bootstrap_ci_high,row.bootstrap_p_raw],[a,b,c,a-b,stat,n-3,2*tdist.sf(abs(stat),n-3),*np.percentile(bd[:,ci,k],[2.5,97.5]),bp])
    for table,fields in [(associations,['p_raw','permutation_p_raw']),(comparisons,['p_raw','bootstrap_p_raw'])]:
        for primary in [True,False]:
            mask=(table.outcome!='capture_original')==primary
            for col in fields:eq('Holm_'+col+'_'+str(len(table))+'_'+str(primary),adjust(table.loc[mask,col]),table.loc[mask,col.replace('_raw','_Holm')])
    om=pd.read_csv(P/'group_omnibus.csv');moder=pd.read_csv(P/'group_moderation.csv');assert len(om)==3 and len(moder)==18
    covsave=np.load(P/'moderation_covariances.npy');bm=np.stack([bd[:,o*3+1]-bd[:,o*3+2] for o in range(3)],axis=1);og=np.stack([od[o*3+1]-od[o*3+2] for o in range(3)]);eq('all_moderation_draws',bm,ar['bootstrap_moderation']);eq('observed_moderation',og,ar['observed_moderation'])
    for o in range(3):
        centered=bm[:,o]-bm[:,o].mean(0);cov=centered.T@centered/19999;eq('covariance_'+str(o),cov,covsave[o]);inv=np.linalg.inv(cov);T=float(og[o]@inv@og[o]);null=bm[:,o]-og[o];Ts=np.sum((null@inv)*null,axis=1);row=om.iloc[o];eq('omnibus_'+str(o),[row.T if False else row['T'],row.p_raw,row.p_chi2_sensitivity],[T,(1+(Ts>=T).sum())/20001,chi2.sf(T,6)],tol=2e-9)
        for k in range(6):
            row=moder.iloc[o*6+k];eq('moderation_'+str(o)+'_'+str(k),[row.delta_r_SS,row.delta_r_DS,row.gamma_SS_minus_DS,row.bootstrap_ci_low,row.bootstrap_ci_high,row.p_raw],[od[o*3+1,k],od[o*3+2,k],og[o,k],*np.percentile(bm[:,o,k],[2.5,97.5]),(1+(abs(bm[:,o,k]-og[o,k])>=abs(og[o,k])).sum())/20001])
    for table in [om,moder]:
        for primary in [True,False]:
            mask=(table.outcome!='capture_original')==primary;eq('moderationHolm_'+str(len(table))+'_'+str(primary),adjust(table.loc[mask,'p_raw']),table.loc[mask,'p_Holm'])
    (P/'independent_verification.json').write_text(json.dumps({'status':'PASS','no_generator_import':True,'method':'independent trial list means, symmetric block matrix means, multiplicity-weighted centered moment bootstrap replay','all_bootstrap_draws_verified':20000,'all_permutations_verified':10000,'all_associations_verified':63,'all_direct_comparisons_verified':54,'all_group_moderation_verified':18,'all_omnibus_verified':3,'all_input_output_and_old_cpm_hashes_pass':True,'maximum_numerical_error':max(errors.values()),'errors':errors},indent=2));print('PASS',max(errors.values()))
if __name__=='__main__':main()
