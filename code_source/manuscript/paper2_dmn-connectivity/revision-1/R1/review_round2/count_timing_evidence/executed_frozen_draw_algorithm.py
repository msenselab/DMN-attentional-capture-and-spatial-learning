#!/usr/bin/env python3
"""Authorized correct+RT-valid statistics, retaining run-first trials."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[6])) + '/manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior');P=Path(__file__).resolve().parent/'rerun_outputs';P.mkdir(parents=True,exist_ok=True);SEED=20261001;R=10000
s=pd.read_csv(SRC/'participant_scores_n33.csv');d=pd.read_csv(SRC/'correct_rt_trials_n33.csv')
g=s.group.to_numpy();groups=[np.flatnonzero(g==q) for q in ['SS','DS']]
x=s.dmn_conn.to_numpy(float);B=s.biased_capture.to_numpy(float);U=s.unbiased_capture.to_numpy(float)
def center(z,idx=groups):
 z=np.array(z,float,copy=True)
 if z.ndim==1:
  for ix in idx:z[ix]-=z[ix].mean()
 else:
  for ix in idx:z[:,ix]-=z[:,ix].mean(axis=1,keepdims=True)
 return z
def corr(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);a=a-a.mean(axis=-1,keepdims=True);b=b-b.mean(axis=-1,keepdims=True)
 return np.sum(a*b,axis=-1)/np.sqrt(np.sum(a*a,axis=-1)*np.sum(b*b,axis=-1))
def rc(a,b,k=0):
 r=float(corr(a,b));n=len(a);df=n-k-2;t=r*np.sqrt(df/(1-r*r));p=float(2*stats.t.sf(abs(t),df));z=np.arctanh(r);ci=np.tanh(z+np.array([-1,1])*stats.norm.ppf(.975)/np.sqrt(n-k-3))
 return dict(N=n,r=r,t=t,df=df,p_raw=p,ci_low=ci[0],ci_high=ci[1],CI_method='Fisher z approximate; effective n=N-k')
def partial(z,covariates):
 C=np.column_stack([np.ones(len(z))]+[z[c].to_numpy(float) for c in covariates]);assert np.linalg.matrix_rank(C)==C.shape[1]
 a=z.dmn_conn.to_numpy(copy=True);b=z.biased_capture.to_numpy(copy=True);a-=C@np.linalg.lstsq(C,a,rcond=None)[0];b-=C@np.linalg.lstsq(C,b,rcond=None)[0]
 return rc(a,b,len(covariates))
def will(a,b,c,n=32):
 det=1-a*a-b*b-c*c+2*a*b*c;t=(a-b)*np.sqrt((n-1)*(1+c))/np.sqrt(2*det*(n-1)/(n-3)+(a+b)**2/4*(1-c)**3)
 return dict(delta_r=a-b,t=t,df=n-3,p_raw=2*stats.t.sf(abs(t),n-3),r_FB=a,r_FU=b,r_BU=c,n_eff=n)
xc=center(x);Bc=center(B);Uc=center(U)
rows=[]
for outcome in ['biased_capture','unbiased_capture','all_capture','planned_weight_capture']:
 rows.append(dict(analysis='centered Pearson',outcome=outcome,scope='pooled',**rc(xc,center(s[outcome]))))
for cov,label in [(['D'],'group only'),(['unbiased_capture','D'],'unbiased+group'),(['biased_priming','D'],'priming+group')]:
 rows.append(dict(analysis='partial '+label,outcome='biased_capture',scope='pooled',**partial(s.copy(),cov)))
for group,z in s.groupby('group'):
 rows.append(dict(analysis='Pearson',outcome='biased_capture',scope=group,**rc(z.dmn_conn,z.biased_capture)))
 rows.append(dict(analysis='partial unbiased',outcome='biased_capture',scope=group,**partial(z.copy(),['unbiased_capture'])))
a=pd.DataFrame(rows);a['p_Holm']=np.nan
mask=~((a.analysis=='centered Pearson')&(a.outcome=='biased_capture'))
a.loc[mask,'p_Holm']=multipletests(a.loc[mask,'p_raw'],method='holm')[1]
a['Holm_family']=np.where(mask,'10 exploratory associations in this table','primary; no adjustment')
a.to_csv(P/'associations.csv',index=False)
w=will(float(corr(xc,Bc)),float(corr(xc,Uc)),float(corr(Bc,Uc)));pd.DataFrame([w]).to_csv(P/'williams.csv',index=False)
# Stratified paired participant percentile bootstrap, recenter each draw.
rng=np.random.default_rng(SEED)
ix=np.column_stack([rng.choice(q,size=(R,len(q)),replace=True) for q in groups])
# This sampled table places SS then DS; define group positions accordingly.
pos=[np.arange(len(groups[0])),np.arange(len(groups[0]),33)]
xb=center(x[ix],pos);bb=center(B[ix],pos);ub=center(U[ix],pos);ab=center(s.all_capture.to_numpy()[ix],pos)
bootB=corr(xb,bb);bootU=corr(xb,ub);bootA=corr(xb,ab);bootDelta=bootB-bootU
perm=np.empty(R)
for k in range(R):
 y=B.copy()
 for q in groups:y[q]=rng.permutation(y[q])
 perm[k]=corr(xc,center(y))
permp=(1+np.sum(np.abs(perm)>=abs(corr(xc,Bc))))/(R+1)
pd.DataFrame(dict(iteration=np.arange(1,R+1),bootstrap_primary_r=bootB,bootstrap_unbiased_r=bootU,bootstrap_all_r=bootA,bootstrap_delta_r=bootDelta,permuted_primary_r=perm)).to_csv(P/'participant_resampling_draws.csv',index=False)
res=[]
for name,point,draw in [('biased r',corr(xc,Bc),bootB),('unbiased r',corr(xc,Uc),bootU),('all r',corr(xc,center(s.all_capture)),bootA),('delta biased-unbiased r',w['delta_r'],bootDelta)]:
 lo,hi=np.quantile(draw,[.025,.975]);res.append(dict(estimand=name,point=point,ci_low=lo,ci_high=hi,CI_method='95% percentile stratified paired participant bootstrap; not BCa',R=R,seed=SEED,p_permutation=permp if name=='biased r' else np.nan))
pd.DataFrame(res).to_csv(P/'participant_resampling_summary.csv',index=False)
loo=[]
for i in range(len(s)):
 z=s.drop(index=i).copy()
 for col in ['dmn_conn','biased_capture']:z[col+'_c']=z[col]-z.groupby('group')[col].transform('mean')
 loo.append(dict(omitted=s.subject.iloc[i],**rc(z.dmn_conn_c,z.biased_capture_c)))
pd.DataFrame(loo).to_csv(P/'leave_one_out.csv',index=False)
# Observed retained-weight decomposition.
ac=center(s.all_capture.to_numpy());den=np.std(xc,ddof=1)*np.std(ac,ddof=1)
parts=[]
for cat in ['biased','unbiased']:
 z=s[cat+'_weight'].to_numpy()*s[cat+'_capture'].to_numpy();zc=center(z);q=np.cov(xc,zc,ddof=1)[0,1]/den
 parts.append(dict(component=cat,standardized_covariance_contribution=q,share_of_all_r=q/corr(xc,ac)))
assert abs(sum(q['standardized_covariance_contribution'] for q in parts)-corr(xc,ac))<1e-10
pd.DataFrame(parts).to_csv(P/'all_capture_decomposition.csv',index=False)
s.groupby('group').biased_weight.agg(['mean','min','max']).to_csv(P/'retained_weight_summary.csv')
# Recorded odd/even global trial parity; same phase and inclusion as capture.
d['half']=np.where(d.global_order.mod(2).eq(1),'odd','even');hh=[]
for (subject,half),z in d.groupby(['subject','half']):
 ar=z.loc[z.category.eq('absent'),'RT_ms'].mean()
 for name,mask in [('biased',z.category.eq('biased')),('unbiased',z.category.eq('unbiased')),('physical_color',z.dimension.eq('Different'))]:
  hh.append(dict(subject=subject,group=z.group.iloc[0],half=half,score=name,capture=z.loc[mask,'RT_ms'].mean()-ar))
h=pd.DataFrame(hh).pivot(index=['subject','group','score'],columns='half',values='capture').reset_index();h.to_csv(P/'split_half_scores.csv',index=False)
rels=[];reldr=[]
for score in ['biased','unbiased','physical_color']:
 z=h.loc[h.score.eq(score)].sort_values('subject').copy();assert len(z)==33 and z[['odd','even']].notna().all().all()
 qg=[np.flatnonzero(z.group.to_numpy()==q) for q in ['SS','DS']];odd=z.odd.to_numpy();even=z.even.to_numpy()
 zz=rc(center(odd,qg),center(even,qg));idx=np.column_stack([rng.choice(q,size=(R,len(q)),replace=True) for q in qg]);positions=[np.arange(len(qg[0])),np.arange(len(qg[0]),33)]
 rboot=corr(center(odd[idx],positions),center(even[idx],positions));SB=2*rboot/(1+rboot);lo,hi=np.quantile(rboot,[.025,.975]);slo,shi=np.quantile(SB,[.025,.975])
 rels.append(dict(score=score,**zz,SB=2*zz['r']/(1+zz['r']),bootstrap_r_low=lo,bootstrap_r_high=hi,bootstrap_SB_low=slo,bootstrap_SB_high=shi,bootstrap_method='10,000 stratified paired participant percentile intervals; group means recalculated',half_definition='original recorded global scanner-trial odd/even parity; RT+correct gate; first retained'))
 reldr.extend(dict(score=score,iteration=k+1,r=rboot[k],SB=SB[k]) for k in range(R))
pd.DataFrame(rels).to_csv(P/'capture_reliability.csv',index=False);pd.DataFrame(reldr).to_csv(P/'capture_reliability_bootstrap.csv',index=False)
# Trial count matching and unbiased within-participant bootstrap: fixed retained absent baseline.
bdraw=np.empty((R,33));udraw=np.empty((R,33))
for col,subject in enumerate(s.subject):
 z=d.loc[d.subject.eq(subject)];ba=z.loc[z.category.eq('biased'),'RT_ms'].to_numpy();ua=z.loc[z.category.eq('unbiased'),'RT_ms'].to_numpy();ar=z.loc[z.category.eq('absent'),'RT_ms'].mean();assert len(ba)>=len(ua)
 for k in range(R):bdraw[k,col]=rng.choice(ba,len(ua),replace=False).mean()-ar
 udraw[:,col]=ua[rng.integers(len(ua),size=(R,len(ua)))].mean(axis=1)-ar
np.save(P/'matched_biased_score_draws.npy',bdraw)
bdcorr=corr(np.broadcast_to(xc,(R,33)),center(bdraw));udcorr=corr(np.broadcast_to(xc,(R,33)),center(udraw));bdp=2*stats.t.sf(abs(bdcorr*np.sqrt(31/(1-bdcorr**2))),31);udp=2*stats.t.sf(abs(udcorr*np.sqrt(31/(1-udcorr**2))),31)
counts=[]
for label,z,pv in [('biased downsampling',bdcorr,bdp),('unbiased trial bootstrap',udcorr,udp)]:
 lo,hi=np.quantile(z,[.025,.975]);counts.append(dict(procedure=label,median_r=np.median(z),mean_r=np.mean(z),ci_low=lo,ci_high=hi,positive_fraction=np.mean(z>0),nominal_p_under_05_fraction=np.mean(pv<.05),CI_method='95% percentile conditional trial-resampling interval; not participant-sampling CI',R=R,seed=SEED))
 for group,q in zip(['SS','DS'],groups):
  zz=corr(np.broadcast_to(x[q],(R,len(q))),bdraw[:,q] if label=='biased downsampling' else udraw[:,q]);lo,hi=np.quantile(zz,[.025,.975]);counts.append(dict(procedure=label+' '+group,median_r=np.median(zz),ci_low=lo,ci_high=hi,CI_method='95% percentile conditional trial-resampling interval',R=R,seed=SEED))
pd.DataFrame(counts).to_csv(P/'trial_resampling_summary.csv',index=False)
pd.DataFrame(dict(iteration=np.arange(1,R+1),biased_downsample_r=bdcorr,biased_downsample_p=bdp,unbiased_bootstrap_r=udcorr,unbiased_bootstrap_p=udp)).to_csv(P/'trial_resampling_draws.csv',index=False)
s['count_matched_biased_mean']=bdraw.mean(axis=0);s.to_csv(P/'participant_scores_with_downsampling.csv',index=False)
wb=will(float(corr(xc,center(bdraw.mean(axis=0)))),float(corr(xc,Uc)),float(corr(center(bdraw.mean(axis=0)),Uc)))
wt=pd.DataFrame([dict(procedure='original correct+RT estimates',**w),dict(procedure='count-matched mean estimate',**wb)]);wt['p_Holm']=multipletests(wt.p_raw,method='holm')[1];wt['Holm_family']='two exploratory Williams comparisons';wt.to_csv(P/'williams_full_and_countmatched.csv',index=False)
# Chronology/exposure-adjusted subject OLS; recorded pre-current exposures.
ests=[];history=[]
for subject,z in d.groupby('subject'):
 for mode,history_cols in [('basic',['prior_biased','prior_unbiased']),('location_specific',['prior_biased_frequent','prior_biased_rare','prior_unbiased_frequent','prior_unbiased_rare'])]:
  z=z.copy();cols=['B','U'];z['run2']=z.run.eq(2).astype(int);cols+=['run2']
  for col in ['log_order']+history_cols:
   name='z_'+col;z[name]=(z[col]-z[col].mean())/z[col].std(ddof=1);cols.append(name)
  C=sm.add_constant(z[cols].astype(float));fit=sm.OLS(z.RT_ms,C).fit();assert np.linalg.matrix_rank(C)==len(cols)+1
  ests.append(dict(subject=subject,group=z.group.iloc[0],mode=mode,dmn_conn=z.dmn_conn.iloc[0],biased_capture=fit.params['B'],unbiased_capture=fit.params['U'],bias_se=fit.bse['B'],unbias_se=fit.bse['U'],n=len(z),df=fit.df_resid,condition_number=np.linalg.cond(C)))
e=pd.DataFrame(ests);e.to_csv(P/'history_subject_coefficients.csv',index=False)
for mode,z in e.groupby('mode'):
 for outcome in ['biased_capture','unbiased_capture']:
  xx=z.dmn_conn-z.groupby('group').dmn_conn.transform('mean');yy=z[outcome]-z.groupby('group')[outcome].transform('mean');history.append(dict(mode=mode,outcome=outcome,**rc(xx,yy)))
 joined=z.merge(s[['subject','biased_capture']],on='subject',suffixes=('_adjusted','_raw'))
 history.append(dict(mode=mode,outcome='raw-versus-adjusted biased capture',**rc(joined.biased_capture_adjusted,joined.biased_capture_raw)))
 z=z.copy();z['D']=z.group.eq('DS').astype(int);z['X']=(z.dmn_conn-z.dmn_conn.mean())/z.dmn_conn.std(ddof=1)
 C=sm.add_constant(pd.DataFrame(dict(X=z.X,D=z.D,XD=z.X*z.D)));f=sm.OLS(z.biased_capture,C).fit();ci=f.conf_int().loc['XD'];history.append(dict(mode=mode,outcome='group moderation DS-SS',N=33,b_ms=f.params['XD'],se_ms=f.bse['XD'],t=f.tvalues['XD'],df=f.df_resid,p_raw=f.pvalues['XD'],ci_low=ci.iloc[0],ci_high=ci.iloc[1],CI_method='OLS t interval'))
hist=pd.DataFrame(history);hist['p_Holm']=multipletests(hist.p_raw,method='holm')[1];hist.to_csv(P/'history_associations.csv',index=False)
# Figure3 uses authorized scores and matching adjusted coefficients, not old figures.
fig,axs=plt.subplots(1,2,figsize=(10,4.5));colors={'SS':'#D98C5F','DS':'#5B4B8A'}
for ax,mode,label in zip(axs,[None,'basic'],['A','B']):
 z=s[['subject','group','dmn_conn','biased_capture']].copy() if mode is None else e.loc[e['mode'].eq(mode),['subject','group','dmn_conn','biased_capture']].copy()
 xx=z.dmn_conn-z.groupby('group').dmn_conn.transform('mean');yy=z.biased_capture-z.groupby('group').biased_capture.transform('mean')
 for group in ['SS','DS']:
  ix=z.group.eq(group);ax.scatter(xx[ix],yy[ix],color=colors[group],label=group,s=35,alpha=.85)
 fit=sm.OLS(yy,sm.add_constant(xx)).fit();grid=np.linspace(xx.min(),xx.max(),200);pred=fit.get_prediction(sm.add_constant(grid)).summary_frame();ax.plot(grid,pred['mean'],color='#444444');ax.fill_between(grid,pred.mean_ci_lower,pred.mean_ci_upper,color='#888888',alpha=.2)
 ax.set_xlabel('Within-group-centered DMN connectivity');ax.set_ylabel('Within-group-centered biased capture (ms)');ax.set_title('Raw capture' if mode is None else 'Trial-history-adjusted capture');ax.text(-.12,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=14);ax.spines[['top','right']].set_visible(False)
axs[0].legend(frameon=False);fig.tight_layout();fig.savefig(P/'Figure3_correct_RT.png',dpi=300);fig.savefig(P/'Figure3_correct_RT.pdf');plt.close(fig)
summary=dict(primary=a.iloc[0].to_dict(),permutation_p=permp,bootstrap_primary_CI=np.quantile(bootB,[.025,.975]).tolist(),LOO_r_range=[min(z['r'] for z in loo),max(z['r'] for z in loo)],LOO_p_range=[min(z['p_raw'] for z in loo),max(z['p_raw'] for z in loo)],williams=w,seed=SEED,R=R,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(P/'summary.json').write_text(json.dumps(summary,indent=2,default=float)+'\n');print(a.to_string(index=False));print(pd.DataFrame(res).to_string(index=False));print(pd.DataFrame(rels).to_string(index=False));print(pd.DataFrame(counts).to_string(index=False));print(hist.to_string(index=False))
