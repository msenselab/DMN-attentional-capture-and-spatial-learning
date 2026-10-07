#!/usr/bin/env python3
"""Exploratory same-cohort physical-dimension comparisons; no manuscript edits."""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent
ROUND = OUT.parent
REV = ROUND.parents[1]
SCORES = ROUND/'resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv'
TRIALS = REV/'attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv'
PRIMARY = ROUND/'resolved_reanalysis/inference/results_common399_n33_optimizer_resolved_v2/participant_associations.csv'
SEED, R = 20261004, 10000
EXPECTED = {SCORES:'cc3cddf8377ff9aefba18e8781d2512151e335f280d232c5b23d791266e6feef',
            TRIALS:'fac39301cb3b828f925937e080213bc5ed39fcaa03cbe0e6c558cf2ea18b34c6'}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def center(a,g):
    b=np.array(a,float,copy=True)
    for group in ('SS','DS'): b[g==group]-=b[g==group].mean(axis=0)
    return b

def correlation_stats(r):
    df=30; t=r*np.sqrt(df/(1-r*r)); se=1/np.sqrt(29)
    lo,hi=np.tanh(np.arctanh(r)+stats.norm.ppf(.975)*se*np.array([-1,1]))
    return dict(N=33,df=df,r=float(r),t=float(t),p_raw=float(2*stats.t.sf(abs(t),df)),fisher95_low=lo,fisher95_high=hi)

def williams(a,b,c):
    n_eff=32; df=29; det=1-a*a-b*b-c*c+2*a*b*c
    assert det>0
    t=(a-b)*np.sqrt((n_eff-1)*(1+c))/np.sqrt(2*det*(n_eff-1)/(n_eff-3)+(a+b)**2/4*(1-c)**3)
    return dict(delta_r=a-b,outcome_correlation=c,t=t,df=df,p_raw=float(2*stats.t.sf(abs(t),df)),
                method='Approximate Williams for one group control; n_eff=N-1')

for p,h in EXPECTED.items(): assert sha(p)==h,(p,sha(p))
s=pd.read_csv(SCORES).sort_values('subject').reset_index(drop=True)
t=pd.read_csv(TRIALS)
assert len(s)==33 and s.subject.is_unique and s.groupby('group').size().to_dict()=={'DS':17,'SS':16}
assert len(t)==10995 and t.subject.nunique()==33
assert (t.correct==1).all() and t.RT_ms.between(200,3000,inclusive='both').all()
assert not t.duplicated(['subject','run','trial']).any()
assert set(t.subject)==set(s.subject)
assert {'absent','biased','unbiased'}==set(t.category)
assert set(t.loc[t.category!='absent','dimension'])=={'Same','Different'}
assert (t.loc[(t.group=='SS') & (t.category=='biased'),'dimension']=='Same').all()
assert (t.loc[(t.group=='DS') & (t.category=='biased'),'dimension']=='Different').all()
means=t.groupby(['subject','category']).RT_ms.mean().unstack().reindex(s.subject)
physical=t.loc[t.category!='absent'].groupby(['subject','dimension']).RT_ms.mean().unstack().reindex(s.subject)
recon={'biased_capture':means.biased.to_numpy()-means.absent.to_numpy(),
       'unbiased_capture':means.unbiased.to_numpy()-means.absent.to_numpy(),
       'same_capture':physical.Same.to_numpy()-means.absent.to_numpy(),
       'diff_capture':physical.Different.to_numpy()-means.absent.to_numpy()}
errors={name:float(np.max(np.abs(v-s[name].to_numpy()))) for name,v in recon.items()}
assert max(errors.values())<1e-8
assert np.allclose(s.biased_capture,np.where(s.group=='SS',s.same_capture,s.diff_capture),atol=1e-10)
assert np.allclose(s.unbiased_capture,np.where(s.group=='SS',s.diff_capture,s.same_capture),atol=1e-10)

# Never use carried legacy X/gc/neural columns.
g=s.group.to_numpy(); cols=['dmn_conn_primary','biased_capture','same_capture','diff_capture']
a=s[cols].to_numpy(float); z=center(a,g); C=np.corrcoef(z,rowvar=False)
rs=C[0,1:]; assert abs(rs[0]-0.5503886479554411)<1e-12
p0=pd.read_csv(PRIMARY).query("scope=='pooled' and outcome=='biased_capture' and analysis=='group-centered Pearson df=N-3'").iloc[0]
assert abs(rs[0]-p0.r)<1e-12

# Paired participant bootstraps within groups; identical draws for all outcomes.
rng=np.random.default_rng(SEED); ids={group:np.flatnonzero(g==group) for group in ('SS','DS')}
I=np.concatenate([rng.choice(ids[group],size=(R,len(ids[group])),replace=True) for group in ('SS','DS')],axis=1)
A=a[I]; split=len(ids['SS'])
A[:,:split,:]-=A[:,:split,:].mean(axis=1,keepdims=True)
A[:,split:,:]-=A[:,split:,:].mean(axis=1,keepdims=True)
Sxx=np.sum(A[:,:,0]**2,axis=1)
Syy=np.sum(A[:,:,1:]**2,axis=1)
Sxy=np.sum(A[:,:,0,None]*A[:,:,1:],axis=1)
rr=Sxy/np.sqrt(Sxx[:,None]*Syy)
assert np.isfinite(rr).all()
labels=['assigned_biased','physical_orientation','physical_color']
rows=[]
for j,label in enumerate(labels):
    row=dict(outcome=label,**correlation_stats(rs[j]))
    row.update(zip(['participant_percentile95_low','participant_percentile95_high'],np.quantile(rr[:,j],[.025,.975])))
    row['status']='existing primary' if j==0 else 'new exploratory physical-score association'
    rows.append(row)
association=pd.DataFrame(rows)
association['p_Holm_new_physical2']=np.nan
association.loc[1:,'p_Holm_new_physical2']=multipletests(association.loc[1:,'p_raw'],method='holm')[1]
association['correction_family']='Original primary unchanged; only new orientation/color associations form Holm-2'
association.to_csv(OUT/'participant_associations_physical.csv',index=False)

pairs=[(0,1),(0,2),(1,2)]; contrasts=[]
for j,k in pairs:
    row=dict(comparison=labels[j]+' minus '+labels[k],**williams(rs[j],rs[k],C[j+1,k+1]))
    row.update(zip(['participant_percentile95_low','participant_percentile95_high'],np.quantile(rr[:,j]-rr[:,k],[.025,.975])))
    contrasts.append(row)
comparisons=pd.DataFrame(contrasts)
comparisons['p_Holm3']=multipletests(comparisons.p_raw,method='holm')[1]
comparisons['correction_family']='three new exploratory dependent-correlation comparisons'
comparisons.to_csv(OUT/'dependent_correlation_comparisons.csv',index=False)

# Partition the primary covariance, not causal mechanisms or explained variance.
x,b=z[:,0],z[:,1]; denom=np.sqrt(np.sum(x*x)*np.sum(b*b))
cross_total=float(np.sum(x*b)); contributions=[]
draw_contributions=np.column_stack([np.sum(A[:,:split,0]*A[:,:split,1],axis=1),np.sum(A[:,split:,0]*A[:,split:,1],axis=1)])
draw_r_contributions=draw_contributions/np.sqrt(Sxx*Syy[:,0])[:,None]
draw_shares=draw_contributions/draw_contributions.sum(axis=1)[:,None]
for j,group in enumerate(('SS','DS')):
    cp=float(np.sum(x[g==group]*b[g==group]))
    row=dict(group=group,physical_biased_dimension='orientation' if group=='SS' else 'color',N=len(ids[group]),
             centered_cross_product=cp,contribution_to_primary_r=cp/denom,covariance_share=cp/cross_total)
    row.update(zip(['r_contribution_percentile95_low','r_contribution_percentile95_high'],np.quantile(draw_r_contributions[:,j],[.025,.975])))
    row.update(zip(['share_percentile95_low','share_percentile95_high'],np.quantile(draw_shares[:,j],[.025,.975])))
    contributions.append(row)
decomp=pd.DataFrame(contributions)
assert abs(decomp.contribution_to_primary_r.sum()-rs[0])<1e-12
assert abs(decomp.covariance_share.sum()-1)<1e-12
assert np.allclose(draw_r_contributions.sum(axis=1),rr[:,0],atol=1e-12)
decomp.to_csv(OUT/'descriptive_primary_covariance_decomposition.csv',index=False)
boot=pd.DataFrame(rr,columns=[v+'_r' for v in labels])
for j,k in pairs:boot[labels[j]+'_minus_'+labels[k]]=rr[:,j]-rr[:,k]
for j,group in enumerate(('SS','DS')):
    boot[group+'_r_contribution']=draw_r_contributions[:,j]
    boot[group+'_covariance_share']=draw_shares[:,j]
boot.to_csv(OUT/'paired_participant_bootstrap_draws.csv',index=False)
pd.DataFrame(I,columns=['sampled_source_row_'+str(j) for j in range(33)]).to_csv(OUT/'participant_bootstrap_indices.csv',index=False)
s[['subject','group']+cols].assign(**{c+'_within_group_centered':z[:,j] for j,c in enumerate(cols)}).to_csv(OUT/'exact_analysis_scores.csv',index=False)

# Factor identifiability on present-trial cell definitions, not arbitrary correlation weights.
P=np.array([1,-1,1,-1]) # SS O, SS C, DS O, DS C
B=np.array([1,-1,-1,1]); G=np.array([1,1,-1,-1])
assert np.array_equal(P*B,G)
M=np.column_stack([np.ones(4),P,B,G,P*B,P*G,B*G,P*B*G])
identifiability=dict(present_cells=['SS_orientation_biased','SS_color_unbiased','DS_orientation_unbiased','DS_color_biased'],
                    group_equals_physical_times_bias_effect_code=True,full_three_factor_columns=8,observed_cell_rank=int(np.linalg.matrix_rank(M)),
                    interpretation='Physical-by-bias interaction is group-aliased; actual exposure remains bias-confounded; no causal dimension-versus-learning percentages.')
assert identifiability['observed_cell_rank']==4
(OUT/'identifiability.json').write_text(json.dumps(identifiability,indent=2)+'\n')
manifest=dict(status='PASS',N=33,trials=10995,groups={'SS':16,'DS':17},seed=SEED,bootstrap_participant_draws=R,
              input_hashes={str(p):sha(p) for p in [SCORES,TRIALS,PRIMARY,OUT/'ANALYSIS_SPEC.md',Path(__file__)]},
              reconstruction_max_errors=errors,primary_r_reproduced=float(rs[0]),primary_input_frozen=True,
              covariance_decomposition_sum_verified=True,bootstrap_denominator_near_zero_count=int((np.abs(draw_contributions.sum(axis=1))<1e-8).sum()),
              output_hashes={p.name:sha(p) for p in OUT.glob('*.csv')},python=platform.python_version(),
              limitations=['New exploratory comparisons, not original correction families','Group-adjusted Williams approximation; pointwise participant percentile intervals',
                           'Covariance shares are not variance explained or causal contributions','Physical scores mix biased and unbiased cells across groups','No manuscript edit or neural/model refit'])
(OUT/'participant_analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(association.to_string(index=False))
print('\nDIRECT DEPENDENT COMPARISONS\n'+comparisons.to_string(index=False))
print('\nDESCRIPTIVE COVARIANCE CONTRIBUTIONS\n'+decomp.to_string(index=False))
print('\nPASS: exact primary, raw score reconstruction, paired bootstrap decomposition, factor aliasing verified.')
