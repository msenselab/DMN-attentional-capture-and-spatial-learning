#!/usr/bin/env python3
"""Independent residual-OLS and projection-matrix verification of new outputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
O=Path(__file__).resolve().parent
S=O.parent/'resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv'
s=pd.read_csv(S).sort_values('subject').reset_index(drop=True)
a=pd.read_csv(O/'participant_associations_physical.csv')
c=pd.read_csv(O/'dependent_correlation_comparisons.csv')
d=pd.read_csv(O/'descriptive_primary_covariance_decomposition.csv')
b=pd.read_csv(O/'paired_participant_bootstrap_draws.csv')
I=pd.read_csv(O/'participant_bootstrap_indices.csv').to_numpy(int)
checks=[]
def ck(name,v):
    assert bool(v),name
    checks.append(name)
G=(s.group=='DS').to_numpy(float)
Z=np.column_stack([np.ones(33),G])
H=np.eye(33)-Z@np.linalg.inv(Z.T@Z)@Z.T
fields=['dmn_conn_primary','biased_capture','same_capture','diff_capture']
V=s[fields].to_numpy(float); W=H@V
rs=[]
for j,f in enumerate(fields[1:]):
    fit=sm.OLS(s[f],np.column_stack([np.ones(33),G,s.dmn_conn_primary])).fit()
    r=stats.pearsonr(W[:,0],W[:,j+1]).statistic;rs.append(r)
    ck(f+'_correlation',np.isclose(r,a.iloc[j].r,atol=1e-12))
    ck(f+'_group_OLS_df',fit.df_resid==a.iloc[j].df==30)
    ck(f+'_group_OLS_t',np.isclose(fit.tvalues.iloc[2],a.iloc[j].t,atol=1e-10))
    ck(f+'_group_OLS_p',np.isclose(fit.pvalues.iloc[2],a.iloc[j].p_raw,atol=1e-12))
    ci=np.tanh(np.arctanh(r)+stats.norm.ppf(.975)/np.sqrt(29)*np.array([-1,1]))
    ck(f+'_Fisher_CI',np.allclose(ci,a.iloc[j][['fisher95_low','fisher95_high']].to_numpy(float),atol=1e-12))
ck('new_physical_Holm2',np.allclose(multipletests(a.p_raw.iloc[1:],method='holm')[1],a.p_Holm_new_physical2.iloc[1:]))
# Independent all-draw group residualization by a 33x33 projection matrix.
Gb=np.r_[np.zeros(16),np.ones(17)]
Zb=np.column_stack([np.ones(33),Gb]);Hb=np.eye(33)-Zb@np.linalg.inv(Zb.T@Zb)@Zb.T
Wb=np.einsum('ij,rjk->rik',Hb,V[I])
rboot=np.sum(Wb[:,:,0,None]*Wb[:,:,1:],axis=1)/np.sqrt(np.sum(Wb[:,:,0]**2,axis=1)[:,None]*np.sum(Wb[:,:,1:]**2,axis=1))
ck('all_10000_paired_bootstrap_correlations',np.allclose(rboot,b.iloc[:,:3].to_numpy(float),atol=1e-12))
ck('all_bootstrap_group_membership',all(set(s.group.iloc[I[k,:16]])=={'SS'} and set(s.group.iloc[I[k,16:]])=={'DS'} for k in range(len(I))))
for j,k in [(0,1),(0,2),(1,2)]:
    i=[(0,1),(0,2),(1,2)].index((j,k)); C=np.corrcoef(W[:,[0,j+1,k+1]],rowvar=False)
    x,y,z=C[0,1],C[0,2],C[1,2];n=32
    denominator=np.sqrt(2*np.linalg.det(C)*(n-1)/(n-3)+(x+y)**2*(1-z)**3/4)
    t=(x-y)*np.sqrt((n-1)*(1+z))/denominator
    p=2*stats.t.sf(abs(t),n-3)
    ck(f'comparison_{i}_delta',np.isclose(x-y,c.iloc[i].delta_r,atol=1e-12))
    ck(f'comparison_{i}_Williams',np.isclose(t,c.iloc[i].t,atol=1e-10) and np.isclose(p,c.iloc[i].p_raw,atol=1e-12))
    ci=np.quantile(rboot[:,j]-rboot[:,k],[.025,.975])
    ck(f'comparison_{i}_participant_percentile',np.allclose(ci,c.iloc[i][['participant_percentile95_low','participant_percentile95_high']].to_numpy(float),atol=1e-12))
ck('comparison_Holm3',np.allclose(multipletests(c.p_raw,method='holm')[1],c.p_Holm3))
# Independently use OLS-residual sum of cross-products, one shared denominator.
SS=s.group=='SS';DS=s.group=='DS';den=np.linalg.norm(W[:,0])*np.linalg.norm(W[:,1])
contrib=np.array([np.dot(W[SS,0],W[SS,1]),np.dot(W[DS,0],W[DS,1])])/den
ck('both_group_covariance_contributions',np.allclose(contrib,d.contribution_to_primary_r,atol=1e-12))
ck('contributions_sum_to_primary_r',np.isclose(contrib.sum(),rs[0],atol=1e-12))
ck('shares_are_covariance_not_variance',np.allclose(contrib/contrib.sum(),d.covariance_share,atol=1e-12))
report=dict(status='PASS',checks_passed=len(checks),checks=checks,all_10000_bootstrap_draws_verified=True,
            input_hash=hashlib.sha256(S.read_bytes()).hexdigest(),verifier_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            interval_test_disagreement='Biased-versus-color pointwise participant percentile interval excludes zero, but declared Williams raw/Holm3 test is not significant. Do not select an inferential procedure after seeing this discrepancy.')
(O/'independent_participant_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
