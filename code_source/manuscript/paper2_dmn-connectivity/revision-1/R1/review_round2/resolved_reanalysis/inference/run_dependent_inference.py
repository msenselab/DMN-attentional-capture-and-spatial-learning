#!/usr/bin/env python3
"""Dependent-analysis rerun for the resolved physical Yeo-7 extraction.

This script is intentionally inert until a completed input manifest is supplied.
It never reads the historical `dmn_conn` as a neural result and never calls CPM.
The only full mixed-model job is delegated once, sequentially, to the paired R script.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
HERE = Path(__file__).resolve().parent
RESOLVED = HERE.parent
EXTRACTION = RESOLVED / 'extraction_common_coverage'
BEHAVIOR_TRIALS = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv'
FROZEN_DRAWS = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/count_timing_evidence/rerun_outputs/matched_biased_score_draws.npy'
FROZEN_SUBJECTS = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/count_timing_evidence/rerun_outputs/participant_scores_with_downsampling.csv'
FROZEN_TRIAL_RESAMPLING = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/count_timing_evidence/rerun_outputs/trial_resampling_draws.csv'
SEED = 20261001
R = 10_000
N_COMMON_PARCELS = 399
EXCLUDED_NATIVE_LABELS = frozenset({323})
COMMON_NATIVE_LABELS = frozenset(range(1, 401)) - EXCLUDED_NATIVE_LABELS
COUNTS = {'Default':91,'SomMot':77,'Vis':61,'Cont':52,'SalVentAttn':47,'DorsAttn':46,'Limbic':25}
VARIANTS = ('primary','minimal_tr1','primary_gsr','primary_task','primary_task_gsr')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def fail(message: str) -> None:
    raise RuntimeError(message)

def ensure(condition: bool, message: str) -> None:
    if not condition: fail(message)

def correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a-a.mean(axis=-1,keepdims=True); b=b-b.mean(axis=-1,keepdims=True)
    den=np.sqrt(np.sum(a*a,axis=-1)*np.sum(b*b,axis=-1))
    if np.any(~np.isfinite(den)) or np.any(den == 0): fail('A correlation had zero/nonfinite variance.')
    return np.sum(a*b,axis=-1)/den

def group_indices(groups: np.ndarray) -> list[np.ndarray]:
    ensure(len(groups)==33 and set(groups)=={'SS','DS'}, 'Expected exactly SS/DS N=33.')
    out=[np.flatnonzero(groups==g) for g in ('SS','DS')]
    ensure([len(x) for x in out]==[16,17], 'Expected SS=16 and DS=17.')
    return out

def center_group(v: np.ndarray, idx: list[np.ndarray]) -> np.ndarray:
    a=np.array(v,float,copy=True)
    if a.ndim == 1:
        for q in idx: a[q] -= a[q].mean()
    elif a.ndim == 2:
        for q in idx: a[:,q] -= a[:,q].mean(axis=1,keepdims=True)
    else: fail('Only vectors or draw-by-participant arrays can be group-centered.')
    return a

def corr_stats(x: np.ndarray, y: np.ndarray, k: int = 0, label: str = '') -> dict:
    r=float(correlation(x,y)); n=len(x); df=n-k-2
    ensure(df > 0 and abs(r) < 1, 'Invalid correlation statistic '+label)
    t=r*np.sqrt(df/(1-r*r)); p=float(2*stats.t.sf(abs(t),df))
    z=np.arctanh(r); se=1/np.sqrt(n-k-3)
    lo,hi=np.tanh(z+np.array([-1,1])*stats.norm.ppf(.975)*se)
    return dict(analysis=label,N=n,r=r,t=t,df=df,p_raw=p,ci_low=float(lo),ci_high=float(hi),
                fisher_se=se,CI_method=f'Fisher z approximate; effective n=N-{k}')

def group_adjusted_corr(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Correlation after the one independent SS/DS group contrast is removed."""
    r=float(correlation(x,y)); n=len(x); df=n-3
    ensure(df>0 and abs(r)<1, 'Invalid group-adjusted correlation: '+label)
    t=r*np.sqrt(df/(1-r*r)); p=float(2*stats.t.sf(abs(t),df)); se=1/np.sqrt(n-4); z=np.arctanh(r)
    lo,hi=np.tanh(z+np.array([-1,1])*stats.norm.ppf(.975)*se)
    return dict(analysis=label,N=n,r=r,t=t,df=df,p_raw=p,ci_low=float(lo),ci_high=float(hi),fisher_se=se,
                CI_method='Fisher z approximate after one group contrast; df=N-3, SE=1/sqrt(N-4)')

def partial_corr(frame: pd.DataFrame, x: str, y: str, covariates: list[str], label: str) -> dict:
    C=np.column_stack([np.ones(len(frame))]+[frame[c].to_numpy(float) for c in covariates])
    ensure(np.linalg.matrix_rank(C)==C.shape[1], 'Rank-deficient covariate design: '+label)
    a=frame[x].to_numpy(float)-C@np.linalg.lstsq(C,frame[x].to_numpy(float),rcond=None)[0]
    b=frame[y].to_numpy(float)-C@np.linalg.lstsq(C,frame[y].to_numpy(float),rcond=None)[0]
    return corr_stats(a,b,len(covariates),label)

def williams(a: float,b: float,c: float,n_eff: int=32) -> dict:
    det=1-a*a-b*b-c*c+2*a*b*c
    ensure(det>0, 'Williams correlation matrix is not positive definite.')
    t=(a-b)*np.sqrt((n_eff-1)*(1+c))/np.sqrt(2*det*(n_eff-1)/(n_eff-3)+(a+b)**2/4*(1-c)**3)
    return dict(comparison='biased versus unbiased dependent correlations',r_primary_biased=a,r_primary_unbiased=b,
                r_biased_unbiased=c,delta_r=a-b,t=t,df=n_eff-3,p_raw=float(2*stats.t.sf(abs(t),n_eff-3)),
                n_eff=n_eff,method='Approximate Williams test; declared df=29')

def holm(frame: pd.DataFrame, column: str, family: str) -> pd.DataFrame:
    out=frame.copy(); out['p_Holm']=multipletests(out[column].to_numpy(float),method='holm')[1]; out['Holm_family']=family; return out

def read_manifest(path: Path) -> dict:
    ensure(path.exists(), f'Input manifest is absent: {path}. Extraction is not complete.')
    m=json.loads(path.read_text())
    ensure(m.get('status') == 'complete', 'Input manifest status must be exactly "complete".')
    return m

def relpath(value: str, manifest: Path) -> Path:
    p=Path(value); return p if p.is_absolute() else manifest.parent/p

def need_file(d: dict, key: str, manifest: Path) -> Path:
    ensure(key in d and 'path' in d[key], f'Manifest missing {key}.path')
    p=relpath(d[key]['path'],manifest); ensure(p.exists(), f'Manifest file missing: {p}'); return p

def load_array(path: Path, key: str | None) -> np.ndarray:
    z=np.load(path, mmap_mode='r', allow_pickle=False)
    if isinstance(z,np.lib.npyio.NpzFile):
        ensure(key is not None and key in z.files, f'NPZ needs manifest array_key; available={z.files}')
        return np.asarray(z[key],float)
    return np.asarray(z,float)

def validate_common_coverage_geometry(manifest: dict) -> None:
    """Reject a stale full-400 handoff before any behavioral or neural calculation."""
    geometry=manifest.get('common_coverage_geometry')
    if not isinstance(geometry,dict): fail('Manifest must contain common_coverage_geometry; full-400 handoffs are not accepted.')
    ensure(int(geometry.get('nominal_atlas_parcels',-1))==400, 'Geometry must declare nominal_atlas_parcels=400.')
    ensure(int(geometry.get('effective_common_coverage_parcels',-1))==N_COMMON_PARCELS, 'Geometry must declare effective_common_coverage_parcels=399.')
    ensure(set(geometry.get('excluded_native_labels',[]))==EXCLUDED_NATIVE_LABELS, 'Geometry must exclude exactly native label 323.')
    ensure(geometry.get('uniform_all_participants_and_runs') is True, 'Geometry must attest uniform all-participant/all-run common coverage.')

def load_inputs(manifest_path: Path) -> tuple[pd.DataFrame,np.ndarray,pd.DataFrame,pd.DataFrame,dict]:
    m=read_manifest(manifest_path)
    validate_common_coverage_geometry(m)
    scores_path=need_file(m,'participant_scores',manifest_path)
    matrix_path=need_file(m,'primary_matrices',manifest_path)
    subjects_path=need_file(m,'matrix_subjects',manifest_path)
    parcels_path=need_file(m,'parcel_metadata',manifest_path)
    runwise_path=need_file(m,'runwise_primary',manifest_path)
    s_cfg=m['participant_scores']; primary=s_cfg.get('primary_column')
    ensure(primary, 'Manifest must name participant_scores.primary_column; guessing network columns is forbidden.')
    variant_columns=s_cfg.get('variant_columns',{})
    ensure(set(variant_columns)==set(VARIANTS), 'Manifest variant_columns must name exactly '+', '.join(VARIANTS))
    raw=pd.read_csv(scores_path); required={'subject','group',primary,*variant_columns.values(),s_cfg.get('mean_fd_column','mean_fd')}
    ensure(required <= set(raw), 'participant_scores missing: '+', '.join(sorted(required-set(raw))))
    # Preserve the extracted primary explicitly even when it is also the primary
    # sensitivity column; never rely on duplicate-name rename semantics.
    s=raw[['subject','group']].copy()
    s['primary_default']=raw[primary].to_numpy()
    for v in VARIANTS: s[v]=raw[variant_columns[v]].to_numpy()
    s['mean_fd']=raw[s_cfg.get('mean_fd_column','mean_fd')].to_numpy()
    ensure({'subject','group','primary_default',*VARIANTS,'mean_fd'} <= set(s), 'Primary and variant columns must all resolve uniquely.')
    ensure(len(s)==33 and s.subject.is_unique and set(s.group)=={'SS','DS'}, 'Scores must have unique N=33 SS/DS subjects.')
    for c in ['primary_default',*VARIANTS,'mean_fd']:
        ensure(np.isfinite(s[c].to_numpy(float)).all(), 'Nonfinite predictor value: '+c)
    s=s.sort_values('subject').reset_index(drop=True)
    mat_subjects=pd.read_csv(subjects_path)
    ensure({'subject','group'} <= set(mat_subjects), 'matrix_subjects must include subject,group.')
    ensure(len(mat_subjects)==33 and mat_subjects.subject.is_unique, 'matrix_subjects must have unique 33 rows.')
    mats=load_array(matrix_path,m['primary_matrices'].get('array_key'))
    ensure(mats.shape==(33,N_COMMON_PARCELS,N_COMMON_PARCELS), f'Expected primary matrices (33,399,399), got {mats.shape}.')
    order=mat_subjects[['subject','group']].copy()
    ensure(set(order.subject)==set(s.subject) and (order.set_index('subject').loc[s.subject,'group'].to_numpy()==s.group.to_numpy()).all(), 'Score and matrix subject/group identities disagree.')
    matrix_reorder=order.set_index('subject').loc[s.subject].index.to_numpy()
    # input arrays are ordered by matrix_subjects; convert to scores' deterministic order
    position={sid:i for i,sid in enumerate(order.subject)}; mats=mats[[position[sid] for sid in s.subject]]
    ensure(np.isfinite(mats).all(), 'Nonfinite matrix values.')
    ensure(np.allclose(mats,mats.swapaxes(1,2),atol=1e-8,rtol=0), 'Participant matrices must be symmetric Fisher-z matrices.')
    ensure(np.allclose(np.diagonal(mats,axis1=1,axis2=2),0,atol=1e-10,rtol=0), 'Matrices must use the declared diagonal-zero convention.')
    parcels=pd.read_csv(parcels_path)
    ensure({'matrix_index','native_label','yeo7_network'} <= set(parcels), 'parcel_metadata must contain matrix_index,native_label,yeo7_network.')
    ensure(len(parcels)==N_COMMON_PARCELS and parcels.matrix_index.nunique()==N_COMMON_PARCELS and set(parcels.matrix_index)==set(range(N_COMMON_PARCELS)), 'Parcel matrix_index must be local 0..398 exactly.')
    parcels=parcels.sort_values('matrix_index').reset_index(drop=True)
    ensure(parcels.native_label.nunique()==N_COMMON_PARCELS and set(parcels.native_label)==COMMON_NATIVE_LABELS, 'Parcel native labels must be 1..400 excluding 323; local matrix_index is not a native label.')
    observed=parcels.yeo7_network.value_counts().to_dict(); ensure(observed==COUNTS, f'Unexpected physical Yeo7 counts: {observed}')
    runwise=pd.read_csv(runwise_path)
    r_cfg=m['runwise_primary']; run1=r_cfg.get('run1_column'); run2=r_cfg.get('run2_column')
    ensure(run1 and run2 and {'subject','group',run1,run2} <= set(runwise), 'runwise_primary must name existing run1_column/run2_column.')
    runwise=runwise[['subject','group',run1,run2]].rename(columns={run1:'run1_primary',run2:'run2_primary'}).sort_values('subject').reset_index(drop=True)
    ensure(runwise.subject.tolist()==s.subject.tolist() and runwise.group.tolist()==s.group.tolist(), 'runwise identities must match participant scores.')
    ensure(np.isfinite(runwise[['run1_primary','run2_primary']].to_numpy(float)).all(), 'Nonfinite runwise primary scores.')
    hashes={str(p):sha256(p) for p in (scores_path,matrix_path,subjects_path,parcels_path,runwise_path)}
    return s,mats,parcels,runwise,dict(manifest=m,hashes=hashes,score_path=str(scores_path))

def physical_network_scores(mats: np.ndarray, parcels: pd.DataFrame) -> pd.DataFrame:
    out={}
    for network,n in COUNTS.items():
        ix=np.flatnonzero(parcels.yeo7_network.to_numpy()==network)
        ensure(len(ix)==n, 'Unexpected parcel count for '+network)
        a=mats[:,ix][:,:,ix]; ii,jj=np.triu_indices(n,1)
        out[network]=a[:,ii,jj].mean(axis=1)
    return pd.DataFrame(out)

def behavioral_table(scores: pd.DataFrame) -> pd.DataFrame:
    ensure(BEHAVIOR_TRIALS.exists(), 'Authorized behavior input missing: '+str(BEHAVIOR_TRIALS))
    d=pd.read_csv(BEHAVIOR_TRIALS)
    req={'subject','group','RT100','RT_ms','B','U','D','run','trial','category','prev_category','global_order','log_order','prior_biased','prior_unbiased','correct'}
    ensure(req <= set(d), 'Authorized trials missing: '+', '.join(sorted(req-set(d))))
    ensure(len(d)==10995 and d.subject.nunique()==33 and d.groupby('group').subject.nunique().to_dict()=={'DS':17,'SS':16}, 'Authorized trial set must be N=33/10,995 SS16 DS17.')
    ensure((d.correct==1).all() and (d.RT_ms > 200).all() and (d.RT_ms <= 3000).all(), 'Authorized correct+RT gate must be strict RT>200 ms and RT<=3000 ms.')
    ensure(set(d.subject)==set(scores.subject), 'Behavior subjects do not equal extraction subjects.')
    if BEHAVIOR_TRIALS.name == 'synthetic_authorized_trials.csv':
        # Synthetic fixture has no raw acquisition source. It still exercises the
        # exact column contract; source-provenance validation is real-data only.
        return d
    # History covariates are frozen behavioral inputs, but verify that they retain
    # source chronology rather than being silently renumbered after exclusions.
    raw_path=ROOT/'derivatives/behavior/vigilance_state/merged_events_with_correct.csv'
    ensure(raw_path.exists(), 'Chronology source required to validate frozen history inputs is absent: '+str(raw_path))
    raw=pd.read_csv(raw_path)
    raw_req={'subject','run','trial','Distractor','Dimension'}
    ensure(raw_req <= set(raw), 'Chronology source missing: '+', '.join(sorted(raw_req-set(raw))))
    raw=raw.copy(); raw['group']=raw.subject.str[:2]
    raw['category']=np.where(raw.Distractor.eq(0),'absent',np.where(((raw.group.eq('SS')&raw.Dimension.eq('Same'))|(raw.group.eq('DS')&raw.Dimension.eq('Different'))),'biased','unbiased'))
    raw=raw.sort_values(['subject','run','trial']).reset_index(drop=True)
    raw['global_order_expected']=raw.groupby('subject',sort=False).cumcount()+1
    raw['prev_category_expected']=raw.groupby(['subject','run'],sort=False).category.shift()
    raw['prior_biased_expected']=raw.groupby('subject',sort=False).category.transform(lambda z:z.eq('biased').cumsum().shift(fill_value=0)).astype(int)
    raw['prior_unbiased_expected']=raw.groupby('subject',sort=False).category.transform(lambda z:z.eq('unbiased').cumsum().shift(fill_value=0)).astype(int)
    check=d.merge(raw[['subject','run','trial','global_order_expected','prev_category_expected','prior_biased_expected','prior_unbiased_expected']],on=['subject','run','trial'],how='left',validate='one_to_one')
    ensure(check.global_order_expected.notna().all(), 'Frozen trials did not map one-to-one to chronology source.')
    ensure((check.global_order.to_numpy()==check.global_order_expected.to_numpy()).all(), 'Frozen global_order differs from pre-filter source chronology.')
    ensure(np.allclose(check.log_order.to_numpy(float),np.log(check.global_order_expected.to_numpy(float))), 'Frozen log_order is not log(original chronological order).')
    ensure((check.prior_biased.to_numpy()==check.prior_biased_expected.to_numpy()).all() and (check.prior_unbiased.to_numpy()==check.prior_unbiased_expected.to_numpy()).all(), 'Frozen prior exposures are not pre-current source-chronological counts.')
    same_prev=check.prev_category.fillna('__FIRST__').eq(check.prev_category_expected.fillna('__FIRST__'))
    ensure(same_prev.all(), 'Frozen previous-category labels differ from pre-filter source chronology.')
    return d

def association_suite(s: pd.DataFrame, networks: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    idx=group_indices(s.group.to_numpy()); x=s.primary_default.to_numpy(float); xc=center_group(x,idx)
    outcomes=['biased_capture','unbiased_capture','all_capture']
    need={'subject','group',*outcomes,'biased_priming'}
    # behavioral score columns are joined upstream; use existing precise authorized calculations.
    ensure(need <= set(s), 'Missing authorized behavioral score columns.')
    rows=[]
    for y in outcomes:
        z=center_group(s[y].to_numpy(float),idx)
        rows.append(dict(scope='pooled',outcome=y,**group_adjusted_corr(xc,z,'group-centered Pearson df=N-3')))
    # Group-only residualization duplicates the primary group-centered Pearson
    # estimand and is intentionally not a second family member.
    for cov,label in [(['unbiased_capture','D'],'partial unbiased+group'),(['biased_priming','D'],'partial measured priming+group')]:
        rows.append(dict(scope='pooled',outcome='biased_capture',**partial_corr(s,'primary_default','biased_capture',cov,label)))
    for group in ('SS','DS'):
        z=s.loc[s.group.eq(group)].copy()
        rows.append(dict(scope=group,outcome='biased_capture',**corr_stats(z.primary_default,z.biased_capture,0,'within-group Pearson')))
        rows.append(dict(scope=group,outcome='biased_capture',**partial_corr(z,'primary_default','biased_capture',['unbiased_capture'],'within-group partial unbiased')))
    a=pd.DataFrame(rows); primary=(a.scope.eq('pooled')&a.outcome.eq('biased_capture')&a.analysis.eq('group-centered Pearson df=N-3'))
    a['p_Holm']=np.nan; a.loc[~primary,'p_Holm']=multipletests(a.loc[~primary,'p_raw'],method='holm')[1]; a['Holm_family']=np.where(primary,'primary; no adjustment','8 prespecified revision-stage exploratory participant associations')
    ensure((~primary).sum()==8, 'Exploratory association family must be exactly 8 after removing redundant group-only and planned-weight rows.')
    rb=float(a.loc[primary,'r'].iloc[0]); ru=float(a.loc[(a.scope=='pooled')&(a.outcome=='unbiased_capture')&(a.analysis.str.contains('group-centered')),'r'].iloc[0]); rbu=float(correlation(center_group(s.biased_capture,idx),center_group(s.unbiased_capture,idx)))
    w=pd.DataFrame([williams(rb,ru,rbu)])
    nrows=[]
    for net in COUNTS:
        nrows.append(dict(network=net,**group_adjusted_corr(center_group(networks[net].to_numpy(float),idx),center_group(s.biased_capture.to_numpy(float),idx),'group-centered Pearson df=N-3')))
    ndf=holm(pd.DataFrame(nrows),'p_raw','seven physical Yeo7 within-network comparators (Holm-7)')
    return a,w,ndf

def sensitivity_suite(s: pd.DataFrame) -> pd.DataFrame:
    idx=group_indices(s.group.to_numpy()); rows=[]
    for v in VARIANTS:
        z=center_group(s[v].to_numpy(float),idx); y=center_group(s.biased_capture.to_numpy(float),idx)
        q=group_adjusted_corr(z,y,'group-centered Pearson df=N-3'); q.update(variant=v,kind='preprocessing sensitivity'); rows.append(q)
    out=pd.DataFrame(rows); out=holm(out,'p_raw','five preprocessing variants: primary, minimal-TR1, GSR, task, task+GSR')
    motion=partial_corr(s,'primary_default','biased_capture',['D','mean_fd'],'motion-adjusted primary association')
    motion.update(variant='primary',kind='mean-FD adjusted; descriptive pointwise')
    return pd.concat([out,pd.DataFrame([motion])],ignore_index=True)

def reconstruct_frozen_conditional_scores(trials: pd.DataFrame, s: pd.DataFrame, frozen_draws: Path) -> tuple[np.ndarray,np.ndarray]:
    """Recreate the authorized RNG score stream without reading an old neural vector.

    The archived biased score matrix is the authorized numerical checkpoint (reconstructed within 1e-10 ms). Recreating the
    unused unbiased score matrix from the same frozen behavioral trials and RNG
    stream is permitted only if the biased checkpoint matches within that tolerance.
    """
    idx=group_indices(s.group.to_numpy()); rng=np.random.default_rng(SEED)
    # Advance through the exact authorized RNG calls before trial resampling.
    _=np.column_stack([rng.choice(q,size=(R,len(q)),replace=True) for q in idx])
    for _ in range(R):
        for q in idx: rng.permutation(q)
    for _ in range(3):
        _=np.column_stack([rng.choice(q,size=(R,len(q)),replace=True) for q in idx])
    bdraw=np.empty((R,33)); udraw=np.empty((R,33))
    for col,subject in enumerate(s.subject):
        z=trials.loc[trials.subject.eq(subject)]
        ba=z.loc[z.category.eq('biased'),'RT_ms'].to_numpy(float); ua=z.loc[z.category.eq('unbiased'),'RT_ms'].to_numpy(float); ar=float(z.loc[z.category.eq('absent'),'RT_ms'].mean())
        ensure(len(ba)>=len(ua)>0 and np.isfinite(ar), 'Frozen conditional score reconstruction has invalid condition counts.')
        for k in range(R): bdraw[k,col]=rng.choice(ba,len(ua),replace=False).mean()-ar
        udraw[:,col]=ua[rng.integers(len(ua),size=(R,len(ua)))].mean(axis=1)-ar
    archived=np.load(frozen_draws,mmap_mode='r',allow_pickle=False)
    ensure(archived.shape==(R,33) and np.allclose(bdraw,np.asarray(archived,float),atol=1e-10,rtol=0), 'Reconstructed biased conditional scores do not match the authorized frozen RNG checkpoint within 1e-10 ms.')
    return bdraw,udraw

def conditional_summary(label: str, values: np.ndarray, method: str) -> dict:
    lo,hi=np.quantile(values,[.025,.975])
    return dict(estimand=label,median=float(np.median(values)),mean=float(np.mean(values)),ci_low=float(lo),ci_high=float(hi),R=R,seed=SEED,scope=method)

def resampling_suite(s: pd.DataFrame, trials: pd.DataFrame, frozen_draws: Path, frozen_subjects: Path, frozen_trial_resampling: Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    idx=group_indices(s.group.to_numpy()); x=s.primary_default.to_numpy(float); B=s.biased_capture.to_numpy(float); U=s.unbiased_capture.to_numpy(float); A=s.all_capture.to_numpy(float); xc=center_group(x,idx)
    rng=np.random.default_rng(SEED); sampled=np.column_stack([rng.choice(q,size=(R,len(q)),replace=True) for q in idx]); pos=[np.arange(16),np.arange(16,33)]
    bootB=correlation(center_group(x[sampled],pos),center_group(B[sampled],pos)); bootU=correlation(center_group(x[sampled],pos),center_group(U[sampled],pos)); bootA=correlation(center_group(x[sampled],pos),center_group(A[sampled],pos)); bootDelta=bootB-bootU
    perm=np.empty(R)
    for k in range(R):
        y=B.copy()
        for q in idx: y[q]=rng.permutation(y[q])
        perm[k]=correlation(xc,center_group(y,idx))
    pointB=float(correlation(xc,center_group(B,idx))); pperm=(1+np.sum(np.abs(perm)>=abs(pointB)))/(R+1)
    draws=pd.DataFrame(dict(iteration=np.arange(1,R+1),bootstrap_primary_r=bootB,bootstrap_unbiased_r=bootU,bootstrap_all_r=bootA,bootstrap_delta_r=bootDelta,permuted_primary_r=perm))
    rows=[]
    for label,point,v in [('biased r',pointB,bootB),('unbiased r',float(correlation(xc,center_group(U,idx))),bootU),('all r',float(correlation(xc,center_group(A,idx))),bootA),('delta biased-unbiased r',pointB-float(correlation(xc,center_group(U,idx))),bootDelta)]:
        lo,hi=np.quantile(v,[.025,.975]); rows.append(dict(estimand=label,point=point,ci_low=lo,ci_high=hi,R=R,seed=SEED,CI_method='95% percentile stratified paired participant bootstrap; group means recalculated per draw',p_two_sided_within_group_permutation=pperm if label=='biased r' else np.nan))
    ensure(frozen_draws.exists() and frozen_subjects.exists() and frozen_trial_resampling.exists(), 'Authorized frozen conditional draw input(s) missing.')
    order=pd.read_csv(frozen_subjects)
    ensure({'subject','group'}<=set(order) and len(order)==33 and order.subject.tolist()==s.subject.tolist() and order.group.tolist()==s.group.tolist(), 'Frozen draw subject order must exactly match sorted analysis subjects.')
    bdraw,udraw=reconstruct_frozen_conditional_scores(trials,s,frozen_draws)
    bdc=center_group(bdraw,idx); udc=center_group(udraw,idx)
    br=correlation(np.broadcast_to(xc,(R,33)),bdc); ur=correlation(np.broadcast_to(xc,(R,33)),udc)
    fixed_ur=float(correlation(xc,center_group(U,idx)))
    delta_fixed=br-fixed_ur; delta_bootstrap=br-ur
    r_bdraw_u=np.asarray(correlation(bdc,np.broadcast_to(center_group(U,idx),(R,33))),float)
    williams_fixed_t=np.array([williams(float(a),fixed_ur,float(c))['t'] for a,c in zip(br,r_bdraw_u)])
    cdraw=pd.DataFrame(dict(iteration=np.arange(1,R+1),countmatched_biased_r=br,unbiased_bootstrap_r=ur,direct_matched_biased_minus_fixed_unbiased_r=delta_fixed,direct_matched_biased_minus_bootstrap_unbiased_r=delta_bootstrap,drawwise_williams_t_fixed_unbiased=williams_fixed_t))
    conditional=pd.DataFrame([
        conditional_summary('DMN--matched-biased association',br,'Conditional trial-resampling distribution; not a participant-sampling CI.'),
        conditional_summary('DMN--unbiased bootstrap association',ur,'Conditional trial-resampling distribution; not a participant-sampling CI.'),
        conditional_summary('Direct matched-biased minus fixed-fullscore-unbiased r',delta_fixed,'Conditional difference distribution; fixed full-score unbiased comparator; not a participant-sampling CI.'),
        conditional_summary('Direct matched-biased minus bootstrap-unbiased r',delta_bootstrap,'Conditional paired-by-RNG-iteration difference; not a participant-sampling CI.'),
        conditional_summary('Drawwise Williams t (fixed unbiased)',williams_fixed_t,'Descriptive conditional statistic; no drawwise p values or significance fraction.')
    ])
    conditional['authorized_frozen_biased_draws_sha256']=sha256(frozen_draws)
    # Preserve legacy nominal-draw fractions only as a provenance audit. They use
    # the historical neural vector and are never a p value for the fresh result.
    audit=pd.read_csv(frozen_trial_resampling)
    need={'iteration','biased_downsample_p','unbiased_bootstrap_p'}
    ensure(len(audit)==R and need <= set(audit), 'Frozen conditional audit file has an unexpected schema.')
    audit_out=pd.DataFrame([dict(
        status='AUDIT_ONLY_NOT_FRESH_INFERENCE',
        source=str(frozen_trial_resampling),source_sha256=sha256(frozen_trial_resampling),R=R,
        biased_nominal_draw_p_lt_05_fraction=float(np.mean(audit.biased_downsample_p.to_numpy(float)<.05)),
        unbiased_nominal_draw_p_lt_05_fraction=float(np.mean(audit.unbiased_bootstrap_p.to_numpy(float)<.05)),
        interpretation='Fractions of historical per-draw nominal p values are not inferential p values and must not be reported as p values for fresh neural results.'
    )])
    matched_mean=bdraw.mean(axis=0); matched_r=float(correlation(xc,center_group(matched_mean,idx))); matched_bu=float(correlation(center_group(matched_mean,idx),center_group(U,idx)))
    w_full=williams(pointB,fixed_ur,float(correlation(center_group(B,idx),center_group(U,idx))))
    w_full['procedure']='full retained scores'; w_matched=williams(matched_r,fixed_ur,matched_bu); w_matched['procedure']='mean of authorized count-matched biased score draws'
    w_all=pd.DataFrame([w_full,w_matched]); w_all['p_Holm2']=multipletests(w_all.p_raw,method='holm')[1]; w_all['Holm_family']='two exploratory Williams comparisons: full retained and mean count-matched biased score'
    return draws,pd.DataFrame(rows),cdraw,conditional,audit_out,w_all

def lopo_and_stability(s: pd.DataFrame, runwise: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    for i in range(33):
        z=s.drop(index=i).reset_index(drop=True); idx=[np.flatnonzero(z.group.to_numpy()==g) for g in ('SS','DS')]
        ensure(len(idx[0])+len(idx[1])==32 and min(map(len,idx))>=15, 'LOPO group membership failure.')
        x=center_group(z.primary_default.to_numpy(float),idx); y=center_group(z.biased_capture.to_numpy(float),idx); q=group_adjusted_corr(x,y,'LOPO group-centered Pearson'); q.update(omitted=s.subject.iloc[i]);rows.append(q)
    r=[];idx=group_indices(runwise.group.to_numpy())
    for scope,ix,k in [('pooled group-adjusted',np.arange(33),0),('SS',idx[0],0),('DS',idx[1],0)]:
        if scope.startswith('pooled'):
            a=center_group(runwise.run1_primary.to_numpy(float),idx);b=center_group(runwise.run2_primary.to_numpy(float),idx);q=group_adjusted_corr(a,b,'exact new primary run1/run2 group-adjusted')
        else: q=corr_stats(runwise.run1_primary.to_numpy(float)[ix],runwise.run2_primary.to_numpy(float)[ix],0,'exact new primary run1/run2 within group')
        q.update(scope=scope);r.append(q)
    return pd.DataFrame(rows),pd.DataFrame(r)

def merge_behavior(scores: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    means=trials.groupby(['subject','group','category'],observed=True).RT_ms.mean().unstack('category')
    ensure({'absent','biased','unbiased'} <= set(means), 'Trial categories must include absent/biased/unbiased.')
    b=means.reset_index(); b['biased_capture']=b.biased-b.absent; b['unbiased_capture']=b.unbiased-b.absent
    counts=trials.groupby(['subject','group','category'],observed=True).size().unstack('category').reset_index(); b=b.merge(counts,on=['subject','group'],suffixes=('','_n'),validate='one_to_one')
    b['biased_weight']=b.biased_n/(b.biased_n+b.unbiased_n);b['unbiased_weight']=1-b.biased_weight;b['all_capture']=b.biased_weight*b.biased_capture+b.unbiased_weight*b.unbiased_capture
    # Inter-trial priming is a lag-defined contrast, not the absolute mean RT of
    # biased trials. The frozen behavioral pipeline defines it on current biased
    # trials with a prior recorded display, comparing prior biased versus all
    # other prior categories.
    q=trials.loc[trials.category.eq('biased') & trials.prev_category.notna()].copy()
    prim=q.groupby(['subject','group'],observed=True).apply(lambda z: z.loc[z.prev_category.eq('biased'),'RT_ms'].mean()-z.loc[~z.prev_category.eq('biased'),'RT_ms'].mean(),include_groups=False).rename('biased_priming').reset_index()
    ensure(len(prim)==33 and np.isfinite(prim.biased_priming.to_numpy(float)).all(), 'Lag-defined biased priming is unavailable for one or more participants.')
    b=b.merge(prim,on=['subject','group'],validate='one_to_one')
    out=scores.merge(b[['subject','group','biased_capture','unbiased_capture','all_capture','biased_priming']],on=['subject','group'],validate='one_to_one')
    out['D']=out.group.eq('DS').astype(int)
    return out

def history_subject_audit(trials: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Descriptive S30 score audit; it does not replace the joint LMM."""
    rows=[]
    raw_means=trials.groupby(['subject','group','category'],observed=True).RT_ms.mean().unstack('category').reset_index()
    for (subject,group),z0 in trials.groupby(['subject','group'],observed=True):
        z=z0.copy(); z['run2']=z.run.eq(2).astype(float)
        for name in ('log_order','prior_biased','prior_unbiased'):
            sd=float(z[name].std(ddof=1)); ensure(np.isfinite(sd) and sd>0, 'Invalid within-subject history scale: '+name+' for '+subject)
            z['z_'+name]=(z[name]-z[name].mean())/sd
        cols=['B','U','run2','z_log_order','z_prior_biased','z_prior_unbiased']
        X=np.column_stack([np.ones(len(z))]+[z[c].to_numpy(float) for c in cols]); y=z.RT100.to_numpy(float)
        ensure(np.linalg.matrix_rank(X)==X.shape[1], 'Rank-deficient subject history design: '+subject)
        beta=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X@beta; df=len(y)-X.shape[1]; sigma2=float(resid@resid/df); se=np.sqrt(np.diag(sigma2*np.linalg.inv(X.T@X)))
        rows.append(dict(subject=subject,group=group,history_biased_coefficient_ms=beta[1]*100,history_biased_coefficient_se_ms=se[1]*100,history_unbiased_coefficient_ms=beta[2]*100,history_unbiased_coefficient_se_ms=se[2]*100,history_residual_df=df,history_fixed_columns=X.shape[1],history_condition_number=float(np.linalg.cond(X))))
    coef=pd.DataFrame(rows).merge(raw_means,on=['subject','group'],validate='one_to_one').merge(scores[['subject','group','primary_default','biased_capture','unbiased_capture','all_capture']],on=['subject','group'],validate='one_to_one')
    coef=coef.rename(columns={'absent':'raw_absent_rt_ms','biased':'raw_biased_rt_ms','unbiased':'raw_unbiased_rt_ms','biased_capture':'raw_biased_capture_ms','unbiased_capture':'raw_unbiased_capture_ms','all_capture':'raw_all_capture_ms'})
    ix=group_indices(coef.group.to_numpy()); tests=[]
    for label,a,b in [
        ('fresh primary vs history-adjusted biased coefficient','primary_default','history_biased_coefficient_ms'),
        ('fresh primary vs history-adjusted unbiased coefficient','primary_default','history_unbiased_coefficient_ms'),
        ('history-adjusted biased coefficient vs raw biased capture','history_biased_coefficient_ms','raw_biased_capture_ms'),
        ('history-adjusted unbiased coefficient vs raw unbiased capture','history_unbiased_coefficient_ms','raw_unbiased_capture_ms')]:
        q=group_adjusted_corr(center_group(coef[a].to_numpy(float),ix),center_group(coef[b].to_numpy(float),ix),'group-centered descriptive diagnostic df=N-3')
        q.update(analysis=label,correction='No multiplicity correction: Table S30 descriptive diagnostic, not a confirmatory family.')
        tests.append(q)
    return coef,pd.DataFrame(tests)

def prepare_lmm_trials(trials: pd.DataFrame, scores: pd.DataFrame, out: Path) -> Path:
    # Participant-level predictor: center over one row per participant, never
    # weight a group mean by the number of that person's eligible trials.
    participant=scores[['subject','group','primary_default']].copy()
    ensure(len(participant)==33 and participant.subject.nunique()==33, 'LMM predictor centering requires all33 unique participants.')
    participant['X_raw_new_primary']=participant.primary_default-participant.groupby('group').primary_default.transform('mean')
    sd=participant.X_raw_new_primary.std(ddof=1)
    d=trials.drop(columns=[c for c in ('dmn_conn','X','primary_default','X_raw_new_primary') if c in trials],errors='ignore').merge(participant,on=['subject','group'],validate='many_to_one')
    ensure(np.isfinite(sd) and sd>0, 'Global N=33 SD for centered new predictor is invalid.')
    d['X']=d.X_raw_new_primary/sd
    ensure(abs(d[['subject','X']].drop_duplicates().X.std(ddof=1)-1)<1e-12, 'X must use the global N=33 SD.')
    path=out/'joint_history_trials_new_primary.csv'; d.to_csv(path,index=False); return path

def run_r_models(trials: Path, out: Path) -> bool:
    model_out=out/'mixed_models'; model_out.mkdir(exist_ok=True)
    proc=subprocess.run(['Rscript',str(HERE/'fit_new_predictor_mixed_models.R'),str(trials),str(model_out)],text=True,capture_output=True,env=os.environ.copy())
    (out/'mixed_models_stdout_stderr.txt').write_text(proc.stdout+'\n--- STDERR ---\n'+proc.stderr)
    if proc.returncode: fail('Sequential R mixed-model job failed; see '+str(out/'mixed_models_stdout_stderr.txt'))
    required=['primary_joint_fixed_effects_complete.csv','primary_joint_history_fixed_effects_complete.csv','primary_joint_history_model_matrix_columns.csv','primary_joint_history_diagnostics.csv','primary_joint_history_omnibus_type3.csv','primary_joint_history_simple_slopes_and_contrasts.csv']
    for n in required: ensure((model_out/n).exists(), 'R model missing output '+n)
    cols=pd.read_csv(model_out/'primary_joint_history_model_matrix_columns.csv'); diag=pd.read_csv(model_out/'primary_joint_history_diagnostics.csv')
    ensure(len(cols)==144 and int(diag.fixed_columns.iloc[0])==144 and int(diag.fixed_rank.iloc[0])==144, 'History model must retain the exact full-rank 144-column fixed design.')
    all_diag=pd.read_csv(model_out/'mixed_model_diagnostics_all.csv')
    ensure({'model','diagnostic_status','singular','bobyqa_converged','nloptwrap_converged'} <= set(all_diag), 'R diagnostics did not report convergence/singularity status.')
    return bool(all(all_diag.diagnostic_status.eq('PASSED')))

def make_synthetic(destination: Path) -> Path:
    """Create labelled fake inputs only; no real neural or behavioral evidence."""
    rng=np.random.default_rng(71); destination.mkdir(parents=True,exist_ok=True)
    subjects=[f'SS{i:03d}' for i in range(1,17)]+[f'DS{i:03d}' for i in range(1,18)]; groups=np.array(['SS']*16+['DS']*17)
    meta=[]; native_labels=iter(sorted(COMMON_NATIVE_LABELS))
    for network,n in COUNTS.items():
        for _ in range(n): meta.append(dict(matrix_index=len(meta),native_label=next(native_labels),yeo7_network=network))
    parcels=pd.DataFrame(meta); ensure(len(parcels)==N_COMMON_PARCELS and 323 not in set(parcels.native_label),'Synthetic common-coverage metadata failure.'); parcels.to_csv(destination/'physical_yeo7_common399_parcels.csv',index=False)
    mats=np.zeros((33,N_COMMON_PARCELS,N_COMMON_PARCELS))
    for i in range(33):
        a=rng.normal(0,.08,(N_COMMON_PARCELS,N_COMMON_PARCELS));a=(a+a.T)/2;np.fill_diagonal(a,0);mats[i]=a
    np.save(destination/'primary_participant_matrices.npy',mats)
    pd.DataFrame({'subject':subjects,'group':groups}).to_csv(destination/'primary_matrix_subjects.csv',index=False)
    default_ix=parcels.index[parcels.yeo7_network.eq('Default')].to_numpy();ii,jj=np.triu_indices(91,1);primary=mats[:,default_ix][:,:,default_ix][:,ii,jj].mean(axis=1)
    scores=pd.DataFrame({'subject':subjects,'group':groups,'primary_default':primary,'minimal_tr1':primary+rng.normal(0,.01,33),'primary_gsr':primary+rng.normal(0,.01,33),'primary_task':primary+rng.normal(0,.01,33),'primary_task_gsr':primary+rng.normal(0,.01,33),'mean_fd':rng.uniform(.05,.3,33)})
    scores.to_csv(destination/'participant_scores_resolved.csv',index=False)
    pd.DataFrame({'subject':subjects,'group':groups,'run1_primary':primary+rng.normal(0,.02,33),'run2_primary':primary+rng.normal(0,.02,33)}).to_csv(destination/'runwise_primary_scores.csv',index=False)
    records=[]
    for si,(subject,group) in enumerate(zip(subjects,groups)):
        n=334 if si<6 else 333
        for t in range(n):
            cycle=('biased','biased','unbiased','absent'); category=cycle[t%4]; B=int(category=='biased');U=int(category=='unbiased');run=1 if t<n//2 else 2
            rt=800+25*B-5*U+rng.normal(0,70)
            records.append(dict(subject=subject,group=group,run=run,trial=t+1,RT100=rt/100,RT_ms=rt,B=B,U=U,D=int(group=='DS'),category=category,correct=1,global_order=t+1,log_order=np.log1p(t),prev_category=None if t==0 else cycle[(t-1)%4],prior_biased=sum((cycle[u%4]=='biased') for u in range(t)),prior_unbiased=sum((cycle[u%4]=='unbiased') for u in range(t))))
    trials=pd.DataFrame(records);ensure(len(trials)==10995,'Synthetic trial count failure.');trials.to_csv(destination/'synthetic_authorized_trials.csv',index=False)
    # Create an exact synthetic counterpart of the authorized RNG checkpoint.
    # This deliberately contains no empirical neural vector.
    frozen_order=np.argsort(np.asarray(subjects)); ordered_subjects=np.asarray(subjects)[frozen_order]; ordered_groups=groups[frozen_order]
    gix=[np.flatnonzero(ordered_groups==g) for g in ('SS','DS')]; frozen_rng=np.random.default_rng(SEED)
    _=np.column_stack([frozen_rng.choice(q,size=(R,len(q)),replace=True) for q in gix])
    for _ in range(R):
        for q in gix: frozen_rng.permutation(q)
    for _ in range(3):
        _=np.column_stack([frozen_rng.choice(q,size=(R,len(q)),replace=True) for q in gix])
    bdraw=np.empty((R,33)); udraw=np.empty((R,33))
    for col,subject in enumerate(ordered_subjects):
        z=trials.loc[trials.subject.eq(subject)]; ba=z.loc[z.category.eq('biased'),'RT_ms'].to_numpy(); ua=z.loc[z.category.eq('unbiased'),'RT_ms'].to_numpy(); ar=z.loc[z.category.eq('absent'),'RT_ms'].mean()
        for k in range(R): bdraw[k,col]=frozen_rng.choice(ba,len(ua),replace=False).mean()-ar
        udraw[:,col]=ua[frozen_rng.integers(len(ua),size=(R,len(ua)))].mean(axis=1)-ar
    np.save(destination/'synthetic_frozen_countmatched_draws.npy',bdraw)
    pd.DataFrame({'subject':ordered_subjects,'group':ordered_groups}).to_csv(destination/'synthetic_frozen_countmatched_subjects.csv',index=False)
    pd.DataFrame({'iteration':np.arange(1,R+1),'biased_downsample_p':np.nan,'unbiased_bootstrap_p':np.nan}).to_csv(destination/'synthetic_frozen_trial_resampling.csv',index=False)
    m={'status':'complete','synthetic_test_only':True,'common_coverage_geometry':{'nominal_atlas_parcels':400,'effective_common_coverage_parcels':N_COMMON_PARCELS,'excluded_native_labels':sorted(EXCLUDED_NATIVE_LABELS),'uniform_all_participants_and_runs':True},'participant_scores':{'path':'participant_scores_resolved.csv','primary_column':'primary_default','variant_columns':{'primary':'primary_default','minimal_tr1':'minimal_tr1','primary_gsr':'primary_gsr','primary_task':'primary_task','primary_task_gsr':'primary_task_gsr'},'mean_fd_column':'mean_fd'},'primary_matrices':{'path':'primary_participant_matrices.npy'},'matrix_subjects':{'path':'primary_matrix_subjects.csv'},'parcel_metadata':{'path':'physical_yeo7_common399_parcels.csv'},'runwise_primary':{'path':'runwise_primary_scores.csv','run1_column':'run1_primary','run2_column':'run2_primary'}}
    p=destination/'inference_input_manifest.json';p.write_text(json.dumps(m,indent=2)+'\n');return p

def main() -> None:
    global BEHAVIOR_TRIALS
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-manifest',type=Path,default=EXTRACTION/'inference_input_manifest.json')
    ap.add_argument('--output',type=Path,default=HERE/'results')
    ap.add_argument('--behavior-trials',type=Path,default=BEHAVIOR_TRIALS)
    ap.add_argument('--frozen-draws',type=Path,default=FROZEN_DRAWS)
    ap.add_argument('--frozen-subjects',type=Path,default=FROZEN_SUBJECTS)
    ap.add_argument('--frozen-trial-resampling',type=Path,default=FROZEN_TRIAL_RESAMPLING)
    ap.add_argument('--synthetic-test',action='store_true',help='Generate and analyse synthetic fixture; explicitly not evidence.')
    args=ap.parse_args()
    if args.synthetic_test:
        args.output=args.output/'SYNTHETIC_TEST_NOT_EVIDENCE'
        synthetic_inputs=args.output.parent/(args.output.name+'_inputs')
        manifest=make_synthetic(synthetic_inputs);args.input_manifest=manifest;args.behavior_trials=synthetic_inputs/'synthetic_authorized_trials.csv';args.frozen_draws=synthetic_inputs/'synthetic_frozen_countmatched_draws.npy';args.frozen_subjects=synthetic_inputs/'synthetic_frozen_countmatched_subjects.csv';args.frozen_trial_resampling=synthetic_inputs/'synthetic_frozen_trial_resampling.csv'
    BEHAVIOR_TRIALS=args.behavior_trials
    ensure(not args.output.exists() or args.synthetic_test, 'Refusing to overwrite output directory: '+str(args.output))
    args.output.mkdir(parents=True,exist_ok=False)
    scores,mats,parcels,runwise,provenance=load_inputs(args.input_manifest)
    trials=behavioral_table(scores);scores=merge_behavior(scores,trials)
    physical=physical_network_scores(mats,parcels)
    ensure(np.allclose(physical.Default.to_numpy(),scores.primary_default.to_numpy(),atol=1e-8,rtol=0), 'Primary vector does not equal Default91 mean from the new all-common-coverage matrices.')
    assoc,williams,networks=association_suite(scores,physical);sens=sensitivity_suite(scores)
    draws,resampling,conditional_draws,conditional,conditional_audit,williams_all=resampling_suite(scores,trials,args.frozen_draws,args.frozen_subjects,args.frozen_trial_resampling)
    lopo,stability=lopo_and_stability(scores,runwise)
    history_coefficients,history_diagnostic=history_subject_audit(trials,scores)
    lmm_trials=prepare_lmm_trials(trials,scores,args.output)
    assoc.to_csv(args.output/'participant_associations.csv',index=False);williams.to_csv(args.output/'williams_primary_vs_unbiased_fullscore_audit.csv',index=False);williams_all.to_csv(args.output/'williams_full_and_countmatched_Holm2.csv',index=False);networks.to_csv(args.output/'physical_yeo7_allnetwork_comparators_holm7_AUDIT_ONLY_NOT_TABLE7.csv',index=False);sens.to_csv(args.output/'preprocessing_motion_gsr_task_sensitivities.csv',index=False);draws.to_csv(args.output/'participant_bootstrap_permutation_draws.csv',index=False);resampling.to_csv(args.output/'participant_bootstrap_permutation_summary.csv',index=False);conditional_draws.to_csv(args.output/'countmatched_conditional_draws.csv',index=False);conditional.to_csv(args.output/'countmatched_conditional_summary.csv',index=False);conditional_audit.to_csv(args.output/'conditional_nominal_draw_fraction_AUDIT_ONLY.csv',index=False);lopo.to_csv(args.output/'leave_one_participant_out.csv',index=False);stability.to_csv(args.output/'new_primary_run_stability.csv',index=False);physical.assign(subject=scores.subject,group=scores.group).to_csv(args.output/'physical_yeo7_network_scores_n33.csv',index=False);history_coefficients.to_csv(args.output/'history_subject_coefficients_and_raw_scores_S30.csv',index=False);history_diagnostic.to_csv(args.output/'history_primary_and_raw_score_diagnostics_S30.csv',index=False)
    models_ready=run_r_models(lmm_trials,args.output)
    status='SYNTHETIC_TEST_NOT_EVIDENCE' if args.synthetic_test else ('complete' if models_ready else 'COMPLETE_WITH_LMM_DIAGNOSTIC_FLAGS_NOT_READY')
    manifest={'status':status,'publication_ready':bool(models_ready and not args.synthetic_test),'scope':'Dependent non-CPM inference only. CPM intentionally excluded; Table 7 network-specific comparisons are parent-owned and this suite emits only a clearly labeled audit file.','N':33,'groups':{'SS':16,'DS':17},'trials':10995,'common_coverage_geometry':{'nominal_atlas_parcels':400,'effective_common_coverage_parcels':N_COMMON_PARCELS,'excluded_native_labels':sorted(EXCLUDED_NATIVE_LABELS),'network_counts':COUNTS},'primary_contract':'physical Yeo7 Default=91 mean upper-triangle Fisher-z from all-common-coverage 399-parcel matrices; group-adjusted df=30; global N=33 SD in LMM X.','input_manifest':str(args.input_manifest),'input_hashes':provenance['hashes'],'behavior_trials':str(args.behavior_trials),'behavior_trials_sha256':sha256(args.behavior_trials),'frozen_countmatched_draws':str(args.frozen_draws),'frozen_countmatched_draws_sha256':sha256(args.frozen_draws),'frozen_trial_resampling':str(args.frozen_trial_resampling),'frozen_trial_resampling_sha256':sha256(args.frozen_trial_resampling),'seed':SEED,'iterations':R,'thread_limits':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'platform':platform.platform(),'python':sys.version,'outputs':'All files in this directory; results are not manuscript evidence when status is synthetic or publication_ready is false.'}
    (args.output/'results_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'output':str(args.output),'history_fixed_columns':144,'CPM':'excluded'},indent=2))
if __name__=='__main__': main()
