#!/usr/bin/env python3
"""Physically crosswalked six-cell CPM, fresh TR1/nuisance-cleaned matrices only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key,'2')
from pathlib import Path
import argparse,hashlib,json,importlib.util,time,platform
import numpy as np
import pandas as pd
from scipy import stats
from nilearn import datasets
import nibabel as nib
ROOT=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
REV=ROOT/'manuscript/paper2_dmn-connectivity/revision-1'
HERE=Path(__file__).resolve().parent
RES=HERE.parent
SOURCE=REV/'attached-v3-correct-rt/cpm/run_cpm_authorized_nested.py'
spec=importlib.util.spec_from_file_location('authorized_cpm_utilities',SOURCE)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
FAMILIES=legacy.FAMILIES

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def mapping(native_labels=None):
    atlas={k:datasets.fetch_atlas_schaefer_2018(n_rois=400,yeo_networks=k,resolution_mm=2,verbose=0) for k in [7,17]}
    im={k:nib.load(atlas[k].maps) for k in [7,17]}
    assert im[7].shape==im[17].shape and np.array_equal(im[7].affine,im[17].affine)
    a=np.asanyarray(im[7].dataobj).astype(np.int16).ravel()
    b=np.asanyarray(im[17].dataobj).astype(np.int16).ravel()
    assert np.array_equal(a>0,b>0)
    pairs=np.unique(np.stack([a[a>0],b[a>0]],axis=1),axis=0)
    assert pairs.shape==(400,2) and len(np.unique(pairs[:,0]))==400 and len(np.unique(pairs[:,1]))==400
    names={}
    for k in [7,17]:
        labs=[x.decode() if isinstance(x,bytes) else str(x) for x in atlas[k].labels]
        if labs[0].lower()=='background':labs=labs[1:]
        assert len(labs)==400
        names[k]=labs
    rows=[]
    for c7,c17 in pairs:
        l7,l17=names[7][int(c7)-1],names[17][int(c17)-1]
        rows.append({'yeo7_id':int(c7),'matrix_column':int(c7)-1,'label7':l7,'network7':l7.split('_')[2],'yeo17_id':int(c17),'label17':l17,'network17':l17.split('_')[2]})
    cross=pd.DataFrame(rows).sort_values('matrix_column').reset_index(drop=True)
    full_pair_count=len(cross)
    if native_labels is not None:
        native_labels=np.array(native_labels,int)
        assert len(native_labels)==399 and set(native_labels)==set(range(1,401))-{323}
        cross=cross.set_index('yeo7_id').loc[native_labels].reset_index()
        cross['matrix_column']=np.arange(len(cross))
    net=cross.network17.to_numpy();ti,tj=np.triu_indices(len(cross),1);ni,nj=net[ti],net[tj]
    family_full=np.full(len(ti),-1,dtype=np.int8)
    pairs17=[('DefaultA','DefaultA'),('DefaultA','DefaultB'),('DefaultA','DefaultC'),('DefaultB','DefaultB'),('DefaultB','DefaultC'),('DefaultC','DefaultC')]
    for f,(s,t) in enumerate(pairs17):
        mask=((ni==s)&(nj==t))|((ni==t)&(nj==s))
        assert not np.any(family_full[mask]>=0)
        family_full[mask]=f
    mask=family_full>=0;fid=family_full[mask]
    counts=np.bincount(fid,minlength=6)
    assert np.array_equal(counts,legacy.EXPECTED_EDGES)
    cov={'n_physical_crosswalk_pairs_nominal':full_pair_count,'n_common_coverage_parcels':len(cross),'excluded_native7_labels':sorted(set(range(1,401))-set(cross.yeo7_id)),'all_bijective':True,'primary_yeo7_Default_n':int((cross.network7=='Default').sum()),'complementary_yeo17_DefaultABC_n':int(cross.network17.str.startswith('Default').sum()),'overlap_Default7_Default17_n':int(((cross.network7=='Default')&cross.network17.str.startswith('Default')).sum()),'six_edge_counts':counts.tolist(),'source_atlas_hashes':{str(atlas[k].maps):digest(atlas[k].maps) for k in [7,17]}}
    return mask,fid,cross,cov

def scalar_predictions(X,y,groups,family_id,rcrit):
    """Independent single-outcome loop; never calls vectorized evaluation."""
    n=len(y);pred=np.zeros((n,6));target=np.zeros(n)
    for test in range(n):
        use=np.arange(n)!=test;xx=X[use];yy=y[use].copy();gt=groups[use]
        for g in np.unique(groups):
            mu=yy[gt==g].mean();yy[gt==g]-=mu
            if groups[test]==g:target[test]=y[test]-mu
        for f in range(6):
            xf=xx[:,family_id==f];selected=np.array([stats.pearsonr(xf[:,e],yy).statistic if xf[:,e].std()>0 else 0 for e in range(xf.shape[1])])
            signs=np.where(np.abs(selected)>rcrit,np.sign(selected),0)
            strength=xf@signs;xt=X[test,family_id==f]@signs
            if np.sum((strength-strength.mean())**2)>1e-14:
                coef=np.linalg.lstsq(np.column_stack([np.ones(len(yy)),strength]),yy,rcond=None)[0]
                pred[test,f]=coef[0]+coef[1]*xt
    return pred,target

def self_test():
    rng=np.random.default_rng(57);X=rng.normal(size=(12,24));y=rng.normal(size=12);g=np.array(['SS']*6+['DS']*6);fid=np.repeat(np.arange(6),4)
    folds=legacy.prepare_folds(X,g);r,p,t,c=legacy.evaluate_batch(y[:,None],g,fid,folds,.3)
    pp,tt=scalar_predictions(X,y,g,fid,.3)
    assert np.allclose(p[0],pp,atol=1e-10) and np.allclose(t[0],tt,atol=1e-12)
    _,fid,cross,cov=mapping([i for i in range(1,401) if i!=323])
    assert len(cross)==399 and cov['complementary_yeo17_DefaultABC_n']==79
    print(json.dumps({'status':'synthetic_algorithm_test_only_not_empirical_evidence','max_prediction_error':float(np.max(np.abs(p[0]-pp))),'crosswalk':cov},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--scores',type=Path,default=RES/'extraction_common_coverage/participant_scores_resolved.csv');ap.add_argument('--matrices',type=Path,default=RES/'extraction_common_coverage/primary_participant_matrices.npy');ap.add_argument('--matrix-subjects',type=Path,default=RES/'extraction_common_coverage/primary_matrix_subjects.csv');ap.add_argument('--n-perm',type=int,default=10000);ap.add_argument('--batch-size',type=int,default=50);ap.add_argument('--seed',type=int,default=20261001)
    ap.add_argument('--parcel-metadata',type=Path,default=RES/'extraction_common_coverage/physical_yeo7_400_parcels.csv')
    args=ap.parse_args()
    if args.self_test:self_test();return
    assert args.n_perm==10000,'Empirical report requires all10k permutations'
    HERE.mkdir(parents=True,exist_ok=True);t0=time.monotonic()
    trialp=REV/'attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv'
    inputs=[args.scores,args.matrices,args.matrix_subjects,args.parcel_metadata,trialp,RES/'analysis_contract.json',RES/'analysis_contract_coverage_amendment.json',RES/'coverage_qc/coverage_audit.json',SOURCE,Path(__file__)]
    fingerprints={str(p):digest(p) for p in inputs}
    runfingerprint=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
    statep=HERE/'input_fingerprints.json'
    if statep.exists():assert json.loads(statep.read_text())==fingerprints,'Stale completed/checkpoint input fingerprints; new version required'
    else:statep.write_text(json.dumps(fingerprints,indent=2)+'\n')
    scores=pd.read_csv(args.scores,dtype={'subject':str});ids=pd.read_csv(args.matrix_subjects,dtype={'subject':str});trial=pd.read_csv(trialp,dtype={'subject':str})
    assert len(scores)==33 and scores.subject.nunique()==33 and scores.group.value_counts().to_dict()=={'DS':17,'SS':16}
    assert np.array_equal(ids.subject.to_numpy(),scores.subject.to_numpy())
    assert trial.subject.nunique()==33 and len(trial)==10995 and (trial.correct==1).all() and ((trial.RT>.2)&(trial.RT<=3)&np.isfinite(trial.RT)).all()
    chk=trial.assign(RT_ms=trial.RT*1000).pivot_table(index='subject',columns='category',values='RT_ms',aggfunc='mean').reindex(scores.subject)
    assert np.allclose(chk.biased-chk.absent,scores.biased_capture,atol=1e-10)
    parcels=pd.read_csv(args.parcel_metadata).sort_values('matrix_index');assert len(parcels)==399 and set(parcels.native_label)==set(range(1,401))-{323}
    conn=np.load(args.matrices,mmap_mode='r');assert conn.shape==(33,399,399) and np.isfinite(conn).all()
    assert np.allclose(conn,conn.swapaxes(1,2),atol=1e-12) and np.allclose(np.diagonal(conn,axis1=1,axis2=2),0)
    union,fid,cross,cov=mapping(parcels.native_label.to_numpy());cross.to_csv(HERE/'physical_yeo7_to_17_crosswalk.csv',index=False)
    (HERE/'physical_crosswalk_verification.json').write_text(json.dumps(cov,indent=2)+'\n')
    ti,tj=np.triu_indices(399,1);X=np.array(conn[:,ti,tj][:,union]);groups=scores.group.to_numpy(str);y=scores.biased_capture.to_numpy(float)
    folds=legacy.prepare_folds(X,groups)
    tc=stats.t.ppf(1-.01/2,30);rcrit=float(tc/np.sqrt(tc*tc+30))
    observed,pred,target,const=legacy.evaluate_batch(y[:,None],groups,fid,folds,rcrit);obs=observed[0];pred=pred[0];target=target[0]
    # Real data independent heldout scalar implementation over all33/6 folds.
    scalar,scalar_target=scalar_predictions(X,y,groups,fid,rcrit)
    assert np.allclose(pred,scalar,atol=1e-8) and np.allclose(target,scalar_target,atol=1e-10)
    counts=legacy.selected_counts_observed(y,groups,fid,folds,rcrit);counts.insert(1,'held_out_subject',np.repeat(scores.subject.to_numpy(),6));counts.to_csv(HERE/'observed_fold_selection_counts.csv',index=False)
    cp=HERE/f'permutations_{runfingerprint[:16]}.npz'
    perm=legacy.run_permutations(y,groups,fid,folds,rcrit,args.n_perm,args.seed,args.batch_size,cp)
    upper=((perm>=obs).sum(axis=0)+1)/(args.n_perm+1);lower=((perm<=obs).sum(axis=0)+1)/(args.n_perm+1)
    p_equal=np.minimum(1,2*np.minimum(upper,lower));p_abs=((np.abs(perm)>=np.abs(obs)).sum(axis=0)+1)/(args.n_perm+1)
    boot=legacy.fixed_prediction_bootstrap(pred,target,groups,np.random.default_rng(args.seed+1),10000)
    rows=[]
    for f,name in enumerate(FAMILIES):
        cf=counts[counts.family==name]
        rows.append({'family':name,'n':33,'possible_edges':int(legacy.EXPECTED_EDGES[f]),'r_loocv':float(obs[f]),'p_raw_equal_tail':float(p_equal[f]),'p_raw_absolute_sensitivity':float(p_abs[f]),'fixed_prediction_bootstrap_ci_low_descriptive':float(np.quantile(boot[:,f],.025)),'fixed_prediction_bootstrap_ci_high_descriptive':float(np.quantile(boot[:,f],.975)),'null_mean':float(perm[:,f].mean()),'null_sd':float(perm[:,f].std(ddof=1)),'null_q025':float(np.quantile(perm[:,f],.025)),'null_q975':float(np.quantile(perm[:,f],.975)),'median_positive_edges':float(cf.positive_edges_selected.median()),'median_negative_edges':float(cf.negative_edges_selected.median()),'zero_strength_folds':int(const[0,f])})
    summary=pd.DataFrame(rows);summary['p_Holm6_equal_tail']=legacy.holm_adjust(p_equal);summary['p_Holm6_absolute_sensitivity']=legacy.holm_adjust(p_abs);summary.to_csv(HERE/'table_cpm_resolved.csv',index=False)
    pf=scores[['subject','group','biased_capture']].copy();pf['observed_fold_centered_ms']=target
    for f,name in enumerate(FAMILIES):pf[f'predicted_{name}_ms']=pred[:,f]
    pf.to_csv(HERE/'cpm_prediction_data.csv',index=False)
    np.savez_compressed(HERE/'cpm_distributions.npz',permutation_r=perm,observed_r=obs,bootstrap_fixed_predictions=boot,family=np.array(FAMILIES))
    result={'status':'completed','duration_seconds':time.monotonic()-t0,'input_fingerprint':runfingerprint,'n':33,'trials':10995,'permutations':10000,'seed':args.seed,'bootstrap_seed':args.seed+1,'crosswalk':cov,'scalar_prediction_max_abs_error':float(np.max(np.abs(pred-scalar))),'selection':'Inherited trainN32 Pearson screen p<.01 two-sided, df30. Within-fold train-only outcome group centering and standardization; signed positive-minus-negative strengths; OLS; zero-strength predicts centered trainingmean0. This is a feature screening threshold, not final association inference.','inference':'Full within-SS/DS permutation nested through centering/selection/fit/prediction; equal-tail and absolute-r sensitivity, each Holm across6; neither tail originally preregistered. No independent-case CV Pearson p. Fixed-prediction pair-bootstrap intervals descriptive, not refitted-prediction uncertainty.','software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'nilearn':__import__('nilearn').__version__}}
    (HERE/'execution_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(summary.to_string(index=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
