#!/usr/bin/env python3
"""Independent empirical replay; imports neither CPM generator nor its helpers."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
ROUND=HERE.parent.parent
REV=ROUND.parent.parent
RES=ROUND/'resolved_reanalysis'
NAMES=['DefaultA','A-B','A-C','DefaultB','B-C','DefaultC']
PAIRS=[('DefaultA','DefaultA'),('DefaultA','DefaultB'),('DefaultA','DefaultC'),('DefaultB','DefaultB'),('DefaultB','DefaultC'),('DefaultC','DefaultC')]
EDGES=[561,1088,442,496,416,78]

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def close(a,b,tol=1e-10):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape,(a.shape,b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    err=float(np.max(np.abs(a-b)))
    assert err<=tol,err
    return err

def correl(y,p):
    yc=y-y.mean(axis=0)
    pc=p-p.mean(axis=0)
    numerator=np.einsum('nb,nbf->bf',yc,pc)
    denominator=np.sqrt(np.sum(yc**2,axis=0)[:,None]*np.sum(pc**2,axis=0))
    return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-14)

def main():
    started=time.monotonic()
    freeze=json.loads((HERE.parent/'input_freeze.json').read_text())
    source_hashes=dict(freeze['input_hashes'])
    source_hashes[str(HERE.parent/'learning_target_scores.csv')]=freeze['target_hash']
    for path,value in source_hashes.items():
        assert sha(path)==value,path
    protected_hashes={path:sha(path) for path in freeze['protected_capture_and_source_hashes']}
    protected_differences={path:{'frozen':value,'verification_start':protected_hashes[path]} for path,value in freeze['protected_capture_and_source_hashes'].items() if protected_hashes[path]!=value}
    (HERE/'additional_verifier_initial_integrity.json').write_text(json.dumps({'immutable_inputs_match_freeze':True,'protected_differences_from_freeze':protected_differences},indent=2)+'\n')
    archive_manifest=json.loads((HERE/'null_archive_manifest.json').read_text())
    frozen_archive_hashes=archive_manifest['archive_files_sha256']
    for path,value in frozen_archive_hashes.items():
        assert sha(HERE/path)==value,path
    ids=pd.read_csv(RES/'extraction_common_coverage/primary_matrix_subjects.csv',dtype={'subject':str})
    target=pd.read_csv(HERE.parent/'learning_target_scores.csv',dtype={'subject':str})
    assert ids[['subject','group']].equals(target[['subject','group']])
    assert len(ids)==33 and ids.subject.nunique()==33 and ids.group.value_counts().to_dict()=={'DS':17,'SS':16}
    trial=pd.read_csv(REV/'attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv',dtype={'subject':str})
    assert len(trial)==10995 and set(trial.subject)==set(ids.subject)
    assert trial.correct.eq(1).all() and np.isfinite(trial.RT_ms).all() and trial.RT_ms.gt(200).all() and trial.RT_ms.le(3000).all()
    assert (trial.group==trial.subject.map(ids.set_index('subject').group)).all()
    close(trial.RT.to_numpy(float)*1000,trial.RT_ms.to_numpy(float))
    biased=trial.loc[trial.category.eq('biased')].copy()
    assert len(biased)==6721 and set(biased.DisRegion)=={'Rare','Frequent'}
    y=[];count_rows=[];participant_rows=[]
    for row in ids.itertuples(index=False):
        one=biased.loc[biased.subject.eq(row.subject)]
        values={region:one.loc[one.DisRegion.eq(region),'RT_ms'].to_numpy(float) for region in ['Rare','Frequent']}
        assert min(map(len,values.values()))>0
        score=float(np.mean(values['Rare'])-np.mean(values['Frequent']))
        y.append(score)
        participant_rows.append({'subject':row.subject,'group':row.group,'rare_mean_ms':float(values['Rare'].mean()),'frequent_mean_ms':float(values['Frequent'].mean()),'n_rare':len(values['Rare']),'n_frequent':len(values['Frequent']),'learning_rare_minus_frequent_ms':score})
        assert set(one.run)=={1,2}
        for run in [1,2]:
            for region in ['Frequent','Rare']:
                v=one.loc[one.run.eq(run)&one.DisRegion.eq(region),'RT_ms']
                assert len(v)>0
                count_rows.append({'subject':row.subject,'group':row.group,'run':run,'DisRegion':region,'size':len(v),'mean':v.mean()})
    y=np.array(y);groups=ids.group.to_numpy(str)
    target_error=close(y,target.learning_rare_minus_frequent_ms.to_numpy(float))
    old=pd.read_csv(ROUND/'spatial_learning_dmn_v1/participant_analysis/participant_learning_scores.csv',dtype={'subject':str}).set_index('subject').loc[ids.subject]
    old_error=close(y,old.learning_rare_minus_frequent_ms.to_numpy(float))
    reconstructed=pd.DataFrame(participant_rows)
    for col in ['rare_mean_ms','frequent_mean_ms','n_rare','n_frequent']:
        close(reconstructed[col].to_numpy(float),target[col].to_numpy(float))
    reconstructed.to_csv(HERE/'verified_participant_learning_scores.csv',index=False)
    run_counts=pd.DataFrame(count_rows)
    frozen_counts=pd.read_csv(HERE.parent/'learning_target_run_counts.csv')
    keys=['subject','group','run','DisRegion']
    comparison=run_counts.merge(frozen_counts,on=keys,validate='one_to_one',suffixes=('_new','_frozen'))
    assert len(comparison)==132
    close(comparison[['size_new','mean_new']].to_numpy(),comparison[['size_frozen','mean_frozen']].to_numpy())
    run_counts.to_csv(HERE/'verified_participant_run_region_counts.csv',index=False)
    parcels=pd.read_csv(RES/'common399_analysis_ready/physical_yeo7_common399_parcels.csv').sort_values('matrix_index')
    cross=pd.read_csv(HERE/'physical_yeo7_to_17_crosswalk_common399.csv').sort_values('matrix_index')
    physical_legacy=pd.read_csv(RES/'cpm/physical_yeo7_to_17_crosswalk.csv')
    assert len(cross)==399 and np.array_equal(cross.matrix_index,np.arange(399)) and set(cross.native_label)==set(range(1,401))-{323}
    assert np.array_equal(cross.native_label,parcels.native_label)
    assert np.array_equal(cross.physical_label,parcels.physical_label)
    assert np.array_equal(cross.yeo7_network,parcels.yeo7_network)
    check=physical_legacy.set_index('yeo7_id').loc[cross.native_label]
    assert np.array_equal(cross.yeo17_label_id,check.yeo17_id)
    assert np.array_equal(cross.physical_yeo17_label,check.label17)
    assert np.array_equal(cross.yeo17_network,check.network17)
    cross_meta=json.loads((HERE/'physical_crosswalk_verification.json').read_text())
    atlas_paths=list(cross_meta['atlas_map_sha256'])
    for path,value in cross_meta['atlas_map_sha256'].items():assert sha(path)==value
    paths={int('17Networks' in p):p for p in atlas_paths}
    im7,im17=nib.load(paths[0]),nib.load(paths[1])
    assert im7.shape==im17.shape and np.array_equal(im7.affine,im17.affine)
    a=np.asarray(im7.dataobj).astype(int).ravel();b=np.asarray(im17.dataobj).astype(int).ravel()
    assert np.array_equal(a>0,b>0)
    voxel_pairs=np.unique(np.c_[a[a>0],b[a>0]],axis=0)
    assert voxel_pairs.shape==(400,2) and len(np.unique(voxel_pairs[:,1]))==400
    mapping=dict(voxel_pairs)
    assert all(mapping[int(r.native_label)]==r.yeo17_label_id for r in cross.itertuples())
    assert cross.yeo7_network.eq('Default').sum()==91 and cross.yeo17_network.str.startswith('Default').sum()==79
    conn=np.load(RES/'extraction_common_coverage/primary_participant_matrices.npy',mmap_mode='r')
    assert conn.shape==(33,399,399) and np.isfinite(conn).all()
    close(conn,conn.swapaxes(1,2),1e-12)
    assert np.all(np.diagonal(conn,axis1=1,axis2=2)==0)
    i,j=np.triu_indices(399,1);net=cross.yeo17_network.to_numpy(str)
    matrices=[]
    for (a,b),num in zip(PAIRS,EDGES):
        select=((net[i]==a)&(net[j]==b))|((net[i]==b)&(net[j]==a))
        assert int(select.sum())==num
        matrices.append(np.array(conn[:,i[select],j[select]],dtype=float))
    pred=pd.read_csv(HERE/'cpm_prediction_data_learning_rare_minus_frequent.csv')
    tab=pd.read_csv(HERE/'table_cpm_learning_rare_minus_frequent.csv')
    saved_counts=pd.read_csv(HERE/'observed_fold_selection_counts_learning.csv')
    assert list(tab.family)==NAMES and tab.possible_edges.tolist()==EDGES
    assert pred[['subject','group']].equals(ids[['subject','group']])
    assert len(saved_counts)==198
    tc=stats.t.isf(.005,30);rcrit=tc/np.sqrt(30+tc*tc)
    scalar=np.zeros((33,6));fold_y=np.zeros(33);observed_counts=[];fallback=np.zeros(6,int)
    prepared=[]
    for test in range(33):
        train=np.delete(np.arange(33),test);gg=groups[train]
        # Group-centering projector, constructed independently of inherited functions.
        projector=np.eye(32)
        for g in ['DS','SS']:
            ix=np.flatnonzero(gg==g);projector[np.ix_(ix,ix)]-=1/len(ix)
        local=y[train].copy()
        mu=y[train][gg==groups[test]].mean()
        fold_y[test]=y[test]-mu
        for g in ['DS','SS']:
            local[gg==g]-=y[train][gg==g].mean()
        cell_prepared=[]
        for f,X in enumerate(matrices):
            train_x=X[train]
            corr=stats.pearsonr(train_x,local[:,None],axis=0)
            signed=np.zeros(X.shape[1]);signed[(corr.pvalue<.01)&(corr.statistic>0)]=1;signed[(corr.pvalue<.01)&(corr.statistic<0)]=-1
            strength=train_x@signed;ssq=np.sum((strength-strength.mean())**2)
            positive=int((signed>0).sum());negative=int((signed<0).sum())
            observed_counts.append([test,ids.subject.iloc[test],NAMES[f],positive,negative,positive+negative])
            if ssq>1e-14:
                beta=np.linalg.lstsq(np.c_[np.ones(32),strength],local,rcond=None)[0]
                scalar[test,f]=np.array([1.,X[test]@signed])@beta
            else:fallback[f]+=1
            xc=train_x-train_x.mean(axis=0)
            norm=np.sqrt(np.sum(xc**2,axis=0))
            normalized=np.divide(xc,norm,out=np.zeros_like(xc),where=norm>0)
            cell_prepared.append((train_x,X[test],normalized))
        prepared.append((train,gg,projector,cell_prepared))
    count_frame=pd.DataFrame(observed_counts,columns=['held_out_index','held_out_subject','family','positive_edges_selected','negative_edges_selected','signed_edges_selected'])
    assert saved_counts[count_frame.columns].equals(count_frame)
    pred_error=close(scalar,pred[[f'predicted_{name}_ms' for name in NAMES]].to_numpy(float),1e-8)
    fold_error=close(fold_y,pred.observed_fold_centered_learning_ms.to_numpy(float))
    close(fallback,tab.zero_strength_folds.to_numpy(float),0)
    obs=correl(fold_y[:,None],scalar[:,None,:])[0]
    r_error=close(obs,tab.r_loocv.to_numpy(float),1e-12)
    # Independently replay every archived within-group map and every nested result.
    ds=np.flatnonzero(groups=='DS');ss=np.flatnonzero(groups=='SS')
    ds_map=np.load(HERE/'null_archive/ds_source_indices.npy');ss_map=np.load(HERE/'null_archive/ss_source_indices.npy')
    assert ds_map.shape==(10000,17) and ss_map.shape==(10000,16)
    assert np.array_equal(np.sort(ds_map,axis=1),np.broadcast_to(np.arange(17),ds_map.shape))
    assert np.array_equal(np.sort(ss_map,axis=1),np.broadcast_to(np.arange(16),ss_map.shape))
    rng=np.random.default_rng(20261004)
    for draw in range(10000):
        assert np.array_equal(ds_map[draw],rng.permutation(np.arange(17))),draw
        assert np.array_equal(ss_map[draw],rng.permutation(np.arange(16))),draw
    assert rng.bit_generator.state==archive_manifest['rng_state_after_source_maps']
    null=np.load(HERE/'null_archive/permutation_r.npy')
    assert null.shape==(10000,6) and np.isfinite(null).all()
    verified_null=np.empty_like(null);max_null_error=0.
    for start in range(0,10000,100):
        stop=min(start+100,10000);size=stop-start
        yp=np.empty((33,size));yp[ds]=y[ds][ds_map[start:stop]].T;yp[ss]=y[ss][ss_map[start:stop]].T
        yp_pred=np.zeros((33,size,6));yp_target=np.empty((33,size))
        for test,(train,gg,P,cell_prepared) in enumerate(prepared):
            yy=yp[train];yt=P@yy
            yp_target[test]=yp[test]-yy[gg==groups[test]].mean(axis=0)
            ynorm=np.sqrt(np.sum(yt**2,axis=0))
            for f,(xt,xte,normalized) in enumerate(cell_prepared):
                edge_r=(normalized.T@yt)/ynorm[None,:]
                weights=np.zeros_like(edge_r);weights[edge_r>rcrit]=1;weights[edge_r<-rcrit]=-1
                strength=xt@weights;z=strength-strength.mean(axis=0)
                denom=np.sum(z*z,axis=0)
                numer=np.sum(z*yt,axis=0)
                coefficient=np.divide(numer,denom,out=np.zeros(size),where=denom>1e-14)
                yp_pred[test,:,f]=coefficient*((xte@weights)-strength.mean(axis=0))
        replay=correl(yp_target,yp_pred)
        max_null_error=max(max_null_error,close(replay,null[start:stop],1e-9))
        verified_null[start:stop]=replay
        if stop%1000==0:print(f'independently replayed {stop}/10000',flush=True)
    np.save(HERE/'independent_replayed_permutation_r.npy',verified_null)
    draw_csv=pd.read_csv(HERE/'cpm_per_draw_null_summaries.csv')
    assert np.array_equal(draw_csv.draw,np.arange(10000))
    close(draw_csv[[f'r_{name}' for name in NAMES]].to_numpy(),null,1e-14)
    upper=np.sum(null>=obs,axis=0);lower=np.sum(null<=obs,axis=0)
    peq=np.minimum(1,2*np.minimum((upper+1)/10001,(lower+1)/10001))
    pabs=(np.sum(np.abs(null)>=np.abs(obs),axis=0)+1)/10001
    close(upper,tab.equal_tail_upper_exceedances.to_numpy(),0);close(lower,tab.equal_tail_lower_exceedances.to_numpy(),0)
    close(peq,tab.p_raw_equal_tail.to_numpy(),1e-14);close(pabs,tab.p_raw_absolute_sensitivity.to_numpy(),1e-14)
    close(multipletests(peq,method='holm')[1],tab.p_Holm6_equal_tail.to_numpy(),1e-14)
    close(multipletests(pabs,method='holm')[1],tab.p_Holm6_absolute_sensitivity.to_numpy(),1e-14)
    for col,vals in [('null_mean',null.mean(axis=0)),('null_sd',null.std(axis=0,ddof=1)),('null_q025',np.quantile(null,.025,axis=0)),('null_q975',np.quantile(null,.975,axis=0))]:close(vals,tab[col].to_numpy(),1e-12)
    # Bootstrap call order inherited alphabetic groups: DS then SS.
    boot=np.load(HERE/'null_archive/fixed_prediction_bootstrap_r.npy');assert boot.shape==(10000,6)
    brng=np.random.default_rng(20261005);max_boot_error=0.
    for draw in range(10000):
        idx=np.r_[brng.choice(ds,len(ds),replace=True),brng.choice(ss,len(ss),replace=True)]
        yy=pred.observed_fold_centered_learning_ms.to_numpy(float)[idx]
        pp=pred[[f'predicted_{name}_ms' for name in NAMES]].to_numpy(float)[idx]
        recomputed=correl(yy[:,None],pp[:,None,:])[0]
        max_boot_error=max(max_boot_error,close(recomputed,boot[draw],1e-12))
    close(np.quantile(boot,.025,axis=0),tab.fixed_prediction_bootstrap_ci_low_descriptive.to_numpy(),1e-12)
    close(np.quantile(boot,.975,axis=0),tab.fixed_prediction_bootstrap_ci_high_descriptive.to_numpy(),1e-12)
    for path,value in source_hashes.items():assert sha(path)==value,path
    for path,value in protected_hashes.items():assert sha(path)==value,path
    for path,value in frozen_archive_hashes.items():assert sha(HERE/path)==value,path
    report={'status':'all_checks_passed','generator_imported':False,'legacy_helpers_imported':False,
      'subjects':33,'group_counts':{'DS':17,'SS':16},'parent_trials':10995,'biased_trials':6721,'participant_run_region_rows':132,
      'target_reconstruction_max_abs_error_ms':target_error,'existing_learning_score_max_abs_error_ms':old_error,
      'physical_atlas_voxel_crosswalk_verified':True,'six_cell_edge_counts':EDGES,'observed_folds_verified':198,
      'max_fold_prediction_error_ms':pred_error,'max_fold_target_error_ms':fold_error,'max_observed_r_error':r_error,
      'completed_permutations':10000,'all_seeded_maps_exact':True,'all_60000_null_r_values_recomputed':True,
      'max_null_r_replay_error':max_null_error,'bootstrap_draws_replayed':10000,'max_bootstrap_r_error':max_boot_error,
      'equal_tail_and_absolute_tail_verified':True,'separate_Holm6_families_verified':True,
      'immutable_inputs_match_freeze':True,'protected_files_unchanged_during_additional_verification':True,
      'all_protected_files_match_freeze':not bool(protected_differences),'protected_differences_from_freeze':protected_differences,
      'frozen_input_paths_checked':len(source_hashes),'protected_paths_checked':len(protected_hashes),
      'archive_hashes_unchanged':True,'independent_replay_sha256':sha(HERE/'independent_replayed_permutation_r.npy'),
      'verifier_sha256':sha(__file__),'duration_seconds':time.monotonic()-started}
    (HERE/'independent_verification_additional.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
