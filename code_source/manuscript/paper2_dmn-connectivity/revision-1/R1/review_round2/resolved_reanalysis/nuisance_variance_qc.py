#!/usr/bin/env python3
"""Descriptive nuisance variance of actual filtered parcel means, not raw voxel BOLD.
Never assigns independent-frame p values/effective temporal df to filtered series.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from nilearn import signal
from nilearn.glm.first_level import make_first_level_design_matrix
HERE=Path(__file__).resolve().parent
MOTION=['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def r2(Y,X):
    Z=np.column_stack([np.ones(len(Y)),X]);fit=Z@np.linalg.lstsq(Z,Y,rcond=None)[0]
    rss=np.sum((Y-fit)**2,axis=0);tss=np.sum((Y-Y.mean(axis=0))**2,axis=0)
    assert np.all(tss>0)
    return 1-rss/tss,int(np.linalg.matrix_rank(Z))

def evaluate(Y,blocks):
    order=['motion24','WM_CSF','global_signal','task_Glover'];X=np.column_stack([blocks[x] for x in order]);full,rank=r2(Y,X);out={'full30':(full,rank)}
    primary,rankp=r2(Y,np.column_stack([blocks['motion24'],blocks['WM_CSF']]));out['primary26']=(primary,rankp)
    for key in order:
        red,rankred=r2(Y,np.column_stack([blocks[k] for k in order if k!=key]));delta=full-red
        assert delta.min()>-1e-10
        out['incremental_'+key]=(np.maximum(delta,0),rank)
    return out

def selftest():
    rng=np.random.default_rng(91);blocks={'motion24':rng.normal(size=(80,24)),'WM_CSF':rng.normal(size=(80,2)),'global_signal':rng.normal(size=(80,1)),'task_Glover':rng.normal(size=(80,3))};Y=blocks['motion24'][:,[0]]@np.array([[1,2,3]])+rng.normal(size=(80,3))
    ans=evaluate(Y,blocks);assert (ans['full30'][0]>=ans['primary26'][0]-1e-10).all();assert len(ans)==6
    # Independent projection/SSE algebra for the full model.
    X=np.column_stack([np.ones(80),*[blocks[k] for k in blocks]]);Q=np.linalg.qr(X,mode='reduced')[0];resid=Y-Q@(Q.T@Y);test=1-np.sum(resid**2,axis=0)/np.sum((Y-Y.mean(axis=0))**2,axis=0)
    assert np.allclose(ans['full30'][0],test,atol=1e-12)
    print(json.dumps({'status':'SYNTHETIC_UNIT_TEST_ONLY','maximum_projection_error':float(np.max(np.abs(ans['full30'][0]-test)))}))

def resolve(value, manifest):
    p=Path(value)
    return p if p.is_absolute() else manifest.parent/p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--extraction',type=Path,default=HERE/'extraction_common_coverage');ap.add_argument('--input-manifest',type=Path);ap.add_argument('--out',type=Path,default=HERE/'nuisance_variance');args=ap.parse_args()
    if args.self_test:selftest();return
    # The raw extraction manifest is intentionally not an analysis-ready manifest.
    # Read the adapter output and use its explicit paths; never guess legacy keys.
    manifest_path=args.input_manifest or (HERE/'common399_analysis_ready'/'analysis_ready_input_manifest.json')
    cfg=json.loads(manifest_path.read_text());assert cfg['status']=='complete' and not cfg.get('synthetic_test_only',False)
    score_path=resolve(cfg['participant_scores']['path'],manifest_path)
    parcel_path=resolve(cfg['parcel_metadata']['path'],manifest_path)
    scores=pd.read_csv(score_path,dtype={'subject':str});assert len(scores)==33 and scores.subject.is_unique
    parcels=pd.read_csv(parcel_path).sort_values('matrix_index');assert len(parcels)==399 and set(parcels.native_label)==set(range(1,401))-{323}
    assert parcels.yeo7_network.value_counts().to_dict()=={'Default':91,'SomMot':77,'Vis':61,'Cont':52,'SalVentAttn':47,'DorsAttn':46,'Limbic':25}
    default=(parcels.yeo7_network=='Default').to_numpy();assert default.sum()==91
    rows=[];summaries=[];sources={str(parcel_path):sha(parcel_path),str(score_path):sha(score_path),str(manifest_path):sha(manifest_path),str(__file__):sha(__file__)}
    for rec in scores.itertuples():
        mp=args.extraction/'manifests'/f'sub-{rec.subject}_manifest.json';m=json.loads(mp.read_text());assert m['status']=='complete';sources[str(mp)]=sha(mp)
        for rr in m['runs']:
            rawp=Path(rr['raw_timeseries_path']);cfp=Path(rr['source_inputs']['confounds']);evp=Path(rr['source_inputs']['events']);sources.update({str(p):sha(p) for p in [rawp,cfp,evp]})
            raw=np.load(rawp);assert raw.shape[1]==399 and np.isfinite(raw).all()
            conf=pd.read_csv(cfp,sep='\t',na_values=['n/a','NA','nan']);assert len(conf)==len(raw)
            removed=rr['nuisance']['nonsteady_removed_indices'];keep=np.ones(len(raw),bool);keep[removed]=False
            num=conf[MOTION+[f'{v}_derivative1' for v in MOTION]+['white_matter','csf','global_signal']].apply(pd.to_numeric,errors='coerce')
            for name in [f'{v}_derivative1' for v in MOTION]:
                bad=np.flatnonzero(~np.isfinite(num[name]));assert len(bad)==0 or np.array_equal(bad,[0])
                if len(bad):num.loc[0,name]=0
            assert np.isfinite(num.to_numpy()).all()
            motion=np.column_stack([num[MOTION],num[[f'{v}_derivative1' for v in MOTION]],num[MOTION].to_numpy()**2,num[[f'{v}_derivative1' for v in MOTION]].to_numpy()**2])
            ev=pd.read_csv(evp,sep='\t');codes={'Same':'Same','Different':'Different','None':'Absent','Absent':'Absent'};assert set(ev.Dimension.astype(str))<=set(codes)
            events=pd.DataFrame({'onset':ev.Onset,'duration':ev.Duration,'trial_type':ev.Dimension.astype(str).map(codes)})
            dm=make_first_level_design_matrix(np.arange(len(raw)),events,hrf_model='glover',drift_model=None)
            blocks={'motion24':motion,'WM_CSF':num[['white_matter','csf']].to_numpy(),'global_signal':num[['global_signal']].to_numpy(),'task_Glover':dm[['Absent','Different','Same']].to_numpy()}
            kwargs=dict(detrend=True,standardize=False,low_pass=.1,high_pass=.01,t_r=1.,sample_mask=np.flatnonzero(keep),ensure_finite=False)
            Y=signal.clean(raw,**kwargs);filtered={k:signal.clean(v,**kwargs) for k,v in blocks.items()}
            estimates=evaluate(Y,filtered)
            for kind,(values,rank) in estimates.items():
                assert np.isfinite(values).all()
                for idx,v in enumerate(values):rows.append({'subject':rec.subject,'group':rec.group,'run':rr['run'],'native_label':int(parcels.native_label.iloc[idx]),'network7':parcels.yeo7_network.iloc[idx],'statistic':kind,'R2_or_incremental_R2':float(v),'frames_retained':len(Y),'algebraic_full_or_primary_design_rank':rank,'algebraic_N_minus_rank_not_effective_temporal_df':len(Y)-rank})
                for name,mask in [('common399',np.ones(399,bool)),('Default91',default)]:
                    v=values[mask];summaries.append({'subject':rec.subject,'group':rec.group,'run':rr['run'],'summary_domain':name,'statistic':kind,'mean_R2':float(v.mean()),'median_R2':float(np.median(v)),'min_R2':float(v.min()),'max_R2':float(v.max()),'frames_retained':len(Y),'design_rank':rank})
    args.out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(args.out/'parcel_level_nuisance_variance.csv',index=False);s=pd.DataFrame(summaries);s.to_csv(args.out/'run_level_nuisance_variance.csv',index=False)
    # Equal participant/run means rather than treating parcels or time samples as independent cases.
    aggregate=s.groupby(['summary_domain','statistic']).mean_R2.agg(['count','mean','std','min','max']).reset_index();aggregate.to_csv(args.out/'nuisance_variance_summary.csv',index=False)
    report={'status':'complete','subjects':33,'runs':66,'active_parcels':399,'primary_Default_parcels':91,'scope':'Descriptive variance explained in jointly detrended/band-pass-filtered parcel mean series after identical nonsteady-state sample masking. Not raw voxel BOLD explained variance.','definitions':'Full30 R2 uses motion24+WMCSF+GSR+3task. Primary26 R2 uses motion24+WMCSF. Incremental block R2 is R2(full30)-R2(full30 excluding that block), holding all other blocks fixed; values are not additive variance partitions.','df_caveat':'Algebraic N-minus-design-rank only; temporal autocorrelation and Butterworth filtering mean it is not an effective temporal degrees of freedom. No independent-frame p values.','software':{'numpy':np.__version__,'nilearn':__import__('nilearn').__version__},'input_sha256':sources}
    (args.out/'variance_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(aggregate.to_string(index=False))
if __name__=='__main__':main()
