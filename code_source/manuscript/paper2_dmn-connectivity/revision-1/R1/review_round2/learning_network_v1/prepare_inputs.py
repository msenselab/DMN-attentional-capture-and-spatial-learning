from pathlib import Path
import hashlib,json,datetime
import pandas as pd
import numpy as np
ROOT=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[6])))
ROUND=ROOT/'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2'
HERE=Path(__file__).resolve().parent
RES=ROUND/'resolved_reanalysis'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
trial=ROUND.parent.parent/'attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv'
# Explicit canonical revision path avoids relying on hierarchy arithmetic.
trial=ROOT/'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv'
inputs=[HERE/'ANALYSIS_SPEC.md',trial,RES/'extraction_common_coverage/participant_scores_resolved.csv',RES/'extraction_common_coverage/primary_participant_matrices.npy',RES/'extraction_common_coverage/primary_matrix_subjects.csv',RES/'common399_analysis_ready/physical_yeo7_common399_parcels.csv',ROUND/'spatial_learning_dmn_v1/participant_analysis/participant_learning_scores.csv',RES/'analysis_contract.json',RES/'analysis_contract_coverage_amendment.json',RES/'cpm/run_resolved_cpm.py',RES/'network_diagnostics/run_network_diagnostics.py',RES/'network_diagnostics/run_enrichment_permutation_archived_v2.py',ROOT/'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/cpm/run_cpm_authorized_nested.py']
freeze=HERE/'input_freeze.json'
assert not freeze.exists(),'Already frozen; refuse overwrite'
d=pd.read_csv(trial,dtype={'subject':str}); ids=pd.read_csv(inputs[4],dtype={'subject':str})
assert len(d)==10995 and d.subject.nunique()==33 and d.correct.eq(1).all() and ((d.RT_ms>200)&(d.RT_ms<=3000)&np.isfinite(d.RT_ms)).all()
b=d[d.category.eq('biased')];assert len(b)==6721 and set(b.DisRegion)=={'Frequent','Rare'}
t=b.groupby(['subject','group','DisRegion']).RT_ms.agg(['mean','size']).unstack('DisRegion')
out=t.reset_index()[['subject','group']].copy();out.columns=['subject','group']
for region in ['Frequent','Rare']:
 out[region.lower()+'_mean_ms']=t[('mean',region)].to_numpy();out['n_'+region.lower()]=t[('size',region)].to_numpy()
out['learning_rare_minus_frequent_ms']=out.rare_mean_ms-out.frequent_mean_ms
out=ids[['subject','group']].merge(out,on=['subject','group'],validate='one_to_one')
old=pd.read_csv(inputs[6],dtype={'subject':str}).set_index('subject').loc[out.subject]
assert np.allclose(out.learning_rare_minus_frequent_ms,old.learning_rare_minus_frequent_ms,atol=1e-10,rtol=0)
assert out.group.value_counts().to_dict()=={'DS':17,'SS':16}
out.to_csv(HERE/'learning_target_scores.csv',index=False)
b.groupby(['subject','group','run','DisRegion']).RT_ms.agg(['size','mean']).reset_index().to_csv(HERE/'learning_target_run_counts.csv',index=False)
protected={}
for folder in [RES/'cpm',RES/'network_diagnostics',RES/'editorial/final_sources_v8_dimension_spatial_learning',RES/'editorial/final_sources_v9_location_standardized_capture',ROUND/'current_verified_author_review_v8_dimension_spatial_learning']:
 for p in sorted(folder.rglob('*')):
  if p.is_file() and '__pycache__' not in str(p):protected[str(p)]=sha(p)
state={'status':'FROZEN_BEFORE_NEW_NETWORK_RESULTS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_hashes':{str(p):sha(p) for p in inputs},'target_hash':sha(HERE/'learning_target_scores.csv'),'protected_capture_and_source_hashes':protected,'n':33,'biased_trials':6721,'parent_trials':10995,'target_reconstruction_max_abs_error':float(np.max(np.abs(out.learning_rare_minus_frequent_ms.to_numpy()-old.learning_rare_minus_frequent_ms.to_numpy())))}
freeze.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({k:v for k,v in state.items() if 'hashes' not in k},indent=2));print('Protected files:',len(protected));print('Spec SHA256:',sha(HERE/'ANALYSIS_SPEC.md'))
