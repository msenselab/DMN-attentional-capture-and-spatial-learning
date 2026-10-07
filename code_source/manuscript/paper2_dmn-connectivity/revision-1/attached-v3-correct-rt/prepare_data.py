#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd
ROOT=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[4])))
OUT=Path(__file__).resolve().parent/'brain_behavior';OUT.mkdir(exist_ok=True)
raw_path=ROOT/'derivatives/behavior/vigilance_state/merged_events_with_correct.csv'
frozen_path=ROOT/'manuscript/paper2/code_and_data/dmn_capture_unified.csv'
f=pd.read_csv(frozen_path);assert f.subject.is_unique and len(f)==33
raw=pd.read_csv(raw_path).sort_values(['subject','run','trial']).reset_index(drop=True)
assert not raw.duplicated(['subject','run','trial']).any()
raw['group']=raw.subject.str[:2];raw['dimension']=np.where(raw.Distractor.eq(0),'Absent',raw.Dimension)
raw['category']=np.where(raw.dimension.eq('Absent'),'absent',np.where((raw.group.eq('SS')&raw.dimension.eq('Same'))|(raw.group.eq('DS')&raw.dimension.eq('Different')),'biased','unbiased'))
raw['global_order']=raw.groupby('subject').cumcount()+1
raw['first_run']=raw.groupby(['subject','run']).cumcount().eq(0)
raw['prev_category']=raw.groupby(['subject','run']).category.shift()
for cat in ['biased','unbiased']:
 raw['prior_'+cat]=raw.groupby('subject').category.transform(lambda s:s.eq(cat).cumsum().shift(fill_value=0))
 for loc in ['Frequent','Rare']:
  flag=(raw.category.eq(cat)&raw.DisRegion.eq(loc)).astype(int)
  raw['prior_'+cat+'_'+loc.lower()]=flag.groupby(raw.subject).transform(lambda s:s.cumsum().shift(fill_value=0))
keep=raw.correct.eq(1)&raw.RT.between(.2,3,inclusive='both')&np.isfinite(raw.RT)
d=raw.loc[keep].copy();d['RT_ms']=d.RT*1000;d['RT100']=d.RT_ms/100;d['B']=d.category.eq('biased').astype(int);d['U']=d.category.eq('unbiased').astype(int);d['D']=d.group.eq('DS').astype(int)
d['log_order']=np.log(d.global_order);d['log_order_c']=d.log_order-d.groupby('subject').log_order.transform('mean')
d.to_csv(OUT/'correct_rt_trials_n35.csv',index=False)
n=d.loc[d.subject.isin(f.subject)].copy().merge(f[['subject','dmn_conn']],on='subject',validate='many_to_one')
s=f[['subject','group','dmn_conn']].copy();s['D']=s.group.eq('DS').astype(int);s['X']=(s.dmn_conn-s.dmn_conn.mean())/s.dmn_conn.std(ddof=1)
n=n.merge(s[['subject','X']],on='subject',validate='many_to_one').sort_values(['subject','run','trial'])
assert n.subject.nunique()==33 and (n.correct.eq(1)&n.RT.between(.2,3)).all()
n.to_csv(OUT/'correct_rt_trials_n33.csv',index=False)
means=n.pivot_table(index='subject',columns='category',values='RT_ms',aggfunc='mean');cnt=n.groupby(['subject','category']).size().unstack()
s=s.merge(means.reset_index(),on='subject').merge(cnt.add_prefix('n_').reset_index(),on='subject')
s['biased_capture']=s.biased-s.absent;s['unbiased_capture']=s.unbiased-s.absent
s['biased_weight']=s.n_biased/(s.n_biased+s.n_unbiased);s['unbiased_weight']=1-s.biased_weight
s['all_capture']=s.biased_weight*s.biased_capture+s.unbiased_weight*s.unbiased_capture
s['planned_weight_capture']=(14*s.biased_capture+5*s.unbiased_capture)/19
for name,dim in [('same_capture','Same'),('diff_capture','Different')]:s[name]=n.loc[n.dimension.eq(dim)].groupby('subject').RT_ms.mean().reindex(s.subject).to_numpy()-s.absent
for subject,z in n.groupby('subject'):
 q=z.loc[z.category.eq('biased')&z.prev_category.notna()]
 same=q.loc[q.prev_category.eq('biased'),'RT_ms'];other=q.loc[q.prev_category.ne('biased'),'RT_ms']
 s.loc[s.subject.eq(subject),'biased_priming']=same.mean()-other.mean()
 s.loc[s.subject.eq(subject),'n_priming_same']=len(same);s.loc[s.subject.eq(subject),'n_priming_other']=len(other)
for col in ['dmn_conn','biased_capture','unbiased_capture','all_capture','planned_weight_capture','biased_priming']:
 s[col+'_gc']=s[col]-s.groupby('group')[col].transform('mean')
s.to_csv(OUT/'participant_scores_n33.csv',index=False)
flow=[]
for label,z in [('N35',raw),('N33',raw.loc[raw.subject.isin(f.subject)])]:
 corr=z.correct.eq(1);valid=z.RT.between(.2,3)&np.isfinite(z.RT);ret=z.loc[corr&valid]
 flow.append(dict(sample=label,N=z.subject.nunique(),assigned=len(z),incorrect=int((~corr).sum()),additional_RT_invalid=int((corr&~valid).sum()),retained=len(ret),first_in_run_retained=int(ret.first_run.sum()),B=int(ret.category.eq('biased').sum()),U=int(ret.category.eq('unbiased').sum()),absent=int(ret.category.eq('absent').sum())))
pd.DataFrame(flow).to_csv(OUT/'trial_flow.csv',index=False)
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw_path,frozen_path,Path(__file__)]}
(OUT/'data_manifest.json').write_text(json.dumps({'input_hashes':manifest,'rule':'correct==1 and RT 200–3000ms inclusive; retain first-in-run','flow':flow,'group_N':s.groupby('group').size().to_dict(),'row_keys_sha256':hashlib.sha256(n[['subject','run','trial']].to_csv(index=False).encode()).hexdigest()},indent=2)+'\n')
print(pd.DataFrame(flow).to_string(index=False));print(s.groupby('group')[['biased_capture','unbiased_capture','all_capture']].agg(['mean','std']))
