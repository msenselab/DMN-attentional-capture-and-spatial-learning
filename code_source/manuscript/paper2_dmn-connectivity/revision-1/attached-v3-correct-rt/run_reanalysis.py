#!/usr/bin/env python3
"""Correct-response 200--3000-ms behavioral refresh for attached v3.

Writes only within this directory. Raw scanner correctness events and training MAT
exports are immutable inputs. First trials are retained by design.
"""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys, textwrap, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[4])))
OUT=ROOT/'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt'
TAB=OUT/'tables'; FIG=OUT/'figures'; ART=OUT/'artifacts'
SEED=20261001; R=10000
COLS=['tar_ori','conditions','sessions','dist_cond','blk','tar_pos','as_pos','response_group','tar_orient_group','c_orient','key','correct','tm_cue','tm_disp','tm_key','rt','tm_end','tm_trigger']
BM={1:1,2:2,5:3,6:4,7:5,8:6}
PATHS={
 'scanner_correctness_events':ROOT/'derivatives/behavior/vigilance_state/merged_events_with_correct.csv',
 'training_mat_directory':ROOT/'code/cross_dimension/data',
 'participants':ROOT/'participants.tsv',
 'frozen_neural_n33':ROOT/'manuscript/paper2/code_and_data/dmn_capture_unified.csv',
 'training_sequence_code':ROOT/'_archive/manuscript_scope_20260922/review_needed_code/code/mri-code/GenerateTrialSequences.m',
}
def sha(p):
 h=hashlib.sha256()
 if p.is_dir(): return 'directory_not_hashed'
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ptxt(p): return '< .001' if p<.001 else f'= {p:.3f}'.replace('0.','.')
def ci_mean(x):
 x=np.asarray(x,float); n=len(x); m=x.mean(); sd=x.std(ddof=1); se=sd/np.sqrt(n); q=stats.t.ppf(.975,n-1)
 return m,sd,se,m-q*se,m+q*se
def dz_boot(x,rng):
 x=np.asarray(x,float); n=len(x); vals=[]
 for _ in range(R):
  z=rng.choice(x,n,replace=True); s=z.std(ddof=1); vals.append(z.mean()/s if s>0 else np.nan)
 return np.nanquantile(vals,[.025,.975])
def one_sample(x,label,rng):
 x=np.asarray(x,float); n=len(x); m,sd,se,lo,hi=ci_mean(x); t,p=stats.ttest_1samp(x,0)
 bci=dz_boot(x,rng)
 return dict(measure=label,n=n,mean_ms=m,sd_ms=sd,sem_ms=se,ci_low_ms=lo,ci_high_ms=hi,t=float(t),df=n-1,p_raw=float(p),cohens_dz=float(t/np.sqrt(n)),dz_boot_ci_low=float(bci[0]),dz_boot_ci_high=float(bci[1]))
def hedges_boot(a,b,rng):
 a=np.asarray(a,float); b=np.asarray(b,float); vals=[]
 for _ in range(R):
  aa=rng.choice(a,len(a),replace=True); bb=rng.choice(b,len(b),replace=True)
  sp=np.sqrt(((len(aa)-1)*aa.var(ddof=1)+(len(bb)-1)*bb.var(ddof=1))/(len(aa)+len(bb)-2))
  j=1-3/(4*(len(aa)+len(bb))-9); vals.append(j*(bb.mean()-aa.mean())/sp if sp else np.nan)
 return np.nanquantile(vals,[.025,.975])
def welch(a,b,label,rng): # b minus a: DS minus SS
 a=np.asarray(a,float); b=np.asarray(b,float); na,nb=len(a),len(b); ma,sa,sea,loa,hia=ci_mean(a); mb,sb,seb,lob,hib=ci_mean(b)
 diff=mb-ma; sed=np.sqrt(sa*sa/na+sb*sb/nb); df=(sa*sa/na+sb*sb/nb)**2/((sa*sa/na)**2/(na-1)+(sb*sb/nb)**2/(nb-1)); t=diff/sed; p=2*stats.t.sf(abs(t),df); q=stats.t.ppf(.975,df)
 sp=np.sqrt(((na-1)*sa*sa+(nb-1)*sb*sb)/(na+nb-2)); j=1-3/(4*(na+nb)-9); g=j*diff/sp; bc=hedges_boot(a,b,rng)
 return dict(measure=label,n_SS=na,n_DS=nb,difference_DS_minus_SS_ms=diff,se_ms=sed,ci_low_ms=diff-q*sed,ci_high_ms=diff+q*sed,t=t,df=df,p_raw=p,hedges_g=g,g_boot_ci_low=bc[0],g_boot_ci_high=bc[1])
def centered_corr(x,y):
 r,p=stats.pearsonr(x,y); n=len(x); z=np.arctanh(r); se=1/np.sqrt(n-3); lo,hi=np.tanh([z-1.96*se,z+1.96*se]); return r,p,lo,hi
def load_map_demo():
 m={}; rows=[]
 for line in PATHS['participants'].read_text().splitlines()[1:]:
  p=line.split('\t'); # header is malformed but data are 5 columns
  if len(p)>=5:
   sid=p[0].replace('sub-',''); origin=p[4].split('_')[0]; m[origin]=sid; rows.append(dict(subject=sid,age=float(p[1]),sex=p[2],group_file=p[3],origin_name=p[4]))
 return m,pd.DataFrame(rows)
def cat(group,dim):
 if dim=='Absent': return 'absent'
 return 'biased' if (group=='SS' and dim=='Same') or (group=='DS' and dim=='Different') else 'unbiased'
def load_training(idmap):
 frames=[]
 for p in sorted(PATHS['training_mat_directory'].glob('*.mat')):
  oid,sess=p.stem.split('_',1)
  if sess not in {'s11','s12','s21','s22'}: continue
  x=pd.DataFrame(loadmat(p)['trials'],columns=COLS); x['source_file']=p.name; x['source_subject']=oid; x['subject']=idmap.get(oid,oid)
  x['group']=np.where(x.sessions.eq(2),'SS','DS'); x['mat_block']=x.blk.astype(int); x=x[x.mat_block.isin(BM)].copy(); x['block']=x.mat_block.map(BM).astype(int)
  x['dimension']=np.select([x.conditions.eq(0),x.conditions.eq(2)],['Absent',np.where(x.group.eq('SS'),'Same','Different')],default=np.where(x.group.eq('SS'),'Different','Same'))
  x['category']=[cat(g,d) for g,d in zip(x.group,x.dimension)]; x['location_direct']=x.dist_cond.map({0:'Absent',1:'Rare',2:'Frequent'}).fillna('Unknown'); x['location']=x.location_direct
  x['rt_ms']=x.rt*1000; x['correct_bool']=x.correct.eq(1); x['rt_valid']=np.isfinite(x.rt_ms)&x.rt_ms.ge(200)&x.rt_ms.le(3000); x['retained']=x.correct_bool&x.rt_valid; x['phase']='training'; x['run']=sess; x['trial']=x.groupby(['source_file','mat_block']).cumcount()+1
  frames.append(x)
 return pd.concat(frames,ignore_index=True)
def load_scanner():
 x=pd.read_csv(PATHS['scanner_correctness_events']).copy(); x['group']=x.subject.str[:2]; x['dimension']=np.where(x.Distractor.eq(0),'Absent',x.Dimension); x['category']=[cat(g,d) for g,d in zip(x.group,x.dimension)]; x['location']=x.DisRegion; x['location_direct']=x.DisRegion
 x['rt_ms']=x.RT*1000; x['correct_bool']=x.correct.eq(1); x['rt_valid']=np.isfinite(x.rt_ms)&x.rt_ms.ge(200)&x.rt_ms.le(3000); x['retained']=x.correct_bool&x.rt_valid; x['phase']='scanner'; x['block']=6+(x.run.astype(int)-1)*4+((x.trial.astype(int)-1)//46+1)
 return x
def flow_table(allx):
 rows=[]
 for key,d in allx.groupby(['phase','group','category','dimension','location'],dropna=False):
  keep=d[d.retained]; rows.append(dict(phase=key[0],group=key[1],category=key[2],dimension=key[3],location=key[4],assigned_trials=len(d),incorrect_trials=int((~d.correct_bool).sum()),rt_invalid_trials=int((~d.rt_valid).sum()),correct_rt_retained_trials=len(keep),correct_rt_retained_percent=100*len(keep)/len(d),rt_mean_retained_ms=keep.rt_ms.mean(),rt_sd_retained_ms=keep.rt_ms.std(ddof=1)))
 return pd.DataFrame(rows).sort_values(['phase','group','category','dimension','location'])
def region_reconcile(tr,sc):
 b=sc[(sc.category=='biased')&(sc.location.isin(['Frequent','Rare']))]
 freq=b[b.location.eq('Frequent')].groupby('subject').DisPos.agg(lambda z: sorted(set(map(int,z)))).to_dict()
 d=tr[(tr.location_direct.isin(['Frequent','Rare']))&(tr.as_pos.ne(0))].copy(); d['scanner_freq_positions']=d.subject.map(freq); d['location_from_scanner_positions']=d.apply(lambda r:'Frequent' if int(r.as_pos) in (r.scanner_freq_positions or []) else 'Rare',axis=1); d['agree']=d.location_direct.eq(d.location_from_scanner_positions)
 summary=pd.DataFrame([dict(training_location_authority='raw dist_cond: acquisition code documents 2=Frequent, 1=Rare',scanner_mapping_authority='participant-specific scanner biased-trial DisRegion/DisPos mapping',rows_compared=len(d),rows_agree=int(d.agree.sum()),rows_disagree=int((~d.agree).sum()),percent_agree=100*d.agree.mean(),decision='Use raw dist_cond mapping for training spatial-learning statistics; scanner-position inference retained only as reconciliation audit.')])
 return d,summary
def make_r(script_path, spatial_input, hier_input, outdir):
 txt=f'''suppressPackageStartupMessages(library(lmerTest))
options(contrasts=c("contr.treatment","contr.poly"))
outdir <- "{outdir}"
cirow <- function(mod,L,label) {{
 z <- contest1D(mod,L=L); est <- as.numeric(z[1,"Estimate"]); se <- as.numeric(z[1,"Std. Error"]); df <- as.numeric(z[1,"df"]); tt <- as.numeric(z[1,"t value"]); pp <- as.numeric(z[1,"Pr(>|t|)"]); q <- qt(.975,df); data.frame(contrast=label,estimate=est,se=se,df=df,t=tt,p_raw=pp,ci_low=est-q*se,ci_high=est+q*se)
}}
s <- read.csv("{spatial_input}"); s$subject<-factor(s$subject); s$ds_group<-as.numeric(s$group=="DS")
m <- lmer(rt_ms ~ block + rare_loc*ds_group + (1|subject), data=s, REML=TRUE)
co <- as.data.frame(summary(m)$coefficients); co$term<-rownames(co); rownames(co)<-NULL; names(co)[names(co)=="Estimate"]<-"estimate"; names(co)[names(co)=="Std. Error"]<-"se"; names(co)[names(co)=="df"]<-"df"; names(co)[names(co)=="t value"]<-"t"; names(co)[names(co)=="Pr(>|t|)"]<-"p_raw"; co$ci_low<-co$estimate-qt(.975,co$df)*co$se; co$ci_high<-co$estimate+qt(.975,co$df)*co$se
write.csv(co,file.path(outdir,"Table_07_spatial_lmm_fixed_effects.csv"),row.names=FALSE)
emmeans::emm_options(lmerTest.limit=20000, pbkrtest.limit=20000); em <- emmeans::emmeans(m, ~ rare_loc | ds_group, at=list(block=mean(s$block)), lmer.df='satterthwaite'); ec <- as.data.frame(summary(emmeans::contrast(em, method='revpairwise'), infer=c(TRUE,TRUE))); write.csv(ec,file.path(outdir,'debug_emmeans_spatial.csv'),row.names=FALSE); ec <- ec[,c('ds_group','estimate','SE','df','t.ratio','p.value','lower.CL','upper.CL')]; names(ec)<-c('ds_group','estimate','se','df','t','p_raw','ci_low','ci_high'); ec$contrast <- ifelse(ec$ds_group==0,'SS rare minus frequent','DS rare minus frequent'); ec<-ec[,c('contrast','estimate','se','df','t','p_raw','ci_low','ci_high')]
ix<-which(co$term=='rare_loc:ds_group'); inter<-data.frame(contrast='DS minus SS learning difference',estimate=co$estimate[ix],se=co$se[ix],df=co$df[ix],t=co$t[ix],p_raw=co$p_raw[ix],ci_low=co$ci_low[ix],ci_high=co$ci_high[ix]); cc<-rbind(ec,inter);write.csv(cc,file.path(outdir,"Table_08_spatial_lmm_contrasts.csv"),row.names=FALSE)
writeLines(capture.output(summary(m)),file.path(outdir,"spatial_lmm_summary.txt"))
h<-read.csv("{hier_input}"); h$subject<-factor(h$subject)
hm<-lmer(rt_ms ~ B*X*D + U*X*D + (1+B+U|subject), data=h, REML=TRUE)
hco<-as.data.frame(summary(hm)$coefficients); hco$term<-rownames(hco);rownames(hco)<-NULL;names(hco)[names(hco)=="Estimate"]<-"estimate";names(hco)[names(hco)=="Std. Error"]<-"se";names(hco)[names(hco)=="df"]<-"df";names(hco)[names(hco)=="t value"]<-"t";names(hco)[names(hco)=="Pr(>|t|)"]<-"p_raw";hco$ci_low<-hco$estimate-qt(.975,hco$df)*hco$se;hco$ci_high<-hco$estimate+qt(.975,hco$df)*hco$se;write.csv(hco,file.path(outdir,"Table_14_hierarchical_lmm_fixed_effects.csv"),row.names=FALSE)
hc<-rbind(transform(hco[hco$term=='B:X',],contrast='SS DMN slope for biased capture'),transform(hco[hco$term=='X:U',],contrast='SS DMN slope for unbiased capture'),transform(hco[hco$term=='B:X:D',],contrast='Group moderation of biased DMN slope (DS-SS)'),transform(hco[hco$term=='X:D:U',],contrast='Group moderation of unbiased DMN slope (DS-SS)'));hc<-hc[,c('contrast','estimate','se','df','t','p_raw','ci_low','ci_high')];write.csv(hc,file.path(outdir,"Table_15_hierarchical_lmm_contrasts.csv"),row.names=FALSE)
writeLines(capture.output(summary(hm)),file.path(outdir,"hierarchical_lmm_summary.txt"));writeLines(capture.output(isSingular(hm,tol=1e-4)),file.path(outdir,"hierarchical_lmm_singularity.txt"))
'''
 script_path.write_text(txt)
def main():
 OUT.mkdir(parents=True,exist_ok=True); TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); ART.mkdir(exist_ok=True)
 manifest=[]
 for n,p in PATHS.items(): manifest.append(dict(label=n,path=str(p),exists=p.exists(),sha256=sha(p) if p.exists() else 'MISSING'))
 pd.DataFrame(manifest).to_csv(TAB/'Table_01_input_manifest.csv',index=False)
 if not all(x['exists'] for x in manifest): raise RuntimeError('missing input')
 idmap,demo=load_map_demo(); tr=load_training(idmap); sc=load_scanner(); allx=pd.concat([tr,sc],ignore_index=True,sort=False)
 # source-backed label reconciliation; the direct acquisition code maps dist_cond 2/1 to frequent/rare.
 rec,recsum=region_reconcile(tr,sc); rec.to_csv(TAB/'Table_03_training_region_reconciliation_rows.csv',index=False); recsum.to_csv(TAB/'Table_02_training_region_reconciliation_summary.csv',index=False)
 # save transparent records without unneeded raw response fields.
 allx[['phase','source_file','source_subject','subject','group','run','trial','block','dimension','category','location','location_direct','rt_ms','correct_bool','rt_valid','retained']].to_csv(ART/'authorized_trial_records.csv',index=False)
 flow=flow_table(allx); flow.to_csv(TAB/'Table_04_trial_flow_by_phase_group_condition_location.csv',index=False)
 avail=[]
 for phase,d in allx.groupby('phase'):
  for group,g in d.groupby('group'):
   avail.append(dict(phase=phase,group=group,participants_recorded=g.subject.nunique(),assigned_trials=len(g),correct_trials=int(g.correct_bool.sum()),rt_valid_trials=int(g.rt_valid.sum()),correct_rt_retained_trials=int(g.retained.sum()),first_trials_retained=True))
 pd.DataFrame(avail).to_csv(TAB/'Table_05_phase_availability_and_gate_counts.csv',index=False)
 # Scanner condition/capture participant scores.
 sk=sc[sc.retained].copy(); means=sk.groupby(['subject','group','dimension'],observed=True).agg(mean_rt_ms=('rt_ms','mean'),n_trials=('rt_ms','size')).reset_index(); wide=means.pivot(index=['subject','group'],columns='dimension',values='mean_rt_ms'); cnt=means.pivot(index=['subject','group'],columns='dimension',values='n_trials'); pscore=wide.copy(); pscore['same_capture']=pscore['Same']-pscore['Absent']; pscore['diff_capture']=pscore['Different']-pscore['Absent']; pscore['biased_capture']=np.where(pscore.index.get_level_values('group') == 'SS',pscore.same_capture,pscore.diff_capture); pscore['unbiased_capture']=np.where(pscore.index.get_level_values('group') == 'SS',pscore.diff_capture,pscore.same_capture); pscore['all_capture']=((wide['Same']*cnt['Same']+wide['Different']*cnt['Different'])/(cnt['Same']+cnt['Different']))-wide['Absent']; pscore=pscore.reset_index(); pscore.to_csv(TAB/'Table_10_scanner_participant_condition_rts_and_capture_scores.csv',index=False)
 rtrows=[]
 for (group,dim),d in means.groupby(['group','dimension']):
  m,sd,se,lo,hi=ci_mean(d.mean_rt_ms); rtrows.append(dict(group=group,condition=dim,participants=len(d),mean_rt_ms=m,sd_ms=sd,sem_ms=se,ci_low_ms=lo,ci_high_ms=hi,mean_retained_trials=d.n_trials.mean(),min_retained_trials=d.n_trials.min(),max_retained_trials=d.n_trials.max()))
 pd.DataFrame(rtrows).to_csv(TAB/'Table_09_scanner_condition_rt_participant_summary.csv',index=False)
 rng=np.random.default_rng(SEED); tests=[]
 for group,g in pscore.groupby('group'):
  for col,label in [('all_capture','All present weighted'),('same_capture','Same dimension'),('diff_capture','Different dimension'),('biased_capture','Biased dimension'),('unbiased_capture','Unbiased dimension')]: tests.append(dict(group=group,**one_sample(g[col],label,rng)))
 tests=pd.DataFrame(tests); tests['p_holm_within_group_5_capture_tests']=np.nan
 for group in tests.group.unique():
  ix=tests.group.eq(group); tests.loc[ix,'p_holm_within_group_5_capture_tests']=multipletests(tests.loc[ix,'p_raw'],method='holm')[1]
 tests.to_csv(TAB/'Table_11_scanner_capture_one_sample_tests.csv',index=False)
 diffs=[]
 for col,label in [('all_capture','All present weighted'),('same_capture','Same dimension'),('diff_capture','Different dimension'),('biased_capture','Biased dimension'),('unbiased_capture','Unbiased dimension')]: diffs.append(welch(pscore.query("group=='SS'")[col],pscore.query("group=='DS'")[col],label,rng))
 diffs=pd.DataFrame(diffs);diffs['p_holm_5_group_capture_tests']=multipletests(diffs.p_raw,method='holm')[1];diffs.to_csv(TAB/'Table_12_scanner_capture_welch_group_comparisons.csv',index=False)
 # Unified spatial LMM data: only biased trials with direct source location coding.
 learn=allx[allx.retained&(allx.category=='biased')&(allx.location.isin(['Frequent','Rare']))].copy(); learn['rare_loc']=learn.location.eq('Rare').astype(int); learn.to_csv(ART/'spatial_learning_lmm_authorized_trials.csv',index=False)
 # Participant rare-frequent summary and tests.
 pm=learn.groupby(['subject','group','location']).rt_ms.mean().unstack(); ps=(pm['Rare']-pm['Frequent']).rename('rare_minus_frequent_ms').reset_index(); ps.to_csv(TAB/'Table_06_spatial_learning_participant_scores.csv',index=False)
 lr=[]
 for group,g in ps.groupby('group'): lr.append(dict(group=group,**one_sample(g.rare_minus_frequent_ms,'Rare minus Frequent',rng)))
 lr=pd.DataFrame(lr);lr['p_holm_two_group_simple_effects']=multipletests(lr.p_raw,method='holm')[1];lr.to_csv(TAB/'Table_13_spatial_learning_participant_simple_effects.csv',index=False)
 # Epochs: two-block means, 1--3 training and 4--7 scanner.
 learn['epoch']=((learn.block-1)//2+1).astype(int); ep=learn.groupby(['subject','group','epoch','location']).rt_ms.mean().unstack(); ep=(ep['Rare']-ep['Frequent']).rename('rare_minus_frequent_ms').reset_index(); ep['phase']=np.where(ep.epoch.le(3),'Training','Scanner'); epstats=[]
 for (group,epoch,phase),g in ep.groupby(['group','epoch','phase']): epstats.append(dict(group=group,epoch=epoch,phase=phase,**one_sample(g.rare_minus_frequent_ms,'Rare minus Frequent',rng)))
 epstats=pd.DataFrame(epstats);epstats['p_holm_within_group_7_epochs']=np.nan
 for group in epstats.group.unique():
  ix=epstats.group.eq(group);epstats.loc[ix,'p_holm_within_group_7_epochs']=multipletests(epstats.loc[ix,'p_raw'],method='holm')[1]
 ep.to_csv(TAB/'Table_16_epoch_participant_learning_scores.csv',index=False);epstats.to_csv(TAB/'Table_17_epoch_learning_summary_and_holm_tests.csv',index=False)
 # N33 correlations from new authorized capture and immutable neural value.
 neural=pd.read_csv(PATHS['frozen_neural_n33'])[['subject','group','dmn_conn']].copy(); n33=neural.merge(pscore,on=['subject','group'],validate='one_to_one'); n33['Xc']=n33.dmn_conn-n33.groupby('group').dmn_conn.transform('mean'); corrrows=[]; bootrows=[]; permrows=[]
 for out in ['biased_capture','unbiased_capture','all_capture','same_capture','diff_capture']:
  y=n33[out]-n33.groupby('group')[out].transform('mean'); r,p,lo,hi=centered_corr(n33.Xc,y); corrrows.append(dict(outcome=out,n=len(n33),df=len(n33)-2,r=r,p_raw=p,fisher_ci_low=lo,fisher_ci_high=hi))
  if out=='biased_capture':
   br=[]; pr=[]
   for i in range(R):
    idx=np.concatenate([rng.choice(n33.index[n33.group.eq(g)],sum(n33.group.eq(g)),replace=True) for g in ['SS','DS']]); b=n33.loc[idx].copy(); bx=b.dmn_conn-b.groupby('group').dmn_conn.transform('mean'); by=b[out]-b.groupby('group')[out].transform('mean'); br.append(stats.pearsonr(bx,by).statistic)
    q=n33.copy(); q[out]=pd.concat([n33.loc[n33.group.eq(g),out].sample(frac=1,replace=False,random_state=int(rng.integers(2**32-1))).set_axis(n33.index[n33.group.eq(g)]) for g in ['SS','DS']]).sort_index(); py=q[out]-q.groupby('group')[out].transform('mean');pr.append(stats.pearsonr(q.Xc,py).statistic)
   bootrows.append(dict(outcome=out,method='group-stratified participant bootstrap; group means recalculated per resample; percentile',iterations=R,seed=SEED,median_r=np.median(br),ci_low=np.quantile(br,.025),ci_high=np.quantile(br,.975)))
   permrows.append(dict(outcome=out,method='within-group permutation; group-centered outcome',iterations=R,seed=SEED,p_two_sided=(sum(abs(np.array(pr))>=abs(r))+1)/(R+1)))
 corr=pd.DataFrame(corrrows);corr['p_holm_5_correlations']=multipletests(corr.p_raw,method='holm')[1];corr.to_csv(TAB/'Table_18_n33_centered_dmn_capture_correlations.csv',index=False);pd.DataFrame(bootrows).to_csv(TAB/'Table_19_n33_biased_capture_bootstrap.csv',index=False);pd.DataFrame(permrows).to_csv(TAB/'Table_20_n33_biased_capture_permutation.csv',index=False)
 n33.to_csv(TAB/'Table_21_n33_immutable_neural_and_recomputed_capture_scores.csv',index=False)
 # Hierarchical raw RT main input; scale group-centered immutable neural X globally N33.
 h=sc[sc.retained].merge(neural,on=['subject','group'],validate='many_to_one');h['B']=h.category.eq('biased').astype(int);h['U']=h.category.eq('unbiased').astype(int);h['D']=h.group.eq('DS').astype(int);h['Xraw']=h.dmn_conn-h.groupby('group').dmn_conn.transform('mean');h['X']=h.Xraw/h[['subject','Xraw']].drop_duplicates().Xraw.std(ddof=1);h[['subject','group','RT','rt_ms','B','U','D','X','correct','trial','run']].to_csv(ART/'hierarchical_lmm_authorized_trials.csv',index=False)
 rpath=ART/'fit_models.R'; make_r(rpath,ART/'spatial_learning_lmm_authorized_trials.csv',ART/'hierarchical_lmm_authorized_trials.csv',str(TAB)); proc=subprocess.run(['Rscript',str(rpath)],text=True,capture_output=True); (ART/'R_model_stdout.txt').write_text(proc.stdout);(ART/'R_model_stderr.txt').write_text(proc.stderr)
 if proc.returncode: raise RuntimeError('R model failure: '+proc.stderr)
 # Demographics, explicitly N33 neural membership. Age Welch + sex Fisher exact.
 dem=n33[['subject','group']].merge(demo,on='subject',validate='one_to_one'); dem.to_csv(TAB/'Table_22_n33_demographic_source_rows.csv',index=False); drows=[]
 for group,g in dem.groupby('group'):
  m,sd,se,lo,hi=ci_mean(g.age); female=int(g.sex.eq('F').sum()); drows.append(dict(group=group,n=len(g),age_mean=m,age_sd=sd,age_sem=se,age_ci_low=lo,age_ci_high=hi,female_n=female,male_n=int(g.sex.eq('M').sum()),female_percent=100*female/len(g)))
 pd.DataFrame(drows).to_csv(TAB/'Table_23_n33_demographic_group_summary.csv',index=False); age=welch(dem.query("group=='SS'").age,dem.query("group=='DS'").age,'Age years (DS minus SS)',rng); sexct=pd.crosstab(dem.group,dem.sex).reindex(index=['SS','DS'],columns=['F','M'],fill_value=0); chi2,pchi,dfchi,expected=stats.chi2_contingency(sexct.to_numpy(),correction=False); v=np.sqrt(chi2/len(dem)); vb=[]
 for _ in range(R):
  boot=np.concatenate([dem.loc[dem.group.eq(g),['group','sex']].sample(n=int(dem.group.eq(g).sum()),replace=True,random_state=int(rng.integers(2**32-1))).to_records(index=False) for g in ['SS','DS']]); bt=pd.DataFrame.from_records(boot); bx=pd.crosstab(bt.group,bt.sex).reindex(index=['SS','DS'],columns=['F','M'],fill_value=0).to_numpy(); ch=stats.chi2_contingency(bx,correction=False)[0]; vb.append(np.sqrt(ch/len(bt)))
 sexrow=dict(test='Pearson chi-square sex distribution, no continuity correction',SS_female=int(sexct.loc['SS','F']),SS_male=int(sexct.loc['SS','M']),DS_female=int(sexct.loc['DS','F']),DS_male=int(sexct.loc['DS','M']),chi_square=chi2,df=dfchi,p_raw=pchi,cramers_v=v,cramers_v_boot_ci_low=float(np.quantile(vb,.025)),cramers_v_boot_ci_high=float(np.quantile(vb,.975)))
 demo_tests=pd.DataFrame([
  dict(test='Welch two-sample t test',outcome='Age, years (DS minus SS)',n_SS=int(age['n_SS']),n_DS=int(age['n_DS']),estimate=float(age['difference_DS_minus_SS_ms']),se=float(age['se_ms']),ci_low=float(age['ci_low_ms']),ci_high=float(age['ci_high_ms']),statistic_type='t',test_statistic=float(age['t']),df=float(age['df']),p_raw=float(age['p_raw']),standardized_effect='Hedges g',effect_estimate=float(age['hedges_g']),effect_ci_low=float(age['g_boot_ci_low']),effect_ci_high=float(age['g_boot_ci_high'])),
  dict(test='Pearson chi-square, no continuity correction',outcome='Sex distribution (F/M)',n_SS=int(sexrow['SS_female']+sexrow['SS_male']),n_DS=int(sexrow['DS_female']+sexrow['DS_male']),estimate=np.nan,se=np.nan,ci_low=np.nan,ci_high=np.nan,statistic_type='chi-square',test_statistic=float(sexrow['chi_square']),df=int(sexrow['df']),p_raw=float(sexrow['p_raw']),standardized_effect="Cramer's V",effect_estimate=float(sexrow['cramers_v']),effect_ci_low=float(sexrow['cramers_v_boot_ci_low']),effect_ci_high=float(sexrow['cramers_v_boot_ci_high']))
 ]); demo_tests.to_csv(TAB/'Table_24_n33_demographic_tests.csv',index=False)
 # Figure 2
 lcon=pd.read_csv(TAB/'Table_08_spatial_lmm_contrasts.csv'); fig,axs=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True); colors={'SS':'#D98C5F','DS':'#5B4B8A'}
 a=axs[0]; q=lcon[lcon.contrast.isin(['SS rare minus frequent','DS rare minus frequent'])].copy();q['group']=['SS','DS']
 for i,r in q.reset_index(drop=True).iterrows(): a.errorbar(i,r.estimate,yerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',ms=9,color=colors[r.group],capsize=5);a.text(i,r.ci_high+6,f"{r.estimate:.1f}",ha='center',fontsize=9)
 a.axhline(0,color='.45',lw=1);a.set_ylim(min(-10,q.ci_low.min()-8),q.ci_high.max()+18);a.set_xticks([0,1],['SS','DS']);a.set_ylabel('Rare − Frequent RT (ms), model estimate');a.set_title('A. Whole-experiment learning');a.set_xlim(-.55,1.55)
 b=axs[1]
 for g in ['SS','DS']:
  z=epstats[epstats.group.eq(g)].sort_values('epoch'); b.errorbar(z.epoch,z.mean_ms,yerr=z.sem_ms,fmt='o-',color=colors[g],label=g,capsize=3)
 b.axhline(0,color='.45',lw=1);b.axvspan(.5,3.5,color='#eeeeee',zorder=-2);b.axvline(3.5,color='.55',ls='--',lw=1);b.text(2, b.get_ylim()[1]*.92,'Training',ha='center',fontsize=9);b.text(5.5,b.get_ylim()[1]*.92,'Scanner',ha='center',fontsize=9);b.set_xticks(range(1,8));b.set_xlabel('Two-block epoch');b.set_ylabel('Rare − Frequent RT (ms), mean ± SEM');b.set_title('B. Learning time course');b.legend(frameon=False,loc='lower left')
 c=axs[2]; labs=['all_capture','same_capture','diff_capture']; names=['All present\n(weighted)','Same\ndimension','Different\ndimension']; offs={'SS':-.15,'DS':.15}
 for g in ['SS','DS']:
  z=pscore[pscore.group.eq(g)]; vals=[]; errs=[]
  for col in labs:
   m,sd,se,lo,hi=ci_mean(z[col]);vals.append(m);errs.append((hi-lo)/2)
  xx=np.arange(3)+offs[g];c.errorbar(xx,vals,yerr=errs,fmt='o',color=colors[g],capsize=4,label=g)
  for j,col in enumerate(labs): c.scatter(np.full(len(z),j+offs[g]),z[col],color=colors[g],alpha=.25,s=14)
 c.axhline(0,color='.45',lw=1);c.set_xticks(range(3),names);c.set_ylabel('Capture: present − absent RT (ms), mean ± 95% CI');c.set_title('C. Scanner capture');c.legend(frameon=False)
 for ax in axs: ax.spines[['top','right']].set_visible(False)
 fig.savefig(FIG/'Figure2_correct_rt_first_trials.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/'Figure2_correct_rt_first_trials.pdf',bbox_inches='tight');plt.close(fig)
 # Write table index and text.
 tables=[]
 for p in sorted(TAB.glob('Table_*')): tables.append(dict(file=p.name,bytes=p.stat().st_size))
 (TAB/'table_manifest.json').write_text(json.dumps(tables,indent=2)+'\n')
 lfix=pd.read_csv(TAB/'Table_07_spatial_lmm_fixed_effects.csv'); lc=lcon.set_index('contrast'); cap=tests[(tests.group.eq('SS'))&(tests.measure.eq('Same dimension'))].iloc[0]; capd=tests[(tests.group.eq('DS'))&(tests.measure.eq('Same dimension'))].iloc[0]; dsimp=lc.loc['DS minus SS learning difference']; ssimp=lc.loc['SS rare minus frequent']; dslearn=lc.loc['DS rare minus frequent']; cc=corr.set_index('outcome').loc['biased_capture']; hg=pd.read_csv(TAB/'Table_15_hierarchical_lmm_contrasts.csv').set_index('contrast').loc['Group moderation of biased DMN slope (DS-SS)']
 results=f'''# Results text for parent: correct + 200–3000 ms + first trials retained

## Methods/update text
Behavioral RT and capture analyses used trials with `correct == 1` and finite RTs from 200 to 3,000 ms inclusive. First trials of scanner runs and training blocks were retained. No IQR trimming or additional temporal exclusion was applied. Scanner behavior comprised all 35 participants (SS=17, DS=18); training records were available for 31 participants, and the joint training-plus-scanner learning model consequently included the union N=35. Training location was coded directly from the raw `dist_cond` field (1=rare, 2=frequent), as documented in the implemented sequence-generation code; a participant-specific scanner-position reconciliation is reported in Table 2.

## Behavioral learning
The joint raw-RT mixed model (RT ~ block + rare location × group + participant random intercept) estimated a rare-minus-frequent effect of {ssimp.estimate:.2f} ms in SS (95% CI [{ssimp.ci_low:.2f}, {ssimp.ci_high:.2f}], t({ssimp.df:.1f})={ssimp.t:.2f}, p {ptxt(ssimp.p_raw)}) and {dslearn.estimate:.2f} ms in DS (95% CI [{dslearn.ci_low:.2f}, {dslearn.ci_high:.2f}], t({dslearn.df:.1f})={dslearn.t:.2f}, p {ptxt(dslearn.p_raw)}). The DS-minus-SS location-learning difference was {dsimp.estimate:.2f} ms (95% CI [{dsimp.ci_low:.2f}, {dsimp.ci_high:.2f}], t({dsimp.df:.1f})={dsimp.t:.2f}, p {ptxt(dsimp.p_raw)}); positive values indicate a larger rare-minus-frequent difference in DS. Participant-level simple-effect tests and Holm correction are in Table 13; epoch-specific descriptive tests use a separate Holm family within each group (Table 17).

## Scanner capture
In the SS group, same-dimension capture was {cap.mean_ms:.2f} ms (95% CI [{cap.ci_low_ms:.2f}, {cap.ci_high_ms:.2f}]), t({int(cap.df)})={cap.t:.2f}, p {ptxt(cap.p_raw)}, dz={cap.cohens_dz:.2f}. In the DS group, the corresponding same-dimension capture was {capd.mean_ms:.2f} ms (95% CI [{capd.ci_low_ms:.2f}, {capd.ci_high_ms:.2f}]), t({int(capd.df)})={capd.t:.2f}, p {ptxt(capd.p_raw)}, dz={capd.cohens_dz:.2f}. Full participant-balanced RT/capture summaries, uncorrected and Holm p values, effect sizes, and percentile bootstrap CIs are in Tables 9–12.

## Scope note for parent
Tables 14–15 are a supplementary N33 model audit only and are **not** recommended for the final DOCX; the parent’s independently verified raw-X standardized full-joint model is the manuscript source. This package’s manuscript-ready behavioral evidence is in Tables 4–13 and 16–17, with N33 demographics in Tables 22–24.
'''
 (OUT/'RESULTS_TEXT_FOR_PARENT.md').write_text(results)
 report=f'''# Evidence package: attached v3 behavioral refresh

**Authorized rule:** correct trial and finite 200–3000-ms RT inclusive; first trials retained. No IQR trimming. This package writes no manuscript/DOCX edits.

## Verified scope
- Scanner behavior: N=35 (SS=17, DS=18), raw correctness-linked events.
- Training: 31 participants with available MAT records; scanner completes the learning-model union at N=35.
- Connectivity: immutable N=33 frozen neural table (SS=16, DS=17), with all capture scores recomputed from authorized scanner trials.
- Bootstrap/permutation seed: {SEED}; R={R:,}; all intervals explicitly labelled percentile where applicable.

## Inputs and reconciliation
Input hashes are in Table 1. The executed sequence generator explicitly defines `dist_cond` 2=frequent and 1=rare. The training-versus-scanner position reconciliation compared {int(recsum.rows_compared.iloc[0]):,} available non-absent training rows: {int(recsum.rows_agree.iloc[0]):,} agreed and {int(recsum.rows_disagree.iloc[0]):,} disagreed ({recsum.percent_agree.iloc[0]:.2f}% agreement). Direct `dist_cond`, not a session-frequency inference, is used for training location statistics; discrepancies are retained row-wise in Table 3.

## Deliverables
- Tables 4–5: raw flow, phase availability, correctness, fixed-gate retention, explicitly retaining first trials.
- Tables 6–8 and 13, 16–17: participant learning scores, unified LMM effects/contrasts, and seven two-block epochs with separate within-group Holm correction.
- Tables 9–12: scanner condition RTs, participant capture values, one-sample tests, Welch group tests, dz/Hedges g and percentile bootstrap CIs.
- Tables 18–21 and 14–15: N33 recomputed centered correlations/bootstrap/permutation and an audit-only hierarchical group-interaction model; do not use Tables 14–15 for the final DOCX because the parent has an independently verified raw-X full-joint model.
- Tables 22–24: N33 demographic source rows, group summary, age Welch/Hedges g and no-continuity-correction Pearson χ² sex test.
- `figures/Figure2_correct_rt_first_trials.png` and `.pdf`: new A/B/C Figure 2 (SS orange, DS violet).
- `RESULTS_TEXT_FOR_PARENT.md`: standalone results prose; no manuscript edits.

## Interpretation guardrails
The data support the stated corrected-trial descriptive/model estimates. A non-significant CI crossing zero is not equivalence evidence. The group-defined biased dimension has differential prior exposure by design, so spatial-learning and DMN/capture results do not isolate probability learning from the broader exposure history.
'''
 (OUT/'REPORT.md').write_text(report)
 env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,seed=SEED,bootstrap_permutation_iterations=R,first_trials_retained=True,rt_gate_inclusive_ms=[200,3000]);(ART/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
 print(report)
if __name__=='__main__': main()
