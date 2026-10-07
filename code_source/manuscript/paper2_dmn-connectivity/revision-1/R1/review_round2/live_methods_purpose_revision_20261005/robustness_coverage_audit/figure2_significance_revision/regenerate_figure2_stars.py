#!/usr/bin/env python3
"""Add authorized Figure 2 significance annotations to saved estimates only."""
from pathlib import Path
import hashlib,json,shutil
import pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageChops
from docx import Document
from docx.shared import Inches
P=Path(__file__).resolve().parent;P.mkdir(exist_ok=True)
ROOT=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[8])) + '/manuscript/paper2_dmn-connectivity/revision-1');T=ROOT/'attached-v3-correct-rt/tables'
files=['Table_08_spatial_lmm_contrasts.csv','Table_17_epoch_learning_summary_and_holm_tests.csv','Table_11_scanner_capture_one_sample_tests.csv','Table_10_scanner_participant_condition_rts_and_capture_scores.csv','Table_12_scanner_capture_welch_group_comparisons.csv']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(T/n):sha(T/n) for n in files}
for n in files:shutil.copy2(T/n,P/n)
source=ROOT/'R1/coherent_revision/figures/Figure2_current.png';shutil.copy2(source,P/'Figure2_approved_source_before.png');shutil.copy2(ROOT/'R1/coherent_revision/regenerate_figure2.py',P/'approved_plotting_source_before.py')
sp,ep,cs,sub,wel=[pd.read_csv(T/n) for n in files];colors={'SS':'#D98C5F','DS':'#5B4B8A'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False})
def stars(p):return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
records=[]
def bracket(ax,x1,x2,y,h,label):
 ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y],color='.18',lw=.9,clip_on=False)
 ax.text((x1+x2)/2,y+h+1.7,label,ha='center',va='bottom',color='.12',fontsize=11)
def draw(annotate):
 f,axs=plt.subplots(1,3,figsize=(8.5,3.9),gridspec_kw={'width_ratios':[.75,1.2,1.15]});a,b,c=axs
 for x,g in enumerate(['SS','DS']):
  z=sp.iloc[x];a.errorbar(x,z.estimate,yerr=[[z.estimate-z.ci_low],[z.ci_high-z.estimate]],fmt='o',color=colors[g],capsize=3,markersize=5)
 a.set(xticks=[0,1],xticklabels=['SS','DS'],ylabel='Rare − frequent RT (ms)',title='A  Whole-experiment learning');a.set_xlim(-.5,1.5);a.axhline(0,color='.6',lw=.7)
 b.axvspan(.5,3.5,color='.94');b.axvline(3.5,ls='--',color='.5',lw=.8)
 for g in ['SS','DS']:
  z=ep.query('group==@g').sort_values('epoch');b.errorbar(z.epoch,z.mean_ms,yerr=z.sem_ms,color=colors[g],fmt='o-',lw=1,capsize=2,markersize=4,label=g)
 b.set(xticks=range(1,8),xlabel='Two-block epoch',ylabel='Rare − frequent RT (ms)',title='B  Learning time course');b.axhline(0,color='.6',lw=.7);b.legend(frameon=False,loc='lower left');b.text(2,105,'Training',ha='center',fontsize=9);b.text(5.5,105,'Scanner',ha='center',fontsize=9);b.set_ylim(-35,122)
 rng=np.random.default_rng(20261001)
 for gi,g in enumerate(['SS','DS']):
  for xi,(measure,col) in enumerate([('All present weighted','all_capture'),('Same dimension','same_capture'),('Different dimension','diff_capture')]):
   z=cs.query('group==@g and measure==@measure').iloc[0];x=xi+(-.12 if g=='SS' else .12);vals=sub.query('group==@g')[col];c.scatter(x+rng.uniform(-.025,.025,len(vals)),vals,s=8,color=colors[g],alpha=.22)
   c.errorbar(x,z.mean_ms,yerr=[[z.mean_ms-z.ci_low_ms],[z.ci_high_ms-z.mean_ms]],fmt='o',color=colors[g],markersize=4,capsize=2,label=g if xi==0 else None)
 c.set(xticks=[0,1,2],xticklabels=['All present\n(weighted)','Same','Different'],ylabel='Present − absent RT (ms)',title='C  Scanner capture');c.axhline(0,color='.6',lw=.7);c.legend(frameon=False,loc='upper left');c.set_ylim(-55,390)
 if annotate:
  # Panel A: already reported nominal fitted-model interaction/SS-DS contrast.
  z=sp.query('contrast=="DS minus SS learning difference"').iloc[0];lab=stars(z.p_raw);a.set_ylim(-10,75);a.set_yticks(range(0,71,10));bracket(a,0,1,64,2,lab)
  records.append({'panel':'A','comparison':z.contrast,'p_raw':z.p_raw,'p_decision':z.p_raw,'correction':'nominal saved-model contrast','label':lab,'source':files[0]})
  # Panel B: user explicitly chose nominal epochs, with Holm-7 warning in caption.
  for _,z in ep.iterrows():
   lab=stars(z.p_raw)
   records.append({'panel':'B','group':z.group,'epoch':int(z.epoch),'comparison':'rare-minus-frequent versus zero','p_raw':z.p_raw,'p_holm':z.p_holm_within_group_7_epochs,'p_decision':z.p_raw,'correction':'nominal (no epoch survives within-group Holm-7)','label':lab,'source':files[1]})
   if lab:
    assert z.mean_ms>0
    y=z.mean_ms+z.sem_ms+5.5 if z.group=='SS' else z.mean_ms-z.sem_ms-8.5
    b.text(z.epoch,y,lab,ha='center',va='bottom' if z.group=='SS' else 'top',color=colors[z.group],fontsize=12,fontweight='bold')
  # Panel C: SS-DS comparisons within each displayed capture condition, Holm-5.
  for xi,(measure,y) in enumerate([('All present weighted',210),('Same dimension',347),('Different dimension',90)]):
   z=wel.query('measure==@measure').iloc[0];lab=stars(z.p_holm_5_group_capture_tests)
   records.append({'panel':'C','comparison':'DS minus SS: '+measure,'p_raw':z.p_raw,'p_holm':z.p_holm_5_group_capture_tests,'p_decision':z.p_holm_5_group_capture_tests,'correction':'Holm-5 between-group capture family','label':lab,'source':files[4]})
   if lab:bracket(c,xi-.12,xi+.12,y,7,lab)
 f.tight_layout(w_pad=1.1)
 stem='Figure2_significance' if annotate else 'Figure2_baseline_replay'
 f.savefig(P/(stem+'.png'),dpi=300);f.savefig(P/(stem+'.pdf'));plt.close(f)
draw(False)
a=Image.open(source).convert('RGB');b=Image.open(P/'Figure2_baseline_replay.png').convert('RGB');baseline_equal=a.size==b.size and ImageChops.difference(a,b).getbbox() is None
assert baseline_equal,'Plotting source does not reproduce approved figure; audit before modifying'
draw(True)
pd.DataFrame(records).to_csv(P/'annotation_decisions.csv',index=False)
sig=ep[ep.p_raw<.05];assert set(zip(sig.group,sig.epoch))=={('SS',3),('SS',5),('SS',7),('DS',2)};assert (ep.p_holm_within_group_7_epochs>=.05).all()
# Match the current native Google Doc raster dimensions to prevent crop/transport changes.
img=Image.open(P/'Figure2_significance.png').convert('RGB');img.resize((2047,939),Image.Resampling.LANCZOS).save(P/'Figure2_significance_live.png',dpi=(300,300))
d=Document();sec=d.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11);sec.left_margin=sec.right_margin=Inches(.7);d.add_picture(str(P/'Figure2_significance_live.png'),width=Inches(6.8));d.save(P/'private_figure2_transport.docx')
assert all(sha(Path(n))==v for n,v in hashes.items())
proof={'status':'PASS','approved_source_replay_pixel_equal':baseline_equal,'source_sha256':hashes,'scope':'Add stars/brackets only; A y-axis gains headroom. Preserve values, CI/SEM, palette, panel order, canvas and jitter seed.','B_nominal_significant_epochs':{'SS':[3,5,7],'DS':[2]},'B_Holm7_significant_epochs':0,'C_Holm5_stars':{'All present weighted':'**','Same dimension':'*','Different dimension':''},'outputs':{n:sha(P/n) for n in ['Figure2_significance.png','Figure2_significance.pdf','Figure2_significance_live.png']}}
(P/'figure2_stars_verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
