#!/usr/bin/env python3
"""Figure2 label-only regeneration from frozen executed CSVs."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent; R=P.parent; DATA=R.parent/'attached-v3-correct-rt'; T=DATA/'tables'; O=P/'figures';O.mkdir(exist_ok=True)
paths=[T/n for n in ['Table_08_spatial_lmm_contrasts.csv','Table_17_epoch_learning_summary_and_holm_tests.csv','Table_11_scanner_capture_one_sample_tests.csv','Table_10_scanner_participant_condition_rts_and_capture_scores.csv']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in paths}
colors={'SS':'#D98C5F','DS':'#5B4B8A'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False})
sp,ep,cs,sub=[pd.read_csv(p) for p in paths]
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
f.tight_layout(w_pad=1.1);f.savefig(O/'Figure2_current.png',dpi=300);f.savefig(O/'Figure2_current.pdf');plt.close(f)
assert all(sha(p)==h for p,h in [(Path(n),h) for n,h in before.items()])
proof={'scope':'Only PanelC y-axis clarified; preserved all saved plotted estimates/intervals, layout, SS/DS colors and seed. Caption revised in current results.md.','source_sha256':before,'outputs':{p.name:sha(p) for p in [O/'Figure2_current.png',O/'Figure2_current.pdf']}}
(P/'figure2_verification.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2))
