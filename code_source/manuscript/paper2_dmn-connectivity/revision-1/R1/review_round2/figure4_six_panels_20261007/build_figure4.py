"""Requested six-panel extension: frozen capture/learning vectors, no model refits."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import t,norm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
F=Path(__file__).resolve().parent;R=F.parent
C=R/'resolved_reanalysis/network_diagnostics/results_common399_n33'
P=R/'resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv'
L=R/'learning_network_pooled_groups_v1/analysis'
NET=['Default','SomMot','Vis','Cont','SalVentAttn','DorsAttn','Limbic'];DISPLAY=['Default','SomMot','Visual','Control','VAN/SN','DAN','Limbic']
x=pd.read_csv(C/'yeo7_network_summary_scores_n33.csv').set_index('subject');s=pd.read_csv(P).set_index('subject').loc[x.index]
assert len(x)==33 and x.index.is_unique and (x.group==s.group).all() and x.group.value_counts().to_dict()=={'DS':17,'SS':16}
assert np.allclose(s.biased_capture,s.biased-s.absent,atol=1e-10,rtol=0)
assert np.allclose(x['Default--Default'],s.dmn_conn_primary,atol=2e-14,rtol=0)
cap=[]
for scope in ['pooled_group_adjusted','SS','DS']:
 z=x if scope=='pooled_group_adjusted' else x[x.group==scope];b=s.loc[z.index,'biased_capture'].to_numpy(copy=True);a=z.drop(columns='group').to_numpy(copy=True)
 if scope=='pooled_group_adjusted':
  for g in ['SS','DS']:
   mask=z.group.to_numpy()==g;a[mask]-=a[mask].mean(axis=0);b[mask]-=b[mask].mean()
 n=len(z);df=n-3 if scope=='pooled_group_adjusted' else n-2
 for k,pair in enumerate(z.drop(columns='group').columns):
  r=float(np.corrcoef(a[:,k],b)[0,1]);stat=r*np.sqrt(df/(1-r*r));p=float(2*t.sf(abs(stat),df));se=1/np.sqrt(n-4 if scope=='pooled_group_adjusted' else n-3);ci=np.tanh(np.arctanh(r)+np.array([-1,1])*norm.ppf(.975)*se)
  cap.append({'scope':scope,'pair':pair.replace('--','__'),'N':n,'df':df,'r':r,'two_sided_raw_p':p,'fisher95_ci_low':ci[0],'fisher95_ci_high':ci[1],'p_holm28':np.nan})
cap=pd.DataFrame(cap)
for scope in cap.scope.unique():
 mask=cap.scope==scope;cap.loc[mask,'p_holm28']=multipletests(cap.loc[mask,'two_sided_raw_p'],method='holm')[1]
original=pd.read_csv(C/'yeo7_all_28_summary_associations_holm28.csv').set_index('network_pair')
for _,v in cap[cap.scope=='pooled_group_adjusted'].iterrows():
 z=original.loc[v.pair.replace('__','--')]
 for key,old in [('r','r_group_adjusted'),('two_sided_raw_p','p_raw'),('p_holm28','p_Holm_28')]:assert abs(float(v[key])-float(z[old]))<2e-13
cap.to_csv(F/'capture_network_summary_all84.csv',index=False)
# Preserve every original learning estimate and both original correction columns.
learn=pd.read_csv(L/'network_associations_all84.csv');assert len(learn)==84
plot=[]
for outcome,data,letters in [('capture',cap,'ABC'),('learning',learn,'DEF')]:
 for scope,letter in zip(['pooled_group_adjusted','SS','DS'],letters):
  d=data[data.scope==scope];assert len(d)==28
  for _,z in d.iterrows():plot.append({'panel':letter,'outcome':outcome,'scope':scope,**{k:z[k] for k in ['pair','N','df','r','two_sided_raw_p','fisher95_ci_low','fisher95_ci_high','p_holm28']}})
plot=pd.DataFrame(plot);assert len(plot)==168;plot.to_csv(F/'Figure4_plotted_statistics.csv',index=False)
sig=plot[plot.p_holm28<.05];sig.to_csv(F/'Figure4_corrected_markers.csv',index=False)
print('CORRECTED CELLS',sig.to_dict('records'));print('CAPTURE DEFAULT',cap[cap.pair=='Default__Default'].to_dict('records'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none'})
fig=plt.figure(figsize=(6.8,5.85));cmap=plt.get_cmap('PuOr_r').copy();cmap.set_bad('white');vmax=float(np.ceil(plot.r.abs().max()*20)/20)
fig.text(.035,.970,'Attentional capture',fontsize=11,fontweight='bold',ha='left',va='top')
fig.text(.035,.510,'Spatial learning',fontsize=11,fontweight='bold',ha='left',va='top')
axes=[]
for row,outcome in enumerate(['capture','learning']):
 for col,scope in enumerate(['pooled_group_adjusted','SS','DS']):
  letter='ABCDEF'[row*3+col];ax=fig.add_axes([.085+.324*col,.620-.460*row,.239,.278]);axes.append(ax);d=plot[plot.panel==letter].set_index('pair');arr=np.full((7,7),np.nan)
  for i,a in enumerate(NET):
   for j in range(i,7):
    z=d.loc[a+'__'+NET[j]];arr[i,j]=z.r;label=f'{z.r:.2f}'.replace('-0.','−.').replace('0.','.')
    if z.p_holm28<.05:
     label+='*';ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='black',lw=1.35,zorder=5))
    ax.text(j,i,label,ha='center',va='center',fontsize=7.0,color='white' if abs(z.r)>vmax*.78 else 'black')
  im=ax.imshow(arr,cmap=cmap,vmin=-vmax,vmax=vmax,interpolation='none',zorder=0)
  ax.set_xticks(range(7),DISPLAY,rotation=50,ha='right',fontsize=7.0);ax.set_yticks(range(7),DISPLAY,fontsize=7.0)
  ax.set_xticks(np.arange(-.5,7,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True);ax.grid(which='minor',color='white',lw=.4);ax.tick_params(which='both',length=0)
  ax.set_xlim(-.5,6.5);ax.set_ylim(6.5,-.5)
  title='Pooled' if scope=='pooled_group_adjusted' else scope
  ax.set_title(f'{letter}. {title} (N = {int(d.N.iloc[0])})',fontsize=8.4,fontweight='bold',loc='left',pad=7)
  for sp in ax.spines.values():sp.set_visible(False)
cax=fig.add_axes([.26,.060,.48,.014]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_label('Correlation r',fontsize=8,labelpad=1);cb.ax.tick_params(labelsize=7,length=2)
for ext in ['png','pdf','svg']:fig.savefig(F/('Figure4_six_panel_capture_learning.'+ext),dpi=300,facecolor='white')
plt.close(fig)
manifest={'status':'REQUESTED_CAPTURE_SUBGROUP_EXTENSION','panels':plot.groupby('panel')[['outcome','scope']].first().to_dict('index'),'capture_rows':84,'learning_rows':84,'total_cells':168,'Holm_family':'28 network pairs separately within each outcome/sample panel','new_capture_subgroup_cells':56,'existing_pooled_capture_and_learning_estimates_preserved':True,'model_refits':0,'corrected_cells':sig.to_dict('records'),'color_scale':[-vmax,vmax],'marker':'asterisk and black outline iff Holm28 p<.05','sources':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,C/'yeo7_network_summary_scores_n33.csv',C/'yeo7_all_28_summary_associations_holm28.csv',L/'network_associations_all84.csv']}}
(F/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
