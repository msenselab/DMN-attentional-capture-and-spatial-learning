"""Presentation-only replay: preserve original capture/learning scopes."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
F=Path(__file__).resolve().parent;R=F.parent
C=R/'resolved_reanalysis/network_diagnostics/results_common399_n33'
L=R/'learning_network_pooled_groups_v1/analysis'
NET=['Default','SomMot','Vis','Cont','SalVentAttn','DorsAttn','Limbic']
DISPLAY=['Default','SomMot','Visual','Control','VAN/SN','DAN','Limbic']
cap=pd.read_csv(C/'yeo7_all_28_summary_associations_holm28.csv');learn=pd.read_csv(L/'network_associations_all84.csv')
capture_counts=pd.read_csv(C/'yeo7_pair_enrichment_hypergeom_holm28.csv');learning_counts=pd.read_csv(L/'descriptive_signed_edge_counts_all168.csv')
assert len(cap)==28 and len(learn)==84 and len(capture_counts)==28 and len(learning_counts)==168
assert (cap.p_Holm_28<.05).sum()==1 and not (learn.p_holm28<.05).any()
assert int(capture_counts.selected_positive.sum())==812 and int(capture_counts.selected_negative.sum())==70
expected={'pooled_group_adjusted':(518,18),'SS':(1447,24),'DS':(98,629)}
for scope,(p,n) in expected.items():
 z=learning_counts[learning_counts.scope==scope];assert z[z.sign=='positive'].selected_count.sum()==p and z[z.sign=='negative'].selected_count.sum()==n
# Preserve identical input connectivity and participant IDs across outcomes.
x=pd.read_csv(C/'yeo7_network_summary_scores_n33.csv').set_index('subject');y=pd.read_csv(L/'participant_network_summary_fisher_z.csv').set_index('subject')
assert x.index.tolist()==y.index.tolist() and (x.group==y.group).all()
assert np.allclose(x.drop(columns='group').to_numpy(),y.drop(columns='group').to_numpy(),atol=2e-14,rtol=0)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'svg.fonttype':'none'})
plot=[]
for label,scope in [('A','capture_pooled'),('B','pooled_group_adjusted'),('C','SS'),('D','DS')]:
 df=cap if scope=='capture_pooled' else learn[learn.scope==scope]
 for _,z in df.iterrows():
  pair=z.network_pair.replace('--','__') if scope=='capture_pooled' else z.pair
  plot.append({'panel':label,'outcome':'capture' if scope=='capture_pooled' else 'learning','scope':scope,'pair':pair,'N':int(z.N),'r':float(z.r_group_adjusted if scope=='capture_pooled' else z.r),'p_raw':float(z.p_raw if scope=='capture_pooled' else z.two_sided_raw_p),'p_holm28':float(z.p_Holm_28 if scope=='capture_pooled' else z.p_holm28)})
plotted=pd.DataFrame(plot);plotted.to_csv(F/'Figure4_plotted_statistics.csv',index=False)
vmax=float(np.ceil(plotted.r.abs().max()*20)/20)
cmap=plt.get_cmap('PuOr_r').copy();cmap.set_bad('white')
def decorate(ax,letter):
 ax.set_xticks(range(7),DISPLAY,rotation=50,ha='right',fontsize=8.2)
 ax.set_yticks(range(7),DISPLAY,fontsize=8.2)
 ax.tick_params(length=0);ax.set_xlim(-.5,6.5);ax.set_ylim(6.5,-.5);ax.set_aspect('equal')
 ax.text(-.31,1.07,letter,transform=ax.transAxes,fontsize=12,fontweight='bold',va='bottom')
 for s in ax.spines.values():s.set_visible(False)
def save(fig,name):
 for ext in ('png','pdf','svg'):fig.savefig(F/(name+'.'+ext),dpi=300,facecolor='white')
 plt.close(fig)
fig=plt.figure(figsize=(6.8,6.75))
# Fixed axes keep panels uniform; caption rather than internal titles explains outcome/scope.
axes=[fig.add_axes([.12,.605,.33,.333]),fig.add_axes([.59,.605,.33,.333]),fig.add_axes([.12,.165,.33,.333]),fig.add_axes([.59,.165,.33,.333])]
for ax,letter in zip(axes,'ABCD'):
 z=plotted[plotted.panel==letter].set_index('pair');arr=np.full((7,7),np.nan)
 for i,a in enumerate(NET):
  for j in range(i,7):
   b=NET[j];v=z.loc[a+'__'+b];arr[i,j]=v.r
   text=f'{v.r:.2f}'.replace('-0.','−.').replace('0.','.')
   ax.text(j,i,text,ha='center',va='center',fontsize=8.0,color='white' if abs(v.r)>vmax*.78 else 'black')
   if v.p_holm28<.05:ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='black',lw=1.3,zorder=5))
 im=ax.imshow(arr,cmap=cmap,vmin=-vmax,vmax=vmax,interpolation='none',zorder=0)
 ax.set_xticks(np.arange(-.5,7,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True);ax.grid(which='minor',color='white',lw=.5);ax.tick_params(which='minor',bottom=False,left=False)
 decorate(ax,letter)
cax=fig.add_axes([.23,.077,.56,.017]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_label('Correlation r',labelpad=1,fontsize=8.5);cb.ax.tick_params(labelsize=7.5,length=2)
save(fig,'Figure4_capture_learning_network_summaries')

def counts_fig(df,scopes,name):
 fig=plt.figure(figsize=(6.8,6.15) if len(scopes)==3 else (5.4,4.8))
 if len(scopes)==3:
  axes=[fig.add_axes([.115,.615,.31,.343]),fig.add_axes([.585,.615,.31,.343]),fig.add_axes([.35,.11,.31,.343])]
 else:axes=[fig.add_axes([.16,.22,.62,.69])]
 pmax=int(df[df.sign=='positive'].selected_count.max());nmax=max(1,int(df[df.sign=='negative'].selected_count.max()))
 orange=plt.get_cmap('Oranges');purple=plt.get_cmap('Purples')
 for ax,scope,letter in zip(axes,scopes,'ABC'):
  d=df[df.scope==scope].pivot(index='pair',columns='sign',values='selected_count')
  for i,a in enumerate(NET):
   for j in range(i,7):
    p,n=map(int,d.loc[a+'__'+NET[j],['positive','negative']])
    ax.add_patch(Polygon([(j-.5,i-.5),(j+.5,i-.5),(j-.5,i+.5)],facecolor=orange(.1+.85*p/pmax),edgecolor='white',lw=.5))
    ax.add_patch(Polygon([(j+.5,i-.5),(j+.5,i+.5),(j-.5,i+.5)],facecolor=purple(.1+.85*n/nmax),edgecolor='white',lw=.5))
    # Place labels in their own triangles, not over the separating diagonal.
    ax.text(j-.17,i-.18,str(p),ha='center',va='center',fontsize=6.7,color='white' if p>pmax*.7 else 'black')
    ax.text(j+.17,i+.18,str(n),ha='center',va='center',fontsize=6.7,color='white' if n>nmax*.7 else 'black')
  decorate(ax,letter)
  if len(scopes)==3 and letter in 'AB':
   ax.texts[-1].set_position((-.31,1.0))
  if len(scopes)==1:
   ax.texts[-1].set_visible(False)
   fig.text(.045,.915,letter,fontsize=12,fontweight='bold')
 if len(scopes)==3:
  c1=fig.add_axes([.85,.15,.017,.32]);c2=fig.add_axes([.935,.15,.017,.32])
 else:c1=fig.add_axes([.81,.3,.025,.50]);c2=fig.add_axes([.92,.3,.025,.50])
 for ca,cm,ma,lab in [(c1,orange,pmax,'Positive'),(c2,purple,nmax,'Negative')]:
  cb=fig.colorbar(ScalarMappable(norm=Normalize(0,ma),cmap=cm),cax=ca);cb.ax.tick_params(labelsize=6.4,length=2);cb.ax.set_title(lab,fontsize=7,pad=5)
 save(fig,name)
cc=[]
for _,z in capture_counts.iterrows():
 for sign,key in [('positive','selected_positive'),('negative','selected_negative')]:cc.append({'scope':'pooled_group_adjusted','pair':z.network_pair.replace('--','__'),'sign':sign,'selected_count':int(z[key]),'possible_edges':int(z.possible_edges)})
cc=pd.DataFrame(cc);cc.to_csv(F/'FigureS4_capture_signed_counts.csv',index=False)
counts_fig(cc,['pooled_group_adjusted'],'FigureS4_capture_descriptive_edge_counts')
counts_fig(learning_counts,['pooled_group_adjusted','SS','DS'],'FigureS1_learning_descriptive_edge_counts')
learning_counts.to_csv(F/'FigureS1_learning_signed_counts.csv',index=False)
manifest={'status':'PRESENTATION_ONLY_REPLAY','new_statistical_analyses':0,'new_models':0,'main_plot_cells':len(plotted),'main_panel_scopes':plotted.groupby('panel').scope.first().to_dict(),'capture_scope_preserved':'pooled only','learning_scopes_preserved':['pooled','SS','DS'],'vmax':vmax,'sign_palette':'positive orange; negative purple','holm28_mark':'black outline; no stars','capture_signed_counts':{'positive':812,'negative':70},'learning_counts':expected,'originals_preserved':True,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [C/'yeo7_all_28_summary_associations_holm28.csv',C/'yeo7_pair_enrichment_hypergeom_holm28.csv',L/'network_associations_all84.csv',L/'descriptive_signed_edge_counts_all168.csv']}}
(F/'figure_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
