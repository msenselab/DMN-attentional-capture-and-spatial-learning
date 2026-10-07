"""Pool-only Figure S1; subset frozen signed-count output, no recalculation."""
from pathlib import Path
import json,hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
F=Path(__file__).resolve().parent;R=F.parent;source=R/'learning_network_pooled_groups_v1/analysis/descriptive_signed_edge_counts_all168.csv'
d=pd.read_csv(source);d=d[d.scope=='pooled_group_adjusted'].copy();assert len(d)==56 and d.N.eq(33).all() and d.df.eq(30).all()
assert set(d.sign)=={'positive','negative'} and d.groupby('sign').selected_count.sum().to_dict()=={'negative':18,'positive':518}
assert d[d.sign=='positive'].possible_edges.sum()==79401
within=d.network_1==d.network_2;assert d[within&(d.sign=='positive')].selected_count.sum()==213 and d[within&(d.sign=='negative')].selected_count.sum()==5
d.to_csv(F/'FigureS1_pooled_signed_counts.csv',index=False)
NET=['Default','SomMot','Vis','Cont','SalVentAttn','DorsAttn','Limbic'];DISPLAY=['Default','SomMot','Visual','Control','VAN/SN','DAN','Limbic']
z=d.pivot(index='pair',columns='sign',values='selected_count');assert len(z)==28
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'svg.fonttype':'none'})
fig=plt.figure(figsize=(5.4,4.8));ax=fig.add_axes([.16,.22,.62,.69]);orange=plt.get_cmap('Oranges');purple=plt.get_cmap('Purples');pmax=int(z.positive.max());nmax=int(z.negative.max());assert pmax==171 and nmax==3
for i,a in enumerate(NET):
 for j in range(i,7):
  p,n=map(int,z.loc[a+'__'+NET[j],['positive','negative']]);ax.add_patch(Polygon([(j-.5,i-.5),(j+.5,i-.5),(j-.5,i+.5)],facecolor=orange(.1+.85*p/pmax),edgecolor='white',lw=.5));ax.add_patch(Polygon([(j+.5,i-.5),(j+.5,i+.5),(j-.5,i+.5)],facecolor=purple(.1+.85*n/nmax),edgecolor='white',lw=.5))
  ax.text(j-.17,i-.18,str(p),ha='center',va='center',fontsize=7.1,color='white' if p>pmax*.7 else 'black');ax.text(j+.17,i+.18,str(n),ha='center',va='center',fontsize=7.1,color='white' if n>nmax*.7 else 'black')
ax.set_xticks(range(7),DISPLAY,rotation=50,ha='right',fontsize=8.2);ax.set_yticks(range(7),DISPLAY,fontsize=8.2);ax.tick_params(length=0);ax.set_xlim(-.5,6.5);ax.set_ylim(6.5,-.5);ax.set_aspect('equal');ax.set_title('Pooled spatial learning (N = 33)',fontsize=10.5,fontweight='bold',loc='left',pad=8)
for sp in ax.spines.values():sp.set_visible(False)
for rect,cmap,maximum,label,ticks in [([.81,.3,.025,.50],orange,pmax,'Positive',[0,50,100,150,171]),([.92,.3,.025,.50],purple,nmax,'Negative',[0,1,2,3])]:
 ca=fig.add_axes(rect);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,maximum),cmap=cmap),cax=ca,ticks=ticks);cb.ax.tick_params(labelsize=6.4,length=2);cb.ax.set_title(label,fontsize=7,pad=5)
for ext in ['png','pdf','svg']:fig.savefig(F/('FigureS1_pooled_learning_edge_counts.'+ext),dpi=300,facecolor='white')
plt.close(fig)
(F/'manifest.json').write_text(json.dumps({'status':'PASS','scope':'pooled group-adjusted learning only','N':33,'df':30,'pair_categories':28,'signed_cells':56,'positive_total':518,'negative_total':18,'edge_universe':79401,'positive_within_between':[213,305],'negative_within_between':[5,13],'source':str(source.relative_to(R)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'new_statistical_analyses':0,'separate_sign_color_scales':[pmax,nmax]},indent=2));print('PASS: 56 signed cells; +518/-18; pooled-only; no refits')
