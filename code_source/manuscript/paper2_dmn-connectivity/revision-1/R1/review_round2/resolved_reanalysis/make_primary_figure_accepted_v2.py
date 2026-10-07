#!/usr/bin/env python3
"""Figure3 from actual verified new Default91 association vectors only."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=HERE/'inference/results_common399_n33_optimizer_resolved_v2');ap.add_argument('--input-manifest',type=Path,default=HERE/'common399_analysis_ready/analysis_ready_input_manifest.json');ap.add_argument('--acceptance',type=Path,default=HERE/'parent_dependent_bundle_acceptance_v2.json');ap.add_argument('--output',type=Path,default=HERE/'figures/current_verified_v2');args=ap.parse_args()
    RESULTS=args.results
    manifest=json.loads((RESULTS/'results_manifest.json').read_text());assert manifest['status']=='complete'
    accepted=json.loads(args.acceptance.read_text());assert accepted['status']=='PARENT_DEPENDENT_BUNDLE_ACCEPTED_FOR_CURRENT_REPORTING' and Path(accepted['results_directory']).resolve()==RESULTS.resolve()
    for path,expected in accepted['files'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    verified=json.loads((RESULTS/'independent_dependent_inference_verification.json').read_text());assert verified['status']=='PASS'
    cfg=json.loads(args.input_manifest.read_text());assert cfg['status']=='complete' and cfg['common_coverage_geometry']['effective_common_coverage_parcels']==399
    score_path=Path(cfg['participant_scores']['path']);score_path=score_path if score_path.is_absolute() else args.input_manifest.parent/score_path
    s=pd.read_csv(score_path);assert len(s)==33 and s.subject.is_unique
    p=cfg['participant_scores']['primary_column'];x=s[p]-s.groupby('group')[p].transform('mean')
    table=pd.read_csv(RESULTS/'participant_associations.csv')
    assert len(table[(table.scope=='pooled')&(table.outcome=='biased_capture')&(table.analysis.str.contains('group-centered'))])==1
    points=s[['subject','group',p,'biased_capture','unbiased_capture']].copy();points['connectivity_group_centered_fisher_z']=x
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.2),constrained_layout=True);palette={'SS':'#D98C5F','DS':'#5B4B8A'}
    for ax,key,label,panel in zip(axes,['biased_capture','unbiased_capture'],['Biased-dimension capture','Unbiased-dimension capture'],['A','B']):
        y=s[key]-s.groupby('group')[key].transform('mean');points[f'{key}_group_centered_ms']=y
        q=table[(table.scope=='pooled')&(table.outcome==key)&table.analysis.str.contains('group-centered')].iloc[0]
        r=float(stats.pearsonr(x,y).statistic);assert abs(r-q.r)<1e-12 and q.df==30
        for g in ['SS','DS']:
            ix=s.group==g;ax.scatter(x[ix],y[ix],s=45,color=palette[g],edgecolors='white',linewidths=.6,label=g)
        xx=np.linspace(float(x.min())*.98,float(x.max())*1.02,120);slope=float(x@y/(x@x));sse=float(np.sum((y-slope*x)**2));se=np.sqrt((sse/30)*xx**2/float(x@x));fit=slope*xx
        ax.plot(xx,fit,color='#333333',lw=1.3);ax.fill_between(xx,fit-stats.t.ppf(.975,30)*se,fit+stats.t.ppf(.975,30)*se,color='#cccccc',alpha=.45)
        ax.axhline(0,color='#dddddd',lw=.7,zorder=0);ax.axvline(0,color='#dddddd',lw=.7,zorder=0)
        ax.set_xlabel('Within-DMN connectivity\n(group-centered Fisher z)');ax.set_ylabel(label+'\n(group-centered, ms)')
        ptext='< .001' if q.p_raw<.001 else '= '+f'{q.p_raw:.3f}'.removeprefix('0')
        ax.text(.04,.96,f'r = {r:+.2f}, p {ptext}',transform=ax.transAxes,ha='left',va='top',fontsize=9)
        ax.text(-.12,1.04,panel,transform=ax.transAxes,fontweight='bold',fontsize=13)
        if panel=='A':ax.legend(frameon=False,loc='lower right',title='Group')
    out=args.output;out.mkdir(parents=True,exist_ok=False);points.to_csv(out/'Figure3_resolved_plot_data.csv',index=False)
    fig.savefig(out/'Figure3_resolved.png',dpi=300);fig.savefig(out/'Figure3_resolved.pdf');plt.close(fig)
    caption='Figure 3. Associations of the anatomically defined 91-parcel Yeo-7 Default network with capture by the biased (A) and unbiased (B) distractor dimension. Connectivity and RT contrasts were centered separately within SS and DS before pooling; each point is one participant (SS n=16, DS n=17). Lines show the pooled group-adjusted fits and shading their pointwise 95% confidence bands conditional on centering. Raw two-sided correlation p values are displayed; adjusted exploratory values and the direct dependent-correlation comparison are in the Supplement. Comparing these two panel p values alone does not test dimension specificity. SS, same-dimension spatial-bias group; DS, different-dimension spatial-bias group.'
    (out/'Figure3_caption.md').write_text(caption+'\n')
    report={'status':'PASS','parent_acceptance_sha256':hashlib.sha256(args.acceptance.read_bytes()).hexdigest(),'source_results_manifest_sha256':hashlib.sha256((RESULTS/'results_manifest.json').read_bytes()).hexdigest(),'source_scores_sha256':hashlib.sha256(score_path.read_bytes()).hexdigest(),'points':33,'panels':2,'df':30,'primary_physical_Default_parcels':91,'plot_data_sha256':hashlib.sha256((out/'Figure3_resolved_plot_data.csv').read_bytes()).hexdigest()}
    (out/'Figure3_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
