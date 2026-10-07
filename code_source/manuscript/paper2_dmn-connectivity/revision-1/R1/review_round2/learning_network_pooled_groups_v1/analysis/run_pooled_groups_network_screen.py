#!/usr/bin/env python3
"""Frozen Figure4-style rare-minus-frequent learning network screen.

Creates all requested analytic tables and figures from frozen raw inputs.  This is
an exploratory association/descriptive screen; it is not CPM, enrichment, or an
edgewise-discovery analysis.
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])) + '/manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2')
OUT_DEFAULT = ROOT / "learning_network_pooled_groups_v1"
NETWORKS = ["Default", "SomMot", "Vis", "Cont", "SalVentAttn", "DorsAttn", "Limbic"]
DISPLAY = {"Default":"Default", "SomMot":"SomMot", "Vis":"Visual", "Cont":"Control", "SalVentAttn":"VAN/SN", "DorsAttn":"DAN", "Limbic":"Limbic"}
SCOPES = [("pooled_group_adjusted", None), ("SS", "SS"), ("DS", "DS")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p); ranked = p[order]
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    out = np.empty_like(p); out[order] = np.minimum(adjusted, 1.0)
    return out


def pearson(x, y, df):
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * math.sqrt(df / max(1e-300, 1-r*r))
    p = float(2 * stats.t.sf(abs(t), df))
    return r, t, p


def fisher_ci(r, n, se_denom):
    z = np.arctanh(np.clip(r, -0.999999999999, 0.999999999999))
    d = 1.959963984540054 / math.sqrt(se_denom)
    return float(np.tanh(z-d)), float(np.tanh(z+d))


def scope_vectors(values, score, groups, scope):
    if scope == "pooled_group_adjusted":
        vv = values.copy().astype(float); yy = score.copy().astype(float)
        for g in ("SS", "DS"):
            ix = np.flatnonzero(groups == g)
            vv[ix] -= vv[ix].mean(axis=0)
            yy[ix] -= yy[ix].mean()
        return vv, yy, len(yy)-3, len(yy)-4
    ix = np.flatnonzero(groups == scope)
    return values[ix].astype(float), score[ix].astype(float), len(ix)-2, len(ix)-3


def ensure(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def load_inputs():
    manifest_path = ROOT / "resolved_reanalysis/common399_analysis_ready/analysis_ready_input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    matrix = Path(manifest["primary_matrices"]["path"])
    subjects = Path(manifest["matrix_subjects"]["path"])
    parcels = Path(manifest["parcel_metadata"]["path"])
    prior_scores = ROOT / "spatial_learning_dmn_v1/participant_analysis/participant_learning_scores.csv"
    trials = ROOT.parents[1] / "attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv"
    paths = {"analysis_ready_manifest": manifest_path, "primary_matrices": matrix,
             "matrix_subjects": subjects, "parcel_metadata": parcels,
             "correct_rt_trials": trials, "prior_learning_scores": prior_scores,
             "prior_dmn_associations": ROOT / "spatial_learning_dmn_v1/participant_analysis/neural_learning_associations.csv"}
    for label, p in paths.items(): ensure(p.exists(), f"missing frozen input: {label}: {p}")
    return manifest, paths


def reconstruct_target(trial_path, matrix_subjects):
    tr = pd.read_csv(trial_path)
    required = {"correct", "RT_ms", "subject", "group", "Dimension", "category", "DisRegion", "run"}
    ensure(required <= set(tr.columns), f"trial schema missing: {sorted(required-set(tr.columns))}")
    # Frozen definition: correct, 200<=RT_ms<=3000; scanner trials only in this file; no order gate.
    eligible = tr[(tr.correct == 1) & tr.RT_ms.between(200, 3000, inclusive="both")].copy()
    biased_rule = ((eligible.group == "SS") & (eligible.Dimension == "Same")) | ((eligible.group == "DS") & (eligible.Dimension == "Different"))
    biased = eligible[biased_rule].copy()
    ensure(len(eligible) == 10995, f"eligible rows {len(eligible)} != frozen expected 10995")
    ensure(len(biased) == 6721, f"biased rows {len(biased)} != frozen expected 6721")
    ensure(set(biased.DisRegion.unique()) == {"Rare", "Frequent"}, "biased distractor-region labels are not exactly Rare/Frequent")
    rows=[]
    for (sub, grp), d in biased.groupby(["subject", "group"], sort=True):
        rare=d.loc[d.DisRegion == "Rare", "RT_ms"]; freq=d.loc[d.DisRegion == "Frequent", "RT_ms"]
        ensure(len(rare)>0 and len(freq)>0, f"missing category cell: {sub}")
        rows.append(dict(subject=sub, group=grp, rare_mean_ms=rare.mean(), frequent_mean_ms=freq.mean(),
                         n_rare=len(rare), n_frequent=len(freq),
                         learning_rare_minus_frequent_ms=rare.mean()-freq.mean(),
                         eligible_all_condition_rows=len(eligible), biased_eligible_rows=len(biased),
                         scoring_rule="correct==1; 200<=RT_ms<=3000; biased only; Rare-Frequent"))
    scores=pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    ensure(len(scores)==33 and (scores.group=="SS").sum()==16 and (scores.group=="DS").sum()==17,
           f"cohort mismatch: N={len(scores)}, SS={(scores.group=='SS').sum()}, DS={(scores.group=='DS').sum()}")
    # Matrix subject map is the required ordering contract.
    sm=pd.read_csv(matrix_subjects).sort_values("row_index").reset_index(drop=True)
    ensure(sm.row_index.tolist()==list(range(33)), "matrix subject map does not have row indices 0..32")
    joined=sm.merge(scores, on=["subject","group"], how="left", validate="one_to_one", indicator=True)
    ensure((joined._merge=="both").all(), "matrix subjects failed target join")
    return joined.drop(columns="_merge"), {"trial_rows_total":int(len(tr)), "eligible_all_condition_rows":int(len(eligible)),
        "biased_eligible_rows":int(len(biased)), "subjects":33, "SS":16, "DS":17,
        "rare_rows":int((biased.DisRegion=="Rare").sum()), "frequent_rows":int((biased.DisRegion=="Frequent").sum()),
        "run_counts":{str(k):int(v) for k,v in biased.run.value_counts().sort_index().items()}}


def make_pairs(net):
    pairs=[]
    for i, a in enumerate(NETWORKS):
        for j in range(i, len(NETWORKS)):
            b=NETWORKS[j]
            if i==j:
                ii=np.flatnonzero(net==a); rr,cc=np.triu_indices(len(ii),1); edge_i,edge_j=ii[rr],ii[cc]
            else:
                edge_i=np.repeat(np.flatnonzero(net==a), int((net==b).sum()))
                edge_j=np.tile(np.flatnonzero(net==b), int((net==a).sum()))
            pairs.append((a,b,f"{a}__{b}",edge_i,edge_j))
    ensure(len(pairs)==28, "not 28 network pairs")
    return pairs


def make_figures(assoc, counts, outfig):
    outfig.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":7, "axes.titlesize":8, "axes.labelsize":7})
    # Figure 6: only the three summary panels; splitting avoids six dense panels at 6.8in.
    fig, axes=plt.subplots(1,3,figsize=(6.8,3.15), constrained_layout=False)
    cmap=plt.get_cmap("PuOr_r").copy(); cmap.set_bad("white")  # positive orange, negative purple
    vmax=max(.35, float(np.ceil(np.max(np.abs(assoc.r))*20)/20))
    panel="ABC"
    for ax,(scope,_), letter in zip(axes, SCOPES, panel):
        d=assoc[assoc.scope==scope].set_index("pair")
        arr=np.full((7,7),np.nan); labels=np.empty((7,7),object)
        for i,a in enumerate(NETWORKS):
            for j,b in enumerate(NETWORKS):
                if j>=i:
                    key=f"{a}__{b}"; z=d.loc[key]
                    arr[i,j]=z.r; labels[i,j]=f"{z.r:+.2f}"
        im=ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="none")
        ax.set_xticks(np.arange(-.5,7,1), minor=True); ax.set_yticks(np.arange(-.5,7,1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.7); ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(7):
            for j in range(i,7): ax.text(j,i,labels[i,j],ha="center",va="center",fontsize=5.6,linespacing=.92)
        ax.set_xticks(range(7),[DISPLAY[x] for x in NETWORKS],rotation=42,ha="right",fontsize=6.3)
        ax.set_yticks(range(7),[DISPLAY[x] for x in NETWORKS],fontsize=6.3)
        n=int(d.N.iloc[0]); suffix="group-adjusted" if scope.startswith("pooled") else scope
        status="H<.05" if (d.p_holm28<.05).any() else "none H<.05"
        ax.set_title(f"{letter}. {suffix} (N={n})\n{status}",fontweight="bold",loc="left",pad=5)
        ax.tick_params(length=0); ax.set_xlim(-.5,6.5); ax.set_ylim(6.5,-.5)
    cax=fig.add_axes([.932,.215,.016,.60]); cb=fig.colorbar(im,cax=cax); cb.set_label("Correlation r",fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.text(.5,.035,"Upper triangle incl. diagonal: cell labels are r for Fisher-z connectivity vs. Rare−Frequent RT learning (ms). H: Holm across 28 summaries within scope.",ha="center",fontsize=6.15)
    fig.subplots_adjust(left=.08,right=.91,top=.84,bottom=.24,wspace=.39)
    for ext in ("png","pdf","svg"): fig.savefig(outfig/f"Figure6_rare_minus_frequent_network_summary.{ext}",dpi=600,bbox_inches=None)
    plt.close(fig)

    # Supplement: dual-colour triangular tiles report positive and negative raw-screen counts independently.
    fig, axes=plt.subplots(1,3,figsize=(6.8,3.15), constrained_layout=False)
    posmax=max(1,int(counts.loc[counts.sign=="positive", "selected_count"].max())); negmax=max(1,int(counts.loc[counts.sign=="negative", "selected_count"].max()))
    porange=plt.get_cmap("Oranges"); ppurple=plt.get_cmap("Purples")
    panel="DEF"
    for ax,(scope,_),letter in zip(axes,SCOPES,panel):
        d=counts[counts.scope==scope].pivot(index="pair",columns="sign",values="selected_count")
        ax.set_xlim(-.5,6.5); ax.set_ylim(6.5,-.5); ax.set_aspect("equal")
        for i,a in enumerate(NETWORKS):
            for j,b in enumerate(NETWORKS):
                if j>=i:
                    z=d.loc[f"{a}__{b}"]; p,n=int(z.positive),int(z.negative)
                    # Orange upper-left triangle is positive; purple lower-right triangle is negative.
                    ax.add_patch(Polygon([(j-.5,i-.5),(j+.5,i-.5),(j-.5,i+.5)],facecolor=porange(.12+.85*p/posmax),edgecolor="white",lw=.45))
                    ax.add_patch(Polygon([(j+.5,i-.5),(j+.5,i+.5),(j-.5,i+.5)],facecolor=ppurple(.12+.85*n/negmax),edgecolor="white",lw=.45))
                    ax.text(j,i,f"+{p}\n−{n}",ha="center",va="center",fontsize=5.8,linespacing=.9)
        ax.set_xticks(range(7),[DISPLAY[x] for x in NETWORKS],rotation=42,ha="right",fontsize=6.3)
        ax.set_yticks(range(7),[DISPLAY[x] for x in NETWORKS],fontsize=6.3)
        n=33 if scope.startswith("pooled") else (16 if scope=="SS" else 17); suffix="group-adjusted" if scope.startswith("pooled") else scope
        ax.set_title(f"{letter}. {suffix} (N={n})",fontweight="bold",loc="left",pad=5); ax.tick_params(length=0)
    c1=fig.add_axes([.900,.49,.014,.32]); c2=fig.add_axes([.940,.49,.014,.32])
    fig.colorbar(ScalarMappable(norm=Normalize(0,posmax),cmap=porange),cax=c1).ax.tick_params(labelsize=5.7)
    fig.colorbar(ScalarMappable(norm=Normalize(0,negmax),cmap=ppurple),cax=c2).ax.tick_params(labelsize=5.7)
    fig.text(.907,.835,"+",ha="center",fontsize=7,fontweight="bold",color="#c65f00"); fig.text(.947,.835,"−",ha="center",fontsize=7,fontweight="bold",color="#542788")
    fig.text(.5,.045,"Each tile: + positive / − negative individual edges with raw two-sided p<.01.\nOrange and purple intensity scale separately across all scopes; descriptive only (not edgewise-corrected discoveries).",ha="center",fontsize=6.15,linespacing=1.2)
    fig.subplots_adjust(left=.08,right=.875,top=.84,bottom=.24,wspace=.39)
    for ext in ("png","pdf","svg"): fig.savefig(outfig/f"FigureS1_rare_minus_frequent_descriptive_edge_counts.{ext}",dpi=600,bbox_inches=None)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=OUT_DEFAULT); args=ap.parse_args()
    out=args.output; ana=out/"analysis"; figs=out/"figures"; arrays=ana/"edge_arrays"
    ana.mkdir(parents=True,exist_ok=True); arrays.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    manifest, inputs=load_inputs()
    hashes={k:{"path":str(v),"sha256":sha256(v)} for k,v in inputs.items()}
    (ana/"source_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    target,qc=reconstruct_target(inputs["correct_rt_trials"],inputs["matrix_subjects"])
    prior=pd.read_csv(inputs["prior_learning_scores"])
    col="learning_rare_minus_frequent_ms"; ensure(col in prior, f"prior score column unavailable: {col}")
    prior_sub=prior[["subject",col]].rename(columns={col:"prior_learning"})
    target=target.merge(prior_sub,on="subject",how="left",validate="one_to_one")
    ensure(target.prior_learning.notna().all(), "prior target has missing subjects")
    delta=np.abs(target.learning_rare_minus_frequent_ms-target.prior_learning).max()
    ensure(delta < 1e-9, f"learning target mismatch vs frozen prior: {delta}")
    target.drop(columns="prior_learning").to_csv(ana/"participant_learning_scores_reconstructed.csv",index=False)
    target.drop(columns="prior_learning").to_csv(ana/"plotted_learning_scores.csv",index=False)

    mats=np.load(inputs["primary_matrices"])
    sm=pd.read_csv(inputs["matrix_subjects"]).sort_values("row_index")
    parcel=pd.read_csv(inputs["parcel_metadata"]).sort_values("matrix_index")
    ensure(mats.shape==(33,399,399),f"matrix shape {mats.shape}, expected (33,399,399)")
    ensure(parcel.shape[0]==399 and set(parcel.yeo7_network)==set(NETWORKS),"parcel networks do not meet physical Yeo7 contract")
    ensure(np.isfinite(mats).all() and np.allclose(mats,mats.transpose(0,2,1),atol=1e-12) and np.allclose(np.diagonal(mats,axis1=1,axis2=2),0,atol=1e-12),"matrix finite/symmetry/zero diagonal failure")
    net=parcel.yeo7_network.to_numpy(); pairs=make_pairs(net); groups=target.group.to_numpy(); score=target.learning_rare_minus_frequent_ms.to_numpy(float)
    tri_i,tri_j=np.triu_indices(399,1); ensure(len(tri_i)==79401,"unique edge count failure")
    pair_ids=np.empty(len(tri_i),dtype=object)
    pair_edge_index={}
    summaries=np.empty((33,28)); pair_rows=[]
    for k,(a,b,key,ii,jj) in enumerate(pairs):
        summaries[:,k]=mats[:,ii,jj].mean(axis=1)
        edge_set=set(zip(np.minimum(ii,jj).tolist(),np.maximum(ii,jj).tolist()))
        loc=np.array([z in edge_set for z in zip(tri_i.tolist(),tri_j.tolist())])
        pair_edge_index[key]=np.flatnonzero(loc); pair_ids[loc]=key
        pair_rows.append({"pair":key,"network_1":a,"network_2":b,"possible_edges":len(ii),"within_network":a==b})
    ensure(pd.isna(pair_ids).sum()==0 and sum(len(v) for v in pair_edge_index.values())==79401,"physical edge membership coverage failure")
    summary_wide=pd.DataFrame(summaries,columns=[x[2] for x in pairs]); summary_wide.insert(0,"group",groups); summary_wide.insert(0,"subject",target.subject)
    summary_wide.to_csv(ana/"participant_network_summary_fisher_z.csv",index=False)
    pd.DataFrame(pair_rows).to_csv(ana/"network_pair_geometry.csv",index=False)

    assoc=[]; countrows=[]; edge_all=mats[:,tri_i,tri_j]
    for scope,_ in SCOPES:
        vals,yy,df,se_denom=scope_vectors(summaries,score,groups,scope)
        for k,(a,b,key,ii,jj) in enumerate(pairs):
            r,t,p=pearson(vals[:,k],yy,df); lo,hi=fisher_ci(r,len(yy),se_denom)
            assoc.append({"scope":scope,"pair":key,"network_1":a,"network_2":b,"possible_edges":len(ii),"N":len(yy),"df":df,"r":r,"t":t,"two_sided_raw_p":p,"fisher95_ci_low":lo,"fisher95_ci_high":hi,"fisher_se":"1/sqrt(N-4)" if scope.startswith("pooled") else "1/sqrt(N-3)","correction_family":"Holm 28 within scope"})
        ev,ey,edf,_=scope_vectors(edge_all,score,groups,scope)
        # Vectorised Pearson correlations and t-tail p values, independently retaining all 79,401 edges.
        ex=ev-ev.mean(0); yc=ey-ey.mean(); rr=(ex*yc[:,None]).sum(0)/np.sqrt((ex*ex).sum(0)*(yc*yc).sum())
        rr=np.clip(rr,-1,1); tt=rr*np.sqrt(edf/np.maximum(1e-300,1-rr*rr)); pp=2*stats.t.sf(np.abs(tt),edf); ph=holm(pp)
        np.savez_compressed(arrays/f"{scope}_79401_edges.npz",edge_i=tri_i,edge_j=tri_j,pair=pair_ids,r=rr,t=tt,two_sided_raw_p=pp,holm79401_p=ph,N=np.array([len(ey)]),df=np.array([edf]))
        edge_df=pd.DataFrame({"scope":scope,"edge_i":tri_i,"edge_j":tri_j,"pair":pair_ids,"r":rr,"t":tt,"df":edf,"N":len(ey),"two_sided_raw_p":pp,"holm79401_p":ph})
        edge_df.to_csv(ana/f"individual_edges_{scope}_79401.csv.gz",index=False,compression="gzip")
        for a,b,key,ii,jj in pairs:
            x=np.flatnonzero(pair_ids==key); pos=(rr[x]>0)&(pp[x]<.01); neg=(rr[x]<0)&(pp[x]<.01)
            for sign, mask in (("positive",pos),("negative",neg)):
                countrows.append({"scope":scope,"pair":key,"network_1":a,"network_2":b,"sign":sign,"possible_edges":len(x),"N":len(ey),"df":edf,"screen_rule":"raw two-sided p<0.01","selected_count":int(mask.sum()),"selected_proportion":float(mask.mean()),"edgewise_correction_note":"full-universe Holm79401 saved per edge; panel counts use raw screen only"})
    assoc=pd.DataFrame(assoc); assoc["p_holm28"]=assoc.groupby("scope")["two_sided_raw_p"].transform(lambda x:holm(x.to_numpy()))
    assoc["p_holm84_global_sensitivity"]=holm(assoc.two_sided_raw_p.to_numpy())
    assoc["holm28_pass_p_lt_05"]=assoc.p_holm28<.05; assoc["global84_sensitivity_pass_p_lt_05"]=assoc.p_holm84_global_sensitivity<.05
    assoc.to_csv(ana/"network_associations_all84.csv",index=False)
    counts=pd.DataFrame(countrows); counts.to_csv(ana/"descriptive_signed_edge_counts_all168.csv",index=False)
    totals=(counts.groupby(["scope","sign"],as_index=False).agg(selected_count=("selected_count","sum")).pivot(index="scope",columns="sign",values="selected_count").reset_index().rename_axis(None,axis=1).rename(columns={"positive":"positive_count","negative":"negative_count"}))
    totals=totals.merge(counts.groupby("scope",as_index=False).agg(N=("N","first"),df=("df","first"),possible_edges=("possible_edges","sum")),on="scope",validate="one_to_one")
    totals["possible_edges"]//=2
    totals["selected_total"]=totals.positive_count+totals.negative_count
    totals["selected_proportion"]=totals.selected_total/totals.possible_edges; totals.to_csv(ana/"descriptive_edge_screen_scope_totals.csv",index=False)
    make_figures(assoc,counts,figs)
    passing=assoc[assoc.holm28_pass_p_lt_05]
    dmn=assoc[assoc.pair=="Default__Default"].copy()
    prior_r=float(pd.read_csv(inputs["prior_dmn_associations"]).query("scope == 'pooled_group_adjusted'").r.iloc[0])
    ensure(abs(float(dmn.query("scope == 'pooled_group_adjusted'").r.iloc[0])-prior_r)<1e-12,"pooled DMN correlation mismatch with prior corrected learning result")
    report=["# Frozen pooled/SS/DS Rare-minus-Frequent learning network screen", "", "## Analysis contract", "Exploratory association/descriptive screen only. Connectivity is the frozen common-399 Fisher-z mean across two task runs. The behavioral target is participant mean correct Rare RT minus correct Frequent RT (ms), restricted to 200–3000 ms and the group-specific biased dimension (SS Same/orientation; DS Different/color). No trial-order exclusion, CPM, enrichment, edgewise discovery claim, interaction, or group comparison was run.", "", "## Input/QC", f"- Reconstructed target: N=33 (SS=16, DS=17); eligible all-condition rows={qc['eligible_all_condition_rows']}; biased rows={qc['biased_eligible_rows']} (Rare={qc['rare_rows']}, Frequent={qc['frequent_rows']}); runs={qc['run_counts']}.", "- Target exactly agrees (maximum absolute difference <1e-9 ms) with frozen `spatial_learning_dmn_v1/participant_analysis/participant_learning_scores.csv`.", "- Physical Yeo-7 geometry: 399 parcels (native label 323 excluded), 79,401 unique undirected non-self edges, Default has all physical parcels including Default91.", "", "## Network associations (all 84)", "Pooled associations are group-demeaned for both summary and learning (N=33, df=30); SS/DS are ordinary within-group correlations (N=16/17, df=14/15). CIs use the declared Fisher SE. Primary `p_holm28` is separately Holm-adjusted across 28 pairs per scope; `p_holm84_global_sensitivity` is the prespecified global sensitivity family.", "", assoc.to_csv(index=False), "", "## Passing primary associations", passing.to_csv(index=False) if len(passing) else "None at Holm-adjusted p<.05 within a 28-pair scope.", "", "## Descriptive raw edge-screen totals", totals.to_csv(index=False), "", "The 168 pair×sign×scope rows are in `descriptive_signed_edge_counts_all168.csv`. Individual edge r, t, raw p, and full-universe Holm79401 p are in one gzipped CSV and one NPZ array per scope. Counts use only raw two-sided p<.01 and are descriptive; they are not corrected discoveries, enrichment, network importance, prediction, causal evidence, or a basis for group-difference claims.", "", "## Figure files", "`figures/Figure6_rare_minus_frequent_network_summary.{png,pdf,svg}` contains three association heatmaps. `figures/FigureS1_rare_minus_frequent_descriptive_edge_counts.{png,pdf,svg}` separately contains descriptive raw count matrices because a six-panel composite would not retain publication-readable numeric labels at 6.8 in width. PNGs are 600 dpi and all figures are exactly 6.8 in wide.", "", "## Reproducibility", "Run `python analysis/run_pooled_groups_network_screen.py` from any directory. Exact frozen paths/hashes are in `analysis/source_hashes.json`; the independent checker is `analysis/verify_pooled_groups_network_screen.py` and writes its own `analysis/independent_verification/validation.json`."]
    (ana/"REPORT.md").write_text("\n".join(report))
    schema = """# Output files and column schemas

## Tabular outputs
- `participant_learning_scores_reconstructed.csv` and `plotted_learning_scores.csv`: `row_index, subject, group, rare_mean_ms, frequent_mean_ms, n_rare, n_frequent, learning_rare_minus_frequent_ms, eligible_all_condition_rows, biased_eligible_rows, scoring_rule`.
- `participant_network_summary_fisher_z.csv`: `subject, group`, followed by 28 columns named `<network_1>__<network_2>`; each is that participant's mean Fisher-z connectivity across the corresponding physical network-pair edges.
- `network_pair_geometry.csv`: `pair, network_1, network_2, possible_edges, within_network`.
- `network_associations_all84.csv`: `scope, pair, network_1, network_2, possible_edges, N, df, r, t, two_sided_raw_p, fisher95_ci_low, fisher95_ci_high, fisher_se, correction_family, p_holm28, p_holm84_global_sensitivity, holm28_pass_p_lt_05, global84_sensitivity_pass_p_lt_05`.
- `descriptive_signed_edge_counts_all168.csv`: `scope, pair, network_1, network_2, sign, possible_edges, N, df, screen_rule, selected_count, selected_proportion, edgewise_correction_note`; exactly 3 scopes × 28 physical pairs × positive/negative sign.
- `descriptive_edge_screen_scope_totals.csv`: `scope, negative_count, positive_count, N, df, possible_edges, selected_total, selected_proportion`.
- `individual_edges_<scope>_79401.csv.gz`: `scope, edge_i, edge_j, pair, r, t, df, N, two_sided_raw_p, holm79401_p`; there is one 79,401-edge table for each scope.

## Full arrays and provenance
- `edge_arrays/<scope>_79401_edges.npz`: named arrays `edge_i, edge_j, pair, r, t, two_sided_raw_p, holm79401_p, N, df`; all 79,401 unique undirected non-self edges are ordered by `numpy.triu_indices(399, 1)`.
- `source_hashes.json`: exact absolute frozen input paths and SHA-256 bindings.
- `output_manifest.json`: run counts and generated output hashes.
- `independent_verification/validation.json`: independent raw-input replay status; it does not import the generator.

## Figures
- `figures/Figure6_rare_minus_frequent_network_summary.{png,pdf,svg}`: three 28-pair association heatmaps.
- `figures/FigureS1_rare_minus_frequent_descriptive_edge_counts.{png,pdf,svg}`: three raw p<.01 signed-count matrices. Both PNGs are 6.8 in wide at 600 dpi.
"""
    (ana/"OUTPUT_SCHEMA.md").write_text(schema)
    output_manifest={"status":"PASS","analysis":"frozen pooled/SS/DS Figure4-style Rare-minus-Frequent RT screen","counts":qc,"association_rows":int(len(assoc)),"signed_count_rows":int(len(counts)),"edge_arrays":{s:str(arrays/f"{s}_79401_edges.npz") for s,_ in SCOPES},"figures":{p.name:sha256(p) for p in figs.glob("*")},"files":{p.name:sha256(p) for p in ana.glob("*") if p.is_file()}}
    (ana/"output_manifest.json").write_text(json.dumps(output_manifest,indent=2)+"\n")
    print(json.dumps({"status":"PASS","output":str(out),"association_rows":len(assoc),"count_rows":len(counts),"primary_passes":int(len(passing)),"DMN":dmn[["scope","r","two_sided_raw_p","p_holm28","p_holm84_global_sensitivity"]].to_dict("records"),"edge_totals":totals.to_dict("records")},indent=2))

if __name__ == "__main__": main()
