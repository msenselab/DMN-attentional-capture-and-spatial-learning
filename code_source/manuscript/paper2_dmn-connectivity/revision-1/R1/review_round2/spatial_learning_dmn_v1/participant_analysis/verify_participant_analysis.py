#!/usr/bin/env python3
"""Independent verifier for run_participant_analysis.py outputs.

This file deliberately does not import the generator. It reconstructs the
specified participant scores, resampling streams, tests, Holm adjustments, and
every saved bootstrap draw from the hash-bound source files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
HERE = Path(__file__).resolve().parent
SCORES = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv"
TRIALS = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv"
SEED, R = 20261004, 10_000


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1 << 20),b""): h.update(block)
    return h.hexdigest()


def corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a-a.mean(); b=b-b.mean()
    return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))


def center(v, groups):
    a=np.array(v,float,copy=True)
    for group in ("SS","DS"):
        ix=np.flatnonzero(np.asarray(groups)==group)
        if a.ndim==1: a[ix]-=a[ix].mean()
        else: a[:,ix]-=a[:,ix].mean(axis=1,keepdims=True)
    return a


def cstats(x,y,adjusted):
    n=len(x); r=corr(x,y); df=n-3 if adjusted else n-2
    t=r*np.sqrt(df/(1-r*r)); p=float(2*stats.t.sf(abs(t),df))
    se=1/np.sqrt(n-4 if adjusted else n-3)
    ci=np.tanh(np.arctanh(r)+np.array([-1,1])*stats.norm.ppf(.975)*se)
    return r,float(t),df,p,float(ci[0]),float(ci[1])


def williams(a,b,c,n_eff=32):
    determinant=1-a*a-b*b-c*c+2*a*b*c
    den_sq=2*determinant*(n_eff-1)/(n_eff-3)+((a+b)**2/4)*(1-c)**3
    t=(a-b)*np.sqrt((n_eff-1)*(1+c)/den_sq)
    return a-b,float(t),n_eff-3,float(2*stats.t.sf(abs(t),n_eff-3)),determinant,den_sq


def indices(groups,rng):
    return np.concatenate([rng.choice(np.flatnonzero(groups==g),size=(R,sum(groups==g)),replace=True) for g in ("SS","DS")],axis=1)


def draws(a,b,ss_n):
    def c2(x,y):
        x=x-x.mean(axis=1,keepdims=True); y=y-y.mean(axis=1,keepdims=True)
        return np.sum(x*y,axis=1)/np.sqrt(np.sum(x*x,axis=1)*np.sum(y*y,axis=1))
    return c2(a,b),c2(a[:,:ss_n],b[:,:ss_n]),c2(a[:,ss_n:],b[:,ss_n:])


def check_close(name, expected, observed, tol=1e-12):
    e=np.asarray(expected,float); o=np.asarray(observed,float)
    if e.shape != o.shape or not np.allclose(e,o,atol=tol,rtol=0,equal_nan=True):
        diff=np.nanmax(np.abs(e-o)) if e.shape==o.shape else float("inf")
        raise AssertionError(f"{name} mismatch; shape={e.shape}/{o.shape}, max_abs_diff={diff}")


def main():
    checks=[]
    source_meta=json.loads((HERE/"source_hashes.json").read_text())
    assert source_meta["seed"]==SEED and source_meta["bootstrap_draws"]==R
    for raw, expected in source_meta["input_sha256"].items():
        assert sha256(Path(raw))==expected, f"source hash mismatch: {raw}"
    checks.append("All hash-bound inputs and generator source match source_hashes.json")
    manifest=json.loads((HERE/"output_manifest.json").read_text())
    assert manifest["status"]=="complete" and manifest["expected"]==dict(N=33,SS=16,DS=17,trial_rows=10995,association_rows=3,reliability_rows=6,draw_rows=10000)
    for name, expected in manifest["output_sha256"].items():
        assert sha256(HERE/name)==expected, f"output hash mismatch: {name}"
    checks.append("All manifest-listed output hashes match")

    s=pd.read_csv(SCORES).sort_values("subject").reset_index(drop=True)
    d=pd.read_csv(TRIALS)
    assert len(s)==33 and s.groupby("group").size().to_dict()=={"DS":17,"SS":16} and s.subject.is_unique
    assert len(d)==10995 and d.subject.nunique()==33 and set(d.subject)==set(s.subject)
    assert d.correct.eq(1).all() and np.isfinite(d.RT_ms).all() and (d.RT_ms>200).all() and (d.RT_ms<=3000).all()
    assert d.global_order.eq((d.run-1)*184+d.trial).all()
    assert d.loc[d.group.eq("SS") & d.category.eq("biased"),"Dimension"].eq("Same").all()
    assert d.loc[d.group.eq("DS") & d.category.eq("biased"),"Dimension"].eq("Different").all()
    checks.append("Authorized N33 scanner trial gate, original global-order parity, and recorded SS/DS biased-dimension mapping verified")

    b=d.loc[d.category.eq("biased") & d.DisRegion.isin(["Rare","Frequent"])].copy()
    b["parity"]=np.where(b.global_order.mod(2).eq(1),"odd","even")
    records=[]
    for scheme, cols in (("overall",[]),("run",["run"]),("global_order_parity",["parity"])):
        z=b.groupby(["subject","group",*cols,"DisRegion"]).RT_ms.agg(["mean","size"]).reset_index()
        m=z.pivot(index=["subject","group",*cols],columns="DisRegion",values="mean")
        n=z.pivot(index=["subject","group",*cols],columns="DisRegion",values="size")
        q=m.join(n,lsuffix="_mean",rsuffix="_n").reset_index()
        q["scheme"]=scheme; q["scope_level"]="all" if not cols else q[cols[0]].astype(str)
        q["learning_rare_minus_frequent_ms"]=q["Rare_mean"]-q["Frequent_mean"]
        records.append(q[["subject","group","scheme","scope_level","learning_rare_minus_frequent_ms"]])
    all_scores=pd.concat(records,ignore_index=True)
    assert not all_scores.learning_rare_minus_frequent_ms.isna().any()
    overall=all_scores.loc[all_scores.scheme.eq("overall")].sort_values("subject")
    rawcap=d.groupby(["subject","category"]).RT_ms.mean().unstack()
    z=s[["subject","group","dmn_conn_primary","biased_capture"]].merge(overall[["subject","learning_rare_minus_frequent_ms"]],on="subject",validate="one_to_one")
    assert np.allclose(z.biased_capture.to_numpy(),(rawcap.loc[z.subject,"biased"]-rawcap.loc[z.subject,"absent"]).to_numpy(),atol=1e-9,rtol=0)
    groups=z.group.to_numpy(); ss=np.flatnonzero(groups=="SS"); ds=np.flatnonzero(groups=="DS")
    x=center(z.dmn_conn_primary.to_numpy(),groups); learn=center(z.learning_rare_minus_frequent_ms.to_numpy(),groups); cap=center(z.biased_capture.to_numpy(),groups)
    checks.append("Rare-minus-frequent scores and resolved biased-capture source match raw scanner recomputation without label recoding")

    # Associations and Holm-3.
    expected_assoc=np.array([cstats(x,learn,True),cstats(x[ss],learn[ss],False),cstats(x[ds],learn[ds],False)])
    got_assoc=pd.read_csv(HERE/"neural_learning_associations.csv")
    check_close("association r/t/df/p/Fisher",expected_assoc,got_assoc[["r","t","df","p_raw","fisher_ci_low","fisher_ci_high"]].to_numpy())
    check_close("association Holm",multipletests(expected_assoc[:,3],method="holm")[1],got_assoc.p_Holm)

    # Descriptive one-sample tests and Holm-2.
    expected_desc=[]
    for ix in (ss,ds):
        y=z.learning_rare_minus_frequent_ms.to_numpy()[ix]; tt=stats.ttest_1samp(y,0); n=len(y); sd=np.std(y,ddof=1); mean=np.mean(y); ci=stats.t.interval(.95,n-1,loc=mean,scale=sd/np.sqrt(n))
        expected_desc.append([mean,sd,tt.statistic,n-1,tt.pvalue,ci[0],ci[1],tt.statistic/np.sqrt(n)])
    expected_desc=np.asarray(expected_desc,float); got_desc=pd.read_csv(HERE/"learning_group_descriptive_tests.csv")
    check_close("descriptive tests",expected_desc,got_desc[["mean_ms","SD_ms","t","df","p_raw","mean_t_ci_low","mean_t_ci_high","dz"]].to_numpy())
    check_close("descriptive Holm",multipletests(expected_desc[:,4],method="holm")[1],got_desc.p_Holm)

    # Reconstruct every saved association/comparison and dz bootstrap draw in specified RNG call order.
    rng=np.random.default_rng(SEED); ix=indices(groups,rng); ssn=len(ss); boot_groups=np.array(["SS"]*ssn+["DS"]*len(ds))
    xb=center(z.dmn_conn_primary.to_numpy()[ix],boot_groups); lb=center(z.learning_rare_minus_frequent_ms.to_numpy()[ix],boot_groups); cb=center(z.biased_capture.to_numpy()[ix],boot_groups)
    rp,rs,rd=draws(xb,lb,ssn); rcap,_,_=draws(xb,cb,ssn); rcl,_,_=draws(cb,lb,ssn)
    expected_draws=np.column_stack([np.arange(1,R+1),rp,rs,rd,rcap,rcl,rcap-rp])
    got_draws=pd.read_csv(HERE/"association_and_comparison_bootstrap_draws.csv")
    check_close("all association/comparison bootstrap draws",expected_draws,got_draws.to_numpy())
    dzix=indices(groups,rng)
    dzss=z.learning_rare_minus_frequent_ms.to_numpy()[dzix[:,:ssn]]; dzds=z.learning_rare_minus_frequent_ms.to_numpy()[dzix[:,ssn:]]
    expected_dz=np.column_stack([np.arange(1,R+1),dzss.mean(axis=1)/dzss.std(axis=1,ddof=1),dzds.mean(axis=1)/dzds.std(axis=1,ddof=1)])
    got_dz=pd.read_csv(HERE/"learning_descriptive_dz_bootstrap_draws.csv")
    check_close("all dz bootstrap draws",expected_dz,got_dz.to_numpy())
    check_close("dz percentile CIs",np.quantile(expected_dz[:,1:],[.025,.975],axis=0).T,got_desc[["dz_bootstrap_ci_low","dz_bootstrap_ci_high"]].to_numpy())
    checks.append("All 10,000 association/comparison and d_z bootstrap draws exactly regenerated")

    # Williams and its bootstrap interval.
    w=williams(corr(x,cap),corr(x,learn),corr(cap,learn))
    got_w=pd.read_csv(HERE/"capture_vs_learning_williams.csv").iloc[0]
    check_close("Williams point values",[w[0],w[1],w[2],w[3],32,w[4],w[5]],[got_w.delta_r,got_w.williams_t,got_w.williams_df,got_w.p_raw,got_w.n_eff,got_w.determinant_R,got_w.denominator_squared])
    check_close("Williams component correlations",[corr(x,cap),corr(x,learn),corr(cap,learn)],[got_w.r_dmn_capture,got_w.r_dmn_learning,got_w.r_capture_learning])
    check_close("Williams bootstrap interval",np.quantile(expected_draws[:,-1],[.025,.975]),[got_w.bootstrap_ci_low,got_w.bootstrap_ci_high])

    # Reliability: recreate split/run scores, all point tests, all paired bootstrap draws, Holm-6.
    wide=all_scores.pivot(index=["subject","group"],columns=["scheme","scope_level"],values="learning_rare_minus_frequent_ms").reindex(pd.MultiIndex.from_frame(z[["subject","group"]]))
    inputs={"odd_even":(wide[("global_order_parity","odd")].to_numpy(),wide[("global_order_parity","even")].to_numpy()),"run1_run2":(wide[("run","1")].to_numpy(),wide[("run","2")].to_numpy())}
    expected_rel=[]; out={"iteration":np.arange(1,R+1)}
    for scheme,(a,bv) in inputs.items():
        for scope,ii,adj in (("pooled_group_adjusted",np.arange(33),True),("SS",ss,False),("DS",ds,False)):
            aa=center(a,groups) if adj else a[ii]; bb=center(bv,groups) if adj else bv[ii]
            r,t,df,p,lo,hi=cstats(aa,bb,adj); expected_rel.append([r,t,df,p,lo,hi])
        ab=center(a[ix],boot_groups); bb=center(bv[ix],boot_groups); p,s1,d1=draws(ab,bb,ssn)
        out[f"{scheme}_pooled_group_adjusted_r"]=p;out[f"{scheme}_SS_r"]=s1;out[f"{scheme}_DS_r"]=d1
    expected_rel=np.asarray(expected_rel,float); got_rel=pd.read_csv(HERE/"learning_reliability.csv")
    check_close("reliability point tests",expected_rel,got_rel[["r","t","df","p_raw","fisher_ci_low","fisher_ci_high"]].to_numpy())
    check_close("reliability Holm",multipletests(expected_rel[:,3],method="holm")[1],got_rel.p_Holm)
    got_rel_draws=pd.read_csv(HERE/"learning_reliability_bootstrap_draws.csv")
    check_close("all reliability bootstrap draws",np.column_stack(list(out.values())),got_rel_draws.to_numpy())
    ci=[]
    for scheme in ("odd_even","run1_run2"):
        for scope in ("pooled_group_adjusted","SS","DS"):
            ci.append(np.quantile(out[f"{scheme}_{scope}_r"],[.025,.975]))
    check_close("reliability percentile CIs",np.asarray(ci),got_rel[["bootstrap_ci_low","bootstrap_ci_high"]].to_numpy())
    checks.append("All six reliability point tests, Holm-6 adjustments, and 10,000 paired bootstrap draws exactly regenerated")

    report={"status":"PASS","seed":SEED,"R":R,"checks":checks,"verified_files":["neural_learning_associations.csv","learning_group_descriptive_tests.csv","association_and_comparison_bootstrap_draws.csv","learning_descriptive_dz_bootstrap_draws.csv","capture_vs_learning_williams.csv","learning_reliability.csv","learning_reliability_bootstrap_draws.csv","source_hashes.json","output_manifest.json"],"limitations":["Recorded DisRegion labels were not recoded. Task-code definitions and event position partitions were checked, but original per-participant sequence .mat files were not available for byte-level reconstruction of every event label.","All results are exploratory, scanner-only associations; reliability uncertainty limits null and trait interpretation."]}
    (HERE/"independent_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
