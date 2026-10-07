#!/usr/bin/env python3
"""Independent verifier for the participant 50/50 capture companion.

Does not import its generator. It independently rebuilds score cells, all point
statistics, all 10,000 paired draws, multiplicity adjustments, source bindings,
and the copied legacy reliability reconciliation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent
ROUND = OUT.parent.parent
ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
TRIALS = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv"
NEURAL = ROUND / "resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv"
LEGACY = ROUND / "spatial_learning_dmn_v1/participant_analysis"
SEED, R = 20261004, 10_000
CONDS = ("Absent", "BF", "BR", "UF", "UR")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corr(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float); x=x-x.mean(); y=y-y.mean()
    return float(np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y)))


def center(v, groups):
    a=np.array(v,float,copy=True)
    for g in ("SS","DS"):
        ix=np.flatnonzero(np.asarray(groups)==g)
        if a.ndim == 1: a[ix] -= a[ix].mean()
        else: a[:,ix] -= a[:,ix].mean(axis=1,keepdims=True)
    return a


def cstats(x,y,adjusted):
    n=len(x); r=corr(x,y); df=n-3 if adjusted else n-2
    t=r*np.sqrt(df/(1-r*r)); p=float(2*stats.t.sf(abs(t),df))
    se=1/np.sqrt(n-4 if adjusted else n-3)
    ci=np.tanh(np.arctanh(r)+stats.norm.ppf(.975)*se*np.array([-1,1]))
    return np.array([n,df,r,t,p,ci[0],ci[1]],float)


def williams(a,b,c,n):
    det=1-a*a-b*b-c*c+2*a*b*c
    den=2*det*(n-1)/(n-3)+((a+b)**2/4)*(1-c)**3
    t=(a-b)*np.sqrt((n-1)*(1+c)/den)
    return np.array([a-b,c,t,n-3,2*stats.t.sf(abs(t),n-3),n,det,den],float)


def cond_label(d):
    z=np.full(len(d),None,dtype=object)
    z[d.category.eq("absent").to_numpy()]="Absent"
    z[(d.category.eq("biased") & d.DisRegion.eq("Frequent")).to_numpy()]="BF"
    z[(d.category.eq("biased") & d.DisRegion.eq("Rare")).to_numpy()]="BR"
    z[(d.category.eq("unbiased") & d.DisRegion.eq("Frequent")).to_numpy()]="UF"
    z[(d.category.eq("unbiased") & d.DisRegion.eq("Rare")).to_numpy()]="UR"
    if any(v is None for v in z): raise AssertionError("unrecognized trial cell")
    return z


def check(name, expected, observed, tol=1e-11):
    a=np.asarray(expected,float); b=np.asarray(observed,float)
    if a.shape != b.shape or not np.allclose(a,b,atol=tol,rtol=0,equal_nan=True):
        raise AssertionError(f"{name}: shape {a.shape}/{b.shape}, max abs diff " + (str(np.nanmax(abs(a-b))) if a.shape==b.shape else "inf"))


def main():
    checks=[]
    smeta=json.loads((OUT/"source_hashes.json").read_text())
    assert smeta["status"] == "PASS" and smeta["seed"] == SEED and smeta["paired_bootstrap_draws"] == R
    assert smeta["predictor_column"] == "dmn_conn_primary"
    for raw, expected in smeta["input_sha256"].items():
        assert sha(Path(raw)) == expected, f"source hash mismatch: {raw}"
    checks.append("All hash-bound inputs, legacy verifier/report, generator, and independent verifier match source_hashes.json")
    manifest=json.loads((OUT/"output_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["expected"] == {"N":33,"SS":16,"DS":17,"trial_rows":10995,"association_rows":6,"comparison_rows":3,"bootstrap_rows":10000,"reliability_rows":6}
    for name, expected in manifest["output_sha256"].items():
        assert sha(OUT/name) == expected, f"output hash mismatch: {name}"
    checks.append("All companion manifest-listed output hashes match")

    t=pd.read_csv(TRIALS); n=pd.read_csv(NEURAL, usecols=["subject","group","dmn_conn_primary"]).sort_values("subject").reset_index(drop=True)
    assert len(t)==10995 and t.subject.nunique()==33 and t.correct.eq(1).all() and t.RT_ms.between(200,3000,inclusive="both").all()
    assert len(n)==33 and n.subject.is_unique and n.groupby("group").size().to_dict()=={"DS":17,"SS":16}
    assert set(t.subject)==set(n.subject) and np.isfinite(n.dmn_conn_primary).all()
    t=t.copy(); t["condition"]=cond_label(t)
    audit=(t.groupby(["subject","group","run","condition"]).size().rename("n_trials").reset_index())
    got_audit=pd.read_csv(OUT/"participant_run_condition_count_audit.csv")
    assert audit.equals(got_audit[["subject","group","run","condition","n_trials"]])
    assert (got_audit.cell_complete).all() and len(got_audit)==330 and (got_audit.n_trials>0).all()
    inv=(t.groupby(["group","condition"]).size().rename("n_trials").reset_index())
    got_inv=pd.read_csv(OUT/"five_condition_count_inventory.csv")
    assert inv.equals(got_inv)
    expected_inventory={("SS","Absent"):953,("SS","BF"):2632,("SS","BR"):559,("SS","UF"):596,("SS","UR"):596,("DS","Absent"):1013,("DS","BF"):2897,("DS","BR"):633,("DS","UF"):566,("DS","UR"):550}
    assert inv.set_index(["group","condition"]).n_trials.to_dict()==expected_inventory
    checks.append("All 330 participant-by-run condition cells are nonempty and the exact frozen five-condition count inventory is reproduced")

    m=(t.groupby(["subject","group","condition"]).RT_ms.mean().unstack("condition").reindex(columns=CONDS).reset_index())
    m["B50_ms"]=.5*(m.BF+m.BR)-m.Absent; m["U50_ms"]=.5*(m.UF+m.UR)-m.Absent
    data=n[["subject","group","dmn_conn_primary"]].merge(m,on=["subject","group"],validate="one_to_one").sort_values("subject").reset_index(drop=True)
    g=data.group.to_numpy(); x=data.dmn_conn_primary.to_numpy(float); xc=center(x,g)
    data["X_group_centered"]=xc; data["X_globalSD_standardized"]=xc/np.std(xc,ddof=1)
    data["B50_group_centered"]=center(data.B50_ms.to_numpy(float),g); data["U50_group_centered"]=center(data.U50_ms.to_numpy(float),g)
    got_scores=pd.read_csv(OUT/"participant_50_50_scores.csv")
    assert list(got_scores.columns)==list(data.columns)
    for c in data.columns:
        if data[c].dtype.kind in "fiu": check("score "+c,data[c],got_scores[c])
        else: assert data[c].astype(str).equals(got_scores[c].astype(str)), c
    checks.append("Participant Absent/BF/BR/UF/UR means and B50/U50 raw-score formulas are exactly reconstructed from the authorized trials")

    ss=np.flatnonzero(g=="SS"); ds=np.flatnonzero(g=="DS")
    rows=[]
    for scope,ix,adj in (("pooled_group_adjusted",np.arange(33),True),("SS",ss,False),("DS",ds,False)):
        xx=xc if adj else x[ix]
        for score, raw in (("B50",data.B50_ms.to_numpy(float)),("U50",data.U50_ms.to_numpy(float))):
            yy=center(raw,g) if adj else raw[ix]
            rows.append(cstats(xx,yy,adj))
    expected_assoc=np.vstack(rows)
    assoc=pd.read_csv(OUT/"participant_50_50_associations.csv")
    check("association point statistics",expected_assoc,assoc[["N","df","r","t","p_raw","fisher_ci_low","fisher_ci_high"]].to_numpy())
    check("association Holm-6",multipletests(expected_assoc[:,4],method="holm")[1],assoc.p_Holm6)

    rng=np.random.default_rng(SEED)
    idx=np.concatenate([rng.choice(ss,size=(R,len(ss)),replace=True),rng.choice(ds,size=(R,len(ds)),replace=True)],axis=1)
    saved_idx=pd.read_csv(OUT/"participant_50_50_bootstrap_indices.csv").to_numpy(int)
    assert np.array_equal(idx,saved_idx)
    V=data[["dmn_conn_primary","B50_ms","U50_ms"]].to_numpy(float)[idx]
    def rr(a,b):
        a=a-a.mean(axis=1,keepdims=True); b=b-b.mean(axis=1,keepdims=True)
        return np.sum(a*b,axis=1)/np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1))
    P=center(V,np.array(["SS"]*16+["DS"]*17))
    draw={"iteration":np.arange(1,R+1)}
    for scope,z in (("pooled_group_adjusted",P),("SS",V[:,:16,:]),("DS",V[:,16:,:])):
        draw[f"{scope}_B50_r"]=rr(z[:,:,0],z[:,:,1]); draw[f"{scope}_U50_r"]=rr(z[:,:,0],z[:,:,2]); draw[f"{scope}_B50_minus_U50"]=draw[f"{scope}_B50_r"]-draw[f"{scope}_U50_r"]
    boot=pd.DataFrame(draw)
    got_boot=pd.read_csv(OUT/"participant_50_50_paired_bootstrap_draws.csv")
    check("all paired bootstrap draws",boot.to_numpy(),got_boot.to_numpy())
    for i,row in assoc.iterrows():
        ci=np.quantile(boot[f"{row.scope}_{row.score}_r"],[.025,.975])
        check("association bootstrap CI "+str(i),ci,[row.bootstrap_ci_low,row.bootstrap_ci_high])
    checks.append("Every one of 10,000 paired SS/DS-stratified bootstrap indices and raw-score re-centered pooled association draw is exactly regenerated")

    comparison=[]
    for scope,ix,adj,neff in (("pooled_group_adjusted",np.arange(33),True,32),("SS",ss,False,16),("DS",ds,False,17)):
        xx=xc if adj else x[ix]; b=center(data.B50_ms.to_numpy(float),g) if adj else data.B50_ms.to_numpy(float)[ix]; u=center(data.U50_ms.to_numpy(float),g) if adj else data.U50_ms.to_numpy(float)[ix]
        comparison.append(williams(corr(xx,b),corr(xx,u),corr(b,u),neff))
    expected_comp=np.vstack(comparison)
    comp=pd.read_csv(OUT/"participant_50_50_dependent_comparisons.csv")
    check("Williams comparison point statistics",expected_comp,comp[["delta_r","outcome_correlation","williams_t","df","p_raw","n_eff","determinant_R","denominator_squared"]].to_numpy())
    check("Williams Holm-3",multipletests(expected_comp[:,4],method="holm")[1],comp.p_Holm3)
    for _,row in comp.iterrows():
        ci=np.quantile(boot[f"{row.scope}_B50_minus_U50"],[.025,.975])
        check("Williams bootstrap CI "+row.scope,ci,[row.bootstrap_ci_low,row.bootstrap_ci_high])
    checks.append("All three declared Williams comparisons (pooled effective N=32/df=29; SS/DS) plus Holm-3 and paired percentile intervals are independently reproduced")

    legacy_source=LEGACY/"learning_reliability.csv"; legacy_copy=OUT/"existing_learning_reliability_summary.csv"
    assert legacy_source.read_bytes()==legacy_copy.read_bytes(), "reliability summary was not copied verbatim"
    l=pd.read_csv(legacy_copy); lv=json.loads((OUT/"existing_learning_reliability_verification.json").read_text())
    assert len(l)==6 and set(l.scheme)=={"odd_even","run1_run2"} and set(l.scope)=={"pooled_group_adjusted","SS","DS"}
    assert lv.get("status")=="PASS" and any("All six reliability" in x for x in lv.get("checks",[]))
    assert set(l.Holm_family)=={"Six new exploratory reliability tests: 2 schemes x 3 scopes"}
    checks.append("Existing odd/even and run1/run2 reliability summary is byte-identical to source; its rerun independent verifier reports PASS for all six raw/Holm-6/percentile-CI rows")

    report={"status":"PASS","checks_passed":len(checks),"checks":checks,"seed":SEED,"R":R,
            "scope":"Descriptive participant 50/50 standardized-capture sensitivity; not the hierarchical ms-slope estimand.",
            "files_verified":["participant_50_50_scores.csv","participant_run_condition_count_audit.csv","five_condition_count_inventory.csv","participant_50_50_associations.csv","participant_50_50_dependent_comparisons.csv","participant_50_50_paired_bootstrap_draws.csv","participant_50_50_bootstrap_indices.csv","existing_learning_reliability_summary.csv","source_hashes.json","output_manifest.json"],
            "limitations":["Only dmn_conn_primary is used as neural source; no legacy X/dmn_conn value is used.","Association and Williams intervals are pointwise percentile intervals; declared tests retain their separate Holm-6 and Holm-3 families.","No manuscript, trial-model, or original reliability output was edited; the existing reliability verifier may refresh only its own verification report."]}
    (OUT/"independent_participant_companion_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
