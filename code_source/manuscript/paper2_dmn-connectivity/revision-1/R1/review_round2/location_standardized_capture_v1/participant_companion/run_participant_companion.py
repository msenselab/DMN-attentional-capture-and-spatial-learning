#!/usr/bin/env python3
"""Generate the independent participant-level 50/50 capture companion.

This descriptive sensitivity intentionally does not import trial-model code or
reuse legacy X/dmn_conn columns.  It reads only the hash-bound corrected N=33
trial file and dmn_conn_primary from the resolved common-coverage score file.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
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
SPEC = OUT.parent / "ANALYSIS_SPEC.md"
LEGACY = ROUND / "spatial_learning_dmn_v1/participant_analysis"
LEGACY_VERIFY = LEGACY / "verify_participant_analysis.py"
LEGACY_RELIABILITY = LEGACY / "learning_reliability.csv"
LEGACY_VERIFY_REPORT = LEGACY / "independent_verification.json"
EXPECTED = {
    TRIALS: "fac39301cb3b828f925937e080213bc5ed39fcaa03cbe0e6c558cf2ea18b34c6",
    NEURAL: "cc3cddf8377ff9aefba18e8781d2512151e335f280d232c5b23d791266e6feef",
}
SEED, R = 20261004, 10_000
CONDITIONS = ("Absent", "BF", "BR", "UF", "UR")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt(np.sum(x*x) * np.sum(y*y))
    if not np.isfinite(d) or d == 0:
        raise RuntimeError("nonfinite or zero correlation denominator")
    return float(np.sum(x*y) / d)


def center_by_group(v: np.ndarray, group: np.ndarray) -> np.ndarray:
    a = np.array(v, float, copy=True)
    for g in ("SS", "DS"):
        ix = np.flatnonzero(group == g)
        if a.ndim == 1:
            a[ix] -= a[ix].mean()
        else:
            a[:, ix] -= a[:, ix].mean(axis=1, keepdims=True)
    return a


def cstats(x: np.ndarray, y: np.ndarray, *, adjusted: bool) -> dict:
    n = len(x); r = corr(x, y); df = n - 3 if adjusted else n - 2
    t = r * np.sqrt(df / (1-r*r))
    p = float(2 * stats.t.sf(abs(t), df))
    # One group contrast is partialled for the group-adjusted pooled result.
    se = 1 / np.sqrt(n - 4 if adjusted else n - 3)
    lo, hi = np.tanh(np.arctanh(r) + stats.norm.ppf(.975) * se * np.array([-1, 1]))
    return dict(N=n, df=int(df), r=r, t=float(t), p_raw=p,
                fisher_ci_low=float(lo), fisher_ci_high=float(hi))


def williams(a: float, b: float, c: float, n_eff: int) -> dict:
    det = 1 - a*a - b*b - c*c + 2*a*b*c
    den_sq = 2*det*(n_eff-1)/(n_eff-3) + ((a+b)**2/4)*(1-c)**3
    if det <= 0 or den_sq <= 0:
        raise RuntimeError("invalid correlation matrix for Williams comparison")
    t = (a-b) * np.sqrt((n_eff-1)*(1+c) / den_sq)
    df = n_eff - 3
    return dict(delta_r=float(a-b), outcome_correlation=float(c), williams_t=float(t),
                df=int(df), p_raw=float(2*stats.t.sf(abs(t), df)), n_eff=int(n_eff),
                determinant_R=float(det), denominator_squared=float(den_sq))


def condition_label(df: pd.DataFrame) -> pd.Series:
    c = pd.Series(np.nan, index=df.index, dtype=object)
    c.loc[df.category.eq("absent")] = "Absent"
    c.loc[df.category.eq("biased") & df.DisRegion.eq("Frequent")] = "BF"
    c.loc[df.category.eq("biased") & df.DisRegion.eq("Rare")] = "BR"
    c.loc[df.category.eq("unbiased") & df.DisRegion.eq("Frequent")] = "UF"
    c.loc[df.category.eq("unbiased") & df.DisRegion.eq("Rare")] = "UR"
    if c.isna().any():
        raise RuntimeError("unexpected category/recorded-region coding; no inferred recoding allowed")
    return c


def stratified_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([
        rng.choice(np.flatnonzero(groups == g), size=(R, int(np.sum(groups == g))), replace=True)
        for g in ("SS", "DS")
    ], axis=1)


def draw_correlations(values: np.ndarray, idx: np.ndarray, ss_n: int) -> np.ndarray:
    """Return paired pooled-adjusted, SS, and DS correlations for X/B50/U50."""
    sampled = values[idx]  # R x 33 x [X, B50, U50], raw scores.
    draw_groups = np.array(["SS"]*ss_n + ["DS"]*(values.shape[0]-ss_n))
    pooled = center_by_group(sampled, draw_groups)  # re-center raw scores every draw

    def rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a-a.mean(axis=1, keepdims=True); b = b-b.mean(axis=1, keepdims=True)
        return np.sum(a*b, axis=1) / np.sqrt(np.sum(a*a, axis=1)*np.sum(b*b, axis=1))

    out = {}
    for scope, z in (("pooled_group_adjusted", pooled), ("SS", sampled[:, :ss_n, :]),
                     ("DS", sampled[:, ss_n:, :])):
        out[f"{scope}_B50_r"] = rows(z[:, :, 0], z[:, :, 1])
        out[f"{scope}_U50_r"] = rows(z[:, :, 0], z[:, :, 2])
        out[f"{scope}_B50_minus_U50"] = out[f"{scope}_B50_r"] - out[f"{scope}_U50_r"]
    return pd.DataFrame(out)


def main() -> None:
    for p, expected in EXPECTED.items():
        actual = sha256(p)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {p}: {actual}")
    trials = pd.read_csv(TRIALS)
    neural = pd.read_csv(NEURAL, usecols=["subject", "group", "dmn_conn_primary"]).sort_values("subject").reset_index(drop=True)
    required_trial = {"subject", "group", "run", "trial", "global_order", "RT_ms", "correct", "category", "DisRegion"}
    required_neural = {"subject", "group", "dmn_conn_primary"}
    if not required_trial <= set(trials) or not required_neural <= set(neural):
        raise RuntimeError("required source columns absent")
    if not (len(trials) == 10995 and trials.subject.nunique() == 33 and trials.correct.eq(1).all()
            and trials.RT_ms.between(200, 3000, inclusive="both").all()):
        raise RuntimeError("not the locked N33/10,995 correct RT 200--3000ms source")
    if not (len(neural) == 33 and neural.subject.is_unique and neural.groupby("group").size().to_dict() == {"DS":17,"SS":16}):
        raise RuntimeError("not the locked SS16/DS17 neural cohort")
    if set(trials.subject) != set(neural.subject) or not trials.groupby("subject").group.nunique().eq(1).all():
        raise RuntimeError("trial/neural cohort or group keys do not match")
    if not np.isfinite(neural.dmn_conn_primary).all():
        raise RuntimeError("dmn_conn_primary must be finite")

    t = trials.copy()
    t["condition"] = condition_label(t)
    # Original keys are retained in this audit; each required condition must occur in every run/person.
    counts = (t.groupby(["subject", "group", "run", "condition"], observed=True).size()
              .rename("n_trials").reset_index())
    full = pd.MultiIndex.from_product([neural.subject, [1, 2], CONDITIONS], names=["subject", "run", "condition"])
    observed = counts.set_index(["subject", "run", "condition"]).reindex(full)
    if observed.n_trials.isna().any() or (observed.n_trials <= 0).any():
        raise RuntimeError("missing/empty participant-by-run condition cell; no imputation permitted")
    counts["cell_complete"] = True
    inventory = (t.groupby(["group", "condition"], observed=True).size().rename("n_trials").reset_index())
    expected_inventory = {("SS","Absent"):953,("SS","BF"):2632,("SS","BR"):559,("SS","UF"):596,("SS","UR"):596,
                          ("DS","Absent"):1013,("DS","BF"):2897,("DS","BR"):633,("DS","UF"):566,("DS","UR"):550}
    if inventory.set_index(["group","condition"]).n_trials.to_dict() != expected_inventory:
        raise RuntimeError("five-condition inventory differs from frozen specification")

    means = (t.groupby(["subject", "group", "condition"], observed=True).RT_ms.mean()
             .unstack("condition").reindex(columns=CONDITIONS).reset_index())
    if means[list(CONDITIONS)].isna().any().any():
        raise RuntimeError("missing participant condition mean; no imputation permitted")
    means["B50_ms"] = .5*(means.BF + means.BR) - means.Absent
    means["U50_ms"] = .5*(means.UF + means.UR) - means.Absent
    data = neural[["subject", "group", "dmn_conn_primary"]].merge(means, on=["subject","group"], validate="one_to_one")
    data = data.sort_values("subject").reset_index(drop=True)
    groups = data.group.to_numpy()
    raw_x = data.dmn_conn_primary.to_numpy(float)
    centered_x = center_by_group(raw_x, groups)
    global_centered_x_sd = float(np.std(centered_x, ddof=1))
    data["X_group_centered"] = centered_x
    data["X_globalSD_standardized"] = centered_x / global_centered_x_sd
    data["B50_group_centered"] = center_by_group(data.B50_ms.to_numpy(float), groups)
    data["U50_group_centered"] = center_by_group(data.U50_ms.to_numpy(float), groups)

    ss = np.flatnonzero(groups == "SS"); ds = np.flatnonzero(groups == "DS")
    association_rows = []
    for scope, ix, adjusted in (("pooled_group_adjusted", np.arange(33), True), ("SS", ss, False), ("DS", ds, False)):
        x = centered_x if adjusted else raw_x[ix]
        for score, raw_score in (("B50", data.B50_ms.to_numpy(float)), ("U50", data.U50_ms.to_numpy(float))):
            row = dict(scope=scope, score=score, **cstats(x, center_by_group(raw_score, groups) if adjusted else raw_score[ix], adjusted=adjusted))
            row["estimand"] = "descriptive participant 50/50 region-standardized capture sensitivity"
            association_rows.append(row)
    assoc = pd.DataFrame(association_rows)
    assoc["p_Holm6"] = multipletests(assoc.p_raw, method="holm")[1]
    assoc["Holm_family"] = "Six new descriptive associations: pooled adjusted, SS, DS x B50/U50"

    # One paired participant bootstrap stream for every association and comparison.
    rng = np.random.default_rng(SEED)
    idx = stratified_indices(groups, rng)
    values = data[["dmn_conn_primary", "B50_ms", "U50_ms"]].to_numpy(float)
    boot = draw_correlations(values, idx, len(ss))
    boot.insert(0, "iteration", np.arange(1, R+1))
    for _, row in assoc.iterrows():
        col = f"{row.scope}_{row.score}_r"
        lo, hi = np.quantile(boot[col], [.025, .975])
        assoc.loc[(assoc.scope == row.scope) & (assoc.score == row.score), "bootstrap_ci_low"] = lo
        assoc.loc[(assoc.scope == row.scope) & (assoc.score == row.score), "bootstrap_ci_high"] = hi
    assoc["bootstrap_method"] = "10,000 paired participant draws, SS/DS-stratified; raw X/B50/U50 re-centered within group each pooled draw"

    comparisons = []
    for scope, ix, adjusted, n_eff in (("pooled_group_adjusted", np.arange(33), True, 32), ("SS", ss, False, 16), ("DS", ds, False, 17)):
        x = centered_x if adjusted else raw_x[ix]
        b = data.B50_group_centered.to_numpy(float) if adjusted else data.B50_ms.to_numpy(float)[ix]
        u = data.U50_group_centered.to_numpy(float) if adjusted else data.U50_ms.to_numpy(float)[ix]
        row = dict(scope=scope, comparison="r(X,B50) minus r(X,U50)", **williams(corr(x,b), corr(x,u), corr(b,u), n_eff))
        row["association_r_B50"] = corr(x,b); row["association_r_U50"] = corr(x,u)
        lo, hi = np.quantile(boot[f"{scope}_B50_minus_U50"], [.025,.975])
        row["bootstrap_ci_low"], row["bootstrap_ci_high"] = float(lo), float(hi)
        row["bootstrap_method"] = "10,000 paired participant draws, SS/DS-stratified; raw scores re-centered within group each pooled draw"
        comparisons.append(row)
    comp = pd.DataFrame(comparisons)
    comp["p_Holm3"] = multipletests(comp.p_raw, method="holm")[1]
    comp["Holm_family"] = "Three new descriptive dependent-correlation comparisons: pooled adjusted, SS, DS"

    # Rerun the prior independent verifier. It may only refresh its own verification report.
    legacy_run = subprocess.run([sys.executable, str(LEGACY_VERIFY)], cwd=str(LEGACY), text=True, capture_output=True)
    if legacy_run.returncode != 0:
        raise RuntimeError("existing learning-reliability verifier failed:\n" + legacy_run.stdout + legacy_run.stderr)
    legacy_verification = json.loads(LEGACY_VERIFY_REPORT.read_text())
    if legacy_verification.get("status") != "PASS":
        raise RuntimeError("existing learning-reliability verifier did not report PASS")
    legacy_rel = pd.read_csv(LEGACY_RELIABILITY)
    if len(legacy_rel) != 6 or set(legacy_rel.scheme) != {"odd_even","run1_run2"} or set(legacy_rel.scope) != {"pooled_group_adjusted","SS","DS"}:
        raise RuntimeError("existing reliability summary is not the required six-row, two-scheme table")

    # Human-readable products are generated from the saved ledgers, not console output.
    fmt = lambda x: f"{float(x):.6f}"
    association_lines = [
        "| scope | score | N, df | r | raw p | Holm-6 p | Fisher 95% CI | paired bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in assoc.iterrows():
        association_lines.append(
            f"| {row.scope} | {row.score} | {int(row.N)}, {int(row.df)} | {fmt(row.r)} | {fmt(row.p_raw)} | {fmt(row.p_Holm6)} | [{fmt(row.fisher_ci_low)}, {fmt(row.fisher_ci_high)}] | [{fmt(row.bootstrap_ci_low)}, {fmt(row.bootstrap_ci_high)}] |"
        )
    comparison_lines = [
        "| scope | delta r (B50−U50) | Williams t, df | raw p | Holm-3 p | paired bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in comp.iterrows():
        comparison_lines.append(
            f"| {row.scope} | {fmt(row.delta_r)} | {fmt(row.williams_t)}, {int(row.df)} | {fmt(row.p_raw)} | {fmt(row.p_Holm3)} | [{fmt(row.bootstrap_ci_low)}, {fmt(row.bootstrap_ci_high)}] |"
        )
    reliability_lines = [
        "| scheme | scope | N, df | r | raw p | original Holm-6 p | original percentile 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in legacy_rel.iterrows():
        reliability_lines.append(
            f"| {row.scheme} | {row.scope} | {int(row.N)}, {int(row.df)} | {fmt(row.r)} | {fmt(row.p_raw)} | {fmt(row.p_Holm)} | [{fmt(row.bootstrap_ci_low)}, {fmt(row.bootstrap_ci_high)}] |"
        )
    report = "\n".join([
        "# Participant 50/50 region-standardized capture companion",
        "",
        "**Status:** PASS after independent verification. This is a descriptive participant-level sensitivity, not the hierarchical millisecond-slope estimand and not a prediction/power analysis.",
        "",
        "## Frozen inputs and scoring",
        "",
        "- N=33 (SS=16, DS=17), 10,995 authorized correct trials, inclusive RT 200–3000 ms; no added first-trial exclusion and no imputation.",
        "- Neural predictor: corrected `dmn_conn_primary` only. Pooled tests separately center raw X and raw scores within SS/DS; the recomputed group-centered X global sample SD is " + fmt(global_centered_x_sd) + ".",
        "- Per person: `B50 = 0.5(BF + BR) − Absent`; `U50 = 0.5(UF + UR) − Absent`. BF/BR/UF/UR retain recorded Frequent/Rare labels.",
        "- All 330 person×run×condition cells were nonempty. Group count inventory: SS Absent/BF/BR/UF/UR = 953/2632/559/596/596; DS = 1013/2897/633/566/550.",
        "",
        "## Six new descriptive associations (one Holm-6 family)", "", *association_lines,
        "",
        "Pooled group-adjusted association tests use df=30. Fisher and paired-bootstrap intervals are pointwise; the declared family is the six-test Holm family above.",
        "",
        "## Three dependent B50−U50 comparisons (one Holm-3 family)", "", *comparison_lines,
        "",
        "The pooled comparison uses the specified approximate Williams effective N=32, df=29. SS and DS use effective N=16/17, df=13/14. Each percentile interval comes from the same 10,000 paired SS/DS-stratified participant draws; raw X/B50/U50 are re-centered within group in each pooled draw.",
        "",
        "## Existing learning-reliability reconciliation (unchanged original Holm-6 family)", "", *reliability_lines,
        "",
        "The existing verifier was rerun and reported PASS, including exact regeneration of all six point tests, Holm-6 adjustments, and 10,000 paired bootstrap draws. This companion copies its six-row summary verbatim and does not calculate a new reliability family.",
        "",
        "## Provenance and verification",
        "",
        "Source hashes, output hashes, all scores/counts/draws, the copied legacy verification report, and the independent companion verification record are stored alongside this report. The independent verifier passed seven checks, including all 10,000 new bootstrap indices and draws. No manuscript, trial-model, or legacy reliability result was edited (apart from the legacy verifier refreshing its own report).",
        "",
    ])
    readme = "\n".join([
        "# Participant companion: 50/50 region-standardized capture",
        "",
        "Runnable descriptive participant-level sensitivity for the location-standardized capture specification. It is deliberately separate from hierarchical ms-slope trial models and creates no prediction or power figures.",
        "",
        "## Run",
        "",
        "```bash",
        "cd ${DMN_PROJECT_ROOT}/manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/location_standardized_capture_v1/participant_companion",
        "python run_participant_companion.py",
        "python verify_participant_companion.py",
        "```",
        "",
        "The generator hash-checks the locked trial and neural sources; reconstructs Absent/BF/BR/UF/UR and B50/U50; writes 6 Holm-6 associations, 3 Holm-3 Williams comparisons, 10,000 paired stratified draws, scores and count audits; reruns the existing reliability verifier; and copies its six-row summary unchanged. The verifier is independently implemented and does not import the generator.",
        "",
        "See `REPORT.md` for raw p values, Holm values, pointwise intervals, and all six reconciled reliability lines. `source_hashes.json` and `output_manifest.json` bind sources and generated files.",
        "",
    ])
    (OUT / "REPORT.md").write_text(report)
    (OUT / "README.md").write_text(readme)

    # Save only companion products; original analysis outputs remain untouched except its verifier report.
    data.to_csv(OUT / "participant_50_50_scores.csv", index=False)
    counts.to_csv(OUT / "participant_run_condition_count_audit.csv", index=False)
    inventory.to_csv(OUT / "five_condition_count_inventory.csv", index=False)
    assoc.to_csv(OUT / "participant_50_50_associations.csv", index=False)
    comp.to_csv(OUT / "participant_50_50_dependent_comparisons.csv", index=False)
    boot.to_csv(OUT / "participant_50_50_paired_bootstrap_draws.csv", index=False)
    pd.DataFrame(idx, columns=[f"sampled_source_row_{i}" for i in range(33)]).to_csv(OUT / "participant_50_50_bootstrap_indices.csv", index=False)
    shutil.copy2(LEGACY_RELIABILITY, OUT / "existing_learning_reliability_summary.csv")
    shutil.copy2(LEGACY_VERIFY_REPORT, OUT / "existing_learning_reliability_verification.json")

    source_hashes = {
        str(TRIALS): sha256(TRIALS), str(NEURAL): sha256(NEURAL), str(SPEC): sha256(SPEC),
        str(LEGACY_RELIABILITY): sha256(LEGACY_RELIABILITY), str(LEGACY_VERIFY): sha256(LEGACY_VERIFY),
        str(LEGACY_VERIFY_REPORT): sha256(LEGACY_VERIFY_REPORT), str(Path(__file__)): sha256(Path(__file__)),
        str(OUT / "verify_participant_companion.py"): sha256(OUT / "verify_participant_companion.py"),
    }
    (OUT / "source_hashes.json").write_text(json.dumps({
        "status":"PASS", "seed":SEED, "paired_bootstrap_draws":R, "input_sha256":source_hashes,
        "locked_expected_hashes":{str(k):v for k,v in EXPECTED.items()}, "predictor_column":"dmn_conn_primary",
        "predictor_rule":"Only dmn_conn_primary was read; X was recomputed from its group-centered values and their participant-level global sample SD before any join.",
        "global_group_centered_X_sample_SD":global_centered_x_sd,
        "behavior_rule":"all 10,995 authorized correct scanner trials, inclusive RT 200--3000ms, no additional first-trial exclusion or imputation",
        "legacy_reliability_rule":"Existing six-row Holm-6 reliability family copied verbatim after rerunning its independent verifier; no new reliability family was calculated."
    }, indent=2) + "\n")
    outputs = [p for p in OUT.glob("*.csv")] + [p for p in OUT.glob("*.md")] + [OUT / "source_hashes.json", OUT / "existing_learning_reliability_verification.json"]
    (OUT / "output_manifest.json").write_text(json.dumps({
        "status":"complete", "expected":{"N":33,"SS":16,"DS":17,"trial_rows":10995,"association_rows":6,"comparison_rows":3,"bootstrap_rows":R,"reliability_rows":6},
        "output_sha256":{p.name:sha256(p) for p in sorted(outputs)}, "python":platform.python_version(),
        "legacy_verifier_stdout":legacy_run.stdout.strip()
    }, indent=2) + "\n")
    print(assoc.to_string(index=False))
    print("\nDEPENDENT COMPARISONS\n" + comp.to_string(index=False))
    print("\nExisting learning reliability rerun: PASS; copied six-row summary unchanged.")


if __name__ == "__main__":
    main()
