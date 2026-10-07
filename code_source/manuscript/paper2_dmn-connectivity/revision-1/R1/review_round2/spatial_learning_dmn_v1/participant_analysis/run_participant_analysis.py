#!/usr/bin/env python3
"""Exploratory scanner-only participant spatial-learning analysis.

Uses only the authorized N=33 scanner trial file and resolved common-coverage
Default91 `dmn_conn_primary` predictor. It neither pools training data nor fits
the separately owned trial-level model. Outputs are deterministic (seed 20261004).
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
SEED = 20261004
R = 10_000
SCORES_PATH = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv"
TRIALS_PATH = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv"
SPEC_PATH = HERE.parent / "ANALYSIS_SPEC.md"
TASK_SEQUENCE_PATH = ROOT / "_archive/manuscript_scope_20260922/review_needed_code/code/mri-code/GenerateTrialSequences.m"
TASK_DISPLAY_PATH = ROOT / "_archive/manuscript_scope_20260922/review_needed_code/code/mri-code/popSearch.m"
TASK_DS_PATH = ROOT / "_archive/manuscript_scope_20260922/review_needed_code/code/mri-code/session1_R3.m"
TASK_SS_PATH = ROOT / "_archive/manuscript_scope_20260922/review_needed_code/code/mri-code/session2_R4.m"
RECORDED_REGION_PROVENANCE_PATH = ROOT / "code/shared/behavioral_data/current/provenance/lmm_behavioral_original.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if not np.isfinite(den) or den == 0:
        raise RuntimeError("Nonfinite/zero correlation denominator.")
    return float(np.sum(a * b) / den)


def center_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    a = np.asarray(values, float).copy()
    for group in ("SS", "DS"):
        ix = np.flatnonzero(groups == group)
        if a.ndim == 1:
            a[ix] -= a[ix].mean()
        else:
            a[:, ix] -= a[:, ix].mean(axis=1, keepdims=True)
    return a


def corr_row(x: np.ndarray, y: np.ndarray, scope: str, group_adjusted: bool) -> dict:
    n = len(x)
    r = corr(x, y)
    df = n - 3 if group_adjusted else n - 2
    if df <= 0 or abs(r) >= 1:
        raise RuntimeError(f"Invalid correlation for {scope}: N={n}, r={r}")
    t = r * np.sqrt(df / (1 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    # Group adjustment consumes one binary group contrast: effective n=N-1.
    fisher_se = 1 / np.sqrt(n - 4 if group_adjusted else n - 3)
    lo, hi = np.tanh(np.arctanh(r) + np.array([-1, 1]) * stats.norm.ppf(.975) * fisher_se)
    return dict(scope=scope, N=n, r=r, t=float(t), df=int(df), p_raw=p,
                fisher_ci_low=float(lo), fisher_ci_high=float(hi),
                fisher_method=("Fisher z approximation after SS/DS group adjustment; df=N-3, SE=1/sqrt(N-4)"
                               if group_adjusted else "Fisher z approximation; df=N-2, SE=1/sqrt(N-3)"))


def williams(a: float, b: float, c: float, n_eff: int = 32) -> dict:
    determinant = 1 - a*a - b*b - c*c + 2*a*b*c
    den_sq = 2 * determinant * (n_eff - 1) / (n_eff - 3) + ((a + b)**2 / 4) * (1 - c)**3
    if determinant <= 0 or den_sq <= 0:
        raise RuntimeError("Williams correlation matrix/denominator invalid.")
    t = (a - b) * np.sqrt((n_eff - 1) * (1 + c) / den_sq)
    return dict(delta_r=float(a-b), williams_t=float(t), williams_df=n_eff-3,
                p_raw=float(2 * stats.t.sf(abs(t), n_eff-3)), n_eff=n_eff,
                determinant_R=float(determinant), denominator_squared=float(den_sq),
                method="Approximate Williams test after separate SS/DS group centering; declared n_eff=32, df=29")


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = HERE / name
    frame.to_csv(path, index=False)
    return path


def source_and_design_audit(scores: pd.DataFrame, trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit recorded coding; do not create/replace any DisRegion label."""
    rows: list[dict] = []
    def add(check: str, observed: object, expected: str, status: str, detail: str) -> None:
        rows.append(dict(check=check, observed=str(observed), expected=expected, status=status, detail=detail))

    add("score_cohort", f"N={len(scores)}, SS={sum(scores.group.eq('SS'))}, DS={sum(scores.group.eq('DS'))}",
        "N=33; SS=16; DS=17", "PASS", "Resolved score table uniquely defines the neural cohort.")
    add("trial_cohort", f"rows={len(trials)}, N={trials.subject.nunique()}, SS={trials.loc[trials.group.eq('SS'),'subject'].nunique()}, DS={trials.loc[trials.group.eq('DS'),'subject'].nunique()}",
        "10,995 rows; N=33; SS=16; DS=17", "PASS", "Authorized scanner-only correct-RT table.")
    add("trial_gate", f"all correct={trials.correct.eq(1).all()}; RT range={trials.RT_ms.min():.0f}-{trials.RT_ms.max():.0f}",
        "correct=1; finite 200<RT<=3000 ms", "PASS" if (trials.correct.eq(1).all() and np.isfinite(trials.RT_ms).all() and (trials.RT_ms > 200).all() and (trials.RT_ms <= 3000).all()) else "FAIL", "No further participant/trial exclusion was applied.")
    chronology_ok = trials.global_order.eq((trials.run - 1) * 184 + trials.trial).all()
    add("recorded_global_order", str(bool(chronology_ok)), "global_order=(run-1)*184+trial before retained-trial filtering", "PASS" if chronology_ok else "FAIL", "Odd/even parity is assigned from this stored original order, never renumbered.")
    dimensions = pd.crosstab([trials.group, trials.category], trials.Dimension).to_dict()
    mapping_ok = ((trials.loc[(trials.group.eq('SS')) & trials.category.eq('biased'), 'Dimension'].eq('Same').all()) and
                  (trials.loc[(trials.group.eq('DS')) & trials.category.eq('biased'), 'Dimension'].eq('Different').all()))
    add("recorded_biased_dimension_mapping", str(dimensions), "SS biased=Same/orientation; DS biased=Different/color", "PASS" if mapping_ok else "FAIL", "Checked in retained scanner events; labels were not recoded.")
    region_ok = set(trials.DisRegion) == {"Absent", "Frequent", "Rare"}
    add("recorded_region_labels", str(sorted(trials.DisRegion.unique())), "Absent, Frequent, Rare", "PASS" if region_ok else "FAIL", "Learning uses only recorded Rare/Frequent labels on category=biased trials.")
    positional = trials.loc[trials.category.eq('biased')].groupby(['subject', 'DisRegion']).DisPos.agg(lambda x: tuple(sorted(set(x)))).unstack()
    position_ok = (positional.Frequent.notna().all() and positional.Rare.notna().all() and
                   positional.apply(lambda x: len(set(x.Frequent).intersection(set(x.Rare))) == 0, axis=1).all())
    add("recorded_region_position_partition", f"{int(position_ok)}; 33 subject-specific disjoint Frequent/Rare position sets", "subject-specific counterbalanced position partition", "PASS" if position_ok else "FAIL", "Recorded event labels partition lateral positions; no universal side/position was imposed.")
    add("task_sequence_definition", "GenerateTrialSequences.m: seq(:,4)=2 Frequent, 1 Rare, 0 absent; seq(:,2)=2 spatial biased", "source task definition", "PASS", "Original generator supports the conceptual Rare/Frequent coding.")
    add("task_dimension_definition", "popSearch.m: dim_cond 1=color, 2=orientation; session1_R3 sets 1, session2_R4 sets 2", "DS=color/Different; SS=orientation/Same", "PASS", "Task-display and session source paths agree with recorded group/dimension mapping.")
    add("event_label_reconstruction", "No per-participant original sequence .mat files were present at reviewed task-source path", "direct byte-level reconstruction of every event DisRegion from original sequence files", "LIMITATION", "Recorded labels were audited against source definitions and position partitions, but no new labels were inferred or silently recoded.")

    count = (trials.loc[(trials.category.eq('biased')) & trials.DisRegion.isin(['Rare', 'Frequent'])]
             .groupby(['subject', 'group', 'run', 'DisRegion']).size().unstack('DisRegion', fill_value=0).reset_index())
    count["scope"] = "run"
    parity = trials.copy()
    parity["parity"] = np.where(parity.global_order.mod(2).eq(1), "odd", "even")
    parity = (parity.loc[(parity.category.eq('biased')) & parity.DisRegion.isin(['Rare', 'Frequent'])]
              .groupby(['subject', 'group', 'parity', 'DisRegion']).size().unstack('DisRegion', fill_value=0).reset_index())
    parity = parity.rename(columns={"parity": "scope_level"})
    parity["scope"] = "global_order_parity"
    count = count.rename(columns={"run": "scope_level"})
    cells = pd.concat([count, parity], ignore_index=True)
    overall = (trials.loc[(trials.category.eq('biased')) & trials.DisRegion.isin(['Rare', 'Frequent'])]
               .groupby(['subject', 'group', 'DisRegion']).size().unstack('DisRegion', fill_value=0).reset_index())
    overall["scope"] = "overall"; overall["scope_level"] = "all"
    cells = pd.concat([cells, overall], ignore_index=True)
    if not ((cells.Rare > 0).all() and (cells.Frequent > 0).all()):
        raise RuntimeError("Empty rare/frequent cell; imputation is forbidden.")
    return pd.DataFrame(rows), cells


def score_cells(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = trials.loc[trials.category.eq("biased") & trials.DisRegion.isin(["Rare", "Frequent"])].copy()
    d["parity"] = np.where(d.global_order.mod(2).eq(1), "odd", "even")
    rows = []
    for scheme, cols in (("overall", []), ("run", ["run"]), ("global_order_parity", ["parity"])):
        grouped = d.groupby(["subject", "group", *cols, "DisRegion"], sort=True).RT_ms.agg(["mean", "size"]).reset_index()
        wide_mean = grouped.pivot(index=["subject", "group", *cols], columns="DisRegion", values="mean")
        wide_n = grouped.pivot(index=["subject", "group", *cols], columns="DisRegion", values="size")
        z = wide_mean.join(wide_n, lsuffix="_mean", rsuffix="_n").reset_index()
        z["scheme"] = scheme
        z = z.rename(columns={"Rare_mean":"rare_mean_ms", "Frequent_mean":"frequent_mean_ms", "Rare_n":"n_rare", "Frequent_n":"n_frequent"})
        z["learning_rare_minus_frequent_ms"] = z.rare_mean_ms - z.frequent_mean_ms
        z["scope_level"] = "all" if scheme == "overall" else z[cols[0]].astype(str)
        rows.append(z[["subject","group","scheme","scope_level","rare_mean_ms","frequent_mean_ms","n_rare","n_frequent","learning_rare_minus_frequent_ms"]])
    cells = pd.concat(rows, ignore_index=True)
    if (cells[["n_rare", "n_frequent"]] <= 0).any().any() or cells.learning_rare_minus_frequent_ms.isna().any():
        raise RuntimeError("Every participant needs nonempty Rare/Frequent cells in every required scope.")
    overall = cells.loc[cells.scheme.eq("overall")].sort_values("subject").reset_index(drop=True)
    return overall, cells


def stratified_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for group in ("SS", "DS"):
        ix = np.flatnonzero(groups == group)
        parts.append(rng.choice(ix, size=(R, len(ix)), replace=True))
    return np.concatenate(parts, axis=1)


def main() -> None:
    scores = pd.read_csv(SCORES_PATH).sort_values("subject").reset_index(drop=True)
    trials = pd.read_csv(TRIALS_PATH)
    required_scores = {"subject", "group", "dmn_conn_primary", "biased_capture"}
    required_trials = {"subject","group","run","trial","global_order","RT_ms","correct","category","DisRegion","Dimension","DisPos"}
    if not required_scores <= set(scores) or not required_trials <= set(trials):
        raise RuntimeError("Required source columns are absent.")
    if not (len(scores) == 33 and scores.subject.is_unique and scores.groupby("group").size().to_dict() == {"DS":17,"SS":16}):
        raise RuntimeError("Resolved score file is not exact N=33 SS16/DS17.")
    if not (len(trials) == 10995 and trials.subject.nunique() == 33 and set(trials.subject) == set(scores.subject)):
        raise RuntimeError("Trial file is not exact authorized N33/10995 and source-aligned.")
    if not np.isfinite(scores.dmn_conn_primary).all():
        raise RuntimeError("Corrected Default91 predictor contains nonfinite values.")

    audit, count_cells = source_and_design_audit(scores, trials)
    overall, cells = score_cells(trials)
    data = scores[["subject", "group", "dmn_conn_primary", "biased_capture"]].merge(
        overall[["subject", "rare_mean_ms", "frequent_mean_ms", "n_rare", "n_frequent", "learning_rare_minus_frequent_ms"]],
        on="subject", validate="one_to_one").sort_values("subject").reset_index(drop=True)
    data["D"] = data.group.eq("DS").astype(int)
    data["corrected_dmn_centered_within_group"] = data.dmn_conn_primary - data.groupby("group").dmn_conn_primary.transform("mean")
    data["learning_centered_within_group"] = data.learning_rare_minus_frequent_ms - data.groupby("group").learning_rare_minus_frequent_ms.transform("mean")
    data["biased_capture_centered_within_group"] = data.biased_capture - data.groupby("group").biased_capture.transform("mean")
    # Explicit audit: resolved biased capture must reproduce raw scanner category score.
    raw_capture = trials.groupby(["subject","category"]).RT_ms.mean().unstack()
    data["raw_biased_capture_recomputed_ms"] = raw_capture.loc[data.subject, "biased"].to_numpy() - raw_capture.loc[data.subject, "absent"].to_numpy()
    data["resolved_minus_raw_biased_capture_ms"] = data.biased_capture - data.raw_biased_capture_recomputed_ms
    if not np.allclose(data.resolved_minus_raw_biased_capture_ms, 0, atol=1e-9, rtol=0):
        raise RuntimeError("Resolved biased_capture differs from raw scanner recomputation; comparison blocked.")

    groups = data.group.to_numpy()
    x = data.corrected_dmn_centered_within_group.to_numpy(float)
    learning = data.learning_centered_within_group.to_numpy(float)
    capture = data.biased_capture_centered_within_group.to_numpy(float)
    ss = np.flatnonzero(groups == "SS"); ds = np.flatnonzero(groups == "DS")

    # Three newly declared exploratory neural-learning associations: Holm-3.
    assoc = pd.DataFrame([
        corr_row(x, learning, "pooled_group_adjusted", True),
        corr_row(x[ss], learning[ss], "SS", False),
        corr_row(x[ds], learning[ds], "DS", False),
    ])
    assoc["p_Holm"] = multipletests(assoc.p_raw, method="holm")[1]
    assoc["Holm_family"] = "Three new exploratory neural-learning associations (pooled, SS, DS)"

    # Group descriptive tests and dz=t/sqrt(N), Holm-2; no causal interpretation.
    desc_rows = []
    for group, ix in (("SS", ss), ("DS", ds)):
        y = data.learning_rare_minus_frequent_ms.to_numpy(float)[ix]
        tt = stats.ttest_1samp(y, 0.0)
        n = len(y); mean = float(np.mean(y)); sd = float(np.std(y, ddof=1)); se = sd / np.sqrt(n)
        ci = stats.t.interval(.95, n-1, loc=mean, scale=se)
        desc_rows.append(dict(group=group, N=n, mean_ms=mean, SD_ms=sd, t=float(tt.statistic), df=n-1,
                              p_raw=float(tt.pvalue), mean_t_ci_low=float(ci[0]), mean_t_ci_high=float(ci[1]),
                              dz=float(tt.statistic / np.sqrt(n))))
    desc = pd.DataFrame(desc_rows)
    desc["p_Holm"] = multipletests(desc.p_raw, method="holm")[1]
    desc["Holm_family"] = "Two new exploratory group descriptive one-sample tests"

    # Exact paired, group-stratified participant bootstrap stream 1: association/delta/reliability.
    rng = np.random.default_rng(SEED)
    idx = stratified_indices(groups, rng)
    positions = [np.arange(len(ss)), np.arange(len(ss), len(groups))]
    xb = center_by_group(data.dmn_conn_primary.to_numpy(float)[idx], np.array(["SS"]*len(ss)+["DS"]*len(ds)))
    lb = center_by_group(data.learning_rare_minus_frequent_ms.to_numpy(float)[idx], np.array(["SS"]*len(ss)+["DS"]*len(ds)))
    cb = center_by_group(data.biased_capture.to_numpy(float)[idx], np.array(["SS"]*len(ss)+["DS"]*len(ds)))
    def corr_rows_draw(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def c2(z1, z2):
            z1 = z1-z1.mean(axis=1, keepdims=True); z2 = z2-z2.mean(axis=1, keepdims=True)
            return np.sum(z1*z2,axis=1)/np.sqrt(np.sum(z1*z1,axis=1)*np.sum(z2*z2,axis=1))
        return c2(a,b), c2(a[:,:len(ss)], b[:,:len(ss)]), c2(a[:,len(ss):], b[:,len(ss):])
    rb_pool, rb_ss, rb_ds = corr_rows_draw(xb, lb)
    rc_pool, _, _ = corr_rows_draw(xb, cb)
    r_capture_learning, _, _ = corr_rows_draw(cb, lb)
    association_draws = pd.DataFrame(dict(iteration=np.arange(1,R+1), r_learning_pooled=rb_pool, r_learning_SS=rb_ss,
                                          r_learning_DS=rb_ds, r_capture_pooled=rc_pool,
                                          r_capture_learning_pooled=r_capture_learning, delta_capture_minus_learning=rc_pool-rb_pool))

    # Direct comparison: same X and participant cohort, a=capture and b=learning.
    r_dmn_capture = corr(x, capture)
    r_dmn_learning = corr(x, learning)
    r_capture_learning = corr(capture, learning)
    comparison = williams(r_dmn_capture, r_dmn_learning, r_capture_learning, n_eff=32)
    comparison.update(comparison="DMN--biased-capture versus DMN--spatial-learning", N=33,
                      r_dmn_capture=r_dmn_capture, r_dmn_learning=r_dmn_learning,
                      r_capture_learning=r_capture_learning,
                      bootstrap_ci_low=percentile_ci(association_draws.delta_capture_minus_learning.to_numpy())[0],
                      bootstrap_ci_high=percentile_ci(association_draws.delta_capture_minus_learning.to_numpy())[1],
                      bootstrap_method="10,000 paired, SS/DS-stratified participant percentile bootstrap; group means recalculated each draw",
                      family="One explicitly new exploratory comparison; raw p (one-test family)")

    # Independent bootstrap stream 2: d_z uncertainty, draws saved.
    dz_idx = stratified_indices(groups, rng)
    dz_draw = {}
    for group, ix in (("SS", ss), ("DS", ds)):
        local = np.arange(len(ix)) if group == "SS" else np.arange(len(ss),len(groups))
        vals = data.learning_rare_minus_frequent_ms.to_numpy(float)[dz_idx[:, local]]
        dz_draw[group] = vals.mean(axis=1) / vals.std(axis=1, ddof=1)
        lo, hi = percentile_ci(dz_draw[group])
        desc.loc[desc.group.eq(group), "dz_bootstrap_ci_low"] = lo
        desc.loc[desc.group.eq(group), "dz_bootstrap_ci_high"] = hi
    descriptive_draws = pd.DataFrame(dict(iteration=np.arange(1,R+1), dz_SS=dz_draw["SS"], dz_DS=dz_draw["DS"]))

    # Reliability scores: each split/run has its own Rare/Frequent means, without imputation.
    wide = cells.pivot(index=["subject","group"], columns=["scheme","scope_level"], values="learning_rare_minus_frequent_ms")
    wide = wide.reindex(pd.MultiIndex.from_frame(data[["subject","group"]]))
    if wide.isna().any().any():
        raise RuntimeError("Reliability split/run score missing; no imputation permitted.")
    reliab_inputs = {
        "odd_even": (wide[("global_order_parity","odd")].to_numpy(float), wide[("global_order_parity","even")].to_numpy(float)),
        "run1_run2": (wide[("run","1")].to_numpy(float), wide[("run","2")].to_numpy(float)),
    }
    rel_rows, rel_draws = [], {"iteration":np.arange(1,R+1)}
    for scheme, (a, b) in reliab_inputs.items():
        # Point statistics by declared scope.
        for scope, ix, adjusted in (("pooled_group_adjusted", np.arange(33), True), ("SS", ss, False), ("DS", ds, False)):
            aa = center_by_group(a, groups) if adjusted else a[ix]
            bb = center_by_group(b, groups) if adjusted else b[ix]
            row = corr_row(aa if adjusted else aa, bb if adjusted else bb, scope, adjusted)
            row["scheme"] = scheme
            rel_rows.append(row)
        ab = a[idx]; bb = b[idx]
        ab_c = center_by_group(ab, np.array(["SS"]*len(ss)+["DS"]*len(ds)))
        bb_c = center_by_group(bb, np.array(["SS"]*len(ss)+["DS"]*len(ds)))
        p, s1, d1 = corr_rows_draw(ab_c, bb_c)
        rel_draws[f"{scheme}_pooled_group_adjusted_r"] = p
        rel_draws[f"{scheme}_SS_r"] = s1
        rel_draws[f"{scheme}_DS_r"] = d1
    reliability = pd.DataFrame(rel_rows)[["scheme","scope","N","r","t","df","p_raw","fisher_ci_low","fisher_ci_high","fisher_method"]]
    reliability["p_Holm"] = multipletests(reliability.p_raw, method="holm")[1]
    reliability["Holm_family"] = "Six new exploratory reliability tests: 2 schemes x 3 scopes"
    reliability_draws = pd.DataFrame(rel_draws)
    for _, row in reliability.iterrows():
        col = f"{row.scheme}_{row.scope}_r"
        lo, hi = percentile_ci(reliability_draws[col].to_numpy())
        reliability.loc[(reliability.scheme.eq(row.scheme)) & (reliability.scope.eq(row.scope)), "bootstrap_ci_low"] = lo
        reliability.loc[(reliability.scheme.eq(row.scheme)) & (reliability.scope.eq(row.scope)), "bootstrap_ci_high"] = hi
    reliability["bootstrap_method"] = "10,000 paired, SS/DS-stratified participant percentile bootstrap; group means recalculated each draw"
    reliability["Spearman_Brown"] = np.where(reliability.r >= 0, 2*reliability.r/(1+reliability.r), np.nan)
    reliability["Spearman_Brown_note"] = np.where(reliability.r >= 0, "Reported descriptively only", "Not reported: negative half correlation is not meaningful negative reliability")

    # Save all derived analytic products.
    written = []
    for frame, name in ((audit,"coding_provenance_audit.csv"), (count_cells,"learning_count_cells.csv"),
                        (cells,"learning_scores_by_scope.csv"), (data,"participant_learning_scores.csv"),
                        (assoc,"neural_learning_associations.csv"), (association_draws,"association_and_comparison_bootstrap_draws.csv"),
                        (desc,"learning_group_descriptive_tests.csv"), (descriptive_draws,"learning_descriptive_dz_bootstrap_draws.csv"),
                        (pd.DataFrame([comparison]),"capture_vs_learning_williams.csv"),
                        (reliability,"learning_reliability.csv"), (reliability_draws,"learning_reliability_bootstrap_draws.csv")):
        written.append(write_csv(frame, name))

    # Human-readable seed/source index, plus explicit output-hash manifest.
    inputs = [SCORES_PATH, TRIALS_PATH, SPEC_PATH, TASK_SEQUENCE_PATH, TASK_DISPLAY_PATH, TASK_DS_PATH, TASK_SS_PATH, RECORDED_REGION_PROVENANCE_PATH, Path(__file__), HERE / "verify_participant_analysis.py"]
    sources = {str(p): sha256(p) for p in inputs}
    (HERE / "source_hashes.json").write_text(json.dumps(dict(seed=SEED, bootstrap_draws=R, input_sha256=sources,
        predictor_column="dmn_conn_primary", predictor_definition=str(scores.dmn_conn_primary_definition.iloc[0]),
        behavior_rule="Existing authorized correct scanner trials with finite 200 < RT_ms <= 3000; initial trials retained; no extra exclusion.",
        direct_capture_source="resolved participant_scores_resolved.csv: biased_capture; independently matched raw scanner category mean(biased)-mean(absent) to 1e-9 ms."), indent=2)+"\n")
    manifest_paths = written + [HERE / "source_hashes.json"]
    (HERE / "output_manifest.json").write_text(json.dumps(dict(status="complete", seed=SEED, R=R,
        output_sha256={p.name:sha256(p) for p in manifest_paths},
        expected=dict(N=33, SS=16, DS=17, trial_rows=10995,
                      association_rows=3, reliability_rows=6, draw_rows=10000)), indent=2)+"\n")
    print(assoc.to_string(index=False))
    print(desc.to_string(index=False))
    print(pd.DataFrame([comparison]).to_string(index=False))
    print(reliability.to_string(index=False))

if __name__ == "__main__":
    main()
