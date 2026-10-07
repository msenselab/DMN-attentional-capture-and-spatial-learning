#!/usr/bin/env python3
"""Reproducible behavioral revision evidence package for Paper 2.

This script is intentionally read-only with respect to source data, manuscript,
and revision plan. It writes only inside its own behavioral-audit directory.

Run from any working directory:
    python ${DMN_PROJECT_ROOT}/manuscript/paper2_dmn-connectivity/revision-1/behavioral-audit/run_behavioral_audit.py

The package distinguishes (a) the frozen N=33 Paper 2 analytic table used to
reconstruct the reported primary association from (b) the current trial tables
used for trial-flow, accuracy, history, and mixed-model sensitivities. A mismatch
between their capture values is reported rather than silently reconciled.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.io import loadmat
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[4])))
OUT = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/behavioral-audit"
RESULTS = OUT / "results"

# Immutable inputs used below. The frozen N=33 analytic table is the only input
# used to reconstruct the manuscript-era primary correlation.
PATHS = {
    "participants": ROOT / "participants.tsv",
    "training_directory": ROOT / "code/cross_dimension/data",
    "scanner_events_with_correctness": ROOT / "derivatives/behavior/vigilance_state/merged_events_with_correct.csv",
    "scanner_analysis_ready": ROOT / "manuscript/data/intermediate/behavioral_data_clean.csv",
    "frozen_paper2_n33": ROOT / "manuscript/paper2/code_and_data/dmn_capture_unified.csv",
    "current_n33_connectivity": ROOT / "manuscript/data/intermediate/dmn_connectivity_merged.csv",
    "active_manifest": ROOT / "code/paper2_dmn-connectivity/ACTIVE_MANIFEST.csv",
    "reviewed_figure2": ROOT / "manuscript/paper2_dmn-connectivity/revision-1/media/image2.png",
    "reviewed_figure2_generator_transcript": ROOT / "ara/src/execution/code__figures_20260602_paper2_high_quality.py",
    "current_figure2v_generator": ROOT / "code/shared/behavioral_data/current/04_figure2v_unified_n35.py",
}

SEED = 20260924
N_BOOT = 10_000
TRAIN_SESSIONS = {"s11", "s12", "s21", "s22"}
TRAIN_BLOCK_MAP = {1: 1, 2: 2, 5: 3, 6: 4, 7: 5, 8: 6}
TRAIN_COLS = [
    "tar_ori", "conditions", "sessions", "dist_cond", "blk", "tar_pos",
    "as_pos", "response_group", "tar_orient_group", "c_orient", "key",
    "correct", "tm_cue", "tm_disp", "tm_key", "rt", "tm_end", "tm_trigger",
]


@dataclass
class Corr:
    n: int
    r: float
    p: float
    ci_low: float
    ci_high: float


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return tuple(np.tanh([z - stats.norm.ppf(.975) * se, z + stats.norm.ppf(.975) * se]))


def pearson(x: Iterable[float], y: Iterable[float]) -> Corr:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 4 or np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
        return Corr(len(x), np.nan, np.nan, np.nan, np.nan)
    r, p = stats.pearsonr(x, y)
    lo, hi = fisher_ci(float(r), len(x))
    return Corr(len(x), float(r), float(p), float(lo), float(hi))


def group_center(frame: pd.DataFrame, columns: list[str], group: str = "group") -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[f"{column}_group_centered"] = out[column] - out.groupby(group)[column].transform("mean")
    return out


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return "< .001" if value < .001 else f"= {value:.3f}".replace("0.", ".")


def pct(numerator: float, denominator: float) -> float:
    return 100 * numerator / denominator if denominator else np.nan


def input_manifest() -> pd.DataFrame:
    rows = []
    for label, path in PATHS.items():
        if path.is_dir():
            rows.append({"label": label, "absolute_path": str(path), "exists": path.exists(),
                         "type": "directory", "bytes": np.nan, "sha256": "directory_not_hashed"})
        else:
            rows.append({"label": label, "absolute_path": str(path), "exists": path.exists(),
                         "type": "file", "bytes": path.stat().st_size if path.exists() else np.nan,
                         "sha256": sha256(path) if path.exists() else "MISSING"})
    return pd.DataFrame(rows)


def load_id_map() -> dict[str, str]:
    # participants.tsv has a malformed final tab in this repository, so parse
    # positional fields exactly as the active N35 source does rather than rely
    # on pandas' header validation.
    mapping: dict[str, str] = {}
    with PATHS["participants"].open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0].startswith("participant_id"):
                continue
            participant_id, origin_name = parts[0].strip(), parts[4].strip()
            if origin_name and "_" in origin_name:
                mapping[origin_name.split("_", 1)[0]] = participant_id.replace("sub-", "")
    return mapping


def biased_category(group: pd.Series, dimension: pd.Series) -> np.ndarray:
    group = group.astype(str)
    dimension = dimension.astype(str).str.lower()
    return np.select(
        [dimension.eq("absent"),
         (group.eq("SS") & dimension.eq("same")) | (group.eq("DS") & dimension.eq("different"))],
        ["absent", "biased"],
        default="unbiased",
    )


def load_training_raw() -> pd.DataFrame:
    id_map = load_id_map()
    pieces = []
    for path in sorted(PATHS["training_directory"].glob("*.mat")):
        original_id, session = path.stem.split("_", 1)
        if session not in TRAIN_SESSIONS:
            continue
        raw = pd.DataFrame(loadmat(path)["trials"], columns=TRAIN_COLS)
        raw["source_subject"] = original_id
        raw["subject"] = id_map.get(original_id, original_id)
        raw["session"] = session
        raw["group"] = np.where(raw["sessions"].eq(2), "SS", "DS")
        raw["mat_block"] = raw["blk"].astype(int)
        raw = raw.loc[raw["mat_block"].isin(TRAIN_BLOCK_MAP)].copy()
        raw["display_block"] = raw["mat_block"].map(TRAIN_BLOCK_MAP).astype(int)
        raw["dimension"] = np.select(
            [raw["conditions"].eq(0),
             (raw["group"].eq("SS") & raw["conditions"].eq(2)) |
             (raw["group"].eq("DS") & raw["conditions"].eq(1))],
            ["absent", "same"], default="different")
        raw["category"] = biased_category(raw["group"], raw["dimension"])
        raw["location"] = raw["dist_cond"].map({0: "Absent", 1: "Rare", 2: "Frequent"}).fillna("Unknown")
        raw["rt_ms"] = raw["rt"] * 1000.0
        # Exact active N35 loader rule: within file, lower 1.5-IQR only; first
        # trial of each original MATLAB block; correctness required.
        q1, q3 = raw["rt"].quantile([.25, .75])
        lower = q1 - 1.5 * (q3 - q1)
        raw["incorrect"] = ~raw["correct"].eq(1)
        raw["rt_invalid"] = raw["rt"].lt(lower) | raw["rt"].isna()
        raw["first_trial"] = raw.groupby("mat_block").cumcount().eq(0)
        raw["retained_rt_rule"] = ~(raw["rt_invalid"] | raw["first_trial"])
        raw["retained_primary"] = ~(
            raw["incorrect"] | raw["rt_invalid"] | raw["first_trial"]
        )
        pieces.append(raw)
    if not pieces:
        raise FileNotFoundError(f"No training files found in {PATHS['training_directory']}")
    out = pd.concat(pieces, ignore_index=True)
    out["phase"] = "training"
    return out


def load_scanner_raw() -> pd.DataFrame:
    raw = pd.read_csv(PATHS["scanner_events_with_correctness"])
    raw["group"] = raw["subject"].str[:2]
    raw["dimension"] = np.where(raw["Distractor"].eq(0), "absent", raw["Dimension"].str.lower())
    raw["category"] = biased_category(raw["group"], raw["dimension"])
    raw["location"] = raw["DisRegion"].astype(str)
    raw["rt_ms"] = raw["RT"] * 1000.0
    raw["incorrect"] = ~raw["correct"].eq(1)
    raw["rt_invalid"] = raw["RT"].lt(.2) | raw["RT"].gt(3) | raw["RT"].isna()
    raw["first_trial"] = raw["trial"].eq(raw.groupby(["subject", "run"])["trial"].transform("min"))
    # This is the current Paper-2 analysis-ready event rule; it does NOT apply
    # correctness because the original analysis-ready CSV has no correctness field.
    raw["retained_rt_rule"] = ~(raw["rt_invalid"] | raw["first_trial"])
    raw["retained_primary"] = raw["retained_rt_rule"]
    raw["retained_correct_rt"] = raw["retained_rt_rule"] & ~raw["incorrect"]
    raw["phase"] = "scanner"
    return raw


def trial_flow(training: pd.DataFrame, scanner: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_data = pd.concat([training, scanner], ignore_index=True, sort=False)
    # Sequential filters preserve an interpretable flow denominator while the
    # component columns separately show overlapping reasons.
    rows = []
    grouping = ["phase", "group", "category", "dimension", "location"]
    for keys, d in all_data.groupby(grouping, dropna=False, observed=True):
        assigned = len(d)
        after_incorrect = d.loc[~d["incorrect"]]
        after_first = after_incorrect.loc[~after_incorrect["first_trial"]]
        retained_correct = after_first.loc[~after_first["rt_invalid"]]
        retained_rt = d.loc[d["retained_rt_rule"]]
        rows.append({
            **dict(zip(grouping, keys)),
            "assigned_trials": assigned,
            "incorrect_trials": int(d["incorrect"].sum()),
            "accuracy_percent": pct((~d["incorrect"]).sum(), assigned),
            "first_trial_flagged": int(d["first_trial"].sum()),
            "rt_invalid_flagged": int(d["rt_invalid"].sum()),
            "retained_rt_rule_trials": int(len(retained_rt)),
            "retained_correct_rt_trials": int(len(retained_correct)),
            "removed_incorrect_sequential": assigned - len(after_incorrect),
            "removed_first_sequential": len(after_incorrect) - len(after_first),
            "removed_rt_sequential": len(after_first) - len(retained_correct),
            "mean_rt_ms_retained_rt_rule": retained_rt["rt_ms"].mean(),
            "sd_rt_ms_retained_rt_rule": retained_rt["rt_ms"].std(ddof=1),
            "mean_rt_ms_retained_correct_rt": retained_correct["rt_ms"].mean(),
            "sd_rt_ms_retained_correct_rt": retained_correct["rt_ms"].std(ddof=1),
        })
    flow = pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)

    # Phase/group totals are retained separately for a concise manuscript table.
    totals = []
    for keys, d in all_data.groupby(["phase", "group"], observed=True):
        assigned = len(d)
        rt_keep = d.loc[d["retained_rt_rule"]]
        correct_mask = d["retained_correct_rt"].fillna(d["retained_primary"]).astype(bool)
        correct_keep = d.loc[correct_mask]
        totals.append({
            "phase": keys[0], "group": keys[1], "subjects": d["subject"].nunique(),
            "assigned_trials": assigned, "incorrect_trials": int(d["incorrect"].sum()),
            "accuracy_percent": pct((~d["incorrect"]).sum(), assigned),
            "rt_rule_retained": len(rt_keep), "correct_rt_retained": len(correct_keep),
            "mean_rt_ms_rt_rule": rt_keep["rt_ms"].mean(), "sd_rt_ms_rt_rule": rt_keep["rt_ms"].std(ddof=1),
        })
    return flow, pd.DataFrame(totals)


def frozen_primary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(PATHS["frozen_paper2_n33"]).copy()
    d["unbiased_capture"] = np.where(d["group"].eq("SS"), d["diff_capture"], d["same_capture"])
    d = group_center(d, ["dmn_conn", "biased_capture", "unbiased_capture", "all_capture", "same_capture", "diff_capture"])
    outcomes = ["biased_capture", "unbiased_capture", "all_capture", "same_capture", "diff_capture"]
    rows = []
    for outcome in outcomes:
        pooled = pearson(d["dmn_conn_group_centered"], d[f"{outcome}_group_centered"])
        rows.append({"scope": "pooled_group_centered", "outcome": outcome, **asdict(pooled)})
        for group, g in d.groupby("group", observed=True):
            within = pearson(g["dmn_conn"], g[outcome])
            rows.append({"scope": f"within_{group}", "outcome": outcome, **asdict(within)})
    correlations = pd.DataFrame(rows)

    # Participant bootstrap: resample people within SS and DS, then recenter all
    # variables inside each resample. Percentile interval, not BCa.
    rng = np.random.default_rng(SEED)
    group_indices = {g: d.index[d.group.eq(g)].to_numpy() for g in ["SS", "DS"]}
    boot = np.full(N_BOOT, np.nan)
    for i in range(N_BOOT):
        index = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in group_indices.values()])
        b = group_center(d.loc[index].copy(), ["dmn_conn", "biased_capture"])
        boot[i] = pearson(b["dmn_conn_group_centered"], b["biased_capture_group_centered"]).r
    boot_df = pd.DataFrame({"iteration": np.arange(1, N_BOOT + 1), "bootstrap_r": boot})
    boot_summary = pd.DataFrame([{
        "target": "DMN x biased capture, frozen N33 table", "iterations_requested": N_BOOT,
        "iterations_finite": int(np.isfinite(boot).sum()), "seed": SEED,
        "method": "group-stratified participant bootstrap; group recentered per resample; percentile interval",
        "median_r": float(np.nanmedian(boot)), "ci_low": float(np.nanquantile(boot, .025)),
        "ci_high": float(np.nanquantile(boot, .975)),
    }])

    loo_rows = []
    for subject in d["subject"]:
        sub = d.loc[~d.subject.eq(subject)].copy()
        sub = group_center(sub, ["dmn_conn", "biased_capture"])
        result = pearson(sub["dmn_conn_group_centered"], sub["biased_capture_group_centered"])
        loo_rows.append({"dropped_subject": subject, **asdict(result)})
    loo = pd.DataFrame(loo_rows)
    return d, correlations, boot_df, boot_summary, loo


def current_trial_capture_and_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the formula implemented by the active history-adjustment source.

    This uses the current analysis-ready scanner CSV and current N33 connectivity
    table. It is a sensitivity analysis only: its raw capture values are checked
    against the frozen N33 table before interpretation.
    """
    beh = pd.read_csv(PATHS["scanner_analysis_ready"]).copy()
    dmn = pd.read_csv(PATHS["current_n33_connectivity"])[["subject", "group", "dmn_mean_conn"]].copy()
    dmn["group"] = dmn["group"].replace({"Same-Dimension": "SS", "Different-Dimension": "DS"})
    beh = beh.merge(dmn, on="subject", how="inner", validate="many_to_one", suffixes=("", "_dmn"))
    beh["group_short"] = beh["subject"].str[:2]
    beh["cond_biased"] = ((beh["group_short"].eq("SS") & beh["distractor_condition"].eq("Same-Dimension Distractor")) |
                          (beh["group_short"].eq("DS") & beh["distractor_condition"].eq("Different-Dimension Distractor"))).astype(int)
    beh["cond_unbiased"] = ((beh["group_short"].eq("SS") & beh["distractor_condition"].eq("Different-Dimension Distractor")) |
                            (beh["group_short"].eq("DS") & beh["distractor_condition"].eq("Same-Dimension Distractor"))).astype(int)
    beh["cond_absent"] = beh["distractor_condition"].eq("No Distractor").astype(int)
    beh["run2"] = beh["run"].eq(2).astype(int)
    beh = beh.sort_values(["subject", "run", "trial", "Onset"]).copy()
    beh["global_trial"] = beh.groupby("subject").cumcount() + 1
    beh["log_global_trial"] = np.log(beh["global_trial"])
    for source, target in [("cond_biased", "prior_biased_n"), ("cond_unbiased", "prior_unbiased_n")]:
        beh[target] = beh.groupby("subject")[source].transform(lambda x: x.cumsum().shift(fill_value=0))
    for col in ["log_global_trial", "prior_biased_n", "prior_unbiased_n"]:
        beh[f"z_{col}"] = beh.groupby("subject")[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
        )

    estimates = []
    diagnostics = []
    predictors = ["cond_biased", "cond_unbiased", "run2", "z_log_global_trial", "z_prior_biased_n", "z_prior_unbiased_n"]
    for subject, sub in beh.groupby("subject", observed=True):
        absent = sub.loc[sub.cond_absent.eq(1), "RT_ms"].mean()
        raw_b = sub.loc[sub.cond_biased.eq(1), "RT_ms"].mean() - absent
        raw_u = sub.loc[sub.cond_unbiased.eq(1), "RT_ms"].mean() - absent
        x = sm.add_constant(sub[predictors].astype(float), has_constant="add")
        fit = sm.OLS(sub["RT_ms"].astype(float), x, missing="drop").fit()
        base = {"subject": subject, "group": sub.group_short.iloc[0], "dmn_mean_conn": sub.dmn_mean_conn.iloc[0],
                "n_trials": int(len(sub)), "n_biased": int(sub.cond_biased.sum()),
                "n_unbiased": int(sub.cond_unbiased.sum()), "n_absent": int(sub.cond_absent.sum())}
        estimates.append({**base, "model": "raw_mean_difference", "biased_capture": raw_b, "unbiased_capture": raw_u})
        estimates.append({**base, "model": "history_adjusted_subject_OLS", "biased_capture": fit.params["cond_biased"],
                          "unbiased_capture": fit.params["cond_unbiased"]})
        diagnostics.append({**base, "model": "history_adjusted_subject_OLS", "r_squared": fit.rsquared,
                            "df_resid": fit.df_resid, "condition_number": fit.condition_number,
                            "biased_se_ms": fit.bse["cond_biased"], "unbiased_se_ms": fit.bse["cond_unbiased"]})
    est = pd.DataFrame(estimates)
    diag = pd.DataFrame(diagnostics)
    associations = []
    for model, sub in est.groupby("model", observed=True):
        sub = group_center(sub, ["dmn_mean_conn", "biased_capture", "unbiased_capture"])
        for outcome in ["biased_capture", "unbiased_capture"]:
            r = pearson(sub["dmn_mean_conn_group_centered"], sub[f"{outcome}_group_centered"])
            associations.append({"model": model, "outcome": outcome, **asdict(r)})
    assoc = pd.DataFrame(associations)
    return beh, est, diag, assoc


def frozen_group_moderation(frozen: pd.DataFrame) -> pd.DataFrame:
    d = frozen.copy()
    d["dmn_z"] = (d["dmn_conn"] - d["dmn_conn"].mean()) / d["dmn_conn"].std(ddof=1)
    d["DS"] = d["group"].eq("DS").astype(int)
    rows = []
    for outcome in ["biased_capture", "unbiased_capture", "all_capture"]:
        fit = smf.ols(f"{outcome} ~ dmn_z * DS", data=d).fit()
        term = "dmn_z:DS"
        ci = fit.conf_int().loc[term]
        rows.append({"outcome": outcome, "formula": f"{outcome} ~ dmn_z * DS", "n": int(fit.nobs),
                     "term": term, "estimate": fit.params[term], "se": fit.bse[term],
                     "ci_low": ci.iloc[0], "ci_high": ci.iloc[1], "t": fit.tvalues[term],
                     "df_resid": fit.df_resid, "p": fit.pvalues[term], "r_squared": fit.rsquared})
    return pd.DataFrame(rows)


def trial_level_mixed_model() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit an explicit N33, accuracy-linked hierarchical log-RT sensitivity.

    It uses raw scanner rows linked to correctness, excludes first run trials and
    RT outside 200--3000 ms, then keeps correct trials. The model is deliberately
    a sensitivity model and not a substitute for the frozen primary correlation.
    """
    raw = load_scanner_raw()
    dmn = pd.read_csv(PATHS["current_n33_connectivity"])[["subject", "dmn_mean_conn"]]
    data = raw.merge(dmn, on="subject", how="inner", validate="many_to_one")
    data = data.loc[data["retained_correct_rt"]].copy()
    data["B"] = data["category"].eq("biased").astype(float)
    data["U"] = data["category"].eq("unbiased").astype(float)
    data["D"] = data["group"].eq("DS").astype(float)
    # X is within-group centered then standardized across the N33 trial table.
    data["X"] = data["dmn_mean_conn"] - data.groupby("group")["dmn_mean_conn"].transform("mean")
    data["X"] = data["X"] / data[["subject", "X"]].drop_duplicates()["X"].std(ddof=1)
    data = data.sort_values(["subject", "run", "trial"]).copy()
    data["log_trial"] = np.log(data.groupby("subject").cumcount() + 1)
    data["log_trial"] = data["log_trial"] - data.groupby("subject")["log_trial"].transform("mean")
    data["log_rt"] = np.log(data["RT"] * 1000.0)
    formula = "log_rt ~ B*X*D + U*X*D + C(run) + log_trial"
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fit = smf.mixedlm(formula, data, groups=data["subject"], re_formula="1 + B + U").fit(
            reml=False, method="lbfgs", maxiter=2000, disp=False
        )
    messages.extend(sorted({f"{warning.category.__name__}: {warning.message}" for warning in captured}))
    names = list(fit.fe_params.index)
    cov = fit.cov_params().loc[names, names]
    fixed_rows = []
    intervals = fit.conf_int().loc[names]
    for term in names:
        fixed_rows.append({"term": term, "estimate_log_rt": fit.fe_params[term], "se": fit.bse_fe[term],
                           "ci_low": intervals.loc[term, 0], "ci_high": intervals.loc[term, 1],
                           "z": fit.fe_params[term] / fit.bse_fe[term],
                           "p": 2 * stats.norm.sf(abs(fit.fe_params[term] / fit.bse_fe[term]))})
    fixed = pd.DataFrame(fixed_rows)

    # Planned simple slopes of DMN for biased and unbiased capture, plus their
    # direct contrast, by group. Coefficients are changes in log RT per 1-SD
    # within-group DMN connectivity; exp(beta)-1 is an approximate percentage.
    def contrast(label: str, weights: dict[str, float]) -> dict:
        vector = np.array([weights.get(term, 0.0) for term in names], dtype=float)
        estimate = float(vector @ fit.fe_params.to_numpy())
        se = float(np.sqrt(vector @ cov.to_numpy() @ vector))
        z = estimate / se
        return {"contrast": label, "estimate_log_rt": estimate, "se": se,
                "ci_low": estimate - stats.norm.ppf(.975) * se,
                "ci_high": estimate + stats.norm.ppf(.975) * se,
                "z": z, "p": float(2 * stats.norm.sf(abs(z))),
                "approx_percent_change": 100 * (np.exp(estimate) - 1)}

    contrasts = pd.DataFrame([
        contrast("SS: DMN slope for biased versus absent", {"B:X": 1}),
        contrast("DS: DMN slope for biased versus absent", {"B:X": 1, "B:X:D": 1}),
        contrast("SS: DMN slope for unbiased versus absent", {"U:X": 1}),
        contrast("DS: DMN slope for unbiased versus absent", {"U:X": 1, "U:X:D": 1}),
        contrast("SS: biased-minus-unbiased DMN slope", {"B:X": 1, "U:X": -1}),
        contrast("DS: biased-minus-unbiased DMN slope", {"B:X": 1, "B:X:D": 1, "U:X": -1, "U:X:D": -1}),
        contrast("Group moderation of biased DMN slope (DS minus SS)", {"B:X:D": 1}),
        contrast("Group moderation of unbiased DMN slope (DS minus SS)", {"U:X:D": 1}),
    ])
    eig = np.linalg.eigvalsh(np.asarray(fit.cov_re))
    diagnostics = pd.DataFrame([{
        "formula": formula, "likelihood": "Gaussian linear mixed model on log(RT_ms)",
        "estimation": "maximum likelihood; LBFGS", "converged": bool(fit.converged),
        "participants": data.subject.nunique(), "SS": int(data.loc[data.group.eq("SS"), "subject"].nunique()),
        "DS": int(data.loc[data.group.eq("DS"), "subject"].nunique()), "trials": len(data),
        "log_likelihood": fit.llf, "aic": fit.aic, "bic": fit.bic, "residual_variance": fit.scale,
        "random_effects": "unstructured participant covariance for intercept, biased indicator, unbiased indicator",
        "random_covariance_min_eigenvalue": float(eig.min()), "random_covariance_max_eigenvalue": float(eig.max()),
        "warnings": " | ".join(messages) if messages else "none",
    }])
    return data, fixed, contrasts, diagnostics


def write_formula_record() -> None:
    text = """# Formula record: Paper 2 behavioral revision analyses

## Coding used throughout

- `SS=0` denotes the same-dimension spatial-bias group; `DS=1` denotes the different-dimension spatial-bias group.
- `B_ij=1` for a current trial from the group-defined biased distractor dimension and 0 otherwise. `U_ij=1` analogously denotes the group-defined unbiased dimension. A distractor-absent trial has `B_ij=U_ij=0`.
- Positive capture is a distractor-present minus distractor-absent RT difference.
- `X_i` is mean within-DMN connectivity, centered within SS/DS; the trial-level model additionally scales this centered value by its across-participant SD.

## Frozen primary N=33 correlation

For participant `i` in group `g(i)`, with `Y_i` equal biased-dimension capture:

`Xc_i = X_i - mean_g(X)` and `Yc_i = Y_i - mean_g(Y)`.

The primary statistic is the ordinary Pearson correlation `cor(Xc_i, Yc_i)` over N=33. The bootstrap resamples participants separately within SS and DS, recomputes both group means and the correlation on every resample, and uses the 2.5th/97.5th percentiles. Leave-one-out recomputes the group-centered correlation after omitting one participant (N=32 each time).

## Unified training-plus-scanner learning model in the current N=35 behavioral source

The current executable source `code/shared/behavioral_data/current/01_rt_interference_spatial_learning_n35.py` fits its spatial-learning model to biased-dimension frequent/rare trials across training and scanner:

`RT_ij = beta0 + beta1*block_ij + beta2*rare_ij + beta3*DS_i + beta4*(rare_ij*DS_i) + b0_i + epsilon_ij`,

with `b0_i ~ N(0, tau^2)`, ML estimation, and BFGS. `rare=1` means rare location; frequent is the reference. The source's exact formula string is `RT_ms ~ block + rare_loc * ds_group` with a participant random intercept. It is an N=35 behavioral model, not an N=33 connectivity model.

## Existing two-stage history-adjusted capture formula

The active Paper-2 history script uses a separate OLS regression for each participant on *raw RT in ms*:

`RT_ij = a_i + bB_i*B_ij + bU_i*U_ij + bR_i*run2_ij + bT_i*z(log(global_trial_ij)) + bHB_i*z(HB_ij) + bHU_i*z(HU_ij) + e_ij`.

`HB_ij` and `HU_ij` are shifted (pre-current) cumulative counts of biased and unbiased distractor trials, respectively, calculated across the retained scanner rows for a participant and not reset at the run boundary. Continuous history covariates are z-scored within participant. The second stage correlates `bB_i` (or `bU_i`) with within-group-centered DMN connectivity. This package reruns that formula only on the current analysis-ready trial table and explicitly checks it against the frozen N=33 primary table.

## New hierarchical trial-level RT sensitivity

The executed accuracy-linked sensitivity uses correct scanner trials after first-run-trial and fixed 200--3000 ms RT exclusions:

`log(RT_ms_ij) = beta0 + betaB*B_ij + betaU*U_ij + betaX*X_i + betaD*DS_i`
` + betaBX*B_ij*X_i + betaUX*U_ij*X_i + betaBD*B_ij*DS_i + betaUD*U_ij*DS_i`
` + betaXD*X_i*DS_i + betaBXD*B_ij*X_i*DS_i + betaUXD*U_ij*X_i*DS_i`
` + betaR*run2_ij + betaT*z_i(log(trial order_ij)) + b0_i + bB_i*B_ij + bU_i*U_ij + e_ij`.

The random effects have an unstructured participant-level covariance. `betaBXD` is group moderation of the biased-condition DMN slope; `betaUXD` is the corresponding unbiased-condition moderation. This sensitivity preserves trial-level clustering and unequal cell counts, but it cannot identify spatial learning separately from differential exposure/training history.
"""
    (OUT / "FORMULAS.md").write_text(text, encoding="utf-8")


def figure2_note() -> None:
    text = f"""# Reviewed Figure 2 provenance and safe regeneration note

## What was verified from the reviewed artifact

- Reviewed artifact: `{PATHS['reviewed_figure2']}`.
- SHA-256: `{sha256(PATHS['reviewed_figure2'])}`.
- Panel B plots the group-defined biased-dimension location-learning difference across seven two-block epochs. Its visible y-axis says only `Location learning (ms)`; the required explicit definition is **Rare minus Frequent RT (ms)**. Positive values therefore mean slower RT at the rare than frequent region.
- The DS (orange) values are below zero at epoch 5 (`-27.44 ms`, 9--10) and epoch 7 (`-25.12 ms`, 13--14), as verified in the archived source summary table. Each is a descriptive two-block group mean with `n=18` and SEMs 14.58 and 13.18 ms, respectively. They should be described as later scanner-phase negative point estimates/variability, not explained mechanistically or treated as a reversal of the full-period group-by-location effect from this figure alone.
- Panel C's y-axis says only `Capture effect (ms)`; it should say **RT difference: distractor-present minus distractor-absent (ms)**. `Weighted` is the observed trial-frequency-weighted present-trial mean minus the same absent baseline; `Same` and `Diff` are the corresponding physical-dimension present-minus-absent scores.
- The Panel C legend (`Weighted`, `Same`, `Diff`) repeats the x-axis labels and can be removed without loss of encoding.

## Generator provenance

The closest code match found is `{PATHS['reviewed_figure2_generator_transcript']}`. Its header says it is a transcribed copy of `code/figures_20260602_paper2_high_quality.py`; `make_fig1()` makes the same A/B/C layout, writes `behavior_biased_learning_by_epoch_*.csv`, includes the Panel C legend, and sets the same generic B/C y-axis labels. The extant archived source tables are `{ROOT / 'manuscript/paper2/code_and_data' / 'behavior_biased_learning_by_epoch_subject.csv'}`, `{ROOT / 'manuscript/paper2/code_and_data' / 'behavior_biased_learning_by_epoch_summary.csv'}`, and `{ROOT / 'manuscript/paper2/code_and_data' / 'behavior_biased_learning_by_epoch_tests.csv'}` for Panel B, and `{ROOT / 'manuscript/paper2/code_and_data/behavior_capture_effects.csv'}` for Panel C. It is **not listed in the active Paper 2 manifest** and its `make_fig1()` writes into `manuscript/paper2/figures/` and `manuscript/paper2/code_and_data/`. Do not execute it in place.

The currently organized N=35 generator `{PATHS['current_figure2v_generator']}` is non-destructive in intent but produces a later, different Figure-2v layout and writes into Paper 1 paths. It is not an exact regeneration route for the reviewed Figure 2.

## Safe regeneration path

1. Copy the closest-match generator into a new revision-owned file under this directory (do not alter the archival/transcribed source).
2. Redirect all `FIG_DIR` and `DATA_DIR` writes to a new timestamped subdirectory under `{OUT}`.
3. Change only the reviewed figure labels/legend: B y-axis to `RT difference: Rare - Frequent (ms)`; C y-axis to `RT difference: distractor-present - distractor-absent (ms)`; remove the Panel C legend.
4. Preserve the source inputs and write the subject/epoch summary tables beside the PNG/PDF. Before using any regenerated figure, reconcile its source table to the N35 unified model and the review baseline; the current frozen Paper-2 behavioral source and later N35 source are not assumed equivalent here.

No manuscript image was modified by this audit.
"""
    (OUT / "FIGURE2_PROVENANCE_AND_SAFE_REGENERATION.md").write_text(text, encoding="utf-8")


def report(manifest: pd.DataFrame, flow_total: pd.DataFrame, frozen_corr: pd.DataFrame,
           boot: pd.DataFrame, loo: pd.DataFrame, current_assoc: pd.DataFrame,
           discrepancy: pd.DataFrame, moderation: pd.DataFrame, lmm_diag: pd.DataFrame,
           lmm_contrasts: pd.DataFrame) -> None:
    focal = frozen_corr.query("scope == 'pooled_group_centered' and outcome == 'biased_capture'").iloc[0]
    focal_boot = boot.iloc[0]
    focal_loo = loo
    current_raw = current_assoc.query("model == 'raw_mean_difference' and outcome == 'biased_capture'").iloc[0]
    history = current_assoc.query("model == 'history_adjusted_subject_OLS' and outcome == 'biased_capture'").iloc[0]
    mod = moderation.query("outcome == 'biased_capture'").iloc[0]
    bias_mod = lmm_contrasts.query("contrast == 'Group moderation of biased DMN slope (DS minus SS)'").iloc[0]
    pref_ss = lmm_contrasts.query("contrast == 'SS: biased-minus-unbiased DMN slope'").iloc[0]
    pref_ds = lmm_contrasts.query("contrast == 'DS: biased-minus-unbiased DMN slope'").iloc[0]
    scanner = flow_total.query("phase == 'scanner'")
    lines = [
        "# Paper 2 behavioral revision evidence package",
        "",
        "## Scope and status",
        "",
        "This package is an independent, bounded audit. It did not edit the manuscript, revision plan, original DOCX/Markdown, or source data. It uses the frozen Paper-2 N=33 analytic table to reconstruct the primary association, and separately uses current trial tables for flow/accuracy and trial-level sensitivities. These sources are not silently substituted for one another.",
        "",
        "## Verified sample identity and trial flow",
        "",
        f"- Scanner behavioral source: N=35 (SS=17, DS=18), {int(scanner.assigned_trials.sum()):,} assigned trials. The N=33 connectivity source contains SS=16 and DS=17; the two N35 participants absent from it are `{', '.join(sorted(set(pd.read_csv(PATHS['scanner_analysis_ready']).subject) - set(pd.read_csv(PATHS['current_n33_connectivity']).subject)))}`.",
        f"- Accuracy-linked scanner source has {int(scanner.incorrect_trials.sum()):,} incorrect trials ({pct(scanner.incorrect_trials.sum(), scanner.assigned_trials.sum()):.1f}%); exact condition/location cells, sequential exclusions, retained counts, and RT summaries are in `results/trial_flow_accuracy_rt_by_group_phase_condition.csv`.",
        "- The current analysis-ready scanner table retains 12,798/12,880 rows after the first-in-run plus fixed RT-window rule. That historical Paper-2 rule does not apply correctness; the accuracy-linked alternative retained-correct table is reported separately rather than redefined as the original analysis sample.",
        "",
        "## Reconstructed frozen N=33 primary association",
        "",
        f"- Group-centered Pearson correlation of within-DMN connectivity with biased-dimension capture: r={focal.r:+.3f}, p {p_text(focal.p)}, N={int(focal.n)}, Fisher 95% CI [{focal.ci_low:+.3f}, {focal.ci_high:+.3f}].",
        f"- Group-stratified participant bootstrap (seed {SEED}, {N_BOOT:,} resamples; percentile interval): median r={focal_boot.median_r:+.3f}, 95% CI [{focal_boot.ci_low:+.3f}, {focal_boot.ci_high:+.3f}].",
        f"- Leave-one-out (each N=32) range: r [{focal_loo.r.min():+.3f}, {focal_loo.r.max():+.3f}], p [{focal_loo.p.min():.4f}, {focal_loo.p.max():.4f}]. Full results are saved, including all dropped IDs.",
        "- Group-centered associations for biased, unbiased, all-trials, and physical-dimension capture plus within-group correlations are in `results/frozen_n33_correlation_reconstruction.csv`.",
        "",
        "## Current trial-table history sensitivity and provenance discrepancy",
        "",
        f"- Recomputing raw capture from the current analysis-ready trial CSV gives r={current_raw.r:+.3f}, p {p_text(current_raw.p)}; the documented participantwise history OLS gives r={history.r:+.3f}, p {p_text(history.p)} for biased capture. Its exact formula is in `FORMULAS.md`.",
        f"- This is not a direct replication of the frozen primary table: the current-versus-frozen biased-capture maximum absolute difference is {discrepancy.max_abs_difference_ms.iloc[0]:.3f} ms across the shared N=33 IDs. The frozen primary was therefore not overwritten or relabeled as a current-trial reconstruction.",
        "",
        "## Moderation and hierarchical trial-level RT sensitivity",
        "",
        f"- Frozen participant-level biased-capture moderation (`capture ~ DMN_z * group`): interaction b={mod.estimate:+.3f} ms, 95% CI [{mod.ci_low:+.3f}, {mod.ci_high:+.3f}], t({mod.df_resid:.0f})={mod.t:+.3f}, p {p_text(mod.p)}.",
        f"- Accuracy-linked N33 log-RT mixed model converged={bool(lmm_diag.converged.iloc[0])}; n={int(lmm_diag.trials.iloc[0]):,} correct, RT-valid, non-first scanner trials; random effects were participant intercept plus biased/unbiased condition slopes. The biased DMN-slope group moderation estimate was {bias_mod.estimate_log_rt:+.4f} log-RT per 1-SD DMN (95% CI [{bias_mod.ci_low:+.4f}, {bias_mod.ci_high:+.4f}], p {p_text(bias_mod.p)}).",
        f"- Within-group biased-minus-unbiased DMN-slope contrasts: SS={pref_ss.estimate_log_rt:+.4f}, p {p_text(pref_ss.p)}; DS={pref_ds.estimate_log_rt:+.4f}, p {p_text(pref_ds.p)}. These are sensitivity estimates, not a replacement for the frozen correlation or proof that learning is separable from exposure.",
        "",
        "## Figure 2 reviewer request",
        "",
        "The audit verified the reviewed A/B/C figure structure, the redundant Panel-C legend, B/C difference-score definitions, DS negative epoch-5/7 values, closest generator provenance, and a safe non-overwriting regeneration procedure in `FIGURE2_PROVENANCE_AND_SAFE_REGENERATION.md`.",
        "",
        "## Limitations/blockers",
        "",
        "1. The frozen N=33 primary analytic table and current analysis-ready scanner table do not yield identical participant capture values. Without a provenance record identifying the exact historical behavioral source for `dmn_capture_unified.csv`, a full raw-trial reconstruction of the frozen primary statistic is blocked.",
        "2. The history model controls measured scanner-run position/exposure only. It neither resets history at run boundaries nor identifies pre-scanner learning separately from exposure; those properties are documented rather than treated as causal isolation.",
        "3. The trial-level model uses correct trials to make RT inference defensible, while the historic Paper-2 analysis-ready capture table does not encode correctness. It is therefore a complementary sensitivity analysis with a different inclusion rule.",
        "4. The reviewed Figure-2 generator source is a transcribed archival copy outside the active manifest and writes into manuscript paths if run unchanged. No exact active, non-destructive generator for the reviewed artifact was established.",
        "",
        "## Reproduction",
        "",
        f"Run: `python {OUT / 'run_behavioral_audit.py'}`",
        "All inputs, hashes, versions, formulas, results, random seed, and output paths are documented in this directory.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = input_manifest()
    missing = manifest.loc[~manifest.exists, "absolute_path"].tolist()
    if missing:
        raise FileNotFoundError("Required inputs missing: " + "; ".join(missing))
    manifest.to_csv(RESULTS / "input_manifest.csv", index=False)
    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
        "seed": SEED, "bootstrap_iterations": N_BOOT,
    }
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    training = load_training_raw()
    scanner = load_scanner_raw()
    flow, flow_total = trial_flow(training, scanner)
    flow.to_csv(RESULTS / "trial_flow_accuracy_rt_by_group_phase_condition.csv", index=False)
    flow_total.to_csv(RESULTS / "trial_flow_accuracy_rt_phase_group_totals.csv", index=False)

    frozen, frozen_corr, boot_dist, boot_sum, loo = frozen_primary()
    frozen.to_csv(RESULTS / "frozen_n33_primary_analytic_table_with_derived_unbiased.csv", index=False)
    frozen_corr.to_csv(RESULTS / "frozen_n33_correlation_reconstruction.csv", index=False)
    boot_dist.to_csv(RESULTS / "frozen_n33_biased_capture_bootstrap_distribution.csv", index=False)
    boot_sum.to_csv(RESULTS / "frozen_n33_biased_capture_bootstrap_summary.csv", index=False)
    loo.to_csv(RESULTS / "frozen_n33_biased_capture_leave_one_out.csv", index=False)

    current_trial, history_est, history_diag, history_assoc = current_trial_capture_and_history()
    history_est.to_csv(RESULTS / "current_trial_history_subject_estimates.csv", index=False)
    history_diag.to_csv(RESULTS / "current_trial_history_subject_OLS_diagnostics.csv", index=False)
    history_assoc.to_csv(RESULTS / "current_trial_history_associations.csv", index=False)
    current_trial[["subject", "group_short", "run", "trial", "RT_ms", "distractor_condition", "cond_biased", "cond_unbiased", "prior_biased_n", "prior_unbiased_n"]].to_csv(
        RESULTS / "current_trial_history_covariates.csv", index=False
    )

    current_raw = history_est.query("model == 'raw_mean_difference'")[["subject", "biased_capture"]].rename(columns={"biased_capture": "current_raw_biased_capture"})
    discrepancy = frozen[["subject", "biased_capture"]].merge(current_raw, on="subject", validate="one_to_one")
    discrepancy["difference_ms_current_minus_frozen"] = discrepancy.current_raw_biased_capture - discrepancy.biased_capture
    discrepancy["absolute_difference_ms"] = discrepancy.difference_ms_current_minus_frozen.abs()
    discrepancy.to_csv(RESULTS / "frozen_vs_current_raw_capture_discrepancy.csv", index=False)
    discrepancy_summary = pd.DataFrame([{
        "shared_subjects": len(discrepancy), "max_abs_difference_ms": discrepancy.absolute_difference_ms.max(),
        "mean_difference_ms": discrepancy.difference_ms_current_minus_frozen.mean(),
        "rmse_ms": float(np.sqrt(np.mean(discrepancy.difference_ms_current_minus_frozen ** 2))),
    }])
    discrepancy_summary.to_csv(RESULTS / "frozen_vs_current_raw_capture_discrepancy_summary.csv", index=False)

    moderation = frozen_group_moderation(frozen)
    moderation.to_csv(RESULTS / "frozen_n33_group_moderation_models.csv", index=False)
    lmm_trials, lmm_fixed, lmm_contrasts, lmm_diag = trial_level_mixed_model()
    lmm_trials[["subject", "group", "run", "trial", "RT", "correct", "category", "B", "U", "D", "X", "log_trial", "log_rt"]].to_csv(
        RESULTS / "hierarchical_rt_model_trial_table.csv", index=False
    )
    lmm_fixed.to_csv(RESULTS / "hierarchical_rt_model_fixed_effects.csv", index=False)
    lmm_contrasts.to_csv(RESULTS / "hierarchical_rt_model_planned_contrasts.csv", index=False)
    lmm_diag.to_csv(RESULTS / "hierarchical_rt_model_diagnostics.csv", index=False)

    write_formula_record()
    figure2_note()
    report(manifest, flow_total, frozen_corr, boot_sum, loo, history_assoc, discrepancy_summary,
           moderation, lmm_diag, lmm_contrasts)
    print(f"Wrote behavioral evidence package to {OUT}")
    print((OUT / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
