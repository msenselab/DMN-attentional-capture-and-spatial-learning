#!/usr/bin/env python3
"""Independent, read-only audit of current N33 reliability and DS color capture.

Reconstructs behavioral scores from the frozen correct-RT trial file, recomputes
Default91/common399 associations from saved inputs, and writes only beside this
script.  It does not refit neuroimaging or mixed models.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
R1 = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2"
ATTACHED = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt"
RESULTS = R1 / "resolved_reanalysis/inference/results_common399_n33_optimizer_resolved_v2"
EXTRACT = R1 / "resolved_reanalysis/extraction_common_coverage"
READY = R1 / "resolved_reanalysis/common399_analysis_ready"
V5 = R1 / "resolved_reanalysis/editorial/final_sources_v5_equations"
OLD_UNCENTERED = ATTACHED / "uncentered-odd-even-comparison"

TRIALS = ATTACHED / "brain_behavior/correct_rt_trials_n33.csv"
SAVED_HALVES = ATTACHED / "brain_behavior/split_half_scores.csv"
SAVED_CENTERED = ATTACHED / "brain_behavior/capture_reliability.csv"
OLD_COMPARISON = OLD_UNCENTERED / "pooled_odd_even_comparison.csv"
SCORES = EXTRACT / "participant_scores_resolved.csv"
MATRICES = EXTRACT / "primary_participant_matrices.npy"
MATRIX_SUBJECTS = EXTRACT / "primary_matrix_subjects.csv"
PARCELS = READY / "physical_yeo7_common399_parcels.csv"
INPUT_MANIFEST = READY / "analysis_ready_input_manifest.json"
RESULTS_MANIFEST = RESULTS / "results_manifest.json"
ASSOCIATIONS = RESULTS / "participant_associations.csv"
FAMILY_PLAN = R1 / "resolved_reanalysis/inference/CORRECTION_FAMILY_PLAN.md"
V5_LEDGER = V5 / "numerical_ledger.json"
V5_TABLE2 = V5 / "main_tables.md"

R = 10_000
# Fixed before computation; same resampled participant identities are used for
# uncentered and centered estimates within every score.
SEED = 20261003
TOL = 2e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tol=TOL) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0, equal_nan=True))


def require(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "pass": bool(condition)})
    if not condition:
        raise AssertionError(message)


def pearson_stats(x: np.ndarray, y: np.ndarray, df: int, fisher_se: float) -> dict:
    r = float(stats.pearsonr(x, y).statistic)
    t = r * math.sqrt(df / (1.0 - r * r))
    p = float(2.0 * stats.t.sf(abs(t), df))
    z = math.atanh(r)
    critical = float(stats.norm.ppf(0.975))
    lo, hi = np.tanh([z - critical * fisher_se, z + critical * fisher_se])
    return {"N": int(len(x)), "r": r, "t": t, "df": int(df), "p_raw": p,
            "ci_low": float(lo), "ci_high": float(hi), "fisher_se": fisher_se}


def residualize(y: np.ndarray, covariates: list[np.ndarray]) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), *covariates])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def centered(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    z = x.astype(float).copy()
    for group in ("SS", "DS"):
        ii = np.flatnonzero(groups == group)
        z[ii] -= z[ii].mean()
    return z


def holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(values) - np.arange(len(values))) * ordered))
    answer = np.empty_like(adjusted)
    answer[order] = adjusted
    return answer


def make_behavior(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full scores and parity half scores, each half with its own absent baseline."""
    means = trials.groupby(["subject", "group", "category"], observed=True).RT_ms.mean().unstack("category")
    full = means.reset_index()
    full["biased_capture"] = full["biased"] - full["absent"]
    full["unbiased_capture"] = full["unbiased"] - full["absent"]
    counts = trials.groupby(["subject", "group", "category"], observed=True).size().unstack("category").reset_index()
    full = full.merge(counts, on=["subject", "group"], validate="one_to_one", suffixes=("", "_n"))
    full["biased_weight"] = full["biased_n"] / (full["biased_n"] + full["unbiased_n"])
    full["unbiased_weight"] = 1.0 - full["biased_weight"]
    full["all_capture"] = full["biased_weight"] * full["biased_capture"] + full["unbiased_weight"] * full["unbiased_capture"]
    prior_biased = trials.loc[trials.category.eq("biased") & trials.prev_category.notna()].copy()
    priming = prior_biased.groupby(["subject", "group"], observed=True).apply(
        lambda z: z.loc[z.prev_category.eq("biased"), "RT_ms"].mean() -
                  z.loc[~z.prev_category.eq("biased"), "RT_ms"].mean(),
        include_groups=False).rename("biased_priming").reset_index()
    full = full.merge(priming, on=["subject", "group"], validate="one_to_one")

    d = trials.copy()
    d["half"] = np.where(d.global_order.mod(2).eq(1), "odd", "even")
    means_half = d.groupby(["subject", "group", "half", "category"], observed=True).RT_ms.agg(["mean", "size"])
    rows = []
    for (subject, group, half), z in means_half.groupby(level=[0, 1, 2], observed=True):
        z = z.droplevel([0, 1, 2])
        if set(z.index) != {"absent", "biased", "unbiased"}:
            raise AssertionError(f"incomplete half categories: {subject} {half}")
        for score in ("biased", "unbiased", "physical_color"):
            category = score
            if score == "physical_color":
                # In DS, the group-biased dimension is color; in SS, physical color is unbiased.
                category = "biased" if group == "DS" else "unbiased"
            rows.append({"subject": subject, "group": group, "half": half, "score": score,
                         "capture": float(z.loc[category, "mean"] - z.loc["absent", "mean"]),
                         "n_present": int(z.loc[category, "size"]), "n_absent": int(z.loc["absent", "size"])})
    half_long = pd.DataFrame(rows).sort_values(["subject", "group", "score", "half"]).reset_index(drop=True)
    return full.sort_values("subject").reset_index(drop=True), half_long


def association_family(neural: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    s = neural[["subject", "group", "dmn_conn_primary"]].merge(
        behavior[["subject", "group", "biased_capture", "unbiased_capture", "all_capture", "biased_priming"]],
        on=["subject", "group"], validate="one_to_one").sort_values("subject").reset_index(drop=True)
    s["D"] = s.group.eq("DS").astype(float)
    x = s.dmn_conn_primary.to_numpy(float)
    g = s.group.to_numpy()
    records = []

    def add(scope, outcome, analysis, xx, yy, df, se):
        z = pearson_stats(xx, yy, df, se)
        records.append({"scope": scope, "outcome": outcome, "analysis": analysis, **z})

    for outcome in ("biased_capture", "unbiased_capture", "all_capture"):
        add("pooled", outcome, "group-centered Pearson df=N-3", centered(x, g), centered(s[outcome].to_numpy(float), g), 30, 1 / math.sqrt(29))
    add("pooled", "biased_capture", "partial unbiased+group",
        residualize(x, [s.unbiased_capture.to_numpy(float), s.D.to_numpy(float)]),
        residualize(s.biased_capture.to_numpy(float), [s.unbiased_capture.to_numpy(float), s.D.to_numpy(float)]), 29, 1 / math.sqrt(28))
    add("pooled", "biased_capture", "partial measured priming+group",
        residualize(x, [s.biased_priming.to_numpy(float), s.D.to_numpy(float)]),
        residualize(s.biased_capture.to_numpy(float), [s.biased_priming.to_numpy(float), s.D.to_numpy(float)]), 29, 1 / math.sqrt(28))
    for group in ("SS", "DS"):
        q = s.loc[s.group.eq(group)]
        add(group, "biased_capture", "within-group Pearson", q.dmn_conn_primary.to_numpy(float), q.biased_capture.to_numpy(float), len(q) - 2, 1 / math.sqrt(len(q) - 3))
        add(group, "biased_capture", "within-group partial unbiased",
            residualize(q.dmn_conn_primary.to_numpy(float), [q.unbiased_capture.to_numpy(float)]),
            residualize(q.biased_capture.to_numpy(float), [q.unbiased_capture.to_numpy(float)]), len(q) - 3, 1 / math.sqrt(len(q) - 4))
    return pd.DataFrame(records), s


def bootstrap_reliability(half_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, draws, cov_rows = [], [], []
    for score_index, score in enumerate(("biased", "unbiased", "physical_color")):
        z = half_long.loc[half_long.score.eq(score)].pivot(index=["subject", "group"], columns="half", values="capture").reset_index()
        z = z.sort_values(["group", "subject"]).reset_index(drop=True)
        groups = z.group.to_numpy()
        x, y = z.odd.to_numpy(float), z.even.to_numpy(float)
        n = len(z)
        require(n == 33 and (groups == "SS").sum() == 16 and (groups == "DS").sum() == 17, f"{score}: N33 SS16 DS17", CHECKS)

        cx, cy = centered(x, groups), centered(y, groups)
        total_cross = float(((x - x.mean()) * (y - y.mean())).sum())
        within_cross = float((cx * cy).sum())
        cov_rows.append({"score": score, "total_crossproduct": total_cross,
                         "within_group_crossproduct": within_cross,
                         "between_group_crossproduct": total_cross - within_cross,
                         "between_group_covariance_share": (total_cross - within_cross) / total_cross})

        rng = np.random.default_rng(SEED + score_index)
        ss = np.flatnonzero(groups == "SS")
        ds = np.flatnonzero(groups == "DS")
        indices = np.column_stack([rng.choice(ss, size=(R, len(ss)), replace=True), rng.choice(ds, size=(R, len(ds)), replace=True)])
        positions = (np.arange(len(ss)), np.arange(len(ss), n))
        for centering in ("uncentered", "group_centered"):
            xx, yy = x[indices].copy(), y[indices].copy()
            is_centered = centering == "group_centered"
            if is_centered:
                for where in positions:
                    xx[:, where] -= xx[:, where].mean(axis=1, keepdims=True)
                    yy[:, where] -= yy[:, where].mean(axis=1, keepdims=True)
                px, py, df = cx, cy, 30
            else:
                px, py, df = x, y, 31
            xx -= xx.mean(axis=1, keepdims=True)
            yy -= yy.mean(axis=1, keepdims=True)
            rb = (xx * yy).sum(axis=1) / np.sqrt((xx * xx).sum(axis=1) * (yy * yy).sum(axis=1))
            sb = 2 * rb / (1 + rb)
            require(np.isfinite(rb).all() and np.isfinite(sb).all(), f"{score} {centering}: finite 10k bootstrap", CHECKS)
            point = pearson_stats(px, py, df, 1 / math.sqrt(n - 2 - int(is_centered)))
            rlo, rhi = np.quantile(rb, [0.025, 0.975])
            sblo, sbhi = np.quantile(sb, [0.025, 0.975])
            summary.append({"score": score, "centering": centering, **point,
                            "r_percentile95_low": float(rlo), "r_percentile95_high": float(rhi),
                            "SB": float(2 * point["r"] / (1 + point["r"])),
                            "SB_percentile95_low": float(sblo), "SB_percentile95_high": float(sbhi),
                            "bootstrap_R": R, "seed": SEED + score_index,
                            "bootstrap_design": "paired participant resamples, stratified SS/DS; group means recalculated in centered resamples"})
            draws.append(pd.DataFrame({"score": score, "centering": centering, "iteration": np.arange(1, R + 1), "r": rb, "SB": sb}))
    return pd.DataFrame(summary), pd.concat(draws, ignore_index=True), pd.DataFrame(cov_rows)


CHECKS: list[dict] = []


def main() -> None:
    required_paths = [TRIALS, SAVED_HALVES, SAVED_CENTERED, OLD_COMPARISON, SCORES, MATRICES, MATRIX_SUBJECTS,
                      PARCELS, INPUT_MANIFEST, RESULTS_MANIFEST, ASSOCIATIONS, FAMILY_PLAN, V5_LEDGER, V5_TABLE2]
    for path in required_paths:
        require(path.is_file(), f"source exists: {path}", CHECKS)

    result_manifest = json.loads(RESULTS_MANIFEST.read_text())
    input_manifest = json.loads(INPUT_MANIFEST.read_text())
    require(result_manifest["behavior_trials_sha256"] == sha256(TRIALS), "results manifest matches frozen behavior trial hash", CHECKS)
    require(result_manifest["input_hashes"][str(SCORES)] == sha256(SCORES), "results manifest matches participant-score hash", CHECKS)
    require(result_manifest["input_hashes"][str(MATRICES)] == sha256(MATRICES), "results manifest matches primary-matrix hash", CHECKS)
    require(result_manifest["input_hashes"][str(MATRIX_SUBJECTS)] == sha256(MATRIX_SUBJECTS), "results manifest matches matrix-subject hash", CHECKS)
    require(input_manifest["common_coverage_geometry"]["effective_common_coverage_parcels"] == 399 and
            input_manifest["common_coverage_geometry"]["excluded_native_labels"] == [323], "common399 geometry and native323 omission", CHECKS)
    require(input_manifest["participant_scores"]["primary_column"] == "dmn_conn_primary", "Default91 primary column binding", CHECKS)

    trials = pd.read_csv(TRIALS)
    require(len(trials) == 10995 and trials.subject.nunique() == 33, "frozen behavioral trial universe: 10,995 rows/N33", CHECKS)
    require(trials.correct.eq(1).all() and trials.RT_ms.between(200, 3000).all(), "correct and RT-valid gate in every frozen row", CHECKS)
    require(not trials.duplicated(["subject", "run", "trial"]).any(), "unique subject/run/trial keys", CHECKS)
    require(trials.groupby("group").subject.nunique().to_dict() == {"DS": 17, "SS": 16}, "N33 group counts DS17 SS16", CHECKS)
    behavior, half_long = make_behavior(trials)
    half_wide = half_long.pivot(index=["subject", "group", "score"], columns="half", values="capture").reset_index().sort_values(["subject", "group", "score"]).reset_index(drop=True)
    saved_halves = pd.read_csv(SAVED_HALVES).sort_values(["subject", "group", "score"]).reset_index(drop=True)
    require(len(half_wide) == 99 and close(half_wide[["odd", "even"]], saved_halves[["odd", "even"]]), "trial-reconstructed 99 half scores equal saved current half-score vector", CHECKS)

    neural = pd.read_csv(SCORES)
    matrix_subjects = pd.read_csv(MATRIX_SUBJECTS)
    parcels = pd.read_csv(PARCELS).sort_values("matrix_index").reset_index(drop=True)
    matrices = np.load(MATRICES, allow_pickle=False)
    require(matrices.shape == (33, 399, 399), "saved common399 matrix shape", CHECKS)
    default_ix = np.flatnonzero(parcels.yeo7_network.to_numpy() == "Default")
    require(len(default_ix) == 91 and 323 not in set(parcels.native_label), "physical Yeo7 Default91 membership / native323 absent", CHECKS)
    subject_pos = {subject: i for i, subject in enumerate(matrix_subjects.subject)}
    matrix_in_neural_order = matrices[[subject_pos[s] for s in neural.subject]]
    sub = matrix_in_neural_order[:, default_ix][:, :, default_ix]
    upper = np.triu_indices(91, 1)
    direct_default = sub[:, upper[0], upper[1]].mean(axis=1)
    require(close(direct_default, neural.dmn_conn_primary.to_numpy(float), tol=2e-12), "Default91 score directly reproduced from saved common399 matrices", CHECKS)

    assoc_recomputed, joined = association_family(neural, behavior)
    saved_assoc = pd.read_csv(ASSOCIATIONS)
    require(len(saved_assoc) == 9 and len(assoc_recomputed) == 9, "association row inventory (primary + 8 declared secondary)", CHECKS)
    # Match eight rows that do not require the measured-priming variable.
    keys = ["scope", "outcome", "analysis"]
    compare = assoc_recomputed.merge(saved_assoc, on=keys, suffixes=("_fresh", "_saved"), validate="one_to_one")
    for column in ("N", "r", "t", "df", "p_raw", "ci_low", "ci_high", "fisher_se"):
        require(close(compare[f"{column}_fresh"], compare[f"{column}_saved"], tol=2e-8), f"association recomputation agrees for {column}", CHECKS)
    # Use the author-declared, existing eight-member family without creating any new correction family.
    primary_mask = (saved_assoc.scope.eq("pooled") & saved_assoc.outcome.eq("biased_capture") &
                    saved_assoc.analysis.eq("group-centered Pearson df=N-3"))
    secondary = saved_assoc.loc[~primary_mask].copy()
    require(len(secondary) == 8 and secondary.Holm_family.nunique() == 1 and secondary.Holm_family.iloc[0].startswith("8 prespecified"),
            "original declared secondary correction family is Holm-8", CHECKS)
    require(close(holm(secondary.p_raw.to_numpy(float)), secondary.p_Holm.to_numpy(float), tol=2e-12), "Holm-8 adjusted p values independently reproduce", CHECKS)
    ds = saved_assoc.loc[(saved_assoc.scope.eq("DS")) & (saved_assoc.outcome.eq("biased_capture")) &
                         (saved_assoc.analysis.eq("within-group Pearson"))].copy()
    require(len(ds) == 1, "one current DS biased/color capture association row", CHECKS)
    # DS's biased dimension is color, so this is the requested DS-color capture association.
    ds_row = ds.iloc[0].to_dict()
    require(close([ds_row["r"], ds_row["p_raw"], ds_row["p_Holm"]], [0.4169398427019184, 0.09591694721715055, 0.1918338944343011], tol=2e-12),
            "current DS color-capture values are not obsolete .601/.054 values", CHECKS)

    reliability, draws, covariance = bootstrap_reliability(half_long)
    saved_centered = pd.read_csv(SAVED_CENTERED)
    centered_new = reliability.loc[reliability.centering.eq("group_centered")].sort_values("score")
    centered_saved = saved_centered.sort_values("score")
    require(close(centered_new.r, centered_saved.r, tol=2e-12), "fresh centered parity r values reproduce archived current values", CHECKS)
    old = pd.read_csv(OLD_COMPARISON)
    old_un = old.loc[old.centering.eq("uncentered") & old.score.isin(["biased", "unbiased"])].sort_values("score")
    fresh_un = reliability.loc[reliability.centering.eq("uncentered") & reliability.score.isin(["biased", "unbiased"])].sort_values("score")
    require(close(fresh_un.r, old_un.r, tol=2e-12), "fresh uncentered biased/unbiased r values reproduce prior .699/.776 on exact rows", CHECKS)
    require(len(draws) == 60_000 and draws.groupby(["score", "centering"]).size().eq(R).all(), "six paired 10,000-draw bootstrap cells", CHECKS)

    half_long.to_csv(OUT / "reconstructed_half_scores_counts.csv", index=False)
    reliability.to_csv(OUT / "reliability_comparison_10000.csv", index=False)
    draws.to_csv(OUT / "reliability_bootstrap_draws_10000.csv", index=False)
    covariance.to_csv(OUT / "covariance_decomposition.csv", index=False)
    assoc_recomputed.to_csv(OUT / "association_family_recomputed.csv", index=False)
    pd.DataFrame([{
        "scope": "DS", "behavioral_measure": "physical color capture (= biased_capture in DS)",
        "neural_measure": "physical Yeo7 Default91 mean upper-triangle Fisher-z, common399",
        "N": int(ds_row["N"]), "r": ds_row["r"], "t": ds_row["t"], "df": int(ds_row["df"]),
        "p_raw": ds_row["p_raw"], "CI_method": ds_row["CI_method"], "ci_low": ds_row["ci_low"], "ci_high": ds_row["ci_high"],
        "p_Holm8_original_declared_secondary_family": ds_row["p_Holm"], "Holm_family": ds_row["Holm_family"],
        "source_association_csv": str(ASSOCIATIONS), "source_association_row_1based": int(ds.index[0]) + 2
    }]).to_csv(OUT / "ds_default91_color_capture_association.csv", index=False)

    source_hashes = {str(path): sha256(path) for path in required_paths}
    manifest = {
        "audit_scope": "read-only saved-data audit; no neuroimaging/model refits; no manuscript sources edited",
        "analysis_contract": {"N": 33, "groups": {"DS": 17, "SS": 16}, "trials": 10995,
                              "coverage": "common399, native label 323 uniformly omitted", "primary": "physical Yeo7 Default91"},
        "sources_sha256": source_hashes,
        "bootstrap": {"R": R, "seed_base": SEED, "design": "paired stratified participant percentile; same draw identities across uncentered/centered per score; group means recalculated for centered resamples"},
        "DS_color_capture": {"r": ds_row["r"], "p_raw": ds_row["p_raw"], "p_Holm8": ds_row["p_Holm"], "CI": [ds_row["ci_low"], ds_row["ci_high"]]},
        "outputs": ["reconstructed_half_scores_counts.csv", "reliability_comparison_10000.csv", "reliability_bootstrap_draws_10000.csv", "covariance_decomposition.csv", "association_family_recomputed.csv", "ds_default91_color_capture_association.csv"],
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "verification.json").write_text(json.dumps({"all_checks_passed": True, "n_checks": len(CHECKS), "checks": CHECKS}, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "n_checks": len(CHECKS), "ds_r": ds_row["r"], "ds_p_raw": ds_row["p_raw"], "ds_p_Holm8": ds_row["p_Holm"], "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
