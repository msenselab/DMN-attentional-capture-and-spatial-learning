#!/usr/bin/env python3
"""All-subject behavioral RT analysis for Paper 1 (N=35).

This is the reader-facing, self-contained analysis used for the Behavioral
Distractor Interference and Spatial-Probability Learning Results. It loads the
training and scanner data, applies the final RT preprocessing, and reproduces:

1. the unified training-plus-scanner learning trajectory;
2. same- and different-dimension distractor interference;
3. biased-dimension spatial-probability learning.

Behavioral data are not restricted by imaging QC. The union sample is N=35
(SS=17, DS=18); 31 participants have training data and all 35 have scanner data.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
from scipy.stats import norm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[4])))
TRAIN_DIR = ROOT / "code" / "cross_dimension" / "data"
OUT = Path(__file__).resolve().parent / "results" / "rt_interference_spatial_n35"
OUT.mkdir(parents=True, exist_ok=True)

RT_MIN, RT_MAX = 0.2, 3.0
TRIALS_PER_BLOCK = 46
N_TRAIN_BLOCKS = 6
TRAIN_BLOCK_MAP = {1: 1, 2: 2, 5: 3, 6: 4, 7: 5, 8: 6}
TRAIN_COLS = [
    "tar_ori", "conditions", "sessions", "dist_cond", "blk", "tar_pos",
    "as_pos", "response_group", "tar_orient_group", "c_orient", "key",
    "correct", "tm_cue", "tm_disp", "tm_key", "rt", "tm_end", "tm_trigger",
]


def load_id_map() -> dict[str, str]:
    """Map original training IDs onto BIDS IDs from participants.tsv."""
    mapping: dict[str, str] = {}
    with (ROOT / "participants.tsv").open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0].startswith("participant_id"):
                continue
            bids_id = parts[0].replace("sub-", "").strip()
            origin = parts[4].strip()
            if origin and "_" in origin:
                mapping[origin.split("_")[0]] = bids_id
    return mapping


def load_training(id_map: dict[str, str]) -> pd.DataFrame:
    """Load and preprocess the six training blocks."""
    rows: list[pd.DataFrame] = []
    valid_sessions = {"s11", "s12", "s21", "s22"}
    for path_string in sorted(glob.glob(str(TRAIN_DIR / "*.mat"))):
        path = Path(path_string)
        parts = path.stem.split("_", 1)
        if len(parts) != 2 or parts[1] not in valid_sessions:
            continue
        original_id, _session = parts
        mat = loadmat(path)
        if "trials" not in mat:
            continue
        raw = pd.DataFrame(mat["trials"], columns=TRAIN_COLS)
        raw["group"] = raw["sessions"].map({2.0: "SS", 1.0: "DS"}).fillna(
            "SS" if original_id.startswith("SS") else "DS"
        )
        group = str(raw["group"].iloc[0])

        def dimension(condition: float) -> str:
            if condition == 0:
                return "Absent"
            if condition == 2:
                return "Same" if group == "SS" else "Different"
            return "Different" if group == "SS" else "Same"

        raw["Dimension"] = raw["conditions"].map(dimension)

        # Infer each participant's frequent positions from biased-dimension trials.
        biased = raw.loc[raw["conditions"] == 2]
        if len(biased):
            counts = biased["as_pos"].value_counts()
            frequent_positions = set(counts[counts > counts.median()].index.astype(int))
        else:
            frequent_positions = set()

        def region(row: pd.Series) -> str:
            if row["conditions"] == 0 or row["as_pos"] == 0:
                return "Absent"
            return "Frequent" if int(row["as_pos"]) in frequent_positions else "Rare"

        raw["DisRegion"] = raw.apply(region, axis=1)
        raw["mat_block"] = raw["blk"].astype(int)
        raw["block"] = raw["mat_block"].map(TRAIN_BLOCK_MAP)
        raw = raw.loc[raw["block"].notna()].copy()
        raw["block"] = raw["block"].astype(int)

        # Final training RT rule: correct trials; lower 1.5-IQR cutoff; remove the
        # first trial of every original block. No upper IQR cutoff was implemented.
        q1, q3 = raw["rt"].quantile([0.25, 0.75])
        lower = q1 - 1.5 * (q3 - q1)
        raw["exclude"] = raw["rt"].lt(lower)
        raw["exclude"] |= raw.groupby("mat_block").cumcount().eq(0)
        raw = raw.loc[raw["correct"].eq(1) & ~raw["exclude"]].copy()
        raw["RT_ms"] = raw["rt"] * 1000.0
        raw["subject"] = id_map.get(original_id, original_id)
        raw["scanner"] = 0.0
        rows.append(raw[[
            "subject", "group", "Dimension", "DisRegion", "RT_ms",
            "scanner", "block",
        ]])
    return pd.concat(rows, ignore_index=True)


def load_scanner() -> pd.DataFrame:
    """Load and preprocess all scanner event files."""
    rows: list[pd.DataFrame] = []
    pattern = ROOT / "sub-*" / "func" / "*_task-suppression_run-*_events.tsv"
    for path_string in sorted(glob.glob(str(pattern))):
        path = Path(path_string)
        parts = path.name.split("_")
        subject = parts[0].replace("sub-", "")
        run = int(parts[2].replace("run-", ""))
        frame = pd.read_csv(path, sep="\t")
        frame["subject"] = subject
        frame["run"] = run
        frame["group"] = "SS" if subject.startswith("SS") else "DS"
        rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    raw["trial_in_run"] = raw.groupby(["subject", "run"]).cumcount() + 1
    raw["exclude"] = raw["RT"].lt(RT_MIN) | raw["RT"].gt(RT_MAX) | raw["RT"].isna()
    raw["exclude"] |= raw["trial_in_run"].eq(1)
    data = raw.loc[~raw["exclude"]].copy()
    data["RT_ms"] = data["RT"] * 1000.0
    data["block_in_run"] = ((data["trial_in_run"] - 1) // TRIALS_PER_BLOCK + 1).clip(upper=4)
    data["block"] = data["block_in_run"] + N_TRAIN_BLOCKS + (data["run"] - 1) * 4
    data["scanner"] = 1.0
    return data[[
        "subject", "group", "Dimension", "DisRegion", "RT_ms", "scanner", "block",
    ]]


def linear_contrast(model, weights: dict[str, float], label: str) -> dict[str, float | str]:
    """Wald contrast using the fitted cluster-robust covariance matrix."""
    names = list(model.params.index)
    vector = np.zeros(len(names))
    for term, weight in weights.items():
        vector[names.index(term)] = weight
    estimate = float(vector @ model.params.to_numpy())
    variance = float(vector @ model.cov_params().to_numpy() @ vector)
    se = float(np.sqrt(variance))
    z = estimate / se
    p = float(2 * norm.sf(abs(z)))
    return {
        "contrast": label,
        "estimate_ms": estimate,
        "se_ms": se,
        "ci_low_ms": estimate - 1.959963984540054 * se,
        "ci_high_ms": estimate + 1.959963984540054 * se,
        "z": z,
        "p_raw": p,
    }


def fit_unified_model(training: pd.DataFrame, scanner: pd.DataFrame):
    """Participant-FE RT model with participant-clustered standard errors."""
    training = training.copy()
    scanner = scanner.loc[scanner["Dimension"].isin(["Absent", "Same", "Different"])].copy()
    for frame in (training, scanner):
        frame["same_dim"] = frame["Dimension"].eq("Same").astype(float)
        frame["diff_dim"] = frame["Dimension"].eq("Different").astype(float)
        frame["ds_group"] = frame["group"].eq("DS").astype(float)
    columns = [
        "subject", "group", "Dimension", "RT_ms", "block", "scanner",
        "same_dim", "diff_dim", "ds_group",
    ]
    both = pd.concat([training[columns], scanner[columns]], ignore_index=True)
    formula = (
        "RT_ms ~ C(subject) + block + block:ds_group + scanner + "
        "scanner:ds_group + same_dim + same_dim:ds_group + "
        "diff_dim + diff_dim:ds_group"
    )
    model = smf.ols(formula, both).fit(
        cov_type="cluster", cov_kwds={"groups": both["subject"]}
    )
    return both, model, formula


def unified_contrasts(model) -> tuple[pd.DataFrame, pd.DataFrame]:
    learning = pd.DataFrame([
        linear_contrast(
            model, {"block": 1, "block:ds_group": 0.5},
            "Group-balanced linear block slope",
        ),
        linear_contrast(
            model, {"block:ds_group": 1},
            "Group x Block (DS minus SS slope)",
        ),
        linear_contrast(
            model, {"scanner": 1, "scanner:ds_group": 0.5},
            "Group-balanced scanner transition",
        ),
        linear_contrast(
            model, {"scanner:ds_group": 1},
            "Group x Scanner transition",
        ),
    ])
    interference = pd.DataFrame([
        linear_contrast(model, {"same_dim": 1}, "SS: Same - Absent"),
        linear_contrast(
            model, {"same_dim": 1, "same_dim:ds_group": 1},
            "DS: Same - Absent",
        ),
        linear_contrast(
            model, {"same_dim:ds_group": 1},
            "Group x Same (DS cost - SS cost)",
        ),
        linear_contrast(model, {"diff_dim": 1}, "SS: Different - Absent"),
        linear_contrast(
            model, {"diff_dim": 1, "diff_dim:ds_group": 1},
            "DS: Different - Absent",
        ),
        linear_contrast(
            model, {"diff_dim:ds_group": 1},
            "Group x Different (DS cost - SS cost)",
        ),
    ])
    interference["p_holm"] = np.nan
    for family in (
        ["SS: Same - Absent", "DS: Same - Absent"],
        ["SS: Different - Absent", "DS: Different - Absent"],
    ):
        selected = interference["contrast"].isin(family)
        interference.loc[selected, "p_holm"] = multipletests(
            interference.loc[selected, "p_raw"], method="holm"
        )[1]
    return learning, interference


def fit_spatial_learning(training: pd.DataFrame, scanner: pd.DataFrame):
    """Biased-dimension Group x Location LMM and subject-level simple effects."""
    training = training.copy()
    scanner = scanner.copy()
    for frame in (training, scanner):
        frame["rare_loc"] = frame["DisRegion"].eq("Rare").astype(float)
        frame["ds_group"] = frame["group"].eq("DS").astype(float)
    columns = [
        "subject", "group", "Dimension", "DisRegion", "RT_ms", "block",
        "rare_loc", "ds_group",
    ]
    combined = pd.concat([training[columns], scanner[columns]], ignore_index=True)
    spatial = pd.concat([
        combined.loc[
            combined["group"].eq("SS") & combined["Dimension"].eq("Same")
            & combined["DisRegion"].isin(["Frequent", "Rare"])
        ],
        combined.loc[
            combined["group"].eq("DS") & combined["Dimension"].eq("Different")
            & combined["DisRegion"].isin(["Frequent", "Rare"])
        ],
    ], ignore_index=True)
    formula = "RT_ms ~ block + rare_loc * ds_group"
    # BFGS avoids the spurious zero-variance/inf-log-likelihood solution that
    # statsmodels' L-BFGS optimizer reaches for this otherwise stable model.
    model = smf.mixedlm(formula, spatial, groups=spatial["subject"]).fit(
        reml=False, method="bfgs", maxiter=1000, disp=False
    )
    if (not model.converged) or float(model.cov_re.iloc[0, 0]) <= 0 or not np.isfinite(model.llf):
        raise RuntimeError("Spatial mixed model did not yield a valid non-boundary fit")
    fixed = []
    ci = model.conf_int()
    for term in model.fe_params.index:
        fixed.append({
            "term": term,
            "estimate_ms": model.fe_params[term],
            "se_ms": model.bse_fe[term],
            "ci_low_ms": ci.loc[term, 0],
            "ci_high_ms": ci.loc[term, 1],
            "z": model.fe_params[term] / model.bse_fe[term],
            "p": model.pvalues[term],
        })
    fixed = pd.DataFrame(fixed)

    subject_means = (
        spatial.groupby(["subject", "group", "DisRegion"], observed=True)["RT_ms"]
        .mean().reset_index()
    )
    simple_rows = []
    for group in ("SS", "DS"):
        wide = subject_means.loc[subject_means["group"].eq(group)].pivot(
            index="subject", columns="DisRegion", values="RT_ms"
        )
        delta = (wide["Rare"] - wide["Frequent"]).dropna()
        t_value, p_value = stats.ttest_1samp(delta, 0)
        se = stats.sem(delta)
        critical = stats.t.ppf(0.975, len(delta) - 1)
        simple_rows.append({
            "group": group,
            "n": len(delta),
            "mean_ms": delta.mean(),
            "se_ms": se,
            "ci_low_ms": delta.mean() - critical * se,
            "ci_high_ms": delta.mean() + critical * se,
            "t": t_value,
            "df": len(delta) - 1,
            "p_raw": p_value,
            "cohens_dz": t_value / np.sqrt(len(delta)),
        })
    simple = pd.DataFrame(simple_rows)
    simple["p_holm"] = multipletests(simple["p_raw"], method="holm")[1]
    return spatial, model, formula, fixed, simple, subject_means


def main() -> None:
    training = load_training(load_id_map())
    scanner = load_scanner()
    both, unified_model, unified_formula = fit_unified_model(training, scanner)
    learning, interference = unified_contrasts(unified_model)
    spatial, spatial_model, spatial_formula, spatial_fixed, spatial_simple, spatial_means = (
        fit_spatial_learning(training, scanner)
    )

    # Sample-scope assertions guard against accidental imaging-QC filtering.
    assert both["subject"].nunique() == 35
    assert both.loc[both["group"].eq("SS"), "subject"].nunique() == 17
    assert both.loc[both["group"].eq("DS"), "subject"].nunique() == 18
    assert training["subject"].nunique() == 31
    assert scanner["subject"].nunique() == 35
    assert spatial["subject"].nunique() == 35

    subjects = sorted(both["subject"].unique())
    ledger = pd.DataFrame({"subject": subjects})
    ledger["group"] = ledger["subject"].str[:2]
    ledger["training"] = ledger["subject"].isin(set(training["subject"]))
    ledger["scanner"] = ledger["subject"].isin(set(scanner["subject"]))
    trial_counts = (
        both.groupby(["subject", "group", "scanner"], observed=True)
        .size().rename("retained_trials").reset_index()
    )

    learning.to_csv(OUT / "learning_trajectory_contrasts.csv", index=False)
    interference.to_csv(OUT / "dimension_interference_contrasts.csv", index=False)
    spatial_fixed.to_csv(OUT / "spatial_learning_fixed_effects.csv", index=False)
    spatial_simple.to_csv(OUT / "spatial_learning_simple_effects_holm.csv", index=False)
    spatial_means.to_csv(OUT / "spatial_learning_subject_means.csv", index=False)
    ledger.to_csv(OUT / "subject_phase_ledger.csv", index=False)
    trial_counts.to_csv(OUT / "retained_trial_counts.csv", index=False)
    (OUT / "unified_rt_model_summary.txt").write_text(
        f"Formula: {unified_formula}\n\n{unified_model.summary().as_text()}\n"
    )
    (OUT / "spatial_learning_model_summary.txt").write_text(
        f"Formula: {spatial_formula}\n\n{spatial_model.summary()}\n"
    )

    spatial_interaction = spatial_fixed.loc[
        spatial_fixed["term"].eq("rare_loc:ds_group")
    ].iloc[0]
    report = [
        "Behavioral RT analysis: all behavior-valid participants",
        "N=35 (SS=17, DS=18); training N=31; scanner N=35.",
        f"Unified RT model: {unified_formula}",
        f"Spatial RT model: {spatial_formula}",
        "",
        "LEARNING TRAJECTORY",
        learning.to_string(index=False),
        "",
        "DISTRACTOR INTERFERENCE",
        interference.to_string(index=False),
        "",
        "SPATIAL-PROBABILITY LEARNING",
        spatial_fixed.to_string(index=False),
        "",
        spatial_simple.to_string(index=False),
        "",
        "Headline Group x Location: "
        f"beta={spatial_interaction['estimate_ms']:.6f} ms, "
        f"z={spatial_interaction['z']:.6f}, p={spatial_interaction['p']:.9f}.",
    ]
    (OUT / "behavioral_results_n35.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
