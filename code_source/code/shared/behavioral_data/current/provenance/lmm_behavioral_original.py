#!/usr/bin/env python3
"""
LMM Behavioral Analysis — Dimension-Specific Capture & Location Learning
=========================================================================
Replaces paired t-tests with linear mixed-effects models (LMMs) to:
  1. Unified capture LMM across training + scanner, with phase step function
     (scanner=0/1). Non-significant phase confirms smooth learning.
  2. Test location learning (Rare > Frequent) with formal Group × Location
     interaction — the key test of whether learning magnitude differs between
     SS and DS groups
  3. Test within-group location learning per dimension (transfer test)

Practice covariate: block (1–6 training, 7–14 scanner) — linear and directly
interpretable (RT change per block).

Output
------
- Console: APA-formatted results for pasting into manuscript
- File: derivatives/behavior/lmm_results.txt

Data
----
Training: .mat files from code/cross_dimension/data/ (sessions s11/s12, s21/s22)
Scanner:  BIDS events.tsv files
Columns used: RT, Dimension, DisRegion, group, subject, run, trial
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.io import loadmat
import pingouin as pg

warnings.filterwarnings("ignore")

BIDS_ROOT  = __import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[5]))
TRAIN_DIR  = os.path.join(BIDS_ROOT, "code", "cross_dimension", "data")
OUTPUT_DIR = os.path.join(BIDS_ROOT, "derivatives", "behavior")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RT_MIN, RT_MAX = 0.2, 3.0   # seconds
TRIALS_PER_BLOCK = 46        # 46 trials per mat block (verified from data)
N_TRAIN_BLKS = 6             # 6 display blocks in training (blocks 1–6)
N_SCAN_BLKS  = 8             # 8 display blocks in scanner (blocks 7–14)

# Mat-block number → display block (1–6)
TRAIN_BLK_MAP = {1: 1, 2: 2, 5: 3, 6: 4, 7: 5, 8: 6}

TRAIN_COLS = [
    "tar_ori", "conditions", "sessions", "dist_cond", "blk",
    "tar_pos",  "as_pos",    "response_group", "tar_orient_group",
    "c_orient", "key",       "correct",        "tm_cue",
    "tm_disp",  "tm_key",    "rt",             "tm_end", "tm_trigger",
]


# =============================================================================
# ID mapping
# =============================================================================

def load_id_map(bids_root):
    """
    Return dict: training_id (e.g. 'DS02') → BIDS subject ID (e.g. 'DS001').
    participants.tsv has a malformed header (space instead of tab between
    'group' and 'origin_name'), so we parse line-by-line.
    """
    tsv = os.path.join(bids_root, "participants.tsv")
    id_map = {}
    with open(tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0].startswith("participant_id"):
                continue
            bids_id = parts[0].replace("sub-", "").strip()
            origin  = parts[4].strip() if len(parts) > 4 else ""
            if origin and "_" in origin:
                id_map[origin.split("_")[0]] = bids_id
    return id_map


# =============================================================================
# Training data loading
# =============================================================================

def load_all_training_data(train_dir, id_map):
    """
    Load training-phase .mat files (sessions s11/s12 for DS, s21/s22 for SS).
    Returns tidy DataFrame with Dimension, DisRegion (biased≈Frequent),
    RT_ms, subject (= BIDS ID), group, scanner=0, block (display block 1–6).
    """
    train_sessions = {"s11", "s12", "s21", "s22"}
    rows = []
    for fpath in sorted(glob.glob(os.path.join(train_dir, "*.mat"))):
        stem  = os.path.basename(fpath).replace(".mat", "")
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        train_id, sess = parts[0], parts[1]
        if sess not in train_sessions:
            continue

        d = loadmat(fpath)
        if "trials" not in d:
            continue
        raw = pd.DataFrame(d["trials"], columns=TRAIN_COLS)

        # Group from sessions column (2=SS, 1=DS) or subject prefix
        raw["group"] = raw["sessions"].map({2.0: "SS", 1.0: "DS"}).fillna(
            "SS" if train_id.startswith("SS") else "DS"
        )
        grp = raw["group"].iloc[0]

        # Dimension
        def dim_label(row):
            c = row["conditions"]
            if c == 0:
                return "Absent"
            elif c == 2:
                return ("Same" if grp == "SS" else "Different")
            else:
                return ("Different" if grp == "SS" else "Same")
        raw["Dimension"] = raw.apply(dim_label, axis=1)

        # DisRegion - infer Frequent vs Rare from position counts (counterbalanced)
        biased_trials = raw[raw["conditions"] == 2]
        if len(biased_trials) > 0:
            pos_counts = biased_trials["as_pos"].value_counts()
            median_count = pos_counts.median()
            freq_pos = set(pos_counts[pos_counts > median_count].index.astype(int))
        else:
            freq_pos = set()
        
        def pos_to_region(row):
            pos = row["as_pos"]
            cond = row["conditions"]
            if cond == 0 or pos == 0:
                return "Absent"
            elif cond == 2:  # biased dimension
                return "Frequent" if int(pos) in freq_pos else "Rare"
            else:  # unbiased dimension - use same position mapping
                return "Frequent" if int(pos) in freq_pos else "Rare"
        raw["DisRegion"] = raw.apply(pos_to_region, axis=1)

        # Display block (1–6) via TRAIN_BLK_MAP
        raw["mat_blk"] = raw["blk"].astype(int)
        raw["block"] = raw["mat_blk"].map(TRAIN_BLK_MAP)
        raw = raw[raw["block"].notna()].copy()
        raw["block"] = raw["block"].astype(int)

        # Outliers: IQR rule + first trial of each mat block
        q1, q3 = raw["rt"].quantile(0.25), raw["rt"].quantile(0.75)
        iqr = q3 - q1
        raw["outlier"] = raw["rt"] < (q1 - 1.5 * iqr)
        raw["outlier"] |= (raw.groupby("mat_blk").cumcount() == 0)

        raw = raw[(raw["correct"] == 1) & (~raw["outlier"])].copy()
        raw["RT_ms"] = raw["rt"] * 1000.0
        raw["subject"] = id_map.get(train_id, train_id)
        raw["scanner"] = 0.0

        rows.append(raw[["subject", "group", "Dimension", "DisRegion",
                          "RT_ms", "scanner", "block"]])

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    return df


# =============================================================================
# Scanner data loading & preprocessing
# =============================================================================

def load_all_scanner_data(bids_root):
    """Load scanner-phase events.tsv files for all subjects."""
    all_data = []
    pattern = os.path.join(bids_root, "sub-*/func/*_task-suppression_run-*_events.tsv")
    for filepath in sorted(glob.glob(pattern)):
        fname = os.path.basename(filepath)
        parts = fname.split("_")
        subject = parts[0].replace("sub-", "")
        run = int(parts[2].replace("run-", ""))
        df = pd.read_csv(filepath, sep="\t")
        df["subject"] = subject
        df["run"] = run
        df["group"] = "SS" if subject.startswith("SS") else "DS"
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def preprocess(df):
    """Exclude outlier trials; add derived columns."""
    df = df.copy()
    df["trial_global"] = df.groupby(["subject", "run"]).cumcount() + 1
    # Exclude: RT out of range, missing, or first trial of each run
    df["outlier"] = (
        (df["RT"] < RT_MIN) | (df["RT"] > RT_MAX) | df["RT"].isna()
    )
    first = df.groupby(["subject", "run"])["trial_global"].transform("min")
    df.loc[df["trial_global"] == first, "outlier"] = True

    df = df[~df["outlier"]].copy()
    df["RT_ms"] = df["RT"] * 1000

    # Block number: 46 trials per block, 4 blocks per run, 2 runs
    # Display block: run 1 → 7–10, run 2 → 11–14
    df["block_in_run"] = ((df["trial_global"] - 1) // TRIALS_PER_BLOCK + 1).clip(upper=4)
    df["block"] = df["block_in_run"] + N_TRAIN_BLKS + (df["run"] - 1) * 4
    # block = 7–14 (global across the full experiment)

    # Dummy codes (reference = Absent / SS / Frequent)
    df["same_dim"]   = (df["Dimension"] == "Same").astype(float)
    df["diff_dim"]   = (df["Dimension"] == "Different").astype(float)
    df["ds_group"]   = (df["group"] == "DS").astype(float)
    df["rare_loc"]   = (df["DisRegion"] == "Rare").astype(float)
    df["scanner"]    = 1.0

    return df


# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_p(p):
    if p < .001:
        return "p < .001"
    return f"p = {p:.3f}"


def lmm_coef_line(res, term, label=None, df_resid=None):
    """Return APA-style coefficient line for one fixed-effect term."""
    label = label or term
    try:
        b   = res.params[term]
        se  = res.bse[term]
        z   = res.tvalues[term]
        p   = res.pvalues[term]
        ci  = res.conf_int().loc[term]
        return (f"  {label}: β = {b:+.2f} ms, SE = {se:.2f}, "
                f"z = {z:.2f}, {fmt_p(p)}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    except KeyError:
        return f"  {label}: [term not in model]"


def section(title):
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}"


# =============================================================================
# Analysis 0: Unified capture LMM — training + scanner, phase step function
# =============================================================================

def analysis_unified_phase(train_df, scan_df):
    """
    LMM: RT ~ block + same_dim * ds_group + diff_dim * ds_group
              + scanner + (1 | subject)

    Combines training (scanner=0, blocks 1–6) and scanner (scanner=1, blocks 7–14).
    Key test: scanner coefficient — if non-significant, training→scanner
    transition does not produce a step-change in RT, confirming the two
    phases form a continuous practice sequence.

    Reference: Absent trials, SS group, training phase.
    """
    lines = [section("ANALYSIS 0: Unified Phase LMM (Training + Scanner)")]
    lines.append(
        "Model: RT ~ block + same_dim * ds_group + diff_dim * ds_group + scanner\n"
        "block = global display block (training 1–6, scanner 7–14)\n"
        "scanner = 0 (training) / 1 (scanner; phase step function)\n"
        "Reference: Absent trials, SS group, training phase\n"
    )

    if train_df.empty:
        lines.append("  [Training data not available — skipping unified analysis]")
        return "\n".join(lines), None

    # Add dummy codes to training df
    train = train_df.copy()
    train["same_dim"] = (train["Dimension"] == "Same").astype(float)
    train["diff_dim"] = (train["Dimension"] == "Different").astype(float)
    train["ds_group"] = (train["group"] == "DS").astype(float)

    scan = scan_df[scan_df["Dimension"].isin(["Same", "Different", "Absent"])].copy()

    # Keep only subjects present in both phases
    shared_subs = set(train["subject"]) & set(scan["subject"])
    lines.append(f"N subjects with both phases: {len(shared_subs)}")
    lines.append(f"N subjects training-only: {train['subject'].nunique() - len(shared_subs)}")
    lines.append(f"N subjects scanner-only: {scan['subject'].nunique() - len(shared_subs)}")
    lines.append(f"  → block: training 1–{N_TRAIN_BLKS}, scanner {N_TRAIN_BLKS+1}–{N_TRAIN_BLKS+N_SCAN_BLKS}\n")

    train_both = train[train["subject"].isin(shared_subs)].copy()
    scan_both  = scan[scan["subject"].isin(shared_subs)].copy()

    both = pd.concat([
        train_both[["subject", "Dimension", "RT_ms", "block",
                    "same_dim", "diff_dim", "ds_group", "scanner"]],
        scan_both[["subject", "Dimension", "RT_ms", "block",
                   "same_dim", "diff_dim", "ds_group", "scanner"]],
    ], ignore_index=True)

    formula = "RT_ms ~ block + same_dim * ds_group + diff_dim * ds_group + scanner"
    md = smf.mixedlm(formula, both, groups=both["subject"])
    res = md.fit(reml=False, method="lbfgs")

    lines.append("Fixed effects:")
    lines.append(lmm_coef_line(res, "Intercept", "Intercept (Absent, SS, training phase)"))
    lines.append(lmm_coef_line(res, "block",      "block (practice; RT change per block)"))
    lines.append(lmm_coef_line(res, "scanner",    "Phase step (scanner vs training; KEY TEST)"))
    lines.append(lmm_coef_line(res, "same_dim",   "Same > Absent (SS group)"))
    lines.append(lmm_coef_line(res, "diff_dim",   "Different > Absent (SS group)"))
    lines.append(lmm_coef_line(res, "ds_group",   "Group main effect (DS vs SS)"))
    lines.append(lmm_coef_line(res, "same_dim:ds_group", "Group × Same interaction"))
    lines.append(lmm_coef_line(res, "diff_dim:ds_group", "Group × Different interaction"))

    return "\n".join(lines), res


# =============================================================================
# Analysis 1: Dimension-specific capture (scanner phase, all N=34)
# =============================================================================

def analysis_capture(df):
    """
    LMM: RT ~ block + same_dim * ds_group + diff_dim * ds_group
                + (1 | subject)

    Key contrasts (reference: Absent trials, SS group):
      same_dim        = Same > Absent capture (SS group)
      diff_dim        = Different > Absent capture (SS group)
      same_dim:ds_group  = Group × Same interaction
      diff_dim:ds_group  = Group × Different interaction
    """
    lines = [section("ANALYSIS 1: Dimension-Specific Capture (LMM)")]
    lines.append(
        "Model: RT ~ block + same_dim * ds_group + diff_dim * ds_group\n"
        "Reference: Absent trials, SS group\n"
        "N subjects = 34 (scanner phase, after motion exclusion)\n"
    )

    mod_df = df[df["Dimension"].isin(["Same", "Different", "Absent"])].copy()

    formula = "RT_ms ~ block + same_dim * ds_group + diff_dim * ds_group"
    md = smf.mixedlm(formula, mod_df, groups=mod_df["subject"])
    res = md.fit(reml=False, method="lbfgs")

    lines.append("Fixed effects:")
    lines.append(lmm_coef_line(res, "Intercept", "Intercept (Absent, SS group)"))
    lines.append(lmm_coef_line(res, "block",      "block (practice)"))
    lines.append(lmm_coef_line(res, "same_dim",   "Same > Absent (SS group)"))
    lines.append(lmm_coef_line(res, "diff_dim",   "Different > Absent (SS group)"))
    lines.append(lmm_coef_line(res, "ds_group",   "Group main effect (DS vs SS, Absent trials)"))
    lines.append(lmm_coef_line(res, "same_dim:ds_group",  "Group × Same interaction"))
    lines.append(lmm_coef_line(res, "diff_dim:ds_group",  "Group × Different interaction"))

    # Simple effects: capture within each group
    lines.append("\nSimple effects — capture within each group:")
    for grp_label, grp_val in [("SS", 0), ("DS", 1)]:
        grp_df = mod_df[mod_df["ds_group"] == grp_val]
        sub_means = grp_df.groupby(["subject", "Dimension"])["RT_ms"].mean().reset_index()
        pivot = sub_means.pivot(index="subject", columns="Dimension", values="RT_ms")
        for dim in ["Same", "Different"]:
            if dim in pivot.columns and "Absent" in pivot.columns:
                diff = pivot[dim] - pivot["Absent"]
                t, p = stats.ttest_1samp(diff.dropna(), 0)
                d = t / np.sqrt(len(diff.dropna()))
                lines.append(
                    f"  {grp_label} group — {dim} > Absent: "
                    f"M = {diff.mean():.1f} ms, t({len(diff)-1}) = {t:.2f}, "
                    f"{fmt_p(p)}, d = {d:.2f}"
                )

    return "\n".join(lines), res


# =============================================================================
# Analysis 2: Location learning — Group × Location interaction (key test)
# =============================================================================

def analysis_location_learning(train_df, scan_df):
    """
    LMM: RT ~ block + rare_loc * ds_group + (1 | subject)

    Restricted to BIASED-dimension distractor trials only:
      - SS group: Same-dimension distractors at Frequent or Rare locations
      - DS group: Different-dimension distractors at Frequent or Rare locations

    Uses combined training + scanner data for better power.
    Reference: Frequent-location trials, SS group.
    """
    lines = [section("ANALYSIS 2: Location Learning — Group × Location (LMM)")]
    lines.append(
        "Combined training + scanner data.\n"
        "Restricted to biased-dimension distractor trials (Frequent or Rare):\n"
        "  SS group: Same-dimension distractors\n"
        "  DS group: Different-dimension distractors\n"
        "Model: RT ~ block + rare_loc * ds_group\n"
        "Reference: Frequent-location trials, SS group\n"
    )
    
    # Combine training and scanner data
    train = train_df.copy()
    train["rare_loc"] = (train["DisRegion"] == "Rare").astype(float)
    train["ds_group"] = (train["group"] == "DS").astype(float)
    
    scan = scan_df.copy()
    
    # Standardize column names
    combined = pd.concat([
        train[["subject", "group", "Dimension", "DisRegion", "RT_ms", "block", "rare_loc", "ds_group"]],
        scan[["subject", "group", "Dimension", "DisRegion", "RT_ms", "block", "rare_loc", "ds_group"]],
    ], ignore_index=True)
    
    df = combined

    ss_biased = df[
        (df["group"] == "SS") &
        (df["Dimension"] == "Same") &
        (df["DisRegion"].isin(["Frequent", "Rare"]))
    ].copy()

    ds_biased = df[
        (df["group"] == "DS") &
        (df["Dimension"] == "Different") &
        (df["DisRegion"].isin(["Frequent", "Rare"]))
    ].copy()

    mod_df = pd.concat([ss_biased, ds_biased], ignore_index=True)
    n_ss = ss_biased["subject"].nunique()
    n_ds = ds_biased["subject"].nunique()
    lines.append(f"N = {mod_df['subject'].nunique()} (SS: {n_ss}, DS: {n_ds})\n")

    formula = "RT_ms ~ block + rare_loc * ds_group"
    md = smf.mixedlm(formula, mod_df, groups=mod_df["subject"])
    res = md.fit(reml=False, method="lbfgs")

    lines.append("Fixed effects:")
    lines.append(lmm_coef_line(res, "Intercept", "Intercept (Frequent, SS group)"))
    lines.append(lmm_coef_line(res, "block",     "block (practice)"))
    lines.append(lmm_coef_line(res, "rare_loc",  "Rare > Frequent (SS group; location learning)"))
    lines.append(lmm_coef_line(res, "ds_group",  "Group main effect (DS vs SS, Frequent trials)"))
    lines.append(lmm_coef_line(res, "rare_loc:ds_group", "Group × Location interaction (KEY TEST)"))

    # Within-group effects
    lines.append("\nSimple effects — location learning within each group:")
    sub_means = mod_df.groupby(["subject", "group", "DisRegion"])["RT_ms"].mean().reset_index()
    for grp in ["SS", "DS"]:
        grp_df = sub_means[sub_means["group"] == grp]
        pivot = grp_df.pivot(index="subject", columns="DisRegion", values="RT_ms")
        if "Rare" in pivot.columns and "Frequent" in pivot.columns:
            diff = pivot["Rare"] - pivot["Frequent"]
            t, p = stats.ttest_1samp(diff.dropna(), 0)
            d = t / np.sqrt(len(diff.dropna()))
            ci_lo = diff.mean() - 1.96 * diff.sem()
            ci_hi = diff.mean() + 1.96 * diff.sem()
            lines.append(
                f"  {grp} group: M = {diff.mean():.1f} ms (Rare − Frequent), "
                f"t({len(diff)-1}) = {t:.2f}, {fmt_p(p)}, d = {d:.2f}, "
                f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}]"
            )

    return "\n".join(lines), res


# =============================================================================
# Analysis 3: Transfer — does location learning cross dimensions? (within SS)
# =============================================================================

def analysis_transfer(df):
    """
    LMM (SS group only): RT ~ block + rare_loc * same_dim + (1 | subject)

    Reference: Frequent-location trials, Different-dimension (unbiased) distractors.
    """
    lines = [section("ANALYSIS 3: Dimension Specificity of Location Learning (SS group LMM)")]
    lines.append(
        "SS group only — all distractor-present trials at Frequent or Rare locations.\n"
        "Model: RT ~ block + rare_loc * same_dim\n"
        "Reference: Frequent-location, Different-dimension (unbiased) trials\n"
    )

    ss_df = df[
        (df["group"] == "SS") &
        (df["Dimension"].isin(["Same", "Different"])) &
        (df["DisRegion"].isin(["Frequent", "Rare"]))
    ].copy()

    lines.append(f"N = {ss_df['subject'].nunique()} SS participants\n")

    formula = "RT_ms ~ block + rare_loc * same_dim"
    md = smf.mixedlm(formula, ss_df, groups=ss_df["subject"])
    res = md.fit(reml=False, method="lbfgs")

    lines.append("Fixed effects:")
    lines.append(lmm_coef_line(res, "Intercept", "Intercept (Frequent, Different-dim)"))
    lines.append(lmm_coef_line(res, "block",     "block (practice)"))
    lines.append(lmm_coef_line(res, "rare_loc",  "Rare > Frequent (Different-dim; transfer effect)"))
    lines.append(lmm_coef_line(res, "same_dim",  "Same > Different at Frequent locations"))
    lines.append(lmm_coef_line(res, "rare_loc:same_dim",
                                "Location × Dimension interaction (dimension specificity of learning)"))

    # Simple effects per dimension
    lines.append("\nSimple effects — location learning by dimension (SS group):")
    sub_means = ss_df.groupby(["subject", "Dimension", "DisRegion"])["RT_ms"].mean().reset_index()
    for dim in ["Same", "Different"]:
        dim_df = sub_means[sub_means["Dimension"] == dim]
        pivot = dim_df.pivot(index="subject", columns="DisRegion", values="RT_ms")
        if "Rare" in pivot.columns and "Frequent" in pivot.columns:
            diff = pivot["Rare"] - pivot["Frequent"]
            t, p = stats.ttest_1samp(diff.dropna(), 0)
            d = t / np.sqrt(len(diff.dropna()))
            lines.append(
                f"  {dim}-dim: M = {diff.mean():.1f} ms, "
                f"t({len(diff)-1}) = {t:.2f}, {fmt_p(p)}, d = {d:.2f}"
            )

    return "\n".join(lines), res


# =============================================================================
# Analysis 4: Full factorial — capture × location × group
# =============================================================================

def analysis_full_factorial(df):
    """
    LMM: RT ~ block + same_dim * rare_loc * ds_group + (1 | subject)

    Restricted to distractor-present trials (Frequent or Rare).
    """
    lines = [section("ANALYSIS 4: Full Factorial — Dimension × Location × Group (LMM)")]
    lines.append(
        "Distractor-present trials (Frequent or Rare) only.\n"
        "Model: RT ~ block + same_dim * rare_loc * ds_group\n"
        "Reference: Different-dim, Frequent-location, SS group\n"
    )

    mod_df = df[df["DisRegion"].isin(["Frequent", "Rare"])].copy()
    n_subj = mod_df["subject"].nunique()
    lines.append(f"N = {n_subj} subjects\n")

    formula = "RT_ms ~ block + same_dim * rare_loc * ds_group"
    md = smf.mixedlm(formula, mod_df, groups=mod_df["subject"])
    res = md.fit(reml=False, method="lbfgs")

    lines.append("Fixed effects:")
    for term, label in [
        ("Intercept",                   "Intercept (Diff-dim, Frequent, SS)"),
        ("block",                       "block (practice)"),
        ("same_dim",                    "Same > Different at Frequent, SS"),
        ("rare_loc",                    "Rare > Frequent (Different-dim, SS)"),
        ("ds_group",                    "DS vs SS (Different-dim, Frequent)"),
        ("same_dim:rare_loc",           "Dim × Location (SS group)"),
        ("same_dim:ds_group",           "Dim × Group (Frequent locations)"),
        ("rare_loc:ds_group",           "Location × Group (Different-dim)"),
        ("same_dim:rare_loc:ds_group",  "Dim × Location × Group (3-way)"),
    ]:
        lines.append(lmm_coef_line(res, term, label))

    return "\n".join(lines), res


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading participant ID map...")
    id_map = load_id_map(BIDS_ROOT)

    print("Loading training data...")
    train_df = load_all_training_data(TRAIN_DIR, id_map)
    if not train_df.empty:
        print(f"  {train_df['subject'].nunique()} subjects, {len(train_df):,} training trials")
    else:
        print("  [No training data found]")

    print("Loading scanner-phase data...")
    raw = load_all_scanner_data(BIDS_ROOT)
    df  = preprocess(raw)
    print(f"  {df['subject'].nunique()} subjects, {len(df):,} scanner trials (after exclusions)")

    results = []

    print("\nRunning Analysis 0: Unified phase LMM (training + scanner)...")
    text0, _ = analysis_unified_phase(train_df, df)
    results.append(text0)

    print("Running Analysis 1: Capture LMM (scanner phase)...")
    text1, _ = analysis_capture(df)
    results.append(text1)

    print("Running Analysis 2: Location learning LMM (key test)...")
    text2, _ = analysis_location_learning(train_df, df)
    results.append(text2)

    print("Running Analysis 3: Transfer / dimension specificity (SS group)...")
    text3, _ = analysis_transfer(df)
    results.append(text3)

    print("Running Analysis 4: Full factorial...")
    text4, _ = analysis_full_factorial(df)
    results.append(text4)

    full_output = "\n\n".join(results)

    # Print to console
    print("\n" + full_output)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "lmm_results.txt")
    with open(out_path, "w") as f:
        f.write(full_output)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
