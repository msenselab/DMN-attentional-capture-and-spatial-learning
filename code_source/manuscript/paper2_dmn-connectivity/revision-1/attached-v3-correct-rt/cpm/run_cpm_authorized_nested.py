#!/usr/bin/env python3
"""Authorized RT-gated, six-cell nested LOOCV CPM for Paper 2 revision 1.

Writes only under this script's directory. Inputs are immutable: N=33 IDs and
DMN values from dmn_capture_unified.csv, frozen whole-brain matrices aligned to
those IDs, and scanner events. The behavioral target is recomputed from the
specified event-level correct/RT gate; no historical capture labels or old
permutation outputs are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set before NumPy import; the analysis evaluates only one heavy batch at a time.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[5])))
OUT = Path(__file__).resolve().parent
NEURAL_CSV = ROOT / "manuscript/paper2/code_and_data/dmn_capture_unified.csv"
PREP_DIR = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior"
SCORES_CSV = PREP_DIR / "participant_scores_n33.csv"
TRIALS_CSV = PREP_DIR / "correct_rt_trials_n33.csv"
PREP_SCRIPT = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/prepare_data.py"
CONN_NPY = ROOT / "manuscript/paper2/results/paper2_wholebrain_cpm_20260603/wholebrain_connectivity_matrices.npy"
OLD_ALIGNED_CSV = ROOT / "manuscript/paper2/results/paper2_wholebrain_cpm_20260603/wholebrain_behavior_aligned.csv"
SPEC = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/ANALYSIS_SPEC.md"
AUDIT = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/connectivity-audit/run_connectivity_audit.py"

FAMILIES = ["DefaultA", "A-B", "A-C", "DefaultB", "B-C", "DefaultC"]
FAMILY_LONG = ["Within DefaultA", "DefaultA--DefaultB", "DefaultA--DefaultC", "Within DefaultB", "DefaultB--DefaultC", "Within DefaultC"]
EXPECTED_EDGES = np.array([561, 1088, 442, 496, 416, 78], dtype=int)
SEED = 20261001
N_PERM = 10_000
EDGE_P = 0.01


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def holm_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q = (len(p) - np.arange(len(p))) * ranked
    q = np.maximum.accumulate(q)
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def raw_network17(label: str) -> str:
    parts = str(label).split("_")
    if len(parts) >= 3 and parts[0].endswith("Networks"):
        return parts[2]
    raise ValueError(f"Unexpected Schaefer/Yeo17 label: {label}")


def family_mapping() -> tuple[np.ndarray, pd.DataFrame]:
    from nilearn import datasets
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2, verbose=0)
    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in atlas["labels"]]
    if labels and labels[0] == "Background":
        labels = labels[1:]
    if len(labels) != 400:
        raise RuntimeError(f"Expected 400 labels after Background removal, got {len(labels)}")
    net = np.array([raw_network17(x) for x in labels], dtype=object)
    triu = np.triu_indices(400, k=1)
    ni, nj = net[triu[0]], net[triu[1]]
    masks = [
        (ni == "DefaultA") & (nj == "DefaultA"),
        ((ni == "DefaultA") & (nj == "DefaultB")) | ((ni == "DefaultB") & (nj == "DefaultA")),
        ((ni == "DefaultA") & (nj == "DefaultC")) | ((ni == "DefaultC") & (nj == "DefaultA")),
        (ni == "DefaultB") & (nj == "DefaultB"),
        ((ni == "DefaultB") & (nj == "DefaultC")) | ((ni == "DefaultC") & (nj == "DefaultB")),
        (ni == "DefaultC") & (nj == "DefaultC"),
    ]
    union = np.logical_or.reduce(masks)
    fid_full = np.full(len(ni), -1, dtype=np.int8)
    for f, mask in enumerate(masks):
        if np.any(fid_full[mask] >= 0):
            raise RuntimeError("Default subdivision families overlap")
        fid_full[mask] = f
    fid = fid_full[union]
    counts = np.bincount(fid, minlength=6)
    if not np.array_equal(counts, EXPECTED_EDGES):
        raise RuntimeError(f"Unexpected family counts {counts.tolist()}, expected {EXPECTED_EDGES.tolist()}")
    roi = pd.DataFrame({"roi": np.arange(400), "schaefer400_yeo17_label": labels, "network17": net})
    return union, fid, roi


def load_authorized_capture(neural: pd.DataFrame, scores: pd.DataFrame, trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the parent-authorized N33 labels and independently verify their source IDs."""
    req_neural = {"subject", "group", "dmn_conn"}
    req_scores = {"subject", "group", "dmn_conn", "biased_capture", "unbiased_capture", "absent", "biased", "unbiased", "n_absent", "n_biased", "n_unbiased"}
    req_trials = {"subject", "group", "run", "trial", "correct", "RT", "category", "first_run"}
    if not req_neural.issubset(neural.columns) or not req_scores.issubset(scores.columns) or not req_trials.issubset(trials.columns):
        raise RuntimeError("Required columns missing from frozen neural, parent score, or parent authorized-trial files")
    if neural.subject.duplicated().any() or scores.subject.duplicated().any() or len(neural) != 33 or len(scores) != 33:
        raise RuntimeError("Neural and parent-score files must each have 33 unique N33 IDs")
    neural = neural.copy(); scores = scores.copy(); trials = trials.copy()
    neural.subject = neural.subject.astype(str); scores.subject = scores.subject.astype(str); trials.subject = trials.subject.astype(str)
    # Parent output is authoritative for capture. Verify subject order and immutable neural values exactly.
    if not np.array_equal(scores.subject.to_numpy(), neural.subject.to_numpy()):
        raise RuntimeError("Parent participant-score ID order differs from frozen neural N33 IDs")
    if not np.array_equal(scores.group.astype(str).to_numpy(), neural.group.astype(str).to_numpy()):
        raise RuntimeError("Parent participant-score groups differ from frozen neural source")
    if not np.allclose(scores.dmn_conn.to_numpy(float), neural.dmn_conn.to_numpy(float), atol=0, rtol=0):
        raise RuntimeError("Parent participant-score neural values differ from frozen neural source")
    if set(trials.subject) != set(neural.subject) or len(trials) != 10995:
        raise RuntimeError("Parent authorized N33 trial table does not have the verified N=33 / 10,995 rows")
    valid = trials.correct.eq(1) & trials.RT.between(.2, 3, inclusive="both") & np.isfinite(pd.to_numeric(trials.RT, errors="coerce"))
    if not bool(valid.all()):
        raise RuntimeError("Parent authorized-trial table contains a row outside the authorized correct/RT gate")
    # Rebuild only as a verification of the parent prepared outcome, never to override it.
    chk = trials.assign(RT_ms=trials.RT.astype(float) * 1000).pivot_table(index="subject", columns="category", values="RT_ms", aggfunc="mean")
    chk = chk.reindex(scores.subject)
    if not np.allclose((chk.biased - chk.absent).to_numpy(float), scores.biased_capture.to_numpy(float), atol=1e-10):
        raise RuntimeError("Parent biased-capture score does not reproduce from parent authorized trial table")
    subject = pd.DataFrame({
        "subject": scores.subject, "group": scores.group, "dmn_conn_frozen": scores.dmn_conn,
        "biased_dimension": np.where(scores.group.eq("SS"), "Same", "Different"),
        "n_retained_authorized": scores.n_absent + scores.n_biased + scores.n_unbiased,
        "n_biased_retained": scores.n_biased, "n_unbiased_retained": scores.n_unbiased, "n_absent_retained": scores.n_absent,
        "biased_rt_mean_ms": scores.biased, "unbiased_rt_mean_ms": scores.unbiased, "absent_rt_mean_ms": scores.absent,
        "biased_capture_raw_ms": scores.biased_capture, "unbiased_capture_raw_ms": scores.unbiased_capture,
    })
    subject["biased_capture_global_group_centered_ms"] = subject["biased_capture_raw_ms"] - subject.groupby("group")["biased_capture_raw_ms"].transform("mean")
    subject["dmn_global_group_centered"] = subject["dmn_conn_frozen"] - subject.groupby("group")["dmn_conn_frozen"].transform("mean")
    gate = trials.groupby(["subject", "run"], as_index=False).agg(n_authorized=("trial", "size"), n_first_trial_retained=("first_run", "sum"))
    return subject, gate


def prepare_folds(X: np.ndarray, groups: np.ndarray) -> list[dict]:
    folds = []
    for test in range(len(groups)):
        train = np.arange(len(groups)) != test
        xt = X[train]
        sd = xt.std(axis=0, ddof=1)
        zt = (xt - xt.mean(axis=0)) / np.where(sd == 0, 1.0, sd)
        folds.append({"train": train, "Xtr": xt, "Xte": X[test], "Ztr": zt, "test_group": groups[test]})
    return folds


def centered_fold_targets(raw_y: np.ndarray, groups: np.ndarray, test: int) -> tuple[np.ndarray, np.ndarray]:
    """Train-only within-group centering; works for y as N x B outcomes."""
    train = np.arange(len(groups)) != test
    yt = raw_y[train].copy()
    test_y = raw_y[test].copy()
    tg = groups[test]
    train_groups = groups[train]
    for g in np.unique(groups):
        idx = train_groups == g
        mu = yt[idx].mean(axis=0, keepdims=True)
        yt[idx] -= mu
        if g == tg:
            test_y = test_y - mu.ravel()
    return yt, test_y


def evaluate_batch(raw_y: np.ndarray, groups: np.ndarray, family_id: np.ndarray, folds: list[dict], rcrit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nested CPM for B outcomes: returns r(B,6), predictions(B,N,6), folded targets(B,N)."""
    n, b = raw_y.shape
    pred = np.zeros((b, n, 6), dtype=np.float64)
    y_eval = np.zeros((b, n), dtype=np.float64)
    const = np.zeros((b, 6), dtype=np.int16)
    for test, fold in enumerate(folds):
        yt, ytest = centered_fold_targets(raw_y, groups, test)
        y_eval[:, test] = ytest
        sy = yt.std(axis=0, ddof=1)
        if np.any(sy <= 0):
            raise RuntimeError("Zero-variance training outcome encountered")
        zy = yt / sy[None, :]
        r_edges = fold["Ztr"].T @ zy / (yt.shape[0] - 1)
        for f in range(6):
            fm = family_id == f
            rf = r_edges[fm]
            signed = np.where(np.abs(rf) > rcrit, np.sign(rf), 0.0)
            Xtr = fold["Xtr"][:, fm]
            strength = Xtr @ signed
            test_strength = fold["Xte"][fm] @ signed
            sm = strength.mean(axis=0)
            sc = strength - sm
            ss = (sc * sc).sum(axis=0)
            slope = np.divide((sc * yt).sum(axis=0), ss, out=np.zeros(b), where=ss > 1e-14)
            p = (-slope * sm) + slope * test_strength  # training target mean = 0 after fold centering
            fallback = ss <= 1e-14
            if np.any(fallback):
                p[fallback] = 0.0
                const[fallback, f] += 1
            pred[:, test, f] = p
    yc = y_eval - y_eval.mean(axis=1, keepdims=True)
    pc = pred - pred.mean(axis=1, keepdims=True)
    denom = np.sqrt((yc * yc).sum(axis=1)[:, None] * (pc * pc).sum(axis=1))
    r = np.divide((yc[:, :, None] * pc).sum(axis=1), denom, out=np.zeros((b, 6)), where=denom > 1e-14)
    return r, pred, y_eval, const


def selected_counts_observed(raw_y: np.ndarray, groups: np.ndarray, family_id: np.ndarray, folds: list[dict], rcrit: float) -> pd.DataFrame:
    rows = []
    for test, fold in enumerate(folds):
        yt, _ = centered_fold_targets(raw_y[:, None], groups, test)
        zy = yt[:, 0] / yt[:, 0].std(ddof=1)
        r_edges = fold["Ztr"].T @ zy / (len(zy) - 1)
        for f, name in enumerate(FAMILIES):
            rf = r_edges[family_id == f]
            rows.append({"held_out_index": test, "family": name, "positive_edges_selected": int((rf > rcrit).sum()), "negative_edges_selected": int((rf < -rcrit).sum()), "signed_edges_selected": int((np.abs(rf) > rcrit).sum())})
    return pd.DataFrame(rows)


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if abs(r) >= 1 or n <= 3:
        return np.nan, np.nan
    z = np.arctanh(r)
    delta = stats.norm.ppf(.975) / np.sqrt(n - 3)
    return float(np.tanh(z - delta)), float(np.tanh(z + delta))


def fixed_prediction_bootstrap(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n_boot: int=10_000) -> np.ndarray:
    """Stratified participant resampling of fixed LOOCV pairs; descriptive only."""
    idx_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    out = np.empty((n_boot, pred.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_group])
        yy = y[idx]
        pp = pred[idx]
        yc = yy - yy.mean()
        pc = pp - pp.mean(axis=0)
        den = np.sqrt((yc @ yc) * (pc * pc).sum(axis=0))
        out[b] = np.divide(yc @ pc, den, out=np.zeros(pred.shape[1]), where=den > 1e-14)
    return out


def make_permutation_batch(raw_y: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    out = np.empty((len(raw_y), n), dtype=np.float64)
    for b in range(n):
        yp = raw_y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            yp[idx] = rng.permutation(yp[idx])
        out[:, b] = yp
    return out


def run_permutations(raw_y: np.ndarray, groups: np.ndarray, family_id: np.ndarray, folds: list[dict], rcrit: float, n_perm: int, seed: int, batch_size: int, checkpoint: Path) -> np.ndarray:
    config = {"n_perm": n_perm, "seed": seed, "batch_size": batch_size, "families": FAMILIES, "rcrit": rcrit, "centering": "training_only_within_group"}
    if checkpoint.exists():
        saved = np.load(checkpoint, allow_pickle=False)
        prior = json.loads(str(saved["config_json"].item()))
        if prior != config:
            raise RuntimeError("Existing checkpoint configuration differs; refuse mixed permutation runs")
        perm_r = saved["permutation_r"]
        done = int(saved["done"])
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(saved["rng_state_json"].item()))
        if perm_r.shape != (n_perm, 6) or done < 0 or done > n_perm:
            raise RuntimeError("Invalid checkpoint dimensions/state")
    else:
        perm_r = np.full((n_perm, 6), np.nan, dtype=np.float64)
        done = 0
        rng = np.random.default_rng(seed)
    while done < n_perm:
        stop = min(done + batch_size, n_perm)
        yp = make_permutation_batch(raw_y, groups, rng, stop - done)
        r, _, _, _ = evaluate_batch(yp, groups, family_id, folds, rcrit)
        perm_r[done:stop] = r
        done = stop
        tmp = checkpoint.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, permutation_r=perm_r, done=done, config_json=json.dumps(config, sort_keys=True), rng_state_json=json.dumps(rng.bit_generator.state, sort_keys=True))
        os.replace(tmp, checkpoint)
        print(f"checkpoint: {done}/{n_perm}", flush=True)
    if not np.isfinite(perm_r).all():
        raise RuntimeError("Permutation distribution contains non-finite values")
    return perm_r


def plot_figure(subject: pd.DataFrame, pred: np.ndarray, observed_folded: np.ndarray, summary: pd.DataFrame) -> None:
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"SS": "#E69F00", "DS": "#7B3294"}
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), constrained_layout=True)
    for f, ax in enumerate(axes.flat):
        for g in ["SS", "DS"]:
            ix = subject.group.eq(g).to_numpy()
            ax.scatter(observed_folded[ix], pred[ix, f], s=44, color=colors[g], edgecolor="white", linewidth=.7, alpha=.92, label=g)
        x = observed_folded
        y = pred[:, f]
        slope, intercept, *_ = stats.linregress(x, y)
        lo, hi = np.nanmin(np.r_[x, y]), np.nanmax(np.r_[x, y])
        pad = max((hi - lo) * .10, 1)
        xs = np.linspace(lo - pad, hi + pad, 100)
        ax.plot(xs, slope * xs + intercept, color="#222222", linewidth=1.4)
        ax.plot(xs, xs, color="#777777", linestyle="--", linewidth=1)
        ax.axhline(0, color="#dddddd", lw=.65, zorder=0); ax.axvline(0, color="#dddddd", lw=.65, zorder=0)
        r = summary.loc[f, "observed_loocv_r"]
        p = summary.loc[f, "perm_p_two_sided_equal_tail"]
        q = summary.loc[f, "perm_qBH6_two_sided_equal_tail"]
        ax.set_title(FAMILIES[f], fontweight="bold")
        ax.text(.04, .96, f"r = {r:+.2f}\npperm = {p:.3f}\nqBH = {q:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#bdbdbd", alpha=.95))
        ax.set_xlabel("Observed authorized capture\n(fold-centered, ms)")
        ax.set_ylabel("Held-out CPM prediction\n(fold-centered, ms)")
        if f == 0: ax.legend(title="Group", frameon=False, loc="lower right")
    fig.savefig(OUT / "Figure5_authorized_nested_cpm.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "Figure5_authorized_nested_cpm.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="Overwrite completed output checkpoint only when deliberate")
    args = ap.parse_args()
    if args.n_perm != 10_000:
        raise ValueError("This authorized run must use exactly 10,000 permutations")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    inputs = [NEURAL_CSV, SCORES_CSV, TRIALS_CSV, PREP_SCRIPT, CONN_NPY, OLD_ALIGNED_CSV, SPEC, AUDIT, Path(__file__)]
    if not all(p.exists() for p in inputs):
        raise FileNotFoundError([str(p) for p in inputs if not p.exists()])
    provenance = pd.DataFrame([{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs])
    provenance.to_csv(OUT / "input_and_script_sha256.csv", index=False)
    neural = pd.read_csv(NEURAL_CSV)
    scores = pd.read_csv(SCORES_CSV)
    trials = pd.read_csv(TRIALS_CSV)
    subject, gate = load_authorized_capture(neural, scores, trials)
    old = pd.read_csv(OLD_ALIGNED_CSV)
    if not np.array_equal(old.subject.astype(str).to_numpy(), subject.subject.to_numpy()):
        raise RuntimeError("Frozen connectivity matrix behavior alignment does not match authoritative neural source IDs")
    conn = np.load(CONN_NPY, mmap_mode="r")
    if conn.shape != (33, 400, 400):
        raise RuntimeError(f"Unexpected frozen connectivity shape {conn.shape}")
    union, family_id, roi = family_mapping()
    triu = np.triu_indices(400, k=1)
    X = np.asarray(conn[:, triu[0], triu[1]][:, union], dtype=np.float64)
    groups = subject.group.to_numpy(str)
    raw_y = subject.biased_capture_raw_ms.to_numpy(float)
    folds = prepare_folds(X, groups)
    # Training N=32, correlation df=30; p<.01 two-sided threshold.
    tcrit = stats.t.ppf(1 - EDGE_P / 2, df=30)
    rcrit = float(tcrit / np.sqrt(tcrit*tcrit + 30))
    observed_r_b, observed_pred_b, observed_folded_b, const_b = evaluate_batch(raw_y[:, None], groups, family_id, folds, rcrit)
    observed_r = observed_r_b[0]; predictions = observed_pred_b[0]; folded_y = observed_folded_b[0]
    if np.any(const_b):
        print(f"Observed constant-model fold counts by family: {const_b[0].tolist()}", flush=True)
    fold_counts = selected_counts_observed(raw_y, groups, family_id, folds, rcrit)
    fold_counts.insert(1, "held_out_subject", np.repeat(subject.subject.to_numpy(), 6))
    subject["biased_capture_fold_centered_ms"] = folded_y
    checkpoint = OUT / "nested_permutation_checkpoint.npz"
    if args.force and checkpoint.exists(): checkpoint.unlink()
    perm_r = run_permutations(raw_y, groups, family_id, folds, rcrit, args.n_perm, args.seed, args.batch_size, checkpoint)
    upper = (perm_r >= observed_r[None, :]).sum(axis=0)
    lower = (perm_r <= observed_r[None, :]).sum(axis=0)
    # Explicit equal-tail two-sided p: chosen because the null can be asymmetric under fallback models.
    p_upper = (upper + 1) / (args.n_perm + 1)
    p_lower = (lower + 1) / (args.n_perm + 1)
    p_two = np.minimum(1.0, 2 * np.minimum(p_upper, p_lower))
    p_abs = ((np.abs(perm_r) >= np.abs(observed_r)[None, :]).sum(axis=0) + 1) / (args.n_perm + 1)
    rng_boot = np.random.default_rng(args.seed + 1)
    boot = fixed_prediction_bootstrap(predictions, folded_y, groups, rng_boot, 10_000)
    rows = []
    for f, name in enumerate(FAMILIES):
        r = float(observed_r[f])
        tval = r * np.sqrt((33 - 2) / (1-r*r)) if abs(r) < 1 else np.sign(r)*np.inf
        nominal = float(2 * stats.t.sf(abs(tval), df=31)) if np.isfinite(tval) else 0.0
        flo, fhi = fisher_ci(r, 33)
        fc = fold_counts[fold_counts.family.eq(name)]
        rows.append({
            "family": name, "family_long": FAMILY_LONG[f], "n_subjects": 33, "n_possible_edges": int(EXPECTED_EDGES[f]),
            "observed_loocv_r": r, "descriptive_pearson_t_df31": float(tval), "descriptive_pearson_nominal_p": nominal,
            "fisher95_ci_low_descriptive": flo, "fisher95_ci_high_descriptive": fhi,
            "fixed_prediction_stratified_participant_bootstrap_ci_low_descriptive": float(np.quantile(boot[:,f], .025)),
            "fixed_prediction_stratified_participant_bootstrap_ci_high_descriptive": float(np.quantile(boot[:,f], .975)),
            "positive_tail_exceedances": int(upper[f]), "lower_tail_exceedances": int(lower[f]),
            "perm_p_positive_tail": float(p_upper[f]), "perm_p_lower_tail": float(p_lower[f]),
            "perm_p_two_sided_equal_tail": float(p_two[f]), "perm_p_abs_r_sensitivity_not_primary": float(p_abs[f]),
            "permutation_r_mean": float(perm_r[:,f].mean()), "permutation_r_sd": float(perm_r[:,f].std(ddof=1)),
            "permutation_r_q025": float(np.quantile(perm_r[:,f], .025)), "permutation_r_q975": float(np.quantile(perm_r[:,f], .975)),
            "median_positive_edges_per_fold": float(fc.positive_edges_selected.median()), "range_positive_edges_per_fold": f"{int(fc.positive_edges_selected.min())}-{int(fc.positive_edges_selected.max())}",
            "median_negative_edges_per_fold": float(fc.negative_edges_selected.median()), "range_negative_edges_per_fold": f"{int(fc.negative_edges_selected.min())}-{int(fc.negative_edges_selected.max())}",
            "observed_constant_model_folds": int(const_b[0,f]),
        })
    summary = pd.DataFrame(rows)
    summary["perm_qBH6_two_sided_equal_tail"] = bh_adjust(summary.perm_p_two_sided_equal_tail.to_numpy())
    summary["perm_pHolm6_two_sided_equal_tail_sensitivity"] = holm_adjust(summary.perm_p_two_sided_equal_tail.to_numpy())
    summary.to_csv(OUT / "table_cpm_sixcell_authorized_nested.csv", index=False)
    subject.to_csv(OUT / "authorized_capture_N33_subject_values.csv", index=False)
    gate.to_csv(OUT / "authorized_trial_gate_by_subject_run.csv", index=False)
    roi.to_csv(OUT / "schaefer400_yeo17_mapping.csv", index=False)
    fold_counts.to_csv(OUT / "observed_fold_selection_counts.csv", index=False)
    pred_frame = subject[["subject", "group", "biased_capture_raw_ms", "biased_capture_fold_centered_ms"]].copy()
    for f, name in enumerate(FAMILIES): pred_frame[f"predicted_{name}_fold_centered_ms"] = predictions[:,f]
    pred_frame.to_csv(OUT / "Figure5_authorized_nested_plot_data.csv", index=False)
    np.savez_compressed(OUT / "nested_permutation_distributions.npz", permutation_r=perm_r, observed_r=observed_r, family=np.array(FAMILIES), seed=args.seed, n_permutations=args.n_perm)
    plot_figure(subject, predictions, folded_y, summary)
    elapsed = time.monotonic() - t0
    method = {
        "analysis": "Six-cell Schaefer-400/Yeo-17 DefaultA/B/C signed CPM LOOCV",
        "families_in_figure_order": FAMILIES,
        "outcome_authorization": "correct == 1 AND finite RT in [200, 3000] ms inclusive; first-in-run trials retained",
        "capture": "mean authorized RT for biased dimension minus mean authorized RT for absent trials; SS=Same and DS=Different",
        "neural_inputs": "Fixed N=33 source IDs and dmn values from dmn_capture_unified.csv; connectivity matrices verified against its identical subject sequence",
        "group_centering": "For every held-out fold, raw capture is centered within group using TRAINING participants only; training group means are subtracted from training and held-out value. Thus plot y and predictions are fold-centered, not globally centered.",
        "selection": {"within_each_training_fold": True, "two_sided_p": EDGE_P, "training_n": 32, "correlation_df": 30, "absolute_r_critical": rcrit, "signed_network_strength": "sum positive selected edges minus sum negative selected edges"},
        "fit": "OLS of training-only fold-centered outcome on training signed network strength; applied to held-out strength; zero-strength fallback predicts 0 (training centered mean).",
        "permutation": {"n": args.n_perm, "seed": args.seed, "exchangeability": "raw outcomes shuffled independently within SS/DS; all fold centering, selection, fit, held-out prediction, and r recomputed", "two_sided_primary": "2*min((upper+1)/(R+1), (lower+1)/(R+1)), bounded at 1", "absolute_r": "reported sensitivity only, not substituted for equal-tail primary"},
        "correction": "BH across six equal-tail permutation p values; fresh Holm across six equal-tail permutation p values as sensitivity",
        "descriptive_intervals": "Fisher 95% CI uses N=33 independent-case formula and is descriptive because LOOCV predictions are dependent. Fixed-prediction stratified participant bootstrap is also descriptive and does not re-fit CPM.",
        "execution": {"utc_finished": datetime.now(timezone.utc).isoformat(), "duration_seconds": elapsed, "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__},
    }
    (OUT / "method_and_execution.json").write_text(json.dumps(method, indent=2) + "\n")
    verification = {
        "n_subjects": len(subject), "group_counts": subject.group.value_counts().sort_index().to_dict(), "subject_order_neural_equals_matrix_alignment": True,
        "connectivity_shape": list(conn.shape), "family_edge_counts": EXPECTED_EDGES.tolist(), "permutation_shape": list(perm_r.shape),
        "finite_observed_r": bool(np.isfinite(observed_r).all()), "finite_permutation_r": bool(np.isfinite(perm_r).all()),
        "authorized_trials_N33": int(subject.n_retained_authorized.sum()), "expected_authorized_trials_N33": 10995,
        "first_trial_rows_retained": int(gate.n_first_trial_retained.sum()), "first_trial_rows_total": int(len(gate)),
        "checkpoint_complete": bool(np.load(checkpoint, allow_pickle=False)["done"] == args.n_perm), "elapsed_seconds": elapsed,
    }
    (OUT / "execution_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(verification, indent=2))

if __name__ == "__main__":
    main()
