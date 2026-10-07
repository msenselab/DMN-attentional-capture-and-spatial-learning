#!/usr/bin/env python3
"""Frozen rare-minus-frequent six-cell CPM (N=33; 10,000 nested permutations)."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import time
from datetime import datetime, timezone

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from scipy import stats

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
ROUND = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2"
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
RES = ROUND / "resolved_reanalysis"
LEGACY_SOURCE = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/cpm/run_cpm_authorized_nested.py"
TARGET = PARENT / "learning_target_scores.csv"
TRIALS = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv"
MATRICES = RES / "extraction_common_coverage/primary_participant_matrices.npy"
MATRIX_SUBJECTS = RES / "extraction_common_coverage/primary_matrix_subjects.csv"
PARCELS = RES / "common399_analysis_ready/physical_yeo7_common399_parcels.csv"
FREEZE = PARENT / "input_freeze.json"
SEED = 20261004
N_PERM = 10000
BATCH = 50


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_legacy():
    spec = importlib.util.spec_from_file_location("authorized_cpm_helpers", LEGACY_SOURCE)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    return legacy


def make_crosswalk(parcels, expected_edges):
    """Atlas-image crosswalk constrained to supplied common399 physical metadata."""
    atlas = {k: datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=k, resolution_mm=2, verbose=0) for k in (7, 17)}
    img7, img17 = nib.load(atlas[7].maps), nib.load(atlas[17].maps)
    if img7.shape != img17.shape or not np.array_equal(img7.affine, img17.affine):
        raise RuntimeError("Schaefer physical Yeo7/Yeo17 maps are not spatially congruent")
    a = np.asanyarray(img7.dataobj).astype(np.int16).ravel()
    b = np.asanyarray(img17.dataobj).astype(np.int16).ravel()
    if not np.array_equal(a > 0, b > 0):
        raise RuntimeError("Schaefer physical Yeo7/Yeo17 foreground differs")
    pairs = np.unique(np.stack([a[a > 0], b[a > 0]], axis=1), axis=0)
    if pairs.shape != (400, 2) or len(np.unique(pairs[:, 0])) != 400 or len(np.unique(pairs[:, 1])) != 400:
        raise RuntimeError("Schaefer physical crosswalk is not bijective")
    labels = {}
    for k in (7, 17):
        lab = [x.decode() if isinstance(x, bytes) else str(x) for x in atlas[k].labels]
        if lab[0].lower() == "background":
            lab = lab[1:]
        if len(lab) != 400:
            raise RuntimeError("Unexpected atlas label count")
        labels[k] = lab
    all_rows = []
    for native7, native17 in pairs:
        label7, label17 = labels[7][native7 - 1], labels[17][native17 - 1]
        all_rows.append({"native_label": int(native7), "yeo17_label_id": int(native17),
                         "physical_label": label7, "yeo7_network": label7.split("_")[2],
                         "physical_yeo17_label": label17, "yeo17_network": label17.split("_")[2]})
    all_cross = pd.DataFrame(all_rows).set_index("native_label")
    required = {"matrix_index", "native_label", "physical_label", "yeo7_network"}
    if not required.issubset(parcels.columns) or len(parcels) != 399:
        raise RuntimeError("Invalid frozen common399 physical metadata")
    parcels = parcels.sort_values("matrix_index").reset_index(drop=True)
    native = parcels.native_label.to_numpy(int)
    if not np.array_equal(parcels.matrix_index.to_numpy(int), np.arange(399)) or set(native) != set(range(1, 401)) - {323}:
        raise RuntimeError("Frozen common399 metadata does not represent native323 omission")
    cross = all_cross.loc[native].reset_index()
    if not np.array_equal(cross.physical_label.to_numpy(str), parcels.physical_label.to_numpy(str)) or not np.array_equal(cross.yeo7_network.to_numpy(str), parcels.yeo7_network.to_numpy(str)):
        raise RuntimeError("Atlas crosswalk conflicts with frozen physical Yeo7 metadata")
    cross.insert(0, "matrix_index", np.arange(399))
    ti, tj = np.triu_indices(399, 1)
    ni, nj = cross.yeo17_network.to_numpy(str)[ti], cross.yeo17_network.to_numpy(str)[tj]
    cell_pairs = [("DefaultA", "DefaultA"), ("DefaultA", "DefaultB"), ("DefaultA", "DefaultC"),
                  ("DefaultB", "DefaultB"), ("DefaultB", "DefaultC"), ("DefaultC", "DefaultC")]
    full = np.full(len(ti), -1, dtype=np.int8)
    for f, (left, right) in enumerate(cell_pairs):
        mask = ((ni == left) & (nj == right)) | ((ni == right) & (nj == left))
        if np.any(full[mask] >= 0):
            raise RuntimeError("Six physical Default cells overlap")
        full[mask] = f
    union = full >= 0
    family_id = full[union]
    counts = np.bincount(family_id, minlength=6)
    if not np.array_equal(counts, expected_edges):
        raise RuntimeError(f"Six-cell edge count mismatch: {counts.tolist()}")
    provenance = {"common399_parcels": 399, "excluded_native_label": [323], "physical_yeo17_DefaultABC_parcels": int(cross.yeo17_network.str.startswith("Default").sum()),
                  "six_cell_edge_counts": counts.tolist(), "atlas_map_sha256": {str(atlas[k].maps): sha256(atlas[k].maps) for k in (7, 17)}}
    return union, family_id, cross, provenance


def scalar_predictions(X, y, groups, family_id, rcrit):
    """Deliberately scalar real-data check, separate from inherited batched helper."""
    n = len(y)
    predictions = np.zeros((n, 6), dtype=float)
    heldout = np.zeros(n, dtype=float)
    selected = np.zeros((n, 6, 2), dtype=int)
    for test in range(n):
        train = np.arange(n) != test
        xx, yy, gg = X[train], y[train].copy(), groups[train]
        for group in np.unique(groups):
            mu = yy[gg == group].mean()
            yy[gg == group] -= mu
            if groups[test] == group:
                heldout[test] = y[test] - mu
        for family in range(6):
            xf = xx[:, family_id == family]
            r = np.empty(xf.shape[1])
            for edge in range(xf.shape[1]):
                r[edge] = stats.pearsonr(xf[:, edge], yy).statistic if np.std(xf[:, edge]) > 0 else 0.0
            signs = np.where(np.abs(r) > rcrit, np.sign(r), 0.0)
            selected[test, family] = ((r > rcrit).sum(), (r < -rcrit).sum())
            strength = xf @ signs
            test_strength = X[test, family_id == family] @ signs
            if np.sum((strength - strength.mean()) ** 2) > 1e-14:
                beta = np.linalg.lstsq(np.c_[np.ones(len(yy)), strength], yy, rcond=None)[0]
                predictions[test, family] = beta[0] + beta[1] * test_strength
    return predictions, heldout, selected


def source_maps(groups, n_perm, seed):
    """Exact within-group maps: DS permutation is drawn before SS each draw."""
    ds = np.flatnonzero(groups == "DS")
    ss = np.flatnonzero(groups == "SS")
    if (len(ds), len(ss)) != (17, 16):
        raise RuntimeError("Expected DS17 and SS16")
    rng = np.random.default_rng(seed)
    ds_src = np.empty((n_perm, len(ds)), dtype=np.int16)
    ss_src = np.empty((n_perm, len(ss)), dtype=np.int16)
    for draw in range(n_perm):
        ds_src[draw] = rng.permutation(len(ds))
        ss_src[draw] = rng.permutation(len(ss))
    return ds, ss, ds_src, ss_src, rng.bit_generator.state


def make_targets(y, ds, ss, ds_src, ss_src, start, stop):
    out = np.empty((len(y), stop - start), dtype=float)
    out[ds] = y[ds][ds_src[start:stop]].T
    out[ss] = y[ss][ss_src[start:stop]].T
    return out


def plot_predictions(frame, families, summary):
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    colours = {"SS": "#E69F00", "DS": "#6A3D9A"}  # orange/purple; deliberately no green
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), constrained_layout=True)
    for f, ax in enumerate(axes.flat):
        x = frame.observed_fold_centered_learning_ms.to_numpy(float)
        y = frame[f"predicted_{families[f]}_ms"].to_numpy(float)
        for group in ("SS", "DS"):
            ix = frame.group.eq(group).to_numpy()
            ax.scatter(x[ix], y[ix], s=44, color=colours[group], edgecolor="white", linewidth=.7, alpha=.92, label=group)
        slope, intercept, *_ = stats.linregress(x, y)
        lo, hi = np.min(np.r_[x, y]), np.max(np.r_[x, y])
        pad = max(.1 * (hi - lo), 1.0)
        grid = np.linspace(lo - pad, hi + pad, 100)
        ax.plot(grid, slope * grid + intercept, color="#222222", lw=1.4)
        ax.plot(grid, grid, color="#777777", linestyle="--", lw=1)
        ax.axhline(0, color="#dddddd", lw=.65, zorder=0)
        ax.axvline(0, color="#dddddd", lw=.65, zorder=0)
        row = summary.iloc[f]
        ax.set_title(families[f], fontweight="bold")
        ax.text(.04, .96, f"r = {row.r_loocv:+.2f}\np_equal = {row.p_raw_equal_tail:.3f}\np_Holm = {row.p_Holm6_equal_tail:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#bdbdbd"))
        ax.set_xlabel("Observed held-out learning\n(rare − frequent RT; fold-centered ms)")
        ax.set_ylabel("Held-out CPM prediction\n(fold-centered ms)")
        if f == 0:
            ax.legend(title="Biased dimension", frameon=False, loc="lower right")
    fig.savefig(HERE / "Figure_CPM_learning_rare_minus_frequent.png", dpi=300, facecolor="white")
    fig.savefig(HERE / "Figure_CPM_learning_rare_minus_frequent.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    args = ap.parse_args()
    if args.n_perm != N_PERM or args.seed != SEED:
        raise ValueError("Frozen empirical CPM requires n_perm=10000 and seed=20261004")
    HERE.mkdir(parents=True, exist_ok=True)
    archive = HERE / "null_archive"
    archive.mkdir(exist_ok=True)
    t0 = time.monotonic()
    freeze = json.loads(FREEZE.read_text())
    required_hashes = freeze["input_hashes"]
    for path, known in required_hashes.items():
        if sha256(path) != known:
            raise RuntimeError(f"Frozen input hash changed: {path}")
    if sha256(TARGET) != freeze["target_hash"]:
        raise RuntimeError("Frozen learning target hash changed")
    # Capture a run-start baseline for old protected outputs.  A frozen manifest can
    # legitimately be older than parallel editorial work; only changes during this
    # CPM execution would be attributable to this run.
    protected_expected = freeze["protected_capture_and_source_hashes"]
    protected_start = {path: sha256(path) if Path(path).exists() else None for path in protected_expected}
    legacy = load_legacy()
    families, expected_edges = legacy.FAMILIES, legacy.EXPECTED_EDGES
    target = pd.read_csv(TARGET, dtype={"subject": str})
    ids = pd.read_csv(MATRIX_SUBJECTS, dtype={"subject": str})
    trials = pd.read_csv(TRIALS, dtype={"subject": str})
    if len(target) != 33 or not np.array_equal(target[["subject", "group"]].to_numpy(str), ids[["subject", "group"]].to_numpy(str)):
        raise RuntimeError("Target is not N33 in immutable matrix order")
    if target.group.value_counts().to_dict() != {"DS": 17, "SS": 16}:
        raise RuntimeError("Expected DS17/SS16")
    valid = trials.correct.eq(1) & np.isfinite(trials.RT_ms) & (trials.RT_ms > 200) & (trials.RT_ms <= 3000)
    biased = trials.loc[valid & trials.category.eq("biased")]
    # Group is verified from the frozen target; aggregate by subject so the matrix-order reindex is unambiguous.
    check = biased.groupby(["subject", "group", "DisRegion"]).RT_ms.mean().unstack("DisRegion").reindex(pd.MultiIndex.from_frame(target[["subject", "group"]]))
    reconstructed = check.Rare.to_numpy() - check.Frequent.to_numpy()
    if len(biased) != 6721 or not np.allclose(reconstructed, target.learning_rare_minus_frequent_ms.to_numpy(float), atol=1e-10, rtol=0):
        raise RuntimeError("Learning target does not exactly reconstruct from 6,721 biased scanner trials")
    parcels = pd.read_csv(PARCELS)
    union, family_id, crosswalk, crosswalk_info = make_crosswalk(parcels, expected_edges)
    crosswalk.to_csv(HERE / "physical_yeo7_to_17_crosswalk_common399.csv", index=False)
    (HERE / "physical_crosswalk_verification.json").write_text(json.dumps(crosswalk_info, indent=2) + "\n")
    conn = np.load(MATRICES, mmap_mode="r")
    if conn.shape != (33, 399, 399) or not np.isfinite(conn).all() or not np.allclose(conn, conn.swapaxes(1, 2), atol=1e-12):
        raise RuntimeError("Invalid frozen common399 connectivity matrices")
    ti, tj = np.triu_indices(399, 1)
    X = np.asarray(conn[:, ti, tj][:, union], dtype=float)
    y = target.learning_rare_minus_frequent_ms.to_numpy(float)
    groups = target.group.to_numpy(str)
    folds = legacy.prepare_folds(X, groups)
    tcrit = stats.t.ppf(1 - .01 / 2, 30)
    rcrit = float(tcrit / np.sqrt(tcrit * tcrit + 30))
    observed_batch, pred_batch, fold_target_batch, constant = legacy.evaluate_batch(y[:, None], groups, family_id, folds, rcrit)
    observed, predictions, folded_target = observed_batch[0], pred_batch[0], fold_target_batch[0]
    scalar_pred, scalar_target, scalar_counts = scalar_predictions(X, y, groups, family_id, rcrit)
    max_scalar_error = float(np.max(np.abs(predictions - scalar_pred)))
    if max_scalar_error > 1e-8 or not np.allclose(folded_target, scalar_target, atol=1e-10):
        raise RuntimeError("Independent scalar observed prediction check failed")
    counts = legacy.selected_counts_observed(y, groups, family_id, folds, rcrit)
    counts.insert(1, "held_out_subject", np.repeat(target.subject.to_numpy(), 6))
    if not np.array_equal(counts[["positive_edges_selected", "negative_edges_selected"]].to_numpy(int), scalar_counts.reshape(33 * 6, 2)):
        raise RuntimeError("Independent scalar selection-count check failed")
    counts.to_csv(HERE / "observed_fold_selection_counts_learning.csv", index=False)
    ds, ss, ds_src, ss_src, rng_end = source_maps(groups, args.n_perm, args.seed)
    np.save(archive / "ds_source_indices.npy", ds_src)
    np.save(archive / "ss_source_indices.npy", ss_src)
    permutation_r = np.empty((args.n_perm, 6), dtype=float)
    for start in range(0, args.n_perm, args.batch_size):
        stop = min(start + args.batch_size, args.n_perm)
        yp = make_targets(y, ds, ss, ds_src, ss_src, start, stop)
        rb, _, _, _ = legacy.evaluate_batch(yp, groups, family_id, folds, rcrit)
        permutation_r[start:stop] = rb
        print(f"completed {stop}/{args.n_perm}", flush=True)
    if not np.isfinite(permutation_r).all():
        raise RuntimeError("Non-finite permutation result")
    np.save(archive / "permutation_r.npy", permutation_r)
    replay_ds, replay_ss, replay_ds_src, replay_ss_src, replay_state = source_maps(groups, args.n_perm, args.seed)
    if not (np.array_equal(ds, replay_ds) and np.array_equal(ss, replay_ss) and np.array_equal(ds_src, replay_ds_src) and np.array_equal(ss_src, replay_ss_src) and rng_end == replay_state):
        raise RuntimeError("Seeded within-group source-map replay failed")
    upper_n = (permutation_r >= observed).sum(axis=0)
    lower_n = (permutation_r <= observed).sum(axis=0)
    p_equal = np.minimum(1., 2 * np.minimum((upper_n + 1) / (N_PERM + 1), (lower_n + 1) / (N_PERM + 1)))
    p_abs = ((np.abs(permutation_r) >= np.abs(observed)).sum(axis=0) + 1) / (N_PERM + 1)
    bootstrap = legacy.fixed_prediction_bootstrap(predictions, folded_target, groups, np.random.default_rng(SEED + 1), 10000)
    np.save(archive / "fixed_prediction_bootstrap_r.npy", bootstrap)
    rows = []
    for family, name in enumerate(families):
        fc = counts[counts.family.eq(name)]
        rows.append({"family": name, "n_subjects": 33, "possible_edges": int(expected_edges[family]), "r_loocv": float(observed[family]),
                     "p_raw_equal_tail": float(p_equal[family]), "p_raw_absolute_sensitivity": float(p_abs[family]),
                     "equal_tail_upper_exceedances": int(upper_n[family]), "equal_tail_lower_exceedances": int(lower_n[family]),
                     "fixed_prediction_bootstrap_ci_low_descriptive": float(np.quantile(bootstrap[:, family], .025)),
                     "fixed_prediction_bootstrap_ci_high_descriptive": float(np.quantile(bootstrap[:, family], .975)),
                     "null_mean": float(permutation_r[:, family].mean()), "null_sd": float(permutation_r[:, family].std(ddof=1)),
                     "null_q025": float(np.quantile(permutation_r[:, family], .025)), "null_q975": float(np.quantile(permutation_r[:, family], .975)),
                     "median_positive_edges": float(fc.positive_edges_selected.median()), "median_negative_edges": float(fc.negative_edges_selected.median()),
                     "zero_strength_folds": int(constant[0, family])})
    summary = pd.DataFrame(rows)
    summary["p_Holm6_equal_tail"] = legacy.holm_adjust(p_equal)
    summary["p_Holm6_absolute_sensitivity"] = legacy.holm_adjust(p_abs)
    summary.to_csv(HERE / "table_cpm_learning_rare_minus_frequent.csv", index=False)
    prediction_data = target[["subject", "group", "learning_rare_minus_frequent_ms"]].copy()
    prediction_data["observed_fold_centered_learning_ms"] = folded_target
    for family, name in enumerate(families):
        prediction_data[f"predicted_{name}_ms"] = predictions[:, family]
    prediction_data.to_csv(HERE / "cpm_prediction_data_learning_rare_minus_frequent.csv", index=False)
    draw_summary = pd.DataFrame(permutation_r, columns=[f"r_{name}" for name in families])
    draw_summary.insert(0, "draw", np.arange(N_PERM))
    draw_summary.to_csv(HERE / "cpm_per_draw_null_summaries.csv", index=False)
    plot_predictions(prediction_data, families, summary)
    archive_files = [archive / "ds_source_indices.npy", archive / "ss_source_indices.npy", archive / "permutation_r.npy", archive / "fixed_prediction_bootstrap_r.npy"]
    archive_manifest = {"status": "complete", "seed": SEED, "n_permutations": N_PERM, "call_order": "For each draw: rng.permutation(DS source positions 0..16), then rng.permutation(SS source positions 0..15).",
                        "group_source_subject_order": {"DS": target.subject.iloc[ds].tolist(), "SS": target.subject.iloc[ss].tolist()},
                        "archive_files_sha256": {str(p.relative_to(HERE)): sha256(p) for p in archive_files},
                        "source_map_replay_exact": True, "permutation_distribution_shape": list(permutation_r.shape), "permutation_distribution_finite": True,
                        "rng_state_after_source_maps": rng_end}
    (HERE / "null_archive_manifest.json").write_text(json.dumps(archive_manifest, indent=2) + "\n")
    protected_end = {path: sha256(path) if Path(path).exists() else None for path in protected_expected}
    changed_during_run = {path: {"start": protected_start[path], "end": protected_end[path]} for path in protected_expected if protected_start[path] != protected_end[path]}
    differs_from_freeze_at_start = {path: {"frozen": known, "run_start": protected_start[path]} for path, known in protected_expected.items() if protected_start[path] != known}
    protected_check = {"protected_files_checked": len(protected_expected), "unchanged_during_this_cpm_run": not bool(changed_during_run),
                       "changes_during_this_cpm_run": changed_during_run, "preexisting_differences_from_freeze": differs_from_freeze_at_start}
    (HERE / "protected_hash_check_after_run.json").write_text(json.dumps(protected_check, indent=2) + "\n")
    if changed_during_run:
        raise RuntimeError("A protected pre-existing output/source changed during CPM run")
    execution = {"status": "completed", "utc_finished": datetime.now(timezone.utc).isoformat(), "duration_seconds": time.monotonic() - t0,
                 "n": 33, "groups": {"SS": 16, "DS": 17}, "biased_trials": int(len(biased)), "target": "learning_rare_minus_frequent_ms = mean RT(biased,Rare) - mean RT(biased,Frequent), ms",
                 "matrix_shape": list(conn.shape), "six_edge_counts": expected_edges.tolist(), "seed": SEED, "n_permutations": N_PERM, "rcrit": rcrit,
                 "observed_scalar_prediction_max_abs_error": max_scalar_error, "source_maps_replayed_exact": True,
                 "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__, "nilearn": __import__("nilearn").__version__, "nibabel": nib.__version__, "matplotlib": mpl.__version__}}
    (HERE / "execution_verification.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(execution, indent=2))

if __name__ == "__main__":
    main()
