#!/usr/bin/env python3
"""Independent verifier for frozen rare-minus-frequent CPM; never imports its runner."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
from pathlib import Path
import hashlib
import json
import platform

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
TARGET = PARENT / "learning_target_scores.csv"
TRIALS = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv"
MATRICES = RES / "extraction_common_coverage/primary_participant_matrices.npy"
MATRIX_SUBJECTS = RES / "extraction_common_coverage/primary_matrix_subjects.csv"
PARCELS = RES / "common399_analysis_ready/physical_yeo7_common399_parcels.csv"
FREEZE = PARENT / "input_freeze.json"
ARCHIVE = HERE / "null_archive"
FAMILIES = ["DefaultA", "A-B", "A-C", "DefaultB", "B-C", "DefaultC"]
EXPECTED_EDGES = np.array([561, 1088, 442, 496, 416, 78])
SEED, N_PERM, BATCH = 20261004, 10000, 50


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    stepped = (len(p) - np.arange(len(p))) * p[order]
    stepped = np.maximum.accumulate(stepped)
    out = np.empty_like(stepped)
    out[order] = np.minimum(stepped, 1.0)
    return out


def independent_family_map(parcels):
    """Rebuild cells from physical atlas maps and frozen common399 parcel order."""
    atlas7 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2, verbose=0)
    atlas17 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2, verbose=0)
    image7, image17 = nib.load(atlas7.maps), nib.load(atlas17.maps)
    raw7 = np.asanyarray(image7.dataobj).astype(np.int16).ravel()
    raw17 = np.asanyarray(image17.dataobj).astype(np.int16).ravel()
    pairs = np.unique(np.c_[raw7[raw7 > 0], raw17[raw7 > 0]], axis=0)
    if pairs.shape != (400, 2) or not np.array_equal(raw7 > 0, raw17 > 0):
        raise RuntimeError("Invalid independent atlas physical crosswalk")
    names7 = [x.decode() if isinstance(x, bytes) else str(x) for x in atlas7.labels]
    names17 = [x.decode() if isinstance(x, bytes) else str(x) for x in atlas17.labels]
    if names7[0].lower() == "background": names7 = names7[1:]
    if names17[0].lower() == "background": names17 = names17[1:]
    lookup = {int(a): int(b) for a, b in pairs}
    parcels = parcels.sort_values("matrix_index").reset_index(drop=True)
    native = parcels.native_label.to_numpy(int)
    if len(parcels) != 399 or not np.array_equal(parcels.matrix_index, np.arange(399)) or set(native) != set(range(1, 401)) - {323}:
        raise RuntimeError("Common399 metadata invalid")
    label7 = np.array([names7[x - 1] for x in native], dtype=object)
    label17 = np.array([names17[lookup[x] - 1] for x in native], dtype=object)
    if not np.array_equal(label7, parcels.physical_label.to_numpy(str)):
        raise RuntimeError("Independent Yeo7 crosswalk conflicts with frozen metadata")
    n17 = np.array([x.split("_")[2] for x in label17], dtype=object)
    upper_i, upper_j = np.triu_indices(399, 1)
    a, b = n17[upper_i], n17[upper_j]
    which = np.full(len(a), -1, np.int8)
    pairs17 = [("DefaultA", "DefaultA"), ("DefaultA", "DefaultB"), ("DefaultA", "DefaultC"),
               ("DefaultB", "DefaultB"), ("DefaultB", "DefaultC"), ("DefaultC", "DefaultC")]
    for code, (one, two) in enumerate(pairs17):
        hit = ((a == one) & (b == two)) | ((a == two) & (b == one))
        if np.any(which[hit] != -1): raise RuntimeError("Overlapping independent CPM cells")
        which[hit] = code
    keep = which >= 0
    fam = which[keep]
    if not np.array_equal(np.bincount(fam, minlength=6), EXPECTED_EDGES):
        raise RuntimeError("Independent CPM cell edge counts mismatch")
    cross = pd.DataFrame({"matrix_index": np.arange(399), "native_label": native, "yeo17_network": n17})
    return keep, fam, cross


def prepare_X(X):
    folds = []
    for heldout in range(len(X)):
        train = np.arange(len(X)) != heldout
        train_X = X[train]
        sd = train_X.std(axis=0, ddof=1)
        z = (train_X - train_X.mean(axis=0)) / np.where(sd == 0, 1., sd)
        folds.append((train, train_X, X[heldout], z))
    return folds


def centered_outcomes(raw, groups, heldout):
    """Independent explicit train-only group centering for an N x B target matrix."""
    train = np.arange(len(groups)) != heldout
    y_train = raw[train].copy()
    y_test = raw[heldout].copy()
    g_train = groups[train]
    for group in ("DS", "SS"):
        loc = g_train == group
        mean = y_train[loc].mean(axis=0, keepdims=True)
        y_train[loc] -= mean
        if groups[heldout] == group:
            y_test = y_test - mean[0]
    return y_train, y_test


def evaluate_vectorized(raw, groups, family, folds, threshold):
    """Independent batched nested LOOCV CPM, including selection/fitting/refitting."""
    n, b = raw.shape
    predicted = np.zeros((b, n, 6))
    assessed = np.zeros((b, n))
    for heldout, (_, x_train, x_test, zx_train) in enumerate(folds):
        y_train, y_test = centered_outcomes(raw, groups, heldout)
        assessed[:, heldout] = y_test
        y_std = y_train.std(axis=0, ddof=1)
        if np.any(y_std <= 0): raise RuntimeError("Zero variance permutation target")
        edge_r = zx_train.T @ (y_train / y_std) / (len(y_train) - 1)
        for cell in range(6):
            signs = np.where(np.abs(edge_r[family == cell]) > threshold, np.sign(edge_r[family == cell]), 0.)
            x_cell = x_train[:, family == cell]
            strength = x_cell @ signs
            center = strength.mean(axis=0)
            centered = strength - center
            denom = (centered * centered).sum(axis=0)
            slopes = np.divide((centered * y_train).sum(axis=0), denom, out=np.zeros(b), where=denom > 1e-14)
            test_strength = x_test[family == cell] @ signs
            predicted[:, heldout, cell] = slopes * (test_strength - center)
    cy = assessed - assessed.mean(axis=1, keepdims=True)
    cp = predicted - predicted.mean(axis=1, keepdims=True)
    denominator = np.sqrt((cy * cy).sum(axis=1)[:, None] * (cp * cp).sum(axis=1))
    return np.divide((cy[:, :, None] * cp).sum(axis=1), denominator, out=np.zeros((b, 6)), where=denominator > 1e-14), predicted, assessed


def scalar_observed(X, y, groups, family, threshold):
    """One edge at a time; this does not share the verifier's vectorized logic."""
    pred = np.zeros((len(y), 6))
    target = np.zeros(len(y))
    for heldout in range(len(y)):
        train = np.arange(len(y)) != heldout
        yy, xx, gg = y[train].copy(), X[train], groups[train]
        for group in ("DS", "SS"):
            mean = yy[gg == group].mean()
            yy[gg == group] -= mean
            if groups[heldout] == group: target[heldout] = y[heldout] - mean
        for cell in range(6):
            xcell = xx[:, family == cell]
            r = np.array([stats.pearsonr(xcell[:, edge], yy).statistic if xcell[:, edge].std() > 0 else 0. for edge in range(xcell.shape[1])])
            signed = np.where(np.abs(r) > threshold, np.sign(r), 0.)
            net = xcell @ signed
            held = X[heldout, family == cell] @ signed
            if ((net - net.mean()) ** 2).sum() > 1e-14:
                coeff = np.linalg.lstsq(np.column_stack((np.ones(len(net)), net)), yy, rcond=None)[0]
                pred[heldout, cell] = coeff[0] + coeff[1] * held
    return pred, target


def recreate_maps(groups):
    ds, ss = np.flatnonzero(groups == "DS"), np.flatnonzero(groups == "SS")
    generator = np.random.default_rng(SEED)
    ds_map = np.empty((N_PERM, 17), np.int16)
    ss_map = np.empty((N_PERM, 16), np.int16)
    for draw in range(N_PERM):
        ds_map[draw] = generator.permutation(17)
        ss_map[draw] = generator.permutation(16)
    return ds, ss, ds_map, ss_map


def targets_from_maps(y, ds, ss, dmap, smap, low, high):
    target = np.empty((33, high-low))
    target[ds] = y[ds][dmap[low:high]].T
    target[ss] = y[ss][smap[low:high]].T
    return target


def main():
    freeze = json.loads(FREEZE.read_text())
    for path, fingerprint in freeze["input_hashes"].items():
        if sha256(path) != fingerprint: raise RuntimeError(f"Frozen input hash changed: {path}")
    if sha256(TARGET) != freeze["target_hash"]: raise RuntimeError("Target hash changed")
    target = pd.read_csv(TARGET, dtype={"subject": str})
    subject_order = pd.read_csv(MATRIX_SUBJECTS, dtype={"subject": str})
    if not np.array_equal(target[["subject", "group"]].to_numpy(str), subject_order[["subject", "group"]].to_numpy(str)):
        raise RuntimeError("Target/matrix identity mismatch")
    trials = pd.read_csv(TRIALS, dtype={"subject": str})
    valid = trials.correct.eq(1) & np.isfinite(trials.RT_ms) & (trials.RT_ms > 200) & (trials.RT_ms <= 3000)
    b = trials[valid & trials.category.eq("biased")]
    rebuilt = b.groupby(["subject", "DisRegion"]).RT_ms.mean().unstack("DisRegion").reindex(target.subject.to_numpy())
    reconstructed = rebuilt.Rare.to_numpy() - rebuilt.Frequent.to_numpy()
    target_error = float(np.max(np.abs(reconstructed - target.learning_rare_minus_frequent_ms.to_numpy(float))))
    if len(b) != 6721 or target_error > 1e-10: raise RuntimeError("Independent target reconstruction failed")
    union, family, cross = independent_family_map(pd.read_csv(PARCELS))
    saved_cross = pd.read_csv(HERE / "physical_yeo7_to_17_crosswalk_common399.csv")
    if not np.array_equal(cross.native_label.to_numpy(), saved_cross.native_label.to_numpy()) or not np.array_equal(cross.yeo17_network.to_numpy(), saved_cross.yeo17_network.to_numpy()):
        raise RuntimeError("Saved runner crosswalk differs from independent reconstruction")
    conn = np.load(MATRICES, mmap_mode="r")
    triu = np.triu_indices(399, 1)
    X = np.asarray(conn[:, triu[0], triu[1]][:, union], dtype=float)
    y, groups = target.learning_rare_minus_frequent_ms.to_numpy(float), target.group.to_numpy(str)
    tcritical = stats.t.ppf(.995, 30)
    rcritical = float(tcritical / np.sqrt(tcritical*tcritical + 30))
    folds = prepare_X(X)
    r_observed, vector_pred, vector_target = evaluate_vectorized(y[:, None], groups, family, folds, rcritical)
    r_observed, vector_pred, vector_target = r_observed[0], vector_pred[0], vector_target[0]
    scalar_pred, scalar_target = scalar_observed(X, y, groups, family, rcritical)
    scalar_prediction_error = float(np.max(np.abs(scalar_pred - vector_pred)))
    if scalar_prediction_error > 1e-8 or not np.allclose(scalar_target, vector_target, atol=1e-10):
        raise RuntimeError("Independent scalar-versus-vector observed verifier mismatch")
    runner_pred = pd.read_csv(HERE / "cpm_prediction_data_learning_rare_minus_frequent.csv", dtype={"subject": str})
    if not np.array_equal(runner_pred.subject.to_numpy(), target.subject.to_numpy()): raise RuntimeError("Saved prediction subject order differs")
    saved_pred = runner_pred[[f"predicted_{name}_ms" for name in FAMILIES]].to_numpy(float)
    observed_prediction_error = float(np.max(np.abs(saved_pred - scalar_pred)))
    saved_target_error = float(np.max(np.abs(runner_pred.observed_fold_centered_learning_ms.to_numpy() - scalar_target)))
    if observed_prediction_error > 1e-8 or saved_target_error > 1e-10: raise RuntimeError("Saved observed predictions fail independent scalar check")
    ds, ss, ds_map, ss_map = recreate_maps(groups)
    disk_ds, disk_ss = np.load(ARCHIVE / "ds_source_indices.npy"), np.load(ARCHIVE / "ss_source_indices.npy")
    maps_exact = bool(np.array_equal(ds_map, disk_ds) and np.array_equal(ss_map, disk_ss))
    if not maps_exact: raise RuntimeError("Archived within-group maps do not replay from frozen seed")
    replay = np.empty((N_PERM, 6))
    for low in range(0, N_PERM, BATCH):
        high = min(low + BATCH, N_PERM)
        perm_y = targets_from_maps(y, ds, ss, ds_map, ss_map, low, high)
        replay[low:high] = evaluate_vectorized(perm_y, groups, family, folds, rcritical)[0]
        print(f"independent replay {high}/{N_PERM}", flush=True)
    archived = np.load(ARCHIVE / "permutation_r.npy")
    full_replay_error = float(np.max(np.abs(replay - archived)))
    if full_replay_error > 1e-10: raise RuntimeError("Full independent permutation replay disagrees with archive")
    p_upper = (replay >= r_observed).sum(axis=0)
    p_lower = (replay <= r_observed).sum(axis=0)
    p_equal = np.minimum(1., 2 * np.minimum((p_upper + 1)/(N_PERM + 1), (p_lower + 1)/(N_PERM + 1)))
    p_abs = ((np.abs(replay) >= np.abs(r_observed)).sum(axis=0) + 1) / (N_PERM + 1)
    saved_summary = pd.read_csv(HERE / "table_cpm_learning_rare_minus_frequent.csv")
    checks = {"observed_r": float(np.max(np.abs(saved_summary.r_loocv.to_numpy() - r_observed))),
              "equal_tail_p": float(np.max(np.abs(saved_summary.p_raw_equal_tail.to_numpy() - p_equal))),
              "absolute_tail_p": float(np.max(np.abs(saved_summary.p_raw_absolute_sensitivity.to_numpy() - p_abs))),
              "equal_holm": float(np.max(np.abs(saved_summary.p_Holm6_equal_tail.to_numpy() - holm(p_equal)))),
              "absolute_holm": float(np.max(np.abs(saved_summary.p_Holm6_absolute_sensitivity.to_numpy() - holm(p_abs))))}
    if max(checks.values()) > 1e-12: raise RuntimeError(f"Independent summary/tail/Holm mismatch: {checks}")
    per_draw = pd.DataFrame(replay, columns=[f"replayed_r_{name}" for name in FAMILIES])
    per_draw.insert(0, "draw", np.arange(N_PERM))
    per_draw["max_abs_difference_from_archived"] = np.max(np.abs(replay - archived), axis=1)
    per_draw.to_csv(HERE / "independent_verification_per_draw_summaries.csv", index=False)
    manifest = json.loads((HERE / "null_archive_manifest.json").read_text())
    archive_hashes_ok = all(sha256(HERE / relative) == known for relative, known in manifest["archive_files_sha256"].items())
    if not archive_hashes_ok: raise RuntimeError("Null archive hashes fail manifest validation")
    result = {"status": "PASS", "method": "standalone verifier; no runner or inherited-helper imports", "n": 33, "groups": {"DS": 17, "SS": 16}, "biased_trials": int(len(b)),
              "target_reconstruction_max_abs_error": target_error, "six_cell_edge_counts": EXPECTED_EDGES.tolist(), "rcrit": rcritical,
              "observed_scalar_vs_independent_vector_max_abs_error": scalar_prediction_error,
              "saved_observed_scalar_prediction_max_abs_error": observed_prediction_error, "saved_fold_target_max_abs_error": saved_target_error,
              "archived_source_maps_recreated_exactly": maps_exact, "full_10000_draw_replay_max_abs_error": full_replay_error,
              "tail_and_holm_max_abs_errors": checks, "archive_hashes_match_manifest": archive_hashes_ok,
              "per_draw_summary_rows": int(len(per_draw)), "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__}}
    (HERE / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
