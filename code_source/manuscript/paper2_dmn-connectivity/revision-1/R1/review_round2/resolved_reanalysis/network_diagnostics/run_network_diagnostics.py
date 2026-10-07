#!/usr/bin/env python3
"""Fresh common-coverage-399/Yeo-7 network, edge-screen, and enrichment diagnostics.

This program is deliberately blocked until the extraction handoff is complete for
all 33 authorized participants.  ``--synthetic-test`` creates labelled fake
inputs and is the only runnable mode before that handoff; its outputs are never
neural evidence.  CPM is intentionally not called or reproduced here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
HERE = Path(__file__).resolve().parent
RESOLVED = HERE.parent
EXTRACTION = RESOLVED / "extraction_common_coverage"
DEFAULT_MANIFEST = EXTRACTION / "inference_input_manifest.json"
DEFAULT_BEHAVIOR = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv"
NETWORK_ORDER = ("Default", "SomMot", "Vis", "Cont", "SalVentAttn", "DorsAttn", "Limbic")
DISPLAY = {"Default": "Default", "SomMot": "SomMot", "Vis": "Visual", "Cont": "Control", "SalVentAttn": "VAN/SN", "DorsAttn": "DAN", "Limbic": "Limbic"}
N_COMMON_PARCELS = 399
EXCLUDED_NATIVE_LABELS = frozenset({323})
COMMON_NATIVE_LABELS = frozenset(range(1, 401)) - EXCLUDED_NATIVE_LABELS
COUNTS = {"Default": 91, "SomMot": 77, "Vis": 61, "Cont": 52, "SalVentAttn": 47, "DorsAttn": 46, "Limbic": 25}
N = 33
EDGE_UNIVERSE = N_COMMON_PARCELS * (N_COMMON_PARCELS - 1) // 2
EDGE_HOLM_COLUMN = f"p_Holm_{EDGE_UNIVERSE}"
SCREEN_P = 0.01
SEED = 20261003


def fail(message: str) -> None:
    raise RuntimeError(message)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def resolve(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def need(manifest: dict, key: str, manifest_path: Path) -> Path:
    ensure(key in manifest and "path" in manifest[key], f"Manifest must contain {key}.path")
    path = resolve(str(manifest[key]["path"]), manifest_path)
    ensure(path.is_file(), f"Required input is absent: {path}")
    return path


def correlation_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Columnwise Pearson r for x:(participants, features), y:(participants,)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ensure(x.ndim == 2 and x.shape[0] == len(y), "Correlation shapes are invalid.")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y))
    ensure(np.isfinite(denominator).all() and np.all(denominator > 0), "A correlation input has zero/nonfinite variance.")
    return np.sum(x * y[:, None], axis=0) / denominator


def within_group_center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    ensure(values.shape[0] == len(groups), "Group-centering dimensions disagree.")
    out = values.copy()
    for group, expected in (("SS", 16), ("DS", 17)):
        ix = np.flatnonzero(groups == group)
        ensure(len(ix) == expected, f"Expected {group}={expected}, got {len(ix)}.")
        out[ix] -= out[ix].mean(axis=0, keepdims=True)
    return out


def group_adjusted_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int, float, float]:
    r = float(correlation_rows(np.asarray(x, float).reshape(-1, 1), y)[0])
    df = N - 3
    ensure(abs(r) < 1, "Correlation is exactly +/-1; inference is undefined.")
    t = r * np.sqrt(df / (1.0 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    se = 1.0 / np.sqrt(N - 4)
    z = np.arctanh(r)
    lo, hi = np.tanh(z + np.array([-1, 1]) * stats.norm.ppf(0.975) * se)
    return r, float(t), p, df, float(lo), float(hi)


def williams(a: float, b: float, c: float) -> tuple[float, float, int]:
    """Documented approximate dependent-correlation comparison, n_eff=N-1."""
    n_eff = N - 1
    determinant = 1 - a * a - b * b - c * c + 2 * a * b * c
    ensure(determinant > 0, "Williams triplet correlation matrix is not positive definite.")
    denominator = np.sqrt(2 * determinant * (n_eff - 1) / (n_eff - 3) + (a + b) ** 2 / 4 * (1 - c) ** 3)
    t = (a - b) * np.sqrt((n_eff - 1) * (1 + c)) / denominator
    df = n_eff - 3
    return float(t), float(2 * stats.t.sf(abs(t), df)), df


def pair_name(left: str, right: str) -> str:
    a, b = NETWORK_ORDER.index(left), NETWORK_ORDER.index(right)
    return f"{left}--{right}" if a <= b else f"{right}--{left}"


def validate_common_coverage_geometry(manifest: dict) -> None:
    geometry = manifest.get("common_coverage_geometry")
    if not isinstance(geometry, dict):
        fail("Manifest must contain common_coverage_geometry; stale full-400 handoffs are not accepted.")
    ensure(int(geometry.get("nominal_atlas_parcels", -1)) == 400, "Geometry must declare nominal_atlas_parcels=400.")
    ensure(int(geometry.get("effective_common_coverage_parcels", -1)) == N_COMMON_PARCELS, "Geometry must declare effective_common_coverage_parcels=399.")
    ensure(set(geometry.get("excluded_native_labels", [])) == EXCLUDED_NATIVE_LABELS, "Geometry must exclude exactly native label 323.")
    ensure(geometry.get("uniform_all_participants_and_runs") is True, "Geometry must attest uniform all-participant/all-run common coverage.")


def validated_handoff(manifest_path: Path) -> dict:
    ensure(manifest_path.is_file(), f"Input manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    ensure(manifest.get("status") == "complete", "Input manifest status must be exactly 'complete'.")
    validate_common_coverage_geometry(manifest)
    run_status = manifest_path.parent / "run_status.json"
    ensure(run_status.is_file(), f"Full-cohort run status is absent: {run_status}")
    status = json.loads(run_status.read_text())
    ensure(status.get("full_cohort_handoff_emitted") is True, "Extraction has not emitted the complete N=33 handoff.")
    ensure(int(status.get("n_requested", -1)) == 33, "Extraction handoff must report n_requested=33.")
    listed = set(status.get("stable_full_cohort_handoff_files", []))
    ensure({"participant_scores_resolved.csv", "primary_participant_matrices.npy", "primary_matrix_subjects.csv"} <= listed,
           "run_status does not declare all three stable full-cohort handoff files.")
    return manifest


def load_inputs(manifest_path: Path, synthetic: bool = False) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
    manifest = json.loads(manifest_path.read_text()) if synthetic else validated_handoff(manifest_path)
    if synthetic:
        ensure(manifest.get("status") == "complete" and manifest.get("synthetic_test_only") is True,
               "Synthetic mode requires a labelled complete synthetic manifest.")
        validate_common_coverage_geometry(manifest)
    score_path = need(manifest, "participant_scores", manifest_path)
    matrix_path = need(manifest, "primary_matrices", manifest_path)
    subject_path = need(manifest, "matrix_subjects", manifest_path)
    parcel_path = need(manifest, "parcel_metadata", manifest_path)
    score_cfg = manifest["participant_scores"]
    primary_column = score_cfg.get("primary_column")
    ensure(isinstance(primary_column, str) and primary_column, "Manifest must declare participant_scores.primary_column.")
    raw_scores = pd.read_csv(score_path)
    ensure({"subject", "group", primary_column} <= set(raw_scores), "Scores lack subject/group/fresh primary column.")
    scores = raw_scores[["subject", "group", primary_column]].rename(columns={primary_column: "primary_default"})
    ensure(len(scores) == N and scores.subject.is_unique and set(scores.group) == {"SS", "DS"}, "Scores must be unique N=33 SS/DS.")
    scores = scores.sort_values("subject").reset_index(drop=True)
    ensure(np.isfinite(scores.primary_default.to_numpy(float)).all(), "Primary score is nonfinite.")
    order = pd.read_csv(subject_path)
    ensure({"subject", "group"} <= set(order) and len(order) == N and order.subject.is_unique, "matrix_subjects must be unique N=33 subject/group rows.")
    matrices = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    ensure(matrices.shape == (N, N_COMMON_PARCELS, N_COMMON_PARCELS), f"Expected matrices (33,399,399), got {matrices.shape}.")
    position = {subject: i for i, subject in enumerate(order.subject.astype(str))}
    ensure(set(position) == set(scores.subject.astype(str)), "Matrix and score subject IDs differ.")
    ordered_groups = order.set_index("subject").loc[scores.subject, "group"].to_numpy()
    ensure(np.array_equal(ordered_groups, scores.group.to_numpy()), "Matrix and score group labels differ.")
    matrices = np.asarray(matrices[[position[subject] for subject in scores.subject]], dtype=float)
    ensure(np.isfinite(matrices).all(), "Matrices contain nonfinite values.")
    ensure(np.allclose(matrices, matrices.swapaxes(1, 2), atol=1e-8, rtol=0), "Matrices must be symmetric Fisher-z matrices.")
    ensure(np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 0.0, atol=1e-10, rtol=0), "Matrices must use diagonal zero.")
    parcels = pd.read_csv(parcel_path).sort_values("matrix_index").reset_index(drop=True)
    ensure({"matrix_index", "native_label", "yeo7_network"} <= set(parcels), "Parcel metadata needs matrix_index/native_label/yeo7_network.")
    ensure(len(parcels) == N_COMMON_PARCELS and parcels.matrix_index.tolist() == list(range(N_COMMON_PARCELS)), "matrix_index must be local 0..398 exactly.")
    ensure(set(parcels.native_label) == COMMON_NATIVE_LABELS and parcels.native_label.nunique() == N_COMMON_PARCELS, "native labels must be 1..400 excluding 323; local matrix_index is not a native label.")
    observed = parcels.yeo7_network.value_counts().to_dict()
    ensure(observed == COUNTS, f"Physical Yeo-7 counts disagree: {observed}.")
    provenance = {"input_manifest": str(manifest_path), "parcel_metadata_source": "manifest_common399_metadata", "input_hashes": {str(p): sha256(p) for p in (score_path, matrix_path, subject_path, parcel_path)}}
    return scores, matrices, parcels, provenance


def behavioral_capture(scores: pd.DataFrame, trials_path: Path) -> pd.DataFrame:
    ensure(trials_path.is_file(), f"Authorized behavior input is absent: {trials_path}")
    trials = pd.read_csv(trials_path)
    required = {"subject", "group", "RT_ms", "category", "correct"}
    ensure(required <= set(trials), f"Behavior trials missing {sorted(required - set(trials))}.")
    ensure(len(trials) == 10995 and trials.subject.nunique() == N, "Behavior input must be exactly the N33/10,995 retained trial set.")
    ensure((trials.correct == 1).all() and trials.RT_ms.between(200, 3000, inclusive="both").all(), "Behavior input fails authorized correct/RT gate.")
    means = trials.groupby(["subject", "group", "category"], observed=True).RT_ms.mean().unstack("category")
    ensure({"absent", "biased", "unbiased"} <= set(means), "Behavior categories must include absent, biased, unbiased.")
    capture = means.reset_index()
    capture["biased_capture"] = capture["biased"] - capture["absent"]
    joined = scores.merge(capture[["subject", "group", "biased_capture"]], on=["subject", "group"], validate="one_to_one")
    ensure(len(joined) == N and np.isfinite(joined.biased_capture.to_numpy(float)).all(), "Could not form finite N33 biased capture.")
    return joined


def network_summary_vectors(matrices: np.ndarray, parcels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    labels = parcels.yeo7_network.to_numpy()
    vectors: dict[str, np.ndarray] = {}
    rows = []
    for i, left in enumerate(NETWORK_ORDER):
        left_ix = np.flatnonzero(labels == left)
        ensure(len(left_ix) == COUNTS[left], f"Wrong count for {left}.")
        for right in NETWORK_ORDER[i:]:
            right_ix = np.flatnonzero(labels == right)
            if left == right:
                a = matrices[:, left_ix][:, :, left_ix]
                ii, jj = np.triu_indices(len(left_ix), 1)
                values = a[:, ii, jj].mean(axis=1)
                possible = len(ii)
                kind = "within"
            else:
                values = matrices[:, left_ix][:, :, right_ix].reshape(N, -1).mean(axis=1)
                possible = len(left_ix) * len(right_ix)
                kind = "between"
            name = pair_name(left, right)
            vectors[name] = values
            rows.append({"network_pair": name, "network_left": left, "network_right": right, "display_pair": f"{DISPLAY[left]}--{DISPLAY[right]}",
                         "kind": kind, "possible_edges": possible})
    summary = pd.DataFrame(rows)
    ensure(len(summary) == 28 and int(summary.possible_edges.sum()) == EDGE_UNIVERSE, f"Yeo7 summary universe must be 28 cells / {EDGE_UNIVERSE:,} edges.")
    return summary, vectors


def edge_table(matrices: np.ndarray, parcels: pd.DataFrame, capture: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    ii, jj = np.triu_indices(N_COMMON_PARCELS, 1)
    edge_values = matrices[:, ii, jj]
    residual_edges = within_group_center(edge_values, groups)
    residual_capture = within_group_center(capture, groups)
    r = correlation_rows(residual_edges, residual_capture)
    df = N - 3
    t = r * np.sqrt(df / (1 - r * r))
    p = 2 * stats.t.sf(np.abs(t), df)
    table = pd.DataFrame({
        "edge_index": np.arange(len(ii)), "matrix_i": ii, "matrix_j": jj,
        "native_label_i": parcels.native_label.to_numpy()[ii], "native_label_j": parcels.native_label.to_numpy()[jj],
        "network_i": parcels.yeo7_network.to_numpy()[ii], "network_j": parcels.yeo7_network.to_numpy()[jj],
        "network_pair": [pair_name(a, b) for a, b in zip(parcels.yeo7_network.to_numpy()[ii], parcels.yeo7_network.to_numpy()[jj])],
        "r_group_adjusted": r, "t": t, "df": df, "p_raw": p,
    })
    table[EDGE_HOLM_COLUMN] = multipletests(table.p_raw.to_numpy(float), method="holm")[1]
    table["screen_p_threshold"] = SCREEN_P
    table["selected_positive_raw_screen"] = (table.p_raw < SCREEN_P) & (table.r_group_adjusted > 0)
    table["selected_negative_raw_screen"] = (table.p_raw < SCREEN_P) & (table.r_group_adjusted < 0)
    ensure(len(table) == EDGE_UNIVERSE and table.edge_index.is_unique, f"Full screen must contain exactly {EDGE_UNIVERSE:,} unique edges.")
    return table


def enrichment(edgewise: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = len(edgewise)
    n_pos = int(edgewise.selected_positive_raw_screen.sum())
    n_neg = int(edgewise.selected_negative_raw_screen.sum())
    pair_rows = []
    for row in summary.itertuples(index=False):
        selected_positive = int(((edgewise.network_pair == row.network_pair) & edgewise.selected_positive_raw_screen).sum())
        selected_negative = int(((edgewise.network_pair == row.network_pair) & edgewise.selected_negative_raw_screen).sum())
        expected = n_pos * row.possible_edges / total
        p = float(stats.hypergeom.sf(selected_positive - 1, total, row.possible_edges, n_pos)) if n_pos else 1.0
        pair_rows.append({**row._asdict(), "universe_edges": total, "selected_positive": selected_positive, "selected_negative": selected_negative,
                          "selected_positive_share": selected_positive / n_pos if n_pos else 0.0, "universe_share": row.possible_edges / total,
                          "expected_positive": expected, "observed_expected_ratio": selected_positive / expected if expected else np.nan,
                          "hypergeom_p_enrichment_raw": p})
    pairs = pd.DataFrame(pair_rows)
    pairs["hypergeom_p_Holm_28"] = multipletests(pairs.hypergeom_p_enrichment_raw.to_numpy(float), method="holm")[1]
    ensure(len(pairs) == 28 and int(pairs.possible_edges.sum()) == total, "Pair enrichment arithmetic is incomplete.")
    endpoint_rows = []
    for network in NETWORK_ORDER:
        possible = int(((edgewise.network_i == network) | (edgewise.network_j == network)).sum())
        selected_positive = int((((edgewise.network_i == network) | (edgewise.network_j == network)) & edgewise.selected_positive_raw_screen).sum())
        selected_negative = int((((edgewise.network_i == network) | (edgewise.network_j == network)) & edgewise.selected_negative_raw_screen).sum())
        expected = n_pos * possible / total
        p = float(stats.hypergeom.sf(selected_positive - 1, total, possible, n_pos)) if n_pos else 1.0
        endpoint_rows.append({"network": network, "display_network": DISPLAY[network], "universe_edges": total, "possible_touching_edges": possible,
                              "selected_positive_touching": selected_positive, "selected_negative_touching": selected_negative,
                              "selected_positive_share": selected_positive / n_pos if n_pos else 0.0, "universe_share": possible / total,
                              "expected_positive": expected, "observed_expected_ratio": selected_positive / expected if expected else np.nan,
                              "hypergeom_p_enrichment_raw": p})
    endpoints = pd.DataFrame(endpoint_rows)
    endpoints["hypergeom_p_Holm_7"] = multipletests(endpoints.hypergeom_p_enrichment_raw.to_numpy(float), method="holm")[1]
    return pairs, endpoints


def network_associations(summary: pd.DataFrame, vectors: dict[str, np.ndarray], scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = scores.group.to_numpy()
    capture = scores.biased_capture.to_numpy(float)
    centered_capture = within_group_center(capture, groups)
    rows = []
    centered_vectors: dict[str, np.ndarray] = {}
    for row in summary.itertuples(index=False):
        value = within_group_center(vectors[row.network_pair], groups)
        centered_vectors[row.network_pair] = value
        r, t, p, df, lo, hi = group_adjusted_stats(value, centered_capture)
        rows.append({**row._asdict(), "N": N, "r_group_adjusted": r, "t": t, "df": df, "p_raw": p, "ci_low": lo, "ci_high": hi,
                     "CI_method": "Fisher-z approximate after one group contrast; df=N-3, SE=1/sqrt(N-4)"})
    associations = pd.DataFrame(rows)
    associations["p_Holm_28"] = multipletests(associations.p_raw.to_numpy(float), method="holm")[1]
    default_name = "Default--Default"
    default_r = float(associations.loc[associations.network_pair.eq(default_name), "r_group_adjusted"].iloc[0])
    comparisons = []
    for network in NETWORK_ORDER:
        if network == "Default":
            continue
        other = f"{network}--{network}"
        other_r = float(associations.loc[associations.network_pair.eq(other), "r_group_adjusted"].iloc[0])
        corr_networks = float(correlation_rows(centered_vectors[default_name].reshape(N, 1), centered_vectors[other])[0])
        t, p, df = williams(default_r, other_r, corr_networks)
        comparisons.append({"primary_network": "Default", "comparator_network": network, "primary_summary": default_name, "comparator_summary": other,
                            "r_capture_default": default_r, "r_capture_comparator": other_r, "r_default_comparator": corr_networks,
                            "delta_r_default_minus_comparator": default_r - other_r, "t_Williams": t, "df": df, "p_raw": p,
                            "method": "Approximate Williams comparison of overlapping group-centered correlations; n_eff=N-1=32, df=29"})
    comparisons = pd.DataFrame(comparisons)
    comparisons["p_Holm_6"] = multipletests(comparisons.p_raw.to_numpy(float), method="holm")[1]
    ensure(len(associations) == 28 and len(comparisons) == 6, "Network inferential families must be 28 and 6.")
    return associations, comparisons


def make_synthetic(destination: Path) -> tuple[Path, Path]:
    """Create a deterministic, explicitly fake N33 fixture for executable tests."""
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    subjects = [f"SS{i:03d}" for i in range(1, 17)] + [f"DS{i:03d}" for i in range(1, 18)]
    groups = np.array(["SS"] * 16 + ["DS"] * 17)
    rows, index = [], 0
    native_labels = iter(sorted(COMMON_NATIVE_LABELS))
    for network in NETWORK_ORDER:
        for _ in range(COUNTS[network]):
            rows.append({"matrix_index": index, "native_label": next(native_labels), "yeo7_network": network})
            index += 1
    parcels = pd.DataFrame(rows)
    ensure(len(parcels) == N_COMMON_PARCELS and 323 not in set(parcels.native_label), "Synthetic common-coverage metadata failure.")
    parcels.to_csv(destination / "physical_yeo7_common399_parcels.csv", index=False)
    latent = rng.normal(size=N)
    matrices = np.empty((N, N_COMMON_PARCELS, N_COMMON_PARCELS), dtype=np.float64)
    default_ix = np.flatnonzero(parcels.yeo7_network.to_numpy() == "Default")
    for person in range(N):
        noise = rng.normal(0, 0.075, size=(N_COMMON_PARCELS, N_COMMON_PARCELS))
        matrix = (noise + noise.T) / 2
        matrix[np.ix_(default_ix, default_ix)] += latent[person] * 0.017
        np.fill_diagonal(matrix, 0.0)
        matrices[person] = matrix
    np.save(destination / "primary_participant_matrices.npy", matrices)
    pd.DataFrame({"subject": subjects, "group": groups}).to_csv(destination / "primary_matrix_subjects.csv", index=False)
    tri_i, tri_j = np.triu_indices(len(default_ix), 1)
    primary = matrices[:, default_ix][:, :, default_ix][:, tri_i, tri_j].mean(axis=1)
    pd.DataFrame({"subject": subjects, "group": groups, "primary_default": primary}).to_csv(destination / "participant_scores_resolved.csv", index=False)
    records = []
    for person, (subject, group) in enumerate(zip(subjects, groups)):
        count = 334 if person < 6 else 333
        for trial in range(count):
            category = ("absent", "biased", "unbiased")[trial % 3]
            offset = 32 * latent[person] if category == "biased" else (12 if category == "unbiased" else 0)
            records.append({"subject": subject, "group": group, "category": category, "correct": 1, "RT_ms": 800 + offset + rng.normal(0, 50)})
    trials = pd.DataFrame(records)
    ensure(len(trials) == 10995, "Synthetic trial fixture must have 10,995 rows.")
    trials_path = destination / "synthetic_authorized_trials.csv"
    trials.to_csv(trials_path, index=False)
    manifest = {"status": "complete", "synthetic_test_only": True,
                "participant_scores": {"path": "participant_scores_resolved.csv", "primary_column": "primary_default"},
                "common_coverage_geometry": {"nominal_atlas_parcels": 400, "effective_common_coverage_parcels": N_COMMON_PARCELS, "excluded_native_labels": sorted(EXCLUDED_NATIVE_LABELS), "uniform_all_participants_and_runs": True},
                "primary_matrices": {"path": "primary_participant_matrices.npy"}, "matrix_subjects": {"path": "primary_matrix_subjects.csv"},
                "parcel_metadata": {"path": "physical_yeo7_common399_parcels.csv"}}
    manifest_path = destination / "network_diagnostics_input_manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path, trials_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_MANIFEST, help="Complete N33 extraction manifest; real mode rejects partial extraction.")
    parser.add_argument("--behavior-trials", type=Path, default=DEFAULT_BEHAVIOR, help="Authorized correct/RT-valid N33 trial file.")
    parser.add_argument("--output", type=Path, default=HERE / "results", help="New empty output directory.")
    parser.add_argument("--synthetic-test", action="store_true", help="Use only generated fake matrices/trials; output is labelled not evidence.")
    args = parser.parse_args()
    output = args.output
    if args.synthetic_test:
        output = output / "SYNTHETIC_TEST_NOT_EVIDENCE"
        inputs = output.parent / (output.name + "_inputs")
        args.input_manifest, args.behavior_trials = make_synthetic(inputs)
    ensure(not output.exists(), f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    try:
        scores, matrices, parcels, provenance = load_inputs(args.input_manifest, synthetic=args.synthetic_test)
        scores = behavioral_capture(scores, args.behavior_trials)
        groups = scores.group.to_numpy()
        summary, vectors = network_summary_vectors(matrices, parcels)
        primary_fresh = vectors["Default--Default"]
        ensure(np.allclose(primary_fresh, scores.primary_default.to_numpy(float), atol=1e-8, rtol=0),
               "Fresh primary score does not equal all-common-coverage physical Yeo7 Default=91 mean.")
        associations, comparisons = network_associations(summary, vectors, scores)
        edges = edge_table(matrices, parcels, scores.biased_capture.to_numpy(float), groups)
        pairs, endpoints = enrichment(edges, summary)
        parcels.to_csv(output / "physical_yeo7_common399_parcels_used.csv", index=False)
        scores[["subject", "group"]].assign(**{name: vectors[name] for name in summary.network_pair}).to_csv(output / "yeo7_network_summary_scores_n33.csv", index=False)
        associations.to_csv(output / "yeo7_all_28_summary_associations_holm28.csv", index=False)
        comparisons.to_csv(output / "yeo7_default_vs_other_within_network_williams_holm6.csv", index=False)
        edges.to_csv(output / f"edgewise_group_adjusted_allcommon{EDGE_UNIVERSE}_holm.csv", index=False)
        edges.loc[edges.selected_positive_raw_screen | edges.selected_negative_raw_screen].to_csv(output / "edgewise_raw_p001_screen_selected_edges.csv", index=False)
        pairs.to_csv(output / "yeo7_pair_enrichment_hypergeom_holm28.csv", index=False)
        endpoints.to_csv(output / "yeo7_endpoint_enrichment_hypergeom_holm7.csv", index=False)
        screen_summary = pd.DataFrame([{"edge_universe": len(edges), "edge_df": N - 3, "screen_threshold_two_sided_p": SCREEN_P,
                                        "selected_positive_raw_screen": int(edges.selected_positive_raw_screen.sum()), "selected_negative_raw_screen": int(edges.selected_negative_raw_screen.sum()),
                                        "selected_any_raw_screen": int((edges.selected_positive_raw_screen | edges.selected_negative_raw_screen).sum()),
                                        f"positive_Holm_{EDGE_UNIVERSE}_lt_05": int((edges.loc[edges.r_group_adjusted > 0, EDGE_HOLM_COLUMN] < .05).sum()),
                                        f"negative_Holm_{EDGE_UNIVERSE}_lt_05": int((edges.loc[edges.r_group_adjusted < 0, EDGE_HOLM_COLUMN] < .05).sum()),
                                        "minimum_p_raw": float(edges.p_raw.min()), f"minimum_p_Holm_{EDGE_UNIVERSE}": float(edges[EDGE_HOLM_COLUMN].min()),
                                        "interpretation": f"Raw p<.01 selection is exploratory visualization only; no q-values are produced; Holm spans all {EDGE_UNIVERSE:,} correlations."}])
        screen_summary.to_csv(output / f"edgewise_allcommon{EDGE_UNIVERSE}_screen_summary.csv", index=False)
        manifest = {"status": "SYNTHETIC_TEST_NOT_EVIDENCE" if args.synthetic_test else "complete", "scope": "Fresh physical Schaefer nominal-400 / common-coverage-399 Yeo7 network diagnostics only; CPM excluded.",
                    "N": N, "groups": {"SS": 16, "DS": 17}, "network_counts": COUNTS, "matrix_shape": [N, N_COMMON_PARCELS, N_COMMON_PARCELS], "common_coverage_geometry": {"nominal_atlas_parcels": 400, "effective_common_coverage_parcels": N_COMMON_PARCELS, "excluded_native_labels": sorted(EXCLUDED_NATIVE_LABELS)},
                    "primary_contract": "Default=91 mean upper-triangle Fisher-z from all-common-coverage 399-parcel physical Yeo7 matrices.",
                    "network_association_family": "28 group-adjusted (df=30) Yeo7 within/between summaries, fresh Holm-28.",
                    "network_comparison_family": "6 approximate Williams Default-versus-other-within-network comparisons, n_eff=32 df=29, Holm-6.",
                    "edge_family": f"{EDGE_UNIVERSE:,} group-centered edge correlations, df=30, raw p plus Holm-{EDGE_UNIVERSE:,}; raw p<.01 screen is exploratory.",
                    "enrichment_families": "One-sided hypergeometric selected-positive enrichment conditional on raw screen: 28 mutually exclusive pair cells Holm-28; 7 overlapping endpoint cells Holm-7. No BH/q values.",
                    "input": provenance, "behavior_trials": str(args.behavior_trials), "behavior_trials_sha256": sha256(args.behavior_trials),
                    "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
                    "outputs": {"summary_scores": "yeo7_network_summary_scores_n33.csv", "summary_associations": "yeo7_all_28_summary_associations_holm28.csv", "comparisons": "yeo7_default_vs_other_within_network_williams_holm6.csv", "edgewise": f"edgewise_group_adjusted_allcommon{EDGE_UNIVERSE}_holm.csv", "raw_screen": "edgewise_raw_p001_screen_selected_edges.csv", "screen_summary": f"edgewise_allcommon{EDGE_UNIVERSE}_screen_summary.csv", "pair_enrichment": "yeo7_pair_enrichment_hypergeom_holm28.csv", "endpoint_enrichment": "yeo7_endpoint_enrichment_hypergeom_holm7.csv"}}
        atomic_json(output / "network_diagnostics_manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "output": str(output), "edge_universe": len(edges), "network_summary_tests": len(associations), "network_comparisons": len(comparisons), "selected_positive": int(edges.selected_positive_raw_screen.sum()), "selected_negative": int(edges.selected_negative_raw_screen.sum())}, indent=2))
    except Exception:
        # Do not leave a directory that could be mistaken for completed inference.
        if output.exists() and not (output / "network_diagnostics_manifest.json").exists():
            for child in output.iterdir():
                child.unlink()
            output.rmdir()
        raise


if __name__ == "__main__":
    main()
