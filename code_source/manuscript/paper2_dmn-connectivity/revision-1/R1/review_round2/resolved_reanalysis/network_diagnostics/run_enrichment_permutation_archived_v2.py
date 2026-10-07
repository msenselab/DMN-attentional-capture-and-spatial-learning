#!/usr/bin/env python3
"""Frozen common399 enrichment permutation with permanent draw archive.

This is an archival repair of run_enrichment_permutation.py: it preserves its
seed, group order, per-draw RNG calls, screen, statistic, and Holm families.
It adds a durable null archive and resumable checkpoints; it makes no
publication claim.
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

HERE = Path(__file__).resolve().parent
RESOLVED = HERE.parent
ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
DEFAULT_MANIFEST = RESOLVED / "common399_analysis_ready/analysis_ready_input_manifest.json"
DEFAULT_BEHAVIOR = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv"
ORIGINAL_RUNNER = HERE / "run_enrichment_permutation.py"
N, N_COMMON = 33, 399
EDGE_UNIVERSE = N_COMMON * (N_COMMON - 1) // 2
EXCLUDED_NATIVE = frozenset({323})
NATIVE_COMMON = frozenset(range(1, 401)) - EXCLUDED_NATIVE
ORDER = ("Default", "SomMot", "Vis", "Cont", "SalVentAttn", "DorsAttn", "Limbic")
COUNTS = {"Default": 91, "SomMot": 77, "Vis": 61, "Cont": 52, "SalVentAttn": 47, "DorsAttn": 46, "Limbic": 25}
SEED, SCREEN_P, DF = 20261001, 0.01, 30
ARCHIVE_VERSION = "enrichment-null-archive-v2"


def fail(message: str) -> None:
    raise RuntimeError(message)


def ensure(ok: bool, message: str) -> None:
    if not ok:
        fail(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def resolve(value: str, manifest: Path) -> Path:
    value_path = Path(value)
    return value_path if value_path.is_absolute() else manifest.parent / value_path


def required_path(manifest: dict, key: str, manifest_path: Path) -> Path:
    ensure(key in manifest and "path" in manifest[key], f"Manifest lacks {key}.path.")
    path = resolve(str(manifest[key]["path"]), manifest_path)
    ensure(path.is_file(), f"Required input absent: {path}")
    return path


def pair_name(left: str, right: str) -> str:
    return f"{left}--{right}" if ORDER.index(left) <= ORDER.index(right) else f"{right}--{left}"


def validate_geometry(manifest: dict) -> None:
    geometry = manifest.get("common_coverage_geometry")
    ensure(isinstance(geometry, dict), "Manifest must contain common_coverage_geometry; a full-400 matrix is rejected.")
    ensure(int(geometry.get("nominal_atlas_parcels", -1)) == 400, "Geometry nominal atlas must be 400.")
    ensure(int(geometry.get("effective_common_coverage_parcels", -1)) == N_COMMON, "Geometry effective common coverage must be 399.")
    ensure(set(geometry.get("excluded_native_labels", [])) == EXCLUDED_NATIVE, "Geometry must exclude native label 323 only.")
    ensure(geometry.get("uniform_all_participants_and_runs") is True, "Geometry must attest uniform all-participant/all-run coverage.")


def load_inputs(manifest_path: Path, behavior_path: Path, synthetic: bool) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, dict]:
    ensure(manifest_path.is_file(), f"Input manifest absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    ensure(manifest.get("status") == "complete", "Input manifest status must be complete.")
    if synthetic:
        ensure(manifest.get("synthetic_test_only") is True, "Synthetic mode requires a labelled synthetic manifest.")
    validate_geometry(manifest)
    score_path, matrix_path = required_path(manifest, "participant_scores", manifest_path), required_path(manifest, "primary_matrices", manifest_path)
    subject_path, parcel_path = required_path(manifest, "matrix_subjects", manifest_path), required_path(manifest, "parcel_metadata", manifest_path)
    score_column = manifest["participant_scores"].get("primary_column")
    ensure(isinstance(score_column, str) and score_column, "Manifest must name the fresh Default91 primary column.")
    scores = pd.read_csv(score_path)
    ensure({"subject", "group", score_column} <= set(scores), "Scores lack subject/group/fresh primary column.")
    scores = scores[["subject", "group", score_column]].sort_values("subject").reset_index(drop=True)
    ensure(len(scores) == N and scores.subject.is_unique and scores.group.value_counts().to_dict() == {"DS": 17, "SS": 16}, "Scores must be N33 SS16/DS17.")
    subjects = pd.read_csv(subject_path)
    ensure({"subject", "group"} <= set(subjects) and len(subjects) == N and subjects.subject.is_unique, "Matrix subject table must be unique N33.")
    position = {subject: i for i, subject in enumerate(subjects.subject.astype(str))}
    ensure(set(position) == set(scores.subject.astype(str)), "Score/matrix subject IDs differ.")
    ensure(np.array_equal(subjects.set_index("subject").loc[scores.subject, "group"].to_numpy(), scores.group.to_numpy()), "Score/matrix group labels differ.")
    matrices = np.asarray(np.load(matrix_path, mmap_mode="r", allow_pickle=False)[[position[s] for s in scores.subject]], float)
    ensure(matrices.shape == (N, N_COMMON, N_COMMON), f"Expected (33,399,399) matrices, got {matrices.shape}.")
    ensure(np.isfinite(matrices).all() and np.allclose(matrices, matrices.swapaxes(1, 2), atol=1e-8, rtol=0), "Matrices must be finite symmetric Fisher-z arrays.")
    ensure(np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 0, atol=1e-10, rtol=0), "Matrices must have zero diagonal.")
    parcels = pd.read_csv(parcel_path).sort_values("matrix_index").reset_index(drop=True)
    ensure({"matrix_index", "native_label", "yeo7_network"} <= set(parcels), "Parcel metadata lacks local/native/network fields.")
    ensure(parcels.matrix_index.tolist() == list(range(N_COMMON)), "Parcel local matrix_index must be 0..398.")
    ensure(set(parcels.native_label) == NATIVE_COMMON and parcels.native_label.nunique() == N_COMMON, "Parcel native labels must be 1..400 excluding 323.")
    ensure(parcels.yeo7_network.value_counts().to_dict() == COUNTS, "Common-coverage Yeo7 counts failed.")
    trials = pd.read_csv(behavior_path)
    required = {"subject", "group", "RT_ms", "category", "correct"}
    ensure(required <= set(trials), f"Behavior trials missing {sorted(required - set(trials))}.")
    ensure(len(trials) == 10995 and trials.subject.nunique() == N, "Behavior must be the authorized N33/10,995 trial table.")
    ensure((trials.correct == 1).all() and (trials.RT_ms > 200).all() and (trials.RT_ms <= 3000).all(), "Behavior must use the current correct RT>200 and <=3000 gate.")
    means = trials.groupby(["subject", "group", "category"], observed=True).RT_ms.mean().unstack("category").reset_index()
    ensure({"absent", "biased"} <= set(means), "Behavior categories must include absent and biased.")
    capture = scores[["subject", "group"]].merge(means.assign(biased_capture=means.biased - means.absent)[["subject", "group", "biased_capture"]], on=["subject", "group"], validate="one_to_one")
    ensure(np.isfinite(capture.biased_capture.to_numpy(float)).all(), "Biased capture is nonfinite.")
    hashes = {str(path): digest(path) for path in (score_path, matrix_path, subject_path, parcel_path, behavior_path)}
    return matrices, scores.group.to_numpy(), parcels, capture.biased_capture.to_numpy(float), {"manifest": str(manifest_path), "hashes": hashes}


def within_group_center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    output = np.asarray(values, float).copy()
    for group, expected in (("SS", 16), ("DS", 17)):
        indices = np.flatnonzero(groups == group)
        ensure(len(indices) == expected, f"Expected {group}={expected}.")
        output[indices] -= output[indices].mean(axis=0, keepdims=True)
    return output


def make_synthetic(destination: Path) -> tuple[Path, Path]:
    """Exact deterministic fixture copied from the frozen runner; never empirical."""
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(71)
    subjects = [f"SS{i:03d}" for i in range(1, 17)] + [f"DS{i:03d}" for i in range(1, 18)]
    groups = np.array(["SS"] * 16 + ["DS"] * 17)
    rows, native = [], iter(sorted(NATIVE_COMMON))
    for network in ORDER:
        for _ in range(COUNTS[network]): rows.append({"matrix_index": len(rows), "native_label": next(native), "yeo7_network": network})
    parcels = pd.DataFrame(rows); parcels.to_csv(destination / "physical_yeo7_common399_parcels.csv", index=False)
    latent = rng.normal(size=N); matrices = np.empty((N, N_COMMON, N_COMMON)); default = np.flatnonzero(parcels.yeo7_network.to_numpy() == "Default")
    for person in range(N):
        noise = rng.normal(0, .075, size=(N_COMMON, N_COMMON)); matrix = (noise + noise.T) / 2
        matrix[np.ix_(default, default)] += .017 * latent[person]; np.fill_diagonal(matrix, 0); matrices[person] = matrix
    np.save(destination / "primary_participant_matrices.npy", matrices)
    pd.DataFrame({"subject": subjects, "group": groups}).to_csv(destination / "primary_matrix_subjects.csv", index=False)
    tri_i, tri_j = np.triu_indices(91, 1); primary = matrices[:, default][:, :, default][:, tri_i, tri_j].mean(axis=1)
    pd.DataFrame({"subject": subjects, "group": groups, "primary_default": primary}).to_csv(destination / "participant_scores_resolved.csv", index=False)
    trial_rows = []
    for person, (subject, group) in enumerate(zip(subjects, groups)):
        for trial in range(334 if person < 6 else 333):
            category = ("absent", "biased", "unbiased")[trial % 3]; shift = 30 * latent[person] if category == "biased" else (10 if category == "unbiased" else 0)
            trial_rows.append({"subject": subject, "group": group, "category": category, "correct": 1, "RT_ms": 800 + shift + rng.normal(0, 50)})
    behavior = pd.DataFrame(trial_rows); ensure(len(behavior) == 10995, "Synthetic trial-count failure.")
    behavior_path = destination / "synthetic_authorized_trials.csv"; behavior.to_csv(behavior_path, index=False)
    manifest = {"status": "complete", "synthetic_test_only": True, "common_coverage_geometry": {"nominal_atlas_parcels": 400, "effective_common_coverage_parcels": N_COMMON, "excluded_native_labels": [323], "uniform_all_participants_and_runs": True}, "participant_scores": {"path": "participant_scores_resolved.csv", "primary_column": "primary_default"}, "primary_matrices": {"path": "primary_participant_matrices.npy"}, "matrix_subjects": {"path": "primary_matrix_subjects.csv"}, "parcel_metadata": {"path": "physical_yeo7_common399_parcels.csv"}}
    atomic_json(destination / "enrichment_permutation_input_manifest.json", manifest)
    return destination / "enrichment_permutation_input_manifest.json", behavior_path


def create_edge_partition(parcels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(N_COMMON, 1); labels = parcels.yeo7_network.to_numpy()
    pairs = [pair_name(left, right) for x, left in enumerate(ORDER) for right in ORDER[x:]]; pair_lookup = {name: index for index, name in enumerate(pairs)}
    pair_code = np.asarray([pair_lookup[pair_name(left, right)] for left, right in zip(labels[ii], labels[jj])])
    possible_pair = np.bincount(pair_code, minlength=len(pairs)).astype(int)
    endpoint_masks = np.column_stack([(labels[ii] == network) | (labels[jj] == network) for network in ORDER])
    ensure(len(ii) == EDGE_UNIVERSE and possible_pair.sum() == EDGE_UNIVERSE, "Common edge partition failed.")
    return ii, jj, pairs, pair_code, possible_pair, endpoint_masks


def selected_statistics(selected: np.ndarray, pair_code: np.ndarray, possible_pair: np.ndarray, endpoint_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    pair_counts = np.bincount(pair_code, weights=selected.astype(int), minlength=28).astype(int)
    endpoint_counts = np.asarray([int(selected[endpoint_masks[:, column]].sum()) for column in range(len(ORDER))], dtype=int)
    total = int(selected.sum()); possible = np.concatenate([possible_pair, endpoint_masks.sum(axis=0).astype(int)]); counts = np.concatenate([pair_counts, endpoint_counts])
    expected = total * possible / EDGE_UNIVERSE; surplus = counts - expected; oe = np.divide(counts, expected, out=np.zeros_like(expected, dtype=float), where=expected > 0)
    return counts, surplus, total, oe


def array_content_sha(array: np.ndarray, completed: int) -> str:
    prefix = np.asarray(array[:completed])
    h = hashlib.sha256(); h.update(array.dtype.str.encode()); h.update(repr(prefix.shape).encode()); h.update(prefix.tobytes(order="C")); return h.hexdigest()


def fsync_memmap(array: np.memmap) -> None:
    array.flush()
    with Path(array.filename).open("rb") as handle: os.fsync(handle.fileno())


class NullArchive:
    SPECS = {"ss_source_indices": (np.int16, 16), "ds_source_indices": (np.int16, 17), "positive_selected_total": (np.int32,), "negative_selected_total": (np.int32,), "positive_category_counts": (np.int32, 35), "negative_category_counts": (np.int32, 35), "positive_surplus": (np.float64, 35), "negative_surplus": (np.float64, 35)}

    def __init__(self, root: Path, permutations: int, create: bool):
        self.root, self.permutations = root, permutations; self.arrays: dict[str, np.memmap] = {}
        if create:
            ensure(not root.exists(), f"Archive already exists: {root}"); root.mkdir(parents=True)
            for name, spec in self.SPECS.items(): self.arrays[name] = np.lib.format.open_memmap(root / f"{name}.npy", mode="w+", dtype=spec[0], shape=(permutations, *spec[1:]))
        else:
            ensure(root.is_dir(), f"Archive absent: {root}")
            for name, spec in self.SPECS.items():
                path = root / f"{name}.npy"; ensure(path.is_file(), f"Archive member absent: {path}")
                array = np.load(path, mmap_mode="r+"); ensure(array.dtype == np.dtype(spec[0]) and array.shape == (permutations, *spec[1:]), f"Archive member contract fails: {path}"); self.arrays[name] = array

    def write(self, row: int, ss: np.ndarray, ds: np.ndarray, pos: tuple[np.ndarray, np.ndarray, int, np.ndarray], neg: tuple[np.ndarray, np.ndarray, int, np.ndarray]) -> None:
        pc, ps, pt, _ = pos; nc, ns, nt, _ = neg
        self.arrays["ss_source_indices"][row] = ss; self.arrays["ds_source_indices"][row] = ds; self.arrays["positive_selected_total"][row] = pt; self.arrays["negative_selected_total"][row] = nt
        self.arrays["positive_category_counts"][row] = pc; self.arrays["negative_category_counts"][row] = nc; self.arrays["positive_surplus"][row] = ps; self.arrays["negative_surplus"][row] = ns

    def flush(self) -> None:
        for array in self.arrays.values(): fsync_memmap(array)

    def hashes(self, completed: int, final: bool = False) -> dict:
        return {name: {"content_sha256": array_content_sha(array, completed), "file_sha256": digest(Path(array.filename)) if final else None, "dtype": array.dtype.str, "shape": list(array.shape)} for name, array in self.arrays.items()}


def checkpoint_payload(config: dict, completed: int, rng: np.random.Generator, ge_positive: np.ndarray, ge_negative: np.ndarray, archive: NullArchive, final: bool) -> dict:
    return {"status": "COMPLETE_ARCHIVED_NOT_PUBLICATION_READY" if final else "IN_PROGRESS_NOT_EVIDENCE", "config": config, "completed_permutations": completed, "rng_state": rng.bit_generator.state, "ge_positive": ge_positive.tolist(), "ge_negative": ge_negative.tolist(), "archive": {"version": ARCHIVE_VERSION, "directory": archive.root.name, "commit_protocol": "draw rows flushed+fsynced; prefix content SHA256 recorded; checkpoint atomically replaced last", "arrays": archive.hashes(completed, final=final)}}


def validate_checkpoint_archive(saved: dict, archive: NullArchive, completed: int) -> None:
    recorded = saved.get("archive", {}).get("arrays", {})
    ensure(set(recorded) == set(archive.arrays), "Checkpoint/archive array membership differs.")
    for name, array in archive.arrays.items():
        item = recorded[name]; ensure(item.get("dtype") == array.dtype.str and item.get("shape") == list(array.shape), f"Checkpoint/archive contract differs: {name}")
        ensure(item.get("content_sha256") == array_content_sha(array, completed), f"Checkpoint/archive prefix SHA256 differs: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_MANIFEST); parser.add_argument("--behavior-trials", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--output", type=Path, default=HERE / "enrichment_permutation_results_common399_n33_archived_v2")
    parser.add_argument("--permutations", type=int, default=10_000); parser.add_argument("--batch-size", type=int, default=20); parser.add_argument("--checkpoint-every-batches", type=int, default=5); parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--synthetic-test", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--stop-after", type=int, help="Test-only clean stop after this draw count; resume is required to complete.")
    args = parser.parse_args()
    ensure(args.permutations > 0 and args.batch_size > 0 and args.checkpoint_every_batches > 0, "Permutation, batch, and checkpoint counts must be positive.")
    ensure(args.stop_after is None or 0 < args.stop_after < args.permutations, "--stop-after must be in 1..permutations-1.")
    output = args.output / "SYNTHETIC_TEST_NOT_EVIDENCE" if args.synthetic_test else args.output
    if args.synthetic_test:
        inputs = output.parent / (output.name + "_inputs")
        if args.resume:
            args.input_manifest, args.behavior_trials = inputs / "enrichment_permutation_input_manifest.json", inputs / "synthetic_authorized_trials.csv"; ensure(args.input_manifest.is_file() and args.behavior_trials.is_file(), f"Synthetic resume inputs absent: {inputs}")
        else:
            ensure(not inputs.exists() and not output.exists(), f"Synthetic paths already exist: {inputs} or {output}"); args.input_manifest, args.behavior_trials = make_synthetic(inputs)
    output.mkdir(parents=True, exist_ok=args.resume); checkpoint = output / "permutation_checkpoint.json"; archive_root = output / "null_archive_v2"
    matrices, groups, parcels, capture, provenance = load_inputs(args.input_manifest, args.behavior_trials, args.synthetic_test)
    config = {"frozen_original_runner_sha256": digest(ORIGINAL_RUNNER), "seed": args.seed, "permutations": args.permutations, "batch_size": args.batch_size, "screen": {"two_sided_p": SCREEN_P, "df": DF}, "statistic": "selected-count surplus = observed selected count - (all selected edges * category possible-edge share); one-sided upper-tail permutation p; O/E is descriptive and set to 0 when no selected edges make its denominator zero.", "families": "56 mutually exclusive pair-by-sign tests Holm-56 and 14 overlapping endpoint-by-sign tests Holm-14; both signs are included in each family conservatively.", "geometry": {"nominal_atlas_parcels": 400, "effective_common_coverage_parcels": N_COMMON, "excluded_native_labels": [323]}, "input": provenance, "code_sha256": digest(Path(__file__).resolve())}
    ii, jj, pair_names, pair_code, possible_pair, endpoint_masks = create_edge_partition(parcels)
    edge_values = within_group_center(matrices[:, ii, jj], groups); edge_norm = np.sqrt(np.sum(edge_values * edge_values, axis=0)); ensure(np.all(edge_norm > 0) and np.isfinite(edge_norm).all(), "Edge matrix has a zero/nonfinite variance edge.")
    observed_y = within_group_center(capture, groups); observed_r = observed_y @ edge_values / (np.sqrt(observed_y @ observed_y) * edge_norm); observed_p = 2 * stats.t.sf(np.abs(observed_r * np.sqrt(DF / (1 - observed_r * observed_r))), DF)
    obs_pos = selected_statistics((observed_p < SCREEN_P) & (observed_r > 0), pair_code, possible_pair, endpoint_masks); obs_neg = selected_statistics((observed_p < SCREEN_P) & (observed_r < 0), pair_code, possible_pair, endpoint_masks)
    obs_pos_counts, obs_pos_surplus, obs_pos_total, obs_pos_oe = obs_pos; obs_neg_counts, obs_neg_surplus, obs_neg_total, obs_neg_oe = obs_neg
    if args.resume:
        ensure(checkpoint.is_file(), f"Resume requires checkpoint: {checkpoint}"); saved = json.loads(checkpoint.read_text()); ensure(saved.get("config") == config, "Checkpoint provenance/configuration differs; refusing resume.")
        completed = int(saved["completed_permutations"]); ensure(0 <= completed < args.permutations, "Checkpoint is complete or has invalid iteration count; it cannot be resumed.")
        archive = NullArchive(archive_root, args.permutations, create=False); validate_checkpoint_archive(saved, archive, completed); rng = np.random.default_rng(); rng.bit_generator.state = saved["rng_state"]; ge_positive, ge_negative = np.asarray(saved["ge_positive"], int), np.asarray(saved["ge_negative"], int)
    else:
        ensure(not checkpoint.exists() and not (output / "enrichment_permutation_manifest.json").exists() and not archive_root.exists(), f"Refusing existing results/checkpoint/archive at {output}; use --resume only for a matching interrupted run.")
        completed, rng, ge_positive, ge_negative = 0, np.random.default_rng(args.seed), np.zeros(35, dtype=int), np.zeros(35, dtype=int); archive = NullArchive(archive_root, args.permutations, create=True)
    group_indices = [np.flatnonzero(groups == group) for group in ("SS", "DS")]
    index_maps = []
    for indices in group_indices:
        ensure(np.unique(capture[indices]).size == indices.size, "Capture values must be unique within group to store exact source-index maps.")
        index_maps.append({float(value): int(index) for value, index in zip(capture[indices], indices)})
    batches = 0
    while completed < args.permutations:
        limit = min(args.permutations, args.stop_after) if args.stop_after is not None else args.permutations; draw_count = min(args.batch_size, limit - completed); ensure(draw_count > 0, "Stop boundary reached unexpectedly.")
        shuffled = np.empty((draw_count, N)); ss_rows = np.empty((draw_count, 16), dtype=np.int16); ds_rows = np.empty((draw_count, 17), dtype=np.int16)
        for row in range(draw_count):
            shuffled[row] = capture
            for group_number, indices in enumerate(group_indices):
                # This is intentionally the frozen RNG call: SS first, then DS, per draw.
                permuted = rng.permutation(capture[indices]); shuffled[row, indices] = permuted
                source = np.asarray([index_maps[group_number][float(value)] for value in permuted], dtype=np.int16)
                (ss_rows if group_number == 0 else ds_rows)[row] = source
        shuffled = within_group_center(shuffled.T, groups).T; norms = np.sqrt(np.sum(shuffled * shuffled, axis=1)); r = (shuffled @ edge_values) / (norms[:, None] * edge_norm[None, :]); p = 2 * stats.t.sf(np.abs(r * np.sqrt(DF / (1 - r * r))), DF)
        for row in range(draw_count):
            positive = selected_statistics((p[row] < SCREEN_P) & (r[row] > 0), pair_code, possible_pair, endpoint_masks); negative = selected_statistics((p[row] < SCREEN_P) & (r[row] < 0), pair_code, possible_pair, endpoint_masks)
            ge_positive += positive[1] >= obs_pos_surplus; ge_negative += negative[1] >= obs_neg_surplus; archive.write(completed + row, ss_rows[row], ds_rows[row], positive, negative)
        completed += draw_count; batches += 1
        must_checkpoint = completed == args.permutations or completed == args.stop_after or batches % args.checkpoint_every_batches == 0
        if must_checkpoint:
            archive.flush(); atomic_json(checkpoint, checkpoint_payload(config, completed, rng, ge_positive, ge_negative, archive, final=completed == args.permutations))
        if args.stop_after is not None and completed == args.stop_after:
            print(json.dumps({"status": "STOPPED_FOR_RESUME_TEST_NOT_EVIDENCE", "completed": completed, "output": str(output)}, indent=2)); return
    category = pair_names + list(ORDER); scope = ["pair"] * 28 + ["endpoint"] * 7; possible = np.concatenate([possible_pair, endpoint_masks.sum(axis=0).astype(int)])
    result = pd.DataFrame({"scope": scope, "category": category, "possible_edges": possible, "positive_selected_observed": obs_pos_counts, "positive_expected_given_screen": obs_pos_total * possible / EDGE_UNIVERSE, "positive_observed_expected_ratio": obs_pos_oe, "positive_surplus_statistic": obs_pos_surplus, "positive_permutation_p_raw": (1 + ge_positive) / (1 + args.permutations), "negative_selected_observed": obs_neg_counts, "negative_expected_given_screen": obs_neg_total * possible / EDGE_UNIVERSE, "negative_observed_expected_ratio": obs_neg_oe, "negative_surplus_statistic": obs_neg_surplus, "negative_permutation_p_raw": (1 + ge_negative) / (1 + args.permutations)})
    result["zero_selection_oe_fallback"] = "O/E=0 only when screen-selected total is zero (expected denominator=0); surplus remains 0."
    for subset, family in ((result.scope.eq("pair"), "pair-by-sign, Holm-56"), (result.scope.eq("endpoint"), "endpoint-by-sign, Holm-14")):
        raw = np.concatenate([result.loc[subset, "positive_permutation_p_raw"], result.loc[subset, "negative_permutation_p_raw"]]); adjusted = multipletests(raw, method="holm")[1]; n_rows = int(subset.sum()); result.loc[subset, "positive_permutation_p_Holm"] = adjusted[:n_rows]; result.loc[subset, "negative_permutation_p_Holm"] = adjusted[n_rows:]; result.loc[subset, "Holm_family"] = family
    table_path = output / "common399_screen_enrichment_permutation.csv"; result.to_csv(table_path, index=False)
    archive_manifest = {"archive_version": ARCHIVE_VERSION, "completed_permutations": completed, "permutation_index_semantics": "Each row maps destination absolute participant positions in sorted-score group order to the absolute source capture position. SS rows are 16 positions and DS rows 17; maps reproduce rng.permutation(capture[indices]) without changing that RNG call.", "group_destination_indices": {"SS": group_indices[0].tolist(), "DS": group_indices[1].tolist()}, "arrays": archive.hashes(completed, final=True)}
    atomic_json(output / "null_archive_manifest.json", archive_manifest)
    # Re-write final checkpoint only after table and archive manifest are durable; never delete it.
    atomic_json(checkpoint, checkpoint_payload(config, completed, rng, ge_positive, ge_negative, archive, final=True))
    manifest = {"status": "COMPLETE_EMPIRICAL_ARCHIVE_REPAIR_NOT_PUBLICATION_READY" if not args.synthetic_test else "SYNTHETIC_TEST_NOT_EVIDENCE", "publication_ready": False, **config, "completed_permutations": completed, "observed_screen": {"edge_universe": EDGE_UNIVERSE, "positive_selected": obs_pos_total, "negative_selected": obs_neg_total}, "outputs": {"permutation_table": table_path.name, "checkpoint": checkpoint.name, "archive_manifest": "null_archive_manifest.json", "archive_directory": archive_root.name}, "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}}
    atomic_json(output / "enrichment_permutation_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "publication_ready": False, "permutations": completed, "edge_universe": EDGE_UNIVERSE, "positive_selected": obs_pos_total, "negative_selected": obs_neg_total, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
