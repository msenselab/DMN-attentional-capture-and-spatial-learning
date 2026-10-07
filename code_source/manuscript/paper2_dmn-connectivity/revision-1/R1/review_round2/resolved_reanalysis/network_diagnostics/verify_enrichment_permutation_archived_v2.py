#!/usr/bin/env python3
"""Independent verifier for the archived common399 enrichment permutation.

It deliberately imports neither the archival runner nor the frozen runner.
It rebuilds the input-derived statistic, seeded SS-then-DS draw stream, table,
and Holm families and checks every archived draw and required SHA256 record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
DEFAULT_OUTPUT = HERE / "enrichment_permutation_results_common399_n33_archived_v2"
DEFAULT_ORIGINAL = HERE / "enrichment_permutation_results_common399_n33/common399_screen_enrichment_permutation.csv"
N, N_COMMON, SEED, DF, SCREEN_P = 33, 399, 20261001, 30, 0.01
EDGE_UNIVERSE = N_COMMON * (N_COMMON - 1) // 2
EXCLUDED_NATIVE = frozenset({323})
NATIVE_COMMON = frozenset(range(1, 401)) - EXCLUDED_NATIVE
ORDER = ("Default", "SomMot", "Vis", "Cont", "SalVentAttn", "DorsAttn", "Limbic")
COUNTS = {"Default": 91, "SomMot": 77, "Vis": 61, "Cont": 52, "SalVentAttn": 47, "DorsAttn": 46, "Limbic": 25}


def fail(message: str) -> None: raise RuntimeError(message)
def ensure(ok: bool, message: str) -> None:
    if not ok: fail(message)
def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp"); temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); temp.replace(path)
def resolve(value: str, manifest: Path) -> Path:
    candidate = Path(value); return candidate if candidate.is_absolute() else manifest.parent / candidate
def required_path(manifest: dict, key: str, manifest_path: Path) -> Path:
    ensure(key in manifest and "path" in manifest[key], f"Input manifest lacks {key}.path")
    path = resolve(str(manifest[key]["path"]), manifest_path); ensure(path.is_file(), f"Input absent: {path}"); return path
def pair_name(left: str, right: str) -> str:
    return f"{left}--{right}" if ORDER.index(left) <= ORDER.index(right) else f"{right}--{left}"


def independently_load_inputs(manifest_path: Path, behavior_path: Path, synthetic: bool) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, dict]:
    ensure(manifest_path.is_file() and behavior_path.is_file(), "Manifest or behavior input absent.")
    manifest = json.loads(manifest_path.read_text()); ensure(manifest.get("status") == "complete", "Input manifest is not complete.")
    ensure(bool(manifest.get("synthetic_test_only", False)) == synthetic, "Synthetic mode/input-manifest label mismatch.")
    geometry = manifest.get("common_coverage_geometry", {}); ensure(geometry.get("nominal_atlas_parcels") == 400 and geometry.get("effective_common_coverage_parcels") == 399 and set(geometry.get("excluded_native_labels", [])) == EXCLUDED_NATIVE and geometry.get("uniform_all_participants_and_runs") is True, "Common399 geometry contract failed.")
    score_path = required_path(manifest, "participant_scores", manifest_path); matrix_path = required_path(manifest, "primary_matrices", manifest_path); subject_path = required_path(manifest, "matrix_subjects", manifest_path); parcel_path = required_path(manifest, "parcel_metadata", manifest_path)
    primary = manifest["participant_scores"].get("primary_column"); ensure(isinstance(primary, str) and primary, "Primary score column absent.")
    scores = pd.read_csv(score_path); ensure({"subject", "group", primary} <= set(scores), "Score schema invalid.")
    scores = scores[["subject", "group", primary]].sort_values("subject").reset_index(drop=True); ensure(len(scores) == N and scores.subject.is_unique and scores.group.value_counts().to_dict() == {"SS": 16, "DS": 17}, "Score N33/SS16/DS17 contract failed.")
    subjects = pd.read_csv(subject_path); ensure({"subject", "group"} <= set(subjects) and len(subjects) == N and subjects.subject.is_unique, "Matrix-subject schema invalid.")
    position = {subject: index for index, subject in enumerate(subjects.subject.astype(str))}; ensure(set(position) == set(scores.subject.astype(str)), "Matrix/score IDs differ.")
    ensure(np.array_equal(subjects.set_index("subject").loc[scores.subject, "group"].to_numpy(), scores.group.to_numpy()), "Matrix/score group order differs.")
    matrices = np.asarray(np.load(matrix_path, mmap_mode="r", allow_pickle=False)[[position[x] for x in scores.subject]], float); ensure(matrices.shape == (N, N_COMMON, N_COMMON) and np.isfinite(matrices).all() and np.allclose(matrices, matrices.swapaxes(1, 2), atol=1e-8, rtol=0) and np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 0, atol=1e-10, rtol=0), "Matrix contract invalid.")
    parcels = pd.read_csv(parcel_path).sort_values("matrix_index").reset_index(drop=True); ensure({"matrix_index", "native_label", "yeo7_network"} <= set(parcels) and parcels.matrix_index.tolist() == list(range(N_COMMON)) and set(parcels.native_label) == NATIVE_COMMON and parcels.native_label.nunique() == N_COMMON and parcels.yeo7_network.value_counts().to_dict() == COUNTS, "Parcel contract invalid.")
    trials = pd.read_csv(behavior_path); ensure({"subject", "group", "RT_ms", "category", "correct"} <= set(trials) and len(trials) == 10995 and trials.subject.nunique() == N and (trials.correct == 1).all() and (trials.RT_ms > 200).all() and (trials.RT_ms <= 3000).all(), "Authorized behavior contract invalid.")
    means = trials.groupby(["subject", "group", "category"], observed=True).RT_ms.mean().unstack("category").reset_index(); ensure({"absent", "biased"} <= set(means), "Behavior category contract invalid.")
    capture_table = scores[["subject", "group"]].merge(means.assign(capture=means.biased - means.absent)[["subject", "group", "capture"]], on=["subject", "group"], validate="one_to_one"); capture = capture_table.capture.to_numpy(float); ensure(np.isfinite(capture).all(), "Nonfinite capture.")
    return matrices, scores.group.to_numpy(), parcels, capture, {str(path): digest(path) for path in (score_path, matrix_path, subject_path, parcel_path, behavior_path)}


def center_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(values, float).copy()
    for label, size in (("SS", 16), ("DS", 17)):
        positions = np.flatnonzero(groups == label); ensure(len(positions) == size, f"Unexpected {label} group size."); result[positions] -= result[positions].mean(axis=0, keepdims=True)
    return result


def independently_partition(parcels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    first, second = np.triu_indices(N_COMMON, 1); labels = parcels.yeo7_network.to_numpy(); names = [pair_name(a, b) for n, a in enumerate(ORDER) for b in ORDER[n:]]; lookup = {name: n for n, name in enumerate(names)}
    codes = np.asarray([lookup[pair_name(a, b)] for a, b in zip(labels[first], labels[second])]); possible_pairs = np.bincount(codes, minlength=28).astype(int); endpoint_masks = np.column_stack([(labels[first] == label) | (labels[second] == label) for label in ORDER])
    ensure(len(first) == EDGE_UNIVERSE and possible_pairs.sum() == EDGE_UNIVERSE, "Independent edge partition failed.")
    return first, second, names, codes, possible_pairs, endpoint_masks


def independently_summarize(selection: np.ndarray, codes: np.ndarray, possible_pairs: np.ndarray, endpoint_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    pair_counts = np.bincount(codes, weights=selection.astype(np.int64), minlength=28).astype(int); endpoint_counts = np.array([np.count_nonzero(selection & endpoint_masks[:, n]) for n in range(7)], dtype=int); all_counts = np.r_[pair_counts, endpoint_counts]; all_possible = np.r_[possible_pairs, endpoint_masks.sum(axis=0).astype(int)]; total = int(np.count_nonzero(selection)); expected = total * all_possible / EDGE_UNIVERSE; surplus = all_counts - expected; ratio = np.divide(all_counts, expected, out=np.zeros(35, dtype=float), where=expected != 0); return all_counts, surplus, total, ratio


def content_sha(array: np.ndarray, completed: int) -> str:
    subset = np.asarray(array[:completed]); h = hashlib.sha256(); h.update(array.dtype.str.encode()); h.update(repr(subset.shape).encode()); h.update(subset.tobytes(order="C")); return h.hexdigest()


def verify_archive_contract(output: Path, permutations: int) -> tuple[dict[str, np.ndarray], dict, dict]:
    archive_manifest_path = output / "null_archive_manifest.json"; checkpoint_path = output / "permutation_checkpoint.json"; ensure(archive_manifest_path.is_file() and checkpoint_path.is_file(), "Final archive manifest or permanent checkpoint is absent.")
    archive_manifest, checkpoint = json.loads(archive_manifest_path.read_text()), json.loads(checkpoint_path.read_text()); ensure(archive_manifest.get("archive_version") == "enrichment-null-archive-v2" and checkpoint.get("status") == "COMPLETE_ARCHIVED_NOT_PUBLICATION_READY" and int(checkpoint.get("completed_permutations", -1)) == permutations and int(archive_manifest.get("completed_permutations", -1)) == permutations, "Final archive/checkpoint status contract failed.")
    expected = {"ss_source_indices": (np.int16, (permutations, 16)), "ds_source_indices": (np.int16, (permutations, 17)), "positive_selected_total": (np.int32, (permutations,)), "negative_selected_total": (np.int32, (permutations,)), "positive_category_counts": (np.int32, (permutations, 35)), "negative_category_counts": (np.int32, (permutations, 35)), "positive_surplus": (np.float64, (permutations, 35)), "negative_surplus": (np.float64, (permutations, 35))}
    archive_root = output / "null_archive_v2"; arrays = {}
    for name, (dtype, shape) in expected.items():
        path = archive_root / f"{name}.npy"; ensure(path.is_file(), f"Archive member absent: {path}"); array = np.load(path, mmap_mode="r"); ensure(array.dtype == dtype and array.shape == shape, f"Archive dtype/shape mismatch: {name}")
        for holder, label in ((archive_manifest.get("arrays", {}).get(name), "archive-manifest"), (checkpoint.get("archive", {}).get("arrays", {}).get(name), "checkpoint")):
            ensure(isinstance(holder, dict) and holder.get("content_sha256") == content_sha(array, permutations) and holder.get("dtype") == array.dtype.str and holder.get("shape") == list(array.shape), f"{label} content contract mismatch: {name}")
            if label == "archive-manifest": ensure(holder.get("file_sha256") == digest(path), f"Archive file SHA256 mismatch: {name}")
        arrays[name] = array
    return arrays, archive_manifest, checkpoint


def build_independent_table(names: list[str], possible_pairs: np.ndarray, endpoint_masks: np.ndarray, observed_positive: tuple[np.ndarray, np.ndarray, int, np.ndarray], observed_negative: tuple[np.ndarray, np.ndarray, int, np.ndarray], ge_positive: np.ndarray, ge_negative: np.ndarray, permutations: int) -> pd.DataFrame:
    pc, ps, pt, po = observed_positive; nc, ns, nt, no = observed_negative; possible = np.r_[possible_pairs, endpoint_masks.sum(axis=0).astype(int)]
    table = pd.DataFrame({"scope": ["pair"] * 28 + ["endpoint"] * 7, "category": names + list(ORDER), "possible_edges": possible, "positive_selected_observed": pc, "positive_expected_given_screen": pt * possible / EDGE_UNIVERSE, "positive_observed_expected_ratio": po, "positive_surplus_statistic": ps, "positive_permutation_p_raw": (1 + ge_positive) / (1 + permutations), "negative_selected_observed": nc, "negative_expected_given_screen": nt * possible / EDGE_UNIVERSE, "negative_observed_expected_ratio": no, "negative_surplus_statistic": ns, "negative_permutation_p_raw": (1 + ge_negative) / (1 + permutations)})
    table["zero_selection_oe_fallback"] = "O/E=0 only when screen-selected total is zero (expected denominator=0); surplus remains 0."
    for mask, label in ((table.scope.eq("pair"), "pair-by-sign, Holm-56"), (table.scope.eq("endpoint"), "endpoint-by-sign, Holm-14")):
        raw = np.r_[table.loc[mask, "positive_permutation_p_raw"], table.loc[mask, "negative_permutation_p_raw"]]; adjusted = multipletests(raw, method="holm")[1]; count = int(mask.sum()); table.loc[mask, "positive_permutation_p_Holm"] = adjusted[:count]; table.loc[mask, "negative_permutation_p_Holm"] = adjusted[count:]; table.loc[mask, "Holm_family"] = label
    return table


def exact_table_match(recomputed: pd.DataFrame, saved_path: Path, label: str) -> None:
    """Require byte-identical CSV serialization, avoiding parser rounding as a tolerance."""
    ensure(saved_path.is_file(), f"{label} table absent: {saved_path}")
    expected = recomputed.to_csv(index=False).encode("utf-8")
    actual = saved_path.read_bytes()
    ensure(expected == actual, f"{label} CSV differs from independently recomputed values.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--input-manifest", type=Path, default=DEFAULT_MANIFEST); parser.add_argument("--behavior-trials", type=Path, default=DEFAULT_BEHAVIOR); parser.add_argument("--original-table", type=Path, default=DEFAULT_ORIGINAL); parser.add_argument("--permutations", type=int, default=10_000); parser.add_argument("--batch-size", type=int, default=20); parser.add_argument("--synthetic-test", action="store_true"); parser.add_argument("--report", type=Path)
    args = parser.parse_args(); ensure(args.permutations > 0 and args.batch_size > 0, "Positive permutations and batch size required.")
    output = args.output / "SYNTHETIC_TEST_NOT_EVIDENCE" if args.synthetic_test and args.output.name != "SYNTHETIC_TEST_NOT_EVIDENCE" else args.output
    report_path = args.report or (output / "independent_verification_report.json")
    report = {"status": "FAIL", "publication_ready": False, "output": str(output), "permutations": args.permutations, "checks": []}
    try:
        manifest_path = output / "enrichment_permutation_manifest.json"; ensure(manifest_path.is_file(), "Result manifest absent."); run_manifest = json.loads(manifest_path.read_text()); ensure(run_manifest.get("publication_ready") is False and int(run_manifest.get("completed_permutations", -1)) == args.permutations, "Result manifest publication/completion contract failed.")
        matrices, groups, parcels, capture, input_hashes = independently_load_inputs(args.input_manifest, args.behavior_trials, args.synthetic_test); ensure(run_manifest.get("input", {}).get("hashes") == input_hashes, "Input provenance hash mismatch."); ensure(run_manifest.get("seed") == SEED and run_manifest.get("permutations") == args.permutations and run_manifest.get("batch_size") == args.batch_size and run_manifest.get("screen") == {"two_sided_p": SCREEN_P, "df": DF}, "Frozen configuration mismatch.")
        ensure(run_manifest.get("frozen_original_runner_sha256") == digest(HERE / "run_enrichment_permutation.py") and run_manifest.get("code_sha256") == digest(HERE / "run_enrichment_permutation_archived_v2.py"), "Runner provenance hash mismatch."); report["checks"].append("input and runner provenance SHA256 PASS")
        arrays, archive_manifest, checkpoint = verify_archive_contract(output, args.permutations); ensure(archive_manifest.get("group_destination_indices") == {"SS": np.flatnonzero(groups == "SS").tolist(), "DS": np.flatnonzero(groups == "DS").tolist()}, "Archive group destination mapping differs."); report["checks"].append("archive members, SHA256, and permanent final checkpoint PASS")
        ii, jj, names, codes, possible_pairs, endpoint_masks = independently_partition(parcels); edge_values = center_by_group(matrices[:, ii, jj], groups); edge_norm = np.sqrt(np.sum(edge_values ** 2, axis=0)); ensure(np.isfinite(edge_norm).all() and np.all(edge_norm > 0), "Independent edge-norm check failed.")
        observed_y = center_by_group(capture, groups); observed_r = observed_y @ edge_values / (np.sqrt(observed_y @ observed_y) * edge_norm); observed_p = 2 * stats.t.sf(np.abs(observed_r * np.sqrt(DF / (1 - observed_r * observed_r))), DF); observed_positive = independently_summarize((observed_p < SCREEN_P) & (observed_r > 0), codes, possible_pairs, endpoint_masks); observed_negative = independently_summarize((observed_p < SCREEN_P) & (observed_r < 0), codes, possible_pairs, endpoint_masks)
        ensure(run_manifest.get("observed_screen") == {"edge_universe": EDGE_UNIVERSE, "positive_selected": observed_positive[2], "negative_selected": observed_negative[2]}, "Observed-screen provenance mismatch.")
        ss_indices, ds_indices = np.flatnonzero(groups == "SS"), np.flatnonzero(groups == "DS"); ensure(np.unique(capture[ss_indices]).size == 16 and np.unique(capture[ds_indices]).size == 17, "Cannot independently map duplicate capture values."); ss_lookup = {float(v): int(i) for v, i in zip(capture[ss_indices], ss_indices)}; ds_lookup = {float(v): int(i) for v, i in zip(capture[ds_indices], ds_indices)}
        generator = np.random.default_rng(SEED); ge_positive = np.zeros(35, dtype=int); ge_negative = np.zeros(35, dtype=int)
        for start in range(0, args.permutations, args.batch_size):
            stop = min(start + args.batch_size, args.permutations); shuffled = np.empty((stop - start, N)); expected_ss = np.empty((stop - start, 16), dtype=np.int16); expected_ds = np.empty((stop - start, 17), dtype=np.int16)
            for row in range(stop - start):
                shuffled[row] = capture
                # Independently replay the required original RNG calls: SS, then DS, every draw.
                ss_values = generator.permutation(capture[ss_indices]); ds_values = generator.permutation(capture[ds_indices]); shuffled[row, ss_indices] = ss_values; shuffled[row, ds_indices] = ds_values
                expected_ss[row] = [ss_lookup[float(x)] for x in ss_values]; expected_ds[row] = [ds_lookup[float(x)] for x in ds_values]
            ensure(np.array_equal(expected_ss, arrays["ss_source_indices"][start:stop]) and np.array_equal(expected_ds, arrays["ds_source_indices"][start:stop]), f"Seeded SS/DS permutation-index mismatch at draws {start}:{stop}.")
            shuffled = center_by_group(shuffled.T, groups).T; norm = np.sqrt(np.sum(shuffled ** 2, axis=1)); correlations = (shuffled @ edge_values) / (norm[:, None] * edge_norm[None, :]); probabilities = 2 * stats.t.sf(np.abs(correlations * np.sqrt(DF / (1 - correlations * correlations))), DF)
            for row in range(stop - start):
                positive = independently_summarize((probabilities[row] < SCREEN_P) & (correlations[row] > 0), codes, possible_pairs, endpoint_masks); negative = independently_summarize((probabilities[row] < SCREEN_P) & (correlations[row] < 0), codes, possible_pairs, endpoint_masks); draw = start + row
                ensure(positive[2] == arrays["positive_selected_total"][draw] and negative[2] == arrays["negative_selected_total"][draw] and np.array_equal(positive[0], arrays["positive_category_counts"][draw]) and np.array_equal(negative[0], arrays["negative_category_counts"][draw]) and np.array_equal(positive[1], arrays["positive_surplus"][draw]) and np.array_equal(negative[1], arrays["negative_surplus"][draw]), f"Archived null statistics mismatch at draw {draw}.")
                ge_positive += positive[1] >= observed_positive[1]; ge_negative += negative[1] >= observed_negative[1]
        ensure(np.array_equal(ge_positive, np.asarray(checkpoint.get("ge_positive"), int)) and np.array_equal(ge_negative, np.asarray(checkpoint.get("ge_negative"), int)), "Independent GE tail counts differ from permanent checkpoint."); report["checks"].append("all seeded SS/DS maps and all archived null totals/counts/surpluses PASS")
        independent = build_independent_table(names, possible_pairs, endpoint_masks, observed_positive, observed_negative, ge_positive, ge_negative, args.permutations); saved_path = output / "common399_screen_enrichment_permutation.csv"; exact_table_match(independent, saved_path, "independent-versus-archived"); report["checks"].append("independent observed statistics, raw p values, and Holm families PASS")
        if not args.synthetic_test:
            ensure(args.original_table.is_file(), f"Frozen original table absent: {args.original_table}"); exact_table_match(independent, args.original_table, "independent-versus-original"); ensure(saved_path.read_bytes() == args.original_table.read_bytes(), "Archived/original table bytes are not exactly equal."); report["checks"].append("exact numeric match with frozen original table PASS")
        report.update({"status": "PASS", "publication_ready": False, "observed_screen": {"positive": observed_positive[2], "negative": observed_negative[2]}, "archive_file_sha256": {name: digest(output / "null_archive_v2" / f"{name}.npy") for name in arrays}})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
