#!/usr/bin/env python3
"""Strict adapter from the immutable common399 extraction handoff to analysis-ready inputs.

The extraction runner's emitted manifest is deliberately not changed: its status is
an extraction handoff, not a claim that dependent analyses are complete.  This
adapter validates that handoff against every declared hash, preserves a byte-identical
copy, and writes a separate analysis-ready directory for downstream consumers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_EXTRACTION = HERE / "extraction_common_coverage"
DEFAULT_READY = HERE / "common399_analysis_ready"
N_SUBJECTS, N_PARCELS = 33, 399
EXCLUDED_NATIVE = 323
NETWORK_COUNTS = {
    "Default": 91, "SomMot": 77, "Vis": 61, "Cont": 52,
    "SalVentAttn": 47, "DorsAttn": 46, "Limbic": 25,
}
RAW_HANDOFF_STATUS = "COMPLETE_33_EXTRACTION_HANDOFF_NOT_PIPELINE_COMPLETE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_json(path: Path) -> dict[str, Any]:
    ensure(path.is_file(), f"Required JSON is absent: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON {path}: {exc}")
    ensure(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def immutable_bytes(path: Path, payload: bytes) -> None:
    """Create once, or require a byte-identical existing provenance artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ensure(path.is_file(), f"Output path is not a file: {path}")
        ensure(path.read_bytes() == payload, f"Refusing ambiguous overwrite of {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def immutable_json(path: Path, value: dict[str, Any]) -> None:
    immutable_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def file_from_config(config: dict[str, Any], key: str, base: Path) -> Path:
    ensure(isinstance(config.get(key), dict) and isinstance(config[key].get("path"), str),
           f"Raw handoff lacks {key}.path")
    path = resolve(config[key]["path"], base)
    ensure(path.is_file(), f"Raw handoff input is absent: {path}")
    return path


def validate_hash_list(strict: dict[str, Any]) -> dict[Path, str]:
    listed: dict[Path, str] = {}
    items = strict.get("files")
    ensure(isinstance(items, list) and items, "strict_analysis_input_manifest.files is missing/empty")
    for item in items:
        ensure(isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str),
               "Malformed strict file hash item")
        path = Path(item["path"])
        ensure(path.is_absolute(), f"Strict file path must be absolute: {path}")
        ensure(path not in listed, f"Duplicate strict file hash entry: {path}")
        ensure(path.is_file() and sha256(path) == item["sha256"], f"Strict output hash mismatch: {path}")
        listed[path] = item["sha256"]
    return listed


def physical_parcels(native_path: Path) -> pd.DataFrame:
    native = pd.read_csv(native_path)
    required = {"local_matrix_index_0based", "native_label_id", "physical_label"}
    ensure(required <= set(native), f"Native-map schema missing {sorted(required - set(native))}")
    native = native.sort_values("local_matrix_index_0based").reset_index(drop=True)
    ensure(len(native) == N_PARCELS and native.local_matrix_index_0based.tolist() == list(range(N_PARCELS)),
           "Native map must encode local matrix indices 0..398")
    labels = native.native_label_id.to_numpy(int)
    ensure(set(labels) == set(range(1, 401)) - {EXCLUDED_NATIVE} and len(set(labels)) == N_PARCELS,
           "Native map must encode physical labels 1..400 excluding 323 exactly")
    names = native.physical_label.astype(str)
    networks = names.str.split("_").str[2]
    ensure(not networks.isna().any() and networks.value_counts().to_dict() == NETWORK_COUNTS,
           f"Physical Yeo7 network counts failed: {networks.value_counts().to_dict()}")
    return pd.DataFrame({
        "matrix_index": native.local_matrix_index_0based.to_numpy(int),
        "native_label": labels,
        "physical_label": names.to_numpy(str),
        "yeo7_network": networks.to_numpy(str),
    })


def matrix_checks(path: Path, label: str) -> np.ndarray:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    ensure(matrix.shape == (N_PARCELS, N_PARCELS), f"{label} has wrong matrix shape: {matrix.shape}")
    ensure(np.isfinite(matrix).all(), f"{label} has nonfinite values")
    ensure(np.allclose(matrix, matrix.T, atol=1e-10, rtol=0), f"{label} is not symmetric")
    ensure(np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0), f"{label} has nonzero diagonal")
    return np.asarray(matrix)


def validate_subject_manifest(path: Path, expected_subject: str, expected_parent: str, expected_amendment: str) -> dict[str, Any]:
    manifest = read_json(path)
    ensure(manifest.get("status") == "complete" and manifest.get("subject") == expected_subject,
           f"Subject manifest is incomplete/misidentified: {path}")
    ensure(manifest.get("parent_contract_sha256") == expected_parent and manifest.get("coverage_amendment_sha256") == expected_amendment,
           f"Frozen contract/amendment mismatch: {path}")
    ensure(manifest.get("nominal_atlas_parcels") == 400 and manifest.get("effective_common_coverage_parcels") == N_PARCELS
           and manifest.get("excluded_native_labels") == [EXCLUDED_NATIVE], f"Common399 geometry mismatch: {path}")
    atlas_hashes = manifest.get("source_atlas_raw_input_hashes")
    ensure(isinstance(atlas_hashes, dict) and atlas_hashes, f"Source atlas/raw hashes missing: {path}")
    for source, expected_hash in atlas_hashes.items():
        source_path = Path(source)
        ensure(source_path.is_file() and sha256(source_path) == expected_hash, f"Source atlas/raw hash mismatch: {source_path}")
    runs = manifest.get("runs")
    ensure(isinstance(runs, list) and {run.get("run") for run in runs} == {1, 2} and len(runs) == 2,
           f"Expected exactly run 1/run 2: {path}")
    primary_names: list[str] | None = None
    fd_rows: list[dict[str, Any]] = []
    participant = manifest.get("participant")
    ensure(isinstance(participant, dict), f"Participant outputs absent: {path}")
    for run in runs:
        source_inputs = run.get("source_inputs")
        source_hashes = run.get("source_input_hashes")
        ensure(isinstance(source_inputs, dict) and set(source_inputs) == {"bold", "mask", "confounds", "events"},
               f"Run source-input schema failed: {path}")
        ensure(isinstance(source_hashes, dict), f"Run source hashes absent: {path}")
        for source in source_inputs.values():
            source_path = Path(source)
            ensure(source_path.is_file() and source_hashes.get(str(source_path)) == sha256(source_path),
                   f"Run source hash mismatch: {source_path}")
        for key, hash_key in (("raw_timeseries_path", "raw_timeseries_sha256"),):
            derived = Path(run.get(key, ""))
            ensure(derived.is_file() and sha256(derived) == run.get(hash_key), f"Derived hash mismatch: {derived}")
        raw = np.load(Path(run["raw_timeseries_path"]), mmap_mode="r", allow_pickle=False)
        ensure(raw.ndim == 2 and raw.shape[1] == N_PARCELS and np.isfinite(raw).all(), f"Raw399 schema failed: {path}")
        atlas = run.get("atlas")
        ensure(isinstance(atlas, dict), f"Atlas metadata absent: {path}")
        atlas_path = Path(atlas.get("atlas_map_path", ""))
        ensure(atlas_path.is_file() and sha256(atlas_path) == atlas.get("atlas_map_sha256"), f"Atlas hash mismatch: {atlas_path}")
        nuisance = run.get("nuisance")
        ensure(isinstance(nuisance, dict), f"Nuisance metadata absent: {path}")
        for key in ("n_fd_valid", "sum_fd_retained", "mean_fd_retained"):
            ensure(key in nuisance and np.isfinite(float(nuisance[key])), f"Nonfinite retained-FD metadata {key}: {path}")
        ensure(int(nuisance["n_fd_valid"]) > 0 and float(nuisance["sum_fd_retained"]) >= 0
               and np.isclose(float(nuisance["mean_fd_retained"]), float(nuisance["sum_fd_retained"]) / int(nuisance["n_fd_valid"]), atol=1e-14, rtol=0),
               f"Retained-FD pooling schema failed: {path}")
        variants = run.get("variants")
        ensure(isinstance(variants, dict) and set(variants) == {"primary", "minimal_tr1", "primary_gsr", "primary_task", "primary_task_gsr"},
               f"Variant schema failed: {path}")
        primary = variants["primary"]
        names = primary.get("confound_names")
        ensure(primary.get("n_confounds") == 26 and isinstance(names, list) and len(names) == 26,
               f"Primary 26-confound declaration failed: {path}")
        if primary_names is None:
            primary_names = names
        ensure(names == primary_names, f"Primary confounds differ within subject: {path}")
        for variant_name, variant in variants.items():
            output = Path(variant.get("matrix_path", ""))
            ensure(output.is_file() and sha256(output) == variant.get("matrix_sha256"),
                   f"Variant matrix hash mismatch ({variant_name}): {output}")
            matrix_checks(output, f"{expected_subject} run{run['run']} {variant_name}")
        fd_rows.append({"subject": expected_subject, "run": int(run["run"]), **{key: nuisance[key] for key in ("n_fd_valid", "sum_fd_retained", "mean_fd_retained")},
                        "primary_run_value": float(primary["within_default91_mean_fisher_z"]), "confound_names": primary_names})
    for variant_name, record in participant.items():
        output = Path(record.get("matrix_path", ""))
        ensure(output.is_file() and sha256(output) == record.get("matrix_sha256"),
               f"Participant matrix hash mismatch ({variant_name}): {output}")
        matrix_checks(output, f"{expected_subject} participant {variant_name}")
    return {"manifest": manifest, "fd_rows": fd_rows, "primary_names": primary_names}


def validate_and_transform(extraction: Path, ready: Path, synthetic_test: bool = False) -> dict[str, Any]:
    raw_path = extraction / "inference_input_manifest.json"
    raw_bytes = raw_path.read_bytes() if raw_path.is_file() else b""
    raw = read_json(raw_path)
    ensure(raw.get("status") == RAW_HANDOFF_STATUS, "Extraction handoff is not the exact complete-N33 handoff status")
    strict_path = extraction / "strict_analysis_input_manifest.json"
    strict = read_json(strict_path)
    ensure(strict.get("status") == "complete_33", "Strict extraction manifest is not complete_33")
    ensure(strict.get("matrix_shape") == [N_SUBJECTS, N_PARCELS, N_PARCELS]
           and strict.get("effective_common_coverage_parcels") == N_PARCELS
           and strict.get("excluded_native_labels") == [EXCLUDED_NATIVE], "Strict matrix/common-coverage geometry failed")
    subjects = strict.get("subjects")
    ensure(isinstance(subjects, list) and len(subjects) == N_SUBJECTS and len(set(subjects)) == N_SUBJECTS,
           "Strict manifest must name exactly 33 unique subjects")
    strict_files = validate_hash_list(strict)
    score_path = file_from_config(raw, "participant_scores", raw_path.parent)
    matrices_path = file_from_config(raw, "primary_matrices", raw_path.parent)
    subjects_path = file_from_config(raw, "matrix_subjects", raw_path.parent)
    native_path = file_from_config(raw, "parcel_metadata", raw_path.parent)
    runlong_path = file_from_config(raw, "runwise_primary", raw_path.parent)
    for required in (score_path, matrices_path, subjects_path, native_path, runlong_path):
        ensure(required in strict_files, f"Raw handoff file is not protected by strict hash list: {required}")
    parent_hash = strict.get("parent_contract_sha256")
    amendment_hash = strict.get("coverage_amendment_sha256")
    ensure(isinstance(parent_hash, str) and isinstance(amendment_hash, str), "Strict frozen hash schema failed")
    entries = strict.get("subject_manifests")
    ensure(isinstance(entries, list) and len(entries) == N_SUBJECTS, "Strict subject-manifest hash list must have 33 entries")
    subject_checks: dict[str, dict[str, Any]] = {}
    for entry in entries:
        ensure(isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("sha256"), str),
               "Malformed strict subject-manifest entry")
        path = Path(entry["path"])
        ensure(path.is_file() and sha256(path) == entry["sha256"], f"Subject-manifest strict hash mismatch: {path}")
        expected_subject = path.name.removeprefix("sub-").removesuffix("_manifest.json")
        ensure(expected_subject in subjects and expected_subject not in subject_checks, f"Unexpected/duplicate subject manifest: {path}")
        subject_checks[expected_subject] = validate_subject_manifest(path, expected_subject, parent_hash, amendment_hash)
    ensure(set(subject_checks) == set(subjects), "Subject manifests do not exactly match strict subject list")
    primary_names = subject_checks[subjects[0]]["primary_names"]
    ensure(all(check["primary_names"] == primary_names for check in subject_checks.values()),
           "Primary 26-confound names are not uniform across the N33 cohort")

    parcels = physical_parcels(native_path)
    scores = pd.read_csv(score_path, dtype={"subject": str}, float_precision="round_trip")
    ensure({"subject", "group", "dmn_conn_primary", "mean_fd_resolved"} <= set(scores), "Participant score schema failed")
    ensure(len(scores) == N_SUBJECTS and scores.subject.is_unique and set(scores.subject) == set(subjects)
           and scores.group.value_counts().to_dict() == {"DS": 17, "SS": 16}, "Participant cohort/group schema failed")
    scores = scores.sort_values("subject").reset_index(drop=True)
    ensure(np.isfinite(scores[["dmn_conn_primary", "mean_fd_resolved"]].to_numpy(float)).all(), "Nonfinite score/FD values")
    order = pd.read_csv(subjects_path, dtype={"subject": str})
    ensure({"row_index", "subject", "group"} <= set(order) and len(order) == N_SUBJECTS and order.subject.is_unique
           and order.row_index.tolist() == list(range(N_SUBJECTS)), "Matrix subject-order schema failed")
    ensure(set(order.subject) == set(subjects) and order.set_index("subject").loc[scores.subject, "group"].tolist() == scores.group.tolist(),
           "Matrix score group identities disagree")
    matrices = np.load(matrices_path, mmap_mode="r", allow_pickle=False)
    ensure(matrices.shape == (N_SUBJECTS, N_PARCELS, N_PARCELS) and np.isfinite(matrices).all()
           and np.allclose(matrices, matrices.swapaxes(1, 2), atol=1e-10, rtol=0)
           and np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 0.0, atol=1e-12, rtol=0), "Primary matrix stack schema failed")
    position = {subject: index for index, subject in enumerate(order.subject)}
    sorted_matrices = np.asarray(matrices[[position[subject] for subject in scores.subject]], dtype=float)
    default = parcels.yeo7_network.eq("Default").to_numpy()
    ii, jj = np.triu_indices(int(default.sum()), 1)
    reconstructed = sorted_matrices[:, default][:, :, default][:, ii, jj].mean(axis=1)
    ensure(np.allclose(reconstructed, scores.dmn_conn_primary.to_numpy(float), atol=1e-8, rtol=0),
           "Fresh primary score does not reconstruct from physical Default91 matrix entries")

    runlong = pd.read_csv(runlong_path, dtype={"subject": str}, float_precision="round_trip")
    required_long = {"subject", "run", "dmn_conn_primary_run", "n_fd_valid", "sum_fd_retained_mm", "mean_fd_retained_mm"}
    ensure(required_long <= set(runlong) and len(runlong) == 2 * N_SUBJECTS, "Long runwise source schema failed")
    ensure(runlong.groupby("subject").run.apply(lambda x: set(x) == {1, 2}).all() and set(runlong.subject) == set(subjects),
           "Long runwise source must have exactly runs 1/2 for every subject")
    runlong["n_fd_valid"] = pd.to_numeric(runlong.n_fd_valid, errors="raise")
    runlong["sum_fd_retained_mm"] = pd.to_numeric(runlong.sum_fd_retained_mm, errors="raise")
    runlong["mean_fd_retained_mm"] = pd.to_numeric(runlong.mean_fd_retained_mm, errors="raise")
    ensure(np.isfinite(runlong[["n_fd_valid", "sum_fd_retained_mm", "mean_fd_retained_mm"]].to_numpy(float)).all()
           and (runlong.n_fd_valid > 0).all() and (runlong.sum_fd_retained_mm >= 0).all(), "Runwise retained FD is invalid")
    ensure(np.allclose(runlong.mean_fd_retained_mm, runlong.sum_fd_retained_mm / runlong.n_fd_valid, atol=1e-14, rtol=0),
           "Runwise retained FD mean does not equal retained-volume sum/count")
    expected_rows = pd.DataFrame([row for subject in subjects for row in subject_checks[subject]["fd_rows"]])
    compare = runlong.merge(expected_rows, on=["subject", "run"], suffixes=("_long", "_manifest"), validate="one_to_one")
    ensure(len(compare) == 2 * N_SUBJECTS
           and np.allclose(compare.n_fd_valid_long, compare.n_fd_valid_manifest, atol=0, rtol=0)
           and np.allclose(compare.sum_fd_retained_mm, compare.sum_fd_retained, atol=1e-14, rtol=0)
           and np.allclose(compare.mean_fd_retained_mm, compare.mean_fd_retained, atol=1e-14, rtol=0)
           and np.allclose(compare.dmn_conn_primary_run, compare.primary_run_value, atol=1e-12, rtol=0),
           "Runwise long source does not match subject-manifest FD/primary records")
    pooled = runlong.groupby("subject", sort=False).agg(n_fd_valid=("n_fd_valid", "sum"), sum_fd_retained_mm=("sum_fd_retained_mm", "sum"))
    pooled["mean_fd_resolved"] = pooled.sum_fd_retained_mm / pooled.n_fd_valid
    ensure(np.allclose(scores.set_index("subject").loc[pooled.index, "mean_fd_resolved"], pooled.mean_fd_resolved, atol=1e-14, rtol=0),
           "Participant FD does not equal finite retained-volume pooled run FD")
    wide = runlong.pivot(index="subject", columns="run", values="dmn_conn_primary_run").rename(columns={1: "run1_primary", 2: "run2_primary"}).reset_index()
    wide = scores[["subject", "group"]].merge(wide, on="subject", validate="one_to_one")
    ensure(np.isfinite(wide[["run1_primary", "run2_primary"]].to_numpy(float)).all(), "Wide run score schema failed")

    ready.mkdir(parents=True, exist_ok=True)
    immutable_bytes(ready / "extraction_handoff_original.json", raw_bytes)
    parcels_path = ready / "physical_yeo7_common399_parcels.csv"
    wide_path = ready / "runwise_primary_scores_wide.csv"
    immutable_bytes(parcels_path, parcels.to_csv(index=False).encode())
    immutable_bytes(wide_path, wide.to_csv(index=False).encode())
    geometry = {
        "nominal_atlas_parcels": 400,
        "effective_common_coverage_parcels": N_PARCELS,
        "excluded_native_labels": [EXCLUDED_NATIVE],
        "uniform_all_participants_and_runs": True,
        "matrix_index_definition": "local matrix indices 0..398 map through physical_yeo7_common399_parcels.csv; native 323 is omitted uniformly and never used as a matrix index",
    }
    transforms = {
        "parcel_metadata": {"input": str(native_path), "input_sha256": sha256(native_path), "schema": {"input_columns": ["local_matrix_index_0based", "native_label_id", "physical_label"], "output_columns": list(parcels.columns), "network_parse": "physical_label split '_' field 2"}, "output": str(parcels_path), "output_sha256": sha256(parcels_path)},
        "runwise_primary": {"input": str(runlong_path), "input_sha256": sha256(runlong_path), "schema": {"input_layout": "long subject/run/dmn_conn_primary_run", "output_columns": list(wide.columns), "run_mapping": {"1": "run1_primary", "2": "run2_primary"}}, "output": str(wide_path), "output_sha256": sha256(wide_path)},
    }
    fd_provenance = {
        "definition": "Raw fMRIPrep framewise_displacement in mm, finite retained nonsteady-state volumes only; frame 0 excluded rather than zero-imputed; participant mean is pooled by retained finite FD-volume count across both runs.",
        "units": "mm",
        "participants_checked": N_SUBJECTS,
        "runs_checked": 2 * N_SUBJECTS,
        "all_participant_means_finite": True,
        "retained_volume_pooling_verified": True,
        "uniform_primary_confounds": {"n_confounds": 26, "names": primary_names, "all_66_runs_identical": True},
        "per_run_source": str(runlong_path),
        "per_run_source_sha256": sha256(runlong_path),
    }
    analysis_manifest = {
        "status": "SYNTHETIC_TEST_NOT_EVIDENCE" if synthetic_test else "complete",
        "synthetic_test_only": bool(synthetic_test),
        "common_coverage_geometry": geometry,
        "participant_scores": {"path": str(score_path), "primary_column": "dmn_conn_primary", "variant_columns": raw["participant_scores"]["variant_columns"], "mean_fd_column": "mean_fd_resolved", "mean_fd_provenance": fd_provenance},
        "primary_matrices": {"path": str(matrices_path)},
        "matrix_subjects": {"path": str(subjects_path)},
        "parcel_metadata": {"path": str(parcels_path)},
        "runwise_primary": {"path": str(wide_path), "run1_column": "run1_primary", "run2_column": "run2_primary"},
        "adapter_provenance": {"raw_extraction_handoff": str(raw_path), "raw_extraction_handoff_sha256": sha256(raw_path), "preserved_original": str(ready / "extraction_handoff_original.json"), "preserved_original_sha256": sha256(ready / "extraction_handoff_original.json"), "strict_manifest": str(strict_path), "strict_manifest_sha256": sha256(strict_path), "strict_output_hashes_verified": len(strict_files), "strict_subject_manifests_verified": N_SUBJECTS, "transformations": transforms, "adapter_code": str(Path(__file__).resolve()), "adapter_code_sha256": sha256(Path(__file__).resolve())},
    }
    manifest_path = ready / "analysis_ready_input_manifest.json"
    immutable_json(manifest_path, analysis_manifest)
    run_status = {"status": "SYNTHETIC_TEST_NOT_EVIDENCE" if synthetic_test else "complete", "full_cohort_handoff_emitted": True, "n_requested": N_SUBJECTS,
                  "stable_full_cohort_handoff_files": ["participant_scores_resolved.csv", "primary_participant_matrices.npy", "primary_matrix_subjects.csv"],
                  "source_extraction_run_status": str(extraction / "run_status.json"), "analysis_ready_input_manifest": str(manifest_path), "analysis_ready_input_manifest_sha256": sha256(manifest_path),
                  "adapter_code_sha256": sha256(Path(__file__).resolve())}
    immutable_json(ready / "run_status.json", run_status)
    record = {"status": "SYNTHETIC_TEST_NOT_EVIDENCE" if synthetic_test else "PASS", "validated": {"subjects": N_SUBJECTS, "runs": 2 * N_SUBJECTS, "strict_file_hashes": len(strict_files), "strict_subject_manifest_hashes": N_SUBJECTS, "uniform_primary_confounds": 26},
              "outputs": {str(path): sha256(path) for path in (ready / "extraction_handoff_original.json", parcels_path, wide_path, manifest_path, ready / "run_status.json")}}
    immutable_json(ready / "analysis_ready_adapter_record.json", record)
    return record


def make_synthetic_fixture(destination: Path) -> Path:
    """Build labelled fake N33 inputs solely for adapter schema/geometry tests."""
    destination.mkdir(parents=True, exist_ok=True)
    extraction = destination / "extraction_common_coverage"
    extraction.mkdir()
    labels = [native for native in range(1, 401) if native != EXCLUDED_NATIVE]
    networks = [name for name, count in NETWORK_COUNTS.items() for _ in range(count)]
    parcels = pd.DataFrame({"local_matrix_index_0based": np.arange(N_PARCELS), "atlas_map_storage_code_1based": np.arange(1, N_PARCELS + 1), "native_label_id": labels,
                            "physical_label": [f"7Networks_LH_{network}_{i + 1}" for i, network in enumerate(networks)]})
    native_path = extraction / "native_ids_local399.csv"; parcels.to_csv(native_path, index=False)
    np.save(extraction / "native_ids_local399.npy", np.asarray(labels, dtype=np.int16))
    subjects = [f"SS{i:03d}" for i in range(1, 17)] + [f"DS{i:03d}" for i in range(1, 18)]
    groups = ["SS"] * 16 + ["DS"] * 17
    matrix = np.zeros((N_PARCELS, N_PARCELS), dtype=np.float32)
    matrix_path = extraction / "synthetic_matrix.npy"; np.save(matrix_path, matrix)
    raw_path = extraction / "synthetic_raw.npy"; np.save(raw_path, np.zeros((24, N_PARCELS), dtype=np.float32))
    atlas_path = extraction / "synthetic_atlas.bin"; atlas_path.write_bytes(b"synthetic labelled local399 atlas")
    source_paths = {}
    for kind in ("bold", "mask", "confounds", "events"):
        path = extraction / f"synthetic_{kind}.txt"; path.write_text(f"synthetic {kind}\n"); source_paths[kind] = str(path)
    source_hashes = {path: sha256(Path(path)) for path in source_paths.values()}
    common_atlas_hashes = {str(atlas_path): sha256(atlas_path)}
    manifests = extraction / "manifests"; manifests.mkdir()
    run_rows = []
    strict_subjects = []
    primary_names = [f"motion_{i}" for i in range(24)] + ["white_matter", "csf"]
    for subject, group in zip(subjects, groups):
        runs = []
        for run in (1, 2):
            nuisance = {"n_fd_valid": 23, "sum_fd_retained": 2.3 + run / 10, "mean_fd_retained": (2.3 + run / 10) / 23}
            variants = {name: {"matrix_path": str(matrix_path), "matrix_sha256": sha256(matrix_path), "n_confounds": 26 if name == "primary" else 0,
                               "confound_names": primary_names if name == "primary" else [], "within_default91_mean_fisher_z": 0.0} for name in ("primary", "minimal_tr1", "primary_gsr", "primary_task", "primary_task_gsr")}
            runs.append({"run": run, "source_inputs": source_paths, "source_input_hashes": source_hashes, "raw_timeseries_path": str(raw_path), "raw_timeseries_sha256": sha256(raw_path),
                         "atlas": {"atlas_map_path": str(atlas_path), "atlas_map_sha256": sha256(atlas_path)}, "nuisance": nuisance, "variants": variants})
            run_rows.append({"subject": subject, "run": run, "dmn_conn_primary_run": 0.0, "n_fd_valid": 23, "sum_fd_retained_mm": nuisance["sum_fd_retained"], "mean_fd_retained_mm": nuisance["mean_fd_retained"]})
        manifest = {"status": "complete", "subject": subject, "parent_contract_sha256": "synthetic-parent", "coverage_amendment_sha256": "synthetic-amendment", "nominal_atlas_parcels": 400,
                    "effective_common_coverage_parcels": N_PARCELS, "excluded_native_labels": [EXCLUDED_NATIVE], "source_atlas_raw_input_hashes": common_atlas_hashes, "runs": runs,
                    "participant": {name: {"matrix_path": str(matrix_path), "matrix_sha256": sha256(matrix_path)} for name in variants}}
        path = manifests / f"sub-{subject}_manifest.json"; path.write_text(json.dumps(manifest, indent=2) + "\n")
        strict_subjects.append({"path": str(path), "sha256": sha256(path)})
    score_path = extraction / "participant_scores_resolved.csv"; pd.DataFrame({"subject": subjects, "group": groups, "dmn_conn": np.zeros(N_SUBJECTS), "dmn_conn_legacy": np.zeros(N_SUBJECTS),
        "dmn_conn_primary": np.zeros(N_SUBJECTS), "dmn_conn_minimal_tr1": np.zeros(N_SUBJECTS), "dmn_conn_primary_gsr": np.zeros(N_SUBJECTS), "dmn_conn_primary_task": np.zeros(N_SUBJECTS), "dmn_conn_primary_task_gsr": np.zeros(N_SUBJECTS), "mean_fd_resolved": [(2.3 + .1 + 2.3 + .2) / 46] * N_SUBJECTS}).to_csv(score_path, index=False)
    stack_path = extraction / "primary_participant_matrices.npy"; np.save(stack_path, np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS), dtype=np.float32))
    order_path = extraction / "primary_matrix_subjects.csv"; pd.DataFrame({"row_index": np.arange(N_SUBJECTS), "subject": subjects, "group": groups}).to_csv(order_path, index=False)
    runlong_path = extraction / "runwise_primary_scores.csv"; pd.DataFrame(run_rows).to_csv(runlong_path, index=False)
    files = [score_path, stack_path, order_path, runlong_path, extraction / "native_ids_local399.npy", native_path]
    strict = {"status": "complete_33", "parent_contract_sha256": "synthetic-parent", "coverage_amendment_sha256": "synthetic-amendment", "matrix_shape": [N_SUBJECTS, N_PARCELS, N_PARCELS],
              "effective_common_coverage_parcels": N_PARCELS, "excluded_native_labels": [EXCLUDED_NATIVE], "subjects": subjects,
              "files": [{"path": str(path), "sha256": sha256(path)} for path in files], "subject_manifests": strict_subjects}
    (extraction / "strict_analysis_input_manifest.json").write_text(json.dumps(strict, indent=2) + "\n")
    raw = {"status": RAW_HANDOFF_STATUS, "synthetic_test_only": True,
           "participant_scores": {"path": str(score_path), "primary_column": "dmn_conn_primary", "variant_columns": {name: f"dmn_conn_{name}" for name in ("primary", "minimal_tr1", "primary_gsr", "primary_task", "primary_task_gsr")}, "mean_fd_column": "mean_fd_resolved"},
           "primary_matrices": {"path": str(stack_path), "shape": [N_SUBJECTS, N_PARCELS, N_PARCELS]}, "matrix_subjects": {"path": str(order_path)}, "parcel_metadata": {"path": str(native_path)},
           "runwise_primary": {"path": str(runlong_path), "run1_column": "dmn_conn_primary_run", "run2_column": "dmn_conn_primary_run"}, "primary_confounds": {"n_confounds": 26, "names": primary_names}}
    (extraction / "inference_input_manifest.json").write_text(json.dumps(raw, indent=2) + "\n")
    return extraction


def self_test(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    extraction = make_synthetic_fixture(destination)
    record = validate_and_transform(extraction, destination / "common399_analysis_ready", synthetic_test=True)
    output_manifest = read_json(destination / "common399_analysis_ready" / "analysis_ready_input_manifest.json")
    ensure(output_manifest["status"] == "SYNTHETIC_TEST_NOT_EVIDENCE" and output_manifest["synthetic_test_only"] is True,
           "Synthetic output is not explicitly labelled non-evidence")
    raw_path = extraction / "inference_input_manifest.json"
    incomplete = read_json(raw_path); incomplete["status"] = "IN_PROGRESS"
    raw_path.write_text(json.dumps(incomplete) + "\n")
    rejected = False
    try:
        validate_and_transform(extraction, destination / "incomplete_should_not_write", synthetic_test=True)
    except RuntimeError as exc:
        rejected = "exact complete-N33" in str(exc)
    ensure(rejected, "Incomplete cohort gate did not reject before writing analysis-ready output")
    report = {"status": "PASS_SYNTHETIC_NOT_EVIDENCE", "geometry": output_manifest["common_coverage_geometry"], "schema": {"parcel_columns": list(pd.read_csv(destination / "common399_analysis_ready" / "physical_yeo7_common399_parcels.csv").columns), "runwide_columns": list(pd.read_csv(destination / "common399_analysis_ready" / "runwise_primary_scores_wide.csv").columns)}, "incomplete_cohort_gate_rejected": True, "adapter_record": record}
    (destination / "adapter_synthetic_test_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, default=DEFAULT_EXTRACTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_READY)
    parser.add_argument("--synthetic-test", action="store_true")
    parser.add_argument("--test-output", type=Path, default=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[6])) + '/scratch/common399_handoff_adapter_synthetic_test'))
    args = parser.parse_args()
    if args.synthetic_test:
        self_test(args.test_output)
        return
    report = validate_and_transform(args.extraction, args.output, synthetic_test=False)
    print(json.dumps({"status": report["status"], "output": str(args.output), "validated": report["validated"]}, indent=2))


if __name__ == "__main__":
    main()
