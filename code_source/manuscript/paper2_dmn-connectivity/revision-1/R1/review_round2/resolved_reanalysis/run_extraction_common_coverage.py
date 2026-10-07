#!/usr/bin/env python3
"""Coverage-amended Schaefer extraction: native 323 is uniformly omitted.

This is a new, standalone derivative runner.  It never modifies the frozen
400-parcel extraction, contract, or output tree.  It can reuse only a complete
pre-amendment raw400 cache whose manifest proves all relevant source hashes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nilearn import image, signal
from nilearn.glm.first_level import make_first_level_design_matrix

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[6])))
HERE = Path(__file__).resolve().parent
PARENT_CONTRACT = HERE / "analysis_contract.json"
AMENDMENT = HERE / "analysis_contract_coverage_amendment.json"
ORIGINAL = HERE / "extraction"
FMRIPREP = ROOT / "derivatives/fmriprep"
SCORES = ROOT / "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/participant_scores_n33.csv"
ATLAS = Path(__import__('os').environ.get('DMN_ATLAS_CACHE', str(__import__('pathlib').Path.home() / 'nilearn_data')) + '/schaefer_2018/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
LABELS = Path(__import__('os').environ.get('DMN_ATLAS_CACHE', str(__import__('pathlib').Path.home() / 'nilearn_data')) + '/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt')
TR = 1.0
NOMINAL = 400
EXCLUDED_NATIVE = 323
N = 399
NATIVE_IDS = np.array([i for i in range(1, NOMINAL + 1) if i != EXCLUDED_NATIVE], dtype=np.int16)
MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
VARIANTS = ("primary", "minimal_tr1", "primary_gsr", "primary_task", "primary_task_gsr")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, values)
    os.replace(tmp, path)


def find_input(subject: str, run: int, kind: str) -> Path:
    for run_text in (str(run), f"0{run}"):
        stem = f"sub-{subject}_task-suppression_run-{run_text}"
        if kind == "bold":
            path = FMRIPREP / f"sub-{subject}/func/{stem}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        elif kind == "mask":
            path = FMRIPREP / f"sub-{subject}/func/{stem}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        elif kind == "confounds":
            path = FMRIPREP / f"sub-{subject}/func/{stem}_desc-confounds_timeseries.tsv"
        elif kind == "events":
            path = ROOT / f"sub-{subject}/func/{stem}_events.tsv"
        else:
            raise ValueError(kind)
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {kind} for sub-{subject}, run {run}")


def parse_labels() -> tuple[list[str], np.ndarray, list[str]]:
    rows = [line.split("\t") for line in LABELS.read_text().splitlines() if line.strip()]
    if len(rows) != NOMINAL:
        raise RuntimeError(f"Expected {NOMINAL} labels, found {len(rows)}")
    physical = []
    for expected, fields in enumerate(rows, 1):
        if len(fields) < 2 or int(fields[0]) != expected:
            raise RuntimeError(f"Malformed native label row {expected}")
        physical.append(fields[1])
    labels = [physical[i - 1] for i in NATIVE_IDS]
    default = np.array(["Default" in label for label in labels], dtype=bool)
    networks = [label.split("_")[2] for label in labels]
    counts = {x: networks.count(x) for x in set(networks)}
    expected = {"Default": 91, "SomMot": 77, "Vis": 61, "Cont": 52, "SalVentAttn": 47, "DorsAttn": 46, "Limbic": 25}
    if int(default.sum()) != 91 or counts != expected:
        raise RuntimeError(f"Unexpected amended labels/counts: Default={int(default.sum())}, networks={counts}")
    return labels, default, physical


def check_bold_header(bold: Path) -> tuple[nib.spatialimages.SpatialImage, int]:
    img = nib.load(str(bold))
    if img.ndim != 4:
        raise RuntimeError(f"Expected 4-D BOLD: {bold}")
    zooms = img.header.get_zooms()
    if len(zooms) < 4 or not np.isclose(float(zooms[3]), TR, atol=1e-7, rtol=0):
        raise RuntimeError(f"NIfTI header TR is not exactly {TR}: {bold} ({zooms})")
    return img, int(img.shape[3])


def make_local_atlas(bold: Path, mask: Path, out_path: Path) -> tuple[np.ndarray, dict]:
    bold_img, _ = check_bold_header(bold)
    mask_img = nib.load(str(mask))
    if mask_img.shape != bold_img.shape[:3] or not np.allclose(mask_img.affine, bold_img.affine, atol=1e-5):
        raise RuntimeError("Brain-mask grid mismatch")
    atlas = image.resample_img(nib.load(str(ATLAS)), target_affine=bold_img.affine,
        target_shape=bold_img.shape[:3], interpolation="nearest", force_resample=True, copy_header=False)
    native = np.rint(np.asanyarray(atlas.dataobj)).astype(np.int16, copy=False)
    if np.any((native < 0) | (native > NOMINAL)):
        raise RuntimeError("Source atlas codes outside native 0..400")
    native = np.where(np.asanyarray(mask_img.dataobj) > 0, native, 0).astype(np.int16, copy=False)
    native_counts = np.bincount(native.ravel(order="F"), minlength=NOMINAL + 1)[1:]
    missing_other = np.flatnonzero((native_counts == 0) & (np.arange(1, NOMINAL + 1) != EXCLUDED_NATIVE)) + 1
    if missing_other.size:
        raise RuntimeError(f"Unexpected missing common native labels: {missing_other.tolist()}")
    if native_counts[EXCLUDED_NATIVE - 1] != 0:
        # The amended rule is still uniform exclusion even if a future mask covers it.
        pass
    lookup = np.zeros(NOMINAL + 1, dtype=np.uint16)
    lookup[NATIVE_IDS] = np.arange(1, N + 1, dtype=np.uint16)  # map storage is 1..399; matrices are 0..398
    local = lookup[native]
    counts = np.bincount(local.ravel(order="F"), minlength=N + 1)[1:]
    if np.any(counts == 0) or int(local.max()) != N:
        raise RuntimeError("Local399 atlas map has empty/invalid labels")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = bold_img.header.copy(); header.set_data_dtype(np.uint16); header.set_data_shape(local.shape)
    nib.save(nib.Nifti1Image(local, bold_img.affine, header), str(out_path))
    saved = nib.load(str(out_path))
    if saved.shape != bold_img.shape[:3] or not np.allclose(saved.affine, bold_img.affine, atol=1e-5):
        raise RuntimeError("Local atlas write grid verification failed")
    return local, {"atlas_map_path": str(out_path), "atlas_map_sha256": sha256(out_path),
        "atlas_map_storage_codes": "0=background; 1..399 correspond to matrix local indices 0..398",
        "native_excluded_label": EXCLUDED_NATIVE, "native_coverage_counts_nominal400": native_counts.astype(int).tolist(),
        "local_coverage_counts_399": counts.astype(int).tolist(), "shape_xyz": list(local.shape),
        "affine": np.asarray(bold_img.affine).round(8).tolist(), "header_tr_seconds": float(bold_img.header.get_zooms()[3])}


def stream_raw399(bold: Path, local: np.ndarray) -> np.ndarray:
    img, ntime = check_bold_header(bold)
    if tuple(img.shape[:3]) != tuple(local.shape):
        raise RuntimeError("BOLD/local atlas mismatch")
    dtype = np.dtype(img.header.get_data_dtype()); nvox = int(np.prod(img.shape[:3]))
    flat = local.ravel(order="F").astype(np.intp, copy=False)
    counts = np.bincount(flat, minlength=N + 1)[1:].astype(float)
    if np.any(counts == 0): raise RuntimeError("Empty local parcel")
    out = np.empty((ntime, N), dtype=np.float64)
    with gzip.open(bold, "rb") as f:
        offset = int(img.dataobj.offset)
        if len(f.read(offset)) != offset: raise RuntimeError("Cannot reach NIfTI data offset")
        for t in range(ntime):
            raw = f.read(nvox * dtype.itemsize)
            if len(raw) != nvox * dtype.itemsize: raise RuntimeError(f"Truncated BOLD frame {t}")
            v = np.frombuffer(raw, dtype=dtype).astype(np.float64, copy=False)
            if img.dataobj.slope != 1.0 or img.dataobj.inter != 0.0: v = v * img.dataobj.slope + img.dataobj.inter
            out[t] = np.bincount(flat, weights=v, minlength=N + 1)[1:] / counts
    if not np.isfinite(out).all(): raise RuntimeError("Nonfinite streamed raw399")
    return out


def valid_original_cache(subject: str, run: int, inputs: dict[str, Path]) -> tuple[np.ndarray | None, dict]:
    """Return a subset only when the original *complete* manifest proves it."""
    manifest_path = ORIGINAL / "manifests" / f"sub-{subject}_manifest.json"
    cache_path = ORIGINAL / "raw_timeseries400" / f"sub-{subject}_run-{run}_raw_means.npy"
    provenance = {"attempted": True, "original_manifest_path": str(manifest_path), "original_raw400_path": str(cache_path)}
    if not manifest_path.exists() or not cache_path.exists():
        provenance.update({"used": False, "reason": "missing_complete_manifest_or_raw400"}); return None, provenance
    try: m = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        provenance.update({"used": False, "reason": "invalid_original_manifest"}); return None, provenance
    if m.get("status") != "complete" or len(m.get("runs", [])) != 2:
        provenance.update({"used": False, "reason": "original_manifest_not_complete"}); return None, provenance
    old = next((r for r in m["runs"] if r.get("run") == run), None)
    if old is None or old.get("raw_timeseries_path") != str(cache_path):
        provenance.update({"used": False, "reason": "missing_original_run_provenance"}); return None, provenance
    for kind, path in inputs.items():
        if m.get("input_hashes", {}).get(str(path)) != sha256(path):
            provenance.update({"used": False, "reason": f"source_hash_mismatch_{kind}"}); return None, provenance
    for path in (ATLAS, LABELS):
        if m.get("input_hashes", {}).get(str(path)) != sha256(path):
            provenance.update({"used": False, "reason": "source_atlas_or_labels_hash_mismatch"}); return None, provenance
    if old.get("raw_timeseries_sha256") != sha256(cache_path):
        provenance.update({"used": False, "reason": "original_raw400_hash_mismatch"}); return None, provenance
    raw400 = np.load(cache_path, mmap_mode="r")
    if raw400.ndim != 2 or raw400.shape[1] != NOMINAL or not np.isfinite(raw400).all():
        provenance.update({"used": False, "reason": "invalid_original_raw400"}); return None, provenance
    provenance.update({"used": True, "reason": "complete_original_manifest_and_source_raw_hashes_validated",
        "original_raw400_sha256": old["raw_timeseries_sha256"], "original_raw400_shape": list(raw400.shape)})
    return np.asarray(raw400[:, NATIVE_IDS - 1]), provenance


def read_confounds(path: Path, n_frames: int) -> tuple[np.ndarray, dict, dict]:
    conf = pd.read_csv(path, sep="\t", na_values=["n/a", "NA", "nan"])
    if len(conf) != n_frames: raise RuntimeError("Confound/BOLD frame drift")
    derivative = [f"{x}_derivative1" for x in MOTION]
    needed = MOTION + derivative + ["white_matter", "csf", "global_signal"]
    if missing := [x for x in needed if x not in conf]: raise RuntimeError(f"Missing confounds: {missing}")
    x = conf[needed].apply(pd.to_numeric, errors="coerce")
    repaired = []
    for col in derivative:
        bad = np.flatnonzero(~np.isfinite(x[col].to_numpy(float)))
        if bad.size:
            if not np.array_equal(bad, np.array([0])): raise RuntimeError(f"Derivative nonfinite beyond frame 0: {col}")
            x.loc[0, col] = 0.; repaired.append(col)
    for col in set(needed) - set(derivative):
        if not np.isfinite(x[col].to_numpy(float)).all(): raise RuntimeError(f"Nonfinite confound {col}")
    flags = [c for c in conf.columns if c.startswith("non_steady_state_outlier")]
    f = conf[flags].apply(pd.to_numeric, errors="coerce").to_numpy(float) if flags else np.zeros((n_frames, 0))
    if not np.isfinite(f).all(): raise RuntimeError("Nonfinite nonsteady flags")
    keep = ~(f > 0).any(axis=1) if flags else np.ones(n_frames, bool)
    if keep.sum() < 20: raise RuntimeError("Too few retained frames")
    motion24 = np.column_stack([x[MOTION].to_numpy(float), x[derivative].to_numpy(float), x[MOTION].to_numpy(float)**2, x[derivative].to_numpy(float)**2])
    payload = {"primary": np.column_stack([motion24, x[["white_matter", "csf"]].to_numpy(float)]), "gsr": x[["global_signal"]].to_numpy(float)}
    fd = pd.to_numeric(conf.get("framewise_displacement", pd.Series(np.nan, index=conf.index)), errors="coerce").to_numpy(float)
    bad_fd = np.flatnonzero(~np.isfinite(fd))
    if bad_fd.size and not np.array_equal(bad_fd, np.array([0])):
        raise RuntimeError(f"framewise_displacement must be finite except undefined frame 0: {bad_fd.tolist()}")
    # The new handoff basis is raw fMRIPrep FD in mm on retained nonsteady volumes.
    # Frame 0 is deliberately excluded, never filled with a synthetic zero.
    fd_valid = keep & np.isfinite(fd) & (np.arange(n_frames) != 0)
    if not fd_valid.any():
        raise RuntimeError("No retained finite fMRIPrep FD volumes")
    meta = {"nonsteady_columns": flags, "nonsteady_removed_indices": np.flatnonzero(~keep).astype(int).tolist(),
        "n_original_frames": n_frames, "n_retained_frames": int(keep.sum()), "first_frame_derivative_zero_repaired": repaired,
        "legacy_mean_fd_all_nonfirst_including_nonsteady": float(pd.Series(fd).iloc[1:].mean()),
        "fd_units": "mm", "fd_source_column": "framewise_displacement", "fd_first_frame_excluded_not_imputed": True,
        "n_fd_valid": int(fd_valid.sum()), "sum_fd_retained": float(fd[fd_valid].sum()), "mean_fd_retained": float(fd[fd_valid].mean()),
        "fd_valid_indices": np.flatnonzero(fd_valid).astype(int).tolist()}
    return keep, payload, meta


def task_regressors(path: Path, n_frames: int) -> tuple[np.ndarray, dict]:
    ev = pd.read_csv(path, sep="\t")
    if any(c not in ev for c in ("Onset", "Duration", "Dimension")): raise RuntimeError("Events lack Onset/Duration/Dimension")
    onset, duration = pd.to_numeric(ev.Onset, errors="coerce").to_numpy(float), pd.to_numeric(ev.Duration, errors="coerce").to_numpy(float)
    mapping = {"Same":"Same", "Different":"Different", "Absent":"Absent", "None":"Absent"}; dim = ev.Dimension.astype(str).to_numpy()
    if set(dim)-set(mapping) or not np.isfinite(onset).all() or not np.isfinite(duration).all() or np.any(duration <= 0): raise RuntimeError("Invalid Glover events")
    if np.any(onset < 0) or np.any(onset + duration > n_frames * TR + 1e-9): raise RuntimeError("Events outside TR=1 bounds")
    design = make_first_level_design_matrix(np.arange(n_frames)*TR, pd.DataFrame({"onset":onset,"duration":duration,"trial_type":[mapping[x] for x in dim]}), hrf_model="glover", drift_model=None)
    cols = ["Absent", "Different", "Same"]
    if any(c not in design for c in cols): raise RuntimeError("Expected all Glover task columns")
    return design[cols].to_numpy(float), {"task_regressor_names":[f"glover_{x}" for x in cols], "event_counts": pd.Series([mapping[x] for x in dim]).value_counts().sort_index().to_dict()}


def variant_confounds(variant: str, payload: dict, task: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
    blocks, names = [], []
    if variant != "minimal_tr1":
        blocks.append(payload["primary"]); names += [*MOTION, *[f"{x}_derivative1" for x in MOTION], *[f"{x}_power2" for x in MOTION], *[f"{x}_derivative1_power2" for x in MOTION], "white_matter", "csf"]
    if variant in ("primary_gsr", "primary_task_gsr"): blocks.append(payload["gsr"]); names.append("global_signal")
    if variant in ("primary_task", "primary_task_gsr"): blocks.append(task); names += ["glover_Absent", "glover_Different", "glover_Same"]
    return (None, names) if not blocks else (np.column_stack(blocks), names)


def fisher(cleaned: np.ndarray) -> np.ndarray:
    if cleaned.shape[1] != N or not np.isfinite(cleaned).all(): raise RuntimeError("Invalid local399 clean data")
    corr = np.corrcoef(cleaned.T)
    if corr.shape != (N, N) or not np.isfinite(corr).all(): raise RuntimeError("Invalid local399 correlation")
    z = np.arctanh(np.clip(corr, -1 + 1e-7, 1 - 1e-7)); np.fill_diagonal(z, 0.); return z


def network_summary(matrix: np.ndarray, labels: list[str]) -> list[dict]:
    networks = sorted({x.split("_")[2] for x in labels}); membership = np.array([x.split("_")[2] for x in labels])
    if len(networks) != 7: raise RuntimeError("Expected 7 networks")
    rows=[]
    for i, a_name in enumerate(networks):
        a=np.flatnonzero(membership==a_name)
        for b_name in networks[i:]:
            b=np.flatnonzero(membership==b_name)
            vals=matrix[np.ix_(a,b)][np.triu_indices(len(a),1)] if a_name==b_name else matrix[np.ix_(a,b)].ravel()
            rows.append({"network_i":a_name,"network_j":b_name,"n_edges":int(vals.size),"mean_fisher_z":float(vals.mean())})
    if len(rows)!=28: raise RuntimeError("Expected 28 network summaries")
    return rows


def manifest_path(out: Path, subject: str) -> Path: return out / "manifests" / f"sub-{subject}_manifest.json"


def process_subject(subject: str, out: Path, labels: list[str], default: np.ndarray, parent_hash: str, amendment_hash: str, force: bool) -> None:
    mp = manifest_path(out, subject)
    if mp.exists() and not force:
        old=json.loads(mp.read_text())
        if old.get("status")=="complete" and old.get("parent_contract_sha256")==parent_hash and old.get("coverage_amendment_sha256")==amendment_hash and old.get("runner_sha256")==sha256(Path(__file__).resolve()):
            print(f"resume: sub-{subject} already complete", flush=True); return
    all_runs=[]; participant_mats={v:[] for v in VARIANTS}; source_hashes={str(ATLAS):sha256(ATLAS),str(LABELS):sha256(LABELS)}
    for run in (1,2):
        inputs={k:find_input(subject,run,k) for k in ("bold","mask","confounds","events")}
        source_hashes.update({str(p):sha256(p) for p in inputs.values()})
        _, n_frames=check_bold_header(inputs["bold"])
        atlas_path=out/"atlasmaps_local399"/f"sub-{subject}_run-{run}_schaefer_local399_uint16.nii.gz"
        local, atlas_info=make_local_atlas(inputs["bold"],inputs["mask"],atlas_path)
        raw, cache=valid_original_cache(subject,run,inputs)
        if raw is None:
            raw=stream_raw399(inputs["bold"],local); cache["streamed_bold_due_to_cache_ineligibility"]=True
        if raw.shape != (n_frames,N): raise RuntimeError("Raw399 frame/parcel shape mismatch")
        raw_path=out/"raw_timeseries399"/f"sub-{subject}_run-{run}_raw_means.npy"; atomic_npy(raw_path,raw)
        keep,payload,nuisance=read_confounds(inputs["confounds"],n_frames); task,task_meta=task_regressors(inputs["events"],n_frames)
        rec={"subject":subject,"run":run,"source_inputs":{k:str(v) for k,v in inputs.items()},"source_input_hashes":{str(v):sha256(v) for v in inputs.values()},"raw_timeseries_path":str(raw_path),"raw_timeseries_sha256":sha256(raw_path),"raw_timeseries_shape":list(raw.shape),"original_raw400_cache_provenance":cache,"atlas":atlas_info,"nuisance":nuisance,"task":task_meta,"variants":{}}
        for variant in VARIANTS:
            conf,names=variant_confounds(variant,payload,task)
            rank=int(np.linalg.matrix_rank(np.ones((int(keep.sum()),1)) if conf is None else np.column_stack([np.ones((int(keep.sum()),1)),conf[keep]])))
            clean=signal.clean(raw,confounds=conf,detrend=True,standardize="zscore_sample",standardize_confounds=True,low_pass=.10,high_pass=.01,t_r=TR,sample_mask=np.flatnonzero(keep),ensure_finite=False)
            if clean.shape != (int(keep.sum()),N) or not np.isfinite(clean).all(): raise RuntimeError(f"Invalid clean output {subject} run{run} {variant}")
            mat=fisher(clean); p=out/"run_matrices399"/variant/f"sub-{subject}_run-{run}_fisherz.npy"; atomic_npy(p,mat); participant_mats[variant].append(mat)
            dvals=mat[np.ix_(default,default)][np.triu_indices(int(default.sum()),1)]
            rec["variants"][variant]={"matrix_path":str(p),"matrix_sha256":sha256(p),"matrix_shape":[N,N],"n_confounds":0 if conf is None else int(conf.shape[1]),"confound_names":names,"design_rank_after_nonsteady_mask":rank,"algebraic_residual_df":int(keep.sum())-rank,"within_default91_mean_fisher_z":float(dvals.mean())}
        all_runs.append(rec); print(f"complete run: sub-{subject} run {run}",flush=True)
    participant={}
    for variant,mats in participant_mats.items():
        combined=(mats[0]+mats[1])/2.; np.fill_diagonal(combined,0.)
        p=out/"participant_matrices399"/variant/f"sub-{subject}_fisherz_meanruns.npy"; atomic_npy(p,combined)
        dvals=combined[np.ix_(default,default)][np.triu_indices(int(default.sum()),1)]
        participant[variant]={"matrix_path":str(p),"matrix_sha256":sha256(p),"matrix_shape":[N,N],"within_default91_mean_fisher_z":float(dvals.mean()),"network_summary":network_summary(combined,labels)}
    atomic_json(mp,{"status":"complete","generated_utc":now(),"subject":subject,"parent_contract_path":str(PARENT_CONTRACT),"parent_contract_sha256":parent_hash,"coverage_amendment_path":str(AMENDMENT),"coverage_amendment_sha256":amendment_hash,"runner_path":str(Path(__file__).resolve()),"runner_sha256":sha256(Path(__file__).resolve()),"nominal_atlas_parcels":NOMINAL,"effective_common_coverage_parcels":N,"excluded_native_labels":[EXCLUDED_NATIVE],"matrix_index_definition":"matrix local indices 0..398 map via explicit native_ids_local399.npy/csv; native 323 excluded","source_atlas_raw_input_hashes":source_hashes,"runs":all_runs,"participant":participant,"software":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"nibabel":nib.__version__,"nilearn":__import__('nilearn').__version__}})
    print(f"complete subject: sub-{subject}",flush=True)


def emit_handoff(out: Path, scores: pd.DataFrame, subjects: list[str], default: np.ndarray, parent_hash: str, amendment_hash: str) -> bool:
    if not all(manifest_path(out,s).exists() and json.loads(manifest_path(out,s).read_text()).get("status")=="complete" for s in subjects): return False
    rows=[]; mats=[]; runrows=[]
    for subject in subjects:
        m=json.loads(manifest_path(out,subject).read_text()); row={"subject":subject}
        for v in VARIANTS: row[f"dmn_conn_{v}"]=m["participant"][v]["within_default91_mean_fisher_z"]
        rows.append(row); mats.append(np.load(m["participant"]["primary"]["matrix_path"],mmap_mode="r"))
        for r in m["runs"]:
            q=r["nuisance"]
            runrows.append({"subject":subject,"run":r["run"],"dmn_conn_primary_run":r["variants"]["primary"]["within_default91_mean_fisher_z"],"mean_fd_legacy_qc_all_nonfirst_including_nonsteady":q["legacy_mean_fd_all_nonfirst_including_nonsteady"],"mean_fd_retained_mm":q["mean_fd_retained"],"n_fd_valid":q["n_fd_valid"],"sum_fd_retained_mm":q["sum_fd_retained"],"n_original_frames":q["n_original_frames"],"n_retained_frames":q["n_retained_frames"],"nonsteady_removed_indices":json.dumps(q["nonsteady_removed_indices"]),"fd_valid_indices":json.dumps(q["fd_valid_indices"]),"raw_cache_reused":r["original_raw400_cache_provenance"]["used"]})
    summary=pd.DataFrame(rows)
    runfd=pd.DataFrame(runrows).groupby("subject",sort=False).agg(n_fd_valid=("n_fd_valid","sum"),sum_fd_retained_mm=("sum_fd_retained_mm","sum")).reset_index()
    runfd["mean_fd_resolved"]=runfd["sum_fd_retained_mm"] / runfd["n_fd_valid"]
    if not np.isfinite(runfd["mean_fd_resolved"]).all() or (runfd["n_fd_valid"] <= 0).any(): raise RuntimeError("Invalid pooled retained-volume FD")
    summary=summary.merge(runfd,on="subject",validate="one_to_one"); merged=scores.merge(summary,on="subject",validate="one_to_one")
    if len(merged)!=33: raise RuntimeError("Full handoff requires 33 scores")
    if "dmn_conn" in merged: merged["dmn_conn_legacy"]=merged["dmn_conn"]
    merged["dmn_conn_primary_definition"]="physical_Yeo7_Default91_mean_upper_triangle_runwise_Fisher_z_then_equal_run_mean; common399 (native323 omitted uniformly)"
    merged.to_csv(out/"participant_scores_resolved.csv",index=False)
    atomic_npy(out/"primary_participant_matrices.npy",np.stack(mats,axis=0))
    order=pd.DataFrame({"row_index":np.arange(33),"subject":subjects,"group":merged.set_index("subject").loc[subjects,"group"].to_numpy()}); order.to_csv(out/"primary_matrix_subjects.csv",index=False)
    pd.DataFrame(runrows).to_csv(out/"runwise_primary_scores.csv",index=False)
    files=[out/"participant_scores_resolved.csv",out/"primary_participant_matrices.npy",out/"primary_matrix_subjects.csv",out/"runwise_primary_scores.csv",out/"native_ids_local399.npy",out/"native_ids_local399.csv"]
    atomic_json(out/"strict_analysis_input_manifest.json",{"status":"complete_33","generated_utc":now(),"parent_contract_sha256":parent_hash,"coverage_amendment_sha256":amendment_hash,"nominal_atlas_parcels":NOMINAL,"effective_common_coverage_parcels":N,"excluded_native_labels":[EXCLUDED_NATIVE],"matrix_shape":[33,N,N],"subjects":subjects,"files":[{"path":str(p),"sha256":sha256(p)} for p in files],"subject_manifests":[{"path":str(manifest_path(out,s)),"sha256":sha256(manifest_path(out,s))} for s in subjects]})
    primary_names=json.loads(manifest_path(out,subjects[0]).read_text())["runs"][0]["variants"]["primary"]["confound_names"]
    if len(primary_names)!=26 or any(json.loads(manifest_path(out,s).read_text())["runs"][r]["variants"]["primary"]["confound_names"] != primary_names for s in subjects for r in range(2)):
        raise RuntimeError("Primary 26-confound metadata is not uniform across the full cohort")
    atomic_json(out/"inference_input_manifest.json",{"status":"COMPLETE_33_EXTRACTION_HANDOFF_NOT_PIPELINE_COMPLETE","generated_utc":now(),"participant_scores":{"path":str(out/"participant_scores_resolved.csv"),"primary_column":"dmn_conn_primary","variant_columns":{v:f"dmn_conn_{v}" for v in VARIANTS},"mean_fd_column":"mean_fd_resolved","mean_fd_provenance":"Raw fMRIPrep framewise_displacement in mm; retained nonsteady-state volumes only; frame 0 excluded rather than zero-imputed; two-run pooled mean weighted by finite retained FD-volume count."},"primary_matrices":{"path":str(out/"primary_participant_matrices.npy"),"shape":[33,N,N]},"matrix_subjects":{"path":str(out/"primary_matrix_subjects.csv")},"parcel_metadata":{"path":str(out/"native_ids_local399.csv"),"matrix_index_definition":"local matrix 0..398 maps explicitly to native IDs; native 323 omitted"},"runwise_primary":{"path":str(out/"runwise_primary_scores.csv"),"run1_column":"dmn_conn_primary_run","run2_column":"dmn_conn_primary_run","fd_columns":{"n_fd_valid":"n_fd_valid","sum_fd_mm":"sum_fd_retained_mm","mean_fd_retained_mm":"mean_fd_retained_mm"}},"primary_confounds":{"n_confounds":26,"names":primary_names,"sample_mask":"non_steady_state_outlier rows removed exactly as runwise metadata"},"parent_contract_sha256":parent_hash,"coverage_amendment_sha256":amendment_hash})
    return True


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--subjects",nargs="+"); ap.add_argument("--output",type=Path,default=HERE/"extraction_common_coverage"); ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    if not all(p.exists() for p in (PARENT_CONTRACT,AMENDMENT,ATLAS,LABELS,SCORES)): raise FileNotFoundError("Parent contract, amendment, atlas, labels, or score source missing")
    parent=json.loads(PARENT_CONTRACT.read_text()); amendment=json.loads(AMENDMENT.read_text()); parent_hash=sha256(PARENT_CONTRACT); amendment_hash=sha256(AMENDMENT)
    if parent_hash != amendment.get("parent_contract_sha256") or amendment.get("excluded_native_labels") != [EXCLUDED_NATIVE] or amendment.get("effective_common_coverage_parcels") != N or parent.get("tr_seconds") != TR: raise RuntimeError("Frozen contract/amendment mismatch")
    labels,default,physical=parse_labels(); args.output.mkdir(parents=True,exist_ok=True)
    atomic_npy(args.output/"native_ids_local399.npy",NATIVE_IDS)
    pd.DataFrame({"local_matrix_index_0based":np.arange(N),"atlas_map_storage_code_1based":np.arange(1,N+1),"native_label_id":NATIVE_IDS,"physical_label":[physical[i-1] for i in NATIVE_IDS]}).to_csv(args.output/"native_ids_local399.csv",index=False)
    scores=pd.read_csv(SCORES,dtype={"subject":str}).sort_values("subject").reset_index(drop=True); all_subjects=scores.subject.tolist()
    if len(all_subjects)!=33 or len(set(all_subjects))!=33: raise RuntimeError("Expected 33 unique frozen score subjects")
    subjects=args.subjects or all_subjects
    if unknown:=sorted(set(subjects)-set(all_subjects)): raise RuntimeError(f"Subjects outside frozen N33: {unknown}")
    for subject in subjects: process_subject(subject,args.output,labels,default,parent_hash,amendment_hash,args.force)
    full=emit_handoff(args.output,scores,all_subjects,default,parent_hash,amendment_hash)
    atomic_json(args.output/"run_status.json",{"generated_utc":now(),"requested_subjects":subjects,"n_requested":len(subjects),"full_cohort_handoff_emitted":full,"parent_contract_sha256":parent_hash,"coverage_amendment_sha256":amendment_hash,"nominal_atlas_parcels":NOMINAL,"effective_common_coverage_parcels":N,"variants":list(VARIANTS)})
    print(json.dumps({"requested_subjects":subjects,"full_cohort_handoff_emitted":full,"output":str(args.output)},indent=2))

if __name__ == "__main__": main()
