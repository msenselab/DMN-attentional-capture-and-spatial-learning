#!/usr/bin/env python3
"""Independent verifier for coverage-amended local399 extraction outputs.

It deliberately does not import the runner.  It validates frozen hashes and
coverage metadata, spot-checks raw means using both manual gzip and nibabel
frame decoding, and independently recleans every variant.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import signal
from nilearn.glm.first_level import make_first_level_design_matrix

HERE=Path(__file__).resolve().parent
PARENT=HERE/"analysis_contract.json"
AMENDMENT=HERE/"analysis_contract_coverage_amendment.json"
TR=1.0; NOMINAL=400; N=399; EXCLUDED=323
MOTION=["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]
VARIANTS=("primary","minimal_tr1","primary_gsr","primary_task","primary_task_gsr")


def sha256(path: Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()

def fail(s:str)->None: raise RuntimeError(s)

def manual_frame(bold:Path,frame:int)->np.ndarray:
 img=nib.load(str(bold)); dtype=np.dtype(img.header.get_data_dtype()); nvox=int(np.prod(img.shape[:3])); nbytes=nvox*dtype.itemsize
 with gzip.open(bold,"rb") as f:
  skip=int(img.dataobj.offset)+frame*nbytes
  if len(f.read(skip))!=skip: fail("Manual decoder cannot reach requested frame")
  raw=f.read(nbytes)
 if len(raw)!=nbytes: fail("Manual decoder got truncated frame")
 x=np.frombuffer(raw,dtype=dtype).astype(np.float64,copy=False)
 return x if (img.dataobj.slope==1.0 and img.dataobj.inter==0.0) else x*img.dataobj.slope+img.dataobj.inter

def means(values:np.ndarray,codes:np.ndarray)->np.ndarray:
 flat=codes.ravel(order="F").astype(np.intp,copy=False); cnt=np.bincount(flat,minlength=N+1)[1:].astype(float)
 if np.any(cnt==0): fail("Saved local atlas has an empty parcel")
 return np.bincount(flat,weights=values,minlength=N+1)[1:]/cnt

def independent_confounds(path:Path,nframes:int)->tuple[dict,np.ndarray,dict]:
 tab=pd.read_csv(path,sep="\t",na_values=["n/a","NA","nan"])
 if len(tab)!=nframes: fail("Confound frame count mismatch")
 derivative=[f"{x}_derivative1" for x in MOTION]; needed=MOTION+derivative+["white_matter","csf","global_signal"]
 if any(x not in tab for x in needed): fail("Required confound missing")
 x=tab[needed].apply(pd.to_numeric,errors="coerce")
 for c in derivative:
  bad=np.flatnonzero(~np.isfinite(x[c].to_numpy(float)))
  if bad.size:
   if not np.array_equal(bad,np.array([0])): fail(f"Derivative nonfinite outside frame 0: {c}")
   x.loc[0,c]=0.
 for c in set(needed)-set(derivative):
  if not np.isfinite(x[c].to_numpy(float)).all(): fail(f"Nonfinite confound: {c}")
 flags=[c for c in tab if c.startswith("non_steady_state_outlier")]
 fv=tab[flags].apply(pd.to_numeric,errors="coerce").to_numpy(float) if flags else np.zeros((nframes,0))
 if not np.isfinite(fv).all(): fail("Nonfinite nonsteady flags")
 keep=~(fv>0).any(axis=1) if flags else np.ones(nframes,bool)
 m24=np.column_stack([x[MOTION].to_numpy(float),x[derivative].to_numpy(float),x[MOTION].to_numpy(float)**2,x[derivative].to_numpy(float)**2])
 fd=pd.to_numeric(tab.get("framewise_displacement",pd.Series(np.nan,index=tab.index)),errors="coerce").to_numpy(float)
 bad=np.flatnonzero(~np.isfinite(fd))
 if bad.size and not np.array_equal(bad,np.array([0])): fail(f"FD nonfinite outside undefined first frame: {bad.tolist()}")
 valid=keep & np.isfinite(fd) & (np.arange(nframes)!=0)
 if not valid.any(): fail("No finite retained FD volumes")
 return {"primary":np.column_stack([m24,x[["white_matter","csf"]].to_numpy(float)]),"gsr":x[["global_signal"]].to_numpy(float)},keep,{"removed":np.flatnonzero(~keep).astype(int).tolist(),"columns":flags,"n_fd_valid":int(valid.sum()),"sum_fd_retained":float(fd[valid].sum()),"mean_fd_retained":float(fd[valid].mean()),"fd_valid_indices":np.flatnonzero(valid).astype(int).tolist()}

def independent_task(path:Path,nframes:int)->np.ndarray:
 ev=pd.read_csv(path,sep="\t")
 if any(c not in ev for c in ("Onset","Duration","Dimension")): fail("Events missing required columns")
 onset=pd.to_numeric(ev.Onset,errors="coerce").to_numpy(float); duration=pd.to_numeric(ev.Duration,errors="coerce").to_numpy(float)
 mapping={"Same":"Same","Different":"Different","Absent":"Absent","None":"Absent"}; dim=ev.Dimension.astype(str).to_numpy()
 if set(dim)-set(mapping) or not np.isfinite(onset).all() or not np.isfinite(duration).all() or np.any(duration<=0): fail("Invalid events")
 if np.any(onset<0) or np.any(onset+duration>nframes*TR+1e-9): fail("Out-of-bounds events")
 design=make_first_level_design_matrix(np.arange(nframes)*TR,pd.DataFrame({"onset":onset,"duration":duration,"trial_type":[mapping[x] for x in dim]}),hrf_model="glover",drift_model=None)
 cols=["Absent","Different","Same"]
 if any(c not in design for c in cols): fail("Glover event design missing a condition")
 return design[cols].to_numpy(float)

def variant(variant:str,payload:dict,task:np.ndarray)->np.ndarray|None:
 blocks=[]
 if variant!="minimal_tr1": blocks.append(payload["primary"])
 if variant in ("primary_gsr","primary_task_gsr"): blocks.append(payload["gsr"])
 if variant in ("primary_task","primary_task_gsr"): blocks.append(task)
 return None if not blocks else np.column_stack(blocks)

def fisher(x:np.ndarray)->np.ndarray:
 c=np.corrcoef(x.T)
 if c.shape!=(N,N) or not np.isfinite(c).all(): fail("Bad independent local399 correlation")
 z=np.arctanh(np.clip(c,-1+1e-7,1-1e-7)); np.fill_diagonal(z,0.); return z

def verify_subject(subject:str,out:Path)->dict:
 mp=out/"manifests"/f"sub-{subject}_manifest.json"
 if not mp.exists(): fail(f"Missing manifest for {subject}")
 m=json.loads(mp.read_text())
 if m.get("status")!="complete": fail("Manifest is not complete")
 if m.get("parent_contract_sha256")!=sha256(PARENT) or m.get("coverage_amendment_sha256")!=sha256(AMENDMENT): fail("Frozen parent/amendment hash mismatch")
 if m.get("nominal_atlas_parcels")!=NOMINAL or m.get("effective_common_coverage_parcels")!=N or m.get("excluded_native_labels")!=[EXCLUDED]: fail("Coverage metadata mismatch")
 native=np.load(out/"native_ids_local399.npy")
 if native.shape!=(N,) or native.dtype.kind not in "iu" or EXCLUDED in native or not np.array_equal(native,np.array([i for i in range(1,401) if i!=EXCLUDED])): fail("Explicit native ID map invalid")
 checks=[]
 for rec in m["runs"]:
  if rec["run"] not in (1,2): fail("Unexpected run")
  paths={k:Path(v) for k,v in rec["source_inputs"].items()}
  if set(paths)!={"bold","mask","confounds","events"}: fail("Incomplete source input provenance")
  for p,d in rec["source_input_hashes"].items():
   if not Path(p).exists() or sha256(Path(p))!=d: fail(f"Source input hash mismatch: {p}")
  bold=paths["bold"]; img=nib.load(str(bold)); zooms=img.header.get_zooms()
  if img.ndim!=4 or len(zooms)<4 or not np.isclose(zooms[3],TR,atol=1e-7,rtol=0): fail("NIfTI TR/header failure")
  atlas_path=Path(rec["atlas"]["atlas_map_path"]); raw_path=Path(rec["raw_timeseries_path"])
  if sha256(atlas_path)!=rec["atlas"]["atlas_map_sha256"] or sha256(raw_path)!=rec["raw_timeseries_sha256"]: fail("Derived atlas/raw hash failure")
  atlas=nib.load(str(atlas_path)); codes=np.asanyarray(atlas.dataobj)
  if codes.dtype.kind!="u" or codes.dtype.itemsize<2 or codes.shape!=img.shape[:3] or int(codes.max())!=N: fail("Local399 atlas storage failure")
  counts=np.bincount(codes.ravel(order="F"),minlength=N+1)[1:]
  if np.any(counts==0) or rec["atlas"]["native_excluded_label"]!=EXCLUDED: fail("Local399 coverage failure")
  if len(rec["atlas"]["native_coverage_counts_nominal400"])!=NOMINAL or rec["atlas"]["native_coverage_counts_nominal400"][EXCLUDED-1] < 0: fail("Invalid native323 coverage metadata")
  raw=np.load(raw_path,mmap_mode="r")
  if raw.shape!=(img.shape[3],N) or not np.isfinite(raw).all(): fail("Raw399 shape/finiteness failure")
  frames=sorted(set([0,raw.shape[0]//2,raw.shape[0]-1])); max_manual=0.; max_nib=0.
  for frame in frames:
   a=means(manual_frame(bold,frame),codes)
   b=means(np.asanyarray(img.dataobj[...,frame],dtype=np.float64).ravel(order="F"),codes)
   max_manual=max(max_manual,float(np.max(np.abs(a-raw[frame])))); max_nib=max(max_nib,float(np.max(np.abs(b-raw[frame]))))
  if max_manual>1e-10 or max_nib>1e-5: fail(f"Raw spot means mismatch: manual={max_manual} nibabel={max_nib}")
  cache=rec["original_raw400_cache_provenance"]
  if cache.get("used"):
   old=Path(cache["original_raw400_path"])
   if not old.exists() or sha256(old)!=cache.get("original_raw400_sha256"): fail("Claimed reused raw400 cache does not hash-match")
   oldx=np.load(old,mmap_mode="r")
   if oldx.shape!=(raw.shape[0],NOMINAL) or np.max(np.abs(oldx[:,native-1]-raw))>0: fail("Raw399 is not exact native-ID subset of claimed raw400 cache")
  payload,keep,nonsteady=independent_confounds(paths["confounds"],raw.shape[0]); task=independent_task(paths["events"],raw.shape[0])
  if nonsteady["removed"]!=rec["nuisance"]["nonsteady_removed_indices"]: fail("Sample-mask mismatch")
  for k in ("n_fd_valid","fd_valid_indices"):
   if nonsteady[k] != rec["nuisance"][k]: fail(f"Retained-volume FD metadata mismatch: {k}")
  for k in ("sum_fd_retained","mean_fd_retained"):
   if not np.isclose(nonsteady[k],rec["nuisance"][k],atol=1e-14,rtol=0): fail(f"Retained-volume FD metadata mismatch: {k}")
  varchecks={}
  for v in VARIANTS:
   conf=variant(v,payload,task)
   clean=signal.clean(np.asarray(raw),confounds=conf,detrend=True,standardize="zscore_sample",standardize_confounds=True,low_pass=.10,high_pass=.01,t_r=TR,sample_mask=np.flatnonzero(keep),ensure_finite=False)
   recomputed=fisher(clean); p=Path(rec["variants"][v]["matrix_path"]); saved=np.load(p,mmap_mode="r")
   if sha256(p)!=rec["variants"][v]["matrix_sha256"] or saved.shape!=(N,N) or not np.isfinite(saved).all() or not np.allclose(saved,saved.T,atol=1e-12) or not np.array_equal(np.diag(saved),np.zeros(N)): fail(f"Saved matrix structural/hash failure: {v}")
   delta=float(np.max(np.abs(recomputed-saved)))
   if delta>1e-10: fail(f"Independent cleaning mismatch {v}: {delta}")
   varchecks[v]={"matrix_max_abs_delta":delta,"n_confounds":0 if conf is None else int(conf.shape[1])}
  checks.append({"run":rec["run"],"header_tr_seconds":float(zooms[3]),"local399_nonempty":True,"spot_frames":frames,"manual_raw_mean_max_abs_delta":max_manual,"nibabel_raw_mean_max_abs_delta":max_nib,"cache_reused":bool(cache.get("used")),"variants":varchecks})
 for v in VARIANTS:
  p=Path(m["participant"][v]["matrix_path"]); saved=np.load(p,mmap_mode="r"); runs=[np.load(Path(r["variants"][v]["matrix_path"]),mmap_mode="r") for r in m["runs"]]
  expected=(runs[0]+runs[1])/2.; np.fill_diagonal(expected,0.)
  delta=float(np.max(np.abs(saved-expected)))
  if sha256(p)!=m["participant"][v]["matrix_sha256"] or delta>1e-12: fail(f"Run mean arithmetic mismatch {v}: {delta}")
 return {"subject":subject,"status":"PASS","runs":checks}

def verify_handoff(out:Path)->dict:
 names=["participant_scores_resolved.csv","primary_participant_matrices.npy","primary_matrix_subjects.csv","runwise_primary_scores.csv","strict_analysis_input_manifest.json","inference_input_manifest.json"]
 present=[x for x in names if (out/x).exists()]
 if not present:return {"status":"not_emitted_for_partial_cohort"}
 if len(present)!=len(names):fail("Incomplete full-cohort handoff")
 scores=pd.read_csv(out/names[0],dtype={"subject":str}); mats=np.load(out/names[1],mmap_mode="r"); order=pd.read_csv(out/names[2],dtype={"subject":str}); runrows=pd.read_csv(out/names[3],dtype={"subject":str}); strict=json.loads((out/names[4]).read_text()); inference=json.loads((out/names[5]).read_text())
 if len(scores)!=33 or mats.shape!=(33,N,N) or len(order)!=33 or not np.array_equal(order.row_index.to_numpy(),np.arange(33)) or len(runrows)!=66:fail("Full handoff shape/count failure")
 if "dmn_conn_legacy" not in scores or "dmn_conn_primary" not in scores or not np.array_equal(scores["dmn_conn_legacy"].to_numpy(),scores["dmn_conn"].to_numpy()):fail("Legacy/primary score preservation failure")
 fd=runrows.groupby("subject",sort=False).agg(n=("n_fd_valid","sum"),total=("sum_fd_retained_mm","sum")); expected=(fd.total/fd.n).reindex(scores.subject).to_numpy()
 if not np.allclose(scores["mean_fd_resolved"].to_numpy(),expected,atol=1e-14,rtol=0) or not np.isfinite(expected).all() or (fd.n<=0).any():fail("Pooled retained-volume FD handoff failure")
 if strict.get("matrix_shape")!=[33,N,N] or len(strict.get("subject_manifests",[]))!=33:fail("Strict handoff manifest failure")
 for item in strict["files"]:
  if not Path(item["path"]).exists() or sha256(Path(item["path"]))!=item["sha256"]:fail("Strict handoff file hash mismatch")
 if inference.get("status")!="COMPLETE_33_EXTRACTION_HANDOFF_NOT_PIPELINE_COMPLETE" or inference.get("participant_scores",{}).get("mean_fd_column")!="mean_fd_resolved" or inference.get("primary_confounds",{}).get("n_confounds")!=26 or len(inference.get("primary_confounds",{}).get("names",[]))!=26:fail("Inference handoff schema/FD/confounds failure")
 if "retained nonsteady-state volumes" not in inference["participant_scores"].get("mean_fd_provenance","") or inference["primary_matrices"].get("shape")!=[33,N,N]:fail("Inference handoff provenance/shape failure")
 return {"status":"PASS","n_subjects":33,"matrix_shape":list(mats.shape),"n_run_rows":len(runrows)}

def main()->None:
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--subjects",nargs="+",default=["DS001"]); ap.add_argument("--output",type=Path,default=HERE/"extraction_common_coverage"); args=ap.parse_args()
 if not PARENT.exists() or not AMENDMENT.exists():fail("Missing frozen parent contract or amendment")
 results=[verify_subject(s,args.output) for s in args.subjects]; result={"status":"PASS","parent_contract_sha256":sha256(PARENT),"coverage_amendment_sha256":sha256(AMENDMENT),"subjects":results,"full_handoff":verify_handoff(args.output)}
 p=args.output/"verification"/("verify_"+"_".join(args.subjects)+".json");p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":"PASS","verification":str(p),"full_handoff":result["full_handoff"]},indent=2))
if __name__=="__main__":main()
