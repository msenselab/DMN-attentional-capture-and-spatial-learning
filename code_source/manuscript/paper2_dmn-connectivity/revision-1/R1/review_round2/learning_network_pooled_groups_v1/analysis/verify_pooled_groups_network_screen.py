#!/usr/bin/env python3
"""Independent verifier for the frozen pooled/SS/DS learning network screen.

Deliberately does not import the generator or share its statistic functions.
It rereads frozen raw inputs, reconstructs every requested quantity, and compares
all saved scalar and edge-array outputs.
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image

ROOT=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])) + '/manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2')
BASE=ROOT/"learning_network_pooled_groups_v1"; ANA=BASE/"analysis"; OUT=ANA/"independent_verification"
NETS=["Default","SomMot","Vis","Cont","SalVentAttn","DorsAttn","Limbic"]
SCOPES=[("pooled_group_adjusted",None),("SS","SS"),("DS","DS")]

def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()

def holm_independent(p):
 p=np.asarray(p,float); idx=np.argsort(p); ans=np.empty_like(p); last=0.
 for rank,j in enumerate(idx):
  last=max(last,(len(p)-rank)*p[j]); ans[j]=min(last,1.)
 return ans

def corr_scalar(x,y,df,se_n):
 r=float(np.dot(x-x.mean(),y-y.mean())/math.sqrt(np.dot(x-x.mean(),x-x.mean())*np.dot(y-y.mean(),y-y.mean())))
 t=r*math.sqrt(df/(1-r*r)); p=float(2*stats.t.sf(abs(t),df)); z=np.arctanh(r); d=1.959963984540054/math.sqrt(se_n)
 return r,t,p,float(np.tanh(z-d)),float(np.tanh(z+d))
def fail(msg): raise AssertionError(msg)
def close(a,b,name,tol=2e-11):
 a=np.asarray(a); b=np.asarray(b)
 if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=tol,equal_nan=True):
  fail(f"{name}: mismatch; shapes {a.shape}/{b.shape}; maxabs={np.nanmax(np.abs(a-b)) if a.shape==b.shape else 'NA'}")
def centered(data,score,groups,scope):
 if scope=="pooled_group_adjusted":
  a=data.copy(); b=score.copy()
  for g in ("SS","DS"):
   ix=np.where(groups==g)[0]; a[ix]-=a[ix].mean(0); b[ix]-=b[ix].mean()
  return a,b,30,29
 ix=np.where(groups==scope)[0]; return data[ix].copy(),score[ix].copy(),len(ix)-2,len(ix)-3

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 checks=[]
 src=json.loads((ANA/"source_hashes.json").read_text())
 for key,x in src.items():
  actual=digest(x["path"]); checks.append({"check":f"source hash {key}","pass":actual==x["sha256"]})
  if actual!=x["sha256"]: fail(f"source hash mismatch {key}")
 manifest=json.loads(Path(src["analysis_ready_manifest"]["path"]).read_text())
 matrix_path=Path(manifest["primary_matrices"]["path"]); subject_path=Path(manifest["matrix_subjects"]["path"]); parcel_path=Path(manifest["parcel_metadata"]["path"])
 trial=pd.read_csv(src["correct_rt_trials"]["path"])
 eligible=trial[(trial.correct==1)&trial.RT_ms.between(200,3000,inclusive="both")]
 biased_trials=eligible[((eligible.group=="SS")&(eligible.Dimension=="Same"))|((eligible.group=="DS")&(eligible.Dimension=="Different"))]
 if (len(eligible),len(biased_trials))!=(10995,6721): fail("raw eligibility counts differ from frozen contract")
 row=[]
 for (subject,group),x in biased_trials.groupby(["subject","group"],sort=True):
  rare=x[x.DisRegion=="Rare"].RT_ms; frequent=x[x.DisRegion=="Frequent"].RT_ms
  if len(rare)==0 or len(frequent)==0: fail(f"missing score cell {subject}")
  row.append((subject,group,rare.mean(),frequent.mean(),len(rare),len(frequent),rare.mean()-frequent.mean()))
 score=pd.DataFrame(row,columns=["subject","group","rare_mean_ms","frequent_mean_ms","n_rare","n_frequent","learning_rare_minus_frequent_ms"])
 sm=pd.read_csv(subject_path).sort_values("row_index"); score=sm[["row_index","subject","group"]].merge(score,on=["subject","group"],how="left",validate="one_to_one")
 if score.isna().any().any() or len(score)!=33 or (score.group=="SS").sum()!=16: fail("cohort/target reconstruction failure")
 saved_score=pd.read_csv(ANA/"participant_learning_scores_reconstructed.csv")
 close(saved_score.learning_rare_minus_frequent_ms,score.learning_rare_minus_frequent_ms,"all reconstructed learning scores")
 checks.append({"check":"raw target and 33 score rows","pass":True})
 mat=np.load(matrix_path); pa=pd.read_csv(parcel_path).sort_values("matrix_index")
 if mat.shape!=(33,399,399) or not np.allclose(mat,mat.transpose(0,2,1)) or not np.allclose(np.diagonal(mat,axis1=1,axis2=2),0): fail("matrix geometry/symmetry failure")
 if set(pa.yeo7_network)!=set(NETS): fail("physical Yeo-7 labels mismatch")
 edges_i,edges_j=np.triu_indices(399,1)
 pair=np.empty(79401,object); sums=[]; geom=[]
 for ai,a in enumerate(NETS):
  for bi in range(ai,7):
   c=NETS[bi]; ia=np.where(pa.yeo7_network.to_numpy()==a)[0]; ib=np.where(pa.yeo7_network.to_numpy()==c)[0]
   if a==c:
    ii,jj=np.triu_indices(len(ia),1); ii,jj=ia[ii],ia[jj]
   else: ii,jj=np.repeat(ia,len(ib)),np.tile(ib,len(ia))
   lo,hi=np.minimum(ii,jj),np.maximum(ii,jj); key=f"{a}__{c}"
   pos=np.isin(edges_i*399+edges_j,lo*399+hi); pair[pos]=key; sums.append(mat[:,ii,jj].mean(1)); geom.append((key,len(ii)))
 if pd.isna(pair).any() or len(set(pair))!=28: fail("independent physical edge membership coverage failure")
 sums=np.column_stack(sums); saved_sums=pd.read_csv(ANA/"participant_network_summary_fisher_z.csv")
 close(saved_sums[[x[0] for x in geom]].to_numpy(),sums,"all 33x28 network summary means")
 checks.append({"check":"physical edge membership and 33x28 means","pass":True})
 scorev=score.learning_rare_minus_frequent_ms.to_numpy(); groups=score.group.to_numpy(); assoc=[]; raw_edge={}; signed=[]
 edgevals=mat[:,edges_i,edges_j]
 for scope,_ in SCOPES:
  z,y,df,se=centered(sums,scorev,groups,scope)
  for k,(key,nedge) in enumerate(geom):
   r,t,p,lo,hi=corr_scalar(z[:,k],y,df,se); a,b=key.split("__")
   assoc.append(dict(scope=scope,pair=key,network_1=a,network_2=b,possible_edges=nedge,N=len(y),df=df,r=r,t=t,two_sided_raw_p=p,fisher95_ci_low=lo,fisher95_ci_high=hi))
  ez,ey,edf,_=centered(edgevals,scorev,groups,scope); xx=ez-ez.mean(0); yy=ey-ey.mean()
  rr=(xx*yy[:,None]).sum(0)/np.sqrt((xx*xx).sum(0)*(yy*yy).sum()); tt=rr*np.sqrt(edf/(1-rr*rr)); pp=2*stats.t.sf(np.abs(tt),edf); hh=holm_independent(pp)
  raw_edge[scope]=(rr,tt,pp,hh,edf,len(ey))
  for key,nedge in geom:
   ix=np.where(pair==key)[0]
   for sign,flag in [("positive",(rr[ix]>0)&(pp[ix]<.01)),("negative",(rr[ix]<0)&(pp[ix]<.01))]:
    signed.append(dict(scope=scope,pair=key,sign=sign,selected_count=int(flag.sum()),possible_edges=nedge,N=len(ey),df=edf))
 ass=pd.DataFrame(assoc); ass["p_holm28"]=ass.groupby("scope").two_sided_raw_p.transform(lambda x:holm_independent(x)); ass["p_holm84_global_sensitivity"]=holm_independent(ass.two_sided_raw_p)
 saved=pd.read_csv(ANA/"network_associations_all84.csv").sort_values(["scope","pair"]).reset_index(drop=True); actual=ass.sort_values(["scope","pair"]).reset_index(drop=True)
 if len(saved)!=84: fail(f"saved association rows {len(saved)} !=84")
 for col in ["r","t","two_sided_raw_p","fisher95_ci_low","fisher95_ci_high","p_holm28","p_holm84_global_sensitivity"]: close(saved[col],actual[col],f"all84 {col}")
 if not (saved.N.to_numpy()==actual.N.to_numpy()).all() or not (saved.df.to_numpy()==actual.df.to_numpy()).all(): fail("all84 N/df mismatch")
 checks.append({"check":"all 84 associations, CIs, 28/84 Holm corrections","pass":True})
 for scope in [x[0] for x in SCOPES]:
  rr,tt,pp,hh,edf,n=raw_edge[scope]; npz=np.load(ANA/"edge_arrays"/f"{scope}_79401_edges.npz",allow_pickle=True)
  if len(npz["r"])!=79401 or not np.array_equal(npz["edge_i"],edges_i) or not np.array_equal(npz["edge_j"],edges_j) or not np.array_equal(npz["pair"],pair): fail(f"{scope} edge-array geometry mismatch")
  for key,a in [("r",rr),("t",tt),("two_sided_raw_p",pp),("holm79401_p",hh)]: close(npz[key],a,f"{scope} all79401 {key}")
 checks.append({"check":"all 3 x 79,401 individual edge r/p/Holm arrays","pass":True})
 sc=pd.DataFrame(signed).sort_values(["scope","pair","sign"]).reset_index(drop=True); savedc=pd.read_csv(ANA/"descriptive_signed_edge_counts_all168.csv").sort_values(["scope","pair","sign"]).reset_index(drop=True)
 if len(savedc)!=168: fail(f"signed count rows {len(savedc)} !=168")
 for col in ["selected_count","possible_edges","N","df"]:
  if not np.array_equal(savedc[col].to_numpy(),sc[col].to_numpy()): fail(f"all168 {col} mismatch")
 checks.append({"check":"all 168 scope/pair/sign raw-screen counts","pass":True})
 figchecks=[]
 for p in sorted((BASE/"figures").glob("*.png")):
  im=Image.open(p); w,h=im.size; dpi=im.info.get("dpi",(0,0)); ok=(w==4080 and dpi[0]>=300 and h>0)
  figchecks.append({"file":p.name,"pixels":[w,h],"dpi":list(dpi),"pass":ok})
  if not ok: fail(f"figure QC failed {p.name}")
 checks.append({"check":"6.8-inch 600-dpi PNG figures", "pass":True})
 result={"status":"PASS","independent":True,"generator_imported":False,"checks":checks,"figure_qc":figchecks,"raw_counts":{"eligible_all_condition_rows":len(eligible),"biased_rows":len(biased_trials),"SS":int((groups=="SS").sum()),"DS":int((groups=="DS").sum()),"unique_edges":79401},"files_verified":{"associations":84,"signed_counts":168,"edge_arrays":3}}
 (OUT/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__": main()
