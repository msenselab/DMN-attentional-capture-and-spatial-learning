#!/usr/bin/env python3
"""Archived signed enrichment rerun for the frozen rare-minus-frequent target.

The archive records every SS-then-DS source map and every signed selected-total,
category-count, and surplus vector. It is intentionally isolated from legacy
analysis code and writes only beneath the requested output directory.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(name,"1")
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
N,P,DF,SCREEN_P,EDGE_N=33,399,30,.01,79401
SEED=20261005
ORDER=("Default","SomMot","Vis","Cont","SalVentAttn","DorsAttn","Limbic")
COUNTS={"Default":91,"SomMot":77,"Vis":61,"Cont":52,"SalVentAttn":47,"DorsAttn":46,"Limbic":25}
HERE=Path(__file__).resolve().parent; BASE=HERE.parent
FREEZE=BASE/"input_freeze.json"; TARGET=BASE/"learning_target_scores.csv"
READY=Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])) + '/manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/common399_analysis_ready/analysis_ready_input_manifest.json')
def die(x): raise RuntimeError(x)
def need(x,msg):
 if not x: die(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');t.replace(p)
def pair(a,b): return f'{a}--{b}' if ORDER.index(a)<=ORDER.index(b) else f'{b}--{a}'
def center(x,g):
 o=np.asarray(x,float).copy()
 for lab,n in (("SS",16),("DS",17)):
  ix=np.flatnonzero(g==lab);need(len(ix)==n,f'{lab} count invalid');o[ix]-=o[ix].mean(0,keepdims=True)
 return o
def resolve(mpath,x):
 p=Path(x);return p if p.is_absolute() else mpath.parent/p
def content_sha(a,n):
 x=np.asarray(a[:n]);h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(repr(x.shape).encode());h.update(x.tobytes(order='C'));return h.hexdigest()

class Archive:
 specs={"ss_source_indices":(np.int16,(16,)),"ds_source_indices":(np.int16,(17,)),"positive_selected_total":(np.int32,()),"negative_selected_total":(np.int32,()),"positive_category_counts":(np.int32,(35,)),"negative_category_counts":(np.int32,(35,)),"positive_surplus":(np.float64,(35,)),"negative_surplus":(np.float64,(35,))}
 def __init__(self,root,n,new):
  self.root=root;self.n=n;self.arr={}
  if new:
   need(not root.exists(),f'archive exists: {root}');root.mkdir(parents=True)
   for k,(dt,tail) in self.specs.items():self.arr[k]=np.lib.format.open_memmap(root/f'{k}.npy',mode='w+',dtype=dt,shape=(n,*tail))
  else:
   for k,(dt,tail) in self.specs.items():
    a=np.load(root/f'{k}.npy',mmap_mode='r+');need(a.dtype==np.dtype(dt) and a.shape==(n,*tail),f'archive shape/dtype {k}');self.arr[k]=a
 def write(self,i,ss,ds,pos,neg):
  self.arr['ss_source_indices'][i]=ss;self.arr['ds_source_indices'][i]=ds
  for sign,data in (("positive",pos),("negative",neg)):
   counts,surplus,total=data;self.arr[f'{sign}_selected_total'][i]=total;self.arr[f'{sign}_category_counts'][i]=counts;self.arr[f'{sign}_surplus'][i]=surplus
 def flush(self):
  for a in self.arr.values():a.flush()
 def records(self,done,final=False):
  return {k:{"dtype":a.dtype.str,"shape":list(a.shape),"content_sha256":content_sha(a,done),"file_sha256":sha(a.filename) if final else None} for k,a in self.arr.items()}

def fixture(root):
 root.mkdir(parents=True);rng=np.random.default_rng(913);subs=np.array([f'SS{i:03d}' for i in range(1,17)]+[f'DS{i:03d}' for i in range(1,18)]);groups=np.array(["SS"]*16+["DS"]*17)
 rows=[];native=iter([x for x in range(1,401) if x!=323])
 for net in ORDER:
  for _ in range(COUNTS[net]):rows.append({"matrix_index":len(rows),"native_label":next(native),"yeo7_network":net})
 parcels=pd.DataFrame(rows);latent=rng.normal(size=N);mats=np.empty((N,P,P));default=np.flatnonzero(parcels.yeo7_network.eq("Default"))
 for i in range(N):
  noise=rng.normal(0,.06,(P,P));m=(noise+noise.T)/2;m[np.ix_(default,default)]+=.015*latent[i];np.fill_diagonal(m,0);mats[i]=m
 np.save(root/'matrices.npy',mats);parcels.to_csv(root/'parcels.csv',index=False);pd.DataFrame({"subject":subs,"group":groups,"learning_rare_minus_frequent_ms":50*latent+rng.normal(0,15,N)}).to_csv(root/'target.csv',index=False)
 return mats,groups,parcels,pd.read_csv(root/'target.csv').learning_rare_minus_frequent_ms.to_numpy(),{"synthetic_input":True,"input_hashes":{str(x):sha(x) for x in (root/'matrices.npy',root/'parcels.csv',root/'target.csv')}}

def load_real():
 freeze=json.loads(FREEZE.read_text());need(freeze.get('status')=='FROZEN_BEFORE_NEW_NETWORK_RESULTS','freeze status invalid');need(sha(TARGET)==freeze['target_hash'],'target hash invalid')
 for p,h in freeze['input_hashes'].items():need(sha(p)==h,f'input hash invalid: {p}')
 m=json.loads(READY.read_text());need(m.get('status')=='complete' and m['common_coverage_geometry']['effective_common_coverage_parcels']==399,'ready manifest invalid')
 def p(k):return resolve(READY,m[k]['path'])
 s=pd.read_csv(p('participant_scores'),dtype={'subject':str})[['subject','group']].sort_values('subject').reset_index(drop=True);sub=pd.read_csv(p('matrix_subjects'),dtype={'subject':str});pos={x:i for i,x in enumerate(sub.subject)};need(set(pos)==set(s.subject),'subject IDs invalid')
 mats=np.asarray(np.load(p('primary_matrices'),mmap_mode='r',allow_pickle=False)[[pos[x] for x in s.subject]],float);parcels=pd.read_csv(p('parcel_metadata')).sort_values('matrix_index').reset_index(drop=True);target=pd.read_csv(TARGET,dtype={'subject':str})
 need(mats.shape==(N,P,P) and np.allclose(mats,mats.swapaxes(1,2),atol=1e-8,rtol=0) and np.allclose(np.diagonal(mats,axis1=1,axis2=2),0,atol=1e-10,rtol=0),'matrix contract invalid')
 need(parcels.matrix_index.tolist()==list(range(P)) and set(parcels.native_label)==set(range(1,401))-{323} and parcels.yeo7_network.value_counts().to_dict()==COUNTS,'parcel contract invalid')
 need(np.array_equal(target.subject,s.subject) and np.array_equal(target.group,s.group) and len(target)==N,'target identity/order invalid')
 return mats,s.group.to_numpy(),parcels,target.learning_rare_minus_frequent_ms.to_numpy(float),{"synthetic_input":False,"input_hashes":{str(x):sha(x) for x in (FREEZE,TARGET,READY,p('primary_matrices'),p('matrix_subjects'),p('parcel_metadata'))}}

def partition(parcels):
 ii,jj=np.triu_indices(P,1);labs=parcels.yeo7_network.to_numpy();names=[pair(a,b) for i,a in enumerate(ORDER) for b in ORDER[i:]];lookup={x:i for i,x in enumerate(names)};code=np.array([lookup[pair(a,b)] for a,b in zip(labs[ii],labs[jj])]);possible=np.bincount(code,minlength=28).astype(int);endpoint=np.column_stack([(labs[ii]==x)|(labs[jj]==x) for x in ORDER]);need(len(ii)==EDGE_N and possible.sum()==EDGE_N,'edge partition invalid');return ii,jj,names,code,possible,endpoint
def stat(mask,code,possible,endpoint):
 pc=np.bincount(code,weights=mask.astype(int),minlength=28).astype(int);ec=np.array([mask[:,][endpoint[:,i]].sum() for i in range(7)],int);counts=np.r_[pc,ec];cap=np.r_[possible,endpoint.sum(0)];total=int(mask.sum());surplus=counts-total*cap/EDGE_N;oe=np.divide(counts,total*cap/EDGE_N,out=np.zeros(35),where=(total*cap)>0);return counts,surplus,total,oe

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--permutations',type=int,default=10000);ap.add_argument('--batch-size',type=int,default=20);ap.add_argument('--seed',type=int,default=SEED);ap.add_argument('--resume',action='store_true');ap.add_argument('--stop-after',type=int);ap.add_argument('--synthetic-test',action='store_true');a=ap.parse_args();need(a.permutations>0 and a.batch_size>0 and a.seed==SEED,'positive counts and frozen seed required');need(a.stop_after is None or 0<a.stop_after<a.permutations,'stop-after must be interior')
 out=a.output/"SYNTHETIC_TEST_NOT_EVIDENCE" if a.synthetic_test else a.output;checkpoint=out/'permutation_checkpoint.json';aroot=out/'null_archive';
 if a.synthetic_test:
  inputs=a.output/'synthetic_inputs'
  if a.resume:need(inputs.is_dir(),'synthetic inputs absent');mats,groups,parcels,y,prov=fixture_load(inputs)
  else:need(not a.output.exists(),'refusing existing synthetic root');mats,groups,parcels,y,prov=fixture(inputs)
 else:mats,groups,parcels,y,prov=load_real()
 out.mkdir(parents=True,exist_ok=a.resume);ii,jj,names,code,possible,endpoint=partition(parcels);edge=center(mats[:,ii,jj],groups);enorm=np.sqrt((edge*edge).sum(0));need((enorm>0).all(),'edge variance invalid');cy=center(y,groups);normy=np.sqrt(cy@cy);r0=cy@edge/(normy*enorm);crit=np.sqrt(stats.t.ppf(1-SCREEN_P/2,DF)**2/(stats.t.ppf(1-SCREEN_P/2,DF)**2+DF));obspos=stat((np.abs(r0)>crit)&(r0>0),code,possible,endpoint);obsneg=stat((np.abs(r0)>crit)&(r0<0),code,possible,endpoint)
 config={"seed":SEED,"permutations":a.permutations,"batch_size":a.batch_size,"call_order":"for each draw: rng.permutation(SS target) then rng.permutation(DS target)","screen":{"two_sided_p":SCREEN_P,"df":DF,"r_critical":float(crit)},"statistic":"signed selected-count surplus = category selected count - signed selected total * category possible-edge share; primary upper-surplus tail with +1 correction","families":"pair categories: 28 cells x both signs, Holm-56; endpoints: 7 overlapping categories x both signs, Holm-14","input":prov,"code_sha256":sha(__file__)}
 if a.resume:
  saved=json.loads(checkpoint.read_text());need(saved.get('config')==config,'resume config mismatch');done=int(saved['completed_permutations']);need(0<done<a.permutations,'resume requires incomplete checkpoint');archive=Archive(aroot,a.permutations,False)
  for k,v in saved['archive']['arrays'].items():need(v['content_sha256']==content_sha(archive.arr[k],done),'checkpoint archive hash mismatch')
  rng=np.random.default_rng();rng.bit_generator.state=saved['rng_state'];gepos=np.array(saved['ge_positive'],int);geneg=np.array(saved['ge_negative'],int)
 else:
  need(not checkpoint.exists() and not aroot.exists() and not (out/'enrichment_manifest.json').exists(),'refusing existing output');done=0;archive=Archive(aroot,a.permutations,True);rng=np.random.default_rng(SEED);gepos=np.zeros(35,int);geneg=np.zeros(35,int)
 ss=np.flatnonzero(groups=='SS');ds=np.flatnonzero(groups=='DS');need(len(ss)==16 and len(ds)==17 and len(np.unique(y[ss]))==16 and len(np.unique(y[ds]))==17,'group source mapping invalid');lookups=[{float(v):int(i) for v,i in zip(y[ix],ix)} for ix in (ss,ds)]
 try:
  limit=a.stop_after or a.permutations
  while done<limit:
   n=min(a.batch_size,limit-done);ys=np.empty((n,N));sm=np.empty((n,16),np.int16);dm=np.empty((n,17),np.int16)
   for row in range(n):
    ys[row]=y
    for ix,lookup,dest in ((ss,lookups[0],sm),(ds,lookups[1],dm)):
     perm=rng.permutation(y[ix]);ys[row,ix]=perm;dest[row]=[lookup[float(z)] for z in perm]
   ys=center(ys.T,groups).T; rr=ys@edge/(np.sqrt((ys*ys).sum(1))[:,None]*enorm[None,:])
   for row in range(n):
    po=stat((np.abs(rr[row])>crit)&(rr[row]>0),code,possible,endpoint);ne=stat((np.abs(rr[row])>crit)&(rr[row]<0),code,possible,endpoint);gepos+=po[1]>=obspos[1];geneg+=ne[1]>=obsneg[1];archive.write(done+row,sm[row],dm[row],po[:3],ne[:3])
   done+=n;archive.flush();payload={"status":"COMPLETE_ARCHIVED" if done==a.permutations else "IN_PROGRESS_NOT_EVIDENCE","config":config,"completed_permutations":done,"rng_state":rng.bit_generator.state,"ge_positive":gepos.tolist(),"ge_negative":geneg.tolist(),"archive":{"directory":"null_archive","arrays":archive.records(done)}};dump(checkpoint,payload)
  if done<a.permutations:print(json.dumps({"status":"STOPPED_FOR_RESUME_TEST_NOT_EVIDENCE","completed":done,"output":str(out)},indent=2));return
  cap=np.r_[possible,endpoint.sum(0)];table=pd.DataFrame({"scope":["pair"]*28+["endpoint"]*7,"category":names+list(ORDER),"possible_edges":cap,"positive_selected_observed":obspos[0],"positive_expected_given_screen":obspos[2]*cap/EDGE_N,"positive_observed_expected_ratio":obspos[3],"positive_surplus_statistic":obspos[1],"positive_permutation_p_raw":(1+gepos)/(1+a.permutations),"negative_selected_observed":obsneg[0],"negative_expected_given_screen":obsneg[2]*cap/EDGE_N,"negative_observed_expected_ratio":obsneg[3],"negative_surplus_statistic":obsneg[1],"negative_permutation_p_raw":(1+geneg)/(1+a.permutations)})
  for mask,label in ((table.scope.eq('pair'),'pair-by-sign Holm-56'),(table.scope.eq('endpoint'),'endpoint-by-sign Holm-14')):
   raw=np.r_[table.loc[mask,'positive_permutation_p_raw'],table.loc[mask,'negative_permutation_p_raw']];adj=multipletests(raw,method='holm')[1];k=int(mask.sum());table.loc[mask,'positive_permutation_p_Holm']=adj[:k];table.loc[mask,'negative_permutation_p_Holm']=adj[k:];table.loc[mask,'Holm_family']=label
  table['tail']='upper surplus, +1 correction';table['interpretation']='O/E and counts descriptive; no hypergeometric result is reported.';table.to_csv(out/'signed_enrichment_permutation.csv',index=False)
  am={"archive_version":"learning-enrichment-v1","completed_permutations":done,"source_index_semantics":"absolute sorted-participant source indices; destination order is SS then DS positions","group_destination_indices":{"SS":ss.tolist(),"DS":ds.tolist()},"arrays":archive.records(done,True)};dump(out/'null_archive_manifest.json',am);payload={"status":"COMPLETE_ARCHIVED","config":config,"completed_permutations":done,"rng_state":rng.bit_generator.state,"ge_positive":gepos.tolist(),"ge_negative":geneg.tolist(),"archive":{"directory":"null_archive","arrays":archive.records(done,True)}};dump(checkpoint,payload)
  man={"status":"COMPLETE_EMPIRICAL_ARCHIVE_NOT_PUBLICATION_READY" if not a.synthetic_test else "SYNTHETIC_TEST_NOT_EVIDENCE","publication_ready":False,**config,"completed_permutations":done,"observed_screen":{"edge_universe":EDGE_N,"positive_selected":obspos[2],"negative_selected":obsneg[2]},"outputs":{"table":"signed_enrichment_permutation.csv","archive":"null_archive","checkpoint":"permutation_checkpoint.json"}};dump(out/'enrichment_manifest.json',man);print(json.dumps({"status":man['status'],"output":str(out),"permutations":done,"positive_selected":obspos[2],"negative_selected":obsneg[2]},indent=2))
 except Exception: raise

def fixture_load(root):
 return np.load(root/'matrices.npy'),pd.read_csv(root/'target.csv').group.to_numpy(),pd.read_csv(root/'parcels.csv'),pd.read_csv(root/'target.csv').learning_rare_minus_frequent_ms.to_numpy(),{"synthetic_input":True,"input_hashes":{str(x):sha(x) for x in (root/'matrices.npy',root/'parcels.csv',root/'target.csv')}}
if __name__=='__main__':main()
