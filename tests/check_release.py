#!/usr/bin/env python3
"""Validate the committed source snapshot; synthetic checks are not evidence."""
from pathlib import Path
import ast,collections,hashlib,importlib.util,json,os,re,shutil,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def main():
 m=json.loads((ROOT/'release_manifest.json').read_text());files=m['files']
 expected={e['path'] for e in files};assert len(expected)==len(files),'Duplicate manifest paths'
 actual={str(p.relative_to(ROOT)) for p in (ROOT/'code_source').rglob('*') if p.is_file()}
 assert actual==expected,('Source allowlist mismatch',sorted(actual^expected))
 counts=collections.Counter();rparsed=0
 for e in files:
  p=ROOT/e['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==e['release_sha256'],str(p)
  assert p.suffix.lower() in {'.py','.r','.m','.md','.sh'},str(p)
  assert not p.is_symlink(),str(p)
  text=p.read_text();counts[p.suffix.lower()]+=1
  assert not re.search(r'/home/[^/\s\"\']+|/dss/studies/|/dss/tmp|\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY',text),('Private path/secret pattern',str(p))
  if p.suffix.lower()=='.py':compile(text,str(p),'exec')
  if p.suffix.lower()=='.r':
   assert shutil.which('Rscript'),'Rscript required for full syntax validation'
   v=subprocess.run(['Rscript','-e','invisible(parse(file=commandArgs(TRUE)[1]));cat("PARSE_OK")',str(p)],capture_output=True,text=True)
   assert v.returncode==0,(str(p),v.stderr);rparsed+=1
  if p.suffix.lower()=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
 for p in [ROOT/'run_analysis.py',Path(__file__)]:compile(p.read_text(),str(p),'exec')
 catalog=json.loads((ROOT/'analysis_catalog.json').read_text())['entry_points'];assert len({e['name'] for e in catalog})==len(catalog)
 for e in catalog:assert 'code_source/'+e['source'] in expected
 by_name={e['name']:e for e in catalog};assert 'connectivity_coverage_audit' in by_name
 cpmtext=(ROOT/'code_source'/by_name['nested_cpm_capture']['source']).read_text()
 assert "default=RES/'common399_analysis_ready/physical_yeo7_common399_parcels.csv'" in cpmtext
 assert "default=RES/'extraction_common_coverage/physical_yeo7_400_parcels.csv'" not in cpmtext
 assert '--check-inputs-only' in cpmtext
 inputsdoc=(ROOT/'docs/INPUT_CONTRACTS.md').read_text()
 assert all(x in inputsdoc for x in ['analysis_contract.json','analysis_contract_coverage_amendment.json','coverage_qc/coverage_audit.json'])
 assert not any(e['study_relative_path'].endswith('finish_extraction_handoff.py') for e in files)
 # Actual released implementation versus the independent scalar training-fold
 # implementation, using labelled synthetic random arrays only.
 import numpy as np
 rel='manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/cpm/run_resolved_cpm.py'
 p=ROOT/'code_source'/rel;spec=importlib.util.spec_from_file_location('released_cpm_check',p)
 assert spec is not None and spec.loader is not None
 cpm=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpm)
 rng=np.random.default_rng(57);X=rng.normal(size=(12,24));y=rng.normal(size=12);groups=np.array(['SS']*6+['DS']*6);families=np.repeat(np.arange(6),4)
 folds=cpm.legacy.prepare_folds(X,groups)
 r,pred,target,selected=cpm.legacy.evaluate_batch(y[:,None],groups,families,folds,.3)
 reference_pred,reference_target=cpm.scalar_predictions(X,y,groups,families,.3)
 assert np.allclose(pred[0],reference_pred,atol=1e-10,rtol=0)
 assert np.allclose(target[0],reference_target,atol=1e-12,rtol=0)
 independently_centered=np.array([y[i]-y[(np.arange(len(y))!=i)&(groups==groups[i])].mean() for i in range(len(y))])
 assert np.allclose(target[0],independently_centered,atol=1e-12,rtol=0)
 result={'status':'PASS_SOURCE_AND_SYNTHETIC_CHECKS_NOT_EMPIRICAL_REPRODUCTION','source_files':len(files),'source_types':dict(sorted(counts.items())),'entry_points':len(catalog),'R_scripts_parsed':rparsed,'synthetic_CPM_scalar_vectorized_agreement':True,'training_only_group_centering_checked':True,'empirical_inputs_in_release':False,'all_10000_empirical_permutations_rerun_for_release':False}
 print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
