#!/usr/bin/env python3
"""Run the released code in an external data/results workspace.

The workspace mirrors the study-relative paths retained in code_source/. No
participant data are shipped. This launcher never changes the release checkout.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--list',action='store_true',help='List released entry points and their roles')
 p.add_argument('--analysis',help='Entry-point name from analysis_catalog.json')
 p.add_argument('--workspace',type=Path,help='External directory containing the documented study-relative inputs')
 p.add_argument('--prepare-only',action='store_true',help='Copy the released source; do not execute an analysis')
 p.add_argument('arguments',nargs=argparse.REMAINDER,help='Arguments passed to the selected scientific script')
 a=p.parse_args(argv);catalog=json.loads((ROOT/'analysis_catalog.json').read_text());entries=catalog['entry_points']
 if a.list:
  for e in entries:print(f"{e['name']}\t{e['language']}\t{e['role']}\t{e['source']}")
  return 0
 if not a.workspace:p.error('--workspace is required unless --list is used')
 work=a.workspace.expanduser().resolve()
 if work==ROOT or ROOT in work.parents:p.error('The data/results workspace must be outside this checkout')
 if not a.prepare_only and not a.analysis:p.error('--analysis is required unless --prepare-only is used')
 candidates=[e for e in entries if e['name']==a.analysis]
 if a.analysis and len(candidates)!=1:p.error(f'Unknown or ambiguous analysis {a.analysis!r}; use --list')
 files=json.loads((ROOT/'release_manifest.json').read_text())['files'];work.mkdir(parents=True,exist_ok=True)
 for e in files:
  src=ROOT/e['path'];rel=Path(e['study_relative_path']);dst=work/rel
  if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Invalid study-relative source path')
  content=src.read_bytes()
  if hashlib.sha256(content).hexdigest()!=e['release_sha256']:raise RuntimeError(f'Release source hash mismatch: {src}')
  if dst.exists() and dst.read_bytes()!=content:raise RuntimeError(f'Workspace already contains different source; preserve it or use a fresh workspace: {dst}')
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 if a.prepare_only:print(f'Prepared {len(files)} source files in external workspace {work}');return 0
 e=candidates[0];script=work/e['source'];env=os.environ.copy();env['DMN_PROJECT_ROOT']=str(work)
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env.setdefault(k,'1')
 args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
 command=([sys.executable] if e['language']=='python' else ['Rscript'] if e['language']=='r' else ['bash'])+[str(script)]+args
 print('Running:', ' '.join(command),flush=True)
 return subprocess.run(command,cwd=work,env=env).returncode
if __name__=='__main__':raise SystemExit(main())
