#!/usr/bin/env python3
"""Integrate a verified primary_joint optimizer recovery without refitting models."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = HERE / 'results_common399_n33'
DEST = HERE / 'results_common399_n33_optimizer_resolved_v2'
DIAG = BASE / 'model_diagnostics' / 'primary_joint_sameformula_v1'
PLAN = BASE / 'optimizer_recovery_execution_plan_v2.json'
READY = BASE / 'common399_analysis_ready' / 'analysis_ready_input_manifest.json'
REPORTER = BASE / 'model_omnibus_reporting.py'
VERIFIER = HERE / 'verify_dependent_inference.py'
EXPORTER = HERE / 'recovery_integration_v2_export_primary.R'
RVALIDATOR = HERE / 'recovery_integration_v2_validate_rds_tables.R'
SELECTED = DIAG / 'primary_joint_bobyqa_restart_from_nloptwrap.rds'
COMPARATOR = DIAG / 'primary_joint_nloptwrap.rds'
ORIGINAL = SOURCE / 'mixed_models' / 'primary_joint.rds'
TRIALS = SOURCE / 'joint_history_trials_new_primary.csv'

PRIMARY_REGENERATED = {
    'mixed_models/primary_joint.rds',
    'mixed_models/primary_joint_simple_slopes_and_contrasts.csv',
    'mixed_models/primary_joint_optimizer_agreement.csv',
    'mixed_models/primary_joint_diagnostics.csv',
    'mixed_models/primary_joint_summary.txt',
    'mixed_models/primary_joint_omnibus_type3.csv',
    'mixed_models/primary_joint_model_matrix_columns.csv',
    'mixed_models/primary_joint_random_covariance_ms2.csv',
    'mixed_models/primary_joint_fixed_effects_complete.csv',
    'mixed_models/mixed_model_diagnostics_all.csv',
    'mixed_models/mixed_model_simple_slopes_all.csv',
}
STALE_PREFIXES = ('reporting_common399_v1/',)
STALE_FILES = {'independent_dependent_inference_verification.json'}

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def req(p: Path) -> None:
    if not p.is_file(): raise FileNotFoundError(p)

def run(cmd: list[str], env: dict[str, str]) -> str:
    cp = subprocess.run(cmd, cwd=BASE, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if cp.returncode:
        raise RuntimeError('command failed (%s):\n%s' % (' '.join(map(str,cmd)), cp.stdout))
    return cp.stdout

def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--output', type=Path, default=DEST)
    args = ap.parse_args()
    source, dest = args.source.resolve(), args.output.resolve()
    if dest.exists(): raise FileExistsError(f'Refusing to overwrite output: {dest}')
    for p in (PLAN, READY, REPORTER, VERIFIER, EXPORTER, RVALIDATOR, SELECTED, COMPARATOR, ORIGINAL, TRIALS, source/'results_manifest.json'):
        req(p)
    plan = json.loads(PLAN.read_text())
    if plan.get('version') != 'common399-sameformula-optimizer-recovery-v2' or plan.get('publication_ready') is not False:
        raise ValueError('Frozen optimizer recovery plan contract mismatch.')
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    source_hashes_before = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    dest.mkdir(parents=True)
    copied, excluded = [], []
    for src in sorted(source.rglob('*')):
        if not src.is_file(): continue
        rel = src.relative_to(source).as_posix()
        reason = None
        if rel in PRIMARY_REGENERATED: reason = 'regenerated_from_saved_selected_RDS_or_coherent_aggregate'
        elif rel == 'results_manifest.json': reason = 'regenerated_manifest_with_complete_status_and_publication_ready_false'
        elif rel in STALE_FILES: reason = 'stale_verifier_artifact_not_copied'
        elif any(rel.startswith(prefix) for prefix in STALE_PREFIXES): reason = 'stale_reporting_artifact_not_copied'
        if reason:
            excluded.append({'source_relative_path': rel, 'reason': reason, 'source_sha256': sha(src)})
            continue
        target = dest / rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, target)
        if sha(src) != sha(target): raise RuntimeError(f'copy hash mismatch: {rel}')
        copied.append({'source_relative_path': rel, 'destination_relative_path': rel, 'sha256': sha(src), 'classification': 'history_or_nonLMM_byteidentical' if not rel.startswith('mixed_models/primary_joint') else 'unchanged_history_LMM_byteidentical'})
    selected_target = dest / 'mixed_models' / 'primary_joint.rds'; selected_target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SELECTED, selected_target)
    if sha(selected_target) != sha(SELECTED): raise RuntimeError('selected recovery RDS copy hash mismatch')
    exporter_log = run(['Rscript', str(EXPORTER), str(selected_target), str(COMPARATOR), str(ORIGINAL), str(TRIALS), str(dest/'mixed_models')], env)
    (dest/'recovery_integration_v2_export_primary.log').write_text(exporter_log)
    # Coherent aggregates must combine recovered primary output with byte-identical history output.
    import csv
    mixed = dest/'mixed_models'
    def combine(names: list[str], target: Path) -> None:
        fields = None; rows = []
        for name in names:
            with (mixed/name).open(newline='') as f:
                r = csv.DictReader(f); now = list(r.fieldnames or [])
                if fields is None: fields = now
                if now != fields: raise ValueError(f'aggregate schema mismatch in {name}')
                rows.extend(r)
        if fields is None:
            raise ValueError('aggregate has no header')
        with target.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    combine(['primary_joint_diagnostics.csv','primary_joint_history_diagnostics.csv'], mixed/'mixed_model_diagnostics_all.csv')
    combine(['primary_joint_simple_slopes_and_contrasts.csv','primary_joint_history_simple_slopes_and_contrasts.csv'], mixed/'mixed_model_simple_slopes_all.csv')
    original_manifest = json.loads((source/'results_manifest.json').read_text())
    manifest = dict(original_manifest)
    manifest.update({
        'status': 'complete', 'publication_ready': False,
        'optimizer_recovery_integration': {
            'plan_path': str(PLAN), 'plan_sha256': sha(PLAN),
            'selected_solution': 'bobyqa restart initialized at converged nloptwrap theta',
            'selected_rds': str(SELECTED), 'selected_rds_sha256': sha(SELECTED),
            'comparator_rds': str(COMPARATOR), 'comparator_rds_sha256': sha(COMPARATOR),
            'original_primary_rds': str(ORIGINAL), 'original_primary_rds_sha256': sha(ORIGINAL),
            'primary_exporter': str(EXPORTER), 'primary_exporter_sha256': sha(EXPORTER),
            'no_model_refit': True,
            'publication_ready_reason': 'False pending parent readback despite numerical recovery gates.',
        },
        'thread_limits': {k: env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
    })
    atomic_json(dest/'results_manifest.json', manifest)
    # Independent checks execute only after manifest status is complete and outputs exist.
    rvalidation_path = dest/'independent_rds_to_table_validation.json'
    rvalidation_log = run(['Rscript', str(RVALIDATOR), str(selected_target), str(mixed), str(rvalidation_path)], env)
    (dest/'recovery_integration_v2_rds_validation.log').write_text(rvalidation_log)
    rvalidation = json.loads(rvalidation_path.read_text())
    if rvalidation.get('status') != 'PASS': raise RuntimeError('RDS-to-table validation did not PASS')
    verifier_log = run(['python', str(VERIFIER), '--input-manifest', str(READY), '--results-dir', str(dest), '--audit-output', str(dest/'independent_dependent_inference_verification.json')], env)
    (dest/'recovery_integration_v2_dependent_verifier.log').write_text(verifier_log)
    dependent = json.loads((dest/'independent_dependent_inference_verification.json').read_text())
    if dependent.get('status') != 'PASS': raise RuntimeError('dependent verifier did not PASS')
    reporting_dir = dest/'reporting_common399_v2'
    mapper_log = run(['python', str(REPORTER), '--plan', str(BASE/'model_claim_reporting_plan.json'), '--mixed-models-dir', str(mixed), '--output-dir', str(reporting_dir), '--models', 'primary_joint', 'primary_joint_history'], env)
    (reporting_dir/'recovery_integration_v2_holm6_mapper.log').write_text(mapper_log)
    mapper = json.loads(mapper_log)
    if mapper.get('status') != 'FROZEN_PLAN_APPLIED_NOT_CLAIM_SELECTION' or len(mapper.get('mapped', [])) != 2:
        raise RuntimeError('frozen Holm6 mapper did not apply to both models')
    # Source immutability and copied-file byte identity are verified after all commands.
    source_hashes_after = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    source_unchanged = source_hashes_before == source_hashes_after
    if not source_unchanged: raise RuntimeError('source result directory changed during integration')
    bad_copies = [x['destination_relative_path'] for x in copied if sha(source/x['source_relative_path']) != sha(dest/x['destination_relative_path'])]
    if bad_copies: raise RuntimeError('post-integration copied-file mismatch: ' + ', '.join(bad_copies))
    source_artifacts = [PLAN, READY, REPORTER, VERIFIER, EXPORTER, RVALIDATOR, SELECTED, COMPARATOR, ORIGINAL, TRIALS, source/'results_manifest.json']
    ledger = {
        'status': 'PASS', 'scope': 'Optimizer recovery integration v2: saved-RDS re-export only; no lmer refit or scientific model change.',
        'source_results_directory': str(source), 'source_results_directory_unchanged': source_unchanged,
        'copied_byteidentical_count': len(copied), 'copied_byteidentical_files': copied,
        'excluded_from_copy': excluded,
        'regenerated_primary_files': sorted(PRIMARY_REGENERATED - {'mixed_models/primary_joint.rds'}),
        'selected_primary_rds_destination': str(selected_target), 'selected_primary_rds_sha256': sha(selected_target),
        'all_named_source_hashes': {str(p): sha(p) for p in source_artifacts},
        'outputs_sha256': {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'recovery_integration_v2_provenance_ledger.json'},
    }
    atomic_json(dest/'recovery_integration_v2_provenance_ledger.json', ledger)
    report = {
        'status': 'PASS', 'output_directory': str(dest), 'publication_ready': False,
        'paths': {
            'manifest': str(dest/'results_manifest.json'), 'ledger': str(dest/'recovery_integration_v2_provenance_ledger.json'),
            'dependent_verifier': str(dest/'independent_dependent_inference_verification.json'),
            'rds_to_table_validator': str(rvalidation_path), 'holm6_reporting_directory': str(reporting_dir),
        },
        'gates': {
            'saved_primary_RDS_reexport_without_refit': 'PASS', 'original_source_directory_unchanged': 'PASS',
            'history_and_nonLMM_copied_byteidentical': 'PASS', 'coherent_aggregate_tables': 'PASS',
            'independent_dependent_verifier': dependent['status'], 'independent_RDS_to_table_validation': rvalidation['status'],
            'frozen_Holm6_mapper': 'PASS', 'publication_ready': 'FALSE_PENDING_PARENT_READBACK',
        },
        'gaps': dependent.get('coverage_gaps', []),
    }
    atomic_json(dest/'recovery_integration_v2_execution_report.json', report)
    print(json.dumps(report, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
