#!/usr/bin/env python3
"""Read-only verification of current behavioral and Default91 run reliability.

Reconstructs run-specific capture from frozen correct/RT-valid N33 and N35 trials,
then derives pooled uncentered and group-centered biased/unbiased Run1--Run2
statistics.  It also recomputes current common399 Default91 Run1--Run2 values
from the authorized saved N33 run-score file.  No model or image data are fit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
ATTACHED = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt'
R1 = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2'
RUN_DIR = ATTACHED / 'run-to-run-reliability'
READY = R1 / 'resolved_reanalysis/common399_analysis_ready'
RESULTS = R1 / 'resolved_reanalysis/inference/results_common399_n33_optimizer_resolved_v2'

TRIALS = {33: ATTACHED / 'brain_behavior/correct_rt_trials_n33.csv',
          35: ATTACHED / 'brain_behavior/correct_rt_trials_n35.csv'}
SAVED_RUN_SCORES = RUN_DIR / 'participant_run_capture_scores.csv'
SAVED_CENTERED = RUN_DIR / 'run_to_run_results.csv'
RUNWISE_DEFAULT91 = READY / 'runwise_primary_scores_wide.csv'
READY_MANIFEST = READY / 'analysis_ready_input_manifest.json'
RESULTS_MANIFEST = RESULTS / 'results_manifest.json'

R = 10_000
SEED = 20261006
TOL = 2e-10
CHECKS: list[dict] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, label: str) -> None:
    CHECKS.append({'check': label, 'pass': bool(value)})
    if not value:
        raise AssertionError(label)


def close(x, y, tol=TOL) -> bool:
    return bool(np.allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float), atol=tol, rtol=0, equal_nan=True))


def center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=float).copy()
    for group in ('SS', 'DS'):
        idx = np.flatnonzero(groups == group)
        z[idx] -= z[idx].mean()
    return z


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind='mergesort')
    q = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(q)
    out[order] = q
    return out


def stats_row(x: np.ndarray, y: np.ndarray, df: int, fisher_se: float) -> dict:
    r = float(stats.pearsonr(x, y).statistic)
    t = r * math.sqrt(df / (1 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    z = math.atanh(r)
    ci = np.tanh([z - stats.norm.ppf(.975) * fisher_se, z + stats.norm.ppf(.975) * fisher_se])
    return {'r': r, 't': t, 'df': int(df), 'p_raw': p, 'fisher95_low': float(ci[0]),
            'fisher95_high': float(ci[1]), 'fisher_se': fisher_se, 'SB': float(2 * r / (1 + r))}


def reconstruct_run_scores(trials: pd.DataFrame, cohort: str) -> pd.DataFrame:
    means = trials.groupby(['subject', 'group', 'run', 'Dimension'], observed=True).RT_ms.mean()
    counts = trials.groupby(['subject', 'group', 'run', 'Dimension'], observed=True).size()
    rows = []
    for (subject, group), unused in trials.groupby(['subject', 'group'], observed=True):
        for physical in ('Same', 'Different'):
            row = {'cohort': cohort, 'subject': subject, 'group': group, 'physical_dimension': physical}
            for run in (1, 2):
                absent = means.loc[(subject, group, run, 'Absent')]
                present = means.loc[(subject, group, run, physical)]
                row[f'run{run}_absent_ms'] = absent
                row[f'run{run}_present_ms'] = present
                row[f'run{run}_capture_ms'] = present - absent
                row[f'run{run}_n_present'] = int(counts.loc[(subject, group, run, physical)])
                row[f'run{run}_n_absent'] = int(counts.loc[(subject, group, run, 'Absent')])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['cohort', 'subject', 'physical_dimension']).reset_index(drop=True)


def behavior_run_reliability(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, draws = [], []
    for cohort in ('N33', 'N35'):
        h = scores.loc[scores.cohort.eq(cohort)].copy()
        expected_n = int(cohort[1:])
        for score_index, score in enumerate(('biased', 'unbiased')):
            biased = ((h.group.eq('SS') & h.physical_dimension.eq('Same')) |
                      (h.group.eq('DS') & h.physical_dimension.eq('Different')))
            z = h.loc[biased if score == 'biased' else ~biased].sort_values(['group', 'subject']).reset_index(drop=True)
            groups = z.group.to_numpy()
            x, y = z.run1_capture_ms.to_numpy(float), z.run2_capture_ms.to_numpy(float)
            require(len(z) == expected_n and (groups == 'SS').sum() == (16 if cohort == 'N33' else 17) and
                    (groups == 'DS').sum() == (17 if cohort == 'N33' else 18), f'{cohort} {score}: authorized pooled group counts')
            rng = np.random.default_rng(SEED + (0 if cohort == 'N33' else 100) + score_index)
            ss, ds = np.flatnonzero(groups == 'SS'), np.flatnonzero(groups == 'DS')
            idx = np.column_stack([rng.choice(ss, (R, len(ss))), rng.choice(ds, (R, len(ds)))])
            locations = (np.arange(len(ss)), np.arange(len(ss), len(z)))
            for centering in ('uncentered', 'group_centered'):
                pooled_centered = centering == 'group_centered'
                xx, yy = x[idx].copy(), y[idx].copy()
                if pooled_centered:
                    for loc in locations:
                        xx[:, loc] -= xx[:, loc].mean(axis=1, keepdims=True)
                        yy[:, loc] -= yy[:, loc].mean(axis=1, keepdims=True)
                    px, py, df = center(x, groups), center(y, groups), len(z) - 3
                    se = 1 / math.sqrt(len(z) - 4)
                else:
                    px, py, df = x, y, len(z) - 2
                    se = 1 / math.sqrt(len(z) - 3)
                xx -= xx.mean(axis=1, keepdims=True)
                yy -= yy.mean(axis=1, keepdims=True)
                rb = (xx * yy).sum(axis=1) / np.sqrt((xx * xx).sum(axis=1) * (yy * yy).sum(axis=1))
                sb = 2 * rb / (1 + rb)
                require(np.isfinite(rb).all() and np.isfinite(sb).all(), f'{cohort} {score} {centering}: finite paired 10k bootstrap')
                point = stats_row(px, py, df, se)
                rows.append({'cohort': cohort, 'measure': 'run1-run2 capture', 'score': score, 'centering': centering,
                             'N': len(z), **point, 'percentile95_low': float(np.quantile(rb, .025)),
                             'percentile95_high': float(np.quantile(rb, .975)), 'SB_percentile95_low': float(np.quantile(sb, .025)),
                             'SB_percentile95_high': float(np.quantile(sb, .975)), 'bootstrap_R': R,
                             'seed': SEED + (0 if cohort == 'N33' else 100) + score_index,
                             'bootstrap_design': 'paired participant resamples stratified SS/DS; same resamples for uncentered and centered row within score'})
                draws.append(pd.DataFrame({'cohort': cohort, 'score': score, 'centering': centering,
                                           'iteration': np.arange(1, R + 1), 'r': rb, 'SB': sb}))
    result = pd.DataFrame(rows)
    for cohort in ('N33', 'N35'):
        for centering in ('uncentered', 'group_centered'):
            idx = result.cohort.eq(cohort) & result.centering.eq(centering)
            result.loc[idx, 'p_Holm2'] = holm(result.loc[idx, 'p_raw'].to_numpy(float))
            result.loc[idx, 'Holm_family'] = f'{cohort}, {centering}: biased and unbiased pooled run correlations (Holm-2)'
    return result, pd.concat(draws, ignore_index=True)


def default91_run_reliability(runwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = runwise.sort_values(['group', 'subject']).reset_index(drop=True)
    require(len(z) == 33 and z.subject.nunique() == 33 and z.groupby('group').size().to_dict() == {'DS': 17, 'SS': 16},
            'authorized Default91 run-score cohort N33 SS16 DS17')
    groups, x, y = z.group.to_numpy(), z.run1_primary.to_numpy(float), z.run2_primary.to_numpy(float)
    rows, draws = [], []
    rng = np.random.default_rng(SEED + 1000)
    ss, ds = np.flatnonzero(groups == 'SS'), np.flatnonzero(groups == 'DS')
    idx = np.column_stack([rng.choice(ss, (R, len(ss))), rng.choice(ds, (R, len(ds)))])
    locations = (np.arange(len(ss)), np.arange(len(ss), len(z)))
    for centering in ('uncentered', 'group_centered'):
        pooled_centered = centering == 'group_centered'
        xx, yy = x[idx].copy(), y[idx].copy()
        if pooled_centered:
            for loc in locations:
                xx[:, loc] -= xx[:, loc].mean(axis=1, keepdims=True)
                yy[:, loc] -= yy[:, loc].mean(axis=1, keepdims=True)
            px, py, df, se = center(x, groups), center(y, groups), 30, 1 / math.sqrt(29)
        else:
            px, py, df, se = x, y, 31, 1 / math.sqrt(30)
        xx -= xx.mean(axis=1, keepdims=True)
        yy -= yy.mean(axis=1, keepdims=True)
        rb = (xx * yy).sum(axis=1) / np.sqrt((xx * xx).sum(axis=1) * (yy * yy).sum(axis=1))
        sb = 2 * rb / (1 + rb)
        require(np.isfinite(rb).all() and np.isfinite(sb).all(), f'Default91 {centering}: finite paired 10k bootstrap')
        rows.append({'cohort': 'N33', 'measure': 'current Default91/common399 connectivity Run1--Run2', 'centering': centering,
                     'N': 33, **stats_row(px, py, df, se), 'percentile95_low': float(np.quantile(rb, .025)),
                     'percentile95_high': float(np.quantile(rb, .975)), 'SB_percentile95_low': float(np.quantile(sb, .025)),
                     'SB_percentile95_high': float(np.quantile(sb, .975)), 'bootstrap_R': R, 'seed': SEED + 1000,
                     'bootstrap_design': 'paired participant resamples stratified SS/DS; same resamples for uncentered and centered rows',
                     'correction': 'single current Default91 run-reliability estimand; no multiplicity adjustment'})
        draws.append(pd.DataFrame({'centering': centering, 'iteration': np.arange(1, R + 1), 'r': rb, 'SB': sb}))
    return pd.DataFrame(rows), pd.concat(draws, ignore_index=True)


def main() -> None:
    sources = [*TRIALS.values(), SAVED_RUN_SCORES, SAVED_CENTERED, RUNWISE_DEFAULT91, READY_MANIFEST, RESULTS_MANIFEST]
    for path in sources:
        require(path.is_file(), f'source exists: {path}')
    results_manifest = json.loads(RESULTS_MANIFEST.read_text())
    ready_manifest = json.loads(READY_MANIFEST.read_text())
    require(results_manifest['input_hashes'][str(RUNWISE_DEFAULT91)] == sha256(RUNWISE_DEFAULT91),
            'results manifest matches authorized Default91 run-score hash')
    require(ready_manifest['runwise_primary']['run1_column'] == 'run1_primary' and ready_manifest['runwise_primary']['run2_column'] == 'run2_primary',
            'analysis-ready manifest binds saved Default91 Run1/Run2 columns')

    reconstructed = []
    for n, path in TRIALS.items():
        trials = pd.read_csv(path)
        require(len(trials) == (10995 if n == 33 else 11660) and trials.subject.nunique() == n, f'N{n}: fixed retained trial universe')
        require(trials.correct.eq(1).all() and trials.RT_ms.between(200, 3000).all() and set(trials.run) == {1, 2}, f'N{n}: correct/RT gate and both runs')
        require(not trials.duplicated(['subject', 'run', 'trial']).any(), f'N{n}: unique subject/run/trial keys')
        reconstructed.append(reconstruct_run_scores(trials, f'N{n}'))
    run_scores = pd.concat(reconstructed, ignore_index=True)
    saved_scores = pd.read_csv(SAVED_RUN_SCORES).sort_values(['cohort', 'subject', 'physical_dimension']).reset_index(drop=True)
    require(len(run_scores) == len(saved_scores) == 136 and close(run_scores[[c for c in run_scores if c.startswith('run')]], saved_scores[[c for c in saved_scores if c.startswith('run')]]),
            'all N33/N35 run-specific capture/count vectors reproduce saved authorized run-score file')

    behavior_summary, behavior_draws = behavior_run_reliability(run_scores)
    saved_centered = pd.read_csv(SAVED_CENTERED)
    saved_pooled = saved_centered.loc[saved_centered.scope.eq('pooled_group_centered'), ['cohort', 'score', 'r', 't', 'df', 'p_raw', 'p_Holm']]
    new_centered = behavior_summary.loc[behavior_summary.centering.eq('group_centered'), ['cohort', 'score', 'r', 't', 'df', 'p_raw', 'p_Holm2']]
    compare = new_centered.merge(saved_pooled, on=['cohort', 'score'], suffixes=('_new', '_saved'), validate='one_to_one')
    for field in ('r', 't', 'df', 'p_raw'):
        require(close(compare[f'{field}_new'], compare[f'{field}_saved'], tol=2e-10), f'current centered pooled run result reproduces saved {field}')
    require(close(compare.p_Holm2, compare.p_Holm, tol=2e-10), 'current centered pooled run Holm-2 reproduces saved values')

    runwise = pd.read_csv(RUNWISE_DEFAULT91)
    default_summary, default_draws = default91_run_reliability(runwise)
    centered_default = default_summary.loc[default_summary.centering.eq('group_centered')].iloc[0]
    require(close([centered_default['r'], centered_default['df']], [0.7332272384011732, 30], tol=2e-10),
            'current Default91/common399 group-centered run reliability is r=.733227, df=30 (not historical .746/.798)')

    run_scores.to_csv(OUT / 'current_run_capture_scores_reconstructed.csv', index=False)
    behavior_summary.to_csv(OUT / 'current_pooled_run_capture_reliability.csv', index=False)
    behavior_draws.to_csv(OUT / 'current_pooled_run_capture_bootstrap_10000.csv', index=False)
    default_summary.to_csv(OUT / 'current_default91_connectivity_run_reliability.csv', index=False)
    default_draws.to_csv(OUT / 'current_default91_connectivity_run_bootstrap_10000.csv', index=False)
    availability = pd.DataFrame([
        {'measure': 'current Default91/common399 connectivity Run1--Run2', 'cohort': 'N33', 'status': 'available',
         'reason': '33-participant authorized imaging cohort with saved runwise_primary_scores_wide.csv.'},
        {'measure': 'current Default91/common399 connectivity Run1--Run2', 'cohort': 'N35', 'status': 'not available',
         'reason': 'No N35 imaging cohort exists in the current Default91/common399 results contract; N35 is behavioral only.'},
    ])
    availability.to_csv(OUT / 'current_default91_connectivity_run_availability.csv', index=False)
    manifest = {'scope': 'read-only saved-score reconstruction; no neuroimaging/model refit; no canonical source edited',
                'sources_sha256': {str(path): sha256(path) for path in sources},
                'bootstrap': {'R': R, 'seed_base': SEED, 'design': 'stratified SS/DS paired participant percentile bootstrap'},
                'outputs': ['current_run_capture_scores_reconstructed.csv', 'current_pooled_run_capture_reliability.csv',
                            'current_pooled_run_capture_bootstrap_10000.csv', 'current_default91_connectivity_run_reliability.csv',
                            'current_default91_connectivity_run_bootstrap_10000.csv', 'current_default91_connectivity_run_availability.csv']}
    (OUT / 'run_reliability_audit_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / 'run_reliability_verification.json').write_text(json.dumps({'all_checks_passed': True, 'n_checks': len(CHECKS), 'checks': CHECKS}, indent=2) + '\n')
    print(json.dumps({'all_checks_passed': True, 'n_checks': len(CHECKS),
                      'behavior_rows': len(behavior_summary), 'default91_rows': len(default_summary),
                      'default91_centered_r': centered_default['r']}, indent=2))


if __name__ == '__main__':
    main()
