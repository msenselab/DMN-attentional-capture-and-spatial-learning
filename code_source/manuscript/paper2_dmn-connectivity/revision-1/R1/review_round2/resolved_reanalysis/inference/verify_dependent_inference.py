#!/usr/bin/env python3
"""Independent verifier for completed dependent-inference rerun outputs.

This program deliberately does not import or execute run_dependent_inference.py.
It recomputes the reported subject-level statistics and saved resampling streams
from the manifest-named fresh inputs plus frozen behavioral checkpoint inputs.
It does not refit the R mixed models; it verifies their input/design, output
binding, rank contract, formula contract, and convergence/optimizer records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[7])))
HERE = Path(__file__).resolve().parent
DEFAULT_BEHAVIOR = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv'
DEFAULT_FROZEN_DRAWS = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/count_timing_evidence/rerun_outputs/matched_biased_score_draws.npy'
DEFAULT_FROZEN_SUBJECTS = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/count_timing_evidence/rerun_outputs/participant_scores_with_downsampling.csv'
DEFAULT_FROZEN_TRIAL = ROOT / 'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/count_timing_evidence/rerun_outputs/trial_resampling_draws.csv'
R = 10_000
SEED = 20261001
NETWORK_COUNTS = {'Default': 91, 'SomMot': 77, 'Vis': 61, 'Cont': 52, 'SalVentAttn': 47, 'DorsAttn': 46, 'Limbic': 25}
VARIANTS = ('primary', 'minimal_tr1', 'primary_gsr', 'primary_task', 'primary_task_gsr')
TOL = 1e-8


class Checker:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.gaps: list[str] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append({'name': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})

    def close(self, name: str, actual, expected, atol: float = TOL) -> bool:
        a = np.asarray(actual, dtype=float)
        e = np.asarray(expected, dtype=float)
        ok = a.shape == e.shape and np.allclose(a, e, atol=atol, rtol=0, equal_nan=True)
        max_error = None if a.shape != e.shape else (float(np.nanmax(np.abs(a - e))) if a.size else 0.0)
        self.check(name, ok, f'expected shape={e.shape}; actual shape={a.shape}; max_abs_error={max_error}; atol={atol}')
        return ok

    def require_file(self, path: Path, label: str) -> bool:
        ok = path.is_file()
        self.check(label, ok, str(path))
        return ok


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def resolved_path(value: str, manifest_path: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else manifest_path.parent / p


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
    if np.any(~np.isfinite(den)) or np.any(den == 0):
        raise ValueError('zero/nonfinite correlation variance')
    return np.sum(a * b, axis=-1) / den


def group_indices(groups: np.ndarray) -> list[np.ndarray]:
    if len(groups) != 33 or set(groups) != {'SS', 'DS'}:
        raise ValueError('expected N=33 with SS/DS groups')
    out = [np.flatnonzero(groups == group) for group in ('SS', 'DS')]
    if [len(x) for x in out] != [16, 17]:
        raise ValueError('expected SS=16, DS=17')
    return out


def center_group(v: np.ndarray, indexes: list[np.ndarray]) -> np.ndarray:
    out = np.array(v, dtype=float, copy=True)
    if out.ndim == 1:
        for ix in indexes:
            out[ix] -= out[ix].mean()
    elif out.ndim == 2:
        for ix in indexes:
            out[:, ix] -= out[:, ix].mean(axis=1, keepdims=True)
    else:
        raise ValueError('only vector or draw-by-participant input is supported')
    return out


def corr_stats(x: np.ndarray, y: np.ndarray, k: int = 0) -> dict:
    r = float(corr(x, y))
    n = len(x)
    df = n - k - 2
    if not (df > 0 and abs(r) < 1):
        raise ValueError('invalid correlation')
    t = r * np.sqrt(df / (1 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    se = 1 / np.sqrt(n - k - 3)
    ci = np.tanh(np.arctanh(r) + np.array([-1, 1]) * stats.norm.ppf(.975) * se)
    return {'N': n, 'r': r, 't': t, 'df': df, 'p_raw': p, 'ci_low': float(ci[0]), 'ci_high': float(ci[1]), 'fisher_se': se}


def group_adjusted(x: np.ndarray, y: np.ndarray) -> dict:
    # The runner first centers both vectors within SS/DS, then uses one group
    # contrast in its stated degrees of freedom and Fisher-z interval.
    return corr_stats(x, y, k=1)


def partial(frame: pd.DataFrame, x: str, y: str, covariates: list[str]) -> dict:
    c = np.column_stack([np.ones(len(frame))] + [frame[z].to_numpy(float) for z in covariates])
    if np.linalg.matrix_rank(c) != c.shape[1]:
        raise ValueError('rank-deficient partial-correlation covariates')
    xv = frame[x].to_numpy(float)
    yv = frame[y].to_numpy(float)
    xr = xv - c @ np.linalg.lstsq(c, xv, rcond=None)[0]
    yr = yv - c @ np.linalg.lstsq(c, yv, rcond=None)[0]
    return corr_stats(xr, yr, k=len(covariates))


def williams(a: float, b: float, c: float, n_eff: int = 32) -> dict:
    determinant = 1 - a*a - b*b - c*c + 2*a*b*c
    if determinant <= 0:
        raise ValueError('Williams correlation matrix not positive definite')
    t = (a - b) * np.sqrt((n_eff - 1) * (1 + c)) / np.sqrt(
        2 * determinant * (n_eff - 1) / (n_eff - 3) + (a + b)**2 / 4 * (1 - c)**3
    )
    df = n_eff - 3
    return {'r_primary_biased': a, 'r_primary_unbiased': b, 'r_biased_unbiased': c,
            'delta_r': a - b, 't': t, 'df': df,
            'p_raw': float(2 * stats.t.sf(abs(t), df)), 'n_eff': n_eff}


def holm(p_values: np.ndarray) -> np.ndarray:
    """Independent Holm adjustment, retaining original input order."""
    p = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid p value for Holm')
    order = np.argsort(p, kind='mergesort')
    ordered = p[order]
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * ordered)
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def output_csv(results: Path, name: str, checker: Checker) -> pd.DataFrame | None:
    path = results / name
    if not checker.require_file(path, f'output file exists: {name}'):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        checker.check(f'parse CSV: {name}', False, repr(exc))
        return None


def expect_columns(frame: pd.DataFrame | None, required: set[str], name: str, checker: Checker) -> bool:
    ok = frame is not None and required <= set(frame.columns)
    missing = [] if frame is None else sorted(required - set(frame.columns))
    checker.check(f'schema: {name}', ok, f'missing={missing}')
    return ok


def merge_behavior(scores: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    means = trials.groupby(['subject', 'group', 'category'], observed=True).RT_ms.mean().unstack('category')
    if not {'absent', 'biased', 'unbiased'} <= set(means.columns):
        raise ValueError('behavior trial categories incomplete')
    b = means.reset_index()
    b['biased_capture'] = b['biased'] - b['absent']
    b['unbiased_capture'] = b['unbiased'] - b['absent']
    counts = trials.groupby(['subject', 'group', 'category'], observed=True).size().unstack('category').reset_index()
    b = b.merge(counts, on=['subject', 'group'], suffixes=('', '_n'), validate='one_to_one')
    b['biased_weight'] = b['biased_n'] / (b['biased_n'] + b['unbiased_n'])
    b['unbiased_weight'] = 1 - b['biased_weight']
    b['all_capture'] = b['biased_weight'] * b['biased_capture'] + b['unbiased_weight'] * b['unbiased_capture']
    q = trials.loc[trials.category.eq('biased') & trials.prev_category.notna()].copy()
    prim = q.groupby(['subject', 'group'], observed=True).apply(
        lambda z: z.loc[z.prev_category.eq('biased'), 'RT_ms'].mean() -
                  z.loc[~z.prev_category.eq('biased'), 'RT_ms'].mean(), include_groups=False
    ).rename('biased_priming').reset_index()
    if len(prim) != 33 or not np.isfinite(prim.biased_priming).all():
        raise ValueError('biased priming unavailable')
    b = b.merge(prim, on=['subject', 'group'], validate='one_to_one')
    merged = scores.merge(b[['subject', 'group', 'biased_capture', 'unbiased_capture', 'all_capture', 'biased_priming']],
                          on=['subject', 'group'], validate='one_to_one')
    merged['D'] = merged.group.eq('DS').astype(int)
    return merged


def load_inputs(manifest_path: Path, checker: Checker) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, dict] | None:
    if not checker.require_file(manifest_path, 'input manifest exists'):
        return None
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        checker.check('input manifest parses', False, repr(exc))
        return None
    checker.check('input manifest status complete', manifest.get('status') == 'complete', repr(manifest.get('status')))
    geom = manifest.get('common_coverage_geometry', {})
    geometry_ok = (geom.get('nominal_atlas_parcels') == 400 and geom.get('effective_common_coverage_parcels') == 399 and
                   set(geom.get('excluded_native_labels', [])) == {323} and geom.get('uniform_all_participants_and_runs') is True)
    checker.check('common-coverage geometry contract', geometry_ok, json.dumps(geom, sort_keys=True))
    try:
        s_cfg = manifest['participant_scores']
        names = {key: resolved_path(manifest[key]['path'], manifest_path) for key in ('primary_matrices', 'matrix_subjects', 'parcel_metadata', 'runwise_primary')}
        names['participant_scores'] = resolved_path(s_cfg['path'], manifest_path)
        for key, path in names.items():
            if not checker.require_file(path, f'input exists: {key}'):
                return None
        primary = s_cfg['primary_column']
        vcols = s_cfg['variant_columns']
        checker.check('manifest has exact five variants', set(vcols) == set(VARIANTS), str(vcols))
        raw = pd.read_csv(names['participant_scores'])
        required = {'subject', 'group', primary, s_cfg.get('mean_fd_column', 'mean_fd'), *vcols.values()}
        checker.check('participant-score columns resolve', required <= set(raw.columns), str(sorted(required - set(raw.columns))))
        if not required <= set(raw.columns):
            return None
        scores = raw[['subject', 'group']].copy()
        scores['primary_default'] = raw[primary].to_numpy(float)
        for variant in VARIANTS:
            scores[variant] = raw[vcols[variant]].to_numpy(float)
        scores['mean_fd'] = raw[s_cfg.get('mean_fd_column', 'mean_fd')].to_numpy(float)
        scores = scores.sort_values('subject').reset_index(drop=True)
        score_ok = len(scores) == 33 and scores.subject.is_unique and scores.groupby('group').size().to_dict() == {'DS': 17, 'SS': 16}
        checker.check('score identity N=33 SS16 DS17', score_ok, str(scores.groupby('group').size().to_dict()))
        checker.check('score values finite', np.isfinite(scores[['primary_default', *VARIANTS, 'mean_fd']].to_numpy(float)).all(), 'all numerical score values')
        matrix_subjects = pd.read_csv(names['matrix_subjects'])
        checker.check('matrix identity schema N=33', {'subject', 'group'} <= set(matrix_subjects.columns) and len(matrix_subjects) == 33 and matrix_subjects.subject.is_unique, str(matrix_subjects.shape))
        matrices = np.load(names['primary_matrices'], allow_pickle=False)
        checker.check('matrix shape common399', matrices.shape == (33, 399, 399), str(matrices.shape))
        if matrices.shape != (33, 399, 399):
            return None
        pos = {s: i for i, s in enumerate(matrix_subjects.subject)}
        checker.check('matrix/score identity agreement', set(pos) == set(scores.subject) and
                      all(matrix_subjects.set_index('subject').loc[s, 'group'] == g for s, g in zip(scores.subject, scores.group)), 'subject and group join')
        matrices = np.asarray(matrices[[pos[s] for s in scores.subject]], dtype=float)
        checker.check('matrices finite/symmetric/zero diagonal', np.isfinite(matrices).all() and
                      np.allclose(matrices, matrices.swapaxes(1, 2), atol=1e-8, rtol=0) and
                      np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 0, atol=1e-10, rtol=0), 'Fisher-z matrix contract')
        parcels = pd.read_csv(names['parcel_metadata']).sort_values('matrix_index').reset_index(drop=True)
        parcel_ok = ({'matrix_index', 'native_label', 'yeo7_network'} <= set(parcels.columns) and len(parcels) == 399 and
                     set(parcels.matrix_index) == set(range(399)) and set(parcels.native_label) == (set(range(1, 401)) - {323}) and
                     parcels.yeo7_network.value_counts().to_dict() == NETWORK_COUNTS)
        checker.check('physical Yeo7 parcel identity/counts', parcel_ok, str(parcels.yeo7_network.value_counts().to_dict()))
        r_cfg = manifest['runwise_primary']
        runwise = pd.read_csv(names['runwise_primary'])
        run1, run2 = r_cfg['run1_column'], r_cfg['run2_column']
        run_ok = {'subject', 'group', run1, run2} <= set(runwise.columns)
        if run_ok:
            runwise = runwise[['subject', 'group', run1, run2]].rename(columns={run1: 'run1_primary', run2: 'run2_primary'}).sort_values('subject').reset_index(drop=True)
            run_ok = (runwise.subject.tolist() == scores.subject.tolist() and runwise.group.tolist() == scores.group.tolist() and
                      np.isfinite(runwise[['run1_primary', 'run2_primary']].to_numpy(float)).all())
        checker.check('runwise identity and finite inputs', run_ok, f'columns={run1!r},{run2!r}')
        paths = {str(path): sha256(path) for path in names.values()}
        return scores, matrices, parcels, runwise, {'manifest': manifest, 'hashes': paths}
    except Exception as exc:
        checker.check('load fresh inputs', False, repr(exc))
        return None


def physical_network_scores(matrices: np.ndarray, parcels: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for network, count in NETWORK_COUNTS.items():
        ix = np.flatnonzero(parcels.yeo7_network.to_numpy() == network)
        if len(ix) != count:
            raise ValueError(f'wrong parcel count for {network}')
        sub = matrices[:, ix][:, :, ix]
        ii, jj = np.triu_indices(count, 1)
        out[network] = sub[:, ii, jj].mean(axis=1)
    return pd.DataFrame(out)


def verify_manifest_hashes(results_manifest: dict, input_hashes: dict, behavior: Path, frozen_draws: Path, frozen_trial: Path, checker: Checker) -> None:
    checker.check('results manifest records N=33/trials=10995', results_manifest.get('N') == 33 and results_manifest.get('trials') == 10995,
                  f"N={results_manifest.get('N')}, trials={results_manifest.get('trials')}")
    recorded = results_manifest.get('input_hashes', {})
    checker.check('fresh-input manifest hash binding', recorded == input_hashes, f'recorded={len(recorded)} current={len(input_hashes)}')
    for label, path, key in [('behavior trial hash', behavior, 'behavior_trials_sha256'),
                             ('frozen countmatched draw hash', frozen_draws, 'frozen_countmatched_draws_sha256'),
                             ('frozen trial-resampling hash', frozen_trial, 'frozen_trial_resampling_sha256')]:
        checker.check(label, path.is_file() and results_manifest.get(key) == sha256(path), str(path))


def verify_associations(results: Path, s: pd.DataFrame, checker: Checker) -> None:
    out = output_csv(results, 'participant_associations.csv', checker)
    required = {'scope', 'outcome', 'analysis', 'N', 'r', 't', 'df', 'p_raw', 'ci_low', 'ci_high', 'fisher_se', 'p_Holm', 'Holm_family'}
    if not expect_columns(out, required, 'participant_associations.csv', checker):
        return
    checker.check('participant association row count', len(out) == 9, f'rows={len(out)}')
    ix = group_indices(s.group.to_numpy())
    expected: dict[tuple[str, str, str], dict] = {}
    x = s.primary_default.to_numpy(float)
    for outcome in ('biased_capture', 'unbiased_capture', 'all_capture'):
        expected[('pooled', outcome, 'group-centered Pearson df=N-3')] = group_adjusted(center_group(x, ix), center_group(s[outcome].to_numpy(float), ix))
    expected[('pooled', 'biased_capture', 'partial unbiased+group')] = partial(s, 'primary_default', 'biased_capture', ['unbiased_capture', 'D'])
    expected[('pooled', 'biased_capture', 'partial measured priming+group')] = partial(s, 'primary_default', 'biased_capture', ['biased_priming', 'D'])
    for group in ('SS', 'DS'):
        z = s.loc[s.group.eq(group)].copy()
        expected[(group, 'biased_capture', 'within-group Pearson')] = corr_stats(z.primary_default, z.biased_capture)
        expected[(group, 'biased_capture', 'within-group partial unbiased')] = partial(z, 'primary_default', 'biased_capture', ['unbiased_capture'])
    keyed = {(r.scope, r.outcome, r.analysis): r for r in out.itertuples(index=False)}
    checker.check('participant association row identities', set(keyed) == set(expected), f'actual={sorted(keyed)}')
    cols = ['N', 'r', 't', 'df', 'p_raw', 'ci_low', 'ci_high', 'fisher_se']
    for key, want in expected.items():
        if key in keyed:
            got = keyed[key]
            checker.close(f'association recomputation: {key}', [getattr(got, c) for c in cols], [want[c] for c in cols], atol=2e-8)
    primary = out.loc[(out.scope == 'pooled') & (out.outcome == 'biased_capture') & (out.analysis == 'group-centered Pearson df=N-3')]
    checker.check('primary is unadjusted df30 Fisher SE=1/sqrt29', len(primary) == 1 and int(primary.df.iloc[0]) == 30 and
                  pd.isna(primary.p_Holm.iloc[0]) and np.isclose(primary.fisher_se.iloc[0], 1 / np.sqrt(29), atol=1e-12), 'one selected primary row')
    exploratory = out.drop(primary.index)
    checker.check('Holm-8 association family exactly declared', len(exploratory) == 8 and exploratory.Holm_family.nunique() == 1 and
                  exploratory.Holm_family.iloc[0].startswith('8 prespecified'), str(exploratory.Holm_family.unique()))
    if len(exploratory) == 8:
        checker.close('Holm-8 association arithmetic', exploratory.p_Holm.to_numpy(), holm(exploratory.p_raw.to_numpy()), atol=2e-12)


def verify_williams(results: Path, s: pd.DataFrame, checker: Checker) -> None:
    ix = group_indices(s.group.to_numpy())
    x = center_group(s.primary_default.to_numpy(float), ix)
    b = center_group(s.biased_capture.to_numpy(float), ix)
    u = center_group(s.unbiased_capture.to_numpy(float), ix)
    expected_full = williams(float(corr(x, b)), float(corr(x, u)), float(corr(b, u)))
    one = output_csv(results, 'williams_primary_vs_unbiased_fullscore_audit.csv', checker)
    if expect_columns(one, set(expected_full) | {'comparison', 'method'}, 'williams_primary_vs_unbiased_fullscore_audit.csv', checker):
        checker.check('full-score Williams one row/df29', len(one) == 1 and int(one.df.iloc[0]) == 29 and int(one.n_eff.iloc[0]) == 32, f'rows={len(one)}')
        if len(one) == 1:
            checker.close('full-score Williams recomputation', [one.iloc[0][c] for c in expected_full], list(expected_full.values()), atol=2e-8)
    both = output_csv(results, 'williams_full_and_countmatched_Holm2.csv', checker)
    if expect_columns(both, set(expected_full) | {'procedure', 'p_Holm2', 'Holm_family'}, 'williams_full_and_countmatched_Holm2.csv', checker):
        checker.check('Williams Holm-2 has two declared procedures', len(both) == 2 and set(both.procedure) == {
            'full retained scores', 'mean of authorized count-matched biased score draws'}, str(both.procedure.tolist()))
        if len(both) == 2:
            full = both.loc[both.procedure.eq('full retained scores')].iloc[0]
            checker.close('Williams Holm-2 full row binding', [full[c] for c in expected_full], list(expected_full.values()), atol=2e-8)
            checker.close('Williams Holm-2 arithmetic', both.p_Holm2.to_numpy(), holm(both.p_raw.to_numpy()), atol=2e-12)
            checker.check('Williams Holm-2 df=29', (both.df == 29).all() and (both.n_eff == 32).all(), str(both[['df', 'n_eff']].to_dict('records')))


def verify_networks_and_sensitivity(results: Path, s: pd.DataFrame, networks: pd.DataFrame, checker: Checker) -> None:
    ix = group_indices(s.group.to_numpy())
    y = center_group(s.biased_capture.to_numpy(float), ix)
    nout = output_csv(results, 'physical_yeo7_allnetwork_comparators_holm7_AUDIT_ONLY_NOT_TABLE7.csv', checker)
    if expect_columns(nout, {'network', 'r', 't', 'df', 'p_raw', 'ci_low', 'ci_high', 'fisher_se', 'p_Holm', 'Holm_family'}, 'network comparator output', checker):
        checker.check('network comparator has seven physical rows', len(nout) == 7 and set(nout.network) == set(NETWORK_COUNTS), str(nout.network.tolist()))
        for row in nout.itertuples(index=False):
            want = group_adjusted(center_group(networks[row.network].to_numpy(float), ix), y)
            checker.close(f'network statistic: {row.network}', [getattr(row, c) for c in ('r','t','df','p_raw','ci_low','ci_high','fisher_se')], list(want.values())[1:], atol=2e-8)
        if len(nout) == 7:
            checker.close('network Holm-7 arithmetic', nout.p_Holm.to_numpy(), holm(nout.p_raw.to_numpy()), atol=2e-12)
            checker.check('network output remains audit-only', nout.Holm_family.astype(str).str.contains('Holm-7').all(), str(nout.Holm_family.unique()))
    sout = output_csv(results, 'preprocessing_motion_gsr_task_sensitivities.csv', checker)
    if expect_columns(sout, {'variant', 'kind', 'r', 't', 'df', 'p_raw', 'ci_low', 'ci_high', 'fisher_se', 'p_Holm', 'Holm_family'}, 'sensitivity output', checker):
        primary_rows = sout.loc[sout.kind.eq('preprocessing sensitivity')]
        checker.check('preprocessing family exactly five variants', len(primary_rows) == 5 and set(primary_rows.variant) == set(VARIANTS), str(primary_rows.variant.tolist()))
        for row in primary_rows.itertuples(index=False):
            want = group_adjusted(center_group(s[getattr(row, 'variant')].to_numpy(float), ix), y)
            checker.close(f'preprocessing statistic: {row.variant}', [getattr(row, c) for c in ('r','t','df','p_raw','ci_low','ci_high','fisher_se')], list(want.values())[1:], atol=2e-8)
        if len(primary_rows) == 5:
            checker.close('preprocessing Holm-5 arithmetic', primary_rows.p_Holm.to_numpy(), holm(primary_rows.p_raw.to_numpy()), atol=2e-12)
        motion = sout.loc[sout.kind.astype(str).str.contains('mean-FD adjusted')]
        checker.check('mean-FD row explicitly descriptive/pointwise', len(motion) == 1 and motion.kind.iloc[0].endswith('descriptive pointwise'), str(motion.kind.tolist()))
        if len(motion) == 1:
            want = partial(s, 'primary_default', 'biased_capture', ['D', 'mean_fd'])
            checker.close('mean-FD partial statistic', [motion.iloc[0][c] for c in ('r','t','df','p_raw','ci_low','ci_high','fisher_se')], list(want.values())[1:], atol=2e-8)


def reconstruct_conditional(trials: pd.DataFrame, s: pd.DataFrame, frozen_draws: Path) -> tuple[np.ndarray, np.ndarray]:
    ix = group_indices(s.group.to_numpy())
    rng = np.random.default_rng(SEED)
    _ = np.column_stack([rng.choice(q, size=(R, len(q)), replace=True) for q in ix])
    for _ in range(R):
        for q in ix:
            rng.permutation(q)
    for _ in range(3):
        _ = np.column_stack([rng.choice(q, size=(R, len(q)), replace=True) for q in ix])
    bdraw = np.empty((R, 33))
    udraw = np.empty((R, 33))
    for col, subject in enumerate(s.subject):
        z = trials.loc[trials.subject.eq(subject)]
        b = z.loc[z.category.eq('biased'), 'RT_ms'].to_numpy(float)
        u = z.loc[z.category.eq('unbiased'), 'RT_ms'].to_numpy(float)
        a = float(z.loc[z.category.eq('absent'), 'RT_ms'].mean())
        if not (len(b) >= len(u) > 0 and np.isfinite(a)):
            raise ValueError('invalid conditional resampling condition counts')
        for draw in range(R):
            bdraw[draw, col] = rng.choice(b, len(u), replace=False).mean() - a
        udraw[:, col] = u[rng.integers(len(u), size=(R, len(u)))].mean(axis=1) - a
    checkpoint = np.load(frozen_draws, allow_pickle=False)
    if checkpoint.shape != (R, 33) or not np.allclose(bdraw, checkpoint, atol=1e-10, rtol=0):
        raise ValueError('reconstructed biased count-matched scores do not match frozen checkpoint')
    return bdraw, udraw


def verify_resampling(results: Path, s: pd.DataFrame, trials: pd.DataFrame, frozen_draws: Path, frozen_subjects: Path, frozen_trial: Path, checker: Checker) -> None:
    ix = group_indices(s.group.to_numpy())
    x = s.primary_default.to_numpy(float)
    b = s.biased_capture.to_numpy(float)
    u = s.unbiased_capture.to_numpy(float)
    a = s.all_capture.to_numpy(float)
    xc = center_group(x, ix)
    rng = np.random.default_rng(SEED)
    sampled = np.column_stack([rng.choice(q, size=(R, len(q)), replace=True) for q in ix])
    draw_groups = [np.arange(16), np.arange(16, 33)]
    boot_b = corr(center_group(x[sampled], draw_groups), center_group(b[sampled], draw_groups))
    boot_u = corr(center_group(x[sampled], draw_groups), center_group(u[sampled], draw_groups))
    boot_a = corr(center_group(x[sampled], draw_groups), center_group(a[sampled], draw_groups))
    boot_delta = boot_b - boot_u
    perm = np.empty(R)
    for k in range(R):
        y = b.copy()
        for q in ix:
            y[q] = rng.permutation(y[q])
        perm[k] = corr(xc, center_group(y, ix))
    draws = output_csv(results, 'participant_bootstrap_permutation_draws.csv', checker)
    required = {'iteration', 'bootstrap_primary_r', 'bootstrap_unbiased_r', 'bootstrap_all_r', 'bootstrap_delta_r', 'permuted_primary_r'}
    if expect_columns(draws, required, 'participant bootstrap/permutation draws', checker):
        checker.check('participant bootstrap/permutation draw count', len(draws) == R and np.array_equal(draws.iteration.to_numpy(), np.arange(1, R + 1)), f'rows={len(draws)}')
        checker.close('participant bootstrap/permutation streams', draws[['bootstrap_primary_r','bootstrap_unbiased_r','bootstrap_all_r','bootstrap_delta_r','permuted_primary_r']].to_numpy(),
                      np.column_stack([boot_b, boot_u, boot_a, boot_delta, perm]), atol=2e-12)
    summary = output_csv(results, 'participant_bootstrap_permutation_summary.csv', checker)
    if expect_columns(summary, {'estimand','point','ci_low','ci_high','R','seed','CI_method','p_two_sided_within_group_permutation'}, 'participant bootstrap summary', checker):
        point_b = float(corr(xc, center_group(b, ix)));
        point_u = float(corr(xc, center_group(u, ix))); point_a = float(corr(xc, center_group(a, ix)))
        expected = [('biased r', point_b, boot_b), ('unbiased r', point_u, boot_u), ('all r', point_a, boot_a), ('delta biased-unbiased r', point_b - point_u, boot_delta)]
        checker.check('participant bootstrap summary four rows', len(summary) == 4 and set(summary.estimand) == {x[0] for x in expected}, str(summary.estimand.tolist()))
        for name, point, distribution in expected:
            got = summary.loc[summary.estimand.eq(name)]
            if len(got) == 1:
                want = [point, *np.quantile(distribution, [.025, .975]), R, SEED]
                checker.close(f'participant bootstrap summary arithmetic: {name}', [got.iloc[0][z] for z in ('point','ci_low','ci_high','R','seed')], want, atol=2e-10)
                if name == 'biased r':
                    p = (1 + np.sum(np.abs(perm) >= abs(point_b))) / (R + 1)
                    checker.close('stratified permutation p arithmetic', [got.iloc[0].p_two_sided_within_group_permutation], [p], atol=2e-14)
    checker.require_file(frozen_subjects, 'frozen countmatched subject-order file exists')
    if frozen_subjects.is_file():
        order = pd.read_csv(frozen_subjects)
        checker.check('frozen countmatched subject order matches analysis', {'subject','group'} <= set(order.columns) and
                      order.subject.tolist() == s.subject.tolist() and order.group.tolist() == s.group.tolist(), str(order.shape))
    try:
        bdraw, udraw = reconstruct_conditional(trials, s, frozen_draws)
        checker.check('frozen biased checkpoint independently reconstructs', True, 'shape=(10000,33), atol=1e-10')
    except Exception as exc:
        checker.check('frozen biased checkpoint independently reconstructs', False, repr(exc))
        return
    bdc, udc = center_group(bdraw, ix), center_group(udraw, ix)
    br = corr(np.broadcast_to(xc, (R, 33)), bdc)
    ur = corr(np.broadcast_to(xc, (R, 33)), udc)
    fixed_ur = float(corr(xc, center_group(u, ix)))
    delta_fixed, delta_boot = br - fixed_ur, br - ur
    rb_u = corr(bdc, np.broadcast_to(center_group(u, ix), (R, 33)))
    wt = np.array([williams(float(q), fixed_ur, float(c))['t'] for q, c in zip(br, rb_u)])
    cdraw = output_csv(results, 'countmatched_conditional_draws.csv', checker)
    needed = {'iteration','countmatched_biased_r','unbiased_bootstrap_r','direct_matched_biased_minus_fixed_unbiased_r',
              'direct_matched_biased_minus_bootstrap_unbiased_r','drawwise_williams_t_fixed_unbiased'}
    if expect_columns(cdraw, needed, 'conditional draw output', checker):
        checker.check('conditional draw count', len(cdraw) == R and np.array_equal(cdraw.iteration.to_numpy(), np.arange(1, R+1)), f'rows={len(cdraw)}')
        checker.close('conditional paired-RNG arithmetic', cdraw[[x for x in needed if x != 'iteration']].sort_index(axis=1).to_numpy(),
                      pd.DataFrame({'countmatched_biased_r':br,'unbiased_bootstrap_r':ur,'direct_matched_biased_minus_fixed_unbiased_r':delta_fixed,
                                    'direct_matched_biased_minus_bootstrap_unbiased_r':delta_boot,'drawwise_williams_t_fixed_unbiased':wt}).sort_index(axis=1).to_numpy(), atol=2e-12)
    csum = output_csv(results, 'countmatched_conditional_summary.csv', checker)
    if expect_columns(csum, {'estimand','median','mean','ci_low','ci_high','R','seed','scope'}, 'conditional summary', checker):
        expected = [('DMN--matched-biased association', br), ('DMN--unbiased bootstrap association', ur),
                    ('Direct matched-biased minus fixed-fullscore-unbiased r', delta_fixed),
                    ('Direct matched-biased minus bootstrap-unbiased r', delta_boot), ('Drawwise Williams t (fixed unbiased)', wt)]
        checker.check('conditional summary five non-inferential rows', len(csum) == 5 and set(csum.estimand) == {x[0] for x in expected}, str(csum.estimand.tolist()))
        for name, values in expected:
            got = csum.loc[csum.estimand.eq(name)]
            if len(got) == 1:
                checker.close(f'conditional percentile arithmetic: {name}', [got.iloc[0][c] for c in ('median','mean','ci_low','ci_high','R','seed')],
                              [np.median(values), np.mean(values), *np.quantile(values, [.025,.975]), R, SEED], atol=2e-10)
        non_sampling = csum.loc[csum.estimand.ne('Drawwise Williams t (fixed unbiased)'), 'scope'].astype(str)
        descriptive_williams = csum.loc[csum.estimand.eq('Drawwise Williams t (fixed unbiased)'), 'scope'].astype(str)
        checker.check('conditional summaries avoid false participant-sampling/p-value claims',
                      non_sampling.str.contains('not a participant-sampling CI').all() and len(descriptive_williams) == 1 and
                      descriptive_williams.iloc[0] == 'Descriptive conditional statistic; no drawwise p values or significance fraction.',
                      'conditional scopes')
    audit = output_csv(results, 'conditional_nominal_draw_fraction_AUDIT_ONLY.csv', checker)
    if expect_columns(audit, {'status','source','source_sha256','R','biased_nominal_draw_p_lt_05_fraction','unbiased_nominal_draw_p_lt_05_fraction'}, 'conditional nominal audit', checker):
        checker.check('historical nominal-draw file is audit-only', len(audit) == 1 and audit.status.iloc[0] == 'AUDIT_ONLY_NOT_FRESH_INFERENCE' and
                      audit.source_sha256.iloc[0] == sha256(frozen_trial), str(audit.status.tolist()))


def verify_lopo_stability(results: Path, s: pd.DataFrame, runwise: pd.DataFrame, checker: Checker) -> None:
    lopo = output_csv(results, 'leave_one_participant_out.csv', checker)
    if expect_columns(lopo, {'omitted','N','r','t','df','p_raw','ci_low','ci_high','fisher_se'}, 'LOPO output', checker):
        checker.check('LOPO one row per participant', len(lopo) == 33 and set(lopo.omitted) == set(s.subject), f'rows={len(lopo)}')
        for i, subject in enumerate(s.subject):
            got = lopo.loc[lopo.omitted.eq(subject)]
            if len(got) == 1:
                z = s.drop(index=i).reset_index(drop=True)
                ix = [np.flatnonzero(z.group.to_numpy() == g) for g in ('SS','DS')]
                want = group_adjusted(center_group(z.primary_default.to_numpy(float), ix), center_group(z.biased_capture.to_numpy(float), ix))
                checker.close(f'LOPO statistic: {subject}', [got.iloc[0][c] for c in ('N','r','t','df','p_raw','ci_low','ci_high','fisher_se')], list(want.values()), atol=2e-8)
    out = output_csv(results, 'new_primary_run_stability.csv', checker)
    if expect_columns(out, {'scope','N','r','t','df','p_raw','ci_low','ci_high','fisher_se'}, 'run-stability output', checker):
        ix = group_indices(runwise.group.to_numpy())
        expected = {
            'pooled group-adjusted': group_adjusted(center_group(runwise.run1_primary.to_numpy(float), ix), center_group(runwise.run2_primary.to_numpy(float), ix)),
            'SS': corr_stats(runwise.loc[runwise.group.eq('SS'), 'run1_primary'], runwise.loc[runwise.group.eq('SS'), 'run2_primary']),
            'DS': corr_stats(runwise.loc[runwise.group.eq('DS'), 'run1_primary'], runwise.loc[runwise.group.eq('DS'), 'run2_primary']),
        }
        checker.check('run-stability three scopes', len(out) == 3 and set(out.scope) == set(expected), str(out.scope.tolist()))
        for scope, want in expected.items():
            got = out.loc[out.scope.eq(scope)]
            if len(got) == 1:
                checker.close(f'run stability statistic: {scope}', [got.iloc[0][c] for c in ('N','r','t','df','p_raw','ci_low','ci_high','fisher_se')], list(want.values()), atol=2e-8)


def verify_lmm_binding(results: Path, s: pd.DataFrame, trials: pd.DataFrame, checker: Checker) -> None:
    handoff = output_csv(results, 'joint_history_trials_new_primary.csv', checker)
    needed = {'subject','group','run','RT100','RT_ms','B','U','D','category','correct','log_order','prior_biased','prior_unbiased','primary_default','X_raw_new_primary','X'}
    if not expect_columns(handoff, needed, 'joint history handoff', checker):
        return
    checker.check('history handoff preserves 10,995 trial identities', len(handoff) == 10995 and handoff[['subject','run','trial']].equals(trials[['subject','run','trial']]) if 'trial' in handoff else len(handoff) == 10995,
                  'trial identifier is not emitted by runner; row order plus frozen columns are checked')
    source = trials.merge(s[['subject','group','primary_default']], on=['subject','group'], validate='many_to_one')
    columns = ['subject','group','run','RT100','RT_ms','B','U','D','category','correct','log_order','prior_biased','prior_unbiased','primary_default']
    checker.check('history handoff source columns exactly bind frozen trials/fresh vector', len(handoff) == len(source) and
                  all(np.array_equal(handoff[c].to_numpy(), source[c].to_numpy()) for c in columns), 'all retained source columns in retained row order')
    subject_x = handoff[['subject','group','primary_default','X_raw_new_primary','X']].drop_duplicates()
    checker.check('history handoff exactly one fresh X per subject', len(subject_x) == 33 and subject_x.subject.is_unique, str(subject_x.shape))
    group_means = s.groupby('group').primary_default.transform('mean')
    raw = s.primary_default - group_means
    sd = raw.std(ddof=1)
    expected_x = pd.DataFrame({'subject':s.subject, 'X_raw_new_primary':raw, 'X':raw/sd})
    got_x = subject_x[['subject','X_raw_new_primary','X']].sort_values('subject').reset_index(drop=True)
    exp_x = expected_x.sort_values('subject').reset_index(drop=True)
    checker.close('history X is group-centered then global N=33 standardized', got_x[['X_raw_new_primary','X']].to_numpy(), exp_x[['X_raw_new_primary','X']].to_numpy(), atol=2e-12)
    checker.check('history X global participant SD exactly one', np.isclose(got_x.X.std(ddof=1), 1, atol=1e-12), str(got_x.X.std(ddof=1)))
    # Independently construct the 144 fixed-effect numeric design. This verifies
    # the model input/design contract without calling R or importing its runner.
    d = handoff.copy()
    d['run2'] = (d.run == 2).astype(float)
    for col in ('log_order','prior_biased','prior_unbiased'):
        d['z_' + col] = d.groupby('subject')[col].transform(lambda v: (v - v.mean()) / v.std(ddof=1))
    core = np.column_stack([np.ones(len(d)), d.B, d.X, d.D, d.B*d.X, d.B*d.D, d.X*d.D, d.B*d.X*d.D,
                            d.U, d.U*d.X, d.U*d.D, d.U*d.X*d.D])
    nuisance = []
    for col in ('run2','z_log_order','z_prior_biased','z_prior_unbiased'):
        nuisance.extend([(d.subject.eq(subject).to_numpy(float) * d[col].to_numpy(float)) for subject in sorted(d.subject.unique())])
    design = np.column_stack([core, *nuisance])
    checker.check('independent history numeric design is 10995x144 full rank', design.shape == (10995, 144) and np.linalg.matrix_rank(design) == 144, f'shape={design.shape}, rank={np.linalg.matrix_rank(design)}')
    model_dir = results / 'mixed_models'
    diag = output_csv(model_dir, 'mixed_model_diagnostics_all.csv', checker)
    expected_formula = {
        'primary_joint': 'RT100 ~ B * X * D + U * X * D + (1 + B + U | subject)',
        'primary_joint_history': 'RT100 ~ B * X * D + U * X * D + subject:run2 + subject:z_log_order + subject:z_prior_biased + subject:z_prior_unbiased + (1 + B + U | subject)',
    }
    if expect_columns(diag, {'model','N','trials','REML','formula','fixed_columns','fixed_rank','expected_fixed_columns','singular','bobyqa_converged','nloptwrap_converged','diagnostic_status'}, 'mixed-model diagnostics', checker):
        checker.check('two model diagnostic rows', len(diag) == 2 and set(diag.model) == set(expected_formula), str(diag.model.tolist()))
        for model, ncol in [('primary_joint',12), ('primary_joint_history',144)]:
            got = diag.loc[diag.model.eq(model)]
            if len(got) == 1:
                r = got.iloc[0]
                formula = ''.join(str(r.formula).split())
                expected = ''.join(expected_formula[model].split())
                checker.check(f'R {model} formula binding', formula == expected, str(r.formula))
                checker.check(f'R {model} design/rank binding', int(r.N) == 33 and int(r.trials) == 10995 and bool(r.REML) and
                              int(r.fixed_columns) == ncol and int(r.fixed_rank) == ncol and int(r.expected_fixed_columns) == ncol,
                              str(r[['N','trials','REML','fixed_columns','fixed_rank','expected_fixed_columns']].to_dict()))
                checker.check(f'R {model} convergence/singularity gate', str(r.diagnostic_status) == 'PASSED' and not bool(r.singular) and
                              bool(r.bobyqa_converged) and bool(r.nloptwrap_converged), str(r[['diagnostic_status','singular','bobyqa_converged','nloptwrap_converged']].to_dict()))
    for model, ncol in [('primary_joint', 12), ('primary_joint_history', 144)]:
        fixed = output_csv(model_dir, f'{model}_fixed_effects_complete.csv', checker)
        matrix_cols = output_csv(model_dir, f'{model}_model_matrix_columns.csv', checker)
        opt = output_csv(model_dir, f'{model}_optimizer_agreement.csv', checker)
        if expect_columns(fixed, {'term','estimate_RT100','estimate_ms','se_RT100','se_ms','df','t','p_raw','ci_low_ms','ci_high_ms'}, f'{model} fixed effects', checker):
            checker.check(f'{model} fixed-effect row count', len(fixed) == ncol and fixed.term.nunique() == ncol, f'rows={len(fixed)}')
            checker.close(f'{model} ms scaling and Wald CI arithmetic', fixed[['estimate_ms','se_ms','ci_low_ms','ci_high_ms']].to_numpy(),
                          np.column_stack([fixed.estimate_RT100*100, fixed.se_RT100*100,
                                           (fixed.estimate_RT100 - stats.t.ppf(.975, fixed.df)*fixed.se_RT100)*100,
                                           (fixed.estimate_RT100 + stats.t.ppf(.975, fixed.df)*fixed.se_RT100)*100]), atol=2e-8)
        if expect_columns(matrix_cols, {'model','column_index','column'}, f'{model} matrix columns', checker):
            checker.check(f'{model} saved matrix column count', len(matrix_cols) == ncol and matrix_cols.column.nunique() == ncol and
                          set(matrix_cols.column_index) == set(range(1, ncol+1)), f'rows={len(matrix_cols)}')
        if expect_columns(opt, {'model','term','bobyqa_RT100','nloptwrap_RT100','abs_difference_RT100'}, f'{model} optimizer agreement', checker):
            checker.check(f'{model} optimizer term coverage', len(opt) == ncol and opt.term.nunique() == ncol, f'rows={len(opt)}')
            checker.close(f'{model} optimizer absolute-difference arithmetic', opt.abs_difference_RT100.to_numpy(), np.abs(opt.bobyqa_RT100 - opt.nloptwrap_RT100), atol=5e-14)
            if fixed is not None and {'term','estimate_RT100'} <= set(fixed):
                joined = fixed[['term','estimate_RT100']].merge(opt[['term','bobyqa_RT100']], on='term', validate='one_to_one')
                checker.close(f'{model} bobyqa output binds fixed effects', joined.bobyqa_RT100.to_numpy(), joined.estimate_RT100.to_numpy(), atol=2e-12)
        contrasts = output_csv(model_dir, f'{model}_simple_slopes_and_contrasts.csv', checker)
        if expect_columns(contrasts, {'contrast','N','trials','p_raw','p_Holm9','Holm_family'}, f'{model} LMM contrasts', checker):
            checker.check(f'{model} LMM contrast family is separate Holm-9', len(contrasts) == 9 and contrasts.Holm_family.astype(str).eq('nine exploratory contrasts from this joint LMM').all(), str(contrasts.Holm_family.unique()))
            if len(contrasts) == 9:
                checker.close(f'{model} LMM Holm-9 arithmetic', contrasts.p_Holm9.to_numpy(), holm(contrasts.p_raw.to_numpy()), atol=2e-12)
        omnibus = output_csv(model_dir, f'{model}_omnibus_type3.csv', checker)
        if expect_columns(omnibus, {'term','NumDF','DenDF','F value','Pr(>F)','model'}, f'{model} type-III omnibus', checker):
            checker.check(f'{model} omnibus is raw type-III output, not silently in Holm-9 family',
                          'p_Holm9' not in omnibus.columns and omnibus.model.astype(str).eq(model).all(), f'columns={list(omnibus.columns)}')
    checker.gaps.append('No Spearman residual-analysis output is produced by the rerun. The verifier therefore does not invent a residualization definition or substitute a Spearman retest.')
    checker.gaps.append('The saved type-III omnibus files have raw p values and no selected-manuscript-family indicator. This verifier distinguishes them from the separate Holm-9 contrast families but cannot verify any manuscript selection without an explicit claim-to-output map.')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-manifest', type=Path, required=True)
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--behavior-trials', type=Path, default=DEFAULT_BEHAVIOR)
    ap.add_argument('--frozen-draws', type=Path, default=DEFAULT_FROZEN_DRAWS)
    ap.add_argument('--frozen-subjects', type=Path, default=DEFAULT_FROZEN_SUBJECTS)
    ap.add_argument('--frozen-trial-resampling', type=Path, default=DEFAULT_FROZEN_TRIAL)
    ap.add_argument('--audit-output', type=Path, default=None, help='Default: RESULTS_DIR/independent_dependent_inference_verification.json')
    args = ap.parse_args()
    checker = Checker()
    results = args.results_dir
    audit = args.audit_output or results / 'independent_dependent_inference_verification.json'
    checker.require_file(results / 'results_manifest.json', 'results manifest exists')
    for p, label in [(args.behavior_trials, 'authorized behavior trial input exists'), (args.frozen_draws, 'frozen countmatched draw input exists'),
                     (args.frozen_subjects, 'frozen subject-order input exists'), (args.frozen_trial_resampling, 'frozen trial-resampling input exists')]:
        checker.require_file(p, label)
    inputs = load_inputs(args.input_manifest, checker)
    results_manifest = None
    try:
        results_manifest = read_json(results / 'results_manifest.json')
        checker.check('results manifest is completed run or labeled synthetic test', results_manifest.get('status') in {'complete','SYNTHETIC_TEST_NOT_EVIDENCE'}, str(results_manifest.get('status')))
    except Exception as exc:
        checker.check('results manifest parses', False, repr(exc))
    if inputs is not None and results_manifest is not None and args.behavior_trials.is_file() and args.frozen_draws.is_file() and args.frozen_trial_resampling.is_file():
        scores, matrices, parcels, runwise, provenance = inputs
        verify_manifest_hashes(results_manifest, provenance['hashes'], args.behavior_trials, args.frozen_draws, args.frozen_trial_resampling, checker)
        try:
            trials = pd.read_csv(args.behavior_trials)
            req = {'subject','group','run','trial','RT100','RT_ms','B','U','D','category','prev_category','correct','global_order','log_order','prior_biased','prior_unbiased'}
            checker.check('behavior source contract N=33/10995/correct RT gate', req <= set(trials) and len(trials) == 10995 and
                          trials.subject.nunique() == 33 and trials.groupby('group').subject.nunique().to_dict() == {'DS':17,'SS':16} and
                          (trials.correct == 1).all() and (trials.RT_ms > 200).all() and (trials.RT_ms <= 3000).all(), 'authorized frozen behavioral rows')
            checker.check('behavior/extraction identities match', set(trials.subject) == set(scores.subject), 'subject sets')
            scores = merge_behavior(scores, trials)
            networks = physical_network_scores(matrices, parcels)
            checker.close('fresh primary vector equals physical Default91 matrix score', scores.primary_default.to_numpy(), networks.Default.to_numpy(), atol=1e-8)
            netout = output_csv(results, 'physical_yeo7_network_scores_n33.csv', checker)
            if expect_columns(netout, {'subject','group',*NETWORK_COUNTS}, 'physical network score output', checker):
                aligned = netout.sort_values('subject').reset_index(drop=True)
                checker.check('saved physical network score identities', aligned.subject.tolist() == scores.subject.tolist() and aligned.group.tolist() == scores.group.tolist(), 'subject/group order')
                checker.close('saved physical network scores bind matrices', aligned[list(NETWORK_COUNTS)].to_numpy(), networks.to_numpy(), atol=2e-12)
            verify_associations(results, scores, checker)
            verify_williams(results, scores, checker)
            verify_networks_and_sensitivity(results, scores, networks, checker)
            verify_resampling(results, scores, trials, args.frozen_draws, args.frozen_subjects, args.frozen_trial_resampling, checker)
            verify_lopo_stability(results, scores, runwise, checker)
            verify_lmm_binding(results, scores, trials, checker)
        except Exception as exc:
            checker.check('independent statistical verification execution', False, repr(exc))
    else:
        checker.check('independent statistical verification execution', False, 'prerequisite file/input failure')
    passed = sum(x['status'] == 'PASS' for x in checker.checks)
    failed = sum(x['status'] == 'FAIL' for x in checker.checks)
    report = {
        'verifier': 'verify_dependent_inference.py', 'independence': 'No import or execution of run_dependent_inference.py; NumPy/SciPy/Pandas implementation only.',
        'status': 'PASS' if failed == 0 else 'FAIL', 'checks_passed': passed, 'checks_failed': failed,
        'results_dir': str(results), 'input_manifest': str(args.input_manifest), 'audit_output': str(audit),
        'checks': checker.checks, 'coverage_gaps': checker.gaps,
    }
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'checks_passed': passed, 'checks_failed': failed, 'audit_output': str(audit), 'coverage_gaps': checker.gaps}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
