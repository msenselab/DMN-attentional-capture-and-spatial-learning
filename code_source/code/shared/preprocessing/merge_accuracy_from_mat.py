#!/usr/bin/env python3
"""Merge trial-wise accuracy (correct/error) from legacy MATLAB .mat files into BIDS events.

Why: BIDS suppression *_events.tsv currently lacks correctness.
Source: code/cross_dimension/data/*.mat contains `trials` matrix where column 12 = correct, col 16 = rt.

Strategy:
- Map BIDS subject -> origin_name prefix (DS02/SS19/...) using participants.tsv.
- For each subject and each run (run-1/run-2), find best-matching .mat file among origin_prefix_*.mat
  by correlating RT vectors (events.tsv RT vs mat trials[:,15])
- Attach `correct` vector (trials[:,11]) to events trials by index (0..183)

Outputs:
- derivatives/behavior/vigilance_state/merged_events_with_correct.csv
- derivatives/behavior/vigilance_state/match_table.csv

"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat

BIDS_ROOT = __import__('os').environ.get('DMN_PROJECT_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[3]))
MAT_DIR = os.path.join(BIDS_ROOT, 'code', 'cross_dimension', 'data')
OUT_DIR = os.path.join(BIDS_ROOT, 'derivatives', 'behavior', 'vigilance_state')

PATTERN = os.path.join(BIDS_ROOT, 'sub-*/func/*_task-suppression_run-*_events.tsv')

# Indices in MATLAB trials matrix (0-based)
IDX_CORRECT = 11
IDX_RT = 15


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a = a[m]
    b = b[m]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def load_participants_map():
    p = os.path.join(BIDS_ROOT, 'participants.tsv')
    # manual parse (file has header formatting quirks)
    rows = []
    with open(p, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            participant_id, age, sex, group, origin_name = parts[:5]
            rows.append({
                'participant_id': participant_id.replace('sub-', ''),
                'origin_name': origin_name.strip(),
            })
    df = pd.DataFrame(rows)
    df['origin_prefix'] = df['origin_name'].str.split('_').str[0]  # DS02, SS19, ...
    return df.set_index('participant_id')


def load_mat_rt_correct(mat_path):
    d = loadmat(mat_path)
    trials = d['trials']
    rt = trials[:, IDX_RT].astype(float)
    correct = trials[:, IDX_CORRECT].astype(float)
    # rt in seconds already in legacy? It appears seconds in example (1.38). keep.
    return rt, correct


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    subj_map = load_participants_map()

    match_rows = []
    merged_rows = []

    for ev_path in sorted(glob.glob(PATTERN)):
        fn = os.path.basename(ev_path)
        subject = fn.split('_')[0].replace('sub-', '')
        run = int(fn.split('_')[2].replace('run-', ''))

        ev = pd.read_csv(ev_path, sep='\t')
        if 'RT' not in ev.columns:
            continue

        # ensure stable order
        ev = ev.reset_index(drop=True)
        ev_rt = ev['RT'].to_numpy(dtype=float)

        if subject not in subj_map.index:
            match_rows.append({
                'subject': subject, 'run': run, 'status': 'no_subject_mapping',
                'best_mat': None, 'best_corr': np.nan
            })
            continue

        origin_prefix = subj_map.loc[subject, 'origin_prefix']
        candidates = sorted(glob.glob(os.path.join(MAT_DIR, f"{origin_prefix}_s*.mat")))

        best = {'mat': None, 'corr': -np.inf, 'rt': None, 'correct': None}

        for mat in candidates:
            try:
                mat_rt, mat_correct = load_mat_rt_correct(mat)
            except Exception:
                continue

            # length guard
            n = min(len(ev_rt), len(mat_rt))
            corr = _safe_corr(ev_rt[:n], mat_rt[:n])
            if np.isfinite(corr) and corr > best['corr']:
                best = {'mat': mat, 'corr': corr, 'rt': mat_rt, 'correct': mat_correct}

        status = 'ok' if best['mat'] is not None else 'no_mat_candidates'
        match_rows.append({
            'subject': subject,
            'run': run,
            'origin_prefix': origin_prefix,
            'status': status,
            'best_mat': best['mat'],
            'best_corr': best['corr'],
            'n_trials_events': len(ev_rt),
            'n_trials_mat': None if best['rt'] is None else len(best['rt']),
        })

        if status != 'ok':
            continue

        # attach correct by trial index
        n = min(len(ev), len(best['correct']))
        ev2 = ev.iloc[:n].copy()
        ev2['correct'] = best['correct'][:n].astype(int)
        ev2['subject'] = subject
        ev2['run'] = run
        ev2['source_events'] = ev_path
        ev2['source_mat'] = best['mat']
        ev2['rt_corr_events_vs_mat'] = best['corr']
        ev2['trial'] = np.arange(1, n + 1)

        merged_rows.append(ev2)

    match_df = pd.DataFrame(match_rows)
    match_df.to_csv(os.path.join(OUT_DIR, 'match_table.csv'), index=False)

    if merged_rows:
        merged = pd.concat(merged_rows, ignore_index=True)
        merged.to_csv(os.path.join(OUT_DIR, 'merged_events_with_correct.csv'), index=False)
    else:
        merged = pd.DataFrame()

    print(f"Wrote: {os.path.join(OUT_DIR, 'match_table.csv')}")
    print(f"Wrote: {os.path.join(OUT_DIR, 'merged_events_with_correct.csv')} (rows={len(merged)})")
    # basic QC
    if len(match_df):
        ok = match_df[match_df['status'] == 'ok']
        print(f"Matched {len(ok)}/{len(match_df)} subject-runs")
        if len(ok):
            print("Correlation summary (events RT vs mat RT):")
            print(ok['best_corr'].describe().to_string())


if __name__ == '__main__':
    main()
