#!/usr/bin/env python3
"""Independent arithmetic verifier for fresh network diagnostics.

This script deliberately does not import ``run_network_diagnostics.py``.  It
rebuilds the tested arithmetic from inputs and checks the saved result tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

N = 33
ORDER = ("Default", "SomMot", "Vis", "Cont", "SalVentAttn", "DorsAttn", "Limbic")
N_COMMON_PARCELS = 399
EXCLUDED_NATIVE_LABELS = frozenset({323})
COMMON_NATIVE_LABELS = frozenset(range(1, 401)) - EXCLUDED_NATIVE_LABELS
COUNTS = {"Default": 91, "SomMot": 77, "Vis": 61, "Cont": 52, "SalVentAttn": 47, "DorsAttn": 46, "Limbic": 25}
EDGE_UNIVERSE = N_COMMON_PARCELS * (N_COMMON_PARCELS - 1) // 2
EDGE_HOLM_COLUMN = f"p_Holm_{EDGE_UNIVERSE}"


def fail(message: str) -> None:
    raise RuntimeError(message)


def check(ok: bool, message: str) -> None:
    if not ok:
        fail(message)


def center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.asarray(values, float).copy()
    for group, expected in (("SS", 16), ("DS", 17)):
        ix = np.flatnonzero(groups == group)
        check(len(ix) == expected, f"Expected {group}={expected}.")
        out[ix] -= out[ix].mean(axis=0, keepdims=True)
    return out


def corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean()
    return np.sum(x * y[:, None], axis=0) / np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y))


def pair(left: str, right: str) -> str:
    return f"{left}--{right}" if ORDER.index(left) <= ORDER.index(right) else f"{right}--{left}"


def williams(a: float, b: float, c: float) -> tuple[float, float]:
    n_eff = 32
    determinant = 1 - a*a - b*b - c*c + 2*a*b*c
    check(determinant > 0, "Non-positive Williams correlation determinant.")
    statistic = (a-b)*np.sqrt((n_eff-1)*(1+c)) / np.sqrt(2*determinant*(n_eff-1)/(n_eff-3) + (a+b)**2/4*(1-c)**3)
    return float(statistic), float(2*stats.t.sf(abs(statistic), n_eff-3))


def close(actual: np.ndarray, expected: np.ndarray, label: str, atol: float) -> float:
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    check(actual.shape == expected.shape, f"{label}: shape mismatch {actual.shape} != {expected.shape}.")
    delta = float(np.nanmax(np.abs(actual - expected))) if actual.size else 0.0
    check(np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=True), f"{label}: maximum difference {delta:g} exceeds {atol:g}.")
    return delta


def check_common_coverage_geometry(value: dict, label: str) -> None:
    geometry = value.get("common_coverage_geometry")
    check(isinstance(geometry, dict), f"{label} lacks common_coverage_geometry.")
    assert isinstance(geometry, dict)
    check(int(geometry.get("nominal_atlas_parcels", -1)) == 400, f"{label} nominal atlas count failed.")
    check(int(geometry.get("effective_common_coverage_parcels", -1)) == N_COMMON_PARCELS, f"{label} common coverage parcel count failed.")
    check(set(geometry.get("excluded_native_labels", [])) == EXCLUDED_NATIVE_LABELS, f"{label} excluded-native-label set failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path, help="Completed network_diagnostics output directory.")
    parser.add_argument("--input-manifest", type=Path, help="Override input manifest recorded in diagnostics manifest.")
    parser.add_argument("--behavior-trials", type=Path, help="Override behavioral trial file recorded in diagnostics manifest.")
    parser.add_argument("--synthetic-test", action="store_true", help="Require labelled synthetic diagnostics; never treat this check as evidence.")
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    directory = args.diagnostics
    result_manifest = json.loads((directory / "network_diagnostics_manifest.json").read_text())
    if args.synthetic_test:
        check(result_manifest.get("status") == "SYNTHETIC_TEST_NOT_EVIDENCE", "Synthetic verifier mode requires synthetic diagnostics output.")
    else:
        check(result_manifest.get("status") == "complete", "Real verification requires complete non-synthetic diagnostics output.")
    check_common_coverage_geometry(result_manifest, "Diagnostics manifest")
    input_manifest = args.input_manifest or Path(result_manifest["input"]["input_manifest"])
    behavior_path = args.behavior_trials or Path(result_manifest["behavior_trials"])
    source_manifest = json.loads(input_manifest.read_text())
    check_common_coverage_geometry(source_manifest, "Input manifest")
    def source_path(key: str) -> Path:
        path = Path(source_manifest[key]["path"])
        return path if path.is_absolute() else input_manifest.parent / path
    score_cfg = source_manifest["participant_scores"]
    score_column = score_cfg["primary_column"]
    scores = pd.read_csv(source_path("participant_scores"))[["subject", "group", score_column]].rename(columns={score_column: "primary_default"})
    scores = scores.sort_values("subject").reset_index(drop=True)
    check(len(scores) == N and scores.subject.is_unique and scores.group.value_counts().to_dict() == {"DS": 17, "SS": 16}, "Input scores are not the authorized N33 groups.")
    subjects = pd.read_csv(source_path("matrix_subjects"))
    matrices = np.load(source_path("primary_matrices"), allow_pickle=False)
    check("parcel_metadata" in source_manifest, "Input manifest lacks required common-coverage parcel metadata.")
    parcels = pd.read_csv(source_path("parcel_metadata")).sort_values("matrix_index").reset_index(drop=True)
    check(matrices.shape == (N, N_COMMON_PARCELS, N_COMMON_PARCELS) and np.allclose(matrices, matrices.swapaxes(1,2), atol=1e-8), "Matrix contract failed.")
    check(parcels.matrix_index.tolist() == list(range(N_COMMON_PARCELS)) and set(parcels.native_label) == COMMON_NATIVE_LABELS, "Common-coverage local/native mapping failed.")
    check(parcels.yeo7_network.value_counts().to_dict() == COUNTS, "Yeo7 parcel counts failed.")
    pos = {s: i for i, s in enumerate(subjects.subject)}
    matrices = matrices[[pos[s] for s in scores.subject]]
    groups = scores.group.to_numpy()
    trials = pd.read_csv(behavior_path)
    check(len(trials) == 10995 and (trials.correct == 1).all() and trials.RT_ms.between(200,3000,inclusive="both").all(), "Behavior gate failed.")
    means = trials.groupby(["subject","group","category"], observed=True).RT_ms.mean().unstack("category").reset_index()
    means["biased_capture"] = means.biased - means.absent
    scores = scores.merge(means[["subject","group","biased_capture"]], on=["subject","group"], validate="one_to_one")
    capture = scores.biased_capture.to_numpy(float)

    score_table = pd.read_csv(directory / "yeo7_network_summary_scores_n33.csv")
    association = pd.read_csv(directory / "yeo7_all_28_summary_associations_holm28.csv")
    comparison = pd.read_csv(directory / "yeo7_default_vs_other_within_network_williams_holm6.csv")
    edge_table = pd.read_csv(directory / f"edgewise_group_adjusted_allcommon{EDGE_UNIVERSE}_holm.csv")
    screen = pd.read_csv(directory / f"edgewise_allcommon{EDGE_UNIVERSE}_screen_summary.csv").iloc[0]
    pair_enrich = pd.read_csv(directory / "yeo7_pair_enrichment_hypergeom_holm28.csv")
    endpoint_enrich = pd.read_csv(directory / "yeo7_endpoint_enrichment_hypergeom_holm7.csv")

    checks: dict[str, float | int] = {}
    check(len(score_table) == 33 and score_table.subject.tolist() == scores.subject.tolist(), "Saved network-score table subject order failed.")
    vectors: dict[str, np.ndarray] = {}
    expected_rows = []
    labels = parcels.yeo7_network.to_numpy()
    for i, left in enumerate(ORDER):
        ix = np.flatnonzero(labels == left)
        for right in ORDER[i:]:
            jx = np.flatnonzero(labels == right)
            name = pair(left, right)
            if left == right:
                ii, jj = np.triu_indices(len(ix), 1)
                vector = matrices[:, ix][:, :, ix][:, ii, jj].mean(axis=1)
                possible = len(ii)
                kind = "within"
            else:
                vector = matrices[:, ix][:, :, jx].reshape(N, -1).mean(axis=1)
                possible = len(ix)*len(jx)
                kind = "between"
            vectors[name] = vector
            expected_rows.append((name, possible, kind))
            checks[f"summary_{name}"] = close(score_table[name], vector, f"network score {name}", args.atol)
    check(len(expected_rows) == 28 and sum(x[1] for x in expected_rows) == EDGE_UNIVERSE, "Summary universe failed.")
    checks["primary_default"] = close(scores.primary_default, vectors["Default--Default"], "fresh Default91 primary", args.atol)

    centered_y = center(capture, groups)
    expected_r, expected_p = [], []
    centered_networks: dict[str, np.ndarray] = {}
    for name, _, _ in expected_rows:
        x = center(vectors[name], groups)
        centered_networks[name] = x
        r = float(corr_columns(x.reshape(N,1), centered_y)[0])
        expected_r.append(r)
        expected_p.append(float(2*stats.t.sf(abs(r*np.sqrt(30/(1-r*r))), 30)))
    check(association.network_pair.tolist() == [x[0] for x in expected_rows], "Association pair order failed.")
    checks["association_r"] = close(association.r_group_adjusted, expected_r, "28 summary r", args.atol)
    checks["association_p"] = close(association.p_raw, expected_p, "28 summary raw p", args.atol)
    checks["association_holm"] = close(association.p_Holm_28, multipletests(expected_p, method="holm")[1], "28 summary Holm", args.atol)

    check(len(comparison) == 6 and comparison.comparator_network.tolist() == [n for n in ORDER if n != "Default"], "Williams comparator family failed.")
    t_values, p_values = [], []
    primary_r = expected_r[0]
    for network in [n for n in ORDER if n != "Default"]:
        other = f"{network}--{network}"
        other_r = expected_r[[name for name,_,_ in expected_rows].index(other)]
        inter = float(corr_columns(centered_networks["Default--Default"].reshape(N,1), centered_networks[other])[0])
        t, p = williams(primary_r, other_r, inter)
        t_values.append(t); p_values.append(p)
    checks["williams_t"] = close(comparison.t_Williams, t_values, "Williams t", args.atol)
    checks["williams_p"] = close(comparison.p_raw, p_values, "Williams raw p", args.atol)
    checks["williams_holm"] = close(comparison.p_Holm_6, multipletests(p_values, method="holm")[1], "Williams Holm6", args.atol)

    ii, jj = np.triu_indices(N_COMMON_PARCELS, 1)
    check(len(edge_table) == EDGE_UNIVERSE and edge_table.edge_index.tolist() == list(range(EDGE_UNIVERSE)), "Edge universe/order failed.")
    edges = center(matrices[:, ii, jj], groups)
    edge_r = corr_columns(edges, centered_y)
    edge_t = edge_r*np.sqrt(30/(1-edge_r*edge_r))
    edge_p = 2*stats.t.sf(np.abs(edge_t), 30)
    checks["edge_r"] = close(edge_table.r_group_adjusted, edge_r, f"{EDGE_UNIVERSE:,} edge r", args.atol)
    checks["edge_p"] = close(edge_table.p_raw, edge_p, f"{EDGE_UNIVERSE:,} edge p", args.atol)
    checks["edge_holm"] = close(edge_table[EDGE_HOLM_COLUMN], multipletests(edge_p,method="holm")[1], f"{EDGE_UNIVERSE:,} edge Holm", args.atol)
    positive = (edge_p < .01) & (edge_r > 0)
    negative = (edge_p < .01) & (edge_r < 0)
    check(np.array_equal(edge_table.selected_positive_raw_screen.to_numpy(bool), positive), "Positive raw-screen mask differs.")
    check(np.array_equal(edge_table.selected_negative_raw_screen.to_numpy(bool), negative), "Negative raw-screen mask differs.")
    check(int(screen.edge_universe) == EDGE_UNIVERSE and int(screen.edge_df) == 30, "Saved screen universe/df failed.")
    check(int(screen.selected_positive_raw_screen) == int(positive.sum()) and int(screen.selected_negative_raw_screen) == int(negative.sum()), "Saved screen count differs.")

    n_pos = int(positive.sum())
    check(len(pair_enrich) == 28 and int(pair_enrich.possible_edges.sum()) == EDGE_UNIVERSE, "Pair enrichment universe failed.")
    pair_p = []
    for row in pair_enrich.itertuples(index=False):
        mask = np.array([pair(a,b) == row.network_pair for a,b in zip(labels[ii], labels[jj])])
        observed = int((mask & positive).sum())
        expected = n_pos*int(mask.sum())/EDGE_UNIVERSE
        p = float(stats.hypergeom.sf(observed-1,EDGE_UNIVERSE,int(mask.sum()),n_pos)) if n_pos else 1.0
        check(int(row.selected_positive) == observed and int(row.possible_edges) == int(mask.sum()), f"Pair enrichment counts failed: {row.network_pair}.")
        checks[f"pair_oe_{row.network_pair}"] = close([row.observed_expected_ratio], [observed/expected if expected else np.nan], f"pair O/E {row.network_pair}", args.atol)
        pair_p.append(p)
    checks["pair_holm"] = close(pair_enrich.hypergeom_p_Holm_28, multipletests(pair_p,method="holm")[1], "pair enrichment Holm28", args.atol)

    endpoint_p = []
    for row in endpoint_enrich.itertuples(index=False):
        mask = (labels[ii] == row.network) | (labels[jj] == row.network)
        observed = int((mask & positive).sum())
        expected = n_pos*int(mask.sum())/EDGE_UNIVERSE
        p = float(stats.hypergeom.sf(observed-1,EDGE_UNIVERSE,int(mask.sum()),n_pos)) if n_pos else 1.0
        check(int(row.selected_positive_touching) == observed and int(row.possible_touching_edges) == int(mask.sum()), f"Endpoint enrichment counts failed: {row.network}.")
        checks[f"endpoint_oe_{row.network}"] = close([row.observed_expected_ratio], [observed/expected if expected else np.nan], f"endpoint O/E {row.network}", args.atol)
        endpoint_p.append(p)
    checks["endpoint_holm"] = close(endpoint_enrich.hypergeom_p_Holm_7, multipletests(endpoint_p,method="holm")[1], "endpoint enrichment Holm7", args.atol)
    report = {"status": "PASS_SYNTHETIC_NOT_EVIDENCE" if args.synthetic_test else "PASS", "diagnostics": str(directory), "common_coverage_geometry": {"nominal_atlas_parcels": 400, "effective_common_coverage_parcels": N_COMMON_PARCELS, "excluded_native_labels": sorted(EXCLUDED_NATIVE_LABELS)}, "checked": {"network_summary_vectors": 28, "network_association_family": 28, "Williams_family": 6, "edgewise_family": EDGE_UNIVERSE, "pair_enrichment_family": 28, "endpoint_enrichment_family": 7}, "maximum_absolute_differences": checks}
    (directory / "independent_arithmetic_verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "edge_universe": EDGE_UNIVERSE, "selected_positive": int(positive.sum()), "selected_negative": int(negative.sum()), "max_difference": max(float(v) for v in checks.values())}, indent=2))


if __name__ == "__main__":
    main()
