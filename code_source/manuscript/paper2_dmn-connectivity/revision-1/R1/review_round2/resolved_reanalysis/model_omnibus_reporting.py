#!/usr/bin/env python3
"""Apply the frozen Type-III connectivity omnibus reporting plan.

This formatter deliberately does not fit models, replace raw Type-III output, or
modify the separate Holm-9 simple-slope contrast family.  For each supplied
model it preserves every source omnibus row and appends a prospective Holm-6
column only for the six frozen connectivity-containing terms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_PLAN = HERE / "model_claim_reporting_plan.json"
DEFAULT_MODELS = ("primary_joint", "primary_joint_history")
P_RAW_COLUMN = "Pr(>F)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_term(term: str) -> str:
    """Canonicalize only colon-separated Type-III term component order."""
    parts = term.split(":")
    if not term or any(not part for part in parts):
        raise ValueError(f"Invalid Type-III term: {term!r}")
    return ":".join(sorted(parts))


def holm(p_values: list[float]) -> list[float]:
    """Holm adjustment returned in the original list order."""
    if not p_values or any(not math.isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise ValueError("Holm requires one or more finite p values in [0, 1].")
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted_sorted: list[float] = []
    running = 0.0
    family_size = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (family_size - rank) * p_values[index])
        adjusted_sorted.append(min(running, 1.0))
    adjusted = [math.nan] * family_size
    for index, value in zip(order, adjusted_sorted):
        adjusted[index] = value
    return adjusted


def read_plan(plan_path: Path) -> tuple[dict[str, Any], list[str], str]:
    if not plan_path.is_file():
        raise FileNotFoundError(f"Frozen reporting plan is absent: {plan_path}")
    plan = json.loads(plan_path.read_text())
    binding = plan.get("mixed_model_omnibus_binding")
    if not isinstance(binding, dict):
        raise ValueError("Plan lacks mixed_model_omnibus_binding.")
    declared = binding.get("eligible_connectivity_terms")
    if not isinstance(declared, list) or len(declared) != 6 or not all(isinstance(x, str) for x in declared):
        raise ValueError("Plan must declare exactly six string eligible_connectivity_terms.")
    eligible = [canonical_term(term) for term in declared]
    if len(set(eligible)) != 6:
        raise ValueError("Frozen eligible connectivity terms are not canonically unique.")
    family = binding.get("declared_reporting_family")
    if not isinstance(family, str) or "Holm6" not in family:
        raise ValueError("Plan does not declare a Holm6 omnibus reporting family.")
    return plan, eligible, family


def read_omnibus(source: Path, model: str) -> tuple[list[dict[str, str]], list[str]]:
    if not source.is_file():
        raise FileNotFoundError(f"Raw Type-III source is absent: {source}")
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"term", "model", P_RAW_COLUMN}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{source} lacks required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source} has no Type-III rows.")
    seen: set[str] = set()
    for row in rows:
        if row.get("model") != model:
            raise ValueError(f"{source} has model={row.get('model')!r}; expected {model!r}.")
        canonical = canonical_term(row["term"])
        if canonical in seen:
            raise ValueError(f"{source} has duplicate canonical Type-III term {canonical!r}.")
        seen.add(canonical)
        try:
            p = float(row[P_RAW_COLUMN])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source} has nonnumeric raw p for {row['term']!r}.") from error
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"{source} has invalid raw p for {row['term']!r}: {p!r}")
    return rows, fields


def render_model(
    *, plan_path: Path, mixed_models_dir: Path, output_dir: Path, model: str
) -> dict[str, Any]:
    plan, eligible, family = read_plan(plan_path)
    source = mixed_models_dir / f"{model}_omnibus_type3.csv"
    rows, source_fields = read_omnibus(source, model)
    source_by_canonical = {canonical_term(row["term"]): row for row in rows}
    missing = [term for term in eligible if term not in source_by_canonical]
    if missing:
        raise ValueError(f"{model} is missing frozen eligible Type-III term(s): {missing}")

    eligible_p = [float(source_by_canonical[term][P_RAW_COLUMN]) for term in eligible]
    adjusted = dict(zip(eligible, holm(eligible_p), strict=True))
    output_rows: list[dict[str, str]] = []
    for row in rows:
        canonical = canonical_term(row["term"])
        formatted = dict(row)  # retain every raw source field and value unchanged
        formatted.update(
            {
                "canonical_term": canonical,
                "source_p_raw": row[P_RAW_COLUMN],
                "in_prospective_Holm6_connectivity_family": "TRUE" if canonical in adjusted else "FALSE",
                "p_Holm6_connectivity_terms": "" if canonical not in adjusted else format(adjusted[canonical], ".17g"),
                "family": family if canonical in adjusted else "",
            }
        )
        output_rows.append(formatted)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{model}_omnibus_reporting_holm6.csv"
    report_fields = [
        *source_fields,
        "canonical_term",
        "source_p_raw",
        "in_prospective_Holm6_connectivity_family",
        "p_Holm6_connectivity_terms",
        "family",
    ]
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(output_rows)

    binding_path = output_dir / f"{model}_omnibus_claim_binding.json"
    binding = {
        "status": "FROZEN_PLAN_APPLIED_NOT_CLAIM_SELECTION",
        "model": model,
        "plan_path": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "source_type3_path": str(source),
        "source_type3_sha256": sha256(source),
        "report_path": str(report_path),
        "report_sha256": sha256(report_path),
        "all_raw_type3_rows_retained": len(output_rows),
        "source_raw_p_column": P_RAW_COLUMN,
        "prospective_reporting_family": family,
        "family_size": 6,
        "eligible_terms_in_frozen_plan_order": eligible,
        "bindings": [
            {
                "canonical_term": term,
                "source_term": source_by_canonical[term]["term"],
                "source_p_raw": float(source_by_canonical[term][P_RAW_COLUMN]),
                "p_Holm6_connectivity_terms": adjusted[term],
                "claim_boundary": "Omnibus connectivity-term reporting only; not a simple-slope contrast, mediation, causality, or model-difference claim.",
            }
            for term in eligible
        ],
        "noneligible_raw_terms_retained_without_holm6": [
            {"canonical_term": canonical_term(row["term"]), "source_term": row["term"]}
            for row in rows
            if canonical_term(row["term"]) not in adjusted
        ],
        "contrast_family_boundary": "The separate nine written contrasts per model remain Holm9 and are not altered by this formatter.",
        "plan_status": plan.get("status"),
    }
    binding_path.write_text(json.dumps(binding, indent=2) + "\n")
    return {
        "model": model,
        "source_rows": len(rows),
        "eligible_rows": len(adjusted),
        "report": str(report_path),
        "claim_binding": str(binding_path),
    }


def self_test() -> None:
    """Synthetic arithmetic/provenance test; never reads empirical data."""
    plan = {
        "status": "synthetic_test_only",
        "mixed_model_omnibus_binding": {
            "eligible_connectivity_terms": ["X", "X:D", "B:X", "U:X", "B:X:D", "U:X:D"],
            "declared_reporting_family": "Holm6 synthetic frozen family",
        },
    }
    rows = [
        {"term": "B", "model": "synthetic", P_RAW_COLUMN: "0.9", "F value": "1"},
        {"term": "X", "model": "synthetic", P_RAW_COLUMN: "0.04", "F value": "2"},
        {"term": "D:X", "model": "synthetic", P_RAW_COLUMN: "0.001", "F value": "3"},
        {"term": "B:X", "model": "synthetic", P_RAW_COLUMN: "0.2", "F value": "4"},
        {"term": "X:U", "model": "synthetic", P_RAW_COLUMN: "0.01", "F value": "5"},
        {"term": "B:D:X", "model": "synthetic", P_RAW_COLUMN: "0.03", "F value": "6"},
        {"term": "U:X:D", "model": "synthetic", P_RAW_COLUMN: "0.05", "F value": "7"},
        {"term": "D", "model": "synthetic", P_RAW_COLUMN: "0.8", "F value": "8"},
    ]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
        root = Path(directory)
        plan_path = root / "plan.json"
        model_dir = root / "mixed_models"
        output_dir = root / "reports"
        model_dir.mkdir()
        plan_path.write_text(json.dumps(plan))
        source = model_dir / "synthetic_omnibus_type3.csv"
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = render_model(plan_path=plan_path, mixed_models_dir=model_dir, output_dir=output_dir, model="synthetic")
        with Path(result["report"]).open(newline="") as handle:
            got = list(csv.DictReader(handle))
        assert len(got) == len(rows)
        assert [row["term"] for row in got] == [row["term"] for row in rows]
        assert [row[P_RAW_COLUMN] for row in got] == [row[P_RAW_COLUMN] for row in rows]
        expected = {"X": 0.12, "D:X": 0.006, "B:X": 0.2, "U:X": 0.05, "B:D:X": 0.12, "D:U:X": 0.12}
        for row in got:
            canonical = row["canonical_term"]
            if canonical in expected:
                assert row["in_prospective_Holm6_connectivity_family"] == "TRUE"
                assert math.isclose(float(row["p_Holm6_connectivity_terms"]), expected[canonical], abs_tol=1e-15)
            else:
                assert row["in_prospective_Holm6_connectivity_family"] == "FALSE"
                assert row["p_Holm6_connectivity_terms"] == ""
        binding = json.loads(Path(result["claim_binding"]).read_text())
        assert binding["family_size"] == 6
        assert binding["all_raw_type3_rows_retained"] == len(rows)
        assert binding["source_type3_sha256"] == sha256(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="Frozen JSON reporting plan.")
    parser.add_argument("--mixed-models-dir", type=Path, help="Directory holding {model}_omnibus_type3.csv files.")
    parser.add_argument("--output-dir", type=Path, help="Directory for reporting CSVs and binding JSONs; default is --mixed-models-dir.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Model stems to map.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic arithmetic/provenance tests only.")
    args = parser.parse_args()
    if args.self_test:
        if args.mixed_models_dir or args.output_dir or args.models != list(DEFAULT_MODELS):
            parser.error("--self-test cannot be combined with model input/output options.")
        self_test()
        print(json.dumps({"status": "PASS", "test": "synthetic Holm6 arithmetic and raw-statistic provenance"}))
        return 0
    if args.mixed_models_dir is None:
        parser.error("--mixed-models-dir is required unless --self-test is used.")
    output_dir = args.output_dir or args.mixed_models_dir
    mapped = [render_model(plan_path=args.plan, mixed_models_dir=args.mixed_models_dir, output_dir=output_dir, model=model) for model in args.models]
    print(json.dumps({"status": "FROZEN_PLAN_APPLIED_NOT_CLAIM_SELECTION", "mapped": mapped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
