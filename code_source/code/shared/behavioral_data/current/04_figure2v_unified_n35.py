#!/usr/bin/env python3
"""Regenerate Figure 2v from the final all-available-data N=35 analysis.

The visual grammar follows the current Figure 2: (A) blockwise trajectory,
(B) grouped condition bars, and (C) spatial-learning delta bars with participant
dots. Unlike the previous export, Panels B and C use all available training and
scanner observations rather than scanner-only descriptive means. Inferential
annotations are read from the saved unified N=35 model outputs.

Outputs are non-destructive review variants; the canonical Figure 2 is not
overwritten.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS_SCRIPT = HERE / "01_rt_interference_spatial_learning_n35.py"
STATS_DIR = HERE / "results" / "rt_interference_spatial_n35"
OUT_DIR = HERE / "results" / "figure2v_unified_n35"
FIG_DIR = ROOT / "manuscript" / "paper1" / "figures"
OUT_PNG = FIG_DIR / "figure2v_behavior_unified_n35.png"
OUT_PDF = FIG_DIR / "figure2v_behavior_unified_n35.pdf"

# Preserve the established dimension palette and figure grammar.
COL_SAME = "#D62728"
COL_DIFF = "#1F77B4"
COL_ABSENT = "#7F7F7F"
N_TRAIN_BLOCKS = 6
N_TOTAL_BLOCKS = 14


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("behavior_n35", ANALYSIS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ANALYSIS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stars(p: float) -> str:
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    if p < .10:
        return "†"
    return "n.s."


def add_bracket(ax, x1, x2, y, text, height=7, fontsize=19, lw=1.1):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            color="black", lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2, y + height + 2, text,
            ha="center", va="bottom", fontsize=fontsize, clip_on=False)


def combined_data():
    analysis = load_analysis_module()
    training = analysis.load_training(analysis.load_id_map()).copy()
    scanner = analysis.load_scanner().copy()
    use = ["subject", "group", "Dimension", "DisRegion", "RT_ms", "block", "scanner"]
    both = pd.concat([training[use], scanner[use]], ignore_index=True)

    # Scope guards: behavioral availability, not imaging QC.
    assert both["subject"].nunique() == 35
    assert both.loc[both["group"].eq("SS"), "subject"].nunique() == 17
    assert both.loc[both["group"].eq("DS"), "subject"].nunique() == 18
    assert training["subject"].nunique() == 31
    assert scanner["subject"].nunique() == 35
    return training, scanner, both


def panel_a(ax, both: pd.DataFrame) -> pd.DataFrame:
    styles = {
        "Same": (COL_SAME, "o", "-", "Same"),
        "Different": (COL_DIFF, "s", "-", "Different"),
        "Absent": (COL_ABSENT, "^", "--", "Absent"),
    }
    subject = (
        both.groupby(["subject", "block", "Dimension"], observed=True)["RT_ms"]
        .mean().reset_index()
    )
    summary = (
        subject.groupby(["block", "Dimension"], observed=True)["RT_ms"]
        .agg(mean_ms="mean", sem_ms="sem", n="count").reset_index()
    )
    for dim, (color, marker, linestyle, label) in styles.items():
        d = summary.loc[summary["Dimension"].eq(dim)].sort_values("block")
        ax.errorbar(d["block"], d["mean_ms"], yerr=d["sem_ms"],
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.8, markersize=5.2, capsize=3, label=label)

    ax.axvline(N_TRAIN_BLOCKS + .5, color="black", linestyle=":", linewidth=1.4)
    ax.text(3.5, 699, "Training", ha="center", va="bottom",
            fontsize=20, style="italic", color="gray")
    ax.text(10.5, 699, "Scanner", ha="center", va="bottom",
            fontsize=20, style="italic", color="gray")
    ax.set(xlim=(.5, 14.5), ylim=(680, 1220), xlabel="Block", ylabel="RT (ms)")
    ax.set_xticks(range(1, N_TOTAL_BLOCKS + 1))
    ax.legend(frameon=False, loc="upper right", fontsize=18)
    return summary


def condition_table(both: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    specs = {
        "SS": ("Same", "Different"),
        "DS": ("Different", "Same"),
    }
    for group, (biased, unbiased) in specs.items():
        d = both.loc[both["group"].eq(group)].copy()

        def classify(row):
            dim, region = row["Dimension"], row["DisRegion"]
            if dim == "Absent":
                return "Absent"
            if dim == biased and region in {"Frequent", "Rare"}:
                return f"{biased}-{region}"
            if dim == unbiased:
                return f"{unbiased}-Unbiased"
            return None

        d["condition"] = d.apply(classify, axis=1)
        frames.append(d.loc[d["condition"].notna(),
                            ["subject", "group", "condition", "RT_ms"]])
    selected = pd.concat(frames, ignore_index=True)
    subject = (
        selected.groupby(["subject", "group", "condition"], observed=True)["RT_ms"]
        .mean().reset_index()
    )

    # Cousineau-normalized SEM within each group, matching the current format.
    out = []
    for group, d in subject.groupby("group", observed=True):
        d = d.copy()
        subject_grand = d.groupby("subject")["RT_ms"].mean()
        grand = subject_grand.mean()
        d["RT_centered"] = d["RT_ms"] - d["subject"].map(subject_grand) + grand
        raw = d.groupby("condition", observed=True)["RT_ms"].mean()
        sem = d.groupby("condition", observed=True)["RT_centered"].sem()
        n = d.groupby("condition", observed=True)["subject"].nunique()
        for condition in raw.index:
            out.append({"group": group, "condition": condition,
                        "mean_ms": raw[condition], "sem_ms": sem[condition],
                        "n": int(n[condition])})
    return subject, pd.DataFrame(out)


def panel_b(ax, both: pd.DataFrame, interference: pd.DataFrame):
    subject, summary = condition_table(both)
    orders = {
        "SS": ["Absent", "Same-Frequent", "Same-Rare", "Different-Unbiased"],
        "DS": ["Absent", "Different-Frequent", "Different-Rare", "Same-Unbiased"],
    }
    labels = {
        "SS": ["Absent", "Same\nFrequent", "Same\nRare", "Different\nUnbiased"],
        "DS": ["Absent", "Different\nFrequent", "Different\nRare", "Same\nUnbiased"],
    }
    colors = {
        "SS": [COL_ABSENT, COL_SAME, COL_SAME, COL_DIFF],
        "DS": [COL_ABSENT, COL_DIFF, COL_DIFF, COL_SAME],
    }
    hatches = ["", "", "///", ""]
    alphas = [.85, .90, .60, .45]
    xs = {"SS": np.arange(4, dtype=float), "DS": np.arange(4, dtype=float) + 5.0}

    lookup = summary.set_index(["group", "condition"])
    for group in ("SS", "DS"):
        for i, condition in enumerate(orders[group]):
            row = lookup.loc[(group, condition)]
            ax.bar(xs[group][i], row["mean_ms"], yerr=row["sem_ms"], width=.62,
                   color=colors[group][i], hatch=hatches[i], alpha=alphas[i],
                   edgecolor="black", linewidth=.8, capsize=3)
        # Same-dimension effect is the pooled model contrast; point the bracket at
        # the midpoint of split Same bars (SS) or the pooled Same bar (DS).
        same_x = np.mean(xs[group][1:3]) if group == "SS" else xs[group][3]
        p = float(interference.loc[
            interference["contrast"].eq(f"{group}: Same - Absent"), "p_holm"
        ].iloc[0])
        add_bracket(ax, xs[group][0], same_x, 1123 if group == "SS" else 1142,
                    stars(p), height=7, fontsize=20)

    for group in ("SS", "DS"):
        ax.axvline((xs[group][0] + xs[group][1]) / 2,
                   color="gray", linestyle=":", lw=.8, alpha=.45)
        ax.axvline((xs[group][2] + xs[group][3]) / 2,
                   color="gray", linestyle=":", lw=.8, alpha=.45)
    ax.axvline(4.0, color="gray", linestyle="--", lw=1.0, alpha=.6)
    ax.text(xs["SS"].mean(), 1204, "SS group", ha="center",
            fontsize=20, fontweight="bold")
    ax.text(xs["DS"].mean(), 1204, "DS group", ha="center",
            fontsize=20, fontweight="bold")
    ax.set_xticks(np.r_[xs["SS"], xs["DS"]])
    ax.set_xticklabels(labels["SS"] + labels["DS"], fontsize=14.5)
    ax.set_ylim(700, 1230)
    ax.set_ylabel("RT (ms)")

    # Compact key preserves current hatch/opacity grammar without covering data.
    legend = [
        mpatches.Patch(facecolor="gray", alpha=.90, edgecolor="black",
                       label="Frequent (biased dimension)"),
        mpatches.Patch(facecolor="gray", alpha=.60, hatch="///", edgecolor="black",
                       label="Rare (biased dimension)"),
        mpatches.Patch(facecolor="gray", alpha=.45, edgecolor="black",
                       label="Unbiased dimension (pooled)"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=12.8,
              loc="lower left", bbox_to_anchor=(0.16, 1.035), ncol=3,
              borderaxespad=0, handlelength=1.4, columnspacing=1.0)
    return subject, summary


def spatial_table(both: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for group, dim in (("SS", "Same"), ("DS", "Different")):
        d = both.loc[
            both["group"].eq(group)
            & both["Dimension"].eq(dim)
            & both["DisRegion"].isin(["Frequent", "Rare"])
        ]
        means = (
            d.groupby(["subject", "DisRegion"], observed=True)["RT_ms"]
            .mean().unstack("DisRegion").dropna()
        )
        delta = means["Rare"] - means["Frequent"]
        pieces.append(pd.DataFrame({"subject": delta.index, "group": group,
                                    "delta_ms": delta.to_numpy()}))
    return pd.concat(pieces, ignore_index=True)


def panel_c(ax, both: pd.DataFrame, simple: pd.DataFrame, fixed: pd.DataFrame):
    delta = spatial_table(both)
    rng = np.random.default_rng(42)
    colors = {"SS": COL_SAME, "DS": COL_DIFF}
    xs = {"SS": 0.0, "DS": 1.0}
    summary_rows = []
    for group in ("SS", "DS"):
        values = delta.loc[delta["group"].eq(group), "delta_ms"]
        mean, sem = values.mean(), values.sem()
        summary_rows.append({"group": group, "n": len(values),
                             "mean_ms": mean, "sem_ms": sem})
        x = xs[group]
        ax.bar(x, mean, yerr=sem, width=.5, color=colors[group], alpha=.80,
               edgecolor="black", linewidth=.8, capsize=4)
        jitter = rng.uniform(-.08, .08, len(values))
        ax.scatter(x + jitter, values, s=26, color=colors[group], alpha=.70,
                   edgecolor="white", linewidth=.5, zorder=3)
        p = float(simple.loc[simple["group"].eq(group), "p_holm"].iloc[0])
        ax.text(x, mean + sem + 8, stars(p), ha="center", va="bottom", fontsize=20)

    p_inter = float(fixed.loc[fixed["term"].eq("rare_loc:ds_group"), "p"].iloc[0])
    y = max(r["mean_ms"] + r["sem_ms"] for r in summary_rows) + 30
    add_bracket(ax, 0, 1, y, stars(p_inter), height=5, fontsize=20)
    ax.axhline(0, color="black", linestyle="--", linewidth=.8, alpha=.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SS group", "DS group"], fontsize=16)
    ax.set_ylabel("RT difference: Rare − Frequent (ms)")
    low = min(-100, np.floor((delta["delta_ms"].min() - 10) / 20) * 20)
    high = max(220, np.ceil((y + 35) / 20) * 20)
    ax.set_ylim(low, high)
    return delta, pd.DataFrame(summary_rows)


def style_axis(ax, letter):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=15)
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.set_title(letter, fontweight="bold", loc="left", fontsize=25, pad=8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    training, scanner, both = combined_data()
    learning = pd.read_csv(STATS_DIR / "learning_trajectory_contrasts.csv")
    interference = pd.read_csv(STATS_DIR / "dimension_interference_contrasts.csv")
    simple = pd.read_csv(STATS_DIR / "spatial_learning_simple_effects_holm.csv")
    fixed = pd.read_csv(STATS_DIR / "spatial_learning_fixed_effects.csv")

    # Exact-stat guards prevent stale/wrong annotation tables.
    assert np.isclose(float(interference.loc[interference["contrast"].str.startswith("Group x Same"), "estimate_ms"].iloc[0]), 57.49911919420363)
    assert np.isclose(float(fixed.loc[fixed["term"].eq("rare_loc:ds_group"), "p"].iloc[0]), .007991063969649399)
    assert list(simple["n"]) == [17, 18]

    with plt.rc_context({"font.size": 16, "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 3, figsize=(25.5, 8.8),
                                 gridspec_kw={"width_ratios": [1.25, 2.0, 1.0]})
        fig.subplots_adjust(left=.055, right=.99, bottom=.18, top=.83, wspace=.31)
        trajectory = panel_a(axes[0], both)
        cond_subject, cond_summary = panel_b(axes[1], both, interference)
        deltas, delta_summary = panel_c(axes[2], both, simple, fixed)
        for ax, letter in zip(axes, "ABC"):
            style_axis(ax, letter)
        fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    trajectory.to_csv(OUT_DIR / "panel_a_blockwise_subject_mean_summary.csv", index=False)
    cond_subject.to_csv(OUT_DIR / "panel_b_subject_condition_means.csv", index=False)
    cond_summary.to_csv(OUT_DIR / "panel_b_plot_summary.csv", index=False)
    deltas.to_csv(OUT_DIR / "panel_c_subject_deltas.csv", index=False)
    delta_summary.to_csv(OUT_DIR / "panel_c_plot_summary.csv", index=False)
    learning.to_csv(OUT_DIR / "annotation_learning_trajectory.csv", index=False)
    interference.to_csv(OUT_DIR / "annotation_dimension_interference.csv", index=False)
    simple.to_csv(OUT_DIR / "annotation_spatial_simple_effects_holm.csv", index=False)
    fixed.loc[fixed["term"].eq("rare_loc:ds_group")].to_csv(
        OUT_DIR / "annotation_spatial_interaction.csv", index=False
    )

    with Image.open(OUT_PNG) as image:
        metadata = {
            "figure": "Figure 2v",
            "scope": "all available training plus scanner behavioral data",
            "union_n": int(both["subject"].nunique()),
            "ss_n": int(both.loc[both["group"].eq("SS"), "subject"].nunique()),
            "ds_n": int(both.loc[both["group"].eq("DS"), "subject"].nunique()),
            "training_n": int(training["subject"].nunique()),
            "scanner_n": int(scanner["subject"].nunique()),
            "png_pixels": list(image.size),
            "png_dpi": list(image.info.get("dpi", ())),
            "canonical_figure_overwritten": False,
            "panel_b_group_comparison_displayed": False,
            "typography_change": "+2 pt for all visible text and numeric labels",
            "annotations": {
                "group_x_same_beta_ms": 57.49911919420363,
                "group_x_same_p": .0060237389994063075,
                "group_x_location_beta_ms": -29.312708303184838,
                "group_x_location_p": .007991063969649399,
                "ss_spatial_p_holm": .020686572238260273,
                "ds_spatial_p_holm": .06862716653167415,
            },
        }
    (OUT_DIR / "figure2v_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"PNG: {OUT_PNG}\nPDF: {OUT_PDF}\nTables: {OUT_DIR}")


if __name__ == "__main__":
    main()
