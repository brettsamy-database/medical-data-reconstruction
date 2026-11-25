"""Figures du README et de la documentation (assets/figures/), générées à partir des CSV de results/."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from config import settings

KIND_LABELS = {"conditions": "Conditions", "medication_requests": "Medication requests"}
PANELS = [
    ("note_vs_recon", "exact", "", "Note vs recon\nExact F1"),
    ("note_vs_recon", "semantic", "_llm", "Note vs recon\nSemantic F1"),
    ("source_vs_recon", "exact", "", "Source vs recon\nExact F1"),
    ("source_vs_recon", "semantic", "_llm", "Source vs recon\nSemantic F1"),
]


def _metric_value(df: pd.DataFrame, metric: str) -> float:
    hit = df.loc[df["metric_name"] == metric, "mean_value"]
    return float(hit.iloc[0]) if not hit.empty else 0.0


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_overall_f1(df_overall: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or settings.FIGURES_DIR / "overall_f1.png"
    x = range(len(PANELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, kind in ((-width / 2, "conditions"), (width / 2, "medication_requests")):
        vals = [_metric_value(df_overall, f"score_{fam}_{kind}_{mode}_f1{suf}") for fam, mode, suf, _ in PANELS]
        ax.bar([i + offset for i in x], vals, width, label=KIND_LABELS[kind])
    ax.set_xticks(list(x))
    ax.set_xticklabels([p[3] for p in PANELS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Overall exact and semantic F1 scores")
    ax.legend()
    return _save(fig, out_path)


def plot_style_exact_f1(df_by_style: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or settings.FIGURES_DIR / "style_exact_f1.png"
    series = [
        ("Note vs recon - conditions", "score_note_vs_recon_conditions_exact_f1"),
        ("Note vs recon - medication requests", "score_note_vs_recon_medication_requests_exact_f1"),
        ("Source vs recon - conditions", "score_source_vs_recon_conditions_exact_f1"),
        ("Source vs recon - medication requests", "score_source_vs_recon_medication_requests_exact_f1"),
    ]
    styles = sorted(df_by_style["note_style_label"].dropna().unique())
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, (label, metric) in enumerate(series):
        vals = [_metric_value(df_by_style[df_by_style["note_style_label"] == s], metric) for s in styles]
        ax.bar([i - 0.4 + width * (j + 0.5) for i in range(len(styles))], vals, width, label=label)
    ax.set_xticks(range(len(styles)))
    ax.set_xticklabels(styles)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact F1")
    ax.set_title("Exact F1 by note style")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_error_categories(df_error_analysis: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or settings.FIGURES_DIR / "error_categories.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    n_total = max(len(df_error_analysis), 1)
    for ax, kind in zip(axes, KIND_LABELS):
        pct = (df_error_analysis[f"error_type_{kind}_overall"].value_counts() / n_total * 100).sort_values()
        labels = [str(i).replace("_", " ").capitalize() for i in pct.index]
        bars = ax.barh(labels, pct.values)
        for b, v in zip(bars, pct.values):
            ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center")
        ax.set_xlim(0, max(pct.max() * 1.12, 1) if len(pct) else 1)
        ax.set_xlabel("Percentage of total patients (%)")
        ax.set_title(f"Dominant error type - {KIND_LABELS[kind]}")
    return _save(fig, out_path)


def plot_verification_by_style(df_verification: pd.DataFrame, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or settings.FIGURES_DIR / "verification_by_style.png"
    table = df_verification.groupby(["note_style_label", "verification_verdict"]).size().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(table)
    for verdict in table.columns:
        vals = table[verdict].tolist()
        ax.bar(table.index, vals, bottom=bottom, label=verdict)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("Number of patients")
    ax.set_title("Verification verdict by note style")
    ax.legend(title="Verdict")
    return _save(fig, out_path)


def make_all_figures(results_dir: Optional[Path] = None) -> List[Path]:
    results_dir = Path(results_dir or settings.RESULTS_DIR)
    summaries = results_dir / "summaries"
    return [
        plot_overall_f1(pd.read_csv(summaries / "metrics" / "df_scores_overall_summary.csv")),
        plot_style_exact_f1(pd.read_csv(summaries / "metrics" / "df_scores_summary_by_note_style.csv")),
        plot_error_categories(pd.read_csv(summaries / "errors" / "df_error_analysis.csv")),
        plot_verification_by_style(pd.read_csv(summaries / "verification" / "df_verification.csv")),
    ]
