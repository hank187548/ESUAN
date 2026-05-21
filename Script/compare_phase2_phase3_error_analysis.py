from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PHASE2_DIR = Path("reports/v2/error_analysis")
DEFAULT_PHASE3_DIR = Path("reports/v3/error_analysis")
DEFAULT_OUTPUT_DIR = Path("reports/v3/error_analysis")


def resolve_path(value: str | Path) -> Path:
    text = str(value).replace("\\", os.sep)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_metric_tables(
    phase2: pd.DataFrame,
    phase3: pd.DataFrame,
    key_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    merged = phase2.merge(phase3, on=key_cols, how="outer", suffixes=("_v2", "_v3"))
    for col in metric_cols:
        if f"{col}_v2" in merged.columns and f"{col}_v3" in merged.columns:
            merged[f"delta_{col}"] = merged[f"{col}_v3"] - merged[f"{col}_v2"]
    if "bias_v2" in merged.columns and "bias_v3" in merged.columns:
        merged["delta_abs_bias"] = merged["bias_v3"].abs() - merged["bias_v2"].abs()
    return merged


def top_error_summary(df: pd.DataFrame, label: str, side: str) -> dict[str, Any]:
    return {
        "phase": label,
        "side": side,
        "n": int(len(df)),
        "mean_abs_error": float(df["abs_error"].mean()),
        "median_abs_error": float(df["abs_error"].median()),
        "max_abs_error": float(df["abs_error"].max()),
        "mean_ape": float(df["ape"].mean()),
        "max_ape": float(df["ape"].max()),
        "min_error": float(df["error"].min()),
        "max_error": float(df["error"].max()),
    }


def build_top_error_comparison(phase2_dir: Path, phase3_dir: Path) -> pd.DataFrame:
    rows = []
    for side, filename in [
        ("under_prediction", "error_top_under_predictions.csv"),
        ("over_prediction", "error_top_over_predictions.csv"),
    ]:
        v2 = read_csv(phase2_dir / filename)
        v3 = read_csv(phase3_dir / filename)
        rows.append(top_error_summary(v2, "phase2", side))
        rows.append(top_error_summary(v3, "phase3", side))
    summary = pd.DataFrame(rows)
    wide = summary.pivot(index="side", columns="phase")
    comparisons = []
    for side in summary["side"].unique():
        row = {"side": side}
        for metric in ["mean_abs_error", "median_abs_error", "max_abs_error", "mean_ape", "max_ape", "min_error", "max_error"]:
            v2_value = float(wide.loc[side, (metric, "phase2")])
            v3_value = float(wide.loc[side, (metric, "phase3")])
            row[f"{metric}_v2"] = v2_value
            row[f"{metric}_v3"] = v3_value
            row[f"delta_{metric}"] = v3_value - v2_value
        comparisons.append(row)
    return pd.DataFrame(comparisons)


def format_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No data._"
    headers = df.columns.astype(str).tolist()
    rows = df.values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(format_scalar(cell) for cell in row) + " |" for row in rows],
        ]
    )


def select_existing(df: pd.DataFrame, key_col: str, values: list[str]) -> pd.DataFrame:
    return df.loc[df[key_col].isin(values)].copy()


def write_phase3_markdown_alias(phase3_dir: Path) -> Path | None:
    source = phase3_dir / "error_analysis_phase1.md"
    if not source.exists():
        return None
    target = phase3_dir / "phase3_error_analysis.md"
    text = source.read_text(encoding="utf-8")
    text = text.replace("# Phase 1C Error Analysis", "# Phase 3C Error Analysis")
    text = text.replace("- 下一階段可加入 time-aware regional market features。", "- 下一階段可做 Phase 3 error deep dive 或調整 comparable feature 設計。")
    text = text.replace("- 之後再考慮 comparable sales features。", "- 後續可比較 v3 comparable features 對各區與各價格帶的改善。")
    target.write_text(text, encoding="utf-8")
    return target


def write_comparison_report(
    output_dir: Path,
    price_cmp: pd.DataFrame,
    district_cmp: pd.DataFrame,
    quarter_cmp: pd.DataFrame,
    top_cmp: pd.DataFrame,
    phase2_summary: dict[str, Any],
    phase3_summary: dict[str, Any],
    focus_districts: list[str],
    focus_quarters: list[str],
) -> Path:
    overall_v2 = phase2_summary.get("overall", {})
    overall_v3 = phase3_summary.get("overall", {})
    overall = pd.DataFrame(
        [
            {"phase": "phase2", **overall_v2},
            {"phase": "phase3", **overall_v3},
        ]
    )
    if overall_v2 and overall_v3:
        delta = {"phase": "phase3 - phase2"}
        for metric in ["mae", "rmse", "mape", "r2", "bias"]:
            delta[metric] = overall_v3.get(metric, np.nan) - overall_v2.get(metric, np.nan)
        overall = pd.concat([overall, pd.DataFrame([delta])], ignore_index=True)

    price_cols = [
        "price_segment",
        "n_v2",
        "n_v3",
        "mae_v2",
        "mae_v3",
        "delta_mae",
        "mape_v2",
        "mape_v3",
        "delta_mape",
        "bias_v2",
        "bias_v3",
        "delta_abs_bias",
    ]
    district_focus = select_existing(district_cmp, "district", focus_districts)
    district_focus_cols = [
        "district",
        "n_v2",
        "n_v3",
        "mae_v2",
        "mae_v3",
        "delta_mae",
        "mape_v2",
        "mape_v3",
        "delta_mape",
        "bias_v2",
        "bias_v3",
        "delta_abs_bias",
    ]
    quarter_focus = select_existing(quarter_cmp, "test_quarter", focus_quarters)
    quarter_focus_cols = [
        "test_quarter",
        "n_v2",
        "n_v3",
        "mae_v2",
        "mae_v3",
        "delta_mae",
        "mape_v2",
        "mape_v3",
        "delta_mape",
        "r2_v2",
        "r2_v3",
        "delta_r2",
        "bias_v2",
        "bias_v3",
    ]

    best_district_mape = district_cmp.sort_values("delta_mape").head(5)
    worst_district_mape = district_cmp.sort_values("delta_mape", ascending=False).head(5)
    best_quarter_mape = quarter_cmp.sort_values("delta_mape").head(5)
    worst_quarter_mape = quarter_cmp.sort_values("delta_mape", ascending=False).head(5)

    content = [
        "# Phase 2 vs Phase 3 Error Analysis Comparison",
        "",
        "## Overall",
        "",
        markdown_table(overall[[col for col in ["phase", "n", "mae", "rmse", "mape", "r2", "bias"] if col in overall.columns]]),
        "",
        "## Price Segment",
        "",
        markdown_table(price_cmp[[col for col in price_cols if col in price_cmp.columns]]),
        "",
        "Interpretation: negative `delta_mae` / `delta_mape` means Phase 3 improved.",
        "",
        "## Focus Districts",
        "",
        markdown_table(district_focus[[col for col in district_focus_cols if col in district_focus.columns]]),
        "",
        "## Districts With Largest MAPE Improvement",
        "",
        markdown_table(best_district_mape[[col for col in district_focus_cols if col in best_district_mape.columns]], max_rows=5),
        "",
        "## Districts With MAPE Worsening",
        "",
        markdown_table(worst_district_mape[[col for col in district_focus_cols if col in worst_district_mape.columns]], max_rows=5),
        "",
        "## Focus Quarters",
        "",
        markdown_table(quarter_focus[[col for col in quarter_focus_cols if col in quarter_focus.columns]]),
        "",
        "## Quarters With Largest MAPE Improvement",
        "",
        markdown_table(best_quarter_mape[[col for col in quarter_focus_cols if col in best_quarter_mape.columns]], max_rows=5),
        "",
        "## Quarters With MAPE Worsening",
        "",
        markdown_table(worst_quarter_mape[[col for col in quarter_focus_cols if col in worst_quarter_mape.columns]], max_rows=5),
        "",
        "## Extreme Error Summary",
        "",
        markdown_table(top_cmp),
        "",
        "Interpretation: negative `delta_mean_abs_error`, `delta_max_abs_error`, or `delta_mean_ape` means Phase 3 reduced extreme errors in the top-50 list.",
        "",
    ]
    path = output_dir / "phase2_vs_phase3_error_comparison.md"
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Phase 2 and Phase 3 error analysis outputs.")
    parser.add_argument("--phase2-dir", default=str(DEFAULT_PHASE2_DIR))
    parser.add_argument("--phase3-dir", default=str(DEFAULT_PHASE3_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--focus-districts", default="北投區,士林區,中正區")
    parser.add_argument("--focus-quarters", default="2024Q3,2026Q1")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    phase2_dir = resolve_path(args.phase2_dir)
    phase3_dir = resolve_path(args.phase3_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase3_alias = write_phase3_markdown_alias(phase3_dir)

    price_cmp = merge_metric_tables(
        read_csv(phase2_dir / "error_by_price_segment.csv"),
        read_csv(phase3_dir / "error_by_price_segment.csv"),
        ["price_segment"],
        ["n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "bias"],
    )
    district_cmp = merge_metric_tables(
        read_csv(phase2_dir / "error_by_district.csv"),
        read_csv(phase3_dir / "error_by_district.csv"),
        ["district"],
        ["n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "bias"],
    )
    quarter_cmp = merge_metric_tables(
        read_csv(phase2_dir / "error_by_test_quarter.csv"),
        read_csv(phase3_dir / "error_by_test_quarter.csv"),
        ["test_quarter", "test_start", "test_end"],
        ["n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "r2", "bias"],
    )
    top_cmp = build_top_error_comparison(phase2_dir, phase3_dir)

    price_cmp.to_csv(output_dir / "phase2_vs_phase3_error_by_price_segment.csv", index=False, encoding="utf-8-sig")
    district_cmp.to_csv(output_dir / "phase2_vs_phase3_error_by_district.csv", index=False, encoding="utf-8-sig")
    quarter_cmp.to_csv(output_dir / "phase2_vs_phase3_error_by_test_quarter.csv", index=False, encoding="utf-8-sig")
    top_cmp.to_csv(output_dir / "phase2_vs_phase3_top_error_summary.csv", index=False, encoding="utf-8-sig")

    report_path = write_comparison_report(
        output_dir=output_dir,
        price_cmp=price_cmp,
        district_cmp=district_cmp,
        quarter_cmp=quarter_cmp,
        top_cmp=top_cmp,
        phase2_summary=read_summary(phase2_dir / "error_analysis_summary.json"),
        phase3_summary=read_summary(phase3_dir / "error_analysis_summary.json"),
        focus_districts=[item.strip() for item in args.focus_districts.split(",") if item.strip()],
        focus_quarters=[item.strip() for item in args.focus_quarters.split(",") if item.strip()],
    )
    print(f"Wrote comparison report: {report_path}")
    if phase3_alias is not None:
        print(f"Wrote Phase 3 alias report: {phase3_alias}")


if __name__ == "__main__":
    main()
