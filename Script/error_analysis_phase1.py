from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PRED_PATH = Path("data/processed/phase1_oof_predictions.csv")
DEFAULT_METRICS_PATH = Path("reports/phase1_model_metrics.csv")
DEFAULT_METRICS_SUMMARY_PATH = Path("reports/phase1_model_metrics_summary.csv")
DEFAULT_MODEL_READY_PATH = Path("data/processed/taipei_house_model_ready.csv")
DEFAULT_FOLDS_PATH = Path("data/processed/rolling_folds.csv")
DEFAULT_REPORT_DIR = Path("reports")

PRICE_SEGMENTS = [
    ("0-50", "0–50 萬/坪", 0, 50),
    ("50-80", "50–80 萬/坪", 50, 80),
    ("80-120", "80–120 萬/坪", 80, 120),
    ("120+", "120 萬/坪以上", 120, np.inf),
]

TOP_ERROR_COLUMNS = [
    "id",
    "trade_date",
    "district",
    "building_type",
    "y_true",
    "y_pred",
    "error",
    "abs_error",
    "ape",
    "fold_id",
    "test_start",
    "test_end",
    "building_age",
    "building_area_ping",
    "floor",
    "total_floor",
    "floor_ratio",
    "rooms",
    "living_rooms",
    "bathrooms",
    "has_parking",
    "physical_condition_flag",
    "renovation_flag",
    "broad_note_flag",
    "address_raw",
    "note_raw",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(value: str | Path) -> Path:
    text = str(value).replace("\\", os.sep)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def calculate_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    n = int(len(y_true_arr))
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan, "bias": np.nan}

    error = y_pred_arr - y_true_arr
    abs_error = np.abs(error)
    squared_error = error**2
    positive_mask = y_true_arr != 0
    mape = np.mean(abs_error[positive_mask] / y_true_arr[positive_mask]) * 100 if positive_mask.any() else np.nan
    sst = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    sse = np.sum(squared_error)
    return {
        "n": n,
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mape": float(mape),
        "r2": float(1 - sse / sst) if sst > 0 else np.nan,
        "bias": float(np.mean(error)),
    }


def assign_price_segment(value: Any) -> str:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not np.isfinite(price):
        return "missing"
    for _, label, lower, upper in PRICE_SEGMENTS:
        if lower <= price < upper:
            return label
    return "missing"


def load_predictions(pred_path: str | Path, model_name: str, split: str) -> pd.DataFrame:
    path = resolve_path(pred_path)
    logging.info("Reading predictions: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"model_name", "split", "y_true", "y_pred", "id", "trade_date", "district", "building_type"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction file missing required columns: {missing}")

    filtered = df.loc[df["model_name"].eq(model_name) & df["split"].eq(split)].copy()
    if filtered.empty:
        raise ValueError(f"No predictions found for model_name={model_name}, split={split}.")
    filtered["trade_date"] = pd.to_datetime(filtered["trade_date"], errors="coerce")
    filtered["test_start"] = pd.to_datetime(filtered["test_start"], errors="coerce")
    filtered["test_end"] = pd.to_datetime(filtered["test_end"], errors="coerce")
    filtered["y_true"] = pd.to_numeric(filtered["y_true"], errors="coerce")
    filtered["y_pred"] = pd.to_numeric(filtered["y_pred"], errors="coerce")
    filtered["error"] = filtered["y_pred"] - filtered["y_true"]
    filtered["abs_error"] = filtered["error"].abs()
    filtered["ape"] = np.where(filtered["y_true"].ne(0), filtered["abs_error"] / filtered["y_true"] * 100, np.nan)
    filtered["test_quarter"] = filtered["test_start"].dt.to_period("Q").astype(str)
    logging.info("Filtered predictions rows=%s", len(filtered))
    return filtered


def load_model_ready_columns(model_ready_path: str | Path, ids: pd.Series) -> pd.DataFrame:
    path = resolve_path(model_ready_path)
    logging.info("Reading model_ready reference columns: %s", path)
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    wanted = [
        "id",
        "building_age",
        "building_area_ping",
        "floor",
        "total_floor",
        "floor_ratio",
        "rooms",
        "living_rooms",
        "bathrooms",
        "has_parking",
        "physical_condition_flag",
        "renovation_flag",
        "broad_note_flag",
        "address_raw",
        "note_raw",
    ]
    usecols = [col for col in wanted if col in header]
    ref = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    ref = ref.loc[ref["id"].isin(set(ids.astype(str)))].copy()
    ref = ref.drop_duplicates(subset=["id"], keep="last")
    return ref


def load_auxiliary_inputs(metrics_path: str | Path, metrics_summary_path: str | Path, folds_path: str | Path) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    metrics_resolved = resolve_path(metrics_path)
    summary_resolved = resolve_path(metrics_summary_path)
    folds_resolved = resolve_path(folds_path)
    if metrics_resolved.exists():
        outputs["metrics"] = pd.read_csv(metrics_resolved, encoding="utf-8-sig")
    if summary_resolved.exists():
        outputs["metrics_summary"] = pd.read_csv(summary_resolved, encoding="utf-8-sig")
    if folds_resolved.exists():
        outputs["folds"] = pd.read_csv(folds_resolved, encoding="utf-8-sig", low_memory=False)
    return outputs


def summarize_frame(df: pd.DataFrame) -> dict[str, float]:
    metrics = calculate_regression_metrics(df["y_true"], df["y_pred"])
    metrics["y_true_mean"] = float(df["y_true"].mean()) if not df.empty else np.nan
    metrics["y_pred_mean"] = float(df["y_pred"].mean()) if not df.empty else np.nan
    return metrics


def summarize_by_price_segment(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    labels = [label for _, label, _, _ in PRICE_SEGMENTS]
    work["price_segment"] = work["y_true"].map(assign_price_segment)
    work["price_segment"] = pd.Categorical(work["price_segment"], categories=labels, ordered=True)
    rows = []
    for segment, group in work.groupby("price_segment", observed=False):
        if group.empty:
            rows.append(
                {
                    "price_segment": str(segment),
                    "n": 0,
                    "y_true_mean": np.nan,
                    "y_pred_mean": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "bias": np.nan,
                }
            )
            continue
        metrics = summarize_frame(group)
        rows.append(
            {
                "price_segment": str(segment),
                "n": int(metrics["n"]),
                "y_true_mean": metrics["y_true_mean"],
                "y_pred_mean": metrics["y_pred_mean"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "bias": metrics["bias"],
            }
        )
    return pd.DataFrame(rows)


def summarize_by_district(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for district, group in df.groupby("district", dropna=False):
        metrics = summarize_frame(group)
        rows.append(
            {
                "district": district,
                "n": int(metrics["n"]),
                "y_true_mean": metrics["y_true_mean"],
                "y_pred_mean": metrics["y_pred_mean"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "bias": metrics["bias"],
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "district"], ascending=[False, True]).reset_index(drop=True)


def summarize_by_test_quarter(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["test_quarter"] = pd.Categorical(
        work["test_quarter"],
        categories=sorted(work["test_quarter"].dropna().unique()),
        ordered=True,
    )
    for quarter, group in work.groupby("test_quarter", observed=True):
        metrics = summarize_frame(group)
        rows.append(
            {
                "test_quarter": str(quarter),
                "test_start": group["test_start"].min().strftime("%Y-%m-%d"),
                "test_end": group["test_end"].max().strftime("%Y-%m-%d"),
                "n": int(metrics["n"]),
                "y_true_mean": metrics["y_true_mean"],
                "y_pred_mean": metrics["y_pred_mean"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
                "bias": metrics["bias"],
            }
        )
    return pd.DataFrame(rows).sort_values("test_quarter").reset_index(drop=True)


def extract_top_errors(df: pd.DataFrame, model_ready_ref: pd.DataFrame, n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = [
        "id",
        "trade_date",
        "district",
        "building_type",
        "y_true",
        "y_pred",
        "error",
        "abs_error",
        "ape",
        "fold_id",
        "test_start",
        "test_end",
    ]
    top_base = df[base_cols].copy()
    top_base["id"] = top_base["id"].astype(str)
    ref = model_ready_ref.copy()
    ref["id"] = ref["id"].astype(str)
    merged = top_base.merge(ref, on="id", how="left")
    for col in ["trade_date", "test_start", "test_end"]:
        merged[col] = pd.to_datetime(merged[col], errors="coerce").dt.strftime("%Y-%m-%d")
    available_cols = [col for col in TOP_ERROR_COLUMNS if col in merged.columns]
    under = merged.sort_values("error", ascending=True).head(n)[available_cols].reset_index(drop=True)
    over = merged.sort_values("error", ascending=False).head(n)[available_cols].reset_index(drop=True)
    return under, over


def write_csv_outputs(
    report_dir: Path,
    price_summary: pd.DataFrame,
    district_summary: pd.DataFrame,
    quarter_summary: pd.DataFrame,
    under: pd.DataFrame,
    over: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "price_segment": report_dir / "error_by_price_segment.csv",
        "district": report_dir / "error_by_district.csv",
        "test_quarter": report_dir / "error_by_test_quarter.csv",
        "under": report_dir / "error_top_under_predictions.csv",
        "over": report_dir / "error_top_over_predictions.csv",
    }
    price_summary.to_csv(paths["price_segment"], index=False, encoding="utf-8-sig")
    district_summary.to_csv(paths["district"], index=False, encoding="utf-8-sig")
    quarter_summary.to_csv(paths["test_quarter"], index=False, encoding="utf-8-sig")
    under.to_csv(paths["under"], index=False, encoding="utf-8-sig")
    over.to_csv(paths["over"], index=False, encoding="utf-8-sig")
    return paths


def _format_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _df_to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
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
            *["| " + " | ".join(_format_scalar(cell) for cell in row) + " |" for row in rows],
        ]
    )


def _rounded_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df[columns].copy()
    for col in output.select_dtypes(include=[np.number]).columns:
        output[col] = output[col].round(4)
    return output


def write_summary_json(
    report_dir: Path,
    overall: dict[str, float],
    price_summary: pd.DataFrame,
    district_summary: pd.DataFrame,
    quarter_summary: pd.DataFrame,
) -> Path:
    highest_mae_segment = price_summary.sort_values("mae", ascending=False).head(1)
    highest_mape_segment = price_summary.sort_values("mape", ascending=False).head(1)
    summary = {
        "overall": overall,
        "highest_mae_price_segment": highest_mae_segment.iloc[0].to_dict() if not highest_mae_segment.empty else {},
        "highest_mape_price_segment": highest_mape_segment.iloc[0].to_dict() if not highest_mape_segment.empty else {},
        "top_districts_by_mae": district_summary.head(3).to_dict(orient="records"),
        "top_districts_by_mape": district_summary.sort_values("mape", ascending=False).head(3).to_dict(orient="records"),
        "best_quarter_by_mae": quarter_summary.sort_values("mae", ascending=True).head(1).to_dict(orient="records"),
        "worst_quarter_by_mae": quarter_summary.sort_values("mae", ascending=False).head(1).to_dict(orient="records"),
    }
    path = report_dir / "error_analysis_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def write_markdown_report(
    report_dir: Path,
    pred_path: Path,
    model_name: str,
    split: str,
    df: pd.DataFrame,
    overall: dict[str, float],
    price_summary: pd.DataFrame,
    district_summary: pd.DataFrame,
    quarter_summary: pd.DataFrame,
    under: pd.DataFrame,
    over: pd.DataFrame,
    output_paths: dict[str, Path],
) -> Path:
    price_mae_max = price_summary.sort_values("mae", ascending=False).head(1).iloc[0]
    price_mape_max = price_summary.sort_values("mape", ascending=False).head(1).iloc[0]
    low_segment = price_summary.loc[price_summary["price_segment"].eq("0–50 萬/坪")]
    high_segment = price_summary.loc[price_summary["price_segment"].eq("120 萬/坪以上")]
    low_bias = float(low_segment.iloc[0]["bias"]) if not low_segment.empty else np.nan
    high_bias = float(high_segment.iloc[0]["bias"]) if not high_segment.empty else np.nan
    high_price_text = "高價區平均低估" if high_bias < 0 else "高價區平均高估" if high_bias > 0 else "高價區平均無明顯偏誤"
    low_price_text = "低價區平均低估" if low_bias < 0 else "低價區平均高估" if low_bias > 0 else "低價區平均無明顯偏誤"

    district_mape = district_summary.sort_values("mape", ascending=False).reset_index(drop=True)
    district_mae_low = district_summary.sort_values("mae", ascending=True).reset_index(drop=True)
    district_mape_low = district_summary.sort_values("mape", ascending=True).reset_index(drop=True)
    small_districts = district_summary.loc[district_summary["n"] < 100, ["district", "n"]]

    best_quarter = quarter_summary.sort_values("mae", ascending=True).head(1).iloc[0]
    worst_quarter = quarter_summary.sort_values("mae", ascending=False).head(1).iloc[0]
    q2026 = quarter_summary.loc[quarter_summary["test_quarter"].eq("2026Q1")]
    median_quarter_n = float(quarter_summary["n"].median()) if not quarter_summary.empty else np.nan
    q2026_note = ""
    if not q2026.empty:
        q2026_n = int(q2026.iloc[0]["n"])
        if q2026_n < median_quarter_n:
            q2026_note = (
                f"2026Q1 為已結束季度，但樣本數較少，n={q2026_n}，"
                "單季結果需謹慎解讀。"
            )

    top_cols = ["id", "trade_date", "district", "building_type", "y_true", "y_pred", "error", "abs_error", "ape"]
    content = [
        "# Phase 1C Error Analysis",
        "",
        "## 1. Analysis Setup",
        "",
        f"- prediction file: `{pred_path}`",
        f"- model_name: `{model_name}`",
        f"- split: `{split}`",
        f"- test prediction rows: {len(df):,}",
        "- target: `unit_price_ping`，單位為萬元/坪",
        "",
        "## 2. Overall Test Performance",
        "",
        _df_to_markdown_table(pd.DataFrame([overall])),
        "",
        "## 3. Price Segment Analysis",
        "",
        _df_to_markdown_table(_rounded_table(price_summary, ["price_segment", "n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "bias"])),
        "",
        f"- MAE 最高價格區間：{price_mae_max['price_segment']}，MAE={price_mae_max['mae']:.4f}",
        f"- MAPE 最高價格區間：{price_mape_max['price_segment']}，MAPE={price_mape_max['mape']:.4f}%",
        f"- {high_price_text}，bias={high_bias:.4f}",
        f"- {low_price_text}，bias={low_bias:.4f}",
        "",
        "## 4. District Analysis",
        "",
        "### MAE 最高前 5 行政區",
        "",
        _df_to_markdown_table(_rounded_table(district_summary.head(5), ["district", "n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "bias"])),
        "",
        "### MAPE 最高前 5 行政區",
        "",
        _df_to_markdown_table(_rounded_table(district_mape.head(5), ["district", "n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "bias"])),
        "",
        "### MAE 最低前 5 行政區",
        "",
        _df_to_markdown_table(_rounded_table(district_mae_low.head(5), ["district", "n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "bias"])),
        "",
        "### MAPE 最低前 5 行政區",
        "",
        _df_to_markdown_table(_rounded_table(district_mape_low.head(5), ["district", "n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "bias"])),
        "",
        "- 高單價行政區的 MAE 較高不一定代表模型較差，需同時看 MAPE。",
        "- n < 100 的行政區需謹慎解讀：" + (" 無。" if small_districts.empty else " " + ", ".join(f"{r.district}:{int(r.n)}" for r in small_districts.itertuples())),
        "",
        "## 5. Test Quarter Stability",
        "",
        _df_to_markdown_table(_rounded_table(quarter_summary, ["test_quarter", "test_start", "test_end", "n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "r2", "bias"])),
        "",
        f"- 最好季度（以 MAE）：{best_quarter['test_quarter']}，MAE={best_quarter['mae']:.4f}",
        f"- 最差季度（以 MAE）：{worst_quarter['test_quarter']}，MAE={worst_quarter['mae']:.4f}",
        f"- {q2026_note}" if q2026_note else "- 2026Q1 樣本數未明顯低於其他季度。",
        "",
        "## 6. Largest Errors",
        "",
        f"- under-predictions top 50: `{output_paths['under']}`",
        f"- over-predictions top 50: `{output_paths['over']}`",
        "",
        "### Top 10 Under-Predictions",
        "",
        _df_to_markdown_table(_rounded_table(under, [col for col in top_cols if col in under.columns]), max_rows=10),
        "",
        "### Top 10 Over-Predictions",
        "",
        _df_to_markdown_table(_rounded_table(over, [col for col in top_cols if col in over.columns]), max_rows=10),
        "",
        "## 7. Next Steps",
        "",
        "- 檢查高價區與高誤差行政區。",
        "- 下一階段可加入 time-aware regional market features。",
        "- 之後再考慮 comparable sales features。",
        "",
    ]
    path = report_dir / "error_analysis_phase1.md"
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1C error analysis for Phase 1 model predictions.")
    parser.add_argument("--pred-path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--metrics-summary-path", default=str(DEFAULT_METRICS_SUMMARY_PATH))
    parser.add_argument("--model-ready-path", default=str(DEFAULT_MODEL_READY_PATH))
    parser.add_argument("--folds-path", default=str(DEFAULT_FOLDS_PATH))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--model-name", default="tree_model")
    parser.add_argument("--split", default="test")
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    pred_path = resolve_path(args.pred_path)
    predictions = load_predictions(pred_path, args.model_name, args.split)
    _ = load_auxiliary_inputs(args.metrics_path, args.metrics_summary_path, args.folds_path)
    model_ready_ref = load_model_ready_columns(args.model_ready_path, predictions["id"])

    overall = calculate_regression_metrics(predictions["y_true"], predictions["y_pred"])
    price_summary = summarize_by_price_segment(predictions)
    district_summary = summarize_by_district(predictions)
    quarter_summary = summarize_by_test_quarter(predictions)
    under, over = extract_top_errors(predictions, model_ready_ref, n=50)

    output_paths = write_csv_outputs(report_dir, price_summary, district_summary, quarter_summary, under, over)
    summary_json = write_summary_json(report_dir, overall, price_summary, district_summary, quarter_summary)
    report_path = write_markdown_report(
        report_dir=report_dir,
        pred_path=pred_path,
        model_name=args.model_name,
        split=args.split,
        df=predictions,
        overall=overall,
        price_summary=price_summary,
        district_summary=district_summary,
        quarter_summary=quarter_summary,
        under=under,
        over=over,
        output_paths=output_paths,
    )

    logging.info("Wrote error analysis report: %s", report_path)
    logging.info("Wrote summary json: %s", summary_json)


if __name__ == "__main__":
    main()
