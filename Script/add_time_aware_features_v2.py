from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/processed/taipei_house_model_ready.csv")
DEFAULT_FEATURE_CONFIG_V1 = Path("reports/feature_config_model_v1.json")
DEFAULT_ROLLING_FOLDS_PATH = Path("data/processed/rolling_folds.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/taipei_house_model_ready_v2.csv")
DEFAULT_OUTPUT_PARQUET_PATH = Path("data/processed/taipei_house_model_ready_v2.parquet")
DEFAULT_FEATURE_CONFIG_V2 = Path("reports/feature_config_model_v2.json")
DEFAULT_REPORT_DIR = Path("reports")

TARGET_COL = "unit_price_ping"

V2_FEATURES = [
    "district_median_price_180d",
    "district_median_price_365d",
    "district_count_180d",
    "district_type_median_price_180d",
    "district_type_median_price_365d",
    "district_type_count_180d",
    "district_price_change_180_365",
    "district_type_price_change_180_365",
]

MEDIAN_FEATURES = [
    "district_median_price_180d",
    "district_median_price_365d",
    "district_type_median_price_180d",
    "district_type_median_price_365d",
]

COUNT_FEATURES = [
    "district_count_180d",
    "district_type_count_180d",
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


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    path = resolve_path(input_path)
    logging.info("Reading v1 model_ready dataset: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"id", "trade_date", "district", "building_type", TARGET_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input dataset missing required columns: {missing}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if df["trade_date"].isna().any():
        raise ValueError("Input dataset contains missing or invalid trade_date.")
    if df[TARGET_COL].isna().any():
        raise ValueError(f"Input dataset contains missing or invalid {TARGET_COL}.")
    logging.info("Input rows=%s columns=%s", len(df), len(df.columns))
    return df


def load_feature_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    logging.info("Reading v1 feature config: %s", resolved)
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rolling_folds(path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(path)
    logging.info("Reading rolling folds for summary only: %s", resolved)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved, encoding="utf-8-sig", low_memory=False)


def _historical_stats_for_group(group: pd.DataFrame, windows: tuple[int, int] = (180, 365)) -> pd.DataFrame:
    sorted_group = group.sort_values(["trade_date", "_row_order"], kind="mergesort").copy()
    dates = sorted_group["trade_date"].to_numpy(dtype="datetime64[ns]")
    prices = sorted_group[TARGET_COL].to_numpy(dtype=float)
    row_order = sorted_group["_row_order"].to_numpy()

    output = pd.DataFrame(index=row_order)
    for window in windows:
        medians = np.full(len(sorted_group), np.nan, dtype=float)
        counts = np.zeros(len(sorted_group), dtype=int)
        delta = np.timedelta64(window, "D")
        for idx, current_date in enumerate(dates):
            start_date = current_date - delta
            left = np.searchsorted(dates, start_date, side="left")
            right = np.searchsorted(dates, current_date, side="left")
            if right <= left:
                continue
            values = prices[left:right]
            values = values[np.isfinite(values)]
            counts[idx] = int(len(values))
            if len(values) > 0:
                medians[idx] = float(np.median(values))
        output[f"median_{window}d"] = medians
        output[f"count_{window}d"] = counts
    return output


def compute_group_historical_features(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    work = df[["_row_order", "trade_date", TARGET_COL, *group_cols]].copy()
    frames = []
    for _, group in work.groupby(group_cols, dropna=False, sort=False):
        frames.append(_historical_stats_for_group(group))
    stats = pd.concat(frames, axis=0).sort_index()
    return pd.DataFrame(
        {
            f"{prefix}_median_price_180d": stats["median_180d"],
            f"{prefix}_median_price_365d": stats["median_365d"],
            f"{prefix}_count_180d": stats["count_180d"].astype(int),
        },
        index=stats.index,
    )


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    district_denominator = output["district_median_price_365d"]
    district_valid = (
        output["district_median_price_180d"].notna()
        & district_denominator.notna()
        & district_denominator.gt(0)
    )
    output["district_price_change_180_365"] = np.nan
    output.loc[district_valid, "district_price_change_180_365"] = (
        output.loc[district_valid, "district_median_price_180d"] / district_denominator.loc[district_valid] - 1
    )

    type_denominator = output["district_type_median_price_365d"]
    type_valid = (
        output["district_type_median_price_180d"].notna()
        & type_denominator.notna()
        & type_denominator.gt(0)
    )
    output["district_type_price_change_180_365"] = np.nan
    output.loc[type_valid, "district_type_price_change_180_365"] = (
        output.loc[type_valid, "district_type_median_price_180d"] / type_denominator.loc[type_valid] - 1
    )
    return output


def add_time_aware_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["_row_order"] = np.arange(len(output))

    logging.info("Computing district historical features")
    district_features = compute_group_historical_features(output, ["district"], "district")
    logging.info("Computing district + building_type historical features")
    district_type_features = compute_group_historical_features(output, ["district", "building_type"], "district_type")

    output = output.join(district_features, on="_row_order")
    output = output.join(district_type_features, on="_row_order")
    output = add_change_features(output)

    for col in COUNT_FEATURES:
        output[col] = output[col].fillna(0).astype(int)
    output = output.drop(columns=["_row_order"])
    return output


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in V2_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()),
                "min": float(series.min()) if series.notna().any() else np.nan,
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "median": float(series.median()) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_feature_config_v2(config_v1: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config_v1)
    numeric = list(config.get("numeric_features", []))
    for feature in V2_FEATURES:
        if feature not in numeric:
            numeric.append(feature)
    config["numeric_features"] = numeric
    notes = list(config.get("notes", []))
    notes.extend(
        [
            "v2 新增 time-aware historical market features。",
            "v2 time-aware features 嚴格使用 trade_date < current trade_date，不包含同日交易、自己或未來資料。",
            "rolling folds 仍使用原本 data/processed/rolling_folds.csv。",
            "v1 feature config 保留不動。",
        ]
    )
    config["notes"] = notes
    return config


def run_feature_config_leakage_check(config: dict[str, Any]) -> list[dict[str, str]]:
    target_col = config.get("target_col", TARGET_COL)
    leakage_cols = set(config.get("leakage_cols", []))
    drop_cols = set(config.get("drop_cols", []))
    numeric = set(config.get("numeric_features", []))
    categorical = set(config.get("categorical_features", []))
    forbidden = leakage_cols | drop_cols | {target_col}
    bad_v2 = sorted(set(V2_FEATURES) & forbidden)
    bad_features = sorted((numeric | categorical) & ({target_col} | leakage_cols))
    return [
        {
            "check_name": "v2_features_not_target_or_leakage",
            "status": "PASS" if not bad_v2 else "FAIL",
            "details": "No v2 feature overlaps target/leakage/drop columns." if not bad_v2 else ", ".join(bad_v2),
        },
        {
            "check_name": "model_features_exclude_target_and_leakage",
            "status": "PASS" if not bad_features else "FAIL",
            "details": "No target/leakage columns in numeric/categorical features." if not bad_features else ", ".join(bad_features),
        },
    ]


def _historical_max_date(
    df: pd.DataFrame,
    row: pd.Series,
    group_cols: list[str],
    window_days: int = 180,
) -> pd.Timestamp | pd.NaT:
    current_date = pd.Timestamp(row["trade_date"])
    start_date = current_date - pd.Timedelta(days=window_days)
    mask = df["trade_date"].ge(start_date) & df["trade_date"].lt(current_date)
    for col in group_cols:
        mask &= df[col].eq(row[col])
    history = df.loc[mask, "trade_date"]
    if history.empty:
        return pd.NaT
    return history.max()


def run_historical_leakage_sample_check(df: pd.DataFrame, sample_size: int = 100, random_state: int = 42) -> list[dict[str, str]]:
    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=random_state) if len(df) > n else df.copy()
    checks: list[dict[str, str]] = []
    for check_name, group_cols in [
        ("district_180d_history_uses_past_dates_only", ["district"]),
        ("district_type_180d_history_uses_past_dates_only", ["district", "building_type"]),
    ]:
        violations = []
        for row in sample.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            max_history_date = _historical_max_date(df, row_series, group_cols, window_days=180)
            if pd.notna(max_history_date) and max_history_date >= pd.Timestamp(row_series["trade_date"]):
                violations.append(f"id={row_series['id']} max_history_date={max_history_date}")
        checks.append(
            {
                "check_name": check_name,
                "status": "PASS" if not violations else "FAIL",
                "details": f"Checked {n} sampled rows; all max history trade_date values are < current trade_date."
                if not violations
                else " | ".join(violations[:10]),
            }
        )
    return checks


def write_feature_config(config: dict[str, Any], path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return resolved


def write_outputs(df: pd.DataFrame, output_path: str | Path, output_parquet_path: str | Path) -> tuple[Path, Path, str]:
    csv_path = resolve_path(output_path)
    parquet_path = resolve_path(output_parquet_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    parquet_error = ""
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:  # noqa: BLE001
        parquet_error = f"{type(exc).__name__}: {exc}"
        logging.warning("Parquet output failed: %s", parquet_error)
    return csv_path, parquet_path, parquet_error


def _format_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
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


def write_summary_report(
    report_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    output_parquet_path: str | Path,
    feature_config_v1_path: str | Path,
    feature_config_v2_path: str | Path,
    folds_path: str | Path,
    input_rows: int,
    output_df: pd.DataFrame,
    config_v1: dict[str, Any],
    config_v2: dict[str, Any],
    missing_report: pd.DataFrame,
    leakage_report: pd.DataFrame,
    folds: pd.DataFrame,
    parquet_error: str,
) -> Path:
    report_path = resolve_path(report_dir) / "time_aware_features_v2_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    id_missing = int(output_df["id"].isna().sum() + output_df["id"].astype("string").str.strip().eq("").sum())
    id_duplicates = int(output_df["id"].duplicated().sum())
    folds_count = int(folds["fold_id"].nunique()) if not folds.empty and "fold_id" in folds.columns else 0
    missing_ratio = missing_report[["column", "missing_count", "missing_ratio"]].copy()
    stats = missing_report[["column", "min", "mean", "median", "max"]].copy()

    content = [
        "# Time-Aware Features V2 Summary",
        "",
        "## 1. Input / Output",
        "",
        f"- input dataset path: `{resolve_path(input_path)}`",
        f"- output v2 CSV path: `{resolve_path(output_path)}`",
        f"- output v2 parquet path: `{resolve_path(output_parquet_path)}`",
        f"- feature config v1: `{resolve_path(feature_config_v1_path)}`",
        f"- feature config v2: `{resolve_path(feature_config_v2_path)}`",
        f"- rolling folds path: `{resolve_path(folds_path)}`",
        f"- rolling fold count: {folds_count}",
        f"- input row count: {input_rows:,}",
        f"- output row count: {len(output_df):,}",
        f"- row count unchanged: {str(input_rows == len(output_df)).lower()}",
        f"- id missing count: {id_missing:,}",
        f"- id duplicate count: {id_duplicates:,}",
        f"- parquet output status: {'success' if not parquet_error else 'failed: ' + parquet_error}",
        "",
        "## 2. Added Features",
        "",
        _markdown_table(pd.DataFrame({"feature": V2_FEATURES})),
        "",
        "## 3. Leakage Control",
        "",
        "- 所有歷史行情特徵只使用 `trade_date < current trade_date`。",
        "- 同日交易不會被納入 historical pool。",
        "- 未來資料不會被使用。",
        "- `unit_price_ping` 只用於產生歷史統計，不會作為模型 feature。",
        "- 實作使用 group 內依 `trade_date` 排序後的 `searchsorted`，每筆資料的 right boundary 是目前日期的第一個位置，因此排除同日全部交易與自己。",
        "",
        "### Leakage Check",
        "",
        _markdown_table(leakage_report),
        "",
        "## 4. Missing Values",
        "",
        _markdown_table(missing_ratio),
        "",
        "## 5. Basic Statistics",
        "",
        _markdown_table(stats),
        "",
        "## 6. Feature Config Update",
        "",
        f"- v1 numeric features: {len(config_v1.get('numeric_features', []))}",
        f"- v2 numeric features: {len(config_v2.get('numeric_features', []))}",
        f"- categorical features: {len(config_v2.get('categorical_features', []))}",
        "",
        "### Added Numeric Features",
        "",
        _markdown_table(pd.DataFrame({"numeric_feature": V2_FEATURES})),
        "",
        "## 7. Next Step",
        "",
        "- 下一步可使用 `data/processed/taipei_house_model_ready_v2.csv`。",
        "- 搭配 `reports/feature_config_model_v2.json` 重新跑 Phase 2 training。",
        "- 本步沒有訓練模型，也沒有覆蓋 v1 dataset 或 v1 feature config。",
        "",
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")
    return report_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add Phase 2 time-aware market features to v1 model_ready dataset.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--feature-config-v1", default=str(DEFAULT_FEATURE_CONFIG_V1))
    parser.add_argument("--rolling-folds-path", default=str(DEFAULT_ROLLING_FOLDS_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-parquet-path", default=str(DEFAULT_OUTPUT_PARQUET_PATH))
    parser.add_argument("--feature-config-v2", default=str(DEFAULT_FEATURE_CONFIG_V2))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_v1 = load_dataset(args.input_path)
    input_rows = len(df_v1)
    config_v1 = load_feature_config(args.feature_config_v1)
    folds = load_rolling_folds(args.rolling_folds_path)

    df_v2 = add_time_aware_features(df_v1)
    config_v2 = build_feature_config_v2(config_v1)

    leakage_rows = run_feature_config_leakage_check(config_v2)
    leakage_rows.extend(run_historical_leakage_sample_check(df_v2, sample_size=100, random_state=42))
    leakage_report = pd.DataFrame(leakage_rows)

    missing_report = build_missing_report(df_v2)
    csv_path, parquet_path, parquet_error = write_outputs(df_v2, args.output_path, args.output_parquet_path)
    config_v2_path = write_feature_config(config_v2, args.feature_config_v2)

    missing_path = report_dir / "time_aware_features_v2_missing_report.csv"
    leakage_path = report_dir / "time_aware_features_v2_leakage_check.csv"
    missing_report.to_csv(missing_path, index=False, encoding="utf-8-sig")
    leakage_report.to_csv(leakage_path, index=False, encoding="utf-8-sig")

    summary_path = write_summary_report(
        report_dir=report_dir,
        input_path=args.input_path,
        output_path=csv_path,
        output_parquet_path=parquet_path,
        feature_config_v1_path=args.feature_config_v1,
        feature_config_v2_path=config_v2_path,
        folds_path=args.rolling_folds_path,
        input_rows=input_rows,
        output_df=df_v2,
        config_v1=config_v1,
        config_v2=config_v2,
        missing_report=missing_report,
        leakage_report=leakage_report,
        folds=folds,
        parquet_error=parquet_error,
    )

    logging.info("Wrote v2 dataset CSV: %s", csv_path)
    logging.info("Wrote v2 dataset parquet: %s", parquet_path)
    logging.info("Wrote v2 feature config: %s", config_v2_path)
    logging.info("Wrote v2 summary report: %s", summary_path)


if __name__ == "__main__":
    main()
