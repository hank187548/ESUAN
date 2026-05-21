from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/processed/taipei_house_model_ready.csv")
DEFAULT_FEATURE_CONFIG = Path("data/processed/feature_config.json")
DEFAULT_REPORT_DIR = Path("reports")

REQUIRED_LEAKAGE_COLS = [
    "total_price",
    "total_price_wan",
    "unit_price_m2",
    "unit_price_ping",
    "parking_price",
    "address_raw",
    "note_raw",
    "id",
    "transfer_id",
    "source_release",
    "source_order",
    "source_file",
    "source_folder",
]

REQUIRED_DROP_COLS = [
    "id",
    "transfer_id",
    "source_release",
    "source_order",
    "source_file",
    "source_folder",
    "trade_date",
    "address_raw",
    "note_raw",
    "total_price",
    "total_price_wan",
    "unit_price_m2",
    "unit_price_ping",
    "parking_price",
]

QUALITY_FLAGS_TO_REMOVE = [
    "abnormal_transaction_flag",
    "special_note_flag",
    "presale_note_flag",
    "separate_registration_flag",
    "layout_outlier_flag",
    "area_outlier_flag",
]

QUALITY_FLAGS_TO_KEEP = [
    "physical_condition_flag",
    "renovation_flag",
    "broad_note_flag",
    "building_age_missing",
    "main_use_missing",
    "has_parking",
    "has_management",
    "has_elevator",
    "is_basement",
    "multi_floor",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value must be true or false.")


def resolve_path(value: str | Path) -> Path:
    text = str(value).replace("\\", os.sep)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    path = resolve_path(data_path)
    logging.info("Reading model_ready dataset: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df


def load_feature_config(feature_config: str | Path) -> dict[str, Any]:
    path = resolve_path(feature_config)
    logging.info("Reading feature config: %s", path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_basic_summary(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    trade_date = df["trade_date"] if "trade_date" in df.columns else pd.Series(pd.NaT, index=df.index)
    id_series = df["id"] if "id" in df.columns else pd.Series(pd.NA, index=df.index)
    target = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series(np.nan, index=df.index)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "id_missing_count": int(id_series.isna().sum()),
        "id_duplicate_count": int(id_series.duplicated(keep=False).sum()),
        "trade_date_min": trade_date.min(),
        "trade_date_max": trade_date.max(),
        "year_counts": _value_counts_df(trade_date.dt.year if pd.api.types.is_datetime64_any_dtype(trade_date) else pd.Series(dtype=int), "trade_year"),
        "quarter_counts": _value_counts_df(trade_date.dt.to_period("Q").astype(str) if pd.api.types.is_datetime64_any_dtype(trade_date) else pd.Series(dtype=str), "trade_yq"),
        "district_counts": _value_counts_df(df.get("district", pd.Series(dtype=str)), "district"),
        "building_type_counts": _value_counts_df(df.get("building_type", pd.Series(dtype=str)), "building_type"),
        "target_summary": target.describe(),
        "target_missing_count": int(target.isna().sum()),
        "target_non_positive_count": int((target <= 0).fillna(False).sum()),
        "district_target_median": _median_by_group(df, "district", target_col),
        "building_type_target_median": _median_by_group(df, "building_type", target_col),
    }


def _value_counts_df(series: pd.Series, name: str) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=[name, "rows"])
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False).sort_index()
    return counts.rename_axis(name).reset_index(name="rows")


def _median_by_group(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    if group_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "unit_price_ping_median", "rows"])
    work = df[[group_col, target_col]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    grouped = (
        work.groupby(group_col, dropna=False)
        .agg(unit_price_ping_median=(target_col, "median"), rows=(target_col, "size"))
        .reset_index()
        .sort_values("unit_price_ping_median", ascending=False)
    )
    return grouped


def generate_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_ratio": missing_count / total if total else np.nan,
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_ratio", "column"], ascending=[False, True])


def generate_zero_variance_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        non_missing = df[col].dropna()
        unique_count = int(non_missing.nunique(dropna=True))
        if unique_count <= 1:
            single_value = "<ALL_MISSING>" if non_missing.empty else non_missing.iloc[0]
            rows.append(
                {
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isna().sum()),
                    "non_missing_count": int(non_missing.size),
                    "unique_count": unique_count,
                    "single_value": single_value,
                }
            )
    return pd.DataFrame(rows).sort_values("column") if rows else pd.DataFrame(
        columns=["column", "dtype", "missing_count", "non_missing_count", "unique_count", "single_value"]
    )


def generate_categorical_levels_report(df: pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in feature_config.get("categorical_features", []):
        if col not in df.columns:
            rows.append(
                {
                    "column": col,
                    "unique_count": 0,
                    "missing_count": total,
                    "missing_ratio": 1.0 if total else np.nan,
                    "rank": pd.NA,
                    "level": "<MISSING_COLUMN>",
                    "count": 0,
                }
            )
            continue
        missing_count = int(df[col].isna().sum())
        counts = df[col].astype("string").fillna("<NA>").value_counts(dropna=False).head(20)
        for rank, (level, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "column": col,
                    "unique_count": int(df[col].nunique(dropna=True)),
                    "missing_count": missing_count,
                    "missing_ratio": missing_count / total if total else np.nan,
                    "rank": rank,
                    "level": level,
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def run_leakage_check(feature_config: dict[str, Any]) -> pd.DataFrame:
    numeric_features = set(feature_config.get("numeric_features", []))
    categorical_features = set(feature_config.get("categorical_features", []))
    drop_cols = set(feature_config.get("drop_cols", []))
    leakage_cols = sorted(set(feature_config.get("leakage_cols", [])) | set(REQUIRED_LEAKAGE_COLS))

    rows = []
    for col in leakage_cols:
        in_numeric = col in numeric_features
        in_categorical = col in categorical_features
        rows.append(
            {
                "column": col,
                "in_numeric_features": in_numeric,
                "in_categorical_features": in_categorical,
                "in_drop_cols": col in drop_cols,
                "status": "FAIL" if in_numeric or in_categorical else "PASS",
            }
        )
    return pd.DataFrame(rows)


def build_model_v1_feature_config(
    df: pd.DataFrame,
    feature_config: dict[str, Any],
    zero_variance_report: pd.DataFrame,
) -> dict[str, Any]:
    target_col = feature_config.get("target_col", "unit_price_ping")
    zero_variance_cols = set(zero_variance_report["column"].tolist()) if not zero_variance_report.empty else set()
    leakage_cols = sorted(set(feature_config.get("leakage_cols", [])) | set(REQUIRED_LEAKAGE_COLS))
    drop_cols = sorted(set(feature_config.get("drop_cols", [])) | set(REQUIRED_DROP_COLS))
    forbidden = zero_variance_cols | set(leakage_cols) | set(drop_cols) | {target_col}

    all_zero_quality_flags = {
        col
        for col in QUALITY_FLAGS_TO_REMOVE
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").fillna(0).eq(0).all()
    }
    forbidden |= all_zero_quality_flags

    numeric_features = [
        col
        for col in feature_config.get("numeric_features", [])
        if col in df.columns and col not in forbidden
    ]
    categorical_features = [
        col
        for col in feature_config.get("categorical_features", [])
        if col in df.columns and col not in forbidden
    ]

    # Explicitly retain informative binary flags if they exist and are not zero variance.
    for col in QUALITY_FLAGS_TO_KEEP:
        if col in df.columns and col not in numeric_features and col not in forbidden:
            numeric_features.append(col)

    zero_variance_removed = sorted(
        zero_variance_cols
        & (set(feature_config.get("numeric_features", [])) | set(feature_config.get("categorical_features", [])))
    )

    return {
        "target_col": target_col,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "drop_cols": drop_cols,
        "leakage_cols": leakage_cols,
        "zero_variance_removed": zero_variance_removed,
        "notes": [
            "時間切分要用 trade_date。",
            "source_release 是發布批次，不是交易日期。",
            "unit_price_ping 是 target，不能放進 feature。",
            "total_price / unit_price_m2 / parking_price 屬於 leakage，不可放入 feature。",
        ],
    }


def write_sanity_report(
    report_dir: str | Path,
    data_path: str | Path,
    feature_config_path: str | Path,
    basic_summary: dict[str, Any],
    missing_report: pd.DataFrame,
    zero_variance_report: pd.DataFrame,
    categorical_report: pd.DataFrame,
    leakage_report: pd.DataFrame,
    model_v1_config: dict[str, Any],
) -> Path:
    report_dir = resolve_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "model_ready_sanity_check.md"

    content = [
        "# Model Ready Sanity Check",
        "",
        "## Inputs",
        "",
        f"- data-path: `{resolve_path(data_path)}`",
        f"- feature-config: `{resolve_path(feature_config_path)}`",
        "",
        "## Basic Dataset Status",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["row_count", basic_summary["row_count"]],
                ["column_count", basic_summary["column_count"]],
                ["id_missing_count", basic_summary["id_missing_count"]],
                ["id_duplicate_count", basic_summary["id_duplicate_count"]],
                ["trade_date_min", _date_str(basic_summary["trade_date_min"])],
                ["trade_date_max", _date_str(basic_summary["trade_date_max"])],
            ],
        ),
        "",
        "## Target Checks",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["target_col", "unit_price_ping"],
                ["missing_count", basic_summary["target_missing_count"]],
                ["non_positive_count", basic_summary["target_non_positive_count"]],
            ],
        ),
        "",
        "### unit_price_ping Summary",
        "",
        _series_to_markdown_table(basic_summary["target_summary"], "stat", "value"),
        "",
        "### 每年筆數",
        "",
        _df_to_markdown_table(basic_summary["year_counts"]),
        "",
        "### 每季筆數",
        "",
        _df_to_markdown_table(basic_summary["quarter_counts"]),
        "",
        "### 各行政區筆數",
        "",
        _df_to_markdown_table(basic_summary["district_counts"]),
        "",
        "### 各 building_type 筆數",
        "",
        _df_to_markdown_table(basic_summary["building_type_counts"]),
        "",
        "### 各行政區 unit_price_ping 中位數",
        "",
        _df_to_markdown_table(basic_summary["district_target_median"]),
        "",
        "### 各 building_type unit_price_ping 中位數",
        "",
        _df_to_markdown_table(basic_summary["building_type_target_median"]),
        "",
        "## Missing Values",
        "",
        f"- output: `{report_dir / 'missing_value_report.csv'}`",
        f"- columns_with_missing: {int((missing_report['missing_count'] > 0).sum())}",
        "",
        "## Zero Variance Columns",
        "",
        f"- output: `{report_dir / 'zero_variance_columns.csv'}`",
        _df_to_markdown_table(zero_variance_report),
        "",
        "## Categorical Levels",
        "",
        f"- output: `{report_dir / 'categorical_levels_report.csv'}`",
        "",
        "## Leakage Check",
        "",
        f"- output: `{report_dir / 'leakage_check_report.csv'}`",
        _df_to_markdown_table(leakage_report),
        "",
        "## feature_config_model_v1",
        "",
        f"- output: `{report_dir / 'feature_config_model_v1.json'}`",
        f"- numeric_features: {len(model_v1_config['numeric_features'])}",
        f"- categorical_features: {len(model_v1_config['categorical_features'])}",
        f"- zero_variance_removed: {', '.join(model_v1_config['zero_variance_removed']) if model_v1_config['zero_variance_removed'] else '(none)'}",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    logging.info("Wrote sanity report: %s", path)
    return path


def _date_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _format_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No data._\n"
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(_format_scalar(cell) for cell in row) + " |" for row in rows],
        ]
    )


def _df_to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No data._\n"
    view = df if max_rows is None else df.head(max_rows)
    return _markdown_table(view.columns.astype(str).tolist(), view.values.tolist())


def _series_to_markdown_table(series: pd.Series, key_name: str, value_name: str) -> str:
    rows = [[idx, value] for idx, value in series.items()]
    return _markdown_table([key_name, value_name], rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Taipei model_ready dataset before modeling.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--feature-config", default=str(DEFAULT_FEATURE_CONFIG))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()

    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_path)
    feature_config = load_feature_config(args.feature_config)
    target_col = feature_config.get("target_col", "unit_price_ping")

    basic_summary = generate_basic_summary(df, target_col)
    missing_report = generate_missing_report(df)
    zero_variance_report = generate_zero_variance_report(df)
    categorical_report = generate_categorical_levels_report(df, feature_config)
    leakage_report = run_leakage_check(feature_config)
    model_v1_config = build_model_v1_feature_config(df, feature_config, zero_variance_report)

    missing_report.to_csv(report_dir / "missing_value_report.csv", index=False, encoding="utf-8-sig")
    zero_variance_report.to_csv(report_dir / "zero_variance_columns.csv", index=False, encoding="utf-8-sig")
    categorical_report.to_csv(report_dir / "categorical_levels_report.csv", index=False, encoding="utf-8-sig")
    leakage_report.to_csv(report_dir / "leakage_check_report.csv", index=False, encoding="utf-8-sig")
    with (report_dir / "feature_config_model_v1.json").open("w", encoding="utf-8") as f:
        json.dump(model_v1_config, f, ensure_ascii=False, indent=2)
        f.write("\n")

    write_sanity_report(
        report_dir=report_dir,
        data_path=args.data_path,
        feature_config_path=args.feature_config,
        basic_summary=basic_summary,
        missing_report=missing_report,
        zero_variance_report=zero_variance_report,
        categorical_report=categorical_report,
        leakage_report=leakage_report,
        model_v1_config=model_v1_config,
    )


if __name__ == "__main__":
    main()
