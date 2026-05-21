from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/processed/taipei_house_model_ready.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/rolling_folds.csv")
DEFAULT_REPORT_DIR = Path("reports")
LOW_DISTRICT_TEST_COUNT_THRESHOLD = 30
DEFAULT_MIN_TRADE_DATE = "2019-01-01"
DEFAULT_MIN_TRAIN_SAMPLES = 1000
DEFAULT_MIN_VALID_SAMPLES = 500
DEFAULT_MIN_TEST_SAMPLES = 500


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
    if "trade_date" not in df.columns:
        raise ValueError("Dataset must contain trade_date for time-based splits.")
    if "id" not in df.columns:
        raise ValueError("Dataset must contain id for rolling_folds output.")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.loc[df["trade_date"].notna()].copy()
    return df


def get_quarter_start(date: Any) -> pd.Timestamp:
    return pd.Timestamp(date).to_period("Q").start_time.normalize()


def get_quarter_end(date: Any) -> pd.Timestamp:
    return pd.Timestamp(date).to_period("Q").end_time.normalize()


def is_incomplete_last_quarter(max_trade_date: Any) -> bool:
    max_date = pd.Timestamp(max_trade_date).normalize()
    return max_date < get_quarter_end(max_date)


def get_available_quarters(
    df: pd.DataFrame,
    min_trade_date: str | pd.Timestamp = DEFAULT_MIN_TRADE_DATE,
    exclude_incomplete_last_quarter: bool = True,
) -> tuple[pd.PeriodIndex, dict[str, Any]]:
    original_min_date = df["trade_date"].min().normalize()
    original_max_date = df["trade_date"].max().normalize()
    min_trade_ts = pd.Timestamp(min_trade_date).normalize()

    before_min = df.loc[df["trade_date"].lt(min_trade_ts)].copy()
    filtered = df.loc[df["trade_date"].ge(min_trade_ts)].copy()

    if filtered.empty:
        return pd.PeriodIndex([], freq="Q"), {
            "trade_date_min": original_min_date,
            "trade_date_max": original_max_date,
            "model_ready_trade_date_min": original_min_date,
            "model_ready_trade_date_max": original_max_date,
            "min_trade_date": min_trade_ts,
            "excluded_before_min_trade_date_rows": int(len(before_min)),
            "excluded_before_min_by_year": _period_count_dict(before_min, "Y"),
            "excluded_before_min_by_quarter": _period_count_dict(before_min, "Q"),
            "exclude_incomplete_last_quarter": exclude_incomplete_last_quarter,
            "last_quarter_incomplete": False,
            "excluded_last_quarter": False,
            "excluded_incomplete_quarter": "",
            "usable_start_date": pd.NaT,
            "usable_end_date": pd.NaT,
            "rolling_trade_date_min": pd.NaT,
            "rolling_trade_date_max": pd.NaT,
        }

    min_date = filtered["trade_date"].min().normalize()
    max_date = filtered["trade_date"].max().normalize()
    incomplete = is_incomplete_last_quarter(max_date)

    usable_max_date = max_date
    excluded_last_quarter = False
    excluded_incomplete_quarter = ""
    if exclude_incomplete_last_quarter and incomplete:
        usable_max_date = get_quarter_start(max_date) - pd.Timedelta(days=1)
        excluded_last_quarter = True
        excluded_incomplete_quarter = str(max_date.to_period("Q"))

    if usable_max_date < min_date:
        return pd.PeriodIndex([], freq="Q"), {
            "trade_date_min": original_min_date,
            "trade_date_max": original_max_date,
            "model_ready_trade_date_min": original_min_date,
            "model_ready_trade_date_max": original_max_date,
            "min_trade_date": min_trade_ts,
            "excluded_before_min_trade_date_rows": int(len(before_min)),
            "excluded_before_min_by_year": _period_count_dict(before_min, "Y"),
            "excluded_before_min_by_quarter": _period_count_dict(before_min, "Q"),
            "exclude_incomplete_last_quarter": exclude_incomplete_last_quarter,
            "last_quarter_incomplete": incomplete,
            "excluded_last_quarter": excluded_last_quarter,
            "excluded_incomplete_quarter": excluded_incomplete_quarter,
            "usable_start_date": pd.NaT,
            "usable_end_date": pd.NaT,
            "rolling_trade_date_min": pd.NaT,
            "rolling_trade_date_max": pd.NaT,
        }

    quarters = pd.period_range(min_date.to_period("Q"), usable_max_date.to_period("Q"), freq="Q")
    rolling_filtered = filtered.loc[filtered["trade_date"].le(usable_max_date)]
    return quarters, {
        "trade_date_min": original_min_date,
        "trade_date_max": original_max_date,
        "model_ready_trade_date_min": original_min_date,
        "model_ready_trade_date_max": original_max_date,
        "min_trade_date": min_trade_ts,
        "excluded_before_min_trade_date_rows": int(len(before_min)),
        "excluded_before_min_by_year": _period_count_dict(before_min, "Y"),
        "excluded_before_min_by_quarter": _period_count_dict(before_min, "Q"),
        "exclude_incomplete_last_quarter": exclude_incomplete_last_quarter,
        "last_quarter_incomplete": incomplete,
        "excluded_last_quarter": excluded_last_quarter,
        "excluded_incomplete_quarter": excluded_incomplete_quarter,
        "usable_start_date": quarters[0].start_time.normalize(),
        "usable_end_date": quarters[-1].end_time.normalize(),
        "rolling_trade_date_min": rolling_filtered["trade_date"].min().normalize() if not rolling_filtered.empty else pd.NaT,
        "rolling_trade_date_max": rolling_filtered["trade_date"].max().normalize() if not rolling_filtered.empty else pd.NaT,
    }


def _period_count_dict(df: pd.DataFrame, freq: str) -> dict[str, int]:
    if df.empty or "trade_date" not in df.columns:
        return {}
    periods = df["trade_date"].dt.to_period(freq).astype(str)
    return {str(k): int(v) for k, v in periods.value_counts().sort_index().items()}


def build_rolling_folds(
    df: pd.DataFrame,
    train_years: int = 3,
    valid_quarters: int = 2,
    test_quarters: int = 1,
    step_quarters: int = 1,
    exclude_incomplete_last_quarter: bool = True,
    min_trade_date: str | pd.Timestamp = DEFAULT_MIN_TRADE_DATE,
    min_train_samples: int = DEFAULT_MIN_TRAIN_SAMPLES,
    min_valid_samples: int = DEFAULT_MIN_VALID_SAMPLES,
    min_test_samples: int = DEFAULT_MIN_TEST_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if train_years <= 0 or valid_quarters <= 0 or test_quarters <= 0 or step_quarters <= 0:
        raise ValueError("train_years, valid_quarters, test_quarters, and step_quarters must be positive.")
    if min_train_samples < 0 or min_valid_samples < 0 or min_test_samples < 0:
        raise ValueError("Minimum sample thresholds must be non-negative.")

    min_trade_ts = pd.Timestamp(min_trade_date).normalize()
    quarters, metadata = get_available_quarters(df, min_trade_ts, exclude_incomplete_last_quarter)
    train_quarters = train_years * 4
    fold_width = train_quarters + valid_quarters + test_quarters
    rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    excluded_candidate_rows: list[dict[str, Any]] = []
    metadata.update(
        {
            "train_years": int(train_years),
            "valid_quarters": int(valid_quarters),
            "test_quarters": int(test_quarters),
            "step_quarters": int(step_quarters),
            "min_train_samples": int(min_train_samples),
            "min_valid_samples": int(min_valid_samples),
            "min_test_samples": int(min_test_samples),
        }
    )

    if len(quarters) < fold_width:
        return pd.DataFrame(
            columns=["fold_id", "id", "split", "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]
        ), pd.DataFrame(), metadata | {"fold_count": 0, "excluded_candidate_folds": []}

    usable_df = df.loc[
        df["trade_date"].between(max(metadata["usable_start_date"], min_trade_ts), metadata["usable_end_date"], inclusive="both")
    ].copy()

    fold_id = 1
    candidate_id = 1
    for start_idx in range(0, len(quarters) - fold_width + 1, step_quarters):
        train_start_q = quarters[start_idx]
        train_end_q = quarters[start_idx + train_quarters - 1]
        valid_start_q = quarters[start_idx + train_quarters]
        valid_end_q = quarters[start_idx + train_quarters + valid_quarters - 1]
        test_start_q = quarters[start_idx + train_quarters + valid_quarters]
        test_end_q = quarters[start_idx + fold_width - 1]

        bounds = {
            "train_start": train_start_q.start_time.normalize(),
            "train_end": train_end_q.end_time.normalize(),
            "valid_start": valid_start_q.start_time.normalize(),
            "valid_end": valid_end_q.end_time.normalize(),
            "test_start": test_start_q.start_time.normalize(),
            "test_end": test_end_q.end_time.normalize(),
        }

        split_counts: dict[str, int] = {}
        for split, start_col, end_col in [
            ("train", "train_start", "train_end"),
            ("valid", "valid_start", "valid_end"),
            ("test", "test_start", "test_end"),
        ]:
            split_df = usable_df.loc[
                usable_df["trade_date"].between(bounds[start_col], bounds[end_col], inclusive="both"),
                ["id"],
            ].copy()
            split_counts[split] = int(len(split_df))
            if split_df.empty:
                continue

        exclusion_reasons = []
        if split_counts.get("train", 0) < min_train_samples:
            exclusion_reasons.append(f"train_rows < {min_train_samples}")
        if split_counts.get("valid", 0) < min_valid_samples:
            exclusion_reasons.append(f"valid_rows < {min_valid_samples}")
        if split_counts.get("test", 0) < min_test_samples:
            exclusion_reasons.append(f"test_rows < {min_test_samples}")

        test_df = usable_df.loc[
            usable_df["trade_date"].between(bounds["test_start"], bounds["test_end"], inclusive="both")
        ].copy()
        district_counts = (
            test_df["district"].astype("string").fillna("<NA>").value_counts().sort_index()
            if "district" in test_df.columns
            else pd.Series(dtype=int)
        )
        low_districts = district_counts[district_counts < LOW_DISTRICT_TEST_COUNT_THRESHOLD]

        summary_row = (
            {
                "candidate_id": candidate_id,
                "fold_id": fold_id,
                **{col: value.strftime("%Y-%m-%d") for col, value in bounds.items()},
                "train_rows": split_counts.get("train", 0),
                "valid_rows": split_counts.get("valid", 0),
                "test_rows": split_counts.get("test", 0),
                "test_quarter": str(test_start_q) if test_quarters == 1 else f"{test_start_q}-{test_end_q}",
                "test_district_min_count": int(district_counts.min()) if not district_counts.empty else 0,
                "test_districts_below_30": ", ".join(f"{idx}:{int(val)}" for idx, val in low_districts.items()),
            }
        )
        if exclusion_reasons:
            excluded_candidate_rows.append(
                {
                    **summary_row,
                    "fold_id": "",
                    "exclusion_reason": "; ".join(exclusion_reasons),
                }
            )
            candidate_id += 1
            continue

        for split, start_col, end_col in [
            ("train", "train_start", "train_end"),
            ("valid", "valid_start", "valid_end"),
            ("test", "test_start", "test_end"),
        ]:
            split_df = usable_df.loc[
                usable_df["trade_date"].between(bounds[start_col], bounds[end_col], inclusive="both"),
                ["id"],
            ].copy()
            if split_df.empty:
                continue
            split_df.insert(0, "fold_id", fold_id)
            split_df["split"] = split
            for col, value in bounds.items():
                split_df[col] = value.strftime("%Y-%m-%d")
            rows.append(split_df)

        summary_rows.append(summary_row)
        fold_id += 1
        candidate_id += 1

    folds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["fold_id", "id", "split", "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]
    )
    summary = pd.DataFrame(summary_rows)
    metadata["fold_count"] = int(len(summary))
    metadata["excluded_candidate_folds"] = excluded_candidate_rows
    return folds, summary, metadata


def write_rolling_summary(
    report_dir: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    metadata: dict[str, Any],
    fold_summary: pd.DataFrame,
    train_years: int,
    valid_quarters: int,
    test_quarters: int,
    step_quarters: int,
    min_trade_date: str | pd.Timestamp,
    min_train_samples: int,
    min_valid_samples: int,
    min_test_samples: int,
) -> Path:
    report_dir = resolve_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "rolling_folds_summary.md"

    content = [
        "# Rolling Folds Summary",
        "",
        "## Settings",
        "",
        f"- data-path: `{resolve_path(data_path)}`",
        f"- output-path: `{resolve_path(output_path)}`",
        f"- min_trade_date: `{pd.Timestamp(min_trade_date).strftime('%Y-%m-%d')}`",
        f"- train_years: {train_years}",
        f"- valid_quarters: {valid_quarters}",
        f"- test_quarters: {test_quarters}",
        f"- step_quarters: {step_quarters}",
        f"- min_train_samples: {min_train_samples}",
        f"- min_valid_samples: {min_valid_samples}",
        f"- min_test_samples: {min_test_samples}",
        f"- exclude_incomplete_last_quarter: {str(metadata.get('exclude_incomplete_last_quarter')).lower()}",
        "",
        "## Date Range",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["model_ready_trade_date_min", _date_str(metadata.get("model_ready_trade_date_min"))],
                ["model_ready_trade_date_max", _date_str(metadata.get("model_ready_trade_date_max"))],
                ["rolling_min_trade_date", _date_str(metadata.get("min_trade_date"))],
                ["rolling_used_trade_date_min", _date_str(metadata.get("rolling_trade_date_min"))],
                ["rolling_used_trade_date_max", _date_str(metadata.get("rolling_trade_date_max"))],
                ["rows_excluded_before_min_trade_date", metadata.get("excluded_before_min_trade_date_rows", 0)],
                ["exclude_incomplete_last_quarter", str(metadata.get("exclude_incomplete_last_quarter")).lower()],
                ["last_quarter_incomplete", str(metadata.get("last_quarter_incomplete")).lower()],
                ["excluded_last_quarter", str(metadata.get("excluded_last_quarter")).lower()],
                ["excluded_incomplete_quarter", metadata.get("excluded_incomplete_quarter", "")],
                ["split_usable_start_date", _date_str(metadata.get("usable_start_date"))],
                ["split_usable_end_date", _date_str(metadata.get("usable_end_date"))],
                ["fold_count", metadata.get("fold_count", 0)],
            ],
        ),
        "",
        "## Rows Excluded Before min_trade_date",
        "",
        "### By Year",
        "",
        _dict_to_markdown_table(metadata.get("excluded_before_min_by_year", {}), "year", "rows"),
        "",
        "### By Quarter",
        "",
        _dict_to_markdown_table(metadata.get("excluded_before_min_by_quarter", {}), "quarter", "rows"),
        "",
        "## Fold Details",
        "",
        _df_to_markdown_table(fold_summary),
        "",
        "## Excluded Candidate Folds",
        "",
        _df_to_markdown_table(pd.DataFrame(metadata.get("excluded_candidate_folds", []))),
        "",
        f"District test count warning threshold: < {LOW_DISTRICT_TEST_COUNT_THRESHOLD} rows.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    logging.info("Wrote rolling summary: %s", path)
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


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._\n"
    return _markdown_table(df.columns.astype(str).tolist(), df.values.tolist())


def _dict_to_markdown_table(values: dict[str, int], key_name: str, value_name: str) -> str:
    if not values:
        return "_No data._\n"
    return _markdown_table([key_name, value_name], [[k, v] for k, v in values.items()])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build rolling time-based folds from model_ready dataset.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--valid-quarters", type=int, default=2)
    parser.add_argument("--test-quarters", type=int, default=1)
    parser.add_argument("--step-quarters", type=int, default=1)
    parser.add_argument("--exclude-incomplete-last-quarter", type=str_to_bool, default=True)
    parser.add_argument("--min-trade-date", default=DEFAULT_MIN_TRADE_DATE)
    parser.add_argument("--min-train-samples", type=int, default=DEFAULT_MIN_TRAIN_SAMPLES)
    parser.add_argument("--min-valid-samples", type=int, default=DEFAULT_MIN_VALID_SAMPLES)
    parser.add_argument("--min-test-samples", type=int, default=DEFAULT_MIN_TEST_SAMPLES)
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()

    df = load_dataset(args.data_path)
    folds, fold_summary, metadata = build_rolling_folds(
        df,
        train_years=args.train_years,
        valid_quarters=args.valid_quarters,
        test_quarters=args.test_quarters,
        step_quarters=args.step_quarters,
        exclude_incomplete_last_quarter=args.exclude_incomplete_last_quarter,
        min_trade_date=args.min_trade_date,
        min_train_samples=args.min_train_samples,
        min_valid_samples=args.min_valid_samples,
        min_test_samples=args.min_test_samples,
    )

    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info("Wrote rolling folds: %s rows=%s", output_path, len(folds))

    write_rolling_summary(
        report_dir=args.report_dir,
        data_path=args.data_path,
        output_path=output_path,
        metadata=metadata,
        fold_summary=fold_summary,
        train_years=args.train_years,
        valid_quarters=args.valid_quarters,
        test_quarters=args.test_quarters,
        step_quarters=args.step_quarters,
        min_trade_date=args.min_trade_date,
        min_train_samples=args.min_train_samples,
        min_valid_samples=args.min_valid_samples,
        min_test_samples=args.min_test_samples,
    )


if __name__ == "__main__":
    main()
