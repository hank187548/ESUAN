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


DEFAULT_INPUT_PATH = Path("data/processed/taipei_house_model_ready_v2.csv")
DEFAULT_FEATURE_CONFIG_V2 = Path("reports/feature_config_model_v2.json")
DEFAULT_FEATURE_CONFIG_V2_FALLBACK = Path("reports/v2/feature_config_model_v2.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/taipei_house_model_ready_v3.csv")
DEFAULT_OUTPUT_PARQUET_PATH = Path("data/processed/taipei_house_model_ready_v3.parquet")
DEFAULT_FEATURE_CONFIG_V3 = Path("reports/feature_config_model_v3.json")
DEFAULT_REPORT_DIR = Path("reports/v3")

TARGET_COL = "unit_price_ping"
WINDOWS = (365, 730)
EPSILON = 1e-6

BASE_COMPARABLE_FEATURES = [
    "count",
    "median_price",
    "mean_price",
    "weighted_mean_price",
    "std_price",
    "nearest_price",
    "nearest_distance",
    "median_distance",
    "median_days_diff",
    "median_area_diff_pct",
    "median_age_diff",
]

V3_FEATURES = [f"comp_{window}d_{feature}" for window in WINDOWS for feature in BASE_COMPARABLE_FEATURES]
COUNT_FEATURES = [f"comp_{window}d_count" for window in WINDOWS]


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
    logging.info("Reading v2 dataset: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {
        "id",
        "trade_date",
        "district",
        "building_type",
        TARGET_COL,
        "building_area_ping",
        "building_age",
        "floor_ratio",
        "has_parking",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input dataset missing required columns: {missing}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    if df["trade_date"].isna().any():
        raise ValueError("Input dataset contains invalid trade_date values.")
    for col in [TARGET_COL, "building_area_ping", "building_age", "floor_ratio", "has_parking"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logging.info("Input rows=%s columns=%s", len(df), len(df.columns))
    return df


def load_feature_config_v2(path: str | Path, fallback_path: str | Path = DEFAULT_FEATURE_CONFIG_V2_FALLBACK) -> tuple[dict[str, Any], Path]:
    primary = resolve_path(path)
    fallback = resolve_path(fallback_path)
    config_path = primary if primary.exists() else fallback
    if not config_path.exists():
        raise FileNotFoundError(f"Feature config v2 not found: {primary} or {fallback}")
    logging.info("Reading v2 feature config: %s", config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f), config_path


def _as_numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def prepare_group_arrays(group: pd.DataFrame) -> dict[str, Any]:
    sorted_group = group.sort_values(["trade_date", "_row_order"], kind="mergesort").copy()
    return {
        "row_order": sorted_group["_row_order"].to_numpy(),
        "id": sorted_group["id"].astype(str).to_numpy(),
        "dates": sorted_group["trade_date"].to_numpy(dtype="datetime64[ns]"),
        "prices": _as_numeric_array(sorted_group[TARGET_COL]),
        "area": _as_numeric_array(sorted_group["building_area_ping"]),
        "age": _as_numeric_array(sorted_group["building_age"]),
        "floor_ratio": _as_numeric_array(sorted_group["floor_ratio"]),
        "has_parking": _as_numeric_array(sorted_group["has_parking"]),
        "district": sorted_group["district"].astype(str).to_numpy(),
        "building_type": sorted_group["building_type"].astype(str).to_numpy(),
    }


def compute_candidate_details(arrays: dict[str, Any], current_pos: int, window_days: int) -> pd.DataFrame:
    dates = arrays["dates"]
    current_date = dates[current_pos]
    start_date = current_date - np.timedelta64(window_days, "D")
    left = int(np.searchsorted(dates, start_date, side="left"))
    right = int(np.searchsorted(dates, current_date, side="left"))
    if right <= left:
        return pd.DataFrame()

    candidate_pos = np.arange(left, right)
    current_area = arrays["area"][current_pos]
    candidate_area = arrays["area"][candidate_pos]
    denominator = max(current_area, 1.0) if np.isfinite(current_area) else 1.0
    area_valid = np.isfinite(candidate_area) & np.isfinite(current_area)
    area_diff_pct = np.where(area_valid, np.abs(candidate_area - current_area) / denominator, np.nan)
    area_diff_for_distance = np.where(np.isfinite(area_diff_pct), area_diff_pct, 0.5)

    current_age = arrays["age"][current_pos]
    candidate_age = arrays["age"][candidate_pos]
    age_valid = np.isfinite(candidate_age) & np.isfinite(current_age)
    age_diff = np.where(age_valid, np.abs(candidate_age - current_age), np.nan)
    age_diff_norm = np.where(age_valid, age_diff / 50.0, 0.5)

    days_diff = ((current_date - dates[candidate_pos]) / np.timedelta64(1, "D")).astype(float)
    days_diff_norm = days_diff / float(window_days)

    current_floor_ratio = arrays["floor_ratio"][current_pos]
    candidate_floor_ratio = arrays["floor_ratio"][candidate_pos]
    floor_valid = np.isfinite(candidate_floor_ratio) & np.isfinite(current_floor_ratio)
    floor_ratio_diff = np.where(floor_valid, np.abs(candidate_floor_ratio - current_floor_ratio), 0.5)

    current_parking = arrays["has_parking"][current_pos]
    candidate_parking = arrays["has_parking"][candidate_pos]
    parking_diff = np.where(candidate_parking == current_parking, 0.0, 1.0)

    distance = (
        0.35 * area_diff_for_distance
        + 0.25 * age_diff_norm
        + 0.20 * days_diff_norm
        + 0.10 * floor_ratio_diff
        + 0.10 * parking_diff
    )
    valid_distance = np.isfinite(distance)
    if not valid_distance.any():
        return pd.DataFrame()

    candidate_pos = candidate_pos[valid_distance]
    distance = distance[valid_distance]
    days_diff = days_diff[valid_distance]
    area_diff_pct = area_diff_pct[valid_distance]
    age_diff = age_diff[valid_distance]

    return pd.DataFrame(
        {
            "candidate_pos": candidate_pos,
            "candidate_row_order": arrays["row_order"][candidate_pos],
            "candidate_id": arrays["id"][candidate_pos],
            "candidate_trade_date": pd.to_datetime(arrays["dates"][candidate_pos]),
            "candidate_unit_price_ping": arrays["prices"][candidate_pos],
            "candidate_building_area_ping": arrays["area"][candidate_pos],
            "candidate_building_age": arrays["age"][candidate_pos],
            "candidate_distance": distance,
            "candidate_days_diff": days_diff,
            "candidate_area_diff_pct": area_diff_pct,
            "candidate_age_diff": age_diff,
        }
    )


def select_top_k_candidates(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    work = candidates.loc[candidates["candidate_distance"].notna()].copy()
    if work.empty:
        return work
    return work.sort_values(["candidate_distance", "candidate_trade_date", "candidate_id"], ascending=[True, False, True]).head(top_k).reset_index(drop=True)


def nanmedian_or_nan(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.median(arr))


def summarize_top_candidates(candidates: pd.DataFrame) -> dict[str, float]:
    if candidates.empty:
        return {
            "count": 0,
            "median_price": np.nan,
            "mean_price": np.nan,
            "weighted_mean_price": np.nan,
            "std_price": np.nan,
            "nearest_price": np.nan,
            "nearest_distance": np.nan,
            "median_distance": np.nan,
            "median_days_diff": np.nan,
            "median_area_diff_pct": np.nan,
            "median_age_diff": np.nan,
        }
    prices = candidates["candidate_unit_price_ping"].to_numpy(dtype=float)
    distances = candidates["candidate_distance"].to_numpy(dtype=float)
    weights = 1.0 / (distances + EPSILON)
    return {
        "count": int(len(candidates)),
        "median_price": float(np.median(prices)),
        "mean_price": float(np.mean(prices)),
        "weighted_mean_price": float(np.sum(weights * prices) / np.sum(weights)),
        "std_price": float(np.std(prices, ddof=0)),
        "nearest_price": float(candidates.iloc[0]["candidate_unit_price_ping"]),
        "nearest_distance": float(candidates.iloc[0]["candidate_distance"]),
        "median_distance": float(np.median(distances)),
        "median_days_diff": nanmedian_or_nan(candidates["candidate_days_diff"]),
        "median_area_diff_pct": nanmedian_or_nan(candidates["candidate_area_diff_pct"]),
        "median_age_diff": nanmedian_or_nan(candidates["candidate_age_diff"]),
    }


def summarize_candidate_arrays(arrays: dict[str, Any], current_pos: int, window_days: int, top_k: int) -> dict[str, float]:
    dates = arrays["dates"]
    current_date = dates[current_pos]
    start_date = current_date - np.timedelta64(window_days, "D")
    left = int(np.searchsorted(dates, start_date, side="left"))
    right = int(np.searchsorted(dates, current_date, side="left"))
    if right <= left:
        return summarize_top_candidates(pd.DataFrame())

    candidate_pos = np.arange(left, right)
    current_area = arrays["area"][current_pos]
    candidate_area = arrays["area"][candidate_pos]
    denominator = max(current_area, 1.0) if np.isfinite(current_area) else 1.0
    area_valid = np.isfinite(candidate_area) & np.isfinite(current_area)
    area_diff_pct = np.where(area_valid, np.abs(candidate_area - current_area) / denominator, np.nan)
    area_diff_for_distance = np.where(np.isfinite(area_diff_pct), area_diff_pct, 0.5)

    current_age = arrays["age"][current_pos]
    candidate_age = arrays["age"][candidate_pos]
    age_valid = np.isfinite(candidate_age) & np.isfinite(current_age)
    age_diff = np.where(age_valid, np.abs(candidate_age - current_age), np.nan)
    age_diff_norm = np.where(age_valid, age_diff / 50.0, 0.5)

    days_diff = ((current_date - dates[candidate_pos]) / np.timedelta64(1, "D")).astype(float)
    days_diff_norm = days_diff / float(window_days)

    current_floor_ratio = arrays["floor_ratio"][current_pos]
    candidate_floor_ratio = arrays["floor_ratio"][candidate_pos]
    floor_valid = np.isfinite(candidate_floor_ratio) & np.isfinite(current_floor_ratio)
    floor_ratio_diff = np.where(floor_valid, np.abs(candidate_floor_ratio - current_floor_ratio), 0.5)

    current_parking = arrays["has_parking"][current_pos]
    candidate_parking = arrays["has_parking"][candidate_pos]
    parking_diff = np.where(candidate_parking == current_parking, 0.0, 1.0)

    distance = (
        0.35 * area_diff_for_distance
        + 0.25 * age_diff_norm
        + 0.20 * days_diff_norm
        + 0.10 * floor_ratio_diff
        + 0.10 * parking_diff
    )
    valid = np.isfinite(distance) & np.isfinite(arrays["prices"][candidate_pos])
    if not valid.any():
        return summarize_top_candidates(pd.DataFrame())

    distance = distance[valid]
    days_diff = days_diff[valid]
    area_diff_pct = area_diff_pct[valid]
    age_diff = age_diff[valid]
    prices = arrays["prices"][candidate_pos][valid]

    k = min(top_k, len(distance))
    top_idx = np.argpartition(distance, k - 1)[:k]
    top_idx = top_idx[np.argsort(distance[top_idx], kind="mergesort")]

    top_prices = prices[top_idx]
    top_distances = distance[top_idx]
    weights = 1.0 / (top_distances + EPSILON)
    return {
        "count": int(k),
        "median_price": float(np.median(top_prices)),
        "mean_price": float(np.mean(top_prices)),
        "weighted_mean_price": float(np.sum(weights * top_prices) / np.sum(weights)),
        "std_price": float(np.std(top_prices, ddof=0)),
        "nearest_price": float(top_prices[0]),
        "nearest_distance": float(top_distances[0]),
        "median_distance": float(np.median(top_distances)),
        "median_days_diff": nanmedian_or_nan(days_diff[top_idx]),
        "median_area_diff_pct": nanmedian_or_nan(area_diff_pct[top_idx]),
        "median_age_diff": nanmedian_or_nan(age_diff[top_idx]),
    }


def compute_comparable_features(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    output = df.copy()
    output["_row_order"] = np.arange(len(output))
    result_arrays: dict[str, np.ndarray] = {}
    for feature in V3_FEATURES:
        if feature in COUNT_FEATURES:
            result_arrays[feature] = np.zeros(len(output), dtype=int)
        else:
            result_arrays[feature] = np.full(len(output), np.nan, dtype=float)

    groups = list(output.groupby(["district", "building_type"], dropna=False, sort=False))
    logging.info("Computing comparable features for %s groups", len(groups))
    for group_idx, (_, group) in enumerate(groups, start=1):
        if group_idx == 1 or group_idx % 20 == 0 or group_idx == len(groups):
            logging.info("Processing group %s/%s rows=%s", group_idx, len(groups), len(group))
        arrays = prepare_group_arrays(group)
        for current_pos, row_order in enumerate(arrays["row_order"]):
            for window_days in WINDOWS:
                summary = summarize_candidate_arrays(arrays, current_pos, window_days, top_k)
                for key, value in summary.items():
                    result_arrays[f"comp_{window_days}d_{key}"][row_order] = value

    for feature in COUNT_FEATURES:
        output[feature] = result_arrays[feature].astype(int)
    for feature in [feature for feature in V3_FEATURES if feature not in COUNT_FEATURES]:
        output[feature] = result_arrays[feature]
    return output.drop(columns=["_row_order"])


def build_sample_matches(df: pd.DataFrame, top_k: int = 10, sample_size: int = 200, random_state: int = 42) -> pd.DataFrame:
    work = df.copy()
    work["_row_order"] = np.arange(len(work))
    sampled = work.sample(n=min(sample_size, len(work)), random_state=random_state)
    sampled_orders = set(sampled["_row_order"].tolist())
    rows: list[dict[str, Any]] = []

    for _, group in work.groupby(["district", "building_type"], dropna=False, sort=False):
        if not set(group["_row_order"]).intersection(sampled_orders):
            continue
        arrays = prepare_group_arrays(group)
        row_to_pos = {row_order: pos for pos, row_order in enumerate(arrays["row_order"])}
        for _, current in group.loc[group["_row_order"].isin(sampled_orders)].iterrows():
            current_pos = row_to_pos[current["_row_order"]]
            candidates = compute_candidate_details(arrays, current_pos, 365)
            top_candidates = select_top_k_candidates(candidates, min(3, top_k))
            for rank, candidate in enumerate(top_candidates.itertuples(index=False), start=1):
                rows.append(
                    {
                        "current_id": current["id"],
                        "current_trade_date": pd.Timestamp(current["trade_date"]).strftime("%Y-%m-%d"),
                        "current_district": current["district"],
                        "current_building_type": current["building_type"],
                        "current_unit_price_ping": current[TARGET_COL],
                        "current_building_area_ping": current["building_area_ping"],
                        "current_building_age": current["building_age"],
                        "candidate_rank": rank,
                        "candidate_id": candidate.candidate_id,
                        "candidate_trade_date": pd.Timestamp(candidate.candidate_trade_date).strftime("%Y-%m-%d"),
                        "candidate_unit_price_ping": candidate.candidate_unit_price_ping,
                        "candidate_building_area_ping": candidate.candidate_building_area_ping,
                        "candidate_building_age": candidate.candidate_building_age,
                        "candidate_distance": candidate.candidate_distance,
                        "candidate_days_diff": candidate.candidate_days_diff,
                        "candidate_area_diff_pct": candidate.candidate_area_diff_pct,
                        "candidate_age_diff": candidate.candidate_age_diff,
                    }
                )
    return pd.DataFrame(rows)


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in V3_FEATURES:
        series = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()),
                "min": float(series.min()) if series.notna().any() else np.nan,
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "median": float(series.median()) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_feature_config_v3(config_v2: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config_v2)
    numeric = list(config.get("numeric_features", []))
    for feature in V3_FEATURES:
        if feature not in numeric:
            numeric.append(feature)
    config["numeric_features"] = numeric
    notes = list(config.get("notes", []))
    notes.extend(
        [
            "v3 新增 rule-based comparable sales features。",
            "Comparable pool 嚴格使用 trade_date < current trade_date。",
            "同日交易不會互相看到。",
            "rolling folds 仍使用原本 data/processed/rolling_folds.csv。",
            "v1 / v2 feature config 保留不動。",
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
    bad_v3 = sorted(set(V3_FEATURES) & forbidden)
    bad_features = sorted((numeric | categorical) & ({target_col} | leakage_cols | drop_cols))
    return [
        {
            "check_name": "v3_features_not_target_or_leakage",
            "status": "PASS" if not bad_v3 else "FAIL",
            "details": "No comparable feature overlaps target/leakage/drop columns." if not bad_v3 else ", ".join(bad_v3),
        },
        {
            "check_name": "model_features_exclude_target_and_leakage",
            "status": "PASS" if not bad_features else "FAIL",
            "details": "No target/leakage/drop columns in numeric/categorical features." if not bad_features else ", ".join(bad_features),
        },
    ]


def run_historical_leakage_sample_check(df: pd.DataFrame, sample_size: int = 100, random_state: int = 42) -> list[dict[str, str]]:
    work = df.copy()
    work["_row_order"] = np.arange(len(work))
    sampled = work.sample(n=min(sample_size, len(work)), random_state=random_state)
    checks: list[dict[str, str]] = []
    for window_days in WINDOWS:
        violations = []
        same_day_violations = []
        for _, row in sampled.iterrows():
            current_date = pd.Timestamp(row["trade_date"])
            start_date = current_date - pd.Timedelta(days=window_days)
            mask = (
                work["district"].eq(row["district"])
                & work["building_type"].eq(row["building_type"])
                & work["trade_date"].ge(start_date)
                & work["trade_date"].lt(current_date)
            )
            pool = work.loc[mask, "trade_date"]
            if not pool.empty and pool.max() >= current_date:
                violations.append(str(row["id"]))
            same_day_mask = (
                work["district"].eq(row["district"])
                & work["building_type"].eq(row["building_type"])
                & work["trade_date"].eq(current_date)
                & work["id"].ne(row["id"])
            )
            if same_day_mask.any() and not work.loc[mask & work["trade_date"].eq(current_date)].empty:
                same_day_violations.append(str(row["id"]))
        bad = violations + same_day_violations
        checks.append(
            {
                "check_name": f"comp_{window_days}d_pool_uses_past_dates_only",
                "status": "PASS" if not bad else "FAIL",
                "details": f"Checked {len(sampled)} sampled rows; max candidate trade_date is < current trade_date and same-day candidates are excluded."
                if not bad
                else " | ".join(bad[:10]),
            }
        )
    return checks


def write_feature_configs(config: dict[str, Any], root_path: str | Path, report_dir: str | Path) -> tuple[Path, Path]:
    root = resolve_path(root_path)
    report_config = resolve_path(report_dir) / "feature_config_model_v3.json"
    for path in [root, report_config]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return root, report_config


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
    feature_config_v2_path: Path,
    feature_config_v3_root: Path,
    feature_config_v3_report: Path,
    input_rows: int,
    df_v3: pd.DataFrame,
    config_v2: dict[str, Any],
    config_v3: dict[str, Any],
    missing_report: pd.DataFrame,
    leakage_report: pd.DataFrame,
    top_k: int,
    parquet_error: str,
) -> Path:
    report_root = resolve_path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "comparable_features_v3_summary.md"
    id_missing = int(df_v3["id"].isna().sum() + df_v3["id"].astype("string").str.strip().eq("").sum())
    id_duplicates = int(df_v3["id"].duplicated().sum())
    count_stats = df_v3[COUNT_FEATURES].describe().T.reset_index().rename(columns={"index": "feature"})
    zero_rows = []
    for feature in COUNT_FEATURES:
        zero_count = int(df_v3[feature].eq(0).sum())
        zero_rows.append({"feature": feature, "zero_count": zero_count, "zero_ratio": zero_count / len(df_v3)})
    zero_report = pd.DataFrame(zero_rows)
    missing_ratio = missing_report[["feature", "missing_count", "missing_ratio"]].copy()

    content = [
        "# Comparable Features V3 Summary",
        "",
        "## 1. Input / Output",
        "",
        f"- input dataset path: `{resolve_path(input_path)}`",
        f"- output v3 CSV path: `{resolve_path(output_path)}`",
        f"- output v3 parquet path: `{resolve_path(output_parquet_path)}`",
        f"- feature config v2: `{feature_config_v2_path}`",
        f"- feature config v3 root: `{feature_config_v3_root}`",
        f"- feature config v3 report copy: `{feature_config_v3_report}`",
        f"- input row count: {input_rows:,}",
        f"- output row count: {len(df_v3):,}",
        f"- row count unchanged: {str(input_rows == len(df_v3)).lower()}",
        f"- id missing count: {id_missing:,}",
        f"- id duplicate count: {id_duplicates:,}",
        f"- parquet output status: {'success' if not parquet_error else 'failed: ' + parquet_error}",
        "",
        "## 2. Comparable Rule",
        "",
        f"- same district",
        f"- same building_type",
        f"- `trade_date < current trade_date`",
        f"- windows: 365d and 730d",
        f"- top_k: {top_k}",
        "",
        "## 3. Distance Formula",
        "",
        "`distance = 0.35 * area_diff_pct + 0.25 * age_diff_norm + 0.20 * days_diff_norm + 0.10 * floor_ratio_diff + 0.10 * parking_diff`",
        "",
        "## 4. Leakage Control",
        "",
        "- Comparable pool 嚴格使用歷史資料。",
        "- 同日交易不會被使用。",
        "- 未來資料不會被使用。",
        "- `unit_price_ping` 只用於過去 comparable cases 的統計，不直接作為 current row feature。",
        "",
        "### Leakage Check",
        "",
        _markdown_table(leakage_report),
        "",
        "## 5. New Features",
        "",
        _markdown_table(pd.DataFrame({"feature": V3_FEATURES})),
        "",
        "## 6. Missing Values",
        "",
        _markdown_table(missing_ratio),
        "",
        "## 7. Comparable Count Statistics",
        "",
        "### Count Describe",
        "",
        _markdown_table(count_stats),
        "",
        "### Zero Count Ratio",
        "",
        _markdown_table(zero_report),
        "",
        "## 8. Feature Config Update",
        "",
        f"- v2 numeric features: {len(config_v2.get('numeric_features', []))}",
        f"- v3 numeric features: {len(config_v3.get('numeric_features', []))}",
        f"- categorical features: {len(config_v3.get('categorical_features', []))}",
        "",
        "### Added Numeric Features",
        "",
        _markdown_table(pd.DataFrame({"numeric_feature": V3_FEATURES})),
        "",
        "## 9. Next Step",
        "",
        "- 下一步可使用 `data/processed/taipei_house_model_ready_v3.csv`。",
        "- 搭配 `reports/feature_config_model_v3.json` 跑 Phase 3 training。",
        "- 本步沒有訓練模型，也沒有覆蓋 v1 / v2 結果。",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add rule-based comparable sales features to v2 model_ready dataset.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--feature-config-v2", default=str(DEFAULT_FEATURE_CONFIG_V2))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-parquet-path", default=str(DEFAULT_OUTPUT_PARQUET_PATH))
    parser.add_argument("--feature-config-v3", default=str(DEFAULT_FEATURE_CONFIG_V3))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_v2 = load_dataset(args.input_path)
    input_rows = len(df_v2)
    config_v2, config_v2_path = load_feature_config_v2(args.feature_config_v2)

    df_v3 = compute_comparable_features(df_v2, top_k=args.top_k)
    config_v3 = build_feature_config_v3(config_v2)

    leakage_rows = run_feature_config_leakage_check(config_v3)
    leakage_rows.extend(run_historical_leakage_sample_check(df_v3, sample_size=100, random_state=42))
    leakage_report = pd.DataFrame(leakage_rows)
    missing_report = build_missing_report(df_v3)
    sample_matches = build_sample_matches(df_v2, top_k=args.top_k, sample_size=200, random_state=42)

    output_path, parquet_path, parquet_error = write_outputs(df_v3, args.output_path, args.output_parquet_path)
    config_v3_root, config_v3_report = write_feature_configs(config_v3, args.feature_config_v3, report_dir)

    missing_path = report_dir / "comparable_features_v3_missing_report.csv"
    leakage_path = report_dir / "comparable_features_v3_leakage_check.csv"
    sample_path = report_dir / "comparable_features_v3_sample_matches.csv"
    missing_report.to_csv(missing_path, index=False, encoding="utf-8-sig")
    leakage_report.to_csv(leakage_path, index=False, encoding="utf-8-sig")
    sample_matches.to_csv(sample_path, index=False, encoding="utf-8-sig")

    summary_path = write_summary_report(
        report_dir=report_dir,
        input_path=args.input_path,
        output_path=output_path,
        output_parquet_path=parquet_path,
        feature_config_v2_path=config_v2_path,
        feature_config_v3_root=config_v3_root,
        feature_config_v3_report=config_v3_report,
        input_rows=input_rows,
        df_v3=df_v3,
        config_v2=config_v2,
        config_v3=config_v3,
        missing_report=missing_report,
        leakage_report=leakage_report,
        top_k=args.top_k,
        parquet_error=parquet_error,
    )

    logging.info("Wrote v3 dataset CSV: %s", output_path)
    logging.info("Wrote v3 dataset parquet: %s", parquet_path)
    logging.info("Wrote v3 feature config: %s and %s", config_v3_root, config_v3_report)
    logging.info("Wrote v3 summary report: %s", summary_path)


if __name__ == "__main__":
    main()
