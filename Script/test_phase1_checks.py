from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_model_ready_dataset import generate_zero_variance_report, run_leakage_check  # noqa: E402
from build_time_splits import build_rolling_folds, is_incomplete_last_quarter  # noqa: E402


def test_leakage_check_flags_feature_leakage():
    config = {
        "numeric_features": ["building_area_m2", "total_price"],
        "categorical_features": ["district"],
        "drop_cols": ["id"],
        "leakage_cols": ["total_price", "unit_price_ping"],
    }

    report = run_leakage_check(config)
    status = dict(zip(report["column"], report["status"], strict=False))

    assert status["total_price"] == "FAIL"
    assert status["unit_price_ping"] == "PASS"


def test_zero_variance_detection_finds_all_zero_column():
    df = pd.DataFrame({"all_zero": [0, 0, 0], "varies": [0, 1, 0]})

    report = generate_zero_variance_report(df)

    assert "all_zero" in set(report["column"])
    assert "varies" not in set(report["column"])


def test_incomplete_last_quarter_detection():
    assert is_incomplete_last_quarter(pd.Timestamp("2026-04-18"))
    assert not is_incomplete_last_quarter(pd.Timestamp("2026-06-30"))


def test_rolling_split_uses_trade_date_and_has_non_overlapping_windows():
    dates = pd.date_range("2020-01-01", periods=28, freq="QS")
    df = pd.DataFrame(
        {
            "id": [f"ID{i:02d}" for i in range(len(dates))],
            "trade_date": dates,
            "district": ["中山區"] * len(dates),
            "source_release": ["CURRENT"] * len(dates),
        }
    )

    folds, summary, _ = build_rolling_folds(
        df,
        train_years=1,
        valid_quarters=2,
        test_quarters=1,
        step_quarters=1,
        exclude_incomplete_last_quarter=False,
    )

    assert not folds.empty
    assert set(folds["split"].unique()) == {"train", "valid", "test"}

    first = summary.iloc[0]
    assert pd.Timestamp(first["valid_start"]) > pd.Timestamp(first["train_end"])
    assert pd.Timestamp(first["test_start"]) > pd.Timestamp(first["valid_end"])

    first_fold = folds.loc[folds["fold_id"].eq(first["fold_id"])]
    train_ids = set(first_fold.loc[first_fold["split"].eq("train"), "id"])
    valid_ids = set(first_fold.loc[first_fold["split"].eq("valid"), "id"])
    test_ids = set(first_fold.loc[first_fold["split"].eq("test"), "id"])
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)

    merged = first_fold.merge(df[["id", "trade_date", "source_release"]], on="id", how="left")
    for _, row in merged.iterrows():
        trade_date = pd.Timestamp(row["trade_date"])
        start = pd.Timestamp(row[f"{row['split']}_start"])
        end = pd.Timestamp(row[f"{row['split']}_end"])
        assert start <= trade_date <= end
        assert row["source_release"] == "CURRENT"
