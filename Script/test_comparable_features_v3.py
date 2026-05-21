from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from add_comparable_features_v3 import (  # noqa: E402
    V3_FEATURES,
    build_feature_config_v3,
    compute_candidate_details,
    compute_comparable_features,
    prepare_group_arrays,
    select_top_k_candidates,
    summarize_top_candidates,
)


def _base_df(rows):
    return pd.DataFrame(rows).assign(
        id=lambda df: [f"ID{i}" for i in range(len(df))],
        trade_date=lambda df: pd.to_datetime(df["trade_date"]),
        building_area_ping=lambda df: df.get("building_area_ping", 30),
        building_age=lambda df: df.get("building_age", 20),
        floor_ratio=lambda df: df.get("floor_ratio", 0.5),
        has_parking=lambda df: df.get("has_parking", 0),
    )


def test_no_future_leakage():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 200},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-03", "unit_price_ping": 300},
        ]
    )
    group = df.assign(_row_order=np.arange(len(df)))
    arrays = prepare_group_arrays(group)

    candidates = compute_candidate_details(arrays, current_pos=1, window_days=365)

    assert candidates["candidate_id"].tolist() == ["ID0"]
    assert "ID2" not in candidates["candidate_id"].tolist()


def test_same_day_exclusion():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 200},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 300},
        ]
    )

    result = compute_comparable_features(df, top_k=10)

    assert result.loc[0, "comp_365d_count"] == 0
    assert result.loc[1, "comp_365d_count"] == 0
    assert result.loc[2, "comp_365d_count"] == 2


def test_same_district_and_building_type_only():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "B", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 500},
            {"district": "A", "building_type": "華廈", "trade_date": "2020-01-01", "unit_price_ping": 600},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 300},
        ]
    )

    result = compute_comparable_features(df, top_k=10)

    assert result.loc[3, "comp_365d_count"] == 1
    assert result.loc[3, "comp_365d_median_price"] == 100


def test_top_k_selection_uses_smallest_distance():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C"],
            "candidate_trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "candidate_distance": [0.3, 0.1, 0.2],
            "candidate_unit_price_ping": [100, 200, 300],
        }
    )

    selected = select_top_k_candidates(candidates, top_k=2)

    assert selected["candidate_id"].tolist() == ["B", "C"]


def test_weighted_mean_price():
    candidates = pd.DataFrame(
        {
            "candidate_unit_price_ping": [100.0, 200.0],
            "candidate_distance": [1.0, 3.0],
            "candidate_days_diff": [10.0, 20.0],
            "candidate_area_diff_pct": [0.1, 0.2],
            "candidate_age_diff": [5.0, 10.0],
        }
    )

    summary = summarize_top_candidates(candidates)
    expected_weights = np.array([1 / (1.0 + 1e-6), 1 / (3.0 + 1e-6)])
    expected = np.sum(expected_weights * np.array([100.0, 200.0])) / np.sum(expected_weights)

    assert summary["weighted_mean_price"] == pytest.approx(expected)


def test_no_candidates_outputs_count_zero_and_nan_features():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
        ]
    )

    result = compute_comparable_features(df, top_k=10)

    assert result.loc[0, "comp_365d_count"] == 0
    assert np.isnan(result.loc[0, "comp_365d_median_price"])
    assert np.isnan(result.loc[0, "comp_365d_mean_price"])
    assert np.isnan(result.loc[0, "comp_365d_weighted_mean_price"])
    assert np.isnan(result.loc[0, "comp_365d_nearest_price"])


def test_feature_config_v3_adds_features_and_excludes_leakage():
    config_v2 = {
        "target_col": "unit_price_ping",
        "numeric_features": ["building_area_m2"],
        "categorical_features": ["district"],
        "drop_cols": ["id", "address_raw", "note_raw"],
        "leakage_cols": ["unit_price_ping", "unit_price_m2", "total_price", "parking_price"],
        "notes": [],
    }

    config_v3 = build_feature_config_v3(config_v2)

    for feature in V3_FEATURES:
        assert feature in config_v3["numeric_features"]
    forbidden = set(config_v3["leakage_cols"]) | {config_v3["target_col"]}
    assert not (forbidden & set(config_v3["numeric_features"]))
    assert not (forbidden & set(config_v3["categorical_features"]))


def test_row_count_unchanged():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 200},
        ]
    )

    result = compute_comparable_features(df, top_k=10)

    assert len(result) == len(df)
