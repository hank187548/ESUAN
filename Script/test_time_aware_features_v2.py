from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from add_time_aware_features_v2 import (  # noqa: E402
    V2_FEATURES,
    add_change_features,
    add_time_aware_features,
    build_feature_config_v2,
)


def _base_df(rows):
    return pd.DataFrame(rows).assign(
        id=lambda df: [f"ID{i}" for i in range(len(df))],
        trade_date=lambda df: pd.to_datetime(df["trade_date"]),
    )


def test_historical_window_excludes_current_row():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 200},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-03", "unit_price_ping": 300},
        ]
    )

    result = add_time_aware_features(df)
    row = result.loc[result["trade_date"].eq(pd.Timestamp("2020-01-03"))].iloc[0]

    assert row["district_median_price_180d"] == 150
    assert row["district_count_180d"] == 2


def test_same_day_transactions_are_excluded_from_history():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 200},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 300},
        ]
    )

    result = add_time_aware_features(df)
    same_day = result.loc[result["trade_date"].eq(pd.Timestamp("2020-01-01"))]
    next_day = result.loc[result["trade_date"].eq(pd.Timestamp("2020-01-02"))].iloc[0]

    assert same_day["district_count_180d"].tolist() == [0, 0]
    assert same_day["district_median_price_180d"].isna().all()
    assert next_day["district_median_price_180d"] == 150


def test_district_type_features_do_not_mix_building_types():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "華廈", "trade_date": "2020-01-01", "unit_price_ping": 500},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 300},
        ]
    )

    result = add_time_aware_features(df)
    row = result.loc[result["id"].eq("ID2")].iloc[0]

    assert row["district_median_price_180d"] == 300
    assert row["district_type_median_price_180d"] == 100
    assert row["district_type_count_180d"] == 1


def test_price_change_feature():
    df = pd.DataFrame(
        {
            "district_median_price_180d": [200.0, 100.0],
            "district_median_price_365d": [150.0, np.nan],
            "district_type_median_price_180d": [120.0, 100.0],
            "district_type_median_price_365d": [100.0, np.nan],
        }
    )

    result = add_change_features(df)

    assert result.loc[0, "district_price_change_180_365"] == pytest.approx(200 / 150 - 1)
    assert result.loc[0, "district_type_price_change_180_365"] == pytest.approx(0.2)
    assert np.isnan(result.loc[1, "district_price_change_180_365"])
    assert np.isnan(result.loc[1, "district_type_price_change_180_365"])


def test_row_count_unchanged():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 200},
        ]
    )

    result = add_time_aware_features(df)

    assert len(result) == len(df)


def test_feature_config_v2_adds_features_and_excludes_leakage():
    config_v1 = {
        "target_col": "unit_price_ping",
        "numeric_features": ["building_area_m2"],
        "categorical_features": ["district"],
        "drop_cols": ["id"],
        "leakage_cols": ["unit_price_ping", "total_price", "parking_price"],
        "notes": [],
    }

    config_v2 = build_feature_config_v2(config_v1)

    for feature in V2_FEATURES:
        assert feature in config_v2["numeric_features"]
    forbidden = set(config_v2["leakage_cols"]) | {config_v2["target_col"]}
    assert not (forbidden & set(config_v2["numeric_features"]))
    assert not (forbidden & set(config_v2["categorical_features"]))
