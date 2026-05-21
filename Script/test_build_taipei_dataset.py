from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_taipei_dataset import (  # noqa: E402
    add_categorical_boolean_features,
    add_numeric_features,
    deduplicate_records,
    parse_chinese_floor,
    parse_roc_date,
    standardize_columns,
)


def test_parse_roc_date_valid_values():
    assert parse_roc_date(1071220) == pd.Timestamp("2018-12-20")
    assert parse_roc_date("1140703") == pd.Timestamp("2025-07-03")
    assert parse_roc_date(930501) == pd.Timestamp("2004-05-01")
    assert parse_roc_date("0930501") == pd.Timestamp("2004-05-01")


def test_parse_roc_date_invalid_values():
    assert pd.isna(parse_roc_date(""))
    assert pd.isna(parse_roc_date(None))
    assert pd.isna(parse_roc_date("--"))
    assert pd.isna(parse_roc_date("abc"))
    assert pd.isna(parse_roc_date("1141332"))


def test_parse_chinese_floor():
    assert parse_chinese_floor("一層") == 1
    assert parse_chinese_floor("十層") == 10
    assert parse_chinese_floor("十一層") == 11
    assert parse_chinese_floor("二十一層") == 21
    assert parse_chinese_floor("地下一層") == -1
    assert parse_chinese_floor("一層，二層") == 1
    assert np.isnan(parse_chinese_floor("全"))


def test_deduplicate_records_keeps_larger_source_order_for_same_id():
    df = pd.DataFrame(
        {
            "id": ["A001", "A001"],
            "source_order": [1121, 1122],
            "source_release": ["112Q1", "112Q2"],
            "district": ["信義區", "信義區"],
            "address_raw": ["臺北市信義區測試路1號", "臺北市信義區測試路1號"],
            "交易年月日": ["1120101", "1120101"],
            "total_price": [1_000_000, 1_100_000],
            "building_area_m2": [30, 30],
        }
    )

    deduped = deduplicate_records(df)

    assert len(deduped) == 1
    assert deduped.loc[0, "source_release"] == "112Q2"
    assert deduped.loc[0, "total_price"] == 1_100_000


def test_has_parking_detection():
    df = pd.DataFrame(
        {
            "車位類別": ["", "", "坡道平面", ""],
            "車位移轉總面積平方公尺": ["0", "12.5", "0", "0"],
            "車位總價元": ["1500000", "0", "0", "0"],
        }
    )
    df = standardize_columns(df)
    df = add_numeric_features(df)
    df = add_categorical_boolean_features(df)

    assert df["has_parking"].tolist() == [1, 1, 1, 0]
