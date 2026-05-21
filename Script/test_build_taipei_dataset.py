from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_taipei_dataset import (  # noqa: E402
    add_area_outlier_flag,
    add_building_age_missing_flag,
    add_categorical_boolean_features,
    add_layout_outlier_flag,
    add_numeric_features,
    add_presale_note_flag,
    add_separate_registration_flag,
    add_special_note_flag,
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


def test_presale_note_flag():
    df = pd.DataFrame(
        {
            "備註": [
                "預售屋、或土地及建物分件登記案件",
                "本案為預售屋買賣",
                "一般交易",
            ]
        }
    )
    df = standardize_columns(df)
    df = add_presale_note_flag(df)

    assert df["presale_note_flag"].tolist() == [1, 1, 0]


def test_separate_registration_flag():
    df = pd.DataFrame({"備註": ["土地及建物分件登記案件", "一般交易"]})
    df = standardize_columns(df)
    df = add_separate_registration_flag(df)

    assert df["separate_registration_flag"].tolist() == [1, 0]


def test_building_age_missing_flag():
    df = pd.DataFrame({"building_age": [np.nan, 12.5]})
    df = add_building_age_missing_flag(df)

    assert df["building_age_missing"].tolist() == [1, 0]


def test_area_outlier_flag():
    df = pd.DataFrame({"building_area_m2": [5, 800, 80, np.nan]})
    df = add_area_outlier_flag(df)

    assert df["area_outlier_flag"].tolist() == [1, 1, 0, 0]


def test_special_note_flag():
    df = pd.DataFrame(
        {
            "note_raw": [
                "親友交易",
                "法院拍賣案件",
                "含裝潢",
                "包含其他約定事項",
                "增建未登記",
                "一般交易",
            ]
        }
    )
    df = add_special_note_flag(df)

    assert df["abnormal_transaction_flag"].tolist() == [1, 1, 0, 0, 0, 0]
    assert df["physical_condition_flag"].tolist() == [0, 0, 0, 0, 1, 0]
    assert df["renovation_flag"].tolist() == [0, 0, 1, 0, 0, 0]
    assert df["broad_note_flag"].tolist() == [0, 0, 0, 1, 0, 0]
    assert df["special_note_flag"].tolist() == [1, 1, 0, 0, 0, 0]


def test_layout_outlier_flag():
    df = pd.DataFrame(
        {
            "rooms": [33, 3, 3, 3],
            "living_rooms": [2, 22, 2, 2],
            "bathrooms": [2, 2, 22, 2],
        }
    )
    df = add_layout_outlier_flag(df)

    assert df["layout_outlier_flag"].tolist() == [1, 1, 1, 0]
