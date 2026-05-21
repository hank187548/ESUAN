from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PING_PER_M2 = 1 / 3.305785
CURRENT_SOURCE_ORDER = 999_999

EXPECTED_COLUMNS = [
    "鄉鎮市區",
    "交易標的",
    "土地位置建物門牌",
    "土地移轉總面積平方公尺",
    "都市土地使用分區",
    "非都市土地使用分區",
    "非都市土地使用編定",
    "交易年月日",
    "交易筆棟數",
    "移轉層次",
    "總樓層數",
    "建物型態",
    "主要用途",
    "主要建材",
    "建築完成年月",
    "建物移轉總面積平方公尺",
    "建物現況格局-房",
    "建物現況格局-廳",
    "建物現況格局-衛",
    "建物現況格局-隔間",
    "有無管理組織",
    "總價元",
    "單價元平方公尺",
    "車位類別",
    "車位移轉總面積平方公尺",
    "車位總價元",
    "備註",
    "編號",
    "主建物面積",
    "附屬建物面積",
    "陽台面積",
    "電梯",
    "移轉編號",
]

NUMERIC_SOURCE_COLUMNS = [
    "土地移轉總面積平方公尺",
    "建物移轉總面積平方公尺",
    "主建物面積",
    "附屬建物面積",
    "陽台面積",
    "建物現況格局-房",
    "建物現況格局-廳",
    "建物現況格局-衛",
    "總價元",
    "單價元平方公尺",
    "車位移轉總面積平方公尺",
    "車位總價元",
]

RESIDENTIAL_BUILDING_TYPES = ["住宅大樓", "華廈", "公寓", "套房", "透天厝"]
MODEL_BUILDING_TYPES = ["住宅大樓", "華廈", "公寓", "套房"]
NON_RESIDENTIAL_USE_KEYWORDS = ["商業用", "工業用", "辦公室", "店鋪", "廠房"]

STRICT_SPECIAL_NOTE_KEYWORDS = [
    "親友",
    "關係人",
    "特殊",
    "急買急賣",
    "拍賣",
    "瑕疵",
    "債權",
    "共有",
    "增建",
    "未登記",
    "包含",
    "其他",
    "畸零",
    "協議",
    "讓與",
    "持分",
    "毛胚",
    "裝潢",
    "夾層",
    "加蓋",
]

ABNORMAL_TRANSACTION_KEYWORDS = [
    "親友",
    "關係人",
    "急買急賣",
    "拍賣",
    "債權",
    "協議",
    "讓與",
    "瑕疵",
    "畸零",
    "持分",
]

PHYSICAL_CONDITION_KEYWORDS = [
    "增建",
    "未登記",
    "加蓋",
    "夾層",
    "毛胚",
]

RENOVATION_KEYWORDS = [
    "裝潢",
]

BROAD_NOTE_KEYWORDS = [
    "包含",
    "其他",
]

PRESALE_NOTE_KEYWORDS = [
    "預售屋買賣",
    "預售屋",
    "預售",
]

SEPARATE_REGISTRATION_KEYWORDS = [
    "土地及建物分件登記",
    "分件登記",
    "分件",
]

MODEL_READY_COLUMNS = [
    "id",
    "transfer_id",
    "source_release",
    "source_order",
    "source_file",
    "source_folder",
    "trade_date",
    "address_raw",
    "note_raw",
    "unit_price_ping",
    "unit_price_m2",
    "district",
    "trade_year",
    "trade_month",
    "trade_quarter",
    "trade_ym",
    "trade_yq",
    "building_type",
    "main_use",
    "main_use_missing",
    "material",
    "building_age",
    "building_age_missing",
    "building_area_m2",
    "building_area_ping",
    "main_building_area_m2",
    "main_building_area_ping",
    "auxiliary_area_m2",
    "balcony_area_m2",
    "land_area_m2",
    "floor",
    "total_floor",
    "floor_ratio",
    "is_basement",
    "multi_floor",
    "rooms",
    "living_rooms",
    "bathrooms",
    "has_management",
    "has_elevator",
    "has_parking",
    "parking_area_m2",
    "area_outlier_flag",
    "layout_outlier_flag",
    "abnormal_transaction_flag",
    "physical_condition_flag",
    "renovation_flag",
    "broad_note_flag",
    "special_note_flag",
    "presale_note_flag",
    "separate_registration_flag",
    "total_price",
    "total_price_wan",
    "parking_price",
]

FEATURE_CONFIG = {
    "target_col": "unit_price_ping",
    "categorical_features": [
        "district",
        "trade_month",
        "trade_quarter",
        "trade_yq",
        "building_type",
        "main_use",
        "material",
    ],
    "numeric_features": [
        "trade_year",
        "main_use_missing",
        "building_age",
        "building_age_missing",
        "building_area_m2",
        "building_area_ping",
        "main_building_area_m2",
        "main_building_area_ping",
        "auxiliary_area_m2",
        "balcony_area_m2",
        "land_area_m2",
        "floor",
        "total_floor",
        "floor_ratio",
        "is_basement",
        "multi_floor",
        "rooms",
        "living_rooms",
        "bathrooms",
        "has_management",
        "has_elevator",
        "has_parking",
        "parking_area_m2",
        "area_outlier_flag",
        "layout_outlier_flag",
        "abnormal_transaction_flag",
        "physical_condition_flag",
        "renovation_flag",
        "broad_note_flag",
        "special_note_flag",
        "presale_note_flag",
        "separate_registration_flag",
    ],
    "drop_cols": [
        "id",
        "transfer_id",
        "source_release",
        "source_order",
        "source_file",
        "source_folder",
        "trade_date",
        "address_raw",
        "note_raw",
    ],
    "leakage_cols": [
        "total_price",
        "total_price_wan",
        "unit_price_m2",
        "unit_price_ping",
        "parking_price",
    ],
}


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


def resolve_path_arg(value: str, must_exist: bool = False) -> Path:
    raw = str(value)
    normalized = raw.replace("\\", os.sep)
    candidates = [Path(raw).expanduser(), Path(normalized).expanduser()]

    if not Path(normalized).is_absolute():
        candidates.append(Path.cwd() / normalized)

    normalized_path = Path(normalized)
    if normalized_path.name:
        candidates.append(Path.cwd() / normalized_path.name)

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate.resolve()

    fallback = Path(normalized).expanduser()
    if not fallback.is_absolute():
        fallback = (Path.cwd() / fallback).resolve()

    if must_exist:
        raise FileNotFoundError(f"Path does not exist: {value}")
    return fallback


def find_main_files(raw_dir: str | Path) -> list[Path]:
    raw_path = Path(raw_dir)
    files = [
        path
        for path in raw_path.rglob("*")
        if path.is_file()
        and path.name.lower() == "a_lvr_land_a.csv"
        and "@eaDir" not in path.parts
    ]
    return sorted(files, key=lambda p: str(p).lower())


def read_csv_with_encoding(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp950", "big5"):
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                encoding=encoding,
                keep_default_na=False,
                low_memory=False,
                on_bad_lines="skip",
            )
            logging.info("Read %s rows from %s with %s", len(df), path, encoding)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
            logging.warning("Failed reading %s with %s: %s", path, encoding, exc)
    raise RuntimeError(f"Failed to read CSV with supported encodings: {path}") from last_error


def parse_source_release(path: str | Path) -> tuple[str, int]:
    folder = Path(path).parent.name
    if folder == "本期":
        return "CURRENT", CURRENT_SOURCE_ORDER

    quarter_map = {
        "第一": 1,
        "第二": 2,
        "第三": 3,
        "第四": 4,
    }
    match = re.match(r"^(?P<year>\d{2,3})年(?P<quarter>第一|第二|第三|第四)季$", folder)
    if match:
        roc_year = int(match.group("year"))
        quarter = quarter_map[match.group("quarter")]
        return f"{roc_year}Q{quarter}", roc_year * 10 + quarter
    return folder, 0


def parse_roc_date(x: Any) -> pd.Timestamp | pd.NaT:
    if x is None or pd.isna(x) or isinstance(x, bool):
        return pd.NaT

    if isinstance(x, (int, np.integer)):
        text = str(int(x))
    elif isinstance(x, (float, np.floating)):
        if not np.isfinite(x) or not float(x).is_integer():
            return pd.NaT
        text = str(int(x))
    else:
        text = str(x).strip()

    if text.lower() in {"", "0", "--", "nan", "none", "null", "<na>"}:
        return pd.NaT

    text = text.replace(",", "").replace(" ", "")
    text = re.sub(r"\.0+$", "", text)
    digits = re.sub(r"\D", "", text)
    if len(digits) > 7 and digits.startswith("0"):
        digits = digits.lstrip("0")

    try:
        if len(digits) == 6:
            roc_year = int(digits[:2])
            month = int(digits[2:4])
            day = int(digits[4:6])
        elif len(digits) == 7:
            roc_year = int(digits[:3])
            month = int(digits[3:5])
            day = int(digits[5:7])
        else:
            return pd.NaT

        if roc_year <= 0:
            return pd.NaT
        return pd.Timestamp(year=roc_year + 1911, month=month, day=day)
    except (ValueError, OverflowError):
        return pd.NaT


def clean_numeric_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False)
        .str.replace("\u3000", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "--": pd.NA,
            "NaN": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
            "<NA>": pd.NA,
        }
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_chinese_number(text: str) -> float:
    text = str(text).strip()
    digit_match = re.search(r"-?\d+", text)
    if digit_match:
        return float(int(digit_match.group()))

    numeral_map = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    allowed = set(numeral_map) | {"十", "百"}
    chars = "".join(ch for ch in text if ch in allowed)
    if not chars:
        return np.nan

    total = 0
    rest = chars
    if "百" in rest:
        hundred_part, rest = rest.split("百", 1)
        total += (numeral_map.get(hundred_part, 1) if hundred_part else 1) * 100
    if "十" in rest:
        ten_part, one_part = rest.split("十", 1)
        total += (numeral_map.get(ten_part, 1) if ten_part else 1) * 10
        if one_part:
            total += numeral_map.get(one_part, 0)
    elif rest:
        total += numeral_map.get(rest, 0)

    return float(total) if total != 0 else np.nan


def parse_chinese_floor(text: Any) -> float:
    if text is None or pd.isna(text):
        return np.nan

    raw = str(text).strip()
    if raw == "" or raw in {"全", "見其他登記事項"} or "見其他登記事項" in raw:
        return np.nan

    first_part = re.split(r"[，,、]", raw, maxsplit=1)[0].strip()
    is_basement = first_part.startswith("地下")
    floor_text = first_part.replace("地下", "", 1) if is_basement else first_part
    floor_text = (
        floor_text.replace("第", "")
        .replace("層", "")
        .replace("樓", "")
        .replace("夾層", "")
        .strip()
    )

    value = _parse_chinese_number(floor_text)
    if pd.isna(value):
        return np.nan
    return -value if is_basement else value


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned_columns = []
    seen: dict[str, int] = {}
    for col in df.columns:
        cleaned = str(col).strip().lstrip("\ufeff")
        if cleaned in seen:
            seen[cleaned] += 1
            cleaned = f"{cleaned}.{seen[cleaned]}"
        else:
            seen[cleaned] = 0
        cleaned_columns.append(cleaned)
    df.columns = cleaned_columns

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def add_metadata(df: pd.DataFrame, source_file: str | Path) -> pd.DataFrame:
    df = df.copy()
    release, order = parse_source_release(source_file)
    source_path = Path(source_file).resolve()
    df["source_file"] = str(source_path)
    df["source_folder"] = source_path.parent.name
    df["source_release"] = release
    df["source_order"] = order
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    trade_dates = work["交易年月日"].apply(parse_roc_date)
    total_price = clean_numeric_series(work["總價元"])
    unit_price = clean_numeric_series(work["單價元平方公尺"])

    key_text = (
        work[["鄉鎮市區", "交易標的", "交易年月日", "總價元", "單價元平方公尺"]]
        .astype("string")
        .fillna("")
        .agg(" ".join, axis=1)
    )
    english_or_header = key_text.str.contains(r"[A-Za-z]{3,}", regex=True, na=False)
    chinese_header = (
        work["鄉鎮市區"].astype("string").str.strip().eq("鄉鎮市區")
        | work["交易年月日"].astype("string").str.strip().eq("交易年月日")
    )
    invalid_payload = trade_dates.isna() & total_price.isna() & unit_price.isna()
    drop_mask = english_or_header | chinese_header | invalid_payload
    return work.loc[~drop_mask].copy()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trade_date"] = df["交易年月日"].apply(parse_roc_date)
    df["trade_year"] = df["trade_date"].dt.year
    df["trade_month"] = df["trade_date"].dt.month
    df["trade_quarter"] = df["trade_date"].dt.quarter
    df["trade_ym"] = df["trade_date"].dt.strftime("%Y-%m")
    df["trade_yq"] = df["trade_date"].apply(
        lambda x: f"{x.year}Q{x.quarter}" if pd.notna(x) else pd.NA
    )

    df["completion_date"] = df["建築完成年月"].apply(parse_roc_date)
    age = (df["trade_date"] - df["completion_date"]).dt.days / 365.25
    invalid_age = (
        df["trade_date"].isna()
        | df["completion_date"].isna()
        | (df["completion_date"] > df["trade_date"])
    )
    df["building_age"] = age.mask(invalid_age)
    return df


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_SOURCE_COLUMNS:
        df[col] = clean_numeric_series(df[col])

    df["land_area_m2"] = df["土地移轉總面積平方公尺"]
    df["building_area_m2"] = df["建物移轉總面積平方公尺"]
    df["building_area_ping"] = df["building_area_m2"] / 3.305785
    df["main_building_area_m2"] = df["主建物面積"]
    df["main_building_area_ping"] = df["main_building_area_m2"] / 3.305785
    df["auxiliary_area_m2"] = df["附屬建物面積"]
    df["balcony_area_m2"] = df["陽台面積"]
    df["rooms"] = df["建物現況格局-房"]
    df["living_rooms"] = df["建物現況格局-廳"]
    df["bathrooms"] = df["建物現況格局-衛"]
    df["total_price"] = df["總價元"]
    df["total_price_wan"] = df["總價元"] / 10000
    df["unit_price_m2"] = df["單價元平方公尺"]
    df["unit_price_ping"] = df["單價元平方公尺"] * 3.305785 / 10000
    df["parking_area_m2"] = df["車位移轉總面積平方公尺"]
    df["parking_price"] = df["車位總價元"]
    return df


def add_floor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["floor"] = df["移轉層次"].apply(parse_chinese_floor)
    df["total_floor"] = df["總樓層數"].apply(parse_chinese_floor)
    df["floor_ratio"] = df["floor"] / df["total_floor"]
    invalid_ratio = df["total_floor"].isna() | (df["total_floor"] <= 0)
    df.loc[invalid_ratio, "floor_ratio"] = np.nan
    df["is_basement"] = (df["floor"] < 0).fillna(False).astype(int)
    df["multi_floor"] = (
        df["移轉層次"].astype("string").fillna("").str.contains(r"[，,、]", regex=True).astype(int)
    )
    return df


def _clean_text_series(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.strip()
    return cleaned.replace(
        {
            "": pd.NA,
            "--": pd.NA,
            "NaN": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
            "<NA>": pd.NA,
        }
    )


def _yes_no_to_numeric(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.strip()
    result = pd.Series(np.nan, index=s.index, dtype="float")
    result.loc[cleaned.eq("有")] = 1
    result.loc[cleaned.eq("無")] = 0
    return result


def add_categorical_boolean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["district"] = _clean_text_series(df["鄉鎮市區"])
    df["building_type"] = _clean_text_series(df["建物型態"])
    df["main_use"] = _clean_text_series(df["主要用途"])
    df["material"] = _clean_text_series(df["主要建材"])
    df["address_raw"] = _clean_text_series(df["土地位置建物門牌"])
    df["note_raw"] = _clean_text_series(df["備註"])
    df["id"] = _clean_text_series(df["編號"])
    df["transfer_id"] = _clean_text_series(df["移轉編號"])

    df["has_management"] = _yes_no_to_numeric(df["有無管理組織"])
    df["has_elevator"] = _yes_no_to_numeric(df["電梯"])

    parking_type = df["車位類別"].fillna("").astype(str).str.strip()
    has_parking_type = (parking_type != "") & ~parking_type.isin(
        ["無", "--", "nan", "NaN", "None", "none", "NULL", "null", "<NA>"]
    )
    has_parking_area = df["parking_area_m2"].fillna(0) > 0
    has_parking_price = df["parking_price"].fillna(0) > 0
    df["has_parking"] = (has_parking_type | has_parking_area | has_parking_price).astype(int)
    df["main_use_missing"] = df["main_use"].isna().astype(int)
    return df


def add_special_note_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    notes = _combined_note_text(df)
    df["abnormal_transaction_flag"] = _keyword_flag(notes, ABNORMAL_TRANSACTION_KEYWORDS)
    df["physical_condition_flag"] = _keyword_flag(notes, PHYSICAL_CONDITION_KEYWORDS)
    df["renovation_flag"] = _keyword_flag(notes, RENOVATION_KEYWORDS)
    df["broad_note_flag"] = _keyword_flag(notes, BROAD_NOTE_KEYWORDS)
    df["strict_special_note_flag"] = _keyword_flag(notes, STRICT_SPECIAL_NOTE_KEYWORDS)
    df["special_note_flag"] = df["abnormal_transaction_flag"]
    return df


def _keyword_flag(text: pd.Series, keywords: list[str]) -> pd.Series:
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    return text.str.contains(pattern, regex=True, na=False).astype(int)


def _combined_note_text(df: pd.DataFrame) -> pd.Series:
    note_raw = df["note_raw"] if "note_raw" in df.columns else pd.Series("", index=df.index)
    original_note = df["備註"] if "備註" in df.columns else pd.Series("", index=df.index)
    return (
        note_raw.fillna("").astype(str)
        + " "
        + original_note.fillna("").astype(str)
    )


def add_building_age_missing_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["building_age_missing"] = df["building_age"].isna().astype(int)
    return df


def add_presale_note_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pattern = "|".join(re.escape(keyword) for keyword in PRESALE_NOTE_KEYWORDS)
    notes = _combined_note_text(df)
    df["presale_note_flag"] = notes.str.contains(pattern, regex=True, na=False).astype(int)
    return df


def add_separate_registration_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pattern = "|".join(re.escape(keyword) for keyword in SEPARATE_REGISTRATION_KEYWORDS)
    notes = _combined_note_text(df)
    df["separate_registration_flag"] = notes.str.contains(pattern, regex=True, na=False).astype(int)
    return df


def add_area_outlier_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    area = df["building_area_m2"]
    df["area_outlier_flag"] = ((area < 10) | (area > 600)).fillna(False).astype(int)
    return df


def add_layout_outlier_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    layout_outlier = (
        (df["rooms"] > 10)
        | (df["living_rooms"] > 5)
        | (df["bathrooms"] > 5)
    )
    df["layout_outlier_flag"] = layout_outlier.fillna(False).astype(int)
    return df


def deduplicate_records(
    df: pd.DataFrame, return_stats: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    work = df.copy()
    if "_row_order" not in work.columns:
        work["_row_order"] = np.arange(len(work))

    id_col = work["id"] if "id" in work.columns else _clean_text_series(work["編號"])
    has_id = id_col.notna() & id_col.astype("string").str.strip().ne("")

    id_part = work.loc[has_id].copy()
    fallback_part = work.loc[~has_id].copy()

    if not id_part.empty:
        id_part = (
            id_part.sort_values(["id", "source_order", "_row_order"], kind="mergesort")
            .drop_duplicates(subset=["id"], keep="last")
            .copy()
        )

    fallback_keys = ["district", "address_raw", "交易年月日", "total_price", "building_area_m2"]
    for col in fallback_keys:
        if col not in fallback_part.columns:
            fallback_part[col] = pd.NA
    if not fallback_part.empty:
        fallback_part["_fallback_key"] = (
            fallback_part[fallback_keys]
            .astype("string")
            .fillna("<NA>")
            .agg("|".join, axis=1)
        )
        fallback_part = (
            fallback_part.sort_values(["_fallback_key", "source_order", "_row_order"], kind="mergesort")
            .drop_duplicates(subset=["_fallback_key"], keep="last")
            .drop(columns=["_fallback_key"])
            .copy()
        )

    deduped = (
        pd.concat([id_part, fallback_part], ignore_index=False)
        .sort_values("_row_order", kind="mergesort")
        .reset_index(drop=True)
    )

    stats = {
        "rows_before": int(len(work)),
        "rows_after": int(len(deduped)),
        "rows_removed": int(len(work) - len(deduped)),
        "id_dedup_rows": int(has_id.sum()),
        "fallback_dedup_rows": int((~has_id).sum()),
    }
    if return_stats:
        return deduped, stats
    return deduped


def append_filter_log(
    filter_log: list[dict[str, Any]],
    step: str,
    rows_before: int,
    rows_after: int,
    description: str,
) -> None:
    filter_log.append(
        {
            "step": step,
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "rows_removed": int(rows_before - rows_after),
            "description": description,
        }
    )


def _contains_any(series: pd.Series, keywords: list[str]) -> pd.Series:
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    return series.fillna("").astype(str).str.contains(pattern, regex=True, na=False)


def filter_clean_all(
    df: pd.DataFrame,
    drop_outliers: bool = True,
    filter_log: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if filter_log is None:
        filter_log = []

    work = df.copy()

    before = len(work)
    mask = work["交易標的"].fillna("").astype(str).str.contains("房地", na=False)
    work = work.loc[mask].copy()
    append_filter_log(filter_log, "filter_transaction_target", before, len(work), "只保留交易標的包含「房地」")

    before = len(work)
    main_use = work["main_use"]
    main_use_text = main_use.fillna("").astype(str)
    main_use_missing = main_use.isna() | main_use_text.str.strip().eq("")
    has_residential_use = main_use_text.str.contains("住家用", na=False)
    has_non_residential_use = _contains_any(main_use, NON_RESIDENTIAL_USE_KEYWORDS)
    mask = main_use_missing | (has_residential_use & ~has_non_residential_use)
    work = work.loc[mask].copy()
    append_filter_log(filter_log, "filter_main_use", before, len(work), "保留住家用或主要用途缺失，排除明顯非住宅用途")

    before = len(work)
    mask = _contains_any(work["building_type"], RESIDENTIAL_BUILDING_TYPES)
    work = work.loc[mask].copy()
    append_filter_log(filter_log, "filter_building_type", before, len(work), "保留住宅大樓、華廈、公寓、套房、透天厝")

    before = len(work)
    mask = work["trade_date"].notna()
    work = work.loc[mask].copy()
    append_filter_log(filter_log, "filter_invalid_dates", before, len(work), "排除 trade_date 缺失")

    before = len(work)
    mask = (
        (work["unit_price_m2"] > 0)
        & (work["total_price"] > 0)
        & (work["building_area_m2"] > 0)
        & work["district"].notna()
        & work["district"].astype("string").str.strip().ne("")
        & work["building_type"].notna()
        & work["building_type"].astype("string").str.strip().ne("")
    )
    work = work.loc[mask].copy()
    append_filter_log(filter_log, "filter_invalid_price_area", before, len(work), "排除價格、面積、行政區或建物型態異常")

    before = len(work)
    age = work["building_age"]
    mask = age.isna() | ((age >= 0) & (age <= 100))
    work = work.loc[mask].copy()
    append_filter_log(filter_log, "filter_invalid_building_age", before, len(work), "排除 building_age < 0 或 > 100")

    outlier_info: dict[str, Any] = {
        "enabled": bool(drop_outliers),
        "q01": np.nan,
        "q99": np.nan,
        "rows_before": int(len(work)),
        "rows_after": int(len(work)),
    }
    before = len(work)
    if drop_outliers and not work.empty:
        q01 = work["unit_price_ping"].quantile(0.01)
        q99 = work["unit_price_ping"].quantile(0.99)
        outlier_info["q01"] = float(q01) if pd.notna(q01) else np.nan
        outlier_info["q99"] = float(q99) if pd.notna(q99) else np.nan
        if pd.notna(q01) and pd.notna(q99):
            work = work.loc[work["unit_price_ping"].between(q01, q99, inclusive="both")].copy()
    outlier_info["rows_after"] = int(len(work))
    description = "以 unit_price_ping 1% 到 99% quantile 過濾" if drop_outliers else "未啟用價格極端值過濾"
    append_filter_log(filter_log, "filter_outliers", before, len(work), description)

    append_filter_log(filter_log, "build_clean_all", len(work), len(work), "建立 clean_all dataset")
    return work.reset_index(drop=True), outlier_info


def build_clean_no_parking(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["has_parking"].fillna(0).eq(0)].copy().reset_index(drop=True)


def build_model_ready(
    df: pd.DataFrame,
    parking_mode: str = "keep",
    exclude_presale_and_separate: bool = True,
    note_filter_mode: str = "abnormal",
    filter_log: list[dict[str, Any]] | None = None,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    stats: dict[str, Any] = {"rows_start": int(len(work))}

    before = len(work)
    if note_filter_mode == "abnormal":
        note_filter_col = "abnormal_transaction_flag"
        note_filter_step = "filter_abnormal_transaction_for_model_ready"
        note_filter_description = "model_ready 排除 abnormal_transaction_flag = 1"
    elif note_filter_mode == "strict":
        note_filter_col = "strict_special_note_flag"
        note_filter_step = "filter_strict_special_note_for_model_ready"
        note_filter_description = "strict model_ready 沿用舊版嚴格備註規則排除"
    else:
        raise ValueError("note_filter_mode must be abnormal or strict")

    work = work.loc[work[note_filter_col].fillna(0).ne(1)].copy()
    if filter_log is not None:
        append_filter_log(
            filter_log,
            note_filter_step,
            before,
            len(work),
            note_filter_description,
        )
    stats["rows_before_note_filter"] = int(before)
    stats["rows_after_note_filter"] = int(len(work))
    stats["note_filter_mode"] = note_filter_mode

    work = work.loc[_contains_any(work["building_type"], MODEL_BUILDING_TYPES)].copy()

    if parking_mode == "drop":
        work = work.loc[work["has_parking"].fillna(0).eq(0)].copy()
    elif parking_mode != "keep":
        raise ValueError("parking_mode must be keep or drop")

    stats["rows_after_base_filters"] = int(len(work))

    before = len(work)
    if exclude_presale_and_separate:
        work = work.loc[work["presale_note_flag"].fillna(0).ne(1)].copy()
    if filter_log is not None:
        append_filter_log(
            filter_log,
            "filter_presale_for_model_ready",
            before,
            len(work),
            "主要 model_ready 排除 presale_note_flag = 1；with_presale 版本不套用此排除",
        )
    stats["rows_before_presale_filter"] = int(before)
    stats["rows_after_presale_filter"] = int(len(work))

    before = len(work)
    if exclude_presale_and_separate:
        work = work.loc[work["separate_registration_flag"].fillna(0).ne(1)].copy()
    if filter_log is not None:
        append_filter_log(
            filter_log,
            "filter_separate_registration_for_model_ready",
            before,
            len(work),
            "主要 model_ready 排除 separate_registration_flag = 1；with_presale 版本不套用此排除",
        )
    stats["rows_before_separate_registration_filter"] = int(before)
    stats["rows_after_separate_registration_filter"] = int(len(work))

    before = len(work)
    work = work.loc[work["area_outlier_flag"].fillna(0).ne(1)].copy()
    if filter_log is not None:
        append_filter_log(
            filter_log,
            "filter_area_outlier_for_model_ready",
            before,
            len(work),
            "model_ready 排除 building_area_m2 < 10 或 > 600 的面積極端案件",
        )
    stats["rows_before_area_outlier_filter"] = int(before)
    stats["rows_after_area_outlier_filter"] = int(len(work))

    before = len(work)
    work = work.loc[work["layout_outlier_flag"].fillna(0).ne(1)].copy()
    if filter_log is not None:
        append_filter_log(
            filter_log,
            "filter_layout_outlier_for_model_ready",
            before,
            len(work),
            "model_ready 排除 rooms > 10、living_rooms > 5 或 bathrooms > 5 的格局極端案件",
        )
    stats["rows_before_layout_outlier_filter"] = int(before)
    stats["rows_after_layout_outlier_filter"] = int(len(work))

    for col in MODEL_READY_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA
    result = work[MODEL_READY_COLUMNS].copy().reset_index(drop=True)
    if return_stats:
        return result, stats
    return result


def write_feature_config(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "feature_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(FEATURE_CONFIG, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logging.info("Wrote feature config to %s", path)
    return path


def _write_dataset(df: pd.DataFrame, base_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")

    try:
        df.to_parquet(parquet_path, index=False)
        results.append({"path": str(parquet_path), "format": "parquet", "status": "ok", "error": ""})
        logging.info("Wrote parquet to %s", parquet_path)
    except Exception as exc:  # noqa: BLE001 - parquet engines vary by environment.
        results.append({"path": str(parquet_path), "format": "parquet", "status": "failed", "error": str(exc)})
        logging.warning("Parquet output failed for %s: %s", parquet_path, exc)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    results.append({"path": str(csv_path), "format": "csv", "status": "ok", "error": ""})
    logging.info("Wrote csv to %s", csv_path)
    return results


def write_outputs(
    output_dir: str | Path,
    report_dir: str | Path,
    raw_combined: pd.DataFrame,
    clean_all: pd.DataFrame,
    clean_no_parking: pd.DataFrame,
    model_ready: pd.DataFrame,
    model_ready_with_presale: pd.DataFrame,
    model_ready_strict: pd.DataFrame,
    filter_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []
    datasets = {
        "taipei_house_raw_combined": raw_combined,
        "taipei_house_clean_all": clean_all,
        "taipei_house_clean_no_parking": clean_no_parking,
        "taipei_house_model_ready": model_ready,
        "taipei_house_model_ready_with_presale": model_ready_with_presale,
        "taipei_house_model_ready_strict": model_ready_strict,
    }
    for name, df in datasets.items():
        outputs.extend(_write_dataset(df, output_dir / name))

    filter_log_path = report_dir / "filter_log.csv"
    pd.DataFrame(filter_log).to_csv(filter_log_path, index=False, encoding="utf-8-sig")
    outputs.append({"path": str(filter_log_path), "format": "csv", "status": "ok", "error": ""})
    logging.info("Wrote filter log to %s", filter_log_path)
    return outputs


def _format_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No data._\n"
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = [
        "| " + " | ".join(_format_scalar(cell).replace("\n", " ") for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines]) + "\n"


def _value_counts_rows(df: pd.DataFrame, col: str, limit: int | None = None) -> list[list[Any]]:
    if df.empty or col not in df.columns:
        return []
    counts = df[col].astype("string").fillna("<NA>").value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    return [[idx, int(value)] for idx, value in counts.items()]


def _date_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _flag_count_ratio_rows(df: pd.DataFrame, col: str) -> list[list[Any]]:
    if df.empty or col not in df.columns:
        return []
    total = len(df)
    flagged = int(df[col].fillna(0).eq(1).sum())
    return [[col, total, flagged, f"{flagged / total:.2%}" if total else ""]]


def _numeric_summary_rows(df: pd.DataFrame, cols: list[str]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for col in cols:
        if col not in df.columns:
            continue
        summary = df[col].describe()
        for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            rows.append([col, stat, summary.get(stat, np.nan)])
    return rows


def write_summary_report(
    report_dir: str | Path,
    raw_dir: str | Path,
    source_files: list[Path],
    file_read_counts: dict[str, int],
    raw_combined: pd.DataFrame,
    deduped: pd.DataFrame,
    clean_all: pd.DataFrame,
    clean_no_parking: pd.DataFrame,
    model_ready: pd.DataFrame,
    model_ready_with_presale: pd.DataFrame,
    dedup_stats: dict[str, int],
    model_ready_stats: dict[str, Any],
    model_ready_with_presale_stats: dict[str, Any],
    model_ready_strict: pd.DataFrame,
    model_ready_strict_stats: dict[str, Any],
    outlier_info: dict[str, Any],
    output_results: list[dict[str, Any]],
    feature_config_path: Path,
    drop_outliers: bool,
    parking_mode: str,
    include_current: bool,
) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "data_build_summary.md"

    source_rows = [[file, rows] for file, rows in sorted(file_read_counts.items())]
    release_rows = _value_counts_rows(raw_combined, "source_release")
    source_order_rows = (
        raw_combined[["source_release", "source_order"]]
        .drop_duplicates()
        .sort_values(["source_order", "source_release"])
        .values.tolist()
        if not raw_combined.empty
        else []
    )

    min_trade_date = clean_all["trade_date"].min() if "trade_date" in clean_all else pd.NaT
    max_trade_date = clean_all["trade_date"].max() if "trade_date" in clean_all else pd.NaT
    yearly_rows = _value_counts_rows(clean_all.sort_values("trade_year"), "trade_year")
    quarterly_rows = _value_counts_rows(clean_all.sort_values("trade_yq"), "trade_yq")
    district_rows = _value_counts_rows(clean_all, "district")
    building_type_rows = _value_counts_rows(clean_all, "building_type")
    main_use_rows = _value_counts_rows(clean_all, "main_use")
    parking_rows = _value_counts_rows(clean_all, "has_parking")
    abnormal_transaction_rows = _value_counts_rows(clean_all, "abnormal_transaction_flag")
    physical_condition_rows = _value_counts_rows(clean_all, "physical_condition_flag")
    renovation_rows = _value_counts_rows(clean_all, "renovation_flag")
    broad_note_rows = _value_counts_rows(clean_all, "broad_note_flag")
    special_note_rows = _value_counts_rows(clean_all, "special_note_flag")
    presale_note_rows = _value_counts_rows(clean_all, "presale_note_flag")
    separate_registration_rows = _value_counts_rows(clean_all, "separate_registration_flag")
    area_outlier_rows = _value_counts_rows(clean_all, "area_outlier_flag")
    layout_outlier_rows = _value_counts_rows(clean_all, "layout_outlier_flag")
    note_flag_ratio_rows = []
    for col in [
        "abnormal_transaction_flag",
        "physical_condition_flag",
        "renovation_flag",
        "broad_note_flag",
        "special_note_flag",
    ]:
        note_flag_ratio_rows.extend(_flag_count_ratio_rows(clean_all, col))
    layout_outlier_ratio_rows = _flag_count_ratio_rows(clean_all, "layout_outlier_flag")
    layout_summary_rows = _numeric_summary_rows(clean_all, ["rooms", "living_rooms", "bathrooms"])

    building_age_missing_rows = []
    for name, dataset in [
        ("clean_all", clean_all),
        ("model_ready", model_ready),
        ("model_ready_with_presale", model_ready_with_presale),
        ("model_ready_strict", model_ready_strict),
    ]:
        missing = int(dataset["building_age"].isna().sum()) if "building_age" in dataset.columns else 0
        total = int(len(dataset))
        missing_ratio = f"{missing / total:.2%}" if total else ""
        building_age_missing_rows.append([name, total, missing, missing_ratio])

    model_ready_filter_rows = [
        ["排除異常交易備註前", model_ready_stats.get("rows_before_note_filter")],
        ["排除異常交易備註後", model_ready_stats.get("rows_after_note_filter")],
        ["base filters 後", model_ready_stats.get("rows_after_base_filters")],
        ["排除預售屋前", model_ready_stats.get("rows_before_presale_filter")],
        ["排除預售屋後", model_ready_stats.get("rows_after_presale_filter")],
        ["排除分件登記前", model_ready_stats.get("rows_before_separate_registration_filter")],
        ["排除分件登記後", model_ready_stats.get("rows_after_separate_registration_filter")],
        ["排除面積極端前", model_ready_stats.get("rows_before_area_outlier_filter")],
        ["排除面積極端後", model_ready_stats.get("rows_after_area_outlier_filter")],
        ["排除格局極端前", model_ready_stats.get("rows_before_layout_outlier_filter")],
        ["排除格局極端後", model_ready_stats.get("rows_after_layout_outlier_filter")],
        ["with_presale 排除異常交易備註前", model_ready_with_presale_stats.get("rows_before_note_filter")],
        ["with_presale 排除異常交易備註後", model_ready_with_presale_stats.get("rows_after_note_filter")],
        ["with_presale base filters 後", model_ready_with_presale_stats.get("rows_after_base_filters")],
        ["with_presale 排除面積極端前", model_ready_with_presale_stats.get("rows_before_area_outlier_filter")],
        ["with_presale 排除面積極端後", model_ready_with_presale_stats.get("rows_after_area_outlier_filter")],
        ["with_presale 排除格局極端前", model_ready_with_presale_stats.get("rows_before_layout_outlier_filter")],
        ["with_presale 排除格局極端後", model_ready_with_presale_stats.get("rows_after_layout_outlier_filter")],
        ["strict 沿用舊版嚴格備註規則前", model_ready_strict_stats.get("rows_before_note_filter")],
        ["strict 沿用舊版嚴格備註規則後", model_ready_strict_stats.get("rows_after_note_filter")],
        ["strict base filters 後", model_ready_strict_stats.get("rows_after_base_filters")],
        ["strict 排除面積極端後", model_ready_strict_stats.get("rows_after_area_outlier_filter")],
        ["strict 排除格局極端後", model_ready_strict_stats.get("rows_after_layout_outlier_filter")],
    ]

    target_stats = model_ready["unit_price_ping"].describe() if not model_ready.empty else pd.Series(dtype=float)
    target_rows = [[idx, value] for idx, value in target_stats.items()]

    parquet_failures = [
        [item["path"], item["error"]]
        for item in output_results
        if item["format"] == "parquet" and item["status"] != "ok"
    ]
    output_rows = [[item["path"], item["format"], item["status"], item["error"]] for item in output_results]

    content = [
        "# 臺北市住宅實價登錄資料建置摘要",
        "",
        "## 1. 原始資料來源",
        "",
        f"- raw-dir: `{Path(raw_dir)}`",
        f"- 找到的主檔數量: {len(source_files):,}",
        f"- parking-mode: `{parking_mode}`",
        f"- include-current: `{str(include_current).lower()}`",
        "",
        _markdown_table(["source_file", "read_rows"], source_rows),
        "",
        "## 2. source_release 統計",
        "",
        _markdown_table(["source_release", "rows"], release_rows),
        "",
        "### source_order 對應表",
        "",
        _markdown_table(["source_release", "source_order"], source_order_rows),
        "",
        "## 3. 合併結果",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["合併後總筆數", len(raw_combined)],
                ["去重複前筆數", dedup_stats.get("rows_before", 0)],
                ["去重複後筆數", dedup_stats.get("rows_after", 0)],
                ["移除重複筆數", dedup_stats.get("rows_removed", 0)],
                ["使用 id 去重複的筆數", dedup_stats.get("id_dedup_rows", 0)],
                ["使用 fallback key 去重複的筆數", dedup_stats.get("fallback_dedup_rows", 0)],
            ],
        ),
        "",
        "## 4. 日期範圍",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["trade_date 最小日期", _date_str(min_trade_date)],
                ["trade_date 最大日期", _date_str(max_trade_date)],
            ],
        ),
        "",
        "### 每年交易筆數",
        "",
        _markdown_table(["trade_year", "rows"], yearly_rows),
        "",
        "### 每季交易筆數",
        "",
        _markdown_table(["trade_yq", "rows"], quarterly_rows),
        "",
        "## 5. 資料分佈",
        "",
        "### 各行政區筆數",
        "",
        _markdown_table(["district", "rows"], district_rows),
        "",
        "### 各建物型態筆數",
        "",
        _markdown_table(["building_type", "rows"], building_type_rows),
        "",
        "### 各主要用途筆數",
        "",
        _markdown_table(["main_use", "rows"], main_use_rows),
        "",
        "### 含車位 / 不含車位筆數",
        "",
        _markdown_table(["has_parking", "rows"], parking_rows),
        "",
        "### abnormal_transaction_flag 筆數",
        "",
        _markdown_table(["abnormal_transaction_flag", "rows"], abnormal_transaction_rows),
        "",
        "### physical_condition_flag 筆數",
        "",
        _markdown_table(["physical_condition_flag", "rows"], physical_condition_rows),
        "",
        "### renovation_flag 筆數",
        "",
        _markdown_table(["renovation_flag", "rows"], renovation_rows),
        "",
        "### broad_note_flag 筆數",
        "",
        _markdown_table(["broad_note_flag", "rows"], broad_note_rows),
        "",
        "### special_note_flag 筆數",
        "",
        _markdown_table(["special_note_flag", "rows"], special_note_rows),
        "",
        "### 各 note flag = 1 筆數與比例",
        "",
        _markdown_table(["flag", "rows", "flagged_rows", "flagged_ratio"], note_flag_ratio_rows),
        "",
        "### building_age 缺失筆數與比例",
        "",
        _markdown_table(["dataset", "rows", "building_age_missing_rows", "missing_ratio"], building_age_missing_rows),
        "",
        "### presale_note_flag 筆數",
        "",
        _markdown_table(["presale_note_flag", "rows"], presale_note_rows),
        "",
        "### separate_registration_flag 筆數",
        "",
        _markdown_table(["separate_registration_flag", "rows"], separate_registration_rows),
        "",
        "### area_outlier_flag 筆數",
        "",
        _markdown_table(["area_outlier_flag", "rows"], area_outlier_rows),
        "",
        "### layout_outlier_flag 筆數",
        "",
        _markdown_table(["layout_outlier_flag", "rows"], layout_outlier_rows),
        "",
        "### layout_outlier_flag = 1 筆數與比例",
        "",
        _markdown_table(["flag", "rows", "flagged_rows", "flagged_ratio"], layout_outlier_ratio_rows),
        "",
        "### rooms / living_rooms / bathrooms 統計",
        "",
        _markdown_table(["column", "stat", "value"], layout_summary_rows),
        "",
        "## 6. 目標值統計",
        "",
        "target: `unit_price_ping`，單位為萬元 / 坪；統計來源為 model_ready dataset。",
        "",
        _markdown_table(["stat", "value"], target_rows),
        "",
        "## 7. outlier 過濾",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["是否啟用 drop-outliers", str(drop_outliers).lower()],
                ["q01 門檻", outlier_info.get("q01")],
                ["q99 門檻", outlier_info.get("q99")],
                ["過濾前筆數", outlier_info.get("rows_before")],
                ["過濾後筆數", outlier_info.get("rows_after")],
            ],
        ),
        "",
        "## 8. 最終輸出",
        "",
        _markdown_table(
            ["dataset", "rows"],
            [
                ["clean_all", len(clean_all)],
                ["clean_no_parking", len(clean_no_parking)],
                ["taipei_house_model_ready.csv", len(model_ready)],
                ["taipei_house_model_ready_with_presale.csv", len(model_ready_with_presale)],
                ["taipei_house_model_ready_strict.csv", len(model_ready_strict)],
            ],
        ),
        "",
        "### model_ready 篩選補充統計",
        "",
        _markdown_table(["step", "rows"], model_ready_filter_rows),
        "",
        f"- feature_config: `{feature_config_path}`",
        "",
        _markdown_table(["path", "format", "status", "error"], output_rows),
        "",
        "### Parquet 輸出狀態",
        "",
        _markdown_table(["path", "error"], parquet_failures)
        if parquet_failures
        else "所有 parquet 輸出成功。\n",
        "",
        "## 9. 特別提醒",
        "",
        "- `source_release` 是資料發布批次，不代表交易日期。",
        "- 未來模型時間切分應該使用 `trade_date`。",
        "- `id` / `transfer_id` 不應作為模型特徵。",
        "- `total_price`、`unit_price_m2`、`unit_price_ping`、`parking_price` 屬於 leakage 或 target 相關欄位，不應放入 feature。",
        "",
    ]

    path.write_text("\n".join(content), encoding="utf-8")
    logging.info("Wrote summary report to %s", path)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Taipei housing clean/model-ready datasets.")
    parser.add_argument("--raw-dir", default=r"Personal\Hank\Esuan\Data")
    parser.add_argument("--output-dir", default=r"data\processed")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--parking-mode", choices=["keep", "drop"], default="keep")
    parser.add_argument("--drop-outliers", type=str_to_bool, choices=[True, False], default=True)
    parser.add_argument("--include-current", type=str_to_bool, choices=[True, False], default=True)
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()

    raw_dir = resolve_path_arg(args.raw_dir, must_exist=True)
    output_dir = resolve_path_arg(args.output_dir)
    report_dir = resolve_path_arg(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Raw dir: %s", raw_dir)
    logging.info("Output dir: %s", output_dir)
    logging.info("Report dir: %s", report_dir)

    source_files = find_main_files(raw_dir)
    if not args.include_current:
        source_files = [path for path in source_files if parse_source_release(path)[0] != "CURRENT"]
    logging.info("Found %s Taipei main files", len(source_files))
    if not source_files:
        raise FileNotFoundError(f"No a_lvr_land_a.csv files found under {raw_dir}")

    frames: list[pd.DataFrame] = []
    file_read_counts: dict[str, int] = {}
    for source_file in source_files:
        df = read_csv_with_encoding(source_file)
        df = standardize_columns(df)
        df = add_metadata(df, source_file)
        frames.append(df)
        file_read_counts[str(Path(source_file).resolve())] = int(len(df))

    raw_with_header_rows = pd.concat(frames, ignore_index=True)
    raw_with_header_rows["_row_order"] = np.arange(len(raw_with_header_rows))
    filter_log: list[dict[str, Any]] = []
    append_filter_log(
        filter_log,
        "raw_combined",
        len(raw_with_header_rows),
        len(raw_with_header_rows),
        "讀取並合併所有臺北市買賣主檔",
    )
    logging.info("Combined raw rows before invalid header removal: %s", len(raw_with_header_rows))

    before = len(raw_with_header_rows)
    raw_combined = remove_invalid_rows(raw_with_header_rows)
    append_filter_log(
        filter_log,
        "remove_invalid_header_rows",
        before,
        len(raw_combined),
        "移除英文欄位說明、重複 header 或明顯非資料列",
    )
    logging.info("Rows after invalid header removal: %s", len(raw_combined))

    raw_combined = add_date_features(raw_combined)
    before = len(raw_combined)
    raw_combined = add_building_age_missing_flag(raw_combined)
    append_filter_log(
        filter_log,
        "add_building_age_missing_flag",
        before,
        len(raw_combined),
        "新增 building_age_missing；building_age 缺失維持 NaN",
    )
    raw_combined = add_numeric_features(raw_combined)
    before = len(raw_combined)
    raw_combined = add_layout_outlier_flag(raw_combined)
    append_filter_log(
        filter_log,
        "flag_layout_outliers",
        before,
        len(raw_combined),
        "新增 layout_outlier_flag；clean_all 保留，model_ready 排除",
    )
    raw_combined = add_floor_features(raw_combined)
    raw_combined = add_categorical_boolean_features(raw_combined)
    before = len(raw_combined)
    raw_combined = add_special_note_flag(raw_combined)
    append_filter_log(
        filter_log,
        "flag_special_note_records",
        before,
        len(raw_combined),
        "新增 abnormal/physical/renovation/broad note flags；special_note_flag = abnormal_transaction_flag",
    )
    before = len(raw_combined)
    raw_combined = add_presale_note_flag(raw_combined)
    append_filter_log(
        filter_log,
        "flag_presale_records",
        before,
        len(raw_combined),
        "新增 presale_note_flag；clean_all 保留，主要 model_ready 排除",
    )
    before = len(raw_combined)
    raw_combined = add_separate_registration_flag(raw_combined)
    append_filter_log(
        filter_log,
        "flag_separate_registration_records",
        before,
        len(raw_combined),
        "新增 separate_registration_flag；clean_all 保留，主要 model_ready 排除",
    )
    before = len(raw_combined)
    raw_combined = add_area_outlier_flag(raw_combined)
    append_filter_log(
        filter_log,
        "flag_area_outliers",
        before,
        len(raw_combined),
        "新增 area_outlier_flag；clean_all 保留，model_ready 排除",
    )
    raw_combined = raw_combined.reset_index(drop=True)

    deduped, dedup_stats = deduplicate_records(raw_combined, return_stats=True)
    append_filter_log(
        filter_log,
        "deduplicate",
        dedup_stats["rows_before"],
        dedup_stats["rows_after"],
        "同 id 保留 source_order 最大；id 缺失時使用 fallback key",
    )
    logging.info(
        "Deduplicated rows: before=%s after=%s removed=%s",
        dedup_stats["rows_before"],
        dedup_stats["rows_after"],
        dedup_stats["rows_removed"],
    )

    clean_all, outlier_info = filter_clean_all(
        deduped,
        drop_outliers=args.drop_outliers,
        filter_log=filter_log,
    )
    logging.info("clean_all rows: %s", len(clean_all))

    before = len(clean_all)
    clean_no_parking = build_clean_no_parking(clean_all)
    append_filter_log(
        filter_log,
        "build_clean_no_parking",
        before,
        len(clean_no_parking),
        "clean_all 中只保留 has_parking = 0",
    )
    logging.info("clean_no_parking rows: %s", len(clean_no_parking))

    before = len(clean_all)
    model_ready, model_ready_stats = build_model_ready(
        clean_all,
        parking_mode=args.parking_mode,
        exclude_presale_and_separate=True,
        note_filter_mode="abnormal",
        filter_log=filter_log,
        return_stats=True,
    )
    append_filter_log(
        filter_log,
        "build_model_ready",
        before,
        len(model_ready),
        "排除 abnormal_transaction_flag = 1、預售屋、分件登記、面積極端、格局極端、透天厝；parking-mode=drop 時排除含車位",
    )
    logging.info("model_ready rows: %s", len(model_ready))

    before = len(clean_all)
    model_ready_with_presale, model_ready_with_presale_stats = build_model_ready(
        clean_all,
        parking_mode=args.parking_mode,
        exclude_presale_and_separate=False,
        note_filter_mode="abnormal",
        filter_log=None,
        return_stats=True,
    )
    append_filter_log(
        filter_log,
        "build_model_ready_with_presale",
        before,
        len(model_ready_with_presale),
        "保留 presale_note_flag 與 separate_registration_flag，其他 model_ready 規則相同",
    )
    logging.info("model_ready_with_presale rows: %s", len(model_ready_with_presale))

    before = len(clean_all)
    model_ready_strict, model_ready_strict_stats = build_model_ready(
        clean_all,
        parking_mode=args.parking_mode,
        exclude_presale_and_separate=True,
        note_filter_mode="strict",
        filter_log=None,
        return_stats=True,
    )
    append_filter_log(
        filter_log,
        "build_model_ready_strict",
        before,
        len(model_ready_strict),
        "沿用舊版嚴格備註規則；保留作為備查",
    )
    logging.info("model_ready_strict rows: %s", len(model_ready_strict))

    feature_config_path = write_feature_config(output_dir)
    output_results = write_outputs(
        output_dir=output_dir,
        report_dir=report_dir,
        raw_combined=raw_combined,
        clean_all=clean_all,
        clean_no_parking=clean_no_parking,
        model_ready=model_ready,
        model_ready_with_presale=model_ready_with_presale,
        model_ready_strict=model_ready_strict,
        filter_log=filter_log,
    )
    write_summary_report(
        report_dir=report_dir,
        raw_dir=raw_dir,
        source_files=source_files,
        file_read_counts=file_read_counts,
        raw_combined=raw_combined,
        deduped=deduped,
        clean_all=clean_all,
        clean_no_parking=clean_no_parking,
        model_ready=model_ready,
        model_ready_with_presale=model_ready_with_presale,
        dedup_stats=dedup_stats,
        model_ready_stats=model_ready_stats,
        model_ready_with_presale_stats=model_ready_with_presale_stats,
        model_ready_strict=model_ready_strict,
        model_ready_strict_stats=model_ready_strict_stats,
        outlier_info=outlier_info,
        output_results=output_results,
        feature_config_path=feature_config_path,
        drop_outliers=args.drop_outliers,
        parking_mode=args.parking_mode,
        include_current=args.include_current,
    )


if __name__ == "__main__":
    main()
