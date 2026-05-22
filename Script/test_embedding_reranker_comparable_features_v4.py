from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from add_embedding_reranker_comparable_features_v4 import (  # noqa: E402
    EMB_FEATURES,
    build_feature_config,
    build_text_representation,
    compute_embedding_comparable_features,
    cosine_top_k,
    get_candidate_indices,
    reranker_top_k,
    softmax,
    summarize_selected_candidates,
    validate_no_leakage,
)


class MockReranker:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def score(self, pairs, batch_size=4):
        return self.scores[: len(pairs)]


def _base_df(rows):
    defaults = {
        "material": "鋼筋混凝土造",
        "building_age": 20.0,
        "building_area_ping": 30.0,
        "main_building_area_ping": 22.0,
        "floor": 3,
        "total_floor": 5,
        "floor_ratio": 0.6,
        "rooms": 3,
        "living_rooms": 2,
        "bathrooms": 1,
        "has_parking": 0,
        "parking_area_m2": 0,
        "has_management": 0,
        "has_elevator": 0,
        "physical_condition_flag": 0,
        "renovation_flag": 1,
        "broad_note_flag": 0,
        "address_raw": "測試路一段",
        "note_raw": "測試備註",
    }
    full_rows = []
    for i, row in enumerate(rows):
        item = {**defaults, **row}
        item.setdefault("id", f"ID{i}")
        full_rows.append(item)
    df = pd.DataFrame(full_rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["text_representation"] = [build_text_representation(row) for _, row in df.iterrows()]
    return df


def test_text_representation_default_excludes_address_and_note():
    row = _base_df(
        [
            {
                "district": "大安區",
                "building_type": "公寓",
                "trade_date": "2020-01-01",
                "unit_price_ping": 100,
            }
        ]
    ).iloc[0]

    text = build_text_representation(row)

    assert "行政區：大安區" in text
    assert "建物型態：公寓" in text
    assert "屋齡" in text
    assert "建物坪數" in text
    assert "樓層" in text
    assert "車位" in text
    assert "裝潢標記：1" in text
    assert "測試路一段" not in text
    assert "測試備註" not in text


def test_candidate_pool_no_future_leakage():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 200},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-03", "unit_price_ping": 300},
        ]
    )

    candidates = get_candidate_indices(df.sort_values("trade_date").reset_index(drop=True), current_pos=1, window_days=730)

    assert candidates.tolist() == [0]


def test_same_day_exclusion():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 200},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 300},
        ]
    ).sort_values("trade_date").reset_index(drop=True)

    assert get_candidate_indices(df, current_pos=0, window_days=730).tolist() == []
    assert get_candidate_indices(df, current_pos=1, window_days=730).tolist() == []
    assert get_candidate_indices(df, current_pos=2, window_days=730).tolist() == [0, 1]


def test_same_district_and_building_type_enforced_by_grouping():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "B", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 500},
            {"district": "A", "building_type": "華廈", "trade_date": "2020-01-01", "unit_price_ping": 600},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 300},
        ]
    )
    embeddings = np.eye(len(df), dtype=float)

    result, _, _ = compute_embedding_comparable_features(
        df,
        embeddings,
        embedding_top_k=10,
        reranker_top_k_value=10,
        use_reranker=False,
        reranker=None,
        reranker_batch_size=4,
    )

    assert result.loc[3, "emb_730d_count"] == 1
    assert result.loc[3, "emb_730d_median_price"] == 100


def test_embedding_top_k_cosine_sorting():
    current = np.array([1.0, 0.0])
    candidates = np.array([[0.0, 1.0], [0.9, 0.1], [0.7, 0.7]])
    selected, sims = cosine_top_k(current, candidates, np.array([10, 11, 12]), top_k=2)

    assert selected.tolist() == [11, 12]
    assert sims[0] > sims[1]


def test_reranker_top_k_sorting():
    selected, scores = reranker_top_k(np.array([1, 2, 3]), np.array([0.2, 0.9, 0.5]), top_k=2)

    assert selected.tolist() == [2, 3]
    assert scores.tolist() == [0.9, 0.5]


def test_weighted_mean_uses_softmax_reranker_scores():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 200},
        ]
    )
    current = pd.Series({"trade_date": pd.Timestamp("2020-01-03"), "building_area_ping": 30, "building_age": 20})
    scores = np.array([1.0, 3.0])
    summary = summarize_selected_candidates(
        current,
        df,
        similarities=np.array([0.1, 0.2]),
        reranker_scores=scores,
        use_reranker=True,
    )
    expected = float(np.sum(softmax(scores) * np.array([100, 200])))

    assert summary["weighted_mean_price"] == pytest.approx(expected)


def test_no_candidates_summary():
    current = pd.Series({"trade_date": pd.Timestamp("2020-01-03"), "building_area_ping": 30, "building_age": 20})
    summary = summarize_selected_candidates(current, pd.DataFrame(), np.array([]), None, use_reranker=False)

    assert summary["count"] == 0
    assert np.isnan(summary["median_price"])
    assert np.isnan(summary["mean_price"])
    assert np.isnan(summary["weighted_mean_price"])
    assert np.isnan(summary["nearest_price"])


def test_feature_configs_add_and_replace():
    v2 = {
        "target_col": "unit_price_ping",
        "numeric_features": ["building_age", "district_median_price_180d"],
        "categorical_features": ["district"],
        "drop_cols": ["id", "address_raw", "note_raw"],
        "leakage_cols": ["unit_price_ping", "unit_price_m2", "total_price", "parking_price"],
        "notes": [],
    }
    v3 = {
        **v2,
        "numeric_features": ["building_age", "comp_365d_mean_price", "comp_730d_mean_price"],
    }

    add = build_feature_config(v3, mode="add")
    replace = build_feature_config(v2, mode="replace")

    for feature in EMB_FEATURES:
        assert feature in add["numeric_features"]
        assert feature in replace["numeric_features"]
    assert "comp_365d_mean_price" in add["numeric_features"]
    assert not any(feature.startswith("comp_365d") or feature.startswith("comp_730d") for feature in replace["numeric_features"])
    assert validate_no_leakage(add) == []
    assert validate_no_leakage(replace) == []


def test_row_count_unchanged():
    df = _base_df(
        [
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-01", "unit_price_ping": 100},
            {"district": "A", "building_type": "公寓", "trade_date": "2020-01-02", "unit_price_ping": 200},
        ]
    )
    embeddings = np.eye(len(df), dtype=float)
    result, _, _ = compute_embedding_comparable_features(
        df,
        embeddings,
        embedding_top_k=10,
        reranker_top_k_value=10,
        use_reranker=False,
        reranker=None,
        reranker_batch_size=4,
    )

    assert len(result) == len(df)
