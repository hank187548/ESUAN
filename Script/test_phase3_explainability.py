from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_phase3_explainability import (  # noqa: E402
    add_residual_columns,
    assign_price_segment,
    calculate_ic,
    classify_feature_group,
    detect_high_correlation_pairs,
    ensure_output_dirs,
    summarize_residual_group,
)


def test_pearson_spearman_ic_ordering():
    y_true = [1, 2, 3, 4]
    same_order = [10, 20, 30, 40]
    reverse_order = [40, 30, 20, 10]

    same = calculate_ic(y_true, same_order)
    reverse = calculate_ic(y_true, reverse_order)

    assert same["spearman_rank_ic"] == pytest.approx(1.0)
    assert reverse["spearman_rank_ic"] == pytest.approx(-1.0)
    assert same["pearson_ic"] == pytest.approx(1.0)


def test_residual_calculation():
    df = pd.DataFrame({"y_true": [100.0, 200.0], "y_pred": [110.0, 180.0]})
    residuals = add_residual_columns(df)
    summary = summarize_residual_group(residuals)

    assert residuals["error"].tolist() == [10.0, -20.0]
    assert residuals["abs_error"].tolist() == [10.0, 20.0]
    assert residuals["ape"].tolist() == [10.0, 10.0]
    assert summary["bias"] == pytest.approx(-5.0)


def test_price_segment_assignment():
    assert assign_price_segment(40) == "0-50"
    assert assign_price_segment(60) == "50-80"
    assert assign_price_segment(100) == "80-120"
    assert assign_price_segment(130) == "120+"


def test_feature_group_assignment():
    assert classify_feature_group("comp_365d_median_price") == "comparable_sales"
    assert classify_feature_group("district_median_price_180d") == "time_aware_market"
    assert classify_feature_group("building_age") == "basic_housing"


def test_high_correlation_pair_detection():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 1, 3, 2, 4],
        }
    )
    pairs = detect_high_correlation_pairs(df.corr(), threshold=0.90)

    assert ((pairs["feature_1"].eq("a") & pairs["feature_2"].eq("b")) | (pairs["feature_1"].eq("b") & pairs["feature_2"].eq("a"))).any()


def test_output_directory_creation(tmp_path):
    root = tmp_path / "analysis" / "phase3_explainability"
    dirs = ensure_output_dirs(root)

    for key in ["feature_importance", "shap", "prediction_ic", "residual_analysis", "correlation", "summary"]:
        assert dirs[key].exists()
        assert dirs[key].is_dir()
