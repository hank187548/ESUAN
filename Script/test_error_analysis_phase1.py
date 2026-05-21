from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from error_analysis_phase1 import (  # noqa: E402
    assign_price_segment,
    calculate_regression_metrics,
    extract_top_errors,
)


def test_calculate_regression_metrics():
    metrics = calculate_regression_metrics([100, 200], [110, 180])

    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(15.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(250.0))
    assert metrics["mape"] == pytest.approx(10.0)
    assert metrics["bias"] == pytest.approx(-5.0)


def test_price_segment_assignment():
    assert assign_price_segment(40) == "0–50 萬/坪"
    assert assign_price_segment(60) == "50–80 萬/坪"
    assert assign_price_segment(100) == "80–120 萬/坪"
    assert assign_price_segment(130) == "120 萬/坪以上"


def test_bias_interpretation_negative_under_prediction():
    metrics = calculate_regression_metrics([100, 200], [90, 180])

    assert metrics["bias"] < 0


def test_bias_interpretation_positive_over_prediction():
    metrics = calculate_regression_metrics([100, 200], [110, 230])

    assert metrics["bias"] > 0


def test_extract_top_errors_under_and_over_predictions():
    predictions = pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "trade_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "district": ["中山區", "中山區", "中山區"],
            "building_type": ["住宅大樓", "住宅大樓", "住宅大樓"],
            "y_true": [100.0, 100.0, 100.0],
            "y_pred": [50.0, 120.0, 180.0],
            "error": [-50.0, 20.0, 80.0],
            "abs_error": [50.0, 20.0, 80.0],
            "ape": [50.0, 20.0, 80.0],
            "fold_id": [1, 1, 1],
            "test_start": ["2025-01-01", "2025-01-01", "2025-01-01"],
            "test_end": ["2025-03-31", "2025-03-31", "2025-03-31"],
        }
    )
    model_ready = pd.DataFrame({"id": ["A", "B", "C"], "building_age": [10, 20, 30]})

    under, over = extract_top_errors(predictions, model_ready, n=1)

    assert under.iloc[0]["id"] == "A"
    assert under.iloc[0]["error"] == pytest.approx(-50.0)
    assert over.iloc[0]["id"] == "C"
    assert over.iloc[0]["error"] == pytest.approx(80.0)
