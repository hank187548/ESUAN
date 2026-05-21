from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_phase1_models import (  # noqa: E402
    build_arg_parser,
    build_ridge_pipeline,
    calculate_metrics,
    fit_district_median_baseline,
    parse_gpu_ids,
    predict_district_median,
    validate_feature_config,
)


def test_calculate_metrics():
    metrics = calculate_metrics(np.array([100, 200]), np.array([110, 180]))

    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(15.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(250.0))
    assert metrics["mape"] == pytest.approx(10.0)
    assert metrics["medae"] == pytest.approx(15.0)


def test_validate_feature_config_blocks_numeric_target_leakage():
    df = pd.DataFrame({"unit_price_ping": [100.0], "building_area_m2": [30.0]})
    config = {
        "target_col": "unit_price_ping",
        "numeric_features": ["building_area_m2", "unit_price_ping"],
        "categorical_features": [],
        "drop_cols": [],
        "leakage_cols": ["unit_price_ping"],
    }

    with pytest.raises(ValueError, match="Leakage"):
        validate_feature_config(df, config)


def test_validate_feature_config_blocks_categorical_address_leakage():
    df = pd.DataFrame({"target": [100.0], "address_raw": ["臺北市測試路1號"]})
    config = {
        "target_col": "target",
        "numeric_features": [],
        "categorical_features": ["address_raw"],
        "drop_cols": [],
        "leakage_cols": ["address_raw"],
    }

    with pytest.raises(ValueError, match="Leakage"):
        validate_feature_config(df, config)


def test_district_median_uses_train_and_unseen_district_falls_back_to_global():
    train = pd.DataFrame(
        {
            "district": ["A", "A", "B"],
            "unit_price_ping": [100.0, 120.0, 200.0],
        }
    )
    valid = pd.DataFrame({"district": ["A", "C"]})

    model = fit_district_median_baseline(train, "unit_price_ping")
    pred = predict_district_median(model, valid)

    assert pred.tolist() == pytest.approx([110.0, 120.0])


def test_preprocessing_is_fit_on_train_only_and_handles_unknown_category():
    train_x = pd.DataFrame({"x": [1.0, np.nan, 3.0], "cat": ["A", "A", "A"]})
    train_y = pd.Series([10.0, 20.0, 30.0])
    valid_x = pd.DataFrame({"x": [999.0], "cat": ["B"]})

    pipeline = build_ridge_pipeline(["x"], ["cat"])
    pipeline.fit(train_x, train_y)

    preprocessor = pipeline.named_steps["preprocess"]
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    cat_onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    assert num_imputer.statistics_[0] == pytest.approx(2.0)
    assert cat_onehot.categories_[0].tolist() == ["A"]
    transformed = preprocessor.transform(valid_x)
    assert transformed.shape[0] == 1
    assert np.isfinite(pipeline.predict(valid_x)[0])


def test_hardware_argument_parsing():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--n-jobs",
            "48",
            "--parallel-folds",
            "true",
            "--num-parallel-jobs",
            "4",
            "--use-gpu",
            "auto",
            "--gpu-ids",
            "0,1,2,3",
        ]
    )

    assert args.n_jobs == 48
    assert args.parallel_folds is True
    assert args.num_parallel_jobs == 4
    assert args.use_gpu == "auto"
    assert parse_gpu_ids(args.gpu_ids) == [0, 1, 2, 3]

    for use_gpu in ["auto", "true", "false"]:
        parsed = parser.parse_args(["--use-gpu", use_gpu])
        assert parsed.use_gpu == use_gpu
