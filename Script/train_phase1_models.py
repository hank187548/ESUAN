from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import joblib
except Exception:  # noqa: BLE001
    joblib = None


DEFAULT_DATA_PATH = Path("data/processed/taipei_house_model_ready.csv")
DEFAULT_FOLDS_PATH = Path("data/processed/rolling_folds.csv")
DEFAULT_FEATURE_CONFIG = Path("reports/feature_config_model_v1.json")
DEFAULT_OUTPUT_PRED_PATH = Path("data/processed/phase1_oof_predictions.csv")
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_MODEL_DIR = Path("models/phase1")

LEAKAGE_FEATURES = {
    "total_price",
    "total_price_wan",
    "unit_price_m2",
    "unit_price_ping",
    "parking_price",
    "address_raw",
    "note_raw",
    "id",
    "transfer_id",
    "source_release",
    "source_order",
    "source_file",
    "source_folder",
}

MODEL_NAMES = [
    "naive_global_median",
    "naive_district_median",
    "naive_district_building_type_median",
    "ridge_regression",
    "tree_model",
]


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


def parse_gpu_ids(value: str) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip() != ""]


def resolve_path(value: str | Path) -> Path:
    text = str(value).replace("\\", os.sep)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    path = resolve_path(data_path)
    logging.info("Reading dataset: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    logging.info("Dataset rows=%s columns=%s", len(df), len(df.columns))
    return df


def load_folds(folds_path: str | Path) -> pd.DataFrame:
    path = resolve_path(folds_path)
    logging.info("Reading rolling folds: %s", path)
    folds = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for col in ["train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]:
        folds[col] = pd.to_datetime(folds[col], errors="coerce")
    logging.info("Fold rows=%s fold_count=%s", len(folds), folds["fold_id"].nunique())
    return folds


def load_feature_config(feature_config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(feature_config_path)
    logging.info("Reading feature config: %s", path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_feature_config(df: pd.DataFrame, feature_config: dict[str, Any]) -> None:
    target_col = feature_config.get("target_col")
    numeric_features = set(feature_config.get("numeric_features", []))
    categorical_features = set(feature_config.get("categorical_features", []))
    feature_cols = numeric_features | categorical_features
    forbidden = set(feature_config.get("leakage_cols", [])) | set(feature_config.get("drop_cols", [])) | LEAKAGE_FEATURES
    forbidden.add(target_col)
    leaked = sorted(feature_cols & forbidden)
    if leaked:
        raise ValueError(f"Leakage/drop/target columns found in features: {leaked}")

    missing_features = sorted(col for col in feature_cols if col not in df.columns)
    if missing_features:
        raise ValueError(f"Configured features not found in dataset: {missing_features}")
    if target_col not in df.columns:
        raise ValueError(f"Target column not found in dataset: {target_col}")


def calculate_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    n = int(len(y_true_arr))
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "medae": np.nan, "r2": np.nan}
    abs_error = np.abs(y_true_arr - y_pred_arr)
    squared_error = (y_true_arr - y_pred_arr) ** 2
    sst = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    sse = np.sum(squared_error)
    return {
        "n": n,
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mape": float(np.mean(abs_error / y_true_arr) * 100),
        "medae": float(np.median(abs_error)),
        "r2": float(1 - sse / sst) if sst > 0 else np.nan,
    }


def fit_global_median_baseline(train_df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    return {"global_median": float(train_df[target_col].median())}


def predict_global_median(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), model["global_median"], dtype=float)


def fit_district_median_baseline(train_df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    global_median = float(train_df[target_col].median())
    district_medians = train_df.groupby("district", dropna=False)[target_col].median().to_dict()
    return {
        "global_median": global_median,
        "district_medians": {str(k): float(v) for k, v in district_medians.items()},
    }


def predict_district_median(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    medians = model["district_medians"]
    global_median = model["global_median"]
    return df["district"].astype(str).map(medians).fillna(global_median).to_numpy(dtype=float)


def fit_district_building_type_median_baseline(train_df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    district_model = fit_district_median_baseline(train_df, target_col)
    combo = (
        train_df.groupby(["district", "building_type"], dropna=False)[target_col]
        .median()
        .to_dict()
    )
    return {
        "global_median": district_model["global_median"],
        "district_medians": district_model["district_medians"],
        "combo_medians": {f"{str(k[0])}||{str(k[1])}": float(v) for k, v in combo.items()},
    }


def predict_district_building_type_median(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    global_median = model["global_median"]
    district_medians = model["district_medians"]
    combo_medians = model["combo_medians"]
    keys = df["district"].astype(str) + "||" + df["building_type"].astype(str)
    combo_pred = keys.map(combo_medians)
    district_pred = df["district"].astype(str).map(district_medians)
    return combo_pred.fillna(district_pred).fillna(global_median).to_numpy(dtype=float)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_ridge_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]), categorical_features),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=1.0))])


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _build_tree_estimator(candidate: dict[str, Any]):
    backend = candidate["backend"]
    params = candidate["params"]
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**params)
    if backend == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    if backend == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**params)
    raise ValueError(f"Unsupported tree backend: {backend}")


def build_tree_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    candidate: dict[str, Any],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]), categorical_features),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline([("preprocess", preprocessor), ("model", _build_tree_estimator(candidate))])


def detect_gpu_info() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        idx, name = line.split(",", 1)
        gpus.append({"index": int(idx.strip()), "name": name.strip()})
    return gpus


def detect_hardware() -> dict[str, Any]:
    cpu_cores = os.cpu_count() or 1
    gpus = detect_gpu_info()
    return {"cpu_cores": cpu_cores, "gpu_count": len(gpus), "gpu_info": gpus}


def resolve_n_jobs(n_jobs: int, parallel_folds: bool, num_parallel_jobs: int, cpu_cores: int) -> int:
    requested = cpu_cores if n_jobs == -1 else max(1, n_jobs)
    if not parallel_folds:
        return max(1, min(requested, cpu_cores))
    per_job_cap = max(1, cpu_cores // max(1, num_parallel_jobs))
    return max(1, min(requested, per_job_cap))


def assign_gpu_to_job(job_index: int, gpu_ids: list[int]) -> int | None:
    if not gpu_ids:
        return None
    return gpu_ids[job_index % len(gpu_ids)]


def build_tree_candidates(
    use_gpu: str,
    gpu_available: bool,
    gpu_id: int | None,
    n_jobs: int,
    random_state: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    wants_gpu = use_gpu in {"auto", "true"} and gpu_available and gpu_id is not None

    if _module_available("lightgbm"):
        if wants_gpu:
            candidates.append(
                {
                    "backend": "lightgbm",
                    "device": "gpu",
                    "label": "LightGBM GPU",
                    "params": {
                        "n_estimators": 1000,
                        "learning_rate": 0.03,
                        "num_leaves": 31,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_state": random_state,
                        "n_jobs": n_jobs,
                        "verbose": -1,
                        "device_type": "gpu",
                        "gpu_device_id": gpu_id,
                    },
                }
            )
        candidates.append(
            {
                "backend": "lightgbm",
                "device": "cpu",
                "label": "LightGBM CPU",
                "params": {
                    "n_estimators": 1000,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": random_state,
                    "n_jobs": n_jobs,
                    "verbose": -1,
                },
            }
        )

    if _module_available("xgboost"):
        if wants_gpu:
            candidates.append(
                {
                    "backend": "xgboost",
                    "device": "gpu",
                    "label": "XGBoost GPU modern",
                    "params": {
                        "n_estimators": 1000,
                        "learning_rate": 0.03,
                        "max_depth": 6,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "objective": "reg:squarederror",
                        "random_state": random_state,
                        "n_jobs": n_jobs,
                        "tree_method": "hist",
                        "device": f"cuda:{gpu_id}",
                    },
                }
            )
            candidates.append(
                {
                    "backend": "xgboost",
                    "device": "gpu",
                    "label": "XGBoost GPU legacy",
                    "params": {
                        "n_estimators": 1000,
                        "learning_rate": 0.03,
                        "max_depth": 6,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "objective": "reg:squarederror",
                        "random_state": random_state,
                        "n_jobs": n_jobs,
                        "tree_method": "gpu_hist",
                        "gpu_id": gpu_id,
                    },
                }
            )
        candidates.append(
            {
                "backend": "xgboost",
                "device": "cpu",
                "label": "XGBoost CPU",
                "params": {
                    "n_estimators": 1000,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "reg:squarederror",
                    "random_state": random_state,
                    "n_jobs": n_jobs,
                    "tree_method": "hist",
                },
            }
        )

    candidates.append(
        {
            "backend": "hist_gradient_boosting",
            "device": "cpu",
            "label": "sklearn HistGradientBoostingRegressor",
            "params": {"random_state": random_state, "max_iter": 300, "learning_rate": 0.05},
        }
    )
    return candidates


def evaluate_model(
    model_name: str,
    fold_id: int,
    split: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    bounds: dict[str, pd.Timestamp],
    tree_backend: str = "",
    tree_device: str = "",
) -> dict[str, Any]:
    metrics = calculate_metrics(y_true, y_pred)
    return {
        "model_name": model_name,
        "fold_id": fold_id,
        "split": split,
        **metrics,
        **{k: v.strftime("%Y-%m-%d") for k, v in bounds.items()},
        "tree_model_backend": tree_backend,
        "tree_model_device": tree_device,
    }


def _prediction_rows(
    model_name: str,
    fold_id: int,
    split: str,
    split_df: pd.DataFrame,
    y_pred: np.ndarray,
    target_col: str,
    bounds: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    output = split_df[["id", "trade_date", "district", "building_type", target_col]].copy()
    output.insert(0, "fold_id", fold_id)
    output.insert(1, "split", split)
    output["y_true"] = output[target_col].astype(float)
    output["y_pred"] = y_pred.astype(float)
    output["model_name"] = model_name
    output["abs_error"] = np.abs(output["y_true"] - output["y_pred"])
    output["ape"] = output["abs_error"] / output["y_true"] * 100
    output = output.drop(columns=[target_col])
    for col, value in bounds.items():
        output[col] = value.strftime("%Y-%m-%d")
    return output[
        [
            "fold_id",
            "split",
            "id",
            "trade_date",
            "district",
            "building_type",
            "y_true",
            "y_pred",
            "model_name",
            "abs_error",
            "ape",
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
        ]
    ]


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(model, path)
    else:
        with path.open("wb") as f:
            pickle.dump(model, f)


def _save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _get_fold_data(df: pd.DataFrame, folds: pd.DataFrame, fold_id: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Timestamp]]:
    fold_rows = folds.loc[folds["fold_id"].eq(fold_id)].copy()
    if fold_rows.empty:
        raise ValueError(f"fold_id not found: {fold_id}")
    first = fold_rows.iloc[0]
    bounds = {
        "train_start": pd.Timestamp(first["train_start"]),
        "train_end": pd.Timestamp(first["train_end"]),
        "valid_start": pd.Timestamp(first["valid_start"]),
        "valid_end": pd.Timestamp(first["valid_end"]),
        "test_start": pd.Timestamp(first["test_start"]),
        "test_end": pd.Timestamp(first["test_end"]),
    }
    indexed = df.set_index("id", drop=False)
    splits = {}
    for split in ["train", "valid", "test"]:
        ids = fold_rows.loc[fold_rows["split"].eq(split), "id"]
        splits[split] = indexed.loc[ids].copy()
    return splits, bounds


def _fit_tree_with_fallback(
    train_df: pd.DataFrame,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    candidates: list[dict[str, Any]],
) -> tuple[Pipeline, dict[str, Any], list[str]]:
    errors = []
    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target_col].astype(float)
    for candidate in candidates:
        try:
            logging.info("Trying tree_model backend=%s device=%s", candidate["backend"], candidate["device"])
            pipeline = build_tree_pipeline(numeric_features, categorical_features, candidate)
            pipeline.fit(X_train, y_train)
            return pipeline, candidate, errors
        except Exception as exc:  # noqa: BLE001
            message = f"{candidate['label']} failed: {type(exc).__name__}: {exc}"
            logging.warning(message)
            errors.append(message)
    raise RuntimeError("All tree_model candidates failed: " + " | ".join(errors))


def train_one_fold(
    fold_id: int,
    df: pd.DataFrame,
    folds: pd.DataFrame,
    feature_config: dict[str, Any],
    model_dir: str | Path,
    random_state: int,
    n_jobs: int,
    use_gpu: str,
    gpu_id: int | None,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    logging.info("Training fold %s", fold_id)
    target_col = feature_config["target_col"]
    numeric_features = feature_config["numeric_features"]
    categorical_features = feature_config["categorical_features"]
    feature_cols = numeric_features + categorical_features
    splits, bounds = _get_fold_data(df, folds, fold_id)
    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]

    metrics_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    tree_info = {"backend": "", "device": "", "fallback_reasons": []}

    baseline_models = {
        "naive_global_median": (
            fit_global_median_baseline(train_df, target_col),
            predict_global_median,
        ),
        "naive_district_median": (
            fit_district_median_baseline(train_df, target_col),
            predict_district_median,
        ),
        "naive_district_building_type_median": (
            fit_district_building_type_median_baseline(train_df, target_col),
            predict_district_building_type_median,
        ),
    }
    _save_json(
        {
            "global_median": baseline_models["naive_global_median"][0],
            "district_median": baseline_models["naive_district_median"][0],
            "district_building_type_median": baseline_models["naive_district_building_type_median"][0],
        },
        Path(model_dir) / f"naive_medians_fold_{fold_id}.json",
    )

    for model_name, (baseline_model, predict_func) in baseline_models.items():
        for split_name, split_df in splits.items():
            y_pred = predict_func(baseline_model, split_df)
            metrics_rows.append(evaluate_model(model_name, fold_id, split_name, split_df[target_col], y_pred, bounds))
            if split_name in {"valid", "test"}:
                oof_frames.append(_prediction_rows(model_name, fold_id, split_name, split_df, y_pred, target_col, bounds))

    logging.info("Fold %s training ridge_regression", fold_id)
    ridge = build_ridge_pipeline(numeric_features, categorical_features)
    ridge.fit(train_df[feature_cols], train_df[target_col].astype(float))
    save_model(ridge, Path(model_dir) / f"ridge_regression_fold_{fold_id}.joblib")
    for split_name, split_df in splits.items():
        y_pred = ridge.predict(split_df[feature_cols])
        metrics_rows.append(evaluate_model("ridge_regression", fold_id, split_name, split_df[target_col], y_pred, bounds))
        if split_name in {"valid", "test"}:
            oof_frames.append(_prediction_rows("ridge_regression", fold_id, split_name, split_df, y_pred, target_col, bounds))

    logging.info("Fold %s training tree_model", fold_id)
    gpu_available = use_gpu in {"auto", "true"} and hardware.get("gpu_count", 0) > 0 and gpu_id is not None
    candidates = build_tree_candidates(use_gpu, gpu_available, gpu_id, n_jobs, random_state)
    tree, selected_candidate, fallback_errors = _fit_tree_with_fallback(
        train_df, target_col, numeric_features, categorical_features, candidates
    )
    tree_info = {
        "backend": selected_candidate["backend"],
        "device": selected_candidate["device"],
        "label": selected_candidate["label"],
        "fallback_reasons": fallback_errors,
    }
    save_model(tree, Path(model_dir) / f"tree_model_fold_{fold_id}.joblib")
    for split_name, split_df in splits.items():
        y_pred = tree.predict(split_df[feature_cols])
        metrics_rows.append(
            evaluate_model(
                "tree_model",
                fold_id,
                split_name,
                split_df[target_col],
                y_pred,
                bounds,
                tree_backend=selected_candidate["backend"],
                tree_device=selected_candidate["device"],
            )
        )
        if split_name in {"valid", "test"}:
            oof_frames.append(_prediction_rows("tree_model", fold_id, split_name, split_df, y_pred, target_col, bounds))

    test_metrics = [row for row in metrics_rows if row["split"] == "test" and row["model_name"] == "tree_model"][0]
    logging.info("Fold %s tree_model test MAE=%.4f MAPE=%.4f", fold_id, test_metrics["mae"], test_metrics["mape"])
    return {
        "metrics": metrics_rows,
        "oof": pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame(),
        "tree_info": tree_info,
    }


def write_metrics(metrics: pd.DataFrame, report_dir: str | Path) -> tuple[Path, Path, pd.DataFrame]:
    report_dir = resolve_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "phase1_model_metrics.csv"
    summary_path = report_dir / "phase1_model_metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary = (
        metrics.groupby(["model_name", "split"], dropna=False)
        .agg(
            folds=("fold_id", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mape=("mape", "mean"),
            std_mape=("mape", "std"),
            mean_medae=("medae", "mean"),
            mean_r2=("r2", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info("Wrote metrics: %s", metrics_path)
    logging.info("Wrote metrics summary: %s", summary_path)
    return metrics_path, summary_path, summary


def write_oof_predictions(oof: pd.DataFrame, output_pred_path: str | Path) -> Path:
    path = resolve_path(output_pred_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("Wrote OOF predictions: %s rows=%s", path, len(oof))
    return path


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    headers = df.columns.astype(str).tolist()
    rows = df.values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(_format_scalar(cell) for cell in row) + " |" for row in rows],
        ]
    )


def _format_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def write_report(
    report_dir: str | Path,
    data_path: str | Path,
    folds_path: str | Path,
    feature_config_path: str | Path,
    output_pred_path: str | Path,
    model_dir: str | Path,
    feature_config: dict[str, Any],
    metrics: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    tree_infos: list[dict[str, Any]],
    hardware: dict[str, Any],
    args: argparse.Namespace,
    fold_ids: list[int],
) -> Path:
    report_dir = resolve_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "phase1_model_report.md"

    test_summary = metrics_summary.loc[metrics_summary["split"].eq("test")].copy()
    best_row = test_summary.sort_values("mean_mae").head(1)
    best_model = best_row.iloc[0]["model_name"] if not best_row.empty else ""
    tree_test = test_summary.loc[test_summary["model_name"].eq("tree_model")]
    baseline_test = test_summary.loc[test_summary["model_name"].eq("naive_district_building_type_median")]
    if not tree_test.empty and not baseline_test.empty:
        tree_mae = float(tree_test.iloc[0]["mean_mae"])
        base_mae = float(baseline_test.iloc[0]["mean_mae"])
        tree_mape = float(tree_test.iloc[0]["mean_mape"])
        base_mape = float(baseline_test.iloc[0]["mean_mape"])
        mae_improvement = base_mae - tree_mae
        mape_improvement = base_mape - tree_mape
    else:
        mae_improvement = np.nan
        mape_improvement = np.nan

    tree_backends = sorted(set(info.get("backend", "") for info in tree_infos if info.get("backend")))
    tree_devices = sorted(set(info.get("device", "") for info in tree_infos if info.get("device")))
    fallback_reasons = sorted(set(reason for info in tree_infos for reason in info.get("fallback_reasons", [])))
    gpu_names = [gpu["name"] for gpu in hardware.get("gpu_info", [])]
    test_starts = pd.to_datetime(metrics.loc[metrics["split"].eq("test"), "test_start"], errors="coerce")
    test_quarters = test_starts.dt.to_period("Q").astype(str).sort_values()
    earliest_test_quarter = test_quarters.iloc[0] if not test_quarters.empty else ""
    latest_test_quarter = test_quarters.iloc[-1] if not test_quarters.empty else ""

    test_fold_metrics = metrics.loc[metrics["split"].eq("test"), [
        "fold_id", "test_start", "test_end", "model_name", "mae", "rmse", "mape", "r2"
    ]].rename(columns={"mae": "test_mae", "rmse": "test_rmse", "mape": "test_mape", "r2": "test_r2"})

    content = [
        "# Phase 1B Model Report",
        "",
        "## 1. 實驗設定",
        "",
        f"- data path: `{resolve_path(data_path)}`",
        f"- rolling folds path: `{resolve_path(folds_path)}`",
        f"- feature config path: `{resolve_path(feature_config_path)}`",
        f"- target: `{feature_config['target_col']}`",
        f"- numeric features: {len(feature_config['numeric_features'])}",
        f"- categorical features: {len(feature_config['categorical_features'])}",
        f"- models: {', '.join(MODEL_NAMES)}",
        "",
        "## 2. Hardware / Runtime",
        "",
        f"- CPU cores detected: {hardware.get('cpu_cores')}",
        f"- n_jobs: {args.n_jobs}",
        f"- resolved per-model n_jobs: {resolve_n_jobs(args.n_jobs, args.parallel_folds, args.num_parallel_jobs, hardware.get('cpu_cores', 1))}",
        f"- parallel_folds: {str(args.parallel_folds).lower()}",
        f"- num_parallel_jobs: {args.num_parallel_jobs}",
        f"- use_gpu setting: {args.use_gpu}",
        f"- detected_gpu_count: {hardware.get('gpu_count')}",
        f"- gpu_ids: {args.gpu_ids}",
        f"- GPU names: {', '.join(gpu_names) if gpu_names else '(none detected)'}",
        f"- tree_model_backend: {', '.join(tree_backends) if tree_backends else ''}",
        f"- tree_model_device: {', '.join(tree_devices) if tree_devices else ''}",
        f"- GPU used successfully: {'yes' if 'gpu' in tree_devices else 'no'}",
        f"- GPU/CPU fallback reasons: {' | '.join(fallback_reasons) if fallback_reasons else '(none)'}",
        "- early stopping: not used in Phase 1B; valid split is evaluated only.",
        "",
        "## 3. 資料切分",
        "",
        f"- fold count: {len(fold_ids)}",
        f"- earliest test quarter: {earliest_test_quarter}",
        f"- latest test quarter: {latest_test_quarter}",
        f"- last test quarter: {latest_test_quarter}",
        "- validation: time-based rolling folds",
        "- random split: not used",
        "",
        "## 4. Leakage Control",
        "",
        "- preprocessing is fit only on each fold's train split.",
        "- median baselines are calculated only from each fold's train split.",
        "- valid/test data are never used to fit imputers, scalers, encoders, medians, or models.",
        "- `total_price`, `unit_price_m2`, `unit_price_ping`, and `parking_price` are not used as features.",
        "- `source_release` is a release batch marker and is not used as a feature.",
        "",
        "## 5. 模型結果",
        "",
        "### Test Summary",
        "",
        _df_to_markdown_table(test_summary),
        "",
        f"- best model by mean test MAE: `{best_model}`",
        f"- tree_model vs naive_district_building_type_median MAE improvement: {_format_scalar(mae_improvement)}",
        f"- tree_model vs naive_district_building_type_median MAPE improvement: {_format_scalar(mape_improvement)}",
        "",
        "## 6. 每個 Fold 的 Test 結果",
        "",
        _df_to_markdown_table(test_fold_metrics),
        "",
        "## 7. 初步結論",
        "",
        "- naive baselines provide district and building-type market reference points.",
        "- Ridge regression provides a linear baseline with train-only preprocessing.",
        "- tree_model is the first nonlinear main model for this time-based validation setup.",
        "- Next phase can focus on error analysis and SHAP.",
        "",
        "## 8. Outputs",
        "",
        f"- metrics: `{report_dir / 'phase1_model_metrics.csv'}`",
        f"- metrics summary: `{report_dir / 'phase1_model_metrics_summary.csv'}`",
        f"- OOF predictions: `{resolve_path(output_pred_path)}`",
        f"- model dir: `{resolve_path(model_dir)}`",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    logging.info("Wrote model report: %s", path)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phase 1B baseline and first main models.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--folds-path", default=str(DEFAULT_FOLDS_PATH))
    parser.add_argument("--feature-config", default=str(DEFAULT_FEATURE_CONFIG))
    parser.add_argument("--output-pred-path", default=str(DEFAULT_OUTPUT_PRED_PATH))
    parser.add_argument("--report-dir", default=str(Path("reports")))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--parallel-folds", type=str_to_bool, default=False)
    parser.add_argument("--num-parallel-jobs", type=int, default=4)
    parser.add_argument("--use-gpu", choices=["true", "false", "auto"], default="auto")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    hardware = detect_hardware()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    fold_n_jobs = resolve_n_jobs(args.n_jobs, args.parallel_folds, args.num_parallel_jobs, hardware["cpu_cores"])
    model_dir = resolve_path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Hardware: cpu_cores=%s gpu_count=%s gpu_ids=%s", hardware["cpu_cores"], hardware["gpu_count"], gpu_ids)
    logging.info("Runtime: n_jobs=%s resolved_fold_n_jobs=%s parallel_folds=%s", args.n_jobs, fold_n_jobs, args.parallel_folds)

    df = load_dataset(args.data_path)
    folds = load_folds(args.folds_path)
    feature_config = load_feature_config(args.feature_config)
    validate_feature_config(df, feature_config)
    logging.info("Features: numeric=%s categorical=%s", len(feature_config["numeric_features"]), len(feature_config["categorical_features"]))

    fold_ids = sorted(folds["fold_id"].unique().tolist())
    if args.parallel_folds:
        if joblib is None:
            raise RuntimeError("parallel_folds=true requires joblib.")
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.num_parallel_jobs)(
            delayed(train_one_fold)(
                fold_id,
                df,
                folds,
                feature_config,
                model_dir,
                args.random_state,
                fold_n_jobs,
                args.use_gpu,
                assign_gpu_to_job(job_index, gpu_ids),
                hardware,
            )
            for job_index, fold_id in enumerate(fold_ids)
        )
    else:
        results = []
        for job_index, fold_id in enumerate(fold_ids):
            results.append(
                train_one_fold(
                    fold_id,
                    df,
                    folds,
                    feature_config,
                    model_dir,
                    args.random_state,
                    fold_n_jobs,
                    args.use_gpu,
                    assign_gpu_to_job(job_index if args.parallel_folds else 0, gpu_ids),
                    hardware,
                )
            )

    metrics = pd.DataFrame([row for result in results for row in result["metrics"]])
    oof = pd.concat([result["oof"] for result in results], ignore_index=True)
    tree_infos = [result["tree_info"] for result in results]

    metrics_path, summary_path, metrics_summary = write_metrics(metrics, args.report_dir)
    oof_path = write_oof_predictions(oof, args.output_pred_path)
    write_report(
        report_dir=args.report_dir,
        data_path=args.data_path,
        folds_path=args.folds_path,
        feature_config_path=args.feature_config,
        output_pred_path=oof_path,
        model_dir=model_dir,
        feature_config=feature_config,
        metrics=metrics,
        metrics_summary=metrics_summary,
        tree_infos=tree_infos,
        hardware=hardware,
        args=args,
        fold_ids=fold_ids,
    )
    logging.info("Phase 1B complete: metrics=%s summary=%s oof=%s", metrics_path, summary_path, oof_path)


if __name__ == "__main__":
    main()
