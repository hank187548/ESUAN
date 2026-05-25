from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import joblib
except Exception:  # noqa: BLE001
    joblib = None


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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(value: str | Path) -> Path:
    text = str(value).replace("\\", os.sep)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def load_dataset(path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(path)
    logging.info("Reading dataset: %s", resolved)
    df = pd.read_csv(resolved, encoding="utf-8-sig", low_memory=False)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    logging.info("Dataset rows=%s cols=%s", len(df), len(df.columns))
    return df


def load_folds(path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(path)
    logging.info("Reading folds: %s", resolved)
    folds = pd.read_csv(resolved, encoding="utf-8-sig", low_memory=False)
    for col in ["train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]:
        folds[col] = pd.to_datetime(folds[col], errors="coerce")
    logging.info("Fold count=%s rows=%s", folds["fold_id"].nunique(), len(folds))
    return folds


def load_feature_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    logging.info("Reading feature config: %s", resolved)
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_feature_config(df: pd.DataFrame, config: dict[str, Any]) -> None:
    target_col = config.get("target_col")
    numeric = set(config.get("numeric_features", []))
    categorical = set(config.get("categorical_features", []))
    features = numeric | categorical
    forbidden = set(config.get("leakage_cols", [])) | set(config.get("drop_cols", [])) | LEAKAGE_FEATURES | {target_col}
    leaked = sorted(features & forbidden)
    if leaked:
        raise ValueError(f"Leakage/drop/target columns found in features: {leaked}")
    missing = sorted(col for col in features if col not in df.columns)
    if missing:
        raise ValueError(f"Configured features not found in dataset: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"Target column not found in dataset: {target_col}")


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_lightgbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict[str, Any],
) -> Pipeline:
    from lightgbm import LGBMRegressor

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LGBMRegressor(**params)),
        ]
    )


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


def get_fold_frame(df_by_id: pd.DataFrame, fold: pd.DataFrame, split: str) -> pd.DataFrame:
    ids = fold.loc[fold["split"].eq(split), "id"].astype(str)
    return df_by_id.loc[ids].reset_index(drop=True)


def summarize_metrics(metrics: pd.DataFrame, selection_split: str) -> pd.DataFrame:
    summary = (
        metrics.groupby(["param_id", "split"], as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            total_n=("n", "sum"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            weighted_mae=("mae", lambda s: np.average(s, weights=metrics.loc[s.index, "n"])),
            mean_rmse=("rmse", "mean"),
            weighted_rmse=("rmse", lambda s: np.average(s, weights=metrics.loc[s.index, "n"])),
            mean_mape=("mape", "mean"),
            weighted_mape=("mape", lambda s: np.average(s, weights=metrics.loc[s.index, "n"])),
            mean_medae=("medae", "mean"),
            mean_r2=("r2", "mean"),
        )
    )
    params = metrics.drop_duplicates("param_id")[
        ["param_id", "num_leaves", "min_child_samples", "learning_rate", "n_estimators", "subsample", "colsample_bytree"]
    ]
    summary = summary.merge(params, on="param_id", how="left")
    rank_mask = summary["split"].eq(selection_split)
    ranks = summary.loc[rank_mask, ["param_id", "mean_mae"]].sort_values("mean_mae").reset_index(drop=True)
    ranks["rank_by_selection_mae"] = np.arange(1, len(ranks) + 1)
    return summary.merge(ranks[["param_id", "rank_by_selection_mae"]], on="param_id", how="left")


def train_param_set(
    df_by_id: pd.DataFrame,
    folds: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
    params: dict[str, Any],
    param_id: int,
    save_models: bool,
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    metrics_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    for fold_id in sorted(folds["fold_id"].unique()):
        fold = folds.loc[folds["fold_id"].eq(fold_id)].copy()
        train_df = get_fold_frame(df_by_id, fold, "train")
        valid_df = get_fold_frame(df_by_id, fold, "valid")
        test_df = get_fold_frame(df_by_id, fold, "test")
        logging.info("param_id=%s fold=%s train=%s valid=%s test=%s", param_id, fold_id, len(train_df), len(valid_df), len(test_df))

        model = build_lightgbm_pipeline(numeric_features, categorical_features, params)
        model.fit(train_df[numeric_features + categorical_features], train_df[target_col])
        if save_models:
            if joblib is None:
                raise RuntimeError("joblib is required to save models.")
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / f"lightgbm_best_fold_{fold_id}.joblib")

        meta = {
            "param_id": param_id,
            "fold_id": int(fold_id),
            "train_start": fold["train_start"].iloc[0],
            "train_end": fold["train_end"].iloc[0],
            "valid_start": fold["valid_start"].iloc[0],
            "valid_end": fold["valid_end"].iloc[0],
            "test_start": fold["test_start"].iloc[0],
            "test_end": fold["test_end"].iloc[0],
            "num_leaves": params["num_leaves"],
            "min_child_samples": params["min_child_samples"],
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
        }
        for split, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            y_pred = model.predict(split_df[numeric_features + categorical_features])
            metric = calculate_metrics(split_df[target_col], y_pred)
            metrics_rows.append({**meta, "split": split, **metric})
            if save_models and split in {"valid", "test"}:
                part = split_df[["id", "trade_date", "district", "building_type"]].copy()
                part["param_id"] = param_id
                part["fold_id"] = int(fold_id)
                part["split"] = split
                part["model_name"] = "lightgbm_best"
                part["y_true"] = split_df[target_col].to_numpy(dtype=float)
                part["y_pred"] = y_pred
                part["abs_error"] = np.abs(part["y_pred"] - part["y_true"])
                part["ape"] = part["abs_error"] / part["y_true"] * 100
                for key in ["train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]:
                    part[key] = meta[key]
                pred_parts.append(part)
    return metrics_rows, pred_parts


def write_report(
    report_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    best_params: dict[str, Any],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    test_rank = summary.loc[summary["split"].eq(args.selection_split)].sort_values("mean_mae")
    lines = [
        "# LightGBM Parameter Search",
        "",
        "## Setup",
        "",
        f"- data path: `{resolve_path(args.data_path)}`",
        f"- folds path: `{resolve_path(args.folds_path)}`",
        f"- feature config: `{resolve_path(args.feature_config)}`",
        f"- selection split: `{args.selection_split}`",
        f"- selection metric: mean MAE",
        f"- parameter combinations: {summary['param_id'].nunique()}",
        "",
        "## Best Params",
        "",
        "```json",
        json.dumps(best_params, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Ranking",
        "",
        "| rank | param_id | num_leaves | min_child_samples | mean_mae | mean_rmse | mean_mape | mean_r2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test_rank.iterrows():
        lines.append(
            f"| {int(row['rank_by_selection_mae'])} | {int(row['param_id'])} | {int(row['num_leaves'])} | "
            f"{int(row['min_child_samples'])} | {row['mean_mae']:.4f} | {row['mean_rmse']:.4f} | "
            f"{row['mean_mape']:.4f} | {row['mean_r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Param Metrics",
            "",
            "| split | folds | mean_mae | mean_rmse | mean_mape | mean_r2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in best_summary.sort_values("split").iterrows():
        lines.append(
            f"| {row['split']} | {int(row['folds'])} | {row['mean_mae']:.4f} | "
            f"{row['mean_rmse']:.4f} | {row['mean_mape']:.4f} | {row['mean_r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Note",
            "",
            "- This search selects parameters using rolling test-period mean MAE, so it is a practical tuning result rather than an unbiased final holdout estimate.",
        ]
    )
    path = report_dir / "lightgbm_param_search_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search 16 LightGBM parameter combinations on rolling folds.")
    parser.add_argument("--data-path", default="data/processed/taipei_house_model_ready_v4_add.csv")
    parser.add_argument("--folds-path", default="data/processed/rolling_folds.csv")
    parser.add_argument("--feature-config", default="reports/feature_config_model_v4_add.json")
    parser.add_argument("--report-dir", default="reports/v4/lightgbm_search")
    parser.add_argument("--model-dir", default="models/v4_lightgbm_search")
    parser.add_argument("--output-pred-path", default="data/processed/v4_lightgbm_search_oof_predictions.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=48)
    parser.add_argument("--num-leaves-list", default="15,31,63,127")
    parser.add_argument("--min-child-samples-list", default="10,20,50,100")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--selection-split", choices=["valid", "test"], default="test")
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    report_dir = resolve_path(args.report_dir)
    model_dir = resolve_path(args.model_dir)
    output_pred_path = resolve_path(args.output_pred_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_pred_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_path)
    folds = load_folds(args.folds_path)
    config = load_feature_config(args.feature_config)
    validate_feature_config(df, config)
    target_col = config.get("target_col")
    numeric_features = list(config.get("numeric_features", []))
    categorical_features = list(config.get("categorical_features", []))
    df_by_id = df.copy()
    df_by_id["id"] = df_by_id["id"].astype(str)
    if df_by_id["id"].duplicated().any():
        raise ValueError("Dataset contains duplicated id values.")
    df_by_id = df_by_id.set_index("id", drop=False)
    folds["id"] = folds["id"].astype(str)

    grid = []
    param_id = 1
    for num_leaves in parse_int_list(args.num_leaves_list):
        for min_child_samples in parse_int_list(args.min_child_samples_list):
            grid.append(
                {
                    "param_id": param_id,
                    "params": {
                        "n_estimators": args.n_estimators,
                        "learning_rate": args.learning_rate,
                        "num_leaves": num_leaves,
                        "min_child_samples": min_child_samples,
                        "subsample": args.subsample,
                        "colsample_bytree": args.colsample_bytree,
                        "random_state": args.random_state,
                        "n_jobs": args.n_jobs,
                        "verbose": -1,
                    },
                }
            )
            param_id += 1
    logging.info("Searching %s LightGBM parameter combinations", len(grid))

    all_metrics: list[dict[str, Any]] = []
    for item in grid:
        logging.info("Searching param_id=%s params=%s", item["param_id"], item["params"])
        rows, _ = train_param_set(
            df_by_id=df_by_id,
            folds=folds,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_col=target_col,
            params=item["params"],
            param_id=item["param_id"],
            save_models=False,
            model_dir=model_dir,
        )
        all_metrics.extend(rows)
        pd.DataFrame(all_metrics).to_csv(report_dir / "lightgbm_param_search_metrics_by_fold.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(report_dir / "lightgbm_param_search_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    summary = summarize_metrics(metrics, args.selection_split)
    summary.to_csv(report_dir / "lightgbm_param_search_results.csv", index=False, encoding="utf-8-sig")

    best_row = summary.loc[summary["split"].eq(args.selection_split)].sort_values("mean_mae").iloc[0]
    best_param_id = int(best_row["param_id"])
    best_params = next(item["params"] for item in grid if item["param_id"] == best_param_id)
    with (report_dir / "lightgbm_best_params.json").open("w", encoding="utf-8") as f:
        json.dump({"param_id": best_param_id, "selection_split": args.selection_split, "params": best_params}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logging.info("Refitting best param_id=%s params=%s", best_param_id, best_params)
    best_metrics_rows, pred_parts = train_param_set(
        df_by_id=df_by_id,
        folds=folds,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
        params=best_params,
        param_id=best_param_id,
        save_models=True,
        model_dir=model_dir,
    )
    best_metrics = pd.DataFrame(best_metrics_rows)
    best_metrics.to_csv(report_dir / "lightgbm_best_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    best_summary = summarize_metrics(best_metrics, args.selection_split)
    best_summary.to_csv(report_dir / "lightgbm_best_metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_parts, ignore_index=True).to_csv(output_pred_path, index=False, encoding="utf-8-sig")
    write_report(report_dir, args, summary, best_summary, {"param_id": best_param_id, **best_params})
    logging.info("Wrote search results: %s", report_dir)
    logging.info("Wrote best OOF predictions: %s", output_pred_path)


if __name__ == "__main__":
    main()
