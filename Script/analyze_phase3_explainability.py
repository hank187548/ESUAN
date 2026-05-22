from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None


DEFAULT_DATASET_PATH = Path("data/processed/taipei_house_model_ready_v3.csv")
DEFAULT_PRED_PATH = Path("data/processed/phase3_oof_predictions.csv")
DEFAULT_FOLDS_PATH = Path("data/processed/rolling_folds.csv")
DEFAULT_FEATURE_CONFIG = Path("reports/feature_config_model_v3.json")
DEFAULT_FEATURE_CONFIG_FALLBACK = Path("reports/v3/feature_config_model_v3.json")
DEFAULT_MODEL_DIR = Path("models/phase3")
DEFAULT_OUTPUT_DIR = Path("analysis/phase3_explainability")

TARGET_COL = "unit_price_ping"
PRICE_SEGMENTS = [
    ("0-50", 0, 50),
    ("50-80", 50, 80),
    ("80-120", 80, 120),
    ("120+", 120, np.inf),
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


def resolve_path(value: str | Path) -> Path:
    text = str(value).replace("\\", os.sep)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = resolve_path(output_dir)
    dirs = {
        "root": root,
        "feature_importance": root / "feature_importance",
        "shap": root / "shap",
        "prediction_ic": root / "prediction_ic",
        "residual_analysis": root / "residual_analysis",
        "correlation": root / "correlation",
        "summary": root / "summary",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_feature_config(path: str | Path, fallback_path: str | Path = DEFAULT_FEATURE_CONFIG_FALLBACK) -> tuple[dict[str, Any], Path]:
    primary = resolve_path(path)
    fallback = resolve_path(fallback_path)
    config_path = primary if primary.exists() else fallback
    if not config_path.exists():
        raise FileNotFoundError(f"Feature config not found: {primary} or {fallback}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f), config_path


def load_predictions(pred_path: str | Path, model_name: str, split: str) -> pd.DataFrame:
    path = resolve_path(pred_path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"model_name", "split", "id", "trade_date", "district", "building_type", "y_true", "y_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction file missing required columns: {missing}")
    work = df.loc[df["model_name"].eq(model_name) & df["split"].eq(split)].copy()
    if work.empty:
        raise ValueError(f"No predictions found for model_name={model_name}, split={split}")
    for col in ["trade_date", "test_start", "test_end"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")
    work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
    work["y_pred"] = pd.to_numeric(work["y_pred"], errors="coerce")
    work = add_residual_columns(work)
    if "test_start" in work.columns:
        work["test_quarter"] = work["test_start"].dt.to_period("Q").astype(str)
    else:
        work["test_quarter"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.to_period("Q").astype(str)
    logging.info("Loaded predictions rows=%s", len(work))
    return work


def load_dataset_features(dataset_path: str | Path, feature_config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(dataset_path)
    usecols = ["id", TARGET_COL] + feature_config["numeric_features"] + feature_config["categorical_features"]
    usecols = list(dict.fromkeys(usecols))
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    usecols = [col for col in usecols if col in header]
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    for col in [TARGET_COL] + feature_config["numeric_features"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logging.info("Loaded dataset feature frame rows=%s cols=%s", len(df), len(df.columns))
    return df


def assign_price_segment(value: Any) -> str:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not np.isfinite(price):
        return "missing"
    for label, lower, upper in PRICE_SEGMENTS:
        if lower <= price < upper:
            return label
    return "missing"


def add_residual_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["error"] = output["y_pred"] - output["y_true"]
    output["abs_error"] = output["error"].abs()
    output["ape"] = np.where(output["y_true"].ne(0), output["abs_error"] / output["y_true"] * 100, np.nan)
    return output


def regression_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan, "bias": np.nan}
    error = y_pred - y_true
    abs_error = np.abs(error)
    sse = np.sum(error**2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(abs_error / y_true) * 100),
        "r2": float(1 - sse / sst) if sst > 0 else np.nan,
        "bias": float(np.mean(error)),
    }


def calculate_ic(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true_series = pd.Series(y_true, dtype="float64")
    y_pred_series = pd.Series(y_pred, dtype="float64")
    mask = y_true_series.notna() & y_pred_series.notna()
    y_true_series = y_true_series[mask]
    y_pred_series = y_pred_series[mask]
    if len(y_true_series) < 2:
        return {"n": int(len(y_true_series)), "pearson_ic": np.nan, "spearman_rank_ic": np.nan}
    pearson = float(y_pred_series.corr(y_true_series, method="pearson"))
    if importlib.util.find_spec("scipy") is not None:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(y_pred_series, y_true_series, nan_policy="omit").statistic)
    else:
        spearman = float(y_pred_series.rank().corr(y_true_series.rank(), method="pearson"))
    return {"n": int(len(y_true_series)), "pearson_ic": pearson, "spearman_rank_ic": spearman}


def summarize_residual_group(df: pd.DataFrame) -> dict[str, float]:
    metrics = regression_metrics(df)
    return {
        **metrics,
        "y_true_mean": float(df["y_true"].mean()) if not df.empty else np.nan,
        "y_pred_mean": float(df["y_pred"].mean()) if not df.empty else np.nan,
        "error_std": float(df["error"].std()) if not df.empty else np.nan,
        "abs_error_median": float(df["abs_error"].median()) if not df.empty else np.nan,
    }


def summarize_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False):
        rows.append({group_col: value, **summarize_residual_group(group)})
    return pd.DataFrame(rows)


def classify_feature_group(feature: str) -> str:
    name = str(feature)
    clean = name.removeprefix("num__").removeprefix("cat__")
    if clean.startswith("comp_") or "comp_" in clean:
        return "comparable_sales"
    if clean.startswith(("district_median_", "district_type_", "district_count_", "district_price_", "district_type_price_")):
        return "time_aware_market"
    if clean.startswith("district") or clean.startswith("trade_") or "trade_yq" in clean:
        return "location_time"
    if any(token in clean for token in ["physical_condition_flag", "renovation_flag", "broad_note_flag"]):
        return "note_flags"
    if any(
        token in clean
        for token in [
            "building_",
            "main_building",
            "auxiliary",
            "balcony",
            "land_area",
            "floor",
            "rooms",
            "bathrooms",
            "parking",
            "has_",
            "is_basement",
            "multi_floor",
            "main_use_missing",
        ]
    ):
        return "basic_housing"
    return "other"


def classify_comparable_feature_type(feature: str) -> str:
    if feature.endswith("_count"):
        return "count"
    if "weighted_mean_price" in feature:
        return "weighted_mean_price"
    if "median_price" in feature:
        return "median_price"
    if "mean_price" in feature:
        return "mean_price"
    if "std_price" in feature:
        return "std_price"
    if "nearest_price" in feature:
        return "nearest_price"
    if "distance" in feature:
        return "distance"
    if "days_diff" in feature:
        return "days_diff"
    if "area_diff" in feature:
        return "area_diff"
    if "age_diff" in feature:
        return "age_diff"
    return "other"


def get_transformed_feature_names(pipeline: Any, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps.get("preprocess")
    if preprocessor is None:
        return numeric_features + categorical_features
    try:
        names = preprocessor.get_feature_names_out()
        return [str(name).replace("num__", "").replace("cat__", "") for name in names]
    except Exception:  # noqa: BLE001
        return numeric_features + categorical_features


def list_tree_model_files(model_dir: str | Path) -> list[Path]:
    root = resolve_path(model_dir)
    files = sorted(root.glob("tree_model_fold_*.joblib"), key=lambda p: int(re.search(r"fold_(\d+)", p.stem).group(1)))
    if not files:
        raise FileNotFoundError(f"No tree_model_fold_*.joblib files found in {root}")
    return files


def extract_feature_importance(
    model_dir: str | Path,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rows = []
    warnings = []
    for path in list_tree_model_files(model_dir):
        fold_match = re.search(r"fold_(\d+)", path.stem)
        fold_id = int(fold_match.group(1)) if fold_match else -1
        pipeline = joblib.load(path)
        model = pipeline.named_steps.get("model")
        feature_names = get_transformed_feature_names(pipeline, numeric_features, categorical_features)
        try:
            gain = model.booster_.feature_importance(importance_type="gain")
            split = model.booster_.feature_importance(importance_type="split")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"fold {fold_id}: feature importance failed: {type(exc).__name__}: {exc}")
            continue
        if len(feature_names) != len(gain):
            warnings.append(f"fold {fold_id}: feature name count {len(feature_names)} != importance count {len(gain)}; using f-index names.")
            feature_names = [f"f{i}" for i in range(len(gain))]
        for feature, gain_value, split_value in zip(feature_names, gain, split, strict=False):
            rows.append(
                {
                    "fold_id": fold_id,
                    "feature": feature,
                    "gain_importance": float(gain_value),
                    "split_importance": float(split_value),
                    "feature_group": classify_feature_group(feature),
                }
            )
    by_fold = pd.DataFrame(rows)
    if by_fold.empty:
        return by_fold, pd.DataFrame(), pd.DataFrame(), warnings

    mean = (
        by_fold.groupby(["feature", "feature_group"], dropna=False)
        .agg(
            mean_gain_importance=("gain_importance", "mean"),
            std_gain_importance=("gain_importance", "std"),
            mean_split_importance=("split_importance", "mean"),
            std_split_importance=("split_importance", "std"),
        )
        .reset_index()
    )
    mean["rank_gain"] = mean["mean_gain_importance"].rank(method="min", ascending=False).astype(int)
    mean["rank_split"] = mean["mean_split_importance"].rank(method="min", ascending=False).astype(int)
    mean = mean.sort_values(["rank_gain", "rank_split", "feature"]).reset_index(drop=True)

    group = (
        mean.groupby("feature_group", dropna=False)
        .agg(
            total_gain_importance=("mean_gain_importance", "sum"),
            total_split_importance=("mean_split_importance", "sum"),
            feature_count=("feature", "nunique"),
        )
        .reset_index()
    )
    total_gain = group["total_gain_importance"].sum()
    total_split = group["total_split_importance"].sum()
    group["gain_share"] = np.where(total_gain > 0, group["total_gain_importance"] / total_gain, np.nan)
    group["split_share"] = np.where(total_split > 0, group["total_split_importance"] / total_split, np.nan)
    group = group.sort_values("total_gain_importance", ascending=False).reset_index(drop=True)
    return by_fold, mean, group, warnings


def plot_bar(df: pd.DataFrame, value_col: str, label_col: str, title: str, path: Path, top_n: int = 30) -> None:
    if plt is None or df.empty:
        return
    work = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, len(work) * 0.28)))
    ax.barh(work[label_col], work[value_col], color="#3b6ea8")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_feature_importance(config: dict[str, Any], model_dir: Path, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    by_fold, mean, group, warnings = extract_feature_importance(
        model_dir,
        config["numeric_features"],
        config["categorical_features"],
    )
    by_fold.to_csv(dirs["feature_importance"] / "feature_importance_by_fold.csv", index=False, encoding="utf-8-sig")
    mean.to_csv(dirs["feature_importance"] / "feature_importance_mean.csv", index=False, encoding="utf-8-sig")
    group.to_csv(dirs["feature_importance"] / "feature_group_importance.csv", index=False, encoding="utf-8-sig")
    plot_bar(mean, "mean_gain_importance", "feature", "Top 30 Gain Importance", dirs["feature_importance"] / "feature_importance_top30_gain.png")
    plot_bar(mean, "mean_split_importance", "feature", "Top 30 Split Importance", dirs["feature_importance"] / "feature_importance_top30_split.png")
    return by_fold, mean, group, warnings


def run_prediction_ic(predictions: pd.DataFrame, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = {**calculate_ic(predictions["y_true"], predictions["y_pred"]), **regression_metrics(predictions)}
    overall_df = pd.DataFrame([overall])
    rows = []
    for quarter, group in predictions.groupby("test_quarter", dropna=False):
        rows.append({"test_quarter": quarter, **calculate_ic(group["y_true"], group["y_pred"]), **regression_metrics(group)})
    by_quarter = pd.DataFrame(rows).sort_values("test_quarter")
    overall_df.to_csv(dirs["prediction_ic"] / "prediction_ic_overall.csv", index=False, encoding="utf-8-sig")
    by_quarter.to_csv(dirs["prediction_ic"] / "prediction_ic_by_quarter.csv", index=False, encoding="utf-8-sig")
    if plt is not None and not by_quarter.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(by_quarter["test_quarter"], by_quarter["pearson_ic"], marker="o", label="Pearson IC")
        ax.plot(by_quarter["test_quarter"], by_quarter["spearman_rank_ic"], marker="o", label="Spearman Rank IC")
        if "2026Q1" in set(by_quarter["test_quarter"]):
            row = by_quarter.loc[by_quarter["test_quarter"].eq("2026Q1")].iloc[0]
            ax.annotate(f"2026Q1 n={int(row['n'])}", xy=("2026Q1", row["spearman_rank_ic"]), xytext=(0, -25), textcoords="offset points", ha="center")
        ax.set_title("Prediction IC by Test Quarter")
        ax.set_ylabel("Correlation")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(dirs["prediction_ic"] / "prediction_ic_by_quarter.png", dpi=160)
        plt.close(fig)
    return overall_df, by_quarter


def run_residual_analysis(predictions: pd.DataFrame, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = predictions.copy()
    work["price_segment"] = work["y_true"].map(assign_price_segment)
    summary = pd.DataFrame([summarize_residual_group(work)])
    by_price = summarize_by(work, "price_segment")
    by_quarter = summarize_by(work, "test_quarter").sort_values("test_quarter")
    by_district = summarize_by(work, "district").sort_values("mae", ascending=False)
    summary.to_csv(dirs["residual_analysis"] / "residual_summary.csv", index=False, encoding="utf-8-sig")
    by_price.to_csv(dirs["residual_analysis"] / "residual_by_price_segment.csv", index=False, encoding="utf-8-sig")
    by_quarter.to_csv(dirs["residual_analysis"] / "residual_by_quarter.csv", index=False, encoding="utf-8-sig")
    by_district.to_csv(dirs["residual_analysis"] / "residual_by_district.csv", index=False, encoding="utf-8-sig")
    if plt is not None:
        sample = work.sample(n=min(20000, len(work)), random_state=42)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sample["y_true"], sample["error"], s=5, alpha=0.25)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("y_true unit_price_ping")
        ax.set_ylabel("residual (y_pred - y_true)")
        ax.set_title("Residual vs True Price")
        fig.tight_layout()
        fig.savefig(dirs["residual_analysis"] / "residual_vs_y_true.png", dpi=160)
        plt.close(fig)

        plot_bar(by_price, "bias", "price_segment", "Bias by Price Segment", dirs["residual_analysis"] / "residual_by_price_segment.png", top_n=len(by_price))
        plot_bar(by_price, "mape", "price_segment", "MAPE by Price Segment", dirs["residual_analysis"] / "ape_by_price_segment.png", top_n=len(by_price))
    return summary, by_price, by_quarter, by_district


def detect_high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    rows = []
    cols = corr.columns.tolist()
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"feature_1": left, "feature_2": right, "correlation": float(value), "abs_correlation": float(abs(value))})
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])


def run_correlation_analysis(dataset: pd.DataFrame, config: dict[str, Any], dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_features = [col for col in config["numeric_features"] if col in dataset.columns]
    work = dataset[[TARGET_COL] + numeric_features].copy()
    corr = work.corr(method="pearson")
    corr_features = corr.loc[numeric_features, numeric_features]
    corr_features.to_csv(dirs["correlation"] / "numeric_feature_correlation.csv", encoding="utf-8-sig")
    high_pairs = detect_high_correlation_pairs(corr_features, threshold=0.90)
    high_pairs.to_csv(dirs["correlation"] / "high_correlation_pairs.csv", index=False, encoding="utf-8-sig")

    target_corr = (
        corr.loc[numeric_features, TARGET_COL]
        .rename("pearson_corr_with_target")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    target_corr["abs_corr"] = target_corr["pearson_corr_with_target"].abs()
    target_corr["feature_group"] = target_corr["feature"].map(classify_feature_group)
    target_corr = target_corr.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    target_corr.to_csv(dirs["correlation"] / "feature_target_correlation.csv", index=False, encoding="utf-8-sig")

    if plt is not None and not target_corr.empty:
        top_features = target_corr.head(30)["feature"].tolist()
        heat = corr.loc[top_features, top_features]
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(heat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(top_features)))
        ax.set_yticks(range(len(top_features)))
        ax.set_xticklabels(top_features, rotation=90, fontsize=7)
        ax.set_yticklabels(top_features, fontsize=7)
        ax.set_title("Correlation Heatmap: Top 30 Target-Correlated Numeric Features")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(dirs["correlation"] / "correlation_heatmap_top_features.png", dpi=160)
        plt.close(fig)
    return corr_features, high_pairs, target_corr


def run_comparable_analysis(
    importance_mean: pd.DataFrame,
    target_corr: pd.DataFrame,
    dataset: pd.DataFrame,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    comp_cols = [col for col in dataset.columns if col.startswith("comp_")]
    missing = dataset[comp_cols].isna().mean().rename("missing_ratio").reset_index().rename(columns={"index": "feature"})
    imp = importance_mean.loc[importance_mean["feature"].isin(comp_cols), ["feature", "rank_gain", "mean_gain_importance"]].rename(columns={"rank_gain": "feature_importance_gain_rank"})
    corr = target_corr.loc[target_corr["feature"].isin(comp_cols), ["feature", "pearson_corr_with_target"]]
    result = (
        pd.DataFrame({"feature": comp_cols})
        .merge(imp, on="feature", how="left")
        .merge(corr, on="feature", how="left")
        .merge(missing, on="feature", how="left")
    )
    result["feature_type"] = result["feature"].map(classify_comparable_feature_type)
    result = result.sort_values(["feature_importance_gain_rank", "feature"]).reset_index(drop=True)
    result.to_csv(dirs["summary"] / "comparable_feature_analysis.csv", index=False, encoding="utf-8-sig")
    return result


def run_shap_analysis(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict[str, Any],
    model_dir: Path,
    dirs: dict[str, Path],
    run_shap: bool,
    shap_sample_size: int,
) -> tuple[bool, str, pd.DataFrame]:
    if not run_shap:
        return False, "SHAP skipped because --run-shap=false.", pd.DataFrame()
    if importlib.util.find_spec("shap") is None:
        return False, "SHAP skipped because package is not installed. Add `shap` to requirements.txt to enable it.", pd.DataFrame()
    if plt is None:
        return False, "SHAP skipped because matplotlib is not available.", pd.DataFrame()

    import shap

    model_files = list_tree_model_files(model_dir)
    last_model_path = model_files[-1]
    fold_id = int(re.search(r"fold_(\d+)", last_model_path.stem).group(1))
    fold_predictions = predictions.loc[predictions["fold_id"].eq(fold_id)].copy()
    shap_note = f"Using last fold model fold_id={fold_id}."
    if len(fold_predictions) < min(shap_sample_size, 1000):
        fold_predictions = predictions.copy()
        shap_note += f" Last fold test n was small, so sampled from all test predictions."
    sample = fold_predictions.sample(n=min(shap_sample_size, len(fold_predictions)), random_state=42)
    dataset_work = dataset.copy()
    dataset_work["id"] = dataset_work["id"].astype(str)
    sample["id"] = sample["id"].astype(str)
    dataset_indexed = dataset_work.set_index("id", drop=False)
    sample_df = dataset_indexed.loc[sample["id"]].copy()
    feature_cols = config["numeric_features"] + config["categorical_features"]
    pipeline = joblib.load(last_model_path)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample_df[feature_cols])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = get_transformed_feature_names(pipeline, config["numeric_features"], config["categorical_features"])
    if len(feature_names) != transformed.shape[1]:
        return False, f"SHAP skipped because transformed feature names length {len(feature_names)} != matrix width {transformed.shape[1]}.", pd.DataFrame()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "id", sample_df["id"].to_numpy())
    sample_path = dirs["shap"] / "shap_values_sample.parquet"
    try:
        shap_df.to_parquet(sample_path, index=False)
    except Exception:  # noqa: BLE001
        sample_path = dirs["shap"] / "shap_values_sample.csv"
        shap_df.to_csv(sample_path, index=False, encoding="utf-8-sig")

    shap_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "feature_group": [classify_feature_group(feature) for feature in feature_names],
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(dirs["shap"] / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")

    shap.summary_plot(shap_values, transformed, feature_names=feature_names, plot_type="bar", show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(dirs["shap"] / "shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(dirs["shap"] / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()
    return True, f"SHAP completed. {shap_note} Sample rows={len(sample_df)}. Values saved to {sample_path}.", shap_importance


def format_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No data._"
    headers = df.columns.astype(str).tolist()
    rows = df.values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(format_scalar(cell) for cell in row) + " |" for row in rows],
        ]
    )


def write_summary_report(
    dirs: dict[str, Path],
    args: argparse.Namespace,
    dataset_path: Path,
    pred_path: Path,
    config_path: Path,
    model_dir: Path,
    row_count: int,
    importance_mean: pd.DataFrame,
    group_importance: pd.DataFrame,
    prediction_ic_overall: pd.DataFrame,
    prediction_ic_quarter: pd.DataFrame,
    residual_summary: pd.DataFrame,
    residual_by_price: pd.DataFrame,
    target_corr: pd.DataFrame,
    high_pairs: pd.DataFrame,
    comparable_analysis: pd.DataFrame,
    shap_success: bool,
    shap_message: str,
    shap_importance: pd.DataFrame,
    importance_warnings: list[str],
) -> Path:
    comparable_top30 = importance_mean.loc[importance_mean["feature_group"].eq("comparable_sales") & importance_mean["rank_gain"].le(30)].copy()
    time_aware_top30 = importance_mean.loc[importance_mean["feature_group"].eq("time_aware_market") & importance_mean["rank_gain"].le(30)].copy()
    ic_best = prediction_ic_quarter.sort_values("spearman_rank_ic", ascending=False).head(1)
    ic_worst = prediction_ic_quarter.sort_values("spearman_rank_ic", ascending=True).head(1)
    high_price = residual_by_price.loc[residual_by_price["price_segment"].eq("120+")]
    low_price = residual_by_price.loc[residual_by_price["price_segment"].eq("0-50")]
    high_bias = float(high_price.iloc[0]["bias"]) if not high_price.empty else np.nan
    low_bias = float(low_price.iloc[0]["bias"]) if not low_price.empty else np.nan
    price_mae_max = residual_by_price.sort_values("mae", ascending=False).head(1)
    price_mape_max = residual_by_price.sort_values("mape", ascending=False).head(1)

    top_cols = ["feature", "mean_gain_importance", "rank_gain", "feature_group"]
    comp_cols = ["feature", "feature_importance_gain_rank", "mean_gain_importance", "pearson_corr_with_target", "missing_ratio", "feature_type"]
    corr_cols = ["feature", "pearson_corr_with_target", "abs_corr", "feature_group"]
    content = [
        "# Phase 3 Explainability Report",
        "",
        "## 1. Analysis Setup",
        "",
        f"- dataset path: `{dataset_path}`",
        f"- predictions path: `{pred_path}`",
        f"- feature config path: `{config_path}`",
        f"- model directory: `{model_dir}`",
        f"- model_name: `{args.model_name}`",
        f"- split: `{args.split}`",
        f"- row count: {row_count:,}",
        "",
        "## 2. Feature Importance Findings",
        "",
        "### Top 20 Gain Importance Features",
        "",
        markdown_table(importance_mean[top_cols], max_rows=20),
        "",
        "### Feature Group Importance",
        "",
        markdown_table(group_importance),
        "",
        f"- Comparable sales features in top 30 gain: {len(comparable_top30)}",
        f"- Time-aware market features in top 30 gain: {len(time_aware_top30)}",
        "- Feature names are transformed pipeline names where categorical one-hot levels are present.",
        "- Importance warnings: " + ("; ".join(importance_warnings) if importance_warnings else "(none)"),
        "",
        "## 3. Prediction IC Findings",
        "",
        markdown_table(prediction_ic_overall),
        "",
        "### By Quarter",
        "",
        markdown_table(prediction_ic_quarter[["test_quarter", "n", "pearson_ic", "spearman_rank_ic", "mae", "mape", "r2", "bias"]]),
        "",
        f"- Best quarter by Rank IC: {ic_best.iloc[0]['test_quarter'] if not ic_best.empty else ''}",
        f"- Worst quarter by Rank IC: {ic_worst.iloc[0]['test_quarter'] if not ic_worst.empty else ''}",
        "- 2026Q1 sample is small and should be interpreted cautiously.",
        "",
        "## 4. Residual Findings",
        "",
        markdown_table(residual_summary),
        "",
        "### By Price Segment",
        "",
        markdown_table(residual_by_price[["price_segment", "n", "y_true_mean", "y_pred_mean", "mae", "rmse", "mape", "bias", "error_std", "abs_error_median"]]),
        "",
        f"- High-price segment bias (`120+`): {high_bias:.4f}; negative means under-prediction.",
        f"- Low-price segment bias (`0-50`): {low_bias:.4f}; positive means over-prediction.",
        f"- Highest MAE segment: {price_mae_max.iloc[0]['price_segment'] if not price_mae_max.empty else ''}",
        f"- Highest MAPE segment: {price_mape_max.iloc[0]['price_segment'] if not price_mape_max.empty else ''}",
        "",
        "## 5. Correlation Findings",
        "",
        "### Top Target-Correlated Features",
        "",
        markdown_table(target_corr[corr_cols], max_rows=20),
        "",
        f"- High correlation feature pairs count (`abs(corr) >= 0.90`): {len(high_pairs)}",
        "",
        "## 6. Comparable Feature Usefulness",
        "",
        markdown_table(comparable_analysis[comp_cols], max_rows=20),
        "",
        "## 7. SHAP Findings",
        "",
        f"- SHAP success: {str(shap_success).lower()}",
        f"- {shap_message}",
        "",
        "### Top SHAP Features",
        "",
        markdown_table(shap_importance[["feature", "mean_abs_shap", "feature_group"]], max_rows=10) if shap_success else "_SHAP not available._",
        "",
        "## 8. Next Steps",
        "",
        "- 檢查高價 under-prediction cases。",
        "- 考慮加入更細的位置資訊，例如路段、捷運距離、社區名稱。",
        "- 若需要，再評估 embedding-based comparable retrieval。",
        "",
    ]
    path = dirs["summary"] / "phase3_explainability_report.md"
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Phase 3 model explainability and prediction IC.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--pred-path", default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--folds-path", default=str(DEFAULT_FOLDS_PATH))
    parser.add_argument("--feature-config", default=str(DEFAULT_FEATURE_CONFIG))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", default="tree_model")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-shap", type=str_to_bool, default=True)
    parser.add_argument("--shap-sample-size", type=int, default=5000)
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    dirs = ensure_output_dirs(args.output_dir)
    dataset_path = resolve_path(args.dataset_path)
    pred_path = resolve_path(args.pred_path)
    model_dir = resolve_path(args.model_dir)
    config, config_path = load_feature_config(args.feature_config)

    predictions = load_predictions(pred_path, args.model_name, args.split)
    dataset = load_dataset_features(dataset_path, config)

    by_fold, importance_mean, group_importance, importance_warnings = run_feature_importance(config, model_dir, dirs)
    logging.info("Feature importance rows: by_fold=%s mean=%s", len(by_fold), len(importance_mean))
    prediction_ic_overall, prediction_ic_quarter = run_prediction_ic(predictions, dirs)
    residual_summary, residual_by_price, _, _ = run_residual_analysis(predictions, dirs)
    _, high_pairs, target_corr = run_correlation_analysis(dataset, config, dirs)
    comparable_analysis = run_comparable_analysis(importance_mean, target_corr, dataset, dirs)
    shap_success, shap_message, shap_importance = run_shap_analysis(
        dataset=dataset,
        predictions=predictions,
        config=config,
        model_dir=model_dir,
        dirs=dirs,
        run_shap=args.run_shap,
        shap_sample_size=args.shap_sample_size,
    )
    report_path = write_summary_report(
        dirs=dirs,
        args=args,
        dataset_path=dataset_path,
        pred_path=pred_path,
        config_path=config_path,
        model_dir=model_dir,
        row_count=len(predictions),
        importance_mean=importance_mean,
        group_importance=group_importance,
        prediction_ic_overall=prediction_ic_overall,
        prediction_ic_quarter=prediction_ic_quarter,
        residual_summary=residual_summary,
        residual_by_price=residual_by_price,
        target_corr=target_corr,
        high_pairs=high_pairs,
        comparable_analysis=comparable_analysis,
        shap_success=shap_success,
        shap_message=shap_message,
        shap_importance=shap_importance,
        importance_warnings=importance_warnings,
    )
    logging.info("Wrote summary report: %s", report_path)


if __name__ == "__main__":
    main()
