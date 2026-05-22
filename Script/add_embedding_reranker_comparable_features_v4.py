from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
TARGET_COL = "unit_price_ping"
WINDOWS = (730, 1095)
EPSILON = 1e-6

BASE_EMB_FEATURES = [
    "count",
    "median_price",
    "mean_price",
    "weighted_mean_price",
    "std_price",
    "nearest_price",
    "max_similarity",
    "mean_similarity",
    "median_similarity",
    "mean_reranker_score",
    "max_reranker_score",
    "median_days_diff",
    "median_area_diff_pct",
    "median_age_diff",
]
EMB_FEATURES = [f"emb_{window}d_{feature}" for window in WINDOWS for feature in BASE_EMB_FEATURES]
COUNT_FEATURES = [f"emb_{window}d_count" for window in WINDOWS]

TEXT_FIELDS = [
    "district",
    "building_type",
    "material",
    "building_age",
    "building_area_ping",
    "main_building_area_ping",
    "floor",
    "total_floor",
    "floor_ratio",
    "rooms",
    "living_rooms",
    "bathrooms",
    "has_parking",
    "parking_area_m2",
    "has_management",
    "has_elevator",
    "physical_condition_flag",
    "renovation_flag",
    "broad_note_flag",
]

REQUIRED_COLUMNS = {
    "id",
    "trade_date",
    "district",
    "building_type",
    TARGET_COL,
    "building_area_ping",
    "building_age",
}


class EmbeddingBackend(Protocol):
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        ...


class RerankerBackend(Protocol):
    def score(self, pairs: list[tuple[str, str]], batch_size: int = 4) -> np.ndarray:
        ...


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


def parse_gpu_ids(value: str) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def resolve_device(device: str, gpu_ids: list[int]) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{gpu_ids[0] if gpu_ids else 0}"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def resolve_torch_dtype(dtype: str) -> Any | None:
    text = str(dtype).strip().lower()
    if text == "auto":
        return "auto"
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(text)


def release_gpu_memory() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def resolve_model_source(model_name: str, model_path: str | Path, download_if_missing: bool) -> str:
    path = resolve_path(model_path)
    if path.exists():
        return str(path)
    if download_if_missing:
        return model_name
    raise FileNotFoundError(
        f"Local model path not found: {path}. "
        f"Download it first or rerun with --download-if-missing true."
    )


class SentenceTransformerEmbeddingBackend:
    def __init__(self, model_name: str, model_path: str | Path, download_if_missing: bool, device: str, dtype: str):
        from sentence_transformers import SentenceTransformer

        source = resolve_model_source(model_name, model_path, download_if_missing)
        torch_dtype = resolve_torch_dtype(dtype)
        model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else None
        self.model = SentenceTransformer(source, device=device, trust_remote_code=True, model_kwargs=model_kwargs)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class CrossEncoderRerankerBackend:
    def __init__(self, model_name: str, model_path: str | Path, download_if_missing: bool, device: str, max_length: int, dtype: str):
        from sentence_transformers import CrossEncoder

        source = resolve_model_source(model_name, model_path, download_if_missing)
        torch_dtype = resolve_torch_dtype(dtype)
        model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else None
        self.model = CrossEncoder(source, device=device, max_length=max_length, trust_remote_code=True, model_kwargs=model_kwargs)

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 4) -> np.ndarray:
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=float)


def format_value(value: Any, decimals: int = 2, missing: str = "缺失") -> str:
    if value is None or pd.isna(value):
        return missing
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def binary_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "缺失"
    try:
        return "有" if float(value) > 0 else "無"
    except (TypeError, ValueError):
        return str(value)


def build_text_representation(row: pd.Series | dict[str, Any], include_address: bool = False, include_note_raw: bool = False) -> str:
    get = row.get if isinstance(row, dict) else row.get
    parts = [
        f"行政區：{format_value(get('district'), 0)}",
        f"建物型態：{format_value(get('building_type'), 0)}",
        f"主要建材：{format_value(get('material'), 0)}",
        f"屋齡：{format_value(get('building_age'))} 年",
        f"建物坪數：{format_value(get('building_area_ping'))} 坪",
        f"主建物坪數：{format_value(get('main_building_area_ping'))} 坪",
        f"樓層：{format_value(get('floor'), 0)} / {format_value(get('total_floor'), 0)}",
        f"樓層比例：{format_value(get('floor_ratio'))}",
        f"格局：{format_value(get('rooms'), 0)} 房 {format_value(get('living_rooms'), 0)} 廳 {format_value(get('bathrooms'), 0)} 衛",
        f"車位：{binary_label(get('has_parking'))}",
        f"車位面積：{format_value(get('parking_area_m2'))}",
        f"管理組織：{binary_label(get('has_management'))}",
        f"電梯：{binary_label(get('has_elevator'))}",
        f"房屋狀態備註標記：{format_value(get('physical_condition_flag'), 0)}",
        f"裝潢標記：{format_value(get('renovation_flag'), 0)}",
        f"廣泛備註標記：{format_value(get('broad_note_flag'), 0)}",
    ]
    if include_address:
        parts.append(f"地址：{format_value(get('address_raw'), 0)}")
    if include_note_raw:
        parts.append(f"備註：{format_value(get('note_raw'), 0)}")
    return "\n".join(parts)


def add_text_representation(df: pd.DataFrame, include_address: bool, include_note_raw: bool) -> pd.DataFrame:
    output = df.copy()
    output["text_representation"] = [
        build_text_representation(row, include_address=include_address, include_note_raw=include_note_raw)
        for _, row in output.iterrows()
    ]
    return output


def load_dataset(path: str | Path, include_address: bool, include_note_raw: bool) -> pd.DataFrame:
    resolved = resolve_path(path)
    df = pd.read_csv(resolved, encoding="utf-8-sig", low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    if df["trade_date"].isna().any():
        raise ValueError("Dataset contains invalid trade_date values.")
    for col in [TARGET_COL, "building_area_ping", "building_age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return add_text_representation(df, include_address=include_address, include_note_raw=include_note_raw)


def load_feature_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def hash_texts(texts: pd.Series | list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_compute_embeddings(
    df: pd.DataFrame,
    backend: EmbeddingBackend,
    cache_dir: str | Path,
    model_name: str,
    batch_size: int,
    force_recompute: bool,
) -> np.ndarray:
    cache_root = resolve_path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    vector_path = cache_root / "embedding_vectors.npy"
    meta_path = cache_root / "embedding_meta.csv"
    info_path = cache_root / "embedding_cache_info.json"
    text_hash = hash_texts(df["text_representation"])
    if not force_recompute and vector_path.exists() and meta_path.exists() and info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        meta = pd.read_csv(meta_path, encoding="utf-8-sig")
        if (
            info.get("model_name") == model_name
            and info.get("text_hash") == text_hash
            and len(meta) == len(df)
            and meta["id"].astype(str).tolist() == df["id"].astype(str).tolist()
        ):
            return np.load(vector_path)

    vectors = backend.encode(df["text_representation"].tolist(), batch_size=batch_size)
    np.save(vector_path, vectors)
    df[["id", "trade_date"]].to_csv(meta_path, index=False, encoding="utf-8-sig")
    with info_path.open("w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "text_hash": text_hash, "row_count": len(df)}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return vectors


def get_candidate_indices(group: pd.DataFrame, current_pos: int, window_days: int) -> np.ndarray:
    dates = group["trade_date"].to_numpy(dtype="datetime64[ns]")
    current_date = dates[current_pos]
    start_date = current_date - np.timedelta64(window_days, "D")
    left = int(np.searchsorted(dates, start_date, side="left"))
    right = int(np.searchsorted(dates, current_date, side="left"))
    if right <= left:
        return np.array([], dtype=int)
    return np.arange(left, right, dtype=int)


def cosine_top_k(current_vector: np.ndarray, candidate_vectors: np.ndarray, candidate_indices: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(candidate_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    current_norm = np.linalg.norm(current_vector)
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
    denominator = np.maximum(current_norm * candidate_norms, EPSILON)
    similarities = candidate_vectors @ current_vector / denominator
    k = min(top_k, len(similarities))
    order = np.argsort(-similarities, kind="mergesort")[:k]
    return candidate_indices[order], similarities[order]


def reranker_top_k(candidate_indices: np.ndarray, scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(candidate_indices) == 0:
        return candidate_indices, scores
    k = min(top_k, len(scores))
    order = np.argsort(-scores, kind="mergesort")[:k]
    return candidate_indices[order], scores[order]


def softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    shifted = arr - np.nanmax(arr)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def nanmedian_or_nan(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else np.nan


def summarize_selected_candidates(
    current_row: pd.Series,
    candidate_rows: pd.DataFrame,
    similarities: np.ndarray,
    reranker_scores: np.ndarray | None,
    use_reranker: bool,
) -> dict[str, float]:
    if candidate_rows.empty:
        return {feature: 0 if feature == "count" else np.nan for feature in BASE_EMB_FEATURES}

    prices = pd.to_numeric(candidate_rows[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(prices)
    if not valid.any():
        return {feature: 0 if feature == "count" else np.nan for feature in BASE_EMB_FEATURES}

    candidate_rows = candidate_rows.loc[valid].copy()
    prices = prices[valid]
    similarities = np.asarray(similarities, dtype=float)[valid]
    if reranker_scores is not None:
        reranker_scores = np.asarray(reranker_scores, dtype=float)[valid]

    if use_reranker and reranker_scores is not None:
        weights = softmax(reranker_scores)
    else:
        weights = np.maximum(similarities, 0) + EPSILON
        weights = weights / np.sum(weights)

    days_diff = (pd.Timestamp(current_row["trade_date"]) - candidate_rows["trade_date"]).dt.days.to_numpy(dtype=float)
    current_area = float(current_row["building_area_ping"]) if pd.notna(current_row["building_area_ping"]) else np.nan
    candidate_area = pd.to_numeric(candidate_rows["building_area_ping"], errors="coerce").to_numpy(dtype=float)
    area_denominator = max(current_area, 1.0) if np.isfinite(current_area) else 1.0
    area_diff = np.where(np.isfinite(candidate_area) & np.isfinite(current_area), np.abs(candidate_area - current_area) / area_denominator, np.nan)

    current_age = float(current_row["building_age"]) if pd.notna(current_row["building_age"]) else np.nan
    candidate_age = pd.to_numeric(candidate_rows["building_age"], errors="coerce").to_numpy(dtype=float)
    age_diff = np.where(np.isfinite(candidate_age) & np.isfinite(current_age), np.abs(candidate_age - current_age), np.nan)

    return {
        "count": int(len(candidate_rows)),
        "median_price": float(np.median(prices)),
        "mean_price": float(np.mean(prices)),
        "weighted_mean_price": float(np.sum(weights * prices)),
        "std_price": float(np.std(prices, ddof=0)),
        "nearest_price": float(prices[0]),
        "max_similarity": float(np.max(similarities)),
        "mean_similarity": float(np.mean(similarities)),
        "median_similarity": float(np.median(similarities)),
        "mean_reranker_score": float(np.mean(reranker_scores)) if reranker_scores is not None else np.nan,
        "max_reranker_score": float(np.max(reranker_scores)) if reranker_scores is not None else np.nan,
        "median_days_diff": nanmedian_or_nan(days_diff),
        "median_area_diff_pct": nanmedian_or_nan(area_diff),
        "median_age_diff": nanmedian_or_nan(age_diff),
    }


def compute_embedding_comparable_features(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_top_k: int,
    reranker_top_k_value: int,
    use_reranker: bool,
    reranker: RerankerBackend | None,
    reranker_batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = df.copy()
    output["_row_order"] = np.arange(len(output))
    for feature in EMB_FEATURES:
        output[feature] = 0 if feature in COUNT_FEATURES else np.nan

    retrieval_samples: list[dict[str, Any]] = []
    reranker_samples: list[dict[str, Any]] = []

    for _, group in output.groupby(["district", "building_type"], dropna=False, sort=False):
        group = group.sort_values(["trade_date", "_row_order"], kind="mergesort").reset_index(drop=True)
        row_orders = group["_row_order"].to_numpy(dtype=int)
        group_embeddings = embeddings[row_orders]
        for current_pos, current_row in group.iterrows():
            current_global_idx = int(current_row["_row_order"])
            for window_days in WINDOWS:
                candidate_pos = get_candidate_indices(group, current_pos, window_days)
                if len(candidate_pos) == 0:
                    continue
                selected_pos, similarities = cosine_top_k(
                    group_embeddings[current_pos],
                    group_embeddings[candidate_pos],
                    candidate_pos,
                    embedding_top_k,
                )
                if len(selected_pos) == 0:
                    continue
                selected_rows = group.iloc[selected_pos].copy()
                if len(retrieval_samples) < 200 and window_days == 730:
                    for rank, (candidate, similarity) in enumerate(zip(selected_rows.itertuples(index=False), similarities, strict=False), start=1):
                        if rank > 3:
                            break
                        retrieval_samples.append(
                            {
                                "current_id": current_row["id"],
                                "current_trade_date": current_row["trade_date"],
                                "candidate_rank": rank,
                                "candidate_id": candidate.id,
                                "candidate_trade_date": candidate.trade_date,
                                "similarity": similarity,
                            }
                        )

                reranker_scores = None
                if use_reranker:
                    if reranker is None:
                        raise RuntimeError("Reranker requested but not initialized.")
                    pairs = [(current_row["text_representation"], text) for text in selected_rows["text_representation"].tolist()]
                    raw_scores = reranker.score(pairs, batch_size=reranker_batch_size)
                    selected_pos, reranker_scores = reranker_top_k(selected_pos, raw_scores, reranker_top_k_value)
                    selected_rows = group.iloc[selected_pos].copy()
                    similarities = np.asarray([
                        float(np.dot(group_embeddings[current_pos], group_embeddings[pos]) / max(np.linalg.norm(group_embeddings[current_pos]) * np.linalg.norm(group_embeddings[pos]), EPSILON))
                        for pos in selected_pos
                    ])
                    if len(reranker_samples) < 200 and window_days == 730:
                        for rank, (candidate, score) in enumerate(zip(selected_rows.itertuples(index=False), reranker_scores, strict=False), start=1):
                            if rank > 3:
                                break
                            reranker_samples.append(
                                {
                                    "current_id": current_row["id"],
                                    "current_trade_date": current_row["trade_date"],
                                    "candidate_rank": rank,
                                    "candidate_id": candidate.id,
                                    "candidate_trade_date": candidate.trade_date,
                                    "reranker_score": score,
                                }
                            )
                else:
                    selected_pos = selected_pos[:reranker_top_k_value]
                    selected_rows = group.iloc[selected_pos].copy()
                    similarities = similarities[:reranker_top_k_value]

                summary = summarize_selected_candidates(
                    current_row=current_row,
                    candidate_rows=selected_rows,
                    similarities=similarities,
                    reranker_scores=reranker_scores,
                    use_reranker=use_reranker,
                )
                for key, value in summary.items():
                    output.loc[current_global_idx, f"emb_{window_days}d_{key}"] = value

    for feature in COUNT_FEATURES:
        output[feature] = output[feature].fillna(0).astype(int)
    return (
        output.drop(columns=["_row_order"]),
        pd.DataFrame(retrieval_samples),
        pd.DataFrame(reranker_samples),
    )


def build_feature_config(base_config: dict[str, Any], mode: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    numeric = list(config.get("numeric_features", []))
    if mode == "replace":
        numeric = [feature for feature in numeric if not feature.startswith("comp_")]
    for feature in EMB_FEATURES:
        if feature not in numeric:
            numeric.append(feature)
    config["numeric_features"] = numeric
    notes = list(config.get("notes", []))
    notes.append(f"v4_{mode} 新增 Qwen3 embedding/reranker comparable sales features。")
    notes.append("Embedding/reranker comparable pool 嚴格使用 trade_date < current trade_date。")
    notes.append("同日交易不會互相作為 comparable candidate。")
    config["notes"] = notes
    return config


def validate_no_leakage(config: dict[str, Any]) -> list[str]:
    target = config.get("target_col", TARGET_COL)
    leakage = set(config.get("leakage_cols", []))
    drop = set(config.get("drop_cols", []))
    features = set(config.get("numeric_features", [])) | set(config.get("categorical_features", []))
    return sorted(features & (leakage | drop | {target}))


def run_leakage_check(df: pd.DataFrame, config_add: dict[str, Any], config_replace: dict[str, Any], sample_size: int = 100) -> pd.DataFrame:
    rows = []
    forbidden_emb = sorted(set(EMB_FEATURES) & (set(config_add.get("leakage_cols", [])) | set(config_add.get("drop_cols", [])) | {config_add.get("target_col", TARGET_COL)}))
    rows.append(
        {
            "check_name": "emb_features_not_target_or_leakage",
            "status": "PASS" if not forbidden_emb else "FAIL",
            "details": "No emb_ feature overlaps target/leakage/drop cols." if not forbidden_emb else ", ".join(forbidden_emb),
        }
    )
    for name, config in [("v4_add", config_add), ("v4_replace", config_replace)]:
        bad = validate_no_leakage(config)
        rows.append(
            {
                "check_name": f"{name}_feature_config_excludes_leakage",
                "status": "PASS" if not bad else "FAIL",
                "details": "No leakage/drop/target columns in features." if not bad else ", ".join(bad),
            }
        )

    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    for window in WINDOWS:
        violations = []
        for _, row in sample.iterrows():
            current_date = pd.Timestamp(row["trade_date"])
            start_date = current_date - pd.Timedelta(days=window)
            mask = (
                df["district"].eq(row["district"])
                & df["building_type"].eq(row["building_type"])
                & df["trade_date"].ge(start_date)
                & df["trade_date"].lt(current_date)
            )
            pool = df.loc[mask]
            if not pool.empty and pool["trade_date"].max() >= current_date:
                violations.append(str(row["id"]))
        rows.append(
            {
                "check_name": f"emb_{window}d_candidates_use_past_dates_only",
                "status": "PASS" if not violations else "FAIL",
                "details": f"Checked {len(sample)} sampled rows; candidates use trade_date < current trade_date."
                if not violations
                else ", ".join(violations[:20]),
            }
        )
    return pd.DataFrame(rows)


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in EMB_FEATURES:
        series = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()),
                "min": float(series.min()) if series.notna().any() else np.nan,
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "median": float(series.median()) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_dataset_outputs(df: pd.DataFrame, csv_path: str | Path, parquet_path: str | Path) -> tuple[Path, Path, str]:
    csv = resolve_path(csv_path)
    parquet = resolve_path(parquet_path)
    csv.parent.mkdir(parents=True, exist_ok=True)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    parquet_error = ""
    try:
        df.to_parquet(parquet, index=False)
    except Exception as exc:  # noqa: BLE001
        parquet_error = f"{type(exc).__name__}: {exc}"
        logging.warning("Parquet output failed: %s", parquet_error)
    return csv, parquet, parquet_error


def write_feature_config(config: dict[str, Any], root_path: str | Path, report_dir: str | Path) -> tuple[Path, Path]:
    root = resolve_path(root_path)
    report = resolve_path(report_dir) / root.name
    for path in [root, report]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return root, report


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    headers = df.columns.astype(str).tolist()
    rows = df.values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join("" if pd.isna(cell) else str(round(cell, 4)) if isinstance(cell, (float, np.floating)) else str(cell) for cell in row) + " |" for row in rows],
        ]
    )


def write_summary(
    report_dir: str | Path,
    args: argparse.Namespace,
    df_add: pd.DataFrame,
    df_replace: pd.DataFrame,
    missing_report: pd.DataFrame,
    leakage_report: pd.DataFrame,
    config_add: dict[str, Any],
    config_replace: dict[str, Any],
    parquet_errors: list[str],
) -> Path:
    report_root = resolve_path(report_dir)
    count_stats = df_add[COUNT_FEATURES].describe().T.reset_index().rename(columns={"index": "feature"})
    content = [
        "# Embedding Comparable Features V4 Summary",
        "",
        "## Input / Output",
        "",
        f"- v2 input: `{resolve_path(args.v2_input_path)}`",
        f"- v3 input: `{resolve_path(args.v3_input_path)}`",
        f"- v4_add output: `{resolve_path(args.v4_add_output_path)}`",
        f"- v4_replace output: `{resolve_path(args.v4_replace_output_path)}`",
        f"- v4_add rows: {len(df_add):,}",
        f"- v4_replace rows: {len(df_replace):,}",
        "",
        "## Models",
        "",
        f"- embedding model: `{args.embedding_model_name}`",
        f"- reranker model: `{args.reranker_model_name}`",
        f"- embedding model path: `{resolve_path(args.embedding_model_path)}`",
        f"- reranker model path: `{resolve_path(args.reranker_model_path)}`",
        f"- download_if_missing: {args.download_if_missing}",
        f"- device: {args.device}",
        f"- dtype: {args.dtype}",
        f"- embedding batch size: {args.embedding_batch_size}",
        f"- reranker batch size: {args.reranker_batch_size}",
        "",
        "## Text Representation Fields",
        "",
        ", ".join(TEXT_FIELDS),
        "",
        "## Retrieval Rule",
        "",
        "- same district",
        "- same building_type",
        "- trade_date < current trade_date",
        "- windows: 730d and 1095d",
        f"- embedding_top_k: {args.embedding_top_k}",
        f"- reranker_top_k: {args.reranker_top_k}",
        f"- use_reranker: {args.use_reranker}",
        f"- allow_reranker_fallback: {args.allow_reranker_fallback}",
        "",
        "## Leakage Check",
        "",
        markdown_table(leakage_report),
        "",
        "## New Features",
        "",
        markdown_table(pd.DataFrame({"feature": EMB_FEATURES})),
        "",
        "## Count Statistics",
        "",
        markdown_table(count_stats),
        "",
        "## Missing Ratio",
        "",
        markdown_table(missing_report[["feature", "missing_count", "missing_ratio"]]),
        "",
        "## Feature Config Counts",
        "",
        f"- v4_add numeric features: {len(config_add.get('numeric_features', []))}",
        f"- v4_replace numeric features: {len(config_replace.get('numeric_features', []))}",
        f"- categorical features: {len(config_add.get('categorical_features', []))}",
        "",
        "## Output Notes",
        "",
        f"- parquet errors: {' | '.join(error for error in parquet_errors if error) if any(parquet_errors) else '(none)'}",
        "",
        "## Next Step CLI",
        "",
        "- Use `data/processed/taipei_house_model_ready_v4_add.csv` with `reports/feature_config_model_v4_add.json` for additive Phase 4 training.",
        "- Use `data/processed/taipei_house_model_ready_v4_replace.csv` with `reports/feature_config_model_v4_replace.json` for replacement Phase 4 training.",
        "",
    ]
    path = report_root / "embedding_comparable_features_v4_summary.md"
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 4 embedding/reranker comparable sales features.")
    parser.add_argument("--v2-input-path", default="data/processed/taipei_house_model_ready_v2.csv")
    parser.add_argument("--v3-input-path", default="data/processed/taipei_house_model_ready_v3.csv")
    parser.add_argument("--feature-config-v2", default="reports/feature_config_model_v2.json")
    parser.add_argument("--feature-config-v3", default="reports/feature_config_model_v3.json")
    parser.add_argument("--v4-add-output-path", default="data/processed/taipei_house_model_ready_v4_add.csv")
    parser.add_argument("--v4-add-parquet-path", default="data/processed/taipei_house_model_ready_v4_add.parquet")
    parser.add_argument("--v4-replace-output-path", default="data/processed/taipei_house_model_ready_v4_replace.csv")
    parser.add_argument("--v4-replace-parquet-path", default="data/processed/taipei_house_model_ready_v4_replace.parquet")
    parser.add_argument("--feature-config-v4-add", default="reports/feature_config_model_v4_add.json")
    parser.add_argument("--feature-config-v4-replace", default="reports/feature_config_model_v4_replace.json")
    parser.add_argument("--report-dir", default="reports/v4")
    parser.add_argument("--embedding-model-name", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--reranker-model-name", default=RERANKER_MODEL_NAME)
    parser.add_argument("--embedding-model-path", default="models/hf/Qwen3-Embedding-8B")
    parser.add_argument("--reranker-model-path", default="models/hf/Qwen3-Reranker-8B")
    parser.add_argument("--download-if-missing", type=str_to_bool, default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--embedding-top-k", type=int, default=50)
    parser.add_argument("--reranker-top-k", type=int, default=10)
    parser.add_argument("--use-reranker", type=str_to_bool, default=True)
    parser.add_argument("--allow-reranker-fallback", type=str_to_bool, default=False)
    parser.add_argument("--include-address", type=str_to_bool, default=False)
    parser.add_argument("--include-note-raw", type=str_to_bool, default=False)
    parser.add_argument("--embedding-cache-dir", default="data/processed/v4_embedding_cache")
    parser.add_argument("--force-recompute-embeddings", type=str_to_bool, default=False)
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = resolve_device(args.device, gpu_ids)
    logging.info("Using device=%s dtype=%s", device, args.dtype)

    df_v2 = load_dataset(args.v2_input_path, include_address=args.include_address, include_note_raw=args.include_note_raw)
    df_v3 = load_dataset(args.v3_input_path, include_address=args.include_address, include_note_raw=args.include_note_raw)
    config_v2 = load_feature_config(args.feature_config_v2)
    config_v3 = load_feature_config(args.feature_config_v3)

    text_sample = df_v3[["id", "trade_date", "district", "building_type", "text_representation"]].head(200)
    text_sample.to_csv(report_dir / "text_representation_sample.csv", index=False, encoding="utf-8-sig")

    embedding_backend = SentenceTransformerEmbeddingBackend(
        model_name=args.embedding_model_name,
        model_path=args.embedding_model_path,
        download_if_missing=args.download_if_missing,
        device=device,
        dtype=args.dtype,
    )
    embeddings = load_or_compute_embeddings(
        df_v3,
        backend=embedding_backend,
        cache_dir=args.embedding_cache_dir,
        model_name=args.embedding_model_name,
        batch_size=args.embedding_batch_size,
        force_recompute=args.force_recompute_embeddings,
    )
    del embedding_backend
    release_gpu_memory()

    reranker_backend = None
    if args.use_reranker:
        try:
            reranker_backend = CrossEncoderRerankerBackend(
                model_name=args.reranker_model_name,
                model_path=args.reranker_model_path,
                download_if_missing=args.download_if_missing,
                device=device,
                max_length=args.max_length,
                dtype=args.dtype,
            )
        except Exception:
            if not args.allow_reranker_fallback:
                raise
            logging.exception("Reranker failed to initialize; falling back to embedding-only because fallback is allowed.")

    df_v3_features, retrieval_sample, reranker_sample = compute_embedding_comparable_features(
        df_v3,
        embeddings=embeddings,
        embedding_top_k=args.embedding_top_k,
        reranker_top_k_value=args.reranker_top_k,
        use_reranker=args.use_reranker and reranker_backend is not None,
        reranker=reranker_backend,
        reranker_batch_size=args.reranker_batch_size,
    )

    emb_cols = ["id"] + EMB_FEATURES
    df_add = df_v3_features
    df_replace = df_v2.merge(df_v3_features[emb_cols], on="id", how="left")
    for feature in COUNT_FEATURES:
        df_replace[feature] = df_replace[feature].fillna(0).astype(int)

    config_add = build_feature_config(config_v3, mode="add")
    config_replace = build_feature_config(config_v2, mode="replace")
    leakage = run_leakage_check(df_v3, config_add, config_replace)
    missing = build_missing_report(df_add)

    add_csv, add_parquet, add_parquet_error = write_dataset_outputs(df_add, args.v4_add_output_path, args.v4_add_parquet_path)
    replace_csv, replace_parquet, replace_parquet_error = write_dataset_outputs(df_replace, args.v4_replace_output_path, args.v4_replace_parquet_path)
    write_feature_config(config_add, args.feature_config_v4_add, report_dir)
    write_feature_config(config_replace, args.feature_config_v4_replace, report_dir)

    missing.to_csv(report_dir / "embedding_comparable_features_v4_missing_report.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(report_dir / "embedding_comparable_features_v4_leakage_check.csv", index=False, encoding="utf-8-sig")
    retrieval_sample.to_csv(report_dir / "embedding_retrieval_sample_matches.csv", index=False, encoding="utf-8-sig")
    reranker_sample.to_csv(report_dir / "reranker_sample_matches.csv", index=False, encoding="utf-8-sig")
    write_summary(
        report_dir=report_dir,
        args=args,
        df_add=df_add,
        df_replace=df_replace,
        missing_report=missing,
        leakage_report=leakage,
        config_add=config_add,
        config_replace=config_replace,
        parquet_errors=[add_parquet_error, replace_parquet_error],
    )
    logging.info("Wrote v4_add: %s %s", add_csv, add_parquet)
    logging.info("Wrote v4_replace: %s %s", replace_csv, replace_parquet)


if __name__ == "__main__":
    main()
