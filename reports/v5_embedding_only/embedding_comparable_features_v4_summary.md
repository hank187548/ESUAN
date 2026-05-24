# Embedding Comparable Features V4 Summary

## Input / Output

- v2 input: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v2.csv`
- v3 input: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v3.csv`
- v4_add output: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v5_embedding_only_add.csv`
- v4_replace output: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v5_embedding_only_replace.csv`
- v4_add rows: 90,481
- v4_replace rows: 90,481

## Models

- embedding model: `Qwen/Qwen3-Embedding-8B`
- reranker model: `Qwen/Qwen3-Reranker-8B`
- embedding model path: `/home/nas2/Personal/Hank/Esuan/models/hf/Qwen3-Embedding-8B`
- reranker model path: `/home/nas2/Personal/Hank/Esuan/models/hf/Qwen3-Reranker-8B`
- download_if_missing: False
- device: cuda:0
- dtype: float16
- embedding batch size: 8
- reranker batch size: 2

## Text Representation Fields

district, building_type, material, building_age, building_area_ping, main_building_area_ping, floor, total_floor, floor_ratio, rooms, living_rooms, bathrooms, has_parking, parking_area_m2, has_management, has_elevator, physical_condition_flag, renovation_flag, broad_note_flag

## Retrieval Rule

- same district
- same building_type
- trade_date < current trade_date
- windows: 730d and 1095d
- embedding_top_k: 10
- reranker_top_k: 10
- use_reranker: False
- allow_reranker_fallback: False

## Leakage Check

| check_name | status | details |
| --- | --- | --- |
| emb_features_not_target_or_leakage | PASS | No emb_ feature overlaps target/leakage/drop cols. |
| v4_add_feature_config_excludes_leakage | PASS | No leakage/drop/target columns in features. |
| v4_replace_feature_config_excludes_leakage | PASS | No leakage/drop/target columns in features. |
| emb_730d_candidates_use_past_dates_only | PASS | Checked 100 sampled rows; candidates use trade_date < current trade_date. |
| emb_1095d_candidates_use_past_dates_only | PASS | Checked 100 sampled rows; candidates use trade_date < current trade_date. |

## New Features

| feature |
| --- |
| emb_730d_count |
| emb_730d_median_price |
| emb_730d_mean_price |
| emb_730d_weighted_mean_price |
| emb_730d_std_price |
| emb_730d_nearest_price |
| emb_730d_max_similarity |
| emb_730d_mean_similarity |
| emb_730d_median_similarity |
| emb_730d_mean_reranker_score |
| emb_730d_max_reranker_score |
| emb_730d_median_days_diff |
| emb_730d_median_area_diff_pct |
| emb_730d_median_age_diff |
| emb_1095d_count |
| emb_1095d_median_price |
| emb_1095d_mean_price |
| emb_1095d_weighted_mean_price |
| emb_1095d_std_price |
| emb_1095d_nearest_price |
| emb_1095d_max_similarity |
| emb_1095d_mean_similarity |
| emb_1095d_median_similarity |
| emb_1095d_mean_reranker_score |
| emb_1095d_max_reranker_score |
| emb_1095d_median_days_diff |
| emb_1095d_median_area_diff_pct |
| emb_1095d_median_age_diff |

## Count Statistics

| feature | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| emb_730d_count | 90481.0 | 9.9643 | 0.5028 | 0.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| emb_1095d_count | 90481.0 | 9.9675 | 0.4815 | 0.0 | 10.0 | 10.0 | 10.0 | 10.0 |

## Missing Ratio

| feature | missing_count | missing_ratio |
| --- | --- | --- |
| emb_730d_count | 0 | 0.0 |
| emb_730d_median_price | 77 | 0.0009 |
| emb_730d_mean_price | 77 | 0.0009 |
| emb_730d_weighted_mean_price | 77 | 0.0009 |
| emb_730d_std_price | 77 | 0.0009 |
| emb_730d_nearest_price | 77 | 0.0009 |
| emb_730d_max_similarity | 77 | 0.0009 |
| emb_730d_mean_similarity | 77 | 0.0009 |
| emb_730d_median_similarity | 77 | 0.0009 |
| emb_730d_mean_reranker_score | 90481 | 1.0 |
| emb_730d_max_reranker_score | 90481 | 1.0 |
| emb_730d_median_days_diff | 77 | 0.0009 |
| emb_730d_median_area_diff_pct | 77 | 0.0009 |
| emb_730d_median_age_diff | 6965 | 0.077 |
| emb_1095d_count | 0 | 0.0 |
| emb_1095d_median_price | 70 | 0.0008 |
| emb_1095d_mean_price | 70 | 0.0008 |
| emb_1095d_weighted_mean_price | 70 | 0.0008 |
| emb_1095d_std_price | 70 | 0.0008 |
| emb_1095d_nearest_price | 70 | 0.0008 |
| emb_1095d_max_similarity | 70 | 0.0008 |
| emb_1095d_mean_similarity | 70 | 0.0008 |
| emb_1095d_median_similarity | 70 | 0.0008 |
| emb_1095d_mean_reranker_score | 90481 | 1.0 |
| emb_1095d_max_reranker_score | 90481 | 1.0 |
| emb_1095d_median_days_diff | 70 | 0.0008 |
| emb_1095d_median_area_diff_pct | 70 | 0.0008 |
| emb_1095d_median_age_diff | 6982 | 0.0772 |

## Feature Config Counts

- v4_add numeric features: 84
- v4_replace numeric features: 62
- categorical features: 6

## Output Notes

- parquet errors: (none)

## Next Step CLI

- Use `data/processed/taipei_house_model_ready_v4_add.csv` with `reports/feature_config_model_v4_add.json` for additive Phase 4 training.
- Use `data/processed/taipei_house_model_ready_v4_replace.csv` with `reports/feature_config_model_v4_replace.json` for replacement Phase 4 training.
