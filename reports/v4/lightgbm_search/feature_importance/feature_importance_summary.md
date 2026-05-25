# V4 Reranker Tuned LightGBM Feature Importance

- model dir: `models/v4_lightgbm_search`
- folds: 15
- importance: LightGBM gain and split, averaged across folds
- categorical one-hot levels are aggregated back to the original feature name

## Top 20 Features by Gain

| rank | feature | group | gain_share | split_share |
|---:|---|---|---:|---:|
| 1 | comp_730d_weighted_mean_price | rule_based_comparable | 45.16% | 1.40% |
| 2 | emb_1095d_weighted_mean_price | embedding_reranker_comparable | 11.11% | 1.60% |
| 3 | comp_730d_mean_price | rule_based_comparable | 8.78% | 0.97% |
| 4 | emb_730d_weighted_mean_price | embedding_reranker_comparable | 6.85% | 1.29% |
| 5 | comp_365d_weighted_mean_price | rule_based_comparable | 3.81% | 0.84% |
| 6 | building_age | basic_housing | 1.42% | 4.02% |
| 7 | floor | basic_housing | 1.09% | 1.53% |
| 8 | land_area_m2 | basic_housing | 0.83% | 3.04% |
| 9 | comp_730d_nearest_price | rule_based_comparable | 0.69% | 1.63% |
| 10 | parking_area_m2 | basic_housing | 0.61% | 1.54% |
| 11 | building_area_m2 | basic_housing | 0.58% | 2.22% |
| 12 | main_building_area_m2 | basic_housing | 0.57% | 2.31% |
| 13 | comp_730d_median_price | rule_based_comparable | 0.57% | 1.09% |
| 14 | auxiliary_area_m2 | basic_housing | 0.53% | 1.90% |
| 15 | comp_730d_std_price | rule_based_comparable | 0.50% | 2.12% |
| 16 | balcony_area_m2 | basic_housing | 0.47% | 2.12% |
| 17 | district_median_price_365d | time_aware_market | 0.46% | 1.44% |
| 18 | total_floor | basic_housing | 0.46% | 1.59% |
| 19 | emb_1095d_nearest_price | embedding_reranker_comparable | 0.46% | 1.55% |
| 20 | district_median_price_180d | time_aware_market | 0.46% | 1.45% |

## Feature Group Importance

| group | features | gain_share | split_share |
|---|---:|---:|---:|
| rule_based_comparable | 22 | 63.63% | 27.57% |
| embedding_reranker_comparable | 28 | 24.63% | 32.78% |
| basic_housing | 21 | 8.10% | 24.86% |
| time_aware_market | 8 | 2.71% | 11.44% |
| location_time | 7 | 0.77% | 2.79% |
| note_or_quality_flags | 4 | 0.17% | 0.56% |

## Embedding / Reranker Comparable Features

| rank | feature | gain_share | split_share |
|---:|---|---:|---:|
| 2 | emb_1095d_weighted_mean_price | 11.11% | 1.60% |
| 4 | emb_730d_weighted_mean_price | 6.85% | 1.29% |
| 19 | emb_1095d_nearest_price | 0.46% | 1.55% |
| 22 | emb_1095d_max_reranker_score | 0.44% | 1.59% |
| 24 | emb_730d_std_price | 0.39% | 1.75% |
| 25 | emb_1095d_median_age_diff | 0.38% | 1.55% |
| 26 | emb_1095d_std_price | 0.38% | 1.76% |
| 27 | emb_730d_median_age_diff | 0.38% | 1.53% |
| 29 | emb_730d_max_reranker_score | 0.37% | 1.42% |
| 30 | emb_730d_nearest_price | 0.37% | 1.43% |
| 36 | emb_1095d_median_days_diff | 0.29% | 1.50% |
| 41 | emb_730d_median_area_diff_pct | 0.28% | 1.42% |
| 43 | emb_730d_median_days_diff | 0.27% | 1.51% |
| 44 | emb_1095d_median_area_diff_pct | 0.27% | 1.40% |
| 49 | emb_1095d_mean_price | 0.25% | 0.93% |