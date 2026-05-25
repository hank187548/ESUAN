# V5 Embedding-Only Feature Importance

- model dir: `models/v5_embedding_only_add`
- folds: 15
- importance: LightGBM gain and split, averaged across folds
- categorical one-hot levels are aggregated back to the original feature name

## Top 20 Features by Gain

| rank | feature | group | gain_share | split_share |
|---:|---|---|---:|---:|
| 1 | comp_730d_weighted_mean_price | rule_based_comparable | 56.10% | 1.87% |
| 2 | comp_730d_mean_price | rule_based_comparable | 8.36% | 1.18% |
| 3 | comp_365d_weighted_mean_price | rule_based_comparable | 4.27% | 0.84% |
| 4 | emb_1095d_weighted_mean_price | embedding_comparable | 2.60% | 0.55% |
| 5 | building_age | basic_housing | 1.75% | 5.32% |
| 6 | emb_1095d_mean_price | embedding_comparable | 1.73% | 1.05% |
| 7 | emb_1095d_nearest_price | embedding_comparable | 1.42% | 1.75% |
| 8 | floor | basic_housing | 1.40% | 2.19% |
| 9 | emb_730d_weighted_mean_price | embedding_comparable | 1.09% | 0.45% |
| 10 | comp_730d_nearest_price | rule_based_comparable | 1.09% | 1.84% |
| 11 | comp_730d_median_price | rule_based_comparable | 0.87% | 1.23% |
| 12 | land_area_m2 | basic_housing | 0.85% | 3.68% |
| 13 | comp_365d_mean_price | rule_based_comparable | 0.84% | 0.92% |
| 14 | emb_1095d_median_price | embedding_comparable | 0.84% | 1.11% |
| 15 | emb_730d_mean_price | embedding_comparable | 0.83% | 0.96% |
| 16 | parking_area_m2 | basic_housing | 0.77% | 2.31% |
| 17 | emb_730d_nearest_price | embedding_comparable | 0.74% | 1.56% |
| 18 | total_floor | basic_housing | 0.71% | 2.41% |
| 19 | main_building_area_m2 | basic_housing | 0.57% | 2.67% |
| 20 | auxiliary_area_m2 | basic_housing | 0.56% | 2.36% |

## Feature Group Importance

| group | features | gain_share | split_share |
|---|---:|---:|---:|
| rule_based_comparable | 22 | 75.07% | 27.54% |
| embedding_comparable | 24 | 12.53% | 25.37% |
| basic_housing | 21 | 9.30% | 31.49% |
| time_aware_market | 8 | 1.98% | 10.90% |
| location_time | 7 | 0.93% | 3.97% |
| note_or_quality_flags | 4 | 0.19% | 0.72% |

## Embedding Comparable Features

| rank | feature | gain_share | split_share |
|---:|---|---:|---:|
| 4 | emb_1095d_weighted_mean_price | 2.60% | 0.55% |
| 6 | emb_1095d_mean_price | 1.73% | 1.05% |
| 7 | emb_1095d_nearest_price | 1.42% | 1.75% |
| 9 | emb_730d_weighted_mean_price | 1.09% | 0.45% |
| 14 | emb_1095d_median_price | 0.84% | 1.11% |
| 15 | emb_730d_mean_price | 0.83% | 0.96% |
| 17 | emb_730d_nearest_price | 0.74% | 1.56% |
| 28 | emb_730d_std_price | 0.35% | 1.70% |
| 32 | emb_1095d_std_price | 0.31% | 1.68% |
| 33 | emb_730d_median_age_diff | 0.31% | 1.54% |
| 34 | emb_1095d_median_age_diff | 0.29% | 1.56% |
| 41 | emb_730d_median_price | 0.24% | 1.06% |
| 42 | emb_1095d_median_area_diff_pct | 0.23% | 1.31% |
| 43 | emb_1095d_median_days_diff | 0.23% | 1.31% |
| 45 | emb_730d_median_days_diff | 0.22% | 1.39% |