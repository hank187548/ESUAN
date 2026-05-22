# Phase 3 Explainability Report

## 1. Analysis Setup

- dataset path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v3.csv`
- predictions path: `/home/nas2/Personal/Hank/Esuan/data/processed/phase3_oof_predictions.csv`
- feature config path: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v3.json`
- model directory: `/home/nas2/Personal/Hank/Esuan/models/phase3`
- model_name: `tree_model`
- split: `test`
- row count: 36,429

## 2. Feature Importance Findings

### Top 20 Gain Importance Features

| feature | mean_gain_importance | rank_gain | feature_group |
| --- | --- | --- | --- |
| comp_730d_weighted_mean_price | 172,817,195.0502 | 1 | comparable_sales |
| comp_730d_mean_price | 33,641,188.9962 | 2 | comparable_sales |
| comp_365d_weighted_mean_price | 11,747,729.6524 | 3 | comparable_sales |
| building_age | 6,191,285.4516 | 4 | basic_housing |
| floor | 4,613,912.6956 | 5 | basic_housing |
| comp_730d_median_price | 3,953,243.2722 | 6 | comparable_sales |
| total_floor | 3,548,937.0180 | 7 | basic_housing |
| land_area_m2 | 3,113,775.1763 | 8 | basic_housing |
| comp_730d_nearest_price | 3,022,060.1099 | 9 | comparable_sales |
| parking_area_m2 | 2,681,490.9410 | 10 | basic_housing |
| rooms | 2,469,970.2484 | 11 | basic_housing |
| district_type_median_price_365d | 2,316,406.4945 | 12 | time_aware_market |
| district_median_price_365d | 2,299,010.1618 | 13 | time_aware_market |
| auxiliary_area_m2 | 2,223,320.1434 | 14 | basic_housing |
| main_building_area_m2 | 2,128,279.7121 | 15 | basic_housing |
| comp_365d_mean_price | 2,062,730.8564 | 16 | comparable_sales |
| building_area_m2 | 1,935,831.0240 | 17 | basic_housing |
| district_median_price_180d | 1,878,959.7262 | 18 | time_aware_market |
| comp_730d_std_price | 1,802,688.8372 | 19 | comparable_sales |
| material_鋼筋混凝土造 | 1,760,731.5884 | 20 | other |

### Feature Group Importance

| feature_group | total_gain_importance | total_split_importance | feature_count | gain_share | split_share |
| --- | --- | --- | --- | --- | --- |
| comparable_sales | 241,208,581.0884 | 11,252.9333 | 22 | 0.8216 | 0.3723 |
| basic_housing | 35,426,349.5227 | 12,305.9333 | 26 | 0.1207 | 0.4072 |
| time_aware_market | 11,569,238.4642 | 4,674.6000 | 8 | 0.0394 | 0.1547 |
| other | 2,490,579.9699 | 430.8444 | 19 | 0.0085 | 0.0143 |
| location_time | 2,301,341.9063 | 1,330.1105 | 55 | 0.0078 | 0.0440 |
| note_flags | 598,479.7497 | 228.4667 | 3 | 0.0020 | 0.0076 |

- Comparable sales features in top 30 gain: 12
- Time-aware market features in top 30 gain: 5
- Feature names are transformed pipeline names where categorical one-hot levels are present.
- Importance warnings: (none)

## 3. Prediction IC Findings

| n | pearson_ic | spearman_rank_ic | mae | rmse | mape | r2 | bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 36,429.0000 | 0.8456 | 0.8357 | 10.1399 | 13.5632 | 14.1311 | 0.7142 | -0.1282 |

### By Quarter

| test_quarter | n | pearson_ic | spearman_rank_ic | mae | mape | r2 | bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022Q3 | 969 | 0.8339 | 0.8257 | 9.8256 | 14.5946 | 0.6923 | -1.0141 |
| 2022Q4 | 1,855 | 0.8451 | 0.8415 | 9.6679 | 14.4917 | 0.7125 | 0.4842 |
| 2023Q1 | 2,572 | 0.8355 | 0.8263 | 9.2995 | 14.0112 | 0.6979 | 0.2218 |
| 2023Q2 | 3,175 | 0.8333 | 0.8214 | 9.8343 | 14.2636 | 0.6939 | -0.2238 |
| 2023Q3 | 3,144 | 0.8535 | 0.8436 | 9.6397 | 13.7235 | 0.7273 | -0.3913 |
| 2023Q4 | 3,475 | 0.8473 | 0.8349 | 9.6701 | 13.6605 | 0.7162 | -0.6826 |
| 2024Q1 | 3,482 | 0.8458 | 0.8280 | 10.2858 | 14.2847 | 0.7116 | -1.2228 |
| 2024Q2 | 4,025 | 0.8479 | 0.8355 | 10.2857 | 13.5438 | 0.7128 | -1.5272 |
| 2024Q3 | 2,827 | 0.8362 | 0.8281 | 10.6145 | 13.8683 | 0.6945 | -1.2850 |
| 2024Q4 | 2,204 | 0.8205 | 0.8104 | 10.8192 | 15.3000 | 0.6680 | 1.8046 |
| 2025Q1 | 2,098 | 0.8324 | 0.8257 | 10.7977 | 14.6647 | 0.6906 | 0.8512 |
| 2025Q2 | 2,307 | 0.8497 | 0.8396 | 10.4383 | 14.3797 | 0.7182 | 1.2898 |
| 2025Q3 | 2,037 | 0.8632 | 0.8576 | 10.1322 | 13.6725 | 0.7414 | 1.1655 |
| 2025Q4 | 1,904 | 0.8444 | 0.8331 | 10.6969 | 14.1634 | 0.7116 | 0.9662 |
| 2026Q1 | 355 | 0.8534 | 0.8610 | 11.4587 | 16.5578 | 0.7241 | 0.6798 |

- Best quarter by Rank IC: 2026Q1
- Worst quarter by Rank IC: 2024Q4
- 2026Q1 sample is small and should be interpreted cautiously.

## 4. Residual Findings

| n | mae | rmse | mape | r2 | bias | y_true_mean | y_pred_mean | error_std | abs_error_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 36,429.0000 | 10.1399 | 13.5632 | 14.1311 | 0.7142 | -0.1282 | 77.3680 | 77.2398 | 13.5628 | 7.8694 |

### By Price Segment

| price_segment | n | y_true_mean | y_pred_mean | mae | rmse | mape | bias | error_std | abs_error_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-50 | 4,292 | 42.4681 | 53.3291 | 11.1768 | 14.1282 | 28.0707 | 10.8609 | 9.0368 | 9.5773 |
| 120+ | 2,312 | 138.9404 | 119.0491 | 21.4181 | 26.5511 | 15.1966 | -19.8913 | 17.5906 | 18.3780 |
| 50-80 | 17,334 | 65.3240 | 68.6042 | 8.3451 | 11.0963 | 13.0194 | 3.2802 | 10.6007 | 6.4566 |
| 80-120 | 12,491 | 94.6769 | 89.7009 | 10.1867 | 12.9064 | 10.6868 | -4.9760 | 11.9090 | 8.4775 |

- High-price segment bias (`120+`): -19.8913; negative means under-prediction.
- Low-price segment bias (`0-50`): 10.8609; positive means over-prediction.
- Highest MAE segment: 120+
- Highest MAPE segment: 0-50

## 5. Correlation Findings

### Top Target-Correlated Features

| feature | pearson_corr_with_target | abs_corr | feature_group |
| --- | --- | --- | --- |
| comp_730d_weighted_mean_price | 0.8198 | 0.8198 | comparable_sales |
| comp_730d_mean_price | 0.8124 | 0.8124 | comparable_sales |
| comp_365d_weighted_mean_price | 0.8123 | 0.8123 | comparable_sales |
| comp_365d_mean_price | 0.8051 | 0.8051 | comparable_sales |
| comp_730d_median_price | 0.8015 | 0.8015 | comparable_sales |
| comp_365d_median_price | 0.7941 | 0.7941 | comparable_sales |
| comp_730d_nearest_price | 0.7372 | 0.7372 | comparable_sales |
| comp_365d_nearest_price | 0.7259 | 0.7259 | comparable_sales |
| district_type_median_price_180d | 0.7060 | 0.7060 | time_aware_market |
| district_type_median_price_365d | 0.7040 | 0.7040 | time_aware_market |
| district_median_price_180d | 0.6243 | 0.6243 | time_aware_market |
| district_median_price_365d | 0.6218 | 0.6218 | time_aware_market |
| comp_365d_std_price | 0.4395 | 0.4395 | comparable_sales |
| comp_730d_std_price | 0.4372 | 0.4372 | comparable_sales |
| total_floor | 0.3364 | 0.3364 | basic_housing |
| has_elevator | 0.3327 | 0.3327 | basic_housing |
| building_age | -0.2889 | 0.2889 | basic_housing |
| trade_year | 0.2523 | 0.2523 | location_time |
| has_management | 0.2451 | 0.2451 | basic_housing |
| floor | 0.2342 | 0.2342 | basic_housing |

- High correlation feature pairs count (`abs(corr) >= 0.90`): 25

## 6. Comparable Feature Usefulness

| feature | feature_importance_gain_rank | mean_gain_importance | pearson_corr_with_target | missing_ratio | feature_type |
| --- | --- | --- | --- | --- | --- |
| comp_730d_weighted_mean_price | 1 | 172,817,195.0502 | 0.8198 | 0.0009 | weighted_mean_price |
| comp_730d_mean_price | 2 | 33,641,188.9962 | 0.8124 | 0.0009 | mean_price |
| comp_365d_weighted_mean_price | 3 | 11,747,729.6524 | 0.8123 | 0.0011 | weighted_mean_price |
| comp_730d_median_price | 6 | 3,953,243.2722 | 0.8015 | 0.0009 | median_price |
| comp_730d_nearest_price | 9 | 3,022,060.1099 | 0.7372 | 0.0009 | nearest_price |
| comp_365d_mean_price | 16 | 2,062,730.8564 | 0.8051 | 0.0011 | mean_price |
| comp_730d_std_price | 19 | 1,802,688.8372 | 0.4372 | 0.0009 | std_price |
| comp_730d_median_age_diff | 21 | 1,693,323.3547 | 0.0095 | 0.0768 | age_diff |
| comp_365d_median_age_diff | 24 | 1,235,294.2936 | 0.0165 | 0.0768 | age_diff |
| comp_365d_std_price | 25 | 1,200,310.4932 | 0.4395 | 0.0011 | std_price |
| comp_730d_nearest_distance | 27 | 1,059,787.6742 | 0.0648 | 0.0009 | distance |
| comp_365d_nearest_price | 29 | 956,396.6504 | 0.7259 | 0.0011 | nearest_price |
| comp_365d_median_price | 32 | 842,824.3471 | 0.7941 | 0.0011 | median_price |
| comp_365d_median_distance | 34 | 815,860.4648 | 0.1273 | 0.0011 | distance |
| comp_730d_median_distance | 35 | 794,029.8950 | 0.0987 | 0.0009 | distance |
| comp_730d_median_area_diff_pct | 37 | 785,076.9795 | 0.0309 | 0.0009 | area_diff |
| comp_365d_nearest_distance | 38 | 763,159.6602 | 0.0836 | 0.0011 | distance |
| comp_730d_median_days_diff | 39 | 697,146.4805 | 0.1630 | 0.0009 | days_diff |
| comp_365d_median_days_diff | 40 | 665,037.4650 | 0.1698 | 0.0011 | days_diff |
| comp_365d_median_area_diff_pct | 41 | 653,496.5560 | 0.0505 | 0.0011 | area_diff |

## 7. SHAP Findings

- SHAP success: true
- SHAP completed. Using last fold model fold_id=15. Last fold test n was small, so sampled from all test predictions. Sample rows=5000. Values saved to /home/nas2/Personal/Hank/Esuan/analysis/phase3_explainability/shap/shap_values_sample.parquet.

### Top SHAP Features

| feature | mean_abs_shap | feature_group |
| --- | --- | --- |
| comp_730d_weighted_mean_price | 10.1766 | comparable_sales |
| building_age | 2.4847 | basic_housing |
| floor | 1.5677 | basic_housing |
| building_area_m2 | 1.5539 | basic_housing |
| comp_730d_mean_price | 1.3538 | comparable_sales |
| district_median_price_365d | 1.3291 | time_aware_market |
| parking_area_m2 | 1.3012 | basic_housing |
| total_floor | 0.9654 | basic_housing |
| district_type_median_price_365d | 0.9270 | time_aware_market |
| district_median_price_180d | 0.9122 | time_aware_market |

## 8. Next Steps

- 檢查高價 under-prediction cases。
- 考慮加入更細的位置資訊，例如路段、捷運距離、社區名稱。
- 若需要，再評估 embedding-based comparable retrieval。
