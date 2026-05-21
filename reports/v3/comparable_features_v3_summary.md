# Comparable Features V3 Summary

## 1. Input / Output

- input dataset path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v2.csv`
- output v3 CSV path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v3.csv`
- output v3 parquet path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v3.parquet`
- feature config v2: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v2.json`
- feature config v3 root: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v3.json`
- feature config v3 report copy: `/home/nas2/Personal/Hank/Esuan/reports/v3/feature_config_model_v3.json`
- input row count: 90,481
- output row count: 90,481
- row count unchanged: true
- id missing count: 0
- id duplicate count: 0
- parquet output status: success

## 2. Comparable Rule

- same district
- same building_type
- `trade_date < current trade_date`
- windows: 365d and 730d
- top_k: 10

## 3. Distance Formula

`distance = 0.35 * area_diff_pct + 0.25 * age_diff_norm + 0.20 * days_diff_norm + 0.10 * floor_ratio_diff + 0.10 * parking_diff`

## 4. Leakage Control

- Comparable pool 嚴格使用歷史資料。
- 同日交易不會被使用。
- 未來資料不會被使用。
- `unit_price_ping` 只用於過去 comparable cases 的統計，不直接作為 current row feature。

### Leakage Check

| check_name | status | details |
| --- | --- | --- |
| v3_features_not_target_or_leakage | PASS | No comparable feature overlaps target/leakage/drop columns. |
| model_features_exclude_target_and_leakage | PASS | No target/leakage/drop columns in numeric/categorical features. |
| comp_365d_pool_uses_past_dates_only | PASS | Checked 100 sampled rows; max candidate trade_date is < current trade_date and same-day candidates are excluded. |
| comp_730d_pool_uses_past_dates_only | PASS | Checked 100 sampled rows; max candidate trade_date is < current trade_date and same-day candidates are excluded. |

## 5. New Features

| feature |
| --- |
| comp_365d_count |
| comp_365d_median_price |
| comp_365d_mean_price |
| comp_365d_weighted_mean_price |
| comp_365d_std_price |
| comp_365d_nearest_price |
| comp_365d_nearest_distance |
| comp_365d_median_distance |
| comp_365d_median_days_diff |
| comp_365d_median_area_diff_pct |
| comp_365d_median_age_diff |
| comp_730d_count |
| comp_730d_median_price |
| comp_730d_mean_price |
| comp_730d_weighted_mean_price |
| comp_730d_std_price |
| comp_730d_nearest_price |
| comp_730d_nearest_distance |
| comp_730d_median_distance |
| comp_730d_median_days_diff |
| comp_730d_median_area_diff_pct |
| comp_730d_median_age_diff |

## 6. Missing Values

| feature | missing_count | missing_ratio |
| --- | --- | --- |
| comp_365d_count | 0 | 0.0000 |
| comp_365d_median_price | 99 | 0.0011 |
| comp_365d_mean_price | 99 | 0.0011 |
| comp_365d_weighted_mean_price | 99 | 0.0011 |
| comp_365d_std_price | 99 | 0.0011 |
| comp_365d_nearest_price | 99 | 0.0011 |
| comp_365d_nearest_distance | 99 | 0.0011 |
| comp_365d_median_distance | 99 | 0.0011 |
| comp_365d_median_days_diff | 99 | 0.0011 |
| comp_365d_median_area_diff_pct | 99 | 0.0011 |
| comp_365d_median_age_diff | 6,947 | 0.0768 |
| comp_730d_count | 0 | 0.0000 |
| comp_730d_median_price | 77 | 0.0009 |
| comp_730d_mean_price | 77 | 0.0009 |
| comp_730d_weighted_mean_price | 77 | 0.0009 |
| comp_730d_std_price | 77 | 0.0009 |
| comp_730d_nearest_price | 77 | 0.0009 |
| comp_730d_nearest_distance | 77 | 0.0009 |
| comp_730d_median_distance | 77 | 0.0009 |
| comp_730d_median_days_diff | 77 | 0.0009 |
| comp_730d_median_area_diff_pct | 77 | 0.0009 |
| comp_730d_median_age_diff | 6,948 | 0.0768 |

## 7. Comparable Count Statistics

### Count Describe

| feature | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| comp_365d_count | 90,481.0000 | 9.9575 | 0.5555 | 0.0000 | 10.0000 | 10.0000 | 10.0000 | 10.0000 |
| comp_730d_count | 90,481.0000 | 9.9643 | 0.5028 | 0.0000 | 10.0000 | 10.0000 | 10.0000 | 10.0000 |

### Zero Count Ratio

| feature | zero_count | zero_ratio |
| --- | --- | --- |
| comp_365d_count | 99 | 0.0011 |
| comp_730d_count | 77 | 0.0009 |

## 8. Feature Config Update

- v2 numeric features: 34
- v3 numeric features: 56
- categorical features: 6

### Added Numeric Features

| numeric_feature |
| --- |
| comp_365d_count |
| comp_365d_median_price |
| comp_365d_mean_price |
| comp_365d_weighted_mean_price |
| comp_365d_std_price |
| comp_365d_nearest_price |
| comp_365d_nearest_distance |
| comp_365d_median_distance |
| comp_365d_median_days_diff |
| comp_365d_median_area_diff_pct |
| comp_365d_median_age_diff |
| comp_730d_count |
| comp_730d_median_price |
| comp_730d_mean_price |
| comp_730d_weighted_mean_price |
| comp_730d_std_price |
| comp_730d_nearest_price |
| comp_730d_nearest_distance |
| comp_730d_median_distance |
| comp_730d_median_days_diff |
| comp_730d_median_area_diff_pct |
| comp_730d_median_age_diff |

## 9. Next Step

- 下一步可使用 `data/processed/taipei_house_model_ready_v3.csv`。
- 搭配 `reports/feature_config_model_v3.json` 跑 Phase 3 training。
- 本步沒有訓練模型，也沒有覆蓋 v1 / v2 結果。
