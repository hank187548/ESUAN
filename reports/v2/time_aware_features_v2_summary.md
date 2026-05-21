# Time-Aware Features V2 Summary

## 1. Input / Output

- input dataset path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready.csv`
- output v2 CSV path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v2.csv`
- output v2 parquet path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v2.parquet`
- feature config v1: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v1.json`
- feature config v2: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v2.json`
- rolling folds path: `/home/nas2/Personal/Hank/Esuan/data/processed/rolling_folds.csv`
- rolling fold count: 15
- input row count: 90,481
- output row count: 90,481
- row count unchanged: true
- id missing count: 0
- id duplicate count: 0
- parquet output status: success

## 2. Added Features

| feature |
| --- |
| district_median_price_180d |
| district_median_price_365d |
| district_count_180d |
| district_type_median_price_180d |
| district_type_median_price_365d |
| district_type_count_180d |
| district_price_change_180_365 |
| district_type_price_change_180_365 |

## 3. Leakage Control

- 所有歷史行情特徵只使用 `trade_date < current trade_date`。
- 同日交易不會被納入 historical pool。
- 未來資料不會被使用。
- `unit_price_ping` 只用於產生歷史統計，不會作為模型 feature。
- 實作使用 group 內依 `trade_date` 排序後的 `searchsorted`，每筆資料的 right boundary 是目前日期的第一個位置，因此排除同日全部交易與自己。

### Leakage Check

| check_name | status | details |
| --- | --- | --- |
| v2_features_not_target_or_leakage | PASS | No v2 feature overlaps target/leakage/drop columns. |
| model_features_exclude_target_and_leakage | PASS | No target/leakage columns in numeric/categorical features. |
| district_180d_history_uses_past_dates_only | PASS | Checked 100 sampled rows; all max history trade_date values are < current trade_date. |
| district_type_180d_history_uses_past_dates_only | PASS | Checked 100 sampled rows; all max history trade_date values are < current trade_date. |

## 4. Missing Values

| column | missing_count | missing_ratio |
| --- | --- | --- |
| district_median_price_180d | 56 | 0.0006 |
| district_median_price_365d | 39 | 0.0004 |
| district_count_180d | 0 | 0.0000 |
| district_type_median_price_180d | 130 | 0.0014 |
| district_type_median_price_365d | 99 | 0.0011 |
| district_type_count_180d | 0 | 0.0000 |
| district_price_change_180_365 | 56 | 0.0006 |
| district_type_price_change_180_365 | 130 | 0.0014 |

## 5. Basic Statistics

| column | min | mean | median | max |
| --- | --- | --- | --- | --- |
| district_median_price_180d | 27.4088 | 67.1809 | 65.0830 | 145.3697 |
| district_median_price_365d | 27.6056 | 66.6831 | 64.4737 | 169.4896 |
| district_count_180d | 0.0000 | 571.7478 | 550.0000 | 1,163.0000 |
| district_type_median_price_180d | 27.4088 | 67.9793 | 65.6370 | 153.0440 |
| district_type_median_price_365d | 27.4258 | 67.4045 | 64.8269 | 169.4896 |
| district_type_count_180d | 0.0000 | 195.4217 | 184.0000 | 599.0000 |
| district_price_change_180_365 | -0.6722 | 0.0073 | 0.0081 | 0.4477 |
| district_type_price_change_180_365 | -0.4586 | 0.0086 | 0.0072 | 0.6801 |

## 6. Feature Config Update

- v1 numeric features: 26
- v2 numeric features: 34
- categorical features: 6

### Added Numeric Features

| numeric_feature |
| --- |
| district_median_price_180d |
| district_median_price_365d |
| district_count_180d |
| district_type_median_price_180d |
| district_type_median_price_365d |
| district_type_count_180d |
| district_price_change_180_365 |
| district_type_price_change_180_365 |

## 7. Next Step

- 下一步可使用 `data/processed/taipei_house_model_ready_v2.csv`。
- 搭配 `reports/feature_config_model_v2.json` 重新跑 Phase 2 training。
- 本步沒有訓練模型，也沒有覆蓋 v1 dataset 或 v1 feature config。
