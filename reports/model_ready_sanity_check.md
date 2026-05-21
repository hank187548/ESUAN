# Model Ready Sanity Check

## Inputs

- data-path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready.csv`
- feature-config: `/home/nas2/Personal/Hank/Esuan/data/processed/feature_config.json`

## Basic Dataset Status

| metric | value |
| --- | --- |
| row_count | 33,894 |
| column_count | 54 |
| id_missing_count | 0 |
| id_duplicate_count | 0 |
| trade_date_min | 2018-06-03 |
| trade_date_max | 2026-04-18 |

## Target Checks

| metric | value |
| --- | --- |
| target_col | unit_price_ping |
| missing_count | 0 |
| non_positive_count | 0 |

### unit_price_ping Summary

| stat | value |
| --- | --- |
| count | 33,894.0000 |
| mean | 77.5631 |
| std | 25.7402 |
| min | 22.0783 |
| 25% | 59.3040 |
| 50% | 73.8712 |
| 75% | 91.2879 |
| max | 182.4522 |

### 每年筆數

| trade_year | rows |
| --- | --- |
| 2018 | 4 |
| 2019 | 71 |
| 2020 | 154 |
| 2021 | 65 |
| 2022 | 2,010 |
| 2023 | 12,378 |
| 2024 | 12,556 |
| 2025 | 6,494 |
| 2026 | 162 |

### 每季筆數

| trade_yq | rows |
| --- | --- |
| 2018Q2 | 1 |
| 2018Q3 | 1 |
| 2018Q4 | 2 |
| 2019Q2 | 5 |
| 2019Q3 | 33 |
| 2019Q4 | 33 |
| 2020Q1 | 22 |
| 2020Q2 | 22 |
| 2020Q3 | 42 |
| 2020Q4 | 68 |
| 2021Q1 | 40 |
| 2021Q2 | 15 |
| 2021Q3 | 7 |
| 2021Q4 | 3 |
| 2022Q1 | 6 |
| 2022Q2 | 25 |
| 2022Q3 | 121 |
| 2022Q4 | 1,858 |
| 2023Q1 | 2,572 |
| 2023Q2 | 3,176 |
| 2023Q3 | 3,150 |
| 2023Q4 | 3,480 |
| 2024Q1 | 3,484 |
| 2024Q2 | 4,029 |
| 2024Q3 | 2,831 |
| 2024Q4 | 2,212 |
| 2025Q1 | 2,093 |
| 2025Q2 | 2,276 |
| 2025Q3 | 1,772 |
| 2025Q4 | 353 |
| 2026Q1 | 137 |
| 2026Q2 | 25 |

### 各行政區筆數

| district | rows |
| --- | --- |
| 中山區 | 4,418 |
| 中正區 | 1,843 |
| 信義區 | 2,653 |
| 內湖區 | 3,842 |
| 北投區 | 3,260 |
| 南港區 | 1,505 |
| 士林區 | 2,867 |
| 大同區 | 1,490 |
| 大安區 | 3,868 |
| 文山區 | 3,375 |
| 松山區 | 2,490 |
| 萬華區 | 2,283 |

### 各 building_type 筆數

| building_type | rows |
| --- | --- |
| 住宅大樓(11層含以上有電梯) | 12,915 |
| 公寓(5樓含以下無電梯) | 11,289 |
| 華廈(10層含以下有電梯) | 9,690 |

### 各行政區 unit_price_ping 中位數

| district | unit_price_ping_median | rows |
| --- | --- | --- |
| 大安區 | 100.5233 | 3,868 |
| 中正區 | 90.8664 | 1,843 |
| 松山區 | 85.7658 | 2,490 |
| 信義區 | 81.5021 | 2,653 |
| 中山區 | 78.4788 | 4,418 |
| 南港區 | 75.3696 | 1,505 |
| 大同區 | 75.0572 | 1,490 |
| 內湖區 | 67.8443 | 3,842 |
| 士林區 | 65.1788 | 2,867 |
| 萬華區 | 60.1051 | 2,283 |
| 文山區 | 57.6704 | 3,375 |
| 北投區 | 56.8912 | 3,260 |

### 各 building_type unit_price_ping 中位數

| building_type | unit_price_ping_median | rows |
| --- | --- | --- |
| 住宅大樓(11層含以上有電梯) | 84.6000 | 12,915 |
| 華廈(10層含以下有電梯) | 75.9547 | 9,690 |
| 公寓(5樓含以下無電梯) | 61.0516 | 11,289 |

## Missing Values

- output: `/home/nas2/Personal/Hank/Esuan/reports/missing_value_report.csv`
- columns_with_missing: 8

## Zero Variance Columns

- output: `/home/nas2/Personal/Hank/Esuan/reports/zero_variance_columns.csv`
| column | dtype | missing_count | non_missing_count | unique_count | single_value |
| --- | --- | --- | --- | --- | --- |
| abnormal_transaction_flag | int64 | 0 | 33,894 | 1 | 0 |
| area_outlier_flag | int64 | 0 | 33,894 | 1 | 0 |
| layout_outlier_flag | int64 | 0 | 33,894 | 1 | 0 |
| main_use | str | 771 | 33,123 | 1 | 住家用 |
| presale_note_flag | int64 | 0 | 33,894 | 1 | 0 |
| separate_registration_flag | int64 | 0 | 33,894 | 1 | 0 |
| special_note_flag | int64 | 0 | 33,894 | 1 | 0 |

## Categorical Levels

- output: `/home/nas2/Personal/Hank/Esuan/reports/categorical_levels_report.csv`

## Leakage Check

- output: `/home/nas2/Personal/Hank/Esuan/reports/leakage_check_report.csv`
| column | in_numeric_features | in_categorical_features | in_drop_cols | status |
| --- | --- | --- | --- | --- |
| address_raw | 0 | 0 | 1 | PASS |
| id | 0 | 0 | 1 | PASS |
| note_raw | 0 | 0 | 1 | PASS |
| parking_price | 0 | 0 | 0 | PASS |
| source_file | 0 | 0 | 1 | PASS |
| source_folder | 0 | 0 | 1 | PASS |
| source_order | 0 | 0 | 1 | PASS |
| source_release | 0 | 0 | 1 | PASS |
| total_price | 0 | 0 | 0 | PASS |
| total_price_wan | 0 | 0 | 0 | PASS |
| transfer_id | 0 | 0 | 1 | PASS |
| unit_price_m2 | 0 | 0 | 0 | PASS |
| unit_price_ping | 0 | 0 | 0 | PASS |

## feature_config_model_v1

- output: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v1.json`
- numeric_features: 26
- categorical_features: 6
- zero_variance_removed: abnormal_transaction_flag, area_outlier_flag, layout_outlier_flag, main_use, presale_note_flag, separate_registration_flag, special_note_flag
