# Model Ready Sanity Check

## Inputs

- data-path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready.csv`
- feature-config: `/home/nas2/Personal/Hank/Esuan/data/processed/feature_config.json`

## Basic Dataset Status

| metric | value |
| --- | --- |
| row_count | 90,481 |
| column_count | 54 |
| id_missing_count | 0 |
| id_duplicate_count | 0 |
| trade_date_min | 2005-05-12 |
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
| count | 90,481.0000 |
| mean | 70.3681 |
| std | 24.2223 |
| min | 20.7825 |
| 25% | 53.0215 |
| 50% | 66.6374 |
| 75% | 83.5299 |
| max | 175.6559 |

### 每年筆數

| trade_year | rows |
| --- | --- |
| 2005 | 1 |
| 2011 | 2 |
| 2012 | 8 |
| 2013 | 5 |
| 2014 | 27 |
| 2015 | 122 |
| 2016 | 223 |
| 2017 | 453 |
| 2018 | 3,521 |
| 2019 | 13,970 |
| 2020 | 15,360 |
| 2021 | 14,406 |
| 2022 | 8,753 |
| 2023 | 12,366 |
| 2024 | 12,538 |
| 2025 | 8,346 |
| 2026 | 380 |

### 每季筆數

| trade_yq | rows |
| --- | --- |
| 2005Q2 | 1 |
| 2011Q3 | 1 |
| 2011Q4 | 1 |
| 2012Q1 | 1 |
| 2012Q2 | 2 |
| 2012Q3 | 4 |
| 2012Q4 | 1 |
| 2013Q1 | 1 |
| 2013Q2 | 2 |
| 2013Q3 | 1 |
| 2013Q4 | 1 |
| 2014Q2 | 6 |
| 2014Q3 | 11 |
| 2014Q4 | 10 |
| 2015Q1 | 23 |
| 2015Q2 | 31 |
| 2015Q3 | 34 |
| 2015Q4 | 34 |
| 2016Q1 | 52 |
| 2016Q2 | 64 |
| 2016Q3 | 61 |
| 2016Q4 | 46 |
| 2017Q1 | 84 |
| 2017Q2 | 119 |
| 2017Q3 | 120 |
| 2017Q4 | 130 |
| 2018Q1 | 184 |
| 2018Q2 | 170 |
| 2018Q3 | 367 |
| 2018Q4 | 2,800 |
| 2019Q1 | 2,900 |
| 2019Q2 | 3,862 |
| 2019Q3 | 3,420 |
| 2019Q4 | 3,788 |
| 2020Q1 | 2,982 |
| 2020Q2 | 3,833 |
| 2020Q3 | 4,370 |
| 2020Q4 | 4,175 |
| 2021Q1 | 3,653 |
| 2021Q2 | 3,284 |
| 2021Q3 | 3,449 |
| 2021Q4 | 4,020 |
| 2022Q1 | 3,169 |
| 2022Q2 | 2,760 |
| 2022Q3 | 969 |
| 2022Q4 | 1,855 |
| 2023Q1 | 2,572 |
| 2023Q2 | 3,175 |
| 2023Q3 | 3,144 |
| 2023Q4 | 3,475 |
| 2024Q1 | 3,482 |
| 2024Q2 | 4,025 |
| 2024Q3 | 2,827 |
| 2024Q4 | 2,204 |
| 2025Q1 | 2,098 |
| 2025Q2 | 2,307 |
| 2025Q3 | 2,037 |
| 2025Q4 | 1,904 |
| 2026Q1 | 355 |
| 2026Q2 | 25 |

### 各行政區筆數

| district | rows |
| --- | --- |
| 中山區 | 11,862 |
| 中正區 | 5,039 |
| 信義區 | 6,874 |
| 內湖區 | 11,599 |
| 北投區 | 8,699 |
| 南港區 | 4,034 |
| 士林區 | 7,418 |
| 大同區 | 3,754 |
| 大安區 | 9,353 |
| 文山區 | 9,793 |
| 松山區 | 6,279 |
| 萬華區 | 5,777 |

### 各 building_type 筆數

| building_type | rows |
| --- | --- |
| 住宅大樓(11層含以上有電梯) | 35,136 |
| 公寓(5樓含以下無電梯) | 27,568 |
| 套房(1房1廳1衛) | 2,485 |
| 華廈(10層含以下有電梯) | 25,292 |

### 各行政區 unit_price_ping 中位數

| district | unit_price_ping_median | rows |
| --- | --- | --- |
| 大安區 | 92.8681 | 9,353 |
| 中正區 | 83.3511 | 5,039 |
| 松山區 | 78.2734 | 6,279 |
| 信義區 | 75.4202 | 6,874 |
| 中山區 | 72.7891 | 11,862 |
| 大同區 | 67.1043 | 3,754 |
| 南港區 | 65.0094 | 4,034 |
| 內湖區 | 61.1831 | 11,599 |
| 士林區 | 59.9013 | 7,418 |
| 萬華區 | 54.2985 | 5,777 |
| 文山區 | 52.8026 | 9,793 |
| 北投區 | 51.2658 | 8,699 |

### 各 building_type unit_price_ping 中位數

| building_type | unit_price_ping_median | rows |
| --- | --- | --- |
| 住宅大樓(11層含以上有電梯) | 75.4056 | 35,136 |
| 華廈(10層含以下有電梯) | 68.5678 | 25,292 |
| 套房(1房1廳1衛) | 64.4109 | 2,485 |
| 公寓(5樓含以下無電梯) | 55.1319 | 27,568 |

## Missing Values

- output: `/home/nas2/Personal/Hank/Esuan/reports/missing_value_report.csv`
- columns_with_missing: 9

## Zero Variance Columns

- output: `/home/nas2/Personal/Hank/Esuan/reports/zero_variance_columns.csv`
| column | dtype | missing_count | non_missing_count | unique_count | single_value |
| --- | --- | --- | --- | --- | --- |
| abnormal_transaction_flag | int64 | 0 | 90,481 | 1 | 0 |
| area_outlier_flag | int64 | 0 | 90,481 | 1 | 0 |
| layout_outlier_flag | int64 | 0 | 90,481 | 1 | 0 |
| main_use | str | 1,077 | 89,404 | 1 | 住家用 |
| presale_note_flag | int64 | 0 | 90,481 | 1 | 0 |
| separate_registration_flag | int64 | 0 | 90,481 | 1 | 0 |
| special_note_flag | int64 | 0 | 90,481 | 1 | 0 |

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
