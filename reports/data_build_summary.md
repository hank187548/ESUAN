# 臺北市住宅實價登錄資料建置摘要

## 1. 原始資料來源

- raw-dir: `/home/nas2/Personal/Hank/Esuan/Data`
- 找到的主檔數量: 13
- parking-mode: `keep`
- include-current: `true`

| source_file | read_rows |
| --- | --- |
| /home/nas2/Personal/Hank/Esuan/Data/112年第一季/a_lvr_land_a.csv | 4,761 |
| /home/nas2/Personal/Hank/Esuan/Data/112年第三季/a_lvr_land_a.csv | 6,392 |
| /home/nas2/Personal/Hank/Esuan/Data/112年第二季/a_lvr_land_a.csv | 6,116 |
| /home/nas2/Personal/Hank/Esuan/Data/112年第四季/a_lvr_land_a.csv | 6,713 |
| /home/nas2/Personal/Hank/Esuan/Data/113年第一季/a_lvr_land_a.csv | 6,251 |
| /home/nas2/Personal/Hank/Esuan/Data/113年第三季/a_lvr_land_a.csv | 7,702 |
| /home/nas2/Personal/Hank/Esuan/Data/113年第二季/a_lvr_land_a.csv | 7,342 |
| /home/nas2/Personal/Hank/Esuan/Data/113年第四季/a_lvr_land_a.csv | 6,093 |
| /home/nas2/Personal/Hank/Esuan/Data/114年第一季/a_lvr_land_a.csv | 5,345 |
| /home/nas2/Personal/Hank/Esuan/Data/114年第三季/a_lvr_land_a.csv | 5,443 |
| /home/nas2/Personal/Hank/Esuan/Data/114年第二季/a_lvr_land_a.csv | 5,744 |
| /home/nas2/Personal/Hank/Esuan/Data/114年第四季/a_lvr_land_a.csv | 5,820 |
| /home/nas2/Personal/Hank/Esuan/Data/本期/a_lvr_land_a.csv | 459 |


## 2. source_release 統計

| source_release | rows |
| --- | --- |
| 113Q3 | 7,701 |
| 113Q2 | 7,341 |
| 112Q4 | 6,712 |
| 112Q3 | 6,391 |
| 113Q1 | 6,250 |
| 112Q2 | 6,115 |
| 113Q4 | 6,092 |
| 114Q4 | 5,819 |
| 114Q2 | 5,743 |
| 114Q3 | 5,442 |
| 114Q1 | 5,344 |
| 112Q1 | 4,760 |
| CURRENT | 458 |


### source_order 對應表

| source_release | source_order |
| --- | --- |
| 112Q1 | 1,121 |
| 112Q2 | 1,122 |
| 112Q3 | 1,123 |
| 112Q4 | 1,124 |
| 113Q1 | 1,131 |
| 113Q2 | 1,132 |
| 113Q3 | 1,133 |
| 113Q4 | 1,134 |
| 114Q1 | 1,141 |
| 114Q2 | 1,142 |
| 114Q3 | 1,143 |
| 114Q4 | 1,144 |
| CURRENT | 999,999 |


## 3. 合併結果

| metric | value |
| --- | --- |
| 合併後總筆數 | 74,168 |
| 去重複前筆數 | 74,168 |
| 去重複後筆數 | 74,168 |
| 移除重複筆數 | 0 |
| 使用 id 去重複的筆數 | 74,168 |
| 使用 fallback key 去重複的筆數 | 0 |


## 4. 日期範圍

| metric | value |
| --- | --- |
| trade_date 最小日期 | 2009-04-29 |
| trade_date 最大日期 | 2026-04-18 |


### 每年交易筆數

| trade_year | rows |
| --- | --- |
| 2024 | 15,371 |
| 2023 | 15,049 |
| 2025 | 7,992 |
| 2022 | 3,464 |
| 2020 | 2,849 |
| 2021 | 2,390 |
| 2019 | 1,045 |
| 2018 | 324 |
| 2026 | 199 |
| 2017 | 14 |
| 2015 | 3 |
| 2009 | 2 |


### 每季交易筆數

| trade_yq | rows |
| --- | --- |
| 2024Q2 | 4,790 |
| 2023Q4 | 4,350 |
| 2024Q1 | 4,270 |
| 2023Q2 | 3,799 |
| 2023Q3 | 3,739 |
| 2024Q3 | 3,435 |
| 2023Q1 | 3,161 |
| 2024Q4 | 2,876 |
| 2025Q2 | 2,735 |
| 2025Q1 | 2,595 |
| 2022Q4 | 2,412 |
| 2025Q3 | 2,169 |
| 2020Q3 | 1,089 |
| 2020Q4 | 900 |
| 2021Q1 | 642 |
| 2021Q4 | 604 |
| 2021Q2 | 588 |
| 2020Q2 | 576 |
| 2021Q3 | 556 |
| 2019Q4 | 497 |
| 2025Q4 | 493 |
| 2022Q1 | 491 |
| 2019Q3 | 365 |
| 2022Q3 | 296 |
| 2020Q1 | 284 |
| 2022Q2 | 265 |
| 2026Q1 | 168 |
| 2019Q2 | 143 |
| 2018Q2 | 110 |
| 2018Q3 | 93 |
| 2018Q1 | 77 |
| 2018Q4 | 44 |
| 2019Q1 | 40 |
| 2026Q2 | 31 |
| 2017Q4 | 10 |
| 2015Q3 | 3 |
| 2017Q3 | 3 |
| 2009Q2 | 1 |
| 2009Q3 | 1 |
| 2017Q1 | 1 |


## 5. 資料分佈

### 各行政區筆數

| district | rows |
| --- | --- |
| 中山區 | 5,902 |
| 文山區 | 5,364 |
| 大安區 | 5,310 |
| 北投區 | 4,811 |
| 內湖區 | 4,691 |
| 士林區 | 4,181 |
| 萬華區 | 3,718 |
| 信義區 | 3,638 |
| 松山區 | 3,257 |
| 南港區 | 2,663 |
| 中正區 | 2,657 |
| 大同區 | 2,510 |


### 各建物型態筆數

| building_type | rows |
| --- | --- |
| 住宅大樓(11層含以上有電梯) | 21,049 |
| 公寓(5樓含以下無電梯) | 14,499 |
| 華廈(10層含以下有電梯) | 12,132 |
| 透天厝 | 1,022 |


### 各主要用途筆數

| main_use | rows |
| --- | --- |
| 住家用 | 47,248 |
| <NA> | 1,454 |


### 含車位 / 不含車位筆數

| has_parking | rows |
| --- | --- |
| 0 | 31,255 |
| 1 | 17,447 |


### special_note_flag 筆數

| special_note_flag | rows |
| --- | --- |
| 0 | 25,650 |
| 1 | 23,052 |


## 6. 目標值統計

target: `unit_price_ping`，單位為萬元 / 坪；統計來源為 model_ready dataset。

| stat | value |
| --- | --- |
| count | 25,424.0000 |
| mean | 81.5755 |
| std | 26.4957 |
| min | 22.0783 |
| 25% | 63.0666 |
| 50% | 77.9788 |
| 75% | 95.0240 |
| max | 182.4562 |


## 7. outlier 過濾

| metric | value |
| --- | --- |
| 是否啟用 drop-outliers | true |
| q01 門檻 | 21.7933 |
| q99 門檻 | 182.7561 |
| 過濾前筆數 | 49,696 |
| 過濾後筆數 | 48,702 |


## 8. 最終輸出

| dataset | rows |
| --- | --- |
| clean_all | 48,702 |
| clean_no_parking | 31,255 |
| model_ready | 25,424 |


- feature_config: `/home/nas2/Personal/Hank/Esuan/data/processed/feature_config.json`

| path | format | status | error |
| --- | --- | --- | --- |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_raw_combined.parquet | parquet | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_raw_combined.csv | csv | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_clean_all.parquet | parquet | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_clean_all.csv | csv | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_clean_no_parking.parquet | parquet | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_clean_no_parking.csv | csv | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready.parquet | parquet | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready.csv | csv | ok |  |
| /home/nas2/Personal/Hank/Esuan/reports/filter_log.csv | csv | ok |  |


### Parquet 輸出狀態

所有 parquet 輸出成功。


## 9. 特別提醒

- `source_release` 是資料發布批次，不代表交易日期。
- 未來模型時間切分應該使用 `trade_date`。
- `id` / `transfer_id` 不應作為模型特徵。
- `total_price`、`unit_price_m2`、`unit_price_ping`、`parking_price` 屬於 leakage 或 target 相關欄位，不應放入 feature。
