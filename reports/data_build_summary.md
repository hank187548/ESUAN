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


### abnormal_transaction_flag 筆數

| abnormal_transaction_flag | rows |
| --- | --- |
| 0 | 42,193 |
| 1 | 6,509 |


### physical_condition_flag 筆數

| physical_condition_flag | rows |
| --- | --- |
| 0 | 31,749 |
| 1 | 16,953 |


### renovation_flag 筆數

| renovation_flag | rows |
| --- | --- |
| 0 | 48,424 |
| 1 | 278 |


### broad_note_flag 筆數

| broad_note_flag | rows |
| --- | --- |
| 0 | 28,697 |
| 1 | 20,005 |


### special_note_flag 筆數

| special_note_flag | rows |
| --- | --- |
| 0 | 42,193 |
| 1 | 6,509 |


### 各 note flag = 1 筆數與比例

| flag | rows | flagged_rows | flagged_ratio |
| --- | --- | --- | --- |
| abnormal_transaction_flag | 48,702 | 6,509 | 13.36% |
| physical_condition_flag | 48,702 | 16,953 | 34.81% |
| renovation_flag | 48,702 | 278 | 0.57% |
| broad_note_flag | 48,702 | 20,005 | 41.08% |
| special_note_flag | 48,702 | 6,509 | 13.36% |


### building_age 缺失筆數與比例

| dataset | rows | building_age_missing_rows | missing_ratio |
| --- | --- | --- | --- |
| clean_all | 48,702 | 9,821 | 20.17% |
| model_ready | 33,894 | 1,116 | 3.29% |
| model_ready_with_presale | 41,300 | 8,419 | 20.38% |
| model_ready_strict | 18,008 | 655 | 3.64% |


### presale_note_flag 筆數

| presale_note_flag | rows |
| --- | --- |
| 0 | 40,748 |
| 1 | 7,954 |


### separate_registration_flag 筆數

| separate_registration_flag | rows |
| --- | --- |
| 0 | 40,746 |
| 1 | 7,956 |


### area_outlier_flag 筆數

| area_outlier_flag | rows |
| --- | --- |
| 0 | 48,512 |
| 1 | 190 |


### layout_outlier_flag 筆數

| layout_outlier_flag | rows |
| --- | --- |
| 0 | 48,461 |
| 1 | 241 |


### layout_outlier_flag = 1 筆數與比例

| flag | rows | flagged_rows | flagged_ratio |
| --- | --- | --- | --- |
| layout_outlier_flag | 48,702 | 241 | 0.49% |


### rooms / living_rooms / bathrooms 統計

| column | stat | value |
| --- | --- | --- |
| rooms | count | 48,702.0000 |
| rooms | mean | 2.4920 |
| rooms | std | 1.2718 |
| rooms | min | 0.0000 |
| rooms | 25% | 2.0000 |
| rooms | 50% | 3.0000 |
| rooms | 75% | 3.0000 |
| rooms | max | 42.0000 |
| living_rooms | count | 48,702.0000 |
| living_rooms | mean | 1.6336 |
| living_rooms | std | 0.7023 |
| living_rooms | min | 0.0000 |
| living_rooms | 25% | 1.0000 |
| living_rooms | 50% | 2.0000 |
| living_rooms | 75% | 2.0000 |
| living_rooms | max | 26.0000 |
| bathrooms | count | 48,702.0000 |
| bathrooms | mean | 1.5901 |
| bathrooms | std | 0.8682 |
| bathrooms | min | 0.0000 |
| bathrooms | 25% | 1.0000 |
| bathrooms | 50% | 2.0000 |
| bathrooms | 75% | 2.0000 |
| bathrooms | max | 32.0000 |


## 6. 目標值統計

target: `unit_price_ping`，單位為萬元 / 坪；統計來源為 model_ready dataset。

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
| taipei_house_model_ready.csv | 33,894 |
| taipei_house_model_ready_with_presale.csv | 41,300 |
| taipei_house_model_ready_strict.csv | 18,008 |


### model_ready 篩選補充統計

| step | rows |
| --- | --- |
| 排除異常交易備註前 | 48,702 |
| 排除異常交易備註後 | 42,193 |
| base filters 後 | 41,510 |
| 排除預售屋前 | 41,510 |
| 排除預售屋後 | 34,104 |
| 排除分件登記前 | 34,104 |
| 排除分件登記後 | 34,102 |
| 排除面積極端前 | 34,102 |
| 排除面積極端後 | 34,037 |
| 排除格局極端前 | 34,037 |
| 排除格局極端後 | 33,894 |
| with_presale 排除異常交易備註前 | 48,702 |
| with_presale 排除異常交易備註後 | 42,193 |
| with_presale base filters 後 | 41,510 |
| with_presale 排除面積極端前 | 41,510 |
| with_presale 排除面積極端後 | 41,443 |
| with_presale 排除格局極端前 | 41,443 |
| with_presale 排除格局極端後 | 41,300 |
| strict 沿用舊版嚴格備註規則前 | 48,702 |
| strict 沿用舊版嚴格備註規則後 | 25,650 |
| strict base filters 後 | 25,424 |
| strict 排除面積極端後 | 18,056 |
| strict 排除格局極端後 | 18,008 |


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
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_with_presale.parquet | parquet | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_with_presale.csv | csv | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_strict.parquet | parquet | ok |  |
| /home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_strict.csv | csv | ok |  |
| /home/nas2/Personal/Hank/Esuan/reports/filter_log.csv | csv | ok |  |


### Parquet 輸出狀態

所有 parquet 輸出成功。


## 9. 特別提醒

- `source_release` 是資料發布批次，不代表交易日期。
- 未來模型時間切分應該使用 `trade_date`。
- `id` / `transfer_id` 不應作為模型特徵。
- `total_price`、`unit_price_m2`、`unit_price_ping`、`parking_price` 屬於 leakage 或 target 相關欄位，不應放入 feature。
