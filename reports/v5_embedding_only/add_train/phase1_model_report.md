# Phase 1B Model Report

## 1. 實驗設定

- data path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v5_embedding_only_add.csv`
- rolling folds path: `/home/nas2/Personal/Hank/Esuan/data/processed/rolling_folds.csv`
- feature config path: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v5_embedding_only_add.json`
- target: `unit_price_ping`
- numeric features: 84
- categorical features: 6
- models: naive_global_median, naive_district_median, naive_district_building_type_median, ridge_regression, tree_model

## 2. Hardware / Runtime

- CPU cores detected: 64
- n_jobs: 24
- resolved per-model n_jobs: 24
- parallel_folds: false
- num_parallel_jobs: 1
- use_gpu setting: false
- detected_gpu_count: 4
- gpu_ids: 0,1,2,3
- GPU names: NVIDIA RTX A6000, NVIDIA RTX A6000, NVIDIA RTX A6000, NVIDIA RTX A6000
- tree_model_backend: lightgbm
- tree_model_device: cpu
- GPU used successfully: no
- GPU/CPU fallback reasons: (none)
- early stopping: not used in Phase 1B; valid split is evaluated only.

## 3. 資料切分

- fold count: 15
- earliest test quarter: 2022Q3
- latest test quarter: 2026Q1
- last test quarter: 2026Q1
- validation: time-based rolling folds
- random split: not used

## 4. Leakage Control

- preprocessing is fit only on each fold's train split.
- median baselines are calculated only from each fold's train split.
- valid/test data are never used to fit imputers, scalers, encoders, medians, or models.
- `total_price`, `unit_price_m2`, `unit_price_ping`, and `parking_price` are not used as features.
- `source_release` is a release batch marker and is not used as a feature.

## 5. 模型結果

### Test Summary

| model_name | split | folds | mean_mae | std_mae | mean_rmse | std_rmse | mean_mape | std_mape | mean_medae | mean_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_district_building_type_median | test | 15 | 15.2770 | 1.1121 | 20.9384 | 1.3068 | 18.7750 | 0.8279 | 11.4075 | 0.3122 |
| naive_district_median | test | 15 | 16.8182 | 1.3442 | 22.8484 | 1.6163 | 20.7737 | 0.9912 | 12.5558 | 0.1827 |
| naive_global_median | test | 15 | 20.5305 | 1.3562 | 27.5635 | 1.6178 | 25.8218 | 1.3619 | 15.9815 | -0.1893 |
| ridge_regression | test | 15 | 10.5738 | 0.4138 | 14.1481 | 0.5507 | 14.8965 | 0.7557 | 8.1851 | 0.6857 |
| tree_model | test | 15 | 10.0886 | 0.5547 | 13.4910 | 0.6946 | 14.2044 | 0.7471 | 7.7941 | 0.7147 |

- best model by mean test MAE: `tree_model`
- tree_model vs naive_district_building_type_median MAE improvement: 5.1884
- tree_model vs naive_district_building_type_median MAPE improvement: 4.5705

## 6. 每個 Fold 的 Test 結果

| fold_id | test_start | test_end | model_name | test_mae | test_rmse | test_mape | test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-07-01 | 2022-09-30 | naive_global_median | 19.0233 | 25.6445 | 25.5998 | -0.1939 |
| 1 | 2022-07-01 | 2022-09-30 | naive_district_median | 15.9131 | 21.7919 | 20.9131 | 0.1378 |
| 1 | 2022-07-01 | 2022-09-30 | naive_district_building_type_median | 14.9213 | 20.2980 | 19.6542 | 0.2520 |
| 1 | 2022-07-01 | 2022-09-30 | ridge_regression | 10.2653 | 13.5326 | 15.5611 | 0.6675 |
| 1 | 2022-07-01 | 2022-09-30 | tree_model | 9.5626 | 12.6744 | 14.3059 | 0.7084 |
| 2 | 2022-10-01 | 2022-12-31 | naive_global_median | 19.4595 | 25.9958 | 26.1832 | -0.1776 |
| 2 | 2022-10-01 | 2022-12-31 | naive_district_median | 15.1520 | 20.8233 | 20.0323 | 0.2444 |
| 2 | 2022-10-01 | 2022-12-31 | naive_district_building_type_median | 14.0440 | 19.6177 | 18.3781 | 0.3293 |
| 2 | 2022-10-01 | 2022-12-31 | ridge_regression | 10.0328 | 13.2597 | 15.2679 | 0.6936 |
| 2 | 2022-10-01 | 2022-12-31 | tree_model | 9.5702 | 12.7106 | 14.3544 | 0.7185 |
| 3 | 2023-01-01 | 2023-03-31 | naive_global_median | 18.0570 | 24.4408 | 24.8811 | -0.1410 |
| 3 | 2023-01-01 | 2023-03-31 | naive_district_median | 14.4326 | 19.8482 | 19.6738 | 0.2475 |
| 3 | 2023-01-01 | 2023-03-31 | naive_district_building_type_median | 13.0665 | 18.3061 | 17.6100 | 0.3599 |
| 3 | 2023-01-01 | 2023-03-31 | ridge_regression | 9.9722 | 13.5546 | 15.5333 | 0.6491 |
| 3 | 2023-01-01 | 2023-03-31 | tree_model | 9.1494 | 12.5197 | 13.8610 | 0.7006 |
| 4 | 2023-04-01 | 2023-06-30 | naive_global_median | 18.9692 | 25.9135 | 24.7399 | -0.1857 |
| 4 | 2023-04-01 | 2023-06-30 | naive_district_median | 15.2811 | 21.0995 | 19.7264 | 0.2139 |
| 4 | 2023-04-01 | 2023-06-30 | naive_district_building_type_median | 14.1862 | 19.7355 | 18.2391 | 0.3123 |
| 4 | 2023-04-01 | 2023-06-30 | ridge_regression | 10.4632 | 14.0191 | 15.3797 | 0.6530 |
| 4 | 2023-04-01 | 2023-06-30 | tree_model | 9.7452 | 13.0932 | 14.1778 | 0.6973 |
| 5 | 2023-07-01 | 2023-09-30 | naive_global_median | 19.8893 | 27.4323 | 25.2722 | -0.1982 |
| 5 | 2023-07-01 | 2023-09-30 | naive_district_median | 16.0101 | 22.2374 | 20.0625 | 0.2126 |
| 5 | 2023-07-01 | 2023-09-30 | naive_district_building_type_median | 14.6245 | 20.5933 | 18.1961 | 0.3248 |
| 5 | 2023-07-01 | 2023-09-30 | ridge_regression | 10.1752 | 13.7896 | 14.6961 | 0.6972 |
| 5 | 2023-07-01 | 2023-09-30 | tree_model | 9.5211 | 12.8946 | 13.6492 | 0.7353 |
| 6 | 2023-10-01 | 2023-12-31 | naive_global_median | 19.5236 | 26.5184 | 24.9371 | -0.1972 |
| 6 | 2023-10-01 | 2023-12-31 | naive_district_median | 15.4561 | 21.3285 | 19.4837 | 0.2255 |
| 6 | 2023-10-01 | 2023-12-31 | naive_district_building_type_median | 14.2873 | 19.9226 | 17.8420 | 0.3243 |
| 6 | 2023-10-01 | 2023-12-31 | ridge_regression | 10.0452 | 13.4926 | 14.3409 | 0.6901 |
| 6 | 2023-10-01 | 2023-12-31 | tree_model | 9.5951 | 12.8441 | 13.5795 | 0.7191 |
| 7 | 2024-01-01 | 2024-03-31 | naive_global_median | 20.8311 | 28.6454 | 25.7659 | -0.2186 |
| 7 | 2024-01-01 | 2024-03-31 | naive_district_median | 16.8812 | 23.2105 | 20.7200 | 0.2000 |
| 7 | 2024-01-01 | 2024-03-31 | naive_district_building_type_median | 15.6646 | 21.7242 | 19.0903 | 0.2991 |
| 7 | 2024-01-01 | 2024-03-31 | ridge_regression | 10.5295 | 14.3819 | 14.6910 | 0.6928 |
| 7 | 2024-01-01 | 2024-03-31 | tree_model | 10.0717 | 13.7218 | 14.0386 | 0.7204 |
| 8 | 2024-04-01 | 2024-06-30 | naive_global_median | 21.6165 | 28.9849 | 25.5539 | -0.3029 |
| 8 | 2024-04-01 | 2024-06-30 | naive_district_median | 17.9350 | 24.0443 | 21.0147 | 0.1034 |
| 8 | 2024-04-01 | 2024-06-30 | naive_district_building_type_median | 16.3711 | 22.1425 | 19.0797 | 0.2396 |
| 8 | 2024-04-01 | 2024-06-30 | ridge_regression | 10.6760 | 14.3813 | 13.9005 | 0.6793 |
| 8 | 2024-04-01 | 2024-06-30 | tree_model | 10.1767 | 13.5987 | 13.3966 | 0.7132 |
| 9 | 2024-07-01 | 2024-09-30 | naive_global_median | 21.9968 | 29.4856 | 25.2849 | -0.3148 |
| 9 | 2024-07-01 | 2024-09-30 | naive_district_median | 18.8142 | 25.4914 | 21.4234 | 0.0173 |
| 9 | 2024-07-01 | 2024-09-30 | naive_district_building_type_median | 17.5200 | 23.5955 | 19.9435 | 0.1580 |
| 9 | 2024-07-01 | 2024-09-30 | ridge_regression | 10.9031 | 14.6399 | 14.0097 | 0.6759 |
| 9 | 2024-07-01 | 2024-09-30 | tree_model | 10.5261 | 14.0681 | 13.7565 | 0.7007 |
| 10 | 2024-10-01 | 2024-12-31 | naive_global_median | 20.3788 | 27.3145 | 25.2619 | -0.1894 |
| 10 | 2024-10-01 | 2024-12-31 | naive_district_median | 17.3569 | 23.4063 | 21.0806 | 0.1266 |
| 10 | 2024-10-01 | 2024-12-31 | naive_district_building_type_median | 15.8714 | 21.3301 | 19.2183 | 0.2747 |
| 10 | 2024-10-01 | 2024-12-31 | ridge_regression | 10.8263 | 14.5287 | 14.8201 | 0.6635 |
| 10 | 2024-10-01 | 2024-12-31 | tree_model | 10.6238 | 14.1932 | 14.9686 | 0.6789 |
| 11 | 2025-01-01 | 2025-03-31 | naive_global_median | 21.3710 | 27.9595 | 25.7042 | -0.2081 |
| 11 | 2025-01-01 | 2025-03-31 | naive_district_median | 17.4876 | 23.2037 | 20.7473 | 0.1679 |
| 11 | 2025-01-01 | 2025-03-31 | naive_district_building_type_median | 15.5538 | 20.8465 | 18.3095 | 0.3284 |
| 11 | 2025-01-01 | 2025-03-31 | ridge_regression | 10.7595 | 14.2545 | 14.3741 | 0.6860 |
| 11 | 2025-01-01 | 2025-03-31 | tree_model | 10.5655 | 13.9608 | 14.4115 | 0.6988 |
| 12 | 2025-04-01 | 2025-06-30 | naive_global_median | 20.9496 | 27.9505 | 25.7365 | -0.1485 |
| 12 | 2025-04-01 | 2025-06-30 | naive_district_median | 17.7622 | 23.7700 | 21.3584 | 0.1694 |
| 12 | 2025-04-01 | 2025-06-30 | naive_district_building_type_median | 15.8193 | 21.3741 | 18.8907 | 0.3284 |
| 12 | 2025-04-01 | 2025-06-30 | ridge_regression | 10.8779 | 14.4361 | 14.7430 | 0.6936 |
| 12 | 2025-04-01 | 2025-06-30 | tree_model | 10.3848 | 13.7551 | 14.2774 | 0.7218 |
| 13 | 2025-07-01 | 2025-09-30 | naive_global_median | 21.5202 | 28.1910 | 26.1976 | -0.1504 |
| 13 | 2025-07-01 | 2025-09-30 | naive_district_median | 17.5137 | 23.3395 | 20.8684 | 0.2115 |
| 13 | 2025-07-01 | 2025-09-30 | naive_district_building_type_median | 15.3472 | 20.6212 | 18.2281 | 0.3844 |
| 13 | 2025-07-01 | 2025-09-30 | ridge_regression | 10.6137 | 13.9277 | 14.4035 | 0.7192 |
| 13 | 2025-07-01 | 2025-09-30 | tree_model | 10.0576 | 13.2631 | 13.6652 | 0.7454 |
| 14 | 2025-10-01 | 2025-12-31 | naive_global_median | 21.3003 | 28.5220 | 25.7179 | -0.1329 |
| 14 | 2025-10-01 | 2025-12-31 | naive_district_median | 17.7508 | 24.0014 | 20.9429 | 0.1978 |
| 14 | 2025-10-01 | 2025-12-31 | naive_district_building_type_median | 15.6198 | 21.5051 | 18.3412 | 0.3560 |
| 14 | 2025-10-01 | 2025-12-31 | ridge_regression | 11.1540 | 14.9085 | 14.8163 | 0.6905 |
| 14 | 2025-10-01 | 2025-12-31 | tree_model | 10.6523 | 14.2238 | 14.1491 | 0.7183 |
| 15 | 2026-01-01 | 2026-03-31 | naive_global_median | 23.0712 | 30.4531 | 30.4904 | -0.0799 |
| 15 | 2026-01-01 | 2026-03-31 | naive_district_median | 18.5267 | 25.1304 | 23.5577 | 0.2646 |
| 15 | 2026-01-01 | 2026-03-31 | naive_district_building_type_median | 16.2584 | 22.4635 | 20.6037 | 0.4124 |
| 15 | 2026-01-01 | 2026-03-31 | ridge_regression | 11.3134 | 15.1143 | 16.9101 | 0.7340 |
| 15 | 2026-01-01 | 2026-03-31 | tree_model | 11.1267 | 14.8444 | 16.4753 | 0.7434 |

## 7. 初步結論

- naive baselines provide district and building-type market reference points.
- Ridge regression provides a linear baseline with train-only preprocessing.
- tree_model is the first nonlinear main model for this time-based validation setup.
- Next phase can focus on error analysis and SHAP.

## 8. Outputs

- metrics: `/home/nas2/Personal/Hank/Esuan/reports/v5_embedding_only/add_train/phase1_model_metrics.csv`
- metrics summary: `/home/nas2/Personal/Hank/Esuan/reports/v5_embedding_only/add_train/phase1_model_metrics_summary.csv`
- OOF predictions: `/home/nas2/Personal/Hank/Esuan/data/processed/v5_embedding_only_add_oof_predictions.csv`
- model dir: `/home/nas2/Personal/Hank/Esuan/models/v5_embedding_only_add`
