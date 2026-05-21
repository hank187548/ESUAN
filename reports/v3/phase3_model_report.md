# Phase 1B Model Report

## 1. 實驗設定

- data path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v3.csv`
- rolling folds path: `/home/nas2/Personal/Hank/Esuan/data/processed/rolling_folds.csv`
- feature config path: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v3.json`
- target: `unit_price_ping`
- numeric features: 56
- categorical features: 6
- models: naive_global_median, naive_district_median, naive_district_building_type_median, ridge_regression, tree_model

## 2. Hardware / Runtime

- CPU cores detected: 64
- n_jobs: 48
- resolved per-model n_jobs: 16
- parallel_folds: true
- num_parallel_jobs: 4
- use_gpu setting: auto
- detected_gpu_count: 4
- gpu_ids: 0,1,2,3
- GPU names: NVIDIA RTX A6000, NVIDIA RTX A6000, NVIDIA RTX A6000, NVIDIA RTX A6000
- tree_model_backend: lightgbm
- tree_model_device: cpu
- GPU used successfully: no
- GPU/CPU fallback reasons: LightGBM GPU failed: LightGBMError: No OpenCL device found
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
| ridge_regression | test | 15 | 10.7839 | 0.4010 | 14.4727 | 0.5416 | 15.1625 | 0.7554 | 8.3527 | 0.6708 |
| tree_model | test | 15 | 10.2311 | 0.5836 | 13.6624 | 0.7680 | 14.3453 | 0.7712 | 7.9721 | 0.7075 |

- best model by mean test MAE: `tree_model`
- tree_model vs naive_district_building_type_median MAE improvement: 5.0459
- tree_model vs naive_district_building_type_median MAPE improvement: 4.4296

## 6. 每個 Fold 的 Test 結果

| fold_id | test_start | test_end | model_name | test_mae | test_rmse | test_mape | test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-07-01 | 2022-09-30 | naive_global_median | 19.0233 | 25.6445 | 25.5998 | -0.1939 |
| 1 | 2022-07-01 | 2022-09-30 | naive_district_median | 15.9131 | 21.7919 | 20.9131 | 0.1378 |
| 1 | 2022-07-01 | 2022-09-30 | naive_district_building_type_median | 14.9213 | 20.2980 | 19.6542 | 0.2520 |
| 1 | 2022-07-01 | 2022-09-30 | ridge_regression | 10.6353 | 14.1182 | 16.1331 | 0.6381 |
| 1 | 2022-07-01 | 2022-09-30 | tree_model | 9.8256 | 13.0195 | 14.5946 | 0.6923 |
| 2 | 2022-10-01 | 2022-12-31 | naive_global_median | 19.4595 | 25.9958 | 26.1832 | -0.1776 |
| 2 | 2022-10-01 | 2022-12-31 | naive_district_median | 15.1520 | 20.8233 | 20.0323 | 0.2444 |
| 2 | 2022-10-01 | 2022-12-31 | naive_district_building_type_median | 14.0440 | 19.6177 | 18.3781 | 0.3293 |
| 2 | 2022-10-01 | 2022-12-31 | ridge_regression | 10.3094 | 13.6866 | 15.7561 | 0.6736 |
| 2 | 2022-10-01 | 2022-12-31 | tree_model | 9.6679 | 12.8451 | 14.4917 | 0.7125 |
| 3 | 2023-01-01 | 2023-03-31 | naive_global_median | 18.0570 | 24.4408 | 24.8811 | -0.1410 |
| 3 | 2023-01-01 | 2023-03-31 | naive_district_median | 14.4326 | 19.8482 | 19.6738 | 0.2475 |
| 3 | 2023-01-01 | 2023-03-31 | naive_district_building_type_median | 13.0665 | 18.3061 | 17.6100 | 0.3599 |
| 3 | 2023-01-01 | 2023-03-31 | ridge_regression | 10.1384 | 13.8101 | 15.7815 | 0.6357 |
| 3 | 2023-01-01 | 2023-03-31 | tree_model | 9.2995 | 12.5756 | 14.0112 | 0.6979 |
| 4 | 2023-04-01 | 2023-06-30 | naive_global_median | 18.9692 | 25.9135 | 24.7399 | -0.1857 |
| 4 | 2023-04-01 | 2023-06-30 | naive_district_median | 15.2811 | 21.0995 | 19.7264 | 0.2139 |
| 4 | 2023-04-01 | 2023-06-30 | naive_district_building_type_median | 14.1862 | 19.7355 | 18.2391 | 0.3123 |
| 4 | 2023-04-01 | 2023-06-30 | ridge_regression | 10.7066 | 14.4085 | 15.6686 | 0.6334 |
| 4 | 2023-04-01 | 2023-06-30 | tree_model | 9.8343 | 13.1668 | 14.2636 | 0.6939 |
| 5 | 2023-07-01 | 2023-09-30 | naive_global_median | 19.8893 | 27.4323 | 25.2722 | -0.1982 |
| 5 | 2023-07-01 | 2023-09-30 | naive_district_median | 16.0101 | 22.2374 | 20.0625 | 0.2126 |
| 5 | 2023-07-01 | 2023-09-30 | naive_district_building_type_median | 14.6245 | 20.5933 | 18.1961 | 0.3248 |
| 5 | 2023-07-01 | 2023-09-30 | ridge_regression | 10.3405 | 14.0630 | 14.8958 | 0.6851 |
| 5 | 2023-07-01 | 2023-09-30 | tree_model | 9.6397 | 13.0859 | 13.7235 | 0.7273 |
| 6 | 2023-10-01 | 2023-12-31 | naive_global_median | 19.5236 | 26.5184 | 24.9371 | -0.1972 |
| 6 | 2023-10-01 | 2023-12-31 | naive_district_median | 15.4561 | 21.3285 | 19.4837 | 0.2255 |
| 6 | 2023-10-01 | 2023-12-31 | naive_district_building_type_median | 14.2873 | 19.9226 | 17.8420 | 0.3243 |
| 6 | 2023-10-01 | 2023-12-31 | ridge_regression | 10.2525 | 13.8206 | 14.5447 | 0.6748 |
| 6 | 2023-10-01 | 2023-12-31 | tree_model | 9.6701 | 12.9109 | 13.6605 | 0.7162 |
| 7 | 2024-01-01 | 2024-03-31 | naive_global_median | 20.8311 | 28.6454 | 25.7659 | -0.2186 |
| 7 | 2024-01-01 | 2024-03-31 | naive_district_median | 16.8812 | 23.2105 | 20.7200 | 0.2000 |
| 7 | 2024-01-01 | 2024-03-31 | naive_district_building_type_median | 15.6646 | 21.7242 | 19.0903 | 0.2991 |
| 7 | 2024-01-01 | 2024-03-31 | ridge_regression | 10.7272 | 14.6905 | 14.9512 | 0.6795 |
| 7 | 2024-01-01 | 2024-03-31 | tree_model | 10.2858 | 13.9368 | 14.2847 | 0.7116 |
| 8 | 2024-04-01 | 2024-06-30 | naive_global_median | 21.6165 | 28.9849 | 25.5539 | -0.3029 |
| 8 | 2024-04-01 | 2024-06-30 | naive_district_median | 17.9350 | 24.0443 | 21.0147 | 0.1034 |
| 8 | 2024-04-01 | 2024-06-30 | naive_district_building_type_median | 16.3711 | 22.1425 | 19.0797 | 0.2396 |
| 8 | 2024-04-01 | 2024-06-30 | ridge_regression | 10.8337 | 14.5804 | 14.1568 | 0.6703 |
| 8 | 2024-04-01 | 2024-06-30 | tree_model | 10.2857 | 13.6094 | 13.5438 | 0.7128 |
| 9 | 2024-07-01 | 2024-09-30 | naive_global_median | 21.9968 | 29.4856 | 25.2849 | -0.3148 |
| 9 | 2024-07-01 | 2024-09-30 | naive_district_median | 18.8142 | 25.4914 | 21.4234 | 0.0173 |
| 9 | 2024-07-01 | 2024-09-30 | naive_district_building_type_median | 17.5200 | 23.5955 | 19.9435 | 0.1580 |
| 9 | 2024-07-01 | 2024-09-30 | ridge_regression | 11.1422 | 15.0686 | 14.3628 | 0.6566 |
| 9 | 2024-07-01 | 2024-09-30 | tree_model | 10.6145 | 14.2122 | 13.8683 | 0.6945 |
| 10 | 2024-10-01 | 2024-12-31 | naive_global_median | 20.3788 | 27.3145 | 25.2619 | -0.1894 |
| 10 | 2024-10-01 | 2024-12-31 | naive_district_median | 17.3569 | 23.4063 | 21.0806 | 0.1266 |
| 10 | 2024-10-01 | 2024-12-31 | naive_district_building_type_median | 15.8714 | 21.3301 | 19.2183 | 0.2747 |
| 10 | 2024-10-01 | 2024-12-31 | ridge_regression | 11.0915 | 14.9516 | 15.2129 | 0.6436 |
| 10 | 2024-10-01 | 2024-12-31 | tree_model | 10.8192 | 14.4315 | 15.3000 | 0.6680 |
| 11 | 2025-01-01 | 2025-03-31 | naive_global_median | 21.3710 | 27.9595 | 25.7042 | -0.2081 |
| 11 | 2025-01-01 | 2025-03-31 | naive_district_median | 17.4876 | 23.2037 | 20.7473 | 0.1679 |
| 11 | 2025-01-01 | 2025-03-31 | naive_district_building_type_median | 15.5538 | 20.8465 | 18.3095 | 0.3284 |
| 11 | 2025-01-01 | 2025-03-31 | ridge_regression | 10.9603 | 14.4924 | 14.6353 | 0.6754 |
| 11 | 2025-01-01 | 2025-03-31 | tree_model | 10.7977 | 14.1484 | 14.6647 | 0.6906 |
| 12 | 2025-04-01 | 2025-06-30 | naive_global_median | 20.9496 | 27.9505 | 25.7365 | -0.1485 |
| 12 | 2025-04-01 | 2025-06-30 | naive_district_median | 17.7622 | 23.7700 | 21.3584 | 0.1694 |
| 12 | 2025-04-01 | 2025-06-30 | naive_district_building_type_median | 15.8193 | 21.3741 | 18.8907 | 0.3284 |
| 12 | 2025-04-01 | 2025-06-30 | ridge_regression | 11.0892 | 14.7224 | 15.0624 | 0.6813 |
| 12 | 2025-04-01 | 2025-06-30 | tree_model | 10.4383 | 13.8450 | 14.3797 | 0.7182 |
| 13 | 2025-07-01 | 2025-09-30 | naive_global_median | 21.5202 | 28.1910 | 26.1976 | -0.1504 |
| 13 | 2025-07-01 | 2025-09-30 | naive_district_median | 17.5137 | 23.3395 | 20.8684 | 0.2115 |
| 13 | 2025-07-01 | 2025-09-30 | naive_district_building_type_median | 15.3472 | 20.6212 | 18.2281 | 0.3844 |
| 13 | 2025-07-01 | 2025-09-30 | ridge_regression | 10.7516 | 14.0421 | 14.4694 | 0.7146 |
| 13 | 2025-07-01 | 2025-09-30 | tree_model | 10.1322 | 13.3652 | 13.6725 | 0.7414 |
| 14 | 2025-10-01 | 2025-12-31 | naive_global_median | 21.3003 | 28.5220 | 25.7179 | -0.1329 |
| 14 | 2025-10-01 | 2025-12-31 | naive_district_median | 17.7508 | 24.0014 | 20.9429 | 0.1978 |
| 14 | 2025-10-01 | 2025-12-31 | naive_district_building_type_median | 15.6198 | 21.5051 | 18.3412 | 0.3560 |
| 14 | 2025-10-01 | 2025-12-31 | ridge_regression | 11.3119 | 15.2606 | 14.8940 | 0.6757 |
| 14 | 2025-10-01 | 2025-12-31 | tree_model | 10.6969 | 14.3921 | 14.1634 | 0.7116 |
| 15 | 2026-01-01 | 2026-03-31 | naive_global_median | 23.0712 | 30.4531 | 30.4904 | -0.0799 |
| 15 | 2026-01-01 | 2026-03-31 | naive_district_median | 18.5267 | 25.1304 | 23.5577 | 0.2646 |
| 15 | 2026-01-01 | 2026-03-31 | naive_district_building_type_median | 16.2584 | 22.4635 | 20.6037 | 0.4124 |
| 15 | 2026-01-01 | 2026-03-31 | ridge_regression | 11.4690 | 15.3751 | 16.9129 | 0.7247 |
| 15 | 2026-01-01 | 2026-03-31 | tree_model | 11.4587 | 15.3917 | 16.5578 | 0.7241 |

## 7. 初步結論

- naive baselines provide district and building-type market reference points.
- Ridge regression provides a linear baseline with train-only preprocessing.
- tree_model is the first nonlinear main model for this time-based validation setup.
- Next phase can focus on error analysis and SHAP.

## 8. Outputs

- metrics: `/home/nas2/Personal/Hank/Esuan/reports/v3/_phase3_tmp/phase1_model_metrics.csv`
- metrics summary: `/home/nas2/Personal/Hank/Esuan/reports/v3/_phase3_tmp/phase1_model_metrics_summary.csv`
- OOF predictions: `/home/nas2/Personal/Hank/Esuan/data/processed/phase3_oof_predictions.csv`
- model dir: `/home/nas2/Personal/Hank/Esuan/models/phase3`
