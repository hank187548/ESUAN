# Taipei District MAE - v4 LightGBM Search

Source: `data/processed/v4_lightgbm_search_oof_predictions.csv` filtered to `split == test`.

Model: `lightgbm_best` / v4 reranker comparable features + tuned LightGBM.

## District Metrics

| district | n | mae | rmse | mape | r2 | bias | y_true_mean | y_true_median | y_pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 中正區 | 1948 | 14.54 | 18.67 | 16.2 | 0.6 | -1.41 | 96.48 | 90.1 | 95.07 |
| 大安區 | 4125 | 12.39 | 16.13 | 12.51 | 0.57 | -1.26 | 104.52 | 100.14 | 103.26 |
| 信義區 | 2863 | 11.51 | 15.85 | 14.16 | 0.54 | 0.08 | 85.89 | 82.06 | 85.98 |
| 士林區 | 3112 | 10.53 | 14.05 | 16.91 | 0.59 | 0.69 | 69.34 | 65.2 | 70.03 |
| 中山區 | 4765 | 9.94 | 13.17 | 12.54 | 0.6 | -0.63 | 81.95 | 78.39 | 81.32 |
| 北投區 | 3566 | 9.78 | 12.6 | 18.53 | 0.57 | 0.08 | 58.86 | 56.95 | 58.94 |
| 松山區 | 2695 | 9.73 | 13.05 | 11.34 | 0.57 | -0.43 | 88.88 | 85.71 | 88.45 |
| 南港區 | 1566 | 9.5 | 12.4 | 14.05 | 0.63 | 1.36 | 76.52 | 75.57 | 77.88 |
| 大同區 | 1552 | 9.26 | 12.84 | 12.83 | 0.65 | -0.24 | 76.8 | 73.61 | 76.55 |
| 內湖區 | 4107 | 8.06 | 10.54 | 12.16 | 0.59 | 0.41 | 70.19 | 68.12 | 70.6 |
| 文山區 | 3648 | 7.17 | 9.28 | 12.71 | 0.63 | 0.38 | 59.98 | 57.88 | 60.36 |
| 萬華區 | 2482 | 6.77 | 9.21 | 11.92 | 0.62 | 0.64 | 61.09 | 60.3 | 61.73 |

## Interpretation Notes

- `MAE` is in 萬元/坪, so high-price districts often have larger absolute error even when percentage error is acceptable.
- Use `MAPE` together with `MAE` when comparing districts with very different price levels.
- `bias < 0` means under-prediction; `bias > 0` means over-prediction.

## Outputs

- `reports/v4/lightgbm_search/台北行政區_MAE/taipei_district_mae_v4_lightgbm_search.csv`
- `reports/v4/lightgbm_search/台北行政區_MAE/taipei_district_mae_v4_lightgbm_search_ppt.csv`
- `reports/v4/lightgbm_search/台北行政區_MAE/taipei_district_mae_v4_lightgbm_search_heatmap.csv`