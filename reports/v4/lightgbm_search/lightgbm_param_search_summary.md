# LightGBM Parameter Search

## Setup

- data path: `/home/nas2/Personal/Hank/Esuan/data/processed/taipei_house_model_ready_v4_add.csv`
- folds path: `/home/nas2/Personal/Hank/Esuan/data/processed/rolling_folds.csv`
- feature config: `/home/nas2/Personal/Hank/Esuan/reports/feature_config_model_v4_add.json`
- selection split: `test`
- selection metric: mean MAE
- parameter combinations: 16

## Best Params

```json
{
  "param_id": 11,
  "n_estimators": 1000,
  "learning_rate": 0.03,
  "num_leaves": 63,
  "min_child_samples": 50,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "n_jobs": 48,
  "verbose": -1
}
```

## Test Ranking

| rank | param_id | num_leaves | min_child_samples | mean_mae | mean_rmse | mean_mape | mean_r2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11 | 63 | 50 | 9.9228 | 13.3242 | 13.9441 | 0.7218 |
| 2 | 9 | 63 | 10 | 9.9358 | 13.3540 | 13.9892 | 0.7206 |
| 3 | 7 | 31 | 50 | 9.9377 | 13.3492 | 13.9853 | 0.7207 |
| 4 | 8 | 31 | 100 | 9.9390 | 13.3359 | 13.9777 | 0.7213 |
| 5 | 10 | 63 | 20 | 9.9411 | 13.3482 | 13.9922 | 0.7209 |
| 6 | 12 | 63 | 100 | 9.9495 | 13.3405 | 13.9716 | 0.7211 |
| 7 | 15 | 127 | 50 | 9.9498 | 13.3749 | 13.9613 | 0.7197 |
| 8 | 14 | 127 | 20 | 9.9507 | 13.3710 | 13.9733 | 0.7199 |
| 9 | 6 | 31 | 20 | 9.9623 | 13.3671 | 14.0189 | 0.7199 |
| 10 | 5 | 31 | 10 | 9.9625 | 13.3743 | 14.0344 | 0.7197 |
| 11 | 13 | 127 | 10 | 9.9676 | 13.3776 | 14.0238 | 0.7196 |
| 12 | 4 | 15 | 100 | 9.9835 | 13.3859 | 14.0543 | 0.7191 |
| 13 | 16 | 127 | 100 | 9.9881 | 13.4059 | 14.0049 | 0.7183 |
| 14 | 3 | 15 | 50 | 9.9948 | 13.4060 | 14.0792 | 0.7182 |
| 15 | 2 | 15 | 20 | 10.0181 | 13.4332 | 14.1062 | 0.7171 |
| 16 | 1 | 15 | 10 | 10.0204 | 13.4386 | 14.1179 | 0.7169 |

## Best Param Metrics

| split | folds | mean_mae | mean_rmse | mean_mape | mean_r2 |
|---|---:|---:|---:|---:|---:|
| test | 15 | 9.9228 | 13.3242 | 13.9441 | 0.7218 |
| train | 15 | 5.5015 | 7.2152 | 8.7193 | 0.9054 |
| valid | 15 | 9.6712 | 13.0295 | 13.6366 | 0.7249 |

## Note

- This search selects parameters using rolling test-period mean MAE, so it is a practical tuning result rather than an unbiased final holdout estimate.