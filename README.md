# Taipei House Price Modeling

This project builds a reproducible and explainable residential unit-price model for Taipei real-estate transactions. It focuses on time-aware validation, district-level error tracking, and leakage-controlled comparable-sale features.

<p align="center">
  <img src="MAE_TAIPEI.png" alt="Taipei district MAE by district" width="780">
</p>

<p align="center">
  <b>V4 Tuned LightGBM district-level MAE</b><br>
  MAE is measured in NT$10,000 per ping. Higher-priced districts tend to have larger absolute errors.
</p>

## Highlights

- Data source: Taipei residential real-estate transaction records, with about 90,481 model-ready rows.
- Prediction target: `unit_price_ping`, the transaction unit price per ping.
- Validation: 15 rolling folds, with test quarters from 2022Q3 through 2026Q1.
- Main model: LightGBM regression pipeline with train-only preprocessing.
- Feature engineering: time-aware market signals, rule-based comparable sales, and Qwen embedding + reranker comparable features.
- Leakage control: historical market and comparable-sale features only use transactions where `trade_date < current trade_date`.

## Current Best Result

V4 combines embedding/reranker comparable-sale features with a LightGBM parameter search. The selected configuration is ranked by rolling test-period mean MAE.

| model | folds | mean MAE | mean RMSE | mean MAPE | mean R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4 tuned LightGBM | 15 | 9.9228 | 13.3242 | 13.9441% | 0.7218 |
| V3 tree model | 15 | 10.2311 | 13.6624 | 14.3453% | 0.7075 |
| District + building-type median baseline | 15 | 15.2770 | 20.9384 | 18.7750% | 0.3122 |

The best V4 model reduces mean MAE by about 35% compared with the district + building-type median baseline.

## What Drives The Model

V4 feature importance shows that comparable-sale signals dominate the model, followed by housing attributes and recent market signals.

| feature group | gain share |
| --- | ---: |
| Comparable Sales | 88.26% |
| Housing Attributes | 8.10% |
| Recent Market Signals | 2.71% |
| Location and Time | 0.77% |
| Note / Quality Flags | 0.17% |

Detailed charts and CSV exports are available in `reports/v4/lightgbm_search/feature_importance/`.

## Project Structure

```text
.
├── Script/                         # data build, feature engineering, training, tests
├── reports/                        # versioned experiment reports and exported tables
│   ├── v1/                         # base dataset, folds, baseline models
│   ├── v2/                         # time-aware market features
│   ├── v3/                         # rule-based comparable features
│   ├── v4/lightgbm_search/         # tuned LightGBM results and diagnostics
│   └── v5_embedding_only/          # embedding-only experiment outputs
├── analysis/phase3_explainability/ # SHAP, residual, correlation analysis
├── requirements.txt
└── MAE_TAIPEI.png
```

Large raw data, processed datasets, model artifacts, and embedding caches are intentionally excluded by `.gitignore`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For V4 embedding/reranker feature generation, download the Hugging Face models first:

```bash
huggingface-cli login
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir models/hf/Qwen3-Embedding-8B
huggingface-cli download Qwen/Qwen3-Reranker-8B --local-dir models/hf/Qwen3-Reranker-8B
```

The full V4 command set is documented in `reports/v4/phase4_cli_commands.md`.

## Typical Workflow

```bash
# 1. Build cleaned Taipei transaction dataset
python Script/build_taipei_dataset.py --help

# 2. Build rolling time splits
python Script/build_time_splits.py --help

# 3. Add time-aware market features
python Script/add_time_aware_features_v2.py --help

# 4. Add rule-based comparable-sale features
python Script/add_comparable_features_v3.py --help

# 5. Add embedding/reranker comparable-sale features
python Script/add_embedding_reranker_comparable_features_v4.py --help

# 6. Run LightGBM parameter search
python Script/search_lightgbm_params.py --help
```

Use `--help` on each script to inspect paths and runtime options before running the full pipeline.

## Key Reports

- `reports/v4/lightgbm_search/lightgbm_param_search_summary.md`
- `reports/v4/lightgbm_search/台北行政區_MAE/taipei_district_mae_v4_lightgbm_search.md`
- `reports/v4/lightgbm_search/feature_importance/feature_importance_summary.md`
- `reports/v4/phase4_cli_commands.md`
- `reports/README.md`

## Tests

```bash
pytest Script/test_*.py
```

Some tests and scripts expect local processed data or model artifacts that are not committed to Git.
