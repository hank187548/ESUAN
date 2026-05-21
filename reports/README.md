# Reports Index

Root keeps active feature configs for CLI compatibility:

- `feature_config_model_v1.json`
- `feature_config_model_v2.json`

## v1

`reports/v1/`

- Dataset build: `data_build_summary.md`, `filter_log.csv`
- Sanity check: `model_ready_sanity_check.md`, `missing_value_report.csv`, `zero_variance_columns.csv`, `categorical_levels_report.csv`, `leakage_check_report.csv`
- Rolling folds: `rolling_folds_summary.md`
- Phase 1 training: `phase1_model_metrics.csv`, `phase1_model_metrics_summary.csv`, `phase1_model_report.md`
- Phase 1 error analysis: `error_analysis_phase1.md`, `error_by_*.csv`, `error_top_*.csv`

## v2

`reports/v2/`

- Time-aware features: `time_aware_features_v2_summary.md`, `time_aware_features_v2_missing_report.csv`, `time_aware_features_v2_leakage_check.csv`
- Phase 2 training: `phase2_model_metrics.csv`, `phase2_model_metrics_summary.csv`, `phase2_model_report.md`
- Phase 2 error analysis: `error_analysis/phase2_error_analysis.md`, `error_analysis/error_by_*.csv`, `error_analysis/error_top_*.csv`
