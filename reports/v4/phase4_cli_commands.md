# Phase 4 CLI Commands

本文件只提供手動執行指令。不要在準備階段自動下載 8B 模型或執行完整 v4 feature generation。

## 1. Hugging Face Login

PowerShell:

```powershell
huggingface-cli login
```

Bash:

```bash
huggingface-cli login
```

## 2. 手動下載模型到 `models/hf`

PowerShell:

```powershell
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir models\hf\Qwen3-Embedding-8B
huggingface-cli download Qwen/Qwen3-Reranker-8B --local-dir models\hf\Qwen3-Reranker-8B
```

Bash:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir models/hf/Qwen3-Embedding-8B
huggingface-cli download Qwen/Qwen3-Reranker-8B --local-dir models/hf/Qwen3-Reranker-8B
```

## 3. 執行 v4 Feature Generation

PowerShell:

```powershell
python Script/add_embedding_reranker_comparable_features_v4.py ^
  --v2-input-path "data\processed\taipei_house_model_ready_v2.csv" ^
  --v3-input-path "data\processed\taipei_house_model_ready_v3.csv" ^
  --feature-config-v2 "reports\feature_config_model_v2.json" ^
  --feature-config-v3 "reports\feature_config_model_v3.json" ^
  --v4-add-output-path "data\processed\taipei_house_model_ready_v4_add.csv" ^
  --v4-add-parquet-path "data\processed\taipei_house_model_ready_v4_add.parquet" ^
  --v4-replace-output-path "data\processed\taipei_house_model_ready_v4_replace.csv" ^
  --v4-replace-parquet-path "data\processed\taipei_house_model_ready_v4_replace.parquet" ^
  --feature-config-v4-add "reports\feature_config_model_v4_add.json" ^
  --feature-config-v4-replace "reports\feature_config_model_v4_replace.json" ^
  --report-dir "reports\v4" ^
  --embedding-model-name "Qwen/Qwen3-Embedding-8B" ^
  --reranker-model-name "Qwen/Qwen3-Reranker-8B" ^
  --embedding-model-path "models\hf\Qwen3-Embedding-8B" ^
  --reranker-model-path "models\hf\Qwen3-Reranker-8B" ^
  --download-if-missing false ^
  --device auto ^
  --gpu-ids "0,1,2,3" ^
  --dtype auto ^
  --max-length 512 ^
  --embedding-batch-size 16 ^
  --reranker-batch-size 4 ^
  --embedding-top-k 50 ^
  --reranker-top-k 10 ^
  --use-reranker true ^
  --allow-reranker-fallback false ^
  --include-address false ^
  --include-note-raw false ^
  --force-recompute-embeddings false
```

Bash:

```bash
python Script/add_embedding_reranker_comparable_features_v4.py \
  --v2-input-path "data/processed/taipei_house_model_ready_v2.csv" \
  --v3-input-path "data/processed/taipei_house_model_ready_v3.csv" \
  --feature-config-v2 "reports/feature_config_model_v2.json" \
  --feature-config-v3 "reports/feature_config_model_v3.json" \
  --v4-add-output-path "data/processed/taipei_house_model_ready_v4_add.csv" \
  --v4-add-parquet-path "data/processed/taipei_house_model_ready_v4_add.parquet" \
  --v4-replace-output-path "data/processed/taipei_house_model_ready_v4_replace.csv" \
  --v4-replace-parquet-path "data/processed/taipei_house_model_ready_v4_replace.parquet" \
  --feature-config-v4-add "reports/feature_config_model_v4_add.json" \
  --feature-config-v4-replace "reports/feature_config_model_v4_replace.json" \
  --report-dir "reports/v4" \
  --embedding-model-name "Qwen/Qwen3-Embedding-8B" \
  --reranker-model-name "Qwen/Qwen3-Reranker-8B" \
  --embedding-model-path "models/hf/Qwen3-Embedding-8B" \
  --reranker-model-path "models/hf/Qwen3-Reranker-8B" \
  --download-if-missing false \
  --device auto \
  --gpu-ids "0,1,2,3" \
  --dtype auto \
  --max-length 512 \
  --embedding-batch-size 16 \
  --reranker-batch-size 4 \
  --embedding-top-k 50 \
  --reranker-top-k 10 \
  --use-reranker true \
  --allow-reranker-fallback false \
  --include-address false \
  --include-note-raw false \
  --force-recompute-embeddings false
```

## 4. 檢查輸出檔案

PowerShell:

```powershell
Test-Path data\processed\taipei_house_model_ready_v4_add.csv
Test-Path data\processed\taipei_house_model_ready_v4_replace.csv
Test-Path reports\v4\embedding_comparable_features_v4_summary.md
Test-Path reports\v4\embedding_comparable_features_v4_leakage_check.csv
```

Bash:

```bash
test -f data/processed/taipei_house_model_ready_v4_add.csv
test -f data/processed/taipei_house_model_ready_v4_replace.csv
test -f reports/v4/embedding_comparable_features_v4_summary.md
test -f reports/v4/embedding_comparable_features_v4_leakage_check.csv
```

## 5. Pytest

```bash
pytest Script/test_embedding_reranker_comparable_features_v4.py
```

## 6. GPU Memory 不夠時

將 v4 generation CLI 中參數調小：

```bash
--embedding-batch-size 8
--reranker-batch-size 2
--dtype float16
```
