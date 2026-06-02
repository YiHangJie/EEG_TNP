#!/usr/bin/env bash
set -euo pipefail

# 先运行 purify/augmentation pipeline，成功结束后再做 tensor ring rank 分析。
# 用法：
#   nohup bash run_purify_then_analyze_consistancy.sh > logs/run_purify_then_analyze_consistancy.log 2>&1 &

EPS_VALUE="${EPS:-0.1}"
CONSISTENCY_VERSION="consistancy"
OUTPUT_DIR="tensor_ring_rank_analysis/results_consistancy"

timestamp="$(date +%Y%m%d_%H%M%S)"
pipeline_log="logs/purify_aug_consistancy_${timestamp}.log"
analysis_log="logs/tr_rank_analysis_consistancy_${timestamp}.log"

mkdir -p logs "${OUTPUT_DIR}"

echo "[$(date '+%F %T')] Start pipeline: EPS=${EPS_VALUE}"
echo "Pipeline log: ${pipeline_log}"
EPS="${EPS_VALUE}" bash purify_aug_consistancy_pipeline.sh > "${pipeline_log}" 2>&1

echo "[$(date '+%F %T')] Pipeline finished, start tensor ring rank analysis"
echo "Analysis log: ${analysis_log}"
python tensor_ring_rank_analysis/analyze_tr_rank_predictions.py \
  --eps "${EPS_VALUE}" \
  --consistency_version "${CONSISTENCY_VERSION}" \
  --output_dir "${OUTPUT_DIR}" \
  > "${analysis_log}" 2>&1

echo "[$(date '+%F %T')] All done"
