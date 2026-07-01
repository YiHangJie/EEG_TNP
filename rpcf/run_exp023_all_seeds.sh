#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-42 43 44}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-torch}"
RUN_TAG="${EXP023_RUN_TAG:-$(date +%Y%m%d_%H%M)}"
CHAIN_ID="${EXP023_CHAIN_ID:-exp023_bpda_pgd10_eps0p03_seeds42-44_${RUN_TAG}}"
CHAIN_ROOT="${EXP023_CHAIN_ROOT:-logs/exp023/${CHAIN_ID}}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${CHAIN_ROOT}"
printf '%s\n' "$$" > "${CHAIN_ROOT}/controller.pid"
cat > "${CHAIN_ROOT}/run_config.txt" <<EOF
EXPERIMENT_ID=EXP-023
CHAIN_ID=${CHAIN_ID}
SEEDS=${SEEDS}
GPU_ID=${GPU_ID}
CONDA_ENV=${CONDA_ENV}
RUN_TAG=${RUN_TAG}
DRY_RUN=${DRY_RUN}
EOF

join_by_comma() {
  local IFS=,
  echo "$*"
}

bpda_paths=()
baseline_summaries=()
for seed in ${SEEDS}; do
  run_id="exp023_bpda_pgd10_seed${seed}_${RUN_TAG}"
  log_root="logs/exp023/${run_id}"
  mkdir -p "${log_root}"

  echo "[$(date -Is)] Start EXP-023 seed${seed}: ${run_id}"
  EXP023_SEED="${seed}" \
  EXP023_RUN_ID="${run_id}" \
  EXP023_LOG_ROOT="${log_root}" \
  GPU_ID="${GPU_ID}" \
  CONDA_ENV="${CONDA_ENV}" \
    bash rpcf/run_exp023_bpda_pgd.sh
  echo "[$(date -Is)] Finished EXP-023 seed${seed}: ${run_id}"

  tag="${run_id//[^A-Za-z0-9_.-]/_}"
  bpda_paths+=("ad_data/exp023/${tag}_rpcf_at_bpda_pgd10_rank25.pth")
  bpda_paths+=("ad_data/exp023/${tag}_rpcf_at_bpda_pgd10_rank30.pth")
  case "${seed}" in
    42)
      baseline_summaries+=("logs/exp021/exp021_rpcf_at_seed42_20260623_1303/comparison/summary.json")
      ;;
    43)
      baseline_summaries+=("logs/exp021/exp021_rpcf_at_seeds43-44_20260624_0955_seed43/comparison/summary.json")
      ;;
    44)
      baseline_summaries+=("logs/exp021/exp021_rpcf_at_seeds43-44_20260624_0955_seed44/comparison/summary.json")
      ;;
  esac
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[$(date -Is)] Skip EXP-023 all-seed summary in dry-run mode."
  echo "[$(date -Is)] EXP-023 all-seed pipeline dry-run finished: ${CHAIN_ID}"
  exit 0
fi

bpda_csv="$(join_by_comma "${bpda_paths[@]}")"
baseline_csv="$(join_by_comma "${baseline_summaries[@]}")"
echo "[$(date -Is)] Summarize EXP-023 all seeds"
conda run -n "${CONDA_ENV}" --no-capture-output python -u \
  -m rpcf.compare_exp023 \
  --bpda_paths "${bpda_csv}" --baseline_summaries "${baseline_csv}" \
  --output_dir "${CHAIN_ROOT}/comparison"

echo "[$(date -Is)] EXP-023 all-seed pipeline finished: ${CHAIN_ID}"
