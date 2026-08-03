#!/usr/bin/env bash

# EXP-026 只读复用 EXP-024/025 的 AT checkpoint、six-rank cache 和 sensitivity。

resolve_exp026_artifacts() {
  local model="$1"
  case "${model}" in
    eegnet)
      AT_CHECKPOINT="checkpoints/thubenchmark_eegnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp025_eegnet_prep_seed42_20260706_1225_at_best.pth"
      RPCF_CACHE="purified_data/exp018/rpcf_train/exp025_eegnet_prep_seed42_20260706_1225_six_rank.pth"
      SENSITIVITY_PATH="logs/exp018/exp025_eegnet_prep_seed42_20260706_1225/sensitivity.json"
      CONTEXT_SUMMARY="logs/exp025/exp025_layer_prefix_seed42_20260706_1225_eegnet/eegnet/comparison/summary.json"
      ;;
    tsception)
      AT_CHECKPOINT="checkpoints/thubenchmark_tsception_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp024_parallel_seed42_20260702_153337_tsception_madry_at_best.pth"
      RPCF_CACHE="purified_data/exp024/rpcf_train/exp024_parallel_seed42_20260702_153337_tsception_six_rank.pth"
      SENSITIVITY_PATH="logs/exp024/exp024_parallel_seed42_20260702_153337_tsception/sensitivity.json"
      CONTEXT_SUMMARY="logs/exp024/exp024_parallel_seed42_20260702_153337_tsception/comparison/summary.json"
      ;;
    atcnet)
      AT_CHECKPOINT="checkpoints/thubenchmark_atcnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp024_parallel_seed42_20260702_153337_atcnet_retry2_madry_at_best.pth"
      RPCF_CACHE="purified_data/exp024/rpcf_train/exp024_parallel_seed42_20260702_153337_atcnet_retry2_six_rank.pth"
      SENSITIVITY_PATH="logs/exp024/exp024_parallel_seed42_20260702_153337_atcnet_retry2/sensitivity.json"
      CONTEXT_SUMMARY="logs/exp024/exp024_parallel_seed42_20260702_153337_atcnet_retry2/comparison/summary.json"
      ;;
    conformer)
      AT_CHECKPOINT="checkpoints/thubenchmark_conformer_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp024_parallel_seed42_20260702_153337_conformer_retry2_madry_at_best.pth"
      RPCF_CACHE="purified_data/exp024/rpcf_train/exp024_parallel_seed42_20260702_153337_conformer_retry2_six_rank.pth"
      SENSITIVITY_PATH="logs/exp024/exp024_parallel_seed42_20260702_153337_conformer_retry2/sensitivity.json"
      CONTEXT_SUMMARY="logs/exp024/exp024_parallel_seed42_20260702_153337_conformer_retry2/comparison/summary.json"
      ;;
    deepconvnet)
      AT_CHECKPOINT="checkpoints/thubenchmark_deepconvnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_deepconvnet_madry_at_best.pth"
      RPCF_CACHE="purified_data/exp024/rpcf_train/exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_deepconvnet_six_rank.pth"
      SENSITIVITY_PATH="logs/exp024/exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_deepconvnet/sensitivity.json"
      CONTEXT_SUMMARY="logs/exp024/exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_deepconvnet/comparison/summary.json"
      ;;
    tcnet)
      AT_CHECKPOINT="checkpoints/thubenchmark_tcnet_train_only_subject_no_ea_subject_split_madry_eps0.03_42_fold0_exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_tcnet_madry_at_best.pth"
      RPCF_CACHE="purified_data/exp024/rpcf_train/exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_tcnet_six_rank.pth"
      SENSITIVITY_PATH="logs/exp024/exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_tcnet/sensitivity.json"
      CONTEXT_SUMMARY="logs/exp024/exp024_deepconvnet_tcnet_seed42_20260707_1435_retry1_tcnet/comparison/summary.json"
      ;;
    *)
      echo "Unsupported EXP-026 backbone: ${model}" >&2
      return 1
      ;;
  esac
  export AT_CHECKPOINT RPCF_CACHE SENSITIVITY_PATH CONTEXT_SUMMARY
}

require_exp026_artifact() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required EXP-026 source artifact not found: ${path}" >&2
    return 1
  fi
}
