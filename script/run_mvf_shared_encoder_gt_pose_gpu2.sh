#!/usr/bin/env bash
# Runs up to three trainings concurrently on one physical GPU.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-mvf_10class_ablation_shared_encoder_gt_pose}"
LOG_DIR="${LOG_DIR:-log/mvf_shared_encoder_gt_pose_gpu${GPU_ID}}"
FAILURES_FILE="${LOG_DIR}/failed_runs.txt"
mkdir -p "${LOG_DIR}"
: > "${FAILURES_FILE}"

run_one() {
  local cfg="$1"
  local name="${cfg##*/}"
  name="${name%.yaml}"
  local log_file="${LOG_DIR}/${name}.log"

  echo "[$(date '+%F %T')] GPU ${GPU_ID}: starting ${cfg}"
  if CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" examples/classification/main.py --cfg "${cfg}" \
      wandb.use_wandb=True \
      "wandb.project=${WANDB_PROJECT}" >"${log_file}" 2>&1; then
    echo "[$(date '+%F %T')] GPU ${GPU_ID}: finished ${cfg}"
  else
    local status=$?
    echo "[$(date '+%F %T')] GPU ${GPU_ID}: FAILED (exit ${status}) ${cfg}" | tee -a "${FAILURES_FILE}"
  fi
}

worker() {
  local cfg
  for cfg in "$@"; do
    run_one "${cfg}"
  done
}

# Seven runs, distributed across three independent serial workers.
worker \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/10_coarse_pose_no_fine_refinement.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/13_icp_aggregated_pointclouds_pointnext_10class_baseline.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/17_5views_no_geometry_alignment.yaml" &
pids[0]=$!

worker \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/11_learned_fine_pose_mlp.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/14_learned_fine_pose_mlp_icp_teacher.yaml" &
pids[1]=$!

worker \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/12_single_view_pointnext_10class_baseline.yaml" \
  "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/15_5views_alignment_only.yaml" &
pids[2]=$!

for pid in "${pids[@]}"; do
  wait "${pid}" || true
done

echo "[$(date '+%F %T')] GPU ${GPU_ID}: queue complete. Failures: ${FAILURES_FILE}"
