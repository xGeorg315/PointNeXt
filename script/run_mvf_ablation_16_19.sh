#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_PROJECT="${WANDB_PROJECT:-mvf_10class_ablation}"

CONFIGS=(
  "cfgs/modelnet40ply2048/mvf_10class_ablation/16_mvf_coarse_pose_fine_icp_main_5v_512_small.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/17_mvf_coarse_pose_fine_icp_main_5v_512_large.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/18_mvf_coarse_pose_fine_icp_main_5v_512_view_pool_softmax.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/19_mvf_coarse_pose_fine_icp_main_5v_512_view_pool_gated_weight.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${cfg}"
  "${PYTHON_BIN}" examples/classification/main.py --cfg "${cfg}" \
    wandb.use_wandb=True \
    "wandb.project=${WANDB_PROJECT}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${cfg}"
done
