#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_PROJECT="${WANDB_PROJECT:-mvf_10class_ablation}"

CONFIGS=(
  "cfgs/modelnet40ply2048/mvf_10class_ablation/12_single_view_pointnext_10class_baseline.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/13_icp_aggregated_pointclouds_pointnext_10class_baseline.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/14_mvf_fine_pose_head_no_icp_teacher.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/15_mvf_fine_pose_head_with_icp_teacher.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${cfg}"
  "${PYTHON_BIN}" examples/classification/main.py --cfg "${cfg}" \
    wandb.use_wandb=True \
    "wandb.project=${WANDB_PROJECT}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${cfg}"
done
