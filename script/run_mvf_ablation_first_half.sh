#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_PROJECT="${WANDB_PROJECT:-mvf_10class_ablation}"

CONFIGS=(
  "cfgs/modelnet40ply2048/mvf_10class_ablation/01_mvf_coarse_pose_fine_icp_main_5v_512_current.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/02_mvf_fine_icp_pointnext_small.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/03_mvf_fine_icp_pointnext_large.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/04_mvf_fine_icp_1view.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/05_mvf_fine_icp_2views.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/06_mvf_fine_icp_3views.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/07_mvf_fine_icp_256pts_per_view.yaml"
  "cfgs/modelnet40ply2048/mvf_10class_ablation/08_mvf_fine_icp_1024pts_per_view.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${cfg}"
  "${PYTHON_BIN}" examples/classification/main.py --cfg "${cfg}" \
    wandb.use_wandb=True \
    "wandb.project=${WANDB_PROJECT}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${cfg}"
done
