#!/usr/bin/env python3
"""Compare the FP32 and FP16 ONNX classifiers on the fixed raw-frame test split."""

from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.validate_mvf_10class_ablation import (  # noqa: E402
    RawFramesClassificationDataset,
    dataset_kwargs,
    load_cfg,
)


RUN_DIR = ROOT / (
    "log/mvf_10class_ablation_shared_encoder_gt_pose/"
    "mvf_10class_ablation_shared_encoder_gt_pose-raw_frames_classification-"
    "01_baseline_5views_512_geometry_gradients-ngpus1-seed8240-"
    "20260711-213551-m2474PJB27sDKzaaskrhxf"
)
CFG = ROOT / "cfgs/modelnet40ply2048/mvf_10class_ablation_shared_encoder_gt_pose/01_baseline_5views_512_geometry_gradients.yaml"


def main() -> None:
    cfg = load_cfg(CFG)
    dataset = RawFramesClassificationDataset(
        **dataset_kwargs(cfg, Path(cfg.raw_frames_root), "test", top_down=False)
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    providers = ["CPUExecutionProvider"]
    model_dir = RUN_DIR / "onnx"
    fp32 = ort.InferenceSession(str(model_dir / "mvf_01_baseline_5views_512.onnx"), providers=providers)
    fp16 = ort.InferenceSession(str(model_dir / "mvf_01_baseline_5views_512_fp16.onnx"), providers=providers)
    correct32 = correct16 = changed = total = 0
    for batch in loader:
        views, mask, labels = batch["views"].numpy(), batch["view_mask"].numpy(), batch["y"].numpy()
        pred32 = fp32.run(None, {"views": views.astype(np.float32), "view_mask": mask})[0].argmax(1)
        pred16 = fp16.run(None, {"views": views.astype(np.float16), "view_mask": mask})[0].argmax(1)
        correct32 += int((pred32 == labels).sum())
        correct16 += int((pred16 == labels).sum())
        changed += int((pred32 != pred16).sum())
        total += len(labels)
    print(f"samples={total}")
    print(f"fp32_accuracy={correct32 / total:.6%} ({correct32}/{total})")
    print(f"fp16_accuracy={correct16 / total:.6%} ({correct16}/{total})")
    print(f"accuracy_delta={(correct16 - correct32) / total:+.6%}")
    print(f"changed_predictions={changed} ({changed / total:.6%})")


if __name__ == "__main__":
    main()
