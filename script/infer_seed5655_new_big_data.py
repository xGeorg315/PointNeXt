#!/usr/bin/env python3
"""Infer PointNeXt seed5655 over /home/georg/new-big-data exports.

Run from anywhere, for example:
  python /home/georg/workspace/PointNeXt/script/infer_seed5655_new_big_data.py

The script scans both gt-pred-diff and gt-pred-same below the exported
new-big-data root. It writes one CSV with all predictions and one CSV with only
rows where the predicted class differs from the current export class.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

POINTNEXT_ROOT = Path(__file__).resolve().parents[1]
if str(POINTNEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTNEXT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_seed3365_mismatches import (  # noqa: E402
    ExportInferenceDataset,
    find_checkpoint,
    iter_export_samples,
    load_class_names,
    move_batch_to_device,
)
from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.utils import EasyConfig, load_checkpoint, set_random_seed  # noqa: E402

SEED = 5655
DEFAULT_EXPORT_ROOT = Path(
    "/home/georg/new-big-data/new-config-dataset-samples-since-2026-05-05-4096"
)
DEFAULT_BUCKETS = ("gt-pred-diff", "gt-pred-same")
DEFAULT_RUN_DIR = Path(
    "/home/georg/workspace/PointNeXt/log/modelnet40ply2048/"
    "modelnet40ply2048-train-pointnext-normal-ngpus1-seed5655-20260527-110123-eYCs6zYLroM43Yj4NS5dx5"
)
DEFAULT_CFG = DEFAULT_RUN_DIR / "pointnext-normal.yaml"
DEFAULT_OUTPUT = Path("/home/georg/new-big-data/pointnext_seed5655_predictions.csv")
DEFAULT_MISMATCH_OUTPUT = Path("/home/georg/new-big-data/pointnext_seed5655_mismatches.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inferiere PointNeXt seed5655 ueber new-big-data/gt-pred-diff "
            "+ gt-pred-same und schreibe Predictions als CSV."
        )
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Default: <run-dir>/checkpoint/*_ckpt_best.pth",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mismatch-output", type=Path, default=DEFAULT_MISMATCH_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional fuer Smoke-Tests: nur die ersten N Samples inferieren.",
    )
    return parser.parse_args()


def build_cfg(cfg_path: Path, class_names: list[str], args: argparse.Namespace) -> EasyConfig:
    cfg = EasyConfig()
    cfg.load(str(cfg_path), recursive=True)
    cfg.seed = SEED
    cfg.rank = 0
    cfg.world_size = 1
    cfg.distributed = False
    cfg.mp = False
    cfg.sync_bn = False
    cfg.num_points = int(args.num_points)
    cfg.classes = class_names
    cfg.num_classes = len(class_names)
    cfg.use_normals = bool(cfg.get("use_normals", False))
    cfg.normal_k = int(cfg.get("normal_k", 16))
    cfg.model.cls_args.num_classes = len(class_names)
    cfg.model.encoder_args.in_channels = 6 if cfg.use_normals else 3
    cfg.model.in_channels = cfg.model.encoder_args.in_channels
    cfg.model.extra_global_channels = int(cfg.model.get("extra_global_channels", 0))
    return cfg


def write_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                "bucket",
                "object_id",
                "track_id",
                "sample_id",
                "current_class",
                "predicted_class",
                "is_match",
                "confidence",
                "point_count",
                "day",
                "run_id",
                "json_path",
                "pcd_path",
            ]
        )


def append_rows(path: Path, rows: list[list[object]]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def prediction_row(sample, predicted_class: str, confidence: float) -> list[object]:
    is_match = predicted_class == sample.current_class
    return [
        sample.bucket,
        sample.object_id,
        sample.track_id,
        sample.sample_id,
        sample.current_class,
        predicted_class,
        int(is_match),
        f"{float(confidence):.6f}",
        "" if sample.point_count is None else sample.point_count,
        sample.day,
        sample.run_id,
        str(sample.json_path),
        str(sample.pcd_path),
    ]


def main() -> int:
    args = parse_args()
    export_root = args.export_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = find_checkpoint(run_dir, args.checkpoint)
    class_names = load_class_names(run_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples = list(iter_export_samples(export_root, args.buckets))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"Keine Samples unter {export_root} fuer Buckets {args.buckets} gefunden")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = build_cfg(args.cfg.expanduser().resolve(), class_names, args)
    set_random_seed(cfg.seed, deterministic=True)

    dataset = ExportInferenceDataset(
        samples=samples,
        class_to_idx=class_to_idx,
        num_points=cfg.num_points,
        use_normals=cfg.use_normals,
        normal_k=cfg.normal_k,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_cfg(cfg.model).to(device)
    load_checkpoint(model, str(checkpoint))
    model.eval()

    output_path = args.output.expanduser().resolve()
    mismatch_path = args.mismatch_output.expanduser().resolve()
    write_csv_header(output_path)
    write_csv_header(mismatch_path)

    mismatch_count = 0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer seed5655", unit="batch"):
            sample_indices = batch.pop("sample_index").cpu().numpy().tolist()
            batch = move_batch_to_device(batch, device)
            points = batch["x"][:, : cfg.num_points]
            batch["pos"] = points[:, :, :3].contiguous()
            batch["x"] = points[:, :, : cfg.model.in_channels].transpose(1, 2).contiguous()

            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            pred = pred.detach().cpu().numpy().tolist()
            conf = conf.detach().cpu().numpy().tolist()

            rows = []
            mismatch_rows = []
            for sample_idx, pred_idx, score in zip(sample_indices, pred, conf):
                sample = samples[int(sample_idx)]
                predicted_class = class_names[int(pred_idx)]
                row = prediction_row(sample, predicted_class, float(score))
                total_count += 1
                rows.append(row)
                if predicted_class != sample.current_class:
                    mismatch_count += 1
                    mismatch_rows.append(row)
            append_rows(output_path, rows)
            append_rows(mismatch_path, mismatch_rows)

    print(f"checkpoint: {checkpoint}")
    print(f"classes: {', '.join(class_names)}")
    print(f"samples inferred: {total_count}")
    print(f"mismatches written: {mismatch_count}")
    print(f"predictions csv: {output_path}")
    print(f"mismatches csv: {mismatch_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
