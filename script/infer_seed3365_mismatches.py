#!/usr/bin/env python3
"""Infer PointNeXt seed3365 over exported review samples and write class mismatches.

Run from anywhere, for example:
  python workspace/PointNeXt/script/infer_seed3365_mismatches.py

The script reads both gt-pred-diff and gt-pred-same below the exported big-dataset
root. It writes one CSV row per sample whose predicted class differs from the
current folder/export class.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

POINTNEXT_ROOT = Path(__file__).resolve().parents[1]
if str(POINTNEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTNEXT_ROOT))

from examples.classification.dataloader import (  # noqa: E402
    PointCloudDataset,
    _load_pcd_xyz,
    _metadata_distance_from_json,
)
from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.utils import EasyConfig, load_checkpoint, set_random_seed  # noqa: E402

DEFAULT_EXPORT_ROOT = Path("/home/georg/big-dataset/new-config-dataset-samples-since-2026-05-05-4096")
DEFAULT_BUCKETS = ("gt-pred-diff", "gt-pred-same")
DEFAULT_RUN_DIR = Path(
    "/home/georg/workspace/PointNeXt/log/modelnet40ply2048/"
    "modelnet40ply2048-train-pointnext-normal-ngpus1-seed3365-20260526-153936-85FdgvGzX5Qi37DNpLdcua"
)
DEFAULT_CFG = POINTNEXT_ROOT / "cfgs/modelnet40ply2048/pointnext-normal.yaml"
DEFAULT_OUTPUT = Path("/home/georg/big-dataset/pointnext_seed3365_mismatches.csv")
DEFAULT_TRAINED_CLASSES = (
    "TLS_VEHICLE_BUS",
    "TLS_VEHICLE_CAR",
    "TLS_VEHICLE_MOTORBIKE",
    "TLS_VEHICLE_SEMI_TRAILER_TRUCK",
    "TLS_VEHICLE_TRAILER",
    "TLS_VEHICLE_TRUCK",
    "TLS_VEHICLE_VAN",
)


@dataclass(frozen=True)
class ExportSample:
    bucket: str
    current_class: str
    sample_id: str
    object_id: str
    json_path: Path
    pcd_path: Path
    point_count: int | None
    day: str
    run_id: str
    track_id: str


class ExportInferenceDataset(Dataset):
    def __init__(
        self,
        samples: list[ExportSample],
        class_to_idx: dict[str, int],
        num_points: int = 1024,
        use_normals: bool = False,
        normal_k: int = 16,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        self.num_classes = len(self.classes)
        # Reuse PointCloudDataset preprocessing helpers without running its
        # directory scanner in __init__.
        self.helper = object.__new__(PointCloudDataset)
        self.helper.samples = []
        self.helper.classes = self.classes
        self.helper.class_to_idx = self.class_to_idx
        self.helper.num_classes = self.num_classes
        self.helper.split = "test"
        self.helper.num_points = int(num_points)
        self.helper.normalize = True
        self.helper.jitter_std = 0.01
        self.helper.task = "classification"
        self.helper.is_completion = False
        self.helper.augment_train = False
        self.helper.augment_dropout = (0.0, 0.0)
        self.helper.use_normals = bool(use_normals)
        self.helper.normal_k = max(3, int(normal_k or 16))
        self.helper._rotation_transform = None
        self.helper._scale_transform = None
        self.helper._jitter_transform = None
        self.helper._translation_transform = None
        self.num_points = int(num_points)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        points = _load_pcd_xyz(sample.pcd_path)
        raw_dist = _metadata_distance_from_json(sample.pcd_path, metadata=None)
        obj_features = self.helper._obj_features(points, raw_dist)

        points = self.helper._normalize(points)
        points = self.helper._apply_point_dropout(points)
        points = self.helper._resample(points, target_count=self.num_points, jitter=False)
        points = self.helper._scale_to_unit_radius(points)[0]
        points = self.helper._apply_geometric_augmentations(points)

        current_label = self.class_to_idx.get(sample.current_class, -1)
        return {
            "x": torch.from_numpy(self.helper._point_features(points)).float(),
            "pos": torch.from_numpy(points).float(),
            "y": torch.tensor(current_label).long(),
            "obj_features": torch.from_numpy(obj_features).float(),
            "sample_index": torch.tensor(idx).long(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferiere PointNeXt seed3365 ueber gt-pred-diff + gt-pred-same und schreibe abweichende Klassen in CSV."
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: <run-dir>/checkpoint/*_ckpt_best.pth")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--limit", type=int, default=0, help="Optional fuer Smoke-Tests: nur die ersten N Samples inferieren.")
    return parser.parse_args()


def find_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint.expanduser().resolve()
    matches = sorted((run_dir / "checkpoint").glob("*_ckpt_best.pth"))
    if not matches:
        raise FileNotFoundError(f"Kein *_ckpt_best.pth unter {run_dir / 'checkpoint'} gefunden")
    return matches[0].resolve()


def load_class_names(run_dir: Path) -> list[str]:
    cm_dir = run_dir / "confusion_matrix"
    candidates = sorted(cm_dir.glob("val_epoch_*_raw.csv"))
    if candidates:
        # Prefer the numerically latest epoch if possible.
        def epoch_num(path: Path) -> int:
            stem = path.stem
            try:
                return int(stem.split("_epoch_")[1].split("_")[0])
            except Exception:
                return -1

        latest = max(candidates, key=epoch_num)
        with latest.open(newline="") as handle:
            header = next(csv.reader(handle))
        if len(header) > 1:
            return header[1:]
    return list(DEFAULT_TRAINED_CLASSES)


def iter_export_samples(export_root: Path, buckets: Iterable[str]) -> Iterable[ExportSample]:
    for bucket in buckets:
        bucket_dir = export_root / bucket
        if not bucket_dir.is_dir():
            continue
        for class_dir in sorted(path for path in bucket_dir.iterdir() if path.is_dir()):
            json_dir = class_dir / "json"
            pcd_dir = class_dir / "pcds"
            if not json_dir.is_dir() or not pcd_dir.is_dir():
                continue
            for json_path in sorted(json_dir.glob("*.json")):
                try:
                    with json_path.open("r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    continue
                sample_id = str(meta.get("sample_id") or json_path.stem)
                pcd_path = pcd_dir / f"{sample_id}.pcd"
                if not pcd_path.exists():
                    continue
                object_id = meta.get("object_id", meta.get("gt_object_id", ""))
                track_id = meta.get("track_id", "")
                point_count = meta.get("point_count")
                yield ExportSample(
                    bucket=bucket,
                    current_class=str(meta.get("export_class_name") or meta.get("class_name") or class_dir.name),
                    sample_id=sample_id,
                    object_id=str(object_id),
                    json_path=json_path,
                    pcd_path=pcd_path,
                    point_count=int(point_count) if point_count is not None else None,
                    day=str(meta.get("day", "")),
                    run_id=str(meta.get("run_id", "")),
                    track_id=str(track_id),
                )


def build_cfg(cfg_path: Path, class_names: list[str], args: argparse.Namespace) -> EasyConfig:
    cfg = EasyConfig()
    cfg.load(str(cfg_path), recursive=True)
    cfg.seed = 3365
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


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    non_blocking = device.type == "cuda"
    for key, value in list(batch.items()):
        if hasattr(value, "to"):
            batch[key] = value.to(device, non_blocking=non_blocking)
    return batch


def write_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bucket",
                "object_id",
                "track_id",
                "sample_id",
                "current_class",
                "predicted_class",
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
    write_header(output_path)

    mismatch_count = 0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", unit="batch"):
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
            for sample_idx, pred_idx, score in zip(sample_indices, pred, conf):
                sample = samples[int(sample_idx)]
                total_count += 1
                predicted_class = class_names[int(pred_idx)]
                if predicted_class == sample.current_class:
                    continue
                mismatch_count += 1
                rows.append(
                    [
                        sample.bucket,
                        sample.object_id,
                        sample.track_id,
                        sample.sample_id,
                        sample.current_class,
                        predicted_class,
                        f"{float(score):.6f}",
                        "" if sample.point_count is None else sample.point_count,
                        sample.day,
                        sample.run_id,
                        str(sample.json_path),
                        str(sample.pcd_path),
                    ]
                )
            append_rows(output_path, rows)

    print(f"checkpoint: {checkpoint}")
    print(f"classes: {', '.join(class_names)}")
    print(f"samples inferred: {total_count}")
    print(f"mismatches written: {mismatch_count}")
    print(f"csv: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
