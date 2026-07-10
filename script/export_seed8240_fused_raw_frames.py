#!/usr/bin/env python3
"""Export fused raw-frame clouds for an arbitrary PointNeXt run.

Examples:
  python script/export_seed8240_fused_raw_frames.py --seed 8240
  python script/export_seed8240_fused_raw_frames.py --seed 42 --cfg cfgs/my-config.yaml
  python script/export_seed8240_fused_raw_frames.py --cfg cfgs/my-config.yaml --checkpoint /path/to/ckpt.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

POINTNEXT_ROOT = Path(__file__).resolve().parents[1]
if str(POINTNEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTNEXT_ROOT))

from examples.classification.dataloader import RawFramesClassificationDataset  # noqa: E402
from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.utils import EasyConfig, load_checkpoint, set_random_seed  # noqa: E402


DEFAULT_CFG = POINTNEXT_ROOT / "cfgs/modelnet40ply2048/pointnext-observed-only-fusion-pose-raw-frames.yaml"
DEFAULT_SEED = 8240
DEFAULT_CHECKPOINT_SEARCH_ROOT = POINTNEXT_ROOT / "log"
DEFAULT_CLASSES = (
    "TLS_VEHICLE_BUS",
    "TLS_VEHICLE_CAR",
    "TLS_VEHICLE_MOTORBIKE",
    "TLS_VEHICLE_SEMI_TRAILER_TRUCK",
    "TLS_VEHICLE_SMALL_TRUCK",
    "TLS_VEHICLE_TRAILER",
    "TLS_VEHICLE_TRUCK",
    "TLS_VEHICLE_VAN",
    "small construction vehicle",
    "small tow truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exportiere fused clouds aus einem beliebigen PointNeXt-Raw-Frames-Lauf."
    )
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="Pfad zur Trainings-Config.")
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Optionaler Override; standardmaessig wird raw_frames_root aus der Config verwendet.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Standard: exports/<config-name>_seed<seed>_fused_raw_frames.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional: Run-Verzeichnis mit checkpoint/*_ckpt_best.pth.")
    parser.add_argument(
        "--checkpoint-search-root", type=Path, default=DEFAULT_CHECKPOINT_SEARCH_ROOT,
        help="Wurzel fuer die automatische rekursive Checkpoint-Suche (Standard: log/).",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--seed", type=int, default=None,
        help=f"Seed-Override; sonst Config-Wert, danach Fallback {DEFAULT_SEED}.",
    )
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Optionaler Override; sonst val_batch_size/batch_size aus der Config.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--confidence-head", default="cfg", choices=("cfg", "on", "off"))
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--fusion-voxel-size", type=float, default=None)
    parser.add_argument("--preload-data", action="store_true")
    parser.add_argument("--save-raw-frames", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true", help="Nur fuer Smoke-Tests: Modell nicht laden.")
    parser.add_argument("--skip-shape-mismatch", action="store_true")
    return parser.parse_args()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def classification_augmentation_kwargs(cfg: EasyConfig) -> dict:
    return dict(
        augment_train=as_bool(cfg.get("augment_train", False)),
        augment_rotation_deg=cfg.get("augment_rotation_deg", 20.0),
        augment_scale=cfg.get("augment_scale", (0.8, 1.2)),
        augment_jitter_sigma=cfg.get("augment_jitter_sigma", 0.01),
        augment_jitter_clip=cfg.get("augment_jitter_clip", 0.05),
        augment_dropout=cfg.get("augment_dropout", (0.2, 0.5)),
        augment_translate=cfg.get("augment_translate", 0.1),
        augment_random_points_ratio=cfg.get("augment_random_points_ratio", 0.0),
        augment_random_points_scale=cfg.get("augment_random_points_scale", 1.0),
        augment_viewpoint_topdown_prob=cfg.get("augment_viewpoint_topdown_prob", 0.0),
        augment_topdown_keep_ratio=cfg.get("augment_topdown_keep_ratio", (0.25, 0.6)),
        augment_topdown_z_squash=cfg.get("augment_topdown_z_squash", (0.5, 1.0)),
        augment_topdown_top_bias=cfg.get("augment_topdown_top_bias", 0.6),
        augment_topdown_xy_jitter=cfg.get("augment_topdown_xy_jitter", 0.0),
        use_normals=as_bool(cfg.get("use_normals", False)),
        normal_k=cfg.get("normal_k", 16),
        preload_data=as_bool(cfg.get("preload_data", False)),
        multi_view=as_bool(cfg.get("multi_view", False)),
        multi_view_axes=cfg.get("multi_view_axes", ("xy", "xz", "yz")),
        multi_view_num_points=cfg.get("multi_view_num_points", 512),
        multi_view_bins=cfg.get("multi_view_bins", 256),
        obj_features_include_sensor_dist=as_bool(cfg.get("obj_features_include_sensor_dist", True)),
    )


def configure_point_feature_channels(cfg: EasyConfig) -> None:
    if not as_bool(cfg.get("use_normals", False)):
        return
    if cfg.model.get("encoder_args", None) is not None:
        cfg.model.encoder_args.in_channels = 6
    cfg.model.in_channels = 6


def move_batch_to_device(data: dict, device: torch.device) -> dict:
    non_blocking = device.type == "cuda"
    for key, value in data.items():
        if hasattr(value, "to"):
            data[key] = value.to(device, non_blocking=non_blocking)
    return data


def consolidate_observed_points(points: torch.Tensor, confidence: torch.Tensor, voxel_size: float, min_confidence: float) -> torch.Tensor:
    keep = confidence >= min_confidence
    points = points[keep]
    confidence = confidence[keep]
    if points.numel() == 0:
        return points.reshape(0, 3)
    if voxel_size <= 0:
        return points
    voxel_keys = torch.floor(points / voxel_size).to(torch.int64)
    _, inverse = torch.unique(voxel_keys, dim=0, return_inverse=True)
    representatives = []
    for voxel_idx in range(int(inverse.max().item()) + 1):
        candidates = torch.nonzero(inverse == voxel_idx, as_tuple=False).flatten()
        best = candidates[confidence[candidates].argmax()]
        representatives.append(points[best])
    return torch.stack(representatives, dim=0)


def write_ascii_ply(path: str | Path, points: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = points.detach().cpu().float().numpy()
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"PLY export expects Nx3+ points, got shape {xyz.shape}")
    xyz = xyz[:, :3]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(handle, xyz, fmt="%.7f %.7f %.7f")



def confidence_to_rgb(confidence: torch.Tensor, kept: torch.Tensor) -> np.ndarray:
    conf = confidence.detach().cpu().float().clamp(0.0, 1.0).numpy()
    keep = kept.detach().cpu().bool().numpy()
    rgb = np.zeros((conf.shape[0], 3), dtype=np.uint8)
    rgb[:, 0] = np.round(255.0 * (1.0 - conf)).astype(np.uint8)
    rgb[:, 1] = np.round(255.0 * conf).astype(np.uint8)
    rgb[:, 2] = 32
    rgb[~keep] = np.array([255, 32, 32], dtype=np.uint8)
    return rgb


def write_confidence_ply(
    path: str | Path,
    points: torch.Tensor,
    confidence: torch.Tensor,
    kept: torch.Tensor,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = points.detach().cpu().float().numpy()
    conf = confidence.detach().cpu().float().clamp(0.0, 1.0).numpy()
    keep = kept.detach().cpu().bool().numpy()
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"PLY export expects Nx3+ points, got shape {xyz.shape}")
    if xyz.shape[0] != conf.shape[0] or xyz.shape[0] != keep.shape[0]:
        raise ValueError(
            "points, confidence and kept mask must have equal length, "
            f"got {xyz.shape[0]}, {conf.shape[0]}, {keep.shape[0]}"
        )
    xyz = xyz[:, :3]
    rgb = confidence_to_rgb(confidence, kept)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property float confidence\nproperty uchar kept\nend_header\n")
        for point, color, value, is_kept in zip(xyz, rgb, conf, keep):
            handle.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(value):.6f} {int(is_kept)}\n"
            )


def find_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.skip_checkpoint:
        return None
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve()
    if args.run_dir is not None:
        matches = sorted((args.run_dir.expanduser() / "checkpoint").glob("*_ckpt_best.pth"))
        if not matches:
            raise FileNotFoundError(f"Kein *_ckpt_best.pth unter {args.run_dir / 'checkpoint'} gefunden.")
        return matches[0].resolve()

    search_root = args.checkpoint_search_root.expanduser().resolve()
    matches = sorted(search_root.rglob(f"*seed{args.seed}*/checkpoint/*_ckpt_best.pth"))
    if not matches:
        raise FileNotFoundError(
            f"Kein Checkpoint fuer Seed {args.seed} unter {search_root} gefunden. "
            "Bitte --checkpoint, --run-dir oder --checkpoint-search-root angeben oder fuer einen "
            "reinen Smoke-Test --skip-checkpoint nutzen."
        )

    cfg_name = args.cfg.stem.lower()
    matching_cfg = [path for path in matches if cfg_name in str(path).lower()]
    candidates = matching_cfg or matches
    if len(candidates) > 1:
        logging.warning(
            "Mehrere Checkpoints passen zu Seed %s; verwende %s. "
            "Fuer eine eindeutige Auswahl --checkpoint angeben.",
            args.seed,
            candidates[-1],
        )
    return candidates[-1].resolve()


def load_cfg(args: argparse.Namespace) -> EasyConfig:
    cfg = EasyConfig()
    args.cfg = args.cfg.expanduser().resolve()
    if not args.cfg.is_file():
        raise FileNotFoundError(f"Config nicht gefunden: {args.cfg}")
    cfg.load(str(args.cfg), recursive=True)

    args.seed = int(args.seed if args.seed is not None else cfg.get("seed", DEFAULT_SEED))
    cfg.seed = args.seed
    cfg.rank = 0
    cfg.world_size = 1
    cfg.distributed = False
    cfg.mp = False
    cfg.sync_bn = False
    configured_dataset_root = cfg.get("raw_frames_root", cfg.get("custom_dataset_root", None))
    if args.dataset_root is None and configured_dataset_root is None:
        raise ValueError(
            "Die Config enthaelt weder raw_frames_root noch custom_dataset_root; "
            "bitte --dataset-root angeben."
        )
    args.dataset_root = Path(args.dataset_root or configured_dataset_root).expanduser().resolve()
    cfg.raw_frames_root = str(args.dataset_root)
    cfg.custom_dataset_root = str(args.dataset_root)

    args.batch_size = int(
        args.batch_size if args.batch_size is not None
        else cfg.get("val_batch_size", cfg.get("batch_size", 16))
    )
    cfg.batch_size = args.batch_size
    cfg.val_batch_size = args.batch_size
    cfg.preload_data = bool(args.preload_data)
    cfg.augment_train = False
    cfg.raw_frames_object_multi_view = True
    cfg.raw_frames_frame_selection = cfg.get("raw_frames_frame_selection", "all")
    cfg.raw_frames_max_views = int(cfg.get("raw_frames_max_views", cfg.model.get("max_views", 5)))
    cfg.raw_frames_view_selection = cfg.get("raw_frames_view_selection", "uniform")
    cfg.model.max_views = cfg.raw_frames_max_views
    if args.confidence_head != "cfg":
        cfg.model.enable_confidence_head = args.confidence_head == "on"
    if args.confidence_threshold is not None:
        cfg.model.confidence_threshold = float(args.confidence_threshold)
    if args.fusion_voxel_size is not None:
        cfg.model.fusion_voxel_size = float(args.fusion_voxel_size)
        cfg.fusion_voxel_size = float(args.fusion_voxel_size)
    cfg.raw_frames_classes = cfg.get("raw_frames_classes", list(DEFAULT_CLASSES))
    cfg.classes = list(cfg.raw_frames_classes)
    cfg.num_classes = len(cfg.classes)
    cfg.model.cls_args.num_classes = cfg.num_classes
    configure_point_feature_channels(cfg)
    return cfg


def configure_output_root(args: argparse.Namespace) -> None:
    if args.output_root is None:
        args.output_root = (
            POINTNEXT_ROOT
            / "exports"
            / f"{safe_name(args.cfg.stem)}_seed{args.seed}_fused_raw_frames"
        )
    else:
        args.output_root = args.output_root.expanduser().resolve()


def build_dataset(cfg: EasyConfig, split: str) -> RawFramesClassificationDataset:
    exclude_classes = list(cfg.get("raw_frames_exclude_classes", cfg.get("exclude_classes", ("reject",))))
    for class_name in cfg.get("exclude_classes", ()):
        if class_name not in exclude_classes:
            exclude_classes.append(class_name)
    dataset_kwargs = classification_augmentation_kwargs(cfg)
    dataset_kwargs["preload_data"] = as_bool(cfg.get("preload_data", False))
    return RawFramesClassificationDataset(
        root_dir=cfg.raw_frames_root,
        split=split,
        num_points=cfg.num_points,
        start_date=cfg.get("raw_frames_start_date", None),
        min_points=cfg.get("raw_frames_min_points", 0),
        split_ratios=cfg.get("raw_frames_split_ratios", (0.8, 0.1, 0.1)),
        exclude_classes=exclude_classes,
        min_points_exempt_classes=cfg.get(
            "raw_frames_min_points_exempt_classes",
            ("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
        ),
        forced_classes=cfg.get("raw_frames_classes", None),
        frame_selection=cfg.get("raw_frames_frame_selection", "all"),
        object_multi_view=True,
        max_views=cfg.get("raw_frames_max_views", 5),
        view_selection=cfg.get("raw_frames_view_selection", "uniform"),
        pose_metadata_root=cfg.get("raw_frames_pose_root", None),
        pose_required=as_bool(cfg.get("raw_frames_pose_required", False)),
        **dataset_kwargs,
    )


def restrict_dataset_classes(dataset: RawFramesClassificationDataset, class_names: list[str]) -> None:
    class_names = [name for name in class_names if name in dataset.class_to_idx]
    keep = set(class_names)
    old_classes = list(dataset.classes)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    filtered_samples = []
    for sample in dataset.samples:
        class_name = sample.get("class_name", old_classes[int(sample["label"])])
        if class_name not in keep:
            continue
        sample = dict(sample)
        sample["label"] = class_to_idx[class_name]
        filtered_samples.append(sample)
    dataset.samples = filtered_samples
    dataset.classes = class_names
    dataset.class_to_idx = class_to_idx
    dataset.num_classes = len(class_names)


def prepare_batch(data: dict, cfg: EasyConfig, device: torch.device) -> dict:
    data = move_batch_to_device(data, device)
    data["views"] = data["views"].contiguous()
    data["pos"] = data["views"][:, 0, :, :3].contiguous()
    data["x"] = data["pos"].transpose(1, 2).contiguous()
    return data


def export_raw_frames(base_path: Path, views: torch.Tensor, valid_views: torch.Tensor) -> None:
    stem = base_path.with_suffix("")
    for output_idx, view_idx in enumerate(torch.nonzero(valid_views, as_tuple=False).flatten().tolist()):
        write_ascii_ply(str(stem) + f"_raw_frame_{output_idx:02d}.ply", views[view_idx])


def export_batch(
    model: torch.nn.Module,
    data: dict,
    dataset: RawFramesClassificationDataset,
    batch_start: int,
    cfg: EasyConfig,
    args: argparse.Namespace,
    counts: dict[str, int],
    manifest: list[dict],
) -> None:
    logits = model(data)
    pred = logits.argmax(dim=1).detach().cpu().tolist()
    output = model.last_output
    points = output["transformed_points"].detach()
    confidence = output["point_confidence"].detach()
    point_mask = output.get("point_mask")
    if point_mask is not None:
        point_mask = point_mask.detach()
    view_mask = output["view_mask"].detach()
    labels = data["y"].detach().cpu().tolist()
    voxel_size = float(cfg.model.get("fusion_voxel_size", cfg.get("fusion_voxel_size", 0.02)))
    use_confidence = bool(getattr(model, "enable_confidence_head", False))

    for batch_idx, label in enumerate(labels):
        sample_idx = batch_start + batch_idx
        if sample_idx >= len(dataset.samples):
            continue
        class_name = dataset.classes[int(label)]
        if counts.get(class_name, 0) >= args.per_class:
            continue

        valid_views = view_mask[batch_idx]
        all_points = points[batch_idx][valid_views].reshape(-1, 3)
        all_confidence = confidence[batch_idx][valid_views].reshape(-1)
        if point_mask is not None:
            selected = point_mask[batch_idx][valid_views].reshape(-1)
        else:
            selected = torch.ones_like(all_confidence, dtype=torch.bool)
        valid_points = all_points[selected]
        valid_confidence = all_confidence[selected]
        merged = consolidate_observed_points(valid_points, valid_confidence, voxel_size, 0.0)

        export_idx = counts.get(class_name, 0)
        sample = dataset.samples[sample_idx]
        class_dir = args.output_root / ("confidence_on" if use_confidence else "confidence_off") / safe_name(class_name)
        path = class_dir / f"sample_{export_idx:02d}_{safe_name(sample.get('object_id', sample_idx))}.ply"
        write_ascii_ply(str(path), merged)
        confidence_path = path.with_name(path.stem + "_confidence.ply")
        write_confidence_ply(confidence_path, all_points, all_confidence, selected)
        if args.save_raw_frames:
            export_raw_frames(path, data["views"][batch_idx].detach(), valid_views)

        manifest.append(
            {
                "path": str(path),
                "class_name": class_name,
                "sample_index": sample_idx,
                "object_id": sample.get("object_id"),
                "sample_id": sample.get("sample_id"),
                "frame_id": sample.get("frame_id"),
                "num_views": int(sample.get("num_views", int(valid_views.sum().item()))),
                "num_fused_points": int(merged.shape[0]),
                "confidence_path": str(confidence_path),
                "num_confidence_points": int(all_points.shape[0]),
                "num_kept_points": int(selected.sum().detach().cpu().item()),
                "num_rejected_points": int((~selected).sum().detach().cpu().item()),
                "true_label": int(label),
                "pred_label": int(pred[batch_idx]),
                "pred_class": dataset.classes[int(pred[batch_idx])] if int(pred[batch_idx]) < len(dataset.classes) else str(pred[batch_idx]),
                "confidence_head": use_confidence,
                "confidence_mean": float(valid_confidence.mean().detach().cpu().item()) if valid_confidence.numel() else 0.0,
            }
        )
        counts[class_name] = export_idx + 1


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA wurde angefordert, ist aber nicht verfuegbar.")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    cfg = load_cfg(args)
    checkpoint = find_checkpoint(args)
    configure_output_root(args)
    set_random_seed(args.seed, deterministic=True)
    dataset = build_dataset(cfg, args.split)
    restrict_dataset_classes(dataset, list(cfg.get("raw_frames_classes", DEFAULT_CLASSES)))
    cfg.classes = list(dataset.classes)
    cfg.num_classes = dataset.num_classes
    cfg.model.cls_args.num_classes = dataset.num_classes

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    model = build_model_from_cfg(cfg.model).to(device)
    if checkpoint is not None:
        logging.info("Lade Checkpoint: %s", checkpoint)
        load_checkpoint(model, str(checkpoint), skip_shape_mismatch=args.skip_shape_mismatch)
    model.eval()

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "export_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "cfg": str(args.cfg),
                "dataset_root": str(args.dataset_root),
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "split": args.split,
                "seed": args.seed,
                "per_class": args.per_class,
                "confidence_head": bool(getattr(model, "enable_confidence_head", False)),
                "classes": list(dataset.classes),
            },
            handle,
            sort_keys=True,
        )

    counts = {class_name: 0 for class_name in dataset.classes}
    manifest = []
    target_total = args.per_class * len(dataset.classes)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader), total=len(loader), desc="Export fused clouds"):
            if sum(counts.values()) >= target_total:
                break
            data = prepare_batch(data, cfg, device)
            export_batch(model, data, dataset, batch_idx * args.batch_size, cfg, args, counts, manifest)

    manifest_path = args.output_root / ("manifest_confidence_on.json" if getattr(model, "enable_confidence_head", False) else "manifest_confidence_off.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"counts": counts, "samples": manifest}, handle, indent=2, sort_keys=True)
    logging.info("Fertig. Exportiert: %s", counts)
    logging.info("Manifest: %s", manifest_path)

    missing = {name: args.per_class - count for name, count in counts.items() if count < args.per_class}
    if missing:
        logging.warning("Nicht genug Samples fuer alle Klassen: %s", missing)


if __name__ == "__main__":
    main()
