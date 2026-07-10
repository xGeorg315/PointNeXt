#!/usr/bin/env python3
"""Run seed 7075 (ultra-light ICP) and export fused, input, and GT clouds.

The dataset may contain arbitrary nested directories. Raw frames and GT clouds
are discovered recursively from filenames and JSON metadata; matching does not
depend on fixed parent-directory positions.

For every selected object this script writes the fused model output, every
valid input view, the raw GT cloud, and the GT cloud normalized into the same
coordinate system as the model inputs.

Example:
  python script/infer_seed7075_ultra_light_icp.py --max-samples-per-class 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
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
from script.export_seed8240_fused_raw_frames import (  # noqa: E402
    as_bool,
    classification_augmentation_kwargs,
    configure_point_feature_channels,
    consolidate_observed_points,
    move_batch_to_device,
    safe_name,
    write_ascii_ply,
)


DEFAULT_SEED = 7075
DEFAULT_CFG = POINTNEXT_ROOT / "cfgs/modelnet40ply2048/ultra-light-icp.yaml"
DEFAULT_DATASET_ROOT = Path("/home/georg/workspace/data")
DEFAULT_CHECKPOINT_SEARCH_ROOT = POINTNEXT_ROOT / "log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inferiert Seed 7075 (Ultra-Light ICP) auf review-dataset-v3 und "
            "exportiert Fused-, Input- und GT-Clouds."
        )
    )
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-search-root", type=Path, default=DEFAULT_CHECKPOINT_SEARCH_ROOT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--split", default="all", choices=("all", "train", "val", "test"),
        help="Standard all inferiert alle Objekte; train/val/test reproduzieren den Hash-Split.",
    )
    parser.add_argument(
        "--max-samples-per-class", "--per-class", type=int, default=0,
        help="Maximale Objektzahl je Klasse; 0 exportiert alle verfuegbaren Samples.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--preload-data", action="store_true")
    parser.add_argument("--skip-shape-mismatch", action="store_true")
    parser.add_argument(
        "--strict-gt-matching", action="store_true",
        help="Bricht bei mehrdeutigen GT-Treffern ab, statt den bestbewerteten zu nehmen.",
    )
    return parser.parse_args()


def find_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint}")
        return checkpoint

    if args.run_dir is not None:
        search_dir = args.run_dir.expanduser().resolve() / "checkpoint"
        matches = sorted(search_dir.glob("*_ckpt_best.pth"))
    else:
        search_root = args.checkpoint_search_root.expanduser().resolve()
        matches = sorted(
            search_root.rglob(f"*ultra-light-icp*seed{args.seed}*/checkpoint/*_ckpt_best.pth")
        )

    if not matches:
        raise FileNotFoundError(
            f"Kein Ultra-Light-ICP-Best-Checkpoint fuer Seed {args.seed} gefunden. "
            "Bitte --checkpoint oder --run-dir angeben."
        )
    if len(matches) > 1:
        logging.warning("Mehrere passende Checkpoints gefunden; verwende den neuesten: %s", matches[-1])
    return matches[-1].resolve()


def load_cfg(args: argparse.Namespace) -> EasyConfig:
    args.cfg = args.cfg.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if not args.cfg.is_file():
        raise FileNotFoundError(f"Config nicht gefunden: {args.cfg}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Datensatz nicht gefunden: {args.dataset_root}")

    cfg = EasyConfig()
    cfg.load(str(args.cfg), recursive=True)
    cfg.seed = int(args.seed)
    cfg.rank = 0
    cfg.world_size = 1
    cfg.distributed = False
    cfg.mp = False
    cfg.sync_bn = False
    cfg.raw_frames_root = str(args.dataset_root)
    cfg.custom_dataset_root = str(args.dataset_root)
    cfg.augment_train = False
    cfg.preload_data = bool(args.preload_data)
    cfg.raw_frames_object_multi_view = True
    cfg.raw_frames_frame_selection = cfg.get("raw_frames_frame_selection", "all")
    cfg.raw_frames_max_views = int(
        cfg.get("raw_frames_max_views", cfg.model.get("max_views", 5))
    )
    cfg.raw_frames_view_selection = cfg.get("raw_frames_view_selection", "uniform")
    cfg.model.max_views = cfg.raw_frames_max_views
    args.batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else cfg.get("val_batch_size", cfg.get("batch_size", 16))
    )
    configure_point_feature_channels(cfg)
    return cfg



def iter_files_recursive(root: Path, suffix: str):
    """Yield files at arbitrary depth while ignoring links and tool/output dirs."""
    ignored = {".git", "__pycache__", "exports", "wandb", "checkpoint", "checkpoints"}
    for current_root, dir_names, file_names in os.walk(root, followlinks=False):
        dir_names[:] = sorted(name for name in dir_names if name not in ignored)
        for name in sorted(file_names):
            if Path(name).suffix.lower() == suffix:
                yield Path(current_root) / name


def metadata_class_name(metadata: dict, path: Path, known_classes: list[str]) -> str | None:
    review = metadata.get("review", {}) if isinstance(metadata, dict) else {}
    values = [
        review.get("final_class_name") if isinstance(review, dict) else None,
        metadata.get("final_class_name"),
        metadata.get("class_name"),
        metadata.get("gt_class_name"),
        metadata.get("obj_class"),
    ]
    known = {str(name).casefold(): str(name) for name in known_classes}
    for value in values:
        matched = known.get(str(value).casefold()) if value not in {None, ""} else None
        if matched:
            return matched
    for part in reversed(path.parts):
        matched = known.get(part.casefold())
        if matched:
            return matched
    return None


def sample_stem_from_path(path: Path, metadata: dict) -> str:
    sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
    if sample_id:
        return str(sample_id)
    return re.sub(r"__frame_[^_]+$", "", path.stem)


class RecursiveRawFramesDataset(RawFramesClassificationDataset):
    """RawFrames dataset whose discovery is independent of directory layout."""

    def _collect_classification_samples(self):
        self.root_dir = self.raw_frames_root_dir
        known_classes = list(self.raw_frames_forced_classes or [])
        records = []
        skipped = Counter()

        for class_name in known_classes:
            if class_name not in self.raw_frames_exclude_classes:
                self.class_to_idx[class_name] = len(self.classes)
                self.classes.append(class_name)

        for pcd_file in iter_files_recursive(self.raw_frames_root_dir, ".pcd"):
            frame_match = re.search(r"__frame_([^_]+)$", pcd_file.stem, flags=re.IGNORECASE)
            metadata = self._load_metadata(pcd_file)
            metadata_is_frame = bool(
                isinstance(metadata, dict)
                and metadata.get("frame_index") is not None
                and (metadata.get("source_frame_index") is not None or metadata.get("selected_frame_ids"))
                and not metadata.get("final_frame_poses")
            )
            if frame_match is None and not metadata_is_frame:
                continue

            class_name = metadata_class_name(metadata, pcd_file, known_classes)
            if class_name is None or class_name in self.raw_frames_exclude_classes:
                skipped["unknown_or_excluded_class"] += 1
                continue
            label = self.class_to_idx.get(class_name)
            if label is None:
                skipped["class_not_in_checkpoint"] += 1
                continue

            day = metadata.get("day") if isinstance(metadata, dict) else None
            if day and self.raw_frames_start_date is not None:
                try:
                    from datetime import datetime
                    if datetime.strptime(str(day), "%Y-%m-%d").date() < self.raw_frames_start_date:
                        continue
                except ValueError:
                    skipped["invalid_day"] += 1

            point_count = metadata.get("point_count") if isinstance(metadata, dict) else None
            if point_count is None:
                from examples.classification.dataloader import _pcd_point_count
                point_count = _pcd_point_count(pcd_file)
            point_count = int(point_count)
            if (
                self.raw_frames_min_points > 0
                and class_name not in self.raw_frames_min_points_exempt_classes
                and point_count < self.raw_frames_min_points
            ):
                continue

            stem = sample_stem_from_path(pcd_file, metadata)
            object_id = str(
                metadata.get("gt_object_id")
                or metadata.get("object_id")
                or self._parse_object_id(pcd_file)
            )
            frame_id = str(
                metadata.get("frame_index")
                or metadata.get("source_frame_index")
                or (frame_match.group(1) if frame_match else self._parse_frame_id(pcd_file))
            )
            records.append(
                {
                    "file": pcd_file,
                    "label": label,
                    "class_name": class_name,
                    "object_id": object_id,
                    "sample_id": str(metadata.get("sample_id") or stem),
                    "group_id": stem,
                    "frame_id": frame_id,
                    "run_id": metadata.get("run_id"),
                    "track_id": metadata.get("track_id") or metadata.get("source_sequence_index"),
                    "dist": self._metadata_distance(pcd_file, metadata=metadata),
                    "extent": self._metadata_extent(pcd_file, metadata=metadata),
                    "point_count": point_count,
                }
            )

        if not records:
            raise RuntimeError(
                f"Keine Raw-Frame-PCDs unter {self.raw_frames_root_dir} gefunden. "
                "Erwartet werden '__frame_<id>.pcd' oder passende Frame-Metadaten."
            )
        records = self._select_raw_frame_records(records)
        split_by_key = self._stratified_split_map(records)
        records = [
            record for record in records
            if split_by_key[self._record_key(record)] == self.split
        ]
        self.samples = self._group_object_views(records)
        self._min_sensor_distance = min((record["dist"] for record in self.samples), default=0.0)
        self.discovery_stats = {
            "raw_frame_records": len(records),
            "grouped_samples": len(self.samples),
            "skipped": dict(skipped),
        }

def build_dataset(cfg: EasyConfig, split: str) -> RawFramesClassificationDataset:
    exclude_classes = list(
        cfg.get("raw_frames_exclude_classes", cfg.get("exclude_classes", ("reject",)))
    )
    kwargs = classification_augmentation_kwargs(cfg)
    kwargs["preload_data"] = as_bool(cfg.get("preload_data", False))
    return RecursiveRawFramesDataset(
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
        **kwargs,
    )


def aggregate_stem(sample: dict) -> str:
    group_id = sample.get("group_id")
    if group_id:
        return str(group_id)
    return re.sub(r"__frame_[^_]+$", "", Path(sample["file"]).stem)



def load_json_safely(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def normalized_tokens(path: Path) -> set[str]:
    tokens = set()
    for part in path.parts:
        tokens.update(token for token in re.split(r"[^a-z0-9]+", part.casefold()) if token)
    return tokens


def gt_candidate_score(path: Path, metadata: dict) -> int:
    tokens = normalized_tokens(path.parent)
    score = 0
    if "gt" in tokens or "groundtruth" in tokens or {"ground", "truth"} <= tokens:
        score += 120
    if "pred" in tokens or "prediction" in tokens:
        score -= 100
    declared_path = str(metadata.get("pcd_path", "")).replace("\\", "/").casefold()
    if "/gt/" in f"/{declared_path.strip('/')}/":
        score += 80
    role = str(metadata.get("role") or metadata.get("type") or metadata.get("cloud_type") or "").casefold()
    if role in {"gt", "ground_truth", "ground-truth", "target"}:
        score += 100
    if isinstance(metadata.get("review"), dict):
        score += 10
    return score


def build_gt_index(dataset_root: Path, known_classes: list[str]) -> tuple[dict, dict]:
    """Recursively index non-frame PCDs without assuming any directory layout."""
    index = {
        "stem": defaultdict(list),
        "run_object": defaultdict(list),
        "object": defaultdict(list),
    }
    stats = Counter()
    for path in iter_files_recursive(dataset_root, ".pcd"):
        metadata = load_json_safely(path.with_suffix(".json"))
        if re.search(r"__frame_[^_]+$", path.stem, flags=re.IGNORECASE):
            stats["raw_frames_ignored"] += 1
            continue
        if metadata.get("final_frame_poses"):
            stats["pose_clouds_ignored"] += 1
            continue

        sample_id = str(metadata.get("sample_id") or path.stem)
        object_id = metadata.get("gt_object_id") or metadata.get("object_id")
        if object_id in {None, ""}:
            match = re.search(r"__object_([^_]+)", sample_id)
            object_id = match.group(1) if match else None
        run_id = metadata.get("run_id")
        if not run_id:
            match = re.match(r"run_(.+?)__track_", sample_id)
            run_id = match.group(1) if match else None
        class_name = metadata_class_name(metadata, path, known_classes)
        candidate = {
            "path": path,
            "class_name": class_name,
            "sample_id": sample_id,
            "run_id": str(run_id) if run_id not in {None, ""} else None,
            "object_id": str(object_id) if object_id not in {None, ""} else None,
            "score": gt_candidate_score(path, metadata),
        }
        for key in {path.stem, sample_id}:
            index["stem"][key].append(candidate)
        if candidate["run_id"] and candidate["object_id"]:
            index["run_object"][(candidate["run_id"], candidate["object_id"])].append(candidate)
        if candidate["object_id"]:
            index["object"][candidate["object_id"]].append(candidate)
        stats["non_frame_candidates"] += 1
    return index, dict(stats)


def resolve_gt(sample: dict, gt_index: dict, strict: bool) -> tuple[Path | None, str | None, bool]:
    class_name = str(sample["class_name"])
    stem = aggregate_stem(sample)
    candidates = list(gt_index["stem"].get(stem, ()))
    strategy = "exact_stem"

    if not candidates and sample.get("run_id") and sample.get("object_id"):
        candidates = list(
            gt_index["run_object"].get((str(sample["run_id"]), str(sample["object_id"])), ())
        )
        strategy = "run_id+object_id"
    if not candidates and sample.get("object_id"):
        object_candidates = list(gt_index["object"].get(str(sample["object_id"]), ()))
        class_candidates = [item for item in object_candidates if item["class_name"] == class_name]
        if len(class_candidates) == 1:
            candidates = class_candidates
            strategy = "unique_class+object_id"

    if not candidates:
        return None, None, False
    matching_class = [item for item in candidates if item["class_name"] == class_name]
    if matching_class:
        candidates = matching_class
    ranked = sorted(
        candidates,
        key=lambda item: (item["score"] + (40 if item["class_name"] == class_name else 0), str(item["path"])),
        reverse=True,
    )
    best_score = ranked[0]["score"] + (40 if ranked[0]["class_name"] == class_name else 0)
    tied = [
        item for item in ranked
        if item["score"] + (40 if item["class_name"] == class_name else 0) == best_score
    ]
    ambiguous = len(tied) > 1
    if ambiguous and strict:
        choices = "\n  ".join(str(item["path"]) for item in tied[:10])
        raise RuntimeError(f"Mehrdeutiges GT-Matching fuer {stem}:\n  {choices}")
    return ranked[0]["path"], strategy, ambiguous


def attach_gt_and_limit(
    dataset: RawFramesClassificationDataset,
    gt_index: dict,
    max_per_class: int,
    strict: bool,
) -> tuple[Counter, Counter, Counter]:
    available = Counter()
    missing = Counter()
    matching = Counter()
    selected_counts = Counter()
    selected = []
    for sample in dataset.samples:
        class_name = str(sample["class_name"])
        gt_path, strategy, ambiguous = resolve_gt(sample, gt_index, strict)
        if gt_path is None:
            missing[class_name] += 1
            continue
        available[class_name] += 1
        matching[f"strategy:{strategy}"] += 1
        if ambiguous:
            matching["ambiguous_best_score"] += 1
        if max_per_class > 0 and selected_counts[class_name] >= max_per_class:
            continue
        sample = dict(sample)
        sample["gt_path"] = gt_path
        sample["gt_match_strategy"] = strategy
        sample["gt_match_ambiguous"] = ambiguous
        selected.append(sample)
        selected_counts[class_name] += 1
    dataset.samples = selected
    return available, missing, matching

def prepare_batch(data: dict, device: torch.device) -> dict:
    data = move_batch_to_device(data, device)
    data["views"] = data["views"].contiguous()
    data["pos"] = data["views"][:, 0, :, :3].contiguous()
    data["x"] = data["pos"].transpose(1, 2).contiguous()
    return data


def normalized_gt(
    dataset: RawFramesClassificationDataset, sample: dict, raw_gt: np.ndarray
) -> np.ndarray:
    """Apply the exact shared normalization used by pose-supervised input views."""
    raw_views = [dataset._load_points(record["file"]) for record in sample["view_records"]]
    if not raw_views or not dataset.normalize:
        return raw_gt.astype(np.float32, copy=False)
    centroid = np.mean(raw_views[0], axis=0)
    centered_views = [points - centroid for points in raw_views]
    radius = dataset._unit_radius(np.concatenate(centered_views, axis=0))
    return dataset._normalize_with_centroid(raw_gt, centroid, radius).astype(
        np.float32, copy=False
    )


def export_one(
    model: torch.nn.Module,
    data: dict,
    batch_idx: int,
    sample_idx: int,
    sample: dict,
    dataset: RawFramesClassificationDataset,
    output_root: Path,
    cfg: EasyConfig,
    prediction: int,
) -> dict:
    output = model.last_output
    transformed = output["transformed_points"][batch_idx].detach()
    confidence = output["point_confidence"][batch_idx].detach()
    view_mask = output["view_mask"][batch_idx].detach().bool()
    point_mask = output.get("point_mask")
    selected = (
        point_mask[batch_idx].detach().bool()[view_mask].reshape(-1)
        if point_mask is not None
        else torch.ones_like(confidence[view_mask].reshape(-1), dtype=torch.bool)
    )
    points = transformed[view_mask].reshape(-1, 3)
    point_confidence = confidence[view_mask].reshape(-1)
    voxel_size = float(cfg.model.get("fusion_voxel_size", cfg.get("fusion_voxel_size", 0.0)))
    fused = consolidate_observed_points(
        points[selected], point_confidence[selected], voxel_size, 0.0
    )

    class_name = str(sample["class_name"])
    stem = aggregate_stem(sample)
    sample_dir = output_root / safe_name(class_name) / safe_name(stem)
    fused_path = sample_dir / "fused.ply"
    write_ascii_ply(fused_path, fused)

    input_paths = []
    input_views = data["views"][batch_idx].detach()
    for input_idx, view_idx in enumerate(torch.nonzero(view_mask, as_tuple=False).flatten().tolist()):
        input_path = sample_dir / "inputs" / f"input_{input_idx:02d}.ply"
        write_ascii_ply(input_path, input_views[view_idx, :, :3])
        input_paths.append(str(input_path))

    raw_gt = dataset._load_xyz(Path(sample["gt_path"]))[:, :3].astype(np.float32, copy=False)
    gt_normalized = normalized_gt(dataset, sample, raw_gt)
    gt_path = sample_dir / "gt.ply"
    gt_raw_path = sample_dir / "gt_raw.ply"
    write_ascii_ply(gt_path, torch.from_numpy(gt_normalized))
    write_ascii_ply(gt_raw_path, torch.from_numpy(raw_gt))

    true_label = int(sample["label"])
    metadata = {
        "sample_index": sample_idx,
        "class_name": class_name,
        "object_id": sample.get("object_id"),
        "sample_id": sample.get("sample_id"),
        "aggregate_stem": stem,
        "true_label": true_label,
        "pred_label": int(prediction),
        "pred_class": dataset.classes[prediction],
        "correct": int(prediction) == true_label,
        "num_views": int(view_mask.sum().item()),
        "num_fused_points": int(fused.shape[0]),
        "num_gt_points": int(raw_gt.shape[0]),
        "fused_path": str(fused_path),
        "input_paths": input_paths,
        "gt_path": str(gt_path),
        "gt_raw_path": str(gt_raw_path),
        "source_gt_path": str(sample["gt_path"]),
        "gt_match_strategy": sample.get("gt_match_strategy"),
        "gt_match_ambiguous": bool(sample.get("gt_match_ambiguous", False)),
        "source_input_paths": [str(record["file"]) for record in sample["view_records"]],
    }
    sample_dir.mkdir(parents=True, exist_ok=True)
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.max_samples_per_class < 0:
        raise ValueError("--max-samples-per-class muss >= 0 sein")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA wurde angefordert, ist aber nicht verfuegbar.")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    cfg = load_cfg(args)
    checkpoint = find_checkpoint(args)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else POINTNEXT_ROOT / "exports/seed7075_ultra_light_icp_review_dataset_v3"
    )
    set_random_seed(args.seed, deterministic=True)

    logging.info("Erzeuge Dataset-Index (%s split) ...", args.split)
    dataset = build_dataset(cfg, args.split)
    gt_index, gt_search_stats = build_gt_index(args.dataset_root, list(dataset.classes))
    logging.info("Rekursive Suche: %s", getattr(dataset, "discovery_stats", {}))
    logging.info("GT-Kandidaten: %s", gt_search_stats)
    available, missing_gt, gt_matching_stats = attach_gt_and_limit(
        dataset, gt_index, args.max_samples_per_class, args.strict_gt_matching
    )
    logging.info("GT-Matching: %s", dict(gt_matching_stats))
    if not dataset.samples:
        raise RuntimeError("Keine Raw-Frame-Samples mit zugehoeriger GT-Cloud gefunden.")

    cfg.classes = list(dataset.classes)
    cfg.num_classes = len(cfg.classes)
    cfg.model.cls_args.num_classes = cfg.num_classes
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_model_from_cfg(cfg.model).to(device)
    logging.info("Lade Checkpoint: %s", checkpoint)
    load_checkpoint(model, str(checkpoint), skip_shape_mismatch=args.skip_shape_mismatch)
    model.eval()

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "export_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "cfg": str(args.cfg),
                "checkpoint": str(checkpoint),
                "dataset_root": str(args.dataset_root),
                "raw_frames_root": str(cfg.raw_frames_root),
                "output_root": str(output_root),
                "seed": args.seed,
                "split": args.split,
                "max_samples_per_class": args.max_samples_per_class,
                "available_with_gt_per_class": dict(available),
                "missing_gt_per_class": dict(missing_gt),
                "raw_discovery": getattr(dataset, "discovery_stats", {}),
                "gt_search": gt_search_stats,
                "gt_matching": dict(gt_matching_stats),
                "selected_per_class": dict(Counter(s["class_name"] for s in dataset.samples)),
            },
            handle,
            sort_keys=True,
        )

    manifest = []
    counts = Counter()
    with torch.inference_mode():
        for batch_number, data in tqdm(
            enumerate(loader), total=len(loader), desc="Seed 7075 inference"
        ):
            batch_start = batch_number * args.batch_size
            data = prepare_batch(data, device)
            logits = model(data)
            predictions = logits.argmax(dim=1).detach().cpu().tolist()
            for batch_idx, prediction in enumerate(predictions):
                sample_idx = batch_start + batch_idx
                sample = dataset.samples[sample_idx]
                record = export_one(
                    model, data, batch_idx, sample_idx, sample, dataset,
                    output_root, cfg, int(prediction),
                )
                manifest.append(record)
                counts[record["class_name"]] += 1

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": args.seed,
                "checkpoint": str(checkpoint),
                "counts": dict(counts),
                "samples": manifest,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    logging.info("Fertig. Exportiert pro Klasse: %s", dict(counts))
    logging.info("Ausgabe: %s", output_root)
    logging.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
