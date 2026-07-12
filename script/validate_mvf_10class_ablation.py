#!/usr/bin/env python3
"""Validate every available MVF 10-class ablation checkpoint.

The model is loaded once and evaluated on (1) the exact training test split and
(2) all labelled PCDs below the top-down root.  Data loading and metric/export
work are excluded from the reported network latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import resource
import sys
import time
from pathlib import Path

# Set these before importing numpy/torch: one host CPU thread, including when
# inference itself runs on CUDA.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.classification.dataloader import RawFramesClassificationDataset
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, load_checkpoint, set_random_seed

DEFAULT_CFG_DIR = ROOT / "cfgs/modelnet40ply2048/mvf_10class_ablation"
DEFAULT_LOG_ROOT = ROOT / "log/mvf_10class_ablation"
DEFAULT_TOP_DOWN = Path("/home/georg/workspace/top-down")
DEFAULT_ICP_AGGREGATES = DEFAULT_TOP_DOWN / "top-down-aggregates"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg-dir", type=Path, default=DEFAULT_CFG_DIR)
    p.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    p.add_argument("--top-down-root", type=Path, default=DEFAULT_TOP_DOWN)
    p.add_argument(
        "--icp-aggregates-root", type=Path, default=DEFAULT_ICP_AGGREGATES,
        help="Root containing Config-13 ICP aggregates in gt-pred-same/ and gt-pred-diff/.")
    p.add_argument("--output-root", type=Path, default=ROOT / "validation/mvf_10class_ablation")
    p.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="Inference device. Use --device cpu for explicit CPU inference.")
    p.add_argument(
        "--top-down-rotation-deg-z", type=float, default=-121.0,
        help="Rotate top-down samples around Z before inference (default: 121.0 degrees).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    p.add_argument("--configs", nargs="*", help="Optional config stems or numeric prefixes, e.g. 01 09.")
    p.add_argument("--warmup-batches", type=int, default=3)
    p.add_argument("--tracking-icp-repetitions", type=int, default=1)
    p.add_argument("--tracking-icp-warmup", type=int, default=1)
    p.add_argument(
        "--benchmark-repetitions", type=int, default=1,
        help="Timed model forwards per non-warm-up batch (default: 1).")
    p.add_argument(
        "--registration-rmse-max-correspondence", type=float, default=0.05,
        help="Maximum nearest-neighbour distance for fusion RMSE in normalized model coordinates; <= 0 disables filtering.")
    p.add_argument("--fair-max-views", type=int, default=5,
                   help="Views per object for fair single-view late-fusion evaluation.")
    p.add_argument("--no-fair-single-view-eval", dest="fair_single_view_eval",
                   action="store_false", help="Disable additional object-level single-view comparison.")
    p.set_defaults(fair_single_view_eval=True)
    p.add_argument("--no-cloud-export", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_cfg(path: Path) -> EasyConfig:
    cfg = EasyConfig()
    cfg.load(str(path), recursive=True)
    return cfg


def dataset_kwargs(cfg: EasyConfig, root: Path, split: str, top_down: bool) -> dict:
    # augment_train is harmless for test/all, but explicitly disabling it makes
    # the evaluation contract clear and avoids future dataset changes leaking in.
    exclude_classes = tuple(cfg.get("raw_frames_exclude_classes", ("reject",)))
    if top_down:
        # This is an export directory for Config 13, not a semantic class.
        exclude_classes = tuple(dict.fromkeys((*exclude_classes, "top-down-aggregates")))
    return dict(
        root_dir=str(root), split=split, num_points=int(cfg.num_points),
        start_date=None if top_down else cfg.get("raw_frames_start_date", None),
        min_points=0 if top_down else cfg.get("raw_frames_min_points", 0),
        split_ratios=cfg.get("raw_frames_split_ratios", (0.8, 0.1, 0.1)),
        exclude_classes=exclude_classes,
        min_points_exempt_classes=cfg.get("raw_frames_min_points_exempt_classes", ()),
        forced_classes=cfg.get("raw_frames_classes", None),
        frame_selection=cfg.get("raw_frames_frame_selection", "all"),
        object_multi_view=as_bool(cfg.get("raw_frames_object_multi_view", False)),
        max_views=int(cfg.get("raw_frames_max_views", cfg.model.get("max_views", 1))),
        view_selection=cfg.get("raw_frames_view_selection", "uniform"),
        pose_metadata_root=None if top_down else cfg.get("raw_frames_pose_root", None),
        pose_required=False if top_down else as_bool(cfg.get("raw_frames_pose_required", False)),
        shared_view_normalization=as_bool(
            cfg.get("raw_frames_shared_view_normalization", False)),
        return_shared_geometry_views=as_bool(
            cfg.get("raw_frames_return_shared_geometry_views", False)),
        augment_train=False, use_normals=as_bool(cfg.get("use_normals", False)),
        normal_k=int(cfg.get("normal_k", 16)), preload_data=as_bool(cfg.get("preload_data", False)),
        obj_features_include_sensor_dist=as_bool(cfg.get("obj_features_include_sensor_dist", True)),
    )


class AllTopDownDataset(RawFramesClassificationDataset):
    """Use every top-down sample instead of applying another 80/10/10 split."""
    def _stratified_split_map(self, records):
        return {self._record_key(record): self.split for record in records}


def build_datasets(
        cfg: EasyConfig, *, include_top_down: bool = True
) -> tuple[RawFramesClassificationDataset, RawFramesClassificationDataset | None]:
    dataset_format = str(cfg.get("classification_dataset_format", "raw_frames")).lower()
    if dataset_format == "review":
        # Config 13 stores ICP aggregates in class/pcds/*.pcd.  This is the
        # raw-frame loader's supported aggregate layout (the review loader
        # expects class/day/bucket/source and would return an empty dataset).
        aggregate_root = Path(cfg.get("review_dataset_root", cfg.get("custom_dataset_root")))
        kwargs = dataset_kwargs(cfg, aggregate_root, "test", top_down=False)
        kwargs.update(
            split_ratios=cfg.get("review_split_ratios", kwargs["split_ratios"]),
            min_points=cfg.get("review_min_points", 0), object_multi_view=False,
            max_views=1, pose_metadata_root=None, pose_required=False)
        train_test = RawFramesClassificationDataset(**kwargs)
    else:
        train_test = RawFramesClassificationDataset(**dataset_kwargs(
            cfg, Path(cfg.raw_frames_root), "test", top_down=False
        ))
    if not include_top_down:
        return train_test, None
    top_down = AllTopDownDataset(**dataset_kwargs(
        cfg, ARGS.top_down_root, "all", top_down=True
    ))
    expected = list(train_test.classes)
    if set(top_down.classes) != set(expected):
        raise RuntimeError(f"Class mismatch: training={expected}, top_down={top_down.classes}")
    class_to_idx = {name: idx for idx, name in enumerate(expected)}
    for sample in top_down.samples:
        sample["label"] = class_to_idx[sample["class_name"]]
    top_down.classes, top_down.class_to_idx, top_down.num_classes = expected, class_to_idx, len(expected)
    return train_test, top_down


def build_icp_aggregate_top_down_datasets(
        cfg: EasyConfig, classes: list[str]) -> dict[str, RawFramesClassificationDataset]:
    """Load Config 13's ICP aggregates separately for same/diff predictions."""
    datasets = {}
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    for bucket in ("gt-pred-same", "gt-pred-diff"):
        root = ARGS.icp_aggregates_root / bucket
        if not root.is_dir():
            raise FileNotFoundError(f"Missing ICP aggregate directory: {root}")
        dataset = AllTopDownDataset(**dataset_kwargs(cfg, root, "all", top_down=True))
        observed = {sample["class_name"] for sample in dataset.samples}
        unknown = observed.difference(class_to_idx)
        if unknown:
            raise RuntimeError(
                f"Unexpected classes in {bucket}: {sorted(unknown)}; expected {classes}")
        # A bucket can legitimately omit classes. Keep the training class order
        # so its confusion matrix remains directly comparable to Config 13.
        for sample in dataset.samples:
            sample["label"] = class_to_idx[sample["class_name"]]
        dataset.classes = list(classes)
        dataset.class_to_idx = class_to_idx
        dataset.num_classes = len(classes)
        datasets[bucket] = dataset
    return datasets


def remap_dataset_classes(dataset, classes: list[str]) -> None:
    if set(dataset.classes) != set(classes):
        raise RuntimeError(f"Class mismatch: expected={classes}, actual={dataset.classes}")
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    for sample in dataset.samples:
        sample["label"] = class_to_idx[sample["class_name"]]
    dataset.classes = list(classes)
    dataset.class_to_idx = class_to_idx
    dataset.num_classes = len(classes)


def build_fair_single_view_object_datasets(cfg: EasyConfig, classes: list[str]):
    """Group the same objects as MVF while retaining per-view normalization."""
    common = dict(
        object_multi_view=True, max_views=int(ARGS.fair_max_views),
        pose_metadata_root=None, pose_required=False,
        shared_view_normalization=False,
    )
    train_kwargs = dataset_kwargs(cfg, Path(cfg.raw_frames_root), "test", top_down=False)
    train_kwargs.update(common)
    top_kwargs = dataset_kwargs(cfg, ARGS.top_down_root, "all", top_down=True)
    top_kwargs.update(common)
    train_test = RawFramesClassificationDataset(**train_kwargs)
    top_down = AllTopDownDataset(**top_kwargs)
    remap_dataset_classes(train_test, classes)
    remap_dataset_classes(top_down, classes)
    return train_test, top_down


def find_checkpoint(cfg_path: Path, cfg: EasyConfig) -> Path | None:
    experiment = str(cfg.get("experiment_name", cfg_path.stem))
    # Training-run names can use either experiment_name or the YAML stem.
    # Shared-encoder configs use a longer internal experiment_name while their
    # run directories retain the shorter YAML stem.
    run_name_tokens = {experiment, cfg_path.stem}
    candidates = []
    for run_dir in ARGS.log_root.iterdir() if ARGS.log_root.is_dir() else ():
        if run_dir.is_dir() and any(token in run_dir.name for token in run_name_tokens):
            candidates.extend((run_dir / "checkpoint").glob(f"*_ckpt_{ARGS.checkpoint}.pth"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def prepare(data: dict, device: torch.device) -> dict:
    for key, value in list(data.items()):
        if torch.is_tensor(value):
            data[key] = value.to(device, non_blocking=device.type == "cuda")
    if "views" in data:
        data["views"] = data["views"].contiguous()
        data["pos"] = data["views"][:, 0, :, :3].contiguous()
    data["x"] = data["pos"].transpose(1, 2).contiguous()
    return data


def rotate_batch_z(data: dict, rotation_deg: float) -> dict:
    """Rotate all XYZ coordinates counter-clockwise around +Z."""
    radians = torch.as_tensor(
        np.deg2rad(rotation_deg), device=data["pos"].device, dtype=data["pos"].dtype)
    cosine, sine = torch.cos(radians), torch.sin(radians)
    rotation = torch.stack((
        torch.stack((cosine, -sine, torch.zeros_like(cosine))),
        torch.stack((sine, cosine, torch.zeros_like(cosine))),
        torch.stack((torch.zeros_like(cosine), torch.zeros_like(cosine), torch.ones_like(cosine))),
    ))
    if "views" in data:
        data["views"][..., :3] = torch.matmul(data["views"][..., :3], rotation.T)
        if "geometry_views" in data:
            data["geometry_views"][..., :3] = torch.matmul(
                data["geometry_views"][..., :3], rotation.T
            )
        data["pos"] = data["views"][:, 0, :, :3].contiguous()
    else:
        data["pos"] = torch.matmul(data["pos"], rotation.T).contiguous()
    data["x"] = data["pos"].transpose(1, 2).contiguous()
    return data


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validation_progress(loader: DataLoader, description: str):
    """Render progress visibly in terminals, IDE consoles, and captured logs."""
    return tqdm(
        loader,
        desc=description,
        unit="batch",
        total=len(loader),
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
    )


def measure_tracking_icp(dataset, indices: list[int]) -> tuple[list[float], int, str | None]:
    """Time the tracking ICP/voxel-fusion core on raw-frame object groups."""
    samples = getattr(dataset, "samples", [])
    if not indices or not all(index < len(samples) and samples[index].get("view_records") for index in indices):
        return [], 0, "dataset has no raw-frame multi-view groups"
    pipeline_src = Path("/home/georg/workspace/minimal-tracking-pipeline/src")
    if str(pipeline_src) not in sys.path:
        sys.path.insert(0, str(pipeline_src))
    try:
        from tracking_pipeline.aggregate_raw_frames import _build_accumulator, _build_track, _lane_box
    except Exception as exc:
        return [], 0, f"tracking ICP import failed: {exc!r}"
    timings, failed = [], 0
    for index in indices:
        sample = samples[index]
        try:
            paths = [record["file"] for record in sample["view_records"]]
            track, _ = _build_track(str(sample.get("group_id", sample["sample_id"])), paths, {})
            operation = lambda: _build_accumulator().accumulate(track, _lane_box())
            for _ in range(ARGS.tracking_icp_warmup):
                operation()
            for _ in range(ARGS.tracking_icp_repetitions):
                started = time.perf_counter()
                operation()
                timings.append(time.perf_counter() - started)
        except Exception:
            failed += 1
    return timings, failed, None


def process_memory_peak_bytes() -> int:
    """Linux process high-water RSS; it includes the whole evaluation process."""
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def benchmark_environment(device: torch.device) -> dict:
    gpu = None
    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu = {"index": index, "name": props.name,
               "total_vram_bytes": int(props.total_memory),
               "compute_capability": f"{props.major}.{props.minor}"}
    return {
        "hardware": {"cpu": platform.processor() or platform.uname().processor or "unknown",
                     "logical_cpu_count": os.cpu_count(), "gpu": gpu},
        "software": {"platform": platform.platform(), "python": sys.version,
                     "numpy": np.__version__, "torch": torch.__version__,
                     "cuda_runtime": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None},
        "threading": {"torch_num_threads": torch.get_num_threads(),
                      "torch_num_interop_threads": torch.get_num_interop_threads(),
                      "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                      "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                      "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                      "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS")},
    }


def begin_memory_measurement(device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return process_memory_peak_bytes()


def latency_statistics(samples_seconds: list[float]) -> dict:
    values_ms = np.asarray(samples_seconds, dtype=np.float64) * 1000.0
    if not len(values_ms):
        return {"median_ms": None, "p95_ms": None, "stddev_ms": None}
    return {"median_ms": float(np.median(values_ms)),
            "p95_ms": float(np.percentile(values_ms, 95)),
            "stddev_ms": float(values_ms.std(ddof=0))}


def make_latency_report(*, scope: str, samples_seconds: list[float], warmup_batches: int,
                        measured_batches: int, device: torch.device, ram_before: int, extra: dict) -> dict:
    memory = {"process_peak_rss_bytes": process_memory_peak_bytes(),
              "process_peak_rss_before_evaluation_bytes": ram_before,
              "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
    if device.type == "cuda":
        memory["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        memory["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return {"scope": scope, "device": str(device), "cpu_threads": torch.get_num_threads(),
            "warmup_batches_excluded": warmup_batches,
            "repetitions_per_batch": ARGS.benchmark_repetitions,
            "measured_batches": measured_batches, "timed_forward_passes": len(samples_seconds),
            "included_steps": ["prepared input already resident on selected device",
                               "CUDA synchronization before timing", "model(data) forward pass",
                               "CUDA synchronization after forward pass"],
            "excluded_steps": ["dataset loading and collation", "host-to-device input preparation",
                               "metrics, predictions, confusion matrix, and cloud export"],
            "memory": memory, **latency_statistics(samples_seconds), **extra}


def write_benchmark_table(output_dir: Path, latency: dict) -> None:
    rows = [("environment_and_threading_table", "../../benchmark_environment.csv", "relative path"),
            ("timing_scope", latency["scope"], ""),
            ("included_steps", " | ".join(latency["included_steps"]), ""),
            ("excluded_steps", " | ".join(latency["excluded_steps"]), ""),
            ("warmup_batches", latency["warmup_batches_excluded"], "batches"),
            ("repetitions_per_batch", latency["repetitions_per_batch"], "forwards"),
            ("timed_forward_passes", latency["timed_forward_passes"], "forwards"),
            ("median", latency["median_ms"], "ms / forward"),
            ("p95", latency["p95_ms"], "ms / forward"),
            ("standard_deviation", latency["stddev_ms"], "ms / forward"),
            ("peak_ram", latency["memory"]["process_peak_rss_bytes"], "bytes (process HWM)"),
            ("peak_vram_allocated", latency["memory"]["cuda_peak_allocated_bytes"], "bytes"),
            ("peak_vram_reserved", latency["memory"]["cuda_peak_reserved_bytes"], "bytes")]
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "unit"])
        writer.writerows(rows)


def empty_registration_residual() -> dict:
    return {"squared_distance_sum": 0.0, "inlier_points": 0,
            "candidate_points": 0, "view_pairs": 0}


def add_pairwise_registration_residual(
        accumulator: dict, points: torch.Tensor, view_mask: torch.Tensor,
        point_mask: torch.Tensor | None, max_correspondence: float) -> None:
    """Accumulate symmetric nearest-neighbour residuals without running ICP."""
    batch_size, num_views, num_points, _ = points.shape
    if point_mask is None:
        point_mask = torch.ones((batch_size, num_views, num_points), dtype=torch.bool,
                                device=points.device)
    max_distance = float("inf") if max_correspondence <= 0 else max_correspondence
    for first in range(num_views - 1):
        for second in range(first + 1, num_views):
            pair_valid = view_mask[:, first].bool() & view_mask[:, second].bool()
            if not torch.any(pair_valid):
                continue
            a, b = points[pair_valid, first], points[pair_valid, second]
            a_mask = point_mask[pair_valid, first].bool()
            b_mask = point_mask[pair_valid, second].bool()
            distances = torch.cdist(a, b)
            a_to_b = distances.masked_fill(~b_mask[:, None, :], float("inf")).amin(dim=-1)
            b_to_a = distances.masked_fill(~a_mask[:, :, None], float("inf")).amin(dim=-2)
            for nearest, source_mask in ((a_to_b, a_mask), (b_to_a, b_mask)):
                candidates = source_mask & torch.isfinite(nearest)
                inliers = candidates & (nearest <= max_distance)
                accumulator["squared_distance_sum"] += float(nearest[inliers].square().sum().item())
                accumulator["inlier_points"] += int(inliers.sum().item())
                accumulator["candidate_points"] += int(candidates.sum().item())
            accumulator["view_pairs"] += int(pair_valid.sum().item())


def finalize_registration_residual(accumulator: dict) -> dict:
    inliers = accumulator["inlier_points"]
    candidates = accumulator["candidate_points"]
    return {
        "symmetric_nn_rmse": (float(np.sqrt(accumulator["squared_distance_sum"] / inliers))
                              if inliers else None),
        "inlier_ratio": float(inliers / candidates) if candidates else None,
        "inlier_points": inliers, "candidate_points": candidates,
        "view_pairs": accumulator["view_pairs"],
    }


def write_environment_table(output_root: Path, environment: dict) -> None:
    rows = [("hardware", "cpu", environment["hardware"]["cpu"]),
            ("hardware", "logical_cpu_count", environment["hardware"]["logical_cpu_count"]),
            ("hardware", "gpu", json.dumps(environment["hardware"]["gpu"], sort_keys=True)),
            *[("software", key, value) for key, value in environment["software"].items()],
            *[("threading", key, value) for key, value in environment["threading"].items()]]
    with (output_root / "benchmark_environment.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "field", "value"])
        writer.writerows(rows)


def confusion_metrics(cm: np.ndarray, classes: list[str]) -> dict:
    total = int(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    per_class = {
        name: {"support": int(support[i]), "accuracy_recall": float(recall[i]),
               "recall": float(recall[i]), "precision": float(precision[i]), "f1": float(f1[i])}
        for i, name in enumerate(classes)
    }
    return {
        "num_samples": total,
        "overall_accuracy": float(tp.sum() / total) if total else 0.0,
        "mean_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "f1_macro": float(f1[present].mean()) if present.any() else 0.0,
        "recall_macro": float(recall[present].mean()) if present.any() else 0.0,
        "per_class": per_class,
    }


def safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def write_ply(path: Path, points: torch.Tensor) -> None:
    xyz = points.detach().float().cpu().numpy().reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, xyz, fmt="%.7f %.7f %.7f")


def export_clouds(model: torch.nn.Module, data: dict, dataset, indices: list[int], out_dir: Path) -> None:
    output = getattr(model, "last_output", None)
    if not isinstance(output, dict) or "transformed_points" not in output:
        return  # single-view BaseCls has no fusion/fine stage
    view_mask = output["view_mask"].bool()
    point_mask = output.get("point_mask")
    stages = {"fused": output["transformed_points"]}
    geometry_model = getattr(model, "geometry_model", None)
    has_icp = bool(getattr(model, "enable_icp_refinement", False)) or bool(
        getattr(geometry_model, "enable_icp_refinement", False)
    )
    if has_icp and output.get("pre_icp_points") is not None:
        stages["pre_icp"] = output["pre_icp_points"]
        stages["post_icp"] = output["transformed_points"]
    elif bool(getattr(model, "enable_residual_correction_head", False)):
        stages["pre_fine"] = output["pre_residual_points"]
        stages["post_fine"] = output["transformed_points"]
    for b, sample_index in enumerate(indices):
        sample = dataset.samples[sample_index]
        stem = f"{sample_index:06d}_{safe(sample.get('object_id', sample.get('sample_id', sample_index)))}"
        class_dir = out_dir / safe(sample["class_name"])
        valid_views = view_mask[b]
        for stage_name, stage_points in stages.items():
            points = stage_points[b][valid_views]
            # The learned fusion mask belongs to the final output. Applying it
            # consistently also gives directly comparable pre/post-fine clouds.
            if point_mask is not None:
                points = points[point_mask[b][valid_views].bool()]
            write_ply(class_dir / f"{stem}_{stage_name}.ply", points)


def evaluate(model, dataset, cfg, device, output_dir: Path, *, rotation_deg_z: float = 0.0) -> dict:
    batch_size = int(ARGS.batch_size or cfg.get("val_batch_size", cfg.get("batch_size", 1)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda", drop_last=False)
    classes = list(dataset.classes)
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    network_seconds = 0.0
    forward_samples = []
    measured_batches = 0
    ram_peak_before = begin_memory_measurement(device)
    residual_pre = empty_registration_residual()
    residual_post = empty_registration_residual()
    tracking_icp_samples: list[float] = []
    tracking_icp_failed = 0
    tracking_icp_reason = None
    sample_offset = 0
    with torch.inference_mode():
        progress = validation_progress(loader, f"{output_dir.parent.name}/{output_dir.name}")
        for batch_idx, data in enumerate(progress):
            data = prepare(data, device)
            if rotation_deg_z:
                data = rotate_batch_z(data, rotation_deg_z)
            logits = None
            for _ in range(ARGS.benchmark_repetitions):
                synchronize(device)
                start = time.perf_counter()
                logits = model(data)
                synchronize(device)
                elapsed = time.perf_counter() - start
                if batch_idx >= ARGS.warmup_batches:
                    network_seconds += elapsed
                    forward_samples.append(elapsed)
            if batch_idx >= ARGS.warmup_batches:
                measured_batches += 1
            labels = data["y"].detach().cpu().numpy().astype(np.int64)
            predictions = logits.argmax(1).detach().cpu().numpy().astype(np.int64)
            np.add.at(cm, (labels, predictions), 1)
            output = getattr(model, "last_output", None)
            if (isinstance(output, dict) and output.get("pre_icp_points") is not None
                    and output.get("transformed_points") is not None):
                view_mask = output.get("view_mask")
                if view_mask is not None:
                    point_mask = output.get("point_mask")
                    add_pairwise_registration_residual(
                        residual_pre, output["pre_icp_points"], view_mask, point_mask,
                        ARGS.registration_rmse_max_correspondence)
                    add_pairwise_registration_residual(
                        residual_post, output["transformed_points"], view_mask, point_mask,
                        ARGS.registration_rmse_max_correspondence)
            indices = list(range(sample_offset, sample_offset + len(labels)))
            icp_samples, icp_failed, icp_reason = measure_tracking_icp(dataset, indices)
            tracking_icp_samples.extend(icp_samples)
            tracking_icp_failed += icp_failed
            tracking_icp_reason = tracking_icp_reason or icp_reason
            if not ARGS.no_cloud_export:
                export_clouds(model, data, dataset, indices, output_dir / "clouds")
            sample_offset += len(labels)
    metrics = confusion_metrics(cm, classes)
    measured_samples = max(0, len(dataset) - min(len(dataset), ARGS.warmup_batches * batch_size))
    metrics["latency"] = make_latency_report(
        scope="model_forward_only", samples_seconds=forward_samples,
        warmup_batches=min(ARGS.warmup_batches, len(loader)), measured_batches=measured_batches,
        device=device, ram_before=ram_peak_before,
        extra={"measured_samples": measured_samples, "total_seconds": network_seconds,
               "mean_batch_ms": 1000.0 * network_seconds / measured_batches if measured_batches else None,
               "mean_sample_ms": 1000.0 * network_seconds / measured_samples if measured_samples else None})
    metrics["input_rotation_deg_z"] = float(rotation_deg_z)
    if residual_pre["view_pairs"]:
        pre = finalize_registration_residual(residual_pre)
        post = finalize_registration_residual(residual_post)
        improvement = (pre["symmetric_nn_rmse"] - post["symmetric_nn_rmse"]
                       if pre["symmetric_nn_rmse"] is not None and post["symmetric_nn_rmse"] is not None
                       else None)
        metrics["fusion_registration_residual"] = {
            "available": True,
            "method": "symmetric nearest-neighbour RMSE; no additional ICP optimization",
            "coordinate_system": "normalized model coordinates (not millimetres)",
            "max_correspondence_distance": ARGS.registration_rmse_max_correspondence,
            "pre_icp": pre, "post_icp": post,
            "improvement": improvement,
            "improvement_percent": (100.0 * improvement / pre["symmetric_nn_rmse"]
                                    if improvement is not None and pre["symmetric_nn_rmse"] else None),
        }
    else:
        metrics["fusion_registration_residual"] = {
            "available": False,
            "reason": "Model output has no multi-view pre/post-ICP point clouds with at least two valid views.",
        }
    if tracking_icp_samples:
        metrics["tracking_icp_latency"] = {
            "available": True, "scope": "tracking ICP registration and voxel fusion only; prebuilt track",
            "device": "cpu", "repetitions_per_object": ARGS.tracking_icp_repetitions,
            "warmup_per_object": ARGS.tracking_icp_warmup, "failed_objects": tracking_icp_failed,
            "timed_runs": len(tracking_icp_samples), **latency_statistics(tracking_icp_samples),
        }
    else:
        metrics["tracking_icp_latency"] = {"available": False, "reason": tracking_icp_reason or "no eligible raw-frame objects"}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_table(output_dir, metrics["latency"])
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["true/pred", *classes])
        for name, row in zip(classes, cm): writer.writerow([name, *row.tolist()])
    return metrics


def evaluate_single_view_object_fusion(
        model, dataset, cfg, device, output_dir: Path, *, rotation_deg_z: float = 0.0) -> dict:
    """Run one trained single-view model per view and mean logits per object."""
    batch_size = int(ARGS.batch_size or cfg.get("val_batch_size", cfg.get("batch_size", 1)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda", drop_last=False)
    classes = list(dataset.classes)
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    network_seconds = 0.0
    forward_samples = []
    measured_batches = measured_objects = measured_views = 0
    ram_peak_before = begin_memory_measurement(device)
    with torch.inference_mode():
        progress = validation_progress(loader, f"{output_dir.parent.name}/{output_dir.name}")
        for batch_idx, data in enumerate(progress):
            data = prepare(data, device)
            if rotation_deg_z:
                data = rotate_batch_z(data, rotation_deg_z)
            valid = data["view_mask"].bool()
            owners = torch.nonzero(valid, as_tuple=False)[:, 0]
            views = data["views"][valid]
            view_data = {
                "pos": views[:, :, :3].contiguous(),
                "x": views.transpose(1, 2).contiguous(),
            }
            if data.get("obj_features") is not None:
                view_data["obj_features"] = data["obj_features"][owners]
            view_logits = None
            for _ in range(ARGS.benchmark_repetitions):
                synchronize(device)
                start = time.perf_counter()
                view_logits = model(view_data)
                synchronize(device)
                elapsed = time.perf_counter() - start
                if batch_idx >= ARGS.warmup_batches:
                    network_seconds += elapsed
                    forward_samples.append(elapsed)
            if batch_idx >= ARGS.warmup_batches:
                measured_batches += 1
                measured_objects += int(data["y"].shape[0])
                measured_views += int(views.shape[0])
            object_logits = view_logits.new_zeros((data["y"].shape[0], view_logits.shape[1]))
            object_logits.index_add_(0, owners, view_logits)
            counts = torch.bincount(owners, minlength=data["y"].shape[0]).clamp_min(1)
            object_logits = object_logits / counts.unsqueeze(1)
            labels = data["y"].detach().cpu().numpy().astype(np.int64)
            predictions = object_logits.argmax(1).detach().cpu().numpy().astype(np.int64)
            np.add.at(cm, (labels, predictions), 1)
    metrics = confusion_metrics(cm, classes)
    metrics["evaluation_mode"] = "object_level_mean_logits_from_single_view_model"
    metrics["max_views"] = int(ARGS.fair_max_views)
    metrics["input_rotation_deg_z"] = float(rotation_deg_z)
    metrics["latency"] = make_latency_report(
        scope="all_single_view_model_forwards_only", samples_seconds=forward_samples,
        warmup_batches=min(ARGS.warmup_batches, len(loader)), measured_batches=measured_batches,
        device=device, ram_before=ram_peak_before,
        extra={"measured_objects": measured_objects, "measured_views": measured_views,
               "total_seconds": network_seconds,
               "mean_object_ms": 1000.0 * network_seconds / measured_objects if measured_objects else None,
               "mean_view_ms": 1000.0 * network_seconds / measured_views if measured_views else None})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_table(output_dir, metrics["latency"])
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["true/pred", *classes])
        for name, row in zip(classes, cm): writer.writerow([name, *row.tolist()])
    return metrics


def selected_configs() -> list[Path]:
    paths = sorted(ARGS.cfg_dir.glob("*.yaml"))
    if not ARGS.configs:
        return paths
    wanted = set(ARGS.configs)
    return [p for p in paths if p.stem in wanted or p.stem.split("_", 1)[0] in wanted]


def main() -> int:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if ARGS.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if ARGS.device == "cuda" or (ARGS.device == "auto" and torch.cuda.is_available()) else "cpu")
    if ARGS.warmup_batches < 0 or ARGS.benchmark_repetitions < 1:
        raise ValueError("--warmup-batches must be >= 0 and --benchmark-repetitions must be >= 1")
    ARGS.output_root.mkdir(parents=True, exist_ok=True)
    environment = benchmark_environment(device)
    with (ARGS.output_root / "benchmark_environment.json").open("w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2, sort_keys=True)
    write_environment_table(ARGS.output_root, environment)
    summary, missing, failed = [], [], []
    for cfg_path in selected_configs():
        cfg = load_cfg(cfg_path)
        checkpoint = find_checkpoint(cfg_path, cfg)
        if checkpoint is None:
            print(f"[SKIP] {cfg_path.stem}: no {ARGS.checkpoint} checkpoint", flush=True)
            missing.append(cfg_path.stem); continue
        run_out = ARGS.output_root / cfg_path.stem
        if (run_out / "summary.json").exists() and not ARGS.overwrite:
            print(f"[SKIP] {cfg_path.stem}: already validated (use --overwrite)", flush=True); continue
        print(f"[RUN ] {cfg_path.stem}: {checkpoint}", flush=True)
        try:
            seed = int(cfg.get("seed", 8240)); set_random_seed(seed, deterministic=True)
            is_icp_aggregate_baseline = cfg_path.stem.split("_", 1)[0] == "13"
            train_test, top_down = build_datasets(
                cfg, include_top_down=not is_icp_aggregate_baseline)
            cfg.classes = list(train_test.classes); cfg.num_classes = len(cfg.classes)
            cfg.model.cls_args.num_classes = cfg.num_classes
            model = build_model_from_cfg(cfg.model).to(device)
            load_checkpoint(model, str(checkpoint)); model.eval()
            results = {
                "config": str(cfg_path), "checkpoint": str(checkpoint), "seed": seed,
                "training_test": evaluate(model, train_test, cfg, device, run_out / "training_test"),
            }
            if is_icp_aggregate_baseline:
                # Config 13 was trained on ICP-aggregated clouds. Its top-down
                # evaluation must use the matching aggregate exports, separated
                # by whether GT and prior prediction agree.
                aggregate_sets = build_icp_aggregate_top_down_datasets(
                    cfg, list(train_test.classes))
                for bucket, dataset in aggregate_sets.items():
                    result_key = "top_down_" + bucket.replace("-", "_")
                    results[result_key] = evaluate(
                        model, dataset, cfg, device, run_out / result_key,
                        rotation_deg_z=ARGS.top_down_rotation_deg_z)
            else:
                assert top_down is not None
                results["top_down"] = evaluate(
                    model, top_down, cfg, device, run_out / "top_down",
                    rotation_deg_z=ARGS.top_down_rotation_deg_z)
            is_raw_single_view = (
                ARGS.fair_single_view_eval
                and str(cfg.get("classification_dataset_format", "")).lower() == "raw_frames"
                and str(cfg.model.get("NAME", "")) == "BaseCls"
                and not as_bool(cfg.get("raw_frames_object_multi_view", False))
            )
            if is_raw_single_view:
                fair_test, fair_top = build_fair_single_view_object_datasets(
                    cfg, list(train_test.classes))
                results["training_test_object_fair"] = evaluate_single_view_object_fusion(
                    model, fair_test, cfg, device, run_out / "training_test_object_fair")
                results["top_down_object_fair"] = evaluate_single_view_object_fusion(
                    model, fair_top, cfg, device, run_out / "top_down_object_fair",
                    rotation_deg_z=ARGS.top_down_rotation_deg_z)
            with (run_out / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, sort_keys=True)
            summary.append(results)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception as exc:
            failed.append({"config": cfg_path.stem, "error": repr(exc)})
            print(f"[FAIL] {cfg_path.stem}: {exc}", file=sys.stderr, flush=True)
    with (ARGS.output_root / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"completed": summary, "missing_checkpoints": missing, "failed": failed}, f, indent=2)
    print(f"Done: {len(summary)} completed, {len(missing)} missing, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    ARGS = parse_args()
    raise SystemExit(main())
