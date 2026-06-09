#!/usr/bin/env python3
"""Randomly match trailer samples to vehicle samples and write merged PCDs.

The input is a flat PointNeXt split dataset:

    root/train/TLS_VEHICLE_TRAILER/*.pcd
    root/train/TLS_VEHICLE_CAR/*.pcd
    ...

For every trailer, the script randomly picks a car, truck, or van from the same
split, rotates the trailer into the vehicle's longitudinal direction, places it
behind the vehicle, and writes one combined point cloud. The older geometry-based
matcher is still available with ``--strategy geometric`` for comparison runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(
    "/home/georg/new-big-data/"
    "pointnext_seed5655_curated_dataset_without_suspicious_review_split_dataset"
)
SPLITS = ("train", "val", "test")
TRAILER_CLASSES = ("TLS_VEHICLE_TRAILER",)
VEHICLE_CLASSES = (
    "TLS_VEHICLE_CAR",
    "TLS_VEHICLE_VAN",
    "TLS_VEHICLE_TRUCK",
)
TRACK_RE = re.compile(r"__track_(\d+)")
OBJECT_RE = re.compile(r"__object_(\d+)")
EPS = 1e-9


@dataclass
class Sample:
    split: str
    class_name: str
    pcd_path: Path
    json_path: Path | None
    run_id: str
    track_id: str
    object_id: str
    gt_frame_index: int | None
    frame_start: int | None
    frame_end: int | None
    timestamp_ns: int | None
    predicted_class_name: str
    predicted_class_score: float | None
    xyz: np.ndarray | None = field(default=None, repr=False)
    bbox_min: np.ndarray | None = field(default=None, repr=False)
    bbox_max: np.ndarray | None = field(default=None, repr=False)
    centroid: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Match:
    trailer: Sample
    vehicle: Sample
    score: float | None
    xy_gap: float | None
    z_gap: float | None
    center_distance_xy: float | None
    frame_delta: int | None
    time_delta_s: float | None
    output_pcd: Path
    strategy: str
    output_class: str
    alignment: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly match trailer point clouds to vehicle point clouds and merge them."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ROOT.parent / "trailer_vehicle_matched_pointclouds",
        help="Output directory for merged PCDs plus matches.csv.",
    )
    parser.add_argument("--splits", nargs="*", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--trailer-classes", nargs="*", default=list(TRAILER_CLASSES))
    parser.add_argument("--vehicle-classes", nargs="*", default=list(VEHICLE_CLASSES))
    parser.add_argument(
        "--strategy",
        choices=("random", "geometric"),
        default="random",
        help="random pairs trailers with random cars/trucks/vans from the same split.",
    )
    parser.add_argument("--seed", type=int, default=5655)
    parser.add_argument(
        "--random-across-splits",
        action="store_true",
        help="Allow random matches across train/val/test. Default keeps split boundaries.",
    )
    parser.add_argument(
        "--max-frame-delta",
        type=int,
        default=30,
        help="Only compare samples whose gt_frame_index differs by at most this value.",
    )
    parser.add_argument(
        "--max-time-delta",
        type=float,
        default=2.0,
        help="Only compare samples whose gt_timestamp_ns differs by at most this many seconds.",
    )
    parser.add_argument(
        "--max-xy-gap",
        type=float,
        default=4.0,
        help="Reject candidate pairs whose XY bounding boxes are farther apart than this many meters.",
    )
    parser.add_argument(
        "--max-center-distance-xy",
        type=float,
        default=18.0,
        help="Reject candidate pairs whose XY centroids are farther apart than this many meters.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Stop after N written matches in one-per-trailer mode. Use 0 for all trailers.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Generate exactly N random train samples. Enables fixed-count mode.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Generate exactly N random val samples. Enables fixed-count mode.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Generate exactly N random test samples. Enables fixed-count mode.",
    )
    parser.add_argument(
        "--class-name-style",
        choices=("tls", "short"),
        default="tls",
        help="Use TLS_VEHICLE_*_WITH_TRAILER class folders by default.",
    )
    parser.add_argument(
        "--min-vehicle-confidence",
        type=float,
        default=0.0,
        help="Minimum predicted_class_score for vehicle candidates.",
    )
    parser.add_argument(
        "--trailer-gap",
        type=float,
        default=0.6,
        help="Meters between the vehicle rear and trailer front after alignment.",
    )
    parser.add_argument(
        "--behind-side",
        choices=("negative", "positive"),
        default="negative",
        help="Which side of the vehicle longitudinal axis receives the trailer.",
    )
    parser.add_argument(
        "--trailer-orientation",
        choices=("flipped", "same"),
        default="flipped",
        help="flipped rotates the trailer 180 degrees so its opposite end faces the vehicle.",
    )
    parser.add_argument(
        "--trailer-lateral-offset",
        type=float,
        default=0.0,
        help="Optional extra trailer shift along the vehicle lateral axis in meters.",
    )
    parser.add_argument(
        "--trailer-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Optional extra trailer yaw correction in degrees after auto-straightening.",
    )
    parser.add_argument(
        "--auto-straight",
        dest="auto_straight",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag; centerline alignment is used by default.",
    )
    parser.add_argument(
        "--no-auto-straight",
        dest="auto_straight",
        action="store_false",
        help="Disable automatic trailer yaw straightening.",
    )
    parser.add_argument(
        "--auto-straight-search-deg",
        type=float,
        default=18.0,
        help="Search range around the PCA angle for automatic straightening.",
    )
    parser.add_argument(
        "--auto-straight-step-deg",
        type=float,
        default=0.25,
        help="Yaw step size for automatic straightening.",
    )
    parser.add_argument(
        "--straighten-centerline",
        dest="straighten_centerline",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag; centerline alignment is used by default.",
    )
    parser.add_argument(
        "--no-straighten-centerline",
        dest="straighten_centerline",
        action="store_false",
        help="Disable rigid trailer centerline straightening.",
    )
    parser.add_argument(
        "--write-unmatched",
        action="store_true",
        help="Include unmatched trailers in matches.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def as_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_pcd_header(path: Path) -> tuple[dict[str, list[str]], int]:
    header: dict[str, list[str]] = {}
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"missing DATA line in {path}")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                return header, handle.tell()


def pcd_header_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            count += 1
            if line.decode("ascii", errors="replace").strip().upper().startswith("DATA"):
                return count
    raise ValueError(f"missing DATA line in {path}")


def pcd_dtype(size: int, typ: str) -> str:
    typ = typ.upper()
    if typ == "F":
        return {4: "<f4", 8: "<f8"}[size]
    if typ == "I":
        return {1: "i1", 2: "<i2", 4: "<i4", 8: "<i8"}[size]
    if typ == "U":
        return {1: "u1", 2: "<u2", 4: "<u4", 8: "<u8"}[size]
    raise ValueError(f"unsupported PCD field type: {typ}")


def read_xyz(path: Path) -> np.ndarray:
    header, offset = parse_pcd_header(path)
    fields = header.get("FIELDS", [])
    sizes = [int(v) for v in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(v) for v in header.get("COUNT", [])] or [1] * len(fields)
    points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
    data = header.get("DATA", [""])[0].lower()
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"PCD has no x/y/z fields: {path}")

    if data == "ascii":
        raw = np.loadtxt(path, comments="#", skiprows=pcd_header_lines(path), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        indices = [fields.index(axis) for axis in ("x", "y", "z")]
        return raw[:, indices].astype(np.float32, copy=False)

    if data != "binary":
        raise ValueError(f"unsupported PCD DATA type {data!r}: {path}")

    dtype_fields = []
    for field_name, size, typ, count in zip(fields, sizes, types, counts):
        dtype = pcd_dtype(size, typ)
        dtype_fields.append((field_name, dtype, (count,)) if count > 1 else (field_name, dtype))
    dtype = np.dtype(dtype_fields)

    with path.open("rb") as handle:
        handle.seek(offset)
        cloud = np.frombuffer(handle.read(points * dtype.itemsize), dtype=dtype, count=points)
    return np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float32, copy=False)


def write_labeled_xyz_pcd(path: Path, vehicle_xyz: np.ndarray, trailer_xyz: np.ndarray) -> None:
    xyz = np.vstack([vehicle_xyz, trailer_xyz]).astype(np.float32, copy=False)
    source = np.concatenate(
        [
            np.zeros(vehicle_xyz.shape[0], dtype=np.float32),
            np.ones(trailer_xyz.shape[0], dtype=np.float32),
        ]
    )
    with path.open("w", encoding="ascii") as handle:
        handle.write("# .PCD v0.7 - Point Cloud Data file format\n")
        handle.write("VERSION 0.7\n")
        handle.write("FIELDS x y z source_id\n")
        handle.write("SIZE 4 4 4 4\n")
        handle.write("TYPE F F F F\n")
        handle.write("COUNT 1 1 1 1\n")
        handle.write(f"WIDTH {xyz.shape[0]}\n")
        handle.write("HEIGHT 1\n")
        handle.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        handle.write(f"POINTS {xyz.shape[0]}\n")
        handle.write("DATA ascii\n")
        for point, source_id in zip(xyz, source):
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {source_id:.0f}\n")


def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_id_from_stem(stem: str) -> str:
    return stem.split("__track_", 1)[0]


def regex_group(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1) if match else ""


def collect_samples(root: Path, splits: list[str], classes: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for split in splits:
        for class_name in classes:
            class_dir = root / split / class_name
            if not class_dir.is_dir():
                continue
            for pcd_path in sorted(class_dir.glob("*.pcd")):
                json_path = pcd_path.with_suffix(".json")
                metadata = load_metadata(json_path if json_path.is_file() else None)
                metrics = metadata.get("metrics") or {}
                stem = pcd_path.stem
                samples.append(
                    Sample(
                        split=split,
                        class_name=class_name,
                        pcd_path=pcd_path,
                        json_path=json_path if json_path.is_file() else None,
                        run_id=run_id_from_stem(stem),
                        track_id=str(metadata.get("track_id") or regex_group(TRACK_RE, stem)),
                        object_id=str(metrics.get("gt_object_id") or regex_group(OBJECT_RE, stem)),
                        gt_frame_index=as_int(metrics.get("gt_frame_index")),
                        frame_start=as_int(metrics.get("chunk_quality_segment_start_frame")),
                        frame_end=as_int(metrics.get("chunk_quality_segment_end_frame")),
                        timestamp_ns=as_int(metrics.get("gt_timestamp_ns")),
                        predicted_class_name=str(metrics.get("predicted_class_name") or ""),
                        predicted_class_score=as_float(metrics.get("predicted_class_score")),
                    )
                )
    return samples


def load_geometry(sample: Sample) -> bool:
    if sample.xyz is not None:
        return True
    xyz = read_xyz(sample.pcd_path)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return False
    sample.xyz = xyz
    sample.bbox_min = xyz.min(axis=0)
    sample.bbox_max = xyz.max(axis=0)
    sample.centroid = xyz.mean(axis=0)
    return True


def normalize_xy(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


def principal_axis_xy(xyz: np.ndarray) -> np.ndarray:
    xy = xyz[:, :2].astype(np.float64, copy=False)
    centered = xy - xy.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3 or float(np.linalg.norm(centered)) <= EPS:
        return np.array([1.0, 0.0], dtype=np.float32)
    cov = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    if axis[1] < 0 or (abs(axis[1]) <= EPS and axis[0] < 0):
        axis = -axis
    return normalize_xy(axis)


def rotate_xy(points: np.ndarray, angle: float, center_xy: np.ndarray) -> np.ndarray:
    result = points.copy()
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    result[:, :2] = (result[:, :2] - center_xy) @ rot.T + center_xy
    return result


def signed_extent(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    values = points[:, :2] @ axis
    return float(values.min()), float(values.max())


def robust_signed_extent(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    values = points[:, :2] @ axis
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def signed_center(points: np.ndarray, axis: np.ndarray) -> float:
    min_value, max_value = robust_signed_extent(points, axis)
    return 0.5 * (min_value + max_value)


def centerline_model(points: np.ndarray) -> dict[str, Any]:
    axis = principal_axis_xy(points)
    lateral = np.array([-axis[1], axis[0]], dtype=np.float32)
    long_min, long_max = robust_signed_extent(points, axis)
    lat_min, lat_max = robust_signed_extent(points, lateral)
    long_center = 0.5 * (long_min + long_max)
    lat_center = 0.5 * (lat_min + lat_max)
    center_xy = long_center * axis + lat_center * lateral
    return {
        "axis": axis,
        "lateral": lateral,
        "center_xy": center_xy.astype(np.float32, copy=False),
        "long_min": float(long_min),
        "long_max": float(long_max),
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "long_center": float(long_center),
        "lat_center": float(lat_center),
    }


def model_to_json(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "axis_xy": [float(model["axis"][0]), float(model["axis"][1])],
        "lateral_xy": [float(model["lateral"][0]), float(model["lateral"][1])],
        "center_xy": [float(model["center_xy"][0]), float(model["center_xy"][1])],
        "long_extent": [float(model["long_min"]), float(model["long_max"])],
        "lat_extent": [float(model["lat_min"]), float(model["lat_max"])],
    }


def lateral_width(points: np.ndarray, lateral_axis: np.ndarray) -> float:
    min_value, max_value = robust_signed_extent(points, lateral_axis)
    return max_value - min_value


def auto_straight_yaw_offset(
    points: np.ndarray,
    center_xy: np.ndarray,
    base_angle: float,
    lateral_axis: np.ndarray,
    search_deg: float,
    step_deg: float,
) -> tuple[float, float]:
    if search_deg <= 0.0 or step_deg <= 0.0:
        rotated = rotate_xy(points, base_angle, center_xy)
        return 0.0, lateral_width(rotated, lateral_axis)

    steps = max(1, int(round((2.0 * search_deg) / step_deg)))
    offsets = np.linspace(-search_deg, search_deg, steps + 1)
    best_offset = 0.0
    best_width = float("inf")
    for offset_deg in offsets:
        rotated = rotate_xy(points, base_angle + math.radians(float(offset_deg)), center_xy)
        width = lateral_width(rotated, lateral_axis)
        if width < best_width:
            best_width = width
            best_offset = float(offset_deg)
    return math.radians(best_offset), best_width


def centerline_slope(
    points: np.ndarray,
    longitudinal_axis: np.ndarray,
    lateral_axis: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    long_values = points[:, :2] @ longitudinal_axis
    lat_values = points[:, :2] @ lateral_axis
    lo, hi = np.percentile(long_values, [5.0, 95.0])
    mask = (long_values >= lo) & (long_values <= hi)
    if int(mask.sum()) < 20 or abs(float(hi - lo)) <= EPS:
        return None, {"applied": False, "reason": "insufficient_points"}

    x = long_values[mask].astype(np.float64, copy=False)
    y = lat_values[mask].astype(np.float64, copy=False)
    x_center = float(np.median(x))
    y_center = float(np.median(y))
    x0 = x - x_center
    y0 = y - y_center
    denom = float(np.dot(x0, x0))
    if denom <= EPS:
        return None, {"applied": False, "reason": "degenerate_longitudinal_extent"}

    slope = float(np.dot(x0, y0) / denom)
    width = float(np.percentile(y, 95.0) - np.percentile(y, 5.0))
    return slope, {
        "applied": True,
        "slope_lateral_per_longitudinal": slope,
        "equivalent_angle_deg": math.degrees(math.atan(slope)),
        "center_longitudinal": x_center,
        "center_lateral": y_center,
        "width_m": width,
    }


def straighten_centerline_xy(
    points: np.ndarray,
    longitudinal_axis: np.ndarray,
    lateral_axis: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    slope, before = centerline_slope(points, longitudinal_axis, lateral_axis)
    if slope is None:
        return points, before

    correction_angle = -math.atan(slope)
    center_xy = points[:, :2].mean(axis=0)
    corrected = rotate_xy(points, correction_angle, center_xy)
    after_slope, after = centerline_slope(corrected, longitudinal_axis, lateral_axis)
    return corrected, {
        "applied": True,
        "method": "rigid_rotation",
        "correction_angle_deg": math.degrees(correction_angle),
        "before": before,
        "after": after,
        "residual_angle_deg": None if after_slope is None else math.degrees(math.atan(after_slope)),
    }


def output_class_for(vehicle: Sample, style: str = "tls") -> str:
    if style == "short":
        if vehicle.class_name == "TLS_VEHICLE_TRUCK":
            return "truck_with_trailer"
        return "car_with_trailer"
    if vehicle.class_name == "TLS_VEHICLE_TRUCK":
        return "TLS_VEHICLE_TRUCK_WITH_TRAILER"
    return "TLS_VEHICLE_CAR_WITH_TRAILER"


def align_trailer_behind_vehicle(
    vehicle_xyz: np.ndarray,
    trailer_xyz: np.ndarray,
    gap: float,
    behind_side: str,
    trailer_orientation: str,
    lateral_offset: float,
    yaw_offset_deg: float,
    auto_straight: bool,
    auto_straight_search_deg: float,
    auto_straight_step_deg: float,
    straighten_centerline: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    del auto_straight, auto_straight_search_deg, auto_straight_step_deg, straighten_centerline

    vehicle_line = centerline_model(vehicle_xyz)
    trailer_line = centerline_model(trailer_xyz)
    vehicle_axis = vehicle_line["axis"]
    if behind_side == "positive":
        vehicle_axis = -vehicle_axis
        vehicle_lateral = np.array([-vehicle_axis[1], vehicle_axis[0]], dtype=np.float32)
        vehicle_line = dict(vehicle_line)
        vehicle_line["axis"] = vehicle_axis
        vehicle_line["lateral"] = vehicle_lateral
        vehicle_line["long_min"], vehicle_line["long_max"] = robust_signed_extent(vehicle_xyz, vehicle_axis)
        vehicle_line["lat_min"], vehicle_line["lat_max"] = robust_signed_extent(vehicle_xyz, vehicle_lateral)
        vehicle_line["long_center"] = 0.5 * (vehicle_line["long_min"] + vehicle_line["long_max"])
        vehicle_line["lat_center"] = 0.5 * (vehicle_line["lat_min"] + vehicle_line["lat_max"])
        vehicle_line["center_xy"] = (
            vehicle_line["long_center"] * vehicle_axis
            + vehicle_line["lat_center"] * vehicle_lateral
        ).astype(np.float32, copy=False)

    vehicle_lateral = vehicle_line["lateral"]
    target_trailer_axis = -vehicle_axis if trailer_orientation == "flipped" else vehicle_axis
    trailer_axis = trailer_line["axis"]
    base_angle = math.atan2(target_trailer_axis[1], target_trailer_axis[0]) - math.atan2(
        trailer_axis[1], trailer_axis[0]
    )
    manual_yaw_rad = math.radians(yaw_offset_deg)
    angle = base_angle + manual_yaw_rad
    aligned = rotate_xy(trailer_xyz, angle, trailer_line["center_xy"])

    rotated_trailer_line = centerline_model(aligned)
    trailer_max_on_vehicle_axis = robust_signed_extent(aligned, vehicle_axis)[1]
    vehicle_min_on_axis, vehicle_max_on_axis = robust_signed_extent(vehicle_xyz, vehicle_axis)
    target_trailer_front = vehicle_min_on_axis - gap
    longitudinal_shift = target_trailer_front - trailer_max_on_vehicle_axis

    vehicle_lat_center = signed_center(vehicle_xyz, vehicle_lateral)
    trailer_lat_center = signed_center(aligned, vehicle_lateral)
    lateral_shift = vehicle_lat_center - trailer_lat_center + lateral_offset
    xy_shift = longitudinal_shift * vehicle_axis + lateral_shift * vehicle_lateral

    z_shift = float(vehicle_xyz[:, 2].min() - aligned[:, 2].min())
    aligned[:, :2] += xy_shift
    aligned[:, 2] += z_shift

    final_trailer_line = centerline_model(aligned)
    final_axis_dot = float(abs(np.dot(final_trailer_line["axis"], vehicle_axis)))
    final_angle_error_deg = math.degrees(math.acos(max(-1.0, min(1.0, final_axis_dot))))
    out_min, out_max = robust_signed_extent(aligned, vehicle_axis)
    alignment = {
        "method": "centerline_axis_alignment",
        "vehicle_centerline": model_to_json(vehicle_line),
        "trailer_centerline_before": model_to_json(trailer_line),
        "trailer_centerline_after_rotation": model_to_json(rotated_trailer_line),
        "trailer_centerline_final": model_to_json(final_trailer_line),
        "target_trailer_axis_xy": [float(target_trailer_axis[0]), float(target_trailer_axis[1])],
        "trailer_orientation": trailer_orientation,
        "base_rotation_angle_rad": float(base_angle),
        "manual_yaw_offset_deg": float(yaw_offset_deg),
        "rotation_angle_rad": float(angle),
        "final_abs_angle_error_deg": float(final_angle_error_deg),
        "behind_side": behind_side,
        "gap_m": float(gap),
        "longitudinal_shift_m": float(longitudinal_shift),
        "lateral_offset_m": float(lateral_offset),
        "lateral_shift_m": float(lateral_shift),
        "z_shift_m": float(z_shift),
        "trailer_extent_after_on_vehicle_axis": [float(out_min), float(out_max)],
        "vehicle_extent_on_axis": [float(vehicle_min_on_axis), float(vehicle_max_on_axis)],
        "source_id": {"0": "vehicle", "1": "aligned_trailer"},
    }
    return aligned.astype(np.float32, copy=False), alignment


def interval_delta(a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None) -> int | None:
    if None in (a_start, a_end, b_start, b_end):
        return None
    assert a_start is not None and a_end is not None and b_start is not None and b_end is not None
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0


def frame_delta(a: Sample, b: Sample) -> int | None:
    if a.gt_frame_index is not None and b.gt_frame_index is not None:
        return abs(a.gt_frame_index - b.gt_frame_index)
    return interval_delta(a.frame_start, a.frame_end, b.frame_start, b.frame_end)


def time_delta_s(a: Sample, b: Sample) -> float | None:
    if a.timestamp_ns is None or b.timestamp_ns is None:
        return None
    return abs(a.timestamp_ns - b.timestamp_ns) / 1_000_000_000.0


def axis_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0.0


def score_pair(trailer: Sample, vehicle: Sample) -> tuple[float, float, float, float]:
    assert trailer.bbox_min is not None and trailer.bbox_max is not None and trailer.centroid is not None
    assert vehicle.bbox_min is not None and vehicle.bbox_max is not None and vehicle.centroid is not None
    gap_x = axis_gap(trailer.bbox_min[0], trailer.bbox_max[0], vehicle.bbox_min[0], vehicle.bbox_max[0])
    gap_y = axis_gap(trailer.bbox_min[1], trailer.bbox_max[1], vehicle.bbox_min[1], vehicle.bbox_max[1])
    gap_z = axis_gap(trailer.bbox_min[2], trailer.bbox_max[2], vehicle.bbox_min[2], vehicle.bbox_max[2])
    xy_gap = float(math.hypot(gap_x, gap_y))
    z_gap = float(gap_z)
    center_distance_xy = float(np.linalg.norm(trailer.centroid[:2] - vehicle.centroid[:2]))
    score = xy_gap + 0.15 * center_distance_xy + 0.5 * z_gap
    return score, xy_gap, z_gap, center_distance_xy


def candidate_allowed(trailer: Sample, vehicle: Sample, args: argparse.Namespace) -> bool:
    if trailer.split != vehicle.split or trailer.run_id != vehicle.run_id:
        return False
    if trailer.track_id and trailer.track_id == vehicle.track_id:
        return False
    if trailer.object_id and trailer.object_id == vehicle.object_id:
        return False
    confidence = vehicle.predicted_class_score
    if confidence is not None and confidence < args.min_vehicle_confidence:
        return False
    delta_frame = frame_delta(trailer, vehicle)
    if delta_frame is not None and delta_frame > args.max_frame_delta:
        return False
    delta_time = time_delta_s(trailer, vehicle)
    if delta_time is not None and delta_time > args.max_time_delta:
        return False
    return True


def random_candidate_allowed(trailer: Sample, vehicle: Sample, args: argparse.Namespace) -> bool:
    if not args.random_across_splits and trailer.split != vehicle.split:
        return False
    if trailer.track_id and trailer.track_id == vehicle.track_id:
        return False
    if trailer.object_id and trailer.object_id == vehicle.object_id:
        return False
    confidence = vehicle.predicted_class_score
    if confidence is not None and confidence < args.min_vehicle_confidence:
        return False
    return True


def find_random_match(
    trailer: Sample,
    vehicles: list[Sample],
    args: argparse.Namespace,
    rng: random.Random,
) -> Sample | None:
    candidates = [
        vehicle
        for vehicle in vehicles
        if random_candidate_allowed(trailer, vehicle, args)
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def find_best_match(trailer: Sample, vehicles: list[Sample], args: argparse.Namespace) -> tuple[Sample, float, float, float, float] | None:
    if not load_geometry(trailer):
        return None
    best: tuple[Sample, float, float, float, float] | None = None
    for vehicle in vehicles:
        if not candidate_allowed(trailer, vehicle, args):
            continue
        if not load_geometry(vehicle):
            continue
        score, xy_gap, z_gap, center_distance_xy = score_pair(trailer, vehicle)
        if xy_gap > args.max_xy_gap or center_distance_xy > args.max_center_distance_xy:
            continue
        if best is None or score < best[1]:
            best = (vehicle, score, xy_gap, z_gap, center_distance_xy)
    return best


def write_match_metadata(path: Path, match: Match) -> None:
    payload = {
        "score": match.score,
        "xy_gap": match.xy_gap,
        "z_gap": match.z_gap,
        "center_distance_xy": match.center_distance_xy,
        "frame_delta": match.frame_delta,
        "time_delta_s": match.time_delta_s,
        "strategy": match.strategy,
        "output_class": match.output_class,
        "alignment": match.alignment,
        "output_pcd": str(match.output_pcd),
        "trailer": sample_to_dict(match.trailer),
        "vehicle": sample_to_dict(match.vehicle),
        "source_id": {"0": "vehicle", "1": "aligned_trailer"},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sample_to_dict(sample: Sample) -> dict[str, Any]:
    return {
        "split": sample.split,
        "class_name": sample.class_name,
        "pcd_path": str(sample.pcd_path),
        "json_path": str(sample.json_path) if sample.json_path else "",
        "run_id": sample.run_id,
        "track_id": sample.track_id,
        "object_id": sample.object_id,
        "gt_frame_index": sample.gt_frame_index,
        "frame_start": sample.frame_start,
        "frame_end": sample.frame_end,
        "timestamp_ns": sample.timestamp_ns,
        "predicted_class_name": sample.predicted_class_name,
        "predicted_class_score": sample.predicted_class_score,
    }


def csv_row(match: Match | None, trailer: Sample, reason: str = "") -> dict[str, Any]:
    vehicle = match.vehicle if match else None
    return {
        "matched": bool(match),
        "reason": reason,
        "strategy": match.strategy if match else "",
        "output_class": match.output_class if match else "",
        "split": trailer.split,
        "trailer_class": trailer.class_name,
        "trailer_pcd": trailer.pcd_path,
        "trailer_track_id": trailer.track_id,
        "trailer_object_id": trailer.object_id,
        "vehicle_class": vehicle.class_name if vehicle else "",
        "vehicle_pcd": vehicle.pcd_path if vehicle else "",
        "vehicle_track_id": vehicle.track_id if vehicle else "",
        "vehicle_object_id": vehicle.object_id if vehicle else "",
        "score": fmt_float(match.score) if match else "",
        "xy_gap": fmt_float(match.xy_gap) if match else "",
        "z_gap": fmt_float(match.z_gap) if match else "",
        "center_distance_xy": fmt_float(match.center_distance_xy) if match else "",
        "frame_delta": match.frame_delta if match else "",
        "time_delta_s": f"{match.time_delta_s:.6f}" if match and match.time_delta_s is not None else "",
        "output_pcd": match.output_pcd if match else "",
    }


def fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def fixed_counts(args: argparse.Namespace) -> dict[str, int]:
    raw = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    counts = {split: int(count) for split, count in raw.items() if count is not None}
    for split, count in counts.items():
        if count < 0:
            raise ValueError(f"{split} count must be non-negative, got {count}")
    return counts


def samples_by_split(samples: list[Sample]) -> dict[str, list[Sample]]:
    grouped: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    for sample in samples:
        grouped.setdefault(sample.split, []).append(sample)
    return grouped


def vehicle_candidates_for_split(
    split: str,
    trailers: list[Sample],
    vehicles: list[Sample],
    args: argparse.Namespace,
) -> list[Sample]:
    dummy = trailers[0] if trailers else None
    candidates = []
    for vehicle in vehicles:
        if not args.random_across_splits and vehicle.split != split:
            continue
        confidence = vehicle.predicted_class_score
        if confidence is not None and confidence < args.min_vehicle_confidence:
            continue
        if dummy is not None and dummy.track_id and dummy.track_id == vehicle.track_id:
            continue
        candidates.append(vehicle)
    return candidates


def output_name(trailer: Sample, vehicle: Sample, sample_index: int | None) -> str:
    suffix = "" if sample_index is None else f"__synthetic_{sample_index:05d}"
    return (
        f"{trailer.pcd_path.stem}__MATCH__{vehicle.class_name}"
        f"__track_{vehicle.track_id}{suffix}.pcd"
    )


def write_match(
    trailer: Sample,
    vehicle: Sample,
    score: float | None,
    xy_gap: float | None,
    z_gap: float | None,
    center_distance_xy: float | None,
    sample_index: int | None,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> bool:
    output_class = output_class_for(vehicle, args.class_name_style)
    split_out = args.out / trailer.split / output_class
    output_pcd = split_out / output_name(trailer, vehicle, sample_index)
    output_json = output_pcd.with_suffix(".json")
    if output_pcd.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists, pass --overwrite: {output_pcd}")

    match = Match(
        trailer=trailer,
        vehicle=vehicle,
        score=score,
        xy_gap=xy_gap,
        z_gap=z_gap,
        center_distance_xy=center_distance_xy,
        frame_delta=frame_delta(trailer, vehicle),
        time_delta_s=time_delta_s(trailer, vehicle),
        output_pcd=output_pcd,
        strategy=args.strategy,
        output_class=output_class,
        alignment={},
    )
    rows.append(csv_row(match, trailer))

    if args.dry_run:
        return True

    if not load_geometry(trailer) or not load_geometry(vehicle):
        rows[-1] = csv_row(None, trailer, "pcd_read_failed")
        return False
    assert trailer.xyz is not None and vehicle.xyz is not None
    aligned_trailer_xyz, alignment = align_trailer_behind_vehicle(
        vehicle.xyz,
        trailer.xyz,
        gap=args.trailer_gap,
        behind_side=args.behind_side,
        trailer_orientation=args.trailer_orientation,
        lateral_offset=args.trailer_lateral_offset,
        yaw_offset_deg=args.trailer_yaw_offset_deg,
        auto_straight=args.auto_straight,
        auto_straight_search_deg=args.auto_straight_search_deg,
        auto_straight_step_deg=args.auto_straight_step_deg,
        straighten_centerline=args.straighten_centerline,
    )
    match = Match(
        trailer=trailer,
        vehicle=vehicle,
        score=score,
        xy_gap=xy_gap,
        z_gap=z_gap,
        center_distance_xy=center_distance_xy,
        frame_delta=frame_delta(trailer, vehicle),
        time_delta_s=time_delta_s(trailer, vehicle),
        output_pcd=output_pcd,
        strategy=args.strategy,
        output_class=output_class,
        alignment=alignment,
    )
    rows[-1] = csv_row(match, trailer)
    split_out.mkdir(parents=True, exist_ok=True)
    write_labeled_xyz_pcd(output_pcd, vehicle.xyz, aligned_trailer_xyz)
    write_match_metadata(output_json, match)
    return True


def main() -> int:
    args = parse_args()
    trailers = collect_samples(args.root, args.splits, args.trailer_classes)
    vehicles = collect_samples(args.root, args.splits, args.vehicle_classes)
    if not trailers:
        raise SystemExit(f"No trailer samples found below {args.root}")
    if not vehicles:
        raise SystemExit(f"No vehicle samples found below {args.root}")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "matches.csv"
    rows: list[dict[str, Any]] = []
    written = 0
    rng = random.Random(args.seed)
    counts = fixed_counts(args)

    if counts:
        trailers_by_split = samples_by_split(trailers)
        vehicles_by_split = samples_by_split(vehicles)
        for split in SPLITS:
            count = counts.get(split, 0)
            if count <= 0:
                continue
            split_trailers = trailers_by_split.get(split, [])
            split_vehicles = vehicles if args.random_across_splits else vehicles_by_split.get(split, [])
            if not split_trailers:
                raise SystemExit(f"No trailer samples found for split {split}")
            if not split_vehicles:
                raise SystemExit(f"No vehicle samples found for split {split}")

            for index in range(count):
                trailer = rng.choice(split_trailers)
                vehicle = find_random_match(trailer, split_vehicles, args, rng)
                if vehicle is None:
                    if args.write_unmatched:
                        rows.append(csv_row(None, trailer, "no_candidate"))
                    continue
                if write_match(trailer, vehicle, None, None, None, None, index, args, rows):
                    written += 1
        mode = "fixed-count"
    else:
        for trailer in trailers:
            if args.strategy == "random":
                vehicle = find_random_match(trailer, vehicles, args, rng)
                best = (vehicle, None, None, None, None) if vehicle is not None else None
            else:
                best = find_best_match(trailer, vehicles, args)

            if best is None:
                if args.write_unmatched:
                    rows.append(csv_row(None, trailer, "no_candidate"))
                continue

            vehicle, score, xy_gap, z_gap, center_distance_xy = best
            if write_match(trailer, vehicle, score, xy_gap, z_gap, center_distance_xy, None, args, rows):
                written += 1
            if args.max_matches and written >= args.max_matches:
                break
        mode = "one-per-trailer"

    fieldnames = list(csv_row(None, trailers[0]).keys())
    if not args.dry_run:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    matched = sum(1 for row in rows if row["matched"])
    print(f"mode: {mode}")
    print(f"trailers scanned: {len(trailers)}")
    print(f"vehicles indexed: {len(vehicles)}")
    print(f"matches written: {matched}")
    print(f"output: {args.out}")
    if args.dry_run:
        print("dry-run: no files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
