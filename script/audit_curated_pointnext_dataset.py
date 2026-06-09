#!/usr/bin/env python3
"""Audit a curated flat PointNeXt dataset for suspicious labels/samples.

The script is intentionally conservative: it does not decide that a sample is
"wrong". It ranks samples that deserve manual review because metadata, model
confidence, point count, or geometry looks unusual for the assigned class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
EPS = 1e-9


@dataclass
class SampleAudit:
    split: str
    dataset_class: str
    sample_id: str
    dataset_pcd: str
    source_pcd: str
    current_class: str
    predicted_class: str
    source_folder_class: str
    bucket: str
    confidence: float | None
    point_count: int | None
    pcd_points: int | None
    x_extent: float | None
    y_extent: float | None
    z_extent: float | None
    volume: float | None
    density: float | None
    aspect_xy: float | None
    aspect_xz: float | None
    score: float
    reasons: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find suspicious labels and low-quality samples in a curated "
            "PointNeXt train/val/test dataset."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/georg/new-big-data/pointnext_seed5655_curated_dataset"),
        help="Curated dataset root containing manifest.jsonl and train/val/test folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/georg/new-big-data/pointnext_seed5655_curated_dataset_audit"),
        help="Directory for audit CSV/Markdown output.",
    )
    parser.add_argument("--low-confidence", type=float, default=0.70)
    parser.add_argument("--outlier-z", type=float, default=4.5)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument(
        "--review-links",
        type=int,
        default=250,
        help="Create symlinks for the top N suspicious samples. Use 0 to disable.",
    )
    return parser.parse_args()


def parse_pcd_header(path: Path) -> tuple[dict[str, list[str]], int]:
    header: dict[str, list[str]] = {}
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("missing DATA line")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                return header, handle.tell()


def pcd_dtype(size: int, typ: str) -> str:
    typ = typ.upper()
    if typ == "F":
        return {4: "<f4", 8: "<f8"}[size]
    if typ == "I":
        return {1: "i1", 2: "<i2", 4: "<i4", 8: "<i8"}[size]
    if typ == "U":
        return {1: "u1", 2: "<u2", 4: "<u4", 8: "<u8"}[size]
    raise ValueError(f"unsupported PCD field type: {typ}")


def pcd_header_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            count += 1
            if line.decode("ascii", errors="replace").strip().upper().startswith("DATA"):
                return count
    raise ValueError("missing DATA line")


def read_xyz(path: Path) -> tuple[np.ndarray, int]:
    header, offset = parse_pcd_header(path)
    fields = header.get("FIELDS", [])
    sizes = [int(v) for v in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(v) for v in header.get("COUNT", [])] or [1] * len(fields)
    points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
    data = header.get("DATA", [""])[0].lower()
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"PCD has no x/y/z fields: {fields}")

    if data == "ascii":
        raw = np.loadtxt(path, comments="#", skiprows=pcd_header_lines(path), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        indices = [fields.index(axis) for axis in ("x", "y", "z")]
        return raw[:, indices], points

    if data != "binary":
        raise ValueError(f"unsupported PCD DATA type: {data}")

    dtype_fields = []
    for field, size, typ, count in zip(fields, sizes, types, counts):
        dtype = pcd_dtype(size, typ)
        dtype_fields.append((field, dtype, (count,)) if count > 1 else (field, dtype))
    dtype = np.dtype(dtype_fields)

    with path.open("rb") as handle:
        handle.seek(offset)
        cloud = np.frombuffer(handle.read(points * dtype.itemsize), dtype=dtype, count=points)
    xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float32, copy=False)
    return xyz, points


def source_folder_class(source_pcd: str) -> str:
    if not source_pcd:
        return ""
    path = Path(source_pcd)
    if path.parent.name == "pcds":
        return path.parent.parent.name
    return path.parent.name


def sample_id_from_path(path: Path) -> str:
    return path.stem


def iter_manifest(root: Path) -> list[dict[str, str]]:
    manifest = root / "manifest.jsonl"
    if manifest.is_file():
        rows = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    rows = []
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            continue
        for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            for pcd in sorted(class_dir.glob("*.pcd")):
                rows.append(
                    {
                        "split": split,
                        "dataset_class": class_dir.name,
                        "dataset_pcd": str(pcd),
                        "sample_id": pcd.stem,
                    }
                )
    return rows


def finite_extents(xyz: np.ndarray) -> tuple[int, tuple[float, float, float] | None]:
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return 0, None
    extents = xyz.max(axis=0) - xyz.min(axis=0)
    return int(xyz.shape[0]), (float(extents[0]), float(extents[1]), float(extents[2]))


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def collect_samples(root: Path) -> list[SampleAudit]:
    rows = []
    for record in iter_manifest(root):
        pcd_path = Path(record.get("dataset_pcd") or "")
        dataset_class = record.get("dataset_class") or (pcd_path.parent.name if pcd_path.parent else "")
        split = record.get("split") or (pcd_path.parent.parent.name if pcd_path.parent.parent else "")
        sample_id = record.get("sample_id") or sample_id_from_path(pcd_path)
        confidence = as_float(record.get("confidence"))
        point_count = as_int(record.get("point_count"))
        pcd_points = None
        x_extent = y_extent = z_extent = volume = density = aspect_xy = aspect_xz = None
        reasons = []
        score = 0.0

        if not pcd_path.is_file():
            reasons.append("missing_pcd")
            score += 100.0
        else:
            try:
                xyz, header_points = read_xyz(pcd_path)
                pcd_points = int(header_points)
                finite_points, extents = finite_extents(xyz)
                if finite_points != pcd_points:
                    reasons.append("non_finite_xyz")
                    score += 20.0
                if extents is None:
                    reasons.append("empty_or_invalid_xyz")
                    score += 100.0
                else:
                    x_extent, y_extent, z_extent = extents
                    volume = max(x_extent * y_extent * z_extent, EPS)
                    density = float(finite_points / volume)
                    aspect_xy = float(max(x_extent, y_extent) / max(min(x_extent, y_extent), EPS))
                    aspect_xz = float(max(x_extent, y_extent) / max(z_extent, EPS))
            except Exception as exc:
                reasons.append(f"pcd_read_error:{type(exc).__name__}")
                score += 100.0

        rows.append(
            SampleAudit(
                split=split,
                dataset_class=dataset_class,
                sample_id=sample_id,
                dataset_pcd=str(pcd_path),
                source_pcd=str(record.get("source_pcd", record.get("pcd_path", ""))),
                current_class=str(record.get("current_class", "")),
                predicted_class=str(record.get("predicted_class", "")),
                source_folder_class=source_folder_class(str(record.get("source_pcd", record.get("pcd_path", "")))),
                bucket=str(record.get("bucket", "")),
                confidence=confidence,
                point_count=point_count,
                pcd_points=pcd_points,
                x_extent=x_extent,
                y_extent=y_extent,
                z_extent=z_extent,
                volume=volume,
                density=density,
                aspect_xy=aspect_xy,
                aspect_xz=aspect_xz,
                score=score,
                reasons=";".join(reasons),
            )
        )
    return rows


def median_mad(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median, max(mad * 1.4826, EPS)


def robust_z(value: float | None, median: float, scale: float, log_value: bool = False) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    v = math.log1p(max(0.0, float(value))) if log_value else float(value)
    return abs((v - median) / scale)


def add_scores(rows: list[SampleAudit], low_confidence: float, min_points: int, outlier_z: float) -> None:
    grouped: dict[str, list[SampleAudit]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset_class].append(row)

    stats: dict[str, dict[str, tuple[float, float, bool]]] = {}
    for class_name, class_rows in grouped.items():
        fields = {
            "point_count": True,
            "x_extent": True,
            "y_extent": True,
            "z_extent": True,
            "volume": True,
            "density": True,
            "aspect_xy": True,
            "aspect_xz": True,
        }
        stats[class_name] = {}
        for field, log_value in fields.items():
            values = [
                math.log1p(max(0.0, float(getattr(row, field)))) if log_value else float(getattr(row, field))
                for row in class_rows
                if getattr(row, field) is not None and math.isfinite(float(getattr(row, field)))
            ]
            if len(values) >= 8:
                stats[class_name][field] = (*median_mad(values), log_value)

    for row in rows:
        reasons = [reason for reason in row.reasons.split(";") if reason]

        if row.predicted_class and row.predicted_class != row.dataset_class:
            reasons.append("dataset_class_ne_predicted_class")
            row.score += 80.0
        if row.confidence is not None and row.confidence < low_confidence:
            reasons.append("low_confidence")
            row.score += 25.0 + (low_confidence - row.confidence) * 50.0
        if row.point_count is not None and row.point_count < min_points:
            reasons.append("low_manifest_point_count")
            row.score += 25.0
        if row.pcd_points is not None and row.pcd_points < min_points:
            reasons.append("low_pcd_point_count")
            row.score += 25.0
        for field, field_stats in stats.get(row.dataset_class, {}).items():
            median, scale, log_value = field_stats
            z = robust_z(getattr(row, field), median, scale, log_value=log_value)
            if z >= outlier_z:
                reasons.append(f"{field}_outlier_z{z:.1f}")
                row.score += min(40.0, z * 4.0)

        row.reasons = ";".join(dict.fromkeys(reasons))


def write_csv(path: Path, rows: list[SampleAudit]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(SampleAudit.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def safe_link_name(row: SampleAudit, rank: int) -> str:
    reasons = row.reasons.split(";")[0] if row.reasons else "suspicious"
    reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reasons)[:45]
    return f"{rank:04d}_{row.dataset_class}_{reason}_{Path(row.dataset_pcd).name}"


def create_review_links(output: Path, rows: list[SampleAudit], limit: int) -> None:
    if limit <= 0:
        return
    review_dir = output / "review_links"
    review_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows[:limit], start=1):
        source = Path(row.dataset_pcd)
        if not source.exists():
            continue
        link = review_dir / safe_link_name(row, rank)
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(source, link)
        json_source = source.with_suffix(".json")
        if json_source.exists():
            json_link = link.with_suffix(".json")
            if json_link.exists() or json_link.is_symlink():
                json_link.unlink()
            os.symlink(json_source, json_link)


def write_report(path: Path, rows: list[SampleAudit], suspicious: list[SampleAudit], args: argparse.Namespace) -> None:
    class_counts = Counter(row.dataset_class for row in rows)
    reason_counts = Counter(
        reason for row in suspicious for reason in row.reasons.split(";") if reason
    )
    top = suspicious[:25]
    lines = [
        "# PointNeXt Curated Dataset Audit",
        "",
        f"Root: `{args.root.expanduser().resolve()}`",
        f"Samples scanned: {len(rows)}",
        f"Suspicious samples: {len(suspicious)}",
        "",
        "## Samples per class",
        "",
    ]
    for class_name, count in sorted(class_counts.items()):
        lines.append(f"- {class_name}: {count}")
    lines.extend(["", "## Most common reasons", ""])
    for reason, count in reason_counts.most_common(20):
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Top suspicious samples", ""])
    for row in top:
        lines.append(
            f"- score={row.score:.1f} class={row.dataset_class} conf={row.confidence} "
            f"sample=`{row.sample_id}` reasons={row.reasons}"
        )
    lines.extend(
        [
            "",
            "## Suggested workflow",
            "",
            "1. Open `high_priority_samples.csv` first; it filters out soft consensus-only warnings.",
            "2. Then use `suspicious_samples.csv` if you want the full broad audit queue.",
            "3. Inspect `review_links/` in a point-cloud viewer; links point back to the real dataset files.",
            "4. For stronger false-label detection, train a model on this curated dataset and run inference on the same root; high-confidence disagreements are the next best review queue.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_high_priority(row: SampleAudit) -> bool:
    hard_prefixes = (
        "dataset_class_ne_predicted_class",
        "low_confidence",
        "low_pcd_point_count",
        "non_finite_xyz",
        "empty_or_invalid_xyz",
        "pcd_read_error",
    )
    reasons = [reason for reason in row.reasons.split(";") if reason]
    if any(reason.startswith(hard_prefixes) for reason in reasons):
        return True
    return any("_outlier_z" in reason for reason in reasons) and row.score >= 80.0


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    output.mkdir(parents=True, exist_ok=True)

    rows = collect_samples(root)
    if not rows:
        raise SystemExit(f"No samples found below: {root}")
    add_scores(rows, args.low_confidence, args.min_points, args.outlier_z)
    rows.sort(key=lambda row: (-row.score, row.dataset_class, row.sample_id))
    suspicious = [row for row in rows if row.score > 0.0]
    high_priority = [row for row in suspicious if is_high_priority(row)]

    write_csv(output / "sample_audit.csv", rows)
    write_csv(output / "suspicious_samples.csv", suspicious)
    write_csv(output / "high_priority_samples.csv", high_priority)
    create_review_links(output, high_priority, args.review_links)
    write_report(output / "audit_report.md", rows, suspicious, args)

    print(f"Scanned samples: {len(rows)}")
    print(f"Suspicious samples: {len(suspicious)}")
    print(f"High-priority samples: {len(high_priority)}")
    print(f"Wrote: {output / 'suspicious_samples.csv'}")
    print(f"Wrote: {output / 'high_priority_samples.csv'}")
    print(f"Wrote: {output / 'audit_report.md'}")
    if args.review_links > 0:
        print(f"Review links: {output / 'review_links'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
