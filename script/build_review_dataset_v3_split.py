#!/usr/bin/env python3
"""Create flat train/val/test PCD datasets from review-dataset-v3."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    source: Path
    class_name: str
    object_id: str
    sample_id: str
    sidecar_json: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build flat 80/10/10 train/val/test datasets from review-dataset-v3. "
            "Aggregated PCDs and raw frame PCDs are written separately."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/georg/review-dataset-v3"),
        help="Path to review-dataset-v3.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/georg/review-dataset-v3-split"),
        help="Output root. Creates pcd/ and raw_frames/ below this path by default.",
    )
    parser.add_argument(
        "--aggregate-out",
        type=Path,
        default=None,
        help="Output root for aggregated PCDs. Default: OUT/pcd.",
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Output root for raw frame PCDs. Default: OUT/raw_frames.",
    )
    parser.add_argument(
        "--aggregate-kind",
        default="gt",
        choices=("gt", "pred"),
        help="Which aggregate subfolder to use from gt-pred-different.",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Split ratios per class. Default: 0.8 0.1 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic per-class shuffling.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize files in the new dataset. Default: copy.",
    )
    parser.add_argument(
        "--include-json",
        action="store_true",
        help="Also copy/link matching .json sidecars when present.",
    )
    parser.add_argument(
        "--only",
        choices=("aggregate", "raw", "both", "flat"),
        default="both",
        help="Which dataset(s) to create. Use flat for train/val/test/<class>/*.pcd inputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing output directories and replacing files.",
    )
    return parser.parse_args()


def normalize_ratios(values: tuple[float, float, float]) -> tuple[float, float, float]:
    if any(value < 0 for value in values):
        raise ValueError(f"Split ratios must be non-negative, got {values}.")
    total = sum(values)
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")
    return tuple(value / total for value in values)


def sidecar_for(path: Path) -> Path | None:
    sidecar = path.with_suffix(".json")
    return sidecar if sidecar.is_file() else None


def extract_object_id(path: Path) -> str:
    for part in path.stem.split("__"):
        if part.startswith("object_"):
            return part.removeprefix("object_")
    return "unknown"


def collect_aggregate_samples(src: Path, aggregate_kind: str) -> list[Sample]:
    samples: list[Sample] = []
    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir() or class_dir.name == "raw_frames":
            continue
        for pcd in sorted(class_dir.glob(f"*/gt-pred-different/{aggregate_kind}/*.pcd")):
            samples.append(
                Sample(
                    source=pcd,
                    class_name=class_dir.name,
                    object_id=extract_object_id(pcd),
                    sample_id=f"{pcd.parents[2].name}__{pcd.stem}",
                    sidecar_json=sidecar_for(pcd),
                )
            )
    return samples


def collect_raw_frame_samples(src: Path) -> list[Sample]:
    raw_root = src / "raw_frames"
    if not raw_root.is_dir():
        return []

    samples: list[Sample] = []
    for class_dir in sorted(raw_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for pcd in sorted(class_dir.glob("object_*/*.pcd")):
            object_id = pcd.parent.name.removeprefix("object_")
            samples.append(
                Sample(
                    source=pcd,
                    class_name=class_dir.name,
                    object_id=object_id,
                    sample_id=f"{pcd.parent.name}__{pcd.stem}",
                    sidecar_json=sidecar_for(pcd),
                )
            )
    return samples


def collect_flat_split_samples(src: Path) -> list[Sample]:
    samples: list[Sample] = []
    split_roots = [src / split for split in SPLITS if (src / split).is_dir()]
    if not split_roots:
        split_roots = [src]

    for split_root in split_roots:
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            for pcd in sorted(class_dir.glob("*.pcd")):
                samples.append(
                    Sample(
                        source=pcd,
                        class_name=class_dir.name,
                        object_id=extract_object_id(pcd),
                        sample_id=pcd.stem,
                        sidecar_json=sidecar_for(pcd),
                    )
                )
    return samples


def split_samples(
    samples: list[Sample],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Sample]]:
    objects_by_id: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        objects_by_id[sample.object_id].append(sample)

    grouped: dict[str, list[list[Sample]]] = defaultdict(list)
    for object_group in objects_by_id.values():
        class_key = "::".join(sorted({sample.class_name for sample in object_group}))
        grouped[class_key].append(object_group)

    split_to_samples = {split: [] for split in SPLITS}
    train_ratio, val_ratio, _ = ratios

    for class_key, object_groups in sorted(grouped.items()):
        rng = random.Random(f"{seed}:{class_key}")
        object_groups = list(object_groups)
        rng.shuffle(object_groups)

        total = len(object_groups)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        if total >= 3:
            train_count = max(1, train_count)
            val_count = max(1, val_count)
            if train_count + val_count >= total:
                train_count = max(1, total - 2)
                val_count = 1
        elif total == 2:
            train_count = 1
            val_count = 0
        elif total == 1:
            train_count = 1
            val_count = 0

        split_to_samples["train"].extend(
            sample
            for object_group in object_groups[:train_count]
            for sample in object_group
        )
        split_to_samples["val"].extend(
            sample
            for object_group in object_groups[train_count : train_count + val_count]
            for sample in object_group
        )
        split_to_samples["test"].extend(
            sample
            for object_group in object_groups[train_count + val_count :]
            for sample in object_group
        )

    return split_to_samples


def validate_no_object_leakage(split_to_samples: dict[str, list[Sample]]) -> None:
    object_to_splits: dict[str, set[str]] = defaultdict(set)
    for split, samples in split_to_samples.items():
        for sample in samples:
            object_to_splits[sample.object_id].add(split)

    leaks = {
        object_id: sorted(splits)
        for object_id, splits in object_to_splits.items()
        if len(splits) > 1
    }
    if leaks:
        preview = ", ".join(
            f"{object_id}:{'/'.join(splits)}"
            for object_id, splits in list(sorted(leaks.items()))[:10]
        )
        raise RuntimeError(f"Object-id leakage across splits detected: {preview}")


def ensure_clean_output(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to add/replace files.")
    path.mkdir(parents=True, exist_ok=True)


def unique_destination(base_dir: Path, source: Path, sample: Sample, used: set[Path]) -> Path:
    candidate = base_dir / source.name
    if candidate not in used and not candidate.exists():
        used.add(candidate)
        return candidate

    candidate = base_dir / f"{sample.sample_id}{source.suffix}"
    if candidate not in used and not candidate.exists():
        used.add(candidate)
        return candidate

    counter = 1
    while True:
        candidate = base_dir / f"{sample.sample_id}__dup{counter}{source.suffix}"
        if candidate not in used and not candidate.exists():
            used.add(candidate)
            return candidate
        counter += 1


def materialize(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        destination.hardlink_to(source)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def write_manifest(
    output_root: Path,
    dataset_name: str,
    assignments: dict[str, list[tuple[Sample, Path]]],
    args: argparse.Namespace,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for split in SPLITS:
            for sample, destination in assignments[split]:
                record = {
                    "dataset": dataset_name,
                    "split": split,
                    "class_name": sample.class_name,
                    "object_id": sample.object_id,
                    "source": str(sample.source),
                    "path": str(destination),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary_path = output_root / "summary.json"
    summary = {
        "dataset": dataset_name,
        "source": str(args.src),
        "output": str(output_root),
        "mode": args.mode,
        "seed": args.seed,
        "split_ratios": dict(zip(SPLITS, args.split)),
        "counts": {
            split: dict(sorted(Counter(sample.class_name for sample, _ in items).items()))
            for split, items in assignments.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_dataset(
    name: str,
    samples: list[Sample],
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    ensure_clean_output(output_root, args.overwrite, args.dry_run)
    split_to_samples = split_samples(samples, normalize_ratios(tuple(args.split)), args.seed)
    validate_no_object_leakage(split_to_samples)

    print(f"\n{name}: {len(samples)} PCD files -> {output_root}")
    assignments: dict[str, list[tuple[Sample, Path]]] = {split: [] for split in SPLITS}
    used: set[Path] = set()

    for split in SPLITS:
        counts = Counter(sample.class_name for sample in split_to_samples[split])
        total = sum(counts.values())
        print(f"  {split}: {total} files")
        for class_name, count in sorted(counts.items()):
            print(f"    {class_name}: {count}")

        for sample in split_to_samples[split]:
            class_dir = output_root / split / sample.class_name
            pcd_destination = unique_destination(class_dir, sample.source, sample, used)
            assignments[split].append((sample, pcd_destination))
            if args.dry_run:
                continue

            materialize(sample.source, pcd_destination, args.mode, args.overwrite)
            if args.include_json and sample.sidecar_json is not None:
                json_destination = pcd_destination.with_suffix(".json")
                materialize(sample.sidecar_json, json_destination, args.mode, args.overwrite)

    write_manifest(output_root, name, assignments, args, args.dry_run)


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    args.src = src
    args.out = args.out.expanduser().resolve()
    aggregate_out = (
        args.aggregate_out.expanduser().resolve()
        if args.aggregate_out is not None
        else args.out / "pcd"
    )
    raw_out = args.raw_out.expanduser().resolve() if args.raw_out is not None else args.out / "raw_frames"

    if not src.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    if args.only in {"aggregate", "both"}:
        aggregate_samples = collect_aggregate_samples(src, args.aggregate_kind)
        create_dataset(f"aggregate_{args.aggregate_kind}", aggregate_samples, aggregate_out, args)

    if args.only in {"raw", "both"}:
        raw_samples = collect_raw_frame_samples(src)
        create_dataset("raw_frames", raw_samples, raw_out, args)

    if args.only == "flat":
        flat_samples = collect_flat_split_samples(src)
        create_dataset("flat", flat_samples, args.out, args)


if __name__ == "__main__":
    main()
