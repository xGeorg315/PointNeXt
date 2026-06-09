#!/usr/bin/env python3
"""Build a PointNeXt split dataset using review samples for val/test.

The curated dataset is treated as trusted train material. Samples from the
review dataset are assigned to val/test first, according to the requested final
split ratios. If review has more samples than val/test need, the remainder is
placed into train.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test")
OBJECT_RE = re.compile(r"(?:^|__)object_([^_]+)")
DEFAULT_EXCLUDE_CLASSES = (
    "TLS_VEHICLE_CAR_WITH_TRAILER",
    "TLS_VEHICLE_TRUCK_WITH_TRAILER",
)


@dataclass(frozen=True)
class Sample:
    source: str
    class_name: str
    object_id: str
    sample_id: str
    pcd_path: Path
    json_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val/test folders from a curated train dataset and a "
            "review dataset, reserving review samples for val/test first."
        )
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=Path("/home/georg/new-big-data/pointnext_seed5655_curated_dataset"),
        help="Existing flat PointNeXt dataset. All samples are eligible for train.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("/home/georg/review-dataset-v3"),
        help="Review dataset root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/georg/new-big-data/pointnext_seed5655_curated_review_split_dataset"),
        help="Output root. Creates train/val/test class folders below it.",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Desired final per-class sample ratios.",
    )
    parser.add_argument("--seed", type=int, default=5655)
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize PCD/JSON files.",
    )
    parser.add_argument("--include-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--review-leaf",
        action="append",
        default=["gt"],
        help=(
            "Review leaf folder name to include. Default: gt. Can be repeated, "
            "for example --review-leaf gt --review-leaf gt_matching."
        ),
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON object mapping source class names to dataset class "
            "names, e.g. {\"TLS_VEHICLE_CAR_WITH_TRAILER\": \"TLS_VEHICLE_TRAILER\"}."
        ),
    )
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=list(DEFAULT_EXCLUDE_CLASSES),
        help="Class name to exclude after class mapping. Can be repeated.",
    )
    return parser.parse_args()


def normalize_ratios(values: Iterable[float]) -> tuple[float, float, float]:
    ratios = tuple(float(value) for value in values)
    if len(ratios) != 3:
        raise ValueError(f"Expected three split ratios, got {ratios}")
    if any(value < 0 for value in ratios):
        raise ValueError(f"Split ratios must be non-negative, got {ratios}")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")
    return tuple(value / total for value in ratios)


def load_class_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("--class-map must point to a JSON object.")
    return {str(key): str(value) for key, value in raw.items()}


def object_id_from_name(stem: str) -> str:
    match = OBJECT_RE.search(stem)
    return match.group(1) if match else stem


def sidecar_json(pcd_path: Path) -> Path | None:
    candidate = pcd_path.with_suffix(".json")
    return candidate if candidate.is_file() else None


def mapped_class(class_name: str, class_map: dict[str, str]) -> str:
    return class_map.get(class_name, class_name)


def collect_curated_samples(root: Path, class_map: dict[str, str]) -> list[Sample]:
    samples: list[Sample] = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for pcd_path in sorted(split_dir.glob("*/*.pcd")):
            class_name = mapped_class(pcd_path.parent.name, class_map)
            samples.append(
                Sample(
                    source=f"curated:{split}",
                    class_name=class_name,
                    object_id=object_id_from_name(pcd_path.stem),
                    sample_id=pcd_path.stem,
                    pcd_path=pcd_path,
                    json_path=sidecar_json(pcd_path),
                )
            )
    return samples


def collect_review_samples(
    root: Path,
    leaf_names: set[str],
    class_map: dict[str, str],
) -> list[Sample]:
    samples: list[Sample] = []
    for pcd_path in sorted(root.glob("*/*/*/*/*.pcd")):
        try:
            relative = pcd_path.relative_to(root)
        except ValueError:
            continue
        parts = relative.parts
        if not parts or parts[0] == "raw_frames":
            continue
        if pcd_path.parent.name not in leaf_names:
            continue
        class_name = mapped_class(parts[0], class_map)
        samples.append(
            Sample(
                source="review",
                class_name=class_name,
                object_id=object_id_from_name(pcd_path.stem),
                sample_id=pcd_path.stem,
                pcd_path=pcd_path,
                json_path=sidecar_json(pcd_path),
            )
        )
    return samples


def desired_eval_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    _, val_ratio, test_ratio = ratios
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    if val_ratio > 0 and total >= 3:
        val_count = max(1, val_count)
    if test_ratio > 0 and total >= 3:
        test_count = max(1, test_count)
    if val_count + test_count > total:
        overflow = val_count + test_count - total
        test_reduce = min(test_count, overflow)
        test_count -= test_reduce
        val_count -= overflow - test_reduce
    return val_count, test_count


def assign_splits(
    curated_samples: list[Sample],
    review_samples: list[Sample],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Sample]]:
    split_to_samples = {split: [] for split in SPLITS}
    curated_by_class: dict[str, list[Sample]] = defaultdict(list)
    review_by_class: dict[str, list[Sample]] = defaultdict(list)

    for sample in curated_samples:
        curated_by_class[sample.class_name].append(sample)
    for sample in review_samples:
        review_by_class[sample.class_name].append(sample)

    for class_name in sorted(set(curated_by_class) | set(review_by_class)):
        curated = sorted(curated_by_class[class_name], key=lambda item: item.sample_id)
        review = list(review_by_class[class_name])
        rng = random.Random(f"{seed}:{class_name}:review")
        rng.shuffle(review)

        total = len(curated) + len(review)
        desired_val, desired_test = desired_eval_counts(total, ratios)

        eval_capacity = desired_val + desired_test
        eval_count = min(len(review), eval_capacity)
        val_count, test_count = split_review_eval_count(
            eval_count,
            desired_val,
            desired_test,
            ratios,
        )

        val_samples = review[:val_count]
        test_samples = review[val_count : val_count + test_count]
        overflow_train_samples = review[val_count + test_count :]

        split_to_samples["train"].extend(curated)
        split_to_samples["train"].extend(overflow_train_samples)
        split_to_samples["val"].extend(val_samples)
        split_to_samples["test"].extend(test_samples)

    for split in SPLITS:
        split_to_samples[split].sort(key=lambda item: (item.class_name, item.sample_id, str(item.pcd_path)))
    return split_to_samples


def split_review_eval_count(
    eval_count: int,
    desired_val: int,
    desired_test: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int]:
    if eval_count <= 0:
        return 0, 0
    _, val_ratio, test_ratio = ratios
    eval_ratio = val_ratio + test_ratio
    if eval_ratio <= 0:
        return 0, 0

    val_count = int((eval_count * val_ratio / eval_ratio) + 0.5)
    test_count = eval_count - val_count

    if val_ratio > 0 and test_ratio > 0 and eval_count >= 2:
        val_count = max(1, val_count)
        test_count = max(1, test_count)
    if val_count > desired_val:
        overflow = val_count - desired_val
        val_count = desired_val
        test_count += overflow
    if test_count > desired_test:
        overflow = test_count - desired_test
        test_count = desired_test
        val_count += overflow

    val_count = min(val_count, desired_val, eval_count)
    test_count = min(test_count, desired_test, eval_count - val_count)
    return val_count, test_count


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


def unique_destination(class_dir: Path, sample: Sample, used: set[Path], suffix: str) -> Path:
    candidate = class_dir / f"{sample.sample_id}{suffix}"
    if candidate not in used and not candidate.exists():
        used.add(candidate)
        return candidate

    counter = 1
    while True:
        candidate = class_dir / f"{sample.sample_id}__dup{counter}{suffix}"
        if candidate not in used and not candidate.exists():
            used.add(candidate)
            return candidate
        counter += 1


def write_manifest(
    out: Path,
    split_to_samples: dict[str, list[Sample]],
    assignments: dict[str, list[tuple[Sample, Path]]],
    args: argparse.Namespace,
) -> None:
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for split in SPLITS:
            for sample, destination in assignments[split]:
                record = {
                    "split": split,
                    "source": sample.source,
                    "class_name": sample.class_name,
                    "object_id": sample.object_id,
                    "sample_id": sample.sample_id,
                    "source_pcd": str(sample.pcd_path),
                    "dataset_pcd": str(destination),
                }
                if sample.json_path is not None:
                    record["source_json"] = str(sample.json_path)
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    counts = {
        split: dict(sorted(Counter(sample.class_name for sample in split_to_samples[split]).items()))
        for split in SPLITS
    }
    source_counts = {
        split: dict(sorted(Counter(sample.source for sample in split_to_samples[split]).items()))
        for split in SPLITS
    }
    summary = {
        "curated": str(args.curated),
        "review": str(args.review),
        "output": str(args.out),
        "mode": args.mode,
        "seed": args.seed,
        "split": dict(zip(SPLITS, args.split)),
        "review_leaf": sorted(set(args.review_leaf)),
        "class_map": str(args.class_map) if args.class_map else None,
        "exclude_class": sorted(set(args.exclude_class)),
        "counts": counts,
        "source_counts": source_counts,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_counts(title: str, samples: list[Sample]) -> None:
    counts = Counter(sample.class_name for sample in samples)
    print(f"\n{title}: {sum(counts.values())}")
    for class_name, count in sorted(counts.items()):
        print(f"  {class_name}: {count}")


def main() -> int:
    args = parse_args()
    args.curated = args.curated.expanduser().resolve()
    args.review = args.review.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    ratios = normalize_ratios(args.split)
    class_map = load_class_map(args.class_map)
    exclude_classes = set(args.exclude_class)

    curated_samples = [
        sample
        for sample in collect_curated_samples(args.curated, class_map)
        if sample.class_name not in exclude_classes
    ]
    review_samples = [
        sample
        for sample in collect_review_samples(args.review, set(args.review_leaf), class_map)
        if sample.class_name not in exclude_classes
    ]
    if not curated_samples and not review_samples:
        raise SystemExit("No samples found.")

    split_to_samples = assign_splits(curated_samples, review_samples, ratios, args.seed)
    assignments: dict[str, list[tuple[Sample, Path]]] = {split: [] for split in SPLITS}
    used: set[Path] = set()

    print(f"Curated: {args.curated}")
    print(f"Review: {args.review}")
    print(f"Output: {args.out}")
    print(f"Mode: {args.mode}")
    print(f"Review leaf folders: {', '.join(sorted(set(args.review_leaf)))}")
    print_counts("Curated samples", curated_samples)
    print_counts("Review samples", review_samples)

    for split in SPLITS:
        print_counts(split, split_to_samples[split])
        for sample in split_to_samples[split]:
            class_dir = args.out / split / sample.class_name
            pcd_destination = unique_destination(class_dir, sample, used, sample.pcd_path.suffix)
            assignments[split].append((sample, pcd_destination))
            if args.dry_run:
                continue
            materialize(sample.pcd_path, pcd_destination, args.mode, args.overwrite)
            if args.include_json and sample.json_path is not None:
                materialize(
                    sample.json_path,
                    pcd_destination.with_suffix(".json"),
                    args.mode,
                    args.overwrite,
                )

    if not args.dry_run:
        write_manifest(args.out, split_to_samples, assignments, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
