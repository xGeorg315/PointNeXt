#!/usr/bin/env python3
"""Build a PointNeXt flat split dataset from an inference CSV.

The CSV is expected to contain paths to PCD files plus class columns such as
current_class and predicted_class. Rows are selected by configurable rules:

  {
    "target_column": "predicted_class",
    "source_column": "folder_class",
    "rules": {
      "TLS_VEHICLE_CAR": {
        "allow_sources": ["TLS_VEHICLE_CAR"],
        "require_equal": ["folder_class", "class_name", "predicted_class"]
      },
      "TLS_VEHICLE_TRAILER": {
        "allow_sources": ["*"],
        "require_equal": ["predicted_class"]
      },
      "*": []
    }
  }

In this example, CAR samples must agree across folder class, JSON class, and
new prediction. TRAILER samples can come from any source folder and need either
the JSON class or the new prediction to be TRAILER.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")
SUPPORTED_COLUMNS = {
    "bucket",
    "current_class",
    "predicted_class",
    "folder_class",
    "is_match",
}
COLUMN_ALIASES = {
    "class_name": "current_class",
    "json_class": "current_class",
}
CLI_COLUMNS = SUPPORTED_COLUMNS | set(COLUMN_ALIASES)


@dataclass(frozen=True)
class RuleConfig:
    allow_sources: set[str]
    disallow_sources: set[str] | None = None
    require_equal: tuple[str, ...] = ()
    match_mode: str = "and"


@dataclass(frozen=True)
class SelectedSample:
    row: dict[str, str]
    source_pcd: Path
    source_json: Path | None
    target_class: str
    source_class: str
    object_id: str
    sample_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a configurable train/val/test PCD dataset from a PointNeXt "
            "inference CSV."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/georg/new-big-data/pointnext_seed5655_predictions.csv"),
        help="Inference CSV containing at least pcd_path and class columns.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/georg/new-big-data/pointnext_seed5655_curated_dataset"),
        help="Output root. Creates train/val/test class folders below it.",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=None,
        help=(
            "JSON rules. Either a target-to-source map or an object with "
            "target_column/source_column/rules."
        ),
    )
    parser.add_argument(
        "--target-column",
        default="predicted_class",
        choices=sorted(CLI_COLUMNS - {"is_match"}),
        help="CSV-derived value used as output class.",
    )
    parser.add_argument(
        "--source-column",
        default="folder_class",
        choices=sorted(CLI_COLUMNS),
        help="CSV-derived value checked against the per-target allow-list.",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Per-class object split ratios. Use '1 0 0' for all train.",
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
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=10000,
        help="Maximum selected samples per target class. Use 0 for unlimited.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional: stop after N selected rows.")
    parser.add_argument(
        "--buckets",
        nargs="*",
        default=None,
        help="Optional bucket allow-list, e.g. gt-pred-diff gt-pred-same.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "matches", "mismatches"),
        default="all",
        help="Filter by is_match column before class rules are applied.",
    )
    return parser.parse_args()


def normalize_column(column: str) -> str:
    column = str(column).strip()
    return COLUMN_ALIASES.get(column, column)


def parse_name_set(value: Any) -> set[str]:
    if value == "*":
        return {"*"}
    if value in (None, False):
        return set()
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def parse_column_list(value: Any) -> tuple[str, ...]:
    if value in (None, False):
        return ()
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    columns = tuple(normalize_column(item) for item in values)
    unsupported = [column for column in columns if column not in SUPPORTED_COLUMNS - {"is_match"}]
    if unsupported:
        raise ValueError(f"Unsupported require_equal column(s): {unsupported}")
    return columns


def load_rules(path: Path | None, args: argparse.Namespace) -> tuple[str, str, dict[str, RuleConfig]]:
    target_column = normalize_column(args.target_column)
    source_column = normalize_column(args.source_column)
    if path is None:
        return target_column, source_column, {"*": RuleConfig(allow_sources={"*"})}

    with path.expanduser().open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    rules_raw: Any = raw
    if isinstance(raw, dict) and "rules" in raw:
        target_column = normalize_column(raw.get("target_column") or target_column)
        source_column = normalize_column(raw.get("source_column") or source_column)
        rules_raw = raw["rules"]

    if target_column not in SUPPORTED_COLUMNS - {"is_match"}:
        raise ValueError(f"Unsupported target_column: {target_column}")
    if source_column not in SUPPORTED_COLUMNS:
        raise ValueError(f"Unsupported source_column: {source_column}")
    if not isinstance(rules_raw, dict):
        raise ValueError("Rules must be a JSON object.")

    rules: dict[str, RuleConfig] = {}
    for target, raw_rule in rules_raw.items():
        if isinstance(raw_rule, dict):
            allowed = parse_name_set(
                raw_rule.get(
                    "allow_sources",
                    raw_rule.get("sources", raw_rule.get("allowed_sources", "*")),
                )
            )
            disallowed = parse_name_set(
                raw_rule.get(
                    "disallow_sources",
                    raw_rule.get("deny_sources", raw_rule.get("disallowed_sources", ())),
                )
            )
            require_equal = parse_column_list(
                raw_rule.get(
                    "require_equal",
                    raw_rule.get("must_equal", raw_rule.get("strict_columns", ())),
                )
            )
            match_mode = str(raw_rule.get("match_mode", raw_rule.get("mode", "and"))).lower()
            if match_mode not in {"and", "or"}:
                raise ValueError(f"Unsupported match_mode for {target}: {match_mode}")
        else:
            # Backward-compatible shorthand: "TARGET": ["SOURCE_A", "SOURCE_B"]
            allowed = parse_name_set(raw_rule)
            disallowed = set()
            require_equal = ()
            match_mode = "and"
        rules[str(target)] = RuleConfig(
            allow_sources=allowed,
            disallow_sources=disallowed,
            require_equal=require_equal,
            match_mode=match_mode,
        )
    return target_column, source_column, rules

def csv_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def folder_class_from_pcd(path: Path) -> str:
    # Expected: .../<bucket>/<class>/pcds/<sample>.pcd
    if path.parent.name == "pcds" and path.parent.parent.name:
        return path.parent.parent.name
    return path.parent.name


def row_value(row: dict[str, str], column: str) -> str:
    if column == "folder_class":
        return folder_class_from_pcd(Path(row["pcd_path"]))
    return str(row.get(column, "")).strip()


def object_id_from_row(row: dict[str, str], sample_id: str) -> str:
    object_id = str(row.get("object_id", "")).strip()
    if object_id:
        return object_id
    for part in sample_id.split("__"):
        if part.startswith("object_"):
            return part[len("object_") :]
    return sample_id


def sidecar_json(row: dict[str, str], pcd_path: Path) -> Path | None:
    json_value = str(row.get("json_path", "")).strip()
    if json_value:
        candidate = Path(json_value)
        if candidate.is_file():
            return candidate
    candidate = pcd_path.with_suffix(".json")
    return candidate if candidate.is_file() else None


def rule_for_target(target: str, rules: dict[str, RuleConfig]) -> RuleConfig:
    return rules.get(target, rules.get("*", RuleConfig(allow_sources=set())))


def allowed_by_rules(
    row: dict[str, str],
    target: str,
    source: str,
    rules: dict[str, RuleConfig],
) -> bool:
    rule = rule_for_target(target, rules)
    disallowed = rule.disallow_sources or set()
    if "*" in disallowed or source in disallowed:
        return False
    if not ("*" in rule.allow_sources or source in rule.allow_sources):
        return False
    if not rule.require_equal:
        return True

    matches = [row_value(row, column) == target for column in rule.require_equal]
    if rule.match_mode == "or":
        return any(matches)
    return all(matches)


def collect_samples(
    csv_path: Path,
    target_column: str,
    source_column: str,
    rules: dict[str, RuleConfig],
    args: argparse.Namespace,
) -> list[SelectedSample]:
    selected = []
    bucket_allow = set(args.buckets or [])

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if bucket_allow and row.get("bucket") not in bucket_allow:
                continue
            if args.only == "matches" and not csv_bool(row.get("is_match", "")):
                continue
            if args.only == "mismatches" and csv_bool(row.get("is_match", "")):
                continue
            try:
                confidence = float(row.get("confidence", "") or 0.0)
            except ValueError:
                confidence = 0.0
            if confidence < args.min_confidence:
                continue

            pcd_path = Path(row["pcd_path"]).expanduser()
            if not pcd_path.is_file():
                continue

            target = row_value(row, target_column)
            source = row_value(row, source_column)
            if not target or not source or not allowed_by_rules(row, target, source, rules):
                continue

            sample_id = str(row.get("sample_id") or pcd_path.stem)
            selected.append(
                SelectedSample(
                    row=row,
                    source_pcd=pcd_path,
                    source_json=sidecar_json(row, pcd_path),
                    target_class=target,
                    source_class=source,
                    object_id=object_id_from_row(row, sample_id),
                    sample_id=sample_id,
                )
            )
            if args.limit and len(selected) >= args.limit:
                break
    return selected


def cap_samples_per_class(
    samples: list[SelectedSample],
    max_per_class: int,
    seed: int,
) -> tuple[list[SelectedSample], dict[str, dict[str, int]]]:
    if max_per_class <= 0:
        counts = Counter(sample.target_class for sample in samples)
        return samples, {
            class_name: {"before": count, "after": count}
            for class_name, count in sorted(counts.items())
        }

    grouped: dict[str, list[SelectedSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.target_class].append(sample)

    capped = []
    cap_summary = {}
    for class_name, class_samples in sorted(grouped.items()):
        before = len(class_samples)
        if before > max_per_class:
            rng = random.Random(f"{seed}:max_per_class:{class_name}")
            class_samples = list(class_samples)
            rng.shuffle(class_samples)
            class_samples = class_samples[:max_per_class]
        # Keep deterministic order after random selection so file creation is stable.
        class_samples = sorted(class_samples, key=lambda sample: sample.sample_id)
        capped.extend(class_samples)
        cap_summary[class_name] = {"before": before, "after": len(class_samples)}
    return capped, cap_summary


def normalize_ratios(values: tuple[float, float, float]) -> tuple[float, float, float]:
    if any(value < 0 for value in values):
        raise ValueError(f"Split ratios must be non-negative, got {values}")
    total = sum(values)
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")
    return tuple(value / total for value in values)


def split_samples(
    samples: list[SelectedSample],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[SelectedSample]]:
    train_ratio, val_ratio, _ = ratios
    grouped: dict[str, list[list[SelectedSample]]] = defaultdict(list)
    by_object: dict[tuple[str, str], list[SelectedSample]] = defaultdict(list)

    for sample in samples:
        by_object[(sample.target_class, sample.object_id)].append(sample)
    for (target_class, _), object_samples in by_object.items():
        grouped[target_class].append(object_samples)

    split_to_samples = {split: [] for split in SPLITS}
    for target_class, object_groups in sorted(grouped.items()):
        rng = random.Random(f"{seed}:{target_class}")
        object_groups = list(object_groups)
        rng.shuffle(object_groups)

        total = len(object_groups)
        train_count = int(total * train_ratio) if train_ratio > 0 else 0
        val_count = int(total * val_ratio) if val_ratio > 0 else 0

        if total >= 3:
            if train_ratio > 0:
                train_count = max(1, train_count)
            if val_ratio > 0:
                val_count = max(1, val_count)
            if train_count + val_count > total:
                overflow = train_count + val_count - total
                if val_count >= overflow:
                    val_count -= overflow
                else:
                    train_count = max(0, train_count - (overflow - val_count))
                    val_count = 0
        elif total == 2:
            train_count = min(2, 1 if train_ratio > 0 else 0)
            val_count = 1 if val_ratio > 0 and train_count < 2 else 0
        elif total == 1:
            train_count = 1 if train_ratio > 0 else 0
            val_count = 1 if train_count == 0 and val_ratio > 0 else 0

        split_to_samples["train"].extend(
            sample for group in object_groups[:train_count] for sample in group
        )
        split_to_samples["val"].extend(
            sample
            for group in object_groups[train_count : train_count + val_count]
            for sample in group
        )
        split_to_samples["test"].extend(
            sample for group in object_groups[train_count + val_count :] for sample in group
        )
    return split_to_samples


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


def unique_destination(class_dir: Path, sample: SelectedSample, used: set[Path], suffix: str) -> Path:
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
    split_to_samples: dict[str, list[SelectedSample]],
    assignments: dict[str, list[tuple[SelectedSample, Path]]],
    target_column: str,
    source_column: str,
    rules: dict[str, RuleConfig],
    cap_summary: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> None:
    manifest_path = out / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for split in SPLITS:
            for sample, destination in assignments[split]:
                record = dict(sample.row)
                record.update(
                    {
                        "split": split,
                        "dataset_class": sample.target_class,
                        "rule_source_class": sample.source_class,
                        "source_pcd": str(sample.source_pcd),
                        "dataset_pcd": str(destination),
                    }
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "csv": str(args.csv),
        "output": str(out),
        "mode": args.mode,
        "seed": args.seed,
        "split": dict(zip(SPLITS, args.split)),
        "max_per_class": args.max_per_class,
        "target_column": target_column,
        "source_column": source_column,
        "rules": {
            key: {
                "allow_sources": sorted(value.allow_sources),
                "disallow_sources": sorted(value.disallow_sources or set()),
                "require_equal": list(value.require_equal),
                "match_mode": value.match_mode,
            }
            for key, value in sorted(rules.items())
        },
        "cap_summary": cap_summary,
        "counts": {
            split: dict(sorted(Counter(sample.target_class for sample in samples).items()))
            for split, samples in split_to_samples.items()
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.csv = args.csv.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    target_column, source_column, rules = load_rules(args.rules, args)

    samples = collect_samples(args.csv, target_column, source_column, rules, args)
    if not samples:
        raise SystemExit("No rows matched the configured filters/rules.")

    selected_before_cap = len(samples)
    samples, cap_summary = cap_samples_per_class(samples, args.max_per_class, args.seed)

    split_to_samples = split_samples(samples, normalize_ratios(tuple(args.split)), args.seed)
    assignments: dict[str, list[tuple[SelectedSample, Path]]] = {split: [] for split in SPLITS}
    used: set[Path] = set()

    print(f"CSV: {args.csv}")
    print(f"Output: {args.out}")
    print(f"Target column: {target_column}")
    print(f"Source column: {source_column}")
    print(f"Selected rows before cap: {selected_before_cap}")
    print(f"Selected rows after cap: {len(samples)}")
    print(f"Max per class: {args.max_per_class if args.max_per_class > 0 else 'unlimited'}")

    for split in SPLITS:
        counts = Counter(sample.target_class for sample in split_to_samples[split])
        print(f"\n{split}: {sum(counts.values())}")
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count}")

        for sample in split_to_samples[split]:
            class_dir = args.out / split / sample.target_class
            pcd_destination = unique_destination(class_dir, sample, used, sample.source_pcd.suffix)
            assignments[split].append((sample, pcd_destination))
            if args.dry_run:
                continue
            materialize(sample.source_pcd, pcd_destination, args.mode, args.overwrite)
            if args.include_json and sample.source_json is not None:
                materialize(
                    sample.source_json,
                    pcd_destination.with_suffix(".json"),
                    args.mode,
                    args.overwrite,
                )

    if not args.dry_run:
        write_manifest(
            args.out,
            split_to_samples,
            assignments,
            target_column,
            source_column,
            rules,
            cap_summary,
            args,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
