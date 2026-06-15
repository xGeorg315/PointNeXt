#!/usr/bin/env python3
"""Audit alternative pseudo-label methods for point-cloud boundaries."""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.classification.dataloader import RawFramesClassificationDataset


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/georg/raw-frames"))
    parser.add_argument("--output-dir", type=Path, default=Path("boundary_label_methods_audit"))
    parser.add_argument("--reference-metrics", type=Path, default=Path("boundary_label_audit/boundary_metrics.json"))
    parser.add_argument("--samples-per-class", type=int, default=12)
    parser.add_argument("--visuals-per-class", type=int, default=3)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--curvature-threshold", type=float, default=0.08)
    parser.add_argument("--jitter-sigma", type=float, default=0.005)
    parser.add_argument("--jitter-clip", type=float, default=0.02)
    parser.add_argument("--jitter-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260614)
    return parser.parse_args()


def build_dataset(args):
    return RawFramesClassificationDataset(
        root_dir=args.dataset_root,
        split="train",
        num_points=args.num_points,
        start_date=None,
        min_points=0,
        split_ratios=(0.8, 0.1, 0.1),
        exclude_classes=(
            "reject",
            "TLS_VEHICLE_CAR_WITH_TRAILER",
            "TLS_VEHICLE_TRUCK_WITH_TRAILER",
        ),
        min_points_exempt_classes=(
            "TLS_VEHICLE_MOTORBIKE",
            "TLS_VEHICLE_TRAILER",
        ),
        forced_classes=DEFAULT_CLASSES,
        frame_selection="all",
        object_multi_view=True,
        max_views=5,
        view_selection="uniform",
        augment_train=False,
        augment_jitter_sigma=args.jitter_sigma,
        augment_jitter_clip=args.jitter_clip,
        preload_data=False,
    )


def selected_indices(dataset, args):
    if args.reference_metrics.is_file():
        rows = json.loads(args.reference_metrics.read_text(encoding="utf-8"))
        indices = sorted({int(row["dataset_index"]) for row in rows})
        if indices and max(indices) < len(dataset):
            return indices

    rng = np.random.default_rng(args.seed)
    by_class = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        by_class[sample["class_name"]].append(index)
    indices = []
    for class_name in dataset.classes:
        candidates = by_class[class_name]
        count = min(args.samples_per_class, len(candidates))
        indices.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return sorted(int(index) for index in indices)


def geometry_scores(points, k):
    neighbor_count = min(int(k), points.shape[0] - 1)
    distances = torch.cdist(points.unsqueeze(0), points.unsqueeze(0))[0]
    neighbor_distances, neighbor_indices = distances.topk(
        neighbor_count + 1, largest=False
    )
    neighbor_distances = neighbor_distances[:, 1:]
    neighbor_indices = neighbor_indices[:, 1:]
    neighbors = points[neighbor_indices]

    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered / float(neighbor_count)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    curvature = eigenvalues[:, 0] / eigenvalues.sum(dim=-1).clamp_min(1e-8)
    normals = eigenvectors[:, :, 0]

    distance_asymmetry = (
        neighbor_distances.std(dim=1, unbiased=False)
        / neighbor_distances.mean(dim=1).clamp_min(1e-8)
    )

    neighbor_normals = normals[neighbor_indices]
    signs = torch.sign((neighbor_normals * normals[:, None]).sum(dim=-1, keepdim=True))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    aligned_neighbor_normals = neighbor_normals * signs
    mean_neighbor_normal = aligned_neighbor_normals.mean(dim=1)
    mean_neighbor_normal = mean_neighbor_normal / torch.linalg.norm(
        mean_neighbor_normal, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    cosine = (normals * mean_neighbor_normal).sum(dim=-1).abs().clamp(0.0, 1.0)
    normal_difference_deg = torch.rad2deg(torch.acos(cosine))

    return {
        "curvature": curvature,
        "distance_asymmetry": distance_asymmetry,
        "normal_difference_deg": normal_difference_deg,
    }


def jaccard(left, right):
    union = (left | right).sum().item()
    return 1.0 if union == 0 else float((left & right).sum().item() / union)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def write_colored_ply(path, points, labels):
    xyz = points.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype(bool)
    colors = np.full((len(xyz), 3), 145, dtype=np.uint8)
    colors[labels] = np.array([255, 35, 35], dtype=np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, colors):
            handle.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def render_comparison(path, title, points, label_sets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = points.detach().cpu().numpy()
    fig, axes = plt.subplots(len(label_sets), 3, figsize=(14, 3.7 * len(label_sets)))
    for row, (method_name, labels) in enumerate(label_sets):
        labels = labels.detach().cpu().numpy().astype(bool)
        for column, (dims, projection) in enumerate(
            (((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ"))
        ):
            axis = axes[row, column]
            axis.scatter(
                xyz[~labels, dims[0]],
                xyz[~labels, dims[1]],
                s=4,
                c="#8f969e",
                alpha=0.5,
            )
            axis.scatter(
                xyz[labels, dims[0]],
                xyz[labels, dims[1]],
                s=10,
                c="#ff2323",
                alpha=0.95,
            )
            axis.set_title(
                f"{method_name} | {projection} | positive={labels.mean():.1%}"
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.15)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def threshold_summary(score_values, thresholds):
    values = np.concatenate(score_values)
    return {
        str(threshold): float(np.mean(values >= threshold))
        for threshold in thresholds
    }


def summarize_rows(rows, keys):
    summary = {"views": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
        }
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dataset = build_dataset(args)
    indices = selected_indices(dataset, args)

    records = []
    score_arrays = {
        "curvature": [],
        "distance_asymmetry": [],
        "normal_difference_deg": [],
    }
    cached_views = []
    visual_counts = defaultdict(int)

    for dataset_index in indices:
        item = dataset[dataset_index]
        for view_index in torch.nonzero(item["view_mask"], as_tuple=False).flatten().tolist():
            points = item["views"][view_index, :, :3].float()
            scores = geometry_scores(points, args.k)
            for name, values in scores.items():
                score_arrays[name].append(values.detach().cpu().numpy())
            cached_views.append(
                {
                    "dataset_index": dataset_index,
                    "view_index": view_index,
                    "class_name": dataset.samples[dataset_index]["class_name"],
                    "object_id": str(item["object_id"]),
                    "sample_id": str(item["sample_id"]),
                    "points": points,
                    "scores": scores,
                }
            )

    curvature_values = np.concatenate(score_arrays["curvature"])
    reference_positive_ratio = float(np.mean(curvature_values >= args.curvature_threshold))
    asymmetry_threshold = float(
        np.quantile(
            np.concatenate(score_arrays["distance_asymmetry"]),
            1.0 - reference_positive_ratio,
        )
    )
    normal_threshold = float(
        np.quantile(
            np.concatenate(score_arrays["normal_difference_deg"]),
            1.0 - reference_positive_ratio,
        )
    )
    thresholds = {
        "curvature": args.curvature_threshold,
        "distance_asymmetry": asymmetry_threshold,
        "normal_difference_deg": normal_threshold,
    }

    for view in cached_views:
        labels = {
            name: values >= thresholds[name]
            for name, values in view["scores"].items()
        }
        jitter_jaccards = defaultdict(list)
        for _ in range(args.jitter_repeats):
            noise = torch.randn_like(view["points"]).mul_(args.jitter_sigma)
            noise.clamp_(-args.jitter_clip, args.jitter_clip)
            jitter_scores = geometry_scores(view["points"] + noise, args.k)
            for name in labels:
                jitter_jaccards[name].append(
                    jaccard(labels[name], jitter_scores[name] >= thresholds[name])
                )

        row = {
            key: view[key]
            for key in (
                "dataset_index",
                "view_index",
                "class_name",
                "object_id",
                "sample_id",
            )
        }
        for name, values in view["scores"].items():
            row[f"{name}_mean"] = float(values.mean().item())
            row[f"{name}_median"] = float(values.median().item())
            row[f"{name}_p90"] = float(torch.quantile(values, 0.90).item())
            row[f"{name}_positive_ratio"] = float(labels[name].float().mean().item())
            row[f"{name}_jitter_jaccard"] = float(np.mean(jitter_jaccards[name]))
        row["asymmetry_vs_curvature_jaccard"] = jaccard(
            labels["distance_asymmetry"], labels["curvature"]
        )
        row["normal_vs_curvature_jaccard"] = jaccard(
            labels["normal_difference_deg"], labels["curvature"]
        )
        row["asymmetry_vs_normal_jaccard"] = jaccard(
            labels["distance_asymmetry"], labels["normal_difference_deg"]
        )
        records.append(row)

        class_name = view["class_name"]
        if (
            view["view_index"] == 0
            and visual_counts[class_name] < args.visuals_per_class
        ):
            visual_index = sum(visual_counts.values())
            stem = (
                f"{visual_index:02d}_{safe_name(class_name)}_"
                f"object_{safe_name(view['object_id'])}"
            )
            label_sets = [
                ("Current curvature", labels["curvature"]),
                ("kNN distance CV", labels["distance_asymmetry"]),
                ("PCA normal difference", labels["normal_difference_deg"]),
            ]
            render_comparison(
                args.output_dir / f"{stem}_comparison.png",
                f"{class_name} | object {view['object_id']} | k={args.k}",
                view["points"],
                label_sets,
            )
            for method_name, method_labels in (
                ("distance_asymmetry", labels["distance_asymmetry"]),
                ("normal_difference", labels["normal_difference_deg"]),
            ):
                write_colored_ply(
                    args.output_dir / f"{stem}_{method_name}.ply",
                    view["points"],
                    method_labels,
                )
            visual_counts[class_name] += 1

    metrics_path = args.output_dir / "view_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_keys = [
        "curvature_positive_ratio",
        "curvature_jitter_jaccard",
        "distance_asymmetry_positive_ratio",
        "distance_asymmetry_jitter_jaccard",
        "normal_difference_deg_positive_ratio",
        "normal_difference_deg_jitter_jaccard",
        "asymmetry_vs_curvature_jaccard",
        "normal_vs_curvature_jaccard",
        "asymmetry_vs_normal_jaccard",
    ]
    summary = {
        "dataset_samples": len(dataset),
        "audited_objects": len(indices),
        "audited_views": len(records),
        "num_points": args.num_points,
        "k": args.k,
        "threshold_policy": "Matched to current curvature global positive ratio",
        "thresholds": thresholds,
        "score_quantiles": {
            name: {
                f"p{quantile:02d}": float(np.quantile(np.concatenate(values), quantile / 100))
                for quantile in (10, 25, 50, 75, 85, 90, 95, 99)
            }
            for name, values in score_arrays.items()
        },
        "threshold_sweeps": {
            "distance_asymmetry": threshold_summary(
                score_arrays["distance_asymmetry"],
                (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00),
            ),
            "normal_difference_deg": threshold_summary(
                score_arrays["normal_difference_deg"],
                (2, 5, 10, 15, 20, 25, 30, 40, 50, 60),
            ),
        },
        "overall": summarize_rows(records, summary_keys),
        "classes": {},
    }
    for class_name in dataset.classes:
        class_rows = [row for row in records if row["class_name"] == class_name]
        if class_rows:
            summary["classes"][class_name] = summarize_rows(class_rows, summary_keys)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Audited objects: {len(indices)}")
    print(f"Audited views: {len(records)}")
    print(f"Reference positive ratio: {reference_positive_ratio:.4f}")
    print(f"Distance-asymmetry threshold: {asymmetry_threshold:.6f}")
    print(f"Normal-difference threshold: {normal_threshold:.3f} degrees")
    for key in (
        "curvature_jitter_jaccard",
        "distance_asymmetry_jitter_jaccard",
        "normal_difference_deg_jitter_jaccard",
        "asymmetry_vs_normal_jaccard",
    ):
        print(f"{key}: {summary['overall'][key]['mean']:.4f}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
