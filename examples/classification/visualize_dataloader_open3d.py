import __init__
import argparse
from collections import defaultdict

import numpy as np

from examples.classification.dataloader import PointCloudDataset


def _parse_classes(raw_classes):
    if not raw_classes:
        return None
    classes = []
    for token in raw_classes:
        parts = [p.strip() for p in token.split(",") if p.strip()]
        for part in parts:
            if part.lstrip("-").isdigit():
                classes.append(int(part))
            else:
                classes.append(part)
    return classes


def _resolve_target_class_indices(dataset, classes):
    class_to_idx = getattr(dataset, "class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if classes is None:
        target_idxs = sorted(idx_to_class.keys())
        return target_idxs, idx_to_class

    target_idxs = []
    for c in classes:
        if isinstance(c, int):
            target_idxs.append(c)
        else:
            if c not in class_to_idx:
                raise ValueError(f"Unbekannte Klasse: {c}")
            target_idxs.append(class_to_idx[c])
    return sorted(set(target_idxs)), idx_to_class


def _build_indices_per_class(dataset, target_idxs):
    indices_per_class = defaultdict(list)
    for ds_idx, (_, label) in enumerate(dataset.samples):
        if label in target_idxs:
            indices_per_class[label].append(ds_idx)
    return indices_per_class


def _collect_samples(dataset, target_idxs, samples_per_class, indices_per_class, cursors):
    collected = defaultdict(list)
    for class_idx in target_idxs:
        class_indices = list(indices_per_class[class_idx])
        if not class_indices:
            continue
        for _ in range(samples_per_class):
            ds_idx = class_indices[cursors[class_idx] % len(class_indices)]
            cursors[class_idx] += 1
            file_path, _ = dataset.samples[ds_idx]
            raw_points = np.load(file_path)
            original_points = int(raw_points.shape[0])

            sampled_points, _ = dataset[ds_idx]
            sampled_points = sampled_points.detach().cpu().numpy()[:, :3]

            collected[class_idx].append(
                {
                    "points": sampled_points,
                    "original_points": original_points,
                    "resampled_points": int(sampled_points.shape[0]),
                    "file_path": str(file_path),
                }
            )

    return collected


def _build_geometries(
    collected,
    target_idxs,
    idx_to_class,
    samples_per_class,
    x_spacing,
    y_spacing,
    show_axis,
    o3d,
):
    geometries = []
    merged_points = []
    total_points = 0
    per_class_points = defaultdict(int)
    total_samples = 0
    print("\nGrid-Layout (row, col -> Klasse/Sample):")

    for row, class_idx in enumerate(target_idxs):
        class_name = idx_to_class.get(class_idx, f"class_{class_idx}")
        samples = collected[class_idx]
        for col in range(samples_per_class):
            if col >= len(samples):
                continue

            sample = samples[col]
            pts = np.array(sample["points"], dtype=np.float64, copy=True)
            finite_mask = np.isfinite(pts).all(axis=1)
            if not finite_mask.all():
                pts = pts[finite_mask]
            pts[:, 0] += col * x_spacing
            pts[:, 1] -= row * y_spacing

            merged_points.append(pts)

            if show_axis:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                axis.translate([col * x_spacing, -row * y_spacing, 0.0])
                geometries.append(axis)

            print(
                f"  ({row}, {col}) -> {class_name} (idx={class_idx}), sample={col + 1}"
            )
            print(f"      original points: {sample['original_points']}")
            print(f"      resampled points: {sample['resampled_points']}")
            print(f"      shown points: {pts.shape[0]}")
            per_class_points[class_idx] += int(sample["original_points"])
            total_points += int(sample["original_points"])
            total_samples += 1

    print("\nPunkt-Summen:")
    for class_idx in target_idxs:
        class_name = idx_to_class.get(class_idx, f"class_{class_idx}")
        print(f"  {class_name} (idx={class_idx}): {per_class_points[class_idx]}")
    print(f"  Gesamtpunkte ORIGINAL (alle visualisierten Samples): {total_points}")
    print(f"  Anzahl visualisierte Samples: {total_samples}")

    if merged_points:
        merged = np.concatenate(merged_points, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)
        pcd.paint_uniform_color([0.68, 0.85, 1.0])
        geometries.insert(0, pcd)

    return geometries


def visualize_classes_open3d(args):
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(f"open3d ist nicht installiert oder nicht nutzbar: {e}") from e

    num_points = args.point_count if args.point_count is not None else args.num_points
    print(f"Configured points per sample: {num_points}")

    dataset = PointCloudDataset(
        root_dir=args.dataset_root,
        split=args.split,
        num_points=num_points,
    )

    class_filter = _parse_classes(args.classes)
    target_idxs, idx_to_class = _resolve_target_class_indices(dataset, class_filter)
    if not target_idxs:
        raise RuntimeError("Keine Klassen gefunden.")

    indices_per_class = _build_indices_per_class(dataset, target_idxs)
    missing = [idx for idx in target_idxs if len(indices_per_class[idx]) == 0]
    if missing:
        names = [idx_to_class.get(i, f"class_{i}") for i in missing]
        raise RuntimeError(f"Keine Samples fuer folgende Klassen gefunden: {names}")
    cursors = defaultdict(int)

    cycle = 0
    print("Endlosschleife aktiv. Mit Ctrl+C beenden.")
    try:
        while True:
            cycle += 1
            print(f"\n===== Zyklus {cycle} =====")
            collected = _collect_samples(
                dataset=dataset,
                target_idxs=target_idxs,
                samples_per_class=args.samples_per_class,
                indices_per_class=indices_per_class,
                cursors=cursors,
            )

            geometries = _build_geometries(
                collected=collected,
                target_idxs=target_idxs,
                idx_to_class=idx_to_class,
                samples_per_class=args.samples_per_class,
                x_spacing=args.x_spacing,
                y_spacing=args.y_spacing,
                show_axis=args.show_axis,
                o3d=o3d,
            )
            if not geometries:
                raise RuntimeError("Keine Samples gesammelt. Pruefe Dataset/Filter.")

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"Class-specific Dataloader Visualization | Zyklus {cycle}",
                width=1600,
                height=950,
            )
            for geom in geometries:
                vis.add_geometry(geom)
            render_option = vis.get_render_option()
            render_option.point_size = float(args.point_size)
            render_option.background_color = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)
            render_option.point_color_option = o3d.visualization.PointColorOption.Color
            render_option.light_on = False
            vis.run()
            vis.destroy_window()
    except KeyboardInterrupt:
        print("\nBeendet durch Benutzer (Ctrl+C).")


def build_parser():
    parser = argparse.ArgumentParser("Open3D class-specific dataloader visualization")
    parser.add_argument("--dataset-root", required=True, help="Root-Pfad mit train/val/test")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument(
        "--point-count",
        type=int,
        default=None,
        help="Alias fuer num-points. Wenn gesetzt, ueberschreibt es --num-points.",
    )
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optionaler Klassenfilter, z.B. --classes lkw pkw oder --classes 0,1,2",
    )
    parser.add_argument("--x-spacing", type=float, default=2.5)
    parser.add_argument("--y-spacing", type=float, default=2.5)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--show-axis", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    visualize_classes_open3d(args)
