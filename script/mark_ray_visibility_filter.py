#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_ascii_ply(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        properties = []
        vertex_count = None
        in_vertex = False
        for line in handle:
            tokens = line.strip().split()
            if tokens[:2] == ["format", "ascii"]:
                continue
            if tokens[:2] == ["element", "vertex"]:
                vertex_count = int(tokens[2])
                in_vertex = True
                continue
            if tokens and tokens[0] == "element":
                in_vertex = False
                continue
            if in_vertex and tokens and tokens[0] == "property":
                properties.append(tokens[-1])
                continue
            if tokens == ["end_header"]:
                break
        if vertex_count is None:
            raise ValueError(f"No vertex element found in {path}")
        data = np.loadtxt(handle, dtype=np.float64, max_rows=vertex_count)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape != (vertex_count, len(properties)):
        raise ValueError(
            f"Expected {(vertex_count, len(properties))}, got {data.shape} in {path}"
        )
    return properties, data


def rigid_transform(source, target):
    if source.shape != target.shape:
        raise ValueError(
            f"Corresponding clouds need equal shapes, got {source.shape} and {target.shape}"
        )
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    transformed = source @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))
    return rotation, translation, rmse


def static_sensor_visibility_filter(
    points_list,
    origin,
    az_res_deg=0.2,
    el_res_deg=0.2,
    depth_margin=0.05,
    mode="nearest",
):
    all_points = np.vstack(points_list)
    rays = all_points - origin
    depths = np.linalg.norm(rays, axis=1)
    dirs = rays / (depths[:, None] + 1e-8)

    az = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
    el = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
    az_idx = np.floor(az / az_res_deg).astype(int)
    el_idx = np.floor(el / el_res_deg).astype(int)
    az_idx -= az_idx.min()
    el_idx -= el_idx.min()

    grid = defaultdict(list)
    for index in range(len(all_points)):
        grid[(az_idx[index], el_idx[index])].append((depths[index], index))

    keep = np.zeros(len(all_points), dtype=bool)
    for cell_points in grid.values():
        cell_depths = np.asarray([depth for depth, _ in cell_points])
        cell_indices = [index for _, index in cell_points]

        if mode == "nearest":
            keep[cell_indices[int(np.argmin(cell_depths))]] = True
        elif mode == "median":
            median_depth = np.median(cell_depths)
            keep[cell_indices[int(np.argmin(np.abs(cell_depths - median_depth)))]] = True
        elif mode == "all_within_margin":
            min_depth = cell_depths.min()
            for depth, index in cell_points:
                if depth <= min_depth + depth_margin:
                    keep[index] = True
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return all_points[keep], keep, depths

def write_marked_ply(
    path,
    points,
    confidence,
    confidence_kept,
    visibility_kept,
    view_indices,
    origins,
    depths,
    comments,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.empty((len(points), 3), dtype=np.uint8)
    colors[visibility_kept] = np.array([190, 190, 190], dtype=np.uint8)
    colors[~visibility_kept] = np.array([255, 32, 32], dtype=np.uint8)
    header = [
        "ply",
        "format ascii 1.0",
        *[f"comment {comment}" for comment in comments],
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float confidence",
        "property uchar confidence_kept",
        "property uchar visibility_kept",
        "property uchar view_index",
        "property float origin_x",
        "property float origin_y",
        "property float origin_z",
        "property float depth",
        "end_header",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        for point, color, conf, conf_keep, vis_keep, view, origin, depth in zip(
            points,
            colors,
            confidence,
            confidence_kept,
            visibility_kept,
            view_indices,
            origins,
            depths,
        ):
            handle.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{color[0]} {color[1]} {color[2]} {conf:.6f} "
                f"{int(conf_keep)} {int(vis_keep)} {int(view)} "
                f"{origin[0]:.7f} {origin[1]:.7f} {origin[2]:.7f} "
                f"{depth:.7f}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("confidence_ply", type=Path)
    parser.add_argument("raw_frames", nargs="+", type=Path)
    parser.add_argument("--local-origin", nargs=3, type=float, required=True)
    parser.add_argument(
        "--static-origin",
        nargs=3,
        type=float,
        help="Fixed sensor origin in fused-cloud coordinates (default: frame 0 origin)",
    )
    parser.add_argument("--az-res-deg", type=float, default=0.2)
    parser.add_argument("--el-res-deg", type=float, default=0.2)
    parser.add_argument("--depth-margin", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    properties, cloud = load_ascii_ply(args.confidence_ply)
    columns = {name: index for index, name in enumerate(properties)}
    required = {"x", "y", "z", "confidence", "kept"}
    if not required.issubset(columns):
        raise ValueError(f"Missing properties: {sorted(required - set(columns))}")
    points = cloud[:, [columns["x"], columns["y"], columns["z"]]]
    confidence = cloud[:, columns["confidence"]]
    confidence_kept = cloud[:, columns["kept"]].astype(bool)

    raw_clouds = []
    for raw_path in args.raw_frames:
        raw_properties, raw_data = load_ascii_ply(raw_path)
        raw_columns = {name: index for index, name in enumerate(raw_properties)}
        raw_clouds.append(
            raw_data[:, [raw_columns["x"], raw_columns["y"], raw_columns["z"]]]
        )
    point_counts = [len(raw) for raw in raw_clouds]
    if sum(point_counts) != len(points):
        raise ValueError(
            f"Raw frame points sum to {sum(point_counts)}, confidence cloud has {len(points)}"
        )

    local_origin = np.asarray(args.local_origin, dtype=np.float64)
    view_indices = np.empty(len(points), dtype=np.uint8)
    transform_rmses = []
    offset = 0
    common_origins = []
    for view_index, raw in enumerate(raw_clouds):
        target = points[offset : offset + len(raw)]
        rotation, translation, rmse = rigid_transform(raw, target)
        common_origin = local_origin @ rotation.T + translation
        view_indices[offset : offset + len(raw)] = view_index
        transform_rmses.append(rmse)
        common_origins.append(common_origin)
        offset += len(raw)

    static_origin = (
        np.asarray(args.static_origin, dtype=np.float64)
        if args.static_origin is not None
        else common_origins[0]
    )
    origins = np.broadcast_to(static_origin, points.shape)
    output_dir = args.output_dir or args.confidence_ply.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    point_offsets = np.cumsum([0] + point_counts[:-1])
    fused_frames = [
        points[offset : offset + count]
        for offset, count in zip(point_offsets, point_counts)
    ]

    print(f"points={len(points)}")
    print(f"static_origin={static_origin.tolist()}")
    for mode in ("nearest", "median", "all_within_margin"):
        _, visibility_kept, depths = static_sensor_visibility_filter(
            fused_frames,
            static_origin,
            az_res_deg=args.az_res_deg,
            el_res_deg=args.el_res_deg,
            depth_margin=args.depth_margin,
            mode=mode,
        )
        stem = f"{args.confidence_ply.stem}_static_visibility_{mode}"
        output = output_dir / f"{stem}_marked.ply"
        filtered_output = output_dir / f"{stem}_filtered.ply"
        comments = [
            "red points would be removed; light gray points are retained",
            f"mode {mode}",
            f"static_origin {chr(32).join(str(value) for value in static_origin)}",
            f"az_res_deg {args.az_res_deg}",
            f"el_res_deg {args.el_res_deg}",
            f"depth_margin {args.depth_margin}",
            f"removed_points {int((~visibility_kept).sum())}",
            f"retained_points {int(visibility_kept.sum())}",
        ]
        write_marked_ply(
            output,
            points,
            confidence,
            confidence_kept,
            visibility_kept,
            view_indices,
            origins,
            depths,
            comments,
        )
        write_marked_ply(
            filtered_output,
            points[visibility_kept],
            confidence[visibility_kept],
            confidence_kept[visibility_kept],
            visibility_kept[visibility_kept],
            view_indices[visibility_kept],
            origins[visibility_kept],
            depths[visibility_kept],
            comments,
        )
        print(f"{mode}_marked_output={output}")
        print(f"{mode}_filtered_output={filtered_output}")
        print(f"{mode}_retained={int(visibility_kept.sum())}")
        print(f"{mode}_removed={int((~visibility_kept).sum())}")

    print(f"transform_rmse={transform_rmses}")
    for view_index, origin in enumerate(common_origins):
        print(f"view_{view_index}_origin={origin.tolist()}")


if __name__ == "__main__":
    main()
