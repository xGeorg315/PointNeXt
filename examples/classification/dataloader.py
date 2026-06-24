import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from openpoints.transforms.point_transform_cpu import RandomJitter, RandomRotate, RandomScale
from openpoints.transforms.point_transformer_gpu import PointCloudTranslation

MAX_DIMS = np.array([25.25, 2.60, 4.00], dtype=np.float32)
MAX_DIST = 20


class BalancedClassBatchSampler(Sampler):
    def __init__(
        self,
        labels,
        batch_size,
        shuffle=True,
        drop_last=False,
        max_repeat_per_sample=None,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_repeat_per_sample = (
            None if max_repeat_per_sample in {None, ""} else float(max_repeat_per_sample)
        )

        self.class_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.class_to_indices.setdefault(int(label), []).append(idx)

        self.classes = sorted(self.class_to_indices)
        self.num_classes = len(self.classes)
        if self.num_classes == 0:
            raise ValueError("BalancedClassBatchSampler braucht mindestens eine Klasse.")
        if self.batch_size < self.num_classes or self.batch_size % self.num_classes != 0:
            raise ValueError(
                "batch_size muss ein Vielfaches der Klassenanzahl sein, "
                f"bekommen: batch_size={self.batch_size}, classes={self.num_classes}"
            )

        self.samples_per_class = self.batch_size // self.num_classes
        max_class_count = max(len(indices) for indices in self.class_to_indices.values())
        self.num_batches = int(np.ceil(max_class_count / self.samples_per_class))
        if self.max_repeat_per_sample is not None:
            min_class_count = min(len(indices) for indices in self.class_to_indices.values())
            capped_batches = int(
                np.floor(min_class_count * self.max_repeat_per_sample / self.samples_per_class)
            )
            self.num_batches = max(1, min(self.num_batches, capped_batches))

    def __iter__(self):
        pools = {}
        for label, indices in self.class_to_indices.items():
            indices = np.asarray(indices, dtype=np.int64)
            needed = self.num_batches * self.samples_per_class
            if self.shuffle:
                indices = np.random.permutation(indices)
            if indices.shape[0] < needed:
                extra = np.random.choice(indices, needed - indices.shape[0], replace=True)
                indices = np.concatenate([indices, extra], axis=0)
            else:
                indices = indices[:needed]
            if self.shuffle:
                indices = np.random.permutation(indices)
            pools[label] = indices

        for batch_idx in range(self.num_batches):
            batch = []
            start = batch_idx * self.samples_per_class
            end = start + self.samples_per_class
            for label in self.classes:
                batch.extend(pools[label][start:end].tolist())
            if self.shuffle:
                np.random.shuffle(batch)
            yield batch

    def epoch_class_counts(self):
        count = self.num_batches * self.samples_per_class
        return {label: count for label in self.classes}

    def __len__(self):
        return self.num_batches


class RepeatUnderrepresentedClassBatchSampler(Sampler):
    def __init__(
        self,
        labels,
        batch_size,
        shuffle=True,
        drop_last=False,
        max_repeat_per_sample=3,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_repeat_per_sample = max(1, int(float(max_repeat_per_sample or 1)))

        self.class_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.class_to_indices.setdefault(int(label), []).append(idx)

        self.classes = sorted(self.class_to_indices)
        if not self.classes:
            raise ValueError("RepeatUnderrepresentedClassBatchSampler braucht mindestens eine Klasse.")

        max_class_count = max(len(indices) for indices in self.class_to_indices.values())
        self.target_counts = {
            label: min(max_class_count, len(indices) * self.max_repeat_per_sample)
            for label, indices in self.class_to_indices.items()
        }
        total = sum(self.target_counts.values())
        if self.drop_last:
            self.num_batches = total // self.batch_size
        else:
            self.num_batches = int(np.ceil(total / self.batch_size))

    def epoch_class_counts(self):
        return dict(self.target_counts)

    def __iter__(self):
        epoch_indices = []
        for label, indices in self.class_to_indices.items():
            indices = np.asarray(indices, dtype=np.int64)
            target_count = self.target_counts[label]
            if self.shuffle:
                indices = np.random.permutation(indices)
            if indices.shape[0] < target_count:
                extra = np.random.choice(indices, target_count - indices.shape[0], replace=True)
                indices = np.concatenate([indices, extra], axis=0)
            else:
                indices = indices[:target_count]
            epoch_indices.extend(indices.tolist())

        if self.shuffle:
            np.random.shuffle(epoch_indices)

        usable = self.num_batches * self.batch_size if self.drop_last else len(epoch_indices)
        for start in range(0, usable, self.batch_size):
            batch = epoch_indices[start:start + self.batch_size]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self):
        return self.num_batches


def _parse_date(value):
    if value in {None, ""}:
        return None
    if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value in {None, ""}:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float_tuple(value):
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, str):
        value = value.strip().strip("[]")
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return tuple(float(item) for item in value)


def _as_float_range(value, default):
    if value is None or value == "":
        return tuple(default)
    values = _as_float_tuple(value)
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) != 2:
        raise ValueError(f"Erwartet einen Float oder zwei Werte, bekommen: {value}")
    return (min(values), max(values))


def _as_xyz_range(value, default):
    if value is None or value == "":
        values = _as_float_tuple(default)
    else:
        values = _as_float_tuple(value)
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) != 3:
        raise ValueError(f"Erwartet einen Float oder drei XYZ-Werte, bekommen: {value}")
    return values


def _as_name_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = value.strip().strip("[]")
        return [item.strip().strip("'\"") for item in value.split(",") if item.strip()]
    return list(value)


def _as_str_list(value):
    return [str(item) for item in _as_name_list(value)]


def _as_axes_list(value):
    axes = [str(item).lower() for item in _as_name_list(value)]
    valid = {"xy", "xz", "yx", "yz", "zx", "zy"}
    if not axes:
        return ["xy", "xz", "yz"]
    for axis_pair in axes:
        if axis_pair not in valid:
            raise ValueError(f"Ungueltige View-Achsenkombination: {axis_pair}")
    return axes


def _read_pcd_header(file_path: Path):
    header = {}
    header_bytes = 0
    with file_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Ungueltiger PCD-Header: {file_path}")
            header_bytes += len(line)
            text = line.decode("utf-8", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                break
    return header, header_bytes


def _pcd_point_count(file_path: Path):
    header, _ = _read_pcd_header(file_path)
    if "POINTS" in header:
        return int(header["POINTS"][0])
    if "WIDTH" in header and "HEIGHT" in header:
        return int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
    raise ValueError(f"Keine Punktanzahl im PCD-Header gefunden: {file_path}")


def _metadata_distance_from_json(file_path: Path, metadata=None):
    if metadata is None:
        meta_path = file_path.with_suffix(".json")
        if not meta_path.exists():
            return 0.0
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            return 0.0

    metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
    for key in ("distance_to_exit_line", "closest_edge_distance"):
        if key in metrics:
            try:
                return abs(float(metrics[key]))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _cloud_files(class_dir: Path):
    yield from sorted(class_dir.glob("*.npy"))
    yield from sorted(class_dir.glob("*.pcd"))
    yield from sorted(class_dir.glob("*.off"))


def _read_off_mesh(file_path: Path):
    lines = []
    with file_path.open("r", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    if not lines:
        raise ValueError(f"Leere OFF-Datei: {file_path}")

    first = lines[0]
    if first == "OFF":
        counts_idx = 1
    elif first.startswith("OFF"):
        lines[0] = first[3:].strip()
        counts_idx = 0
    else:
        raise ValueError(f"Keine OFF-Datei: {file_path}")

    counts = lines[counts_idx].split()
    n_vertices, n_faces = int(counts[0]), int(counts[1])
    vertices_start = counts_idx + 1
    vertices = np.array(
        [[float(value) for value in lines[vertices_start + idx].split()[:3]] for idx in range(n_vertices)],
        dtype=np.float64,
    )

    faces = []
    face_start = vertices_start + n_vertices
    for idx in range(n_faces):
        parts = [int(value) for value in lines[face_start + idx].split()]
        count = parts[0]
        face = parts[1:1 + count]
        if count >= 3:
            for tri_idx in range(1, count - 1):
                faces.append([face[0], face[tri_idx], face[tri_idx + 1]])
    return vertices, np.asarray(faces, dtype=np.int32)


def _sample_off_surface(file_path: Path, num_points=4096):
    vertices, faces = _read_off_mesh(file_path)
    if vertices.shape[0] == 0:
        raise ValueError(f"OFF enthaelt keine Vertices: {file_path}")
    if faces.shape[0] == 0:
        return vertices.astype(np.float32, copy=False)

    valid_face = (faces >= 0).all(axis=1) & (faces < vertices.shape[0]).all(axis=1)
    if np.any(valid_face):
        valid_face[valid_face] &= np.isfinite(vertices[faces[valid_face]]).all(axis=(1, 2))
    faces = faces[valid_face]
    if faces.shape[0] == 0:
        return vertices.astype(np.float32, copy=False)

    triangles = vertices[faces].astype(np.float64, copy=False)
    centered = triangles - triangles.mean(axis=(0, 1), keepdims=True)
    scale = float(np.max(np.linalg.norm(centered.reshape(-1, 3), axis=1)))
    if scale > 1e-12 and np.isfinite(scale):
        triangles = centered / scale

    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    ) * 0.5
    valid = np.isfinite(areas) & (areas > 1e-12)
    area_sum = float(areas[valid].sum()) if np.any(valid) else 0.0
    if area_sum <= 0 or not np.isfinite(area_sum):
        return vertices.astype(np.float32, copy=False)

    triangles = triangles[valid]
    probs = areas[valid] / area_sum
    digest = hashlib.md5(str(file_path).encode("utf-8")).hexdigest()
    rng = np.random.RandomState(int(digest[:8], 16))
    choice = rng.choice(triangles.shape[0], size=int(num_points), replace=True, p=probs)
    chosen = triangles[choice]
    uv = rng.rand(int(num_points), 2).astype(np.float32)
    flip = uv.sum(axis=1) > 1.0
    uv[flip] = 1.0 - uv[flip]
    samples = chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    return samples.astype(np.float32, copy=False)


def _has_flat_split_layout(root_dir: Path) -> bool:
    return all((root_dir / split).is_dir() for split in ("train", "val"))


def _has_bucket_first_review_layout(root_dir: Path) -> bool:
    return any(
        bucket_dir.is_dir()
        and bucket_dir.name.startswith("gt-pred")
        and any(class_dir.is_dir() for class_dir in bucket_dir.iterdir())
        for bucket_dir in root_dir.iterdir()
    )


def _load_pcd_xyz(file_path: Path):
    header, header_bytes = _read_pcd_header(file_path)
    data_type = header.get("DATA", [""])[0].lower()
    fields = header.get("FIELDS")
    sizes = [int(v) for v in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields or []))]
    points = int(header.get("POINTS", [0])[0])

    if not fields or not sizes or not types:
        raise ValueError(f"Unvollstaendiger PCD-Header: {file_path}")
    if points <= 0:
        raise ValueError(f"Leere Pointcloud: {file_path}")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"PCD enthaelt keine x/y/z Felder: {file_path}")

    if data_type == "ascii":
        with file_path.open("rb") as handle:
            handle.seek(header_bytes)
            arr = np.loadtxt(handle, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, [fields.index("x"), fields.index("y"), fields.index("z")]].astype(
            np.float32,
            copy=False,
        )
    if data_type != "binary":
        raise ValueError(f"Nicht unterstuetztes PCD-DATA-Format '{data_type}': {file_path}")

    dtype_fields = []
    for field, size, typ, count in zip(fields, sizes, types, counts):
        if typ == "F" and size == 4:
            dtype = np.float32
        elif typ == "F" and size == 8:
            dtype = np.float64
        elif typ == "I" and size == 1:
            dtype = np.int8
        elif typ == "I" and size == 2:
            dtype = np.int16
        elif typ == "I" and size == 4:
            dtype = np.int32
        elif typ == "U" and size == 1:
            dtype = np.uint8
        elif typ == "U" and size == 2:
            dtype = np.uint16
        elif typ == "U" and size == 4:
            dtype = np.uint32
        else:
            raise ValueError(f"Nicht unterstuetztes PCD-Feld {field}: TYPE={typ} SIZE={size}")
        dtype_fields.append((field, dtype) if count == 1 else (field, dtype, (count,)))

    structured = np.fromfile(
        file_path,
        dtype=np.dtype(dtype_fields),
        count=points,
        offset=header_bytes,
    )
    xyz = np.column_stack([structured["x"], structured["y"], structured["z"]])
    return xyz.astype(np.float32, copy=False)


def _sample_label(sample):
    if isinstance(sample, tuple):
        return sample[1]
    if isinstance(sample, dict) and "label" in sample:
        return sample["label"]
    raise ValueError(f"Kann Klassenlabel fuer Sample nicht bestimmen: {sample}")


def _set_sample_label(sample, label):
    label = int(label)
    if isinstance(sample, tuple):
        if len(sample) == 3:
            return (sample[0], label, sample[2])
        if len(sample) == 2:
            return (sample[0], label)
        raise ValueError(f"Unsupported tuple sample format for label randomization: {sample}")
    if isinstance(sample, dict) and "label" in sample:
        sample.setdefault("original_label", int(sample["label"]))
        sample["label"] = label
        return sample
    raise ValueError(f"Kann Klassenlabel fuer Sample nicht setzen: {sample}")


def randomize_dataset_labels(dataset, seed=0, mode="permute"):
    """Randomize dataset labels in-place before samplers are built."""
    labels = np.asarray([_sample_label(sample) for sample in dataset.samples], dtype=np.int64)
    if labels.size == 0:
        dataset.randomized_labels = True
        return dataset

    mode = str(mode or "permute").lower()
    rng = np.random.RandomState(int(seed or 0))
    if mode in {"permute", "shuffle", "permutation"}:
        randomized = labels.copy()
        rng.shuffle(randomized)
    elif mode in {"uniform", "random"}:
        num_classes = int(getattr(dataset, "num_classes", 0) or len(getattr(dataset, "classes", [])))
        if num_classes <= 0:
            num_classes = int(labels.max()) + 1
        randomized = rng.randint(0, num_classes, size=labels.shape[0])
    else:
        raise ValueError(f"Unbekannter random_label_mode: {mode}")

    dataset.samples = [
        _set_sample_label(sample, label)
        for sample, label in zip(dataset.samples, randomized)
    ]
    dataset.randomized_labels = True
    dataset.random_label_mode = mode
    dataset.random_label_seed = int(seed or 0)
    return dataset


def _make_dataloader(
    dataset,
    split,
    batch_size,
    shuffle,
    num_workers,
    class_balanced_batches=False,
    balanced_max_repeat_per_sample=None,
    class_rebalance_mode=None,
    rebalance_max_repeat_per_sample=None,
):
    rebalance_mode = str(class_rebalance_mode or "").strip().lower()
    if rebalance_mode in {"repeat_underrepresented", "repeat_underrepresented_classes", "oversample_minority"}:
        labels = [_sample_label(sample) for sample in dataset.samples]
        batch_sampler = RepeatUnderrepresentedClassBatchSampler(
            labels=labels,
            batch_size=batch_size,
            shuffle=shuffle if split == "train" else False,
            drop_last=(split == "train"),
            max_repeat_per_sample=rebalance_max_repeat_per_sample or balanced_max_repeat_per_sample or 3,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)

    if class_balanced_batches:
        labels = [_sample_label(sample) for sample in dataset.samples]
        batch_sampler = BalancedClassBatchSampler(
            labels=labels,
            batch_size=batch_size,
            shuffle=shuffle if split == "train" else False,
            drop_last=(split == "train"),
            max_repeat_per_sample=balanced_max_repeat_per_sample,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        drop_last=(split == "train"),
    )


class PointCloudDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_points=1024,
        num_complete=2048,
        normalize=True,
        jitter_std=0.01,
        task="classification",
        target_strategy="aggregate",
        partial_choice="random_far",
        completion_min_dist_delta=0.0,
        symmetry_completion=False,
        symmetry_axis="x",
        symmetry_source="partial",
        symmetry_keep_side="both",
        completion_mask=False,
        completion_mask_min_keep_ratio=0.35,
        completion_mask_max_keep_ratio=0.75,
        completion_mask_parts=2,
        exclude_classes=(),
        augment_train=False,
        augment_rotation_deg=20.0,
        augment_scale=(0.8, 1.2),
        augment_jitter_sigma=0.01,
        augment_jitter_clip=0.05,
        augment_dropout=(0.2, 0.5),
        augment_translate=0.1,
        augment_random_points_ratio=0.0,
        augment_random_points_scale=1.0,
        use_normals=False,
        normal_k=16,
        preload_data=False,
        max_samples_per_class=None,
        multi_view=False,
        multi_view_axes=("xy", "xz", "yz"),
        multi_view_num_points=512,
        multi_view_bins=256,
        obj_features_include_sensor_dist=True,
    ):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.num_points = num_points
        self.num_complete = num_complete
        self.normalize = normalize
        self.jitter_std = jitter_std
        self.task = task
        self.is_completion = task in {"completion", "completion_cls", "completion_classification"}
        self.target_strategy = target_strategy
        self.partial_choice = partial_choice
        self.completion_min_dist_delta = float(completion_min_dist_delta)
        self.symmetry_completion = _as_bool(symmetry_completion)
        self.symmetry_axis = symmetry_axis
        self.symmetry_source = symmetry_source
        self.symmetry_keep_side = symmetry_keep_side
        self.completion_mask = _as_bool(completion_mask)
        self.completion_mask_min_keep_ratio = float(completion_mask_min_keep_ratio)
        self.completion_mask_max_keep_ratio = float(completion_mask_max_keep_ratio)
        self.completion_mask_parts = int(completion_mask_parts)
        self.exclude_classes = set(_as_name_list(exclude_classes))
        self.augment_train = _as_bool(augment_train)
        self.augment_dropout = _as_float_range(augment_dropout, (0.2, 0.5))
        self.augment_translate = _as_xyz_range(augment_translate, (0.1, 0.1, 0.1))
        self.augment_random_points_ratio = max(0.0, float(augment_random_points_ratio or 0.0))
        self.augment_random_points_scale = max(0.0, float(augment_random_points_scale or 1.0))
        self.use_normals = _as_bool(use_normals)
        self.normal_k = max(3, int(normal_k or 16))
        self.preload_data = _as_bool(preload_data)
        self.multi_view = _as_bool(multi_view)
        self.multi_view_axes = _as_axes_list(multi_view_axes)
        self.multi_view_num_points = int(multi_view_num_points or 512)
        self.multi_view_bins = int(multi_view_bins or 256)
        self.obj_features_include_sensor_dist = _as_bool(obj_features_include_sensor_dist)
        self._point_cache = {}
        self._rotation_transform = None
        self._scale_transform = None
        self._jitter_transform = None
        self._translation_transform = None
        rotation = float(augment_rotation_deg or 0.0)
        scale = _as_float_range(augment_scale, (0.8, 1.2))
        jitter_sigma = float(augment_jitter_sigma or 0.0)
        jitter_clip = float(augment_jitter_clip or 0.0)
        if rotation > 0:
            angle = rotation / 180.0
            self._rotation_transform = RandomRotate(angle=[angle, angle, angle])
        if scale != (1.0, 1.0):
            self._scale_transform = RandomScale(scale=list(scale))
        if jitter_sigma > 0 and jitter_clip > 0:
            self._jitter_transform = RandomJitter(
                jitter_sigma=jitter_sigma,
                jitter_clip=jitter_clip,
            )
        if any(value > 0 for value in self.augment_translate):
            self._translation_transform = PointCloudTranslation(shift=list(self.augment_translate))

        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        self._min_sensor_distance = None

        if self.is_completion:
            self._collect_completion_samples()
        else:
            self._collect_classification_samples()
        if self.preload_data:
            self._preload_samples()

        self.num_classes = len(self.class_to_idx)

    def _collect_classification_samples(self):
        # Klassen einsammeln
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name in self.exclude_classes:
                continue

            idx = len(self.classes)
            self.class_to_idx[class_dir.name] = idx
            self.classes.append(class_dir.name)

            for f in _cloud_files(class_dir):
                dist = self._file_distance(f)
                self.samples.append((f, idx, dist))

        if self.samples:
            self._min_sensor_distance = min(item[2] for item in self.samples)
        else:
            self._min_sensor_distance = 0.0

    def _collect_completion_samples(self):
        grouped = {}
        dists = []

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name in self.exclude_classes:
                continue

            idx = len(self.classes)
            self.class_to_idx[class_dir.name] = idx
            self.classes.append(class_dir.name)

            for f in _cloud_files(class_dir):
                dist = self._file_distance(f)
                sample_id, object_id = self._parse_sample_and_object_id(f)
                key = (class_dir.name, object_id)
                grouped.setdefault(
                    key,
                    {
                        "class_name": class_dir.name,
                        "label": idx,
                        "object_id": object_id,
                        "items": [],
                    },
                )
                grouped[key]["items"].append(
                    {
                        "file": f,
                        "dist": dist,
                        "sample_id": sample_id,
                    }
                )
                dists.append(dist)

        for group in grouped.values():
            group["items"].sort(key=lambda item: item["dist"])
            if group["items"]:
                self.samples.append(group)

        self._min_sensor_distance = min(dists) if dists else 0.0

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _parse_sensor_distance(file_path: Path) -> float:
        # Erwartetes Muster: "..._19m.npy" -> 19.0
        stem = file_path.stem
        match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)m$", stem)
        if match is None:
            raise ValueError(f"Keine Distanz im Dateinamen gefunden: {file_path.name}")
        return float(match.group(1))

    @staticmethod
    def _file_distance(file_path: Path) -> float:
        if file_path.suffix.lower() == ".npy":
            return PointCloudDataset._parse_sensor_distance(file_path)
        return _metadata_distance_from_json(file_path)

    @staticmethod
    def _parse_sample_and_object_id(file_path: Path):
        # Erwartetes Muster: "<sample>_<object_id>_<dist>m.npy"
        parts = file_path.stem.split("_")
        if len(parts) < 3:
            raise ValueError(f"Keine Objekt-ID im Dateinamen gefunden: {file_path.name}")
        sample_id = "_".join(parts[:-2])
        object_id = parts[-2]
        return sample_id, object_id

    @staticmethod
    def _load_xyz(file_path: Path):
        suffix = file_path.suffix.lower()
        if suffix == ".pcd":
            return _load_pcd_xyz(file_path)
        if suffix == ".off":
            return _sample_off_surface(file_path)
        points = np.load(file_path)
        points = points[:, :3].astype(np.float32, copy=False)
        if points.shape[0] == 0:
            raise ValueError(f"Leere Pointcloud: {file_path}")
        return points

    def _load_points(self, file_path):
        file_path = Path(file_path)
        cached = self._point_cache.get(file_path)
        if cached is not None:
            return cached.copy()
        return self._load_xyz(file_path)

    def _cache_file(self, file_path):
        file_path = Path(file_path)
        if file_path not in self._point_cache:
            self._point_cache[file_path] = self._load_xyz(file_path).astype(np.float32, copy=False)

    def _preload_samples(self):
        for sample in self.samples:
            if isinstance(sample, tuple):
                self._cache_file(sample[0])
            elif isinstance(sample, dict):
                for key in ("file", "partial_file", "complete_file"):
                    if sample.get(key) is not None:
                        self._cache_file(sample[key])
                for item in sample.get("items", []):
                    if item.get("file") is not None:
                        self._cache_file(item["file"])

    def _fps_indices(self, points, target_count, candidate_idx=None):
        n = points.shape[0]
        if candidate_idx is None:
            candidate_idx = np.arange(n, dtype=np.int64)
        else:
            candidate_idx = np.asarray(candidate_idx, dtype=np.int64)

        if target_count >= candidate_idx.shape[0]:
            return candidate_idx

        candidates = points[candidate_idx]
        # Bei sehr grossen Clouds zuerst random vorreduzieren, dann FPS.
        max_candidates = min(candidates.shape[0], max(target_count * 2, 4096))
        if candidates.shape[0] > max_candidates:
            keep = np.random.choice(candidates.shape[0], max_candidates, replace=False)
            candidate_idx = candidate_idx[keep]
            candidates = points[candidate_idx]

        m = candidates.shape[0]
        k = min(target_count, m)

        selected = np.empty(k, dtype=np.int64)
        distances = np.full(m, np.inf, dtype=np.float64)
        current = np.random.randint(0, m)

        for i in range(k):
            selected[i] = current
            diff = candidates - candidates[current]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            distances = np.minimum(distances, dist2)
            current = int(np.argmax(distances))

        return candidate_idx[selected]

    def _stratified_fps_indices(self, points, target_count, bins=8):
        n = points.shape[0]
        if target_count >= n:
            return np.arange(n, dtype=np.int64)

        axis = int(np.argmax(np.ptp(points, axis=0)))
        axis_values = points[:, axis]

        # Gleich breite Bins entlang der Hauptachse (typisch Front-Back-Richtung).
        edges = np.linspace(axis_values.min(), axis_values.max(), bins + 1)
        bin_indices = []
        for i in range(bins):
            if i == bins - 1:
                idx = np.where((axis_values >= edges[i]) & (axis_values <= edges[i + 1]))[0]
            else:
                idx = np.where((axis_values >= edges[i]) & (axis_values < edges[i + 1]))[0]
            if idx.size > 0:
                bin_indices.append(idx)

        if not bin_indices:
            return self._fps_indices(points, target_count)

        quota = target_count // len(bin_indices)
        selected_parts = []

        for idx in bin_indices:
            take = min(quota, idx.shape[0])
            if take > 0:
                selected_parts.append(self._fps_indices(points, take, candidate_idx=idx))

        selected = (
            np.concatenate(selected_parts, axis=0)
            if selected_parts
            else np.empty(0, dtype=np.int64)
        )

        if selected.shape[0] >= target_count:
            choice = np.random.choice(selected.shape[0], target_count, replace=False)
            return selected[choice]

        missing = target_count - selected.shape[0]
        pool = np.setdiff1d(np.arange(n, dtype=np.int64), selected, assume_unique=False)
        if pool.shape[0] == 0:
            return selected
        extra = self._fps_indices(points, missing, candidate_idx=pool)
        return np.concatenate([selected, extra], axis=0)

    # ---------- Resampling ----------
    def _resample(self, points, target_count=None, jitter=True):
        if target_count is None:
            target_count = self.num_points
        n = points.shape[0]

        if n >= target_count:
            idx = self._stratified_fps_indices(points=points, target_count=target_count, bins=8)
            points = points[idx]
        else:
            base_idx = self._fps_indices(points=points, target_count=n)
            base_points = points[base_idx]

            missing = target_count - n
            repeat_idx = torch.randint(
                low=0,
                high=base_points.shape[0],
                size=(missing,),
            ).numpy()
            extra_points = base_points[repeat_idx]
            points = np.concatenate([base_points, extra_points], axis=0)

        return points.astype(np.float32, copy=False)

    def _should_augment_classification(self):
        return self.augment_train and self.split == "train" and not self.is_completion

    def _apply_point_dropout(self, points):
        if not self._should_augment_classification() or points.shape[0] <= 1:
            return points

        drop_min, drop_max = self.augment_dropout
        drop_prob = float(np.random.uniform(drop_min, drop_max))
        if drop_prob <= 0:
            return points

        keep_mask = np.random.rand(points.shape[0]) > min(drop_prob, 0.95)
        if not np.any(keep_mask):
            keep_mask[np.random.randint(0, points.shape[0])] = True
        return points[keep_mask].astype(np.float32, copy=False)

    def _include_random_points(self, points):
        if not self._should_augment_classification() or self.augment_random_points_ratio <= 0:
            return points
        if points.shape[0] == 0:
            return points

        count = int(round(points.shape[0] * self.augment_random_points_ratio))
        if count <= 0:
            return points

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)
        half_extent = 0.5 * (maxs - mins) * self.augment_random_points_scale
        half_extent = np.maximum(half_extent, 1e-3)
        random_points = np.random.uniform(
            center - half_extent,
            center + half_extent,
            size=(count, 3),
        ).astype(np.float32)
        return np.concatenate([points, random_points], axis=0).astype(np.float32, copy=False)

    def _apply_geometric_augmentations(self, points):
        if not self._should_augment_classification():
            return points.astype(np.float32, copy=False)

        data = {"pos": points.astype(np.float32, copy=True)}
        for transform in (self._rotation_transform, self._scale_transform, self._jitter_transform):
            if transform is not None:
                data = transform(data)

        if self._translation_transform is not None:
            data = {"pos": torch.from_numpy(data["pos"])}
            data = self._translation_transform(data)
            return data["pos"].numpy().astype(np.float32, copy=False)
        return data["pos"].astype(np.float32, copy=False)

    def _estimate_normals(self, points):
        points = points[:, :3].astype(np.float32, copy=False)
        n = points.shape[0]
        normals = np.zeros_like(points, dtype=np.float32)
        if n < 3:
            normals[:, 2] = 1.0
            return normals

        k = min(self.normal_k, n - 1)
        for idx in range(n):
            diff = points - points[idx]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            nn_idx = np.argpartition(dist2, k + 1)[: k + 1]
            nn_idx = nn_idx[nn_idx != idx][:k]
            neighbors = points[nn_idx]
            centered = neighbors - neighbors.mean(axis=0, keepdims=True)
            cov = centered.T @ centered / max(1, centered.shape[0])
            _, vecs = np.linalg.eigh(cov)
            normal = vecs[:, 0].astype(np.float32)
            if np.dot(normal, points[idx]) < 0:
                normal = -normal
            norm = float(np.linalg.norm(normal))
            normals[idx] = normal / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return normals

    def _point_features(self, points):
        points = points.astype(np.float32, copy=False)
        if not self.use_normals:
            return points
        normals = self._estimate_normals(points)
        return np.concatenate([points, normals], axis=1).astype(np.float32, copy=False)

    def _silhouette_points(self, points, axes):
        axis_map = {"x": 0, "y": 1, "z": 2}
        idx0, idx1 = axis_map[axes[0]], axis_map[axes[1]]
        proj = points[:, [idx0, idx1]].astype(np.float32, copy=False)
        if proj.shape[0] == 0:
            return np.zeros((1, 3), dtype=np.float32)

        a = proj[:, 0]
        b = proj[:, 1]
        silhouette = []
        if float(a.max() - a.min()) > 1e-8:
            a_edges = np.linspace(a.min(), a.max(), self.multi_view_bins + 1)
            for bin_idx in range(self.multi_view_bins):
                mask = (a >= a_edges[bin_idx]) & (a < a_edges[bin_idx + 1])
                if np.any(mask):
                    local = proj[mask]
                    silhouette.append(local[np.argmin(local[:, 1])])
                    silhouette.append(local[np.argmax(local[:, 1])])
        if float(b.max() - b.min()) > 1e-8:
            b_edges = np.linspace(b.min(), b.max(), max(32, self.multi_view_bins // 3) + 1)
            for bin_idx in range(len(b_edges) - 1):
                mask = (b >= b_edges[bin_idx]) & (b < b_edges[bin_idx + 1])
                if np.any(mask):
                    local = proj[mask]
                    silhouette.append(local[np.argmin(local[:, 0])])
                    silhouette.append(local[np.argmax(local[:, 0])])

        if silhouette:
            proj = np.unique(np.asarray(silhouette, dtype=np.float32), axis=0)

        embedded = np.zeros((proj.shape[0], 3), dtype=np.float32)
        embedded[:, idx0] = proj[:, 0]
        embedded[:, idx1] = proj[:, 1]
        return embedded

    def _multi_view_points(self, points):
        views = []
        for axes in self.multi_view_axes:
            view = self._silhouette_points(points[:, :3], axes)
            view = self._resample(
                view,
                target_count=self.multi_view_num_points,
                jitter=self.split == "train",
            )
            views.append(view.astype(np.float32, copy=False))
        return np.stack(views, axis=0).astype(np.float32, copy=False)

    def _add_multi_view_if_needed(self, output, points):
        if self.multi_view:
            output["views"] = torch.from_numpy(self._multi_view_points(points)).float()
        return output

    # ---------- Normalisierung ----------
    def _normalize(self, points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        radius = self._unit_radius(points)
        return points / radius

    @staticmethod
    def _unit_radius(points):
        radius = float(np.max(np.linalg.norm(points, axis=1)))
        return radius if radius > 1e-8 else 1.0

    @staticmethod
    def _normalize_with_centroid(points, centroid, radius):
        return (points - centroid) / radius

    def _completion_normalization_params(self, partial_raw, complete_raw):
        centroid = np.mean(partial_raw, axis=0)
        complete_centered = complete_raw - centroid
        radius = self._unit_radius(complete_centered)
        return centroid, radius

    def _scale_to_unit_radius(self, *clouds):
        radius = self._unit_radius(np.concatenate(clouds, axis=0))
        return tuple(cloud / radius for cloud in clouds)

    @staticmethod
    def _axis_index(axis):
        if isinstance(axis, int):
            if axis not in {0, 1, 2}:
                raise ValueError(f"Ungueltige Symmetrie-Achse: {axis}")
            return axis
        axis_map = {"x": 0, "y": 1, "z": 2}
        key = str(axis).lower()
        if key not in axis_map:
            raise ValueError(f"Ungueltige Symmetrie-Achse: {axis}")
        return axis_map[key]

    def _symmetry_complete(self, points, center=None):
        points = points[:, :3].astype(np.float32, copy=False)
        if points.shape[0] == 0:
            return points

        axis = self._axis_index(self.symmetry_axis)
        if center is None:
            center = 0.5 * (float(points[:, axis].min()) + float(points[:, axis].max()))
        else:
            center = float(center)
        negative = points[points[:, axis] <= center]
        positive = points[points[:, axis] >= center]

        keep_side = str(self.symmetry_keep_side).lower()
        if keep_side in {"both", "all", "full", "original"}:
            kept = points
        elif keep_side == "auto":
            kept = negative if negative.shape[0] >= positive.shape[0] else positive
        elif keep_side in {"negative", "neg", "left", "lower"}:
            kept = negative
        elif keep_side in {"positive", "pos", "right", "upper"}:
            kept = positive
        else:
            raise ValueError(f"Unbekannte symmetry_keep_side: {self.symmetry_keep_side}")

        if kept.shape[0] == 0:
            kept = points

        mirrored = kept.copy()
        mirrored[:, axis] = (2.0 * center) - mirrored[:, axis]
        return np.concatenate([kept, mirrored], axis=0).astype(np.float32, copy=False)

    def _completion_target_from_sources(self, partial_raw, complete_raw, symmetry_center=None):
        if not self.symmetry_completion:
            return complete_raw

        source = str(self.symmetry_source).lower()
        if source in {"partial", "input", "pred"}:
            return self._symmetry_complete(partial_raw, center=symmetry_center)
        if source in {"complete", "target", "gt"}:
            return self._symmetry_complete(complete_raw, center=symmetry_center)
        raise ValueError(f"Unbekannte symmetry_source: {self.symmetry_source}")

    def _completion_mask_rng(self, key=None):
        if self.split == "train" or key in {None, ""}:
            return np.random
        digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
        return np.random.RandomState(int(digest[:8], 16))

    def _mask_completion_input(self, complete_raw, key=None):
        points = complete_raw[:, :3].astype(np.float32, copy=False)
        if not self.completion_mask or points.shape[0] <= 1:
            return points

        min_keep = max(0.05, min(1.0, self.completion_mask_min_keep_ratio))
        max_keep = max(min_keep, min(1.0, self.completion_mask_max_keep_ratio))
        rng = self._completion_mask_rng(key)
        keep_ratio = float(rng.uniform(min_keep, max_keep))
        target_keep = max(1, int(round(points.shape[0] * keep_ratio)))
        remove_count = max(0, points.shape[0] - target_keep)
        if remove_count == 0:
            return points

        keep_mask = np.ones(points.shape[0], dtype=bool)
        parts = max(1, self.completion_mask_parts)
        remaining_remove = remove_count
        all_indices = np.arange(points.shape[0], dtype=np.int64)
        for part_idx in range(parts):
            active = all_indices[keep_mask]
            if active.size == 0 or remaining_remove <= 0:
                break

            chunks_left = parts - part_idx
            take = int(np.ceil(remaining_remove / chunks_left))
            take = min(take, active.size)
            center_idx = int(active[rng.randint(0, active.size)])
            dist2 = np.einsum("ij,ij->i", points[active] - points[center_idx], points[active] - points[center_idx])
            remove_idx = active[np.argpartition(dist2, take - 1)[:take]]
            keep_mask[remove_idx] = False
            remaining_remove -= remove_idx.size

        masked = points[keep_mask]
        if masked.shape[0] == 0:
            return points
        return masked.astype(np.float32, copy=False)

    def _choose_partial_item(self, items):
        if len(items) == 1:
            return items[0]

        min_dist = items[0]["dist"]
        candidates = [
            item for item in items
            if item["dist"] > min_dist + self.completion_min_dist_delta
        ]
        if not candidates:
            candidates = items[1:] if len(items) > 1 else items

        if self.split == "train" and self.partial_choice == "random_far":
            return candidates[np.random.randint(0, len(candidates))]
        if self.partial_choice == "nearest_far":
            return candidates[0]
        return candidates[-1]

    def _load_complete_target(self, items):
        if self.target_strategy == "nearest":
            return self._load_points(items[0]["file"])
        if self.target_strategy != "aggregate":
            raise ValueError(f"Unbekannte target_strategy: {self.target_strategy}")

        clouds = [self._load_points(item["file"]) for item in items]
        return np.concatenate(clouds, axis=0)

    def _obj_features(self, points, raw_dist):
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        obj_dims = ((maxs - mins) / MAX_DIMS).astype(np.float32)
        if not self.obj_features_include_sensor_dist:
            return obj_dims

        sensor_dist = float(raw_dist) / MAX_DIST
        return np.concatenate(
            [obj_dims, np.array([sensor_dist], dtype=np.float32)]
        )

    def _getitem_completion(self, idx):
        group = self.samples[idx]
        partial_item = self._choose_partial_item(group["items"])

        source_raw = self._load_points(partial_item["file"])
        complete_source_raw = self._load_complete_target(group["items"])
        complete_raw = self._completion_target_from_sources(source_raw, complete_source_raw)
        partial_raw = self._mask_completion_input(
            complete_raw,
            key=f"{group['class_name']}::{group['object_id']}::{partial_item['sample_id']}",
        )
        obj_features = self._obj_features(partial_raw, partial_item["dist"])

        if self.normalize:
            centroid, radius = self._completion_normalization_params(partial_raw, complete_raw)
            partial = self._normalize_with_centroid(partial_raw, centroid, radius)
            complete = self._normalize_with_centroid(complete_raw, centroid, radius)
        else:
            partial = partial_raw
            complete = complete_raw

        partial = self._resample(
            partial,
            target_count=self.num_points,
            jitter=self.split == "train",
        )
        complete = self._resample(
            complete,
            target_count=self.num_complete,
            jitter=False,
        )
        if self.normalize:
            partial, complete = self._scale_to_unit_radius(partial, complete)

        partial = torch.from_numpy(partial).float()
        complete = torch.from_numpy(complete).float()
        label = torch.tensor(group["label"]).long()
        obj_features = torch.from_numpy(obj_features).float()

        return {
            "partial": partial,
            "complete": complete,
            "x": partial,
            "pos": partial,
            "y": label,
            "obj_features": obj_features,
            "object_id": group["object_id"],
            "sample_id": partial_item["sample_id"],
            "partial_dist": torch.tensor(partial_item["dist"]).float(),
        }

    def __getitem__(self, idx):
        if self.is_completion:
            return self._getitem_completion(idx)

        file_path, label, raw_dist = self.samples[idx]

        points = self._load_points(file_path)

        obj_features = self._obj_features(points, raw_dist)

        # 1) Normalisieren (wichtig: VOR Oversampling)
        if self.normalize:
            points = self._normalize(points)

        points = self._apply_point_dropout(points)
        points = self._include_random_points(points)

        # 2) Resampling + ggf. Jitter
        points = self._resample(points)
        if self.normalize:
            points = self._scale_to_unit_radius(points)[0]
        points = self._apply_geometric_augmentations(points)

        point_features = torch.from_numpy(self._point_features(points)).float()
        points = torch.from_numpy(points).float()
        label = torch.tensor(label).long()
        obj_features = torch.from_numpy(obj_features).float()

        output = {
            "x": point_features,
            "pos": points,
            "y": label,
            "obj_features": obj_features,
        }
        return self._add_multi_view_if_needed(output, points.numpy())


def get_dataloader(
    dataset_root,
    split,
    batch_size=32,
    num_points=1024,
    shuffle=True,
    num_workers=4,
    num_complete=2048,
    task="classification",
    target_strategy="aggregate",
    partial_choice="random_far",
    completion_min_dist_delta=0.0,
    symmetry_completion=False,
    symmetry_axis="x",
    symmetry_source="partial",
    symmetry_keep_side="auto",
    completion_mask=False,
    completion_mask_min_keep_ratio=0.35,
    completion_mask_max_keep_ratio=0.75,
    completion_mask_parts=2,
    class_balanced_batches=False,
    balanced_max_repeat_per_sample=None,
    class_rebalance_mode=None,
    rebalance_max_repeat_per_sample=None,
    exclude_classes=(),
    randomize_labels=False,
    random_label_seed=0,
    random_label_mode="permute",
    augment_train=False,
    augment_rotation_deg=20.0,
    augment_scale=(0.8, 1.2),
    augment_jitter_sigma=0.01,
    augment_jitter_clip=0.05,
    augment_dropout=(0.2, 0.5),
    augment_translate=0.1,
    augment_random_points_ratio=0.0,
    augment_random_points_scale=1.0,
    use_normals=False,
    normal_k=16,
    preload_data=False,
    max_samples_per_class=None,
    multi_view=False,
    multi_view_axes=("xy", "xz", "yz"),
    multi_view_num_points=512,
    multi_view_bins=256,
    obj_features_include_sensor_dist=True,
):
    dataset = PointCloudDataset(
        root_dir=dataset_root,
        split=split,
        num_points=num_points,
        num_complete=num_complete,
        task=task,
        target_strategy=target_strategy,
        partial_choice=partial_choice,
        completion_min_dist_delta=completion_min_dist_delta,
        symmetry_completion=symmetry_completion,
        symmetry_axis=symmetry_axis,
        symmetry_source=symmetry_source,
        symmetry_keep_side=symmetry_keep_side,
        completion_mask=completion_mask,
        completion_mask_min_keep_ratio=completion_mask_min_keep_ratio,
        completion_mask_max_keep_ratio=completion_mask_max_keep_ratio,
        completion_mask_parts=completion_mask_parts,
        exclude_classes=exclude_classes,
        augment_train=augment_train,
        augment_rotation_deg=augment_rotation_deg,
        augment_scale=augment_scale,
        augment_jitter_sigma=augment_jitter_sigma,
        augment_jitter_clip=augment_jitter_clip,
        augment_dropout=augment_dropout,
        augment_translate=augment_translate,
        augment_random_points_ratio=augment_random_points_ratio,
        augment_random_points_scale=augment_random_points_scale,
        use_normals=use_normals,
        normal_k=normal_k,
        preload_data=preload_data,
        max_samples_per_class=max_samples_per_class,
        multi_view=multi_view,
        multi_view_axes=multi_view_axes,
        multi_view_num_points=multi_view_num_points,
        multi_view_bins=multi_view_bins,
        obj_features_include_sensor_dist=obj_features_include_sensor_dist,
    )
    if randomize_labels:
        randomize_dataset_labels(dataset, seed=random_label_seed, mode=random_label_mode)

    return _make_dataloader(
        dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        class_balanced_batches=class_balanced_batches,
        balanced_max_repeat_per_sample=balanced_max_repeat_per_sample,
        class_rebalance_mode=class_rebalance_mode,
        rebalance_max_repeat_per_sample=rebalance_max_repeat_per_sample,
    )


class ReviewCompletionDataset(PointCloudDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_points=1024,
        num_complete=2048,
        normalize=True,
        jitter_std=0.01,
        start_date=None,
        min_points=0,
        split_ratios=(0.8, 0.1, 0.1),
        bucket="gt-pred-same",
        buckets=None,
        partial_dir="pred",
        complete_dir="pred",
        exclude_classes=("reject",),
        min_points_exempt_classes=("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
        symmetry_completion=False,
        symmetry_axis="x",
        symmetry_source="partial",
        symmetry_keep_side="both",
        completion_mask=True,
        completion_mask_min_keep_ratio=0.35,
        completion_mask_max_keep_ratio=0.75,
        completion_mask_parts=2,
    ):
        self.review_root_dir = Path(root_dir)
        self.review_start_date = _parse_date(start_date)
        self.review_min_points = int(min_points or 0)
        self.review_split_ratios = _as_float_tuple(split_ratios)
        self.review_buckets = _as_str_list(buckets if buckets is not None else [bucket])
        self.review_partial_dir = partial_dir
        self.review_complete_dir = complete_dir
        self.review_exclude_classes = set(_as_name_list(exclude_classes))
        self.review_min_points_exempt_classes = set(_as_name_list(min_points_exempt_classes))
        super().__init__(
            root_dir=root_dir,
            split=split,
            num_points=num_points,
            num_complete=num_complete,
            normalize=normalize,
            jitter_std=jitter_std,
            task="completion_cls",
            target_strategy="paired_gt",
            partial_choice="paired_pred",
            symmetry_completion=symmetry_completion,
            symmetry_axis=symmetry_axis,
            symmetry_source=symmetry_source,
            symmetry_keep_side=symmetry_keep_side,
            completion_mask=completion_mask,
            completion_mask_min_keep_ratio=completion_mask_min_keep_ratio,
            completion_mask_max_keep_ratio=completion_mask_max_keep_ratio,
            completion_mask_parts=completion_mask_parts,
        )

    def _collect_completion_samples(self):
        self.root_dir = self.review_root_dir
        all_records = []

        class_dirs = [
            p for p in sorted(self.review_root_dir.iterdir())
            if p.is_dir() and p.name not in self.review_exclude_classes
        ]
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            self.classes.append(class_dir.name)

            for day_dir in sorted(class_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                try:
                    day = _parse_date(day_dir.name)
                except ValueError:
                    continue
                if self.review_start_date is not None and day < self.review_start_date:
                    continue

                for bucket in self.review_buckets:
                    pair_root = day_dir / bucket
                    partial_root = pair_root / self.review_partial_dir
                    complete_root = pair_root / self.review_complete_dir
                    if not partial_root.is_dir() or not complete_root.is_dir():
                        continue

                    for partial_file in sorted(partial_root.glob("*.pcd")):
                        complete_file = partial_file if partial_root == complete_root else complete_root / partial_file.name
                        if not complete_file.exists():
                            continue
                        partial_count = _pcd_point_count(partial_file)
                        complete_count = _pcd_point_count(complete_file)
                        if (
                            self.review_min_points > 0
                            and class_dir.name not in self.review_min_points_exempt_classes
                            and min(partial_count, complete_count) < self.review_min_points
                        ):
                            continue

                        sample_id, object_id = self._parse_review_ids(partial_file)
                        record = {
                            "class_name": class_dir.name,
                            "label": idx,
                            "object_id": object_id,
                            "sample_id": sample_id,
                            "date": day_dir.name,
                            "bucket": bucket,
                            "partial_file": partial_file,
                            "complete_file": complete_file,
                            "partial_count": partial_count,
                            "complete_count": complete_count,
                        }
                        all_records.append(record)

        split_by_key = self._stratified_split_map(all_records)
        self.samples = [
            record
            for record in all_records
            if split_by_key[self._record_key(record)] == self.split
        ]
        self._min_sensor_distance = 0.0

    @staticmethod
    def _parse_review_ids(file_path: Path):
        stem = file_path.stem
        object_match = re.search(r"__object_([^_]+)", stem)
        object_id = object_match.group(1) if object_match else stem
        sample_id = stem[: object_match.start()] if object_match else stem
        return sample_id, object_id

    @staticmethod
    def _record_key(record):
        key = f"{record['class_name']}::{record['date']}::{record['object_id']}::{record['sample_id']}"
        return key

    @staticmethod
    def _record_group_key(record):
        return str(record["object_id"])

    def _record_sort_key(self, record):
        key = self._record_key(record)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return digest, key

    def _group_sort_key(self, group_key):
        digest = hashlib.md5(group_key.encode("utf-8")).hexdigest()
        return digest, group_key

    def _split_counts(self, n_records):
        ratios = np.asarray(self.review_split_ratios, dtype=np.float64)
        if ratios.shape[0] != 3:
            raise ValueError(f"review_split_ratios muss 3 Werte enthalten: {self.review_split_ratios}")
        ratio_sum = float(ratios.sum())
        if ratio_sum <= 0:
            raise ValueError(f"review_split_ratios muss eine positive Summe haben: {self.review_split_ratios}")
        ratios = ratios / ratio_sum

        raw = ratios * n_records
        counts = np.floor(raw).astype(np.int64)
        remainder = int(n_records - counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for idx in order[:remainder]:
                counts[idx] += 1
        return counts.tolist()

    def _stratified_split_map(self, records):
        if self.split not in {"train", "val", "test"}:
            return {self._record_key(record): self.split for record in records}

        groups_by_id = {}
        for record in records:
            groups_by_id.setdefault(self._record_group_key(record), []).append(record)

        groups_by_class_key = {}
        for group_key, group_records in groups_by_id.items():
            class_key = "::".join(sorted({record["class_name"] for record in group_records}))
            groups_by_class_key.setdefault(class_key, {})[group_key] = group_records

        split_by_key = {}
        split_names = ("train", "val", "test")
        for class_groups in groups_by_class_key.values():
            group_keys = sorted(class_groups, key=self._group_sort_key)
            counts = self._split_counts(len(group_keys))
            offset = 0
            for split_name, count in zip(split_names, counts):
                for group_key in group_keys[offset: offset + count]:
                    for record in sorted(class_groups[group_key], key=self._record_sort_key):
                        split_by_key[self._record_key(record)] = split_name
                offset += count
        return split_by_key

    @staticmethod
    def _load_xyz(file_path: Path):
        if file_path.suffix.lower() == ".pcd":
            return _load_pcd_xyz(file_path)
        return PointCloudDataset._load_xyz(file_path)

    @staticmethod
    def _metadata_distance(file_path: Path):
        meta_path = file_path.with_suffix(".json")
        if not meta_path.exists():
            return 0.0
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle).get("metrics", {})
        except (OSError, json.JSONDecodeError):
            return 0.0
        for key in ("distance_to_exit_line", "closest_edge_distance"):
            if key in metrics:
                return abs(float(metrics[key]))
        return 0.0

    def _metadata_symmetry_center(self, file_path: Path):
        meta_path = file_path.with_suffix(".json")
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle).get("metrics", {})
        except (OSError, json.JSONDecodeError, TypeError):
            return None

        axis_key = str(self.symmetry_axis).lower()
        meta_axis = str(
            metrics.get("symmetry_completion_lateral_axis")
            or metrics.get("vehicle_width_axis")
            or ""
        ).lower()
        if meta_axis and meta_axis != axis_key:
            return None

        value = metrics.get("symmetry_completion_plane_coordinate")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _getitem_completion(self, idx):
        sample = self.samples[idx]
        source_raw = self._load_points(sample["partial_file"])
        complete_source_raw = self._load_points(sample["complete_file"])
        symmetry_center = self._metadata_symmetry_center(sample["partial_file"])
        complete_raw = self._completion_target_from_sources(
            source_raw,
            complete_source_raw,
            symmetry_center=symmetry_center,
        )
        partial_raw = self._mask_completion_input(
            complete_raw,
            key=f"{sample['class_name']}::{sample['object_id']}::{sample['sample_id']}",
        )
        raw_dist = self._metadata_distance(sample["partial_file"])
        obj_features = self._obj_features(partial_raw, raw_dist)

        if self.normalize:
            centroid, radius = self._completion_normalization_params(partial_raw, complete_raw)
            partial = self._normalize_with_centroid(partial_raw, centroid, radius)
            complete = self._normalize_with_centroid(complete_raw, centroid, radius)
        else:
            partial = partial_raw
            complete = complete_raw

        partial = self._resample(partial, target_count=self.num_points, jitter=self.split == "train")
        complete = self._resample(complete, target_count=self.num_complete, jitter=False)
        if self.normalize:
            partial, complete = self._scale_to_unit_radius(partial, complete)

        partial = torch.from_numpy(partial).float()
        complete = torch.from_numpy(complete).float()
        label = torch.tensor(sample["label"]).long()
        obj_features = torch.from_numpy(obj_features).float()

        return {
            "partial": partial,
            "complete": complete,
            "x": partial,
            "pos": partial,
            "y": label,
            "obj_features": obj_features,
            "object_id": sample["object_id"],
            "sample_id": sample["sample_id"],
            "partial_dist": torch.tensor(raw_dist).float(),
        }


class ReviewClassificationDataset(ReviewCompletionDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_points=1024,
        normalize=True,
        jitter_std=0.01,
        start_date=None,
        min_points=0,
        split_ratios=(0.8, 0.1, 0.1),
        bucket="gt-pred-different",
        buckets=None,
        source_dir="pred",
        exclude_classes=("reject",),
        min_points_exempt_classes=("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
        augment_train=False,
        augment_rotation_deg=20.0,
        augment_scale=(0.8, 1.2),
        augment_jitter_sigma=0.01,
        augment_jitter_clip=0.05,
        augment_dropout=(0.2, 0.5),
        augment_translate=0.1,
        augment_random_points_ratio=0.0,
        augment_random_points_scale=1.0,
        use_normals=False,
        normal_k=16,
        preload_data=False,
        max_samples_per_class=None,
        multi_view=False,
        multi_view_axes=("xy", "xz", "yz"),
        multi_view_num_points=512,
        multi_view_bins=256,
        obj_features_include_sensor_dist=True,
    ):
        self.review_root_dir = Path(root_dir)
        self.review_start_date = _parse_date(start_date)
        self.review_min_points = int(min_points or 0)
        self.review_split_ratios = _as_float_tuple(split_ratios)
        self.review_buckets = _as_str_list(buckets if buckets is not None else [bucket])
        self.review_source_dir = source_dir
        self.review_exclude_classes = set(_as_name_list(exclude_classes))
        self.review_min_points_exempt_classes = set(_as_name_list(min_points_exempt_classes))
        self.review_max_samples_per_class = (
            None if max_samples_per_class in {None, ""} else int(max_samples_per_class)
        )
        PointCloudDataset.__init__(
            self,
            root_dir=root_dir,
            split=split,
            num_points=num_points,
            normalize=normalize,
            jitter_std=jitter_std,
            task="classification",
            augment_train=augment_train,
            augment_rotation_deg=augment_rotation_deg,
            augment_scale=augment_scale,
            augment_jitter_sigma=augment_jitter_sigma,
            augment_jitter_clip=augment_jitter_clip,
            augment_dropout=augment_dropout,
            augment_translate=augment_translate,
            augment_random_points_ratio=augment_random_points_ratio,
            augment_random_points_scale=augment_random_points_scale,
            use_normals=use_normals,
            normal_k=normal_k,
            preload_data=preload_data,
            multi_view=multi_view,
            multi_view_axes=multi_view_axes,
            multi_view_num_points=multi_view_num_points,
            multi_view_bins=multi_view_bins,
            obj_features_include_sensor_dist=obj_features_include_sensor_dist,
        )

    def _collect_classification_samples(self):
        self.root_dir = self.review_root_dir
        all_records = []

        if _has_flat_split_layout(self.review_root_dir):
            self._collect_flat_classification_samples()
            return
        if _has_bucket_first_review_layout(self.review_root_dir):
            self._collect_bucket_first_classification_samples()
            return

        class_dirs = [
            p for p in sorted(self.review_root_dir.iterdir())
            if p.is_dir() and p.name != "raw_frames" and p.name not in self.review_exclude_classes
        ]
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            self.classes.append(class_dir.name)

            for day_dir in sorted(class_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                try:
                    day = _parse_date(day_dir.name)
                except ValueError:
                    continue
                if self.review_start_date is not None and day < self.review_start_date:
                    continue

                for bucket in self.review_buckets:
                    source_root = day_dir / bucket / self.review_source_dir
                    if not source_root.is_dir():
                        continue

                    for pcd_file in sorted(source_root.glob("*.pcd")):
                        point_count = _pcd_point_count(pcd_file)
                        if (
                            self.review_min_points > 0
                            and class_dir.name not in self.review_min_points_exempt_classes
                            and point_count < self.review_min_points
                        ):
                            continue

                        sample_id, object_id = self._parse_review_ids(pcd_file)
                        all_records.append(
                            {
                                "class_name": class_dir.name,
                                "label": idx,
                                "object_id": object_id,
                                "sample_id": sample_id,
                                "date": day_dir.name,
                                "bucket": bucket,
                                "file": pcd_file,
                                "point_count": point_count,
                                "dist": self._metadata_distance(pcd_file),
                            }
                        )

        all_records = self._limit_records_per_class(all_records)
        split_by_key = self._stratified_split_map(all_records)
        self.samples = [
            record
            for record in all_records
            if split_by_key[self._record_key(record)] == self.split
        ]
        self._min_sensor_distance = 0.0

    def _collect_bucket_first_classification_samples(self):
        class_names = sorted(
            {
                class_dir.name
                for bucket_dir in self.review_root_dir.iterdir()
                if bucket_dir.is_dir() and bucket_dir.name.startswith("gt-pred")
                for class_dir in bucket_dir.iterdir()
                if class_dir.is_dir() and class_dir.name not in self.review_exclude_classes
            }
        )
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)

        all_records = []
        for class_name in class_names:
            label = self.class_to_idx[class_name]
            for bucket in self.review_buckets:
                bucket_root = self.review_root_dir / bucket / class_name
                if not bucket_root.is_dir():
                    continue
                pcd_root = bucket_root / "pcds" if (bucket_root / "pcds").is_dir() else bucket_root
                for pcd_file in sorted(pcd_root.glob("*.pcd")):
                    point_count = _pcd_point_count(pcd_file)
                    if (
                        self.review_min_points > 0
                        and class_name not in self.review_min_points_exempt_classes
                        and point_count < self.review_min_points
                    ):
                        continue

                    sample_id, object_id = self._parse_review_ids(pcd_file)
                    all_records.append(
                        {
                            "class_name": class_name,
                            "label": label,
                            "object_id": object_id,
                            "sample_id": sample_id,
                            "date": "bucket-first",
                            "bucket": bucket,
                            "file": pcd_file,
                            "point_count": point_count,
                            "dist": self._metadata_distance(pcd_file),
                        }
                    )

        all_records = self._limit_records_per_class(all_records)
        split_by_key = self._stratified_split_map(all_records)
        self.samples = [
            record
            for record in all_records
            if split_by_key[self._record_key(record)] == self.split
        ]
        self._min_sensor_distance = 0.0

    def _limit_records_per_class(self, records):
        if self.review_max_samples_per_class is None or self.review_max_samples_per_class <= 0:
            return records

        limited = []
        for class_name in self.classes:
            class_records = [record for record in records if record["class_name"] == class_name]
            class_records = sorted(class_records, key=self._record_sort_key)
            limited.extend(class_records[: self.review_max_samples_per_class])
        return limited

    def _collect_flat_classification_samples(self):
        split_root = self.review_root_dir / self.split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        class_names = sorted(
            {
                class_dir.name
                for split_name in ("train", "val", "test")
                if (self.review_root_dir / split_name).is_dir()
                for class_dir in (self.review_root_dir / split_name).iterdir()
                if class_dir.is_dir() and class_dir.name not in self.review_exclude_classes
            }
        )
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)

        for class_name in class_names:
            class_dir = split_root / class_name
            if not class_dir.is_dir():
                continue
            label = self.class_to_idx[class_name]
            for pcd_file in sorted(class_dir.glob("*.pcd")):
                point_count = _pcd_point_count(pcd_file)
                if (
                    self.review_min_points > 0
                    and class_name not in self.review_min_points_exempt_classes
                    and point_count < self.review_min_points
                ):
                    continue

                sample_id, object_id = self._parse_review_ids(pcd_file)
                self.samples.append(
                    {
                        "class_name": class_name,
                        "label": label,
                        "object_id": object_id,
                        "sample_id": sample_id,
                        "date": "flat",
                        "bucket": "flat",
                        "file": pcd_file,
                        "point_count": point_count,
                        "dist": self._metadata_distance(pcd_file),
                    }
                )
        self._min_sensor_distance = 0.0

    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = self._load_points(sample["file"])
        raw_dist = sample.get("dist", self._metadata_distance(sample["file"]))
        obj_features = self._obj_features(points, raw_dist)

        if self.normalize:
            points = self._normalize(points)

        points = self._apply_point_dropout(points)
        points = self._include_random_points(points)
        points = self._resample(points, target_count=self.num_points, jitter=self.split == "train")
        if self.normalize:
            points = self._scale_to_unit_radius(points)[0]
        points = self._apply_geometric_augmentations(points)

        output = {
            "x": torch.from_numpy(self._point_features(points)).float(),
            "pos": torch.from_numpy(points).float(),
            "y": torch.tensor(sample["label"]).long(),
            "obj_features": torch.from_numpy(obj_features).float(),
            "object_id": sample["object_id"],
            "sample_id": sample["sample_id"],
            "date": sample["date"],
            "bucket": sample["bucket"],
        }
        return self._add_multi_view_if_needed(output, points)


class ModelNet40OffClassificationDataset(PointCloudDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_points=1024,
        normalize=True,
        jitter_std=0.01,
        augment_train=False,
        augment_rotation_deg=20.0,
        augment_scale=(0.8, 1.2),
        augment_jitter_sigma=0.01,
        augment_jitter_clip=0.05,
        augment_dropout=(0.0, 0.0),
        augment_translate=0.0,
        augment_random_points_ratio=0.0,
        augment_random_points_scale=1.0,
        use_normals=False,
        normal_k=16,
        preload_data=False,
        multi_view=False,
        multi_view_axes=("xy", "xz", "yz"),
        multi_view_num_points=512,
        multi_view_bins=256,
    ):
        self.modelnet40_root_dir = Path(root_dir)
        super().__init__(
            root_dir=root_dir,
            split=split,
            num_points=num_points,
            normalize=normalize,
            jitter_std=jitter_std,
            task="classification",
            augment_train=augment_train,
            augment_rotation_deg=augment_rotation_deg,
            augment_scale=augment_scale,
            augment_jitter_sigma=augment_jitter_sigma,
            augment_jitter_clip=augment_jitter_clip,
            augment_dropout=augment_dropout,
            augment_translate=augment_translate,
            augment_random_points_ratio=augment_random_points_ratio,
            augment_random_points_scale=augment_random_points_scale,
            use_normals=use_normals,
            normal_k=normal_k,
            preload_data=preload_data,
            multi_view=multi_view,
            multi_view_axes=multi_view_axes,
            multi_view_num_points=multi_view_num_points,
            multi_view_bins=multi_view_bins,
        )

    def _collect_classification_samples(self):
        split_name = "train" if self.split == "train" else "test"
        self.root_dir = self.modelnet40_root_dir
        class_dirs = [p for p in sorted(self.modelnet40_root_dir.iterdir()) if p.is_dir()]
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            self.classes.append(class_dir.name)
            split_dir = class_dir / split_name
            if not split_dir.is_dir():
                continue
            for off_file in sorted(split_dir.glob("*.off")):
                self.samples.append((off_file, idx, 0.0))
        self._min_sensor_distance = 0.0


def get_modelnet40_off_classification_dataloader(
    dataset_root,
    split,
    batch_size=32,
    num_points=1024,
    shuffle=True,
    num_workers=4,
    class_balanced_batches=False,
    balanced_max_repeat_per_sample=None,
    augment_train=False,
    augment_rotation_deg=20.0,
    augment_scale=(0.8, 1.2),
    augment_jitter_sigma=0.01,
    augment_jitter_clip=0.05,
    augment_dropout=(0.0, 0.0),
    augment_translate=0.0,
    augment_random_points_ratio=0.0,
    augment_random_points_scale=1.0,
    use_normals=False,
    normal_k=16,
    preload_data=False,
    multi_view=False,
    multi_view_axes=("xy", "xz", "yz"),
    multi_view_num_points=512,
    multi_view_bins=256,
):
    dataset = ModelNet40OffClassificationDataset(
        root_dir=dataset_root,
        split=split,
        num_points=num_points,
        augment_train=augment_train,
        augment_rotation_deg=augment_rotation_deg,
        augment_scale=augment_scale,
        augment_jitter_sigma=augment_jitter_sigma,
        augment_jitter_clip=augment_jitter_clip,
        augment_dropout=augment_dropout,
        augment_translate=augment_translate,
        augment_random_points_ratio=augment_random_points_ratio,
        augment_random_points_scale=augment_random_points_scale,
        use_normals=use_normals,
        normal_k=normal_k,
        preload_data=preload_data,
        multi_view=multi_view,
        multi_view_axes=multi_view_axes,
        multi_view_num_points=multi_view_num_points,
        multi_view_bins=multi_view_bins,
    )
    return _make_dataloader(
        dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        class_balanced_batches=class_balanced_batches,
        balanced_max_repeat_per_sample=balanced_max_repeat_per_sample,
    )


def get_review_classification_dataloader(
    dataset_root,
    split,
    batch_size=32,
    num_points=1024,
    shuffle=True,
    num_workers=4,
    start_date=None,
    min_points=0,
    split_ratios=(0.8, 0.1, 0.1),
    buckets=None,
    source_dir="pred",
    exclude_classes=("reject",),
    min_points_exempt_classes=("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
    class_balanced_batches=True,
    balanced_max_repeat_per_sample=None,
    class_rebalance_mode=None,
    rebalance_max_repeat_per_sample=None,
    randomize_labels=False,
    random_label_seed=0,
    random_label_mode="permute",
    augment_train=False,
    augment_rotation_deg=20.0,
    augment_scale=(0.8, 1.2),
    augment_jitter_sigma=0.01,
    augment_jitter_clip=0.05,
    augment_dropout=(0.2, 0.5),
    augment_translate=0.1,
    augment_random_points_ratio=0.0,
    augment_random_points_scale=1.0,
    use_normals=False,
    normal_k=16,
    preload_data=False,
    max_samples_per_class=None,
    multi_view=False,
    multi_view_axes=("xy", "xz", "yz"),
    multi_view_num_points=512,
    multi_view_bins=256,
    obj_features_include_sensor_dist=True,
):
    dataset = ReviewClassificationDataset(
        root_dir=dataset_root,
        split=split,
        num_points=num_points,
        start_date=start_date,
        min_points=min_points,
        split_ratios=split_ratios,
        buckets=buckets,
        source_dir=source_dir,
        exclude_classes=exclude_classes,
        min_points_exempt_classes=min_points_exempt_classes,
        augment_train=augment_train,
        augment_rotation_deg=augment_rotation_deg,
        augment_scale=augment_scale,
        augment_jitter_sigma=augment_jitter_sigma,
        augment_jitter_clip=augment_jitter_clip,
        augment_dropout=augment_dropout,
        augment_translate=augment_translate,
        augment_random_points_ratio=augment_random_points_ratio,
        augment_random_points_scale=augment_random_points_scale,
        use_normals=use_normals,
        normal_k=normal_k,
        preload_data=preload_data,
        max_samples_per_class=max_samples_per_class,
        multi_view=multi_view,
        multi_view_axes=multi_view_axes,
        multi_view_num_points=multi_view_num_points,
        multi_view_bins=multi_view_bins,
        obj_features_include_sensor_dist=obj_features_include_sensor_dist,
    )
    if randomize_labels:
        randomize_dataset_labels(dataset, seed=random_label_seed, mode=random_label_mode)

    return _make_dataloader(
        dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        class_balanced_batches=class_balanced_batches,
        balanced_max_repeat_per_sample=balanced_max_repeat_per_sample,
        class_rebalance_mode=class_rebalance_mode,
        rebalance_max_repeat_per_sample=rebalance_max_repeat_per_sample,
    )


class RawFramesClassificationDataset(PointCloudDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_points=1024,
        normalize=True,
        jitter_std=0.01,
        start_date=None,
        min_points=0,
        split_ratios=(0.8, 0.1, 0.1),
        exclude_classes=("reject",),
        min_points_exempt_classes=("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
        forced_classes=None,
        frame_selection="all",
        object_multi_view=False,
        max_views=5,
        view_selection="uniform",
        pose_metadata_root=None,
        pose_required=False,
        augment_train=False,
        augment_rotation_deg=20.0,
        augment_scale=(0.8, 1.2),
        augment_jitter_sigma=0.01,
        augment_jitter_clip=0.05,
        augment_dropout=(0.2, 0.5),
        augment_translate=0.1,
        augment_random_points_ratio=0.0,
        augment_random_points_scale=1.0,
        use_normals=False,
        normal_k=16,
        preload_data=False,
        multi_view=False,
        multi_view_axes=("xy", "xz", "yz"),
        multi_view_num_points=512,
        multi_view_bins=256,
        obj_features_include_sensor_dist=True,
    ):
        self.raw_frames_root_dir = Path(root_dir)
        self.raw_frames_start_date = _parse_date(start_date)
        self.raw_frames_min_points = int(min_points or 0)
        self.raw_frames_split_ratios = _as_float_tuple(split_ratios)
        self.raw_frames_exclude_classes = set(_as_name_list(exclude_classes))
        self.raw_frames_min_points_exempt_classes = set(_as_name_list(min_points_exempt_classes))
        self.raw_frames_forced_classes = _as_name_list(forced_classes)
        self.raw_frames_frame_selection = str(frame_selection or "all").lower()
        self.raw_frames_object_multi_view = _as_bool(object_multi_view)
        self.raw_frames_max_views = max(1, int(max_views or 5))
        self.raw_frames_view_selection = str(view_selection or "uniform").lower()
        self.raw_frames_pose_metadata_root = (
            Path(pose_metadata_root) if pose_metadata_root else None
        )
        self.raw_frames_pose_required = _as_bool(pose_required)
        self._raw_frames_pose_cache = {}
        super().__init__(
            root_dir=root_dir,
            split=split,
            num_points=num_points,
            normalize=normalize,
            jitter_std=jitter_std,
            task="classification",
            augment_train=augment_train,
            augment_rotation_deg=augment_rotation_deg,
            augment_scale=augment_scale,
            augment_jitter_sigma=augment_jitter_sigma,
            augment_jitter_clip=augment_jitter_clip,
            augment_dropout=augment_dropout,
            augment_translate=augment_translate,
            augment_random_points_ratio=augment_random_points_ratio,
            augment_random_points_scale=augment_random_points_scale,
            use_normals=use_normals,
            normal_k=normal_k,
            preload_data=preload_data,
            multi_view=multi_view,
            multi_view_axes=multi_view_axes,
            multi_view_num_points=multi_view_num_points,
            multi_view_bins=multi_view_bins,
            obj_features_include_sensor_dist=obj_features_include_sensor_dist,
        )


    def _raw_frames_class_names(self, available_names):
        available = sorted({name for name in available_names if name not in self.raw_frames_exclude_classes})
        if not self.raw_frames_forced_classes:
            return available
        class_names = [name for name in self.raw_frames_forced_classes if name not in self.raw_frames_exclude_classes]
        extras = [name for name in available if name not in class_names]
        return class_names + extras

    def _collect_classification_samples(self):
        self.root_dir = self.raw_frames_root_dir
        all_records = []

        if _has_flat_split_layout(self.raw_frames_root_dir):
            self._collect_flat_classification_samples()
            return

        class_dirs = {
            p.name: p for p in sorted(self.raw_frames_root_dir.iterdir())
            if p.is_dir() and p.name not in self.raw_frames_exclude_classes
        }
        class_names = self._raw_frames_class_names(class_dirs.keys())
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)

        for class_name in class_names:
            class_dir = class_dirs.get(class_name)
            if class_dir is None:
                continue
            label = self.class_to_idx[class_name]
            pcd_roots = [class_dir / "pcds"] if (class_dir / "pcds").is_dir() else [
                p for p in sorted(class_dir.iterdir()) if p.is_dir()
            ]
            for pcd_root in pcd_roots:
                object_id_from_dir = None if pcd_root.name == "pcds" else pcd_root.name

                for pcd_file in sorted(pcd_root.glob("*.pcd")):
                    metadata = self._load_metadata(pcd_file)
                    object_id = object_id_from_dir or self._parse_object_id(pcd_file)
                    day = metadata.get("day")
                    if day:
                        try:
                            parsed_day = _parse_date(day)
                        except ValueError:
                            parsed_day = None
                        if (
                            self.raw_frames_start_date is not None
                            and parsed_day is not None
                            and parsed_day < self.raw_frames_start_date
                        ):
                            continue

                    point_count = metadata.get("point_count")
                    if point_count is None:
                        point_count = _pcd_point_count(pcd_file)
                    point_count = int(point_count)
                    if (
                        self.raw_frames_min_points > 0
                        and class_dir.name not in self.raw_frames_min_points_exempt_classes
                        and point_count < self.raw_frames_min_points
                    ):
                        continue

                    frame_id = metadata.get("frame_index") or self._parse_frame_id(pcd_file)
                    sample_id = metadata.get("sample_id") or pcd_file.stem
                    record = {
                        "file": pcd_file,
                        "label": label,
                        "class_name": class_dir.name,
                        "object_id": str(metadata.get("gt_object_id") or object_id),
                        "sample_id": str(sample_id),
                        "group_id": str(metadata.get("sample_id") or re.sub(r"__frame_[^_]+$", "", pcd_file.stem)),
                        "frame_id": str(frame_id),
                        "dist": self._metadata_distance(pcd_file, metadata=metadata),
                        "extent": self._metadata_extent(pcd_file, metadata=metadata),
                        "point_count": point_count,
                    }
                    all_records.append(record)

        all_records = self._select_raw_frame_records(all_records)
        split_by_key = self._stratified_split_map(all_records)
        split_records = [
            record
            for record in all_records
            if split_by_key[self._record_key(record)] == self.split
        ]
        self.samples = self._group_object_views(split_records)
        self._min_sensor_distance = min((record["dist"] for record in self.samples), default=0.0)

    def _collect_flat_classification_samples(self):
        split_root = self.raw_frames_root_dir / self.split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        available_class_names = {
            class_dir.name
            for split_name in ("train", "val", "test")
            if (self.raw_frames_root_dir / split_name).is_dir()
            for class_dir in (self.raw_frames_root_dir / split_name).iterdir()
            if class_dir.is_dir()
        }
        class_names = self._raw_frames_class_names(available_class_names)
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)

        for class_name in class_names:
            class_dir = split_root / class_name
            if not class_dir.is_dir():
                continue
            label = self.class_to_idx[class_name]
            for pcd_file in sorted(class_dir.glob("*.pcd")):
                metadata = self._load_metadata(pcd_file)
                point_count = metadata.get("point_count") if isinstance(metadata, dict) else None
                if point_count is None:
                    point_count = _pcd_point_count(pcd_file)
                point_count = int(point_count)
                if (
                    self.raw_frames_min_points > 0
                    and class_name not in self.raw_frames_min_points_exempt_classes
                    and point_count < self.raw_frames_min_points
                ):
                    continue

                object_id = str(metadata.get("gt_object_id") or self._parse_object_id(pcd_file))
                frame_id = metadata.get("frame_index") or self._parse_frame_id(pcd_file)
                sample_id = metadata.get("sample_id") or pcd_file.stem
                self.samples.append(
                    {
                        "file": pcd_file,
                        "label": label,
                        "class_name": class_name,
                        "object_id": object_id,
                        "group_id": str(metadata.get("sample_id") or re.sub(r"__frame_[^_]+$", "", pcd_file.stem)),
                        "sample_id": str(sample_id),
                        "frame_id": str(frame_id),
                        "dist": self._metadata_distance(pcd_file, metadata=metadata),
                        "extent": self._metadata_extent(pcd_file, metadata=metadata),
                        "point_count": point_count,
                    }
                )
        self.samples = self._select_raw_frame_records(self.samples)
        self.samples = self._group_object_views(self.samples)
        self._min_sensor_distance = min((record["dist"] for record in self.samples), default=0.0)

    def _frame_pose_transform(self, record):
        if self.raw_frames_pose_metadata_root is None:
            return None
        aggregate_stem = re.sub(
            r"__frame_[^_]+$", "", Path(record["file"]).stem
        )
        metadata_path = (
            self.raw_frames_pose_metadata_root
            / record["class_name"] / "json" / f"{aggregate_stem}.json"
        )
        cached = self._raw_frames_pose_cache.get(metadata_path)
        if cached is None:
            pose_by_frame = {}
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                for item in metadata.get("final_frame_poses", []):
                    pose = item.get("pose_6d_radians")
                    transform = item.get("transform")
                    if not item.get("accepted", False) or pose is None or transform is None:
                        continue
                    pose_by_frame[str(int(item["frame_id"]))] = np.asarray(
                        transform, dtype=np.float32
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pose_by_frame = {}
            self._raw_frames_pose_cache[metadata_path] = pose_by_frame
            cached = pose_by_frame
        try:
            frame_id = str(int(record["frame_id"]))
        except (TypeError, ValueError):
            frame_id = str(record["frame_id"])
        return cached.get(frame_id)

    def _filter_pose_records(self, records):
        if self.raw_frames_pose_metadata_root is None:
            return records
        filtered = []
        for record in records:
            transform = self._frame_pose_transform(record)
            if transform is None and self.raw_frames_pose_required:
                continue
            record = dict(record)
            record["pose_transform"] = transform
            filtered.append(record)
        return filtered

    def _group_object_views(self, records):
        if not self.raw_frames_object_multi_view:
            return records

        grouped = {}
        for record in records:
            key = self._record_group_key(record)
            grouped.setdefault(key, []).append(record)

        samples = []
        for key in sorted(grouped):
            views = self._select_object_views(grouped[key])
            representative = max(views, key=self._largest_extent_sort_key)
            sample = dict(representative)
            sample["view_records"] = views
            sample["num_views"] = len(views)
            samples.append(sample)
        return samples

    def _select_object_views(self, records):
        records = sorted(records, key=self._frame_order_key)
        if len(records) <= self.raw_frames_max_views:
            return records
        if self.raw_frames_view_selection in {"largest_extent", "max_extent"}:
            selected = sorted(
                records,
                key=self._largest_extent_sort_key,
                reverse=True,
            )[:self.raw_frames_max_views]
            return sorted(selected, key=self._frame_order_key)
        if self.raw_frames_view_selection not in {"uniform", "even"}:
            raise ValueError(
                f"Unsupported raw_frames_view_selection: {self.raw_frames_view_selection}"
            )
        indices = np.linspace(0, len(records) - 1, self.raw_frames_max_views, dtype=np.int64)
        return [records[int(index)] for index in indices]

    def _frame_order_key(self, record):
        frame_id = str(record.get("frame_id", ""))
        try:
            frame_key = (0, int(frame_id))
        except ValueError:
            frame_key = (1, frame_id)
        return frame_key, self._record_sort_key(record)

    def _select_raw_frame_records(self, records):
        records = self._filter_pose_records(records)
        if self.raw_frames_frame_selection in {"all", "", "none"}:
            return records
        if self.raw_frames_frame_selection not in {"largest_extent", "max_extent"}:
            raise ValueError(
                f"Unsupported raw_frames_frame_selection: {self.raw_frames_frame_selection}"
            )

        selected = {}
        for record in records:
            group_key = self._record_group_key(record)
            previous = selected.get(group_key)
            if previous is None or self._largest_extent_sort_key(record) > self._largest_extent_sort_key(previous):
                selected[group_key] = record
        return [selected[key] for key in sorted(selected)]

    def _largest_extent_sort_key(self, record):
        return (
            float(record.get("extent", 0.0) or 0.0),
            int(record.get("point_count", 0) or 0),
            self._record_sort_key(record),
        )

    @staticmethod
    def _parse_object_id(file_path: Path):
        match = re.search(r"__object_([^_]+)", file_path.stem)
        return match.group(1) if match else file_path.stem

    @staticmethod
    def _load_metadata(file_path: Path):
        candidates = [file_path.with_suffix(".json")]
        if file_path.parent.name == "pcds":
            stem_without_frame = re.sub(r"__frame_[^_]+$", "", file_path.stem)
            candidates.append(file_path.parent.parent / "json" / f"{stem_without_frame}.json")
        for meta_path in candidates:
            if not meta_path.exists():
                continue
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
        return {}

    @staticmethod
    def _parse_frame_id(file_path: Path):
        match = re.search(r"__frame_([^_]+)$", file_path.stem)
        return match.group(1) if match else file_path.stem

    @staticmethod
    def _record_key(record):
        return f"{record['class_name']}::{record['object_id']}::{record['sample_id']}::{record['frame_id']}"

    @staticmethod
    def _record_group_key(record):
        return f"{record['class_name']}::{record.get('group_id', record['object_id'])}"

    def _record_sort_key(self, record):
        key = self._record_key(record)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return digest, key

    def _group_sort_key(self, group_key):
        digest = hashlib.md5(group_key.encode("utf-8")).hexdigest()
        return digest, group_key

    def _split_counts(self, n_records):
        ratios = np.asarray(self.raw_frames_split_ratios, dtype=np.float64)
        if ratios.shape[0] != 3:
            raise ValueError(f"raw_frames_split_ratios muss 3 Werte enthalten: {self.raw_frames_split_ratios}")
        ratio_sum = float(ratios.sum())
        if ratio_sum <= 0:
            raise ValueError(f"raw_frames_split_ratios muss eine positive Summe haben: {self.raw_frames_split_ratios}")
        ratios = ratios / ratio_sum

        raw = ratios * n_records
        counts = np.floor(raw).astype(np.int64)
        remainder = int(n_records - counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for idx in order[:remainder]:
                counts[idx] += 1
        return counts.tolist()

    def _stratified_split_map(self, records):
        if self.split not in {"train", "val", "test"}:
            return {self._record_key(record): self.split for record in records}

        groups_by_id = {}
        for record in records:
            groups_by_id.setdefault(self._record_group_key(record), []).append(record)

        groups_by_class_key = {}
        for group_key, group_records in groups_by_id.items():
            class_key = "::".join(sorted({record["class_name"] for record in group_records}))
            groups_by_class_key.setdefault(class_key, {})[group_key] = group_records

        split_by_key = {}
        split_names = ("train", "val", "test")
        for class_groups in groups_by_class_key.values():
            group_keys = sorted(class_groups, key=self._group_sort_key)
            counts = self._split_counts(len(group_keys))
            offset = 0
            for split_name, count in zip(split_names, counts):
                for group_key in group_keys[offset: offset + count]:
                    for record in sorted(class_groups[group_key], key=self._record_sort_key):
                        split_by_key[self._record_key(record)] = split_name
                offset += count
        return split_by_key

    @staticmethod
    def _load_xyz(file_path: Path):
        if file_path.suffix.lower() == ".pcd":
            return _load_pcd_xyz(file_path)
        return PointCloudDataset._load_xyz(file_path)

    @staticmethod
    def _metadata_distance(file_path: Path, metadata=None):
        if metadata is None:
            metadata = RawFramesClassificationDataset._load_metadata(file_path)
        metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
        for key in ("distance_to_exit_line", "closest_edge_distance"):
            if key in metrics:
                return abs(float(metrics[key]))
        return 0.0

    @staticmethod
    def _metadata_extent(file_path: Path, metadata=None):
        if metadata is None:
            metadata = RawFramesClassificationDataset._load_metadata(file_path)
        metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
        extents = []
        for key in ("x_extent", "y_extent", "z_extent"):
            value = metrics.get(key)
            if value is not None:
                extents.append(abs(float(value)))
        if extents:
            return max(extents)

        points = RawFramesClassificationDataset._load_xyz(file_path)
        mins = points[:, :3].min(axis=0)
        maxs = points[:, :3].max(axis=0)
        return float(np.max(maxs - mins))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.raw_frames_object_multi_view:
            return self._get_object_views(sample)

        points = self._load_points(sample["file"])
        obj_features = self._obj_features(points, sample["dist"])

        if self.normalize:
            points = self._normalize(points)

        points = self._apply_point_dropout(points)
        points = self._include_random_points(points)
        points = self._resample(points, target_count=self.num_points, jitter=self.split == "train")
        if self.normalize:
            points = self._scale_to_unit_radius(points)[0]
        points = self._apply_geometric_augmentations(points)

        output = {
            "x": torch.from_numpy(self._point_features(points)).float(),
            "pos": torch.from_numpy(points).float(),
            "y": torch.tensor(sample["label"]).long(),
            "obj_features": torch.from_numpy(obj_features).float(),
            "object_id": sample["object_id"],
            "sample_id": sample["sample_id"],
            "frame_id": sample["frame_id"],
        }
        return self._add_multi_view_if_needed(output, points)

    def _prepare_view_points(self, record, apply_geometric_augmentations=True):
        points = self._load_points(record["file"])
        if self.normalize:
            points = self._normalize(points)
        points = self._apply_point_dropout(points)
        points = self._include_random_points(points)
        points = self._resample(
            points,
            target_count=self.num_points,
            jitter=self.split == "train",
        )
        if self.normalize:
            points = self._scale_to_unit_radius(points)[0]
        if apply_geometric_augmentations:
            points = self._apply_geometric_augmentations(points)
        return points

    def _get_object_views(self, sample):
        view_records = sample["view_records"]
        pose_supervision = self.raw_frames_pose_metadata_root is not None
        normalization_centroid = np.zeros(3, dtype=np.float32)
        normalization_radius = 1.0

        if pose_supervision:
            raw_views = [self._load_points(record["file"]) for record in view_records]
            if self.normalize:
                normalization_centroid = np.mean(raw_views[0], axis=0)
                centered_views = [
                    points - normalization_centroid for points in raw_views
                ]
                normalization_radius = self._unit_radius(
                    np.concatenate(centered_views, axis=0)
                )
            prepared_views = []
            for points in raw_views:
                if self.normalize:
                    points = self._normalize_with_centroid(
                        points, normalization_centroid, normalization_radius
                    )
                points = self._apply_point_dropout(points)
                points = self._include_random_points(points)
                points = self._resample(
                    points, target_count=self.num_points,
                    jitter=self.split == "train",
                )
                prepared_views.append(points.astype(np.float32, copy=False))
        else:
            prepared_views = [
                self._prepare_view_points(
                    record, apply_geometric_augmentations=False
                )
                for record in view_records
            ]
            if prepared_views:
                stacked_views = np.concatenate(prepared_views, axis=0)
                stacked_views = self._apply_geometric_augmentations(stacked_views)
                prepared_views = list(
                    stacked_views.reshape(
                        len(prepared_views), self.num_points,
                        stacked_views.shape[-1],
                    )
                )

        feature_dim = self._point_features(prepared_views[0]).shape[1]
        views = np.zeros(
            (self.raw_frames_max_views, self.num_points, feature_dim),
            dtype=np.float32,
        )
        view_mask = np.zeros(self.raw_frames_max_views, dtype=np.bool_)
        pose_rotations = np.tile(
            np.eye(3, dtype=np.float32),
            (self.raw_frames_max_views, 1, 1),
        )
        pose_translations = np.zeros(
            (self.raw_frames_max_views, 3), dtype=np.float32
        )
        pose_mask = np.zeros(self.raw_frames_max_views, dtype=np.bool_)
        view_origins = np.zeros(
            (self.raw_frames_max_views, 3), dtype=np.float32
        )
        if pose_supervision and self.normalize:
            view_origins[:] = (-normalization_centroid / normalization_radius).astype(
                np.float32, copy=False
            )

        transforms = [record.get("pose_transform") for record in view_records]
        anchor_transform = transforms[0] if transforms else None
        anchor_inverse = (
            np.linalg.inv(anchor_transform)
            if anchor_transform is not None else None
        )
        for view_idx, (record, points) in enumerate(
                zip(view_records, prepared_views)):
            views[view_idx] = self._point_features(points)
            view_mask[view_idx] = True
            transform = record.get("pose_transform")
            if transform is None or anchor_inverse is None:
                continue
            relative = np.matmul(anchor_inverse, transform)
            rotation = relative[:3, :3].astype(np.float32, copy=False)
            translation = relative[:3, 3].astype(np.float32, copy=False)
            if self.normalize:
                translation = (
                    np.matmul(rotation, normalization_centroid)
                    + translation - normalization_centroid
                ) / normalization_radius
            pose_rotations[view_idx] = rotation
            pose_translations[view_idx] = translation
            pose_mask[view_idx] = True

        representative_points = prepared_views[0]
        sensor_dist = float(np.mean([record["dist"] for record in view_records]))
        obj_features = self._obj_features(representative_points, sensor_dist)
        return {
            "views": torch.from_numpy(views).float(),
            "view_mask": torch.from_numpy(view_mask),
            "pose_rotations": torch.from_numpy(pose_rotations).float(),
            "pose_translations": torch.from_numpy(pose_translations).float(),
            "pose_mask": torch.from_numpy(pose_mask),
            "view_origins": torch.from_numpy(view_origins).float(),
            "x": torch.from_numpy(views[0]).float(),
            "pos": torch.from_numpy(representative_points).float(),
            "y": torch.tensor(sample["label"]).long(),
            "obj_features": torch.from_numpy(obj_features).float(),
            "object_id": sample["object_id"],
            "sample_id": sample["sample_id"],
            "frame_id": sample["frame_id"],
        }


def get_raw_frames_classification_dataloader(
    dataset_root,
    split,
    batch_size=32,
    num_points=1024,
    shuffle=True,
    num_workers=4,
    start_date=None,
    min_points=0,
    split_ratios=(0.8, 0.1, 0.1),
    exclude_classes=("reject","TLS_VEHICLE_CAR_WITH_TRAILER", "TLS_VEHICLE_TRUCK_WITH_TRAILER"),
    min_points_exempt_classes=("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
    class_balanced_batches=True,
    balanced_max_repeat_per_sample=None,
    forced_classes=None,
    frame_selection="all",
    object_multi_view=False,
    max_views=5,
    view_selection="uniform",
    pose_metadata_root=None,
    pose_required=False,
    randomize_labels=False,
    random_label_seed=0,
    random_label_mode="permute",
    augment_train=False,
    augment_rotation_deg=20.0,
    augment_scale=(0.8, 1.2),
    augment_jitter_sigma=0.01,
    augment_jitter_clip=0.05,
    augment_dropout=(0.2, 0.5),
    augment_translate=0.1,
    augment_random_points_ratio=0.0,
    augment_random_points_scale=1.0,
    use_normals=False,
    normal_k=16,
    preload_data=False,
    multi_view=False,
    multi_view_axes=("xy", "xz", "yz"),
    multi_view_num_points=512,
    multi_view_bins=256,
    obj_features_include_sensor_dist=True,
):
    dataset = RawFramesClassificationDataset(
        root_dir=dataset_root,
        split=split,
        num_points=num_points,
        start_date=start_date,
        min_points=min_points,
        split_ratios=split_ratios,
        exclude_classes=exclude_classes,
        min_points_exempt_classes=min_points_exempt_classes,
        forced_classes=forced_classes,
        frame_selection=frame_selection,
        object_multi_view=object_multi_view,
        max_views=max_views,
        view_selection=view_selection,
        pose_metadata_root=pose_metadata_root,
        pose_required=pose_required,
        augment_train=augment_train,
        augment_rotation_deg=augment_rotation_deg,
        augment_scale=augment_scale,
        augment_jitter_sigma=augment_jitter_sigma,
        augment_jitter_clip=augment_jitter_clip,
        augment_dropout=augment_dropout,
        augment_translate=augment_translate,
        augment_random_points_ratio=augment_random_points_ratio,
        augment_random_points_scale=augment_random_points_scale,
        use_normals=use_normals,
        normal_k=normal_k,
        preload_data=preload_data,
        multi_view=multi_view,
        multi_view_axes=multi_view_axes,
        multi_view_num_points=multi_view_num_points,
        multi_view_bins=multi_view_bins,
        obj_features_include_sensor_dist=obj_features_include_sensor_dist,
    )
    if randomize_labels:
        randomize_dataset_labels(dataset, seed=random_label_seed, mode=random_label_mode)

    return _make_dataloader(
        dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        class_balanced_batches=class_balanced_batches,
        balanced_max_repeat_per_sample=balanced_max_repeat_per_sample,
    )



def _remap_classification_dataset(dataset, class_names):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    old_classes = list(dataset.classes)

    remapped_samples = []
    for sample in dataset.samples:
        if isinstance(sample, dict):
            class_name = sample.get("class_name", old_classes[int(sample["label"])])
            sample["label"] = class_to_idx[class_name]
            remapped_samples.append(sample)
        elif isinstance(sample, tuple):
            file_path, label, dist = sample
            class_name = old_classes[int(label)]
            remapped_samples.append((file_path, class_to_idx[class_name], dist))
        else:
            raise ValueError(f"Unsupported sample type for remapping: {sample}")

    dataset.samples = remapped_samples
    dataset.classes = list(class_names)
    dataset.class_to_idx = class_to_idx
    dataset.num_classes = len(class_names)
    return dataset


class MixedClassificationDataset(Dataset):
    def __init__(self, datasets, names):
        if len(datasets) != len(names):
            raise ValueError("datasets and names must have the same length")
        if not datasets:
            raise ValueError("MixedClassificationDataset needs at least one dataset")

        self.datasets = list(datasets)
        self.names = list(names)
        self.classes = sorted({class_name for dataset in self.datasets for class_name in dataset.classes})
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.num_points = self.datasets[0].num_points

        for dataset, name in zip(self.datasets[1:], self.names[1:]):
            if dataset.num_points != self.num_points:
                raise ValueError(
                    f"num_points mismatch in mixed dataset for {name}: "
                    f"{dataset.num_points} != {self.num_points}"
                )

        self.index = []
        self.samples = []
        for dataset_idx, (dataset, name) in enumerate(zip(self.datasets, self.names)):
            for sample_idx, sample in enumerate(dataset.samples):
                self.index.append((dataset_idx, sample_idx))
                if isinstance(sample, dict):
                    source_classes = list(dataset.classes)
                    class_name = sample.get("class_name", source_classes[int(sample["label"])])
                    label = self.class_to_idx[class_name]
                elif isinstance(sample, tuple):
                    source_classes = list(dataset.classes)
                    class_name = source_classes[int(sample[1])]
                    label = self.class_to_idx[class_name]
                else:
                    raise ValueError(f"Unsupported sample type in mixed dataset: {sample}")
                self.samples.append(
                    {
                        "label": int(label),
                        "class_name": class_name,
                        "source_dataset": name,
                    }
                )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        item = self.datasets[dataset_idx][sample_idx]
        label = self.samples[idx]["label"]
        if isinstance(item, dict):
            item = dict(item)
            item["y"] = torch.tensor(label).long()
            item["source_dataset"] = self.names[dataset_idx]
            for key in ("object_id", "sample_id", "frame_id", "date", "bucket"):
                item.setdefault(key, "")
        return item


def get_mixed_classification_dataloaders(
    normal_root,
    raw_frames_root,
    split,
    batch_size=32,
    num_points=1024,
    shuffle=True,
    num_workers=4,
    normal_start_date=None,
    normal_min_points=0,
    normal_split_ratios=(0.8, 0.1, 0.1),
    normal_buckets=None,
    normal_source_dir="pred",
    normal_exclude_classes=("reject",),
    raw_frames_start_date=None,
    raw_frames_min_points=0,
    raw_frames_split_ratios=(0.8, 0.1, 0.1),
    raw_frames_exclude_classes=("reject",),
    min_points_exempt_classes=("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
    class_balanced_batches=True,
    balanced_max_repeat_per_sample=None,
    randomize_labels=False,
    random_label_seed=0,
    random_label_mode="permute",
    augment_train=False,
    augment_rotation_deg=20.0,
    augment_scale=(0.8, 1.2),
    augment_jitter_sigma=0.01,
    augment_jitter_clip=0.05,
    augment_dropout=(0.2, 0.5),
    augment_translate=0.1,
    augment_random_points_ratio=0.0,
    augment_random_points_scale=1.0,
    use_normals=False,
    normal_k=16,
    preload_data=False,
    multi_view=False,
    multi_view_axes=("xy", "xz", "yz"),
    multi_view_num_points=512,
    multi_view_bins=256,
):
    normal_dataset = ReviewClassificationDataset(
        root_dir=normal_root,
        split=split,
        num_points=num_points,
        start_date=normal_start_date,
        min_points=normal_min_points,
        split_ratios=normal_split_ratios,
        buckets=normal_buckets,
        source_dir=normal_source_dir,
        exclude_classes=normal_exclude_classes,
        min_points_exempt_classes=min_points_exempt_classes,
        augment_train=augment_train,
        augment_rotation_deg=augment_rotation_deg,
        augment_scale=augment_scale,
        augment_jitter_sigma=augment_jitter_sigma,
        augment_jitter_clip=augment_jitter_clip,
        augment_dropout=augment_dropout,
        augment_translate=augment_translate,
        augment_random_points_ratio=augment_random_points_ratio,
        augment_random_points_scale=augment_random_points_scale,
        use_normals=use_normals,
        normal_k=normal_k,
        preload_data=preload_data,
        multi_view=multi_view,
        multi_view_axes=multi_view_axes,
        multi_view_num_points=multi_view_num_points,
        multi_view_bins=multi_view_bins,
    )
    raw_dataset = RawFramesClassificationDataset(
        root_dir=raw_frames_root,
        split=split,
        num_points=num_points,
        start_date=raw_frames_start_date,
        min_points=raw_frames_min_points,
        split_ratios=raw_frames_split_ratios,
        exclude_classes=raw_frames_exclude_classes,
        min_points_exempt_classes=min_points_exempt_classes,
        augment_train=augment_train,
        augment_rotation_deg=augment_rotation_deg,
        augment_scale=augment_scale,
        augment_jitter_sigma=augment_jitter_sigma,
        augment_jitter_clip=augment_jitter_clip,
        augment_dropout=augment_dropout,
        augment_translate=augment_translate,
        augment_random_points_ratio=augment_random_points_ratio,
        augment_random_points_scale=augment_random_points_scale,
        use_normals=use_normals,
        normal_k=normal_k,
        preload_data=preload_data,
        multi_view=multi_view,
        multi_view_axes=multi_view_axes,
        multi_view_num_points=multi_view_num_points,
        multi_view_bins=multi_view_bins,
    )

    class_names = sorted(set(normal_dataset.classes) | set(raw_dataset.classes))
    normal_dataset = _remap_classification_dataset(normal_dataset, class_names)
    raw_dataset = _remap_classification_dataset(raw_dataset, class_names)

    if split == "train":
        dataset = MixedClassificationDataset(
            [normal_dataset, raw_dataset],
            ["normal", "raw_frames"],
        )
        if randomize_labels:
            randomize_dataset_labels(dataset, seed=random_label_seed, mode=random_label_mode)
        return _make_dataloader(
            dataset,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            class_balanced_batches=class_balanced_batches,
            balanced_max_repeat_per_sample=balanced_max_repeat_per_sample,
        )

    return {
        "normal": _make_dataloader(
            normal_dataset,
            split=split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            class_balanced_batches=False,
        ),
        "raw_frames": _make_dataloader(
            raw_dataset,
            split=split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            class_balanced_batches=False,
        ),
    }

def get_review_completion_dataloader(
    dataset_root,
    split,
    batch_size=32,
    num_partial=1024,
    num_complete=2048,
    shuffle=True,
    num_workers=4,
    start_date=None,
    min_points=0,
    split_ratios=(0.8, 0.1, 0.1),
    buckets=None,
    exclude_classes=("reject",),
    min_points_exempt_classes=("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
    symmetry_completion=False,
    symmetry_axis="x",
    symmetry_source="partial",
    symmetry_keep_side="both",
    completion_mask=True,
    completion_mask_min_keep_ratio=0.35,
    completion_mask_max_keep_ratio=0.75,
    completion_mask_parts=2,
    class_balanced_batches=True,
):
    dataset = ReviewCompletionDataset(
        root_dir=dataset_root,
        split=split,
        num_points=num_partial,
        num_complete=num_complete,
        start_date=start_date,
        min_points=min_points,
        split_ratios=split_ratios,
        buckets=buckets,
        exclude_classes=exclude_classes,
        min_points_exempt_classes=min_points_exempt_classes,
        symmetry_completion=symmetry_completion,
        symmetry_axis=symmetry_axis,
        symmetry_source=symmetry_source,
        symmetry_keep_side=symmetry_keep_side,
        completion_mask=completion_mask,
        completion_mask_min_keep_ratio=completion_mask_min_keep_ratio,
        completion_mask_max_keep_ratio=completion_mask_max_keep_ratio,
        completion_mask_parts=completion_mask_parts,
    )

    return _make_dataloader(
        dataset,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        class_balanced_batches=class_balanced_batches,
    )


def get_completion_dataloader(
    dataset_root,
    split,
    batch_size=32,
    num_partial=1024,
    num_complete=2048,
    shuffle=True,
    num_workers=4,
    target_strategy="aggregate",
    partial_choice="random_far",
    completion_min_dist_delta=0.0,
    dataset_format="auto",
    review_start_date=None,
    review_min_points=0,
    review_split_ratios=(0.8, 0.1, 0.1),
    review_buckets=None,
    review_exclude_classes=("reject",),
    review_min_points_exempt_classes=("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
    symmetry_completion=False,
    symmetry_axis="x",
    symmetry_source="partial",
    symmetry_keep_side="both",
    completion_mask=True,
    completion_mask_min_keep_ratio=0.35,
    completion_mask_max_keep_ratio=0.75,
    completion_mask_parts=2,
    class_balanced_batches=True,
):
    if dataset_format == "auto":
        root = Path(dataset_root)
        has_standard_splits = any((root / name).is_dir() for name in ("train", "val", "test"))
        dataset_format = "standard" if has_standard_splits else "review"

    if dataset_format == "review":
        return get_review_completion_dataloader(
            dataset_root=dataset_root,
            split=split,
            batch_size=batch_size,
            num_partial=num_partial,
            num_complete=num_complete,
            shuffle=shuffle,
            num_workers=num_workers,
            start_date=review_start_date,
            min_points=review_min_points,
            split_ratios=review_split_ratios,
            buckets=review_buckets,
            exclude_classes=review_exclude_classes,
            min_points_exempt_classes=review_min_points_exempt_classes,
            symmetry_completion=symmetry_completion,
            symmetry_axis=symmetry_axis,
            symmetry_source=symmetry_source,
            symmetry_keep_side=symmetry_keep_side,
            completion_mask=completion_mask,
            completion_mask_min_keep_ratio=completion_mask_min_keep_ratio,
            completion_mask_max_keep_ratio=completion_mask_max_keep_ratio,
            completion_mask_parts=completion_mask_parts,
            class_balanced_batches=class_balanced_batches,
        )

    return get_dataloader(
        dataset_root=dataset_root,
        split=split,
        batch_size=batch_size,
        num_points=num_partial,
        shuffle=shuffle,
        num_workers=num_workers,
        num_complete=num_complete,
        task="completion_cls",
        target_strategy=target_strategy,
        partial_choice=partial_choice,
        completion_min_dist_delta=completion_min_dist_delta,
        symmetry_completion=symmetry_completion,
        symmetry_axis=symmetry_axis,
        symmetry_source=symmetry_source,
        symmetry_keep_side=symmetry_keep_side,
        completion_mask=completion_mask,
        completion_mask_min_keep_ratio=completion_mask_min_keep_ratio,
        completion_mask_max_keep_ratio=completion_mask_max_keep_ratio,
        completion_mask_parts=completion_mask_parts,
        class_balanced_batches=class_balanced_batches,
    )
