import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re

MAX_DIMS = np.array([25.25, 2.60, 4.00], dtype=np.float32)
MAX_DIST = 20

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

        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        self._min_sensor_distance = None

        if self.is_completion:
            self._collect_completion_samples()
        else:
            self._collect_classification_samples()

        self.num_classes = len(self.class_to_idx)

    def _collect_classification_samples(self):
        # Klassen einsammeln
        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue

            self.class_to_idx[class_dir.name] = idx
            self.classes.append(class_dir.name)

            for f in class_dir.glob("*.npy"):
                dist = self._parse_sensor_distance(f)
                self.samples.append((f, idx, dist))

        if self.samples:
            self._min_sensor_distance = min(item[2] for item in self.samples)
        else:
            self._min_sensor_distance = 0.0

    def _collect_completion_samples(self):
        grouped = {}
        dists = []

        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue

            self.class_to_idx[class_dir.name] = idx
            self.classes.append(class_dir.name)

            for f in class_dir.glob("*.npy"):
                dist = self._parse_sensor_distance(f)
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
        points = np.load(file_path)
        points = points[:, :3].astype(np.float32, copy=False)
        if points.shape[0] == 0:
            raise ValueError(f"Leere Pointcloud: {file_path}")
        return points

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
            repeat_idx = np.random.choice(base_points.shape[0], missing, replace=True)
            extra_points = base_points[repeat_idx]
            points = np.concatenate([base_points, extra_points], axis=0)

            # Jitter NUR beim Oversampling
            if jitter and self.jitter_std > 0:
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.jitter_std,
                    size=points.shape
                )
                points = points + noise

        return points.astype(np.float32, copy=False)

    # ---------- Normalisierung ----------
    def _normalize(self, points):
        centroid = np.mean(points, axis=0)
        points = points - centroid

        return points

    @staticmethod
    def _normalize_with_centroid(points, centroid):
        return points - centroid

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
            return self._load_xyz(items[0]["file"])
        if self.target_strategy != "aggregate":
            raise ValueError(f"Unbekannte target_strategy: {self.target_strategy}")

        clouds = [self._load_xyz(item["file"]) for item in items]
        return np.concatenate(clouds, axis=0)

    def _obj_features(self, points, raw_dist):
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        obj_dims = (maxs - mins) / MAX_DIMS
        sensor_dist = float(raw_dist) / MAX_DIST
        return np.concatenate(
            [obj_dims.astype(np.float32), np.array([sensor_dist], dtype=np.float32)]
        )

    def _getitem_completion(self, idx):
        group = self.samples[idx]
        partial_item = self._choose_partial_item(group["items"])

        partial_raw = self._load_xyz(partial_item["file"])
        complete_raw = self._load_complete_target(group["items"])
        obj_features = self._obj_features(partial_raw, partial_item["dist"])

        if self.normalize:
            centroid = np.mean(partial_raw, axis=0)
            partial = self._normalize_with_centroid(partial_raw, centroid)
            complete = self._normalize_with_centroid(complete_raw, centroid)
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

        points = self._load_xyz(file_path)

        obj_features = self._obj_features(points, raw_dist)

        # 1) Normalisieren (wichtig: VOR Oversampling)
        if self.normalize:
            points = self._normalize(points)

        # 2) Resampling + ggf. Jitter
        points = self._resample(points)

        points = torch.from_numpy(points).float()
        label = torch.tensor(label).long()
        obj_features = torch.from_numpy(obj_features).float()

        return {
            "x": points,
            "y": label,
            "obj_features": obj_features,
        }


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
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        drop_last=(split == "train"),
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
):
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
    )
