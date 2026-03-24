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
        normalize=True,
        jitter_std=0.01
    ):
        self.root_dir = Path(root_dir) / split
        self.num_points = num_points
        self.normalize = normalize
        self.jitter_std = jitter_std

        self.samples = []
        self.class_to_idx = {}
        self._min_sensor_distance = None

        # Klassen einsammeln
        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue

            self.class_to_idx[class_dir.name] = idx

            for f in class_dir.glob("*.npy"):
                dist = self._parse_sensor_distance(f)
                self.samples.append((f, idx, dist))

        if self.samples:
            self._min_sensor_distance = min(item[2] for item in self.samples)
        else:
            self._min_sensor_distance = 0.0

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _parse_sensor_distance(file_path: Path) -> float:
        # Erwartetes Muster: "..._19m.npy" -> 19.0
        stem = file_path.stem
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)m$", stem)
        if match is None:
            raise ValueError(f"Keine Distanz im Dateinamen gefunden: {file_path.name}")
        return float(match.group(1))

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
        max_candidates = min(candidates.shape[0], max(target_count * 8, 4096))
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
    def _resample(self, points):
        n = points.shape[0]

        if n >= self.num_points:
            idx = self._stratified_fps_indices(points=points, target_count=self.num_points, bins=8)
            points = points[idx]
        else:
            base_idx = self._fps_indices(points=points, target_count=n)
            base_points = points[base_idx]

            missing = self.num_points - n
            repeat_idx = np.random.choice(base_points.shape[0], missing, replace=True)
            extra_points = base_points[repeat_idx]
            points = np.concatenate([base_points, extra_points], axis=0)

            # Jitter NUR beim Oversampling
            noise = np.random.normal(
                loc=0.0,
                scale=self.jitter_std,
                size=points.shape
            )
            points = points + noise

        return points

    # ---------- Normalisierung ----------
    def _normalize(self, points):
        centroid = np.mean(points, axis=0)
        points = points - centroid

        return points

    def __getitem__(self, idx):
        file_path, label, raw_dist = self.samples[idx]

        points = np.load(file_path)

        # nur xyz verwenden
        points = points[:, :3]

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        obj_dims = (maxs - mins) / MAX_DIMS
        sensor_dist = float(raw_dist) / MAX_DIST

        # 1) Normalisieren (wichtig: VOR Oversampling)
        if self.normalize:
            points = self._normalize(points)

        # 2) Resampling + ggf. Jitter
        points = self._resample(points)

        points = torch.from_numpy(points).float()
        label = torch.tensor(label).long()
        obj_features = torch.from_numpy(
            np.concatenate([obj_dims.astype(np.float32), np.array([sensor_dist], dtype=np.float32)])
        ).float()

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
    num_workers=4
):
    dataset = PointCloudDataset(
        root_dir=dataset_root,
        split=split,
        num_points=num_points
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        drop_last=(split == "train"),
    )
