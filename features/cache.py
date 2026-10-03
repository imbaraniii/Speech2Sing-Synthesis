"""Save/load numpy feature arrays to a unified cache tree.

Layout:  cache_root / <feature_type> / <split> / <speaker>_<utt_id>.npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatureCache:
    def __init__(self, cache_root):
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, feature_type: str, split: str, key: str) -> Path:
        return self.root / feature_type / split / f"{key}.npy"

    def save(self, feature_type: str, split: str, key: str, array) -> None:
        path = self._path(feature_type, split, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    def load(self, feature_type: str, split: str, key: str) -> np.ndarray:
        return np.load(self._path(feature_type, split, key), allow_pickle=True)

    def exists(self, feature_type: str, split: str, key: str) -> bool:
        return self._path(feature_type, split, key).exists()

    def list_keys(self, feature_type: str, split: str) -> list:
        d = self.root / feature_type / split
        if not d.exists():
            return []
        return [p.stem for p in sorted(d.glob("*.npy"))]

    def summary(self) -> dict:
        counts = {}
        for feature_dir in sorted(self.root.iterdir()):
            if not feature_dir.is_dir():
                continue
            for split_dir in sorted(feature_dir.iterdir()):
                n = len(list(split_dir.glob("*.npy")))
                counts[f"{feature_dir.name}/{split_dir.name}"] = n
        return counts