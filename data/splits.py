"""Train/validation/test split logic (speaker-disjoint by default)."""

from __future__ import annotations

import random
from typing import Dict, List

SPEAKERS = [
    "ADIZ", "JLEE", "JTAN", "KENN", "MCUR", "MPOL",
    "MPUR", "NJAT", "PMAR", "SAMF", "VKOW", "ZHIY",
]


def speaker_disjoint_split(
    speakers: List[str],
    seed: int = 42,
    n_train: int = 10,
    n_val: int = 1,
) -> Dict[str, str]:
    """Return {speaker_id: 'train'|'val'|'test'} with disjoint speaker sets."""
    if n_train + n_val > len(speakers):
        raise ValueError("n_train + n_val must be <= len(speakers)")
    shuffled = list(speakers)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    split_map: Dict[str, str] = {}
    for i, spk in enumerate(shuffled):
        if i < n_train:
            split_map[spk] = "train"
        elif i < n_train + n_val:
            split_map[spk] = "val"
        else:
            split_map[spk] = "test"
    return split_map


def list_speakers(data_root) -> List[str]:
    """Return sorted speaker IDs present in the corpus directory."""
    from pathlib import Path

    data_root = Path(data_root)
    speakers = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            speakers.append(d.name)
    return speakers