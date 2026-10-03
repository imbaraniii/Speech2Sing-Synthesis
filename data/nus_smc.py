"""Dataset class that indexes the corpus and builds a split manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .annotations import load_all_annotations
from .splits import speaker_disjoint_split

DEFAULT_UTT_IDS = None  # auto-discovered per speaker when None


class NUS_SMC_Dataset:
    """Indexes all (read, sing) utterance pairs for a given split map."""

    def __init__(
        self,
        data_root,
        split_map: Optional[Dict[str, str]] = None,
        utterance_ids: Optional[List[str]] = None,
        seed: int = 42,
        drop_unpaired: bool = True,
    ):
        self.data_root = Path(data_root)
        self.utterance_ids = utterance_ids  # None -> auto-discovered per speaker
        self.annotations = load_all_annotations(self.data_root)

        if split_map is None:
            split_map = speaker_disjoint_split(
                self.speakers_in_corpus(), seed=seed
            )
        self.split_map = split_map

        self.items = self._build_index(drop_unpaired)

    def speakers_in_corpus(self) -> List[str]:
        return sorted(
            d.name for d in self.data_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _speaker_utt_ids(self, speaker_dir: Path) -> List[str]:
        """Intersection of read/ and sing/ utterance ids for one speaker."""
        read_ids = {p.stem for p in (speaker_dir / "read").glob("*.txt")}
        sing_ids = {p.stem for p in (speaker_dir / "sing").glob("*.txt")}
        return sorted(read_ids & sing_ids)

    def _build_index(self, drop_unpaired: bool) -> List[dict]:
        items = []
        for speaker_dir in sorted(self.data_root.iterdir()):
            if not speaker_dir.is_dir() or speaker_dir.name.startswith("."):
                continue
            speaker_id = speaker_dir.name
            split = self.split_map.get(speaker_id, "train")
            for utt_id in self._speaker_utt_ids(speaker_dir):
                read_dir = speaker_dir / "read"
                sing_dir = speaker_dir / "sing"
                read_wav, read_txt = read_dir / f"{utt_id}.wav", read_dir / f"{utt_id}.txt"
                sing_wav, sing_txt = sing_dir / f"{utt_id}.wav", sing_dir / f"{utt_id}.txt"
                has_all = all(p.exists() for p in [read_wav, read_txt, sing_wav, sing_txt])

                if not has_all:
                    if drop_unpaired:
                        continue
                    read_wav = read_wav if read_wav.exists() else None
                    read_txt = read_txt if read_txt.exists() else None
                    sing_wav = sing_wav if sing_wav.exists() else None
                    sing_txt = sing_txt if sing_txt.exists() else None

                items.append(
                    {
                        "speaker_id": speaker_id,
                        "utt_id": utt_id,
                        "split": split,
                        "read_wav": str(read_wav),
                        "sing_wav": str(sing_wav),
                        "read_txt": str(read_txt),
                        "sing_txt": str(sing_txt),
                        "read_phones": self.annotations.get(f"{speaker_id}/read/{utt_id}", []),
                        "sing_phones": self.annotations.get(f"{speaker_id}/sing/{utt_id}", []),
                    }
                )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def get_split_items(self, split: str) -> List[dict]:
        return [it for it in self.items if it["split"] == split]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.items)
        for col in ["read_phones", "sing_phones"]:
            df[col] = df[col].apply(len)
        return df.rename(columns={"read_phones": "n_read_phones", "sing_phones": "n_sing_phones"})

    def save_manifest(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)