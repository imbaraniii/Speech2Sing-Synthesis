"""Parse NUS SMC Corpus phone-level annotation (.txt) files.

Format of each line:  start_time  end_time  phone
e.g.  0.000000 7.086957 sil
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

PHONE_PATTERN = re.compile(r"^\s*([0-9.]+)\s+([0-9.]+)\s+(\S+)\s*$")


def parse_annotation(txt_path) -> List[dict]:
    """Parse a single .txt annotation file into a list of phone dicts."""
    txt_path = Path(txt_path)
    phones: List[dict] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = PHONE_PATTERN.match(line)
            if not m:
                continue
            start, end, phone = float(m.group(1)), float(m.group(2)), m.group(3)
            phones.append(
                {
                    "phone": phone,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
            )
    return phones


def load_all_annotations(data_root) -> Dict[str, List[dict]]:
    """Load every annotation file in the corpus.

    Returns {key: phone_list} where key = f"{speaker}/{mode}/{utt_id}".
    """
    data_root = Path(data_root)
    annotations: Dict[str, List[dict]] = {}
    for speaker_dir in sorted(data_root.iterdir()):
        if not speaker_dir.is_dir() or speaker_dir.name.startswith("."):
            continue
        speaker_id = speaker_dir.name
        for mode in ["read", "sing"]:
            mode_dir = speaker_dir / mode
            if not mode_dir.is_dir():
                continue
            for txt_file in sorted(mode_dir.glob("*.txt")):
                key = f"{speaker_id}/{mode}/{txt_file.stem}"
                annotations[key] = parse_annotation(txt_file)
    return annotations


def validate_alignments(phones: List[dict], tol: float = 1e-3) -> List[str]:
    """Sanity-check a phone list: continuity, order, non-negative durations."""
    issues: List[str] = []
    prev_end = 0.0
    for i, p in enumerate(phones):
        if p["duration"] < 0:
            issues.append(f"[{i}] negative duration {p}")
        if i == 0 and p["start"] > tol:
            issues.append(f"[{i}] first phone starts at {p['start']} (not 0)")
        elif i > 0 and p["start"] < prev_end - tol:
            issues.append(f"[{i}] overlap: starts {p['start']} < prev end {prev_end}")
        prev_end = p["end"]
    return issues