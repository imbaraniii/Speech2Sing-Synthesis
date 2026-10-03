#!/usr/bin/env python3
"""Verify split manifest: no speaker leakage, per-split stats, coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

with open(ROOT / "config" / "data.yaml") as f:
    data_cfg = yaml.safe_load(f)

manifest = Path(data_cfg["cache_root"]) / "split_manifest.csv"
if not manifest.exists():
    print(f"ERROR: manifest not found at {manifest}. Run scripts/preprocess.py first.")
    sys.exit(1)

df = pd.read_csv(manifest)
print(f"total items: {len(df)}")
print("\nper-split counts:")
print(df["split"].value_counts().sort_index().to_string())

print("\nspeakers per split:")
for split in sorted(df["split"].unique()):
    spks = df[df["split"] == split]["speaker_id"].unique()
    print(f"  {split}: {len(spks)} -> {sorted(spks)}")

print("\nleakage check (speaker appearing in >1 split):")
leakage = df.groupby("speaker_id")["split"].nunique()
bad = leakage[leakage > 1]
if bad.empty:
    print("  OK - no speaker leakage")
else:
    print("  FAIL - leakage detected:")
    print(bad.to_string())

print("\nutterances per speaker:")
print(
    df.groupby(["speaker_id", "split"]).size().unstack(fill_value=0).to_string()
)

# Coverage: every speaker has both read and sing? (manifest records file paths)
missing = 0
for _, row in df.iterrows():
    for col in ["read_wav", "sing_wav", "read_txt", "sing_txt"]:
        if not Path(row[col]).exists():
            missing += 1
            print(f"  MISSING FILE: {row[col]}")
print(f"\nmissing files: {missing}" if missing else "\nall referenced files exist: OK")