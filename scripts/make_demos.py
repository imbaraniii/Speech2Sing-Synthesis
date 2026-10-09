#!/usr/bin/env python3
"""Cut aligned listening excerpts of every val/test song -> outputs/demo/<SPK_UTT>_<n>_<system>.wav.

Every singing clip covers the same lyrics window (the first ~25 s from the first sung
phone), so the systems can be compared back to back; the spoken input is cut to the
same phones. Systems whose output folder is missing are skipped.

Usage:
    python scripts/make_demos.py              # all val + test songs
    python scripts/make_demos.py --seconds 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.annotations import parse_annotation  # noqa: E402
from features.audio import load_numpy  # noqa: E402

# (clip suffix, outputs/<folder>; "gt" = the real singing)
SYSTEMS = [
    ("2_world_baseline", "world"),
    ("2b_world_predicted_durations", "world_pred"),
    ("3_world_copysyn", "world_copysyn"),
    ("4_real_singing", "gt"),
    ("5_neural_real_durations", "acoustic_oracle"),
    ("6_neural_predicted_durations", "acoustic_pred"),
    ("7_hybrid_real_durations", "hybrid_oracle"),
    ("8_hybrid_predicted_durations", "hybrid_pred"),
    ("9_vocoder_copysyn", "nsf_copysyn"),
    ("10_hybrid_gan_real_durations", "hybrid_gan_oracle"),
    ("11_hybrid_gan_predicted_durations", "hybrid_gan_pred"),
]
SILENCE = ("sil", "sp")


def window(sing_ph, read_ph, seconds):
    """(sung start, sung end, spoken start, spoken end) for the first `seconds` of singing."""
    first = next(i for i, p in enumerate(sing_ph) if p["phone"] not in SILENCE)
    a = sing_ph[first]["start"] - 0.3
    last = max(i for i, p in enumerate(sing_ph) if p["end"] <= a + seconds and p["phone"] not in SILENCE)
    return a, sing_ph[last]["end"] + 0.3, read_ph[first]["start"] - 0.3, read_ph[last]["end"] + 0.3


def cut(path, start, end, sr, out):
    x = load_numpy(path, sr)[max(0, int(start * sr)):int(end * sr)]
    sf.write(out, (0.9 * x / max(np.abs(x).max(), 1e-9)).astype(np.float32), sr, subtype="PCM_16")


def main(args=None):
    ap = argparse.ArgumentParser(description="Cut aligned demo excerpts")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default="outputs/demo")
    args = ap.parse_args(args)

    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml"))
    sr = data_cfg["sample_rate"]
    manifest = pd.read_csv(ROOT / data_cfg["cache_root"] / "split_manifest.csv", dtype={"utt_id": str})
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for r in manifest[manifest["split"].isin(args.splits)].itertuples():
        key = f"{r.speaker_id}_{r.utt_id}"
        a, b, ra, rb = window(parse_annotation(r.sing_txt), parse_annotation(r.read_txt), args.seconds)
        cut(r.read_wav, ra, rb, sr, out / f"{key}_1_spoken_input.wav")
        done = ["1_spoken_input"]
        for suffix, folder in SYSTEMS:
            src = Path(r.sing_wav) if folder == "gt" else ROOT / "outputs" / folder / f"{key}.wav"
            if src.exists():
                cut(src, a, b, sr, out / f"{key}_{suffix}.wav")
                done.append(suffix.split("_", 1)[0])
        print(f"{key}: {b - a:.1f} s sung / {rb - ra:.1f} s spoken -> clips {', '.join(done)}")


if __name__ == "__main__":
    main()
