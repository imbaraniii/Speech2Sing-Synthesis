#!/usr/bin/env python3
"""Classical WORLD speech-to-singing baseline (lower bound).

For each (read, sing) pair:
  1. WORLD-analyse the spoken audio -> f0, spectral envelope (sp), aperiodicity (ap)
  2. time-stretch sp/ap phone-by-phone to the sung phone durations
     (read/ and sing/ annotations are 1:1 phone-aligned)
  3. replace f0 with the sung Harvest F0 (cache "f0", the "melody")
  4. WORLD-synthesise

Also writes WORLD copy-synthesis of the real singing (analysis -> resynthesis),
the vocoder upper bound.

With --durations predicted, phone durations come from the Phase 2 duration
predictor (results/duration_predictions.csv, run scripts/train_duration.py
first) instead of the sung annotations, rescaled per song to the melody's
length -> outputs/world_pred/.

Usage:
    python baselines/world_baseline.py                 # val + test speakers
    python baselines/world_baseline.py --splits train val test
    python baselines/world_baseline.py --durations predicted
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message="pkg_resources")
import pyworld  # noqa: E402

from data.annotations import parse_annotation  # noqa: E402
from features.cache import FeatureCache  # noqa: E402

FRAME_MS = 5.0
FRAME_S = FRAME_MS / 1000.0
F0_RATE = 100.0  # cache "f0": Harvest, 10 ms frames, 0 Hz = unvoiced


def world_analyse(wav: np.ndarray, sr: int):
    wav = wav.astype(np.float64)
    f0, t = pyworld.harvest(wav, sr, frame_period=FRAME_MS)
    sp = pyworld.cheaptrick(wav, f0, t, sr)
    ap = pyworld.d4c(wav, f0, t, sr)
    return f0, sp, ap


def stretch_index(src_phones, tgt_durations) -> np.ndarray:
    """Fractional source-frame index for every target frame.

    Phone i covers source frames of src_phones[i] and must last tgt_durations[i]
    seconds in the output. Cumulative rounding keeps the total length exact.
    """
    idx = []
    tgt_edges = np.round(np.concatenate([[0.0], np.cumsum(tgt_durations)]) / FRAME_S).astype(int)
    for p, t0, t1 in zip(src_phones, tgt_edges[:-1], tgt_edges[1:]):
        n_tgt = t1 - t0
        if n_tgt <= 0:
            continue
        s0, s1 = p["start"] / FRAME_S, p["end"] / FRAME_S
        idx.append(s0 + (np.arange(n_tgt) + 0.5) * (s1 - s0) / n_tgt)
    return np.concatenate(idx)


def resample_frames(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linear interpolation between neighbouring frames of (T, D) x."""
    idx = np.clip(idx, 0, len(x) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (idx - i0)[:, None]
    return (1 - w) * x[i0] + w * x[i1]


def melody_f0(f0: np.ndarray, n_frames: int) -> np.ndarray:
    """Sung F0 (100 Hz) -> WORLD frame rate (nearest frame keeps 0 Hz = unvoiced)."""
    src = np.minimum(np.round(np.arange(n_frames) * FRAME_S * F0_RATE).astype(int), len(f0) - 1)
    return f0[src].astype(np.float64)


def convert(read_wav, read_phones, tgt_durations, f0_melody, sr) -> np.ndarray:
    """Speech -> singing with given per-phone target durations and melody F0."""
    wav, file_sr = sf.read(read_wav)
    assert file_sr == sr, f"{read_wav}: {file_sr} Hz != {sr} Hz"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    _, sp, ap = world_analyse(wav, sr)
    idx = stretch_index(read_phones, tgt_durations)
    # interpolate the envelope in the log domain
    sp_t = np.exp(resample_frames(np.log(sp + 1e-12), idx))
    ap_t = resample_frames(ap, idx)
    f0_t = melody_f0(f0_melody, len(idx))
    out = pyworld.synthesize(f0_t, np.ascontiguousarray(sp_t), np.ascontiguousarray(ap_t), sr, FRAME_MS)
    return out


def copy_synthesis(sing_wav, sr) -> np.ndarray:
    wav, _ = sf.read(sing_wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    f0, sp, ap = world_analyse(wav, sr)
    return pyworld.synthesize(f0, sp, ap, sr, FRAME_MS)


def normalize(wav: np.ndarray) -> np.ndarray:
    return (0.9 * wav / max(np.abs(wav).max(), 1e-9)).astype(np.float32)


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--durations", choices=["oracle", "predicted"], default="oracle")
    args = ap.parse_args(args)
    predicted = args.durations == "predicted"

    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml"))
    sr = data_cfg["sample_rate"]
    cache = FeatureCache(ROOT / data_cfg["cache_root"])
    manifest = pd.read_csv(ROOT / data_cfg["cache_root"] / "split_manifest.csv", dtype={"utt_id": str})
    out_world = ROOT / args.out / ("world_pred" if predicted else "world")
    if predicted:
        preds = pd.read_csv(ROOT / "results" / "duration_predictions.csv").sort_values(["utt", "idx"])
        pred_durs = preds.groupby("utt")["transformer_notes"].apply(list).to_dict()
    out_copy = ROOT / args.out / "world_copysyn"
    out_world.mkdir(parents=True, exist_ok=True)
    out_copy.mkdir(parents=True, exist_ok=True)

    for _, row in manifest[manifest["split"].isin(args.splits)].iterrows():
        key = f"{row.speaker_id}_{row.utt_id}"
        read_ph = parse_annotation(row.read_txt)
        sing_ph = parse_annotation(row.sing_txt)
        assert len(read_ph) == len(sing_ph), f"{key}: read/sing phones not 1:1"
        f0 = cache.load("f0", row.split, key)

        if predicted:
            # log-MSE training under-predicts long notes (~10% short overall); the melody's
            # length is known at inference, so rescale the phones to fill it
            durs = np.array(pred_durs[key])
            durs *= (len(f0) / F0_RATE) / durs.sum()
        else:
            durs = [p["duration"] for p in sing_ph]
        assert len(durs) == len(read_ph), f"{key}: {len(durs)} durations for {len(read_ph)} phones"
        out = convert(row.read_wav, read_ph, durs, f0, sr)
        sf.write(out_world / f"{key}.wav", normalize(out), sr)
        if not predicted:
            sf.write(out_copy / f"{key}.wav", normalize(copy_synthesis(row.sing_wav, sr)), sr)
        print(f"{key}: {len(out) / sr:6.1f}s (sung {sing_ph[-1]['end']:6.1f}s)")

    print(f"-> {out_world}" + ("" if predicted else f"\n-> {out_copy}"))


if __name__ == "__main__":
    main()
