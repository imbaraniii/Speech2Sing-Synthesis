#!/usr/bin/env python3
"""Objective evaluation of generated singing against ground-truth singing.

Metrics per utterance:
  mcd_db        MFCC-style mel-cepstral distortion (c1..c24), frame-aligned,
                non-silent reference frames only. Cepstrum = DCT of the
                log-amplitude 80-band mel spectrum; fine for comparing systems
                here, not directly comparable to SPTK-mcep MCD in papers
  f0_rmse_hz    F0 RMSE on frames voiced in both (WORLD Harvest, 10 ms)
  f0_rmse_cent  same, in cents (gender-independent)
  f0_corr       Pearson correlation of F0 on those frames
  spk_cos       ECAPA cosine(generated, target speaker's read speech)

Systems are folders outputs/<system>/<SPK_UTT>.wav; "gt" is the real singing.

Usage:
    python scripts/evaluate.py                             # world, world_copysyn, gt
    python scripts/evaluate.py --systems world gt --splits test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import scipy.fft
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.cache import FeatureCache  # noqa: E402
from features.ecapa import extract_ecapa, load_ecapa  # noqa: E402
from features.world_f0 import extract_f0_harvest  # noqa: E402

MCD_K = 10.0 / np.log(10.0) * np.sqrt(2.0)


def mel_cepstrum(wav: np.ndarray, sr: int, hop: int = 256) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=hop, n_mels=80, power=1.0)
    log_mel = np.log(mel + 1e-8)
    # real-cepstrum scaling (unnormalised DCT-II / 2M), not librosa's orthonormal MFCC
    ceps = scipy.fft.dct(log_mel, type=2, axis=0, norm=None) / (2 * log_mel.shape[0])
    return ceps[1:25].T  # (T, 24), drop energy c0


def mcd(gen: np.ndarray, ref: np.ndarray, sr: int, hop: int = 256, silence_db: float = -40.0) -> float:
    # ponytail: frame-aligned (outputs share the melody's timeline); add DTW if timelines diverge
    cg, cr = mel_cepstrum(gen, sr, hop), mel_cepstrum(ref, sr, hop)
    rms = librosa.feature.rms(y=ref, frame_length=1024, hop_length=hop)[0]
    speech = librosa.amplitude_to_db(rms, ref=np.max) > silence_db
    n = min(len(cg), len(cr), len(speech))
    diff = (cg[:n] - cr[:n])[speech[:n]]
    return float(np.mean(MCD_K * np.sqrt(np.sum(diff**2, axis=1))))


def f0_metrics(f0_g, f0_r) -> dict:
    n = min(len(f0_g), len(f0_r))
    both = (f0_g[:n] > 0) & (f0_r[:n] > 0)
    g, r = f0_g[:n][both], f0_r[:n][both]
    if len(g) < 2:
        return {"f0_rmse_hz": np.nan, "f0_rmse_cent": np.nan, "f0_corr": np.nan, "voiced_frac": 0.0}
    return {
        "f0_rmse_hz": float(np.sqrt(np.mean((g - r) ** 2))),
        "f0_rmse_cent": float(np.sqrt(np.mean((1200 * np.log2(g / r)) ** 2))),
        "f0_corr": float(np.corrcoef(g, r)[0, 1]),
        "voiced_frac": float(both.mean()),
    }


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_file(path, ref, f0_r, spk_ref, ecapa, fcfg, sr) -> dict:
    """All metrics for one generated wav vs the real singing (ref waveform, its F0, read-speech ECAPA)."""
    gen, _ = librosa.load(path, sr=sr)
    f0_g = extract_f0_harvest(path, fcfg["frame_period_ms"], fcfg["fmin"], fcfg["fmax"])
    return {"mcd_db": mcd(gen, ref, sr), **f0_metrics(f0_g, f0_r),
            "spk_cos": cosine(extract_ecapa(path, ecapa), spk_ref)}


def main(args=None):
    ap = argparse.ArgumentParser(description="Objective evaluation")
    ap.add_argument("--systems", nargs="+", default=["world", "world_copysyn", "gt"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--csv", default="results/eval.csv")
    args = ap.parse_args(args)

    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml"))
    extract_cfg = yaml.safe_load(open(ROOT / "config" / "extract.yaml"))
    sr = data_cfg["sample_rate"]
    cache = FeatureCache(ROOT / data_cfg["cache_root"])
    manifest = pd.read_csv(ROOT / data_cfg["cache_root"] / "split_manifest.csv", dtype={"utt_id": str})
    ecapa = load_ecapa(extract_cfg["ecapa"]["model"], str(ROOT / extract_cfg["ecapa"]["savedir"]), "cpu")
    fcfg = extract_cfg["f0"]

    rows = []
    for _, row in manifest[manifest["split"].isin(args.splits)].iterrows():
        key = f"{row.speaker_id}_{row.utt_id}"
        ref, _ = librosa.load(row.sing_wav, sr=sr)
        f0_r = cache.load("f0", row.split, key)
        spk_ref = cache.load("ecapa", row.split, key)  # read-speech embedding

        for system in args.systems:
            path = Path(row.sing_wav) if system == "gt" else ROOT / args.outputs / system / f"{key}.wav"
            if not path.exists():
                print(f"  missing {path}, skipped")
                continue
            r = {"system": system, "split": row.split, "utt": key,
                 **score_file(path, ref, f0_r, spk_ref, ecapa, fcfg, sr)}
            rows.append(r)
            print(f"{system:14s} {key}: MCD {r['mcd_db']:5.2f}  F0 RMSE {r['f0_rmse_hz']:6.1f} Hz  "
                  f"CORR {r['f0_corr']:.3f}  spk {r['spk_cos']:.3f}")

    df = pd.DataFrame(rows)
    out = ROOT / args.csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    cols = ["mcd_db", "f0_rmse_hz", "f0_rmse_cent", "f0_corr", "spk_cos"]
    summary = df.groupby(["system", "split"])[cols].mean().round(3)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"))
    print("\n" + summary.to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
