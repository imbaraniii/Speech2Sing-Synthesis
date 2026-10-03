#!/usr/bin/env python3
"""Sanity-check the cached features.

Checks:
  - every feature type has the expected number of cached files
  - feature shapes are sane
  - ECAPA similarity: same-speaker > cross-speaker (train set)
  - F0 overlay plot on a mel spectrogram (saved to figures/)
  - Griffin-Lim inversion of a mel spectrogram (audible check, saved wav)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features.cache import FeatureCache  # noqa: E402

with open(ROOT / "config" / "data.yaml") as f:
    data_cfg = yaml.safe_load(f)
with open(ROOT / "config" / "extract.yaml") as f:
    extract_cfg = yaml.safe_load(f)

cache = FeatureCache(data_cfg["cache_root"])
figures = Path(data_cfg["figures_dir"])
figures.mkdir(parents=True, exist_ok=True)

FEATURE_TYPES = ["hubert", "ecapa", "crepe", "crepe_conf", "f0", "mel", "notes"]


def main():
    summary = cache.summary()
    print("cache summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ---- shape checks ----
    print("\n--- shape checks ---")
    for ft in FEATURE_TYPES:
        for split in ["train", "val", "test"]:
            keys = cache.list_keys(ft, split)
            if not keys:
                continue
            arr = cache.load(ft, split, keys[0])
            shape = getattr(arr, "shape", f"list[{len(arr)}]")
            print(f"  {ft}/{split}: n={len(keys):2d}  shape[0]={shape}")

    # ---- ECAPA similarity matrix ----
    print("\n--- ECAPA same-vs-cross speaker similarity ---")
    import seaborn as sns

    train_keys = cache.list_keys("ecapa", "train")
    if train_keys:
        spk_sample = {}
        for k in train_keys:
            spk = k.split("_")[0]
            spk_sample.setdefault(spk, []).append(k)
        # keep speakers with at least 2 utterances for a real same-vs-cross test
        spks = sorted(s for s, ks in spk_sample.items() if len(ks) >= 2)[:6]

        if len(spks) >= 2:
            same = []
            cross = []
            for i, s in enumerate(spks):
                a, b = spk_sample[s][:2]
                va = cache.load("ecapa", "train", a)
                vb = cache.load("ecapa", "train", b)
                same.append(float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))))
                for j, t in enumerate(spks):
                    if i == j:
                        continue
                    vc = cache.load("ecapa", "train", spk_sample[t][0])
                    cross.append(float(np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc))))

            print(f"  mean same-speaker: {np.mean(same):.3f}")
            print(f"  mean cross-speaker: {np.mean(cross):.3f}")
            print(f"  separation: {np.mean(same) - np.mean(cross):.3f}")

            sim = np.zeros((len(spks), len(spks)))
            for i, s in enumerate(spks):
                for j, t in enumerate(spks):
                    va = cache.load("ecapa", "train", spk_sample[s][0])
                    vb = cache.load("ecapa", "train", spk_sample[t][0])
                    sim[i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

            plt.figure(figsize=(7, 6))
            sns.heatmap(sim, xticklabels=spks, yticklabels=spks, annot=True,
                        cmap="RdBu_r", center=0, vmin=-1, vmax=1)
            plt.title("ECAPA cosine similarity (train speakers)")
            plt.tight_layout()
            plt.savefig(figures / "ecapa_similarity.png", dpi=120)
            plt.close()
            print(f"  saved {figures / 'ecapa_similarity.png'}")

    # ---- F0 overlay on mel ----
    print("\n--- F0 overlay plot ---")
    mel_keys = cache.list_keys("mel", "train")
    f0_keys = cache.list_keys("f0", "train")
    if mel_keys and f0_keys:
        key = mel_keys[0]
        mel = cache.load("mel", "train", key)
        f0 = cache.load("f0", "train", key)  # Harvest melody F0, 0 Hz = unvoiced
        f0 = np.where(f0 > 0, f0, np.nan)
        sr = data_cfg["sample_rate"]
        mel_hop = extract_cfg["mel"]["hop_length"]
        f0_frame_s = extract_cfg["f0"]["frame_period_ms"] / 1000.0

        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        t_mel = np.arange(mel.shape[0]) * mel_hop / sr
        im = ax[0].imshow(mel.T, aspect="auto", origin="lower",
                          extent=[0, t_mel[-1], extract_cfg["mel"]["fmin"], extract_cfg["mel"]["fmax"]],
                          cmap="magma")
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title(f"mel spectrogram: {key}")
        ax[0].set_ylabel("Hz")

        t_f0 = np.arange(f0.shape[0]) * f0_frame_s
        ax[1].plot(t_f0, f0, lw=0.6, color="c")
        ax[1].set_xlim(0, t_mel[-1])
        ax[1].set_title("Melody F0 (WORLD Harvest)")
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("Hz")
        plt.tight_layout()
        fig_save = figures / "f0_overlay.png"
        plt.savefig(fig_save, dpi=120)
        plt.close()
        print(f"  saved {fig_save}")

    # ---- Griffin-Lim inversion ----
    print("\n--- Griffin-Lim mel inversion (audible check) ---")
    if mel_keys:
        from features.mel_invert import invert_mel

        key = mel_keys[0]
        mel = cache.load("mel", "train", key)
        wav = invert_mel(mel, sr=sr, hop_length=mel_hop, n_fft=1024, win_length=1024)
        out_path = figures / f"griffinlim_{key}.wav"
        import soundfile as sf

        sf.write(out_path, wav, sr)
        print(f"  saved {out_path} (len={len(wav)/sr:.1f}s, sr={sr})")

    print("\nDone.")


if __name__ == "__main__":
    main()