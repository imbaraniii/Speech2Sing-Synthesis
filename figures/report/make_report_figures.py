"""Figures for PROJECT_REPORT.md (kept out of git). Run from the project root."""
import sys
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from data.annotations import parse_annotation  # noqa: E402

OUT = Path("figures/report")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
WORLD, PURE, HYBRID = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white"})
VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}


def logmel(path):
    y, sr = librosa.load(path, sr=44100)
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=40, fmax=16000)
    return librosa.power_to_db(m, ref=np.max), len(y) / sr


def excerpt_window(spk, u):  # same window as outputs/demo clips
    sp = parse_annotation(f"nus-smc-corpus_48/{spk}/sing/{u}.txt")
    first = next(i for i, p in enumerate(sp) if p["phone"] not in ("sil", "sp"))
    a = sp[first]["start"] - 0.3
    last = max(i for i, p in enumerate(sp) if p["end"] <= a + 25 and p["phone"] not in ("sil", "sp"))
    return a, sp[last]["end"] + 0.3


# ---------------------------------------------------------------- 1. pipeline flow
def pipeline_flow(spk="VKOW", u="05"):
    a, b = excerpt_window(spk, u)
    rows = [("1  INPUT — spoken lyrics (target speaker)", f"outputs/demo/{spk}_{u}_1_spoken_input.wav"),
            ("2  INPUT — reference melody (F0 contour)", None),
            ("3  WORLD baseline output", f"outputs/demo/{spk}_{u}_2_world_baseline.wav"),
            ("4  Pure Conformer output", f"outputs/demo/{spk}_{u}_5_neural_real_durations.wav"),
            ("5  Hybrid Conformer output (ours)", f"outputs/demo/{spk}_{u}_7_hybrid_real_durations.wav"),
            ("6  Ground truth — real singing", f"outputs/demo/{spk}_{u}_4_real_singing.wav")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 13))
    for ax, (title, path) in zip(axes, rows):
        if path is None:
            f0 = np.load(f"features/cache/f0/test/{spk}_{u}.npy")
            t = np.arange(len(f0)) * 0.01
            sel = (t >= a) & (t <= b)
            v = np.where(f0[sel] > 0, f0[sel], np.nan)
            ax.plot(t[sel] - a, v, color=INK, lw=1.2)
            ax.set_ylabel("F0 (Hz)")
            ax.set_xlim(0, b - a)
            ax.grid(color=GRID, lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        else:
            m, dur = logmel(path)
            ax.imshow(m, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0, extent=[0, dur, 0, 128])
            ax.set_ylabel("mel bin")
            if "spoken" in path:
                ax.set_xlim(0, b - a)  # same seconds scale as the singing rows
                ax.text(dur + 0.3, 64, f"speech is only {dur:.1f} s\nfor the same lyrics", va="center", color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Input → output flow on the unseen test speaker ({spk}_{u}, first 25 s)", fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(OUT / "pipeline_flow.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------- 2. time warping
def time_warping(spk="VKOW", u="05", n=14):
    rp = parse_annotation(f"nus-smc-corpus_48/{spk}/read/{u}.txt")
    sp = parse_annotation(f"nus-smc-corpus_48/{spk}/sing/{u}.txt")
    idx = [i for i, p in enumerate(sp) if p["phone"] not in ("sil", "sp")][:n]
    i0, i1 = idx[0], idx[-1]
    r0, s0 = rp[i0]["start"], sp[i0]["start"]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for i in range(i0, i1 + 1):
        for p, base, y in [(rp[i], r0, 1.0), (sp[i], s0, 0.0)]:
            if p["duration"] <= 0:
                continue
            ph = p["phone"]
            col = PURE if ph in VOWELS else (GRID if ph in ("sil", "sp") else WORLD)
            ax.barh(y, p["duration"], left=p["start"] - base, height=0.32, color=col, edgecolor="white", lw=1.5)
            if p["duration"] > 0.06:
                ax.text(p["start"] - base + p["duration"] / 2, y, ph, ha="center", va="center", fontsize=8,
                        color="white" if ph not in ("sil", "sp") else INK2)
        if rp[i]["phone"] not in ("sil",):
            ax.plot([rp[i]["start"] - r0, sp[i]["start"] - s0], [0.84, 0.16], color=INK2, lw=0.6, alpha=0.6)
    ax.set_yticks([0, 1], ["sung", "spoken"])
    ax.set_xlabel("time from first phone (s)")
    ax.set_title(f"Phone-by-phone time warping, {spk}_{u}: the same {n} phones, spoken vs sung "
                 "(orange = vowels, blue = consonants)", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "time_warping.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------- 3. training curves
def training_curves():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, f, col in [("pure Conformer", "results/acoustic_train.csv", PURE),
                         ("hybrid Conformer", "results/hybrid_train.csv", HYBRID)]:
        d = pd.read_csv(f)
        ax.plot(d.step, d.val_l1, color=col, lw=2, label=f"{name} — validation (JLEE)")
        ax.plot(d.step, d.train_l1, color=col, lw=2, ls=(0, (3, 2)), alpha=0.7, label=f"{name} — training")
        b = d.loc[d.val_l1.idxmin()]
        ax.plot(b.step, b.val_l1, "o", ms=9, color=col, mec=SURF, mew=2)
        text_xy = (1200, 0.97) if "hybrid" in name else (11000, 0.97)
        ax.annotate(f"{name.split()[0]} best: step {int(b.step)}, {b.val_l1:.3f}", (b.step, b.val_l1),
                    xytext=text_xy, color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.axhline(1.536, color=WORLD, lw=1.5, ls=(0, (1, 2)))
    ax.text(40000, 1.55, "WORLD conversion alone (hybrid's starting point): 1.536", ha="right", color=INK2, fontsize=9)
    ax.set_xlabel("training step")
    ax.set_ylabel("L1 on log-mel (lower is better)")
    ax.set_title("Training curves: both models overfit early; checkpoints chosen by validation loss", loc="left")
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.93))
    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------- 4. results comparison
def results_comparison():
    r = pd.concat([pd.read_csv("results/eval_summary.csv"), pd.read_csv("results/eval_hybrid_summary.csv")])
    r = r[r.split == "test"].set_index("system")
    systems = [("WORLD", WORLD, "world", "world_pred"), ("Pure Conformer", PURE, "acoustic_oracle", "acoustic_pred"),
               ("Hybrid Conformer", HYBRID, "hybrid_oracle", "hybrid_pred")]
    metrics = [("mcd_db", "MCD (dB) — lower is better", "{:.2f}"),
               ("f0_rmse_hz", "F0 RMSE (Hz) — lower is better", "{:.1f}"),
               ("spk_cos", "Speaker similarity — higher is better", "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    w = 0.26
    for ax, (col, title, fmt) in zip(axes, metrics):
        for k, (label, color, oracle, pred) in enumerate(systems):
            vals = [r.loc[oracle, col], r.loc[pred, col]]
            xs = np.arange(2) + (k - 1) * (w + 0.02)
            ax.bar(xs, vals, width=w, color=color, label=label, edgecolor="white", lw=2)
            for x, v in zip(xs, vals):
                ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
        if col == "spk_cos":
            ax.axhline(r.loc["gt", col], color=INK2, lw=1.2, ls=(0, (4, 2)))
            ax.text(1.45, r.loc["gt", col] + 0.01, f"real singing {r.loc['gt', col]:.2f}", ha="right",
                    color=INK2, fontsize=8.5)
        ax.set_xticks([0, 1], ["real durations", "predicted durations\n(fully automatic)"])
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Test speaker VKOW (unseen): WORLD vs pure Conformer vs hybrid", fontsize=12.5)
    plt.tight_layout()
    plt.savefig(OUT / "results_comparison.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    pipeline_flow()
    time_warping()
    training_curves()
    results_comparison()
    print("saved:", sorted(p.name for p in OUT.glob("*.png")))
