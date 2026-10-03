#!/usr/bin/env python3
"""Data study: compute summary statistics and save plots for the review.

Produces:
  figures/duration_dist.png         speech vs singing durations
  figures/duration_ratio_phones.png singing/speech duration ratio by phone
  figures/f0_range.png              speech vs singing F0 distributions
  figures/phone_inventory.png       most frequent phones
  data_study_stats.csv              numeric summary table
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.nus_smc import NUS_SMC_Dataset  # noqa: E402
from data.splits import list_speakers, speaker_disjoint_split  # noqa: E402
from features.cache import FeatureCache  # noqa: E402

with open(ROOT / "config" / "data.yaml") as f:
    data_cfg = yaml.safe_load(f)
with open(ROOT / "config" / "extract.yaml") as f:
    extract_cfg = yaml.safe_load(f)

data_root = Path(data_cfg["data_root"])
figures = Path(data_cfg["figures_dir"])
figures.mkdir(parents=True, exist_ok=True)
cache = FeatureCache(data_cfg["cache_root"])

# fixed splits only — leave-one-song-out would need explicit folds
split_map = speaker_disjoint_split(
    list_speakers(data_root),
    seed=data_cfg["splits"]["seed"],
    n_train=data_cfg["splits"]["train_speakers"],
    n_val=data_cfg["splits"]["val_speakers"],
)
dataset = NUS_SMC_Dataset(data_root, split_map, utterance_ids=None)


def phone_stat_table():
    """Read- vs sing- phone durations per phone across the corpus."""
    rows = []
    for it in dataset:
        for p in it["read_phones"]:
            rows.append({"speaker": it["speaker_id"], "utt": it["utt_id"],
                         "mode": "speech", "phone": p["phone"], "dur": p["duration"]})
        for p in it["sing_phones"]:
            rows.append({"speaker": it["speaker_id"], "utt": it["utt_id"],
                         "mode": "singing", "phone": p["phone"], "dur": p["duration"]})
    df = pd.DataFrame(rows)
    return df


def main():
    print("Building phone table from annotations...")
    ptab = phone_stat_table()
    ptab.to_csv(ROOT / "data_study_stats.csv", index=False)
    print(f"saved data_study_stats.csv ({len(ptab)} phones)")

    # ---- duration distribution ----
    plt.figure(figsize=(8, 4))
    for mode, color in [("speech", "tab:blue"), ("singing", "tab:orange")]:
        d = ptab[ptab["mode"] == mode]["dur"]
        plt.hist(d[d < 1.0], bins=80, alpha=0.5, label=f"{mode} (median {d.median():.3f}s)",
                 color=color, density=True)
    plt.xlabel("phone duration (s, phones under 1 s shown)")
    plt.ylabel("density")
    plt.title("Phone duration distribution: speech vs singing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "duration_dist.png", dpi=120)
    plt.close()
    print("saved figures/duration_dist.png")

    # ---- duration ratio by phone (sing/speech) ----
    pivot = ptab.pivot_table(index=["speaker", "utt", "phone"], columns="mode",
                             values="dur", aggfunc="mean")
    ratio = pivot.assign(ratio=pivot["singing"] / pivot["speech"].clip(lower=1e-6)).dropna(subset=["ratio"])
    med = ratio.groupby("phone")["ratio"].median().sort_values(ascending=False)
    top = med.head(15)
    plt.figure(figsize=(10, 4))
    plt.barh(range(len(top)), top.values, color="tab:orange")
    plt.yticks(range(len(top)), top.index)
    plt.xlabel("singing/speech duration ratio")
    plt.title("Median duration ratio by phone (top 15)")
    plt.tight_layout()
    plt.savefig(figures / "duration_ratio_phones.png", dpi=120)
    plt.close()
    print("saved figures/duration_ratio_phones.png")

    # ---- F0 range: speech vs singing, one song per speaker (Harvest, the melody tracker) ----
    print("\nComputing speech F0 with Harvest (one song per speaker)...")
    from features.world_f0 import extract_f0_harvest

    fcfg = extract_cfg["f0"]
    f0_rows = []
    for spk in sorted({it["speaker_id"] for it in dataset}):
        it = next(i for i in dataset if i["speaker_id"] == spk)
        key = f"{spk}_{it['utt_id']}"
        f0_read = extract_f0_harvest(it["read_wav"], fcfg["frame_period_ms"], fcfg["fmin"], fcfg["fmax"])
        f0_sing = cache.load("f0", it["split"], key)
        for mode, f0 in [("speech", f0_read), ("singing", f0_sing)]:
            v = f0[f0 > 0]  # Harvest: 0 Hz = unvoiced
            f0_rows.append({"speaker": spk, "mode": mode, "median": float(np.median(v)),
                            "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))})
    fdf = pd.DataFrame(f0_rows)
    order = fdf[fdf["mode"] == "speech"].sort_values("median")["speaker"].tolist()
    y = {spk: i for i, spk in enumerate(order)}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mode, color, dy in [("speech", "#2a78d6", -0.15), ("singing", "#eb6834", 0.15)]:
        d = fdf[fdf["mode"] == mode]
        ys = [y[s] + dy for s in d["speaker"]]
        ax.hlines(ys, d["p5"], d["p95"], color=color, lw=2, alpha=0.6)
        ax.plot(d["median"], ys, "o", ms=8, color=color, mec="#fcfcfb", mew=2, label=f"{mode}: median, bar = 5–95%")
    ax.set_yticks(range(len(order)), order)
    ax.set_xscale("log")
    ax.set_xticks([100, 150, 200, 300, 400, 600], ["100", "150", "200", "300", "400", "600"])
    ax.set_xlabel("voiced F0 (Hz, log scale)")
    ax.set_title("Speech vs singing pitch per speaker (one song each)")
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(figures / "f0_range.png", dpi=120)
    plt.close()
    print("saved figures/f0_range.png")

    # ---- phone inventory ----
    top_phones = ptab["phone"].value_counts().head(20)
    plt.figure(figsize=(10, 5))
    top_phones.plot(kind="bar", color="tab:green")
    plt.title("Top 20 phone types in corpus")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figures / "phone_inventory.png", dpi=120)
    plt.close()
    print("saved figures/phone_inventory.png")

    # ---- speaker balance ----
    spk_df = dataset.to_dataframe()
    bal = spk_df.groupby("speaker_id").agg(
        n_utts=("utt_id", "count"), split=("split", "first")
    ).sort_index()
    print("\nspeaker table:")
    print(bal.to_string())
    bal.to_csv(ROOT / "speaker_table.csv")

    print("\nStudy complete. Plots in figures/, tables in *.csv")


if __name__ == "__main__":
    main()