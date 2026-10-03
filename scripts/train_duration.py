#!/usr/bin/env python3
"""Phase 2: phone duration predictor (speech-time -> singing-time).

read/ and sing/ annotations are 1:1 phone-aligned, so each phone gives one
training row:  inputs from speech + melody  ->  sung duration (log seconds).

Per-phone inputs:
  phone id, log spoken duration, HuBERT mean-pooled over the spoken phone,
  note features (MIDI, log note duration, has_note) of the note that overlaps
  the sung phone most. NOTE: this phone->note assignment uses the sung timing,
  i.e. it is equivalent to a score with lyric->note assignment (as in SVS
  systems like DiffSinger). Arbitrary-melody inference needs an automatic
  phone->note aligner.

Compares, as duration MAE (ms) on val + test speakers:
  global_ratio      spoken dur x train sung/spoken ratio
  phone_ratio       spoken dur x per-phone median ratio (train)
  transformer_nonotes   2-layer Transformer, note features zeroed (ablation)
  transformer_notes     2-layer Transformer, full inputs

Usage:
    python scripts/train_duration.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.annotations import parse_annotation  # noqa: E402
from features.cache import FeatureCache  # noqa: E402

HUBERT_RATE = 50.0
SILENCES = {"sil", "sp"}
VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}
WINDOW = 128


# --------------------------------------------------------------------------- data

def norm_phone(p: str) -> str:
    return "sil" if p in SILENCES else p


def pool_hubert(hub: np.ndarray, start: float, end: float) -> np.ndarray:
    a = min(int(start * HUBERT_RATE), len(hub) - 1)
    b = min(int(math.ceil(end * HUBERT_RATE)), len(hub))
    return hub[a:b].mean(axis=0) if b > a else hub[a]


def assign_notes(sing_ph, notes) -> np.ndarray:
    """(N, 3): midi, log note duration, has_note for the max-overlap note."""
    on = np.array([n["onset_time"] for n in notes])
    off = np.array([n["offset_time"] for n in notes])
    out = np.zeros((len(sing_ph), 3), dtype=np.float32)
    for i, p in enumerate(sing_ph):
        ov = np.clip(np.minimum(p["end"], off) - np.maximum(p["start"], on), 0, None)
        j = int(np.argmax(ov)) if len(ov) else -1
        if j >= 0 and ov[j] > 0:
            out[i] = [notes[j]["midi"], np.log(notes[j]["duration_time"]), 1.0]
    return out


def load_utterances(manifest, cache):
    utts = []
    for _, row in manifest.iterrows():
        key = f"{row.speaker_id}_{row.utt_id}"
        read_ph, sing_ph = parse_annotation(row.read_txt), parse_annotation(row.sing_txt)
        assert len(read_ph) == len(sing_ph), f"{key}: read/sing phones not 1:1"
        hub = cache.load("hubert", row.split, key)
        notes = list(cache.load("notes", row.split, key))
        utts.append({
            "key": key,
            "split": row.split,
            "phones": [norm_phone(p["phone"]) for p in sing_ph],
            "spoken": np.array([max(p["duration"], 1e-3) for p in read_ph], dtype=np.float32),
            "sung": np.array([max(p["duration"], 1e-3) for p in sing_ph], dtype=np.float32),
            "hubert": np.stack([pool_hubert(hub, p["start"], p["end"]) for p in read_ph]),
            "notes": assign_notes(sing_ph, notes),
        })
    return utts


def to_tensors(u, vocab):
    ids = torch.tensor([vocab.get(p, 1) for p in u["phones"]])
    n = u["notes"]
    has = n[:, 2]
    scal = np.stack([np.log(u["spoken"]), (n[:, 0] - 60.0) / 12.0 * has, n[:, 1], has], axis=1)
    return ids, torch.from_numpy(u["hubert"]), torch.from_numpy(scal.astype(np.float32)), \
        torch.from_numpy(np.log(u["sung"]))


# -------------------------------------------------------------------------- model

class DurationPredictor(nn.Module):
    def __init__(self, n_phones, d=128, hubert_dim=768, n_scalar=4, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_phones, 64, padding_idx=0)
        self.hub = nn.Sequential(nn.LayerNorm(hubert_dim), nn.Linear(hubert_dim, d))
        self.inp = nn.Linear(64 + d + n_scalar, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 1)
        pos = torch.arange(WINDOW * 16)[:, None] / (10000 ** (torch.arange(0, d, 2) / d))
        self.register_buffer("pe", torch.cat([pos.sin(), pos.cos()], dim=1))

    def forward(self, ids, hub, scal, pad_mask=None):
        x = self.inp(torch.cat([self.emb(ids), self.hub(hub), scal], dim=-1))
        x = x + self.pe[: x.shape[1]]
        return self.head(self.enc(x, src_key_padding_mask=pad_mask)).squeeze(-1)


def predict(model, u, vocab, use_notes):
    """Sung durations (s) for a whole utterance, in non-overlapping windows."""
    ids, hub, scal, _ = to_tensors(u, vocab)
    if not use_notes:
        scal[:, 1:] = 0
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(ids), WINDOW):
            sl = slice(s, s + WINDOW)
            out.append(model(ids[None, sl], hub[None, sl], scal[None, sl])[0])
    return torch.cat(out).exp().numpy()


def train(train_utts, val_utts, vocab, use_notes, steps, batch, lr, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [to_tensors(u, vocab) for u in train_utts]
    lengths = np.array([len(d[0]) for d in data], dtype=float)
    model = DurationPredictor(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state = float("inf"), None

    for step in range(1, steps + 1):
        model.train()
        # sample windows proportional to utterance length
        picks = rng.choice(len(data), size=batch, p=lengths / lengths.sum())
        cols = [[], [], [], []]
        for k in picks:
            L = len(data[k][0])
            s = rng.integers(0, max(1, L - WINDOW + 1))
            for c, t in zip(cols, data[k]):
                c.append(t[s:s + WINDOW])
        ids = nn.utils.rnn.pad_sequence(cols[0], batch_first=True)
        hub, scal, y = (nn.utils.rnn.pad_sequence(c, batch_first=True) for c in cols[1:])
        if not use_notes:
            scal[..., 1:] = 0
        mask = ids == 0
        pred = model(ids, hub, scal, mask)
        loss = ((pred - y) ** 2)[~mask].mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0 or step == steps:
            val = mae_ms(val_utts, lambda u: predict(model, u, vocab, use_notes))["mae_ms_nonsil"]
            if val < best:
                best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  [{'notes' if use_notes else 'nonotes'}] step {step:4d}  "
                  f"train log-MSE {loss.item():.4f}  val MAE(non-sil) {val:6.1f} ms  best {best:6.1f}")

    model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------------------ metrics

def mae_ms(utts, predict_fn) -> dict:
    err, nonsil = [], []
    for u in utts:
        e = np.abs(predict_fn(u) - u["sung"]) * 1000
        err.append(e)
        nonsil.append(e[[p != "sil" for p in u["phones"]]])
    return {"mae_ms_all": float(np.concatenate(err).mean()),
            "mae_ms_nonsil": float(np.concatenate(nonsil).mean())}


def main(args=None):
    ap = argparse.ArgumentParser(description="Train/evaluate the duration predictor")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(args)

    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml"))
    cache = FeatureCache(ROOT / data_cfg["cache_root"])
    manifest = pd.read_csv(ROOT / data_cfg["cache_root"] / "split_manifest.csv", dtype={"utt_id": str})
    utts = load_utterances(manifest, cache)
    split = {s: [u for u in utts if u["split"] == s] for s in ["train", "val", "test"]}
    print({s: f"{len(v)} utts / {sum(len(u['phones']) for u in v)} phones" for s, v in split.items()})

    # self-check: phone->note assignment must cover (almost) all sung vowels
    vowel_has = np.concatenate([u["notes"][[p in VOWELS for p in u["phones"]], 2] for u in utts])
    print(f"sung vowels with an assigned note: {vowel_has.mean():.1%}")
    assert vowel_has.mean() >= 0.9, "phone->note assignment coverage too low"

    vocab = {"<pad>": 0, "<unk>": 1}
    for p in sorted({p for u in split["train"] for p in u["phones"]}):
        vocab[p] = len(vocab)

    # ---- ratio baselines (fit on train)
    tr_phones = np.concatenate([u["phones"] for u in split["train"]])
    tr_ratio = np.concatenate([u["sung"] / u["spoken"] for u in split["train"]])
    tr_nonsil = tr_phones != "sil"
    g = float(np.concatenate([u["sung"] for u in split["train"]])[tr_nonsil].sum()
              / np.concatenate([u["spoken"] for u in split["train"]])[tr_nonsil].sum())
    per_phone = pd.Series(tr_ratio).groupby(tr_phones).median().to_dict()
    systems = {
        "global_ratio": lambda u: u["spoken"] * g,
        "phone_ratio": lambda u: u["spoken"] * np.array([per_phone.get(p, g) for p in u["phones"]]),
    }
    print(f"global sung/spoken ratio (train, non-sil): {g:.2f}")

    # ---- transformers
    models = {}
    for use_notes in (False, True):
        name = "transformer_notes" if use_notes else "transformer_nonotes"
        print(f"\ntraining {name}...")
        models[name] = train(split["train"], split["val"], vocab, use_notes,
                             args.steps, args.batch, args.lr, args.seed)
        systems[name] = (lambda m, n: lambda u: predict(m, u, vocab, n))(models[name], use_notes)

    # ---- results table
    rows = []
    for name, fn in systems.items():
        for s in ["val", "test"]:
            rows.append({"system": name, "split": s, **mae_ms(split[s], fn)})
    res = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results" / "duration.csv", index=False)
    print("\n" + res.pivot(index="system", columns="split", values=["mae_ms_nonsil", "mae_ms_all"])
          .round(1).loc[list(systems)].to_string())

    # ---- per-phone predictions (val + test) for analysis / WORLD re-synthesis
    pred_rows = []
    for u in split["val"] + split["test"]:
        preds = {name: fn(u) for name, fn in systems.items()}
        for i, p in enumerate(u["phones"]):
            pred_rows.append({"utt": u["key"], "split": u["split"], "idx": i, "phone": p,
                              "spoken": u["spoken"][i], "sung": u["sung"][i],
                              **{name: float(v[i]) for name, v in preds.items()}})
    pd.DataFrame(pred_rows).to_csv(ROOT / "results" / "duration_predictions.csv", index=False)

    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save({"state_dict": models["transformer_notes"].state_dict(), "vocab": vocab},
               ROOT / "checkpoints" / "duration.pt")

    # ---- plot one test song
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = split["test"][0]
    sl = slice(1, 81)  # skip leading silence
    x = np.arange(sl.start, sl.stop)
    plt.figure(figsize=(14, 4))
    plt.plot(x, u["sung"][sl], "k-o", ms=3, lw=1.5, label="true (sung)")
    plt.plot(x, u["spoken"][sl], color="0.6", lw=1, label="spoken")
    plt.plot(x, systems["phone_ratio"](u)[sl], "--", lw=1, label="per-phone ratio")
    plt.plot(x, systems["transformer_notes"](u)[sl], lw=1.5, label="Transformer + notes")
    plt.xticks(x, u["phones"][sl], rotation=90, fontsize=7)
    plt.ylabel("duration (s)")
    plt.title(f"Phone durations, test speaker {u['key']} (phones 1-80)")
    plt.legend()
    plt.tight_layout()
    fig = ROOT / data_cfg["figures_dir"] / f"duration_pred_{u['key']}.png"
    plt.savefig(fig, dpi=120)
    print(f"\n-> results/duration.csv, results/duration_predictions.csv, checkpoints/duration.pt, {fig}")


if __name__ == "__main__":
    main()
