#!/usr/bin/env python3
"""Phase 3: Conformer acoustic model (content + speaker + melody -> singing mel).

Per mel frame on the singing timeline (NSF-HiFiGAN frames, 44.1 kHz / 512 = 86.1 Hz):
  content  HuBERT of the *spoken* lyrics, time-warped phone-by-phone onto the
           singing timeline (sung phone durations: annotations for training,
           Phase 2 predictions at inference)
  speaker  ECAPA embedding of the spoken utterance (broadcast over time)
  melody   sung F0 (Harvest) as a 256-bin coarse pitch embedding + voiced flag
Target: 128-bin log-mel of the real singing (cache "vmel"), L1 loss. The
pretrained NSF-HiFiGAN vocoder turns predicted mel + melody F0 into audio.

Writes (val + test speakers):
  outputs/acoustic_oracle/   annotated sung phone durations
  outputs/acoustic_pred/     Phase 2 predicted durations (results/duration_predictions.csv)
  outputs/nsf_copysyn/       vocoder on the real mel (upper bound)

Usage (GPU host, needs `preprocess.py --components vmel` first):
    CUDA_VISIBLE_DEVICES=1 python scripts/train_acoustic.py
    python scripts/train_acoustic.py --generate-only      # reuse checkpoints/acoustic.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchaudio.models import Conformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.annotations import parse_annotation  # noqa: E402
from features.cache import FeatureCache  # noqa: E402
from features.vocoder import Vocoder, interp_f0  # noqa: E402

F0_RATE = 100.0       # cache "f0": Harvest, 10 ms frames
HUBERT_RATE = 50.0
FRAME_RATE = 44100 / 512
SEG = 512             # training crop, frames (~5.9 s)
OVERLAP = 128         # inference chunk overlap, frames
F0_MIN, F0_MAX = 60.0, 1100.0


# --------------------------------------------------------------------------- conditioning

def phone_edges(phones) -> np.ndarray:
    # a few annotated phones overlap their neighbours; force non-decreasing boundaries
    return np.maximum.accumulate(np.array([phones[0]["start"]] + [p["end"] for p in phones]))


def warp_content(hub: np.ndarray, read_edges, sung_edges, n_frames: int) -> np.ndarray:
    """HuBERT frames of the speech, resampled onto the singing timeline phone by phone."""
    t = (np.arange(n_frames) + 0.5) / FRAME_RATE              # singing time of each mel frame
    s = np.interp(t, sung_edges, read_edges)                   # matching time in the speech
    pos = np.clip(s * HUBERT_RATE - 0.5, 0, len(hub) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(hub) - 1)
    w = (pos - i0)[:, None]
    return ((1 - w) * hub[i0] + w * hub[i1]).astype(np.float16)


def frame_f0(f0: np.ndarray, n_frames: int) -> np.ndarray:
    """Harvest F0 (100 Hz) at mel frame centres (nearest frame keeps 0 Hz = unvoiced)."""
    idx = np.round((np.arange(n_frames) + 0.5) / FRAME_RATE * F0_RATE).astype(int)
    return f0[np.minimum(idx, len(f0) - 1)].astype(np.float32)


def f0_coarse(f0: np.ndarray):
    """256 mel-spaced pitch bins of the gap-filled F0, plus the unvoiced flag."""
    filled, uv = interp_f0(f0)
    m = 1127 * np.log1p(np.clip(filled, F0_MIN, F0_MAX) / 700)
    lo, hi = 1127 * np.log1p(F0_MIN / 700), 1127 * np.log1p(F0_MAX / 700)
    return np.rint((m - lo) * 255 / (hi - lo)).astype(np.int64), uv.astype(np.int64)


def build_utterance(row, cache, sung_durations=None) -> dict:
    """Frame-level inputs + target for one song. sung_durations=None -> annotated durations."""
    key = f"{row.speaker_id}_{row.utt_id}"
    read_ph, sing_ph = parse_annotation(row.read_txt), parse_annotation(row.sing_txt)
    mel = cache.load("vmel", row.split, key)
    f0 = frame_f0(cache.load("f0", row.split, key), len(mel))
    if sung_durations is None:
        sung_edges = phone_edges(sing_ph)
    else:
        sung_edges = np.concatenate([[0.0], np.cumsum(sung_durations)])
    coarse, uv = f0_coarse(f0)
    return {
        "key": key, "split": row.split, "mel": mel, "f0": f0, "coarse": coarse, "uv": uv,
        "spk": cache.load("ecapa", row.split, key),
        "hub": warp_content(cache.load("hubert", row.split, key), phone_edges(read_ph), sung_edges, len(mel)),
    }


def to_device(u: dict, device) -> dict:
    return {
        "hub": torch.from_numpy(u["hub"]).to(device),
        "coarse": torch.from_numpy(u["coarse"]).to(device),
        "uv": torch.from_numpy(u["uv"]).to(device),
        "spk": torch.from_numpy(u["spk"]).to(device),
        "mel": torch.from_numpy(u["mel"]).to(device),
    }


# ----------------------------------------------------------------------------------- model

class AcousticModel(nn.Module):
    def __init__(self, d=256, layers=6, heads=4, ffn=1024, kernel=31, n_mels=128, dropout=0.1):
        super().__init__()
        self.content = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d))
        self.spk = nn.Linear(192, d)
        self.pitch = nn.Embedding(256, d)
        self.voicing = nn.Embedding(2, d)
        self.conformer = Conformer(input_dim=d, num_heads=heads, ffn_dim=ffn, num_layers=layers,
                                   depthwise_conv_kernel_size=kernel, dropout=dropout)
        self.out = nn.Linear(d, n_mels)

    def forward(self, hub, spk, coarse, uv):
        x = self.content(hub.float()) + self.spk(spk)[:, None] + self.pitch(coarse) + self.voicing(uv)
        lengths = torch.full((x.shape[0],), x.shape[1], device=x.device)
        x, _ = self.conformer(x, lengths)
        return self.out(x)


@torch.no_grad()
def infer(model, g: dict) -> torch.Tensor:
    """Whole-song mel in SEG-frame chunks (the training length), cross-faded over OVERLAP frames."""
    model.eval()
    T = len(g["coarse"])
    out = torch.zeros(T, model.out.out_features, device=g["hub"].device)
    weight = torch.zeros(T, 1, device=out.device)
    ramp = torch.linspace(0, 1, OVERLAP, device=out.device)
    for s in range(0, T, SEG - OVERLAP):
        e = min(s + SEG, T)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(g["hub"][None, s:e], g["spk"][None], g["coarse"][None, s:e], g["uv"][None, s:e])[0].float()
        w = torch.ones(e - s, 1, device=out.device)
        if s > 0:
            w[:OVERLAP, 0] = ramp[: e - s]
        if e < T:
            w[-OVERLAP:, 0] = torch.minimum(w[-OVERLAP:, 0], ramp.flip(0))
        out[s:e] += y * w
        weight[s:e] += w
        if e == T:
            break
    return out / weight.clamp_min(1e-6)


def val_l1(model, val) -> float:
    return float(np.mean([F.l1_loss(infer(model, g), g["mel"]).item() for g in val]))


def sample_batch(train, lengths, batch, rng):
    picks = rng.choice(len(train), size=batch, p=lengths / lengths.sum())
    cols = {k: [] for k in ["hub", "coarse", "uv", "mel"]}
    for k in picks:
        g = train[k]
        s = int(rng.integers(0, len(g["coarse"]) - SEG + 1))
        for c in cols:
            cols[c].append(g[c][s:s + SEG])
    out = {c: torch.stack(v) for c, v in cols.items()}
    out["spk"] = torch.stack([train[k]["spk"] for k in picks])
    return out


# ------------------------------------------------------------------------------------ main

def main(args=None):
    ap = argparse.ArgumentParser(description="Train/generate with the Conformer acoustic model")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many evaluations without val improvement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.5,
                    help="cap on this process's share of the GPU (shared machine)")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args(args)

    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml"))
    cache = FeatureCache(ROOT / data_cfg["cache_root"])
    manifest = pd.read_csv(ROOT / data_cfg["cache_root"] / "split_manifest.csv", dtype={"utt_id": str})

    t0 = time.time()
    utts = {r.speaker_id + "_" + r.utt_id: build_utterance(r, cache) for r in manifest.itertuples()}
    gpu = {k: to_device(u, device) for k, u in utts.items()}
    train = [gpu[k] for k, u in utts.items() if u["split"] == "train"]
    val = [gpu[k] for k, u in utts.items() if u["split"] == "val"]
    lengths = np.array([len(g["coarse"]) for g in train], dtype=float)
    print(f"loaded {len(utts)} songs ({lengths.sum() / FRAME_RATE / 60:.0f} min train) in {time.time() - t0:.0f}s")

    model = AcousticModel().to(device)
    print(f"acoustic model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    ckpt_path = ROOT / "checkpoints" / "acoustic.pt"
    ckpt_path.parent.mkdir(exist_ok=True)

    if not args.generate_only:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
        best, since_best, log, t0 = float("inf"), 0, [], time.time()
        for step in range(1, args.steps + 1):
            model.train()
            b = sample_batch(train, lengths, args.batch, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(b["hub"], b["spk"], b["coarse"], b["uv"])
            loss = F.l1_loss(pred.float(), b["mel"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % args.eval_every == 0 or step == args.steps:
                v = val_l1(model, val)
                since_best += 1
                if v < best:
                    best, since_best = v, 0
                    torch.save({"state_dict": model.state_dict(), "step": step, "val_l1": v}, ckpt_path)
                log.append({"step": step, "train_l1": loss.item(), "val_l1": v})
                print(f"step {step:6d}  train L1 {loss.item():.4f}  val L1 {v:.4f}  best {best:.4f}  "
                      f"({time.time() - t0:.0f}s, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
                if since_best >= args.patience:
                    print(f"early stop: no val improvement for {since_best} evaluations")
                    break
        (ROOT / "results").mkdir(exist_ok=True)
        pd.DataFrame(log).to_csv(ROOT / "results" / "acoustic_train.csv", index=False)

    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["state_dict"])
    print(f"using checkpoint from step {ck['step']} (val L1 {ck['val_l1']:.4f})")

    # ---- generation: val + test speakers
    vocoder = Vocoder(device)
    preds = pd.read_csv(ROOT / "results" / "duration_predictions.csv").sort_values(["utt", "idx"])
    pred_durs = preds.groupby("utt")["transformer_notes"].apply(np.array).to_dict()
    outs = {name: ROOT / "outputs" / name for name in ["acoustic_oracle", "acoustic_pred", "nsf_copysyn"]}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)

    def write(path, wav):
        sf.write(path, (0.9 * wav / max(np.abs(wav).max(), 1e-9)).astype(np.float32), vocoder.sr)

    for r in manifest[manifest["split"].isin(["val", "test"])].itertuples():
        key = f"{r.speaker_id}_{r.utt_id}"
        u = utts[key]
        mel_oracle = infer(model, gpu[key]).cpu().numpy()
        # predicted durations, rescaled to fill the melody (as in the WORLD baseline)
        durs = pred_durs[key] * (len(u["mel"]) / FRAME_RATE) / pred_durs[key].sum()
        mel_pred = infer(model, to_device(build_utterance(r, cache, durs), device)).cpu().numpy()
        write(outs["acoustic_oracle"] / f"{key}.wav", vocoder(mel_oracle, u["f0"]))
        write(outs["acoustic_pred"] / f"{key}.wav", vocoder(mel_pred, u["f0"]))
        write(outs["nsf_copysyn"] / f"{key}.wav", vocoder(u["mel"], u["f0"]))
        print(f"{key}: mel L1 oracle {np.abs(mel_oracle - u['mel']).mean():.3f}  "
              f"predicted-dur {np.abs(mel_pred - u['mel']).mean():.3f}")

    # ---- figure: real vs generated mel, 10 s of the first test song
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = manifest[manifest["split"] == "test"].iloc[0]
    key = f"{first.speaker_id}_{first.utt_id}"
    s = int(np.argmax(utts[key]["uv"] == 0))  # first voiced frame
    sl = slice(s, s + int(10 * FRAME_RATE))
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for a, m, title in [(ax[0], utts[key]["mel"], "real singing"),
                        (ax[1], infer(model, gpu[key]).cpu().numpy(), "Conformer (oracle durations)")]:
        a.imshow(m[sl].T, origin="lower", aspect="auto", cmap="magma", vmin=-11, vmax=1)
        a.set_title(f"{key}: {title}")
        a.set_ylabel("mel bin")
    ax[1].set_xlabel("frame (86 Hz)")
    plt.tight_layout()
    fig_path = ROOT / data_cfg["figures_dir"] / f"acoustic_mel_{key}.png"
    plt.savefig(fig_path, dpi=120)
    print("-> " + ", ".join(str(d.relative_to(ROOT)) for d in outs.values()) + f", {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
