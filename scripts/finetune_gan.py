#!/usr/bin/env python3
"""Adversarial fine-tuning of the hybrid acoustic model (the optional GAN step in approach.md).

L1-trained mel predictors are over-smoothed. This adds a 2-D patch discriminator on
mel crops (LSGAN + feature matching) and fine-tunes checkpoints/hybrid.pt:

    loss_G = L1(mel) + lambda_adv * adversarial + lambda_fm * feature-matching

Snapshots of each run are scored on the validation speaker (vocoded, then MCD, F0 RMSE
and speaker similarity with scripts/evaluate.score_file), next to the unmodified
hybrid. A snapshot passes the gate only if

    MCD < hybrid MCD,  F0 RMSE <= hybrid + --f0-tol,  speaker sim >= hybrid - --spk-tol

The best passing snapshot (lowest MCD) is saved to checkpoints/hybrid_gan.pt and only
then rendered for val + test (outputs/hybrid_gan_{oracle,pred}). The test speaker
never influences the decision. Gate table: results/gan_gate.csv.

Usage (GPU host, after `train_acoustic.py --world-input`):
    CUDA_VISIBLE_DEVICES=1 python scripts/finetune_gan.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.cache import FeatureCache  # noqa: E402
from features.ecapa import load_ecapa  # noqa: E402
from features.vocoder import Vocoder  # noqa: E402
from scripts.evaluate import score_file  # noqa: E402
from scripts.train_acoustic import AcousticModel, generate_outputs, infer, load_data, sample_batch  # noqa: E402


class MelDiscriminator(nn.Module):
    """2-D patch discriminator over (time, mel-bin) crops. Returns logits map + hidden features."""

    def __init__(self, ch=32):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.layers = nn.ModuleList([
            wn(nn.Conv2d(1, ch, (3, 9), padding=(1, 4))),
            wn(nn.Conv2d(ch, ch * 2, (3, 9), stride=(1, 2), padding=(1, 4))),
            wn(nn.Conv2d(ch * 2, ch * 4, (3, 9), stride=(2, 2), padding=(1, 4))),
            wn(nn.Conv2d(ch * 4, ch * 4, (3, 3), stride=(2, 2), padding=1)),
        ])
        self.post = wn(nn.Conv2d(ch * 4, 1, (3, 3), padding=1))

    def forward(self, mel):  # (B, T, n_mels)
        x, feats = mel[:, None], []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
            feats.append(x)
        return self.post(x), feats


def finetune(base_state, train, lengths, lambda_adv, lambda_fm, steps, save_every, batch, lr_g, lr_d, rng, device):
    """Fine-tune a copy of the hybrid with the discriminator. Returns {step: CPU state_dict}."""
    G = AcousticModel(world_input=True).to(device)
    G.load_state_dict(base_state)
    D = MelDiscriminator().to(device)
    opt_g = torch.optim.AdamW(G.parameters(), lr=lr_g, betas=(0.8, 0.99), weight_decay=1e-2)
    opt_d = torch.optim.AdamW(D.parameters(), lr=lr_d, betas=(0.8, 0.99))
    snaps, t0 = {}, time.time()
    for step in range(1, steps + 1):
        G.train(), D.train()
        b = sample_batch(train, lengths, batch, rng)
        real = b["mel"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fake = G(b["hub"], b["spk"], b["coarse"], b["uv"], b["wmel"]).float()

        # discriminator: real -> 1, generated -> 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d_real, _ = D(real)
            d_fake, _ = D(fake.detach())
        loss_d = ((d_real.float() - 1) ** 2).mean() + (d_fake.float() ** 2).mean()
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        opt_d.step()

        # generator: L1 + fool D + match D's features on real mels
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d_fake, f_fake = D(fake)
            with torch.no_grad():
                _, f_real = D(real)
        adv = ((d_fake.float() - 1) ** 2).mean()
        fm = sum(F.l1_loss(a.float(), r.float()) for a, r in zip(f_fake, f_real)) / len(f_fake)
        rec = F.l1_loss(fake, real)
        loss_g = rec + lambda_adv * adv + lambda_fm * fm
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt_g.step()

        if step % 500 == 0:
            print(f"  [adv {lambda_adv}] step {step:5d}  L1 {rec.item():.4f}  adv {adv.item():.3f}  "
                  f"fm {fm.item():.3f}  D {loss_d.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if step % save_every == 0:
            snaps[step] = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
    return snaps


def main(args=None):
    ap = argparse.ArgumentParser(description="Adversarial fine-tuning of the hybrid acoustic model")
    ap.add_argument("--lambda-adv", type=float, nargs="+", default=[0.1, 0.5])
    ap.add_argument("--lambda-fm", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-g", type=float, default=5e-5)
    ap.add_argument("--lr-d", type=float, default=1e-4)
    ap.add_argument("--f0-tol", type=float, default=0.5, help="allowed F0 RMSE increase (Hz)")
    ap.add_argument("--spk-tol", type=float, default=0.01, help="allowed speaker-similarity drop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.5)
    args = ap.parse_args(args)

    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml"))
    extract_cfg = yaml.safe_load(open(ROOT / "config" / "extract.yaml"))
    sr = data_cfg["sample_rate"]
    cache = FeatureCache(ROOT / data_cfg["cache_root"])
    manifest = pd.read_csv(ROOT / data_cfg["cache_root"] / "split_manifest.csv", dtype={"utt_id": str})
    utts, gpu, train, _ = load_data(manifest, cache, True, device)
    lengths = np.array([len(g["coarse"]) for g in train], dtype=float)
    base = torch.load(ROOT / "checkpoints" / "hybrid.pt", map_location=device)["state_dict"]

    candidates = [("hybrid", 0.0, 0, {k: v.cpu() for k, v in base.items()})]
    for lam in args.lambda_adv:
        print(f"== fine-tuning with lambda_adv={lam}, lambda_fm={args.lambda_fm}")
        snaps = finetune(base, train, lengths, lam, args.lambda_fm, args.steps, args.save_every,
                         args.batch, args.lr_g, args.lr_d, rng, device)
        candidates += [(f"adv{lam}_step{s}", lam, s, st) for s, st in snaps.items()]

    # ---- validation gate (val speaker only)
    vocoder = Vocoder(device)
    ecapa = load_ecapa(extract_cfg["ecapa"]["model"], str(ROOT / extract_cfg["ecapa"]["savedir"]), "cpu")
    val_rows = list(manifest[manifest["split"] == "val"].itertuples())
    refs = {f"{r.speaker_id}_{r.utt_id}": (librosa.load(r.sing_wav, sr=sr)[0],
                                           cache.load("f0", r.split, f"{r.speaker_id}_{r.utt_id}"),
                                           cache.load("ecapa", r.split, f"{r.speaker_id}_{r.utt_id}"))
            for r in val_rows}
    table = []
    for tag, lam, step, state in candidates:
        model = AcousticModel(world_input=True).to(device)
        model.load_state_dict(state)
        out_dir = ROOT / "outputs" / "gan_val" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        scores = []
        for key, (ref, f0_r, spk_ref) in refs.items():
            mel = infer(model, gpu[key]).cpu().numpy()
            wav = vocoder(mel, utts[key]["f0"])
            sf.write(out_dir / f"{key}.wav", (0.9 * wav / max(np.abs(wav).max(), 1e-9)).astype(np.float32), sr)
            scores.append({**score_file(out_dir / f"{key}.wav", ref, f0_r, spk_ref, ecapa, extract_cfg["f0"], sr),
                           "val_l1": float(np.abs(mel - utts[key]["mel"]).mean())})
        m = pd.DataFrame(scores)[["mcd_db", "f0_rmse_hz", "f0_corr", "spk_cos", "val_l1"]].mean().to_dict()
        table.append({"candidate": tag, "lambda_adv": lam, "step": step, **m})
        print(f"{tag:22s} MCD {m['mcd_db']:.3f}  F0 RMSE {m['f0_rmse_hz']:.2f} Hz  CORR {m['f0_corr']:.3f}  "
              f"spk {m['spk_cos']:.3f}  L1 {m['val_l1']:.4f}", flush=True)
        del model

    df = pd.DataFrame(table)
    b = df.iloc[0]
    df["passes"] = (df["mcd_db"] < b["mcd_db"]) & (df["f0_rmse_hz"] <= b["f0_rmse_hz"] + args.f0_tol) \
        & (df["spk_cos"] >= b["spk_cos"] - args.spk_tol) & (df["step"] > 0)
    (ROOT / "results").mkdir(exist_ok=True)
    df.round(4).to_csv(ROOT / "results" / "gan_gate.csv", index=False)
    print("\n" + df.round(3).to_string(index=False))

    if not df["passes"].any():
        print("\nGATE FAILED: no adversarial snapshot beats the hybrid on validation; nothing saved.")
        return
    best = df[df["passes"]].sort_values("mcd_db").iloc[0]
    state = next(st for tag, _, _, st in candidates if tag == best["candidate"])
    torch.save({"state_dict": state, "candidate": best["candidate"], "lambda_adv": float(best["lambda_adv"]),
                "step": int(best["step"])}, ROOT / "checkpoints" / "hybrid_gan.pt")
    print(f"\nGATE PASSED: {best['candidate']} -> checkpoints/hybrid_gan.pt")
    model = AcousticModel(world_input=True).to(device)
    model.load_state_dict(state)
    generate_outputs(model, "hybrid_gan", True, utts, gpu, manifest, cache, device, data_cfg["figures_dir"])


if __name__ == "__main__":
    main()
