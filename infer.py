#!/usr/bin/env python3
"""Speech-to-singing inference: spoken lyrics + reference singing -> the speaker singing them.

    python infer.py --speech read/05.wav --speech-phones read/05.txt \\
                    --melody sing/05.wav --melody-phones sing/05.txt --out out.wav

Everything is recomputed from the audio (HuBERT, ECAPA, Harvest F0, notes, WORLD).
The phone annotations (NUS-48E format: "start end phone" per line) must contain the
same phone sequence; sil/sp may differ. The reference singing provides the melody F0
and, with --durations reference, the timing of every phone; with --durations predicted
the Phase 2 duration model predicts the timing from the speech and the melody's notes
(the notes are still matched to phones through the reference annotation).

Needs the trained checkpoints (checkpoints/{hybrid,acoustic,duration}.pt) and the
vocoder assets in vocoder/ (see features/vocoder.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselines.world_baseline import convert as world_convert  # noqa: E402
from data.annotations import parse_annotation  # noqa: E402
from features.audio import load_numpy  # noqa: E402
from features.ecapa import extract_ecapa, load_ecapa  # noqa: E402
from features.hubert import extract_hubert, load_hubert, resolve_device  # noqa: E402
from features.notes import segment_notes  # noqa: E402
from features.vocoder import Vocoder  # noqa: E402
from features.world_f0 import extract_f0_harvest  # noqa: E402
from scripts.train_acoustic import (FRAME_RATE, AcousticModel, infer, make_conditioning,  # noqa: E402
                                    phone_edges, to_device)
from scripts.train_duration import (DurationPredictor, assign_notes, norm_phone,  # noqa: E402
                                    pool_hubert, predict)


def predict_durations(read_ph, sing_ph, hub, f0, extract_cfg, device) -> np.ndarray:
    """Phase 2 duration model: sung duration (s) of every phone."""
    ck = torch.load(ROOT / "checkpoints" / "duration.pt", map_location="cpu")
    model = DurationPredictor(len(ck["vocab"]))
    model.load_state_dict(ck["state_dict"])
    ncfg, fcfg = extract_cfg["notes"], extract_cfg["f0"]
    notes = segment_notes(f0, sing_ph, sr=1000, hop_length=fcfg["frame_period_ms"],
                          stability_semitones=ncfg["f0_stability_semitones"], min_note_frames=ncfg["min_note_frames"])
    u = {"phones": [norm_phone(p["phone"]) for p in read_ph],
         "spoken": np.array([max(p["duration"], 1e-3) for p in read_ph], dtype=np.float32),
         "sung": np.ones(len(read_ph), dtype=np.float32),  # unused by predict()
         "hubert": np.stack([pool_hubert(hub, p["start"], p["end"]) for p in read_ph]),
         "notes": assign_notes(sing_ph, notes)}
    return predict(model, u, ck["vocab"], use_notes=True)


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", required=True, help="spoken lyrics (wav)")
    ap.add_argument("--speech-phones", required=True, help="phone annotation of the speech (txt)")
    ap.add_argument("--melody", required=True, help="reference singing providing the melody (wav)")
    ap.add_argument("--melody-phones", required=True, help="phone annotation of the reference singing (txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["hybrid", "acoustic"], default="hybrid")
    ap.add_argument("--checkpoint", help="acoustic checkpoint (default: checkpoints/<model>.pt)")
    ap.add_argument("--durations", choices=["reference", "predicted"], default="reference")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(args)

    extract_cfg = yaml.safe_load(open(ROOT / "config" / "extract.yaml"))
    sr = yaml.safe_load(open(ROOT / "config" / "data.yaml"))["sample_rate"]
    device = resolve_device(args.device)

    read_ph, sing_ph = parse_annotation(args.speech_phones), parse_annotation(args.melody_phones)
    norm = lambda ph: [norm_phone(p["phone"]) for p in ph]  # noqa: E731
    if norm(read_ph) != norm(sing_ph):
        sys.exit(f"phone sequences differ ({len(read_ph)} spoken vs {len(sing_ph)} sung phones): "
                 "the speech and the reference singing must have the same lyrics/phones")

    # speech: content + speaker
    hub = extract_hubert(args.speech, load_hubert(extract_cfg["hubert"]["model"], device),
                         layer=extract_cfg["hubert"]["layer"], device=device)
    spk = extract_ecapa(args.speech, load_ecapa(extract_cfg["ecapa"]["model"],
                                                str(ROOT / extract_cfg["ecapa"]["savedir"]), "cpu"))
    # melody: F0 contour + length of the output on the vocoder's frame grid
    fcfg = extract_cfg["f0"]
    f0 = extract_f0_harvest(args.melody, fcfg["frame_period_ms"], fcfg["fmin"], fcfg["fmax"])
    vocoder = Vocoder(device)
    n_frames = len(vocoder.mel(load_numpy(args.melody, vocoder.sr)))

    # timing on the singing timeline
    if args.durations == "reference":
        durs = np.array([p["duration"] for p in sing_ph])
        sung_edges = phone_edges(sing_ph)
    else:
        durs = predict_durations(read_ph, sing_ph, hub, f0, extract_cfg, device)
        durs *= (n_frames / FRAME_RATE) / durs.sum()  # fill the melody, as in training/evaluation
        sung_edges = np.concatenate([[0.0], np.cumsum(durs)])

    hybrid = args.model == "hybrid"
    world_mel = vocoder.mel(world_convert(args.speech, read_ph, durs, f0, sr).astype(np.float32)) if hybrid else None
    cond = make_conditioning(hub, read_ph, sung_edges, f0, n_frames, spk, world_mel)

    model = AcousticModel(world_input=hybrid).to(device)
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / "checkpoints" / f"{args.model}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    mel = infer(model, to_device(cond, device)).cpu().numpy()

    wav = vocoder(mel, cond["f0"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, (0.9 * wav / max(np.abs(wav).max(), 1e-9)).astype(np.float32), vocoder.sr)
    print(f"{args.out}: {len(wav) / vocoder.sr:.1f} s ({args.model}, {args.durations} durations, "
          f"{len(read_ph)} phones, speech {len(hub) / 50:.1f} s)")


if __name__ == "__main__":
    main()
