#!/usr/bin/env python3
"""Speech-to-singing inference: spoken lyrics + reference singing -> the speaker singing them.

    python infer.py --speech read/05.wav --speech-phones read/05.txt \\
                    --melody sing/05.wav --melody-phones sing/05.txt --out out.wav

Everything is recomputed from the audio (HuBERT, ECAPA, Harvest F0, notes, WORLD).
The phone annotations (NUS-48E format: "start end phone" per line) must describe the
same lyrics. Identical phone sequences (sil/sp may differ) map 1:1; another singer's
rendition, whose annotation differs in places, is aligned phone by phone (align_phones).
The reference singing provides the melody F0 (optionally --transpose'd) and the timing:
  --durations reference  every phone's timing from the reference singing
  --durations phrase     the Phase 2 duration model predicts every phone; only the melody's
                         rests (pauses >= MIN_REST s) stay in place, like rests in a score
  --durations predicted  the duration model's timing, only stretched to the melody's length
(for the duration model the notes are matched to phones through the reference annotation).
--postfilter sharpen / sharpen_gv undoes part of the neural mel's over-smoothing (see postfilter).

Needs the trained checkpoints (checkpoints/{hybrid,acoustic,duration}.pt) and the
vocoder assets in vocoder/ (see features/vocoder.py).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.ndimage import uniform_filter1d

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

EXTRACT_CFG = yaml.safe_load(open(ROOT / "config" / "extract.yaml"))
SR = yaml.safe_load(open(ROOT / "config" / "data.yaml"))["sample_rate"]
SYSTEMS = ("world", "acoustic", "hybrid")
MIN_REST = 0.5  # s: sung pauses at least this long anchor --durations phrase
POSTFILTERS = ("none", "sharpen", "sharpen_gv")
# per-bin std(real)/std(hybrid) of the log-mel over time on JLEE (val), as a piecewise-linear fit
GV_BINS, GV_RATIO = [0, 64, 80, 92, 127], [1.03, 1.08, 1.21, 1.35, 1.29]


def edit_ops(a, b) -> list[tuple]:
    """Minimum-edit (Levenshtein) alignment as difflib-style opcodes (tag, i0, i1, j0, j1).

    difflib's greedy longest-block matching can pair a repeated chorus with the wrong repeat.
    """
    D = [list(range(len(b) + 1))]
    for i, x in enumerate(a, 1):
        prev, cur = D[-1], [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j - 1] + (x != y), prev[j] + 1, cur[j - 1] + 1)
        D.append(cur)
    steps, i, j = [], len(a), len(b)  # (equal?, i0, i1, j0, j1) per step, backwards
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            steps.append((a[i - 1] == b[j - 1], i - 1, i, j - 1, j))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            steps.append((False, i - 1, i, j, j))
            i -= 1
        else:
            steps.append((False, i, i, j - 1, j))
            j -= 1
    ops = []
    for eq, i0, i1, j0, j1 in reversed(steps):
        if ops and ops[-1][0] == eq:
            ops[-1][2], ops[-1][4] = i1, j1
        else:
            ops.append([eq, i0, i1, j0, j1])
    return [("equal" if eq else "replace" if i1 > i0 and j1 > j0 else "delete" if i1 > i0 else "insert",
             i0, i1, j0, j1) for eq, i0, i1, j0, j1 in ops]


def align_phones(read_ph, sing_ph) -> tuple[list[dict], float]:
    """Each spoken phone's start/end/duration on the singing timeline, and the share of spoken
    phones matched exactly.

    Equal stretches (and same-length substitutions) map phone by phone. A mismatched stretch's
    sung time is shared by its spoken phones in proportion to their spoken durations; a stretch
    with phones on one side only is merged into its neighbour.
    """
    a, b = [norm_phone(p["phone"]) for p in read_ph], [norm_phone(p["phone"]) for p in sing_ph]
    ops = edit_ops(a, b)
    segs = []  # [spoken i0, i1, sung j0, j1]
    for tag, i0, i1, j0, j1 in ops:
        if i1 - i0 == j1 - j0:
            segs += [[i, i + 1, j, j + 1] for i, j in zip(range(i0, i1), range(j0, j1))]
        elif segs and (i0 == i1 or j0 == j1):
            segs[-1][1], segs[-1][3] = i1, j1
        else:
            segs.append([i0, i1, j0, j1])
    if segs[0][0] == segs[0][1] or segs[0][2] == segs[0][3]:  # one-sided stretch at the start
        if len(segs) == 1:
            raise ValueError("no phones in common")
        nxt = segs.pop(1)
        segs[0][1], segs[0][3] = nxt[1], nxt[3]

    edges = phone_edges(sing_ph)
    out = []
    for i0, i1, j0, j1 in segs:
        if i1 - i0 == 1 and j1 - j0 == 1:  # 1:1 keeps the annotation exactly
            s = sing_ph[j0]
            out.append({"phone": read_ph[i0]["phone"], "start": s["start"], "end": s["end"], "duration": s["duration"]})
            continue
        d = np.array([max(p["duration"], 1e-3) for p in read_ph[i0:i1]])
        t = edges[j0] + (edges[j1] - edges[j0]) * np.concatenate([[0.0], np.cumsum(d)]) / d.sum()
        out += [{"phone": read_ph[i]["phone"], "start": t[k], "end": t[k + 1], "duration": t[k + 1] - t[k]}
                for k, i in enumerate(range(i0, i1))]
    matched = sum(i1 - i0 for tag, i0, i1, _, _ in ops if tag == "equal") / len(a)
    return out, matched


@lru_cache
def _hubert(device):
    return load_hubert(EXTRACT_CFG["hubert"]["model"], device)


@lru_cache
def _ecapa():
    return load_ecapa(EXTRACT_CFG["ecapa"]["model"], str(ROOT / EXTRACT_CFG["ecapa"]["savedir"]), "cpu")


@lru_cache
def _vocoder(device):
    return Vocoder(device)


@lru_cache
def _acoustic(system, device, checkpoint=None):
    model = AcousticModel(world_input=system == "hybrid").to(device)
    ckpt = Path(checkpoint) if checkpoint else ROOT / "checkpoints" / f"{system}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    return model


@lru_cache
def _duration_model():
    ck = torch.load(ROOT / "checkpoints" / "duration.pt", map_location="cpu")
    model = DurationPredictor(len(ck["vocab"]))
    model.load_state_dict(ck["state_dict"])
    return model, ck["vocab"]


def predict_durations(read_ph, sing_ph, mapped_ph, hub, f0) -> np.ndarray:
    """Phase 2 duration model: sung duration (s) of every spoken phone."""
    model, vocab = _duration_model()
    ncfg, fcfg = EXTRACT_CFG["notes"], EXTRACT_CFG["f0"]
    notes = segment_notes(f0, sing_ph, sr=1000, hop_length=fcfg["frame_period_ms"],
                          stability_semitones=ncfg["f0_stability_semitones"], min_note_frames=ncfg["min_note_frames"])
    u = {"phones": [norm_phone(p["phone"]) for p in read_ph],
         "spoken": np.array([max(p["duration"], 1e-3) for p in read_ph], dtype=np.float32),
         "sung": np.ones(len(read_ph), dtype=np.float32),  # unused by predict()
         "hubert": np.stack([pool_hubert(hub, p["start"], p["end"]) for p in read_ph]),
         "notes": assign_notes(mapped_ph, notes)}
    return predict(model, u, vocab, use_notes=True)


def anchor_rests(durs, mapped_ph) -> np.ndarray:
    """Predicted durations with the melody's rests kept in place: rests (pauses >= MIN_REST s on
    the singing timeline) take their reference duration, each phrase between them is rescaled
    to its reference span."""
    ref = np.diff(phone_edges(mapped_ph))
    rest = np.array([norm_phone(p["phone"]) == "sil" for p in mapped_ph]) & (ref >= MIN_REST)
    out, i = np.where(rest, ref, durs), 0
    for is_rest, run in itertools.groupby(rest):
        n = len(list(run))
        if not is_rest:
            out[i:i + n] *= ref[i:i + n].sum() / out[i:i + n].sum()
        i += n
    return out


def postfilter(mel, kind="none") -> np.ndarray:
    """The neural mel varies less over time than real singing (JLEE: 0.81x the frame-to-frame detail,
    0.8x the std in the upper bins). sharpen: temporal unsharp mask restoring the detail;
    sharpen_gv: also stretches every bin around its song mean by GV_RATIO."""
    if kind == "sharpen_gv":
        mu = mel.mean(0)
        mel = mu + np.interp(np.arange(mel.shape[1]), GV_BINS, GV_RATIO) * (mel - mu)
    if kind in ("sharpen", "sharpen_gv"):
        mel = mel + 0.25 * (mel - uniform_filter1d(mel, 9, axis=0))
    return mel.astype(np.float32)


def synthesize(speech, read_ph, melody, sing_ph, systems=("hybrid",), durations=("reference",),
               transpose=0.0, device="cpu", checkpoint=None, post="none") -> dict:
    """{(system, durations): waveform at SR} for every requested system x timing.

    speech/melody: wav paths; read_ph/sing_ph: their phone annotations; transpose: semitones;
    post: postfilter for the neural systems.
    """
    mapped, _ = align_phones(read_ph, sing_ph)
    # speech: content + speaker
    hub = extract_hubert(speech, _hubert(device), layer=EXTRACT_CFG["hubert"]["layer"], device=device)
    spk = extract_ecapa(speech, _ecapa())
    # melody: F0 contour + length of the output on the vocoder's frame grid
    fcfg = EXTRACT_CFG["f0"]
    f0 = extract_f0_harvest(melody, fcfg["frame_period_ms"], fcfg["fmin"], fcfg["fmax"]) * 2 ** (transpose / 12)
    vocoder = _vocoder(device)
    n_frames = len(vocoder.mel(load_numpy(melody, vocoder.sr)))

    out = {}
    for timing in durations:
        if timing == "reference":
            durs = np.array([p["duration"] for p in mapped])
            sung_edges = phone_edges(mapped)
        else:
            durs = predict_durations(read_ph, sing_ph, mapped, hub, f0)
            if timing == "phrase":
                durs = anchor_rests(durs, mapped)
            else:  # fill the melody, as in training/evaluation
                durs *= (n_frames / FRAME_RATE) / durs.sum()
            sung_edges = np.concatenate([[0.0], np.cumsum(durs)])
        world = None
        if {"world", "hybrid"} & set(systems):
            world = world_convert(speech, read_ph, durs, f0, SR).astype(np.float32)
        for system in systems:
            if system == "world":
                out[system, timing] = world
                continue
            cond = make_conditioning(hub, read_ph, sung_edges, f0, n_frames, spk,
                                     vocoder.mel(world) if system == "hybrid" else None)
            mel = postfilter(infer(_acoustic(system, device, checkpoint), to_device(cond, device)).cpu().numpy(), post)
            out[system, timing] = vocoder(mel, cond["f0"])
    return out


def write_wav(path, wav):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (0.9 * wav / max(np.abs(wav).max(), 1e-9)).astype(np.float32), SR)


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speech", required=True, help="spoken lyrics (wav)")
    ap.add_argument("--speech-phones", required=True, help="phone annotation of the speech (txt)")
    ap.add_argument("--melody", required=True, help="reference singing providing the melody (wav)")
    ap.add_argument("--melody-phones", required=True, help="phone annotation of the reference singing (txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=SYSTEMS, default="hybrid", help="world = the classical baseline")
    ap.add_argument("--checkpoint", help="acoustic checkpoint (default: checkpoints/<model>.pt)")
    ap.add_argument("--durations", choices=["reference", "phrase", "predicted"], default="reference")
    ap.add_argument("--transpose", type=float, default=0.0, help="shift the melody by this many semitones")
    ap.add_argument("--postfilter", choices=POSTFILTERS, default="none",
                    help="sharpen the neural mel (less muffled; not applied to --model world)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(args)

    read_ph, sing_ph = parse_annotation(args.speech_phones), parse_annotation(args.melody_phones)
    _, matched = align_phones(read_ph, sing_ph)
    if matched < 0.7:  # same-song pairs in NUS-48E: 77-89 %, different songs: <= 56 %
        sys.exit(f"only {matched:.0%} of the {len(read_ph)} spoken phones match the {len(sing_ph)} sung phones: "
                 "the speech and the reference singing must have the same lyrics")

    wav = synthesize(args.speech, read_ph, args.melody, sing_ph, (args.model,), (args.durations,),
                     args.transpose, resolve_device(args.device), args.checkpoint,
                     args.postfilter)[args.model, args.durations]
    write_wav(args.out, wav)
    print(f"{args.out}: {len(wav) / SR:.1f} s ({args.model}, {args.durations} durations, {len(read_ph)} phones, "
          f"{matched:.0%} matched, transpose {args.transpose:+g}, postfilter {args.postfilter})")


if __name__ == "__main__":
    main()
