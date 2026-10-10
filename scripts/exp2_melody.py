#!/usr/bin/env python3
"""Exp 2 - different melody: a speaker's spoken lyrics sung to a melody they were not recorded with.

(a) cross:     every val/test song's speech + another singer's rendition of the same song, cropped to
               the longest stretch where the two phone annotations differ by at most MAX_EDIT phones
               at a time (infer.align_phones bridges those), starting and ending at a shared pause.
               The melody is shifted by whole octaves towards the speaker: the octave closest to the
               pitch gap between the two speaking voices (female melody -> male speaker: -12).
(b) transpose: the speaker's own melody shifted by -3 / 0 / +3 semitones (0 = the Exp 1 control).

Every condition is rendered by WORLD, the pure and the hybrid model with the three timings of
infer.py: the melody's own ("reference"), predicted with the melody's rests in place ("phrase") and
predicted ("predicted"). No real singing exists for these, so the metrics are
melody adherence (output F0 vs the shifted melody F0: RMSE in cents, correlation), speaker similarity
to the speaker's speech (spk_cos) and to the melody singer's speech (leak_cos). For the 0-shift control
MCD vs the real singing is added, to check that it reproduces Exp 1.

    python scripts/exp2_melody.py        # -> outputs/exp2/<condition>/, results/exp2{,_summary}.csv
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.annotations import parse_annotation  # noqa: E402
from features.audio import load_numpy  # noqa: E402
from features.cache import FeatureCache  # noqa: E402
from features.ecapa import extract_ecapa  # noqa: E402
from features.hubert import resolve_device  # noqa: E402
from features.world_f0 import extract_f0_harvest  # noqa: E402
from infer import EXTRACT_CFG, SR, SYSTEMS, _ecapa, edit_ops, synthesize, write_wav  # noqa: E402
from scripts.evaluate import cosine, f0_metrics, mcd  # noqa: E402
from scripts.train_acoustic import phone_edges  # noqa: E402
from scripts.train_duration import norm_phone  # noqa: E402

SHIFTS = (-3, 0, 3)
MAX_EDIT = 5  # phones
TIMINGS = ("reference", "phrase", "predicted")


def clean_region(read_ph, sing_ph):
    """(i0, i1, j0, j1): longest stretch (sung seconds) with no edit over MAX_EDIT phones, cut at shared pauses.

    Pauses are left out of the search: as extra matching tokens they would split a long stretch of
    different lyrics into small edits.
    """
    a, b = [norm_phone(p["phone"]) for p in read_ph], [norm_phone(p["phone"]) for p in sing_ph]
    ia, ib = [i for i, p in enumerate(a) if p != "sil"], [j for j, p in enumerate(b) if p != "sil"]
    edges = phone_edges(sing_ph)
    best, run = (0.0, None), []
    ops = edit_ops([a[i] for i in ia], [b[j] for j in ib])
    for op in ops + [None]:
        if op is not None and (op[0] == "equal" or max(op[2] - op[1], op[4] - op[3]) <= MAX_EDIT):
            run.append(op)
            continue
        if run and run[-1][2] > run[0][1] and run[-1][4] > run[0][3]:
            i0, i1, j0, j1 = ia[run[0][1]], ia[run[-1][2] - 1] + 1, ib[run[0][3]], ib[run[-1][4] - 1] + 1
            if edges[j1] - edges[j0] > best[0]:
                best = (edges[j1] - edges[j0], (i0, i1, j0, j1))
        run = []
    i0, i1, j0, j1 = best[1]
    # take in the pauses on either side, then cut at the outermost pauses both annotations share
    if i0 > 0 and j0 > 0 and a[i0 - 1] == b[j0 - 1] == "sil":
        i0, j0 = i0 - 1, j0 - 1
    if i1 < len(a) and j1 < len(b) and a[i1] == b[j1] == "sil":
        i1, j1 = i1 + 1, j1 + 1
    pauses = [(i0 + i, j0 + j) for tag, x0, x1, y0, _ in
              edit_ops(a[i0:i1], b[j0:j1]) if tag == "equal"
              for i, j in zip(range(x0, x1), range(y0, y0 + x1 - x0)) if a[i0 + i] == "sil"]
    if len(pauses) >= 2:
        return pauses[0][0], pauses[-1][0] + 1, pauses[0][1], pauses[-1][1] + 1
    return i0, i1, j0, j1


def crop(wav_path, phones, out):
    """Write the audio under `phones` to `out`; return the phones shifted to start at 0."""
    t0, t1 = phones[0]["start"], phones[-1]["end"]
    sf.write(out, load_numpy(wav_path, SR)[int(t0 * SR):int(t1 * SR)], SR)
    return [{**p, "start": p["start"] - t0, "end": p["end"] - t0} for p in phones]


def harvest(path):
    fcfg = EXTRACT_CFG["f0"]
    return extract_f0_harvest(path, fcfg["frame_period_ms"], fcfg["fmin"], fcfg["fmax"])


def conditions(manifest, out_dir, pool):
    """One dict per Exp 2 condition (inputs written to out_dir/<cond>/ for the cross pairs)."""
    rows = {(r.speaker_id, r.utt_id): r for r in manifest.itertuples()}
    test = [r for r in manifest.itertuples() if r.split in ("val", "test")]
    # median spoken F0 of every speaker involved, for the octave choice
    speakers = sorted({s for s, u in rows if any(u == t.utt_id for t in test)})
    reads = {s: next(r.read_wav for (s2, _), r in rows.items() if s2 == s) for s in speakers}
    f0s = dict(zip(speakers, pool.map(harvest, reads.values())))
    med = {s: np.median(f[f > 0]) for s, f in f0s.items()}

    conds = []
    for r in test:
        key = f"{r.speaker_id}_{r.utt_id}"
        read_ph, sing_ph = parse_annotation(r.read_txt), parse_annotation(r.sing_txt)
        for shift in SHIFTS:
            conds.append({"cond": f"{key}_own{shift:+d}", "variant": "transpose", "utt": r.utt_id, "speaker": r.speaker_id,
                          "melody_from": r.speaker_id, "shift": shift, "speech": r.read_wav, "read_ph": read_ph,
                          "melody": r.sing_wav, "sing_ph": sing_ph, "real": r.sing_wav if shift == 0 else None})
        for (o, u), m in rows.items():
            if u != r.utt_id or o == r.speaker_id:
                continue
            m_ph = parse_annotation(m.sing_txt)
            i0, i1, j0, j1 = clean_region(read_ph, m_ph)
            d = out_dir / f"{key}_from_{o}"
            d.mkdir(parents=True, exist_ok=True)
            conds.append({"cond": d.name, "variant": "cross", "utt": r.utt_id, "speaker": r.speaker_id, "melody_from": o,
                          "shift": 12 * int(np.round(np.log2(med[r.speaker_id] / med[o]))),
                          "speech": d / "speech.wav", "read_ph": crop(r.read_wav, read_ph[i0:i1], d / "speech.wav"),
                          "melody": d / "melody.wav", "sing_ph": crop(m.sing_wav, m_ph[j0:j1], d / "melody.wav"),
                          "real": None})
    return conds


def main(args=None):
    ap = argparse.ArgumentParser(description="Exp 2: different melody")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=8, help="Harvest processes for the evaluation")
    ap.add_argument("--only", nargs="+", help="condition names to run (default: all)")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.5)
    args = ap.parse_args(args)

    device = resolve_device(args.device)
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml"))
    cache = FeatureCache(ROOT / data_cfg["cache_root"])
    manifest = pd.read_csv(ROOT / data_cfg["cache_root"] / "split_manifest.csv", dtype={"utt_id": str})
    split = dict(zip(manifest.speaker_id + "_" + manifest.utt_id, manifest.split))
    out_dir = ROOT / "outputs" / "exp2"
    pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"))

    conds = conditions(manifest, out_dir, pool)
    if args.only:
        conds = [c for c in conds if c["cond"] in args.only]
    print(f"{len(conds)} conditions on {device}", flush=True)

    # ---- render (resumable: only timings with missing files are rendered)
    for c in conds:
        d = out_dir / c["cond"]
        missing = tuple(t for t in TIMINGS if not all((d / f"{s}_{t}.wav").exists() for s in SYSTEMS))
        if not missing:
            continue
        wavs = synthesize(str(c["speech"]), c["read_ph"], str(c["melody"]), c["sing_ph"], SYSTEMS, missing,
                          c["shift"], device)
        for (s, t), wav in wavs.items():
            write_wav(d / f"{s}_{t}.wav", wav)
        print(f"{c['cond']:22s} shift {c['shift']:+3d}  {len(wav) / SR:6.1f} s  {', '.join(missing)}", flush=True)

    # ---- evaluate
    ecapa = _ecapa()
    jobs = [(c, s, t, out_dir / c["cond"] / f"{s}_{t}.wav") for c in conds for s in SYSTEMS for t in TIMINGS]
    f0_out = list(pool.map(harvest, [str(j[3]) for j in jobs]))
    f0_mel = dict(zip([c["cond"] for c in conds], pool.map(harvest, [str(c["melody"]) for c in conds])))
    rows = []
    for (c, s, t, path), f0_g in zip(jobs, f0_out):
        spk = lambda who: cache.load("ecapa", split[f"{who}_{c['utt']}"], f"{who}_{c['utt']}")  # noqa: E731
        emb = extract_ecapa(path, ecapa)
        r = {k: c[k] for k in ("cond", "variant", "utt", "speaker", "melody_from", "shift")}
        r.update(system=s, timing=t, seconds=len(f0_g) * EXTRACT_CFG["f0"]["frame_period_ms"] / 1000,
                 **f0_metrics(f0_g, f0_mel[c["cond"]] * 2 ** (c["shift"] / 12)),
                 spk_cos=cosine(emb, spk(c["speaker"])),
                 leak_cos=cosine(emb, spk(c["melody_from"])) if c["variant"] == "cross" else np.nan,
                 base_cos=cosine(spk(c["speaker"]), spk(c["melody_from"])) if c["variant"] == "cross" else np.nan)
        if c["real"]:
            r["mcd_db"] = mcd(librosa.load(path, sr=SR)[0], librosa.load(c["real"], sr=SR)[0], SR)
        rows.append(r)
    pool.shutdown()

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.round(4).to_csv(ROOT / "results" / "exp2.csv", index=False)
    df["group"] = np.where(df.variant == "cross", "cross", "own" + df["shift"].map("{:+d}".format))
    cols = ["f0_rmse_cent", "f0_corr", "spk_cos", "leak_cos", "base_cos", "mcd_db"]
    summary = df.groupby(["group", "system", "timing"])[[k for k in cols if k in df]].mean().round(3)
    summary.to_csv(ROOT / "results" / "exp2_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
