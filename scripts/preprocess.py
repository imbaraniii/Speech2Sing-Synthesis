#!/usr/bin/env python3
"""Main preprocessing script: extract + cache all features for the corpus.

Usage:
    python scripts/preprocess.py                 # use config/*.yaml defaults
    python scripts/preprocess.py --force         # re-extract even if cached
    python scripts/preprocess.py --components hubert ecapa crepe mel notes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.nus_smc import NUS_SMC_Dataset  # noqa: E402
from data.splits import speaker_disjoint_split, list_speakers  # noqa: E402
from features.cache import FeatureCache  # noqa: E402


def load_configs():
    with open(ROOT / "config" / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "extract.yaml") as f:
        extract_cfg = yaml.safe_load(f)
    return data_cfg, extract_cfg


def build_split_map(data_cfg):
    strategy = data_cfg["splits"]["strategy"]
    speakers = list_speakers(data_cfg["data_root"])
    seed = data_cfg["splits"]["seed"]

    if strategy == "leave_one_song_out":
        # Special handling: all-speaker train/test, song held out per fold.
        # For pre-processing we simply assign all to 'train' (fold logic is
        # evaluated later during training).
        return {spk: "train" for spk in speakers}

    return speaker_disjoint_split(
        speakers,
        seed=seed,
        n_train=data_cfg["splits"]["train_speakers"],
        n_val=data_cfg["splits"]["val_speakers"],
    )


def main(args=None):
    parser = argparse.ArgumentParser(description="Extract all features")
    parser.add_argument("--force", action="store_true", help="re-extract even if cached")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["hubert", "ecapa", "crepe", "f0", "mel", "vmel", "notes", "aligned"],
        default=["hubert", "ecapa", "crepe", "f0", "mel", "notes"],
    )
    args = parser.parse_args(args)

    data_cfg, extract_cfg = load_configs()
    device = extract_cfg.get("device", "auto")

    data_root = Path(data_cfg["data_root"])
    split_map = build_split_map(data_cfg)
    dataset = NUS_SMC_Dataset(data_root, split_map, utterance_ids=None)
    cache = FeatureCache(data_cfg["cache_root"])

    manifest_path = Path(data_cfg["cache_root"]) / "split_manifest.csv"
    dataset.save_manifest(manifest_path)
    print(f"manifest -> {manifest_path} ({len(dataset)} items)")
    print("split distribution:", dataset.to_dataframe()["split"].value_counts().to_dict())

    from features.hubert import resolve_device

    device = resolve_device(extract_cfg.get("device", "auto"))
    # ECAPA (SpeechBrain) is kept on CPU: it is fast (~1.5s/utt) and avoids
    # MPS compatibility issues. HuBERT/CREPE use the resolved device.
    ecapa_device = "cpu"

    models = {}
    if "hubert" in args.components:
        from features.hubert import load_hubert

        print(f"Loading HuBERT ({extract_cfg['hubert']['model']})...")
        models["hubert"] = load_hubert(extract_cfg["hubert"]["model"], device)

    if "ecapa" in args.components:
        from features.ecapa import load_ecapa

        print(f"Loading ECAPA-TDNN ({extract_cfg['ecapa']['model']})...")
        models["ecapa"] = load_ecapa(
            extract_cfg["ecapa"]["model"],
            extract_cfg["ecapa"].get("savedir", "pretrained_models/"),
            ecapa_device,
        )

    print(f"device: hubert/crepe={device}, ecapa={ecapa_device}")
    count_done = {c: 0 for c in args.components}
    count_skipped = {c: 0 for c in args.components}

    # ---- Harvest F0 (song): slow per file, so run in parallel up front ----
    if "f0" in args.components:
        from concurrent.futures import ProcessPoolExecutor

        from features.world_f0 import extract_f0_harvest

        fcfg = extract_cfg["f0"]
        todo = [it for it in dataset
                if args.force or not cache.exists("f0", it["split"], f"{it['speaker_id']}_{it['utt_id']}")]
        count_skipped["f0"] = len(dataset) - len(todo)
        with ProcessPoolExecutor(fcfg.get("workers", 4)) as pool:
            futures = [pool.submit(extract_f0_harvest, it["sing_wav"], fcfg["frame_period_ms"],
                                   fcfg["fmin"], fcfg["fmax"]) for it in todo]
            for it, fut in tqdm(zip(todo, futures), total=len(todo), desc="harvest f0"):
                cache.save("f0", it["split"], f"{it['speaker_id']}_{it['utt_id']}", fut.result())
                count_done["f0"] += 1

    # ---- Vocoder mel (song): 128-bin NSF-HiFiGAN mel, the acoustic-model target ----
    if "vmel" in args.components:
        from features.audio import load_numpy
        from features.vocoder import Vocoder

        vocoder = Vocoder(device)
        for it in tqdm(dataset, desc="vocoder mel"):
            key = f"{it['speaker_id']}_{it['utt_id']}"
            if cache.exists("vmel", it["split"], key) and not args.force:
                count_skipped["vmel"] += 1
                continue
            cache.save("vmel", it["split"], key, vocoder.mel(load_numpy(it["sing_wav"], vocoder.sr)))
            count_done["vmel"] += 1
        del vocoder

    for it in tqdm(dataset, desc="utterances"):
        spk, utt, split = it["speaker_id"], it["utt_id"], it["split"]
        key = f"{spk}_{utt}"

        # ---- HuBERT (speech) -------------------------------------------------
        if "hubert" in args.components:
            if cache.exists("hubert", split, key) and not args.force:
                count_skipped["hubert"] += 1
            else:
                from features.hubert import extract_hubert

                feats = extract_hubert(
                    it["read_wav"],
                    models["hubert"],
                    layer=extract_cfg["hubert"]["layer"],
                    device=device,
                    sr=data_cfg["sample_rate"],
                )
                cache.save("hubert", split, key, feats)
                count_done["hubert"] += 1

        # ---- ECAPA (speech) --------------------------------------------------
        if "ecapa" in args.components:
            if cache.exists("ecapa", split, key) and not args.force:
                count_skipped["ecapa"] += 1
            else:
                from features.ecapa import extract_ecapa

                emb = extract_ecapa(it["read_wav"], models["ecapa"], ecapa_device)
                cache.save("ecapa", split, key, emb)
                count_done["ecapa"] += 1

        # ---- CREPE F0 (song) -------------------------------------------------
        f0 = conf = None
        if "crepe" in args.components:
            if cache.exists("crepe", split, key) and not args.force:
                count_skipped["crepe"] += 1
            else:
                from features.crepe_f0 import extract_f0

                f0, conf = extract_f0(
                    it["sing_wav"],
                    model_capacity=extract_cfg["crepe"]["model_capacity"],
                    hop_length=extract_cfg["crepe"]["hop_length"],
                    device=device,
                    sr=data_cfg["sample_rate"],
                    batch_size=extract_cfg["crepe"].get("batch_size", 512),
                )
                cache.save("crepe", split, key, f0)
                cache.save("crepe_conf", split, key, conf)
                count_done["crepe"] += 1

        # ---- Mel (song) ------------------------------------------------------
        if "mel" in args.components:
            if cache.exists("mel", split, key) and not args.force:
                count_skipped["mel"] += 1
            else:
                from features.mel import extract_mel

                mcfg = extract_cfg["mel"]
                mel = extract_mel(
                    it["sing_wav"],
                    sr=data_cfg["sample_rate"],
                    n_fft=mcfg["n_fft"],
                    hop_length=mcfg["hop_length"],
                    win_length=mcfg["win_length"],
                    n_mels=mcfg["n_mels"],
                    fmin=mcfg["fmin"],
                    fmax=mcfg.get("fmax"),
                    log_floor=mcfg.get("log_floor", 1e-5),
                )
                cache.save("mel", split, key, mel)
                count_done["mel"] += 1

        # ---- Notes (song) ----------------------------------------------------
        if "notes" in args.components:
            if cache.exists("notes", split, key) and not args.force:
                count_skipped["notes"] += 1
            else:
                from features.notes import segment_notes

                # melody F0 = Harvest (0 Hz = unvoiced), 10 ms frames
                ncfg = extract_cfg["notes"]
                notes = segment_notes(
                    cache.load("f0", split, key),
                    it["sing_phones"],
                    sr=1000,
                    hop_length=extract_cfg["f0"]["frame_period_ms"],
                    stability_semitones=ncfg["f0_stability_semitones"],
                    min_note_frames=ncfg["min_note_frames"],
                )
                cache.save("notes", split, key, np.array(notes, dtype=object))
                count_done["notes"] += 1

        # ---- Aligned (to target frame rate) ----------------------------------
        if "aligned" in args.components:
            from features.align import upsample_to_target

            target_rate = extract_cfg["frame_rates"]["target"]
            rates = {
                "hubert": extract_cfg["frame_rates"]["hubert"],
                "crepe": extract_cfg["frame_rates"]["crepe"],
                "f0": extract_cfg["frame_rates"]["f0"],
                "mel": extract_cfg["frame_rates"]["mel"],
            }
            tag = f"aligned_{target_rate}"
            if cache.exists(tag, split, key) and not args.force:
                count_skipped["aligned"] += 1
            else:
                bundle = {}
                for ft, rate in rates.items():
                    try:
                        data = cache.load(ft, split, key)
                        bundle[ft] = upsample_to_target(data, rate, target_rate)
                    except FileNotFoundError:
                        pass
                cache.save(tag, split, key, np.array(bundle, dtype=object))
                count_done["aligned"] += 1

    print("\n=== extraction summary ===")
    for c in args.components:
        print(f"{c:10s} done={count_done[c]:3d}  skipped(cached)={count_skipped[c]:3d}")

    print("\nCache contents:")
    for k, v in cache.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()