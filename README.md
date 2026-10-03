# Speech-to-Singing Synthesis — Data Preprocessing & Feature Extraction

Preprocessing and offline feature-extraction pipeline for the
**NUS SMC Corpus (nus-smc-corpus_48)**: 12 speakers, 48 paired
speech (`read/`) + singing (`sing/`) utterances with phone-level
time-aligned annotations.

## Setup

```bash
pip install -r requirements.txt
```

> Note: the PyPI `crepe` package is broken on modern Python; this project
> uses `torchcrepe` (pure-PyTorch CREPE). `torchaudio` >= 2.11 additionally
> requires `torchcodec`.

## Config

- `config/data.yaml` — corpus path, sample rate, split strategy, cache path.
- `config/extract.yaml` — model checkpoints, frame rates, mel/note params,
  device (`auto` picks `cuda` > `mps` > `cpu`).

Important: utterance/song IDs are **auto-discovered per speaker** (each
speaker sings a different set of 4 songs, e.g. ADIZ: 01/09/13/18,
JLEE: 05/08/11/15).

## Run

```bash
# 1. Extract + cache all features (48 utterances)
python scripts/preprocess.py

# 2. Verify split integrity (no speaker leakage, file coverage)
python scripts/verify_splits.py

# 3. Sanity-check features (shapes, ECAPA similarity, F0 overlay,
#    Griffin-Lim audibility) -> figures/
python scripts/sanity_check.py

# 4. Data study (durations, phone inventory, F0 ranges) -> figures/ + CSV
python scripts/data_study.py

# 5. Optional: interactive exploration
jupyter lab notebooks/01_data_study.ipynb

# 6. WORLD baseline (oracle durations + sung F0) -> outputs/world/, outputs/world_copysyn/
python baselines/world_baseline.py

# 7. Objective metrics (MCD, F0 RMSE/CORR, speaker cosine) -> results/eval*.csv
python scripts/evaluate.py

# 8. Phase 2: duration predictor + baselines -> results/duration*.csv, checkpoints/duration.pt
python scripts/train_duration.py

# 9. WORLD with *predicted* durations (needs step 8) -> outputs/world_pred/, then score it
python baselines/world_baseline.py --durations predicted
python scripts/evaluate.py --systems world_pred world world_copysyn gt
```

> Melody F0 comes from WORLD Harvest (`f0/`), not CREPE: torchcrepe-tiny gives
> near-zero confidence / octave errors on about half the singers (e.g. MPUR,
> JTAN). The CREPE cache is kept for reference; it never outputs 0 Hz, so mask
> it with `crepe_conf` if you use it.

## Feature cache layout

```
features/cache/
├── split_manifest.csv          # speaker, utt, split, wav/txt paths, phone counts
├── hubert/<split>/SPK_UTT.npy  # (T, 768) content features @ 50 Hz
├── ecapa/<split>/SPK_UTT.npy   # (192,) speaker embedding
├── crepe/<split>/SPK_UTT.npy   # (T,) F0 Hz @ 100 Hz
├── crepe_conf/<split>/...      # (T,) periodicity confidence
├── f0/<split>/SPK_UTT.npy      # (T,) Harvest melody F0 Hz @ 100 Hz, 0 = unvoiced
├── mel/<split>/SPK_UTT.npy     # (T, 80) log-mel @ 172 Hz
└── notes/<split>/SPK_UTT.npy   # note segments from f0/ [{midi, onset_time, ...}]
```

Frame rates are documented in `config/extract.yaml` under `frame_rates`
(all feature streams will be interpolated to `target` before model training).

## Module map

| File | Purpose |
|------|---------|
| `data/annotations.py` | parse `.txt` phone annotations |
| `data/splits.py` | speaker-disjoint train/val/test split |
| `data/nus_smc.py` | dataset index + split manifest |
| `features/audio.py` | wav load/resample helpers |
| `features/hubert.py` | HuBERT content features |
| `features/ecapa.py` | ECAPA-TDNN speaker embeddings |
| `features/crepe_f0.py` | CREPE F0 (torchcrepe), reference only |
| `features/world_f0.py` | Harvest melody F0 (used downstream) |
| `features/mel.py` | log-mel spectrograms |
| `features/notes.py` | F0 → MIDI note segmentation |
| `features/align.py` | frame-rate interpolation |
| `features/mel_invert.py` | Griffin-Lim inversion (audibility check) |
| `features/cache.py` | `.npy` feature cache |
| `scripts/preprocess.py` | run everything |
| `scripts/verify_splits.py` | split integrity check |
| `scripts/sanity_check.py` | feature sanity checks + plots |
| `scripts/data_study.py` | corpus statistics + plots |
| `baselines/world_baseline.py` | WORLD speech-to-singing baseline + copy-synthesis |
| `scripts/evaluate.py` | objective metrics for generated singing |
| `scripts/train_duration.py` | Phase 2 duration predictor + ratio baselines |
