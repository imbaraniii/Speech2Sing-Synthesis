# Speaker-Preserving Speech-to-Singing Synthesis

Turns a person's speech into their singing. They read the lyrics, a reference singing gives the
melody and timing, and the system sings the lyrics in the speaker's voice.

- **Data:** NUS SMC Corpus (NUS-48E). 12 speakers, 48 paired spoken (`read/`) and sung (`sing/`)
  songs, all with phone-level annotations.
- **Splits:** speaker-disjoint. 10 speakers train, JLEE is validation and VKOW is test, so
  neither is seen in training.
- **Hybrid model:**
  1. WORLD converts the speech: it stretches every phone to its sung duration and replaces the
     pitch with the melody's F0.
  2. A 9.5M-parameter Conformer refines that WORLD mel. It is conditioned on HuBERT content, an
     ECAPA speaker embedding and the melody.
  3. The pretrained NSF-HiFiGAN turns the mel into audio.
- **Timing:** without reference timing, a 393k-parameter Transformer predicts every phone's sung
  duration.
- **Own voices:** `script.py` runs the system on new recordings, using the Montreal Forced Aligner
  for the phone annotations.

**Documents:**
- `report/A09.pdf`: the IEEE paper, built from `report/ieee_report.tex`.
- `PROJECT_REPORT.md`: the full write-up, covering background, design, every experiment and viva notes.

## Results (test speaker VKOW, unseen)

| System | MCD (dB) ↓ | F0 RMSE (Hz) ↓ | F0 corr ↑ | Speaker sim ↑ |
|---|---|---|---|---|
| WORLD conversion | 4.32 | 17.6 | 0.891 | 0.657 |
| Pure Conformer | **3.73** | 29.2 | 0.739 | 0.000 |
| **Hybrid Conformer** | 4.12 | **13.3** | **0.931** | 0.354 |
| Real singing | 0 | 0 | 1.000 | 0.436 |

These use the annotated sung durations. The full tables, with predicted durations, validation,
ablations, GAN, regularisation, Exp 2, post-filter and our own voices, are in `PROJECT_REPORT.md` §10.

Speaker similarity is the ECAPA cosine with the speaker's *spoken* recording. Speech-like output
(WORLD) therefore scores above real singing.

## Setup

```bash
python3 -m venv env && source env/bin/activate     # tested with Python 3.13
pip install -r requirements.txt
```

> `requirements.txt` pins `setuptools<81` because pyworld 0.3.5 still imports `pkg_resources`. The PyPI
> `crepe` package is broken on modern Python, so CREPE comes from `torchcrepe`.

**Vocoder.** Training and inference need OpenVPI's NSF-HiFiGAN and DiffSinger's code for it in `vocoder/`:

```bash
mkdir -p vocoder && cd vocoder
git clone --depth 1 https://github.com/openvpi/DiffSinger.git
curl -LO https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-44.1k-hop512-128bin-2024.02/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip
unzip nsf_hifigan_44.1k_hop512_128bin_2024.02.zip && cd ..
```

> The vocoder weights are CC BY-NC-SA 4.0, research use only. The corpus has a research-only licence too.

**Checkpoints.** `checkpoints/{hybrid,acoustic,duration}.pt` are not in git. Train them with the steps
below on a CUDA GPU, or copy them from someone who has them.

**Montreal Forced Aligner** (only for `script.py`). MFA breaks on paths with spaces, so it lives outside the project:

```bash
micromamba create -p ~/mfa-env -c conda-forge montreal-forced-aligner   # or: conda create -p ~/mfa-env ...
mkdir -p ~/mfa-env/models   # put english_us_arpa.dict and english_us_arpa_acoustic.zip here
```

Both files come from the `english_us_arpa` releases at
<https://github.com/MontrealCorpusTools/mfa-models/releases>.

## Your own voice (`script.py`)

```bash
python3 script.py --speech my_speech.mp3 --melody melody.mp3 --lyrics lyrics.txt \
                  --out outputs/custom/my_song.wav [--transpose -12] [--postfilter none]
```

**Inputs:**
- `--speech`: you reading the lyrics.
- `--melody`: an a cappella singing of the same lyrics. It must be vocals only.
- `--lyrics`: the English text.

**What the script does:**
1. Checks every lyric word against the MFA dictionary, and stops with a list of any words it doesn't know.
2. Aligns both recordings with MFA.
3. Writes the alignments as NUS-48E phone files next to `--out`: `<stem>_speech.txt` and `<stem>_melody.txt`.
4. Runs `infer.py` on them, with `--postfilter sharpen_gv` by default.

Any other `infer.py` option is passed through. If the melody sits outside your range, shift it by an
octave with `--transpose -12` or `--transpose 12`.

## Inference with phone annotations (`infer.py`)

```bash
python infer.py --speech read/05.wav --speech-phones read/05.txt \
                --melody sing/05.wav --melody-phones sing/05.txt --out out.wav \
                [--model hybrid|acoustic|world] [--durations reference|phrase|predicted] \
                [--transpose -12] [--postfilter none|sharpen|sharpen_gv]
```

It recomputes every feature from the audio, and it runs on a Mac CPU.

**Annotations:** the two annotations must describe the same lyrics. Another singer's rendition of
the song works too: phones are aligned by minimum edit distance, and at least 70% must match.

**Melody:** the reference singing gives the melody, and `--transpose` shifts it in semitones.

**Timing** (`--durations`):
- `reference`: every phone's timing comes from the reference singing.
- `phrase`: the duration model times every phone, and only the melody's rests (pauses of 0.5 s or
  more) stay in place. This is the best fully automatic option.
- `predicted`: the duration model's timing, stretched to the melody's length.

**Post-filter** (`--postfilter`): the neural mel is over-smoothed in time, which makes it sound muffled.
- `sharpen` adds an unsharp mask along time. It is nearly free: test MCD rises by 0.003 dB and
  speaker similarity rises by 0.03.
- `sharpen_gv` also stretches each band's variance towards that of real singing. It sounds
  clearest, but MCD is about 0.1 dB worse.

> The melody F0 comes from WORLD Harvest (`f0/`), not CREPE. torchcrepe-tiny gives near-zero
> confidence or octave errors on about half the singers (for example MPUR and JTAN). The CREPE cache
> is kept for reference. It never outputs 0 Hz, so mask it with `crepe_conf` if you use it.

## Reproducing the experiments

Configuration:
- `config/data.yaml`: corpus path, sample rate, split strategy, cache path.
- `config/extract.yaml`: model checkpoints, frame rates, mel and note parameters, and the device.
  `auto` picks `cuda`, then `mps`, then `cpu`.

Song IDs are discovered per speaker, because each speaker sings a different set of 4 songs (for
example ADIZ sings 01/09/13/18 and JLEE sings 05/08/11/15).

Steps 10 to 16 need a CUDA GPU; choose one with `CUDA_VISIBLE_DEVICES=<id>`. By default,
`train_acoustic.py` uses at most half of the GPU's memory.

```bash
# 1. Extract and cache all features (48 songs)
python scripts/preprocess.py

# 2. Verify the split (no speaker leakage, every file covered)
python scripts/verify_splits.py

# 3. Sanity-check the features (shapes, ECAPA similarity, F0 overlay, Griffin-Lim audibility) -> figures/
python scripts/sanity_check.py

# 4. Data study (durations, phone inventory, F0 ranges) -> figures/ and a CSV
python scripts/data_study.py

# 5. Optional: interactive exploration
jupyter lab notebooks/01_data_study.ipynb

# 6. WORLD baseline (annotated sung durations and sung F0) -> outputs/world/, outputs/world_copysyn/
python baselines/world_baseline.py

# 7. Objective metrics (MCD, F0 RMSE and correlation, speaker cosine) -> results/eval*.csv
python scripts/evaluate.py

# 8. Duration predictor and ratio baselines -> results/duration*.csv, checkpoints/duration.pt
python scripts/train_duration.py

# 9. WORLD with predicted durations (needs step 8) -> outputs/world_pred/, then score it
python baselines/world_baseline.py --durations predicted
python scripts/evaluate.py --systems world_pred world world_copysyn gt

# 10. Pure Conformer: vocoder-matched mels -> Conformer acoustic model -> NSF-HiFiGAN
python scripts/preprocess.py --components vmel
python scripts/train_acoustic.py               # -> outputs/acoustic_{oracle,pred}/, nsf_copysyn/
python scripts/evaluate.py --systems acoustic_oracle acoustic_pred nsf_copysyn world world_pred world_copysyn gt

# 11. Hybrid: the Conformer refines the WORLD conversion, keeping the speaker's timbre and the pitch
python scripts/preprocess.py --components wmel
python scripts/train_acoustic.py --world-input # -> outputs/hybrid_{oracle,pred}/
python scripts/evaluate.py --systems hybrid_oracle hybrid_pred --csv results/eval_hybrid.csv

# 12. Ablations: drop the melody or the speaker input, for the pure and hybrid models
python scripts/train_acoustic.py --no-pitch                  # -> acoustic_nopitch_*
python scripts/train_acoustic.py --world-input --no-speaker  # -> hybrid_nospk_*
python scripts/evaluate.py --systems acoustic_nopitch_oracle acoustic_nospk_oracle \
    hybrid_nopitch_oracle hybrid_nospk_oracle --csv results/eval_ablation.csv

# 13. Optional GAN fine-tuning of the hybrid, gated on validation -> results/gan_gate.csv
#     It saves checkpoints/hybrid_gan.pt only if a snapshot beats the hybrid. None did.
python scripts/finetune_gan.py

# 14. Listening excerpts of every val/test song for all systems -> outputs/demo/
python scripts/make_demos.py

# 15. Exp 2: the lyrics sung to another singer's melody, and to the song's own melody shifted by
#     -3/0/+3 semitones. WORLD, pure and hybrid, each with reference/phrase/predicted timing -> results/exp2*.csv
python scripts/exp2_melody.py

# 16. Regularisation of the hybrid: EMA, ECAPA input noise, lower learning rate. None passed the
#     validation gate. --tag names the run, so checkpoints/hybrid.pt is kept.
python scripts/train_acoustic.py --world-input --tag ema --ema 0.999
python scripts/train_acoustic.py --world-input --tag ema_spk05 --ema 0.999 --spk-noise 0.5
python scripts/train_acoustic.py --world-input --tag ema_spk10 --ema 0.999 --spk-noise 1.0
python scripts/train_acoustic.py --world-input --tag lr1 --lr 1e-4 --eval-every 500 --patience 12
python scripts/evaluate.py --systems hybrid_lr1_oracle hybrid_lr1_pred --csv results/eval_hybrid_lr1.csv   # same for each tag
```

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
├── vmel/<split>/SPK_UTT.npy    # (T, 128) NSF-HiFiGAN log-mel @ 86 Hz (acoustic target)
├── wmel/<split>/SPK_UTT.npy    # (T, 128) same mel of the WORLD conversion (hybrid input)
└── notes/<split>/SPK_UTT.npy   # note segments from f0/ [{midi, onset_time, ...}]
```

Frame rates are listed under `frame_rates` in `config/extract.yaml`.

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
| `features/vocoder.py` | pretrained NSF-HiFiGAN: mel extraction + vocoding |
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
| `scripts/train_duration.py` | duration predictor + ratio baselines |
| `scripts/train_acoustic.py` | Conformer acoustic model (pure / hybrid / ablations / regularisation) |
| `scripts/finetune_gan.py` | validation-gated adversarial fine-tuning of the hybrid |
| `scripts/make_demos.py` | aligned listening excerpts → `outputs/demo/` |
| `scripts/exp2_melody.py` | Exp 2: cross-singer melodies and transpositions |
| `infer.py` | end-to-end inference from raw audio + phone annotations, with the post-filter |
| `script.py` | own recordings: MFA alignment + `infer.py` in one command |
| `report/` | IEEE paper (`ieee_report.tex`, `A09.pdf`) |
| `PROJECT_REPORT.md` | full project report |
