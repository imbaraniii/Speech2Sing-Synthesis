# Speech-to-Singing Synthesis with Melody Conditioning
## Technical Approach Document

---

## 1. Problem Overview

**Goal:** Generate singing audio from a target speaker's spoken lyrics, following a reference melody.

**Inputs at inference:**
1. Spoken audio of target speaker reading lyrics
2. Reference melody (audio or MIDI)

**Output:** Singing waveform in target speaker's voice, following the reference melody.

---

## 2. Dataset: NUS SMC Corpus (nus-smc-corpus_48)

| Property | Value |
|----------|-------|
| Speakers | 12 (6 male, 6 female) |
| Songs | 20 unique English songs |
| Recordings | 48 paired (speech + singing) |
| Duration | ~169 minutes total |
| Sample rate | 44.1 kHz |
| Annotations | Phone-level time-aligned `.txt` for **both** `read/` and `sing/` |

**Directory structure per speaker:**
```
SPEAKER_ID/
├── read/
│   ├── 05.wav, 05.txt
│   ├── 08.wav, 08.txt
│   ├── 11.wav, 11.txt
│   └── 15.wav, 15.txt
└── sing/
    ├── 05.wav, 05.txt
    ├── 08.wav, 08.txt
    ├── 11.wav, 11.txt
    └── 15.wav, 15.txt
```

Each numbered pair (05, 08, 11, 15) = same lyrics, same melody, same speaker.

---

## 3. Core Insight: Melody as Training Supervision

The dataset provides **paired (speech, singing) with identical lyrics and melody**. During training:

- **Melody features are extracted from the ground-truth singing audio** (`sing/` folder)
- The model learns: `content + speaker + melody_features → singing`
- At inference, melody features come from **any reference** (different song, MIDI, humming)

This is supervised learning where the melody condition is derived from the target domain itself.

---

## 4. Complete Pipeline

### 4.1 Offline Feature Extraction (once, before training)

| Source | Tool | Output | Shape |
|--------|------|--------|-------|
| Speech (`read/`) | HuBERT-base (pretrained) / PPG | Content features | `(T, 768)` or `(T, 256)` |
| Speech (`read/`) | ECAPA-TDNN (pretrained) | Speaker embedding | `(192,)` per utterance |
| Singing (`sing/`) | CREPE + phone alignments | F0 contour | `(T,)` |
| Singing (`sing/`) | Phone annotations → note segmentation | Note onsets, durations, MIDI pitches | Sequence |
| Singing (`sing/`) | STFT (n_fft=1024, hop=256) | Mel spectrogram (target) | `(T, 80)` |

**Phone annotations usage:** The `.txt` files in both `read/` and `sing/` give phone-level timestamps. These enable:
- Forced alignment for HuBERT feature extraction
- Mapping speech phones → sung phones for duration modeling
- Note segmentation from sung F0 + phone boundaries

### 4.2 Alignment Module (Duration Predictor)

**Problem:** Speech and singing have different temporal structures. Speech is faster, syllable-timed; singing stretches vowels to note durations.

**Solution:** Train a duration predictor that maps:
```
Input:  Content features + Melody note durations
Output: Predicted phone durations in singing time
```

Architecture: Small Transformer/Conformer (2-3 layers) or FastSpeech-style duration predictor.
Training target: Ground-truth phone durations derived from `sing/` phone annotations.

### 4.3 Acoustic Model (Main Trainable Component)

```
┌─────────────────────────────────────────────────────────────┐
│                    ACOUSTIC MODEL                           │
├─────────────────────────────────────────────────────────────┤
│  Inputs (concatenated per frame):                           │
│    • Content features (HuBERT/PPG)          [T, D_c]        │
│    • Speaker embedding (broadcast)          [T, D_s]        │
│    • Melody F0 contour (interpolated)       [T, 1]          │
│    • Note-level embedding (pitch, position) [T, D_n]        │
│                                                             │
│  Encoder: 4-6 layer Transformer/Conformer                   │
│  Decoder: 4-6 layer Transformer (autoregressive or          │
│           non-autoregressive with duration prediction)      │
│                                                             │
│  Output: Mel spectrogram [T, 80]                            │
└─────────────────────────────────────────────────────────────┘
```

**Losses:**
- L1 Mel-spectrogram loss (primary)
- F0 MSE loss (auxiliary, on predicted vs. ground-truth F0)
- Optional: Adversarial loss (discriminator on mel)
- Optional: Duration prediction loss (if joint training)

### 4.4 Vocoder (Pretrained, Frozen)

- **HiFi-GAN** (pretrained on LibriTTS / VCTK / singing data)
- Input: Generated mel spectrogram → Output: 44.1 kHz waveform
- No fine-tuning needed; works out-of-box for singing

---

## 5. Training Procedure

### Data Split
- **Train:** 10 speakers (40 utterances)
- **Validation:** 1 speaker (4 utterances)
- **Test:** 1 speaker (4 utterances) — unseen speaker evaluation

### Training Steps

```python
# Per batch:
for speech_wav, sing_wav, speech_phones, sing_phones in dataloader:
    # 1. Extract features (precomputed, loaded from disk)
    content = hubert(speech_wav)                    # [T, 768]
    spk_emb = ecapa(speech_wav)                     # [192]
    melody_f0 = crepe(sing_wav)                     # [T_mel,]
    note_seq = extract_notes(melody_f0, sing_phones) # [(pitch, dur), ...]
    target_mel = melspectrogram(sing_wav)           # [T_mel, 80]

    # 2. Align content to melody timeline using duration predictor
    phone_durs = duration_predictor(content, note_seq)  # [N_phones]
    content_aligned = stretch(content, phone_durs)      # [T_mel, 768]

    # 3. Broadcast speaker embedding
    spk_emb_broadcast = spk_emb.unsqueeze(0).repeat(T_mel, 1)  # [T_mel, 192]

    # 4. Note embeddings per frame
    note_emb = note_encoder(note_seq, T_mel)         # [T_mel, D_n]

    # 5. Forward pass
    pred_mel = acoustic_model(
        content_aligned, spk_emb_broadcast, 
        melody_f0, note_emb
    )  # [T_mel, 80]

    # 6. Loss
    loss = L1(pred_mel, target_mel) + \
           lambda_f0 * MSE(pred_f0, melody_f0) + \
           lambda_dur * duration_loss
    loss.backward()
```

### Hyperparameters (suggested)
| Parameter | Value |
|-----------|-------|
| Batch size | 8-16 (depends on GPU memory) |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-6 |
| Scheduler | Cosine annealing with warmup (4000 steps) |
| Epochs | 200-500 (early stopping on val loss) |
| Mixed precision | Yes (FP16) |
| Gradient clipping | 1.0 |

---

## 6. Inference Pipeline

```
┌────────────────────────────────────────────────────────────────┐
                        INFERENCE                                 
├────────────────────────────────────────────────────────────────┤
                                                                 │
  USER INPUTS:                                                   │
    ├─ speech.wav    (target speaker reading lyrics)            │
    └─ melody.wav    (reference melody: any song, humming, MIDI)│
                                                                 │
  STEP 1: Feature Extraction                                     │
    ├─ content = HuBERT(speech.wav)                              │
    ├─ spk_emb = ECAPA(speech.wav)                               │
    └─ melody_f0, note_seq = CREPE(melody.wav) + note_segment    │
                                                                 │
  STEP 2: Alignment                                              │
    ├─ phone_durs = duration_predictor(content, note_seq)       │
    └─ content_aligned = stretch(content, phone_durs)            │
                                                                 │
  STEP 3: Acoustic Model                                         │
    └─ pred_mel = acoustic_model(                                │
           content_aligned, spk_emb, melody_f0, note_emb)        │
                                                                 │
  STEP 4: Vocoder                                                │
    └─ waveform = HiFi-GAN(pred_mel)                             │
                                                                 │
  OUTPUT: singing.wav                                            │
                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Key point:** The melody reference can be:
- A different song's audio (melody extracted automatically)
- A MIDI file (converted to F0 contour + note sequence)
- User humming/singing the target melody
- Symbolic score (MusicXML → note sequence → F0)

---

## 7. Evaluation Protocol

### Objective Metrics (on test set)
| Metric | Description | Target |
|--------|-------------|--------|
| **MCD** (Mel-Cepstral Distortion) | Spectral similarity to ground-truth singing | < 5.0 dB |
| **F0 RMSE** (Hz) | Pitch accuracy vs. ground-truth F0 | < 30 Hz |
| **F0 CORR** | Pearson correlation of F0 contours | > 0.85 |
| **Speaker Cosine Sim** | ECAPA embedding: generated vs. target speech | > 0.75 |
| **Phone Accuracy** | Forced alignment on generated → phone error rate | < 15% |

### Subjective Evaluation (MUSHRA-style)
- **Naturalness** (1-5): How natural does the singing sound?
- **Speaker Similarity** (1-5): Does it sound like the target speaker?
- **Melody Adherence** (1-5): Does it follow the reference melody?
- **Overall Quality** (1-5): Holistic rating

Minimum 10 listeners, 20 samples per system.

---

## 8. Baseline Comparisons

| System | Description | Expected MCD |
|--------|-------------|--------------|
| **WORLD F0-transfer** | Pitch-shift speech F0 to melody F0, stretch durations, WORLD resynthesis | ~7-8 dB |
| **Proposed (ours)** | Neural acoustic model + pretrained components | ~4-5 dB |
| **Upper bound** | Ground-truth singing (copy-synthesis via vocoder) | ~1-2 dB |

The WORLD baseline is implemented first as a sanity check and lower bound.

---

## 9. Implementation Checklist

### Phase 1: Data Preparation (Week 1-2)
- [ ] Parse all `.txt` phone annotations → interval tiers
- [ ] Verify `read/` and `sing/` phone alignment per utterance
- [ ] Extract HuBERT features for all speech files (cache to disk)
- [ ] Extract ECAPA embeddings for all speech files
- [ ] Extract CREPE F0 for all singing files
- [ ] Segment notes from F0 + phone annotations (heuristic: sustained F0 + vowel phones)
- [ ] Compute mel spectrograms for all singing files
- [ ] Create train/val/test splits (speaker-disjoint)

### Phase 2: Duration Predictor (Week 2-3)
- [ ] Implement duration predictor (Transformer, 2 layers)
- [ ] Train on ground-truth phone durations from `sing/` annotations
- [ ] Evaluate duration MAE on validation set

### Phase 3: Acoustic Model (Week 3-5)
- [ ] Implement acoustic model (Conformer encoder-decoder)
- [ ] Train with L1 mel loss + F0 loss
- [ ] Monitor validation MCD (via WORLD resynthesis for speed)
- [ ] Add adversarial loss if needed

### Phase 4: Vocoder & Full Pipeline (Week 5-6)
- [ ] Integrate pretrained HiFi-GAN
- [ ] End-to-end inference pipeline
- [ ] Generate test set samples

### Phase 5: Evaluation (Week 6-7)
- [ ] Compute objective metrics
- [ ] Conduct subjective listening test
- [ ] Prepare results, ablation studies

---

## 10. Key References

1. **SpeaKing** (NUS, 2023) — Speech-to-singing with phoneme posteriorgram + F0
2. **VISinger2** (2022) — End-to-end singing synthesis with variational inference
3. **DiffSinger** (2022) — Diffusion-based singing synthesis
4. **Sinsy** (2013) — HMM-based singing voice synthesis (baseline)
5. **FastSpeech 2** (2020) — Duration predictor architecture
6. **HiFi-GAN** (2020) — Vocoder
7. **HuBERT** (2021) / **WavLM** (2022) — Content encoders
8. **ECAPA-TDNN** (2020) — Speaker embeddings
9. **CREPE** (2018) — F0 extraction

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Small dataset (48 pairs) | Use pretrained encoders; only train acoustic model + duration predictor |
| Alignment errors | Ground-truth phone annotations provide supervision; use CTC loss as auxiliary |
| Speaker leakage in content features | HuBERT/PPG is speaker-agnostic; verify with speaker classification probe |
| Melody generalization | Augment training with pitch-shifted / time-stretched melody features |
| Vocoder artifacts on singing | Use singing-pretrained HiFi-GAN (e.g., from OpenCpop/M4Singer) |

---

## 12. Expected Deliverables

1. **Trained models:** Duration predictor, acoustic model
2. **Inference script:** `python infer.py --speech speech.wav --melody melody.wav --out out.wav`
3. **Evaluation results:** Objective metrics table, subjective test results
4. **Demo samples:** 10-15 generated singing examples covering different speakers/melodies
5. **Report:** 6-8 page technical report with methodology, results, analysis

---

*Document version: 1.0*  
*Based on NUS SMC Corpus (nus-smc-corpus_48)*