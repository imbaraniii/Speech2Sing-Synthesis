# Speech-to-Singing Synthesis with Melody Conditioning
## An Explanation from the Speech Processing Perspective

---

## 1. Feature Extraction (Speech Front-End)

### 1.1 HuBERT Content Encoding (What is being spoken?)
- **Subfield:** Self-supervised speech representation learning
- **What it does:** Encodes the spoken audio into frame-level features that capture *what was said* (phonetic/linguistic content) while suppressing *who said it*
- **Role in project:** The modern successor to MFCCs and PPGs — richer, context-aware representation of the lyrics' linguistic content

### 1.2 ECAPA-TDNN Speaker Embedding (Who is singing?)
- **Subfield:** Speaker verification / speaker recognition
- **What it does:** Compresses the speech into a single utterance-level vector (speaker embedding) capturing vocal tract characteristics, timbre, and identity
- **Role in project:** Provides the identity conditioning — "who is singing"

### 1.3 CREPE F0 Estimation
- **Subfield:** Pitch tracking / fundamental frequency extraction (prosodic analysis)
- **What it does:** Produces a frame-level F0 contour from the melody audio, including voiced/unvoiced detection
- **Role in project:** Represents the melody's pitch trajectory — one of the three prosodic dimensions (pitch, duration, loudness)

### 1.4 Mel Spectrogram
- **Subfield:** Acoustic front-end / spectral analysis
- **What it does:** STFT → mel filterbank → compression, producing a perceptually-scaled spectral representation
- **Role in project:** The standard acoustic target used to train the synthesis model

---

## 2. Speech-Singing Alignment (Prosody + Time Alignment)

This is the most speech-processing-intensive stage of the project.

### 2.1 Phone-Level Annotation Parsing
- **Subfield:** Forced alignment / speech annotation
- **What it does:** Parses the corpus's time-aligned phone transcriptions to obtain phone boundaries in both the spoken and sung recordings
- **Role in project:** Ground truth for measuring and modeling the speech↔singing temporal difference

### 2.2 Duration Prediction
- **Subfield:** Duration modeling (a core prosody subfield)
- **What it does:** Estimates how long each spoken phoneme must be stretched to fit the melody's note durations — analogous to the duration predictor in TTS (e.g., FastSpeech), but conditioned on musical note lengths
- **Role in project:** Converts speech-time phonemes into singing-time durations

### 2.3 Temporal Expansion (Time-Scale Modification)
- **Subfield:** Time-scale modification (e.g., PSOLA / WSOLA techniques)
- **What it does:** Expands the frame-level content representation onto the singing timeline
- **Role in project:** In the classical WORLD baseline this is done directly on the waveform; in the neural pipeline it is done in the feature domain via the predicted durations

---

## 3. Singing Acoustics Generation (Speech Synthesis)

### 3.1 Conformer Acoustic Model
- **Subfield:** Speech/singing synthesis (acoustic modeling)
- **What it does:** Converts the aligned content + speaker embedding + F0/note features into a mel spectrogram — the acoustic-model stage of modern TTS pipelines (FastSpeech/VITS)
- **Role in project:** Trained with L1 mel loss + F0 loss (prosody accuracy) + duration loss

### 3.2 HiFi-GAN Vocoder
- **Subfield:** Neural vocoding (waveform generation)
- **What it does:** Converts spectral features back into a raw waveform
- **Role in project:** The final waveform synthesis stage (successor to classical vocoders like WORLD/STRAIGHT)

---

## 4. Classical Baseline System (Source-Filter Speech Processing)

- **Subfield:** Source-filter modeling (WORLD vocoder)
- **What it does:** Decomposes speech into a spectral envelope (filter = timbre) and F0 (source = excitation), then replaces the F0 with the melody's F0 and stretches durations
- **Role in project:** A textbook exercise in prosody manipulation — pitch modification + duration modification on the source component while preserving the filter. Provides the lower-bound baseline the neural system is compared against

---

## 5. Evaluation (Speech Processing Metrics)

| Metric | Subfield | What it measures |
|--------|----------|------------------|
| **MCD** (Mel-Cepstral Distortion) | Speech synthesis evaluation | Spectral distance between generated and reference singing |
| **F0 RMSE / F0 Correlation** | Prosody evaluation | Pitch accuracy of the generated singing |
| **Speaker Cosine Similarity** | Speaker verification | How well the generated voice matches the target speaker's embedding |

---

## 6. Summary

The project is a cross-section of core speech processing techniques unified into a single pipeline:

```
Feature Extraction (HuBERT, ECAPA, CREPE, Mel)
        +
Speech-Singing Alignment (Phone alignment, Duration modeling, Time-scale modification)
        +
Singing Synthesis (Conformer acoustic model, HiFi-GAN vocoder)
        +
Classical Baseline (WORLD source-filter manipulation)
        +
Evaluation (MCD, F0 metrics, speaker verification similarity)
```

- **Speech recognition / representation learning** → HuBERT content features
- **Speaker recognition** → ECAPA-TDNN identity embedding
- **Pitch / prosody analysis** → CREPE F0 contour
- **Duration modeling** → alignment of speech to musical timing
- **Speech synthesis** → Conformer acoustic model + HiFi-GAN vocoder
- **Source-filter processing** → WORLD baseline
- **Synthesis + speaker evaluation** → MCD, F0 RMSE, speaker cosine similarity
