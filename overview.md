# Speaker-Preserving Speech-to-Singing Synthesis with Melody Conditioning

## 1. Project Definition

### Title

**Speaker-Preserving Speech-to-Singing Synthesis with Melody
Conditioning**

### Core idea

The system takes:

1.  **Spoken lyrics recorded by a target speaker**
2.  **A reference melody**

and generates:

> **The target speaker singing the spoken lyrics while following the
> reference melody.**

The central separation is:

-   **Speech → linguistic content + speaker identity**
-   **Melody → pitch + musical timing**
-   **Synthesis model → singing acoustic representation**
-   **HiFi-GAN → final waveform**

------------------------------------------------------------------------

## 2. Problem Statement

Most singing voice conversion and singing synthesis systems either
require a singing recording from the target speaker or assume that the
target singing voice is already represented in the training data.

This project investigates:

> **Can a singing performance be generated from a target speaker's
> spoken lyrics while preserving the speaker's identity and
> independently controlling the melody?**

### Example

**Input 1 --- Spoken lyrics**

A target speaker reads:

> "Happy birthday to you"

**Input 2 --- Reference melody**

A melody provides:

-   pitch contour
-   note sequence
-   note duration
-   timing/rhythm

**Output**

The target speaker singing:

> "Happy birthday to you"

according to the reference melody.

The melody controls **how the lyrics are sung**; it does not provide the
target speaker's identity or linguistic content.

------------------------------------------------------------------------

## 3. Primary and Extended Objectives

### Primary objective

Develop a controlled speech-to-singing system that:

-   accepts spoken lyrics from a target speaker,
-   extracts linguistic content,
-   extracts speaker identity,
-   extracts melody information,
-   aligns spoken content with singing-time durations,
-   generates singing while preserving speaker identity,
-   follows the reference melody.

### Extended objective

Investigate cross-melody transfer.

Example:

``` text
Spoken lyrics:
"Twinkle Twinkle Little Star"

+

Reference melody:
Birthday-song melody

↓

Generated:
"Twinkle Twinkle Little Star"
following the birthday-song melody
```

The extended objective is evaluated after the primary speech-to-singing
pipeline is working.

------------------------------------------------------------------------

# 4. Research Questions

### RQ1 --- Content representation

Can linguistic content be extracted from speech while minimizing
unwanted speaker information?

### RQ2 --- Speaker preservation

Can a speaker embedding extracted from speech preserve the target
speaker's identity in generated singing?

### RQ3 --- Melody conditioning

Can an independent melody representation control the pitch and timing of
generated singing?

### RQ4 --- Speech-to-singing alignment

Can phone-level information and a duration predictor bridge the temporal
difference between speech and singing?

------------------------------------------------------------------------

# 5. Core Architecture

The fixed model stack for the project is:

> **HuBERT + ECAPA-TDNN + CREPE + Duration Predictor + Conformer +
> HiFi-GAN + optional Adversarial Training**

``` text
                    SPOKEN LYRICS
                         |
             +-----------+-----------+
             |                       |
             v                       v
        HuBERT Encoder        ECAPA-TDNN Encoder
             |                       |
             v                       v
      Content Features         Speaker Embedding
             |                       |
             +-----------+-----------+
                         |
                         v
                 Alignment Module
                         ^
                         |
                   Melody Input
                         |
                       CREPE
                         |
                         v
                  F0 / Pitch Curve
                         |
                  Note Representation
                         |
                         v
                 Duration Predictor
                         |
                         v
              Singing-time Alignment
                         |
             +-----------+-----------+
             |                       |
             v                       v
      Aligned Content         Melody Features
             |                       |
             +-----------+-----------+
                         |
                         v
                    CONFORMER
                  Acoustic Model
                         |
                         v
                  Mel Spectrogram
                         |
                         v
                     HiFi-GAN
                      Vocoder
                         |
                         v
                 Generated Singing
```

------------------------------------------------------------------------

# 6. Component Responsibilities

  -----------------------------------------------------------------------
  Component               Model / Tool            Responsibility
  ----------------------- ----------------------- -----------------------
  Content encoder         **HuBERT**              Extract
                                                  linguistic/content
                                                  representation from
                                                  speech

  Speaker encoder         **ECAPA-TDNN**          Extract target speaker
                                                  identity

  Melody extractor        **CREPE**               Extract F0/pitch
                                                  contour from reference
                                                  melody

  Duration predictor      **Transformer /         Predict singing-time
                          FastSpeech-style**      phoneme durations

  Acoustic model          **Conformer**           Generate singing mel
                                                  spectrogram

  Vocoder                 **HiFi-GAN**            Convert mel spectrogram
                                                  to waveform

  Optional discriminator  Adversarial training    Improve acoustic
                                                  realism
  -----------------------------------------------------------------------

Pretrained HuBERT, ECAPA-TDNN, and HiFi-GAN components should initially
be frozen. The main trainable components are the duration/alignment
module and Conformer acoustic model.

------------------------------------------------------------------------

# 7. Dataset Strategy

## Primary development dataset: NUS SMC Corpus / NUS-48E

The proposed technical approach uses the NUS SMC Corpus
(`nus-smc-corpus_48`) as the primary development dataset.

Reported properties:

  Property            Value
  ------------------- -------------------------------
  Speakers            12
  Gender              6 male, 6 female
  Unique songs        20
  Paired recordings   48
  Total duration      \~169 minutes
  Sample rate         44.1 kHz
  Recording types     Read speech + singing
  Annotation          Phone-level time-aligned text

Each speaker contains paired `read/` and `sing/` recordings.

``` text
SPEAKER_ID/
├── read/
│   ├── 05.wav
│   ├── 05.txt
│   ├── 08.wav
│   └── ...
│
└── sing/
    ├── 05.wav
    ├── 05.txt
    ├── 08.wav
    └── ...
```

Each numbered pair corresponds to the same lyrics performed by the same
speaker in spoken and sung form.

### Why this dataset is useful

The phone-level annotations support:

-   speech/singing alignment,
-   phoneme duration extraction,
-   phoneme-to-note mapping,
-   duration-model supervision.

------------------------------------------------------------------------

## Optional additional dataset: NHSS

NHSS can be incorporated if access is obtained through its research-use
license.

Potential uses:

-   additional speech/singing training data,
-   cross-dataset evaluation,
-   testing generalization.

The core pipeline should remain executable using the primary development
dataset.

------------------------------------------------------------------------

# 8. Why Phone-Level Annotation Matters

A phone-level annotation gives the start and end time of individual
speech sounds.

Example:

``` text
Word: "cat"

 /k/       /æ/       /t/
 |---------|---------|
0.00      0.15      0.40 sec
```

This matters because speech and singing have different temporal
structures.

A short spoken vowel may be stretched across a long musical note:

``` text
Speech:

"Happy"
|--hap--|--py--|

Singing:

"Haaaaa-py"
|---------|---|
```

Phone-level annotations provide supervision for:

-   phoneme durations,
-   speech-to-singing timing,
-   phoneme-to-note correspondence,
-   vowel stretching.

------------------------------------------------------------------------

# 9. Phase 1 --- Dataset Preparation

### Objectives

Prepare the raw corpus for feature extraction and training.

### Tasks

1.  Organize speaker directories.
2.  Match every `read/` recording with its corresponding `sing/`
    recording.
3.  Parse phone-level annotation files.
4.  Verify phone timestamps.
5.  Normalize audio.
6.  Resample audio where required.
7.  Create speaker-disjoint train/validation/test splits.

### Output

For each training example:

``` text
speech waveform
+
speech phone sequence
+
speech phone durations
+
speaker ID
+
singing waveform
+
singing phone alignment
+
singing mel spectrogram
+
singing F0
```

------------------------------------------------------------------------

# 10. Phase 2 --- Offline Feature Extraction

Features are extracted once and cached before model training.

## 10.1 HuBERT

``` text
spoken waveform
      |
    HuBERT
      |
      v
content representation
```

Purpose:

Capture linguistic information while reducing dependence on
speaker-specific acoustic characteristics.

------------------------------------------------------------------------

## 10.2 ECAPA-TDNN

``` text
target speech
      |
  ECAPA-TDNN
      |
      v
speaker embedding
```

Purpose:

Represent the target speaker's vocal identity.

------------------------------------------------------------------------

## 10.3 CREPE

``` text
reference melody
      |
    CREPE
      |
      v
F0 / pitch contour
```

Purpose:

Represent the melody's fundamental-frequency trajectory.

------------------------------------------------------------------------

## 10.4 Mel Spectrogram

``` text
singing waveform
      |
STFT / Mel filterbank
      |
      v
target mel spectrogram
```

Purpose:

Provide the acoustic target for Conformer training.

------------------------------------------------------------------------

# 11. Phase 3 --- Melody Representation

The melody is represented at two levels.

### Frame-level

-   F0
-   voiced/unvoiced information
-   pitch trajectory

### Note-level

-   pitch
-   onset
-   offset
-   duration
-   note position

Conceptually:

``` text
Note 1       Note 2       Note 3
  C4           D4           E4
|------------|------------|
    duration     duration
```

The frame-level F0 and note-level representation are passed to the
alignment/acoustic system.

------------------------------------------------------------------------

# 12. Phase 4 --- Speech-to-Singing Alignment

This is one of the main technical challenges.

### Problem

Speech and singing use different temporal structures.

``` text
Speech:
phoneme → short duration

Singing:
phoneme → potentially long musical duration
```

Therefore the speech representation must be transformed onto a singing
timeline.

### Duration Predictor

The proposed duration predictor receives:

``` text
Content representation
+
Melody note information
```

and predicts:

``` text
Singing-time duration
for each phoneme
```

Possible architecture:

-   2--3 layer Transformer
-   FastSpeech-style duration predictor

The duration predictor is trained using ground-truth singing phone
durations.

------------------------------------------------------------------------

# 13. Phase 5 --- Content-to-Melody Alignment

After predicting singing durations:

``` text
Speech content
      |
      v
Duration Predictor
      |
      v
Predicted phone durations
      |
      v
Temporal expansion
      |
      v
Content representation
on singing timeline
```

The aligned content is combined with:

-   speaker embedding,
-   F0 contour,
-   note representation.

The result is a unified representation for singing generation.

------------------------------------------------------------------------

# 14. Phase 6 --- Conformer Acoustic Model

The Conformer is the main trainable acoustic model.

### Inputs

1.  Aligned HuBERT content features
2.  ECAPA speaker embedding
3.  CREPE F0 contour
4.  Note-level melody features

### Architecture

``` text
Aligned Content
      |
      |
Speaker Embedding ----+
      |               |
      |               v
      +---------> Conformer
                   Acoustic
                    Model
                       |
                       v
                Mel Spectrogram
```

A practical starting configuration:

-   4--6 Conformer layers
-   moderate hidden dimension
-   multi-head self-attention
-   convolution modules
-   dropout

The exact dimensions are selected based on GPU memory and validation
performance.

------------------------------------------------------------------------

# 15. Phase 7 --- Training Objectives

## Primary loss: L1 Mel Loss

``` text
L_mel = |Predicted Mel - Ground Truth Mel|
```

Purpose:

Train the acoustic model to reproduce the target singing spectrum.

------------------------------------------------------------------------

## Auxiliary F0 Loss

``` text
L_F0 =
MSE(Predicted F0, Ground Truth F0)
```

Purpose:

Improve pitch/melody adherence.

------------------------------------------------------------------------

## Duration Loss

``` text
L_duration =
difference between predicted
and ground-truth singing phone durations
```

Purpose:

Improve speech-to-singing temporal alignment.

------------------------------------------------------------------------

## Combined loss

``` text
L_total =
L_mel
+ λ_F0 L_F0
+ λ_dur L_duration
```

The weights are tuned experimentally.

------------------------------------------------------------------------

# 16. Phase 8 --- Optional Adversarial Training

After the basic acoustic model is stable, adversarial training can be
investigated.

``` text
                 Real Mel
                    |
                    v
              Discriminator
                    ^
                    |
             Generated Mel
```

Purpose:

Improve:

-   spectral realism,
-   fine acoustic detail,
-   perceptual naturalness.

Adversarial training is optional and should not be introduced until the
main model is stable.

------------------------------------------------------------------------

# 17. Phase 9 --- HiFi-GAN Vocoder

The Conformer generates:

``` text
Mel Spectrogram
```

HiFi-GAN converts it to:

``` text
Waveform
```

``` text
Generated Mel
     |
  HiFi-GAN
     |
     v
singing.wav
```

The initial vocoder is pretrained and frozen.

------------------------------------------------------------------------

# 18. Complete Training Pipeline

``` text
                    READ SPEECH
                         |
              +----------+----------+
              |                     |
              v                     v
           HuBERT                ECAPA
              |                     |
              v                     v
       Content Features       Speaker Embedding
              |                     |
              +----------+----------+
                         |
                         v
                 Duration Predictor
                         ^
                         |
                         |
                  SINGING / MELODY
                         |
                       CREPE
                         |
                         v
                   F0 + Notes
                         |
                         v
                Melody Representation
                         |
              +----------+----------+
              |                     |
              v                     v
       Aligned Content        Melody Features
              |                     |
              +----------+----------+
                         |
                         v
                    CONFORMER
                  Acoustic Model
                         |
                         v
                  Mel Spectrogram
                         |
                         v
                     HiFi-GAN
                         |
                         v
                  Singing Waveform
```

------------------------------------------------------------------------

# 19. Phase 10 --- Inference Pipeline

### User inputs

``` text
speech.wav
melody.wav
```

### Step 1 --- Content

``` text
speech.wav
   |
 HuBERT
   |
   v
content representation
```

### Step 2 --- Speaker identity

``` text
speech.wav
   |
 ECAPA-TDNN
   |
   v
speaker embedding
```

### Step 3 --- Melody

``` text
melody.wav
   |
 CREPE
   |
   v
F0 contour
```

### Step 4 --- Alignment

``` text
content
+
melody
   |
Duration Predictor
   |
   v
aligned singing content
```

### Step 5 --- Acoustic synthesis

``` text
aligned content
+
speaker embedding
+
melody features
   |
   v
Conformer
   |
   v
mel spectrogram
```

### Step 6 --- Waveform synthesis

``` text
mel spectrogram
      |
   HiFi-GAN
      |
      v
singing.wav
```

------------------------------------------------------------------------

# 20. Example End-to-End Scenario

### Input speech

``` text
Speaker reads:

"Twinkle Twinkle Little Star"
```

### Input melody

``` text
A reference melody
```

### Processing

``` text
Speech
  |
  +--> HuBERT --> linguistic content
  |
  +--> ECAPA --> speaker identity

Melody
  |
  +--> CREPE --> F0/pitch
  |
  +--> note extraction --> note timing

Content + melody
  |
  +--> duration prediction
  |
  +--> temporal alignment

All representations
  |
  +--> Conformer
  |
  +--> mel spectrogram
  |
  +--> HiFi-GAN
```

### Output

``` text
The target speaker singing:

"Twinkle Twinkle Little Star"

following the reference melody.
```

------------------------------------------------------------------------

# 21. Experimental Design

## Experiment 1 --- Original/Compatible Melody

Input:

``` text
Speech lyrics
+
corresponding melody
```

Purpose:

Verify the basic speech-to-singing pipeline.

------------------------------------------------------------------------

## Experiment 2 --- Different Melody

Input:

``` text
Speech lyrics
+
different compatible melody
```

Purpose:

Evaluate independent melody conditioning.

------------------------------------------------------------------------

## Experiment 3 --- Unseen Speaker

Input:

``` text
Speech from an unseen speaker
+
reference melody
```

Purpose:

Evaluate speaker generalization.

------------------------------------------------------------------------

## Experiment 4 --- Melody Ablation

Compare:

``` text
Full model
vs.
model without melody conditioning
```

Purpose:

Measure the contribution of melody conditioning.

------------------------------------------------------------------------

## Experiment 5 --- Speaker Ablation

Compare:

``` text
Full model
vs.
model without ECAPA speaker embedding
```

Purpose:

Measure the contribution of explicit speaker conditioning.

------------------------------------------------------------------------

# 22. Baseline

A classical feature-based baseline should be implemented before the
complete neural model.

``` text
Speech
   |
Phone alignment
   |
F0 / melody transfer
   |
Duration modification
   |
WORLD vocoder
   |
Generated singing
```

This provides a lower-complexity comparison against the proposed neural
architecture.

The comparison becomes:

``` text
Classical WORLD baseline
            VS
HuBERT + ECAPA + CREPE
+ Duration Predictor
+ Conformer + HiFi-GAN
```

------------------------------------------------------------------------

# 23. Evaluation Metrics

## Objective metrics

### MCD --- Mel-Cepstral Distortion

Measures spectral difference between generated and reference singing.

Lower is better.

### F0 RMSE

Measures pitch error.

Lower is better.

### F0 Correlation

Measures similarity between generated and reference pitch trajectories.

Higher is better.

### Speaker Cosine Similarity

Compare:

``` text
Target speaker speech embedding
            vs.
Generated singing embedding
```

Higher similarity indicates stronger speaker preservation.

------------------------------------------------------------------------

## Subjective metrics

Listeners rate:

### Naturalness

How natural does the generated singing sound?

### Speaker Similarity

How similar is the generated voice to the target speaker?

### Melody Adherence

How accurately does the generated singing follow the reference melody?

### Overall Quality

Overall perceptual quality.

------------------------------------------------------------------------

# 24. Evaluation Matrix

  System                 Speaker          Melody             Purpose
  ---------------------- ---------------- ------------------ ------------------------
  WORLD baseline         Speech           Original           Classical baseline
  Proposed model         Speech           Original           Main system
  Proposed model         Unseen speaker   Original           Speaker generalization
  Proposed model         Speech           Different melody   Melody transfer
  No speaker embedding   Speech           Original           Speaker ablation
  No melody condition    Speech           Original           Melody ablation

------------------------------------------------------------------------

# 25. Phase-Based Development Plan

## Phase 1 --- Dataset Preparation

-   Acquire dataset.
-   Organize speech/singing pairs.
-   Parse phone annotations.
-   Normalize audio.
-   Create speaker-disjoint splits.

**Deliverable:** Clean structured dataset.

------------------------------------------------------------------------

## Phase 2 --- Feature Extraction

-   Extract HuBERT content features.
-   Extract ECAPA speaker embeddings.
-   Extract CREPE F0.
-   Generate mel spectrograms.
-   Generate note/duration representations.

**Deliverable:** Cached training features.

------------------------------------------------------------------------

## Phase 3 --- Alignment and Duration Modeling

-   Establish phone-level speech/singing correspondence.
-   Extract singing phone durations.
-   Map phones to musical timing.
-   Implement duration predictor.
-   Train and evaluate duration prediction.

**Deliverable:** Singing-time aligned content representation.

------------------------------------------------------------------------

## Phase 4 --- Classical Baseline

-   Implement phone alignment.
-   Perform F0 transfer.
-   Perform duration modification.
-   Generate audio with WORLD.
-   Establish baseline metrics.

**Deliverable:** First working speech-to-singing system.

------------------------------------------------------------------------

## Phase 5 --- Conformer Acoustic Model

-   Combine aligned content.
-   Add ECAPA speaker embedding.
-   Add F0 and note features.
-   Train Conformer.
-   Optimize mel reconstruction.
-   Add F0 and duration losses.

**Deliverable:** Neural singing acoustic model.

------------------------------------------------------------------------

## Phase 6 --- HiFi-GAN Integration

-   Connect predicted mel spectrogram to HiFi-GAN.
-   Generate waveform.
-   Validate audio quality.
-   Build complete inference pipeline.

**Deliverable:** End-to-end speech + melody → singing system.

------------------------------------------------------------------------

## Phase 7 --- Optional Adversarial Enhancement

-   Add mel discriminator.
-   Introduce adversarial loss.
-   Compare against the non-adversarial model.

**Deliverable:** Enhanced model if it improves evaluation results.

------------------------------------------------------------------------

## Phase 8 --- Evaluation and Ablation

Evaluate:

-   MCD
-   F0 RMSE
-   F0 correlation
-   speaker cosine similarity
-   subjective ratings

Run:

-   baseline comparison,
-   speaker ablation,
-   melody ablation,
-   unseen-speaker evaluation,
-   different-melody evaluation.

**Deliverable:** Final experimental results and analysis.

------------------------------------------------------------------------

# 26. Risk Management

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Limited paired data                 Use pretrained HuBERT and ECAPA

  Model overfitting                   Speaker-disjoint splits, dropout,
                                      augmentation

  Alignment errors                    Use phone-level annotations

  Poor F0 extraction                  Visually and acoustically validate
                                      CREPE output

  Poor duration prediction            Begin with annotation-based
                                      alignment before learned prediction

  Weak speaker similarity             Improve ECAPA conditioning and
                                      speaker-related loss

  Poor singing quality                Use pretrained HiFi-GAN

  Adversarial instability             Keep adversarial training optional

  Poor melody transfer                Use melody augmentation and
                                      pitch/time transformations
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 27. Expected Output

The completed system should accept:

``` text
speech.wav
melody.wav
```

and produce:

``` text
singing.wav
```

where:

-   the lyrics correspond to the spoken input,
-   the speaker resembles the target speaker,
-   the pitch follows the reference melody,
-   the timing follows the musical structure,
-   the output has perceptually recognizable singing characteristics.

------------------------------------------------------------------------

# 28. Main Technical Contribution

The project focuses on independently modeling and combining:

``` text
        LINGUISTIC CONTENT
                +
        SPEAKER IDENTITY
                +
             MELODY
                |
                v
       SPEECH-TO-SINGING
            SYNTHESIS
                |
                v
             SINGING
```

The central technical challenge is:

> **Bridging the temporal and acoustic gap between speech and singing
> while independently controlling speaker identity and melody.**

The duration predictor and phone-level alignment mechanism explicitly
address the difference between speech-time and singing-time
representations.

------------------------------------------------------------------------

# 29. Final Architecture Summary

### Fixed architecture

**HuBERT + ECAPA-TDNN + CREPE + Duration Predictor + Conformer +
HiFi-GAN + optional Adversarial Training**

### Data flow

``` text
Spoken Lyrics
      |
      +--> HuBERT --> Content
      |
      +--> ECAPA --> Speaker Identity
      |
      |
Reference Melody
      |
      +--> CREPE --> F0
      |
      +--> Note Extraction --> Musical Timing
      |
      |
      +-----------------------------+
                                    |
                                    v
                            Duration Predictor
                                    |
                                    v
                           Content-Melody Alignment
                                    |
                                    v
                           +-------------------+
                           |     Conformer     |
                           |  Acoustic Model   |
                           +---------+---------+
                                     |
                                     v
                              Mel Spectrogram
                                     |
                                     v
                                 HiFi-GAN
                                     |
                                     v
                              Singing Waveform
```

------------------------------------------------------------------------

# 30. Final Project Statement

> **This project proposes a speaker-preserving speech-to-singing
> synthesis system that takes spoken lyrics from a target speaker and an
> independent reference melody. HuBERT is used to extract linguistic
> content, ECAPA-TDNN to represent speaker identity, and CREPE to
> extract melody information. A duration predictor aligns speech
> phonemes with the musical timeline. These representations are combined
> in a Conformer-based acoustic model to generate a singing mel
> spectrogram, which is converted into waveform audio using HiFi-GAN.
> Optional adversarial training can be investigated to improve
> perceptual quality.**

### Core transformation

``` text
Spoken Lyrics
      +
Speaker Identity
      +
Reference Melody
      ↓
Speech-to-Singing Alignment
      ↓
Conformer Acoustic Model
      ↓
Mel Spectrogram
      ↓
HiFi-GAN
      ↓
Generated Singing
```

### Primary research focus

**Speech-to-singing conversion with independent speaker-identity and
melody conditioning.**
