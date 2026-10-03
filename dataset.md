# NUS SMC Corpus / NUS-48E — Dataset Description

## Overview

The **NUS SMC Corpus (NUS-48E)** is a parallel **speech-and-singing corpus** designed for studying the acoustic differences between spoken and sung versions of the same lyrics.

The dataset is particularly relevant to speech-to-singing research because it provides **spoken and sung recordings from the same speakers**, allowing the relationship between speech and singing to be studied directly.

## Dataset Statistics

| Property | Details |
|---|---|
| Dataset | NUS SMC Corpus / NUS-48E |
| Speakers | 12 |
| Gender distribution | 6 male, 6 female |
| Unique songs | 20 English songs |
| Paired recordings | 48 |
| Total duration | Approximately 169 minutes |
| Sample rate | 44.1 kHz |
| Recording types | Read speech + singing |
| Annotations | Phone-level time-aligned `.txt` annotations |

## Dataset Structure

```text
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

Each numbered pair represents the **same lyrics performed by the same speaker** in two forms:

- `read/` → spoken/read version
- `sing/` → sung version

## What Does a Pair Represent?

For example:

```text
Speaker 01

read/05.wav
    ↓
Speaker reads the lyrics

sing/05.wav
    ↓
The same speaker sings the lyrics
```

This paired structure allows the project to study:

```text
Spoken Lyrics
      ↓
Speech Characteristics
      ↓
Singing Characteristics
```

while keeping the speaker and linguistic content controlled.

## Phone-Level Annotations

A phone is an individual speech sound.

For example:

```text
"cat"

 /k/       /æ/       /t/
 |---------|---------|
0.00      0.15      0.40 sec
```

The annotation provides:

- which phone occurs,
- when the phone begins,
- when the phone ends.

The dataset provides these annotations for both the `read/` and `sing/` recordings.

## Why Phone-Level Annotation Is Important

Speech and singing have different temporal characteristics.

For example, in speech:

```text
"Happy"

|--hap--|--py--|
```

In singing, a vowel can be extended:

```text
"Haaaaa-py"

|---------|---|
```

Therefore, speech-to-singing conversion requires the system to understand how phone durations change between speech and singing.

The phone-level annotations provide supervision for:

- speech/singing alignment,
- phoneme duration modelling,
- phoneme-to-note correspondence,
- singing-time duration prediction.

## Relevance to the Proposed Project

The proposed project is:

```text
Spoken Lyrics
      +
Reference Melody
      ↓
Speech-to-Singing System
      ↓
Generated Singing
```

NUS-48E supports the speech-to-singing component because it provides:

### 1. Spoken input

The `read/` recordings provide spoken lyrics from the target speaker.

### 2. Singing target

The corresponding `sing/` recordings provide ground-truth singing from the same speaker.

### 3. Speaker information

Because the same speaker performs both spoken and sung recordings, the dataset supports speaker-preservation experiments.

### 4. Temporal alignment

Phone-level annotations provide precise timing information for both speech and singing.

## How the Dataset Fits the Proposed Architecture

```text
                  READ SPEECH
                       |
              +--------+--------+
              |                 |
              v                 v
           HuBERT            ECAPA
              |                 |
              v                 v
       Content Features   Speaker Embedding
              |                 |
              +--------+--------+
                       |
                       v
                 Alignment /
                Duration Model
                       ^
                       |
                SINGING DATA
                       |
                       v
                Ground Truth
              Melody + Timing
                       |
                       v
                  Conformer
                       |
                       v
                Mel Spectrogram
                       |
                       v
                  HiFi-GAN
                       |
                       v
              Generated Singing
```

## Role During Training

For each paired example:

```text
READ AUDIO
    |
    +--> HuBERT
    |      |
    |      └── Content representation
    |
    +--> ECAPA
           |
           └── Speaker embedding


SING AUDIO
    |
    +--> Ground-truth F0 / melody
    |
    +--> Ground-truth phone durations
    |
    +--> Ground-truth mel spectrogram
```

The model learns:

```text
Content
   +
Speaker Identity
   +
Melody
   +
Duration
   ↓
Singing Acoustic Representation
```

The predicted mel spectrogram is compared against the ground-truth singing mel spectrogram during training.

## Strengths

- Parallel speech and singing recordings
- Same speaker for speech and singing
- Same lyrics across paired recordings
- Phone-level temporal annotations
- English lyrics
- Multiple speakers
- Suitable for speech-to-singing analysis
- Relatively compact for feature extraction and experimentation

## Limitations

The dataset is relatively small:

- 12 speakers
- 48 paired recordings
- approximately 169 minutes

It also does not by itself provide a large independent collection of arbitrary melody references for unrestricted melody-transfer experiments.

Therefore, the dataset is strongest for:

> **Learning and evaluating the speech-to-singing relationship, speaker preservation, and speech–singing temporal alignment.**

Independent melody transfer can be treated as an additional experimental condition.

## Dataset Role in the Project

The dataset will primarily be used for:

1. Speech feature extraction
2. Speaker embedding extraction
3. Phone-level alignment
4. Duration modelling
5. Singing acoustic-model training
6. Objective evaluation
7. Speaker-disjoint testing

## One-Sentence Description for Presentation

> **NUS-48E is a parallel speech-and-singing corpus containing 48 paired recordings from 12 speakers across 20 English songs, with phone-level time-aligned annotations for both spoken and sung recordings, making it suitable for speech-to-singing conversion, speaker preservation, and speech–singing temporal alignment research.**
