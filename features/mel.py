"""Log-mel spectrogram extraction (target acoustic representation).

Output: (T, n_mels) log-mel, frame rate = sr / hop_length.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from .audio import load_waveform


def extract_mel(
    wav_path,
    sr: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = None,
    log_floor: float = 1e-5,
) -> np.ndarray:
    waveform, _ = load_waveform(wav_path, sr)
    if fmax is None:
        fmax = sr / 2.0

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
    )
    mel = transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)
    mel = torch.log(torch.clamp(mel, min=float(log_floor)))
    return mel.float().numpy().astype(np.float32)