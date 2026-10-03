"""Invert a log-mel spectrogram back to a (rough) waveform via
Griffin-Lim on the linear magnitude spectrogram. Used only as an
audibility sanity check, not part of the training pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


def invert_mel(
    log_mel: np.ndarray,
    sr: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    log_floor: float = 1e-5,
    n_iter: int = 100,
) -> np.ndarray:
    """Return mono float32 waveform (shape (N,))."""
    log_mel_t = torch.from_numpy(log_mel).float().transpose(0, 1)  # (n_mels, T)

    mel_inv = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sr,
        f_min=0.0,
        f_max=sr / 2.0,
        norm="slaney",
        mel_scale="slaney",
    )
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_iter=n_iter
    )

    with torch.no_grad():
        mag_stft = mel_inv(torch.exp(log_mel_t))  # (1+n_fft/2, T)
        wav = griffin(mag_stft)  # (1, N)
    return wav.squeeze(0).numpy().astype(np.float32)