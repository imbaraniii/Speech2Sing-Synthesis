"""Shared audio loading / resampling helpers.

Files are read with soundfile (torchaudio >= 2.9 needs torchcodec + FFmpeg
for loading); resampling uses torchaudio.functional.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_waveform(wav_path, sr: int = 44100, channels_first: bool = True):
    """Load a wav, resample to `sr`, return mono torch.Tensor of shape (1, N)."""
    data, orig_sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # (N, C)
    waveform = torch.from_numpy(data.T.copy())
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def load_numpy(wav_path, sr: int = 44100) -> np.ndarray:
    """Return mono float32 numpy array (1D)."""
    waveform, _ = load_waveform(wav_path, sr)
    return waveform.squeeze(0).numpy().astype(np.float32)
