"""Shared audio loading / resampling helpers (torchaudio-based)."""

from __future__ import annotations

import torch
import torchaudio


def load_waveform(wav_path, sr: int = 44100, channels_first: bool = True):
    """Load a wav, resample to `sr`, return mono torch.Tensor of shape (1, N)."""
    waveform, orig_sr = torchaudio.load(wav_path)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def load_numpy(wav_path, sr: int = 44100) -> "np.ndarray":
    """Return mono float32 numpy array (1D)."""
    import numpy as np

    waveform, _ = load_waveform(wav_path, sr)
    return waveform.squeeze(0).numpy().astype(np.float32)