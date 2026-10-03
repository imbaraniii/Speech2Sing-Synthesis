"""CREPE F0 / pitch extraction (from sung `sing/` audio).

Uses torchcrepe (pure-PyTorch port of CREPE). The CREPE model is trained
on 16 kHz audio, so we resample to 16 kHz before prediction and express
hop_length in 16 kHz samples.

Frame rate = 16000 / hop_length. Default hop_length=160 -> 100 Hz.
"""

from __future__ import annotations

import numpy as np
import torch

from .audio import load_waveform
from .hubert import resolve_device

CREPE_SR = 16000


def extract_f0(
    wav_path,
    model_capacity: str = "full",
    hop_length: int = 160,
    device="auto",
    sr: int = 44100,
    fmin: float = 50.0,
    fmax: float = 2006.0,
    batch_size: int = None,
):
    """Return (f0_hz, confidence) arrays at a frame rate of CREPE_SR / hop_length.

    torchcrepe never outputs 0 Hz: unvoiced frames get arbitrary F0 (often
    ~fmax in silence). Mask them with `confidence` (e.g. >= 0.5) instead.
    """
    import torchcrepe

    device = resolve_device(device)

    # load + resample to 16 kHz (CREPE native rate)
    waveform, _ = load_waveform(wav_path, CREPE_SR)
    audio = waveform.to(device)

    f0, confidence = torchcrepe.predict(
        audio,
        CREPE_SR,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model=model_capacity,
        batch_size=batch_size,
        device=device,
        pad=True,
        return_periodicity=True,
    )
    f0 = f0.squeeze().float().cpu().numpy()
    confidence = confidence.squeeze().float().cpu().numpy()
    return f0.astype(np.float32), confidence.astype(np.float32)