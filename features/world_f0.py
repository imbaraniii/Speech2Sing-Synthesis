"""WORLD Harvest F0 extraction (from sung `sing/` audio).

Used as the melody F0 instead of CREPE: torchcrepe-tiny is unreliable on
about half of the NUS singers (periodicity ~0, octave errors vs Harvest),
while full CREPE is too slow on CPU/MPS.

Frame rate = 1000 / frame_period_ms (default 10 ms -> 100 Hz, same as the
CREPE cache). Harvest returns 0 Hz for unvoiced frames.
"""

from __future__ import annotations

import warnings

import numpy as np

from .audio import load_numpy

HARVEST_SR = 16000  # F0 < 1.1 kHz, so 16 kHz is plenty and ~3x faster than 44.1 kHz


def extract_f0_harvest(wav_path, frame_period_ms: float = 10.0,
                       fmin: float = 60.0, fmax: float = 1100.0) -> np.ndarray:
    """Return (T,) float32 F0 in Hz, 0.0 = unvoiced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pyworld

    wav = load_numpy(wav_path, HARVEST_SR).astype(np.float64)
    f0, _ = pyworld.harvest(wav, HARVEST_SR, f0_floor=fmin, f0_ceil=fmax,
                            frame_period=frame_period_ms)
    return f0.astype(np.float32)
