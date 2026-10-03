"""HuBERT content feature extraction (from spoken `read/` audio).

Output: (T, 768) frame-level features at ~50 Hz.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import HubertModel

from .audio import load_waveform

# HuBERT (facebook/hubert-base-ls960) is trained on 16 kHz audio.
HUBERT_SR = 16000


def load_hubert(model_name: str = "facebook/hubert-base-ls960", device="auto"):
    device = resolve_device(device)
    model = HubertModel.from_pretrained(model_name).to(device).eval()
    return model


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_hubert(
    wav_path,
    model,
    layer: int = 9,
    device="auto",
    sr: int = 44100,
    chunk_sec: float = 30.0,
    overlap_sec: float = 1.0,
) -> np.ndarray:
    """Extract layer `layer` hidden states. Returns (T, hidden_size) float32.

    Audio is resampled to 16 kHz (HuBERT native rate), so the output frame
    rate is 50 Hz (conv downsample factor 320).
    """
    device = resolve_device(device)
    waveform, _ = load_waveform(wav_path, HUBERT_SR)
    waveform = waveform.to(device)  # (1, N)

    n = waveform.shape[1]
    chunk = int(chunk_sec * HUBERT_SR)
    overlap = int(overlap_sec * HUBERT_SR)
    if n <= chunk:
        return _extract_segment(waveform, model, layer)

    # Long audio: process in overlapping chunks to bound attention memory,
    # then drop the overlap region between adjacent chunks.
    hop = chunk - overlap
    keep_frames = hop // 320  # HuBERT downsample factor is 320 samples/frame

    pieces = []
    for start in range(0, n, hop):
        end = min(start + chunk, n)
        feats = _extract_segment(waveform[:, start:end], model, layer)
        pieces.append(feats)
        if end == n:
            break

    result = []
    for i, feats in enumerate(pieces):
        is_last = i == len(pieces) - 1
        result.append(feats if is_last else feats[:keep_frames])
    out = np.concatenate(result, axis=0)
    return out.astype(np.float32)


def _extract_segment(waveform, model, layer: int) -> np.ndarray:
    with torch.no_grad():
        outputs = model(waveform, output_hidden_states=True)
        features = outputs.hidden_states[layer].squeeze(0).float().cpu().numpy()
    return features.astype(np.float32)