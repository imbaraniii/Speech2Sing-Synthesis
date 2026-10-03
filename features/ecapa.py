"""ECAPA-TDNN speaker embedding extraction (from spoken `read/` audio).

Output: (192,) utterance-level embedding. Community implementations expose
either 192-dim (speechbrain spkrec-ecapa-voxceleb) or 1024-dim.
"""

from __future__ import annotations

import numpy as np
import torch

from .audio import load_waveform

# speechbrain spkrec-ecapa-voxceleb is trained on 16 kHz audio.
ECAPA_SR = 16000


def load_ecapa(model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
               savedir: str = "pretrained_models/", device: str = "cpu"):
    from speechbrain.inference.speaker import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source=model_name, savedir=savedir, run_opts={"device": device}
    )
    return classifier


def extract_ecapa(wav_path, classifier, device="cpu") -> np.ndarray:
    """Return the speaker embedding vector (float32). Audio is resampled to 16 kHz."""
    signal, _ = load_waveform(wav_path, ECAPA_SR)
    signal = signal.to(device)
    with torch.no_grad():
        emb = classifier.encode_batch(signal).squeeze().float().cpu().numpy()
    return emb.astype(np.float32)