"""Pretrained NSF-HiFiGAN singing vocoder (OpenVPI, 44.1 kHz, 128 mels, hop 512).

Mel extraction and the generator come from the DiffSinger code, so training
targets match what the vocoder was trained on exactly. Expected layout:

  vocoder/DiffSinger/                                   git clone --depth 1 https://github.com/openvpi/DiffSinger
  vocoder/nsf_hifigan_44.1k_hop512_128bin_2024.02/      model.ckpt + config.json from
      https://github.com/openvpi/vocoders/releases (weights: CC BY-NC-SA 4.0, research use)

Mel frame t is centred at (t + 0.5) * hop / sr seconds.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
VOCODER_DIR = ROOT / "vocoder"
CKPT = VOCODER_DIR / "nsf_hifigan_44.1k_hop512_128bin_2024.02" / "model.ckpt"


def _import_diffsinger():
    sys.path.insert(0, str(VOCODER_DIR / "DiffSinger"))
    try:
        import lightning  # noqa: F401
    except ImportError:
        # models.py imports lightning only for a log helper; stub it instead of installing lightning
        for name in ["lightning", "lightning.pytorch", "lightning.pytorch.utilities",
                     "lightning.pytorch.utilities.rank_zero"]:
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["lightning.pytorch.utilities.rank_zero"].rank_zero_info = print
    from modules.nsf_hifigan.models import load_model
    from modules.nsf_hifigan.nvSTFT import STFT
    return load_model, STFT


def interp_f0(f0: np.ndarray):
    """Fill unvoiced (0 Hz) frames by linear interpolation. Returns (f0_filled, uv)."""
    f0 = np.asarray(f0, dtype=np.float32)
    uv = f0 <= 0
    if uv.all():
        return np.full_like(f0, 100.0), uv
    idx = np.arange(len(f0))
    filled = np.interp(idx, idx[~uv], f0[~uv]).astype(np.float32)
    return filled, uv


class Vocoder:
    def __init__(self, device="cuda"):
        load_model, STFT = _import_diffsinger()
        self.device = device
        self.model, self.h = load_model(CKPT)
        self.model.to(device)
        h = self.h
        self.sr, self.hop, self.n_mels = h.sampling_rate, h.hop_size, h.num_mels
        self.stft = STFT(sr=h.sampling_rate, n_mels=h.num_mels, n_fft=h.n_fft, win_size=h.win_size,
                         hop_length=h.hop_size, fmin=h.fmin, fmax=h.fmax, device=device)

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop

    def mel(self, wav: np.ndarray) -> np.ndarray:
        """(N,) waveform at self.sr -> (T, n_mels) natural-log mel, float32."""
        y = torch.from_numpy(np.clip(wav, -1.0, 1.0)).float().to(self.device)[None]
        with torch.no_grad():
            return self.stft.get_mel(y)[0].T.cpu().numpy().astype(np.float32)

    def __call__(self, mel: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """(T, n_mels) log-mel + (T,) F0 in Hz (0 = unvoiced) -> (T * hop,) waveform."""
        f0_filled, _ = interp_f0(f0)
        c = torch.from_numpy(mel).float().to(self.device).T[None]
        f = torch.from_numpy(f0_filled).float().to(self.device)[None]
        with torch.no_grad():
            return self.model(c, f).view(-1).cpu().numpy().astype(np.float32)
