"""Frame-rate alignment / interpolation utilities.

All feature streams are interpolated to a common target frame rate before
being fed to the acoustic model.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def upsample_to_target(features: np.ndarray, src_rate: float, target_rate: float) -> np.ndarray:
    """Linear-interpolate (T_src, D) features to the target frame rate."""
    if src_rate == target_rate:
        return features.astype(np.float32)
    if features.ndim == 1:
        features = features[:, None]
        single = True
    else:
        single = False

    T_src, D = features.shape
    duration = T_src / src_rate
    T_tgt = max(1, int(round(duration * target_rate)))
    src_times = np.linspace(0.0, duration, T_src)
    tgt_times = np.linspace(0.0, duration, T_tgt)
    interp = interp1d(src_times, features, axis=0, kind="linear", fill_value="extrapolate")
    out = interp(tgt_times).astype(np.float32)
    return out[:, 0] if single else out