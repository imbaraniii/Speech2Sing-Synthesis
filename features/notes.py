"""Heuristic note segmentation from F0 contour + singing phone boundaries.

Groups contiguous voiced frames with stable F0 and quantizes each group to
the nearest MIDI pitch.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

CONCERT_A_HZ = 440.0


def hz_to_midi(freq: float) -> int:
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / CONCERT_A_HZ)))


def midi_to_hz(midi: float) -> float:
    return CONCERT_A_HZ * (2.0 ** ((midi - 69) / 12.0))


def frames_to_time(frame: int, sr: int, hop_length: int) -> float:
    return (frame * hop_length) / sr


def segment_notes(
    f0: np.ndarray,
    phones: List[dict],
    sr: int = 44100,
    hop_length: int = 512,
    stability_semitones: float = 1.0,
    min_note_frames: int = 3,
    min_conf: float = 0.5,
    confidence: np.ndarray = None,
) -> List[Dict]:
    """Segment F0 into stable-pitch notes.

    Args:
        f0: (T,) frames of F0 in Hz (0 = unvoiced).
        phones: sing-mode phone list (for final bookkeeping).
        confidence: optional (T,) voicing confidence for masking.
    Returns:
        List of {midi, onset_frame, offset_frame, duration_frames, onset_time, offset_time, duration_time}.
    """
    notes: List[Dict] = []
    voiced = f0 > 0
    if confidence is not None:
        voiced = voiced & (confidence >= min_conf)

    i = 0
    n = len(f0)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        midi_ref = hz_to_midi(f0[i])
        j = i
        while j < n and voiced[j]:
            if abs(hz_to_midi(f0[j]) - midi_ref) > stability_semitones:
                break
            j += 1
        duration = j - i
        if duration >= min_note_frames:
            notes.append(
                {
                    "midi": midi_ref,
                    "onset_frame": int(i),
                    "offset_frame": int(j),
                    "duration_frames": int(duration),
                    "onset_time": frames_to_time(i, sr, hop_length),
                    "offset_time": frames_to_time(j, sr, hop_length),
                    "duration_time": (j - i) * hop_length / sr,
                }
            )
        i = j
    return notes