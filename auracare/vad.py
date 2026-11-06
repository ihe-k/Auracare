"""
Energy-based voice activity detection.

Deliberately dependency-free (numpy only) so the pause measurements are
reproducible and inspectable rather than hidden inside a neural VAD.

Two decisions here matter for correctness, and both differ from a naive
"fraction of frames below threshold" implementation:

1.  The pause ratio is measured over the *active span* only - from the first
    speech onset to the last speech offset. A patient who takes three seconds
    to start talking should not be scored as though they paused for three
    seconds, because leading silence is a reaction-time artefact, not a
    disfluency.

2.  Gaps shorter than `min_pause_ms` are not pauses. Stop consonants in normal
    speech produce 20-80 ms of near-silence; counting those would put every
    healthy speaker over a 45 % pause threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .config import DEFAULT_SIGNAL_PARAMS, SignalParams

Segment = Tuple[float, float]


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------
def frame_signal(
    samples: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float
) -> np.ndarray:
    """Split into overlapping frames, shape (n_frames, frame_len)."""
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))

    if samples.size < frame_len:
        pad = np.zeros(frame_len - samples.size, dtype=np.float32)
        samples = np.concatenate([samples, pad])

    n_frames = 1 + (samples.size - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return samples[idx]


def frame_energy_db(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    floor_db: float = -100.0,
) -> np.ndarray:
    """Per-frame RMS in dB (not dBFS - relative, used only for contrast)."""
    frames = frame_signal(samples, sample_rate, frame_ms, hop_ms)
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1) + 1e-20)
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return np.maximum(db, floor_db)


# ---------------------------------------------------------------------------
# Mask post-processing
# ---------------------------------------------------------------------------
def _runs(mask: np.ndarray, value: bool) -> List[Tuple[int, int]]:
    """Index runs [start, end) where mask == value."""
    out: List[Tuple[int, int]] = []
    if mask.size == 0:
        return out
    m = mask.astype(bool)
    target = bool(value)
    start = None
    for i, v in enumerate(m):
        if bool(v) == target and start is None:
            start = i
        elif bool(v) != target and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(m)))
    return out


def _fill_short_gaps(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Bridge silent runs shorter than `min_frames` that sit between speech."""
    out = mask.copy()
    for a, b in _runs(mask, False):
        if a == 0 or b == len(mask):
            continue  # leading/trailing silence is never bridged
        if (b - a) < min_frames:
            out[a:b] = True
    return out


def _drop_short_speech(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Remove speech runs too short to be a syllable (clicks, taps, breaths)."""
    out = mask.copy()
    for a, b in _runs(mask, True):
        if (b - a) < min_frames:
            out[a:b] = False
    return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class VadResult:
    """Speech/silence structure of one recording."""

    mask: np.ndarray
    hop_s: float
    threshold_db: float
    noise_floor_db: float
    speech_segments: List[Segment] = field(default_factory=list)
    pause_segments: List[Segment] = field(default_factory=list)
    active_start_s: float = 0.0
    active_end_s: float = 0.0
    total_duration_s: float = 0.0

    # -- derived measures ---------------------------------------------------
    @property
    def active_span_s(self) -> float:
        return max(0.0, self.active_end_s - self.active_start_s)

    @property
    def speech_time_s(self) -> float:
        return float(sum(b - a for a, b in self.speech_segments))

    @property
    def pause_time_s(self) -> float:
        return float(sum(b - a for a, b in self.pause_segments))

    @property
    def pause_ratio(self) -> float:
        """Silent time as a fraction of the active span. 0.0 if no speech."""
        span = self.active_span_s
        if span <= 0:
            return 0.0
        return float(min(1.0, self.pause_time_s / span))

    @property
    def n_pauses(self) -> int:
        return len(self.pause_segments)

    @property
    def mean_pause_s(self) -> float:
        if not self.pause_segments:
            return 0.0
        return self.pause_time_s / len(self.pause_segments)

    @property
    def longest_pause_s(self) -> float:
        if not self.pause_segments:
            return 0.0
        return float(max(b - a for a, b in self.pause_segments))

    @property
    def longest_speech_s(self) -> float:
        if not self.speech_segments:
            return 0.0
        return float(max(b - a for a, b in self.speech_segments))

    @property
    def has_speech(self) -> bool:
        return bool(self.speech_segments)

    def as_dict(self) -> dict:
        return {
            "has_speech": self.has_speech,
            "speech_time_s": round(self.speech_time_s, 3),
            "pause_time_s": round(self.pause_time_s, 3),
            "pause_ratio": round(self.pause_ratio, 4),
            "n_pauses": self.n_pauses,
            "mean_pause_s": round(self.mean_pause_s, 3),
            "longest_pause_s": round(self.longest_pause_s, 3),
            "longest_speech_s": round(self.longest_speech_s, 3),
            "active_span_s": round(self.active_span_s, 3),
            "active_start_s": round(self.active_start_s, 3),
            "active_end_s": round(self.active_end_s, 3),
            "total_duration_s": round(self.total_duration_s, 3),
            "threshold_db": round(self.threshold_db, 2),
            "noise_floor_db": round(self.noise_floor_db, 2),
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def detect_voice_activity(
    samples: np.ndarray,
    sample_rate: int,
    params: SignalParams | None = None,
) -> VadResult:
    """
    Adaptive energy VAD.

    The threshold is the higher of (noise floor + offset) and
    (peak - peak_floor). The first term adapts to a quiet room, the second
    stops a very noisy room from marking everything as speech.
    """
    p = params or DEFAULT_SIGNAL_PARAMS
    samples = np.asarray(samples, dtype=np.float32)
    hop_s = p.hop_ms / 1000.0
    total_s = float(samples.size) / float(sample_rate) if sample_rate else 0.0

    if samples.size == 0 or sample_rate <= 0:
        return VadResult(
            mask=np.zeros(0, dtype=bool),
            hop_s=hop_s,
            threshold_db=-120.0,
            noise_floor_db=-120.0,
            total_duration_s=total_s,
        )

    db = frame_energy_db(samples, sample_rate, p.frame_ms, p.hop_ms)

    noise_floor = float(np.percentile(db, 10.0))
    peak = float(np.max(db))
    threshold = max(noise_floor + p.vad_floor_offset_db, peak - p.vad_peak_floor_db)

    mask = db > threshold
    min_speech_frames = max(1, int(round(p.vad_min_speech_ms / p.hop_ms)))
    min_pause_frames = max(1, int(round(p.vad_min_pause_ms / p.hop_ms)))

    # Order matters: bridge articulatory gaps first, then discard transients,
    # otherwise a syllable split by a stop closure gets dropped twice over.
    mask = _fill_short_gaps(mask, min_pause_frames)
    mask = _drop_short_speech(mask, min_speech_frames)

    speech_runs = _runs(mask, True)
    if not speech_runs:
        return VadResult(
            mask=mask,
            hop_s=hop_s,
            threshold_db=threshold,
            noise_floor_db=noise_floor,
            total_duration_s=total_s,
        )

    def to_time(i: int) -> float:
        return min(total_s, i * hop_s)

    speech_segments = [(to_time(a), to_time(b)) for a, b in speech_runs]
    active_start = speech_segments[0][0]
    active_end = speech_segments[-1][1]

    # Pauses: silent runs strictly inside the active span and long enough to count.
    pause_segments: List[Segment] = []
    for a, b in _runs(mask, False):
        if a == 0 or b == len(mask):
            continue
        if (b - a) < min_pause_frames:
            continue
        pause_segments.append((to_time(a), to_time(b)))

    return VadResult(
        mask=mask,
        hop_s=hop_s,
        threshold_db=threshold,
        noise_floor_db=noise_floor,
        speech_segments=speech_segments,
        pause_segments=pause_segments,
        active_start_s=active_start,
        active_end_s=active_end,
        total_duration_s=total_s,
    )


__all__ = [
    "Segment",
    "VadResult",
    "detect_voice_activity",
    "frame_energy_db",
    "frame_signal",
]
