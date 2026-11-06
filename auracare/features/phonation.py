"""
Task 1 - Respiratory and laryngeal health: sustained phonation /a:/.

The headline measure is phonation duration, which the Auracare protocol uses as
its respiratory safety intercept (critical below 2.5 s). Getting that number
right needs more care than "how long is the file":

* The patient does not start phonating the instant recording begins, and
  usually inhales audibly first. Duration must be the longest *continuously
  voiced* run, not the file length and not total voiced time.
* Praat's voicing decision drops out for a frame or two mid-vowel on creaky or
  breathy voices. A short unvoiced gap inside an otherwise continuous vowel is
  bridged, so one dropped frame does not halve the reported duration.
* Jitter and shimmer are measured on the sustained segment alone. Measured over
  the whole file they would include the inhalation and the trailing silence,
  which inflates perturbation badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..audio_io import Audio
from ..config import DEFAULT_SIGNAL_PARAMS, SignalParams
from ..vad import frame_energy_db
from .voice import VoiceIndicators, analyse_voice, pitch_track

# A vowel is not considered to have ended until voicing has been absent this long.
_MAX_VOICING_GAP_S = 0.15
# Ignore runs shorter than this when looking for "the" sustained vowel.
_MIN_RUN_S = 0.10


@dataclass
class PhonationResult:
    """Task 1 measurements."""

    phonation_duration_s: float = 0.0
    phonation_start_s: Optional[float] = None
    phonation_end_s: Optional[float] = None
    total_voiced_s: float = 0.0
    recording_duration_s: float = 0.0

    mean_f0_hz: Optional[float] = None
    f0_cv: Optional[float] = None
    amplitude_decay_db_s: Optional[float] = None

    voice: VoiceIndicators = field(default_factory=VoiceIndicators)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        def r(v, nd=3):
            return None if v is None else round(float(v), nd)

        return {
            "phonation_duration_s": r(self.phonation_duration_s),
            "phonation_start_s": r(self.phonation_start_s),
            "phonation_end_s": r(self.phonation_end_s),
            "total_voiced_s": r(self.total_voiced_s),
            "recording_duration_s": r(self.recording_duration_s),
            "mean_f0_hz": r(self.mean_f0_hz, 2),
            "f0_cv": r(self.f0_cv, 4),
            "amplitude_decay_db_s": r(self.amplitude_decay_db_s, 2),
            "voice": self.voice.as_dict(),
            "notes": list(self.notes),
        }


def longest_voiced_run(
    times: np.ndarray,
    f0: np.ndarray,
    max_gap_s: float = _MAX_VOICING_GAP_S,
    min_run_s: float = _MIN_RUN_S,
) -> Tuple[float, float, float]:
    """
    Longest continuously-voiced interval, bridging gaps up to `max_gap_s`.

    Returns (start_s, end_s, duration_s); all zero when nothing is voiced.
    """
    if times.size == 0 or f0.size == 0:
        return 0.0, 0.0, 0.0

    voiced = f0 > 0
    if not np.any(voiced):
        return 0.0, 0.0, 0.0

    step = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    max_gap_frames = max(1, int(round(max_gap_s / step)))

    # Bridge short unvoiced gaps that sit between voiced frames.
    bridged = voiced.copy()
    idx = np.flatnonzero(voiced)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < (b - a) <= max_gap_frames + 1:
            bridged[a:b] = True

    best = (0.0, 0.0, 0.0)
    start = None
    for i, v in enumerate(bridged):
        if v and start is None:
            start = i
        elif not v and start is not None:
            dur = times[i - 1] - times[start] + step
            if dur > best[2]:
                best = (float(times[start]), float(times[i - 1] + step), float(dur))
            start = None
    if start is not None:
        dur = times[-1] - times[start] + step
        if dur > best[2]:
            best = (float(times[start]), float(times[-1] + step), float(dur))

    if best[2] < min_run_s:
        return 0.0, 0.0, 0.0
    return best


def _amplitude_decay(
    samples: np.ndarray, sample_rate: int, params: SignalParams
) -> Optional[float]:
    """
    Slope of frame level over the segment, dB per second.

    A steeply negative slope on a sustained vowel indicates the patient ran out
    of breath support rather than stopping deliberately.
    """
    db = frame_energy_db(samples, sample_rate, params.frame_ms, params.hop_ms)
    if db.size < 5:
        return None
    t = np.arange(db.size) * (params.hop_ms / 1000.0)
    # Fit only over frames above the segment's own noise floor.
    keep = db > (np.max(db) - 30.0)
    if keep.sum() < 5:
        return None
    slope = float(np.polyfit(t[keep], db[keep], 1)[0])
    return slope


def analyse_phonation(
    audio: Audio,
    params: SignalParams | None = None,
) -> PhonationResult:
    """Measure a sustained-vowel recording."""
    p = params or DEFAULT_SIGNAL_PARAMS
    out = PhonationResult(recording_duration_s=audio.duration_s)

    if audio.samples.size == 0:
        out.notes.append("empty recording")
        return out

    times, f0 = pitch_track(audio.samples, audio.sample_rate, p)
    if times.size == 0:
        out.notes.append("pitch tracking unavailable; cannot measure phonation time")
        out.voice = analyse_voice(audio.samples, audio.sample_rate, p)
        return out

    step = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    out.total_voiced_s = float(np.sum(f0 > 0) * step)

    start_s, end_s, duration_s = longest_voiced_run(times, f0)
    out.phonation_duration_s = duration_s

    if duration_s <= 0:
        out.notes.append(
            "no sustained voicing detected - the recording may be silence, "
            "breath noise or background only"
        )
        out.voice = analyse_voice(audio.samples, audio.sample_rate, p)
        return out

    out.phonation_start_s = start_s
    out.phonation_end_s = end_s

    # Restrict every voice measure to the sustained segment itself.
    segment = audio.slice_seconds(start_s, end_s)
    out.voice = analyse_voice(segment.samples, segment.sample_rate, p)

    in_run = (times >= start_s) & (times < end_s)
    run_f0 = f0[in_run]
    run_f0 = run_f0[run_f0 > 0]
    if run_f0.size:
        out.mean_f0_hz = float(np.mean(run_f0))
        if out.mean_f0_hz > 0:
            out.f0_cv = float(np.std(run_f0) / out.mean_f0_hz)

    out.amplitude_decay_db_s = _amplitude_decay(
        segment.samples, segment.sample_rate, p
    )

    if duration_s < 1.0:
        out.notes.append(
            "sustained segment under 1 s - perturbation measures are unreliable"
        )
    return out


__all__ = ["PhonationResult", "analyse_phonation", "longest_voiced_run"]
