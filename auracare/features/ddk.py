"""
Task 3 - Neuromotor speech control: diadochokinetic rate (/pa-ta-ka/).

Syllable nuclei are found as prominent peaks in the smoothed intensity
envelope. This is the classic De Jong & Wempe approach rather than a learned
detector, which matters here: it can be validated against a signal whose true
syllable rate is known by construction, and the test suite does exactly that.

Rate estimation uses inter-onset intervals, (N-1) / (t_last - t_first), not
N / duration. For N events spanning T seconds the former is exact for a
periodic sequence while the latter is biased low by roughly one period - at a
typical 4-second DDK task that bias is about 5 %, which is enough to push a
healthy speaker below a 5 Hz threshold.

The same nucleus detector is reused for connected-speech rate in Task 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks

from ..audio_io import Audio
from ..config import DEFAULT_SIGNAL_PARAMS, SignalParams
from ..vad import frame_energy_db

# Envelope resolution for nucleus detection. Finer than the VAD hop because
# syllables in a fast DDK task are only ~150 ms apart.
_ENVELOPE_HOP_MS = 5.0
_ENVELOPE_FRAME_MS = 20.0
# Backstop only. Peak acceptance is governed by voice activity and local
# prominence; this just discards anything in the deep noise floor.
#
# An earlier version used `max(envelope) - 25 dB` as the acceptance floor and
# it failed badly on a real recording: 385_AUDIO.wav is quiet (median frame
# level -68 dB) with a few loud transients peaking at -32 dB, so the floor
# landed at -57 dB and excluded 90 % of the actual speech. It found 26
# syllables where the transcript had 52 words. Anchoring acceptance to a global
# maximum is not robust to transients; anchoring it to voice activity is.
_PEAK_FLOOR_DB = 45.0


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average, edge-padded so length is preserved."""
    if width <= 1 or x.size == 0:
        return x
    width = min(width, x.size)
    kernel = np.ones(width, dtype=np.float64) / float(width)
    pad = width // 2
    padded = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    out = np.convolve(padded, kernel, mode="same")
    return out[pad : pad + x.size]


def detect_syllable_nuclei(
    audio: Audio,
    params: SignalParams | None = None,
    min_distance_ms: Optional[float] = None,
    prominence_db: Optional[float] = None,
    restrict_to_speech: bool = True,
) -> np.ndarray:
    """
    Times, in seconds, of detected syllable nuclei.

    `min_distance_ms` caps the maximum detectable rate: at the 90 ms default the
    ceiling is about 11 syllables/s, comfortably above any human DDK rate but
    tight enough to reject the two-part burst of a plosive release.

    Peaks are accepted on two criteria: local prominence in the smoothed
    envelope, and falling inside a region the VAD marks as speech. Both are
    local measures, which is what makes this work on a quiet recording with
    loud transients - see the note on `_PEAK_FLOOR_DB`.
    """
    p = params or DEFAULT_SIGNAL_PARAMS
    min_distance_ms = (
        p.syllable_min_distance_ms if min_distance_ms is None else min_distance_ms
    )
    prominence_db = (
        p.syllable_prominence_db if prominence_db is None else prominence_db
    )

    if audio.samples.size == 0:
        return np.zeros(0)

    db = frame_energy_db(
        audio.samples, audio.sample_rate, _ENVELOPE_FRAME_MS, _ENVELOPE_HOP_MS
    )
    if db.size < 3:
        return np.zeros(0)

    smooth_frames = max(1, int(round(p.syllable_smooth_ms / _ENVELOPE_HOP_MS)))
    env = _smooth(db, smooth_frames)

    distance = max(1, int(round(min_distance_ms / _ENVELOPE_HOP_MS)))
    peaks, _ = find_peaks(env, distance=distance, prominence=prominence_db)
    if peaks.size == 0:
        return np.zeros(0)

    # Backstop: discard anything in the deep noise floor.
    peaks = peaks[env[peaks] >= (float(np.max(env)) - _PEAK_FLOOR_DB)]
    if peaks.size == 0:
        return np.zeros(0)

    times = peaks * (_ENVELOPE_HOP_MS / 1000.0)

    if restrict_to_speech:
        # Local, level-independent acceptance: a syllable nucleus has to sit
        # inside speech. On silence this yields nothing, which is correct.
        from ..vad import detect_voice_activity

        vad = detect_voice_activity(audio.samples, audio.sample_rate, p)
        if not vad.has_speech:
            return np.zeros(0)
        tolerance = _ENVELOPE_FRAME_MS / 1000.0
        keep = np.zeros(times.size, dtype=bool)
        for start, end in vad.speech_segments:
            keep |= (times >= start - tolerance) & (times <= end + tolerance)
        times = times[keep]

    return times


@dataclass
class DdkResult:
    """Task 3 measurements."""

    n_syllables: int = 0
    syllable_rate_hz: Optional[float] = None
    span_s: float = 0.0
    recording_duration_s: float = 0.0

    mean_interval_s: Optional[float] = None
    sd_interval_s: Optional[float] = None
    rhythm_cv: Optional[float] = None
    rhythm_regularity: Optional[float] = None
    max_interval_s: Optional[float] = None
    min_interval_s: Optional[float] = None

    nuclei_s: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        def r(v, nd=3):
            return None if v is None else round(float(v), nd)

        return {
            "n_syllables": self.n_syllables,
            "syllable_rate_hz": r(self.syllable_rate_hz),
            "span_s": r(self.span_s),
            "recording_duration_s": r(self.recording_duration_s),
            "mean_interval_s": r(self.mean_interval_s, 4),
            "sd_interval_s": r(self.sd_interval_s, 4),
            "rhythm_cv": r(self.rhythm_cv, 4),
            "rhythm_regularity": r(self.rhythm_regularity, 4),
            "max_interval_s": r(self.max_interval_s, 4),
            "min_interval_s": r(self.min_interval_s, 4),
            "nuclei_s": [round(float(x), 4) for x in self.nuclei_s],
            "notes": list(self.notes),
        }


def analyse_ddk(
    audio: Audio,
    params: SignalParams | None = None,
) -> DdkResult:
    """Measure a /pa-ta-ka/ repetition recording."""
    p = params or DEFAULT_SIGNAL_PARAMS
    out = DdkResult(recording_duration_s=audio.duration_s)

    if audio.samples.size == 0:
        out.notes.append("empty recording")
        return out

    nuclei = detect_syllable_nuclei(audio, p)
    out.nuclei_s = [float(x) for x in nuclei]
    out.n_syllables = int(nuclei.size)

    if nuclei.size < 2:
        out.notes.append(
            "fewer than two syllables detected - cannot estimate a rate. "
            "Check the recording actually contains repeated /pa-ta-ka/."
        )
        return out

    out.span_s = float(nuclei[-1] - nuclei[0])
    intervals = np.diff(nuclei)

    if out.span_s > 0:
        # Inter-onset-interval rate: exact for a periodic sequence.
        out.syllable_rate_hz = float((nuclei.size - 1) / out.span_s)

    out.mean_interval_s = float(np.mean(intervals))
    out.sd_interval_s = float(np.std(intervals))
    out.max_interval_s = float(np.max(intervals))
    out.min_interval_s = float(np.min(intervals))

    if out.mean_interval_s > 0:
        out.rhythm_cv = float(out.sd_interval_s / out.mean_interval_s)
        out.rhythm_regularity = float(max(0.0, 1.0 - out.rhythm_cv))

    if nuclei.size < 8:
        out.notes.append(
            f"only {nuclei.size} syllables detected - rhythm variability needs "
            "a longer sample to be meaningful"
        )
    return out


__all__ = ["DdkResult", "analyse_ddk", "detect_syllable_nuclei"]
