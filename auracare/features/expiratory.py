"""
Task 2 - Expiratory airway dynamics: forced exhalation.

A forced exhalation is turbulent noise, not voiced sound, so none of the
pitch-based machinery applies. What is measurable acoustically:

* onset and duration of the manoeuvre, from the energy envelope;
* rise time to peak intensity - a proxy for how briskly flow is generated;
* the post-peak decay slope - a proxy for flow deceleration;
* a "wheeze index": the prominence, in dB, of the strongest narrow tonal peak
  over the surrounding broadband spectrum. Wheeze and stridor are periodic
  oscillations of the airway wall, so they appear as a tonal peak sitting above
  the noise floor of otherwise broadband breath sound.

Honest limits, stated here and in the UI: these are acoustic proxies on an
uncalibrated consumer microphone. They are not spirometry. There is no valid
way to recover litres per second from an uncalibrated recording, so no number
here is labelled as a peak flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch

from ..audio_io import Audio
from ..config import DEFAULT_SIGNAL_PARAMS, SignalParams
from ..vad import frame_energy_db

# Band searched for wheeze/stridor tonality.
_WHEEZE_BAND_HZ = (300.0, 2000.0)
# Width of the median window used as the local broadband reference, in Hz.
_WHEEZE_REFERENCE_SPAN_HZ = 400.0


@dataclass
class ExpiratoryResult:
    """Task 2 measurements."""

    exhalation_duration_s: float = 0.0
    onset_s: Optional[float] = None
    peak_s: Optional[float] = None
    rise_time_s: Optional[float] = None
    decay_slope_db_s: Optional[float] = None
    peak_level_db: Optional[float] = None
    recording_duration_s: float = 0.0

    wheeze_index_db: Optional[float] = None
    wheeze_peak_hz: Optional[float] = None
    spectral_centroid_hz: Optional[float] = None
    high_freq_ratio: Optional[float] = None

    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        def r(v, nd=3):
            return None if v is None else round(float(v), nd)

        return {
            "exhalation_duration_s": r(self.exhalation_duration_s),
            "onset_s": r(self.onset_s),
            "peak_s": r(self.peak_s),
            "rise_time_s": r(self.rise_time_s),
            "decay_slope_db_s": r(self.decay_slope_db_s, 2),
            "peak_level_db": r(self.peak_level_db, 2),
            "recording_duration_s": r(self.recording_duration_s),
            "wheeze_index_db": r(self.wheeze_index_db, 2),
            "wheeze_peak_hz": r(self.wheeze_peak_hz, 1),
            "spectral_centroid_hz": r(self.spectral_centroid_hz, 1),
            "high_freq_ratio": r(self.high_freq_ratio, 4),
            "notes": list(self.notes),
        }


def _envelope_db(
    audio: Audio, params: SignalParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Frame-level energy envelope in dB plus its time axis."""
    db = frame_energy_db(audio.samples, audio.sample_rate, params.frame_ms, params.hop_ms)
    t = np.arange(db.size) * (params.hop_ms / 1000.0)
    return t, db


def wheeze_index(
    samples: np.ndarray,
    sample_rate: int,
    band_hz: Tuple[float, float] = _WHEEZE_BAND_HZ,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Prominence in dB of the strongest tonal peak over the local broadband level.

    Returns (prominence_db, peak_frequency_hz). Broadband noise gives a small
    value because every bin sits near its own neighbourhood median; a sustained
    whistle gives a large one.
    """
    if samples.size < 512:
        return None, None

    nperseg = min(2048, max(256, 2 ** int(np.floor(np.log2(samples.size / 4)))))
    freqs, psd = welch(
        samples.astype(np.float64), fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2
    )
    if freqs.size < 8:
        return None, None

    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if band.sum() < 8:
        return None, None

    bin_hz = float(freqs[1] - freqs[0])
    half = max(2, int(round(_WHEEZE_REFERENCE_SPAN_HZ / (2 * bin_hz))))

    best_prom = -np.inf
    best_f = None
    band_idx = np.flatnonzero(band)
    for i in band_idx:
        lo = max(0, i - half)
        hi = min(psd_db.size, i + half + 1)
        neighbourhood = np.concatenate([psd_db[lo:i], psd_db[i + 1 : hi]])
        if neighbourhood.size < 4:
            continue
        prom = float(psd_db[i] - np.median(neighbourhood))
        if prom > best_prom:
            best_prom = prom
            best_f = float(freqs[i])

    if not np.isfinite(best_prom):
        return None, None
    return best_prom, best_f


def _spectral_shape(
    samples: np.ndarray, sample_rate: int
) -> Tuple[Optional[float], Optional[float]]:
    """Spectral centroid, and the share of energy above 1 kHz."""
    if samples.size < 512:
        return None, None
    nperseg = min(2048, max(256, 2 ** int(np.floor(np.log2(samples.size / 4)))))
    freqs, psd = welch(
        samples.astype(np.float64), fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2
    )
    total = float(np.sum(psd))
    if total <= 0:
        return None, None
    centroid = float(np.sum(freqs * psd) / total)
    hf = float(np.sum(psd[freqs > 1000.0]) / total)
    return centroid, hf


def analyse_expiratory(
    audio: Audio,
    params: SignalParams | None = None,
    active_drop_db: float = 12.0,
) -> ExpiratoryResult:
    """
    Measure a forced-exhalation recording.

    `active_drop_db` sets where the manoeuvre is considered to start and end:
    frames within this many dB of the peak. 12 dB corresponds to roughly a
    quarter of peak amplitude, chosen to be robust to room noise while still
    capturing the bulk of the breath.
    """
    p = params or DEFAULT_SIGNAL_PARAMS
    out = ExpiratoryResult(recording_duration_s=audio.duration_s)

    if audio.samples.size == 0:
        out.notes.append("empty recording")
        return out

    t, db = _envelope_db(audio, p)
    if db.size < 3:
        out.notes.append("recording too short to measure an exhalation")
        return out

    peak_db = float(np.max(db))
    peak_i = int(np.argmax(db))
    noise_floor = float(np.percentile(db, 10.0))

    if (peak_db - noise_floor) < 6.0:
        out.notes.append(
            "no clear exhalation burst found - level never rises more than 6 dB "
            "above the room floor"
        )
        return out

    threshold = peak_db - active_drop_db
    active = db >= threshold

    # Take the contiguous active run containing the peak, so a cough elsewhere
    # in the recording does not extend the measured manoeuvre.
    start_i = peak_i
    while start_i > 0 and active[start_i - 1]:
        start_i -= 1
    end_i = peak_i
    while end_i < active.size - 1 and active[end_i + 1]:
        end_i += 1

    hop_s = p.hop_ms / 1000.0
    out.onset_s = float(t[start_i])
    out.peak_s = float(t[peak_i])
    out.exhalation_duration_s = float((end_i - start_i + 1) * hop_s)
    out.rise_time_s = float(t[peak_i] - t[start_i])
    out.peak_level_db = peak_db

    # Decay: linear fit from peak to the end of the active run.
    if end_i - peak_i >= 2:
        seg_t = t[peak_i : end_i + 1]
        seg_db = db[peak_i : end_i + 1]
        out.decay_slope_db_s = float(np.polyfit(seg_t, seg_db, 1)[0])
    else:
        out.notes.append("post-peak segment too short for a decay estimate")

    # Spectral measures over the active portion only.
    seg = audio.slice_seconds(t[start_i], t[end_i] + hop_s)
    prom, peak_hz = wheeze_index(seg.samples, seg.sample_rate)
    out.wheeze_index_db = prom
    out.wheeze_peak_hz = peak_hz
    centroid, hf = _spectral_shape(seg.samples, seg.sample_rate)
    out.spectral_centroid_hz = centroid
    out.high_freq_ratio = hf

    if out.exhalation_duration_s < 0.2:
        out.notes.append(
            "exhalation shorter than 200 ms - check the microphone was not "
            "blocked or the recording cut short"
        )
    return out


__all__ = ["ExpiratoryResult", "analyse_expiratory", "wheeze_index"]
