"""
Parselmouth / Praat voice indicators.

Wraps the standard Praat measurement recipe (pitch -> point process -> jitter /
shimmer, plus harmonicity and intensity) with the guards that recipe normally
lacks: Praat returns *undefined* rather than an error when a recording has too
few glottal periods to measure, and that undefined value arrives in Python as
NaN. Letting a NaN through would surface in the UI as a confident-looking
"0.00 %" jitter, so every value is validated and missing ones stay ``None``.

All measures here are experimental voice indicators, not diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import DEFAULT_SIGNAL_PARAMS, SignalParams

try:  # pragma: no cover - import guard exercised only without the dependency
    import parselmouth
    from parselmouth.praat import call as praat_call

    PARSELMOUTH_AVAILABLE = True
    PARSELMOUTH_ERROR: Optional[str] = None
except Exception as _exc:  # pragma: no cover
    parselmouth = None  # type: ignore[assignment]
    praat_call = None  # type: ignore[assignment]
    PARSELMOUTH_AVAILABLE = False
    PARSELMOUTH_ERROR = str(_exc)


# Praat's own defaults for the period-based measures.
_PERIOD_FLOOR = 0.0001
_PERIOD_CEILING = 0.02
_MAX_PERIOD_FACTOR = 1.3
_MAX_AMPLITUDE_FACTOR = 1.6

# Below this harmonics-to-noise ratio, jitter and shimmer describe the room
# rather than the voice: period boundaries cannot be located accurately in
# noise, so the perturbation measures inflate. Praat's published norms
# (jitter < 1.04 %, shimmer < 3.81 %) come from clean studio recordings, and
# applying them to a laptop microphone in an ordinary room grades almost
# everyone abnormal. 10 dB is a conservative floor - a healthy sustained vowel
# recorded properly sits above 20 dB.
_MIN_HNR_FOR_PERTURBATION_DB = 10.0


def _clean(value) -> Optional[float]:
    """Praat 'undefined' -> None. Also traps inf and the -200 dB HNR sentinel."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


@dataclass
class VoiceIndicators:
    """Praat voice measures for one recording (or one segment of one)."""

    available: bool = False
    unavailable_reason: Optional[str] = None

    # Pitch
    mean_f0_hz: Optional[float] = None
    sd_f0_hz: Optional[float] = None
    min_f0_hz: Optional[float] = None
    max_f0_hz: Optional[float] = None
    f0_cv: Optional[float] = None
    voiced_fraction: Optional[float] = None

    # Perturbation
    jitter_local_pct: Optional[float] = None
    jitter_rap_pct: Optional[float] = None
    jitter_ppq5_pct: Optional[float] = None
    shimmer_local_pct: Optional[float] = None
    shimmer_apq3_pct: Optional[float] = None
    shimmer_apq11_pct: Optional[float] = None

    # Noise and level
    hnr_db: Optional[float] = None
    mean_intensity_db: Optional[float] = None

    n_pulses: Optional[int] = None
    # False when the recording is too noisy for jitter/shimmer to mean
    # anything. The values are still reported, but must not be graded.
    perturbation_reliable: bool = True
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        def r(v, nd=3):
            return None if v is None else round(float(v), nd)

        return {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "mean_f0_hz": r(self.mean_f0_hz, 2),
            "sd_f0_hz": r(self.sd_f0_hz, 2),
            "min_f0_hz": r(self.min_f0_hz, 2),
            "max_f0_hz": r(self.max_f0_hz, 2),
            "f0_cv": r(self.f0_cv, 4),
            "voiced_fraction": r(self.voiced_fraction, 4),
            "jitter_local_pct": r(self.jitter_local_pct),
            "jitter_rap_pct": r(self.jitter_rap_pct),
            "jitter_ppq5_pct": r(self.jitter_ppq5_pct),
            "shimmer_local_pct": r(self.shimmer_local_pct),
            "shimmer_apq3_pct": r(self.shimmer_apq3_pct),
            "shimmer_apq11_pct": r(self.shimmer_apq11_pct),
            "hnr_db": r(self.hnr_db, 2),
            "mean_intensity_db": r(self.mean_intensity_db, 2),
            "n_pulses": self.n_pulses,
            "perturbation_reliable": self.perturbation_reliable,
            "notes": list(self.notes),
        }


def _to_sound(samples: np.ndarray, sample_rate: int):
    """Build a parselmouth Sound from a float array."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return parselmouth.Sound(arr, sampling_frequency=float(sample_rate))


def pitch_track(
    samples: np.ndarray,
    sample_rate: int,
    params: SignalParams | None = None,
    time_step: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (times_s, f0_hz) with 0.0 at unvoiced frames.

    Used by the Task 1 phonation-duration measurement, which needs voicing
    continuity rather than a single summary number.
    """
    p = params or DEFAULT_SIGNAL_PARAMS
    if not PARSELMOUTH_AVAILABLE or np.asarray(samples).size == 0:
        return np.zeros(0), np.zeros(0)

    snd = _to_sound(samples, sample_rate)
    try:
        pitch = praat_call(snd, "To Pitch", time_step, p.f0_min_hz, p.f0_max_hz)
    except Exception:
        return np.zeros(0), np.zeros(0)

    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    times = np.asarray(pitch.xs(), dtype=np.float64)
    return times, f0


def analyse_voice(
    samples: np.ndarray,
    sample_rate: int,
    params: SignalParams | None = None,
) -> VoiceIndicators:
    """
    Run the full Praat measurement set over `samples`.

    Expects a mostly-voiced input for the perturbation measures to be
    meaningful - call it on the sustained vowel, or on the voiced portion of
    connected speech, not on a breath manoeuvre.
    """
    p = params or DEFAULT_SIGNAL_PARAMS

    if not PARSELMOUTH_AVAILABLE:
        return VoiceIndicators(
            available=False,
            unavailable_reason=f"parselmouth not importable: {PARSELMOUTH_ERROR}",
        )

    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return VoiceIndicators(available=False, unavailable_reason="empty recording")

    min_samples = int(sample_rate * 0.1)
    if samples.size < min_samples:
        return VoiceIndicators(
            available=False,
            unavailable_reason=f"recording shorter than 100 ms ({samples.size} samples)",
        )

    out = VoiceIndicators(available=True)
    snd = _to_sound(samples, sample_rate)

    # -- pitch --------------------------------------------------------------
    try:
        pitch = praat_call(snd, "To Pitch", 0.0, p.f0_min_hz, p.f0_max_hz)
        out.mean_f0_hz = _clean(praat_call(pitch, "Get mean", 0, 0, "Hertz"))
        out.sd_f0_hz = _clean(
            praat_call(pitch, "Get standard deviation", 0, 0, "Hertz")
        )
        out.min_f0_hz = _clean(
            praat_call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        )
        out.max_f0_hz = _clean(
            praat_call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        )

        f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
        f0 = np.nan_to_num(f0, nan=0.0)
        if f0.size:
            out.voiced_fraction = float(np.mean(f0 > 0))
        if out.mean_f0_hz and out.sd_f0_hz is not None and out.mean_f0_hz > 0:
            out.f0_cv = float(out.sd_f0_hz / out.mean_f0_hz)
    except Exception as exc:
        out.notes.append(f"pitch analysis failed: {exc}")

    # -- perturbation via point process -------------------------------------
    try:
        point_process = praat_call(
            snd, "To PointProcess (periodic, cc)", p.f0_min_hz, p.f0_max_hz
        )
        n_pulses = _clean(praat_call(point_process, "Get number of points"))
        out.n_pulses = int(n_pulses) if n_pulses is not None else None

        # Jitter needs at least a handful of consecutive periods; ppq5 needs 5.
        if out.n_pulses is not None and out.n_pulses >= 3:
            jit_args = (0, 0, _PERIOD_FLOOR, _PERIOD_CEILING, _MAX_PERIOD_FACTOR)
            local = _clean(praat_call(point_process, "Get jitter (local)", *jit_args))
            rap = _clean(praat_call(point_process, "Get jitter (rap)", *jit_args))
            ppq5 = _clean(praat_call(point_process, "Get jitter (ppq5)", *jit_args))
            out.jitter_local_pct = None if local is None else local * 100.0
            out.jitter_rap_pct = None if rap is None else rap * 100.0
            out.jitter_ppq5_pct = None if ppq5 is None else ppq5 * 100.0

            shim_args = (
                0,
                0,
                _PERIOD_FLOOR,
                _PERIOD_CEILING,
                _MAX_PERIOD_FACTOR,
                _MAX_AMPLITUDE_FACTOR,
            )
            s_local = _clean(
                praat_call([snd, point_process], "Get shimmer (local)", *shim_args)
            )
            s_apq3 = _clean(
                praat_call([snd, point_process], "Get shimmer (apq3)", *shim_args)
            )
            s_apq11 = _clean(
                praat_call([snd, point_process], "Get shimmer (apq11)", *shim_args)
            )
            out.shimmer_local_pct = None if s_local is None else s_local * 100.0
            out.shimmer_apq3_pct = None if s_apq3 is None else s_apq3 * 100.0
            out.shimmer_apq11_pct = None if s_apq11 is None else s_apq11 * 100.0
        else:
            out.notes.append(
                "too few glottal pulses for jitter/shimmer "
                f"(found {out.n_pulses}); needs sustained voicing"
            )
    except Exception as exc:
        out.notes.append(f"perturbation analysis failed: {exc}")

    # -- harmonicity --------------------------------------------------------
    # Gated on detected voicing. Praat happily returns a plausible-looking
    # negative HNR for a recording of room noise - noise does have some
    # autocorrelation structure - and -200 dB is only used when every frame is
    # undefined. Reporting "-5.8 dB" for silence would look like a measurement
    # of a very breathy voice rather than the absence of one, so harmonicity is
    # only published when the pitch tracker and the point process agree that
    # something was actually voiced.
    try:
        harmonicity = praat_call(
            snd, "To Harmonicity (cc)", 0.01, p.f0_min_hz, 0.1, 1.0
        )
        hnr = _clean(praat_call(harmonicity, "Get mean", 0, 0))
        has_voicing = bool(out.n_pulses and out.n_pulses >= 3) or bool(
            out.voiced_fraction and out.voiced_fraction > 0.05
        )
        if hnr is None or hnr <= -199.0:
            out.hnr_db = None
            out.notes.append("harmonicity undefined (no voiced frames)")
        elif not has_voicing:
            out.hnr_db = None
            out.notes.append(
                "harmonicity withheld: no voicing detected, so the value would "
                "describe background noise rather than a voice"
            )
        else:
            out.hnr_db = hnr
    except Exception as exc:
        out.notes.append(f"harmonicity analysis failed: {exc}")

    # -- is the signal clean enough for perturbation to mean anything? ------
    if out.hnr_db is None:
        # Cleanliness could not be verified, so perturbation cannot be trusted
        # either. Conservative is correct here: the cost of withholding a
        # number is a prompt to re-record, the cost of publishing a bad one is
        # a false abnormal result.
        out.perturbation_reliable = False
    elif out.hnr_db < _MIN_HNR_FOR_PERTURBATION_DB:
        out.perturbation_reliable = False
        out.notes.append(
            f"harmonics-to-noise ratio is only {out.hnr_db:.1f} dB, below the "
            f"{_MIN_HNR_FOR_PERTURBATION_DB:.0f} dB needed for jitter and "
            "shimmer to be meaningful. The values are reported but not graded "
            "- move closer to the microphone, reduce background noise, and "
            "hold one steady note."
        )

    # -- intensity ----------------------------------------------------------
    try:
        # "To Intensity" needs at least 6.4 / minimum_pitch seconds of signal.
        min_needed = 6.4 / p.f0_min_hz
        if samples.size / sample_rate > min_needed:
            intensity = praat_call(snd, "To Intensity", p.f0_min_hz, 0.0, "yes")
            out.mean_intensity_db = _clean(
                praat_call(intensity, "Get mean", 0, 0, "energy")
            )
        else:
            out.notes.append("recording too short for intensity analysis")
    except Exception as exc:
        out.notes.append(f"intensity analysis failed: {exc}")

    return out


__all__ = [
    "PARSELMOUTH_AVAILABLE",
    "PARSELMOUTH_ERROR",
    "_MIN_HNR_FOR_PERTURBATION_DB",
    "VoiceIndicators",
    "analyse_voice",
    "pitch_track",
]
