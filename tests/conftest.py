"""
Synthetic signal generators with known ground truth.

The point of this file is that every acoustic measurement in the package can be
checked against a signal whose true value is known by construction, rather than
against a previous run of the same code. If `analyse_ddk` says 5.0 syllables/s
for a signal built with exactly 5.0 modulations per second, the detector works;
if it says 4.75, the (N-1)/span versus N/duration bug is back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auracare.audio_io import Audio  # noqa: E402

SR = 16_000


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def silence(duration_s: float, sr: int = SR, noise_db: float = -70.0) -> np.ndarray:
    """Near-silence with a realistic room floor (never exactly zero)."""
    n = int(round(duration_s * sr))
    amp = 10.0 ** (noise_db / 20.0)
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) * amp).astype(np.float32)


def _formant_filter(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Two resonances approximating the vowel /a/."""

    def reson(sig, f, bw):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        return lfilter([1 - r], [1.0, -2 * r * np.cos(theta), r * r], sig)

    return reson(reson(x, 730.0, 80.0), 1090.0, 90.0)


def pulse_train(
    f0: float,
    duration_s: float,
    sr: int = SR,
    jitter_fraction: float = 0.0,
    shimmer_fraction: float = 0.0,
) -> np.ndarray:
    """
    Glottal-style pulse train with exactly controlled perturbation.

    `jitter_fraction` d alternates period length between T(1+d) and T(1-d).
    Praat's jitter (local) is the mean absolute difference between consecutive
    periods over the mean period, so the expected reading is exactly 2d.

    `shimmer_fraction` a alternates pulse amplitude between (1+a) and (1-a),
    giving an expected shimmer (local) of 2a in amplitude ratio terms.
    """
    n = int(round(duration_s * sr))
    out = np.zeros(n, dtype=np.float64)
    period = 1.0 / f0

    # Pulses are placed with sub-sample accuracy by splitting each impulse
    # across the two neighbouring samples. Rounding to the nearest sample
    # instead would quantise every period to a multiple of 1/sr, which at
    # 16 kHz injects about 0.4 % jitter of its own - enough to swamp the
    # perturbation being tested.
    t = 0.0
    k = 0
    while True:
        sign = 1.0 if (k % 2 == 0) else -1.0
        this_period = period * (1.0 + sign * jitter_fraction)
        pos = t * sr
        i0 = int(np.floor(pos))
        if i0 >= n - 1:
            break
        frac = pos - i0
        amp = 1.0 + sign * shimmer_fraction
        out[i0] += amp * (1.0 - frac)
        out[i0 + 1] += amp * frac
        t += this_period
        k += 1

    voiced = _formant_filter(out, sr)
    peak = np.max(np.abs(voiced))
    if peak > 0:
        voiced = voiced / peak * 0.5
    return voiced.astype(np.float32)


def vowel(
    f0: float = 120.0,
    duration_s: float = 5.0,
    sr: int = SR,
    jitter_fraction: float = 0.0,
    shimmer_fraction: float = 0.0,
) -> np.ndarray:
    """A sustained vowel."""
    return pulse_train(f0, duration_s, sr, jitter_fraction, shimmer_fraction)


def am_syllable_train(
    rate_hz: float,
    duration_s: float,
    sr: int = SR,
    f0: float = 120.0,
    depth: float = 0.95,
) -> np.ndarray:
    """
    Voiced carrier amplitude-modulated at exactly `rate_hz`.

    Produces `rate_hz * duration_s` envelope maxima, which is the ground truth
    for the syllable-nucleus detector.
    """
    carrier = pulse_train(f0, duration_s, sr)
    t = np.arange(carrier.size) / sr
    # Peaks at t = 0.25/rate, then every 1/rate.
    env = 0.5 * (1.0 - np.cos(2 * np.pi * rate_hz * t - np.pi))
    env = (1.0 - depth) + depth * env
    return (carrier * env).astype(np.float32)


def noise_burst(
    duration_s: float,
    sr: int = SR,
    rise_s: float = 0.08,
    lowpass_hz: float = 3000.0,
    tone_hz: float | None = None,
    tone_level: float = 0.35,
    seed: int = 1,
) -> np.ndarray:
    """
    A forced-exhalation-like noise burst with a fast rise and slower decay.

    `tone_hz` adds a sustained sinusoid to simulate a wheeze.
    """
    n = int(round(duration_s * sr))
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)

    # Simple one-pole lowpass to shape the breath spectrum.
    a = np.exp(-2.0 * np.pi * lowpass_hz / sr)
    x = lfilter([1 - a], [1.0, -a], x)

    t = np.arange(n) / sr
    rise_n = max(1, int(round(rise_s * sr)))
    env = np.ones(n)
    env[:rise_n] = np.linspace(0.0, 1.0, rise_n)
    decay_tau = max(1e-3, (duration_s - rise_s) / 2.5)
    env[rise_n:] = np.exp(-(t[rise_n:] - t[rise_n]) / decay_tau)

    y = x * env
    if tone_hz is not None:
        y = y + tone_level * np.sin(2 * np.pi * tone_hz * t) * env

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.6
    return y.astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])


def as_audio(samples: np.ndarray, sr: int = SR, name: str = "synthetic") -> Audio:
    return Audio(samples=np.asarray(samples, dtype=np.float32), sample_rate=sr,
                 source_name=name)


def speech_with_pauses(
    speech_blocks_s,
    pause_blocks_s,
    sr: int = SR,
    lead_s: float = 0.0,
    trail_s: float = 0.0,
) -> np.ndarray:
    """
    Alternating speech and silence with exactly known durations.

    Starts and ends with speech, so the active span is unambiguous. `lead_s`
    and `trail_s` add silence outside the active span, which must NOT count
    towards the pause ratio.
    """
    parts = [silence(lead_s, sr)] if lead_s > 0 else []
    for i, sp in enumerate(speech_blocks_s):
        parts.append(am_syllable_train(4.0, sp, sr))
        if i < len(pause_blocks_s):
            parts.append(silence(pause_blocks_s[i], sr))
    if trail_s > 0:
        parts.append(silence(trail_s, sr))
    return concat(*parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sr() -> int:
    return SR


@pytest.fixture(scope="session")
def clean_vowel_5s() -> Audio:
    return as_audio(concat(silence(0.4), vowel(120.0, 5.0), silence(0.4)))


@pytest.fixture(scope="session")
def short_vowel_1_5s() -> Audio:
    """Below the 2.5 s respiratory safety threshold."""
    return as_audio(concat(silence(0.3), vowel(130.0, 1.5), silence(0.3)))


@pytest.fixture(scope="session")
def ddk_5hz_4s() -> Audio:
    return as_audio(am_syllable_train(5.0, 4.0))


@pytest.fixture(scope="session")
def exhalation_1_2s() -> Audio:
    return as_audio(concat(silence(0.3), noise_burst(1.2), silence(0.3)))


@pytest.fixture(scope="session")
def triage_repo() -> Path | None:
    from auracare.paths import find_triage_repo

    return find_triage_repo()
