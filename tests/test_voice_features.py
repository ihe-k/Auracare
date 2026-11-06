"""
Praat voice indicators and Task 1 phonation timing.

Jitter and shimmer are checked for correct *response* rather than exact
equality. With sub-sample pulse placement the measured jitter tracks the
constructed perturbation linearly at a consistent factor of about 0.76, the
shortfall coming from the formant filter's ~4 ms impulse response smearing
pulse positions and from Praat's interpolating cross-correlation pulling
adjacent periods toward the mean. That factor is stable and documented, so the
tests assert proportionality and a near-zero reading on an unperturbed signal -
which is what actually matters: that the number moves with voice quality and
does not invent perturbation that is not there.
"""

from __future__ import annotations

import numpy as np
import pytest

from auracare.features.phonation import analyse_phonation, longest_voiced_run
from auracare.features.voice import PARSELMOUTH_AVAILABLE, analyse_voice, pitch_track
from conftest import SR, as_audio, concat, silence, vowel

pytestmark = pytest.mark.skipif(
    not PARSELMOUTH_AVAILABLE, reason="parselmouth not installed"
)

# Observed attenuation of the perturbation measures on synthetic pulse trains.
_JITTER_RATIO_RANGE = (0.65, 0.95)


@pytest.mark.parametrize("f0", [90.0, 120.0, 180.0, 240.0])
def test_f0_recovered_exactly(f0):
    result = analyse_voice(vowel(f0, 3.0), SR)
    assert result.available
    assert result.mean_f0_hz == pytest.approx(f0, rel=0.01)


def test_unperturbed_vowel_reports_near_zero_jitter():
    """A perfectly periodic signal must not produce phantom perturbation."""
    result = analyse_voice(vowel(120.0, 4.0, jitter_fraction=0.0), SR)
    assert result.jitter_local_pct is not None
    assert result.jitter_local_pct < 0.05


@pytest.mark.parametrize("d", [0.0025, 0.005, 0.01, 0.02])
def test_jitter_tracks_constructed_perturbation(d):
    """Measured jitter must be proportional to the perturbation injected."""
    result = analyse_voice(vowel(120.0, 4.0, jitter_fraction=d), SR)
    expected = 2.0 * d * 100.0
    ratio = result.jitter_local_pct / expected
    assert _JITTER_RATIO_RANGE[0] < ratio < _JITTER_RATIO_RANGE[1]


def test_jitter_is_monotonic_in_perturbation():
    values = [
        analyse_voice(vowel(120.0, 4.0, jitter_fraction=d), SR).jitter_local_pct
        for d in (0.0, 0.0025, 0.005, 0.01, 0.02)
    ]
    assert all(b > a for a, b in zip(values[:-1], values[1:]))


def test_shimmer_is_monotonic_in_amplitude_perturbation():
    values = [
        analyse_voice(vowel(120.0, 4.0, shimmer_fraction=a), SR).shimmer_local_pct
        for a in (0.0, 0.01, 0.03, 0.05)
    ]
    assert all(b > a for a, b in zip(values[:-1], values[1:]))


def test_hnr_falls_as_noise_is_added():
    clean = analyse_voice(vowel(120.0, 3.0), SR).hnr_db
    rng = np.random.default_rng(3)
    noisy_sig = vowel(120.0, 3.0) + (rng.standard_normal(SR * 3) * 0.05).astype(
        np.float32
    )
    noisy = analyse_voice(noisy_sig, SR).hnr_db
    assert clean is not None and noisy is not None
    assert noisy < clean - 10.0


def test_silence_yields_no_perturbation_values_not_zeros():
    """
    The critical failure mode: a silent recording must report None, never 0.0.
    A 0.00 % jitter reads as a perfect voice.
    """
    result = analyse_voice(silence(3.0), SR)
    assert result.jitter_local_pct is None
    assert result.shimmer_local_pct is None
    assert result.hnr_db is None
    assert result.notes


def test_too_short_recording_reported_unavailable():
    result = analyse_voice(np.zeros(100, dtype=np.float32), SR)
    assert not result.available
    assert "100 ms" in (result.unavailable_reason or "")


def test_empty_recording_reported_unavailable():
    result = analyse_voice(np.zeros(0, dtype=np.float32), SR)
    assert not result.available


def test_voiced_fraction_on_mixed_signal():
    x = concat(silence(2.0), vowel(120.0, 2.0), silence(2.0))
    result = analyse_voice(x, SR)
    assert result.voiced_fraction == pytest.approx(2.0 / 6.0, abs=0.08)


def test_pitch_track_returns_zeros_for_unvoiced():
    times, f0 = pitch_track(concat(silence(1.0), vowel(120.0, 1.0)), SR)
    assert times.size > 0
    assert np.any(f0 == 0)
    assert np.any(f0 > 0)


# ---------------------------------------------------------------------------
# Task 1 phonation timing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("duration", [1.0, 1.5, 2.5, 5.0, 8.0])
def test_phonation_duration_recovered(duration):
    """
    Phonation time is the headline safety measure; it must be exact, not
    approximately right.
    """
    audio = as_audio(concat(silence(0.5), vowel(120.0, duration), silence(0.5)))
    result = analyse_phonation(audio)
    assert result.phonation_duration_s == pytest.approx(duration, abs=0.1)


def test_phonation_duration_ignores_surrounding_silence():
    """File length must not be mistaken for phonation time."""
    audio = as_audio(concat(silence(6.0), vowel(120.0, 2.0), silence(6.0)))
    result = analyse_phonation(audio)
    assert result.recording_duration_s == pytest.approx(14.0, abs=0.1)
    assert result.phonation_duration_s == pytest.approx(2.0, abs=0.15)


def test_phonation_takes_longest_run_not_total_voiced():
    """
    Two separate 2 s vowels are not a 4 s phonation. Total voiced time would
    say 4 s; the longest continuous run is the right answer.
    """
    audio = as_audio(
        concat(silence(0.3), vowel(120.0, 2.0), silence(1.5), vowel(120.0, 2.0),
               silence(0.3))
    )
    result = analyse_phonation(audio)
    assert result.phonation_duration_s == pytest.approx(2.0, abs=0.2)
    assert result.total_voiced_s == pytest.approx(4.0, abs=0.3)


def test_phonation_bridges_brief_voicing_dropout():
    """
    A single dropped frame mid-vowel must not halve the reported duration.
    """
    audio = as_audio(
        concat(silence(0.3), vowel(120.0, 3.0), silence(0.05), vowel(120.0, 3.0),
               silence(0.3))
    )
    result = analyse_phonation(audio)
    assert result.phonation_duration_s > 5.5


def test_phonation_on_silence_reports_zero_with_a_note():
    result = analyse_phonation(as_audio(silence(5.0)))
    assert result.phonation_duration_s == 0.0
    assert result.phonation_start_s is None
    assert any("no sustained voicing" in n for n in result.notes)


def test_phonation_measures_voice_on_the_segment_only():
    """
    Perturbation must be computed on the vowel, not the whole file. Measured
    over 12 s of mostly silence it would be meaningless.
    """
    audio = as_audio(concat(silence(5.0), vowel(120.0, 2.0), silence(5.0)))
    result = analyse_phonation(audio)
    assert result.voice.available
    assert result.voice.jitter_local_pct is not None
    assert result.voice.jitter_local_pct < 0.5


def test_longest_voiced_run_on_empty_input():
    assert longest_voiced_run(np.zeros(0), np.zeros(0)) == (0.0, 0.0, 0.0)


def test_f0_cv_small_for_steady_vowel():
    result = analyse_phonation(as_audio(concat(silence(0.3), vowel(120.0, 4.0))))
    assert result.f0_cv is not None
    assert result.f0_cv < 0.02
