"""
Tasks 2, 3 and 4 feature extraction.

Task 2's exhalation duration is checked against an analytically derived value
rather than the nominal burst length, because the measure is defined as "time
spent within 12 dB of peak" and the synthetic burst decays exponentially. For a
burst with rise r and decay constant tau, the -12 dB crossing is at
r + tau * (12/20) * ln(10), so that is what the detector should report. Testing
against the nominal length instead would be testing the wrong thing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from auracare.features.ddk import analyse_ddk, detect_syllable_nuclei
from auracare.features.expiratory import analyse_expiratory, wheeze_index
from auracare.features.fluency import analyse_fluency
from auracare.stt import Transcript, analyse_language
from conftest import (
    SR,
    am_syllable_train,
    as_audio,
    concat,
    noise_burst,
    silence,
    speech_with_pauses,
    vowel,
)


def _expected_burst_duration(nominal_s: float, rise_s: float = 0.08) -> float:
    tau = max(1e-3, (nominal_s - rise_s) / 2.5)
    return rise_s + tau * (12.0 / 20.0) * math.log(10.0)


# ---------------------------------------------------------------------------
# Task 3: diadochokinetic rate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rate", [2.0, 3.0, 4.0, 5.0, 6.5, 8.0])
def test_ddk_rate_recovered_exactly(rate):
    """
    This is the test that catches the N/duration versus (N-1)/span bias. At
    5 Hz over 4 s the biased estimator reads about 4.75 Hz, which would cross a
    5 Hz clinical threshold in the wrong direction.
    """
    result = analyse_ddk(as_audio(am_syllable_train(rate, 4.0)))
    assert result.syllable_rate_hz == pytest.approx(rate, rel=0.02)


@pytest.mark.parametrize("rate,duration", [(5.0, 2.0), (5.0, 4.0), (5.0, 8.0)])
def test_ddk_rate_independent_of_recording_length(rate, duration):
    result = analyse_ddk(as_audio(am_syllable_train(rate, duration)))
    assert result.syllable_rate_hz == pytest.approx(rate, rel=0.03)


def test_ddk_syllable_count_matches_construction():
    result = analyse_ddk(as_audio(am_syllable_train(5.0, 4.0)))
    # 5 Hz over 4 s gives 20 modulation peaks; the first may fall at t=0.05.
    assert 18 <= result.n_syllables <= 21


def test_ddk_rhythm_cv_near_zero_for_metronomic_input():
    result = analyse_ddk(as_audio(am_syllable_train(5.0, 6.0)))
    assert result.rhythm_cv is not None
    assert result.rhythm_cv < 0.05
    assert result.rhythm_regularity > 0.95


def test_ddk_rhythm_cv_rises_for_irregular_input():
    """Concatenating two different rates must raise measured variability."""
    irregular = concat(
        am_syllable_train(3.0, 2.0), am_syllable_train(7.0, 2.0)
    )
    steady = analyse_ddk(as_audio(am_syllable_train(5.0, 4.0)))
    uneven = analyse_ddk(as_audio(irregular))
    assert uneven.rhythm_cv > steady.rhythm_cv + 0.1


def test_ddk_on_silence_reports_nothing_measurable():
    result = analyse_ddk(as_audio(silence(4.0)))
    assert result.n_syllables < 2
    assert result.syllable_rate_hz is None
    assert any("fewer than two" in n for n in result.notes)


def test_ddk_on_sustained_vowel_finds_no_syllable_structure():
    """A steady vowel has no amplitude modulation, so no nuclei."""
    result = analyse_ddk(as_audio(vowel(120.0, 4.0)))
    assert result.n_syllables <= 2


def test_detect_syllable_nuclei_returns_times_in_range():
    nuclei = detect_syllable_nuclei(as_audio(am_syllable_train(5.0, 3.0)))
    assert nuclei.size > 10
    assert nuclei.min() >= 0.0
    assert nuclei.max() <= 3.0


# ---------------------------------------------------------------------------
# Task 2: forced exhalation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nominal", [0.3, 0.8, 1.5, 2.5])
def test_exhalation_duration_matches_analytic_expectation(nominal):
    audio = as_audio(concat(silence(0.4), noise_burst(nominal), silence(0.4)))
    result = analyse_expiratory(audio)
    expected = _expected_burst_duration(nominal)
    assert result.exhalation_duration_s == pytest.approx(expected, rel=0.15)


def test_exhalation_rise_time_detected():
    audio = as_audio(concat(silence(0.4), noise_burst(1.5, rise_s=0.08)))
    result = analyse_expiratory(audio)
    assert result.rise_time_s == pytest.approx(0.08, abs=0.05)


def test_exhalation_slow_rise_is_distinguished_from_fast():
    fast = analyse_expiratory(
        as_audio(concat(silence(0.4), noise_burst(1.5, rise_s=0.05)))
    )
    slow = analyse_expiratory(
        as_audio(concat(silence(0.4), noise_burst(1.5, rise_s=0.50)))
    )
    assert slow.rise_time_s > fast.rise_time_s + 0.2


def test_exhalation_decay_slope_is_negative():
    result = analyse_expiratory(as_audio(concat(silence(0.4), noise_burst(1.5))))
    assert result.decay_slope_db_s is not None
    assert result.decay_slope_db_s < 0.0


def test_wheeze_index_separates_tonal_from_broadband():
    plain = analyse_expiratory(
        as_audio(concat(silence(0.3), noise_burst(1.2), silence(0.3)))
    )
    wheezy = analyse_expiratory(
        as_audio(
            concat(
                silence(0.3),
                noise_burst(1.2, tone_hz=700.0, tone_level=0.5),
                silence(0.3),
            )
        )
    )
    assert plain.wheeze_index_db < 8.0
    assert wheezy.wheeze_index_db > 14.0
    assert wheezy.wheeze_peak_hz == pytest.approx(700.0, abs=25.0)


@pytest.mark.parametrize("tone_hz", [450.0, 700.0, 1200.0])
def test_wheeze_peak_frequency_localised(tone_hz):
    burst = noise_burst(1.2, tone_hz=tone_hz, tone_level=0.5)
    prominence, peak = wheeze_index(burst, SR)
    assert prominence is not None and prominence > 12.0
    assert peak == pytest.approx(tone_hz, rel=0.06)


def test_exhalation_on_silence_reports_no_burst():
    result = analyse_expiratory(as_audio(silence(2.0)))
    assert result.exhalation_duration_s == 0.0
    assert any("no clear exhalation" in n for n in result.notes)


def test_exhalation_takes_run_containing_peak_not_a_later_cough():
    """A second transient must not extend the measured manoeuvre."""
    audio = as_audio(
        concat(
            silence(0.3),
            noise_burst(1.0, seed=2),
            silence(2.0),
            noise_burst(0.2, seed=3) * 0.4,
            silence(0.3),
        )
    )
    result = analyse_expiratory(audio)
    assert result.exhalation_duration_s < 1.2


# ---------------------------------------------------------------------------
# Task 4: connected speech
# ---------------------------------------------------------------------------
def test_fluency_pause_ratio_matches_construction():
    audio = as_audio(
        speech_with_pauses([3.0, 3.0], [2.0], lead_s=1.0, trail_s=1.0)
    )
    result = analyse_fluency(audio)
    assert result.has_speech
    assert result.pause_ratio == pytest.approx(0.25, abs=0.03)
    assert result.n_pauses == 1


def test_fluency_separates_speech_rate_from_articulation_rate():
    """
    Someone who articulates normally but pauses heavily should show a normal
    articulation rate and a depressed speech rate. Collapsing the two loses
    exactly the distinction that matters.
    """
    audio = as_audio(
        speech_with_pauses([2.0, 2.0], [4.0], lead_s=0.5, trail_s=0.5)
    )
    result = analyse_fluency(audio)
    assert result.articulation_rate_sylls is not None
    assert result.speech_rate_sylls is not None
    assert result.articulation_rate_sylls > result.speech_rate_sylls * 1.5


def test_fluency_without_transcript_still_measures_acoustics():
    audio = as_audio(speech_with_pauses([3.0, 3.0], [1.0]))
    result = analyse_fluency(audio, transcript=None)
    assert result.has_speech
    assert result.speech_rate_sylls is not None
    assert result.n_words == 0
    assert result.lexical_diversity is None
    assert any("content measures skipped" in n for n in result.notes)


def test_fluency_with_transcript_adds_content_measures():
    audio = as_audio(speech_with_pauses([4.0, 4.0], [1.0]))
    transcript = Transcript(
        text=(
            "I can see a park with children playing on the grass and a dog "
            "running near the water while people sit on benches"
        ),
        available=True,
        backend="manual",
    )
    result = analyse_fluency(audio, transcript=transcript)
    assert result.n_words > 15
    assert result.lexical_diversity is not None
    assert result.words_per_minute is not None
    assert result.language.analysed


def test_fluency_on_silence_reports_no_speech():
    result = analyse_fluency(as_audio(silence(10.0)))
    assert not result.has_speech
    assert any("no speech detected" in n for n in result.notes)


def test_fluency_flags_negative_language_from_transcript():
    audio = as_audio(speech_with_pauses([4.0, 4.0], [1.0]))
    transcript = Transcript(
        text=(
            "I feel hopeless and worthless, nothing matters any more and I am "
            "so tired of everything"
        ),
        available=True,
        backend="manual",
    )
    result = analyse_fluency(audio, transcript=transcript)
    assert result.language.negative_ratio > 0.05
    assert "hopelessness" in result.language.category_counts
    assert "worthlessness" in result.language.category_counts
    assert not result.language.safety_critical
