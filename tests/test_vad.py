"""
Voice activity detection against constructed ground truth.

The pause ratio is the measure the protocol escalates on, so these tests pin
down the two behaviours that a naive implementation gets wrong: excluding
silence outside the active span, and not counting articulatory gaps as pauses.
"""

from __future__ import annotations

import numpy as np
import pytest

from auracare.config import SignalParams
from auracare.vad import detect_voice_activity, frame_signal
from conftest import (
    SR,
    am_syllable_train,
    as_audio,
    concat,
    silence,
    speech_with_pauses,
    vowel,
)


@pytest.mark.parametrize(
    "speech,pauses",
    [
        ([3.0, 3.0], [2.0]),
        ([2.0, 2.0, 2.0], [1.0, 1.0]),
        ([5.0], []),
        ([1.0, 1.0], [4.0]),
        ([4.0, 1.0, 1.0], [0.5, 2.5]),
    ],
)
def test_pause_ratio_matches_construction(speech, pauses):
    """Measured pause ratio must match the ratio the signal was built with."""
    x = speech_with_pauses(speech, pauses, lead_s=1.5, trail_s=1.5)
    result = detect_voice_activity(x, SR)

    span = sum(speech) + sum(pauses)
    expected = sum(pauses) / span

    assert result.has_speech
    assert result.pause_ratio == pytest.approx(expected, abs=0.02)
    assert result.n_pauses == len(pauses)
    assert result.active_span_s == pytest.approx(span, abs=0.1)


def test_leading_and_trailing_silence_excluded_from_pause_ratio():
    """
    Silence before the first word and after the last is reaction time, not a
    pause. Counting it would flag anyone slow to start.
    """
    speech, pauses = [2.0, 2.0], [1.0]
    tight = detect_voice_activity(
        speech_with_pauses(speech, pauses, lead_s=0.0, trail_s=0.0), SR
    )
    padded = detect_voice_activity(
        speech_with_pauses(speech, pauses, lead_s=6.0, trail_s=6.0), SR
    )

    assert padded.pause_ratio == pytest.approx(tight.pause_ratio, abs=0.02)
    assert padded.total_duration_s > tight.total_duration_s + 10.0
    # The naive calculation would have been badly inflated.
    naive = (padded.total_duration_s - padded.speech_time_s) / padded.total_duration_s
    assert naive > padded.pause_ratio + 0.4


def test_short_gaps_are_not_counted_as_pauses():
    """
    Sub-250 ms gaps are stop closures, not disfluencies. A detector that counts
    them puts every healthy speaker over the 45 % threshold.
    """
    # 4 Hz syllable train: inter-syllable gaps around 100-150 ms throughout.
    result = detect_voice_activity(am_syllable_train(4.0, 8.0), SR)

    assert result.has_speech
    assert result.n_pauses == 0
    assert result.pause_ratio < 0.10


def test_silence_only_reports_no_speech():
    result = detect_voice_activity(silence(5.0), SR)
    assert not result.has_speech
    assert result.pause_ratio == 0.0
    assert result.n_pauses == 0


def test_empty_input_is_safe():
    result = detect_voice_activity(np.zeros(0, dtype=np.float32), SR)
    assert not result.has_speech
    assert result.pause_ratio == 0.0
    assert result.total_duration_s == 0.0


def test_longest_pause_identified():
    x = speech_with_pauses([1.0, 1.0, 1.0], [0.5, 3.0], lead_s=0.5, trail_s=0.5)
    result = detect_voice_activity(x, SR)
    assert result.longest_pause_s == pytest.approx(3.0, abs=0.15)


def test_threshold_adapts_to_noise_floor():
    """A louder room floor must not turn noise into detected speech."""
    quiet = detect_voice_activity(
        concat(silence(1.0, noise_db=-70.0), vowel(120.0, 2.0),
               silence(1.0, noise_db=-70.0)), SR
    )
    noisy = detect_voice_activity(
        concat(silence(1.0, noise_db=-45.0), vowel(120.0, 2.0),
               silence(1.0, noise_db=-45.0)), SR
    )
    assert quiet.has_speech and noisy.has_speech
    assert noisy.threshold_db > quiet.threshold_db
    assert noisy.speech_time_s == pytest.approx(quiet.speech_time_s, abs=0.3)


def test_frame_signal_shape_and_padding():
    x = np.arange(1000, dtype=np.float32)
    frames = frame_signal(x, 1000, frame_ms=100.0, hop_ms=50.0)
    assert frames.shape[1] == 100
    assert frames.shape[0] == 1 + (1000 - 100) // 50

    short = frame_signal(np.ones(10, dtype=np.float32), 1000, 100.0, 50.0)
    assert short.shape == (1, 100)


def test_custom_min_pause_changes_pause_count():
    """The gap threshold is configurable and actually takes effect."""
    x = speech_with_pauses([1.0, 1.0], [0.4], lead_s=0.3, trail_s=0.3)

    strict = detect_voice_activity(x, SR, SignalParams(vad_min_pause_ms=250.0))
    lenient = detect_voice_activity(x, SR, SignalParams(vad_min_pause_ms=600.0))

    assert strict.n_pauses == 1
    assert lenient.n_pauses == 0
