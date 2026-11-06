"""
Regressions for defects found while testing against the client's real audio.

Each test here corresponds to a bug that the synthetic-signal suite did not
catch, because synthetic signals are well-behaved and real recordings are not.
"""

from __future__ import annotations

import numpy as np
import pytest

from auracare.audio_io import check_quality, load_audio
from auracare.features.ddk import detect_syllable_nuclei
from auracare.features.fluency import analyse_fluency
from auracare.paths import find_practice_files
from conftest import SR, am_syllable_train, as_audio, concat, noise_burst, silence


def test_syllable_detection_survives_loud_transients():
    """
    Regression: the acceptance floor was `max(envelope) - 25 dB`, so a single
    loud transient raised the floor above the speech and suppressed detection.
    On 385_AUDIO.wav that produced 26 syllables for 52 transcribed words.

    Here a quiet 5 Hz syllable train is given one loud click. The detected rate
    must be unaffected.
    """
    quiet = am_syllable_train(5.0, 4.0) * 0.02
    clean = detect_syllable_nuclei(as_audio(quiet))

    spiked = quiet.copy()
    spiked[int(2.0 * SR)] = 1.0  # one full-scale click, 50 dB above the speech
    with_spike = detect_syllable_nuclei(as_audio(spiked))

    assert clean.size >= 15
    # Allow the click itself to add at most one detection.
    assert abs(with_spike.size - clean.size) <= 1


def test_syllable_detection_ignores_silence_regions():
    """Nuclei must not be invented in the silent part of a recording."""
    signal = concat(silence(3.0), am_syllable_train(5.0, 2.0), silence(3.0))
    nuclei = detect_syllable_nuclei(as_audio(signal))
    assert nuclei.size > 5
    assert nuclei.min() > 2.5
    assert nuclei.max() < 5.5


def test_syllable_detection_returns_nothing_for_pure_silence():
    assert detect_syllable_nuclei(as_audio(silence(5.0))).size == 0


def test_restrict_to_speech_can_be_disabled():
    audio = as_audio(am_syllable_train(5.0, 3.0))
    assert detect_syllable_nuclei(audio, restrict_to_speech=False).size > 10


def test_quality_gate_warns_about_occasional_clipping_without_blocking():
    """
    Regression: a recording peaking at exactly 1.0 passed as fine with no
    comment. A few clipped samples should not block analysis, but should be
    reported.
    """
    x = (am_syllable_train(4.0, 3.0) * 0.3).astype(np.float32)
    x[1000] = 1.0
    report = check_quality(as_audio(x))

    assert report.usable
    assert report.problems == []
    assert any("clipping" in w for w in report.warnings)


def test_quality_gate_warns_about_quiet_recordings():
    # Around -50 dBFS: quiet enough to be worth flagging, loud enough to use.
    x = (am_syllable_train(4.0, 3.0) * 0.03).astype(np.float32)
    report = check_quality(as_audio(x))
    assert -55.0 < report.rms_dbfs < -40.0
    assert report.usable
    assert any("quiet" in w for w in report.warnings)


def test_heavy_clipping_still_blocks():
    report = check_quality(as_audio(np.ones(SR * 2, dtype=np.float32)))
    assert not report.usable
    assert any("clipped" in p for p in report.problems)


# ---------------------------------------------------------------------------
# Against the client's actual files
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_wavs():
    directory = find_practice_files()
    if directory is None:
        pytest.skip("Practice_Files not available")
    wavs = sorted(directory.glob("*.wav"))
    if not wavs:
        pytest.skip("no sample recordings found")
    return wavs


def test_real_recordings_decode(real_wavs):
    for path in real_wavs:
        audio = load_audio(path)
        assert audio.sample_rate == 16_000
        assert audio.duration_s > 60.0
        assert np.isfinite(audio.samples).all()


def test_real_recordings_yield_plausible_articulation_rates(real_wavs):
    """
    Conversational articulation rate sits around 4-6 syllables/s. This is the
    check that caught the transient bug: before the fix one recording reported
    1.4 syll/s, which no speaker produces.
    """
    for path in real_wavs:
        audio = load_audio(path)
        excerpt = audio.slice_seconds(60.0, 90.0)
        result = analyse_fluency(excerpt)

        assert result.has_speech
        assert result.articulation_rate_sylls is not None
        assert 2.5 < result.articulation_rate_sylls < 9.0, (
            f"{path.name}: implausible articulation rate "
            f"{result.articulation_rate_sylls:.2f} syll/s"
        )


def test_real_recordings_pause_ratio_in_range(real_wavs):
    for path in real_wavs:
        excerpt = load_audio(path).slice_seconds(60.0, 90.0)
        result = analyse_fluency(excerpt)
        assert 0.0 <= result.pause_ratio <= 1.0


def test_real_recordings_voice_measures_are_defined(real_wavs):
    """Connected speech must yield real perturbation numbers, not None."""
    for path in real_wavs:
        excerpt = load_audio(path).slice_seconds(60.0, 90.0)
        voice = analyse_fluency(excerpt).voice
        assert voice.available
        assert voice.mean_f0_hz is not None
        assert 60.0 < voice.mean_f0_hz < 400.0
        assert voice.jitter_local_pct is not None
        assert voice.hnr_db is not None


# ---------------------------------------------------------------------------
# Perturbation measures must not be graded on noisy recordings
# ---------------------------------------------------------------------------
def _noisy_vowel(noise_amp: float):
    """Amplitude 0.08 reproduces the real laptop capture: HNR ~6 dB, pitch
    still trackable. Above ~0.2 the tracker loses voicing entirely, which is a
    different failure mode."""
    """A sustained vowel buried in white noise."""
    from conftest import vowel

    rng = np.random.default_rng(7)
    clean = vowel(120.0, 4.0)
    noise = rng.standard_normal(clean.size).astype(np.float32) * noise_amp
    return as_audio((clean + noise).astype(np.float32))


def test_clean_vowel_is_graded_normally():
    from auracare.features.voice import analyse_voice

    voice = analyse_voice(_noisy_vowel(0.0).samples, SR)
    assert voice.perturbation_reliable
    assert voice.hnr_db is not None and voice.hnr_db > 10.0


def test_noisy_vowel_marks_perturbation_unreliable():
    """
    Regression found on a real laptop recording: HNR of 4.6 dB, yet jitter and
    shimmer were confidently graded Elevated against studio-derived norms.
    """
    from auracare.features.voice import analyse_voice

    voice = analyse_voice(_noisy_vowel(0.08).samples, SR)
    assert voice.hnr_db is not None and voice.hnr_db < 10.0
    assert not voice.perturbation_reliable
    # The values are still reported, just not trusted.
    assert voice.jitter_local_pct is not None
    assert any("not graded" in n or "meaningful" in n for n in voice.notes)


def test_noisy_recording_is_not_graded_elevated():
    """A noisy capture must read 'not measured', never 'Elevated'."""
    from auracare.config import Thresholds
    from auracare.features.phonation import analyse_phonation
    from auracare.risk import NOT_MEASURED, assess_task1

    assessment = assess_task1(analyse_phonation(_noisy_vowel(0.08)), Thresholds())
    graded = {i.key: i.level for i in assessment.indicators}

    assert graded["jitter_local_pct"] == NOT_MEASURED
    assert graded["shimmer_local_pct"] == NOT_MEASURED
    assert graded["f0_cv"] == NOT_MEASURED
    # Phonation time is a duration, not a voice-quality measure, so it stays.
    assert graded["phonation_duration_s"] != NOT_MEASURED


def test_clean_recording_still_grades_perturbation():
    """The gate must not disable grading on good recordings."""
    from auracare.config import Thresholds
    from auracare.features.phonation import analyse_phonation
    from auracare.risk import NORMAL, assess_task1

    assessment = assess_task1(analyse_phonation(_noisy_vowel(0.0)), Thresholds())
    graded = {i.key: i.level for i in assessment.indicators}
    assert graded["jitter_local_pct"] == NORMAL
    assert graded["shimmer_local_pct"] == NORMAL
