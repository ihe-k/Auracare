"""Decoding, resampling and the quality gate."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
import soundfile as sf

from auracare.audio_io import (
    Audio,
    AudioDecodeError,
    check_quality,
    load_audio,
    resample,
    to_mono,
)
from conftest import SR, as_audio, concat, silence, vowel


def test_resample_preserves_duration_and_tone():
    sr_in, sr_out = 44_100, 16_000
    t = np.arange(int(sr_in * 1.0)) / sr_in
    x = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    y = resample(x, sr_in, sr_out)

    assert abs(y.size / sr_out - 1.0) < 0.01
    # The 440 Hz peak must survive the rate change.
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    peak_hz = np.fft.rfftfreq(y.size, 1.0 / sr_out)[int(np.argmax(spec))]
    assert abs(peak_hz - 440.0) < 2.0


def test_resample_is_identity_at_same_rate():
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    assert np.array_equal(resample(x, SR, SR), x)


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(100), np.full(100, 3.0)], axis=1)
    mono = to_mono(stereo)
    assert mono.shape == (100,)
    assert np.allclose(mono, 2.0)


def test_load_audio_from_wav_bytes_roundtrip():
    x = vowel(150.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    buf.seek(0)

    audio = load_audio(buf.getvalue())

    assert audio.sample_rate == SR
    assert abs(audio.duration_s - 1.0) < 0.01
    assert audio.peak > 0.1


def test_load_audio_resamples_to_target():
    x = vowel(150.0, 1.0, sr=48_000)
    buf = io.BytesIO()
    sf.write(buf, x, 48_000, format="WAV", subtype="PCM_16")

    audio = load_audio(buf.getvalue(), target_sr=16_000)

    assert audio.sample_rate == 16_000
    assert abs(audio.duration_s - 1.0) < 0.01


def test_load_audio_strips_dc_offset():
    x = np.full(SR, 0.5, dtype=np.float32) + vowel(120.0, 1.0)[:SR] * 0.1
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="FLOAT")

    audio = load_audio(buf.getvalue(), remove_dc=True)

    assert abs(float(np.mean(audio.samples))) < 1e-4


def test_load_audio_rejects_unsupported_source():
    with pytest.raises(AudioDecodeError):
        load_audio(12345)


def test_load_audio_rejects_missing_file(tmp_path):
    with pytest.raises(AudioDecodeError):
        load_audio(tmp_path / "nope.wav")


def test_quality_gate_flags_silence():
    report = check_quality(as_audio(silence(2.0)))
    assert not report.usable
    assert any("silent" in p for p in report.problems)


def test_quality_gate_flags_too_short():
    report = check_quality(as_audio(vowel(120.0, 0.2)))
    assert not report.usable
    assert any("long" in p for p in report.problems)


def test_quality_gate_flags_clipping():
    x = np.ones(SR * 2, dtype=np.float32)
    report = check_quality(as_audio(x))
    assert not report.usable
    assert any("clipped" in p for p in report.problems)


def test_quality_gate_passes_good_recording():
    report = check_quality(as_audio(concat(silence(0.2), vowel(120.0, 3.0))))
    assert report.usable
    assert report.problems == []


def test_audio_slice_seconds_bounds():
    audio = as_audio(vowel(120.0, 2.0))
    assert abs(audio.slice_seconds(0.5, 1.5).duration_s - 1.0) < 0.01
    # Out-of-range slices clamp rather than raise.
    assert audio.slice_seconds(5.0, 6.0).samples.size == 0
    assert abs(audio.slice_seconds(-1.0, 1.0).duration_s - 1.0) < 0.01


def test_empty_audio_properties_are_safe():
    audio = Audio(np.zeros(0, dtype=np.float32), SR)
    assert audio.duration_s == 0.0
    assert audio.peak == 0.0
    assert audio.rms_dbfs == -120.0
