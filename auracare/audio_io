"""
Audio decoding for the Auracare prototype.

Accepts whatever the browser recorder hands back (``st.audio_input`` returns a
WAV-wrapped ``UploadedFile``), plus plain paths and raw bytes, and normalises
everything to mono float32 at a single sample rate so that every downstream
extractor sees the same thing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import DEFAULT_SIGNAL_PARAMS


class AudioDecodeError(RuntimeError):
    """Raised when an input cannot be turned into samples."""


@dataclass
class Audio:
    """A decoded mono recording."""

    samples: np.ndarray  # float32, shape (n,)
    sample_rate: int
    source_name: str = "recording"

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)

    @property
    def peak(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.max(np.abs(self.samples)))

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))

    @property
    def rms_dbfs(self) -> float:
        r = self.rms
        return float(20.0 * np.log10(r)) if r > 0 else -120.0

    def slice_seconds(self, start_s: float, end_s: float) -> "Audio":
        a = max(0, int(round(start_s * self.sample_rate)))
        b = min(len(self.samples), int(round(end_s * self.sample_rate)))
        if b <= a:
            return Audio(np.zeros(0, dtype=np.float32), self.sample_rate, self.source_name)
        return Audio(self.samples[a:b].copy(), self.sample_rate, self.source_name)

    def summary(self) -> dict:
        return {
            "source": self.source_name,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "peak": round(self.peak, 5),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "n_samples": int(self.samples.size),
        }


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def _read_any(source: Any) -> Tuple[np.ndarray, int, str]:
    """Pull (samples, sample_rate, name) out of a path / bytes / file-like."""
    name = "recording"

    if isinstance(source, Audio):
        return source.samples, source.sample_rate, source.source_name

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise AudioDecodeError(f"audio file not found: {path}")
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sr), path.name

    if isinstance(source, (bytes, bytearray, memoryview)):
        buf = io.BytesIO(bytes(source))
    elif hasattr(source, "getvalue"):  # Streamlit UploadedFile / BytesIO
        name = getattr(source, "name", name) or name
        buf = io.BytesIO(source.getvalue())
    elif hasattr(source, "read"):
        name = getattr(source, "name", name) or name
        raw = source.read()
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        buf = io.BytesIO(raw)
    else:
        raise AudioDecodeError(f"unsupported audio source type: {type(source)!r}")

    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioDecodeError(
            "could not decode audio. libsndfile handles WAV/FLAC/OGG; a browser "
            "that returns WebM/Opus needs ffmpeg available on PATH. "
            f"underlying error: {exc}"
        ) from exc

    return data, int(sr), name


def to_mono(data: np.ndarray) -> np.ndarray:
    """Average any channel layout down to a single channel."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # soundfile with always_2d gives (frames, channels)
        return arr.mean(axis=1).astype(np.float32)
    raise AudioDecodeError(f"unexpected audio shape {arr.shape}")


def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample. Exact ratio, no drift, linear phase."""
    if sr_in == sr_out or samples.size == 0:
        return np.asarray(samples, dtype=np.float32)
    g = np.gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out // g), int(sr_in // g)
    out = resample_poly(samples.astype(np.float64), up, down)
    return out.astype(np.float32)


def load_audio(
    source: Any,
    target_sr: int | None = None,
    remove_dc: bool = True,
) -> Audio:
    """
    Decode `source` to mono float32 at `target_sr`.

    `remove_dc` strips any constant offset, which some phone microphones add and
    which would otherwise inflate every RMS-based measurement.
    """
    target_sr = int(target_sr or DEFAULT_SIGNAL_PARAMS.sample_rate)
    data, sr, name = _read_any(source)
    mono = to_mono(data)

    if mono.size == 0:
        raise AudioDecodeError(f"'{name}' decoded to zero samples")

    if remove_dc and mono.size > 1:
        mono = mono - float(np.mean(mono, dtype=np.float64))

    mono = resample(mono, sr, target_sr)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Audio(samples=mono, sample_rate=target_sr, source_name=name)


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
@dataclass
class QualityReport:
    """Whether a recording is even worth analysing."""

    usable: bool
    problems: list
    peak: float
    rms_dbfs: float
    duration_s: float
    clipped_fraction: float
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "usable": self.usable,
            "problems": list(self.problems),
            "warnings": list(self.warnings),
            "peak": round(self.peak, 5),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "duration_s": round(self.duration_s, 3),
            "clipped_fraction": round(self.clipped_fraction, 5),
        }


def check_quality(
    audio: Audio,
    min_duration_s: float = 0.5,
    silence_dbfs: float = -55.0,
    clip_threshold: float = 0.999,
    max_clipped_fraction: float = 0.01,
    low_level_dbfs: float = -40.0,
) -> QualityReport:
    """
    Reject recordings that cannot support a measurement.

    This runs before the feature extractors so the UI can say "too quiet, try
    again" instead of reporting a confident number derived from noise.

    `problems` block analysis; `warnings` do not but are worth showing. The
    split exists because of a real case: `385_AUDIO.wav` has a peak of exactly
    1.0 - a handful of clipped samples, three parts per million - while
    averaging -46 dBFS. Neither fact alone makes it unusable, but a recording
    that is simultaneously very quiet and occasionally clipping has awkward
    dynamics, and silently passing it as fine hides that from whoever is
    reading the numbers.
    """
    problems: list = []
    warnings: list = []
    n = audio.samples.size
    clipped = 0.0
    if n:
        clipped = float(np.mean(np.abs(audio.samples) >= clip_threshold))

    if audio.duration_s < min_duration_s:
        problems.append(f"recording is only {audio.duration_s:.2f} s long")
    if audio.rms_dbfs < silence_dbfs:
        problems.append(f"recording is near-silent ({audio.rms_dbfs:.1f} dBFS)")
    if clipped > max_clipped_fraction:
        problems.append(f"{clipped * 100:.1f} % of samples are clipped")
    elif clipped > 0.0:
        warnings.append(
            f"{clipped * 100:.4f} % of samples reach full scale - some clipping"
        )

    if silence_dbfs <= audio.rms_dbfs < low_level_dbfs:
        warnings.append(
            f"recording is quiet ({audio.rms_dbfs:.1f} dBFS); measurements are "
            "usable but less reliable"
        )

    return QualityReport(
        usable=not problems,
        problems=problems,
        warnings=warnings,
        peak=audio.peak,
        rms_dbfs=audio.rms_dbfs,
        duration_s=audio.duration_s,
        clipped_fraction=clipped,
    )


__all__ = [
    "Audio",
    "AudioDecodeError",
    "QualityReport",
    "check_quality",
    "load_audio",
    "resample",
    "to_mono",
]
