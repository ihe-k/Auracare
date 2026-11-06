"""
Task 4 - Cognitive and language fluency: picture description.

Combines three independent views of the same 30-second recording:

* timing, from voice activity detection - pause ratio (the protocol's second
  safety intercept), pause count and length distribution;
* rate, from syllable-nucleus detection - speech rate over the whole active
  span and articulation rate over voiced time only. The two separate because
  someone who speaks quickly but pauses a lot has a normal articulation rate
  and a low speech rate, and that distinction is the interesting one;
* content, from the transcript - lexical diversity as an information-density
  proxy, plus the negative-language lexicon.

Voice perturbation is measured over the active span rather than over
concatenated speech segments: stitching segments together introduces
discontinuities at every join, which Praat would read as enormous jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..audio_io import Audio
from ..config import DEFAULT_SIGNAL_PARAMS, SignalParams
from ..stt import LanguageFlags, Transcript, analyse_language
from ..vad import VadResult, detect_voice_activity
from .ddk import detect_syllable_nuclei
from .voice import VoiceIndicators, analyse_voice


@dataclass
class FluencyResult:
    """Task 4 measurements."""

    # Timing
    pause_ratio: float = 0.0
    n_pauses: int = 0
    mean_pause_s: float = 0.0
    longest_pause_s: float = 0.0
    speech_time_s: float = 0.0
    active_span_s: float = 0.0
    recording_duration_s: float = 0.0
    has_speech: bool = False

    # Rate
    n_syllables: int = 0
    speech_rate_sylls: Optional[float] = None
    articulation_rate_sylls: Optional[float] = None

    # Content
    n_words: int = 0
    words_per_minute: Optional[float] = None
    lexical_diversity: Optional[float] = None

    voice: VoiceIndicators = field(default_factory=VoiceIndicators)
    language: LanguageFlags = field(default_factory=LanguageFlags)
    transcript: Transcript = field(default_factory=Transcript)
    vad: Optional[VadResult] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        def r(v, nd=3):
            return None if v is None else round(float(v), nd)

        return {
            "pause_ratio": r(self.pause_ratio, 4),
            "n_pauses": self.n_pauses,
            "mean_pause_s": r(self.mean_pause_s),
            "longest_pause_s": r(self.longest_pause_s),
            "speech_time_s": r(self.speech_time_s),
            "active_span_s": r(self.active_span_s),
            "recording_duration_s": r(self.recording_duration_s),
            "has_speech": self.has_speech,
            "n_syllables": self.n_syllables,
            "speech_rate_sylls": r(self.speech_rate_sylls),
            "articulation_rate_sylls": r(self.articulation_rate_sylls),
            "n_words": self.n_words,
            "words_per_minute": r(self.words_per_minute, 1),
            "lexical_diversity": r(self.lexical_diversity, 4),
            "voice": self.voice.as_dict(),
            "language": self.language.as_dict(),
            "transcript": self.transcript.as_dict(),
            "vad": self.vad.as_dict() if self.vad else None,
            "notes": list(self.notes),
        }


def analyse_fluency(
    audio: Audio,
    params: SignalParams | None = None,
    transcript: Transcript | None = None,
) -> FluencyResult:
    """
    Measure a connected-speech recording.

    `transcript` is optional. Pass one to enable the content measures; omit it
    and the timing and rate measures still work, with the content fields left
    as None rather than zero.
    """
    p = params or DEFAULT_SIGNAL_PARAMS
    out = FluencyResult(recording_duration_s=audio.duration_s)
    out.transcript = transcript or Transcript()

    if audio.samples.size == 0:
        out.notes.append("empty recording")
        return out

    # -- timing -------------------------------------------------------------
    vad = detect_voice_activity(audio.samples, audio.sample_rate, p)
    out.vad = vad
    out.has_speech = vad.has_speech
    out.pause_ratio = vad.pause_ratio
    out.n_pauses = vad.n_pauses
    out.mean_pause_s = vad.mean_pause_s
    out.longest_pause_s = vad.longest_pause_s
    out.speech_time_s = vad.speech_time_s
    out.active_span_s = vad.active_span_s

    if not vad.has_speech:
        out.notes.append(
            "no speech detected - the recording may be silence or background noise"
        )
        return out

    # -- rate ---------------------------------------------------------------
    active = audio.slice_seconds(vad.active_start_s, vad.active_end_s)
    nuclei = detect_syllable_nuclei(active, p)
    out.n_syllables = int(nuclei.size)

    if out.active_span_s > 0 and out.n_syllables > 0:
        out.speech_rate_sylls = float(out.n_syllables / out.active_span_s)
    if out.speech_time_s > 0 and out.n_syllables > 0:
        out.articulation_rate_sylls = float(out.n_syllables / out.speech_time_s)

    # -- voice over the active span ----------------------------------------
    out.voice = analyse_voice(active.samples, active.sample_rate, p)

    # -- content ------------------------------------------------------------
    if out.transcript.available and out.transcript.text.strip():
        flags = analyse_language(out.transcript.text)
        out.language = flags
        out.n_words = flags.n_tokens
        out.lexical_diversity = flags.lexical_diversity
        if out.active_span_s > 0:
            out.words_per_minute = float(out.n_words / out.active_span_s * 60.0)
    else:
        reason = out.transcript.unavailable_reason or "no transcript supplied"
        out.notes.append(
            f"content measures skipped ({reason}); acoustic measures unaffected"
        )

    if out.speech_time_s < 3.0:
        out.notes.append(
            f"only {out.speech_time_s:.1f} s of speech - rate and pause measures "
            "need a longer sample to be stable"
        )
    return out


__all__ = ["FluencyResult", "analyse_fluency"]
