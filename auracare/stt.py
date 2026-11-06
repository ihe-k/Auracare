"""
Speech-to-text and negative-language flagging for Task 4.

Transcription is pluggable and entirely optional. If no backend is installed
the rest of the pipeline still runs - the acoustic measures do not depend on
the transcript - and the UI says so instead of silently reporting zero
negative-language hits, which would read as a reassuring result.

The lexicon below is a transparent, auditable word list, not a model. That is a
deliberate trade-off: it will miss paraphrase and it will over-trigger on
quoted or hypothetical speech, but a clinician can read exactly why a phrase
was flagged, which is not true of a sentiment classifier. Every flag is
surfaced with the matched text so it can be checked against the transcript.

The `risk` category is handled differently from the rest: it is not a severity
input, it is a safety trigger. Any match routes to the crisis pathway
regardless of PHQ-8 score or acoustic measures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio_io import Audio

# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------
# Ordered so that the safety-critical category is impossible to miss.
RISK_CATEGORY = "risk"

LEXICON: Dict[str, List[str]] = {
    # Inflected forms are listed explicitly. Matching is literal, with no
    # stemming, so "wants to die" and "wanted to die" must each appear or they
    # are missed - and a missed self-harm disclosure is the worst failure this
    # system can have. Any addition here should include its inflections.
    RISK_CATEGORY: [
        "kill myself", "killing myself", "killed myself",
        "end my life", "ending my life", "end my own life",
        "take my own life", "taking my own life", "took my own life",
        "end it all", "ending it all",
        "suicide", "suicidal",
        "better off dead", "rather be dead",
        "want to die", "wants to die", "wanted to die", "wanting to die",
        "wish i was dead", "wish i were dead", "wished i was dead",
        "harm myself", "harming myself", "harmed myself",
        "hurt myself", "hurting myself",
        "cut myself", "cutting myself",
        "overdose", "overdosed",
        "do not want to live", "don't want to live", "dont want to live",
        "does not want to live", "doesn't want to live",
        "did not want to live", "didn't want to live",
        "no reason to live", "nothing to live for", "not worth living",
        "cannot go on", "can't go on", "cant go on", "could not go on",
        "cannot carry on", "can't carry on", "cant carry on",
    ],
    "hopelessness": [
        "hopeless", "no hope", "no point", "pointless", "no future",
        "never get better", "never going to get better", "nothing matters",
        "no way out", "give up", "giving up", "gave up", "despair",
        "trapped", "stuck forever",
    ],
    "worthlessness": [
        "worthless", "useless", "a failure", "i am a failure", "failure",
        "burden", "burdensome", "my fault", "blame myself", "blaming myself",
        "not good enough", "ashamed", "shame", "guilty", "hate myself",
        "disgusted with myself", "let everyone down",
    ],
    "anhedonia": [
        "no interest", "lost interest", "do not enjoy", "don't enjoy",
        "dont enjoy", "nothing interests me", "cannot enjoy", "can't enjoy",
        "numb", "empty", "hollow", "do not care", "don't care", "dont care",
        "nothing is fun", "no pleasure",
    ],
    "negative_affect": [
        "sad", "unhappy", "depressed", "depressing", "miserable", "awful",
        "terrible", "horrible", "crying", "i cry", "tearful", "upset",
        "distressed", "hurts", "painful", "suffering", "dreadful",
    ],
    "anxiety": [
        "anxious", "anxiety", "worried", "worrying", "scared", "afraid",
        "frightened", "panic", "panicking", "on edge", "nervous", "dread",
        "terrified", "restless",
    ],
    "fatigue": [
        "exhausted", "exhausting", "no energy", "drained", "worn out",
        "so tired", "always tired", "cannot get up", "can't get up",
        "cant get up", "heavy", "sluggish", "fatigue",
    ],
    "isolation": [
        "alone", "lonely", "loneliness", "nobody", "no one cares",
        "no-one cares", "isolated", "by myself", "no friends", "left out",
        "abandoned", "unwanted",
    ],
    "cognitive": [
        "cannot concentrate", "can't concentrate", "cant concentrate",
        "cannot focus", "can't focus", "cant focus", "cannot think",
        "can't think", "forget", "forgetful", "confused", "foggy",
        "brain fog", "muddled", "lose track",
    ],
}

# Documented linguistic markers measured alongside the lexicon.
FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself", "i'm", "im", "i've", "id"}
ABSOLUTIST = {
    "always", "never", "nothing", "everything", "completely", "totally",
    "entirely", "absolutely", "constantly", "all", "every", "none", "nobody",
    "everyone", "must", "definitely",
}

_WORD_RE = re.compile(r"[a-z']+")


def _normalise(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenise(text: str) -> List[str]:
    return _WORD_RE.findall(_normalise(text))


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------
@dataclass
class Transcript:
    """A transcription attempt - successful or not."""

    text: str = ""
    available: bool = False
    backend: str = "none"
    language: Optional[str] = None
    duration_s: Optional[float] = None
    unavailable_reason: Optional[str] = None
    segments: List[Tuple[float, float, str]] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return len(tokenise(self.text))

    def as_dict(self) -> Dict[str, object]:
        return {
            "available": self.available,
            "backend": self.backend,
            "language": self.language,
            "text": self.text,
            "n_words": self.n_words,
            "unavailable_reason": self.unavailable_reason,
            "n_segments": len(self.segments),
        }


def available_backends() -> List[str]:
    """Which transcription backends can actually be used right now."""
    found: List[str] = []
    try:
        import faster_whisper  # noqa: F401

        found.append("faster-whisper")
    except Exception:
        pass
    return found


_WHISPER_CACHE: Dict[str, object] = {}


def _transcribe_faster_whisper(
    audio: Audio, model_size: str = "base.en", language: str = "en"
) -> Transcript:
    from faster_whisper import WhisperModel

    key = f"{model_size}"
    model = _WHISPER_CACHE.get(key)
    if model is None:
        # int8 on CPU keeps this usable on a laptop and on Streamlit Cloud.
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _WHISPER_CACHE[key] = model

    samples = np.asarray(audio.samples, dtype=np.float32)
    if audio.sample_rate != 16_000:
        from .audio_io import resample

        samples = resample(samples, audio.sample_rate, 16_000)

    segments_iter, info = model.transcribe(  # type: ignore[attr-defined]
        samples, language=language, beam_size=1, vad_filter=False
    )
    segments: List[Tuple[float, float, str]] = []
    parts: List[str] = []
    for seg in segments_iter:
        segments.append((float(seg.start), float(seg.end), seg.text.strip()))
        parts.append(seg.text.strip())

    return Transcript(
        text=" ".join(p for p in parts if p).strip(),
        available=True,
        backend="faster-whisper",
        language=getattr(info, "language", language),
        duration_s=getattr(info, "duration", audio.duration_s),
        segments=segments,
    )


def transcribe(
    audio: Audio,
    backend: str = "auto",
    model_size: str = "base.en",
    language: str = "en",
) -> Transcript:
    """
    Transcribe `audio`, degrading gracefully when no backend is installed.

    `backend="none"` short-circuits deliberately, which is how the test suite
    and the offline demo path exercise the no-transcript branch.
    """
    if backend == "none":
        return Transcript(
            available=False,
            backend="none",
            unavailable_reason="transcription disabled",
        )

    backends = available_backends()
    if backend in ("auto", "faster-whisper") and "faster-whisper" in backends:
        try:
            return _transcribe_faster_whisper(audio, model_size, language)
        except Exception as exc:
            return Transcript(
                available=False,
                backend="faster-whisper",
                unavailable_reason=f"transcription failed: {exc}",
            )

    return Transcript(
        available=False,
        backend="none",
        unavailable_reason=(
            "no speech-to-text backend installed. "
            "Install one with: pip install faster-whisper"
        ),
    )


# ---------------------------------------------------------------------------
# Negative-language analysis
# ---------------------------------------------------------------------------
@dataclass
class LanguageFlags:
    """Lexicon hits and linguistic markers for one transcript."""

    analysed: bool = False
    n_tokens: int = 0
    n_unique_tokens: int = 0
    lexical_diversity: Optional[float] = None

    category_counts: Dict[str, int] = field(default_factory=dict)
    matched_terms: Dict[str, List[str]] = field(default_factory=dict)
    negative_token_count: int = 0
    negative_ratio: Optional[float] = None

    first_person_singular_ratio: Optional[float] = None
    absolutist_ratio: Optional[float] = None

    safety_critical: bool = False
    safety_terms: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        def r(v, nd=4):
            return None if v is None else round(float(v), nd)

        return {
            "analysed": self.analysed,
            "n_tokens": self.n_tokens,
            "n_unique_tokens": self.n_unique_tokens,
            "lexical_diversity": r(self.lexical_diversity),
            "category_counts": dict(self.category_counts),
            "matched_terms": {k: list(v) for k, v in self.matched_terms.items()},
            "negative_token_count": self.negative_token_count,
            "negative_ratio": r(self.negative_ratio),
            "first_person_singular_ratio": r(self.first_person_singular_ratio),
            "absolutist_ratio": r(self.absolutist_ratio),
            "safety_critical": self.safety_critical,
            "safety_terms": list(self.safety_terms),
        }


def analyse_language(text: str) -> LanguageFlags:
    """
    Run the lexicon over a transcript.

    Longer phrases are matched before their constituent words so that
    "kill myself" is counted once as a risk phrase rather than also feeding
    other categories.
    """
    out = LanguageFlags()
    norm = _normalise(text)
    tokens = tokenise(text)
    out.n_tokens = len(tokens)

    if not tokens:
        return out

    out.analysed = True
    out.n_unique_tokens = len(set(tokens))
    out.lexical_diversity = out.n_unique_tokens / out.n_tokens

    # Mask consumed spans so a phrase is not double-counted by a sub-phrase.
    consumed = np.zeros(len(norm), dtype=bool)
    negative_tokens = 0

    for category, terms in LEXICON.items():
        hits: List[str] = []
        # Longest first so "no interest" wins over "interest".
        for term in sorted(terms, key=len, reverse=True):
            pattern = r"\b" + re.escape(term) + r"\b"
            for match in re.finditer(pattern, norm):
                a, b = match.span()
                if consumed[a:b].any():
                    continue
                consumed[a:b] = True
                hits.append(term)
                negative_tokens += len(term.split())
        if hits:
            out.category_counts[category] = len(hits)
            out.matched_terms[category] = sorted(set(hits))

    out.negative_token_count = negative_tokens
    out.negative_ratio = negative_tokens / out.n_tokens

    fps = sum(1 for t in tokens if t in FIRST_PERSON_SINGULAR)
    out.first_person_singular_ratio = fps / out.n_tokens
    abso = sum(1 for t in tokens if t in ABSOLUTIST)
    out.absolutist_ratio = abso / out.n_tokens

    if RISK_CATEGORY in out.matched_terms:
        out.safety_critical = True
        out.safety_terms = list(out.matched_terms[RISK_CATEGORY])

    return out


__all__ = [
    "ABSOLUTIST",
    "FIRST_PERSON_SINGULAR",
    "LEXICON",
    "RISK_CATEGORY",
    "LanguageFlags",
    "Transcript",
    "analyse_language",
    "available_backends",
    "tokenise",
    "transcribe",
]
