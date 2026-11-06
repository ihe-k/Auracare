"""
Negative-language flagging and the transcription abstraction.

The lexicon's known weaknesses - it cannot tell disclosure from quotation,
negation or hypothetical speech - are asserted here as *documented behaviour*
rather than quietly hoped away, so nobody later mistakes a flag for a finding.
The UI surfaces the matched text for exactly this reason.
"""

from __future__ import annotations

import pytest

from auracare.stt import (
    LEXICON,
    RISK_CATEGORY,
    Transcript,
    analyse_language,
    available_backends,
    tokenise,
    transcribe,
)
from conftest import as_audio, vowel


def test_tokenise_normalises_punctuation_and_case():
    assert tokenise("I'm Tired, so TIRED!") == ["i'm", "tired", "so", "tired"]


def test_empty_transcript_is_not_analysed():
    flags = analyse_language("")
    assert not flags.analysed
    assert flags.n_tokens == 0
    assert flags.negative_ratio is None
    assert not flags.safety_critical


def test_neutral_description_produces_no_flags():
    text = (
        "A group of children are playing football on the grass in a park. "
        "Two adults sit on a bench nearby and a dog runs towards the pond."
    )
    flags = analyse_language(text)
    assert flags.analysed
    assert flags.negative_token_count == 0
    assert flags.category_counts == {}
    assert not flags.safety_critical


def test_depressive_language_flagged_by_category():
    text = "I feel hopeless and worthless and nothing matters any more"
    flags = analyse_language(text)
    assert "hopelessness" in flags.category_counts
    assert "worthlessness" in flags.category_counts
    assert flags.negative_ratio > 0.1


@pytest.mark.parametrize(
    "phrase",
    [
        "I want to die",
        "I would be better off dead",
        "I have thought about suicide",
        "sometimes I want to hurt myself",
        "I don't want to live like this",
    ],
)
def test_self_harm_phrases_are_safety_critical(phrase):
    flags = analyse_language(phrase)
    assert flags.safety_critical
    assert flags.safety_terms
    assert RISK_CATEGORY in flags.category_counts


def test_ordinary_sadness_is_not_safety_critical():
    """The crisis pathway must not fire on general low mood."""
    flags = analyse_language("I have been feeling very sad and tired lately")
    assert not flags.safety_critical
    assert "negative_affect" in flags.category_counts


def test_longer_phrase_wins_over_its_substring():
    """
    'kill myself' must count once as a risk phrase, not also feed another
    category through an overlapping shorter term.
    """
    flags = analyse_language("I think about how I might kill myself")
    assert flags.category_counts.get(RISK_CATEGORY) == 1
    # "kill myself" is two tokens, counted once.
    assert flags.negative_token_count == 2


def test_lexical_diversity_bounds():
    repetitive = analyse_language("the the the the the the the the")
    varied = analyse_language("every single word here is entirely distinct from others")
    assert repetitive.lexical_diversity < 0.2
    assert varied.lexical_diversity > 0.9


def test_first_person_singular_ratio_measured():
    high = analyse_language("I think my life is my own and I blame myself")
    low = analyse_language("the park looks busy today and the weather is fine")
    assert high.first_person_singular_ratio > low.first_person_singular_ratio


def test_absolutist_ratio_measured():
    flags = analyse_language("nothing ever works and everything is always completely wrong")
    assert flags.absolutist_ratio > 0.3


def test_known_limitation_negation_is_not_understood():
    """
    Documented weakness: the lexicon matches surface strings. "I am not
    hopeless" still flags. This is why every flag is shown with its matched
    text for clinician review rather than being acted on automatically.
    """
    flags = analyse_language("I am not hopeless at all")
    assert "hopelessness" in flags.category_counts


def test_known_limitation_attribution_is_not_understood():
    """
    Documented weakness: the lexicon has no notion of who is speaking about
    whom. A patient reporting someone else's words flags identically to a
    first-person disclosure.
    """
    flags = analyse_language("my brother said he wants to die")
    assert flags.safety_critical


@pytest.mark.parametrize(
    "phrase",
    [
        "he wants to die",
        "he wanted to die",
        "I am wanting to die",
        "I cannot go on",
        "I can't carry on",
        "I have nothing to live for",
        "I would rather be dead",
        "I have been cutting myself",
        "I took an overdose",
    ],
)
def test_risk_inflections_are_covered(phrase):
    """
    Matching is literal with no stemming, so every inflection has to be
    enumerated. This test exists because "wants to die" and "wanted to die"
    were both missed by the first draft of the lexicon, and a missed self-harm
    disclosure is the worst failure this system can produce.
    """
    assert analyse_language(phrase).safety_critical


def test_known_limitation_paraphrase_outside_the_list_is_missed():
    """
    Documented weakness: the lexicon cannot generalise. A phrasing that is not
    enumerated will not flag, however clear it is to a human reader. This is
    why the transcript is always shown to the clinician in full rather than
    only its flags.
    """
    assert not analyse_language(
        "I have been making arrangements for after I am gone"
    ).safety_critical


def test_lexicon_categories_are_non_empty_and_lowercase():
    for category, terms in LEXICON.items():
        assert terms
        for term in terms:
            assert term == term.lower()


def test_transcribe_backend_none_short_circuits():
    result = transcribe(as_audio(vowel(120.0, 1.0)), backend="none")
    assert not result.available
    assert result.backend == "none"
    assert result.unavailable_reason


def test_transcript_word_count():
    assert Transcript(text="one two three", available=True).n_words == 3
    assert Transcript().n_words == 0


def test_available_backends_returns_a_list():
    assert isinstance(available_backends(), list)


def test_unavailable_transcript_yields_no_false_reassurance():
    """
    A failed transcription must not look like "no negative language found".
    """
    result = transcribe(as_audio(vowel(120.0, 1.0)), backend="none")
    flags = analyse_language(result.text)
    assert not flags.analysed
    assert flags.negative_ratio is None
