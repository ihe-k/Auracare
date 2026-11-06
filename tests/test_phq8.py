"""PHQ-8 scoring, banding and missing-data handling."""

from __future__ import annotations

import pytest

from auracare.phq8 import (
    ITEM_KEYS,
    MAX_SCORE,
    MIN_ANSWERED_FOR_PRORATION,
    PHQ8_ITEMS,
    RESPONSE_OPTIONS,
    band_for_score,
    empty_responses,
    item_by_key,
    score_phq8,
)


def all_at(level: int) -> dict:
    return {k: level for k in ITEM_KEYS}


def test_instrument_shape():
    assert len(PHQ8_ITEMS) == 8
    assert MAX_SCORE == 24
    assert len(RESPONSE_OPTIONS) == 4
    assert len(set(ITEM_KEYS)) == 8


def test_item_keys_match_severity_model_feature_names():
    """
    The keys deliberately match the DAIC-WOZ feature names in the repository
    severity model, so the two can be lined up without a translation table.
    """
    assert ITEM_KEYS == [
        "PHQ8_NoInterest",
        "PHQ8_Depressed",
        "PHQ8_Sleep",
        "PHQ8_Tired",
        "PHQ8_Appetite",
        "PHQ8_Failure",
        "PHQ8_Concentrating",
        "PHQ8_Moving",
    ]


def test_every_item_has_a_spoken_prompt():
    for item in PHQ8_ITEMS:
        assert item.spoken_prompt
        assert item.spoken_prompt != item.text


@pytest.mark.parametrize(
    "level,expected", [(0, 0), (1, 8), (2, 16), (3, 24)]
)
def test_summation_is_exact(level, expected):
    result = score_phq8(all_at(level))
    assert result.total_score == expected
    assert result.complete
    assert result.scorable
    assert not result.prorated


@pytest.mark.parametrize(
    "score,band,index",
    [
        (0, "None/minimal", 0),
        (4, "None/minimal", 0),
        (5, "Mild", 1),
        (9, "Mild", 1),
        (10, "Moderate", 2),
        (14, "Moderate", 2),
        (15, "Moderately severe", 3),
        (19, "Moderately severe", 3),
        (20, "Severe", 4),
        (24, "Severe", 4),
    ],
)
def test_published_band_boundaries(score, band, index):
    """Exact Kroenke 2009 cut-points, including both ends of every band."""
    assert band_for_score(score) == (band, index)


def test_case_finding_cutpoint_is_ten():
    assert not score_phq8({k: 1 for k in ITEM_KEYS[:4]} | {k: 0 for k in ITEM_KEYS[4:]}).clinically_significant
    nine = dict.fromkeys(ITEM_KEYS, 0)
    for k in ITEM_KEYS[:3]:
        nine[k] = 3
    assert sum(nine.values()) == 9
    assert not score_phq8(nine).clinically_significant

    ten = dict(nine)
    ten[ITEM_KEYS[3]] = 1
    assert sum(ten.values()) == 10
    assert score_phq8(ten).clinically_significant


def test_missing_items_are_not_treated_as_zero():
    """
    Treating unanswered items as 0 biases the total downward, which on a
    screening instrument means under-referral.
    """
    partial = {k: 3 for k in ITEM_KEYS[:6]}  # 6 answered, sum 18
    result = score_phq8(partial)

    assert result.n_answered == 6
    assert result.raw_sum == 18
    assert result.prorated
    assert result.scorable
    # Prorated to the full 8 items: 18 * 8/6 = 24, not 18.
    assert result.total_score == 24
    assert result.band == "Severe"


def test_too_few_answers_is_not_scored():
    result = score_phq8({k: 3 for k in ITEM_KEYS[:3]})
    assert not result.scorable
    assert result.total_score is None
    assert result.band == "Not scored"
    assert any("only 3 of 8" in n for n in result.notes)


def test_proration_can_be_disabled():
    partial = {k: 2 for k in ITEM_KEYS[:7]}
    assert score_phq8(partial, allow_proration=True).scorable
    assert not score_phq8(partial, allow_proration=False).scorable


def test_proration_threshold_is_respected():
    at_threshold = {k: 1 for k in ITEM_KEYS[:MIN_ANSWERED_FOR_PRORATION]}
    below = {k: 1 for k in ITEM_KEYS[: MIN_ANSWERED_FOR_PRORATION - 1]}
    assert score_phq8(at_threshold).scorable
    assert not score_phq8(below).scorable


def test_out_of_range_responses_rejected_not_clamped():
    result = score_phq8(all_at(0) | {"PHQ8_Sleep": 7})
    assert result.n_answered == 7
    assert any("out of range" in n for n in result.notes)


def test_non_numeric_responses_rejected():
    result = score_phq8(all_at(1) | {"PHQ8_Tired": "often"})
    assert result.n_answered == 7
    assert any("non-numeric" in n for n in result.notes)


def test_elevated_items_are_those_at_two_or_more():
    responses = all_at(0)
    responses["PHQ8_Depressed"] = 2
    responses["PHQ8_Sleep"] = 3
    responses["PHQ8_Tired"] = 1
    result = score_phq8(responses)
    assert set(result.elevated_items) == {"PHQ8_Depressed", "PHQ8_Sleep"}


def test_empty_responses_helper_is_blank_and_unscorable():
    blank = empty_responses()
    assert set(blank) == set(ITEM_KEYS)
    assert all(v is None for v in blank.values())
    assert not score_phq8(blank).scorable


def test_item_by_key_round_trip():
    for key in ITEM_KEYS:
        assert item_by_key(key).key == key
    with pytest.raises(KeyError):
        item_by_key("PHQ8_NotAnItem")


def test_as_dict_is_serialisable():
    import json

    payload = score_phq8(all_at(2)).as_dict()
    assert json.loads(json.dumps(payload))["total_score"] == 16
