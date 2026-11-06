"""
PHQ-8 questionnaire, scoring and severity banding.

The PHQ-8 is the PHQ-9 with the self-harm item removed. It is scored by simple
summation over eight items rated 0-3, giving 0-24, and interpreted against
published cut-points. There is no model here and there does not need to be one:
the instrument *is* the validated instrument, and summing it is the validated
scoring procedure.

Reference
    Kroenke K, Strine TW, Spitzer RL, Williams JBW, Berry JT, Mokdad AH.
    The PHQ-8 as a measure of current depression in the general population.
    J Affect Disord. 2009;114(1-3):163-173.

Each item carries a `spoken_prompt`: the same question phrased for a voice
agent to read aloud, since the screening flow asks these conversationally
rather than presenting a paper form.

Note on the self-harm item: because PHQ-8 omits it, this module cannot detect
suicidal ideation. The separate `safety_critical` flag from the transcript
lexicon in `stt.py` exists to cover that gap, and `risk.py` routes on it
independently of the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import PROVISIONAL_DEFAULTS

# Response anchors, 0-3.
RESPONSE_OPTIONS: List[str] = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day",
]

RECALL_STEM = (
    "Over the last 2 weeks, how often have you been bothered by "
    "any of the following problems?"
)


@dataclass(frozen=True)
class Phq8Item:
    """One PHQ-8 item."""

    key: str  # matches the DAIC-WOZ / severity-model feature name
    short: str
    text: str
    spoken_prompt: str


# Order follows the published instrument. `key` deliberately matches the
# feature names in the repository severity model so the two can be lined up.
PHQ8_ITEMS: List[Phq8Item] = [
    Phq8Item(
        key="PHQ8_NoInterest",
        short="Little interest or pleasure",
        text="Little interest or pleasure in doing things",
        spoken_prompt=(
            "Over the last two weeks, how often have you had little interest "
            "or pleasure in doing things? You can say: not at all, several "
            "days, more than half the days, or nearly every day."
        ),
    ),
    Phq8Item(
        key="PHQ8_Depressed",
        short="Feeling down or hopeless",
        text="Feeling down, depressed, or hopeless",
        spoken_prompt=(
            "And how often have you been feeling down, depressed, or hopeless?"
        ),
    ),
    Phq8Item(
        key="PHQ8_Sleep",
        short="Sleep trouble",
        text="Trouble falling or staying asleep, or sleeping too much",
        spoken_prompt=(
            "How often have you had trouble falling asleep, staying asleep, "
            "or sleeping too much?"
        ),
    ),
    Phq8Item(
        key="PHQ8_Tired",
        short="Tired or low energy",
        text="Feeling tired or having little energy",
        spoken_prompt=(
            "How often have you been feeling tired, or having little energy?"
        ),
    ),
    Phq8Item(
        key="PHQ8_Appetite",
        short="Appetite change",
        text="Poor appetite or overeating",
        spoken_prompt="How often have you had a poor appetite, or been overeating?",
    ),
    Phq8Item(
        key="PHQ8_Failure",
        short="Feeling bad about yourself",
        text=(
            "Feeling bad about yourself, or that you are a failure, or have let "
            "yourself or your family down"
        ),
        spoken_prompt=(
            "How often have you been feeling bad about yourself, or that you "
            "are a failure, or have let yourself or your family down?"
        ),
    ),
    Phq8Item(
        key="PHQ8_Concentrating",
        short="Concentration trouble",
        text=(
            "Trouble concentrating on things, such as reading the newspaper or "
            "watching television"
        ),
        spoken_prompt=(
            "How often have you had trouble concentrating on things, such as "
            "reading or watching television?"
        ),
    ),
    Phq8Item(
        key="PHQ8_Moving",
        short="Psychomotor change",
        text=(
            "Moving or speaking so slowly that other people could have noticed, "
            "or being so fidgety or restless that you have been moving around a "
            "lot more than usual"
        ),
        spoken_prompt=(
            "And last one. How often have you been moving or speaking so slowly "
            "that other people could have noticed? Or the opposite, being so "
            "fidgety or restless that you move around a lot more than usual?"
        ),
    ),
]

ITEM_KEYS: List[str] = [item.key for item in PHQ8_ITEMS]
MAX_SCORE = 3 * len(PHQ8_ITEMS)  # 24

# Minimum answered items before prorating is considered defensible.
MIN_ANSWERED_FOR_PRORATION = 6


@dataclass
class Phq8Result:
    """A scored PHQ-8 administration."""

    responses: Dict[str, Optional[int]] = field(default_factory=dict)
    total_score: Optional[int] = None
    raw_sum: int = 0
    n_answered: int = 0
    complete: bool = False
    prorated: bool = False
    scorable: bool = False

    band: str = "Not scored"
    band_index: int = -1
    clinically_significant: bool = False
    item_scores: Dict[str, int] = field(default_factory=dict)
    elevated_items: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_score": self.total_score,
            "max_score": MAX_SCORE,
            "raw_sum": self.raw_sum,
            "n_answered": self.n_answered,
            "n_items": len(PHQ8_ITEMS),
            "complete": self.complete,
            "prorated": self.prorated,
            "scorable": self.scorable,
            "band": self.band,
            "band_index": self.band_index,
            "clinically_significant": self.clinically_significant,
            "item_scores": dict(self.item_scores),
            "elevated_items": list(self.elevated_items),
            "notes": list(self.notes),
        }


def band_for_score(score: float, cutoffs: Dict[str, float] | None = None) -> tuple:
    """
    Map a PHQ-8 total to its published severity band.

    Returns (band_name, band_index) with index 0-4.
    """
    c = cutoffs or PROVISIONAL_DEFAULTS["phq8"]
    if score >= c["severe"]:
        return "Severe", 4
    if score >= c["moderately_severe"]:
        return "Moderately severe", 3
    if score >= c["moderate"]:
        return "Moderate", 2
    if score >= c["mild"]:
        return "Mild", 1
    return "None/minimal", 0


def score_phq8(
    responses: Dict[str, Optional[int]],
    cutoffs: Dict[str, float] | None = None,
    allow_proration: bool = True,
) -> Phq8Result:
    """
    Score a set of PHQ-8 responses.

    Missing items are handled explicitly rather than treated as zero. Treating
    an unanswered item as 0 biases the total downward, which on a screening
    instrument means under-referral - the costlier error. With at least
    `MIN_ANSWERED_FOR_PRORATION` items answered the total is prorated to the
    full eight-item scale and flagged as such; below that it is not scored.
    """
    c = cutoffs or PROVISIONAL_DEFAULTS["phq8"]
    out = Phq8Result(responses=dict(responses or {}))

    valid: Dict[str, int] = {}
    for item in PHQ8_ITEMS:
        raw = (responses or {}).get(item.key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            out.notes.append(f"{item.key}: non-numeric response ignored")
            continue
        if not 0 <= value <= 3:
            out.notes.append(f"{item.key}: response {value} out of range 0-3, ignored")
            continue
        valid[item.key] = value

    out.item_scores = valid
    out.n_answered = len(valid)
    out.raw_sum = sum(valid.values())
    out.complete = out.n_answered == len(PHQ8_ITEMS)
    out.elevated_items = sorted(k for k, v in valid.items() if v >= 2)

    if out.complete:
        out.total_score = out.raw_sum
        out.scorable = True
    elif allow_proration and out.n_answered >= MIN_ANSWERED_FOR_PRORATION:
        prorated = out.raw_sum * (len(PHQ8_ITEMS) / out.n_answered)
        out.total_score = int(round(prorated))
        out.prorated = True
        out.scorable = True
        out.notes.append(
            f"{len(PHQ8_ITEMS) - out.n_answered} item(s) unanswered; total "
            f"prorated from {out.n_answered} answered items"
        )
    else:
        out.notes.append(
            f"only {out.n_answered} of {len(PHQ8_ITEMS)} items answered - "
            "not scored (a PHQ-8 total needs at least "
            f"{MIN_ANSWERED_FOR_PRORATION})"
        )
        return out

    out.band, out.band_index = band_for_score(out.total_score, c)
    out.clinically_significant = out.total_score >= c["clinically_significant"]
    return out


def empty_responses() -> Dict[str, Optional[int]]:
    """A blank response set, for seeding UI state."""
    return {item.key: None for item in PHQ8_ITEMS}


def item_by_key(key: str) -> Phq8Item:
    for item in PHQ8_ITEMS:
        if item.key == key:
            return item
    raise KeyError(f"unknown PHQ-8 item: {key}")


__all__ = [
    "ITEM_KEYS",
    "MAX_SCORE",
    "MIN_ANSWERED_FOR_PRORATION",
    "PHQ8_ITEMS",
    "RECALL_STEM",
    "RESPONSE_OPTIONS",
    "Phq8Item",
    "Phq8Result",
    "band_for_score",
    "empty_responses",
    "item_by_key",
    "score_phq8",
]
