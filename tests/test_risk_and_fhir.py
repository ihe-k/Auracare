"""
Triage grading, escalation routing and FHIR-style document construction.

The behaviours pinned here are the ones that decide whether a patient is
escalated, so they are tested at the exact threshold boundaries and for the
failure mode that matters most: a recording that could not be measured must not
trigger an emergency.
"""

from __future__ import annotations

import json

import pytest

from auracare.config import Thresholds
from auracare.features.expiratory import analyse_expiratory
from auracare.features.fluency import analyse_fluency
from auracare.features.phonation import analyse_phonation
from auracare.fhir import (
    SIMULATION_BANNER,
    build_bundle,
    communication_request,
    questionnaire_response,
    summarise_bundle,
)
from auracare.phq8 import ITEM_KEYS, score_phq8
from auracare.risk import (
    CRITICAL,
    ELEVATED,
    NORMAL,
    NOT_MEASURED,
    PATHWAY_CRISIS,
    PATHWAY_EMERGENCY,
    PATHWAY_REVIEW,
    PATHWAY_ROUTINE,
    PATHWAY_URGENT,
    assess_screening,
    assess_task1,
    worst,
)
from auracare.stt import Transcript, analyse_language
from conftest import as_audio, concat, noise_burst, silence, speech_with_pauses, vowel

PATIENT = "NHS-948-204-11"


def phq(level: int) -> dict:
    return {k: level for k in ITEM_KEYS}


# ---------------------------------------------------------------------------
# Level algebra
# ---------------------------------------------------------------------------
def test_worst_ignores_unmeasured():
    assert worst([NORMAL, NOT_MEASURED, ELEVATED]) == ELEVATED
    assert worst([NOT_MEASURED, NOT_MEASURED]) == NOT_MEASURED
    assert worst([]) == NOT_MEASURED
    assert worst([NORMAL, CRITICAL]) == CRITICAL


# ---------------------------------------------------------------------------
# Task 1 grading and the respiratory intercept
# ---------------------------------------------------------------------------
def test_short_phonation_triggers_respiratory_intercept():
    """1.5 s is below the 2.5 s protocol threshold."""
    audio = as_audio(concat(silence(0.3), vowel(130.0, 1.5), silence(0.3)))
    assessment = assess_task1(analyse_phonation(audio), Thresholds())

    assert assessment.measured
    assert assessment.intercept_triggered
    assert assessment.level == CRITICAL
    assert "2.5" in assessment.intercept_reason


def test_adequate_phonation_does_not_trigger_intercept():
    audio = as_audio(concat(silence(0.3), vowel(120.0, 6.0), silence(0.3)))
    assessment = assess_task1(analyse_phonation(audio), Thresholds())
    assert assessment.measured
    assert not assessment.intercept_triggered


def test_unmeasurable_recording_does_not_trigger_intercept():
    """
    The most important negative case. A silent recording yields a phonation
    duration of 0.0 s. An implementation that grades 0.0 against "< 2.5 s"
    dispatches an ambulance for a muted microphone.
    """
    assessment = assess_task1(analyse_phonation(as_audio(silence(10.0))), Thresholds())

    assert not assessment.measured
    assert not assessment.intercept_triggered
    assert assessment.indicators[0].value is None
    assert assessment.indicators[0].level == NOT_MEASURED


# ---------------------------------------------------------------------------
# Task 2 grading is two-sided
# ---------------------------------------------------------------------------
def _task2_level(nominal_s: float):
    from auracare.risk import assess_task2

    audio = as_audio(concat(silence(0.4), noise_burst(nominal_s), silence(0.4)))
    assessment = assess_task2(analyse_expiratory(audio), Thresholds())
    return assessment, assessment.indicators[0].level


def test_normal_exhalation_grades_normal():
    """
    Regression: the low bound was set against the nominal breath length while
    the extractor measures time within 12 dB of peak, so every healthy
    exhalation graded Elevated and every screening escalated.
    """
    for nominal in (1.5, 2.5, 4.0):
        _, level = _task2_level(nominal)
        assert level == NORMAL, f"{nominal}s nominal graded {level}"


def test_very_short_exhalation_is_critical_and_intercepts():
    assessment, level = _task2_level(0.25)
    assert level == CRITICAL
    assert assessment.intercept_triggered


def test_prolonged_exhalation_is_flagged_not_called_normal():
    """Prolonged forced expiration is the obstruction sign; it must not pass."""
    _, level = _task2_level(7.0)
    assert level != NORMAL


def test_exhalation_grading_is_two_sided():
    levels = {n: _task2_level(n)[1] for n in (0.25, 0.8, 2.5, 7.0)}
    assert levels[0.25] == CRITICAL
    assert levels[2.5] == NORMAL
    assert levels[7.0] != NORMAL


def test_threshold_is_configurable_and_changes_the_outcome():
    audio = as_audio(concat(silence(0.3), vowel(130.0, 4.0), silence(0.3)))
    result = analyse_phonation(audio)

    default = assess_task1(result, Thresholds())
    strict = Thresholds()
    strict.set("task1", "phonation_critical_s", 5.0)

    assert not default.intercept_triggered
    assert assess_task1(result, strict).intercept_triggered


# ---------------------------------------------------------------------------
# Whole-screening routing
# ---------------------------------------------------------------------------
def _good_tasks():
    t1 = analyse_phonation(as_audio(concat(silence(0.3), vowel(120.0, 8.0))))
    t2 = analyse_expiratory(as_audio(concat(silence(0.3), noise_burst(1.5))))
    from auracare.features.ddk import analyse_ddk
    from conftest import am_syllable_train

    t3 = analyse_ddk(as_audio(am_syllable_train(6.0, 4.0)))
    t4 = analyse_fluency(
        as_audio(speech_with_pauses([5.0, 5.0], [1.5], lead_s=0.5, trail_s=0.5)),
        transcript=Transcript(
            text=(
                "I can see a sunny park with several children playing football "
                "on the grass while two adults talk beside a wooden bench and a "
                "small brown dog runs towards the pond"
            ),
            available=True,
            backend="manual",
        ),
    )
    return t1, t2, t3, t4


def test_routine_screening_produces_no_escalation():
    t1, t2, t3, t4 = _good_tasks()
    assessment = assess_screening(t1, t2, t3, t4, score_phq8(phq(0)))

    assert assessment.primary_pathway in (PATHWAY_ROUTINE, PATHWAY_REVIEW)
    assert not assessment.is_emergency
    assert not assessment.safety_critical
    assert assessment.n_tasks_measured == 4


def test_severe_phq8_routes_to_urgent():
    t1, t2, t3, t4 = _good_tasks()
    assessment = assess_screening(t1, t2, t3, t4, score_phq8(phq(3)))

    assert PATHWAY_URGENT in assessment.pathways
    assert assessment.primary_pathway == PATHWAY_URGENT
    assert any("severe band" in r for r in assessment.reasons)


def test_moderate_phq8_routes_to_review():
    t1, t2, t3, t4 = _good_tasks()
    responses = phq(0)
    for k in ITEM_KEYS[:4]:
        responses[k] = 3  # total 12 -> moderate
    assessment = assess_screening(t1, t2, t3, t4, score_phq8(responses))

    assert PATHWAY_REVIEW in assessment.pathways
    assert PATHWAY_URGENT not in assessment.pathways


def test_respiratory_emergency_outranks_everything():
    _, t2, t3, t4 = _good_tasks()
    short = analyse_phonation(as_audio(concat(silence(0.3), vowel(130.0, 1.2))))
    assessment = assess_screening(short, t2, t3, t4, score_phq8(phq(3)))

    assert assessment.is_emergency
    assert assessment.primary_pathway == PATHWAY_EMERGENCY
    # The severe PHQ-8 is retained, not discarded.
    assert PATHWAY_URGENT in assessment.pathways


def test_self_harm_language_opens_a_separate_crisis_pathway():
    """
    PHQ-8 omits the self-harm item, so the transcript lexicon is the only route
    to this pathway. It must fire independently of the score.
    """
    t1, t2, t3, _ = _good_tasks()
    t4 = analyse_fluency(
        as_audio(speech_with_pauses([5.0, 5.0], [1.0], lead_s=0.5, trail_s=0.5)),
        transcript=Transcript(
            text="sometimes I think I would be better off dead",
            available=True,
            backend="manual",
        ),
    )
    assessment = assess_screening(t1, t2, t3, t4, score_phq8(phq(0)))

    assert assessment.safety_critical
    assert PATHWAY_CRISIS in assessment.pathways
    assert assessment.safety_terms
    assert any("116 123" in m for m in assessment.patient_messages)


def test_emergency_and_crisis_can_coexist():
    _, t2, t3, _ = _good_tasks()
    short = analyse_phonation(as_audio(concat(silence(0.3), vowel(130.0, 1.0))))
    t4 = analyse_fluency(
        as_audio(speech_with_pauses([4.0], [], lead_s=0.5, trail_s=0.5)),
        transcript=Transcript(
            text="I want to die", available=True, backend="manual"
        ),
    )
    assessment = assess_screening(short, t2, t3, t4, score_phq8(phq(0)))

    assert PATHWAY_EMERGENCY in assessment.pathways
    assert PATHWAY_CRISIS in assessment.pathways
    assert assessment.primary_pathway == PATHWAY_EMERGENCY


def _high_pause_assessment():
    t1, t2, t3, _ = _good_tasks()
    # 1 s speech, 1 s speech, 4 s pause -> ratio 0.667, over the 0.45 threshold
    t4 = analyse_fluency(
        as_audio(speech_with_pauses([1.0, 1.0], [4.0], lead_s=0.5, trail_s=0.5))
    )
    return assess_screening(t1, t2, t3, t4, score_phq8(phq(0)))


def test_high_pause_ratio_triggers_intercept():
    assessment = _high_pause_assessment()
    assert assessment.task(4).intercept_triggered
    assert PATHWAY_URGENT in assessment.pathways


def test_pause_ratio_breach_is_not_a_respiratory_emergency():
    """
    A pause ratio measures language and cognition, not the airway. Routing it
    to the respiratory pathway would have the agent tell a merely hesitant
    patient that their breathing sounds strained.
    """
    assessment = _high_pause_assessment()
    assert not assessment.is_emergency
    assert PATHWAY_EMERGENCY not in assessment.pathways

    joined = " ".join(assessment.patient_messages).lower()
    assert "breathing sounds strained" not in joined
    assert "paused quite a lot" in joined


def test_pause_ratio_breach_carries_a_calibration_warning():
    """
    Measured on ordinary conversational speech this indicator runs 0.38-0.61,
    so the protocol's 0.45 threshold fires on normal speakers. The clinician
    view has to say so rather than presenting it as a finding.
    """
    assessment = _high_pause_assessment()
    actions = " ".join(assessment.clinician_actions).lower()
    assert "calibration" in actions
    assert "recalibrate" in actions


def test_respiratory_intercept_still_routes_to_emergency():
    """The Task 1 route must be unaffected by the Task 4 reclassification."""
    _, t2, t3, t4 = _good_tasks()
    short = analyse_phonation(as_audio(concat(silence(0.3), vowel(130.0, 1.0))))
    assessment = assess_screening(short, t2, t3, t4, score_phq8(phq(0)))
    assert assessment.is_emergency
    assert "breathing sounds strained" in " ".join(
        assessment.patient_messages
    ).lower()


def test_all_silent_recordings_produce_no_escalation():
    """Four failed recordings must produce "could not measure", not an alarm."""
    quiet = as_audio(silence(10.0))
    from auracare.features.ddk import analyse_ddk

    assessment = assess_screening(
        analyse_phonation(quiet),
        analyse_expiratory(quiet),
        analyse_ddk(quiet),
        analyse_fluency(quiet),
        score_phq8({}),
    )

    assert not assessment.is_emergency
    assert assessment.n_tasks_measured == 0
    assert assessment.primary_pathway == PATHWAY_ROUTINE
    assert any("could not be measured" in m for m in assessment.patient_messages)


def test_patient_messages_always_include_the_disclaimer():
    t1, t2, t3, t4 = _good_tasks()
    for level in (0, 1, 2, 3):
        assessment = assess_screening(t1, t2, t3, t4, score_phq8(phq(level)))
        assert any("not a diagnosis" in m for m in assessment.patient_messages)


def test_high_phq8_messages_include_professional_help_guidance():
    t1, t2, t3, t4 = _good_tasks()
    responses = phq(0)
    for k in ITEM_KEYS[:6]:
        responses[k] = 3  # total 18 -> moderately severe
    assessment = assess_screening(t1, t2, t3, t4, score_phq8(responses))

    joined = " ".join(assessment.patient_messages)
    assert "GP" in joined or "mental health professional" in joined
    assert "116 123" in joined


# ---------------------------------------------------------------------------
# FHIR documents
# ---------------------------------------------------------------------------
def test_questionnaire_response_structure():
    resource = questionnaire_response(score_phq8(phq(2)), PATIENT)

    assert resource["resourceType"] == "QuestionnaireResponse"
    assert resource["status"] == "completed"
    assert len(resource["item"]) == 8
    assert resource["subject"]["reference"] == f"Patient/{PATIENT}"
    totals = [
        e["valueInteger"]
        for e in resource["extension"]
        if e["url"].endswith("phq8-total")
    ]
    assert totals == [16]


def test_questionnaire_response_marks_unanswered_items():
    resource = questionnaire_response(score_phq8({k: 1 for k in ITEM_KEYS[:6]}), PATIENT)
    empty = [i for i in resource["item"] if i["answer"] == []]
    assert len(empty) == 2
    assert resource["status"] == "in-progress"


def test_every_resource_carries_the_test_data_tag():
    """
    HTEST marks data that must never be treated as a real clinical record.
    Its absence is how simulated data ends up in a live system.
    """
    t1, t2, t3, t4 = _good_tasks()
    bundle = build_bundle(assess_screening(t1, t2, t3, t4, score_phq8(phq(1))), PATIENT)

    assert bundle["entry"]
    for entry in bundle["entry"]:
        tags = entry["resource"]["meta"]["security"]
        assert any(t["code"] == "HTEST" for t in tags)


def test_bundle_contains_expected_resource_types():
    t1, t2, t3, t4 = _good_tasks()
    bundle = build_bundle(assess_screening(t1, t2, t3, t4, score_phq8(phq(1))), PATIENT)
    counts = summarise_bundle(bundle)

    assert counts["QuestionnaireResponse"] == 1
    assert counts["RiskAssessment"] == 1
    assert counts["CommunicationRequest"] == 1
    assert counts["Observation"] >= 10
    assert bundle["total"] == len(bundle["entry"])


def test_unmeasured_indicators_use_data_absent_reason_not_zero():
    """A missing measurement must be absent, not recorded as a value of 0."""
    quiet = as_audio(silence(8.0))
    from auracare.features.ddk import analyse_ddk

    assessment = assess_screening(
        analyse_phonation(quiet),
        analyse_expiratory(quiet),
        analyse_ddk(quiet),
        analyse_fluency(quiet),
        score_phq8({}),
    )
    bundle = build_bundle(assessment, PATIENT)

    observations = [
        e["resource"] for e in bundle["entry"]
        if e["resource"]["resourceType"] == "Observation"
    ]
    absent = [o for o in observations if "dataAbsentReason" in o]
    assert absent
    for o in absent:
        assert "valueQuantity" not in o


def test_communication_request_priority_by_pathway():
    t1, t2, t3, t4 = _good_tasks()

    routine = communication_request(
        assess_screening(t1, t2, t3, t4, score_phq8(phq(0))), PATIENT
    )
    assert routine["priority"] in ("routine",)

    urgent = communication_request(
        assess_screening(t1, t2, t3, t4, score_phq8(phq(3))), PATIENT
    )
    assert urgent["priority"] == "urgent"

    short = analyse_phonation(as_audio(concat(silence(0.3), vowel(130.0, 1.0))))
    stat = communication_request(
        assess_screening(short, t2, t3, t4, score_phq8(phq(0))), PATIENT
    )
    assert stat["priority"] == "stat"
    assert "emergency-triage-queue" in stat["recipient"][0]["reference"]


def test_communication_request_payload_is_marked_simulated():
    t1, t2, t3, t4 = _good_tasks()
    resource = communication_request(
        assess_screening(t1, t2, t3, t4, score_phq8(phq(0))), PATIENT
    )
    payloads = [p["contentString"] for p in resource["payload"]]
    assert any(SIMULATION_BANNER == p for p in payloads)
    assert any("SIMULATED" in p for p in payloads)


def test_bundle_is_json_serialisable():
    t1, t2, t3, t4 = _good_tasks()
    bundle = build_bundle(assess_screening(t1, t2, t3, t4, score_phq8(phq(2))), PATIENT)
    assert json.loads(json.dumps(bundle))["resourceType"] == "Bundle"


def test_bundle_ids_are_deterministic_for_identical_input():
    t1, t2, t3, t4 = _good_tasks()
    assessment = assess_screening(t1, t2, t3, t4, score_phq8(phq(1)))
    ts = "2026-01-01T00:00:00Z"
    a = build_bundle(assessment, PATIENT, timestamp=ts)
    b = build_bundle(assessment, PATIENT, timestamp=ts)
    assert a["id"] == b["id"]


def test_filtering_normal_observations_reduces_the_bundle():
    t1, t2, t3, t4 = _good_tasks()
    assessment = assess_screening(t1, t2, t3, t4, score_phq8(phq(1)))
    full = build_bundle(assessment, PATIENT, include_normal_observations=True)
    trimmed = build_bundle(assessment, PATIENT, include_normal_observations=False)
    assert trimmed["total"] <= full["total"]
