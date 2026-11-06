"""
Streamlit app smoke tests.

Runs the real `app.py` through Streamlit's own test harness so that render
errors, bad widget arguments and broken state transitions fail here rather
than in front of the client. Recordings are injected directly into session
state, because `st.audio_input` has no scripted equivalent.

The two behaviours asserted at the end are the specific defects the client
asked to have fixed: the emergency banner appearing before anything had been
measured, and the "check-in complete" message showing during an escalation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auracare.features.ddk import analyse_ddk
from auracare.features.expiratory import analyse_expiratory
from auracare.features.fluency import analyse_fluency
from auracare.features.phonation import analyse_phonation
from auracare.phq8 import ITEM_KEYS
from auracare.stt import Transcript
from conftest import (
    am_syllable_train,
    as_audio,
    concat,
    noise_burst,
    silence,
    speech_with_pauses,
    vowel,
)

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 120


def run_app(**state):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def healthy_results(transcript_text: str | None = None):
    t4_audio = as_audio(
        speech_with_pauses([5.0, 5.0], [1.5], lead_s=0.5, trail_s=0.5)
    )
    transcript = (
        Transcript(text=transcript_text, available=True, backend="manual")
        if transcript_text
        else None
    )
    return {
        1: analyse_phonation(
            as_audio(concat(silence(0.3), vowel(120.0, 9.0), silence(0.3)))
        ),
        2: analyse_expiratory(
            as_audio(concat(silence(0.3), noise_burst(1.5), silence(0.3)))
        ),
        3: analyse_ddk(as_audio(am_syllable_train(6.0, 4.0))),
        4: analyse_fluency(t4_audio, transcript=transcript),
    }


def emergency_results():
    results = healthy_results()
    results[1] = analyse_phonation(
        as_audio(concat(silence(0.3), vowel(135.0, 1.2), silence(0.3)))
    )
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_app_starts_without_exception():
    at = run_app()
    assert not at.exception
    assert at.title[0].value.startswith("\U0001fa7a Auracare")


def test_all_five_tabs_render():
    at = run_app()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert labels == [
        "Check-in",
        "Your results",
        "Clinical / EHR view",
        "MRI classifier",
        "Method & limits",
    ]


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_each_task_step_renders(step):
    at = run_app(step=step)
    assert not at.exception, f"step {step} raised: {at.exception}"


def test_phq8_step_renders_all_eight_items():
    at = run_app(step=5)
    assert not at.exception
    keys = {r.key for r in at.radio}
    for item_key in ITEM_KEYS:
        assert f"phq_{item_key}" in keys


def test_completion_step_renders():
    at = run_app(step=6, finished=True, results=healthy_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception


def test_patient_and_clinical_views_render_with_results():
    at = run_app(
        step=6,
        finished=True,
        results=healthy_results(),
        phq8_responses={k: 1 for k in ITEM_KEYS},
    )
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Auracare" in body


def test_views_render_before_any_recording():
    """Empty state must not raise; it should invite the user to start."""
    at = run_app()
    assert not at.exception
    info = " ".join(i.value for i in at.info)
    assert "Complete the check-in first" in info


# ---------------------------------------------------------------------------
# The two reported defects
# ---------------------------------------------------------------------------
# The verbatim opening of the emergency banner. Matched in full rather than by
# keyword: the Method & limits tab discusses the phrase "breathing sounds
# strained" while explaining why a pause-ratio breach must not use it, and a
# loose substring check picks that explanation up as a false positive.
EMERGENCY_BANNER = "based on your check-in, your breathing sounds strained"
EMERGENCY_ROUTING = "routing your assessment to emergency triage"


def _error_text(at) -> str:
    return " ".join(e.value for e in at.error).lower()


def test_no_emergency_banner_before_anything_is_measured():
    """
    Reported defect: the emergency banner appeared at the start of the flow.
    With no recordings at all, nothing may claim an emergency.
    """
    at = run_app()
    assert not at.exception
    text = _error_text(at)
    assert EMERGENCY_BANNER not in text
    assert EMERGENCY_ROUTING not in text


def test_no_emergency_banner_when_recordings_are_unmeasurable():
    """Silent recordings must not be read as a breach."""
    quiet = as_audio(silence(10.0))
    results = {
        1: analyse_phonation(quiet),
        2: analyse_expiratory(quiet),
        3: analyse_ddk(quiet),
        4: analyse_fluency(quiet),
    }
    at = run_app(step=6, finished=True, results=results,
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    text = _error_text(at)
    assert EMERGENCY_BANNER not in text
    assert EMERGENCY_ROUTING not in text


def test_emergency_banner_appears_when_threshold_is_breached():
    at = run_app(step=6, finished=True, results=emergency_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    text = _error_text(at)
    assert EMERGENCY_BANNER in text
    assert EMERGENCY_ROUTING in text


def test_completion_message_suppressed_during_escalation():
    """
    Reported defect: "check-in complete" was shown during an escalation,
    contradicting the emergency banner directly above it.
    """
    at = run_app(step=6, finished=True, results=emergency_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    success_text = " ".join(s.value for s in at.success)
    assert "Check-in complete" not in success_text


def test_completion_message_shown_on_a_routine_result():
    at = run_app(step=6, finished=True, results=healthy_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    success_text = " ".join(s.value for s in at.success)
    assert "Check-in complete" in success_text


def test_crisis_language_surfaces_support_resources():
    results = healthy_results(
        transcript_text="honestly I think I would be better off dead"
    )
    at = run_app(step=6, finished=True, results=results,
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    text = " ".join(e.value for e in at.error)
    assert "116 123" in text or "harming yourself" in text


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def test_threshold_controls_are_present_and_editable():
    at = run_app()
    assert not at.exception
    keys = {n.key for n in at.number_input}
    assert "thr_task1_phonation_critical_s" in keys
    assert "thr_task4_pause_ratio_critical" in keys


def test_changing_a_threshold_changes_the_outcome():
    """A 4 s phonation is fine by default and a breach at a 5 s threshold."""
    results = healthy_results()
    results[1] = analyse_phonation(
        as_audio(concat(silence(0.3), vowel(130.0, 4.0), silence(0.3)))
    )
    base = run_app(step=6, finished=True, results=results,
                   phq8_responses={k: 0 for k in ITEM_KEYS})
    assert "emergency triage" not in " ".join(e.value for e in base.error).lower()

    strict = run_app(step=6, finished=True, results=results,
                     phq8_responses={k: 0 for k in ITEM_KEYS})
    strict.number_input(key="thr_task1_phonation_critical_s").set_value(5.0).run()
    assert not strict.exception
    assert "emergency triage" in " ".join(e.value for e in strict.error).lower()


def test_patient_id_is_configurable():
    at = run_app()
    assert not at.exception
    assert at.text_input(key="patient_id").value == "NHS-948-204-11"
