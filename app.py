"""
Auracare - guided voice screening prototype.

Run with:  streamlit run app.py

Scope and honesty, stated once here and repeated in the UI:
  * This is a screening prototype, not a medical device and not a diagnosis.
  * Acoustic thresholds are provisional engineering values, tunable from the
    sidebar, and separate from the published clinical reference values.
  * Depression severity is scored from the PHQ-8 against published cut-points.
  * All emergency dispatches and FHIR messages are simulated. Nothing is
    transmitted anywhere.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from auracare import paths, ui
from auracare.audio_io import AudioDecodeError, check_quality, load_audio
from auracare.config import (
    CLINICAL_REFERENCE,
    PROVISIONAL_DEFAULTS,
    SignalParams,
    Thresholds,
)
from auracare.features.ddk import analyse_ddk
from auracare.features.expiratory import analyse_expiratory
from auracare.features.fluency import analyse_fluency
from auracare.features.phonation import analyse_phonation
from auracare.features.voice import PARSELMOUTH_AVAILABLE
from auracare.fhir import SIMULATION_BANNER, build_bundle, summarise_bundle
from auracare.phq8 import (
    MAX_SCORE,
    PHQ8_ITEMS,
    RECALL_STEM,
    RESPONSE_OPTIONS,
    empty_responses,
    score_phq8,
)
from auracare.risk import (
    CRISIS_RESOURCES_UK,
    ELEVATED,
    LEVEL_RANK,
    NOT_MEASURED,
    PATHWAY_CRISIS,
    PATHWAY_EMERGENCY,
    PATHWAY_LABEL,
    PATHWAY_ROUTINE,
    assess_screening,
)
from auracare.stt import analyse_language, available_backends, transcribe

st.set_page_config(
    page_title="Auracare voice screening",
    page_icon="\U0001fa7a",
    layout="wide",
)

ANALYSERS = {
    1: analyse_phonation,
    2: analyse_expiratory,
    3: analyse_ddk,
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "step": 1,
        "recordings": {},       # task_id -> raw bytes
        "fingerprints": {},     # task_id -> sha1 of the bytes analysed
        "results": {},          # task_id -> analysis result object
        "quality": {},          # task_id -> QualityReport
        "transcript": None,
        "phq8_responses": empty_responses(),
        "thresholds": Thresholds(),
        "patient_id": "NHS-948-204-11",
        "log": [],
        "intercepts": [],
        "finished": False,
        "spoken": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_agent(text: str) -> None:
    if not st.session_state.log or st.session_state.log[-1] != ("agent", text):
        st.session_state.log.append(("agent", text))


def log_patient(text: str) -> None:
    st.session_state.log.append(("patient", text))


def log_system(text: str) -> None:
    st.session_state.log.append(("system", text))


def reset_all() -> None:
    for key in (
        "step", "recordings", "fingerprints", "results", "quality",
        "transcript", "phq8_responses", "log", "intercepts", "finished",
        "spoken",
    ):
        st.session_state.pop(key, None)
    init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> Dict[str, object]:
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.text_input("Patient ID", key="patient_id")

    st.sidebar.markdown("#### Speech-to-text")
    backends = available_backends()
    if backends:
        stt_on = st.sidebar.toggle(
            "Transcribe Task 4", value=True,
            help="Needed for negative-language flagging and word-rate measures.",
        )
        model_size = st.sidebar.selectbox(
            "Whisper model",
            ["tiny.en", "base.en", "small.en"],
            index=1,
            help="tiny is roughly 3x faster; base is more accurate. "
                 "Expect transcription to take about as long as the recording.",
        )
    else:
        stt_on = False
        model_size = "base.en"
        st.sidebar.warning(
            "No speech-to-text backend installed. Acoustic measures still "
            "work; transcript-based measures will be skipped.\n\n"
            "`pip install faster-whisper`"
        )

    speak = st.sidebar.toggle(
        "Speak prompts aloud", value=False,
        help="Uses your browser's built-in speech synthesis. No audio leaves "
             "this device.",
    )

    st.sidebar.markdown("#### Provisional thresholds")
    st.sidebar.caption(
        "Prototype operating points. Tunable, and deliberately separate from "
        "the published clinical values below."
    )
    thresholds: Thresholds = st.session_state.thresholds
    group_labels = {
        "task1": "Task 1 · phonation",
        "task2": "Task 2 · exhalation",
        "task3": "Task 3 · DDK",
        "task4": "Task 4 · fluency",
    }
    for group, label in group_labels.items():
        with st.sidebar.expander(label):
            for key, default in PROVISIONAL_DEFAULTS[group].items():
                current = thresholds.get(group, key)
                step = 0.01 if abs(default) < 2 else 0.1
                new = st.number_input(
                    key.replace("_", " "),
                    value=float(current),
                    step=float(step),
                    format="%.3f",
                    key=f"thr_{group}_{key}",
                )
                if new != current:
                    thresholds.set(group, key, new)

    with st.sidebar.expander("PHQ-8 cut-points (published)"):
        st.caption(
            "These are the Kroenke 2009 cut-points. They are shown for "
            "transparency and are not intended to be tuned."
        )
        for key, value in PROVISIONAL_DEFAULTS["phq8"].items():
            st.text(f"{key.replace('_', ' '):<24}{value}")

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Reset thresholds", use_container_width=True):
        thresholds.reset()
        st.rerun()
    st.sidebar.download_button(
        "Export JSON",
        data=thresholds.to_json(),
        file_name="auracare_thresholds.json",
        mime="application/json",
        use_container_width=True,
    )
    uploaded = st.sidebar.file_uploader(
        "Import thresholds", type=["json"], key="thr_upload"
    )
    if uploaded is not None:
        try:
            st.session_state.thresholds = Thresholds.from_dict(
                json.loads(uploaded.getvalue().decode("utf-8"))
            )
            st.sidebar.success("Thresholds loaded.")
        except Exception as exc:
            st.sidebar.error(f"Could not read that file: {exc}")

    st.sidebar.markdown("#### Environment")
    found = paths.describe()
    st.sidebar.caption(
        f"Parselmouth: {'ready' if PARSELMOUTH_AVAILABLE else 'MISSING'}  \n"
        f"Speech-to-text: {', '.join(backends) or 'none'}  \n"
        f"Severity model: {'found' if found['severity_model'] else 'not found'}  \n"
        f"Image model: {'found' if found['image_model'] else 'not found'}"
    )

    if st.sidebar.button("\U0001f501 Start over", type="secondary",
                         use_container_width=True):
        reset_all()
        st.rerun()

    return {"stt_on": stt_on, "model_size": model_size, "speak": speak}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse_task(task_id: int, raw: bytes, settings: Dict[str, object]) -> None:
    """Decode, quality-check and analyse one task recording."""
    fingerprint = hashlib.sha1(raw).hexdigest()
    if st.session_state.fingerprints.get(task_id) == fingerprint:
        return  # already analysed this exact recording

    params = SignalParams()
    try:
        audio = load_audio(raw)
    except AudioDecodeError as exc:
        st.error(f"Could not read that recording: {exc}")
        return

    quality = check_quality(audio)
    st.session_state.quality[task_id] = quality

    with st.spinner(f"Analysing Task {task_id}…"):
        if task_id == 4:
            transcript = None
            if settings["stt_on"]:
                transcript = transcribe(
                    audio, model_size=str(settings["model_size"])
                )
                st.session_state.transcript = transcript
            result = analyse_fluency(audio, params, transcript=transcript)
        else:
            result = ANALYSERS[task_id](audio, params)

    st.session_state.results[task_id] = result
    st.session_state.fingerprints[task_id] = fingerprint

    log_patient(f"(recorded Task {task_id}, {audio.duration_s:.1f} s)")
    log_system(summarise_result(task_id, result))


def summarise_result(task_id: int, result) -> str:
    """One compact telemetry line for the conversation log."""
    if task_id == 1:
        return (
            f"telemetry · phonation {ui.fmt_value(result.phonation_duration_s, 's')}"
            f" · jitter {ui.fmt_value(result.voice.jitter_local_pct, '%')}"
            f" · shimmer {ui.fmt_value(result.voice.shimmer_local_pct, '%')}"
            f" · HNR {ui.fmt_value(result.voice.hnr_db, 'dB')}"
        )
    if task_id == 2:
        return (
            f"telemetry · exhalation {ui.fmt_value(result.exhalation_duration_s, 's')}"
            f" · rise {ui.fmt_value(result.rise_time_s, 's')}"
            f" · wheeze index {ui.fmt_value(result.wheeze_index_db, 'dB')}"
        )
    if task_id == 3:
        return (
            f"telemetry · rate {ui.fmt_value(result.syllable_rate_hz, 'syll/s')}"
            f" · {result.n_syllables} syllables"
            f" · rhythm CV {ui.fmt_value(result.rhythm_cv, '', 3)}"
        )
    return (
        f"telemetry · pause ratio {ui.fmt_value(result.pause_ratio, '', 3)}"
        f" · speech rate {ui.fmt_value(result.speech_rate_sylls, 'syll/s')}"
        f" · {result.n_pauses} pauses"
    )


def current_assessment():
    """Grade whatever has been measured so far."""
    return assess_screening(
        st.session_state.results.get(1),
        st.session_state.results.get(2),
        st.session_state.results.get(3),
        st.session_state.results.get(4),
        score_phq8(st.session_state.phq8_responses),
        st.session_state.thresholds,
    )


def check_intercept(task_id: int) -> Optional[str]:
    """
    Did this task trip a safety intercept?

    Only fires on a *measured* breach. A recording that could not be measured
    yields None from the extractor and is never an intercept - the original
    draft showed its emergency banner before any analysis had run at all.
    """
    assessment = current_assessment()
    task = assessment.task(task_id)
    if task is not None and task.intercept_triggered:
        return task.intercept_reason
    return None


# ---------------------------------------------------------------------------
# Screening flow
# ---------------------------------------------------------------------------
def render_task_step(step: int, settings: Dict[str, object]) -> None:
    script = ui.script_for(step)
    done = {i: i in st.session_state.results for i in range(1, 5)}

    st.markdown(
        ui.progress_dots(step, 4, done)
        + f"&nbsp;&nbsp;<small>Task {step} of 4 &middot; target window "
          f"{script.window}</small>",
        unsafe_allow_html=True,
    )

    col_chat, col_cue = st.columns([3, 2], gap="large")

    with col_chat:
        log_agent(script.prompt)
        for role, text in st.session_state.log:
            if role == "agent":
                st.markdown(ui.agent_bubble(text), unsafe_allow_html=True)
            elif role == "patient":
                st.markdown(ui.patient_bubble(text), unsafe_allow_html=True)
            else:
                st.markdown(ui.system_bubble(text), unsafe_allow_html=True)

        if settings["speak"]:
            nonce = f"task{step}"
            if nonce not in st.session_state.spoken:
                components.html(ui.speak_script(script.prompt, nonce), height=0)
                st.session_state.spoken.add(nonce)

        st.markdown(f"**{script.instruction}**")
        st.caption(
            f"Aim for about {script.record_seconds} seconds. Press record, "
            "complete the task, then press stop."
        )

        raw_file = st.audio_input(
            f"Record Task {step}", key=f"rec_{step}", label_visibility="collapsed"
        )
        if raw_file is not None:
            analyse_task(step, raw_file.getvalue(), settings)

    with col_cue:
        if script.cue_class:
            st.markdown(ui.cue_html(script), unsafe_allow_html=True)
        else:
            stimulus = paths.ASSETS_DIR / "stimulus_park.png"
            if stimulus.exists():
                st.image(str(stimulus), use_container_width=True)
                st.caption(
                    "Describe what you see happening. Original artwork - the "
                    "established clinical cards are under copyright."
                )
            else:
                st.warning(
                    "Stimulus image missing. Generate it with "
                    "`python scripts/make_stimulus.py`."
                )
        st.caption(f"**Safety check:** {script.safety_check}")

    # -- results for this task ---------------------------------------------
    result = st.session_state.results.get(step)
    if result is None:
        return

    quality = st.session_state.quality.get(step)
    if quality is not None:
        for problem in quality.problems:
            st.error(f"Recording problem: {problem}")
        for warning in quality.warnings:
            st.warning(warning)

    st.divider()
    render_task_metrics(step, result)

    for note in getattr(result, "notes", []):
        st.info(note)

    reason = check_intercept(step)
    if reason:
        if step not in st.session_state.intercepts:
            st.session_state.intercepts.append(step)
        st.error(
            f"**Safety intercept — Task {step}.** {reason}\n\n"
            f"{ui.EMERGENCY_PROMPT}"
        )
        if settings["speak"]:
            nonce = f"intercept{step}"
            if nonce not in st.session_state.spoken:
                components.html(ui.speak_script(ui.EMERGENCY_PROMPT, nonce),
                                height=0)
                st.session_state.spoken.add(nonce)
        log_agent(ui.EMERGENCY_PROMPT)

    st.write("")
    col_a, col_b = st.columns([1, 3])
    label = "Continue to the questions" if step == 4 else f"Continue to Task {step + 1}"
    if col_a.button(label, type="primary", use_container_width=True):
        st.session_state.step = step + 1
        st.rerun()
    if reason:
        col_b.caption(
            "You can continue the check-in; the intercept has already been "
            "recorded and appears in the clinical view."
        )


def render_task_metrics(task_id: int, result) -> None:
    """Near-real-time telemetry for the task just recorded."""
    st.markdown(f"##### Task {task_id} measurements")
    assessment = current_assessment()
    task = assessment.task(task_id)

    cols = st.columns(min(4, max(1, len(task.indicators))))
    for i, indicator in enumerate(task.indicators):
        with cols[i % len(cols)]:
            st.metric(
                indicator.label,
                ui.fmt_value(indicator.value, indicator.unit, 2),
            )
            st.markdown(ui.level_pill(indicator.level), unsafe_allow_html=True)
            st.caption(indicator.detail)

    if task_id == 4:
        transcript = st.session_state.transcript
        if transcript is not None and transcript.available:
            with st.expander("Transcript and language flags", expanded=True):
                st.write(transcript.text or "_(no speech transcribed)_")
                flags = analyse_language(transcript.text)
                if flags.matched_terms:
                    st.markdown("**Lexicon matches**")
                    for category, terms in flags.matched_terms.items():
                        st.markdown(f"- `{category}`: {', '.join(terms)}")
                    st.caption(
                        "Matched text is shown so each flag can be checked "
                        "against the transcript. The lexicon cannot tell "
                        "disclosure from quotation or negation."
                    )
                else:
                    st.caption("No lexicon matches.")
        elif transcript is not None:
            st.info(
                f"Transcript unavailable ({transcript.unavailable_reason}). "
                "Acoustic measures above are unaffected."
            )


def render_phq8_step(settings: Dict[str, object]) -> None:
    st.markdown("### A few questions about how you have been feeling")
    st.caption(RECALL_STEM)
    st.info(
        "These eight questions are the PHQ-8, a standard screening "
        "questionnaire. Your answers are scored by simple addition against "
        "published cut-points."
    )

    responses = dict(st.session_state.phq8_responses)
    for index, item in enumerate(PHQ8_ITEMS, start=1):
        st.markdown(ui.agent_bubble(item.spoken_prompt), unsafe_allow_html=True)
        current = responses.get(item.key)
        choice = st.radio(
            f"{index}. {item.text}",
            options=list(range(4)),
            format_func=lambda v: RESPONSE_OPTIONS[v],
            index=None if current is None else int(current),
            key=f"phq_{item.key}",
            horizontal=True,
        )
        responses[item.key] = choice
        st.write("")
    st.session_state.phq8_responses = responses

    result = score_phq8(responses)
    if result.scorable:
        st.success(
            f"PHQ-8 total: **{result.total_score} / {MAX_SCORE}** "
            f"— {result.band}"
            + ("  (prorated from partial answers)" if result.prorated else "")
        )
    else:
        answered = result.n_answered
        st.caption(f"{answered} of 8 answered. At least 6 are needed to score.")

    col_a, col_b = st.columns([1, 3])
    if col_a.button("Finish check-in", type="primary", use_container_width=True):
        st.session_state.finished = True
        st.session_state.step = 6
        st.rerun()
    col_b.caption(
        "You can finish with questions unanswered; the score is prorated and "
        "flagged as such, or withheld if too few are answered."
    )


def render_completion(settings: Dict[str, object]) -> None:
    assessment = current_assessment()
    escalating = assessment.primary_pathway != PATHWAY_ROUTINE

    if PATHWAY_EMERGENCY in assessment.pathways:
        st.error(f"### ⚠️ {ui.EMERGENCY_PROMPT}")
        spoken = ui.EMERGENCY_PROMPT
    elif PATHWAY_CRISIS in assessment.pathways:
        st.error(f"### {ui.CRISIS_PROMPT}")
        spoken = ui.CRISIS_PROMPT
    elif escalating:
        st.warning(
            "### Check-in complete, and we would like a clinician to look at "
            "your results."
        )
        spoken = (
            "Check-in complete. We would like a clinician to look at your "
            "results, and someone will be in touch."
        )
    else:
        # Only say "complete" when nothing is being escalated. The original
        # draft showed this message during an escalation, which contradicted
        # the emergency banner directly above it.
        st.success(f"### ✅ {ui.COMPLETION_PROMPT}")
        spoken = ui.COMPLETION_PROMPT

    if settings["speak"] and "completion" not in st.session_state.spoken:
        components.html(ui.speak_script(spoken, "completion"), height=0)
        st.session_state.spoken.add("completion")

    st.caption(
        "Open the **Your results** and **Clinical / EHR view** tabs above for "
        "the full picture."
    )
    if st.button("Start a new check-in"):
        reset_all()
        st.rerun()


# ---------------------------------------------------------------------------
# Result views
# ---------------------------------------------------------------------------
def render_patient_view() -> None:
    st.markdown("### Your results")
    if not st.session_state.results and not st.session_state.finished:
        st.info("Complete the check-in first and your results will appear here.")
        return

    assessment = current_assessment()

    if PATHWAY_EMERGENCY in assessment.pathways:
        st.error("**We are routing your assessment to emergency triage.**")
    if PATHWAY_CRISIS in assessment.pathways:
        st.error(f"**Support is available right now.** {CRISIS_RESOURCES_UK}")

    for message in assessment.patient_messages:
        st.write(f"- {message}")

    st.divider()
    phq8 = assessment.phq8
    if phq8 is not None and phq8.scorable:
        st.markdown("#### How you have been feeling")
        st.progress(
            min(1.0, (phq8.total_score or 0) / MAX_SCORE),
            text=f"PHQ-8 {phq8.total_score} of {MAX_SCORE} — {phq8.band}",
        )

    st.markdown("#### What we measured")
    st.caption(
        "Plain-language summary. These are experimental voice indicators, not "
        "a diagnosis."
    )
    friendly = {
        1: "How long you could hold a note, and how steady your voice was",
        2: "How strongly and quickly you could breathe out",
        3: "How fast and evenly you could repeat sounds",
        4: "How you spoke when describing the picture",
    }
    for task in assessment.tasks:
        if not task.measured:
            st.write(f"- **{friendly[task.task_id]}** — could not be measured")
            continue
        st.write(
            f"- **{friendly[task.task_id]}** — "
            f"{'looks typical' if LEVEL_RANK.get(task.level, 0) <= 1 else 'worth a closer look'}"
        )


def render_clinical_view() -> None:
    st.markdown("### Clinical / EHR view")
    if not st.session_state.results and not st.session_state.finished:
        st.info("Complete the check-in first.")
        return

    assessment = current_assessment()
    patient_id = st.session_state.patient_id

    st.markdown(
        f'<div class="sim-banner">{SIMULATION_BANNER}</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Overall indicator level", assessment.overall_level)
    col_b.metric("Tasks measured", f"{assessment.n_tasks_measured}/4")
    phq8 = assessment.phq8
    col_c.metric(
        "PHQ-8",
        f"{phq8.total_score}/{MAX_SCORE}" if (phq8 and phq8.scorable) else "not scored",
        phq8.band if (phq8 and phq8.scorable) else None,
    )
    col_d.metric("Pathway", PATHWAY_LABEL[assessment.primary_pathway])

    if assessment.reasons:
        st.markdown("#### Reasons")
        for reason in assessment.reasons:
            st.write(f"- {reason}")

    st.markdown("#### Recommended actions")
    for action in assessment.clinician_actions:
        st.write(f"- {action}")

    st.markdown("#### Indicator table")
    rows = []
    for task in assessment.tasks:
        for indicator in task.indicators:
            rows.append(
                {
                    "Task": task.task_id,
                    "Domain": task.title,
                    "Indicator": indicator.label,
                    "Value": (
                        None if indicator.value is None
                        else round(float(indicator.value), 4)
                    ),
                    "Unit": indicator.unit,
                    "Level": indicator.level,
                    "Threshold": indicator.detail,
                }
            )
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)

    elevated = frame[frame["Level"].isin(["Elevated", "Critical"])]
    if not elevated.empty:
        st.warning(
            f"{len(elevated)} indicator(s) at Elevated or Critical. "
            "These are provisional prototype thresholds."
        )

    st.markdown("#### Simulated FHIR dispatch")
    bundle = build_bundle(assessment, patient_id)
    counts = summarise_bundle(bundle)
    st.caption(
        " · ".join(f"{v}× {k}" for k, v in sorted(counts.items()))
        + f"  — {bundle['total']} resources, all tagged HTEST"
    )

    comm = next(
        e["resource"] for e in bundle["entry"]
        if e["resource"]["resourceType"] == "CommunicationRequest"
    )
    priority = comm["priority"]
    recipient = comm["recipient"][0]["reference"]
    if priority == "stat":
        st.error(f"**STAT** → `{recipient}`")
    elif priority == "urgent":
        st.warning(f"**URGENT** → `{recipient}`")
    else:
        st.success(f"**ROUTINE** → `{recipient}`")
    st.code(comm["payload"][0]["contentString"], language=None)

    with st.expander("Full FHIR-style bundle"):
        st.json(bundle, expanded=False)
    st.download_button(
        "Download bundle (JSON)",
        data=json.dumps(bundle, indent=2),
        file_name=f"auracare_{patient_id}_bundle.json",
        mime="application/json",
    )

    transcript = st.session_state.transcript
    if transcript is not None and transcript.available:
        with st.expander("Task 4 transcript (verbatim)"):
            st.write(transcript.text)
            st.caption(
                "Always review the transcript directly. The lexicon matches "
                "surface strings and cannot interpret negation, quotation or "
                "hypothetical speech."
            )


def render_reference_view() -> None:
    st.markdown("### Method, thresholds and limitations")

    st.markdown("#### What this prototype does and does not claim")
    st.markdown(
        "- It is a **screening prototype**, not a medical device, and nothing "
        "here is a diagnosis.\n"
        "- Depression severity is scored from the **PHQ-8 by summation "
        "against published cut-points**. No model is involved in that number.\n"
        "- The acoustic measures in Tasks 1-4 are **experimental indicators**. "
        "They are graded against provisional thresholds chosen to drive this "
        "prototype, not against validated clinical cut-offs.\n"
        "- Task 2's measures are acoustic proxies on an uncalibrated "
        "microphone. They are **not spirometry** and no number here is a peak "
        "flow.\n"
        "- All emergency dispatch and FHIR messaging is **simulated**. Nothing "
        "is transmitted to any system or service."
    )

    st.error(
        "**Calibration warning — Task 4 pause ratio.** The protocol "
        "specifies an intercept at 0.45. Measured across 76 thirty-second "
        "windows of the supplied sample recordings, this indicator ranged "
        "0.24–0.72 with a median of **0.48** and exceeded 0.45 in "
        "**57%** of windows. The threshold therefore sits below the median of "
        "ordinary speech and, as specified, cannot separate a struggling "
        "patient from an ordinary one. It is kept as the default because it is "
        "the specified value, and it is tunable in the sidebar. Recalibrate it "
        "against real Task 4 recordings before relying on it — "
        "`scripts/calibrate_pause_ratio.py` reproduces the measurement.\n\n"
        "A breach routes to clinical review, not to the respiratory emergency "
        "pathway: a pause ratio measures language and cognition, so telling "
        "the patient their breathing sounds strained would be wrong."
    )

    st.markdown("#### Provisional thresholds in use")
    rows = []
    for group, items in st.session_state.thresholds.as_dict().items():
        for key, value in items.items():
            rows.append(
                {
                    "Group": group,
                    "Threshold": key,
                    "Current": value,
                    "Default": PROVISIONAL_DEFAULTS[group][key],
                    "Changed": value != PROVISIONAL_DEFAULTS[group][key],
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Published clinical reference values")
    st.caption(
        "Shown for comparison only. Nothing in the app branches on these "
        "numbers; they exist so the distance between a tuned demo threshold "
        "and the literature stays visible."
    )
    for group, entries in CLINICAL_REFERENCE.items():
        with st.expander(group):
            for label, value in entries.items():
                st.write(f"**{label}** — {value}")


def render_imaging_view() -> None:
    st.markdown("### MRI dementia-stage classifier")

    model_path = paths.find_image_model()
    if model_path is None:
        st.warning(
            "No classifier artifact found. Set `AURACARE_TRIAGE_REPO` to your "
            "Cognitive_AI_Triage checkout, or put the .pkl in `models/`."
        )
        return

    from auracare.imaging.alzheimer import (
        classify_image,
        load_image_model,
    )

    @st.cache_resource(show_spinner="Loading classifier…")
    def _load(path_str: str):
        return load_image_model(path_str)

    try:
        bundle = _load(str(model_path))
    except Exception as exc:
        st.error(f"Could not load the classifier: {exc}")
        return

    st.caption(f"Artifact: `{model_path.name}` · {bundle.n_features} features")
    st.info(
        "**Fixed in this build.** The previous inference path built 44 "
        "features and tiled them to 50,436; the model was trained on 50,176 "
        "flattened pixels plus 4 statistics plus a 256-bin histogram. Every "
        "input was in the wrong place, so predictions collapsed onto one or "
        "two classes. This build reproduces the training extraction exactly."
    )

    note = bundle.coverage_note()
    if note:
        st.warning(f"**Model coverage.** {note}")

    from auracare.imaging.alzheimer import MODEL_CAVEAT

    st.caption(f"⚠️ {MODEL_CAVEAT}")

    uploaded = st.file_uploader(
        "Upload an MRI slice", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )
    practice = paths.find_practice_files()
    sample = None
    if practice is not None:
        options = sorted(
            p for p in practice.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if options:
            chosen = st.selectbox(
                "…or use a sample image",
                ["(none)"] + [p.name for p in options],
            )
            if chosen != "(none)":
                sample = next(p for p in options if p.name == chosen)

    source = uploaded if uploaded is not None else sample
    if source is None:
        return

    col_img, col_res = st.columns([1, 1], gap="large")
    with col_img:
        st.image(source if uploaded is None else uploaded,
                 use_container_width=True)

    with col_res:
        with st.spinner("Classifying…"):
            prediction = classify_image(
                source if uploaded is None else uploaded.getvalue(), bundle
            )
        if not prediction.probabilities:
            st.error("Classification failed. " + " ".join(prediction.notes))
            return
        st.metric("Predicted stage", prediction.display_label,
                  f"{prediction.confidence * 100:.1f}% confidence")
        frame = pd.DataFrame(
            sorted(
                ({"Class": k, "Probability": v}
                 for k, v in prediction.probabilities.items()),
                key=lambda r: -r["Probability"],
            )
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)
        for n in prediction.notes:
            st.caption(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    st.markdown(ui.CSS, unsafe_allow_html=True)

    st.title("\U0001fa7a Auracare — guided voice screening")
    st.caption(
        "Four-domain bioacoustic check-in with PHQ-8. Screening prototype — "
        "not a diagnosis. Emergency dispatch and FHIR messaging are simulated."
    )

    settings = render_sidebar()

    tab_flow, tab_patient, tab_clinical, tab_image, tab_ref = st.tabs(
        [
            "Check-in",
            "Your results",
            "Clinical / EHR view",
            "MRI classifier",
            "Method & limits",
        ]
    )

    with tab_flow:
        step = int(st.session_state.step)
        if step <= 4:
            render_task_step(step, settings)
        elif step == 5:
            render_phq8_step(settings)
        else:
            render_completion(settings)

    with tab_patient:
        render_patient_view()
    with tab_clinical:
        render_clinical_view()
    with tab_image:
        render_imaging_view()
    with tab_ref:
        render_reference_view()


if __name__ == "__main__":
    main()
