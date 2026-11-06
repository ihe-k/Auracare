"""
Presentation helpers for the Streamlit app: CSS, animated task cues, spoken
prompts and the protocol script.

Kept out of `app.py` so the protocol copy - the words the agent actually says -
sits in one reviewable place rather than scattered through UI code.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .risk import (
    BORDERLINE,
    CRITICAL,
    ELEVATED,
    LEVEL_COLOUR,
    NORMAL,
    NOT_MEASURED,
)


# ---------------------------------------------------------------------------
# Protocol script
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskScript:
    """One step of the guided check-in."""

    task_id: int
    domain: str
    window: str
    record_seconds: int
    prompt: str
    instruction: str
    cue_class: str
    cue_label: str
    cue_hint: str
    colour: str
    safety_check: str


TASK_SCRIPTS: List[TaskScript] = [
    TaskScript(
        task_id=1,
        domain="Respiratory & laryngeal health",
        window="0:00 - 0:25",
        record_seconds=10,
        prompt=(
            "Hi there, I'm Auracare. Let's do your quick check-in. Follow the "
            "blue breathing guide on your screen: take a deep breath in, and "
            "hold the vowel sound 'Ah' for as long as you comfortably can."
        ),
        instruction="Inhale as the circle grows, then hold **\u201cAhhh\u201d**.",
        cue_class="circle-phonation",
        cue_label="Task 1 \u00b7 Sustained phonation /a:/",
        cue_hint="Hold the note steadily until you run out of breath.",
        colour="#38bdf8",
        safety_check=(
            "Sustained phonation time and vocal perturbation. Phonation below "
            "the critical threshold raises the respiratory intercept."
        ),
    ),
    TaskScript(
        task_id=2,
        domain="Expiratory airway dynamics",
        window="0:25 - 0:45",
        record_seconds=10,
        prompt=(
            "Thank you. Now follow the orange guide: take a deep breath in "
            "through your nose, and exhale as hard and fast as you can through "
            "your mouth, as if blowing out a candle."
        ),
        instruction="Deep breath in\u2026 then **blow out hard and fast**.",
        cue_class="circle-exhale",
        cue_label="Task 2 \u00b7 Forced exhalation",
        cue_hint="One sharp, complete breath out. Keep going until empty.",
        colour="#f97316",
        safety_check=(
            "Exhalation duration, rise time and the tonal-peak prominence used "
            "as a wheeze proxy."
        ),
    ),
    TaskScript(
        task_id=3,
        domain="Neuromotor speech control",
        window="0:45 - 1:05",
        record_seconds=10,
        prompt=(
            "Great. Now follow the purple rhythm guide and repeat the sounds "
            "'pa-ta-ka' as fast and steadily as you can on one breath."
        ),
        instruction="Match the rhythm: **\u201cpa - ta - ka - pa - ta - ka\u201d**",
        cue_class="circle-ddk",
        cue_label="Task 3 \u00b7 Diadochokinetic rate",
        cue_hint="As fast as you can while staying clear and even.",
        colour="#c084fc",
        safety_check=(
            "Syllable repetition rate and the variability of inter-syllable "
            "timing."
        ),
    ),
    TaskScript(
        task_id=4,
        domain="Cognitive & language fluency",
        window="1:05 - 1:45",
        record_seconds=30,
        prompt=(
            "Last step. Please look at the picture on your screen and describe "
            "in your own words everything you see happening."
        ),
        instruction="Describe **everything you can see happening** in the picture.",
        cue_class="",
        cue_label="Task 4 \u00b7 Picture description",
        cue_hint="Keep going for the full thirty seconds if you can.",
        colour="#34d399",
        safety_check=(
            "Speech pause ratio, speech rate, lexical diversity and the "
            "negative-language lexicon."
        ),
    ),
]

COMPLETION_PROMPT = (
    "Check-in complete. Your results have been added to your health record. "
    "Have a good day."
)

EMERGENCY_PROMPT = (
    "Based on your check-in, your breathing sounds strained. We are routing "
    "your assessment to emergency triage now. Please stay on the line. If you "
    "are in severe distress, call 999 straight away."
)

CRISIS_PROMPT = (
    "Thank you for telling me that. Some of what you said suggests you may be "
    "having thoughts of harming yourself. Please talk to someone today. "
    "Samaritans are free on 116 123, any time."
)


def script_for(task_id: int) -> TaskScript:
    for script in TASK_SCRIPTS:
        if script.task_id == task_id:
            return script
    raise KeyError(f"no script for task {task_id}")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
  .aura-agent {
      background: #0f172a; padding: 14px 18px; border-radius: 12px;
      margin-bottom: 10px; border-left: 4px solid #0284c7; color: #f8fafc;
      font-size: 0.97rem; line-height: 1.5;
  }
  .aura-user {
      background: #334155; padding: 12px 18px; border-radius: 12px;
      margin-bottom: 10px; border-right: 4px solid #10b981; color: #f8fafc;
      text-align: right; font-size: 0.93rem;
  }
  .aura-system {
      background: #1e293b; padding: 10px 16px; border-radius: 10px;
      margin-bottom: 10px; border-left: 4px solid #64748b; color: #cbd5e1;
      font-size: 0.86rem; font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .cue-box {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; padding: 26px 18px; background: #0f172a;
      border-radius: 16px; border: 1px solid #334155; text-align: center;
      min-height: 250px;
  }
  .cue-title { margin-top: 18px; margin-bottom: 4px; font-weight: 600; }
  .cue-hint { color: #94a3b8; font-size: 0.86rem; margin: 0; }

  .circle-phonation, .circle-exhale, .circle-ddk {
      width: 118px; height: 118px; border-radius: 50%;
  }
  .circle-phonation {
      background: radial-gradient(circle, #38bdf8 0%, #0284c7 100%);
      animation: pulse-chest 4s infinite ease-in-out;
  }
  .circle-exhale {
      background: radial-gradient(circle, #f97316 0%, #ea580c 100%);
      animation: burst-exhale 2s infinite ease-in-out;
  }
  .circle-ddk {
      background: radial-gradient(circle, #a855f7 0%, #9333ea 100%);
      animation: rhythm-ddk 0.2s infinite alternate ease-in-out;
  }
  @keyframes pulse-chest {
      0%   { transform: scale(0.85); box-shadow: 0 0 10px rgba(56,189,248,0.2); }
      50%  { transform: scale(1.25); box-shadow: 0 0 35px rgba(56,189,248,0.7); }
      100% { transform: scale(0.85); box-shadow: 0 0 10px rgba(56,189,248,0.2); }
  }
  @keyframes burst-exhale {
      0%   { transform: scale(1.3); opacity: 1; }
      70%  { transform: scale(0.65); opacity: 0.5; }
      100% { transform: scale(1.3); opacity: 1; }
  }
  @keyframes rhythm-ddk {
      0%   { transform: scale(0.9); }
      100% { transform: scale(1.12); }
  }
  @media (prefers-reduced-motion: reduce) {
      .circle-phonation, .circle-exhale, .circle-ddk { animation: none; }
  }

  .pill {
      display: inline-block; padding: 2px 10px; border-radius: 999px;
      font-size: 0.76rem; font-weight: 700; color: #0b1120;
      letter-spacing: 0.02em;
  }
  .sim-banner {
      background: #422006; border: 1px solid #a16207; color: #fde68a;
      padding: 10px 14px; border-radius: 10px; font-size: 0.85rem;
      margin-bottom: 12px;
  }
  .step-dot {
      display: inline-block; width: 11px; height: 11px; border-radius: 50%;
      margin-right: 7px; border: 2px solid #475569;
  }
</style>
"""


def level_pill(level: str) -> str:
    colour = LEVEL_COLOUR.get(level, "#64748b")
    return f'<span class="pill" style="background:{colour}">{html.escape(level)}</span>'


def agent_bubble(text: str) -> str:
    return (
        '<div class="aura-agent"><strong>Auracare</strong><br>'
        f"{html.escape(text)}</div>"
    )


def patient_bubble(text: str) -> str:
    return (
        '<div class="aura-user"><strong>You</strong><br>'
        f"{html.escape(text)}</div>"
    )


def system_bubble(text: str) -> str:
    return f'<div class="aura-system">{html.escape(text)}</div>'


def cue_html(script: TaskScript) -> str:
    """The animated visual pacing guide for one task."""
    if not script.cue_class:
        return ""
    return f"""
    <div class="cue-box">
      <div class="{script.cue_class}"></div>
      <div class="cue-title" style="color:{script.colour}">{script.cue_label}</div>
      <p class="cue-hint">{script.cue_hint}</p>
    </div>
    """


def speak_script(text: str, nonce: str, rate: float = 0.95) -> str:
    """
    Speak `text` with the browser's own speech synthesis.

    Real text-to-speech with no API key and no network call - the voice is the
    one already installed on the patient's device. `nonce` must change whenever
    a new utterance should be spoken, because Streamlit re-renders components
    on every interaction and the utterance would otherwise repeat.

    Browsers block speech until the page has had a user gesture, so the first
    prompt may stay silent until the patient clicks something. That is a
    browser policy, not a bug, and the text is always shown as well.
    """
    payload = json.dumps(text)
    return f"""
    <script>
      (function () {{
        const key = "auracare-spoken-{nonce}";
        try {{
          if (window.sessionStorage.getItem(key)) return;
          window.sessionStorage.setItem(key, "1");
        }} catch (e) {{ /* private mode: speak anyway */ }}
        const synth = window.speechSynthesis;
        if (!synth) return;
        synth.cancel();
        const u = new SpeechSynthesisUtterance({payload});
        u.rate = {rate};
        u.pitch = 1.0;
        u.lang = "en-GB";
        synth.speak(u);
      }})();
    </script>
    """


def stop_speech() -> str:
    return (
        "<script>if (window.speechSynthesis) window.speechSynthesis.cancel();"
        "</script>"
    )


def progress_dots(current: int, total: int, done: Dict[int, bool]) -> str:
    out = []
    for i in range(1, total + 1):
        if done.get(i):
            colour = "#10b981"
        elif i == current:
            colour = "#38bdf8"
        else:
            colour = "transparent"
        out.append(
            f'<span class="step-dot" style="background:{colour}"></span>'
        )
    return "".join(out)


def fmt_value(value: Optional[float], unit: str = "", nd: int = 2) -> str:
    """Render a measurement, or an explicit dash when it could not be taken."""
    if value is None:
        return "not measured"
    text = f"{value:.{nd}f}"
    return f"{text} {unit}".strip()


__all__ = [
    "COMPLETION_PROMPT",
    "CRISIS_PROMPT",
    "CSS",
    "EMERGENCY_PROMPT",
    "TASK_SCRIPTS",
    "TaskScript",
    "agent_bubble",
    "cue_html",
    "fmt_value",
    "level_pill",
    "patient_bubble",
    "progress_dots",
    "script_for",
    "speak_script",
    "stop_speech",
    "system_bubble",
]
