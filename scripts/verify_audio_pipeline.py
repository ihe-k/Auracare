"""
End-to-end verification of the voice screening pipeline.

Three parts:

  1. The client's real recordings. `Practice_Files/300_AUDIO.wav` and
     `385_AUDIO.wav` are full DAIC-WOZ interview sessions (10.8 and 8.9
     minutes), so two caveats apply and are printed with the results: they
     contain the interviewer as well as the participant, and they are
     connected speech rather than the four short task manoeuvres. They are
     therefore valid input for the Task 4 analysis and not for Tasks 1-3.

  2. A routine screening, end to end, using synthesised task recordings whose
     true values are known, plus a real 30-second excerpt for Task 4.

  3. An emergency screening, end to end, demonstrating that the respiratory
     intercept fires and produces a stat dispatch.

Usage
    python scripts/verify_audio_pipeline.py [--no-stt] [--model tiny.en]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
warnings.filterwarnings("ignore")

from auracare.audio_io import check_quality, load_audio  # noqa: E402
from auracare.config import Thresholds  # noqa: E402
from auracare.features.ddk import analyse_ddk  # noqa: E402
from auracare.features.expiratory import analyse_expiratory  # noqa: E402
from auracare.features.fluency import analyse_fluency  # noqa: E402
from auracare.features.phonation import analyse_phonation  # noqa: E402
from auracare.fhir import build_bundle, summarise_bundle  # noqa: E402
from auracare.paths import find_practice_files  # noqa: E402
from auracare.phq8 import ITEM_KEYS, score_phq8  # noqa: E402
from auracare.risk import PATHWAY_LABEL, assess_screening  # noqa: E402
from auracare.stt import Transcript, analyse_language, transcribe  # noqa: E402

PATIENT = "NHS-948-204-11"


def hr(title: str, char: str = "=") -> None:
    print()
    print(char * 78)
    print(title)
    print(char * 78)


def fmt(value, nd=3, dash="  --  "):
    if value is None:
        return dash
    return f"{value:.{nd}f}"


# ---------------------------------------------------------------------------
# Part 1: the real recordings
# ---------------------------------------------------------------------------
def characterise_real_recordings(use_stt: bool, model_size: str) -> None:
    directory = find_practice_files()
    if directory is None:
        print("Practice_Files not found; skipping real-recording checks.")
        return

    wavs = sorted(directory.glob("*.wav"))
    if not wavs:
        print(f"no .wav files in {directory}")
        return

    hr("PART 1  THE CLIENT'S REAL RECORDINGS")
    print("These are full DAIC-WOZ interview sessions. Two things follow from")
    print("that, and both are limitations of the input, not of the analysis:")
    print("  * they contain the interviewer's speech as well as the participant's,")
    print("    so pause and rate measures describe the conversation, not one speaker;")
    print("  * they are connected speech, so they exercise Task 4 only. There is no")
    print("    sustained vowel, forced exhalation or /pa-ta-ka/ in them.")

    for path in wavs:
        audio = load_audio(path)
        quality = check_quality(audio)

        print()
        print("-" * 78)
        print(f"{path.name}")
        print("-" * 78)
        print(f"  duration        {audio.duration_s:>10.1f} s "
              f"({audio.duration_s / 60:.1f} min)")
        print(f"  sample rate     {audio.sample_rate:>10} Hz")
        print(f"  peak amplitude  {audio.peak:>10.4f}")
        print(f"  level           {audio.rms_dbfs:>10.1f} dBFS")
        print(f"  usable          {str(quality.usable):>10}  {quality.problems}")

        # Task 4 over a 30 s excerpt, matching the protocol window.
        start = 60.0
        excerpt = audio.slice_seconds(start, start + 30.0)

        transcript = Transcript()
        if use_stt:
            t0 = time.time()
            transcript = transcribe(excerpt, model_size=model_size)
            elapsed = time.time() - t0
            print(f"  transcription   {elapsed:>10.1f} s for a 30 s excerpt "
                  f"({transcript.backend})")

        result = analyse_fluency(excerpt, transcript=transcript)
        print(f"  --- Task 4 analysis of the {start:.0f}-{start + 30:.0f} s excerpt ---")
        print(f"    speech detected      {str(result.has_speech):>10}")
        print(f"    pause ratio          {fmt(result.pause_ratio, 4):>10}")
        print(f"    pauses               {result.n_pauses:>10}")
        print(f"    longest pause        {fmt(result.longest_pause_s):>10} s")
        print(f"    speech time          {fmt(result.speech_time_s):>10} s")
        print(f"    syllables            {result.n_syllables:>10}")
        print(f"    speech rate          {fmt(result.speech_rate_sylls):>10} syll/s")
        print(f"    articulation rate    {fmt(result.articulation_rate_sylls):>10} syll/s")
        v = result.voice
        print(f"    mean F0              {fmt(v.mean_f0_hz, 1):>10} Hz")
        print(f"    jitter (local)       {fmt(v.jitter_local_pct):>10} %")
        print(f"    shimmer (local)      {fmt(v.shimmer_local_pct):>10} %")
        print(f"    HNR                  {fmt(v.hnr_db, 1):>10} dB")
        print(f"    voiced fraction      {fmt(v.voiced_fraction, 3):>10}")

        if transcript.available:
            print(f"    words                {result.n_words:>10}")
            print(f"    words per minute     {fmt(result.words_per_minute, 1):>10}")
            print(f"    lexical diversity    {fmt(result.lexical_diversity, 3):>10}")
            flags = result.language
            print(f"    negative share       {fmt(flags.negative_ratio, 4):>10}")
            print(f"    first-person share   "
                  f"{fmt(flags.first_person_singular_ratio, 4):>10}")
            print(f"    safety critical      {str(flags.safety_critical):>10}")
            if flags.matched_terms:
                print(f"    lexicon hits         {flags.matched_terms}")
            print(f"    transcript: {transcript.text[:220]}...")
        for note in result.notes:
            print(f"    note: {note}")


# ---------------------------------------------------------------------------
# Parts 2 and 3: end-to-end screenings
# ---------------------------------------------------------------------------
def _print_assessment(assessment, label: str) -> None:
    print()
    print(f"  overall level    : {assessment.overall_level}")
    print(f"  tasks measured   : {assessment.n_tasks_measured}/4")
    print(f"  primary pathway  : {PATHWAY_LABEL[assessment.primary_pathway]}")
    print(f"  all pathways     : {assessment.pathways}")
    print(f"  emergency        : {assessment.is_emergency}")
    print(f"  safety critical  : {assessment.safety_critical}")
    if assessment.phq8 and assessment.phq8.scorable:
        print(f"  PHQ-8            : {assessment.phq8.total_score}/24 "
              f"({assessment.phq8.band})")
    print()
    print("  per-task indicators:")
    for task in assessment.tasks:
        print(f"    Task {task.task_id}  {task.title:<34} {task.level}")
        for ind in task.indicators:
            print(f"        {ind.label:<44}{fmt(ind.value, 3):>10} "
                  f"{ind.unit:<8} {ind.level}")
    print()
    print("  reasons:")
    for reason in assessment.reasons or ["(none)"]:
        print(f"    - {reason}")
    print()
    print("  PATIENT VIEW:")
    for msg in assessment.patient_messages:
        print(f"    * {msg}")
    print()
    print("  CLINICIAN VIEW:")
    for action in assessment.clinician_actions:
        print(f"    * {action}")

    bundle = build_bundle(assessment, PATIENT)
    print()
    print(f"  FHIR bundle      : {bundle['total']} resources "
          f"{summarise_bundle(bundle)}")
    comm = [
        e["resource"] for e in bundle["entry"]
        if e["resource"]["resourceType"] == "CommunicationRequest"
    ][0]
    print(f"  dispatch priority: {comm['priority']}")
    print(f"  dispatch to      : {comm['recipient'][0]['reference']}")
    print(f"  dispatch payload : {comm['payload'][0]['contentString'][:160]}")


def routine_screening(use_stt: bool, model_size: str) -> None:
    from conftest import (
        am_syllable_train,
        as_audio,
        concat,
        noise_burst,
        silence,
        vowel,
    )

    hr("PART 2  ROUTINE SCREENING, END TO END")
    print("Tasks 1-3 are synthesised with known true values so the reported")
    print("numbers can be checked. Task 4 uses a real 30 s excerpt.")
    print()
    print("  Task 1 built as a 9.0 s sustained vowel at 120 Hz")
    print("  Task 2 built as a 1.5 s forced-exhalation burst")
    print("  Task 3 built as /pa-ta-ka/ at exactly 6.0 syllables/s for 4.0 s")

    t1 = analyse_phonation(as_audio(concat(silence(0.4), vowel(120.0, 9.0), silence(0.4))))
    t2 = analyse_expiratory(as_audio(concat(silence(0.4), noise_burst(1.5), silence(0.4))))
    t3 = analyse_ddk(as_audio(am_syllable_train(6.0, 4.0)))

    print()
    print(f"  Task 1 measured phonation : {fmt(t1.phonation_duration_s)} s "
          f"(true 9.000)")
    print(f"  Task 2 measured exhalation: {fmt(t2.exhalation_duration_s)} s")
    print(f"  Task 3 measured rate      : {fmt(t3.syllable_rate_hz)} syll/s "
          f"(true 6.000)")

    directory = find_practice_files()
    transcript = Transcript()
    if directory and sorted(directory.glob("*.wav")):
        audio = load_audio(sorted(directory.glob("*.wav"))[0])
        excerpt = audio.slice_seconds(120.0, 150.0)
        if use_stt:
            transcript = transcribe(excerpt, model_size=model_size)
        t4 = analyse_fluency(excerpt, transcript=transcript)
    else:
        from conftest import speech_with_pauses

        t4 = analyse_fluency(
            as_audio(speech_with_pauses([8.0, 8.0], [2.0], lead_s=0.5, trail_s=0.5))
        )

    responses = {k: 0 for k in ITEM_KEYS}
    responses["PHQ8_Sleep"] = 1
    responses["PHQ8_Tired"] = 2
    phq8 = score_phq8(responses)

    assessment = assess_screening(t1, t2, t3, t4, phq8, Thresholds())
    _print_assessment(assessment, "routine")


def emergency_screening(use_stt: bool, model_size: str) -> None:
    from conftest import (
        am_syllable_train,
        as_audio,
        concat,
        noise_burst,
        silence,
        speech_with_pauses,
        vowel,
    )

    hr("PART 3  EMERGENCY SCREENING, END TO END")
    print("Task 1 is built as a 1.8 s sustained vowel, below the 2.5 s")
    print("respiratory safety threshold. Task 4 carries a transcript containing")
    print("self-harm language, so the crisis pathway should open alongside.")

    t1 = analyse_phonation(as_audio(concat(silence(0.3), vowel(135.0, 1.8), silence(0.3))))
    t2 = analyse_expiratory(
        as_audio(concat(silence(0.3), noise_burst(0.5, tone_hz=750.0, tone_level=0.6),
                        silence(0.3)))
    )
    t3 = analyse_ddk(as_audio(am_syllable_train(3.0, 4.0)))
    t4 = analyse_fluency(
        as_audio(speech_with_pauses([1.5, 1.5], [3.5], lead_s=0.5, trail_s=0.5)),
        transcript=Transcript(
            text=(
                "I do not know what to say about the picture. I feel hopeless "
                "most days and honestly sometimes I think I would be better "
                "off dead. Nothing matters any more."
            ),
            available=True,
            backend="manual",
        ),
    )

    responses = {k: 3 for k in ITEM_KEYS}
    assessment = assess_screening(t1, t2, t3, t4, score_phq8(responses), Thresholds())
    _print_assessment(assessment, "emergency")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stt", action="store_true", help="skip transcription")
    ap.add_argument("--model", default="base.en", help="faster-whisper model size")
    args = ap.parse_args()
    use_stt = not args.no_stt

    characterise_real_recordings(use_stt, args.model)
    routine_screening(use_stt, args.model)
    emergency_screening(use_stt, args.model)

    hr("DONE")
    print("No network request was made to any clinical system. Every dispatch")
    print("above is a JSON document built in memory and tagged HTEST.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
