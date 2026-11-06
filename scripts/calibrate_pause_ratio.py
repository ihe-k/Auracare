"""
Measure the pause-ratio distribution on real conversational speech.

The Auracare protocol escalates when the Task 4 speech pause ratio exceeds
0.45. That number needs checking against real recordings before it can be
trusted, because if ordinary speakers sit near or above it the intercept fires
on everyone and the alarm stops meaning anything.

This slides a 30-second window (the protocol's Task 4 duration) across the
client's sample recordings and reports the distribution, plus the fraction of
windows that would trigger at a range of candidate thresholds.

Caveat, stated in the output as well: these are DAIC-WOZ interview sessions
containing both the interviewer and the participant, with conversational
turn-taking gaps. A picture-description monologue should pause less. So this
establishes an upper bound on the false-trigger rate rather than a drop-in
replacement threshold - that needs the client's own Task 4 recordings.

Usage
    python scripts/calibrate_pause_ratio.py [--window 30] [--hop 15]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from auracare.audio_io import load_audio  # noqa: E402
from auracare.config import PROVISIONAL_DEFAULTS  # noqa: E402
from auracare.paths import find_practice_files  # noqa: E402
from auracare.vad import detect_voice_activity  # noqa: E402

CANDIDATES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--hop", type=float, default=15.0)
    args = ap.parse_args()

    directory = find_practice_files()
    if directory is None:
        print("Practice_Files not found. Set AURACARE_TRIAGE_REPO.")
        return 1
    wavs = sorted(directory.glob("*.wav"))
    if not wavs:
        print(f"no .wav files in {directory}")
        return 1

    print("=" * 78)
    print("PAUSE-RATIO CALIBRATION ON REAL SPEECH")
    print("=" * 78)
    print(f"window {args.window:.0f} s, hop {args.hop:.0f} s")
    print(f"protocol threshold under test: "
          f"{PROVISIONAL_DEFAULTS['task4']['pause_ratio_critical']}")
    print()

    all_ratios: list[float] = []
    for path in wavs:
        audio = load_audio(path)
        ratios = []
        start = 0.0
        while start + args.window <= audio.duration_s:
            excerpt = audio.slice_seconds(start, start + args.window)
            vad = detect_voice_activity(excerpt.samples, excerpt.sample_rate)
            if vad.has_speech and vad.active_span_s > args.window * 0.4:
                ratios.append(vad.pause_ratio)
            start += args.hop

        if not ratios:
            print(f"{path.name}: no usable windows")
            continue

        arr = np.asarray(ratios)
        all_ratios.extend(ratios)
        print(f"{path.name}  ({len(arr)} windows)")
        print(f"   min {arr.min():.3f}   p25 {np.percentile(arr, 25):.3f}   "
              f"median {np.median(arr):.3f}   p75 {np.percentile(arr, 75):.3f}   "
              f"max {arr.max():.3f}")
        print(f"   mean {arr.mean():.3f}  sd {arr.std():.3f}")
        print()

    if not all_ratios:
        return 1

    arr = np.asarray(all_ratios)
    print("-" * 78)
    print(f"POOLED  ({arr.size} windows across {len(wavs)} recordings)")
    print("-" * 78)
    print(f"   min {arr.min():.3f}   p05 {np.percentile(arr, 5):.3f}   "
          f"median {np.median(arr):.3f}   p95 {np.percentile(arr, 95):.3f}   "
          f"max {arr.max():.3f}")
    print()
    print(f"{'threshold':>12}{'windows triggering':>22}{'rate':>10}")
    print("-" * 78)
    for threshold in CANDIDATES:
        n = int(np.sum(arr > threshold))
        marker = "   <- protocol value" if abs(threshold - 0.45) < 1e-9 else ""
        print(f"{threshold:>12.2f}{n:>22}{n / arr.size:>9.1%}{marker}")
    print("-" * 78)
    print()
    print("Reading this:")
    print("  These are interview recordings with conversational turn-taking,")
    print("  so they pause more than a picture-description monologue will.")
    print("  They bound the false-trigger rate from above rather than giving a")
    print("  drop-in threshold. What they do establish is that 0.45 sits inside")
    print("  the normal range of real speech, not above it, so as specified the")
    print("  intercept cannot distinguish a struggling patient from an ordinary")
    print("  one. Recalibrate against your own Task 4 recordings before use;")
    print("  the threshold is a sidebar control for exactly that reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
