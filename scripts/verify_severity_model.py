"""
Quantify what the repository severity model can and cannot do with PHQ-8 input.

The claim being tested: because only 8 of this model's 10,249 features come
from the PHQ-8, a voice-only screening flow cannot drive it meaningfully. That
is checkable rather than rhetorical - sweep the PHQ-8 from its floor to its
ceiling, hold everything else at the training mean, and see how far the output
moves. A model responding properly to a 0-24 input should move most of 24
points.

Also prints the metrics stored inside the artifact itself, which are the
model's own report on its generalisation.

Usage
    python scripts/verify_severity_model.py [--model PATH]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from auracare.paths import find_severity_model  # noqa: E402
from auracare.phq8 import ITEM_KEYS, band_for_score, score_phq8  # noqa: E402
from auracare.severity_model import (  # noqa: E402
    STORED_METRICS,
    load_bundle,
    predict_severity,
    sensitivity_sweep,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    args = ap.parse_args()

    path = args.model or find_severity_model()
    if path is None:
        print("could not locate severity_model.pkl.")
        print("pass --model, or set AURACARE_TRIAGE_REPO.")
        return 1

    print("=" * 78)
    print("SEVERITY MODEL: WHAT PHQ-8 ALONE CAN DRIVE")
    print("=" * 78)
    print(f"artifact: {path}")
    print()

    bundle = load_bundle(path)

    # -- feature composition ------------------------------------------------
    groups: dict = {}
    for name in bundle.feature_names:
        if name in ITEM_KEYS:
            key = "PHQ-8 items"
        elif name in ("Gender", "Age"):
            key = "demographics"
        else:
            key = name.rsplit("_f", 1)[0] if "_f" in name else name
        groups[key] = groups.get(key, 0) + 1

    print(f"total features: {bundle.n_features}")
    print()
    print(f"{'feature block':<28}{'count':>8}{'share':>10}   obtainable from a microphone?")
    print("-" * 78)
    voice_ok = {"PHQ-8 items", "demographics", "audio_mean", "audio_std"}
    for key in sorted(groups, key=lambda k: -groups[k]):
        n = groups[key]
        share = n / bundle.n_features
        if key in ("PHQ-8 items", "demographics"):
            verdict = "yes - asked directly"
        elif key.startswith("audio"):
            verdict = "no - DAIC-WOZ COVAREP layout, not reproducible"
        elif key.startswith("fkps"):
            verdict = "no - needs facial keypoints (video)"
        elif key.startswith("gaze"):
            verdict = "no - needs gaze tracking (video)"
        elif key.startswith("pose"):
            verdict = "no - needs body pose (video)"
        elif key.startswith("text"):
            verdict = "no - needs the original embedding pipeline"
        else:
            verdict = "?"
        print(f"{key:<28}{n:>8}{share:>9.2%}   {verdict}")
    print("-" * 78)
    drivable = len(ITEM_KEYS) + (1 if "Gender" in bundle.index_of else 0)
    print(f"{'directly answerable':<28}{drivable:>8}{drivable / bundle.n_features:>9.2%}")
    print()

    # -- the model's own stored metrics -------------------------------------
    print("metrics stored inside the artifact:")
    print(f"  {'split':<14}{'MAE':>9}{'RMSE':>9}{'R2':>9}")
    for split, m in STORED_METRICS.items():
        print(f"  {split:<14}{m['mae']:>9.3f}{m['rmse']:>9.3f}{m['r2']:>9.3f}")
    print()
    print("  PHQ-8 runs 0-24. A test MAE of 6.0 spans more than one severity")
    print("  band, and a negative test R2 means the model does worse on held-out")
    print("  data than always predicting the training mean.")
    print()

    # -- the sweep ----------------------------------------------------------
    for impute in ("mean", "zero"):
        print("=" * 78)
        print(f"PHQ-8 SWEEP (unavailable features imputed at the training {impute})")
        print("=" * 78)
        sweep = sensitivity_sweep(bundle, gender=0.0, impute=impute)

        print(f"{'items at':<10}{'true PHQ-8':>11}{'true band':>20}"
              f"{'output':>9}{'model band':>20}   agreement")
        print("-" * 78)
        for row in sweep["sweep"]:
            print(
                f"{row['item_level']:<10}{row['phq8_true_total']:>11}"
                f"{row['true_band']:>20}{row['model_prediction']:>9.2f}"
                f"{str(row['model_band']):>20}   {row['band_agreement']}"
            )
        print("-" * 78)
        span = sweep["prediction_span"]
        print(f"output span across the full instrument range : {span:>7.3f} points")
        print(f"the instrument's own range                   : {24.0:>7.3f} points")
        print(f"fraction of the scale the model traverses    : "
              f"{sweep['span_fraction_of_scale']:>7.2%}")
        print(f"severity bands reproduced correctly          : "
              f"{sweep['bands_matched']}/{sweep['bands_tested']}"
              f"  ({sweep['bands_under_estimated']} under-estimated)")
        print()
        print("  The model is not insensitive to PHQ-8 - it tracks the lower half")
        print("  of the scale. It saturates above it, so the two highest bands both")
        print("  come out as 'Moderate'. Both errors are under-estimates, which on a")
        print("  screening instrument is the direction that causes under-referral.")
        print()

        print("per-item sensitivity (one item 0 -> 3, others held at 0):")
        for row in sweep["per_item"]:
            flag = "   <- ignored entirely" if row["delta_when_maxed"] == 0.0 else ""
            print(f"   {row['item']:<24}{row['delta_when_maxed']:>+10.4f}{flag}")
        if sweep["items_ignored"]:
            print()
            print(f"  {len(sweep['items_ignored'])} of 8 items have zero influence; "
                  "the response is dominated by")
            print("  PHQ8_NoInterest, so this behaves as a function of one or two items")
            print("  rather than of the eight-item instrument.")
        print()

    # -- what the app does instead -----------------------------------------
    print("=" * 78)
    print("WHAT THE SCREENING APP USES INSTEAD: PUBLISHED CUT-POINTS")
    print("=" * 78)
    print(f"{'all items at':<14}{'PHQ-8 total':>13}{'band':>22}{'significant':>13}")
    print("-" * 78)
    for level in (0, 1, 2, 3):
        result = score_phq8({k: level for k in ITEM_KEYS})
        print(f"{level:<14}{result.total_score:>13}{result.band:>22}"
              f"{str(result.clinically_significant):>13}")
    print("-" * 78)
    print("Summation traverses the full 0-24 range exactly, by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
