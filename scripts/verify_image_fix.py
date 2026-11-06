"""
Prove the MRI image-classification fix, rather than assert it.

Runs both feature pipelines - the current repository one and the corrected one -
through the *same* shipped model and scaler, and reports:

  1. Standardised feature magnitudes, |z| = |(x - scaler.mean_) / scaler.scale_|.
     This is the decisive measurement. The model was trained on standardised
     features whose |z| is O(1) by construction. A pipeline whose features
     standardise to O(1000) is feeding the forest inputs it has never seen, and
     no amount of threshold tuning downstream can repair that.

  2. The predicted class and full probability distribution for each image.

  3. How many distinct classes each pipeline can produce across the inputs.

Usage
    python scripts/verify_image_fix.py [--model PATH] [--images DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auracare.imaging.alzheimer import (  # noqa: E402
    EXPECTED_N_FEATURES,
    classify_image,
    extract_features_legacy_buggy,
    extract_training_features,
    load_image_model,
    preprocess_legacy_app,
    preprocess_like_training,
)
from auracare.paths import find_image_model, find_practice_files  # noqa: E402

# The practice filenames encode their ground truth.
TRUTH = {
    "Mild_dementia.jpg": "Mild Dementia",
    "Non-demented.jpg": "Non Demented",
    "Moderate_dementia.jpg": "Moderate Dementia",
    "Very_mild_dementia.jpg": "Very mild Dementia",
}


def z_stats(features: np.ndarray, scaler) -> dict:
    """Standardised-magnitude summary for one feature vector."""
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    n = min(features.size, mean.size)
    z = (features[:n].astype(np.float64) - mean[:n]) / np.where(
        scale[:n] == 0, 1.0, scale[:n]
    )
    az = np.abs(z)
    return {
        "median_abs_z": float(np.median(az)),
        "p95_abs_z": float(np.percentile(az, 95)),
        "max_abs_z": float(np.max(az)),
        "frac_beyond_6_sigma": float(np.mean(az > 6.0)),
    }


def predict_with(features: np.ndarray, bundle) -> tuple:
    """Run a raw feature vector through the scaler and classifier."""
    n = bundle.n_features
    f = features[:n] if features.size >= n else np.pad(features, (0, n - features.size))
    scaled = bundle.scaler.transform(f.reshape(1, -1).astype(np.float32))
    probs = np.asarray(bundle.classifier.predict_proba(scaled)[0], dtype=np.float64)
    names = [str(c) for c in bundle.label_encoder.inverse_transform(
        bundle.classifier.classes_
    )]
    best = int(np.argmax(probs))
    return names[best], float(probs[best]), dict(zip(names, probs.tolist()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--images", type=Path, default=None)
    args = ap.parse_args()

    model_path = args.model or find_image_model()
    images_dir = args.images or find_practice_files()

    if model_path is None:
        print("could not locate the Alzheimer classifier artifact.")
        print("pass --model, or set AURACARE_TRIAGE_REPO to the "
              "Cognitive_AI_Triage checkout.")
        return 1
    if images_dir is None:
        print("could not locate Practice_Files. pass --images.")
        return 1

    print("=" * 78)
    print("MRI CLASSIFICATION: BUGGY vs CORRECTED FEATURE PIPELINE")
    print("=" * 78)
    print(f"model : {model_path.resolve()}")
    print(f"images: {images_dir.resolve()}")
    print()

    bundle = load_image_model(model_path)
    print(f"classes in artifact      : {bundle.classes}")
    print(f"model expects n_features : {bundle.n_features}")
    print(f"corrected pipeline yields: {EXPECTED_N_FEATURES}")
    print(f"stored test accuracy     : {bundle.stored_test_accuracy}")
    print()

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        print(f"no images found in {images_dir}")
        return 1

    legacy_labels, fixed_labels = [], []
    rows = []

    for path in image_paths:
        img = preprocess_like_training(path)
        fixed_feats = extract_training_features(img)
        # Reproduce the shipped app end to end: its preprocessing AND its
        # feature step, so the symptom matches what the client observed.
        legacy_img = preprocess_legacy_app(path)
        legacy_feats = extract_features_legacy_buggy(legacy_img, bundle.n_features)

        z_fixed = z_stats(fixed_feats, bundle.scaler)
        z_legacy = z_stats(legacy_feats, bundle.scaler)

        l_label, l_conf, l_probs = predict_with(legacy_feats, bundle)
        f_label, f_conf, f_probs = predict_with(fixed_feats, bundle)
        legacy_labels.append(l_label)
        fixed_labels.append(f_label)

        truth = TRUTH.get(path.name, "unknown")
        rows.append((path.name, truth, l_label, l_conf, f_label, f_conf))

        print("-" * 78)
        print(f"IMAGE: {path.name}    ground truth: {truth}")
        print("-" * 78)
        print("  standardised feature magnitudes |z| = |(x - mean_)/scale_|")
        print(
            f"    buggy    : median {z_legacy['median_abs_z']:>12.2f}   "
            f"p95 {z_legacy['p95_abs_z']:>12.2f}   max {z_legacy['max_abs_z']:>14.2f}   "
            f"beyond 6 sigma {z_legacy['frac_beyond_6_sigma'] * 100:6.2f} %"
        )
        print(
            f"    corrected: median {z_fixed['median_abs_z']:>12.2f}   "
            f"p95 {z_fixed['p95_abs_z']:>12.2f}   max {z_fixed['max_abs_z']:>14.2f}   "
            f"beyond 6 sigma {z_fixed['frac_beyond_6_sigma'] * 100:6.2f} %"
        )
        print()
        print(f"  buggy pipeline    -> {l_label:<22} p={l_conf:.4f}")
        for k, v in sorted(l_probs.items(), key=lambda kv: -kv[1]):
            print(f"        {k:<24} {v:.4f}")
        print(f"  corrected pipeline-> {f_label:<22} p={f_conf:.4f}")
        for k, v in sorted(f_probs.items(), key=lambda kv: -kv[1]):
            print(f"        {k:<24} {v:.4f}")
        print()

        api = classify_image(path, bundle, drop_non_diagnostic=True)
        print(f"  public API (PET class dropped) -> {api.display_label} "
              f"p={api.confidence:.4f}")
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'image':<26}{'truth':<22}{'buggy':<22}{'corrected':<22}")
    for name, truth, l_label, l_conf, f_label, f_conf in rows:
        print(f"{name:<26}{truth:<22}{l_label:<22}{f_label:<22}")
    print()
    print(f"distinct classes predicted, buggy    : "
          f"{sorted(set(legacy_labels))}")
    print(f"distinct classes predicted, corrected: "
          f"{sorted(set(fixed_labels))}")
    print()

    n_correct_legacy = sum(
        1 for n, t, l, _, _, _ in rows if t != "unknown" and l == t
    )
    n_correct_fixed = sum(
        1 for n, t, _, _, f, _ in rows if t != "unknown" and f == t
    )
    n_known = sum(1 for _, t, *_ in rows if t != "unknown")
    print(f"correct against filename ground truth: "
          f"buggy {n_correct_legacy}/{n_known}, corrected {n_correct_fixed}/{n_known}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
