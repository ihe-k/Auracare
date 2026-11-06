"""
MRI feature-pipeline correctness.

The headline assertion is quantitative: the corrected pipeline produces
features that standardise to O(1) under the model's own scaler, while the
shipped pipeline produces features that standardise to O(10,000). That is the
difference between feeding a Random Forest inputs from its training
distribution and feeding it noise.

Tests that need the model artifact skip cleanly when it is not present, so the
suite still runs on a machine without the 23-40 MB pickles.
"""

from __future__ import annotations

import numpy as np
import pytest

from auracare.imaging.alzheimer import (
    EXPECTED_DIAGNOSTIC_STAGES,
    EXPECTED_N_FEATURES,
    NON_DIAGNOSTIC_CLASSES,
    TARGET_SIZE,
    classify_image,
    extract_features_legacy_buggy,
    extract_training_features,
    load_image_model,
    preprocess_legacy_app,
    preprocess_like_training,
)
from auracare.paths import find_image_model, find_practice_files


@pytest.fixture(scope="module")
def bundle():
    path = find_image_model()
    if path is None:
        pytest.skip("Alzheimer classifier artifact not available")
    return load_image_model(path)


@pytest.fixture(scope="module")
def practice_images():
    directory = find_practice_files()
    if directory is None:
        pytest.skip("Practice_Files not available")
    images = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        pytest.skip("no practice images found")
    return images


def _synthetic_mri(seed: int = 0) -> np.ndarray:
    """A brain-like grayscale image: dark border, bright centre, texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:256, 0:256]
    cy, cx = 128.0, 128.0
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    base = np.clip(1.0 - (r / 110.0) ** 2, 0.0, 1.0)
    texture = rng.normal(0.0, 0.05, size=base.shape)
    img = np.clip(base * 0.8 + texture * base, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Feature geometry
# ---------------------------------------------------------------------------
def test_expected_feature_count_is_pixels_plus_stats_plus_histogram():
    assert EXPECTED_N_FEATURES == 224 * 224 + 4 + 256 == 50_436


def test_preprocessing_shape_and_range():
    img = preprocess_like_training(_synthetic_mri())
    assert img.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 1)
    assert img.dtype == np.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


def test_corrected_features_have_the_training_length():
    features = extract_training_features(preprocess_like_training(_synthetic_mri()))
    assert features.size == EXPECTED_N_FEATURES


def test_corrected_features_are_the_flattened_image_then_stats_then_histogram():
    """Layout must match train_model.py exactly, not merely in length."""
    img = preprocess_like_training(_synthetic_mri())
    features = extract_training_features(img)

    n_px = 224 * 224
    assert np.allclose(features[:n_px], img.flatten(), atol=1e-6)
    assert features[n_px] == pytest.approx(float(np.mean(img)), abs=1e-5)
    assert features[n_px + 1] == pytest.approx(float(np.std(img)), abs=1e-5)
    assert features[n_px + 2] == pytest.approx(float(np.min(img)), abs=1e-5)
    assert features[n_px + 3] == pytest.approx(float(np.max(img)), abs=1e-5)
    histogram = features[n_px + 4 :]
    assert histogram.size == 256
    assert float(histogram.sum()) == pytest.approx(1.0, abs=1e-4)


def test_legacy_pipeline_is_44_distinct_values_tiled():
    """Confirms the nature of the bug: only 44 unique numbers, repeated."""
    img = preprocess_like_training(_synthetic_mri())
    legacy = extract_features_legacy_buggy(img, EXPECTED_N_FEATURES)

    assert legacy.size == EXPECTED_N_FEATURES
    assert np.unique(legacy).size <= 44
    # The vector is periodic with period 44.
    assert np.allclose(legacy[:44], legacy[44:88], atol=1e-6)


def test_legacy_and_corrected_features_differ_fundamentally():
    """
    The corrected vector carries per-pixel information and is not periodic;
    the legacy one is 44 values on repeat. Distinct-value counts are bounded
    by the 8-bit source depth, so the periodicity check is the sharper test.
    """
    img = preprocess_like_training(_synthetic_mri())
    corrected = extract_training_features(img)
    legacy = extract_features_legacy_buggy(img, EXPECTED_N_FEATURES)

    assert np.unique(corrected).size > 200
    assert np.unique(corrected).size > np.unique(legacy).size * 4
    assert not np.allclose(corrected, legacy)

    # Sample from the middle of the image, not the corner: the first rows of an
    # MRI are uniformly black, so a corner slice is legitimately constant.
    # (That zero variance is exactly why the bug is so destructive - those
    # features have a near-zero scaler scale_, so substituting mean(image)
    # there produces an enormous z-score.)
    mid = 224 * 112
    assert not np.allclose(
        corrected[mid : mid + 44], corrected[mid + 44 : mid + 88], atol=1e-6
    )
    assert np.std(corrected[mid : mid + 224]) > 0.01


# ---------------------------------------------------------------------------
# The decisive measurement
# ---------------------------------------------------------------------------
def _abs_z(features: np.ndarray, scaler) -> np.ndarray:
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    return np.abs((features.astype(np.float64) - mean) / scale)


def test_corrected_features_standardise_to_order_one(bundle, practice_images):
    for path in practice_images:
        features = extract_training_features(preprocess_like_training(path))
        az = _abs_z(features, bundle.scaler)
        assert np.median(az) < 2.0
        assert np.max(az) < 50.0
        assert float(np.mean(az > 6.0)) < 0.01


def test_legacy_features_standardise_far_outside_training_range(
    bundle, practice_images
):
    """
    The proof that the bug is a feature-space error and not a tuning problem:
    the shipped pipeline puts a meaningful share of inputs thousands of
    standard deviations from anything the forest was trained on.
    """
    for path in practice_images:
        features = extract_features_legacy_buggy(
            preprocess_legacy_app(path), bundle.n_features
        )
        az = _abs_z(features, bundle.scaler)
        assert np.max(az) > 1_000.0
        assert float(np.mean(az > 6.0)) > 0.02


def test_corrected_pipeline_is_orders_of_magnitude_closer_to_training(
    bundle, practice_images
):
    path = practice_images[0]
    legacy = _abs_z(
        extract_features_legacy_buggy(preprocess_legacy_app(path), bundle.n_features),
        bundle.scaler,
    )
    corrected = _abs_z(
        extract_training_features(preprocess_like_training(path)), bundle.scaler
    )
    assert np.max(legacy) / max(np.max(corrected), 1e-9) > 1_000.0


# ---------------------------------------------------------------------------
# Classification behaviour
# ---------------------------------------------------------------------------
def test_practice_images_classified_correctly(bundle, practice_images):
    """
    The practice filenames encode ground truth. The corrected pipeline must get
    them right; the shipped one does not.
    """
    truth = {
        "Mild_dementia.jpg": "Mild Dementia",
        "Non-demented.jpg": "Non Demented",
        "Moderate_dementia.jpg": "Moderate Dementia",
        "Very_mild_dementia.jpg": "Very mild Dementia",
    }
    checked = 0
    for path in practice_images:
        expected = truth.get(path.name)
        if expected is None or expected not in bundle.classes:
            continue
        prediction = classify_image(path, bundle)
        assert prediction.label == expected, (
            f"{path.name}: expected {expected}, got {prediction.label}"
        )
        assert prediction.confidence > 0.5
        checked += 1
    assert checked >= 1


def test_probabilities_are_normalised(bundle, practice_images):
    prediction = classify_image(practice_images[0], bundle)
    assert prediction.probabilities
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_non_diagnostic_class_is_dropped_when_present(bundle, practice_images):
    prediction = classify_image(practice_images[0], bundle, drop_non_diagnostic=True)
    for name in NON_DIAGNOSTIC_CLASSES:
        assert name not in prediction.probabilities
    if any(c in NON_DIAGNOSTIC_CLASSES for c in bundle.classes):
        assert prediction.dropped_classes
        assert any("PET" in n for n in prediction.notes)


def test_keeping_non_diagnostic_classes_is_opt_in(bundle, practice_images):
    kept = classify_image(practice_images[0], bundle, drop_non_diagnostic=False)
    assert set(kept.probabilities) == set(bundle.classes)


def test_model_reports_which_stages_it_cannot_predict(bundle):
    """
    The deployed artifact has only three classes and cannot output "Very mild
    Dementia" at all. The UI must be able to say so rather than silently never
    showing that label.
    """
    missing = bundle.missing_stages
    assert isinstance(missing, list)
    for stage in missing:
        assert stage in EXPECTED_DIAGNOSTIC_STAGES
        assert stage not in bundle.classes
    if missing:
        assert "cannot predict" in (bundle.coverage_note() or "")
    else:
        assert bundle.coverage_note() is None


def test_prediction_carries_the_accuracy_caveat(bundle, practice_images):
    prediction = classify_image(practice_images[0], bundle)
    assert "not trustworthy" in prediction.caveat


def test_classify_accepts_raw_bytes(bundle, practice_images):
    raw = practice_images[0].read_bytes()
    from_path = classify_image(practice_images[0], bundle)
    from_bytes = classify_image(raw, bundle)
    assert from_bytes.label == from_path.label
    assert from_bytes.confidence == pytest.approx(from_path.confidence, abs=1e-9)


def test_classification_is_deterministic(bundle, practice_images):
    a = classify_image(practice_images[0], bundle)
    b = classify_image(practice_images[0], bundle)
    assert a.label == b.label
    assert a.confidence == pytest.approx(b.confidence, abs=1e-12)
