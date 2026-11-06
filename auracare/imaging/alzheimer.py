"""
MRI dementia-stage classification - corrected inference pipeline.

The bug
    `streamlit_inference.py` in the Cognitive_AI_Triage repository builds a
    44-element feature vector (4 summary statistics, a 32-bin histogram, 8
    quadrant statistics) and then `np.tile`s it ~1,147 times to reach the
    50,436 inputs the model expects. The model was trained on something
    completely different: 50,176 raw flattened pixels, then 4 statistics, then
    a 256-bin histogram.

    So `pixel_0`, which during training received the top-left corner pixel of
    the MRI (near-black, near-zero variance across the training set), receives
    `mean(image)` at inference. After StandardScaler divides by that feature's
    tiny training standard deviation, the result is an enormous z-score. Every
    one of the 50,436 inputs is wrong in this way, the Random Forest is pushed
    far outside its training distribution, and its output collapses onto one or
    two leaves - which is why only "Mild Dementia" (class index 0) and
    "Non Demented" (class index 2) are ever predicted.

The fix
    `extract_training_features` below reproduces the training-time extraction
    exactly: same resize, same scaling order, same grayscale conversion, same
    flatten, same 256-bin histogram with the same normalisation. `verify.py`
    quantifies the difference by comparing standardised feature magnitudes.

Two further issues found in the shipped artifact, handled here
    1.  It has five classes, and one of them is "PET" - an imaging modality,
        not a diagnosis. `drop_non_diagnostic` removes it and renormalises the
        remaining probabilities.
    2.  Its reported 99.3 % test accuracy is not trustworthy: 13,445 of 17,295
        test samples are a single class, and the split is random over
        individual OASIS slices, so neighbouring slices of the same brain
        appear in both train and test. `MODEL_CAVEAT` carries this to the UI.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np

TARGET_SIZE: Tuple[int, int] = (224, 224)
HIST_BINS = 256
# 224*224 pixels + 4 statistics + 256 histogram bins
EXPECTED_N_FEATURES = TARGET_SIZE[0] * TARGET_SIZE[1] + 4 + HIST_BINS  # 50436

# Classes in the shipped artifact that are not diagnoses.
NON_DIAGNOSTIC_CLASSES = {"PET"}

DISPLAY_NAMES = {
    "Non Demented": "Non-demented",
    "Very mild Dementia": "Very mild dementia",
    "Mild Dementia": "Mild dementia",
    "Moderate Dementia": "Moderate dementia",
    "PET": "PET (imaging modality, not a diagnosis)",
}

MODEL_CAVEAT = (
    "Reported test accuracy for these artifacts (99.3-99.5 %) is not "
    "trustworthy: the class balance is extreme and the train/test split is "
    "random over individual MRI slices, so adjacent slices of the same brain "
    "appear on both sides of the split. Expect substantially lower accuracy on "
    "unseen patients. Retraining with a subject-level split is required before "
    "the accuracy figure means anything."
)

# The four dementia stages the client's README describes.
EXPECTED_DIAGNOSTIC_STAGES = (
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia",
)


class ImageModelUnavailable(RuntimeError):
    """Raised when the classifier artifact cannot be loaded."""


# ---------------------------------------------------------------------------
# Preprocessing - must match train_model.py exactly
# ---------------------------------------------------------------------------
def decode_image(source: Any) -> np.ndarray:
    """
    Decode to a 3-channel BGR uint8 array, matching cv2.imread's output.

    Accepts a path, raw bytes, or a Streamlit UploadedFile.
    """
    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ImageModelUnavailable(f"image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ImageModelUnavailable(f"could not decode image: {path}")
        return img
    else:
        if isinstance(source, (bytes, bytearray, memoryview)):
            raw = bytes(source)
        elif hasattr(source, "getvalue"):
            raw = source.getvalue()
        elif hasattr(source, "read"):
            raw = source.read()
        else:
            raise ImageModelUnavailable(
                f"unsupported image source type: {type(source)!r}"
            )
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageModelUnavailable("could not decode uploaded image bytes")
        return img

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def preprocess_like_training(source: Any) -> np.ndarray:
    """
    Reproduce train_model.py's `load_and_preprocess_image`.

    Order matters and is preserved deliberately:
      resize on the 3-channel image -> divide by 255 -> BGR to grayscale.
    Converting to grayscale before resizing, or scaling after conversion,
    both shift the pixel values slightly and would reintroduce a mismatch.
    """
    img = decode_image(source)
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
    return img


def extract_training_features(img: np.ndarray) -> np.ndarray:
    """
    Reproduce train_model.py's `extract_image_features`.

    Returns 50,436 features: 50,176 flattened pixels, 4 statistics, then a
    256-bin histogram normalised to sum to 1.
    """
    if img is None or img.size == 0:
        return np.zeros(0, dtype=np.float32)

    arr = np.ascontiguousarray(img, dtype=np.float32)
    pixels = arr.flatten()

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    hist = cv2.calcHist([arr], [0], None, [HIST_BINS], [0, 1])
    total = float(np.sum(hist))
    hist = (hist.flatten() / total) if total > 0 else hist.flatten()

    return np.concatenate(
        [pixels, [mean_val, std_val, min_val, max_val], hist]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# The original buggy pipeline, retained only so the fix can be measured
# ---------------------------------------------------------------------------
def preprocess_legacy_app(source: Any) -> np.ndarray:
    """
    Reproduce `streamlit_inference.py`'s `load_and_preprocess_image`.

    Differs from the training preprocessing in three further ways beyond the
    feature bug, each of which shifts pixel values on its own:
      * a 2nd/98th-percentile contrast stretch that training never applied;
      * PIL LANCZOS resampling instead of cv2 INTER_LINEAR;
      * grayscale conversion before resizing rather than after.
    Kept so the observed symptom can be reproduced exactly rather than
    approximated. Do not use for inference.
    """
    from PIL import Image, ImageOps

    if isinstance(source, (str, Path)):
        pil_img = Image.open(str(source))
    elif isinstance(source, (bytes, bytearray, memoryview)):
        import io

        pil_img = Image.open(io.BytesIO(bytes(source)))
    elif hasattr(source, "getvalue"):
        import io

        pil_img = Image.open(io.BytesIO(source.getvalue()))
    else:
        raise ImageModelUnavailable(f"unsupported source: {type(source)!r}")

    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.convert("L")

    img_np = np.array(pil_img, dtype=np.float32)
    p2, p98 = np.percentile(img_np, (2, 98))
    if p98 > p2:
        img_np = np.clip((img_np - p2) / (p98 - p2), 0, 1)
    else:
        img_np = img_np / 255.0

    resized = Image.fromarray((img_np * 255).astype(np.uint8)).resize(
        TARGET_SIZE, Image.Resampling.LANCZOS
    )
    return np.array(resized, dtype=np.float32) / 255.0


def extract_features_legacy_buggy(
    img_gray_2d: np.ndarray, target_features: int = EXPECTED_N_FEATURES
) -> np.ndarray:
    """
    The extraction used by the current `streamlit_inference.py`.

    Kept verbatim (44 features tiled up to the target length) so
    `scripts/verify_image_fix.py` can compare the two side by side. Do not use
    this for inference.
    """
    arr = np.asarray(img_gray_2d, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    mean_val, std_val = np.mean(arr), np.std(arr)
    min_val, max_val = np.min(arr), np.max(arr)

    hist, _ = np.histogram(arr.flatten(), bins=32, range=(0, 1))
    hist = hist.astype(np.float32) / (np.sum(hist) + 1e-6)

    h, w = arr.shape
    h2, w2 = h // 2, w // 2
    quadrants = [arr[:h2, :w2], arr[:h2, w2:], arr[h2:, :w2], arr[h2:, w2:]]
    quad_stats: List[float] = []
    for q in quadrants:
        quad_stats.extend([float(np.mean(q)), float(np.std(q))])

    base = np.concatenate(
        [[mean_val, std_val, min_val, max_val], hist, np.array(quad_stats, np.float32)]
    )
    if base.size < target_features:
        repeats = int(np.ceil(target_features / base.size))
        return np.tile(base, repeats)[:target_features].astype(np.float32)
    return base[:target_features].astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@dataclass
class ImageModelBundle:
    """A loaded Alzheimer classifier artifact."""

    classifier: Any
    scaler: Any
    label_encoder: Any
    classes: List[str] = field(default_factory=list)
    n_features: int = 0
    stored_test_accuracy: Optional[float] = None

    @property
    def diagnostic_classes(self) -> List[str]:
        return [c for c in self.classes if c not in NON_DIAGNOSTIC_CLASSES]

    @property
    def missing_stages(self) -> List[str]:
        """
        Stages the client's README promises that this artifact cannot output.

        The two artifacts differ: the copy in `artifacts/` has five classes
        (including a spurious "PET"), while the GitHub Release asset the
        deployed app actually downloads has only three and omits "Very mild
        Dementia" entirely. A three-class model cannot report four stages no
        matter how the features are built, so this is surfaced in the UI
        separately from the feature bug.
        """
        present = set(self.classes)
        return [s for s in EXPECTED_DIAGNOSTIC_STAGES if s not in present]

    def coverage_note(self) -> Optional[str]:
        missing = self.missing_stages
        if not missing:
            return None
        return (
            f"This artifact has {len(self.diagnostic_classes)} diagnostic "
            f"class(es): {', '.join(self.diagnostic_classes)}. It cannot "
            f"predict: {', '.join(missing)}. Retraining is required to report "
            "those stages."
        )


def load_image_model(path: str | Path) -> ImageModelBundle:
    """
    Load the classifier from a .pkl or gzipped .pkl.gz.

    Tries gzip+joblib first (how the repository ships it), then plain joblib,
    then plain pickle.
    """
    p = Path(path)
    if not p.exists():
        raise ImageModelUnavailable(f"classifier not found at {p}")

    artifact = None
    errors: List[str] = []
    for loader_name, loader in (
        ("gzip+joblib", lambda: joblib.load(gzip.open(p, "rb"))),
        ("joblib", lambda: joblib.load(p)),
        ("gzip+pickle", lambda: pickle.load(gzip.open(p, "rb"))),
        ("pickle", lambda: pickle.load(open(p, "rb"))),
    ):
        try:
            artifact = loader()
            break
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")
    if artifact is None:
        raise ImageModelUnavailable(
            f"could not load {p}. Tried: " + " | ".join(errors)
        )

    if not isinstance(artifact, dict):
        raise ImageModelUnavailable(
            f"expected a dict artifact, got {type(artifact).__name__}"
        )

    missing = [k for k in ("classifier", "scaler", "label_encoder") if k not in artifact]
    if missing:
        raise ImageModelUnavailable(f"artifact missing keys: {missing}")

    label_encoder = artifact["label_encoder"]
    classes = list(
        artifact.get("classes")
        or getattr(label_encoder, "classes_", [])
    )
    scaler = artifact["scaler"]

    return ImageModelBundle(
        classifier=artifact["classifier"],
        scaler=scaler,
        label_encoder=label_encoder,
        classes=[str(c) for c in classes],
        n_features=int(getattr(scaler, "n_features_in_", EXPECTED_N_FEATURES)),
        stored_test_accuracy=artifact.get("test_accuracy"),
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
@dataclass
class ImagePrediction:
    """One classification result."""

    label: str = ""
    display_label: str = ""
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    dropped_classes: Dict[str, float] = field(default_factory=dict)
    n_features: int = 0
    caveat: str = MODEL_CAVEAT
    notes: List[str] = field(default_factory=list)
    preprocessed: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "display_label": self.display_label,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {
                k: round(float(v), 6) for k, v in self.probabilities.items()
            },
            "dropped_classes": {
                k: round(float(v), 6) for k, v in self.dropped_classes.items()
            },
            "n_features": self.n_features,
            "notes": list(self.notes),
            "caveat": self.caveat,
        }


def classify_image(
    source: Any,
    bundle: ImageModelBundle,
    drop_non_diagnostic: bool = True,
) -> ImagePrediction:
    """
    Classify an MRI image using the corrected feature pipeline.

    `drop_non_diagnostic` removes the "PET" class from the returned
    distribution and renormalises, because PET is an imaging modality that was
    mistakenly trained as a diagnostic label.
    """
    out = ImagePrediction()

    img = preprocess_like_training(source)
    out.preprocessed = img
    features = extract_training_features(img)
    out.n_features = int(features.size)

    expected = bundle.n_features
    if features.size != expected:
        out.notes.append(
            f"feature count {features.size} does not match the model's "
            f"{expected}; the artifact may have changed"
        )
        if features.size > expected:
            features = features[:expected]
        else:
            features = np.pad(features, (0, expected - features.size))

    scaled = bundle.scaler.transform(features.reshape(1, -1))

    classes = list(bundle.classes)
    if hasattr(bundle.classifier, "predict_proba"):
        probs = np.asarray(bundle.classifier.predict_proba(scaled)[0], dtype=np.float64)
    else:
        idx = int(bundle.classifier.predict(scaled)[0])
        probs = np.zeros(len(classes))
        probs[idx] = 1.0
        out.notes.append("classifier has no predict_proba; confidence is nominal")

    # Align the probability vector to label order via the classifier's own
    # classes_ attribute rather than assuming positional order.
    class_order = getattr(bundle.classifier, "classes_", None)
    if class_order is not None and len(class_order) == len(probs):
        try:
            names = list(bundle.label_encoder.inverse_transform(class_order))
            classes = [str(n) for n in names]
        except Exception:
            pass

    full = {classes[i]: float(probs[i]) for i in range(min(len(classes), probs.size))}

    if drop_non_diagnostic:
        dropped = {k: v for k, v in full.items() if k in NON_DIAGNOSTIC_CLASSES}
        kept = {k: v for k, v in full.items() if k not in NON_DIAGNOSTIC_CLASSES}
        total = sum(kept.values())
        if total > 0:
            kept = {k: v / total for k, v in kept.items()}
        out.dropped_classes = dropped
        if dropped:
            out.notes.append(
                "the 'PET' class was removed and probabilities renormalised; "
                "PET is an imaging modality, not a diagnosis"
            )
        out.probabilities = kept
    else:
        out.probabilities = full

    if not out.probabilities:
        out.notes.append("no diagnostic classes available in this model")
        return out

    out.label = max(out.probabilities, key=out.probabilities.get)
    out.confidence = out.probabilities[out.label]
    out.display_label = DISPLAY_NAMES.get(out.label, out.label)
    return out


__all__ = [
    "DISPLAY_NAMES",
    "EXPECTED_DIAGNOSTIC_STAGES",
    "EXPECTED_N_FEATURES",
    "MODEL_CAVEAT",
    "NON_DIAGNOSTIC_CLASSES",
    "TARGET_SIZE",
    "ImageModelBundle",
    "ImageModelUnavailable",
    "ImagePrediction",
    "classify_image",
    "decode_image",
    "extract_features_legacy_buggy",
    "extract_training_features",
    "load_image_model",
    "preprocess_legacy_app",
    "preprocess_like_training",
]
