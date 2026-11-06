"""
Optional adapter for the repository's pre-trained severity model.

This exists because it was asked for, and it is wired up honestly rather than
flatteringly. Read this before using its output for anything.

What the model is
    A RandomForestRegressor over 10,249 features, trained on DAIC-WOZ. The
    features are 10 blocks of 1,024 (audio mean/std, facial-keypoint mean/std,
    gaze-confidence mean/std, pose-confidence mean/std, text-embedding
    mean/std), then `Gender`, then the 8 PHQ-8 item scores.

What a microphone can supply
    8 of 10,249 inputs are PHQ-8 items - 0.08 % of the feature space - plus
    `Gender`. The other 10,240 require synchronised video (facial keypoints,
    gaze, pose) and the original DAIC-WOZ audio and text-embedding pipelines,
    none of which a voice-only screening flow has. Those blocks are imputed at
    the scaler's own training means, which is the most defensible imputation
    available because it standardises to exactly zero, the training-set centre.

What it actually does, measured rather than assumed
    Running `sensitivity_sweep()` against the shipped artifact gives a result
    worth stating precisely, because the obvious guess is wrong. The model is
    *not* insensitive to PHQ-8: sweeping all eight items from 0 to 3 moves the
    output from 0.42 to 12.63, traversing about 51 % of the 0-24 scale. The
    problem is the shape of that response, not its size:

      true PHQ-8   true band            model output   model band
               0   None/minimal                 0.42   None/minimal   ok
               8   Mild                         8.13   Mild           ok
              16   Moderately severe           10.78   Moderate       under
              24   Severe                      12.63   Moderate       under

    It tracks the bottom half of the scale and then saturates, compressing
    everything above 16 into "moderate". Both banding errors are
    under-estimates, which on a screening instrument is the dangerous
    direction: a severely depressed patient is reported as moderate.

    Per-item sensitivity is also lopsided. Maxing one item at a time gives
    +4.95 for PHQ8_NoInterest against roughly +1 for most others, and exactly
    +0.00 for PHQ8_Moving - the model ignores that item entirely. So it behaves
    as a function of one or two items rather than of the instrument.

What its own metrics say
    Stored inside the artifact: train R2 0.964 / MAE 0.75, validation R2 0.711
    / MAE 2.43, test R2 -0.751 / MAE 6.01. A negative test R2 means the model
    performs worse on held-out data than always predicting the training mean.
    On a 0-24 scale an MAE of 6.0 cannot separate "none/minimal" from
    "moderately severe".

Therefore
    `risk.py` scores depression severity from the PHQ-8 sum against published
    cut-points, which traverses 0-24 exactly by construction. This module is
    available for comparison, is clearly labelled experimental everywhere it
    surfaces, and never feeds the triage decision. `sensitivity_sweep()`
    regenerates the table above so none of this has to be taken on trust.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .phq8 import ITEM_KEYS, PHQ8_ITEMS

RELEASE_URL = (
    "https://github.com/ihe-k/Cognitive_AI_Triage/releases/download/"
    "v1.0.0/severity_model.pkl.2"
)

EXPECTED_N_FEATURES = 10_249

# Metrics as stored in the artifact itself, for display alongside predictions.
STORED_METRICS = {
    "train": {"mae": 0.750, "rmse": 1.034, "r2": 0.964},
    "validation": {"mae": 2.428, "rmse": 3.490, "r2": 0.711},
    "test": {"mae": 6.013, "rmse": 8.473, "r2": -0.751},
}

CAVEAT = (
    "EXPERIMENTAL - shown for comparison only, never used for triage. "
    "8 of this model's 10,249 inputs come from the PHQ-8; the other 10,240 "
    "require synchronised video and the original DAIC-WOZ feature pipelines, "
    "so they are imputed at training means. Measured behaviour: the output "
    "tracks the lower half of the scale then saturates, reporting a true PHQ-8 "
    "of 24 (severe) as 12.6 (moderate), and it ignores the PHQ8_Moving item "
    "entirely. Its own held-out test R2 is -0.751, worse than predicting the "
    "mean. Both banding errors are under-estimates."
)


class SeverityModelUnavailable(RuntimeError):
    """Raised when the artifact cannot be loaded."""


@dataclass
class SeverityModelBundle:
    """A loaded severity-model artifact."""

    model: object
    scaler: object
    feature_names: List[str]
    index_of: Dict[str, int] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def load_bundle(path: str | Path) -> SeverityModelBundle:
    """Load the pickled artifact from disk."""
    p = Path(path)
    if not p.exists():
        raise SeverityModelUnavailable(f"severity model not found at {p}")
    try:
        with open(p, "rb") as fh:
            artifact = pickle.load(fh)
    except Exception as exc:
        raise SeverityModelUnavailable(f"could not unpickle {p}: {exc}") from exc

    missing = [k for k in ("model", "scaler", "feature_names") if k not in artifact]
    if missing:
        raise SeverityModelUnavailable(f"artifact missing keys: {missing}")

    names = list(artifact["feature_names"])
    bundle = SeverityModelBundle(
        model=artifact["model"],
        scaler=artifact["scaler"],
        feature_names=names,
        index_of={name: i for i, name in enumerate(names)},
    )
    if bundle.n_features != EXPECTED_N_FEATURES:
        # Not fatal - just means the artifact changed since this was written.
        pass
    return bundle


def build_feature_vector(
    bundle: SeverityModelBundle,
    phq8_scores: Dict[str, int],
    gender: Optional[float] = None,
    impute: str = "mean",
) -> np.ndarray:
    """
    Build one input row for the model.

    `impute="mean"` fills the 10,240 unavailable features with the scaler's own
    training means, so they standardise to exactly zero and contribute the
    training-set centre rather than an out-of-distribution value.
    `impute="zero"` fills them with literal zeros, which is what a naive
    integration would do; it is supported only so the two can be compared.
    """
    n = bundle.n_features

    if impute == "mean" and hasattr(bundle.scaler, "mean_"):
        x = np.asarray(bundle.scaler.mean_, dtype=np.float64).copy()
    elif impute in ("mean", "zero"):
        x = np.zeros(n, dtype=np.float64)
    else:
        raise ValueError(f"unknown impute mode: {impute!r}")

    if x.size != n:  # pragma: no cover - guards a malformed artifact
        x = np.zeros(n, dtype=np.float64)

    for key, value in (phq8_scores or {}).items():
        idx = bundle.index_of.get(key)
        if idx is not None:
            x[idx] = float(value)

    if gender is not None:
        idx = bundle.index_of.get("Gender")
        if idx is not None:
            x[idx] = float(gender)

    return x.reshape(1, -1)


@dataclass
class SeverityPrediction:
    """One experimental model prediction, with its caveat attached."""

    predicted_phq8: Optional[float] = None
    available: bool = False
    impute_mode: str = "mean"
    n_features: int = 0
    n_features_supplied: int = 0
    caveat: str = CAVEAT
    stored_metrics: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: dict(STORED_METRICS)
    )
    unavailable_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "available": self.available,
            "predicted_phq8": (
                None if self.predicted_phq8 is None else round(self.predicted_phq8, 3)
            ),
            "impute_mode": self.impute_mode,
            "n_features": self.n_features,
            "n_features_supplied": self.n_features_supplied,
            "fraction_supplied": (
                round(self.n_features_supplied / self.n_features, 6)
                if self.n_features
                else None
            ),
            "caveat": self.caveat,
            "stored_metrics": self.stored_metrics,
            "unavailable_reason": self.unavailable_reason,
        }


def predict_severity(
    bundle: SeverityModelBundle,
    phq8_scores: Dict[str, int],
    gender: Optional[float] = None,
    impute: str = "mean",
) -> SeverityPrediction:
    """Run the experimental model. Never used for triage."""
    out = SeverityPrediction(impute_mode=impute, n_features=bundle.n_features)
    supplied = [k for k in (phq8_scores or {}) if k in bundle.index_of]
    if gender is not None and "Gender" in bundle.index_of:
        supplied.append("Gender")
    out.n_features_supplied = len(supplied)

    try:
        x = build_feature_vector(bundle, phq8_scores, gender, impute)
        x_scaled = bundle.scaler.transform(x)  # type: ignore[attr-defined]
        pred = float(bundle.model.predict(x_scaled)[0])  # type: ignore[attr-defined]
    except Exception as exc:
        out.unavailable_reason = f"prediction failed: {exc}"
        return out

    out.predicted_phq8 = pred
    out.available = True
    return out


def sensitivity_sweep(
    bundle: SeverityModelBundle,
    gender: Optional[float] = None,
    impute: str = "mean",
) -> Dict[str, object]:
    """
    Quantify how much the PHQ-8 items can move this model's output.

    Sweeps every item together from 0 to 3 (the full 0-24 range of the
    instrument) and reports the resulting prediction range. If that range is
    small relative to the 0-24 scale, the model is not meaningfully responding
    to the only inputs a voice flow can supply - which is the point being
    tested.
    """
    from .phq8 import band_for_score

    rows = []
    for level in (0, 1, 2, 3):
        scores = {k: level for k in ITEM_KEYS}
        pred = predict_severity(bundle, scores, gender, impute)
        true_total = level * len(ITEM_KEYS)
        true_band, true_idx = band_for_score(true_total)
        if pred.predicted_phq8 is None:
            model_band, model_idx = None, None
            direction = None
        else:
            model_band, model_idx = band_for_score(pred.predicted_phq8)
            if model_idx == true_idx:
                direction = "match"
            elif model_idx < true_idx:
                direction = "under-estimate"
            else:
                direction = "over-estimate"
        rows.append(
            {
                "item_level": level,
                "phq8_true_total": true_total,
                "true_band": true_band,
                "model_prediction": pred.predicted_phq8,
                "model_band": model_band,
                "band_agreement": direction,
            }
        )

    preds = [r["model_prediction"] for r in rows if r["model_prediction"] is not None]
    span = (max(preds) - min(preds)) if preds else None
    n_match = sum(1 for r in rows if r["band_agreement"] == "match")
    n_under = sum(1 for r in rows if r["band_agreement"] == "under-estimate")

    # Per-item sensitivity: move one item 0 -> 3, hold the rest at 0.
    per_item = []
    base = predict_severity(bundle, {k: 0 for k in ITEM_KEYS}, gender, impute)
    for item in PHQ8_ITEMS:
        scores = {k: 0 for k in ITEM_KEYS}
        scores[item.key] = 3
        p = predict_severity(bundle, scores, gender, impute)
        delta = (
            None
            if (p.predicted_phq8 is None or base.predicted_phq8 is None)
            else round(p.predicted_phq8 - base.predicted_phq8, 4)
        )
        per_item.append({"item": item.key, "delta_when_maxed": delta})

    ignored = [r["item"] for r in per_item if r["delta_when_maxed"] == 0.0]

    return {
        "impute_mode": impute,
        "sweep": rows,
        "prediction_span": None if span is None else round(span, 4),
        "true_span": 24.0,
        "span_fraction_of_scale": None if span is None else round(span / 24.0, 4),
        "bands_matched": n_match,
        "bands_under_estimated": n_under,
        "bands_tested": len(rows),
        "per_item": per_item,
        "items_ignored": ignored,
        "n_features": bundle.n_features,
        "n_features_drivable_by_voice": len(ITEM_KEYS) + 1,
    }


__all__ = [
    "CAVEAT",
    "EXPECTED_N_FEATURES",
    "RELEASE_URL",
    "STORED_METRICS",
    "SeverityModelBundle",
    "SeverityModelUnavailable",
    "SeverityPrediction",
    "build_feature_vector",
    "load_bundle",
    "predict_severity",
    "sensitivity_sweep",
]
