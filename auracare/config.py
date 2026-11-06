"""
Threshold configuration for the Auracare screening prototype.

Two separate, clearly-labelled sets of numbers live here:

  PROVISIONAL_DEFAULTS
      Operating points this *prototype* uses to decide what to show on screen.
      They are demo/engineering values, tunable at runtime from the sidebar, and
      they are NOT diagnostic. They exist so the flow can be driven and tested.

  CLINICAL_REFERENCE
      Published reference values, kept read-only and displayed alongside the
      provisional numbers so a clinician can see how far the demo operating
      point sits from the literature. Nothing in the app branches on these.

Keeping them apart is deliberate: it stops a tuned demo threshold from ever
being mistaken for a validated clinical cut-off.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Provisional (tunable, prototype-only) operating points
# ---------------------------------------------------------------------------
PROVISIONAL_DEFAULTS: Dict[str, Dict[str, float]] = {
    # -- Task 1: sustained phonation /a:/ -----------------------------------
    "task1": {
        # Safety intercept from the Auracare protocol spec.
        "phonation_critical_s": 2.5,
        "phonation_low_s": 5.0,
        "phonation_target_s": 10.0,
        "jitter_warn_pct": 1.04,
        "jitter_high_pct": 2.00,
        "shimmer_warn_pct": 3.81,
        "shimmer_high_pct": 6.00,
        "hnr_warn_db": 20.0,
        "hnr_low_db": 12.0,
        "f0_cv_warn": 0.05,
    },
    # -- Task 2: forced expiratory manoeuvre --------------------------------
    # Exhalation duration is graded on BOTH sides, because the two abnormal
    # directions mean opposite things: too short means the patient could not
    # generate flow at all, too long suggests airflow obstruction (a prolonged
    # forced expiratory time is the classic auscultatory sign).
    #
    # The numbers are scaled to what the extractor actually measures - time
    # spent within 12 dB of peak, not the full breath. A healthy forced
    # exhalation measures around 0.8-1.4 s on that definition. An earlier
    # draft set the low bound at 1.00 s against the nominal breath length and
    # consequently graded every normal exhalation as Elevated.
    "task2": {
        "exhalation_critical_s": 0.40,
        "exhalation_low_s": 0.60,
        "exhalation_normal_max_s": 2.50,
        "exhalation_prolonged_s": 3.50,
        "rise_time_warn_s": 0.30,
        "rise_time_high_s": 0.60,
        # Tonal-peak prominence over the local noise spectrum, in dB.
        "wheeze_index_warn_db": 9.0,
        "wheeze_index_high_db": 14.0,
        "decay_slope_warn_db_s": -6.0,
    },
    # -- Task 3: diadochokinetic rate /pa-ta-ka/ ----------------------------
    "task3": {
        "ddk_rate_low_hz": 5.00,
        "ddk_rate_critical_hz": 3.50,
        # Coefficient of variation of inter-syllable intervals.
        "rhythm_cv_warn": 0.15,
        "rhythm_cv_high": 0.30,
        "min_syllables": 8,
    },
    # -- Task 4: connected speech, VAD + PHQ-8 ------------------------------
    # CALIBRATION WARNING on pause_ratio_critical. The protocol specifies 0.45.
    # Measured across 76 thirty-second windows of the client's own sample
    # recordings, the pause ratio ranged 0.24-0.72 with a median of 0.48 - so
    # 0.45 sits *below* the median of ordinary speech and triggered on 57 % of
    # windows. The spec value is kept as the default because it is the spec,
    # and it is exposed as a sidebar control, but it needs recalibrating
    # against real Task 4 recordings before it can carry any weight. See
    # scripts/calibrate_pause_ratio.py. The breach routes to clinical review,
    # not to the respiratory emergency pathway - see risk.py.
    "task4": {
        "pause_ratio_critical": 0.45,
        "pause_ratio_warn": 0.35,
        "speech_rate_low_sylls": 2.50,
        "speech_rate_high_sylls": 6.50,
        "min_speech_s": 5.0,
        "lexical_diversity_low": 0.40,
        # Negative-language token share of all spoken tokens.
        "negative_language_warn": 0.030,
        "negative_language_high": 0.060,
        "first_person_singular_warn": 0.085,
    },
    # -- PHQ-8 scoring (standard published cut-offs) ------------------------
    # Held here so they are visible and auditable, but see CLINICAL_REFERENCE:
    # these are the real Kroenke cut-points and should not be tuned casually.
    "phq8": {
        "mild": 5,
        "moderate": 10,
        "moderately_severe": 15,
        "severe": 20,
        # Kroenke et al. (2009): PHQ-8 >= 10 identifies current depression.
        "clinically_significant": 10,
    },
}


# ---------------------------------------------------------------------------
# Clinical reference values (read-only, display only, never branched on)
# ---------------------------------------------------------------------------
CLINICAL_REFERENCE: Dict[str, Dict[str, Any]] = {
    "task1": {
        "Maximum phonation time, adult male": "25-35 s typical; < 10 s abnormal",
        "Maximum phonation time, adult female": "15-25 s typical; < 7 s abnormal",
        "Jitter (local), Praat default": "< 1.040 % considered normal",
        "Shimmer (local), Praat default": "< 3.810 % considered normal",
        "Harmonics-to-noise ratio, sustained /a/": "> 20 dB typical normal voice",
    },
    "task2": {
        "Forced expiratory time (auscultated)": "< 6 s normal; > 9 s suggests obstruction",
        "Wheeze fundamental frequency": "commonly 100-1000 Hz; > 400 Hz for high-pitched",
        "Note": "Acoustic peak-flow proxies are NOT calibrated spirometry.",
    },
    "task3": {
        "/pa-ta-ka/ sequential motion rate, healthy adult": "approx. 6.2-6.8 syllables/s",
        "Alternating motion rate /pa/": "approx. 5.5-7.0 syllables/s",
        "Reduced rate threshold commonly cited": "< 5 syllables/s warrants attention",
    },
    "task4": {
        "Speech rate, conversational English": "approx. 4-6 syllables/s",
        "Silent pause proportion, healthy connected speech": "approx. 15-30 %",
        "Note": "Pause-ratio cut-offs for depression are not standardised.",
    },
    "phq8": {
        "Source": "Kroenke K, et al. J Affect Disord. 2009;114(1-3):163-73.",
        "Severity bands": "0-4 none/minimal, 5-9 mild, 10-14 moderate, "
                          "15-19 moderately severe, 20-24 severe",
        "Case-finding cut-point": "total score >= 10",
        "Scale range": "0-24 (eight items, 0-3 each)",
    },
}


# ---------------------------------------------------------------------------
# Signal-processing parameters (not clinical; shared by the extractors)
# ---------------------------------------------------------------------------
@dataclass
class SignalParams:
    """Frame geometry and detector settings used across the feature extractors."""

    sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0

    # Voice activity detection
    vad_floor_offset_db: float = 8.0
    vad_peak_floor_db: float = 35.0
    vad_min_speech_ms: float = 120.0
    vad_min_pause_ms: float = 250.0

    # Pitch analysis (Praat)
    f0_min_hz: float = 75.0
    f0_max_hz: float = 500.0

    # Syllable-nucleus detection
    syllable_min_distance_ms: float = 90.0
    syllable_prominence_db: float = 3.0
    syllable_smooth_ms: float = 25.0


@dataclass
class Thresholds:
    """Mutable, serialisable provisional thresholds."""

    values: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: copy.deepcopy(PROVISIONAL_DEFAULTS)
    )

    def get(self, group: str, key: str) -> float:
        try:
            return float(self.values[group][key])
        except KeyError as exc:  # pragma: no cover - guards config typos
            raise KeyError(f"unknown threshold {group}.{key}") from exc

    def set(self, group: str, key: str, value: float) -> None:
        if group not in self.values or key not in self.values[group]:
            raise KeyError(f"unknown threshold {group}.{key}")
        self.values[group][key] = float(value)

    def reset(self) -> None:
        self.values = copy.deepcopy(PROVISIONAL_DEFAULTS)

    # -- persistence --------------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.values, indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Dict[str, float]]) -> "Thresholds":
        merged = copy.deepcopy(PROVISIONAL_DEFAULTS)
        for group, items in (raw or {}).items():
            if group in merged:
                for key, value in items.items():
                    if key in merged[group]:
                        merged[group][key] = float(value)
        return cls(values=merged)

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return copy.deepcopy(self.values)


DEFAULT_SIGNAL_PARAMS = SignalParams()

__all__ = [
    "PROVISIONAL_DEFAULTS",
    "CLINICAL_REFERENCE",
    "SignalParams",
    "Thresholds",
    "DEFAULT_SIGNAL_PARAMS",
]
