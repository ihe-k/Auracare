
Locating the project's own files and the sibling Cognitive_AI_Triage checkout.

The model artifacts and the sample recordings live in the client's existing
repository, which may sit anywhere relative to this one. Rather than hardcode a
path - the current `streamlit_inference.py` hardcodes `/tmp`, which cannot work
on Windows - this searches a short list of plausible locations and lets an
environment variable override everything.

    AURACARE_TRIAGE_REPO   path to the Cognitive_AI_Triage checkout
    AURACARE_MODEL_DIR     path to a directory holding the .pkl artifacts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "config"

SEVERITY_MODEL_FILENAMES = ("severity_model.pkl", "severity_model.pkl.2")
IMAGE_MODEL_FILENAMES = (
    # The GitHub Release asset first: it is the artifact the deployed app
    # downloads, so a local run matches production by default. It is a raw
    # (ungzipped) pickle despite the extension, and has 3 classes where the
    # in-repo copy has 5.
    "alzheimer_classifier_release.pkl",
    "alzheimer_classifier.pkl.gz",
    "alzheimer_classifier.pkl",
    "alzheimer_classifier.pkl.2",
)


def _candidate_repo_roots() -> List[Path]:
    """Plausible locations for the Cognitive_AI_Triage checkout."""
    env = os.environ.get("AURACARE_TRIAGE_REPO")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        PROJECT_ROOT / "Cognitive_AI_Triage",
        PROJECT_ROOT.parent / "Cognitive_AI_Triage",
        PROJECT_ROOT.parent.parent / "Cognitive_AI_Triage",
        Path.cwd() / "Cognitive_AI_Triage",
        Path.cwd().parent / "Cognitive_AI_Triage",
    ]
    return candidates


def find_triage_repo() -> Optional[Path]:
    """The Cognitive_AI_Triage checkout, or None."""
    for path in _candidate_repo_roots():
        if (path / "streamlit_inference.py").exists() or (path / "artifacts").is_dir():
            return path.resolve()
    return None


def _model_search_dirs() -> List[Path]:
    dirs: List[Path] = []
    env = os.environ.get("AURACARE_MODEL_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(PROJECT_ROOT / "models")
    repo = find_triage_repo()
    if repo:
        dirs += [
            repo / "artifacts",
            repo / "predictive_cognitive_final" / "artifacts",
        ]
    return dirs


def find_severity_model() -> Optional[Path]:
    """Path to the PHQ-8 severity-model artifact, or None."""
    for directory in _model_search_dirs():
        for name in SEVERITY_MODEL_FILENAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()
    return None


def find_image_model() -> Optional[Path]:
    """Path to the Alzheimer classifier artifact, or None."""
    for directory in _model_search_dirs():
        for name in IMAGE_MODEL_FILENAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()
    return None


def find_practice_files() -> Optional[Path]:
    """The sample-recordings directory in the client's repo, or None."""
    repo = find_triage_repo()
    if repo and (repo / "Practice_Files").is_dir():
        return (repo / "Practice_Files").resolve()
    local = PROJECT_ROOT / "Practice_Files"
    return local.resolve() if local.is_dir() else None


def describe() -> dict:
    """Everything the UI needs to tell the user what it did and did not find."""
    repo = find_triage_repo()
    return {
        "project_root": str(PROJECT_ROOT),
        "triage_repo": str(repo) if repo else None,
        "severity_model": str(find_severity_model() or "") or None,
        "image_model": str(find_image_model() or "") or None,
        "practice_files": str(find_practice_files() or "") or None,
    }


__all__ = [
    "ASSETS_DIR",
    "CONFIG_DIR",
    "PROJECT_ROOT",
    "describe",
    "find_image_model",
    "find_practice_files",
    "find_severity_model",
    "find_triage_repo",
]
