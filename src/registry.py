"""A lightweight, in-repo model registry.

Replaces the old pattern of overwriting models/model.joblib in place on every
scheduled retrain (which leaves no history and no way to roll back a bad
model) with a versioned directory per retrain plus a pointer/history file.
Everything still lives inside the repo and is committed by the same daily
GitHub Action — no external registry service (e.g. MLflow) is provisioned.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

REGISTRY_DIRNAME = "registry"
POINTER_FILENAME = "registry.json"
MAX_VERSIONS = 30  # cap on-disk model artifacts kept; older history stays as metadata only


def _registry_root(models_dir: Path) -> Path:
    return Path(models_dir) / REGISTRY_DIRNAME


def _pointer_path(models_dir: Path) -> Path:
    return _registry_root(models_dir) / POINTER_FILENAME


def _load_pointer(models_dir: Path) -> dict:
    pointer_path = _pointer_path(models_dir)
    if not pointer_path.exists():
        return {"latest": None, "history": []}
    return json.loads(pointer_path.read_text())


def save_version(pipeline, metrics: dict, models_dir: Path) -> str:
    """Save a new registry version, update the pointer, and prune old artifacts.

    Returns the new version string.
    """
    # Microsecond resolution avoids version collisions when retrains happen in
    # quick succession (e.g. manual reruns, or tests exercising this in a loop).
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    root = _registry_root(models_dir)
    version_dir = root / version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, version_dir / "model.joblib")
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pointer = _load_pointer(models_dir)
    pointer["latest"] = version
    pointer["history"].insert(
        0,
        {
            "version": version,
            "trained_at_utc": metrics.get("trained_at_utc"),
            "best_model": metrics.get("best_model"),
            "mean_roc_auc": metrics.get("comparison", {}).get(metrics.get("best_model"), {}).get("mean_roc_auc"),
        },
    )
    _prune(root, pointer)
    _pointer_path(models_dir).write_text(json.dumps(pointer, indent=2))
    return version


def _prune(root: Path, pointer: dict) -> None:
    """Keep on-disk artifacts (model.joblib + metrics.json) for only the most
    recent MAX_VERSIONS entries; older entries keep their metadata line in
    the pointer's history but their version directory is removed."""
    for entry in pointer["history"][MAX_VERSIONS:]:
        version_dir = root / entry["version"]
        if version_dir.exists():
            shutil.rmtree(version_dir)


def load_latest(models_dir: Path):
    """Returns (pipeline, metrics) for the current production version.

    Raises FileNotFoundError if no version has been saved yet.
    """
    pointer = _load_pointer(models_dir)
    latest = pointer.get("latest")
    if latest is None:
        raise FileNotFoundError(
            f"No model versions found in {_registry_root(models_dir)}. "
            "Run `python -m src.model` first (or wait for the scheduled retrain)."
        )
    version_dir = _registry_root(models_dir) / latest
    pipeline = joblib.load(version_dir / "model.joblib")
    metrics = json.loads((version_dir / "metrics.json").read_text())
    return pipeline, metrics


def latest_version(models_dir: Path) -> str | None:
    return _load_pointer(models_dir).get("latest")
