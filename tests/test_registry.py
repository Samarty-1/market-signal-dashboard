import json

import pytest
from sklearn.linear_model import LogisticRegression

from src import registry


def _fake_metrics(name="logistic_regression", auc=0.5):
    return {
        "trained_at_utc": "2026-01-01T00:00:00+00:00",
        "best_model": name,
        "comparison": {name: {"mean_roc_auc": auc}},
    }


def test_load_latest_raises_when_registry_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        registry.load_latest(tmp_path)


def test_save_version_then_load_latest_round_trips(tmp_path):
    model = LogisticRegression()
    metrics = _fake_metrics(auc=0.51)

    version = registry.save_version(model, metrics, tmp_path)
    loaded_model, loaded_metrics = registry.load_latest(tmp_path)

    assert registry.latest_version(tmp_path) == version
    assert loaded_metrics["comparison"]["logistic_regression"]["mean_roc_auc"] == 0.51
    assert isinstance(loaded_model, LogisticRegression)


def test_save_version_appends_history_and_updates_pointer(tmp_path):
    model = LogisticRegression()
    v1 = registry.save_version(model, _fake_metrics(auc=0.50), tmp_path)
    v2 = registry.save_version(model, _fake_metrics(auc=0.55), tmp_path)

    pointer = json.loads((tmp_path / "registry" / "registry.json").read_text())
    assert pointer["latest"] == v2
    versions_in_history = [h["version"] for h in pointer["history"]]
    assert versions_in_history[:2] == [v2, v1]  # newest first


def test_save_version_prunes_old_artifacts_beyond_max_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MAX_VERSIONS", 2)
    model = LogisticRegression()
    versions = [registry.save_version(model, _fake_metrics(auc=0.5 + i * 0.01), tmp_path) for i in range(4)]

    root = tmp_path / "registry"
    pointer = json.loads((root / "registry.json").read_text())
    # All 4 retain their metadata line in history...
    assert len(pointer["history"]) == 4
    # ...but only the 2 most recent still have on-disk model artifacts.
    for old_version in versions[:2]:
        assert not (root / old_version).exists()
    for recent_version in versions[2:]:
        assert (root / recent_version / "model.joblib").exists()
