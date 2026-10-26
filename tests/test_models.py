"""Amendment 1.4: candidate development on train + val only, the freeze and the sealed test."""
import json
import shutil

import numpy as np
import pytest
import yaml

from src.utils import paths
from tests.synthetic import CIC_FEATURES, write_cic

TINY = {  # same structure as config/models.yaml, tiny settings so the test runs in seconds
    "version": "test", "development": {"partitions": ["train", "val"], "grid_seed": 11, "tie_margin": 0.0005},
    "proposed": {"rule": "best_val_macro_f1", "feature_space": "cic", "tie_margin": 0.0005},
    "feature_spaces": {"cic": {}, "unsw_canonical": {}},
    "candidates": {
        "logreg": {"family": "linear", "fixed": {"max_iter": 200}, "grid": {"C": [0.1, 1.0], "class_weight": [None]}},
        "random_forest": {"family": "tree", "fixed": {"n_estimators": 10, "n_jobs": 1},
                          "grid": {"max_depth": [None], "min_samples_leaf": [1, 3], "class_weight": [None]}},
        "xgboost": {"family": "tree", "fixed": {"n_estimators": 20, "early_stopping_rounds": 5, "tree_method": "hist",
                                                "eval_metric": "logloss", "n_jobs": 1},
                    "grid": {"max_depth": [3], "learning_rate": [0.3], "scale_pos_weight": [1, "balanced"]}},
        "mlp": {"family": "neural", "fixed": {"hidden": 2, "epochs": 2, "patience": 1, "batch_size": 512},
                "grid": {"width": [32], "dropout": [0.1], "class_weight": [None]}},
        "cnn_lstm": {"family": "neural", "fixed": {"conv_channels": [4, 8], "lstm_units": 8, "epochs": 2,
                                                   "patience": 1, "batch_size": 512},
                     "grid": {"dropout": [0.1], "class_weight": ["balanced"]}}}}


@pytest.fixture
def setup(drive, tmp_path, monkeypatch):
    from src.data import harmonise
    from src.data.pipeline import preprocess
    from src.models import develop
    write_cic(drive / "data" / "raw")
    preprocess("cicids2017_improved", log=lambda *a: None)
    mdir = tmp_path / "mapping"
    mdir.mkdir()
    shutil.copy(harmonise.MAPPING_DIR / "cic_to_unsw_mapping.yaml", mdir)
    (mdir / "cic_shared_features.yaml").write_text(yaml.safe_dump({"features": CIC_FEATURES}))
    monkeypatch.setattr(harmonise, "MAPPING_DIR", mdir)
    monkeypatch.setattr(paths, "MODELS", drive / "models")
    cfg = tmp_path / "models.yaml"
    cfg.write_text(yaml.safe_dump(TINY, sort_keys=False))
    monkeypatch.setattr(develop, "MODELS_CONFIG", cfg)
    monkeypatch.setenv("PHD_ALLOW_CPU_NEURAL", "1")       # tiny neural models in tests only
    return cfg


def test_best_threshold_is_exact():
    from src.models.candidates import best_threshold
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1).round(2)
    thr, mf = best_threshold(y, p)
    from sklearn.metrics import f1_score
    brute = max((f1_score(y, (p >= t).astype(int), average="macro"), -t) for t in np.r_[np.unique(p), np.inf])
    assert mf == pytest.approx(brute[0]) and thr == pytest.approx(-brute[1])


def test_development_never_loads_sealed_partitions(setup, monkeypatch):
    from src.data import datasets
    from src.models import develop
    seen = []
    real = datasets.load
    monkeypatch.setattr(datasets, "load", lambda *a, **k: seen.append(k.get("partitions")) or real(*a, **k))
    develop.develop("cic", "logreg", cfg_path=setup, log=lambda *a: None)
    assert seen and all(set(p) <= {"train", "val"} for p in seen)
    with pytest.raises(PermissionError):
        develop.load_dev("cic", partitions=("train", "test"))


def test_every_candidate_develops_resumes_and_freezes(setup):
    from src.models import develop, seal
    for space in TINY["feature_spaces"]:
        for name in TINY["candidates"]:
            r = develop.develop(space, name, cfg_path=setup, log=lambda *a: None)
            assert r["complete"] and len(r["final"]) == 5 and 0 <= r["val_macro_f1_mean"] <= 1
            assert r["selected_params"] in [g["params"] for g in r["grid"]]
    again = develop.develop("cic", "logreg", cfg_path=setup, log=lambda *a: None)      # resumes: nothing to do
    assert again["complete"]
    m = develop.load_model("cic", "xgboost", 23, json.loads(develop.record_path("cic", "xgboost").read_text())
                           ["final"]["23"]["model_sha256"])
    assert m.predict_proba(np.zeros((3, len(CIC_FEATURES)), np.float32)).shape == (3,)
    with pytest.raises(seal.Sealed):
        seal.require_frozen(require_git=False)
    fr = seal.freeze(cfg_path=setup, log=lambda *a: None)
    ranking = fr["ranking_val_macro_f1"]
    assert ranking[fr["proposed"]] >= max(ranking.values()) - 0.0005
    assert seal.require_frozen(require_git=False)["proposed"] == fr["proposed"]
    with pytest.raises(seal.Sealed):
        seal.freeze(cfg_path=setup, log=lambda *a: None)                    # only once
    f = paths.MODELS / next(iter(fr["model_files_sha256"]))
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(seal.Sealed, match="changed after the freeze"):
        seal.require_frozen(require_git=False)


def test_freeze_refuses_incomplete_development(setup):
    from src.models import develop, seal
    develop.develop("cic", "logreg", cfg_path=setup, log=lambda *a: None)
    with pytest.raises(seal.Sealed, match="not complete"):
        seal.freeze(cfg_path=setup, log=lambda *a: None)


def test_record_from_another_config_is_not_resumed(setup):
    from src.models import develop
    develop.develop("cic", "logreg", cfg_path=setup, log=lambda *a: None)
    setup.write_text(setup.read_text() + "\n# changed\n")
    with pytest.raises(RuntimeError, match="another config"):
        develop.develop("cic", "logreg", cfg_path=setup, log=lambda *a: None)
