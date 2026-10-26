"""Amendment 1.5: Level 3 labels with censoring, false-alarm threshold, early alerts, development and freeze."""
import numpy as np
import pandas as pd
import pytest
import yaml

from src.models import lanl_develop as L
from src.utils import paths
from tests.test_lanl import make_lanl

H, D = 3600, 86400


def _hh(rows):
    return pd.DataFrame(rows, columns=["computer", "hour", "auth_in"]).assign(day=lambda f: f.hour * H // D + 1)


def test_labels_censoring_and_post_compromise_rows():
    comp = pd.DataFrame({"comp": ["A", "B"], "compromise_time": [30 * H + 600, 5 * D + 100]})
    hh = _hh([("A", h, 1) for h in range(0, 40)] + [("B", h, 1) for h in range(0, 48)])
    f, hosts = L.period_frame(hh, comp, (1, 2))             # days 1-2: B's compromise (day 6) is unknown
    a = f[f.computer == "A"]
    assert a.hour.max() == 29                                # hour 30 ends after A's compromise: dropped
    lead = (30 * H + 600) - (a.hour + 1) * H
    assert (a._y == ((lead >= 0) & (lead < D)).astype(int)).all() and a._y.sum() == 24
    assert f[f.computer == "B"]._y.sum() == 0 and list(hosts.index) == ["A"]
    assert not f[f.computer == "A"]._eligible_day.any()        # A's days 1-2 touch the 24 h before t
    assert f[f.computer == "B"]._eligible_day.all()


def test_trailing_sums_are_causal_and_per_host():
    hh = _hh([("A", 0, 1), ("A", 1, 2), ("A", 30, 4), ("B", 1, 8)])
    out = L.add_trailing(hh, ["auth_in"], 24)
    got = out.set_index(["computer", "hour"])
    assert got.loc[("A", 1), "auth_in_sum24h"] == 3 and got.loc[("A", 30), "auth_in_sum24h"] == 4
    assert got.loc[("B", 1), "auth_in_sum24h"] == 8 and got.loc[("A", 1), "active_hours_24h"] == 2


def test_fa_threshold_keeps_budget():
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"computer": rng.integers(0, 50, 5000).astype(str), "day": rng.integers(1, 5, 5000),
                      "_eligible_day": True, "_y": 0, "_t": np.inf, "hour": 0})
    s = rng.random(5000)
    thr, n = L.fa_threshold(s, f, 0.01)
    day_max = pd.DataFrame({"h": f.computer, "d": f.day, "s": s}).groupby(["h", "d"]).s.max()
    assert (day_max >= thr).mean() <= 0.01 and (day_max >= np.nextafter(thr, -np.inf) - 1e-12).sum() >= 1


@pytest.fixture
def lanl_setup(drive, declared, tmp_path, monkeypatch):
    from src.data import lanl
    make_lanl(drive / "data" / "raw")
    lanl.preprocess_lanl(log=lambda *a: None)
    cfg = yaml.safe_load(L.CONFIG.read_text())
    cfg["candidates"] = {"logreg": {"family": "linear", "fixed": {"max_iter": 200}, "grid": {"C": [1.0], "class_weight": [None, "balanced"]}},
                         "xgboost": {"family": "tree", "fixed": {"n_estimators": 10, "early_stopping_rounds": 3,
                                     "tree_method": "hist", "eval_metric": "logloss", "n_jobs": 1},
                                     "grid": {"max_depth": [3], "learning_rate": [0.3], "scale_pos_weight": [1]}}}
    p = tmp_path / "models_lanl.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    monkeypatch.setattr(L, "CONFIG", p)
    monkeypatch.setattr(paths, "MODELS", drive / "models")
    return p


def test_lanl_development_reads_no_test_days_and_freezes(lanl_setup, monkeypatch):
    from src.models import seal
    seen = []
    real = L.read_host_hours
    monkeypatch.setattr(L, "read_host_hours", lambda last: seen.append(last) or real(last))
    for name in ("logreg", "xgboost"):
        r = L.develop(name, cfg_path=lanl_setup, log=lambda *a: None)
        assert r["complete"] and len(r["final"]) == 5
        for v in r["final"].values():
            assert v["val"]["fa_rate"] <= 0.01 + 1e-12
    assert set(seen) == {L.split()["val_days"][1]} and seen[0] < L.split()["test_days"][0]
    fr = seal.freeze_lanl(cfg_path=lanl_setup, log=lambda *a: None)
    assert fr["proposed"] in ("logreg", "xgboost") and fr["prior_exposure_sha256"]
    assert set(fr["val_hosts_alerted_per_seed"]) == {"logreg", "xgboost"}
    assert "Predictive model of H2" in seal.write_summary().read_text()
    assert seal.require_frozen(require_git=False, track="lanl")["proposed"] == fr["proposed"]
    with pytest.raises(seal.Sealed):
        seal.freeze_lanl(cfg_path=lanl_setup, log=lambda *a: None)
