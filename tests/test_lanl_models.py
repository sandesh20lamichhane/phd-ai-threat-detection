"""Amendments 1.5 and 1.6: Level 3 labels with censoring, false-alarm threshold, early alerts, development, freeze."""
import json

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
    f, hosts = L.period_frame(hh, comp, (1, 2), unresolved="negative")   # 1.5 rule; B's day-6 compromise unknown
    a = f[f.computer == "A"]
    assert a.hour.max() == 29                                # hour 30 ends after A's compromise: dropped
    lead = (30 * H + 600) - (a.hour + 1) * H
    assert (a._y == ((lead >= 0) & (lead < D)).astype(int)).all() and a._y.sum() == 24
    assert f[f.computer == "B"]._y.sum() == 0 and list(hosts.index) == ["A"]
    assert not f[f.computer == "A"]._eligible_day.any()        # A's days 1-2 touch the 24 h before t
    assert f[f.computer == "B"]._eligible_day.all()


def test_unresolved_rows_at_the_period_end_leave_the_population():
    """Amendment 1.6: a row whose 24 h window runs past the period end, of a host with no known compromise, has no
    known outcome. It is neither negative nor a false-alarm host-day, even if the host is compromised just after."""
    comp = pd.DataFrame({"comp": ["A", "B"], "compromise_time": [2 * D + 2 * H, 30 * H + 600]})
    hh = _hh([("A", h, 1) for h in range(0, 48)] + [("B", h, 1) for h in range(0, 40)] + [("C", h, 1) for h in range(48)])
    f, hosts = L.period_frame(hh, comp, (1, 2))               # A is compromised 2 h after the period ends
    for c in ("A", "C"):                                      # known outcome only for windows ending by day 2
        r = f[f.computer == c]
        assert r.hour.max() == 23 and (r._y == 0).all() and r._eligible_day.all()
    b = f[f.computer == "B"]                                  # known compromise inside the period: resolved
    assert b.hour.max() == 29 and b._y.sum() == 24 and list(hosts.index) == ["B"]
    g, _ = L.period_frame(hh, comp, (1, 3))                   # a later period knows A's compromise
    assert g[(g.computer == "A") & (g.day == 2)]._y.sum() == 22 and not g[(g.computer == "A") & (g.day == 2)]._eligible_day.any()
    with pytest.raises(ValueError):
        L.period_frame(hh, comp, (1, 2), unresolved="drop")


def test_trailing_sums_are_causal_and_per_host():
    hh = _hh([("A", 0, 1), ("A", 1, 2), ("A", 30, 4), ("B", 1, 8)])
    out = L.add_trailing(hh, ["auth_in"], 24)
    got = out.set_index(["computer", "hour"])
    assert got.loc[("A", 1), "auth_in_sum24h"] == 3 and got.loc[("A", 30), "auth_in_sum24h"] == 4
    assert got.loc[("B", 1), "auth_in_sum24h"] == 8 and got.loc[("A", 1), "active_hours_24h"] == 2


def test_trailing_sums_one_column_at_a_time_match_the_development_code():
    """The column-wise sums (amendment 1.6, for memory) give exactly the numbers of the matrix version used for the
    1.5 development records."""
    rng = np.random.default_rng(3)
    n = 20_000
    hh = pd.DataFrame({"computer": rng.integers(0, 300, n).astype(str), "hour": rng.integers(0, 600, n)})
    hh = hh.drop_duplicates(["computer", "hour"]).reset_index(drop=True)
    feats = ["a", "b", "c"]
    for c in feats:
        hh[c] = (rng.pareto(1.5, len(hh)) * 1e4).astype(np.float32)
    s = hh.sort_values(["computer", "hour"], kind="stable").reset_index(drop=True)
    key = pd.factorize(s["computer"])[0].astype(np.int64) * (1 << 32) + s["hour"].to_numpy(np.int64)
    start = np.searchsorted(key, key - 23, side="left")
    V = s[feats].to_numpy(np.float64)
    cs = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, axis=0)])
    idx = np.arange(len(s))
    old = pd.concat([s, pd.DataFrame((cs[idx + 1] - cs[start]).astype(np.float32), columns=[f"{c}_sum24h" for c in feats]),
                     pd.DataFrame({"active_hours_24h": (idx + 1 - start).astype(np.float32)})], axis=1)
    new = L.add_trailing(hh, feats, 24)
    pd.testing.assert_frame_equal(new, old, check_exact=True)


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
    # the fixture's red team gives a one-day val period; 1.6 leaves the last day of a period unresolved, so use
    # three val days, as in the real split (days 11-13)
    sp = paths.RESULTS / "data_audit" / "lanl_cyber1" / "level3_split.json"
    comp = pd.read_parquet(paths.PROCESSED / "lanl_cyber1" / "v1" / "compromises.parquet")
    sp.write_text(json.dumps({"rule": "test fixture", "train_days": [1, 6], "val_days": [7, 9], "test_days": [10, 58],
                              "test_compromised_hosts": int((comp.day >= 10).sum())}))
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


def test_superseded_level3_records_stay_where_they_were():
    """The 1.5 freeze and records are kept as the historical record and can still be checked."""
    from src.models import seal
    assert seal._record("lanl/logreg") == paths.RESULTS / "models" / "development" / "lanl" / "logreg.json"
    assert seal._record("lanl_1.6/logreg") == L.record_path("logreg")
    assert seal.freeze_path("lanl_1.5").name == "freeze_lanl.json" and seal.freeze_path("lanl").name == "freeze_lanl_1.6.json"
    assert L.model_path("mlp", 11, "1.5").parent.name == "mlp" and L.model_path("mlp", 11, "1.5").parent.parent.name == "lanl"
    with pytest.raises(RuntimeError, match="superseded"):
        L.develop("logreg", cfg_path=L.CONFIG_1_5)

