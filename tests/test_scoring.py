"""One-time scoring of the frozen models and the confirmatory analysis (ADR-0007; preregistration §6, §6b, §7)."""
import functools
import importlib.util
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import f1_score

from src.evaluation import confirmatory as C
from src.evaluation import stats
from src.utils import paths
from tests.synthetic import CIC_FEATURES, write_cic, write_cse, write_unsw
from tests.test_lanl import make_lanl
from tests.test_models import TINY

CANDS = ("logreg", "random_forest", "xgboost", "mlp")          # tree, linear and the neural path, quickly


def QUIET(*a, **k):
    pass


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script", paths.REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def world(drive, declared, tmp_path, monkeypatch):
    """Synthetic CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15 and LANL, preprocessed, with every S7 record, the Level 1-2
    candidates developed and frozen, the Level 3 candidates developed under the 1.6 rules and frozen."""
    from src.data import harmonise, lanl, near
    from src.data.pipeline import preprocess
    from src.models import develop, seal
    from src.models import lanl_develop as L
    write_cic(drive / "data" / "raw")
    write_cse(drive / "data" / "interim")
    write_unsw(drive / "data" / "interim")
    for ds in ("cicids2017_improved", "csecicids2018_improved", "unsw_nb15"):
        preprocess(ds, log=QUIET)
    mdir = tmp_path / "mapping"
    mdir.mkdir()
    shutil.copy(harmonise.MAPPING_DIR / "cic_to_unsw_mapping.yaml", mdir)
    (mdir / "cic_shared_features.yaml").write_text(yaml.safe_dump({"features": CIC_FEATURES}))
    monkeypatch.setattr(harmonise, "MAPPING_DIR", mdir)
    monkeypatch.setattr(near, "WITHIN", ["cicids2017_improved"])
    near._checked.clear()
    _script("preprocess").cross_overlap()
    near.build("cicids2017_improved", log=QUIET)
    monkeypatch.setattr(paths, "MODELS", drive / "models")
    monkeypatch.setattr(paths, "SCORES", drive / "scores")
    monkeypatch.setenv("PHD_ALLOW_CPU_NEURAL", "1")
    tiny = {**TINY, "candidates": {c: TINY["candidates"][c] for c in CANDS}}
    cfg = tmp_path / "models.yaml"
    cfg.write_text(yaml.safe_dump(tiny, sort_keys=False))
    monkeypatch.setattr(develop, "MODELS_CONFIG", cfg)
    for space in tiny["feature_spaces"]:
        for c in CANDS:
            develop.develop(space, c, cfg_path=cfg, log=QUIET)
    # Level 3, as in tests/test_lanl_models.py, with three val days
    make_lanl(drive / "data" / "raw")
    lanl.preprocess_lanl(log=QUIET)
    lc = yaml.safe_load(L.CONFIG.read_text())
    lc["candidates"] = {"logreg": {"family": "linear", "fixed": {"max_iter": 200},
                                   "grid": {"C": [1.0], "class_weight": [None, "balanced"]}},
                        "xgboost": {"family": "tree", "fixed": {"n_estimators": 10, "early_stopping_rounds": 3,
                                    "tree_method": "hist", "eval_metric": "logloss", "n_jobs": 1},
                                    "grid": {"max_depth": [3], "learning_rate": [0.3], "scale_pos_weight": [1]}}}
    lp = tmp_path / "models_lanl_1.6.yaml"
    lp.write_text(yaml.safe_dump(lc, sort_keys=False))
    monkeypatch.setattr(L, "CONFIG", lp)
    comp = pd.read_parquet(paths.PROCESSED / "lanl_cyber1" / "v1" / "compromises.parquet")
    (paths.RESULTS / "data_audit" / "lanl_cyber1" / "level3_split.json").write_text(json.dumps(
        {"rule": "test fixture", "train_days": [1, 6], "val_days": [7, 9], "test_days": [10, 58],
         "test_compromised_hosts": int((comp.day >= 10).sum())}))
    for c in lc["candidates"]:
        L.develop(c, cfg_path=lp, log=QUIET)
    _script("develop").lanl_status()
    assert "beside the superseded 1.5 rules" in (paths.RESULTS / "models" / "development" / "lanl_1.6_summary.md").read_text()
    # no git in the test tree: everything else of the seal is checked
    monkeypatch.setattr(seal, "require_frozen", functools.partial(seal.require_frozen, require_git=False))
    return {"cfg": cfg, "lanl_cfg": lp}


def _rows(ds, split, partitions):
    from src.data import datasets
    return datasets.load(ds, split=split, partitions=partitions)


def test_scoring_end_to_end(world):
    from src.data import harmonise, near
    from src.models import develop, seal
    from src.models import lanl_develop as L
    from src.models import score as S
    from src.models.candidates import Candidate

    # sealed until the freeze, for both levels
    with pytest.raises(seal.Sealed):
        S.score_level12(log=QUIET)
    with pytest.raises(seal.Sealed):
        S.score_lanl(log=QUIET)
    fr = seal.freeze(cfg_path=world["cfg"], log=QUIET)
    fr3 = seal.freeze_lanl(cfg_path=world["lanl_cfg"], log=QUIET)
    assert fr3["rules_version"] == "1.6" and S.frozen_candidates(fr3) == ["logreg", "xgboost"]
    S.score_level12(log=QUIET)
    S.score_lanl(log=QUIET)
    assert not S.level12_todo(fr)
    stamp = {p: p.stat().st_mtime_ns for p in (paths.RESULTS / "confirmatory").rglob("seed*.json")}
    S.score_level12(log=QUIET)                                   # resumes: nothing scored twice
    S.score_lanl(log=QUIET)
    assert stamp == {p: p.stat().st_mtime_ns for p in (paths.RESULTS / "confirmatory").rglob("seed*.json")}

    # the groups reproduce the data: per-row scores of the proposed detector give the same macro-F1 as the counts
    prop = fr["proposed"]
    l12 = C.level12(log=QUIET)
    seeds = develop.seeds()

    def direct_f1(eval_id, space, mask_col=None):
        es = S.EVAL_SETS[eval_id]
        X, meta = _rows(es["dataset"], es["split"], es["partitions"])
        side = pd.read_parquet(near.sidecar_path(es["dataset"], es["against"])).set_index("_key")
        keep = np.ones(len(meta), bool) if mask_col is None else ~side.loc[meta["_key"], mask_col].to_numpy()
        rec = json.loads(develop.record_path(space, prop).read_text())
        f1s = []
        for s in seeds:
            sc = pd.read_parquet(S.scores_path(eval_id, space, prop, s)).set_index("_key")["score"]
            p = sc.loc[meta["_key"]].to_numpy() >= rec["final"][str(s)]["threshold"]
            f1s.append(f1_score(meta["_y"].to_numpy()[keep], p[keep].astype(int), average="macro", labels=[0, 1],
                                zero_division=0))
        return float(np.mean(f1s))

    h1a = l12["claims"]["H1a"]
    assert h1a["estimate"] == pytest.approx(direct_f1("cicids2017_test", "cic"), abs=1e-12)
    h1b = l12["claims"]["H1b"]
    assert h1b["target_figures"]["macro_f1"] == pytest.approx(direct_f1("csecicids2018", "cic"), abs=1e-12)
    assert h1b["estimate"] == pytest.approx(h1b["source_figures"]["macro_f1"] - h1b["target_figures"]["macro_f1"])
    h1c = l12["claims"]["H1c"]
    assert h1c["target_figures"]["macro_f1"] == pytest.approx(direct_f1("unsw_nb15", "unsw_canonical"), abs=1e-12)
    # S7 subset from the groups = S7 subset from the sidecar
    g, _ = C.load_groups("csecicids2018")
    al, _ = C.load_alerts("csecicids2018", "cic", prop, *_load_ids("csecicids2018", fr))
    c7 = C.block_counts(g, al, ~g["seen_sig3"].to_numpy())
    assert C.metrics(c7)["macro_f1"] == pytest.approx(direct_f1("csecicids2018", "cic", "seen_sig3"), abs=1e-12)

    # a baseline unit equals a direct prediction with its frozen model
    X, meta = _rows("unsw_nb15", "external", ("external",))
    rec = json.loads(develop.record_path("unsw_canonical", "logreg").read_text())
    f = rec["final"][str(seeds[0])]
    m = Candidate.load(paths.MODELS / f["model_file"])
    p = m.predict_proba(harmonise.to_canonical_unsw(X, "unsw").to_numpy(np.float32))
    u = json.loads(S.unit_path("unsw_nb15", "unsw_canonical", "logreg", seeds[0]).read_text())
    assert u["alerted_rows"] == int((p >= f["threshold"]).sum()) and u["rows"] == len(meta)
    assert not S.scores_path("unsw_nb15", "unsw_canonical", "logreg", seeds[0]).exists()   # proposed only

    # Level 3 units
    u3 = json.loads(S.lanl_unit_path(fr3["proposed"], seeds[0]).read_text())
    sp = L.split()
    assert u3["hosts"] == sp["test_compromised_hosts"] and u3["rules_version"] == "1.6"
    assert u3["hosts_alerted"] == sum(h["lead_hours"] is not None for h in u3["per_host"])
    assert u3["early_alert_rate_test_days_only"] <= u3["early_alert_rate"]
    for h in u3["per_host"]:
        assert h["day"] >= sp["test_days"][0] and (h["lead_hours"] is None or 0 <= h["lead_hours"] < 24)
        assert h["window_on_previous_day"] == (h["compromise_time"] - 86400 < (sp["test_days"][0] - 1) * 86400)
        assert h["lead_hours_test_days_only"] is None or h["lead_hours"] is not None
    rb = json.loads((S.out_dir() / "level3" / "random_baseline.json").read_text())
    assert rb["val_fa_rate"] <= 0.01 + 1e-9 and rb["early_alert_rate_q025"] <= rb["expected_early_alert_rate"] \
        <= rb["early_alert_rate_q975"]

    # the analysis, once; a second run reproduces it; a changed record stops it
    out = C.analyse(log=QUIET)
    assert list(out["decisions"]["claims"]) == list(C.FAMILY)
    assert out["decisions"]["pending"] == ["H4-healthcare", "H4-IoT", "H4-government"]
    for h in ("H1a", "H1b", "H1c", "H2"):
        assert out["decisions"]["claims"][h]["decision"] in ("supported", "not supported", "inconclusive")
    assert "| H2 |" in (S.out_dir() / "decisions.md").read_text()
    assert set(out["level1_2"]["comparisons"]) == {"cic", "unsw_canonical"}
    assert out["level1_2"]["comparisons"]["cic"]["pairs"] == 2 * len(seeds)
    again = C.analyse(log=QUIET)
    assert again["generated_utc"] == out["generated_utc"]
    _script("score").status()                                    # the notebook's progress table
    d = json.loads((S.out_dir() / "decisions.json").read_text())
    d["decisions"]["claims"]["H1a"]["estimate"] += 1e-9
    (S.out_dir() / "decisions.json").write_text(json.dumps(d))
    with pytest.raises(AssertionError, match="second run gives different numbers"):
        C.analyse(log=QUIET)


def _load_ids(eval_id, fr):
    from src.models import seal
    from src.utils.io import sha256_file
    return C.load_groups(eval_id)[1], fr, sha256_file(seal.freeze_path("l12"))


# ---- the analysis rules on small hand-made inputs -------------------------------------------------------------
def test_counts_from_groups_match_sklearn():
    rng = np.random.default_rng(1)
    n = 5000
    df = pd.DataFrame({"block": rng.integers(0, 40, n), "label": rng.choice(["BENIGN", "DoS", "Scan"], n),
                       "seen_sig3": rng.random(n) < .2, "seen_sig2": rng.random(n) < .4})
    df["y"] = (df["label"] != "BENIGN").astype(int)
    pred = np.stack([(rng.random(n) < np.where(df["y"] == 1, .9, .05)) for _ in range(3)])
    key = ["block", "label", "y", "seen_sig3", "seen_sig2"]
    gid = df.groupby(key, sort=True).ngroup().to_numpy()
    g = df.groupby(key, sort=True).size().rename("n").reset_index()
    alerts = np.stack([np.bincount(gid, p, len(g)).astype(np.int64) for p in pred])
    for mask_rows, mask_g in ((np.ones(n, bool), np.ones(len(g), bool)),
                              (~df["seen_sig2"].to_numpy(), ~g["seen_sig2"].to_numpy())):
        c = C.block_counts(g, alerts, mask_g)
        want = np.mean([f1_score(df["y"][mask_rows], p[mask_rows], average="macro") for p in pred])
        assert C.metrics(c)["macro_f1"] == pytest.approx(want, abs=1e-12)
        bs = stats.block_bootstrap(c, B=200)
        assert bs["estimate"] == pytest.approx(want, abs=1e-12)


def test_holm_decisions_with_h4_pending():
    def cl(est, lo, hi, ps, pa, margin, claim):
        return {"estimate": est, "ci_low": lo, "ci_high": hi, "p_support": ps, "p_against": pa, "margin": margin,
                "claim": claim, "unadjusted_outcome": stats.three_outcome(lo, hi, margin, claim)}
    l12 = {"claims": {"H1a": cl(.99, .985, .995, 1e-4, 1.0, .95, "at_least"),
                      "H1b": cl(.30, .25, .35, 1.0, 1e-4, .05, "below"),
                      "H1c": cl(.05, .01, .09, .40, .45, .05, "below")}}
    l3 = {"claim": {**cl(.6, .52, .7, .003, 1.0, .5, "at_least"), "above_random_baseline": True}}
    d = C.decide(l12, l3)["claims"]
    assert d["H1a"]["decision"] == "supported" and d["H1a"]["final"]
    assert d["H1b"]["decision"] == "not supported" and d["H1b"]["final"]
    assert d["H1c"]["decision"] == "inconclusive" and not d["H1c"]["final"]
    assert d["H2"]["decision"] == "supported"                       # 0.003 x 6 remaining < 0.025
    assert d["H4-IoT"]["decision"].startswith("pending")
    l3["claim"]["above_random_baseline"] = False
    assert C.decide(l12, l3)["claims"]["H2"]["decision"] == "inconclusive"
    l3["claim"]["p_support"] = 0.004                                # Holm: 7 x 0.004 > 0.025 if ranked first
    l12["claims"]["H1a"]["p_support"] = 0.01
    l3["claim"]["above_random_baseline"] = True
    d = C.decide(l12, l3)["claims"]
    assert d["H2"]["holm_p_support"] == pytest.approx(7 * 0.004) and d["H2"]["decision"] == "inconclusive"


def test_account_bootstrap_resamples_accounts_not_hosts():
    alerted = np.array([[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 1]], float)    # 6 hosts x 2 seeds
    acc = ["a", "a", "a", "b", "b", "c"]
    bs = C._account_bootstrap(alerted, acc, B=4000)
    assert bs["estimate"] == pytest.approx((3 / 6 + 4 / 6) / 2) and bs["accounts"] == 3
    # a replicate draws whole accounts: with a, b, c drawn once each it is the estimate; all-a gives 1, all-b 0
    assert bs["ci_high"] == pytest.approx(1.0) and bs["ci_low"] == pytest.approx(0.0)
    one = C._account_bootstrap(alerted, ["x"] * 6, B=100)
    assert one["ci_low"] == one["ci_high"] == pytest.approx(one["estimate"])


def test_random_baseline_matches_simulation():
    from src.models import score as S
    rng = np.random.default_rng(5)
    hosts = [f"H{i}" for i in range(30)]
    rows = []
    for d in (1, 2):                                                # val days: no compromise
        for h in hosts:
            for hr in rng.choice(24, rng.integers(1, 10), replace=False):
                rows.append((h, (d - 1) * 24 + hr, d, 0, True))
    val = pd.DataFrame(rows, columns=["computer", "hour", "day", "_y", "_eligible_day"])
    t_comp = pd.Series({h: 4 * 86400 + 3600 * (i % 20) for i, h in enumerate(hosts[:12])}, dtype=float)
    rows = []
    for h in hosts:
        for hr in range(48, 96):
            if rng.random() < .3:
                t = t_comp.get(h, np.inf)
                end = (hr + 1) * 3600
                if end > t:
                    continue
                rows.append((h, hr, hr // 24 + 1, int(0 <= t - end < 86400), not (t - 86400 < (hr // 24 + 1) * 86400)))
    test = pd.DataFrame(rows, columns=["computer", "hour", "day", "_y", "_eligible_day"])
    rb = S.random_baseline({"val": val, "frame": test, "hosts": t_comp, "days": (3, 4)}, 0.05)
    assert rb["val_fa_rate"] == pytest.approx(0.05, abs=1e-9)
    sims = []
    for _ in range(4000):
        s = rng.random(len(test))
        hit = test[(test["_y"] == 1) & (s >= rb["threshold"])]["computer"].unique()
        sims.append(len(hit) / len(t_comp))
    assert np.mean(sims) == pytest.approx(rb["expected_early_alert_rate"], abs=0.01)
    lo, hi = np.quantile(sims, [0.025, 0.975])
    assert rb["early_alert_rate_q025"] <= lo + 1 / len(t_comp) and hi <= rb["early_alert_rate_q975"] + 1 / len(t_comp)
