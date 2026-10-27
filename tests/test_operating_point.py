"""Exploratory analysis E1, ranking versus operating point (docs/protocol/exploratory_analyses.md)."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.evaluation import operating_point as E
from src.evaluation import stats
from src.evaluation.metrics import fpr_at_detection_rate


def _toy(rng, n=4000, blocks=40, families=3):
    """Scores with many ties (as a random forest gives), attack families and blocks."""
    y = (rng.random(n) < 0.2).astype(np.int64)
    fam = np.where(y == 1, rng.integers(0, families, n), -1)
    s = np.round(np.clip(rng.normal(0.2 + 0.25 * y + 0.1 * (fam == 1), 0.2), 0, 1) * 30) / 30
    s[(fam == 2) & (rng.random(n) < 0.5)] = 0.0                  # a family tied with benign at the lowest score
    return s.astype(np.float32), y, fam, rng.integers(0, blocks, n)


def _figures(s, y, fam, blk, thr, w=None, n_blocks=None):
    """E1 figures through the counts path, for block weights w (None: every block once)."""
    F = int(fam.max()) + 1
    cls = np.where(y == 1, fam + 1, 0)
    nb = n_blocks or int(blk.max()) + 1
    u, H, lay = E.seed_counts(s, cls, blk, F + 1, nb)
    w = np.ones(nb) if w is None else w
    jf = int(np.searchsorted(u, np.float32(thr), side="left"))
    return E.measures((H @ w)[None, :], lay, jf, curve=True)


def _brute_point(s, y, alert):
    tp, fp = (alert & (y == 1)).sum(), (alert & (y == 0)).sum()
    return tp / (y == 1).sum(), fp / (y == 0).sum(), tp / max(tp + fp, 1)


def test_figures_match_per_row_definitions():
    rng = np.random.default_rng(7)
    s, y, fam, blk = _toy(rng)
    thr = 0.4
    m = {k: v[0] for k, v in _figures(s, y, fam, blk, thr).items()}
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert m["fpr_at_tpr_0.95"] == pytest.approx(fpr_at_detection_rate(y, s, 0.95), abs=1e-12)
    assert m["prevalence"] == pytest.approx(y.mean()) and m["ap_lift"] == pytest.approx(m["pr_auc"] - y.mean())
    assert tuple(m[f"frozen_{k}"] for k in ("recall", "fpr", "precision")) == pytest.approx(_brute_point(s, y, s >= thr))
    vals = np.unique(s)

    def split_ties(x, target):
        """Alert on rows of x above the cut-off value c and on a share q of the rows at c, so that the alerted
        count of x is exactly target (ties split in proportion)."""
        j = next(i for i, v in enumerate(vals) if (x >= v).sum() <= target) if (x >= vals[-1]).sum() <= target \
            else len(vals)
        c = vals[j - 1]
        return c, (target - (x > c).sum()) / (x == c).sum()

    def interp_point(c, q):
        return [((s[y == k] > c).sum() + q * (s[y == k] == c).sum()) / (y == k).sum() for k in (1, 0)]

    for a in E.FPR_LEVELS:
        c, q = split_ties(s[y == 0], a * (y == 0).sum())
        rec, fpr = interp_point(c, q)
        assert m[f"oracle_fpr_{a:g}_recall"] == pytest.approx(rec, abs=1e-12)
        assert m[f"oracle_fpr_{a:g}_fpr"] == pytest.approx(a, abs=1e-12) == fpr
        assert np.interp(a, *roc_curve(y, s, drop_intermediate=False)[:2]) == pytest.approx(rec, abs=1e-12)
    for b in E.BUDGETS:
        c, q = split_ties(s, b * len(s))
        assert m[f"budget_{b:g}_recall"] == pytest.approx(interp_point(c, q)[0], abs=1e-12)
        assert m[f"budget_{b:g}_alert_share"] == pytest.approx(b, abs=1e-12)
    sa = s[y == 1]
    c = max(v for v in vals if (sa >= v).sum() >= 0.95 * len(sa))
    q = (0.95 * len(sa) - (sa > c).sum()) / (sa == c).sum()
    assert m["fpr_at_tpr_0.95_interpolated"] == pytest.approx(interp_point(c, q)[1], abs=1e-12)
    assert m["fpr_at_tpr_0.95_interpolated"] <= m["fpr_at_tpr_0.95"]
    low = s[y == 0].min()
    assert m["undetectable_share"] == pytest.approx((s[y == 1] <= low).mean())
    assert m["attack_share_at_lowest_score"] == pytest.approx((s[y == 1] == vals[0]).mean())
    for f in range(3):
        keep = (y == 0) | (fam == f)
        assert m["family_roc_auc"][f] == pytest.approx(roc_auc_score(y[keep], s[keep]), abs=1e-12)
        assert m["family_frozen_recall"][f] == pytest.approx((s[fam == f] >= thr).mean())
        assert m["family_undetectable_share"][f] == pytest.approx((s[fam == f] <= low).mean())
    assert m["family_undetectable_share"][2] > 0.4                 # the tied family is found as such
    assert np.all(np.diff(m["curve_recall"]) >= 0) and m["curve_recall"][-1] == 1.0


def test_a_replicate_equals_the_resampled_rows():
    """A bootstrap replicate from counts equals the figures of the rows of the drawn blocks, repeated."""
    rng = np.random.default_rng(3)
    s, y, fam, blk = _toy(rng)
    W = E.draws(40, B=3, seed=11)
    for w in W:
        rep = {k: v[0] for k, v in _figures(s, y, fam, blk, 0.4, w=w).items()}
        idx = np.concatenate([np.repeat(np.flatnonzero(blk == b), int(k)) for b, k in enumerate(w)])
        direct = {k: v[0] for k, v in _figures(s[idx], y[idx], fam[idx], blk[idx], 0.4, n_blocks=40).items()}
        for k in rep:
            np.testing.assert_allclose(rep[k], direct[k], rtol=0, atol=1e-12, err_msg=k)
        assert rep["pr_auc"] == pytest.approx(average_precision_score(y[idx], s[idx]), abs=1e-12)


def test_draws_are_the_confirmatory_draws():
    """E1 resamples exactly the blocks the confirmatory bootstrap of the same evaluation set resampled."""
    rng = np.random.default_rng(5)
    counts = rng.integers(0, 50, (2, 30, 2, 3))
    W = E.draws(30, B=40)
    mine = stats.macro_f1_from_counts((W @ counts.transpose(1, 0, 2, 3).reshape(30, -1)).reshape(40, 2, 2, 3)).mean(axis=1)
    theirs = stats._replicate_f1(counts, np.random.default_rng(stats.BOOTSTRAP_SEED), 40, batch=7)
    np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-12)


def e1_end_to_end(fr):
    """Called from tests/test_scoring.py once the synthetic world is scored (one fixture build for both)."""
    from src.evaluation import confirmatory as C
    from src.models import develop as D
    from src.models import score as S
    from src.utils import paths

    def quiet(*a, **k):
        pass

    E.run(log=quiet)
    prop = fr["proposed"]
    for e, sp in E.SETS:
        x = json.loads(E.result_path(e, sp).read_text())
        g, gsha = C.load_groups(e)
        al, _ = C.load_alerts(e, sp, prop, gsha, fr, x["freeze_sha256"])
        cm = C.metrics(C.block_counts(g, al, np.ones(len(g), bool)))
        m = x["measures"]
        assert m["frozen_recall"]["estimate"] == pytest.approx(cm["attack_recall"], abs=1e-12)
        assert m["frozen_fpr"]["estimate"] == pytest.approx(cm["fpr"], abs=1e-12)
        assert m["prevalence"]["estimate"] == pytest.approx(C.composition(g, np.ones(len(g), bool))["attack_share"])
        fr_ = C.family_recall(g, al, np.ones(len(g), bool))
        for f, fm in x["family_measures"].items():
            assert fm["frozen_recall"]["estimate"] == pytest.approx(fr_[f], abs=1e-12)
        units = [json.loads(S.unit_path(e, sp, prop, s).read_text()) for s in D.seeds()]
        assert m["pr_auc"]["per_seed"] == pytest.approx([u["pr_auc"] for u in units], abs=1e-6)
        assert x["bootstrap"]["B"] == stats.BOOTSTRAP_B and len(m["pr_auc"]["per_seed"]) == len(D.seeds())
        assert m["pr_auc"]["ci_low"] <= m["pr_auc"]["ci_high"]
        assert m["pr_auc"]["replicates_used"] >= 0.99 * stats.BOOTSTRAP_B     # tiny world: a few replicates draw no attack
        assert x["plan_sha256"] and x["checks"]["groups"] == "reproduced"
    summary = (E.out_dir() / "summary.md").read_text()
    from tests.test_scoring import _script
    _script("explore").status()                                  # the notebook's progress table
    assert "Oracle" in summary and "Not yet run" not in summary
    stamp = {p: p.stat().st_mtime_ns for p in E.out_dir().glob("*.json")}
    E.run(log=quiet)                                            # written once: a second run skips
    assert stamp == {p: p.stat().st_mtime_ns for p in E.out_dir().glob("*.json")}
    # a changed scores file stops it before anything is written
    e, sp = E.SETS[-1]
    E.result_path(e, sp).unlink()
    p = S.scores_path(e, sp, prop, D.seeds()[0])
    df = pd.read_parquet(p)
    df.loc[0, "score"] = 1 - df.loc[0, "score"]
    df.to_parquet(p, index=False)
    with pytest.raises(AssertionError, match="SHA-256"):
        E.run([(e, sp)], log=quiet)
    assert not E.result_path(e, sp).exists()
    assert paths.RESULTS in E.out_dir().parents
