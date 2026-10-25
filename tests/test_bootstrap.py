import numpy as np
from sklearn.metrics import f1_score

from src.evaluation import stats


def _counts(y, p, blk):
    return stats.block_counts(y, p, blk, [0, 1])[0]


def test_counts_reproduce_sklearn_macro_f1():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = np.where(rng.random(5000) < 0.9, y, 1 - y)
    blk = np.repeat(np.arange(50), 100)
    c = _counts(y, p, blk)
    assert np.isclose(stats.macro_f1_from_counts(c.sum(axis=0)), f1_score(y, p, average="macro"))


def test_bootstrap_drop_and_decisions():
    rng = np.random.default_rng(1)
    def seeds(acc, n=20000, nb=200):
        out = []
        for s in range(3):
            y = rng.integers(0, 2, n)
            p = np.where(rng.random(n) < acc, y, 1 - y)
            out.append(_counts(y, p, np.repeat(np.arange(nb), n // nb)))
        return np.stack(out)
    src, tgt = seeds(0.97), seeds(0.85)
    r = stats.block_bootstrap(tgt, src, B=2000)
    assert 0.10 < r["estimate"] < 0.14 and r["ci_low"] < r["estimate"] < r["ci_high"]
    assert stats.three_outcome(r["ci_low"], r["ci_high"], 0.05, "below") == "not supported"
    assert stats.three_outcome(0.01, 0.04, 0.05, "below") == "supported"
    assert stats.three_outcome(0.03, 0.07, 0.05, "below") == "inconclusive"
    assert stats.one_sided_p(r["replicates"], 0.05, "below") > 0.5


def test_holm():
    out = stats.holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
    assert out["a"][1] and out["b"][1] and out["c"][1]
    out = stats.holm({"a": 0.001, "b": 0.03, "c": 0.04}, alpha=0.05)
    assert out["a"][1] and not out["b"][1] and not out["c"][1]
