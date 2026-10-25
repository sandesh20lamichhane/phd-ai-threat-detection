"""Statistical comparison of models across seeds or datasets (Demsar 2006 procedure).

Protocol: at least 5 seeds; report mean and standard deviation; Wilcoxon signed-rank for
two models, Friedman with Nemenyi post-hoc for more than two.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats


def summarise(scores):
    """scores: dict model -> list of per-seed values. Returns a mean/std/n table."""
    rows = [{"model": m, "mean": np.mean(v), "std": np.std(v, ddof=1) if len(v) > 1 else 0.0, "n": len(v)}
            for m, v in scores.items()]
    return pd.DataFrame(rows).set_index("model")


def wilcoxon(a, b):
    res = stats.wilcoxon(a, b, zero_method="zsplit")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def friedman_nemenyi(table, alpha=0.05):
    """table: DataFrame, rows = blocks (seeds or datasets), columns = models, higher is better.

    Returns Friedman statistic and p-value, average ranks, the Nemenyi critical difference,
    and the model pairs whose rank difference exceeds it.
    """
    table = table.dropna()
    n, k = table.shape
    stat, p = stats.friedmanchisquare(*[table[c].values for c in table.columns])
    ranks = table.rank(axis=1, ascending=False).mean()
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    diffs = [(a, b, abs(ranks[a] - ranks[b])) for a, b in itertools.combinations(table.columns, 2)]
    return {"friedman_stat": float(stat), "p_value": float(p), "avg_rank": ranks.sort_values().to_dict(),
            "critical_difference": float(cd),
            "significant_pairs": [(a, b, float(d)) for a, b, d in diffs if d > cd]}


# ---- confirmatory analysis of protocol v1.0 (docs/protocol/preregistration.md §6) --------------------------
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20261025


def block_counts(y_true, y_pred, block, classes):
    """Per-block TP, FP, FN for every class: array (n_blocks, n_classes, 3), plus the block ids."""
    y_true, y_pred, block = map(np.asarray, (y_true, y_pred, block))
    ids, b = np.unique(block, return_inverse=True)
    out = np.zeros((len(ids), len(classes), 3), dtype=np.int64)
    for k, c in enumerate(classes):
        t, p = y_true == c, y_pred == c
        out[:, k, 0] = np.bincount(b, weights=t & p, minlength=len(ids))
        out[:, k, 1] = np.bincount(b, weights=~t & p, minlength=len(ids))
        out[:, k, 2] = np.bincount(b, weights=t & ~p, minlength=len(ids))
    return out, ids


def macro_f1_from_counts(c):
    """c: (..., n_classes, 3) summed counts -> macro-F1 over classes (classes absent from truth and
    predictions count as 0, as sklearn's zero_division=0)."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros(den.shape, dtype=float), where=den > 0)
    return f1.mean(axis=-1)


def _replicate_f1(counts, rng, B, batch=500):
    """counts: (n_seeds, n_blocks, K, 3), same blocks for every seed. Returns (B,) seed-mean macro-F1."""
    s, nb = counts.shape[:2]
    flat = counts.transpose(1, 0, 2, 3).reshape(nb, -1).astype(float)
    out = np.empty(B)
    for i in range(0, B, batch):
        m = min(batch, B - i)
        w = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(m)]).astype(float)
        summed = (w @ flat).reshape(m, s, *counts.shape[2:])
        out[i:i + m] = macro_f1_from_counts(summed).mean(axis=1)
    return out


def block_bootstrap(target, source=None, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Group-aware bootstrap of the seed-mean macro-F1, or of the drop F1(source) - F1(target).

    target, source: arrays (n_seeds, n_blocks, n_classes, 3) from `block_counts` (stack the seeds; every seed
    must use the same blocks). Blocks are resampled with replacement, independently in source and target.
    Returns the point estimate, the percentile CI and the replicates.
    """
    rng = np.random.default_rng(seed)
    target = np.asarray(target)
    f_t = macro_f1_from_counts(target.sum(axis=1)).mean()
    rep_t = _replicate_f1(target, rng, B)
    if source is None:
        est, reps = f_t, rep_t
    else:
        source = np.asarray(source)
        f_s = macro_f1_from_counts(source.sum(axis=1)).mean()
        est, reps = f_s - f_t, _replicate_f1(source, rng, B) - rep_t
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(est), "ci_low": float(lo), "ci_high": float(hi), "B": B, "seed": seed,
            "replicates": reps}


def three_outcome(ci_low, ci_high, margin, claim):
    """Decision rules of docs/protocol/hypotheses.md.

    claim "below": supported if the whole CI is below the margin (e.g. drop < X).
    claim "at_least": supported if the whole CI is at or above the margin (e.g. F1 >= 0.95).
    """
    if claim == "below":
        return "supported" if ci_high < margin else "not supported" if ci_low > margin else "inconclusive"
    if claim == "at_least":
        return "supported" if ci_low >= margin else "not supported" if ci_high < margin else "inconclusive"
    raise ValueError(claim)


def one_sided_p(replicates, margin, claim):
    """Bootstrap p-value for the claim: share of replicates on the wrong side of the margin."""
    r = np.asarray(replicates)
    bad = (r >= margin) if claim == "below" else (r < margin)
    return float((bad.sum() + 1) / (len(r) + 1))


def holm(pvalues, alpha=0.025):
    """Holm step-down. pvalues: dict name -> p. Returns dict name -> (adjusted p, rejected)."""
    names = sorted(pvalues, key=pvalues.get)
    m, running, out, stop = len(names), 0.0, {}, False
    for i, n in enumerate(names):
        adj = min(1.0, (m - i) * pvalues[n])
        running = max(running, adj)
        stop = stop or running > alpha
        out[n] = (running, not stop)
    return out
