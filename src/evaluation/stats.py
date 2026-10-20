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
