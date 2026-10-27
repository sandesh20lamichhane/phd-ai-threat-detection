"""Exploratory analysis E1: ranking versus operating point of the frozen proposed detector
(docs/protocol/exploratory_analyses.md). Post hoc and outside the Holm family: no p-values, no decision.

Reads the per-row scores written once by src/models/score.py and the labels, attack families and blocks of the same
rows, checks that together they reproduce the confirmatory records exactly, and writes

    results/exploratory/operating_point/<eval>__<space>.json   per-seed values, seed means, 95 % CIs, the checks
    results/exploratory/operating_point/summary.md             the tables

Every figure is a function of the row counts per 300 s block, class and distinct score value, so each bootstrap
replicate is computed exactly (no binning of scores). A result file is written once; a second run skips it.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from ..data import datasets, near
from ..models import develop as D
from ..models import score as S
from ..models import seal
from ..utils import paths
from ..utils.io import sha256_file, write_json
from ..utils.logging import run_info
from . import stats
from .metrics import fpr_at_detection_rate, pr_auc

SETS = (("cicids2017_test", "cic"), ("cicids2017_test", "unsw_canonical"), ("unsw_nb15", "unsw_canonical"),
        ("csecicids2018", "cic"))
FPR_LEVELS = (0.001, 0.01, 0.05)
BUDGETS = (0.01, 0.05)
CURVE_FPR = tuple(m * 10.0 ** e for e in range(-4, 0) for m in (1, 2, 5)) + (1.0,)
UNIT_TOL = 1e-6           # float32 scores against the float64 figures of the unit records
CELLS = 2e7               # counts held per bootstrap batch (memory)
PLAN = paths.REPO / "docs" / "protocol" / "exploratory_analyses.md"


def out_dir():
    return paths.RESULTS / "exploratory" / "operating_point"


def result_path(eval_id, space):
    return out_dir() / f"{eval_id}__{space}.json"


# ---- the rows -------------------------------------------------------------------------------------------------
def rows(eval_id, log=print):
    """Key, binary label, family code, block and S7 flags of every scored row of an evaluation set, in the order
    src/models/score.py scored them."""
    es = S.EVAL_SETS[eval_id]
    ds, against = es["dataset"], es["against"]
    labels, keys, ys, codes, blocks = {}, [], [], [], []
    for meta in datasets.iter_meta(ds, es["split"], es["partitions"]):
        lab = meta["_label"].astype(str).to_numpy()
        for v in pd.unique(lab):
            labels.setdefault(v, len(labels))
        keys.append(meta["_key"].to_numpy())
        ys.append(meta["_y"].to_numpy(np.int8))
        codes.append(pd.Series(lab).map(labels).to_numpy(np.int16))
        blocks.append(meta["_block"].to_numpy(np.int64))
    key = np.concatenate(keys)
    near.require_sidecar(ds, against)
    side = pd.read_parquet(near.sidecar_path(ds, against), columns=["_key", "seen_sig3", "seen_sig2"])
    pos = pd.Index(side["_key"].to_numpy()).get_indexer(key)
    if (pos < 0).any():
        raise AssertionError(f"S7 sidecar of {ds} does not cover every evaluation row")
    r = {"key": key, "y": np.concatenate(ys), "code": np.concatenate(codes), "block": np.concatenate(blocks),
         "labels": list(labels), "s3": side["seen_sig3"].to_numpy(bool)[pos], "s2": side["seen_sig2"].to_numpy(bool)[pos]}
    log(f"[{eval_id}] {len(key):,} rows read")
    return r


def _encode(block, code, y, s3, s2):
    block, code, y, s3, s2 = (np.asarray(v, dtype=np.int64) for v in (block, code, y, s3, s2))
    return (((block + 1) * S.LABEL_CODES + code) * 2 + y) * 4 + s3 * 2 + s2


def group_index(r, groups):
    """Index into the confirmatory groups file of every row; the rows must reproduce every group's count."""
    code = groups["label"].map({v: i for i, v in enumerate(r["labels"])})
    if code.isna().any():
        raise AssertionError("the groups file has a label family the rows read now do not")
    kg = _encode(groups["block"], code, groups["y"], groups["seen_sig3"], groups["seen_sig2"])
    order = np.argsort(kg, kind="stable")
    ks = kg[order]
    if len(np.unique(ks)) != len(ks):
        raise AssertionError("duplicate groups in the groups file")
    kr = _encode(r["block"], r["code"], r["y"], r["s3"], r["s2"])
    pos = np.minimum(np.searchsorted(ks, kr), len(ks) - 1)
    if not (ks[pos] == kr).all():
        raise AssertionError("rows read now fall in a group the confirmatory groups file does not have")
    gi = order[pos]
    if not np.array_equal(np.bincount(gi, minlength=len(groups)), groups["n"].to_numpy(np.int64)):
        raise AssertionError("the rows read now do not reproduce the row counts of the confirmatory groups file")
    return gi


def draws(n_blocks, B=stats.BOOTSTRAP_B, seed=stats.BOOTSTRAP_SEED):
    """(B, n_blocks) multiplicities of the resampled blocks, drawn exactly as stats.block_bootstrap draws them."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
                     for _ in range(B)]).astype(np.float64)


# ---- figures from counts --------------------------------------------------------------------------------------
def _rev(x):
    """Rows scored >= u_j, for j = 0..U (j = U: alert on nothing)."""
    c = np.cumsum(x[..., ::-1], axis=-1)[..., ::-1]
    return np.concatenate([c, np.zeros(c.shape[:-1] + (1,), c.dtype)], axis=-1)


def _at(a, j):
    """a[i, ..., j[i]] for every replicate i."""
    idx = np.broadcast_to(j.reshape((-1,) + (1,) * (a.ndim - 1)), a.shape[:-1] + (1,))
    return np.take_along_axis(a, idx, axis=-1)[..., 0]


def _div(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def _q(a, b):
    """a / b for count arrays whose numerator is 0 wherever the denominator is (0 / 0 = NaN), without a copy."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b


class Layout:
    """Which (class, distinct score value) pairs occur. Class 0 is benign, class 1 + f attack family f. The counts
    matrix has one row per occurring pair, ordered by class and then by value, so each class is one slice."""

    def __init__(self, U, cls, j, n_classes):
        self.U, self.j = U, j
        edges = np.searchsorted(cls, np.arange(n_classes + 1))
        self.slices = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def seed_counts(score, cls, blk, n_classes, n_blocks):
    """Distinct score values u, the sparse (pairs, n_blocks) matrix of rows per occurring (class, value) pair and
    block, and its Layout."""
    u, j = np.unique(score, return_inverse=True)
    U = len(u)
    width = n_classes * U
    k, c = np.unique(blk.astype(np.int64) * width + cls.astype(np.int64) * U + j.ravel(), return_counts=True)
    pair, col = k % width, k // width
    pairs, row = np.unique(pair, return_inverse=True)
    H = sparse.csr_matrix((c.astype(np.float64), (row.ravel(), col)), shape=(len(pairs), n_blocks))
    return u, H, Layout(U, pairs // U, pairs % U, n_classes)


def measures(R, lay, jf, curve=False):
    """Every E1 figure from counts. R (m, pairs): rows of each occurring (class, value) pair in m replicates (or
    m = 1, every block once); jf: index of the frozen threshold (alert when the value index j >= jf).
    Returns name -> (m,) array, or (m, F) for names starting with "family_"."""
    m, U = len(R), lay.U
    N, at = np.zeros((m, U)), np.zeros((m, U))
    b = lay.slices[0]
    N[:, lay.j[b]] = R[:, b]
    for s in lay.slices[1:]:
        at[:, lay.j[s]] += R[:, s]                               # values are distinct within one family
    FP, TP = _rev(N), _rev(at)
    nb, p = FP[:, 0], TP[:, 0]
    n = nb + p
    ok = (p > 0) & (nb > 0)
    out = {"prevalence": _div(p, n)}
    prec = TP[:, :U] / np.maximum(TP[:, :U] + FP[:, :U], 1)       # precision at score >= u_j (counts: 0 rows -> 0)
    out["pr_auc"] = _div((at * prec).sum(axis=1), p)
    half = (nb[:, None] - FP[:, :U]) + 0.5 * N                   # benign rows below u_j, ties count one half
    out["roc_auc"] = _div((at * half).sum(axis=1), p * nb)
    out["ap_lift"] = out["pr_auc"] - out["prevalence"]
    out["lift_ratio"] = _div(out["pr_auc"], out["prevalence"])
    fams = []                                                    # per family: its value indices, cumulative rows
    for s in lay.slices[1:]:
        Rf = R[:, s]
        fams.append((lay.j[s], _rev(Rf)))
    pf = np.stack([c[:, 0] for _, c in fams], axis=1) if fams else np.zeros((m, 0))
    out["family_roc_auc"] = _div(np.stack([(R[:, s] * half[:, lay.j[s]]).sum(axis=1) for s in lay.slices[1:]], axis=1)
                                 if fams else np.zeros((m, 0)), pf * nb[:, None])
    fpr = _q(FP, nb[:, None])

    def fam_at(j):
        """Rows of each family scored >= u_j, for each replicate's own j."""
        if not fams:
            return np.zeros((m, 0))
        return np.stack([_at(c, np.searchsorted(jj, j, side="left")) for jj, c in fams], axis=1)

    def cut(X, target, below=True):
        """Where a non-increasing X (counts at score >= u_j) reaches the target: the first value index j with
        X(j) <= target (below=False: X(j) < target), and the share q of the rows at value j - 1 alerted as well, so
        that X reaches the target exactly. Ties at the cut-off are split in proportion: the straight segment of the
        ROC curve between two thresholds (the convention under which ROC-AUC counts ties one half)."""
        j = ((X > target[:, None]) if below else (X >= target[:, None])).sum(axis=1)
        lo, hi = _at(X, np.maximum(j - 1, 0)), _at(X, np.minimum(j, U))
        return j, np.where(j > 0, np.nan_to_num(_q(target - hi, lo - hi)), 0.0)

    def mix(get, j, q):
        """(1 - q) X(j) + q X(j - 1), for X given by get(j)."""
        qq = q[:, None] if get is fam_at else q
        return (1 - qq) * get(np.minimum(j, U)) + qq * get(np.maximum(j - 1, 0))

    def tp_at(j):
        return _at(TP, j)

    def fp_at(j):
        return _at(FP, j)

    def point(j, q, tag):
        tp, fp = mix(tp_at, j, q), mix(fp_at, j, q)
        out[f"{tag}_recall"], out[f"{tag}_fpr"] = _div(tp, p), _div(fp, nb)
        out[f"{tag}_precision"], out[f"{tag}_alert_share"] = _div(tp, tp + fp), _div(tp + fp, n)
        out[f"family_{tag}_recall"] = _div(mix(fam_at, j, q), pf)

    point(np.full(m, jf), np.zeros(m), "frozen")                 # as scored: no split of ties
    for a in FPR_LEVELS:
        point(*cut(FP, a * nb), f"oracle_fpr_{a:g}")             # benign alert rate exactly a
    for bud in BUDGETS:
        point(*cut(TP + FP, bud * n), f"budget_{bud:g}")         # alert share exactly bud
    rec = _q(TP, p[:, None])
    j95 = np.maximum((rec >= 0.95).sum(axis=1) - 1, 0)          # highest u_j still detecting >= 95 %
    out["fpr_at_tpr_0.95"] = np.where(ok, _at(fpr, j95), np.nan)  # as in the unit records (first threshold)
    j, q = cut(TP, 0.95 * p, below=False)                        # recall exactly 0.95, ties split in proportion
    out["fpr_at_tpr_0.95_interpolated"] = np.where(ok, _div(mix(fp_at, j, q), nb), np.nan)
    jmin = np.minimum((FP >= nb[:, None]).sum(axis=1), U)       # first u_j above the lowest-scored benign rows
    out["undetectable_share"] = np.where(ok, 1 - _div(_at(TP, jmin), p), np.nan)
    out["family_undetectable_share"] = np.where(ok[:, None], 1 - _div(fam_at(jmin), pf), np.nan)
    out["attack_share_at_lowest_score"] = _div(at[:, 0], p)
    out["benign_share_at_lowest_score"] = _div(N[:, 0], nb)
    if curve:
        out["curve_recall"] = np.stack([_div(mix(tp_at, *cut(FP, a * nb)), p) for a in CURVE_FPR], axis=1)
    return out


# ---- one evaluation set and feature space -------------------------------------------------------------------
def _unit(eval_id, space, name, seed, gsha, fr, fr_sha):
    up = S.unit_path(eval_id, space, name, seed)
    u = json.loads(up.read_text())
    if u["groups_sha256"] != gsha or u["freeze_sha256"] != fr_sha or not u["proposed"]:
        raise AssertionError(f"{up} was not scored for the proposed detector against these groups and this freeze")
    if fr["model_files_sha256"].get(u["model_file"]) != u["model_sha256"]:
        raise AssertionError(f"{up}: model file not the frozen one")
    sp = S.scores_path(eval_id, space, name, seed)
    if sha256_file(sp) != u["scores_sha256"]:
        raise AssertionError(f"{sp} does not match the SHA-256 in its unit record")
    return u, pd.read_parquet(sp)


def _summary(point, reps):
    """Seed mean, percentile 95 % CI of the replicate seed means, replicates used, per-seed values."""
    est = point.mean(axis=0)
    rm = reps.mean(axis=1)
    used = (~np.isnan(rm)).sum(axis=0)
    with warnings.catch_warnings():                 # a figure undefined in every replicate stays NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = (np.nanquantile(rm, q, axis=0) for q in (0.025, 0.975))
    return est, lo, hi, used, point


def _num(v):
    v = float(v)
    return None if np.isnan(v) else v


def analyse_set(eval_id, space, name, r, gi, groups, gsha, fr, fr_sha, W, seeds, log=print):
    t0 = time.time()
    att = r["y"] == 1
    fams = sorted({r["labels"][c] for c in np.unique(r["code"][att])})
    fam_of = {r["labels"].index(f): i for i, f in enumerate(fams)}
    cls = np.zeros(len(r["y"]), np.int64)
    cls[att] = pd.Series(r["code"][att]).map(fam_of).to_numpy(np.int64) + 1
    C = 1 + len(fams)
    blocks = np.unique(groups["block"].to_numpy())
    blk = np.searchsorted(blocks, r["block"])
    nb, B = len(blocks), len(W)
    points, reps, checks, thresholds = [], None, {"distinct_scores": [], "pr_auc_diff_to_unit": [],
                                                  "fpr_at_tpr_0.95_diff_to_unit": []}, []
    for si, s in enumerate(seeds):
        u, df = _unit(eval_id, space, name, s, gsha, fr, fr_sha)
        if not np.array_equal(df["_key"].to_numpy(), r["key"]):
            raise AssertionError(f"rows of {S.scores_path(eval_id, space, name, s).name} are not the rows read now")
        sc = df["score"].to_numpy(np.float32)
        thr = np.float32(u["threshold"])
        alerts = np.bincount(gi, weights=sc >= thr, minlength=len(groups)).astype(np.int64)
        if not np.array_equal(alerts, np.asarray(u["alerts"], np.int64)):
            raise AssertionError(f"seed {s}: alerts at the frozen threshold do not reproduce the unit record")
        y = r["y"].astype(np.int64)
        ap, f95 = pr_auc(y, sc), fpr_at_detection_rate(y, sc, 0.95)
        checks["pr_auc_diff_to_unit"].append(abs(ap - u["pr_auc"]))
        checks["fpr_at_tpr_0.95_diff_to_unit"].append(abs(f95 - u["fpr_at_tpr_0.95"]))
        if max(checks["pr_auc_diff_to_unit"][-1], checks["fpr_at_tpr_0.95_diff_to_unit"][-1]) > UNIT_TOL:
            raise AssertionError(f"seed {s}: PR-AUC or FPR at 95 % detection differ from the unit record by more than "
                                 f"{UNIT_TOL}")
        vals, H, lay = seed_counts(sc, cls, blk, C, nb)
        jf = int(np.searchsorted(vals, thr, side="left"))
        pt = measures((H @ np.ones(nb))[None, :], lay, jf, curve=True)
        if abs(pt["pr_auc"][0] - ap) > 1e-9 or abs(pt["fpr_at_tpr_0.95"][0] - f95) > 1e-9:
            raise AssertionError(f"seed {s}: the counts do not reproduce the per-row PR-AUC or FPR at 95 % detection")
        points.append(pt)
        checks["distinct_scores"].append(int(len(vals)))
        thresholds.append(float(u["threshold"]))
        step = max(1, min(500, int(CELLS // (H.shape[0] + 2 * lay.U))))
        for b0 in range(0, B, step):
            v = measures((H @ W[b0:b0 + step].T).T, lay, jf)
            if reps is None:
                reps = {k: np.full((B, len(seeds)) + a.shape[1:], np.nan) for k, a in v.items()}
            for k, a in v.items():
                reps[k][b0:b0 + step, si] = a
        log(f"[{eval_id}/{space}] seed {s}: {len(vals):,} distinct scores, PR-AUC {ap:.6f}, "
            f"{time.time() - t0:.0f} s so far")
    overall, family = {}, {f: {} for f in fams}
    for k in reps:
        est, lo, hi, used, per = _summary(np.stack([pt[k][0] for pt in points]), reps[k])
        if k.startswith("family_"):
            for i, f in enumerate(fams):
                family[f][k[len("family_"):]] = {"estimate": _num(est[i]), "ci_low": _num(lo[i]), "ci_high": _num(hi[i]),
                                                 "replicates_used": int(used[i]),
                                                 "per_seed": [_num(v) for v in per[:, i]]}
        else:
            overall[k] = {"estimate": _num(est), "ci_low": _num(lo), "ci_high": _num(hi), "replicates_used": int(used),
                          "per_seed": [_num(v) for v in per]}
    rows_by = dict(zip(r["labels"], np.bincount(r["code"], minlength=len(r["labels"])).tolist()))
    return {"analysis": "E1 ranking versus operating point (exploratory, outside the Holm family)",
            "plan": PLAN.relative_to(paths.REPO).as_posix(), "plan_sha256": sha256_file(PLAN),
            "eval": eval_id, "space": space, "candidate": name, "seeds": list(seeds), "thresholds": thresholds,
            "freeze_sha256": fr_sha, "groups_sha256": gsha, "rows": int(len(r["y"])), "attack_rows": int(att.sum()),
            "blocks": nb, "families": fams, "rows_by_family": rows_by,
            "bootstrap": {"B": B, "seed": stats.BOOTSTRAP_SEED, "unit": "300 s block", "statistic": "seed mean",
                          "ci": "percentile 95 %"},
            "checks": {"scores_sha256": "match", "rows": "same keys in the same order", "groups": "reproduced",
                       "alerts_at_frozen_threshold": "reproduced",
                       "distinct_scores_per_seed": checks["distinct_scores"],
                       "pr_auc_max_abs_diff_to_unit": max(checks["pr_auc_diff_to_unit"]),
                       "fpr_at_tpr_0.95_max_abs_diff_to_unit": max(checks["fpr_at_tpr_0.95_diff_to_unit"])},
            "measures": {k: v for k, v in overall.items() if k != "curve_recall"},
            "family_measures": family,
            "curve": {"fpr": list(CURVE_FPR), "recall_per_seed": [[_num(v) for v in pt["curve_recall"][0]] for pt in points],
                      "note": "oracle: threshold chosen from target labels"},
            "seconds": round(time.time() - t0, 1), "generated_utc": S._now(), "run": run_info()}


def run(sets=None, log=print):
    """E1 for every (evaluation set, feature space) not yet written, then the summary."""
    fr = seal.require_frozen(track="l12")
    fr_sha = sha256_file(seal.freeze_path("l12"))
    if "## E1." not in PLAN.read_text():
        raise AssertionError("the plan of E1 is not in docs/protocol/exploratory_analyses.md")
    todo = [s for s in (sets or SETS) if not result_path(*s).exists()]
    for e, sp in (sets or SETS):
        if (e, sp) not in todo:
            log(f"[{e}/{sp}] already written (E1 runs once)")
    for e in dict.fromkeys(e for e, _ in todo):
        groups = pd.read_parquet(S.groups_path(e))
        gsha = sha256_file(S.groups_path(e))
        r = rows(e, log)
        gi = group_index(r, groups)
        W = draws(groups["block"].nunique())
        for _, sp in [t for t in todo if t[0] == e]:
            res = analyse_set(e, sp, fr["proposed"], r, gi, groups, gsha, fr, fr_sha, W, D.seeds(), log)
            write_json(result_path(e, sp), res)
            log(f"[{e}/{sp}] written ({res['seconds']} s)")
    write_summary()


# ---- tables ---------------------------------------------------------------------------------------------------
def _ci(m, d=3):
    if m is None or m["estimate"] is None:
        return "n/a"
    s = f"{m['estimate']:.{d}f}"
    return s + (f" [{m['ci_low']:.{d}f}, {m['ci_high']:.{d}f}]" if m["ci_low"] is not None else "")


def _pct(m, d=2):
    return "n/a" if m is None or m["estimate"] is None else f"{100 * m['estimate']:.{d}f} %"


def write_summary():
    done = [(e, sp) for e, sp in SETS if result_path(e, sp).exists()]
    res = {s: json.loads(result_path(*s).read_text()) for s in done}
    lines = ["# Exploratory analysis E1: ranking versus operating point", "",
             "_Post hoc and outside the Holm family (plan: `docs/protocol/exploratory_analyses.md`, written after the "
             "confirmatory results). Frozen proposed detector, 5 seeds; estimate = seed mean, 95 % CI = 300 s block "
             "bootstrap (B = 10 000). **Oracle** columns choose the threshold from target labels: upper bounds that no "
             "deployed detector could reach. Budget columns alert on the top-scored rows and use no labels. Oracle, "
             "budget and \"ties split\" figures split the rows tied at the cut-off score in proportion (the straight "
             "segment of the ROC curve); the frozen threshold and the first FPR at 95 % detection do not, as scored._", "",
             "## Ranking and operating points", "",
             "| evaluation / space | attack share | PR-AUC | AP-lift | ROC-AUC | recall at frozen threshold | its FPR | "
             "oracle recall, FPR 0.1 % | oracle recall, FPR 1 % | oracle recall, FPR 5 % | recall, top 1 % | "
             "recall, top 5 % | FPR at 95 % detection | same, ties split | attacks undetectable below 100 % FPR |",
             "|---|---:|---|---|---|---|---:|---|---|---|---|---|---|---|---|"]
    for (e, sp), x in res.items():
        m = x["measures"]
        lines.append(f"| {e} / {sp} | {_pct(m['prevalence'])} | {_ci(m['pr_auc'])} | {_ci(m['ap_lift'])} | "
                     f"{_ci(m['roc_auc'])} | {_ci(m['frozen_recall'])} | {_pct(m['frozen_fpr'], 3)} | "
                     f"{_ci(m['oracle_fpr_0.001_recall'])} | {_ci(m['oracle_fpr_0.01_recall'])} | "
                     f"{_ci(m['oracle_fpr_0.05_recall'])} | {_ci(m['budget_0.01_recall'])} | {_ci(m['budget_0.05_recall'])} | "
                     f"{_ci(m['fpr_at_tpr_0.95'])} | {_ci(m['fpr_at_tpr_0.95_interpolated'])} | "
                     f"{_ci(m['undetectable_share'])} |")
    lines += ["", "## Attack families", "",
              "| evaluation / space | family | rows | recall at frozen threshold | oracle recall, FPR 1 % | recall, top 5 % | "
              "ROC-AUC against benign | undetectable below 100 % FPR |", "|---|---|---:|---|---|---|---|---|"]
    for (e, sp), x in res.items():
        for f, m in x["family_measures"].items():
            lines.append(f"| {e} / {sp} | {f} | {x['rows_by_family'][f]:,} | {_ci(m['frozen_recall'])} | "
                         f"{_ci(m['oracle_fpr_0.01_recall'])} | {_ci(m['budget_0.05_recall'])} | {_ci(m['roc_auc'])} | "
                         f"{_ci(m['undetectable_share'])} |")
    lines += ["", "## Ties at the lowest score", "",
              "| evaluation / space | lowest-score share, attack rows | lowest-score share, benign rows |", "|---|---|---|"]
    for (e, sp), x in res.items():
        m = x["measures"]
        lines.append(f"| {e} / {sp} | {_ci(m['attack_share_at_lowest_score'])} | {_ci(m['benign_share_at_lowest_score'])} |")
    lines += ["", "## Checks (every one must hold, or nothing is written)", "",
              "| evaluation / space | rows | blocks | distinct scores per seed | PR-AUC vs unit record | "
              "FPR at 95 % vs unit record |", "|---|---:|---:|---|---:|---:|"]
    for (e, sp), x in res.items():
        c = x["checks"]
        lines.append(f"| {e} / {sp} | {x['rows']:,} | {x['blocks']} | {', '.join(f'{v:,}' for v in c['distinct_scores_per_seed'])} | "
                     f"{c['pr_auc_max_abs_diff_to_unit']:.1e} | {c['fpr_at_tpr_0.95_max_abs_diff_to_unit']:.1e} |")
    lines += ["", "Scores SHA-256, row order, groups and alerts at the frozen threshold reproduce the confirmatory "
              "records exactly for every seed; PR-AUC and FPR at 95 % detection are recomputed from the float32 "
              "scores and differ from the float64 unit records by at most the figures shown."]
    missing = [f"{e} / {sp}" for e, sp in SETS if (e, sp) not in done]
    if missing:
        lines += ["", f"Not yet run: {', '.join(missing)}."]
    out_dir().mkdir(parents=True, exist_ok=True)
    (out_dir() / "summary.md").write_text("\n".join(lines) + "\n")
    return out_dir() / "summary.md"
