"""One-time scoring of the frozen models on the sealed test data (ADR-0007; preregistration §6, §6b, §7, §8.1).

Nothing here runs unless `seal.require_frozen` passes for the track it scores: the freeze record is committed
unchanged and every model file and development record still matches its SHA-256. Nothing is fitted: the
frozen models, their val thresholds and the frozen feature mappings are applied unchanged.

Levels 1-2. A unit is (evaluation set, feature space, candidate, seed). Evaluation rows are grouped by
(300 s block, label family, binary label, seen at 3 s.f., seen at 2 s.f.); a unit stores how many rows of each
group the model alerted on. Macro-F1, the block bootstrap, the S7 subsets and the per-class figures all follow
from these counts (src/evaluation/confirmatory.py), so the analysis never reads the data again.

    results/confirmatory/level12/groups/<eval>.parquet                     groups and their row counts
    results/confirmatory/level12/units/<eval>/<space>/<candidate>/seed<s>.json
    <Drive>/scores/<eval>/<space>/<proposed>/seed<s>.parquet               per-row scores, proposed detector only

Level 3. A unit is (candidate, seed) on the LANL test period:
    results/confirmatory/level3/units/<candidate>/seed<s>.json    per test host: alerted early?, lead time,
          score of its compromise hour (reactive baseline); realised false-alarm rate; PR-AUC
    results/confirmatory/level3/random_baseline.json              random alerting at the same budget

A unit is written once; running again skips finished units, so an interrupted run resumes where it stopped.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import datasets, harmonise, near
from ..evaluation.metrics import fpr_at_detection_rate, pr_auc
from ..utils import paths
from ..utils.io import sha256_file, write_json, write_parquet
from ..utils.logging import run_info
from . import develop as D
from . import seal
from .candidates import NEURAL, Candidate

EVAL_SETS = {   # order = scoring order: the small sets first
    "cicids2017_test": {"dataset": "cicids2017_improved", "split": "primary", "partitions": ("test",),
                        "side": "cic", "spaces": ("cic", "unsw_canonical"), "against": None},
    "unsw_nb15": {"dataset": "unsw_nb15", "split": "external", "partitions": ("external",),
                  "side": "unsw", "spaces": ("unsw_canonical",), "against": "cicids2017_improved"},
    "csecicids2018": {"dataset": "csecicids2018_improved", "split": "external", "partitions": ("external",),
                      "side": "cic", "spaces": ("cic",), "against": "cicids2017_improved"},
}
MAPPINGS = {"cic": "cic_shared_features.yaml", "unsw_canonical": "cic_to_unsw_mapping.yaml"}
CHUNK = 500_000
LABEL_CODES = 128


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def out_dir():
    return paths.RESULTS / "confirmatory"


def cache_dir(eval_id):
    return Path(os.environ.get("PHD_SCRATCH", "/content/scratch")) / "score_cache" / eval_id


def groups_path(eval_id):
    return out_dir() / "level12" / "groups" / f"{eval_id}.parquet"


def unit_path(eval_id, space, name, seed):
    return out_dir() / "level12" / "units" / eval_id / space / name / f"seed{seed}.json"


def lanl_unit_path(name, seed):
    return out_dir() / "level3" / "units" / name / f"seed{seed}.json"


def scores_path(eval_id, space, name, seed):
    return paths.SCORES / eval_id / space / name / f"seed{seed}.parquet"


def _features(space, side, X):
    return D.features("cic", X) if space == "cic" else harmonise.to_canonical_unsw(X, side)


def _s7_record(ds, against):
    """The S7 record of this evaluation set (row counts the groups must reproduce)."""
    if against:
        return near._record(ds, against)
    return near._record(ds)["subsets"]


def _s7_rows(rec, sig, against):
    r = rec[f"sig{sig}"] if against else rec[f"sig{sig}"]["test"]
    return r["rows"], r["unseen_rows"]


def data_provenance(eval_id):
    es = EVAL_SETS[eval_id]
    ds = es["dataset"]
    return {"dataset": ds, "split": es["split"], "partitions": list(es["partitions"]),
            "provenance": near.provenance(ds), "sidecar_sha256": sha256_file(near.sidecar_path(ds, es["against"])),
            "mappings_sha256": {sp: sha256_file(harmonise.MAPPING_DIR / MAPPINGS[sp]) for sp in es["spaces"]}}


def _decode(k, labels):
    s2, s3, y = k & 1, (k >> 1) & 1, (k >> 2) & 1
    rest = k >> 3
    return pd.DataFrame({"block": rest // LABEL_CODES - 1, "label": np.asarray(labels, dtype=object)[rest % LABEL_CODES],
                         "y": y.astype(np.int8), "seen_sig3": s3.astype(bool), "seen_sig2": s2.astype(bool)})


def prepare(eval_id, log=print):
    """Features, labels and groups of one evaluation set, cached on the Colab disk; writes the groups file.

    The groups must reproduce the row counts of the S7 record exactly (all rows, and rows unseen at 3 and
    2 s.f.), and an existing groups file must be reproduced exactly, or the run stops.
    """
    es = EVAL_SETS[eval_id]
    ds, against = es["dataset"], es["against"]
    cdir = cache_dir(eval_id)
    prov = data_provenance(eval_id)
    man = cdir / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        if m.get("data") == prov and groups_path(eval_id).exists() and \
                sha256_file(groups_path(eval_id)) == m.get("groups_sha256"):
            return m
    near.require_sidecar(ds, against)
    side = pd.read_parquet(near.sidecar_path(ds, against), columns=["_key", "seen_sig3", "seen_sig2"])
    index = pd.Index(side["_key"].to_numpy())
    s3, s2 = side["seen_sig3"].to_numpy(bool), side["seen_sig2"].to_numpy(bool)
    del side
    cdir.mkdir(parents=True, exist_ok=True)
    labels, columns, gkeys, n_parts = {}, {}, [], 0
    for i, (X, meta) in enumerate(datasets.iter_parts(ds, es["split"], es["partitions"])):
        pos = index.get_indexer(meta["_key"].to_numpy())
        if (pos < 0).any():
            raise AssertionError(f"S7 sidecar of {ds} does not cover every evaluation row; run --near-duplicates "
                                 "or --cross-overlap again")
        lab = meta["_label"].astype(str).to_numpy()
        for v in pd.unique(lab):
            labels.setdefault(v, len(labels))
        if len(labels) > LABEL_CODES:
            raise AssertionError(f"more than {LABEL_CODES} label families in {ds}")
        code = pd.Series(lab).map(labels).to_numpy(np.int64)
        y = meta["_y"].to_numpy(np.int64)
        block = meta["_block"].to_numpy(np.int64)
        if (block < -1).any():
            raise AssertionError("unexpected block id")
        k = (((block + 1) * LABEL_CODES + code) * 2 + y) * 4 + s3[pos].astype(np.int64) * 2 + s2[pos]
        gkeys.append(k)
        np.save(cdir / f"y_{i:05d}.npy", y.astype(np.int8))
        np.save(cdir / f"key_{i:05d}.npy", meta["_key"].to_numpy())
        for sp in es["spaces"]:
            F = _features(sp, es["side"], X)
            columns[sp] = list(F.columns)
            np.save(cdir / f"X_{sp}_{i:05d}.npy", F.to_numpy(np.float32))
        n_parts = i + 1
        log(f"[{eval_id}] cached part {i + 1}: {len(y):,} rows")
    allk = np.unique(np.concatenate(gkeys))
    n = np.zeros(len(allk), np.int64)
    for i, k in enumerate(gkeys):
        g = np.searchsorted(allk, k).astype(np.int32)
        np.save(cdir / f"g_{i:05d}.npy", g)
        n += np.bincount(g, minlength=len(allk))
    inv = [None] * len(labels)
    for v, c in labels.items():
        inv[c] = v
    groups = _decode(allk, inv)
    groups["n"] = n
    rec = _s7_record(ds, against)
    for sig in (3, 2):
        rows, unseen = _s7_rows(rec, sig, against)
        got = (int(n.sum()), int(n[~groups[f"seen_sig{sig}"].to_numpy()].sum()))
        if got != (rows, unseen):
            raise AssertionError(f"[{eval_id}] rows {got} differ from the S7 record ({rows}, {unseen}) at {sig} s.f.")
    gp = groups_path(eval_id)
    if gp.exists():
        old = pd.read_parquet(gp)
        if not old.equals(groups):
            raise AssertionError(f"{gp.name} exists and the evaluation data no longer reproduce it")
        gsha = sha256_file(gp)
    else:
        gsha = write_parquet(groups, gp)
    m = {"eval": eval_id, "data": prov, "parts": n_parts, "rows": int(n.sum()), "groups": int(len(groups)),
         "groups_sha256": gsha, "columns": columns, "built_utc": _now()}
    write_json(man, m)
    log(f"[{eval_id}] {m['rows']:,} rows in {m['groups']:,} groups (block x label x seen)")
    return m


def _load_frozen(rel, fr):
    path = paths.MODELS / rel
    if fr["model_files_sha256"].get(rel) is None:
        raise seal.Sealed(f"{rel} is not in the freeze record")
    if sha256_file(path) != fr["model_files_sha256"][rel]:
        raise seal.Sealed(f"model file {rel} changed after the freeze")
    return Candidate.load(path)


def predict(m, X, device=None):
    """P(attack) in chunks, with the model's own predict_proba (the development path). Neural candidates run on the
    GPU when there is one, with the same preprocessing; `gpu_check` records how far that is from the CPU path."""
    use_gpu = False
    if m.name in NEURAL:
        import torch
        use_gpu = (device or ("cuda" if torch.cuda.is_available() else "cpu")) != "cpu"
    if not use_gpu:
        return np.concatenate([m.predict_proba(X[i:i + CHUNK]) for i in range(0, len(X), CHUNK)] or [np.zeros(0)])
    dev = torch.device(device or "cuda")
    net, out = m.model.to(dev), []
    with torch.no_grad():
        for i in range(0, len(X), CHUNK):
            Z = torch.from_numpy(m.prep.transform(X[i:i + CHUNK])).to(dev)
            for j in range(0, len(Z), 65536):
                out.append(torch.sigmoid(net(Z[j:j + 65536]).squeeze(1)).cpu().numpy())
    m.model = net.cpu()
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def gpu_check(m, X, thr, rows=200_000):
    """Neural scores on the GPU against the CPU path used in development, on the first rows of the set."""
    import torch
    if m.name not in NEURAL or not torch.cuda.is_available():
        return None
    Xs = np.asarray(X[:rows])
    a, b = predict(m, Xs, "cuda"), predict(m, Xs, "cpu")
    return {"rows": int(len(Xs)), "max_abs_diff": float(np.max(np.abs(a - b))) if len(Xs) else 0.0,
            "decisions_differing": int(((a >= thr) != (b >= thr)).sum())}


def score_unit(eval_id, space, name, seed, man, fr, fr_sha, log=print):
    rp = unit_path(eval_id, space, name, seed)
    if rp.exists():
        return json.loads(rp.read_text())
    rec = json.loads(D.record_path(space, name).read_text())
    if rec["features"] != man["columns"][space]:
        raise AssertionError(f"{space} features of {eval_id} differ from those {name} was trained on")
    f = rec["final"][str(seed)]
    m = _load_frozen(f["model_file"], fr)
    thr, t0, cdir = float(f["threshold"]), time.time(), cache_dir(eval_id)
    n_groups = man["groups"]
    alerts, scores, ys, keys, check = np.zeros(n_groups, np.int64), [], [], [], None
    for i in range(man["parts"]):
        X = np.load(cdir / f"X_{space}_{i:05d}.npy", mmap_mode="r")
        if i == 0:
            check = gpu_check(m, X, thr)
        p = predict(m, X)
        g = np.load(cdir / f"g_{i:05d}.npy")
        alerts += np.bincount(g, weights=(p >= thr), minlength=n_groups).astype(np.int64)
        scores.append(p.astype(np.float32))
        ys.append(np.load(cdir / f"y_{i:05d}.npy"))
        if name == fr["proposed"]:
            keys.append(np.load(cdir / f"key_{i:05d}.npy"))
    p, y = np.concatenate(scores), np.concatenate(ys).astype(np.int64)
    out = {"eval": eval_id, "space": space, "candidate": name, "seed": seed, "proposed": name == fr["proposed"],
           "model_file": f["model_file"], "model_sha256": f["model_sha256"], "threshold": thr,
           "freeze_sha256": fr_sha, "data": man["data"], "groups_sha256": man["groups_sha256"],
           "rows": int(len(y)), "alerted_rows": int(alerts.sum()), "alerts": alerts.tolist(),
           "pr_auc": pr_auc(y, p) if 0 < y.sum() < len(y) else None,
           "fpr_at_tpr_0.95": fpr_at_detection_rate(y, p, 0.95) if 0 < y.sum() < len(y) else None,
           "gpu_check": check, "seconds": round(time.time() - t0, 1), "scored_utc": _now(), "run": run_info()}
    if name == fr["proposed"]:
        sp = scores_path(eval_id, space, name, seed)
        out["scores_file"] = sp.relative_to(paths.SCORES).as_posix()
        out["scores_sha256"] = write_parquet(pd.DataFrame({"_key": np.concatenate(keys), "score": p}), sp)
    write_json(rp, out)
    log(f"[{eval_id}/{space}/{name}] seed {seed}: {out['alerted_rows']:,} of {out['rows']:,} rows alerted "
        f"({out['seconds']} s)")
    return out


def frozen_candidates(fr):
    """The candidates of a freeze record, in config order (the record's ranking keeps it)."""
    return list(fr.get("ranking_val_macro_f1") or fr["ranking_val_early_alert_rate_and_pr_auc"])


def level12_todo(fr, evals=None, spaces=None, candidates=None):
    return [(e, sp, c, s) for e in (evals or EVAL_SETS) for sp in EVAL_SETS[e]["spaces"] if not spaces or sp in spaces
            for c in frozen_candidates(fr) if not candidates or c in candidates for s in D.seeds()
            if not unit_path(e, sp, c, s).exists()]


def score_level12(evals=None, spaces=None, candidates=None, log=print):
    fr = seal.require_frozen(track="l12")
    fr_sha = sha256_file(seal.freeze_path("l12"))
    for e in (evals or EVAL_SETS):
        todo = level12_todo(fr, [e], spaces, candidates)
        if not todo:
            log(f"[{e}] every unit already scored")
            continue
        if EVAL_SETS[e]["spaces"] and any(c in NEURAL for _, _, c, _ in todo):
            import torch
            if not torch.cuda.is_available() and not os.environ.get("PHD_ALLOW_CPU_NEURAL"):
                raise RuntimeError("the MLP and CNN-LSTM baselines need a GPU runtime to score in reasonable time "
                                   "(Runtime -> Change runtime type -> T4 GPU)")
        man = prepare(e, log)
        for _, sp, c, s in todo:
            score_unit(e, sp, c, s, man, fr, fr_sha, log)


# ---- Level 3 ---------------------------------------------------------------------------------------------------
def _take(hh, cols, rows):
    """Feature matrix of the given rows, one column at a time (no copy of the whole table)."""
    X = np.empty((len(rows), len(cols)), np.float32)
    for j, c in enumerate(cols):
        X[:, j] = hh[c].to_numpy()[rows]
    return X


def _lanl_frames(cfg, L):
    """Test frame (with the day before the test period, for the 24 h windows of early test compromises), its
    features, the test hosts, the val frame (random baseline) and the features of each test host's compromise
    hour (reactive baseline). The 58-day table is dropped once these are taken from it."""
    sp = L.split()
    d0, d1 = sp["test_days"]
    horizon = cfg["label"]["horizon_seconds"]
    hh, comp = L.read_host_hours(d1)
    feats, fc = L.base_features(), cfg["features"]
    hh = L.add_trailing(hh, feats, fc["trailing_hours"], fc["active_hours"])
    cols = feats + [c for c in hh.columns if c.endswith(f"_sum{fc['trailing_hours']}h") or c.startswith("active_hours")]
    rule = L.unresolved_rule(cfg)
    light = hh[["computer", "hour", "day"]].assign(_row=np.arange(len(hh)))
    frame, hosts = L.period_frame(light, comp, (d0 - 1, d1), horizon, rule)
    hosts = hosts[hosts > (d0 - 1) * L.DAY]                         # first compromised inside the test period
    if len(hosts) != sp["test_compromised_hosts"]:
        raise AssertionError(f"{len(hosts)} test hosts, the frozen split says {sp['test_compromised_hosts']}")
    val, _ = L.period_frame(light, comp, sp["val_days"], horizon, rule)
    want = pd.DataFrame({"computer": hosts.index.to_numpy(), "hour": (hosts.to_numpy() // L.HOUR).astype(np.int64)})
    have = light[light["computer"].isin(want["computer"])][["computer", "hour", "_row"]]
    r = want.merge(have.astype({"hour": np.int64}), on=["computer", "hour"], how="left")["_row"].to_numpy()
    found = ~np.isnan(r.astype(float))
    Xr = np.full((len(want), len(cols)), np.nan, np.float32)
    Xr[found] = _take(hh, cols, r[found].astype(np.int64))
    X = _take(hh, cols, frame["_row"].to_numpy())
    del hh, light
    return {"hh_cols": cols, "frame": frame, "X": X, "hosts": hosts, "val": val, "react": Xr, "comp": comp,
            "days": (d0, d1), "horizon": horizon}


def random_baseline(fx, budget):
    """Random alerting at the same budget: i.i.d. uniform scores per host-hour, threshold set on val exactly as
    for the models. A host-day with h active hours alerts with probability 1 - t^h; t solves the val budget.
    The early-alert count is then a sum of independent Bernoulli(1 - t^w) over test hosts (w = host-hours in
    the host's 24 h window), whose exact distribution gives the baseline's 95 % range."""
    def day_hours(f, mask):
        g = pd.DataFrame({"h": f["computer"].to_numpy()[mask], "d": f["day"].to_numpy()[mask]})
        return g.groupby(["h", "d"]).size().to_numpy(np.float64)
    v = fx["val"]
    hv = day_hours(v, v["_eligible_day"].to_numpy())
    lo, hi = 0.0, 1.0
    for _ in range(200):                                          # mean(1 - t^h) is decreasing in t
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if np.mean(1 - mid ** hv) <= budget else (mid, hi)
    t = hi
    f, (d0, _) = fx["frame"], fx["days"]
    ht = day_hours(f, f["_eligible_day"].to_numpy() & (f["day"].to_numpy() >= d0))
    pos = f[f["_y"].to_numpy() == 1]
    w = pos.groupby("computer").size().reindex(fx["hosts"].index).fillna(0).to_numpy(np.float64)
    ph = 1 - t ** w
    pmf = np.array([1.0])
    for q in ph:                                                  # Poisson-binomial by convolution
        pmf = np.convolve(pmf, [1 - q, q])
    cdf = np.cumsum(pmf)
    n = len(ph)
    return {"threshold": t, "val_fa_rate": float(np.mean(1 - t ** hv)), "test_fa_rate": float(np.mean(1 - t ** ht)),
            "hosts": n, "window_host_hours": w.astype(int).tolist(),
            "expected_early_alert_rate": float(ph.mean()),
            "early_alert_rate_q025": float(np.searchsorted(cdf, 0.025) / n),
            "early_alert_rate_q975": float(np.searchsorted(cdf, 0.975) / n)}


def lanl_config(fr):
    """The Level 3 rules the freeze was made under (labels, horizon, features, budget)."""
    from . import lanl_develop as L
    if sha256_file(L.CONFIG) != fr["models_config_sha256"]:
        raise seal.Sealed(f"{L.CONFIG.name} is not the config of the Level 3 freeze")
    return L.load_config()


def score_lanl(candidates=None, log=print):
    from . import lanl_develop as L
    fr = seal.require_frozen(track="lanl")
    fr_sha = sha256_file(seal.freeze_path("lanl"))
    cfg = lanl_config(fr)
    todo = [(c, s) for c in frozen_candidates(fr) if not candidates or c in candidates for s in D.seeds()
            if not lanl_unit_path(c, s).exists()]
    rb = out_dir() / "level3" / "random_baseline.json"
    if not todo and rb.exists():
        log("[lanl] every unit already scored")
        return
    fx = _lanl_frames(cfg, L)
    frame, hosts, (d0, d1) = fx["frame"], fx["hosts"], fx["days"]
    budget = float(cfg["false_alarm"]["budget"])
    if not rb.exists():
        write_json(rb, {**random_baseline(fx, budget), "budget": budget, "freeze_sha256": fr_sha, "run": run_info()})
    X, Xr = fx["X"], fx["react"]
    have_r = ~np.isnan(Xr).all(axis=1)
    day = frame["day"].to_numpy()
    is_test = (day >= d0) | ((frame["_y"].to_numpy() == 1) & frame["computer"].isin(hosts.index).to_numpy())
    fa_rows = frame[day >= d0]
    acct = fx["comp"].set_index("comp")["user"]
    for name, s in todo:
        rec = json.loads(L.record_path(name, cfg["version"]).read_text())
        if rec["features"] != fx["hh_cols"]:
            raise AssertionError(f"LANL features differ from those {name} was trained on")
        f = rec["final"][str(s)]
        m = _load_frozen(f["model_file"], fr)
        thr, t0 = float(f["threshold"]), time.time()
        p = predict(m, X)
        rate, per_host = L.early_alerts(p, frame, hosts, thr, fx["horizon"])
        on_test = day >= d0                                       # boundary sensitivity: alerts on test days only
        rate_t, per_host_t = L.early_alerts(p[on_test], frame[on_test], hosts, thr, fx["horizon"])
        el = fa_rows["_eligible_day"].to_numpy()
        g = pd.DataFrame({"h": fa_rows["computer"].to_numpy()[el], "d": fa_rows["day"].to_numpy()[el],
                          "s": p[day >= d0][el]})
        day_alert = g.groupby(["h", "d"])["s"].max().to_numpy() >= thr
        pr = np.full(len(Xr), np.nan)
        if have_r.any():
            pr[have_r] = predict(m, Xr[have_r])
        yt = frame["_y"].to_numpy()[is_test]
        pos = frame[frame["_y"].to_numpy() == 1].groupby("computer").size()
        out = {"candidate": name, "seed": s, "proposed": name == fr["proposed"], "model_file": f["model_file"],
               "model_sha256": f["model_sha256"], "threshold": thr, "freeze_sha256": fr_sha,
               "data": {"dataset": L.DS, "provenance": near.provenance(L.DS), "test_days": [d0, d1]},
               "rules_version": str(cfg["version"]), "early_alert_rate": rate, "hosts": len(per_host),
               "hosts_alerted": int(sum(v is not None for v in per_host.values())),
               "early_alert_rate_test_days_only": rate_t,
               "fa_rate": float(day_alert.mean()), "eligible_host_days": int(len(day_alert)),
               "pr_auc": pr_auc(yt, p[is_test]) if 0 < yt.sum() < len(yt) else None,
               "positives": int(yt.sum()), "rows": int(is_test.sum()),
               "per_host": [{"host": str(h), "account": str(acct.get(h)), "compromise_time": float(t),
                             "day": int(t // L.DAY) + 1, "window_host_hours": int(pos.get(h, 0)),
                             "lead_hours": None if per_host[h] is None else per_host[h] / L.HOUR,
                             "lead_hours_test_days_only": None if per_host_t[h] is None else per_host_t[h] / L.HOUR,
                             "window_on_previous_day": bool(t - fx["horizon"] < (d0 - 1) * L.DAY),
                             "compromise_hour_score": None if np.isnan(pr[i]) else float(pr[i]),
                             "reactive_alert": bool(pr[i] >= thr) if not np.isnan(pr[i]) else False}
                            for i, (h, t) in enumerate(hosts.items())],
               "seconds": round(time.time() - t0, 1), "scored_utc": _now(), "run": run_info()}
        write_json(lanl_unit_path(name, s), out)
        log(f"[lanl/{name}] seed {s}: {out['hosts_alerted']} of {out['hosts']} test hosts alerted early, "
            f"false alarms {out['fa_rate']:.4%} of {out['eligible_host_days']:,} host-days")
