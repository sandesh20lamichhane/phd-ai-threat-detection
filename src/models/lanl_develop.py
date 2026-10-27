"""Level 3 (LANL, H2) model development on the train and val days only (amendments 1.5 and 1.6, §7a).

Unit: host-hour. Alert time of a host-hour = end of that hour; its features use only events before that time.
For a period ending at E, a host's compromise time t is known only if t <= E (censoring):
    rows of a host after its known compromise (end of hour > t) leave the population;
    a row is positive if 0 <= t - end of hour < 24 h;
    amendment 1.6: a row of a host with no known compromise whose 24 h window runs past E is unresolved (its
    outcome is not known inside the period) and leaves the population, instead of counting as negative.
False alarms are counted on active host-days that neither overlap the 24 h before a known compromise nor come
after it. The alert threshold is the lowest one keeping that share at or below 1 % on val.

Records of the superseded 1.5 rules stay where they were written (results/models/development/lanl/, models in
development/lanl/, freeze_lanl.json); the 1.6 ones go to the lanl_1.6 folders.
"""
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from ..data import near
from ..utils import paths
from ..utils.io import sha256_file, write_json
from ..utils.logging import run_info
from .candidates import Candidate, make
from .develop import seeds

CONFIG = paths.CONFIG / "models_lanl_1.6.yaml"
CONFIG_1_5 = paths.CONFIG / "models_lanl.yaml"     # superseded by amendment 1.6; kept with its records and freeze
VERSION = "1.6"
DS = "lanl_cyber1"
HOUR, DAY = 3600, 86400


def dev_key(version=None):
    """Folder and freeze-record key of a rule version: 'lanl' for 1.5 (where it was written), else 'lanl_<v>'."""
    v = str(version or VERSION)
    return "lanl" if v == "1.5" else f"lanl_{v}"


def load_config(path=None):
    return yaml.safe_load(Path(path or CONFIG).read_text())


def split():
    return json.loads((paths.RESULTS / "data_audit" / DS / "level3_split.json").read_text())


def record_path(name, version=None):
    return paths.RESULTS / "models" / "development" / dev_key(version) / f"{name}.json"


def model_path(name, seed, version=None):
    return paths.MODELS / "development" / dev_key(version) / name / f"seed{seed}.pkl"


def base_features():
    return json.loads((paths.RESULTS / "data_audit" / DS / "report.json").read_text())["features"]


def read_host_hours(last_day):
    """Host-hour rows of days 1..last_day only (the later days are never read here)."""
    from ..data.verify import require_current
    require_current(DS, json.loads((paths.RESULTS / "data_audit" / DS / "report.json").read_text()))
    f = paths.PROCESSED / DS / "v1" / "host_hour.parquet"
    hh = pq.read_table(f, filters=[("day", "<=", int(last_day))]).to_pandas()
    comp = pd.read_parquet(paths.PROCESSED / DS / "v1" / "compromises.parquet")
    return hh, comp


def add_trailing(hh, feats, hours=24, active=True):
    """Per host, sum of each count over the current and previous hours-1 hours (causal), plus active hours.

    One column at a time (float64 running sums, stored as float32), so the 58-day table of the test scoring fits
    in memory; the numbers are identical to summing all columns at once.
    """
    hh = hh.sort_values(["computer", "hour"], kind="stable").reset_index(drop=True)
    codes = pd.factorize(hh["computer"])[0].astype(np.int64)
    key = codes * (1 << 32) + hh["hour"].to_numpy(np.int64)
    start = np.searchsorted(key, key - (hours - 1), side="left")        # first row of the window, same host
    idx = np.arange(len(hh))
    for c in feats:
        cs = np.concatenate([[0.0], np.cumsum(hh[c].to_numpy(np.float64))])
        hh[f"{c}_sum{hours}h"] = (cs[idx + 1] - cs[start]).astype(np.float32)
    if active:
        hh[f"active_hours_{hours}h"] = (idx + 1 - start).astype(np.float32)
    return hh


def period_frame(hh, comp, days, horizon=DAY, unresolved="exclude"):
    """Rows of the period with label, eligibility and compromise bookkeeping (censored at the period end).

    unresolved: "exclude" (amendment 1.6) drops rows of hosts with no known compromise whose window
    (alert time, alert time + horizon] runs past the period end; "negative" is the superseded 1.5 rule.
    """
    if unresolved not in ("exclude", "negative"):
        raise ValueError(f"unresolved must be 'exclude' or 'negative', not {unresolved}")
    d0, d1 = days
    end_p = d1 * DAY
    tc = comp.set_index("comp")["compromise_time"].astype(float)
    tc = tc[tc <= end_p]                                               # censoring
    f = hh[(hh["day"] >= d0) & (hh["day"] <= d1)].copy()
    t = f["computer"].map(tc).to_numpy(dtype=float)
    t = np.where(np.isnan(t), np.inf, t)
    alert_time = (f["hour"].to_numpy(np.int64) + 1) * HOUR
    keep = alert_time <= t                                             # drop rows after the known compromise
    if unresolved == "exclude":
        keep &= ~(np.isinf(t) & (alert_time + horizon > end_p))       # outcome not known inside the period
    f, t, alert_time = f[keep], t[keep], alert_time[keep]
    lead = t - alert_time
    f["_y"] = ((lead >= 0) & (lead < horizon)).astype(np.int64)
    f["_t"] = t
    day = f["day"].to_numpy(np.int64)
    overlap = ((day - 1) * DAY < t) & (day * DAY > t - horizon)        # host-day touches the 24 h before t
    f["_eligible_day"] = ~overlap                                       # later days of the host were dropped above
    hosts = tc[(tc > (d0 - 1) * DAY) & (tc <= end_p)]                  # compromised inside this period
    return f.reset_index(drop=True), hosts


def fa_threshold(scores, frame, budget):
    """Lowest threshold with at most `budget` of eligible host-days alerting (alert = score >= threshold)."""
    el = frame["_eligible_day"].to_numpy()
    g = pd.DataFrame({"h": frame["computer"].to_numpy()[el], "d": frame["day"].to_numpy()[el], "s": scores[el]})
    day_max = np.sort(g.groupby(["h", "d"])["s"].max().to_numpy())[::-1]
    if not len(day_max):
        raise RuntimeError("no eligible host-day to set the false-alarm threshold on")
    k = int(np.floor(budget * len(day_max)))
    if k >= len(day_max):
        return -np.inf, len(day_max)
    return float(np.nextafter(day_max[k], np.inf)), len(day_max)      # the (k+1)-th largest day stays silent


def early_alerts(scores, frame, hosts, thr, horizon=DAY):
    """Per compromised host: alerted early?, and the lead time of its first alert in the window."""
    m = frame["_y"].to_numpy() == 1
    w = pd.DataFrame({"h": frame["computer"].to_numpy()[m], "s": scores[m],
                      "lead": (frame["_t"].to_numpy()[m] - (frame["hour"].to_numpy()[m] + 1) * HOUR)})
    w = w[w["s"] >= thr]
    first = w.groupby("h")["lead"].max()                              # earliest alert = largest lead
    out = {h: (float(first[h]) if h in first.index else None) for h in hosts.index}
    rate = float(np.mean([v is not None for v in out.values()])) if out else None
    return rate, out


def evaluate(scores, frame, hosts, thr):
    from ..evaluation.metrics import pr_auc
    el = frame["_eligible_day"].to_numpy()
    g = pd.DataFrame({"h": frame["computer"].to_numpy()[el], "d": frame["day"].to_numpy()[el], "s": scores[el]})
    day_alert = g.groupby(["h", "d"])["s"].max().to_numpy() >= thr
    rate, per_host = early_alerts(scores, frame, hosts, thr)
    leads = [v for v in per_host.values() if v is not None]
    y = frame["_y"].to_numpy()
    return {"early_alert_rate": rate, "hosts": len(per_host), "hosts_alerted": len(leads),
            "median_lead_hours": float(np.median(leads) / HOUR) if leads else None,
            "fa_rate": float(day_alert.mean()) if len(day_alert) else None, "eligible_host_days": int(len(day_alert)),
            "pr_auc": pr_auc(y, scores) if y.any() else None, "positives": int(y.sum()), "rows": int(len(y))}


def unresolved_rule(cfg):
    return cfg["label"].get("unresolved_horizon", "negative")       # absent in the 1.5 config


def load_dev(cfg):
    sp = split()
    last = sp["val_days"][1]
    hh, comp = read_host_hours(last)
    feats = base_features()
    fc = cfg["features"]
    hh = add_trailing(hh, feats, fc["trailing_hours"], fc["active_hours"])
    cols = feats + [c for c in hh.columns if c.endswith(f"_sum{fc['trailing_hours']}h") or c.startswith("active_hours")]
    out = {}
    for p in ("train", "val"):
        f, hosts = period_frame(hh, comp, sp[f"{p}_days"], cfg["label"]["horizon_seconds"], unresolved_rule(cfg))
        out[p] = {"X": f[cols].to_numpy(np.float32), "frame": f[["computer", "hour", "day", "_y", "_t", "_eligible_day"]],
                  "hosts": hosts}
    return out, cols, sp


def _grid(spec):
    g = spec.get("grid", {})
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))] or [{}]


def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _key(r):
    return (r["early_alert_rate"] if r["early_alert_rate"] is not None else -1, r["pr_auc"] or 0.0)


def develop(name, cfg_path=CONFIG, log=print):
    cfg = load_config(cfg_path)
    if str(cfg["version"]) == "1.5":
        raise RuntimeError("the 1.5 Level 3 rules were superseded by amendment 1.6; their records and freeze are "
                           "kept unchanged and are not developed further")
    version = str(cfg["version"])
    spec, budget = cfg["candidates"][name], float(cfg["false_alarm"]["budget"])
    if spec["family"] == "neural" and not os.environ.get("PHD_ALLOW_CPU_NEURAL"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name} needs a GPU runtime (Runtime -> Change runtime type -> T4 GPU)")
    data, cols, sp = load_dev(cfg)
    tr, va = data["train"], data["val"]
    rp = record_path(name, version)
    rec = json.loads(rp.read_text()) if rp.exists() else None
    if rec and rec["models_config_sha256"] != sha256_file(cfg_path):
        raise RuntimeError(f"{rp.name} was made under another Level 3 config; it cannot be resumed")
    if rec and rec["source_provenance"] != near.provenance(DS):
        raise RuntimeError(f"{DS} outputs changed since {rp.name} was started")
    if rec is None:
        rec = {"candidate": name, "source": DS, "rules_version": version, "spec": spec,
               "models_config_sha256": sha256_file(cfg_path),
               "seeds": seeds(), "source_provenance": near.provenance(DS), "split": sp, "features": cols,
               "rows": {p: {"rows": int(len(d["frame"])), "positives": int(d["frame"]["_y"].sum()),
                            "compromised_hosts": int(len(d["hosts"]))} for p, d in data.items()},
               "grid": [], "final": {}, "complete": False, "runs": []}
    if rec["complete"]:
        log(f"[lanl/{name}] already complete")
        return rec
    rec["runs"].append(run_info(Path(cfg_path)))
    grid, gseed = _grid(spec), cfg["development"]["grid_seed"]
    done = {json.dumps(g["params"], sort_keys=True) for g in rec["grid"]}
    yt, yv = tr["frame"]["_y"].to_numpy(), va["frame"]["_y"].to_numpy()
    for params in grid:
        if json.dumps(params, sort_keys=True) in done:
            continue
        m = make(name, spec, params, gseed).fit(tr["X"], yt, va["X"], yv)
        s = m.predict_proba(va["X"])
        thr, n_days = fa_threshold(s, va["frame"], budget)
        ev = evaluate(s, va["frame"], va["hosts"], thr)
        rec["grid"].append({"params": params, "seed": gseed, "threshold": thr, "val": ev, "info": m.info})
        write_json(rp, rec)
        log(f"[lanl/{name}] grid {len(rec['grid'])}/{len(grid)} {params}: early-alert {_fmt(ev['early_alert_rate'])} "
            f"at FA {_fmt(ev['fa_rate'], 4)}, PR-AUC {_fmt(ev['pr_auc'], 4)} ({m.info['train_seconds']} s)")
    order = {json.dumps(g, sort_keys=True): i for i, g in enumerate(grid)}
    scored = sorted(rec["grid"], key=lambda g: order[json.dumps(g["params"], sort_keys=True)])
    best = max(scored, key=lambda g: _key(g["val"]))       # max keeps the first of equal keys: grid order
    rec["selected_params"] = best["params"]
    for sd in seeds():
        if str(sd) in rec["final"]:
            continue
        m = make(name, spec, best["params"], sd).fit(tr["X"], yt, va["X"], yv)
        s = m.predict_proba(va["X"])
        thr, _ = fa_threshold(s, va["frame"], budget)
        path = model_path(name, sd, version)
        sha = m.save(path)
        rec["final"][str(sd)] = {"threshold": thr, "val": evaluate(s, va["frame"], va["hosts"], thr), "info": m.info,
                                 "model_file": path.relative_to(paths.MODELS).as_posix(), "model_sha256": sha}
        write_json(rp, rec)
        log(f"[lanl/{name}] seed {sd}: early-alert {_fmt(rec['final'][str(sd)]['val']['early_alert_rate'])}")
    v = [rec["final"][str(sd)]["val"]["early_alert_rate"] for sd in seeds()]
    pa = [rec["final"][str(sd)]["val"]["pr_auc"] for sd in seeds()]
    if any(x is None for x in v + pa):
        raise RuntimeError("val has no compromised host or no positive host-hour; Level 3 cannot be developed")
    rec.update({"val_early_alert_rate_mean": float(np.mean(v)), "val_early_alert_rate_std": float(np.std(v, ddof=1)),
                "val_pr_auc_mean": float(np.mean(pa)), "complete": True,
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_json(rp, rec)
    log(f"[lanl/{name}] done: val early-alert rate {rec['val_early_alert_rate_mean']:.3f} ± "
        f"{rec['val_early_alert_rate_std']:.3f}")
    return rec


def summary_table(cfg_path=CONFIG):
    cfg = load_config(cfg_path)
    rows = []
    for name, spec in cfg["candidates"].items():
        rp = record_path(name, cfg["version"])
        r = json.loads(rp.read_text()) if rp.exists() else {}
        rows.append({"candidate": name, "complete": r.get("complete", False),
                     "grid_done": f"{len(r.get('grid', []))}/{len(_grid(spec))}", "seeds_done": len(r.get("final", {})),
                     "selected": r.get("selected_params"), "val_early_alert_rate_mean": r.get("val_early_alert_rate_mean"),
                     "val_early_alert_rate_std": r.get("val_early_alert_rate_std"), "val_pr_auc_mean": r.get("val_pr_auc_mean"),
                     "val_hosts_alerted": [(f["val"]["hosts_alerted"], f["val"]["hosts"]) for f in r.get("final", {}).values()]})
    return rows


def load_model(name, seed, expected_sha=None, version=None):
    path = model_path(name, seed, version)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"{path} differs from its recorded SHA-256")
    return Candidate.load(path)
