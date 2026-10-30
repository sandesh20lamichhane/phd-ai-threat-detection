"""Level 4 extraction support: extractor check and ToN_IoT labels (amendments 1.7-1.8).

Rules and rationale: docs/protocol/level4_extraction.md; values: config/level4_extraction.yaml.

    extractor_check     plan §3: re-extracted CICIDS2017 flows against the corrected CSV of the same day
    label_ton_flows     plan §4.3: CICFlowMeter flows matched to the ToN_IoT ground-truth connection records
    clock_offset        plan §4.4: offset between the capture clock and the ground-truth ts

Matching works on an endpoint key: protocol plus the unordered pair of endpoints ("address|port", address only
for ICMP). Each record covers the interval [ts - margin_before, ts + duration + margin_after]. For every flow,
`merge_asof` finds the record of the same key with the latest start at or before the flow start; a running
maximum of the interval ends tells whether an earlier record also covers the flow, and only those (rare) flows are
resolved one by one.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..utils import paths

CONFIG_PATH = paths.CONFIG / "level4_extraction.yaml"
GT_COLUMNS = ["ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "type"]
_INF = {"infinity": np.inf, "+infinity": np.inf, "inf": np.inf, "-infinity": -np.inf, "-inf": -np.inf}
_NAN = {"", "nan", "-", "none", "null"}


class Level4Stop(RuntimeError):
    """A stop rule of the extraction plan fired: nothing later may run."""


def load_config(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------- parsing

def numeric(series):
    """Numbers as written by CICFlowMeter or pandas: Infinity/-Infinity kept as +-inf, NaN tokens as NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    s = series.astype(str).str.strip()
    low = s.str.lower()
    out = pd.to_numeric(s.where(~low.isin(_NAN | set(_INF))), errors="coerce").astype("float64")
    for tok, val in _INF.items():
        out[low == tok] = val
    return out


def timestamp_us(series):
    """CICFlowMeter `Timestamp` -> integer microseconds since the epoch (the JVM runs in UTC, plan §2.2).

    Accepts `2017-07-06 08:59:50.181316` (the corrected releases) and `06/07/2017 08:59:50 AM` (original
    CICFlowMeter); a value that parses in neither format stops the run."""
    s = series.astype(str).str.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S"):
        t = pd.to_datetime(s, format=fmt, errors="coerce")
        if t.notna().all():
            return t.astype("datetime64[us]").astype("int64")
    bad = s[pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce").isna()].head(3).tolist()
    raise Level4Stop(f"unparseable Timestamp values, e.g. {bad}")


def _proto_name(proto, proto_map):
    if pd.api.types.is_numeric_dtype(proto):
        return proto.map({int(k): v for k, v in proto_map.items()})
    p = proto.astype(str).str.strip().str.lower()
    known = set(proto_map.values())
    return p.where(p.isin(known))


def endpoint_key(src_ip, src_port, dst_ip, dst_port, proto, proto_map, ports_ignored_for=("icmp",)):
    """uint64 key of protocol + unordered endpoint pair; NaN-protocol rows (not tcp/udp/icmp) get key 0.

    Also returns the normalised (proto, lo, hi) strings so a match can be checked against hash collisions."""
    name = _proto_name(proto, proto_map)
    ip_s, ip_d = src_ip.astype(str).str.strip(), dst_ip.astype(str).str.strip()
    port_s = pd.to_numeric(src_port, errors="coerce").astype("Int64").astype(str)
    port_d = pd.to_numeric(dst_port, errors="coerce").astype("Int64").astype(str)
    no_port = name.isin(list(ports_ignored_for))
    a = ip_s.where(no_port, ip_s + "|" + port_s)
    b = ip_d.where(no_port, ip_d + "|" + port_d)
    lo, hi = a.where(a <= b, b), b.where(a <= b, a)
    text = name.fillna("") + "#" + lo + "#" + hi
    key = pd.util.hash_pandas_object(text, index=False).to_numpy(np.uint64)
    key = np.where(name.isna().to_numpy(), np.uint64(0), key)
    return key, text.where(name.notna())


# ---------------------------------------------------------------- extractor check (plan §3)

def _mode(values):
    v, c = np.unique(np.asarray(values), return_counts=True)
    return (int(v[np.argmax(c)]), int(c.max()), int(c.sum())) if len(v) else (None, 0, 0)


def _agree(a, b, rel_tol, abs_tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    fin = np.isfinite(a) & np.isfinite(b)
    close = np.zeros(len(a), bool)
    close[fin] = np.abs(a[fin] - b[fin]) <= np.maximum(abs_tol, rel_tol * np.abs(b[fin]))
    return both_nan | both_inf | close


def extractor_check(extracted, reference, features, cfg):
    """Compare re-extracted flows with the corrected CSV of the same capture day. Returns the record of plan §3.

    `extracted` and `reference` are DataFrames with the key columns, `Timestamp` and the shared features."""
    key_cols, tcol = cfg["key"], cfg["time_column"]
    res_us = int(cfg.get("time_resolution_us", 1))

    def keyed(df):
        k = df[key_cols].astype(str).apply(lambda c: c.str.strip()).agg("#".join, axis=1)
        out = pd.DataFrame({"k": k.to_numpy(), "t": timestamp_us(df[tcol]) // res_us * res_us,
                            "row": np.arange(len(df))})
        return out

    e, r = keyed(extracted), keyed(reference)
    # clock offset from keys that occur once on each side (plan §3 step 2)
    ue = e[~e["k"].duplicated(keep=False)]
    ur = r[~r["k"].duplicated(keep=False)]
    pair = ue.merge(ur, on="k", suffixes=("_e", "_r"))
    diff = (pair["t_e"] - pair["t_r"]).to_numpy()
    diff = diff[np.abs(diff) <= 86_400 * 1_000_000]
    delta, n_mode, n_pairs = _mode(diff)
    record = {"rows_extracted": len(e), "rows_reference": len(r), "offset_pairs": n_pairs,
              "offset_us": delta, "offset_mode_count": n_mode}
    step = int(cfg["offset_multiple_s"]) * 1_000_000
    if delta is None or delta % step != 0:
        record.update(passed=False, reason="clock offset is not a whole multiple of "
                                           f"{cfg['offset_multiple_s']} s")
        return record

    e["t"] = e["t"] - delta
    for d in (e, r):
        d["rank"] = d.groupby(["k", "t"]).cumcount()
    m = r.merge(e, on=["k", "t", "rank"], suffixes=("_r", "_e"))
    matched_share = len(m) / max(len(r), 1)
    per_feature, worst = {}, []
    for f in features:
        a = numeric(extracted[f]).to_numpy()[m["row_e"].to_numpy()]
        b = numeric(reference[f]).to_numpy()[m["row_r"].to_numpy()]
        ok = _agree(a, b, cfg["rel_tol"], cfg["abs_tol"])
        per_feature[f] = float(ok.mean()) if len(ok) else 0.0
        fin = np.isfinite(a) & np.isfinite(b)
        worst.append((per_feature[f], f, float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else None))
    worst.sort(key=lambda x: (x[0], x[1]))
    passed = (matched_share >= cfg["min_matched_share"]
              and all(v >= cfg["min_feature_agreement"] for v in per_feature.values()))
    record.update(matched=len(m), matched_share=matched_share, feature_agreement=per_feature,
                  lowest_agreement=[{"feature": f, "agreement": v, "max_abs_difference": d}
                                    for v, f, d in worst[:20]],
                  min_matched_share=cfg["min_matched_share"],
                  min_feature_agreement=cfg["min_feature_agreement"], passed=bool(passed))
    if not passed:
        record["reason"] = "matched share or feature agreement below the preregistered minimum"
    return record


# ---------------------------------------------------------------- ToN_IoT ground truth (plan §4)

def read_ground_truth(files):
    """The 18 ground-truth network files, schema checked, exact duplicates removed (plan §4.1-4.2)."""
    parts = []
    for f in sorted(files, key=lambda p: (len(Path(p).name), Path(p).name)):
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        cols = [c.strip() for c in df.columns]
        if cols != GT_COLUMNS:
            raise Level4Stop(f"{Path(f).name}: columns {cols} differ from {GT_COLUMNS}")
        df.columns = cols
        parts.append(df.apply(lambda c: c.str.strip()))
    gt = pd.concat(parts, ignore_index=True)
    n = len(gt)
    gt = gt.drop_duplicates(ignore_index=True)
    gt["ts"] = pd.to_numeric(gt["ts"], errors="raise").astype("int64")
    return gt, {"rows_read": n, "exact_duplicates_removed": n - len(gt), "rows": len(gt)}


def attach_durations(gt, processed):
    """Duration of each ground-truth record from the processed Zeek record with the same ts, five-tuple, proto and
    type (largest if several, 0 if none or missing). `processed` holds those columns plus `duration`."""
    on = GT_COLUMNS
    p = processed[on + ["duration"]].astype({c: str for c in on if c != "ts"}).copy()
    p["ts"] = pd.to_numeric(p["ts"], errors="coerce").astype("Int64")
    for c in on[1:]:
        p[c] = p[c].str.strip()
    p["duration"] = numeric(p["duration"]).fillna(0.0).clip(lower=0.0)
    dur = p.groupby(on, dropna=False)["duration"].max().reset_index()
    g = gt.copy()
    g["ts"] = g["ts"].astype("Int64")
    out = g.merge(dur, on=on, how="left")
    joined = out["duration"].notna()
    out["duration"] = out["duration"].fillna(0.0)
    out["ts"] = out["ts"].astype("int64")
    return out, {"joined_share": float(joined.mean()) if len(out) else 0.0}


def intervals(records, cfg, label_col):
    """Sorted interval table: key, start, end (seconds), label; with running max of the end within each key."""
    key, text = endpoint_key(records["src_ip"], records["src_port"], records["dst_ip"], records["dst_port"],
                             records["proto"], cfg["proto_map"], cfg["ports_ignored_for"])
    dur = records["duration"].to_numpy(float) if "duration" in records else np.zeros(len(records))
    iv = pd.DataFrame({"key": key, "text": text.to_numpy(),
                       "start": records["ts"].to_numpy(float) - cfg["margin_before_s"],
                       "ts": records["ts"].to_numpy(float),
                       "end": records["ts"].to_numpy(float) + dur + cfg["margin_after_s"],
                       "label": records[label_col].to_numpy()})
    iv = iv[iv["key"] != 0].sort_values(["key", "start", "end"], kind="mergesort").reset_index(drop=True)
    g = iv.groupby("key", sort=False)["end"]
    iv["cummax_end"] = g.cummax()
    iv["prev_cummax_end"] = g.shift(1)
    iv["prev_cummax_end"] = iv.groupby("key", sort=False)["prev_cummax_end"].cummax().fillna(-np.inf)
    iv["pos"] = np.arange(len(iv))
    return iv


def match(flow_key, flow_text, flow_t, iv):
    """For each flow: matched?, the deciding label (latest-starting covering record), and whether the covering
    records carry more than one label (plan §4.3). `flow_t` is in seconds, already corrected by the offset."""
    n = len(flow_key)
    f = pd.DataFrame({"key": flow_key, "t": np.asarray(flow_t, float), "i": np.arange(n)})
    f = f[f["key"] != 0].sort_values("t", kind="mergesort")
    right = iv[["key", "start", "end", "label", "cummax_end", "prev_cummax_end", "pos", "text"]] \
        .sort_values("start", kind="mergesort")
    m = pd.merge_asof(f, right, left_on="t", right_on="start", by="key", direction="backward")
    matched = np.zeros(n, bool)
    label = np.full(n, None, dtype=object)
    multi = np.zeros(n, bool)
    has = m["pos"].notna().to_numpy()
    covered_any = has & (m["cummax_end"].to_numpy() >= m["t"].to_numpy())
    direct = has & (m["end"].to_numpy() >= m["t"].to_numpy())
    others = has & (m["prev_cummax_end"].to_numpy() >= m["t"].to_numpy())
    idx = m["i"].to_numpy()
    if (has & (m["text"].to_numpy() != np.asarray(flow_text, dtype=object)[idx])).any():
        raise Level4Stop("endpoint-key hash collision between a flow and a record")
    simple = direct & ~others
    matched[idx[simple]] = True
    label[idx[simple]] = m["label"].to_numpy()[simple]
    slow = np.flatnonzero(covered_any & ~simple)
    if len(slow):
        keys = iv["key"].to_numpy()
        starts, ends, labels = iv["start"].to_numpy(), iv["end"].to_numpy(), iv["label"].to_numpy()
        pos, ts_ = m["pos"].to_numpy(), m["t"].to_numpy()
        for j in slow:
            p, t = int(pos[j]), ts_[j]
            lo = int(np.searchsorted(keys, keys[p], side="left"))   # iv is sorted by key
            cand = np.arange(lo, p + 1)
            cov = cand[(starts[cand] <= t) & (ends[cand] >= t)]
            if not len(cov):
                continue
            last = cov[np.argmax(starts[cov] + 1e-12 * np.arange(len(cov)))]
            matched[idx[j]] = True
            label[idx[j]] = labels[last]
            multi[idx[j]] = len(set(labels[cov])) > 1
    return matched, label, multi


def clock_offset(flow_key, flow_t, gt_key, gt_ts, acfg):
    """Plan §4.4: mode of round(t_f) - ts_g over key-equal pairs within the window; keys with more pairs than
    the cap are left out. Returns (delta_s, record); raises Level4Stop if the mode is not an allowed offset."""
    f = pd.DataFrame({"key": flow_key, "t": np.round(np.asarray(flow_t, float)).astype("int64")})
    g = pd.DataFrame({"key": gt_key, "ts": np.asarray(gt_ts, "int64")})
    f, g = f[f["key"] != 0], g[g["key"] != 0]
    nf, ng = f["key"].value_counts(), g["key"].value_counts()
    both = nf.index.intersection(ng.index)
    pairs = nf[both] * ng[both]
    keep = pairs.index[pairs <= acfg["max_pairs_per_key"]]
    p = f[f["key"].isin(keep)].merge(g[g["key"].isin(keep)], on="key")
    d = (p["t"] - p["ts"]).to_numpy()
    d = d[np.abs(d) <= acfg["window_s"]]
    mode, n_mode, n = _mode(d)
    hist = {}
    if mode is not None:
        near = d[np.abs(d - mode) <= 5]
        v, c = np.unique(near, return_counts=True)
        hist = {int(a): int(b) for a, b in zip(v, c)}
    rec = {"pairs": int(n), "keys_left_out": int((pairs > acfg["max_pairs_per_key"]).sum()),
           "mode_s": mode, "mode_count": n_mode, "histogram_near_mode": hist}
    if mode is None:
        raise Level4Stop("no key-equal pairs between the first capture and the ground truth")
    if abs(mode) <= acfg["zero_if_within_s"]:
        rec["delta_s"] = 0
        return 0, rec
    mult = acfg["timezone_multiple_s"]
    nearest = int(round(mode / mult)) * mult
    if abs(mode - nearest) <= acfg["timezone_tolerance_s"]:
        rec["delta_s"] = nearest
        return nearest, rec
    rec["delta_s"] = None
    raise Level4Stop(f"clock offset {mode} s is neither ~0 nor a whole multiple of {mult} s: {rec}")


def flow_keys(flows, cfg):
    """Endpoint keys and start times (seconds) of extractor flows."""
    missing = [c for c in ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"] if c not in flows]
    if missing:
        raise Level4Stop(f"extractor output lacks {missing}")
    key, text = endpoint_key(flows["Src IP"], flows["Src Port"], flows["Dst IP"], flows["Dst Port"],
                             pd.to_numeric(flows["Protocol"], errors="coerce"), cfg["proto_map"],
                             cfg["ports_ignored_for"])
    return key, text, timestamp_us(flows["Timestamp"]).to_numpy() / 1e6


def label_ton_flows(flows, gt_iv, delta_s, cfg):
    """Raw labels for extractor flows (plan §4.3): the deciding record's `type`, or `normal` if none matches.

    Returns (labels DataFrame with Label, _gt_match, _gt_multi; key; text; corrected start time)."""
    key, text, t = flow_keys(flows, cfg)
    t = t - delta_s
    matched, label, multi = match(key, text, t, gt_iv)
    raw = np.where(matched, label, cfg["benign_raw_label"]).astype(object)
    out = pd.DataFrame({"Label": raw, "_gt_match": matched, "_gt_multi": multi}, index=flows.index)
    return out, key, text, t


def zeek_agreement(key, text, t, zeek_iv, flow_binary):
    """Plan §4.5: coverage of the flows by processed Zeek records (both classes) and the 2 x 2 table of the flow
    label against the Zeek `label` of the deciding record, with agreement and Cohen's kappa."""
    matched, zlabel, multi = match(key, text, t, zeek_iv)
    y = np.asarray(flow_binary, int)
    z = pd.to_numeric(pd.Series(zlabel), errors="coerce").to_numpy()
    ok = matched & ~np.isnan(z)
    yy, zz = y[ok], z[ok].astype(int)
    table = {f"flow{a}_zeek{b}": int(((yy == a) & (zz == b)).sum()) for a in (0, 1) for b in (0, 1)}
    n = len(yy)
    po = float((yy == zz).mean()) if n else None
    pe = float(yy.mean() * zz.mean() + (1 - yy.mean()) * (1 - zz.mean())) if n else None
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else None
    return {"flows": int(len(y)), "matched_share": float(matched.mean()) if len(y) else None,
            "multi_label_matches": int(multi.sum()), "table": table, "agreement": po, "cohen_kappa": kappa}, \
        matched, zlabel


def ground_truth_coverage(gt_iv, key, t, t_range):
    """Plan §4.5: share of ground-truth records (ts inside the capture time range) that cover at least one flow,
    per label. Records are matched to flows on the same key and interval."""
    lo, hi = t_range
    g = gt_iv[(gt_iv["ts"] >= lo) & (gt_iv["ts"] <= hi)][["key", "start", "end", "label"]].copy()
    g["gi"] = np.arange(len(g))
    f = pd.DataFrame({"key": key, "t": t})
    f = f[f["key"] != 0].sort_values("t", kind="mergesort")
    # for every record, the first flow of the same key starting at or after its start
    m = pd.merge_asof(g.sort_values("start", kind="mergesort"), f, left_on="start", right_on="t", by="key",
                      direction="forward")
    hit = m["t"].notna() & (m["t"] <= m["end"])
    per = m.assign(hit=hit).groupby("label")["hit"].agg(["size", "mean"])
    return {"records_in_range": int(len(g)), "covered_share": float(hit.mean()) if len(g) else None,
            "by_type": {str(k): {"records": int(v["size"]), "covered_share": float(v["mean"])}
                        for k, v in per.iterrows()}}
