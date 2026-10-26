"""LANL Cyber1 -> host-hour feature table, red-team compromises and the frozen Level 3 split.

Operational definitions (docs/protocol/preregistration.md §7): one row per (computer, hour) built from auth, proc,
flows and dns events of that hour; a compromise is the first red-team event whose destination is the host. The
files are time-ordered and large (auth is about 1 billion lines), so they are streamed in chunks and every hour
is aggregated once it is complete. Each source writes its own parquet, so an interrupted run resumes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import paths
from ..utils.io import sha256_file, write_json, write_parquet
from ..utils.logging import run_info
from .load import CONFIG_PATH, dataset_fingerprint, load_config

COLUMNS = {
    "auth": ["time", "src_user", "dst_user", "src_comp", "dst_comp", "auth_type", "logon_type", "orientation",
             "outcome"],
    "proc": ["time", "user", "comp", "process", "action"],
    "flows": ["time", "duration", "src_comp", "src_port", "dst_comp", "dst_port", "protocol", "pkts", "bytes"],
    "dns": ["time", "src_comp", "resolved"],
    "redteam": ["time", "user", "src_comp", "dst_comp"],
}
USE = {"auth": ["time", "src_user", "src_comp", "dst_comp", "auth_type", "logon_type", "orientation", "outcome"],
       "proc": ["time", "comp", "process", "action"],
       "flows": ["time", "src_comp", "dst_comp", "pkts", "bytes"],
       "dns": ["time", "src_comp", "resolved"]}
LOGON_TYPES = ["Network", "Interactive", "RemoteInteractive", "Batch", "Service", "Unlock", "NewCredentials",
               "CachedInteractive"]
AUTH_TYPES = ["NTLM", "Kerberos", "Negotiate"]
DAY = 86400


class Codebook:
    """Persistent string -> int32 codes across chunks (computer and user names)."""

    def __init__(self):
        self.index = pd.Index([], dtype=object)

    def encode(self, s):
        if isinstance(s.dtype, pd.CategoricalDtype):         # pyarrow dictionary columns: map categories once
            cats = self._encode_values(pd.Series(s.cat.categories, dtype=object))
            codes = s.cat.codes.to_numpy()
            return np.where(codes >= 0, cats[np.maximum(codes, 0)], -1).astype(np.int32)
        return self._encode_values(s.astype(object))

    def _encode_values(self, s):
        new = pd.Index(pd.unique(s)).difference(self.index)
        if len(new):
            self.index = self.index.append(new)
        return self.index.get_indexer(s).astype(np.int32)

    def decode(self, codes):
        return self.index.to_numpy()[codes]


def _count(df, key, name, mask=None):
    d = df if mask is None else df[mask]
    return d.groupby([key, "hour"]).size().rename(name)


def _distinct(df, key, col, name):
    return df[[key, "hour", col]].drop_duplicates().groupby([key, "hour"]).size().rename(name)


class NewPairs:
    """Counts first-ever occurrences of (a, b) pairs, in time order."""

    def __init__(self):
        self.seen = set()

    def __call__(self, df, a, b):
        first = df.groupby([a, b], sort=False)["hour"].min().reset_index()
        packed = first[a].to_numpy(np.int64) << 32 | first[b].to_numpy(np.int64)
        fresh = np.fromiter((p not in self.seen for p in packed), bool, len(packed))
        self.seen.update(packed[fresh].tolist())
        return first[fresh]


def aggregate(kind, df, state):
    """Host-hour aggregates of one complete-hours block of events. Returns a frame indexed by (comp, hour)."""
    out = []
    if kind == "auth":
        fail = df.outcome == "Fail"
        out += [_count(df, "dst_comp", "auth_in"), _count(df, "dst_comp", "auth_in_fail", fail),
                _distinct(df, "dst_comp", "src_comp", "auth_in_distinct_src"),
                _distinct(df, "dst_comp", "src_user", "auth_in_distinct_users"),
                _count(df, "src_comp", "auth_out"), _count(df, "src_comp", "auth_out_fail", fail),
                _distinct(df, "src_comp", "dst_comp", "auth_out_distinct_dst")]
        out += [_count(df, "dst_comp", f"logon_{t}", df.logon_type == t) for t in LOGON_TYPES]
        out += [_count(df, "dst_comp", f"authtype_{t}", df.auth_type == t) for t in AUTH_TYPES]
        remote = df[df.src_comp != df.dst_comp]
        new = state["edges"](remote, "src_comp", "dst_comp")
        out += [new.groupby(["dst_comp", "hour"]).size().rename("auth_in_new_src"),
                new.groupby(["src_comp", "hour"]).size().rename("auth_out_new_dst")]
        out = [s.rename_axis(["comp", "hour"]) for s in out]
    elif kind == "proc":
        start = df.action == "Start"
        new = state["procs"](df[start], "comp", "process")
        out = [_count(df, "comp", "proc_start", start), _distinct(df[start], "comp", "process", "proc_distinct"),
               new.groupby(["comp", "hour"]).size().rename("proc_new")]
    elif kind == "flows":
        g_out = df.groupby(["src_comp", "hour"])
        g_in = df.groupby(["dst_comp", "hour"])
        out = [g_out.size().rename("flows_out"), g_out["bytes"].sum().rename("bytes_out"),
               g_out["pkts"].sum().rename("pkts_out"), _distinct(df, "src_comp", "dst_comp", "flows_out_distinct_dst"),
               g_in.size().rename("flows_in"), g_in["bytes"].sum().rename("bytes_in")]
        out = [s.rename_axis(["comp", "hour"]) for s in out]
    elif kind == "dns":
        out = [_count(df, "src_comp", "dns_queries"), _distinct(df, "src_comp", "resolved", "dns_distinct")]
        out = [s.rename_axis(["comp", "hour"]) for s in out]
    return pd.concat(out, axis=1)


def _encode(kind, chunk, books, hour_seconds):
    chunk["hour"] = (chunk["time"].astype(np.int64) // hour_seconds).astype(np.int32)
    for c in ("src_comp", "dst_comp", "comp", "resolved"):
        if c in chunk:
            chunk[c] = books["comp"].encode(chunk[c])
    if "src_user" in chunk:
        chunk["src_user"] = books["user"].encode(chunk["src_user"])
    if "process" in chunk:
        chunk["process"] = books["proc"].encode(chunk["process"])
    for c in ("pkts", "bytes"):
        if c in chunk:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").fillna(0).astype(np.float64)
    return chunk.drop(columns=["time"])


def _batches(path, kind, block_bytes=256 << 20):
    """Stream a gzipped LANL file as pandas frames (pyarrow parser, string columns dictionary-encoded)."""
    import pyarrow as pa
    from pyarrow import csv

    ro = csv.ReadOptions(column_names=COLUMNS[kind], block_size=block_bytes)
    co = csv.ConvertOptions(include_columns=USE[kind], auto_dict_encode=True, auto_dict_max_cardinality=1 << 30,
                            column_types={"time": pa.int64()})
    with pa.input_stream(str(path), compression="gzip") as f:
        for batch in csv.open_csv(f, read_options=ro, convert_options=co):
            yield batch.to_pandas()


def stream(kind, path, books, hour_seconds, chunksize=None, max_rows=None, log=print):
    """Aggregate one source file hour by hour. Returns (host-hour frame, rows read)."""
    state = {"edges": NewPairs(), "procs": NewPairs()}
    parts, pending, n, k = [], None, 0, 0
    for chunk in _batches(path, kind):
        n += len(chunk)
        k += 1
        chunk = _encode(kind, chunk, books, hour_seconds)
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        last = chunk["hour"].max()
        done, pending = chunk[chunk.hour < last], chunk[chunk.hour == last]
        if len(done):
            parts.append(aggregate(kind, done, state))
        if k % 25 == 0:
            log(f"[lanl {kind}] {n:,} rows, day {last * hour_seconds // DAY + 1}")
        if max_rows and n >= max_rows:
            break
    if pending is not None and len(pending):
        parts.append(aggregate(kind, pending, state))
    return pd.concat(parts).groupby(level=[0, 1]).sum(), n


def redteam(path):
    rt = pd.read_csv(path, header=None, names=COLUMNS["redteam"], compression="gzip")
    rt["day"] = rt["time"] // DAY + 1
    first = rt.sort_values("time").drop_duplicates("dst_comp")
    comp = first.rename(columns={"time": "compromise_time", "dst_comp": "comp"})[
        ["comp", "compromise_time", "day", "user", "src_comp"]]
    return rt, comp.reset_index(drop=True)


def level3_split(comp, spec):
    """Train/test day ranges per preregistration §7, including the contingency rule."""
    lo, hi = spec["train_days"], spec["test_days"]
    in_test = comp[(comp.day >= hi[0]) & (comp.day <= hi[1])]
    need = spec["min_test_compromised_hosts"]
    if len(in_test) >= need:
        return {"rule": "primary", "train_days": lo, "val_days": None, "test_days": hi,
                "test_compromised_hosts": int(len(in_test))}
    days = np.sort(comp.day.to_numpy())
    if len(days) < need:
        raise ValueError(f"only {len(days)} compromised hosts in total; Level 3 needs at least {need}")
    cut = int(days[-need])                      # latest day with >= need hosts first compromised on or after it
    early = cut - 1
    n_train = max(1, int(np.floor(spec["train_frac_of_early"] * early)))
    return {"rule": "contingency (fewer than %d compromised hosts in days %d-%d)" % (need, hi[0], hi[1]),
            "train_days": [1, n_train], "val_days": [n_train + 1, early] if early > n_train else None,
            "test_days": [cut, hi[1]], "test_compromised_hosts": int((comp.day >= cut).sum())}


def preprocess_lanl(config_path=CONFIG_PATH, max_rows=None, sources=("auth", "proc", "flows", "dns"), log=print):
    cfg = load_config(config_path)
    ds = cfg["datasets"]["lanl_cyber1"]
    raw = paths.RAW / "lanl_cyber1"
    out = paths.PROCESSED / "lanl_cyber1" / "v1"
    out.mkdir(parents=True, exist_ok=True)
    books = {"comp": Codebook(), "user": Codebook(), "proc": Codebook()}
    rt, comp = redteam(raw / "redteam.txt.gz")
    books["comp"].encode(pd.concat([comp["comp"], comp["src_comp"]]))
    tables, rows = [], {}
    for kind in sources:
        tag = "" if max_rows is None else f"_pilot{max_rows}"
        part = out / f"{kind}_host_hour{tag}.parquet"
        books_file = out / f"codebook_comp{tag}.parquet"
        if part.exists() and books_file.exists():
            log(f"[lanl {kind}] reusing {part.name}")
            t = pd.read_parquet(part)
            books["comp"].index = pd.Index(pd.read_parquet(books_file)["name"].to_numpy(), dtype=object)
        else:
            t, rows[kind] = stream(kind, raw / f"{kind}.txt.gz", books, ds["hour_seconds"], max_rows=max_rows, log=log)
            t = t.reset_index()
            t.to_parquet(part, index=False)
            pd.DataFrame({"name": books["comp"].index.to_numpy()}).to_parquet(books_file, index=False)
        tables.append(t.set_index(["comp", "hour"]))
    hh = pd.concat(tables, axis=1).fillna(0).astype("float32").reset_index()
    hh["computer"] = books["comp"].decode(hh["comp"].to_numpy())
    hh["day"] = (hh["hour"] * ds["hour_seconds"] // DAY + 1).astype(np.int16)
    hh = hh.drop(columns=["comp"])
    split = level3_split(comp, ds["split"])
    tag = "" if max_rows is None else f"_pilot{max_rows}"
    shas = {f"host_hour{tag}.parquet": write_parquet(hh, out / f"host_hour{tag}.parquet"),
            "compromises.parquet": write_parquet(comp, out / "compromises.parquet")}
    per_day = rt.groupby("day").size().to_dict()
    first_per_day = comp.groupby("day").size().to_dict()
    report = {"dataset": "lanl_cyber1", "role": "temporal", "protocol_version": cfg["version"],
              "run": run_info(Path(config_path)), "max_rows": max_rows,
              "dataset_config_sha256": dataset_fingerprint(cfg, "lanl_cyber1"),
              "inputs": {f.name: {"bytes": f.stat().st_size} for f in sorted(raw.glob("*.txt.gz"))},
              "rows_read": rows, "redteam_events": int(len(rt)), "compromised_hosts": int(len(comp)),
              "redteam_accounts": int(rt["user"].nunique()),
              "redteam_events_per_day": per_day, "first_compromises_per_day": first_per_day,
              "host_hours": int(len(hh)), "computers": int(hh["computer"].nunique()),
              "features": [c for c in hh.columns if c not in ("computer", "hour", "day")],
              "level3_split": split, "outputs": {"dir": str(out), "parquet_sha256": shas}}
    write_json(paths.SPLITS / "lanl_cyber1" / "level3_split.json", split)
    write_json(paths.RESULTS / "data_audit" / "lanl_cyber1" / "level3_split.json", split)   # same record on GitHub
    write_json(paths.RESULTS / "data_audit" / "lanl_cyber1" / "report.json", report)
    (paths.DATA_CARDS / "lanl_cyber1.md").write_text(render_card(report))
    return report


def render_card(r):
    s = r["level3_split"]
    lines = ["# LANL Comprehensive, Multi-Source Cyber-Security Events", "",
             "_Generated by `scripts/preprocess.py --dataset lanl_cyber1`; do not edit by hand._", "",
             "- **Citation:** Kent, A. D. Comprehensive, Multi-Source Cyber-Security Events. LANL, 2015. "
             "doi:10.17021/1179829",
             f"- **Code commit:** `{r['run']['git_commit']}`; config SHA-256 `{r['run'].get('config_sha256')}`",
             f"- **Unit:** host-hour; {r['host_hours']:,} host-hours over {r['computers']:,} computers"
             + (f" (pilot, first {r['max_rows']:,} rows per source)" if r.get("max_rows") else ""),
             f"- **Red team:** {r['redteam_events']:,} events, {r['compromised_hosts']:,} compromised hosts, "
             f"{r['redteam_accounts']:,} accounts",
             f"- **Level 3 split ({s['rule']}):** train days {s['train_days']}, val days {s['val_days']}, "
             f"test days {s['test_days']}; {s['test_compromised_hosts']} compromised hosts in test",
             "", "## Rows read", ""] + [f"- {k}: {v:,}" for k, v in r["rows_read"].items()] + [
             "", "## First compromises per day", "", "| day | hosts |", "|---:|---:|"] + [
             f"| {d} | {n} |" for d, n in sorted(r["first_compromises_per_day"].items())] + [
             "", "## Features", "", ", ".join(f"`{f}`" for f in r["features"])]
    return "\n".join(lines) + "\n"
