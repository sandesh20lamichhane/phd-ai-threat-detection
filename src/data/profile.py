"""Fast schema check of every dataset against config/preprocessing.yaml (preregistration §9 item 4).

Reads only headers and small samples, so it runs in minutes. Writes results/data_audit/schema_profile.{json,md}.
Nothing here trains or scores a model.
"""
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import paths
from ..utils.io import write_json
from .clean import identifier_columns, normalise_name
from .labels import UnmappedLabel, compile_rules, map_labels
from .load import CONFIG_PATH, _epoch_seconds, discover, load_config

SAMPLE_ROWS = 5000
MAX_FILES_SAMPLED = 40


def _read_sample(src, ds, nrows):
    header = None if ds.get("header") == "none" else 0
    with src.open() as f:
        return pd.read_csv(f, header=header, names=ds.get("columns") if header is None else None, nrows=nrows,
                           dtype=str, keep_default_na=False, encoding_errors="replace", on_bad_lines="skip")


def _tree(ds_id):
    out = {}
    for base in (paths.RAW, paths.INTERIM):
        root = base / ds_id
        if not root.exists():
            continue
        c = Counter()
        size = Counter()
        for p in root.rglob("*"):
            if p.is_file():
                key = p.parent.relative_to(root).as_posix() + "/*" + (p.suffix or "")
                c[key] += 1
                size[key] += p.stat().st_size
        out[base.name] = {k: {"files": c[k], "gb": round(size[k] / 1e9, 3)} for k in sorted(c)}
        for z in root.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                dirs = Counter(Path(i.filename).parent.as_posix() for i in zf.infolist() if not i.is_dir())
            out.setdefault("zip_members", {})[z.name] = dict(sorted(dirs.items()))
    return out


def profile_table(ds_id, ds, rules):
    sources = discover(ds)
    rep = {"files_matched": len(sources), "gb_matched": round(sum(s.size for s in sources) / 1e9, 3),
           "first_files": [s.rel for s in sources[:5]], "tree": _tree(ds_id)}
    if not sources:
        rep["problem"] = "no files match files.include; check the tree above"
        return rep
    step = max(1, len(sources) // MAX_FILES_SAMPLED)
    picked = sources[::step][:MAX_FILES_SAMPLED]
    frames, cols_per_file = [], {}
    for s in picked:
        df = _read_sample(s, ds, SAMPLE_ROWS)
        df.columns = [normalise_name(c) for c in df.columns]
        cols_per_file[s.rel] = list(df.columns)
        df["__file"] = s.rel
        frames.append(df)
    first_cols = cols_per_file[picked[0].rel]
    rep["columns"] = first_cols
    rep["files_with_different_columns"] = [f for f, c in cols_per_file.items() if c != first_cols]
    sample = pd.concat(frames, ignore_index=True)
    rep["rows_sampled"] = int(len(sample))
    checks = {}
    lab = ds.get("label", {})
    if "column" in lab:
        col = normalise_name(lab["column"])
        checks["label_column_present"] = col in sample
        raw = sample[col] if col in sample else pd.Series([], dtype=str)
    elif "from_filename" in lab or "from_path" in lab:
        rx = re.compile(lab.get("from_filename") or lab["from_path"])

        def from_name(f):
            inner = f.split("::")[-1]
            m = rx.match(Path(inner).name) if "from_filename" in lab else rx.search(inner)
            return m.group("label") if m else None
        raw = sample["__file"].map(from_name)
        checks["label_from_path_ok"] = bool(raw.notna().all())
    else:
        raw = pd.Series([], dtype=str)          # reference-only dataset: no label rules
    if len(raw) and lab:
        try:
            fam, mapping = map_labels(raw, compile_rules(rules[lab["rules"]]))
            rep["label_values_sampled"] = dict(Counter(raw.fillna("").astype(str).str.strip()).most_common(60))
            rep["label_map_sampled"] = mapping
            checks["all_sampled_labels_mapped"] = True
        except UnmappedLabel as e:
            checks["all_sampled_labels_mapped"] = False
            rep["unmapped_labels"] = str(e)
    bf = ds.get("binary_from")
    if isinstance(bf, dict):
        checks["binary_column_present"] = normalise_name(bf["column"]) in sample
    t = ds.get("time")
    if t:
        col = normalise_name(t["column"])
        checks["time_column_present"] = col in sample
        if col in sample:
            ts = _epoch_seconds(sample[col], t.get("unit"))
            ok = np.isfinite(ts)
            checks["time_parsed_share"] = float(ok.mean())
            if ok.any():
                rep["time_range_sampled"] = [str(pd.to_datetime(np.nanmin(ts), unit="s")),
                                             str(pd.to_datetime(np.nanmax(ts), unit="s"))]
            rep["time_examples"] = sample[col].head(3).tolist()
    d = ds.get("duration")
    if d:
        checks["duration_column_present"] = normalise_name(d["column"]) in sample
    explicit = [normalise_name(c) for c in ds.get("identifier_columns", [])]
    rep["identifier_columns_found"] = sorted(set(identifier_columns(first_cols)) |
                                            {c for c in first_cols if c in explicit})
    rep["configured_identifiers_absent"] = [c for c in explicit if c not in first_cols]
    rep["drop_columns_absent"] = [c for c in ds.get("drop_columns", []) if normalise_name(c) not in first_cols]
    rep["categorical_absent"] = [c for c in ds.get("categorical", []) if normalise_name(c) not in first_cols]
    num_like = []
    for c in first_cols:
        if c in rep["identifier_columns_found"] or c == normalise_name(lab.get("column", "")):
            continue
        v = pd.to_numeric(sample[c].replace({"Infinity": "inf", "-Infinity": "-inf"}), errors="coerce")
        share = float(v.notna().mean())
        if share < 0.99:
            num_like.append({"column": c, "numeric_share": round(share, 3),
                             "examples": sample.loc[v.isna(), c].astype(str).value_counts().head(5).index.tolist()})
    rep["non_numeric_columns"] = num_like
    rep["checks"] = checks
    return rep


def profile_lanl():
    raw = paths.RAW / "lanl_cyber1"
    rep = {"files": {p.name: round(p.stat().st_size / 1e9, 3) for p in sorted(raw.glob("*.gz"))}}
    rt = pd.read_csv(raw / "redteam.txt.gz", header=None, names=["time", "user", "src", "dst"])
    rt["day"] = rt.time // 86400 + 1
    first = rt.sort_values("time").drop_duplicates("dst")
    rep["redteam"] = {"events": int(len(rt)), "compromised_hosts": int(len(first)), "accounts": int(rt.user.nunique()),
                      "first_day": int(rt.day.min()), "last_day": int(rt.day.max()),
                      "first_compromises_per_day": {int(k): int(v) for k, v in first.groupby("day").size().items()},
                      "hosts_first_compromised_in_days_41_58": int(((first.day >= 41) & (first.day <= 58)).sum())}
    rep["heads"] = {}
    for name in ("auth", "proc", "flows", "dns"):
        rep["heads"][name] = pd.read_csv(raw / f"{name}.txt.gz", header=None, nrows=3).astype(str).values.tolist()
    return rep


def profile_parquets(ds_id):
    import pyarrow.parquet as pq
    out = {}
    for p in sorted((paths.RAW / ds_id).glob("*.parquet")):
        md = pq.ParquetFile(p).metadata
        out[p.name] = {"rows": md.num_rows, "columns": [md.schema.column(i).name for i in range(md.num_columns)]}
    return out


def profile_all(config_path=CONFIG_PATH, only=None, log=print):
    cfg = load_config(config_path)
    out = {}
    for ds_id, ds in cfg["datasets"].items():
        if only and ds_id not in only:
            continue
        log(f"profiling {ds_id}")
        try:
            if ds.get("reader") == "lanl":
                out[ds_id] = profile_lanl()
            elif ds_id == "ids_adapt_processed_legacy":
                out[ds_id] = profile_parquets(ds_id)
            else:
                out[ds_id] = profile_table(ds_id, ds, cfg["label_rules"] if "label" in ds else {})
        except Exception as e:                      # report and carry on with the other datasets
            out[ds_id] = {"error": f"{type(e).__name__}: {e}"}
    write_json(paths.RESULTS / "data_audit" / "schema_profile.json", out)
    (paths.RESULTS / "data_audit" / "schema_profile.md").write_text(summary(out))
    return out


def summary(out):
    lines = ["# Schema profile", "", "_Generated by `scripts/preprocess.py --profile` (samples only)._", "",
             "| dataset | files | GB | checks | problems |", "|---|---:|---:|---|---|"]
    for k, r in out.items():
        if "error" in r:
            lines.append(f"| {k} |  |  |  | {r['error']} |")
            continue
        if "redteam" in r:
            rt = r["redteam"]
            lines.append(f"| {k} | {len(r['files'])} |  | red team days {rt['first_day']}-{rt['last_day']}, "
                         f"{rt['compromised_hosts']} hosts | {rt['hosts_first_compromised_in_days_41_58']} hosts "
                         "first compromised in days 41-58 |")
            continue
        if "files_matched" not in r:
            lines.append(f"| {k} | {len(r)} |  | parquet schema only |  |")
            continue
        bad = [c for c, v in r.get("checks", {}).items() if v is False]
        probs = bad + (["columns differ between files"] if r.get("files_with_different_columns") else []) + \
            ([r["problem"]] if "problem" in r else []) + ([r["unmapped_labels"]] if "unmapped_labels" in r else [])
        ok = ", ".join(f"{c}={v}" for c, v in r.get("checks", {}).items() if v is not False)
        lines.append(f"| {k} | {r['files_matched']} | {r['gb_matched']} | {ok} | {'; '.join(probs) or 'none'} |")
    return "\n".join(lines) + "\n"
