"""Find each dataset's files in Drive and read them in chunks as (features, metadata) pairs.

Everything a model must never see (identifiers, time, labels, audit fields) goes into the metadata frame, whose
columns all start with "_". The feature frame holds only model inputs. Rules come from config/preprocessing.yaml.
"""
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..utils import paths
from ..utils.io import HashingReader
from .clean import identifier_columns, normalise_name
from .labels import ATTEMPTED, binary_from_family, compile_rules, map_labels

CONFIG_PATH = paths.CONFIG / "preprocessing.yaml"
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def load_config(path=CONFIG_PATH):
    cfg = yaml.safe_load(Path(path).read_text())
    for ds_id, ds in cfg["datasets"].items():
        merged = dict(cfg["defaults"])
        merged.update(ds)
        merged["identifier_columns"] = list(cfg["defaults"].get("identifier_columns", [])) + list(
            ds.get("identifier_columns", []))
        merged["id"] = ds_id
        cfg["datasets"][ds_id] = merged
    return cfg


@dataclass
class Source:
    """One input file, either on disk or inside a zip in data/raw."""
    rel: str
    path: Path
    member: str = None
    size: int = 0
    stats: dict = field(default_factory=dict)

    def open(self):
        if self.member is None:
            return open(self.path, "rb")
        return zipfile.ZipFile(self.path).open(self.member)


def _root_dirs(ds):
    root = ds.get("files", {}).get("root", "interim")
    base = {"raw": paths.RAW, "interim": paths.INTERIM}[root]
    return base / ds["id"]


def _sort_key(rel):
    stem = Path(rel).name.split(".")[0].lower()
    return (WEEKDAYS.index(stem) if stem in WEEKDAYS else 99, rel.lower())


def discover(ds):
    """List the dataset's input files, sorted so that file order follows time where the name says so."""
    spec = ds.get("files", {})
    pat = re.compile(spec.get("include", r"\.csv$"))
    root = _root_dirs(ds)
    found = []
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and "_stage" not in p.parts:
                rel = p.relative_to(root).as_posix()
                if pat.search(rel) or pat.search(p.name):
                    found.append(Source(rel, p, size=p.stat().st_size))
    if not found and spec.get("zip_fallback"):
        z = paths.RAW / ds["id"] / spec["zip_fallback"]
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                for info in zf.infolist():
                    if not info.is_dir() and (pat.search(info.filename) or pat.search(Path(info.filename).name)):
                        found.append(Source(f"{z.name}::{info.filename}", z, member=info.filename,
                                            size=info.file_size))
    return sorted(found, key=lambda s: _sort_key(s.rel))


# ---------------------------------------------------------------------------------------------------------
def _numeric(series, missing_tokens):
    """Coerce to float64; returns (values, n_unparseable) where unparseable excludes declared missing tokens."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64"), 0
    s = series.astype("string").str.strip()
    out = pd.to_numeric(s, errors="coerce")
    low = {t.lower() for t in missing_tokens}
    is_tok = s.isna() | s.str.lower().isin(low)
    out = out.where(~s.str.lower().isin({"infinity", "inf"}), np.inf)
    out = out.where(~s.str.lower().isin({"-infinity", "-inf"}), -np.inf)
    bad = int((out.isna() & ~is_tok).sum())
    return out.astype("float64"), bad


def _port(series):
    """Ports as float; handles UNSW hex strings ('0x000b') and '-'."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    s = series.astype("string").str.strip()
    hexmask = s.str.lower().str.startswith("0x").fillna(False)
    out = pd.to_numeric(s.where(~hexmask), errors="coerce")
    if hexmask.any():
        out[hexmask] = s[hexmask].map(lambda v: int(v, 16)).astype("float64")
    return out.astype("float64")


def _epoch_seconds(series, unit=None):
    if unit == "s" or pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()
    t = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    out = (t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    return out.astype("float64").to_numpy()


class Standardiser:
    """Turns raw chunks of one dataset into (features, meta). Keeps running counts for the report."""

    def __init__(self, ds, label_rules):
        self.ds = ds
        self.rules = compile_rules(label_rules[ds["label"]["rules"]])
        self.missing = ds.get("missing_tokens", [])
        self.cat = {normalise_name(c) for c in ds.get("categorical", [])}
        self.label_map, self.unparseable, self.inf, self.overflow = {}, {}, {}, {}
        self.removed = {}
        self.time_unparsed = 0
        self.binary_disagreements = 0
        self.feature_columns = None

    def _classify_columns(self, columns, label_cols):
        ds = self.ds
        explicit = {normalise_name(c).lower() for c in ds.get("identifier_columns", [])}
        ids = set(identifier_columns(columns)) | {c for c in columns if c.lower() in explicit}
        drops = {c for c in columns if c.lower() in {normalise_name(d).lower() for d in ds.get("drop_columns", [])}}
        removed = {}
        for c in columns:
            if c in label_cols:
                removed[c] = "label"
            elif c in ids:
                removed[c] = "identifier (kept as metadata where configured)"
            elif c in drops:
                removed[c] = "excluded by protocol (label-derived, free text or meaningless)"
        return [c for c in columns if c not in removed], removed

    def __call__(self, df, src, file_idx, row0):
        ds = self.ds
        df.columns = [normalise_name(c) for c in df.columns]
        lab = ds["label"]
        label_cols = set()
        if "column" in lab:
            raw = df[normalise_name(lab["column"])]
            label_cols.add(normalise_name(lab["column"]))
        else:
            m = re.match(lab["from_filename"], Path(src.rel.split("::")[-1]).name)
            if not m:
                raise ValueError(f"cannot read a label from file name {src.rel}")
            raw = pd.Series([m.group("label")] * len(df), index=df.index)
        family, mapping = map_labels(raw, self.rules)
        self.label_map.update(mapping)

        y = binary_from_family(family)
        bf = ds.get("binary_from", "family")
        if isinstance(bf, dict):
            col = normalise_name(bf["column"])
            label_cols.add(col)
            yb = pd.to_numeric(df[col], errors="coerce").fillna(-9).astype(int).to_numpy()
            fam_np = family.to_numpy()
            valid = fam_np != ATTEMPTED
            self.binary_disagreements += int(((yb != y) & valid).sum())
            y = np.where(valid, yb, -1).astype(np.int8)   # the official binary label wins; disagreement reported

        meta = pd.DataFrame(index=df.index)
        meta["_file"] = np.int16(file_idx)
        meta["_row"] = np.arange(row0, row0 + len(df), dtype=np.int64)
        meta["_label_raw"] = raw.fillna("").astype(str).str.strip().to_numpy()
        meta["_label"] = family.to_numpy()
        meta["_y"] = y

        tcfg = ds.get("time")
        if tcfg:
            ts = _epoch_seconds(df[normalise_name(tcfg["column"])], tcfg.get("unit"))
            self.time_unparsed += int(np.isnan(ts).sum())
            meta["_ts"] = ts
        dcfg = ds.get("duration")
        if dcfg and normalise_name(dcfg["column"]) in df:
            dur = pd.to_numeric(df[normalise_name(dcfg["column"])], errors="coerce").to_numpy(dtype="float64")
            meta["_dur"] = dur * float(dcfg.get("unit_seconds", 1.0))
        day_from = ds.get("day_from")
        if day_from == "filename":
            meta["_day"] = Path(src.rel.split("::")[-1]).name.split(".")[0].capitalize()
        elif day_from == "time" and "_ts" in meta:
            meta["_day"] = pd.to_datetime(meta["_ts"], unit="s", errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        for col, name in (ds.get("keep_meta") or {}).items():
            col = normalise_name(col)
            if col in df:
                meta[name] = _port(df[col])
        part = ds.get("official_partition_from_path")
        if part:
            hit = [k for k, v in part.items() if v.lower() in src.rel.lower()]
            if len(hit) != 1:
                raise ValueError(f"cannot tell the official partition of {src.rel}")
            meta["_partition_official"] = hit[0]

        feats, removed = self._classify_columns(list(df.columns), label_cols)
        self.removed.update(removed)
        if self.feature_columns is None:
            self.feature_columns = feats
        elif feats != self.feature_columns:
            extra = sorted(set(feats) - set(self.feature_columns))
            miss = sorted(set(self.feature_columns) - set(feats))
            if extra or miss:
                raise ValueError(f"{src.rel}: feature columns differ from the first file (extra {extra}, missing {miss})")
            feats = self.feature_columns

        X = pd.DataFrame(index=df.index)
        fdt = ds.get("float_dtype", "float32")
        for c in feats:
            if c in self.cat:
                s = df[c].astype("string").str.strip()
                X[c] = s.mask(s.isin(self.missing)).astype(object)
                continue
            v, bad = _numeric(df[c], self.missing)
            if bad:
                self.unparseable[c] = self.unparseable.get(c, 0) + bad
            arr = v.to_numpy()
            ninf = int(np.isinf(arr).sum())
            if ninf:
                self.inf[c] = self.inf.get(c, 0) + ninf
                arr = np.where(np.isinf(arr), np.nan, arr)
            with np.errstate(over="ignore"):
                cast = arr.astype(fdt)
            over = int((np.isinf(cast) & np.isfinite(arr)).sum())
            if over:
                self.overflow[c] = self.overflow.get(c, 0) + over
                cast = np.where(np.isinf(cast), np.nan, cast).astype(fdt)
            X[c] = cast
        return X.reset_index(drop=True), meta.reset_index(drop=True)

    def report(self):
        return {"label_map": dict(sorted(self.label_map.items())),
                "removed_columns": self.removed,
                "feature_columns": self.feature_columns,
                "unparseable_values_to_nan": self.unparseable,
                "inf_values_to_nan": self.inf,
                "float32_overflow_to_nan": self.overflow,
                "timestamps_unparsed": self.time_unparsed,
                "binary_label_disagreements": self.binary_disagreements}


def _string_columns(ds, columns):
    """Raw column names that must be read as text: labels, categoricals, identifiers kept as metadata."""
    want = {normalise_name(c).lower() for c in ds.get("categorical", [])}
    lab = ds.get("label", {})
    if "column" in lab:
        want.add(normalise_name(lab["column"]).lower())
    want |= {normalise_name(c).lower() for c in (ds.get("keep_meta") or {})}
    t = ds.get("time") or {}
    if t and t.get("unit") != "s":
        want.add(normalise_name(t["column"]).lower())
    return {c: str for c in columns if normalise_name(c).lower() in want}


def read_chunks(ds, sources, standardiser):
    """Yield (features, meta, source) for every chunk of every file; fills source.stats with sha256/bytes/rows.

    Numeric columns are parsed by pandas' C parser ("inf"/"Infinity" become inf); a column holding stray text
    arrives as object and is coerced in Standardiser, which counts what could not be parsed.
    """
    header = None if ds.get("header") == "none" else 0
    names = ds.get("columns") if header is None else None
    na = [t for t in ds.get("missing_tokens", []) if t.lower() not in {"infinity", "-infinity", "inf", "-inf"}]
    for file_idx, src in enumerate(sources):
        if header is None:
            cols = names
        else:
            with src.open() as f:
                cols = list(pd.read_csv(f, nrows=0, encoding_errors="replace").columns)
        raw = src.open()
        hr = HashingReader(raw)
        buf = io.BufferedReader(hr, buffer_size=8 * 1024 * 1024)
        rows = 0
        reader = pd.read_csv(buf, header=header, names=names, chunksize=ds["chunksize"],
                             dtype=_string_columns(ds, cols), na_values=na, encoding_errors="replace",
                             on_bad_lines="warn", low_memory=False)
        for chunk in reader:
            X, meta = standardiser(chunk, src, file_idx, rows)
            rows += len(chunk)
            yield X, meta, src
        sha, nbytes = hr.finish()
        src.stats = {"rows": rows, "bytes_read": nbytes,
                     ("sha256" if src.member is None else "sha256_of_member"): sha}
        buf.close()
