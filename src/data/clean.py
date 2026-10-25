"""Data-quality rules applied to every dataset before training (docs/protocol/data_quality_rules.md).

1. drop identifier columns (flow IDs, IPs, ports, timestamps) so models learn behaviour, not hosts
2. normalise column names, turn +/-inf into NaN, drop rows that are entirely NaN
3. drop exact duplicate rows BEFORE any split
Every step is counted so the thesis can report exactly what was removed.
"""
import re

import numpy as np
import pandas as pd

IDENTIFIER_PATTERNS = [
    r"^flow[_ ]?id$", r"^(src|dst|source|destination)[_ ]?ip$", r"^(src|dst|source|destination)[_ ]?port$",
    r"^(src|dst)ip_?addr$", r"^ip_(src|dst)$", r"^(s|d)addr$", r"^(s|d)port$", r"^sport$", r"^dsport$",
    r"^timestamp$", r"^ts$", r"^(stime|ltime)$", r"^date(time)?$", r"^time$", r"^id$", r"^unnamed: ?\d+$",
]


def normalise_name(c):
    return re.sub(r"\s+", " ", str(c)).strip()


def normalise_columns(df):
    df = df.copy()
    df.columns = [normalise_name(c) for c in df.columns]
    return df


def identifier_columns(columns, extra=()):
    pats = [re.compile(p, re.I) for p in IDENTIFIER_PATTERNS] + [re.compile(p, re.I) for p in extra]
    return [c for c in columns if any(p.search(c.replace(" ", "_")) or p.search(c) for p in pats)]


def clean(df, label_col="Label", extra_identifiers=(), keep=()):
    """Return (clean_df, report). `keep` lists columns that must survive (e.g. a day column used for splitting)."""
    report = {"rows_in": len(df), "cols_in": df.shape[1]}
    df = normalise_columns(df)
    ids = [c for c in identifier_columns(df.columns, extra_identifiers) if c not in keep and c != label_col]
    df = df.drop(columns=ids)
    report["identifier_columns_dropped"] = ids

    num = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[num].to_numpy(dtype=float, na_value=np.nan)).sum()) if len(num) else 0
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    report["inf_values_to_nan"] = n_inf

    before = len(df)
    df = df.dropna(how="all")
    report["all_nan_rows_dropped"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_dropped"] = before - len(df)

    report["rows_out"], report["cols_out"] = len(df), df.shape[1]
    if label_col in df:
        report["label_counts"] = df[label_col].value_counts().to_dict()
    return df.reset_index(drop=True), report


# ---- row hashing for the streaming pipeline (src/data/pipeline.py) ------------------------------------------
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def row_hash(X):
    """64-bit hash per row of a feature frame (NaN-aware, column-order dependent)."""
    return pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)


def combine(h, values):
    """Combine a row hash with a per-row value (e.g. the canonical label)."""
    hv = pd.util.hash_array(np.asarray(values, dtype=object))
    with np.errstate(over="ignore"):
        return h ^ (hv * _GOLDEN)


def round_sig(X, sig=3):
    """Round every numeric column to `sig` significant figures (near-duplicate rule, preregistration §4.7)."""
    out = X.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            v = out[c].to_numpy(dtype="float64")
            with np.errstate(divide="ignore", invalid="ignore"):
                mag = np.where((v == 0) | ~np.isfinite(v), 1.0, 10.0 ** (np.floor(np.log10(np.abs(v))) - (sig - 1)))
            out[c] = np.round(v / mag) * mag
    return out
