"""Learned preprocessing that is fitted once, on the source training partition, and then frozen.

Median imputation, optional standardisation and one-hot encoding of categoricals. `fit` may be called only
once; `transform` refuses columns it was not fitted on. The fitted state is saved with a SHA-256 so every
result can name the exact transform it used (preregistration §5).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class FrozenTransform:
    def __init__(self, scale=False, max_categories=64):
        self.scale, self.max_categories = scale, max_categories
        self.state = None

    def fit(self, X, fitted_on):
        """fitted_on: a short description such as 'cicids2017_improved/primary/train' (recorded in the state)."""
        if self.state is not None:
            raise RuntimeError("FrozenTransform is already fitted; fit a new one instead")
        if any(t in fitted_on for t in ("test", "external", "val")):
            raise ValueError(f"transforms are fitted on training data only, not on '{fitted_on}'")
        num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat = [c for c in X.columns if c not in num]
        med = X[num].median()
        st = {"fitted_on": fitted_on, "rows": int(len(X)), "numeric": num, "categorical": {},
              "median": {c: (None if pd.isna(v) else float(v)) for c, v in med.items()}}
        if self.scale:
            filled = X[num].fillna(med)
            st["mean"] = {c: float(v) for c, v in filled.mean().items()}
            st["std"] = {c: float(v) if v > 0 else 1.0 for c, v in filled.std().items()}
        for c in cat:
            top = X[c].value_counts().index[: self.max_categories].tolist()
            st["categorical"][c] = [str(v) for v in top]
        self.state = st
        return self

    def transform(self, X):
        st = self.state
        if st is None:
            raise RuntimeError("fit first")
        missing = [c for c in st["numeric"] + list(st["categorical"]) if c not in X]
        if missing:
            raise KeyError(f"columns missing at transform time: {missing}")
        out = X[st["numeric"]].astype("float64").fillna({c: v for c, v in st["median"].items() if v is not None})
        out = out.fillna(0.0)
        if self.scale:
            out = (out - pd.Series(st["mean"])) / pd.Series(st["std"])
        parts = [out.astype("float32")]
        for c, levels in st["categorical"].items():
            v = X[c].astype(str)
            oh = pd.DataFrame({f"{c}={lv}": (v == lv).astype("float32") for lv in levels}, index=X.index)
            oh[f"{c}=other"] = (~v.isin(levels)).astype("float32")
            parts.append(oh)
        return pd.concat(parts, axis=1)

    def save(self, path):
        text = json.dumps(self.state, sort_keys=True, indent=1)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def load(cls, path):
        t = cls()
        t.state = json.loads(Path(path).read_text())
        t.scale = "mean" in t.state
        return t
