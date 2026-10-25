"""Raw label string -> canonical family, by the ordered regex rules in config/preprocessing.yaml."""
import re

import numpy as np
import pandas as pd

BENIGN, ATTEMPTED = "BENIGN", "Attempted"


class UnmappedLabel(ValueError):
    pass


def compile_rules(rules):
    return [(re.compile(p, re.I), fam) for p, fam in rules]


def map_labels(raw, rules):
    """raw: Series of label strings. Returns (family Series, mapping dict raw -> family).

    Mapping is done once per distinct raw value, so it is cheap on millions of rows.
    """
    raw = raw.fillna("").astype(str).str.strip()
    uniq = pd.unique(raw)
    mapping, missing = {}, []
    for u in uniq:
        fam = next((f for p, f in rules if p.search(u)), None)
        if fam is None:
            missing.append(u)
        mapping[u] = fam
    if missing:
        raise UnmappedLabel(f"raw labels with no rule in config/preprocessing.yaml: {sorted(missing)[:20]}")
    return raw.map(mapping), mapping


def binary_from_family(family):
    """0 benign, 1 attack, -1 excluded (Attempted)."""
    fam = np.asarray(family, dtype=object)
    y = np.where(fam == BENIGN, 0, 1).astype(np.int8)
    y[fam == ATTEMPTED] = -1
    return y
