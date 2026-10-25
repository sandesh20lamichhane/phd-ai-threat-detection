"""Map datasets onto the shared feature sets in config/feature_mapping/ for the Level 2 tests.

Deterministic, nothing is fitted: the same formula is applied to the source and the target data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..utils import paths

MAPPING_DIR = paths.CONFIG / "feature_mapping"
IANA = {"hopopt": 0, "icmp": 1, "igmp": 2, "ggp": 3, "ip": 4, "st2": 5, "tcp": 6, "egp": 8, "igp": 9, "udp": 17,
        "rdp": 27, "ipv6": 41, "ipv6-route": 43, "ipv6-frag": 44, "rsvp": 46, "gre": 47, "esp": 50, "ah": 51,
        "ipv6-icmp": 58, "ipv6-no": 59, "ipv6-opts": 60, "eigrp": 88, "ospf": 89, "ipip": 94, "pim": 103,
        "sctp": 132, "udplite": 136, "arp": 254}
OTHER_PROTOCOL = 255


def load_mapping(name):
    return yaml.safe_load((MAPPING_DIR / name).read_text())


def _col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df:
            return df[c]
        if c.lower() in lower:
            return df[lower[c.lower()]]
    raise KeyError(f"none of {candidates} in the data (columns: {list(df.columns)[:12]}...)")


def _primitive(df, spec):
    if "sum" in spec:
        out = sum(_col(df, c).astype("float64") for c in spec["sum"])
    else:
        out = _col(df, spec["column"])
        if spec.get("map") == "iana":
            out = out.astype(str).str.strip().str.lower().map(IANA).fillna(OTHER_PROTOCOL)
        out = pd.to_numeric(out, errors="coerce").astype("float64")
    return out * float(spec.get("scale", 1.0))


def to_canonical_unsw(df, side, mapping=None):
    """side: 'cic' or 'unsw'. Returns a float32 frame of the canonical features (primitives + derived)."""
    m = mapping or load_mapping("cic_to_unsw_mapping.yaml")
    out = pd.DataFrame({name: _primitive(df, spec[side]) for name, spec in m["primitives"].items()
                        if spec["status"] != "unavailable"}, index=df.index)
    dur = np.maximum(out["duration_s"].to_numpy(), 1e-6)
    out["fwd_bytes_per_s"] = out["fwd_bytes"] / dur
    out["bwd_bytes_per_s"] = out["bwd_bytes"] / dur
    out["fwd_pkts_per_s"] = out["fwd_pkts"] / dur
    out["bwd_pkts_per_s"] = out["bwd_pkts"] / dur
    out["fwd_mean_pkt_bytes"] = out["fwd_bytes"] / np.maximum(out["fwd_pkts"], 1)
    out["bwd_mean_pkt_bytes"] = out["bwd_bytes"] / np.maximum(out["bwd_pkts"], 1)
    missing = set(m["derived"]) - set(out.columns)
    if missing:
        raise AssertionError(f"derived features in the mapping but not computed: {missing}")
    return out.astype("float32")


def shared_cic_features(cols_2017, cols_2018):
    """Intersection rule of preregistration §8, keeping the 2017 column order."""
    s18 = set(cols_2018)
    shared = [c for c in cols_2017 if c in s18]
    return {"features": shared, "only_in_2017": [c for c in cols_2017 if c not in s18],
            "only_in_2018": [c for c in cols_2018 if c not in set(cols_2017)]}


def check_or_freeze_shared(cols_2017, cols_2018, path=None):
    """First call writes the list into cic_shared_features.yaml; later calls stop if it has changed."""
    path = Path(path or MAPPING_DIR / "cic_shared_features.yaml")
    cfg = yaml.safe_load(path.read_text())
    now = shared_cic_features(cols_2017, cols_2018)
    if not cfg.get("features"):
        text = path.read_text().split("\nfeatures:")[0]
        rest = {"features": now["features"], "only_in_2017": now["only_in_2017"], "only_in_2018": now["only_in_2018"],
                "label_map": cfg.get("label_map", {})}
        path.write_text(text + "\n" + yaml.safe_dump(rest, sort_keys=False, width=110))
        return now, "frozen"
    if cfg["features"] != now["features"]:
        raise AssertionError("shared CIC feature set differs from the frozen list in "
                             f"{path.name}: added {set(now['features']) - set(cfg['features'])}, "
                             f"removed {set(cfg['features']) - set(now['features'])}")
    return now, "unchanged"
