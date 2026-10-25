import gzip

import numpy as np
import pandas as pd

from src.data import lanl


def _write(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def make_lanl(raw, days=60, seed=0):
    rng = np.random.default_rng(seed)
    d = raw / "lanl_cyber1"
    d.mkdir(parents=True)
    t = np.sort(rng.integers(1, days * 86400, 20000))
    comps = [f"C{i}" for i in range(40)]
    auth = [(int(x), f"U{rng.integers(9)}@DOM1", f"U{rng.integers(9)}@DOM1", rng.choice(comps), rng.choice(comps),
             rng.choice(["NTLM", "Kerberos", "?"]), rng.choice(["Network", "Interactive"]), "LogOn",
             rng.choice(["Success", "Fail"], p=[.95, .05])) for x in t]
    _write(d / "auth.txt.gz", auth)
    _write(d / "proc.txt.gz", [(int(x), "U1@DOM1", rng.choice(comps), f"P{rng.integers(30)}", "Start") for x in t[::4]])
    _write(d / "flows.txt.gz", [(int(x), 3, rng.choice(comps), "N1", rng.choice(comps), "N2", 6, 5, 300) for x in t[::5]])
    _write(d / "dns.txt.gz", [(int(x), rng.choice(comps), rng.choice(comps)) for x in t[::7]])
    rt_t = np.sort(rng.integers(2 * 86400, 30 * 86400, 60))      # red team stops before day 41
    _write(d / "redteam.txt.gz", [(int(x), "U66@DOM1", "C17693", comps[i % 35]) for i, x in enumerate(rt_t)])
    return d


def test_lanl_host_hour_and_split(drive):
    make_lanl(drive / "data" / "raw")
    rep = lanl.preprocess_lanl(log=lambda *a: None)
    hh = pd.read_parquet(drive / "data/processed/lanl_cyber1/v1/host_hour.parquet")
    assert {"auth_in", "auth_out", "auth_in_new_src", "proc_new", "flows_out", "dns_queries"} <= set(hh.columns)
    assert not hh.duplicated(["computer", "hour"]).any()
    assert hh["auth_in"].sum() + 0 == 20000 and hh["auth_out"].sum() == 20000
    # new-edge counts never exceed distinct pairs
    assert hh["auth_in_new_src"].sum() <= 40 * 40
    # red team ends before day 41 in this fixture, so the contingency rule must fire
    s = rep["level3_split"]
    assert s["rule"].startswith("contingency") and s["test_compromised_hosts"] >= 30
    assert s["train_days"][1] < s["test_days"][0]


def test_lanl_primary_split_when_test_has_enough_hosts():
    comp = pd.DataFrame({"comp": [f"C{i}" for i in range(80)], "day": np.r_[np.full(40, 10), np.full(40, 45)]})
    s = lanl.level3_split(comp, {"train_days": [1, 40], "test_days": [41, 58], "min_test_compromised_hosts": 30,
                                 "train_frac_of_early": 0.8})
    assert s["rule"] == "primary" and s["test_compromised_hosts"] == 40
