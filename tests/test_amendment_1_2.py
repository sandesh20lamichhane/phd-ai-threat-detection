"""Amendment 1.2: stale outputs refused, task-specific S4, near-duplicate sensitivity S7, output verification."""
import json

import numpy as np
import pandas as pd
import pytest

from src.utils import paths
from tests.synthetic import write_cic


@pytest.fixture
def cic(drive):
    from src.data.pipeline import preprocess
    write_cic(drive / "data" / "raw")
    return preprocess("cicids2017_improved", log=lambda *a: None)


def test_report_carries_dataset_fingerprint(cic):
    from src.data.load import dataset_fingerprint, load_config
    assert cic["dataset_config_sha256"] == dataset_fingerprint(load_config(), "cicids2017_improved")


def test_outputs_built_under_other_rules_are_refused(cic):
    from src.data import datasets
    from src.data.verify import StaleOutput
    rep = paths.RESULTS / "data_audit" / "cicids2017_improved" / "report.json"
    r = json.loads(rep.read_text())
    r["dataset_config_sha256"] = "0" * 64
    rep.write_text(json.dumps(r))
    with pytest.raises(StaleOutput):
        datasets.load("cicids2017_improved")


def test_fingerprint_ignores_other_datasets_and_sees_defaults():
    from src.data.load import CONFIG_PATH, dataset_fingerprint, load_config
    text = CONFIG_PATH.read_text()
    base = load_config(text=text)
    other = load_config(text=text.replace("role: baseline", "role: baseline_x"))      # nsl_kdd only
    assert dataset_fingerprint(base, "cicids2017_improved") == dataset_fingerprint(other, "cicids2017_improved")
    assert dataset_fingerprint(base, "nsl_kdd") != dataset_fingerprint(other, "nsl_kdd")
    dflt = load_config(text=text.replace("min_blocks_per_stratum: 3", "min_blocks_per_stratum: 4"))
    assert dataset_fingerprint(base, "cicids2017_improved") != dataset_fingerprint(dflt, "cicids2017_improved")


def test_s4_uses_the_conflict_flag_of_the_task(cic):
    from src.data import datasets
    _, m = datasets.load("cicids2017_improved")
    _, mb = datasets.load("cicids2017_improved", sensitivity=("S4_drop_conflicts",))
    _, mm = datasets.load("cicids2017_improved", sensitivity=("S4_drop_conflicts",), task="multiclass")
    assert len(m) - len(mb) == int(m["_conflict_bin"].sum()) > 0
    assert len(m) - len(mm) == int(m["_conflict"].sum())
    assert (mb["_conflict_bin"] == 0).all() and (mm["_conflict"] == 0).all()


def test_near_duplicate_sidecar_reproduces_report_and_filters_eval_only(cic):
    from src.data import datasets, near
    s = near.build("cicids2017_improved", log=lambda *a: None)
    for k, v in s["check_against_report"].items():
        assert v["report"] == pytest.approx(v["recomputed"], abs=0), k
    side = pd.read_parquet(near.sidecar_path("cicids2017_improved"))
    sp = pd.read_parquet(paths.SPLITS / "cicids2017_improved" / "primary.parquet")
    part = sp["partition"].to_numpy()
    assert not side["seen_sig3"].to_numpy()[part == near.TRAIN].any()
    # 2 significant figures is stricter: every row seen at 3 is also seen at 2
    assert (side["seen_sig2"] >= side["seen_sig3"]).all()
    Xtr, _ = datasets.load("cicids2017_improved", partitions=("train",))
    Xtr7, _ = datasets.load("cicids2017_improved", partitions=("train",), sensitivity=("S7_unseen_eval",))
    assert len(Xtr) == len(Xtr7)
    _, mt = datasets.load("cicids2017_improved", partitions=("test",))
    _, mt7 = datasets.load("cicids2017_improved", partitions=("test",), sensitivity=("S7_unseen_eval",))
    seen = side.set_index("_key")["seen_sig3"]
    assert len(mt) - len(mt7) == int(seen.loc[mt["_key"]].sum())
    t = s["subsets"]["sig3"]["test"]
    assert t["unseen_rows"] == len(mt7)
    # unseen test rows really have no near-duplicate in train or val
    hs = side.set_index("_key")["near_sig3"]
    trv = set(hs.loc[sp["_key"][np.isin(part, (near.TRAIN, near.VAL))]])
    assert not set(hs.loc[mt7["_key"]]) & trv


def test_verify_detects_changed_files(cic):
    from src.data.verify import verify_dataset
    assert verify_dataset("cicids2017_improved")["ok"]
    f = sorted((paths.PROCESSED / "cicids2017_improved" / "v1").glob("part-*.parquet"))[0]
    f.write_bytes(f.read_bytes() + b"x")
    r = verify_dataset("cicids2017_improved")
    assert not r["ok"] and any("checksum" in p for p in r["problems"])


def test_external_near_duplicates(drive):
    from src.data import near
    src = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0], "float32"), "b": np.array([10.0, 20.0, 30.0], "float32")})
    tgt = pd.DataFrame({"a": np.array([1.0004, 2.5, 3.0], "float32"), "b": np.array([10.0, 20.0, 30.04], "float32")})
    meta = pd.DataFrame({"_key": [0, 1, 2], "_y": [0, 1, 1], "_label": ["BENIGN", "DoS", "DoS"]})
    s = near.external("unsw_nb15", src, tgt, meta, np.array([0, 1, 2]))
    side = pd.read_parquet(near.sidecar_path("unsw_nb15", "cicids2017_improved"))
    assert side["seen_sig3"].tolist() == [True, False, True]
    assert s["sig3"]["unseen_rows"] == 1 and not s["sig3"]["estimable"]
