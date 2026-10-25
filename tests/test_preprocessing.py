import json

import numpy as np
import pandas as pd
import pytest

from src.data import split as S
from src.data.labels import UnmappedLabel, compile_rules, map_labels
from src.data.load import load_config
from tests.synthetic import write_cic


def test_label_rules_cover_known_cic_labels():
    rules = compile_rules(load_config()["label_rules"]["cic"])
    raw = pd.Series(["BENIGN", "DoS Hulk - Attempted", "Web Attack - Brute Force", "FTP-Patator", "DDoS-LOIC-HTTP",
                     "DoS-GoldenEye", "Botnet Ares", "Infiltration - Portscan", "Portscan", "Heartbleed",
                     "SSH-BruteForce", "Web Attack - SQL Injection"])
    fam, _ = map_labels(raw, rules)
    assert list(fam) == ["BENIGN", "Attempted", "WebAttack", "BruteForce", "DDoS", "DoS", "Bot", "Infiltration",
                         "PortScan", "Heartbleed", "BruteForce", "WebAttack"]
    with pytest.raises(UnmappedLabel):
        map_labels(pd.Series(["Something new"]), rules)


def test_chronological_block_split_orders_time_within_each_stratum():
    ts = np.arange(0, 300 * 40, 30.0)             # 40 blocks, 10 flows each
    fam = np.where((ts >= 300 * 20), "DoS", "BENIGN").astype(object)
    y = (fam != "BENIGN").astype(np.int8)
    blk = S.blocks(ts, 300)
    part, diag = S.chronological_block_split(blk, np.array(["d"] * len(ts), dtype=object), fam, y,
                                             {"train": .7, "val": .15, "test": .15})
    for f in ("BENIGN", "DoS"):
        m = fam == f
        assert ts[m][part[m] == S.TRAIN].max() < ts[m][part[m] == S.TEST].min()
        assert set(part[m]) == {S.TRAIN, S.VAL, S.TEST}
    # purging removes flows that run into a block of another partition
    dur = np.full(len(ts), 400.0)
    p2, n = S.purge(part, blk, np.array(["d"] * len(ts), dtype=object), ts, dur, 300)
    assert n > 0 and (p2 == S.PURGED).sum() == n


def test_pipeline_end_to_end_on_synthetic_cic(drive):
    from src.data.pipeline import preprocess
    write_cic(drive / "data" / "raw")
    rep = preprocess("cicids2017_improved", log=lambda *a: None)
    c = rep["counts"]
    assert c["exact_duplicates_removed"] >= 5 * 40 - 5        # sampled rows may repeat
    assert c["conflict_rows"] >= 10 and c["attempted_rows"] > 0
    sch = rep["schema"]
    assert "Dst Port" not in sch["feature_columns"] and "Attempted Category" not in sch["feature_columns"]
    assert "Timestamp" not in sch["feature_columns"] and "Protocol" in sch["feature_columns"]
    assert sch["inf_values_to_nan"].get("Flow Bytes/s", 0) > 0
    prim = rep["splits"]["primary"]
    assert set(prim["rows_per_partition"]) >= {"train", "val", "test"}
    # every attack family is present in train and test
    for fam in ("BruteForce", "DoS", "WebAttack", "PortScan", "DDoS"):
        assert prim["family_by_partition"]["train"].get(fam, 0) > 0
        assert prim["family_by_partition"]["test"].get(fam, 0) > 0
    s1 = rep["splits"]["s1_by_day"]
    assert s1["family_by_partition"]["test"].get("BruteForce", 0) == 0
    # outputs exist and line up with the split keys
    df = pd.concat(pd.read_parquet(p) for p in sorted((drive / "data/processed/cicids2017_improved/v1").glob("*.parquet")))
    sp = pd.read_parquet(drive / "data/splits/cicids2017_improved/primary.parquet")
    assert len(df) == len(sp) == c["rows_out"] and (df["_key"].to_numpy() == sp["_key"].to_numpy()).all()
    assert df.duplicated([*sch["feature_columns"], "_label"]).sum() == 0
    card = (drive / "data_cards" / "cicids2017_improved.md").read_text()
    assert "Exact duplicates" in card or "exact duplicates" in card
    json.dumps(rep, default=str)


def test_profile_flags_schema(drive):
    from src.data.profile import profile_all
    write_cic(drive / "data" / "raw")
    out = profile_all(only=["cicids2017_improved"], log=lambda *a: None)
    r = out["cicids2017_improved"]
    assert r["checks"]["all_sampled_labels_mapped"] and r["checks"]["time_parsed_share"] == 1.0
    assert "Dst Port" in r["identifier_columns_found"]
    assert (drive / "results/data_audit/schema_profile.md").exists()


def test_frozen_transform_refuses_target_fit():
    from src.features.transform import FrozenTransform
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "p": ["tcp", "udp", "tcp"]})
    t = FrozenTransform(scale=True).fit(X, "cicids2017_improved/primary/train")
    Z = t.transform(pd.DataFrame({"a": [np.nan], "p": ["icmp"]}))
    assert Z.loc[0, "p=other"] == 1 and np.isfinite(Z.loc[0, "a"])
    with pytest.raises(RuntimeError):
        t.fit(X, "again/train")
    with pytest.raises(ValueError):
        FrozenTransform().fit(X, "csecicids2018_improved/external")


def test_datasets_loader_drops_attempted_and_meta(drive):
    from src.data import datasets
    from src.data.pipeline import preprocess
    write_cic(drive / "data" / "raw")
    preprocess("cicids2017_improved", log=lambda *a: None)
    X, meta = datasets.load("cicids2017_improved", partitions=("train",))
    assert not any(c.startswith("_") for c in X.columns) and (meta["_y"] >= 0).all()
    X3, meta3 = datasets.load("cicids2017_improved", partitions=("train",), sensitivity=("S3_attempted_benign", "S2_ports"))
    assert len(X3) > len(X) and "dst_port" in X3


def test_unsw_headerless_files(drive):
    from src.data.pipeline import preprocess
    from tests.synthetic import write_unsw
    write_unsw(drive / "data" / "interim")
    rep = preprocess("unsw_nb15", log=lambda *a: None)
    assert len(rep["inputs"]) == 4
    assert set(rep["classes"]["family_out"]) == {"BENIGN", "Exploits", "Fuzzers", "Backdoor"}
    f = rep["schema"]["feature_columns"]
    assert not {"srcip", "sport", "dstip", "dsport", "Stime", "Ltime", "stcpb", "attack_cat", "Label"} & set(f)
    assert rep["schema"]["binary_label_disagreements"] == 0
    assert rep["splits"]["external"]["rows_per_partition"] == {"external": rep["counts"]["rows_out"]}


def test_ciciomt_labels_from_file_names_and_official_partitions(drive):
    from src.data.pipeline import preprocess
    from tests.synthetic import write_ciciomt
    write_ciciomt(drive / "data" / "raw")
    rep = preprocess("ciciomt2024", log=lambda *a: None)
    lm = rep["schema"]["label_map"]
    assert lm["TCP_IP-DDoS-ICMP"] == "DDoS" and lm["MQTT-Malformed_Data"] == "MQTT-Malformed"
    sp = rep["splits"]["primary"]["family_by_partition"]
    assert sp["test"]["DDoS"] == 400 and sp["train"]["DDoS"] + sp["val"]["DDoS"] == 400


def test_pilot_outputs_are_refused(drive):
    from src.data import datasets
    from src.data.pipeline import preprocess
    write_cic(drive / "data" / "raw")
    preprocess("cicids2017_improved", max_rows=3000, log=lambda *a: None)
    with pytest.raises(RuntimeError):
        datasets.load("cicids2017_improved")
