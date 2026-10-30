"""Level 4 extraction support (amendments 1.7-1.8): extractor check and ToN_IoT label matching."""
import numpy as np
import pandas as pd
import pytest

from src.data import level4 as L4

CFG = L4.load_config()
TON = CFG["ton_iot"]
CHECK = CFG["extractor_check"]


def _flows(rows):
    return pd.DataFrame(rows, columns=["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"])


def _gt(rows):
    return pd.DataFrame(rows, columns=L4.GT_COLUMNS)


def test_numeric_tokens():
    s = pd.Series(["1.5", "Infinity", "-Infinity", "NaN", "", "7"])
    out = L4.numeric(s).tolist()
    assert out[0] == 1.5 and out[1] == np.inf and out[2] == -np.inf and np.isnan(out[3]) and np.isnan(out[4])
    assert out[5] == 7.0


def test_timestamp_formats():
    a = L4.timestamp_us(pd.Series(["2019-04-27 10:42:14.250000"]))
    b = L4.timestamp_us(pd.Series(["27/04/2019 10:42:14 AM"]))
    assert a[0] == 1556361734_250000 and b[0] == 1556361734_000000
    with pytest.raises(L4.Level4Stop):
        L4.timestamp_us(pd.Series(["not a time"]))


def test_endpoint_key_unordered_and_icmp_ports_ignored():
    k, _ = L4.endpoint_key(pd.Series(["10.0.0.1", "10.0.0.2", "10.0.0.1", "10.0.0.1"]),
                           pd.Series([1234, 80, 8, 0]),
                           pd.Series(["10.0.0.2", "10.0.0.1", "10.0.0.2", "10.0.0.2"]),
                           pd.Series([80, 1234, 0, 3]),
                           pd.Series([6, 6, 1, 1]), TON["proto_map"], TON["ports_ignored_for"])
    assert k[0] == k[1]                 # direction ignored
    assert k[2] == k[3]                 # ICMP: ports ignored
    assert k[0] != k[2]
    k2, _ = L4.endpoint_key(pd.Series(["a"]), pd.Series([1]), pd.Series(["b"]), pd.Series([2]),
                            pd.Series([47]), TON["proto_map"], TON["ports_ignored_for"])
    assert k2[0] == 0                   # GRE etc. never match


def _iv(gt, durations=None):
    g = gt.copy()
    g["ts"] = g["ts"].astype("int64")
    g["duration"] = durations if durations is not None else 0.0
    return L4.intervals(g, TON, "type")


def test_label_matching_rules():
    gt = _gt([[1000, "192.168.1.35", "40128", "192.168.1.195", "80", "tcp", "xss"],
              [2000, "192.168.1.35", "40130", "192.168.1.195", "80", "tcp", "scanning"],
              [2005, "192.168.1.35", "40130", "192.168.1.195", "80", "tcp", "dos"],       # overlaps the one above
              [3000, "192.168.1.40", "0", "192.168.1.1", "8", "icmp", "ddos"],
              [4000, "192.168.1.50", "5000", "192.168.1.1", "53", "udp", "password"],
              [4010, "192.168.1.50", "5000", "192.168.1.1", "53", "udp", "dos"]])
    iv = _iv(gt, [0.0, 100.0, 0.0, 0.0, 500.0, 0.0])
    fl = _flows([
        ["192.168.1.35", 40128, "192.168.1.195", 80, 6, "2019-01-01 00:00:00.0"],      # placeholder times below
        ["192.168.1.195", 80, "192.168.1.35", 40128, 6, "2019-01-01 00:00:00.0"],      # reversed direction
        ["192.168.1.35", 40128, "192.168.1.195", 80, 6, "2019-01-01 00:00:00.0"],      # too late
        ["192.168.1.35", 40130, "192.168.1.195", 80, 6, "2019-01-01 00:00:00.0"],      # two covering, types differ
        ["192.168.1.40", 0, "192.168.1.1", 0, 1, "2019-01-01 00:00:00.0"],             # ICMP, ports differ
        ["192.168.1.50", 5000, "192.168.1.1", 53, 17, "2019-01-01 00:00:00.0"],        # earlier long record covers
        ["192.168.1.99", 1, "192.168.1.1", 2, 6, "2019-01-01 00:00:00.0"],             # no record
    ])
    starts = [1000.4, 999.2, 1002.5, 2005.0, 3000.9, 4300.0, 1000.0]
    fl["Timestamp"] = pd.to_datetime(starts, unit="s").strftime("%Y-%m-%d %H:%M:%S.%f")
    lab, *_ = L4.label_ton_flows(fl, iv, 0, TON)
    assert lab["Label"].tolist() == ["xss", "xss", "normal", "dos", "ddos", "password", "normal"]
    assert lab["_gt_multi"].tolist() == [False, False, False, True, False, False, False]
    assert lab["_gt_match"].sum() == 5


def test_offset_shifts_flow_times():
    gt = _gt([[1000, "10.0.0.1", "1", "10.0.0.2", "2", "tcp", "dos"]])
    iv = _iv(gt)
    fl = _flows([["10.0.0.1", 1, "10.0.0.2", 2, 6, "x"]])
    fl["Timestamp"] = pd.to_datetime([1000 + 3600], unit="s").strftime("%Y-%m-%d %H:%M:%S.%f")
    assert L4.label_ton_flows(fl, iv, 0, TON)[0]["Label"][0] == "normal"
    assert L4.label_ton_flows(fl, iv, 3600, TON)[0]["Label"][0] == "dos"


def test_clock_offset_rules():
    acfg = TON["alignment"]
    rng = np.random.default_rng(0)
    keys = rng.integers(1, 2**63, 500, dtype=np.int64).astype(np.uint64)
    ts = rng.integers(1_556_000_000, 1_556_100_000, 500)
    d, rec = L4.clock_offset(keys, ts + 1, keys, ts, acfg)
    assert d == 0 and rec["mode_s"] == 1
    d, _ = L4.clock_offset(keys, ts + 36_000 + 1, keys, ts, acfg)
    assert d == 36_000
    with pytest.raises(L4.Level4Stop):
        L4.clock_offset(keys, ts + 437, keys, ts, acfg)


def test_ground_truth_schema_and_durations(tmp_path):
    p = tmp_path / "GroundTruth_Network_1.csv"
    p.write_text("﻿ts,src_ip,src_port,dst_ip,dst_port,proto,type\n"
                 "1556361734,192.168.1.35,40128,192.168.1.195,80,tcp,xss\n"
                 "1556361734,192.168.1.35,40128,192.168.1.195,80,tcp,xss\n"
                 "1556361800,192.168.1.35,40200,192.168.1.195,80,tcp,xss\n")
    gt, rec = L4.read_ground_truth([p])
    assert rec == {"rows_read": 3, "exact_duplicates_removed": 1, "rows": 2}
    proc = pd.DataFrame({"ts": [1556361734, 1556361734], "src_ip": ["192.168.1.35"] * 2,
                         "src_port": [40128, 40128], "dst_ip": ["192.168.1.195"] * 2, "dst_port": [80, 80],
                         "proto": ["tcp", "tcp"], "type": ["xss", "xss"], "duration": ["0.5", "2.0"],
                         "label": [1, 1]})
    out, jr = L4.attach_durations(gt, proc)
    assert out["duration"].tolist() == [2.0, 0.0] and jr["joined_share"] == 0.5
    bad = tmp_path / "GroundTruth_Network_2.csv"
    bad.write_text("ts,date,time,type\n1,a,b,dos\n")
    with pytest.raises(L4.Level4Stop):
        L4.read_ground_truth([bad])


def test_zeek_agreement_and_coverage():
    zeek = pd.DataFrame({"ts": [100, 200, 300], "src_ip": ["a", "a", "a"], "src_port": [1, 2, 3],
                         "dst_ip": ["b", "b", "b"], "dst_port": [9, 9, 9], "proto": ["tcp"] * 3,
                         "duration": [0.0, 0.0, 0.0], "label": [0, 1, 1]})
    ziv = L4.intervals(zeek, TON, "label")
    key, text = L4.endpoint_key(pd.Series(["a", "a", "a", "a"]), pd.Series([1, 2, 3, 4]),
                                pd.Series(["b"] * 4), pd.Series([9] * 4), pd.Series([6] * 4),
                                TON["proto_map"], TON["ports_ignored_for"])
    t = np.array([100.0, 200.0, 300.0, 400.0])
    rec, matched, _ = L4.zeek_agreement(key, text, t, ziv, [0, 1, 0, 0])
    assert rec["matched_share"] == 0.75 and rec["table"] == {"flow0_zeek0": 1, "flow0_zeek1": 1,
                                                             "flow1_zeek0": 0, "flow1_zeek1": 1}
    cov = L4.ground_truth_coverage(ziv, key[:2], t[:2], (0, 1000))
    assert cov["records_in_range"] == 3 and abs(cov["covered_share"] - 2 / 3) < 1e-12


def _check_frames(offset_s=0, perturb=None):
    feats = ["Flow Duration", "Flow Bytes/s"]
    ref = pd.DataFrame({"Src IP": ["1.1.1.1", "1.1.1.1", "2.2.2.2"], "Src Port": [1, 1, 3],
                        "Dst IP": ["9.9.9.9"] * 3, "Dst Port": [80, 80, 443], "Protocol": [6, 6, 6],
                        "Timestamp": ["2017-07-06 12:00:00.000001", "2017-07-06 12:00:00.000001",
                                      "2017-07-06 12:05:00.500000"],
                        "Flow Duration": [10.0, 20.0, 30.0], "Flow Bytes/s": ["Infinity", "5.0", "NaN"]})
    ext = ref.copy()
    t = pd.to_datetime(ext["Timestamp"]) + pd.Timedelta(seconds=offset_s)
    ext["Timestamp"] = t.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    if perturb:
        ext.loc[perturb[0], perturb[1]] = perturb[2]
    return ext, ref, feats


def test_extractor_check_pass_offset_and_fail():
    ext, ref, feats = _check_frames()
    rec = L4.extractor_check(ext, ref, feats, CHECK)
    assert rec["passed"] and rec["matched_share"] == 1.0 and rec["offset_us"] == 0
    ext, ref, feats = _check_frames(offset_s=-3 * 3600)
    rec = L4.extractor_check(ext, ref, feats, CHECK)
    assert rec["passed"] and rec["offset_us"] == -3 * 3600 * 1_000_000
    ext, ref, feats = _check_frames(offset_s=7)
    assert not L4.extractor_check(ext, ref, feats, CHECK)["passed"]
    ext, ref, feats = _check_frames(perturb=(1, "Flow Duration", 20.001))
    rec = L4.extractor_check(ext, ref, feats, CHECK)
    assert not rec["passed"] and rec["feature_agreement"]["Flow Duration"] == pytest.approx(2 / 3)
    ext, ref, feats = _check_frames(perturb=(1, "Flow Duration", 20.0000001))
    assert L4.extractor_check(ext, ref, feats, CHECK)["passed"]
