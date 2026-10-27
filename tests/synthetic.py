"""Small synthetic files shaped like the real datasets, for pipeline tests."""
import numpy as np
import pandas as pd

CIC_FEATURES = ["Protocol", "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
                "Total Length of Bwd Packet", "Fwd Header Length", "Bwd Header Length", "Fwd IAT Mean",
                "Bwd IAT Mean", "FWD Init Win Bytes", "Bwd Init Win Bytes", "Flow Bytes/s"]
DAYS = {"monday": [], "tuesday": [("FTP-Patator", 9.5, 10.5)], "wednesday": [("DoS Hulk", 10, 11)],
        "thursday": [("Web Attack - XSS", 9.5, 10.2)], "friday": [("Portscan", 13.9, 14.6), ("DDoS", 15.9, 16.3)]}
DATES = {"monday": "2017-07-03", "tuesday": "2017-07-04", "wednesday": "2017-07-05", "thursday": "2017-07-06",
         "friday": "2017-07-07"}


def cic_day(day, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(DATES[day] + " 08:00:00")
    t = start + pd.to_timedelta(np.sort(rng.uniform(0, 9 * 3600, n)), unit="s")
    hours = (t - t.normalize()) / pd.Timedelta(hours=1)
    label = np.array(["BENIGN"] * n, dtype=object)
    for name, a, b in DAYS[day]:
        m = (hours >= a) & (hours < b) & (rng.random(n) < 0.8)
        label[m] = name
        att = m & (rng.random(n) < 0.1)
        label[att] = name + " - Attempted"
    df = pd.DataFrame({"id": np.arange(n), "Flow ID": [f"f{i}" for i in range(n)],
                       "Src IP": rng.choice(["192.168.10.5", "192.168.10.8", "205.174.165.73"], n),
                       "Src Port": rng.integers(1024, 65535, n), "Dst IP": "192.168.10.50",
                       "Dst Port": rng.choice([80, 21, 443], n), "Protocol": rng.choice([6, 17], n),
                       "Timestamp": t.strftime("%Y-%m-%d %H:%M:%S.%f")})
    attack = label != "BENIGN"
    df["Flow Duration"] = rng.integers(10, 5_000_000, n) * np.where(attack, 3, 1)
    for c in CIC_FEATURES[2:]:
        if c not in df:
            df[c] = rng.integers(0, 2000, n) + attack * 500
    df["Flow Bytes/s"] = df["Flow Bytes/s"].astype(object)
    df.loc[rng.random(n) < 0.01, "Flow Bytes/s"] = "Infinity"
    df["Label"] = label
    df["Attempted Category"] = np.where(pd.Series(label).str.contains("Attempted"), 1, -1)
    # exact duplicates of some rows (same features, same label), and a conflicting-label copy
    dups = df.sample(40, random_state=seed).copy()
    dups["id"] += n
    conflict = df[df.Label == "BENIGN"].head(5).copy()
    conflict["Label"] = "DoS Hulk" if day == "wednesday" else conflict["Label"]
    return pd.concat([df, dups, conflict], ignore_index=True).sort_values("Timestamp", kind="stable")


def write_cic(root):
    d = root / "cicids2017_improved"
    d.mkdir(parents=True, exist_ok=True)
    for i, day in enumerate(DAYS):
        cic_day(day, seed=i).to_csv(d / f"{day}.csv", index=False)
    return d


def write_cse(interim, seed=10):
    """CSE-CIC-IDS2018-shaped files: CICIDS2017-like days moved to 2018, with attacks spread over the day."""
    d = interim / "csecicids2018_improved"
    d.mkdir(parents=True, exist_ok=True)
    for i, (day, date) in enumerate((("wednesday", "2018-02-14"), ("friday", "2018-02-16"))):
        df = cic_day(day, n=3000, seed=seed + i)
        t = pd.to_datetime(df["Timestamp"])
        df["Timestamp"] = (pd.Timestamp(date) + (t - t.dt.normalize())).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        df.to_csv(d / f"{date}.csv", index=False)
    return d


def write_unsw(interim, n=3000, seed=0):
    from src.data.load import load_config
    cols = load_config()["datasets"]["unsw_nb15"]["columns"]
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 1000, n) for c in cols})
    df["srcip"], df["dstip"] = "59.166.0.1", "149.171.126.6"
    df["sport"] = np.where(rng.random(n) < .05, "0x000b", rng.integers(1, 65535, n).astype(str))
    df["dsport"] = rng.choice(["80", "53", "-"], n)
    df["proto"], df["state"], df["service"] = rng.choice(["tcp", "udp"], n), "FIN", rng.choice(["-", "http"], n)
    df["Stime"] = 1421927414 + np.sort(rng.integers(0, 86400, n))
    df["Ltime"] = df["Stime"] + 1
    lab = rng.choice(["", "Exploits", " Fuzzers ", "Backdoors"], n, p=[.85, .05, .05, .05])
    df["attack_cat"], df["Label"] = lab, (lab != "").astype(int)
    d = interim / "unsw_nb15" / "UNSW-NB15 - CSV Files"
    d.mkdir(parents=True)
    for i, idx in enumerate(np.array_split(np.arange(n), 4), 1):
        df.iloc[idx].to_csv(d / f"UNSW-NB15_{i}.csv", header=False, index=False)
    pd.DataFrame({"x": [1]}).to_csv(d / "UNSW_NB15_training-set.csv", index=False)   # must be ignored


def write_ciciomt(raw, seed=0):
    rng = np.random.default_rng(seed)
    cols = ["Header_Length", "Protocol Type", "Duration", "Rate", "Tot sum"]
    for part in ("train", "test"):
        d = raw / "ciciomt2024" / part
        d.mkdir(parents=True)
        for name in ("Benign", "TCP_IP-DDoS-ICMP1", "TCP_IP-DDoS-ICMP2", "MQTT-Malformed_Data", "Recon-Port_Scan"):
            pd.DataFrame({c: rng.random(200) for c in cols}).to_csv(d / f"{name}_{part}.pcap.csv", index=False)


def write_ciciot_zip(raw, seed=0):
    """CIC-IoT2023 layout: CSV.zip with CSV/<label>/<capture>.pcap.csv and no label column."""
    import zipfile
    rng = np.random.default_rng(seed)
    d = raw / "ciciot2023"
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / "CSV.zip", "w") as z:
        z.writestr("CSV/README.pdf", b"%PDF")
        for label, n_files in (("Benign_Final", 2), ("DDoS-ICMP_Flood", 3), ("Recon-PortScan", 1), ("XSS", 1)):
            for i in range(n_files):
                df = pd.DataFrame({"Header_Length": rng.random(300), "Protocol Type": rng.integers(0, 17, 300),
                                   "Time_To_Live": rng.random(300)})
                z.writestr(f"CSV/{label}/{label}{i}.pcap.csv", df.to_csv(index=False))


def write_ton(interim, seed=0):
    rng = np.random.default_rng(seed)
    d = interim / "ton_iot" / "Processed_datasets" / "Processed_Network_dataset"
    d.mkdir(parents=True)
    t0 = 1554198358
    for i in (1, 2, 10):
        n = 2000
        df = pd.DataFrame({"ts": t0 + i * 20000 + np.sort(rng.integers(0, 20000, n)), "src_ip": "192.168.1.1",
                           "src_port": rng.integers(1, 9, n), "dst_ip": "192.168.1.2", "dst_port": 80,
                           "proto": rng.choice(["tcp", "udp"], n), "service": "-", "duration": rng.random(n),
                           "src_bytes": rng.integers(0, 999, n), "conn_state": "SF", "dns_query": "x.com",
                           "label": 0, "type": "normal"})
        att = rng.random(n) < 0.3
        df.loc[att, "type"], df.loc[att, "label"] = "scanning", 1
        if i == 10:
            df.insert(1, "uid", [f"C{k}" for k in range(n)])
        df.to_csv(d / f"Network_dataset_{i}.csv", index=False)
