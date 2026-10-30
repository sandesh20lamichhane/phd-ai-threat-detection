# Level 4 extraction plan

**Author:** Sandesh Lamichhane, Assistant Professor, Yenepoya (Deemed to be University)

*Written 2026-10-30 under amendments 1.7 and 1.8 ([preregistration §9a](preregistration.md)), before any Level 4
capture was downloaded into Drive or processed and before the extractor was built or run. Known at the time: the
ToN_IoT documentation (`ReadMe.pdf`, `Description_stats_datasets`), the header and first rows of the ToN_IoT
ground-truth files, the CICIoMT2024 and CIC-IoT2023 READMEs of the released CSVs, the capture listings and sizes,
and the data cards of the released CSVs. No flow has been extracted from any capture and no model has scored any
Level 4 row.*

The machine-readable values of this plan (paths, tolerances, address range, pinned extractor commit) are in
[`config/level4_extraction.yaml`](../../config/level4_extraction.yaml). Where this file and the YAML differ, the
pipeline stops.

## 1. Order of work

1. Pin the extractor commit (§2.1) in `config/level4_extraction.yaml`, chosen by reading the code, never by running it.
2. Build the extractor and record it (§2.2).
3. Extractor check on the CICIDS2017 Thursday capture (§3). If it fails, Level 4 stops (§9a).
4. ToN_IoT: manifest of the captures, extraction (§2.3), labels (§4), preprocessing (§6), `--verify`.
5. CICIoMT2024, then CIC-IoT2023: the same, with the labels of §5.
6. Level 4 data record `results/level4/data_freeze.json` (§9a), then scoring once.

Each step writes its record to GitHub (`results/level4/`) and to Drive before the next step starts. A step that
stops is reported with its record; nothing later runs.

## 2. Extractor

### 2.1 Source and version

The corrected CICIDS2017 and CSE-CIC-IDS2018 releases used as the source (Engelen, Rimmer and Joosen 2021; Liu,
Engelen, Lynar, Essam and Joosen 2022) were produced by the DistriNet fixed CICFlowMeter, published at
`github.com/GintsEngelen/CICFlowMeter`. It ends TCP flows on the mutual FIN exchange and on RST, removes the absolute
timestamps from the Active and Idle features and counts the PSH and URG flags correctly. The 2022 release adds the
columns `ICMP Code`, `ICMP Type` and `Total TCP Flow Time`, which are among the 82 shared features.

- A copy of that repository is kept as a fork in the study's GitHub account (`sandesh20lamichhane/CICFlowMeter`),
  so the exact code stays available. The notebook builds from the fork.
- The commit used is the newest commit of the repository whose CSV writer emits every one of the 82 shared features
  under the names of `config/feature_mapping/cic_shared_features.yaml`. It is chosen by reading the source code and
  recorded in `config/level4_extraction.yaml` (`extractor.commit`) before the extractor is built. If no commit emits
  all 82, Level 4 stops; no other tool is tried without a new amendment.
- Flow parameters are the defaults of the command-line entry point of that commit (CICFlowMeter: flow timeout
  120 s, activity timeout 5 s). They are read from the code into the YAML and never changed.

### 2.2 Build

Built in Colab from the pinned commit, with the Java version and the libpcap binding the repository requires.
`results/level4/extractor.json` records:

- the fork URL and the commit;
- the Java version and the operating system;
- the SHA-256 of the built jar or class tree and of the native pcap library;
- the flow parameters;
- the exact command line.

The JVM runs with `-Duser.timezone=UTC`, so the `Timestamp` column is UTC wall time.

### 2.3 Running on a capture

- One capture file at a time, copied (or, for a zip, extracted member by member) from Drive to the Colab disk,
  processed, and deleted from the Colab disk. Captures are never modified and never filtered.
- A capture that the extractor cannot open (for example pcapng) is converted to classic pcap with `editcap -F pcap`
  and nothing else. Both SHA-256 values are recorded.
- Per capture, `results/level4/<dataset>/extraction.json` records:
  - the SHA-256 of the input and output;
  - the flows written and the first and last flow start time;
  - the run time;
  - the extractor's warnings and errors.
- Flows that span two capture files are split at the file boundary, as in the CIC releases.
- Output CSVs are stored in Drive under `data/interim/<dataset id>/extracted/`, never on GitHub.
- If capture time ranges overlap, the overlap and the share of flows with identical 5-tuple and start time are
  reported. Exact duplicates are then removed by the protocol 1.1 rule (§6).

## 3. Extractor check (CICIDS2017, Thursday 6 July 2017)

**Capture.** `Thursday-WorkingHours.pcap` of the official CIC-IDS2017 release. It is the smallest capture day
(7.8 GB on the dataset page) and contains web attacks, infiltration and benign traffic. It is stored in Drive under
`data/raw/cicids2017_improved/pcap/`, with its SHA-256 in the manifest.

**Reference.** `thursday.csv` of the corrected release already in Drive (362,076 rows; SHA-256 in
`results/data_manifest/`).

**Matching.**

1. Key: (`Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Protocol`), in the order both files give it, plus the
   `Timestamp` to the microsecond.
2. Clock offset δ. Take the most frequent difference (extractor minus CSV) over key-equal pairs within 24 h,
   rounded to 1 µs. If it is not 0, it must be a whole multiple of 900 s, and is then applied. Otherwise the check
   fails.
3. Several rows sharing a key and timestamp are paired in order of appearance.

**Agreement.** For each of the 82 shared features and each matched pair, the values agree if
|a − b| <= max(1e-9, 1e-6 · |b|). Two missing values agree, and two infinite values agree if they have the same
sign.

**Pass.** At least 99 % of the CSV's rows are matched, and for every one of the 82 features at least 99 % of matched
pairs agree.

**Record.** `results/level4/extractor_check.json`: rows on each side, matched share, δ, and the agreement share per
feature. Also the 20 features with the lowest agreement and their largest absolute differences (numbers only, no
addresses).

## 4. ToN_IoT labels

### 4.1 What the ground truth is

`SecuityEvents_GroundTruth_datasets/SecurityEvents_Network_datasets/GroundTruth_Network_1.csv` to `_18.csv`
(spelling as released) hold one row per attack connection:

```
ts,src_ip,src_port,dst_ip,dst_port,proto,type
1556361734,192.168.1.35,40128,192.168.1.195,80,tcp,xss
```

`ts` is the Zeek connection start in whole epoch seconds, `proto` is `tcp`, `udp` or `icmp`, and `type` is the
attack. The processed records `Processed_Network_dataset/Network_dataset_1.csv` to `_23.csv` (already preprocessed
as `ton_iot`) carry the same five-tuple and `ts`, together with `duration`, `label` and `type`, for normal and attack
connections alike.

`ReadMe.pdf` says that the datasets "were labelled based on tagging IP addresses (192.168.159.30-39) and their
timestamps". The ground-truth file, however, lists connections, not time windows, and its first source address
(192.168.1.35) lies outside that range. Amendment 1.8 therefore labels by matching to the ground-truth connections
(§4.3) and turns the address range into a sensitivity analysis (§4.6).

**Schema stop.** The pipeline stops if any ground-truth file lacks exactly these seven columns, or if any `type` is
not mapped by the `ton` rules of `config/preprocessing.yaml`.

### 4.2 Ground-truth intervals

- The 18 files are read whole and exact duplicate rows are removed (count reported).
- Each record g gets a duration d_g from the processed record with the same `ts`, five-tuple, `proto` and `type`. If
  several processed records match, the largest duration is taken. If none matches, or the duration is missing, d_g = 0.
- The share of records that find a processed record is reported.
- The interval of g is [ts_g − 1 s, ts_g + d_g + 1 s]. The 1 s margins cover the truncation of `ts` to whole
  seconds.

### 4.3 Matching rule

Let a flow f of the extractor have protocol p_f, endpoints (Src IP, Src Port) and (Dst IP, Dst Port), and start
time t_f (epoch seconds, from `Timestamp`). The flow f matches the ground-truth record g when all three conditions
hold:

1. **Protocol.** The protocols agree: 6 = `tcp`, 17 = `udp`, 1 = `icmp`. Other protocols never match.
2. **Endpoints.** For TCP and UDP, the two (address, port) pairs are equal as an unordered pair, so direction is
   ignored. For ICMP, the two addresses are equal as an unordered pair and ports are ignored: Zeek stores the ICMP
   type and code in the port fields, CICFlowMeter stores 0.
3. **Time.** t_f − δ lies in the interval of g, where δ is the clock offset of §4.4.

**Labels.**

- A flow that matches at least one record is an attack. Its raw label is the `type` of that record.
- If it matches several records, the one with the latest ts_g decides its type. If those records carry different
  types, the flow is flagged `_gt_multi`.
- A flow that matches no record is `normal`.
- Raw labels map to families by the `ton` rules of `config/preprocessing.yaml` (unchanged). The binary label is 0
  for `normal` and 1 otherwise.
- As for the CIC IoT targets, there is no "Attempted" class at Level 4: every matched flow is an attack. The share
  of attack flows with no forward payload (`Total Length of Fwd Packet` = 0) is reported for each family.

### 4.4 Clock alignment

The capture clock and the ground-truth `ts` are expected to agree (Zeek read the same captures). This is checked
before any flow is labelled.

- Take all pairs (f, g) of the first `normal_attack` capture in name order that meet rules 1 and 2. The most
  frequent value of round(t_f) − ts_g within ±24 h is δ. Endpoint keys with more than 10,000 such pairs are left
  out of the count, and their number is recorded.
- If |δ| <= 2 s, δ = 0.
- Otherwise δ must lie within 2 s of a whole multiple of 900 s, a time-zone shift, and is set to that multiple and
  applied to every capture.
- Otherwise the pipeline stops.

δ and the histogram of differences around it are recorded.

### 4.5 Reported with the labels (`results/level4/ton_iot_flows/labels.json`)

- **Counts.** Flows per capture and per family, next to the record counts of `Statistics of Network Records`.
- **Ground-truth coverage.** Of the records whose `ts` falls inside the time range of the captures, the share that
  match at least one flow, per `type`.
- **Agreement with the published Zeek labels** (§9a). Every flow is matched by rules 1-3 above to the processed
  records of both classes. The report gives the share of flows that match any record and the 2 × 2 table of the
  flow label against the Zeek `label` of the matched record (on several matches, the rule of §4.3 decides), with
  overall agreement and Cohen's κ.
- **Other checks.**
  - The number of `_gt_multi` flows.
  - The number of flows in `normal_pcaps` that match a ground-truth record.
  - The zero-payload shares of §4.3.

These are descriptive. None of them changes a label.

### 4.6 Sensitivity analyses (reported beside H4, no decision)

- **S-L1, documented attacker range.** Flows labelled `normal` with an endpoint in 192.168.159.30-39 are removed.
- **S-L2, label agreement.** Only flows whose label agrees with the matched Zeek record's `label` are kept. Flows
  that match no processed record are removed.

## 5. CICIoMT2024 and CIC-IoT2023 labels

- **CICIoMT2024.** The raw label is the capture file name, read by the regular expression of the released CSVs
  (`^(?P<label>.+?)\d*_(train|test)\.pcap$`). Families come from the `ciciomt` rules, so `Benign_train` and
  `Benign_test` are benign. Both the train and test captures are used, all rows, split type `external`.
- **CIC-IoT2023.** The raw label is the capture folder name (`PCAP/<label>/`). Families come from the `ciciot` rules,
  and `Benign_Final` is benign.
- **Stop rule.** A name that matches no rule stops the pipeline (§9a). Every flow of a capture takes that capture's
  label, including any background traffic inside an attack capture, as in the released CSVs (threats to validity).
- **Attacking-endpoint sensitivity** (§9a). The CICIoMT2024 README (`README_Wifi_and_MQTT_attacks.pdf`) and the
  CIC-IoT2023 CSV READMEs name no attacker addresses (read 2026-10-30), so for now there is none. The CIC-IoT2023
  capture README is read when its captures are downloaded. If it names attacker addresses, the sensitivity is added
  to this file and committed before the first CIC-IoT2023 capture is processed.

## 6. Preprocessing

Protocol 1.1 rules apply unchanged (§9a). New entries in `config/preprocessing.yaml` reuse the existing label
rules, which leaves the fingerprints of the Level 1-3 datasets unchanged (`src/data/verify.py`).

| id | role | input | label | time | identifiers removed from the features |
|---|---|---|---|---|---|
| `ton_iot_flows` | domain_target | labelled extractor CSVs, `data/interim/ton_iot_flows/labelled/` | `Label` (from §4) | `Timestamp` | as `cicids2017_improved`, plus the `_gt_*` audit columns |
| `ciciomt2024_flows` | domain_target | `data/interim/ciciomt2024_flows/labelled/` | `Label` (from §5) | `Timestamp` | as `cicids2017_improved` |
| `ciciot2023_flows` | domain_target | `data/interim/ciciot2023_flows/labelled/` | `Label` (from §5) | `Timestamp` | as `cicids2017_improved` |

- Each dataset uses split type `external` (all rows) and 300 s blocks from the flow start times.
- Every dataset gets a data card, the S7 comparison with CICIDS2017 train ∪ val, and `--verify`.
- The extractor's own `Label` column ("NeedManualLabel") is replaced by the label of §4 or §5 before preprocessing.

## 7. Records

- **Drive** (raw data only in Drive, §11): captures under `data/raw/<dataset>/pcap/`, with SHA-256 in
  `MANIFEST.json`, and for a zip, of each member. Extracted and labelled CSVs go under
  `data/interim/<dataset id>/`, processed data under `data/processed/<dataset id>/`.
- **GitHub and Drive:** `results/data_manifest/<dataset>.json`, `results/level4/` (extractor, check, extraction,
  labels), data cards, `config/level4_extraction.yaml`, the notebooks with their outputs.

## 8. Compute

- Colab, one capture at a time; restarts resume from the last finished capture (outputs are checked by SHA-256 before
  they are skipped).
- ToN_IoT (48.1 GB) runs first. Throughput measured on it is used to plan the CIC IoT captures, which may need
  several sessions.
- No capture, flow or row is sampled or skipped to save time (§10).
