# AI-driven threat detection and automated response (Ph.D. code)

Sandesh Lamichhane, Assistant Professor, Yenepoya (Deemed to be University).

Code and results for the thesis *AI-Driven Cybersecurity: Predictive Threat Detection and Automated Response*.

**Where things live**

| What | GitHub (this repo) | Google Drive (`MyDrive/PhD-AI-Cyber/`) |
|---|---|---|
| Code, notebooks, configs | yes | yes, as a clone in `code/` |
| Results (manifests, metrics, figures) | yes, in `results/` | yes, same clone |
| Raw datasets | **never** (see `.gitignore`) | `data/raw/<dataset>/` |

## Start here

Open [`notebooks/00_setup_and_download.ipynb`](notebooks/00_setup_and_download.ipynb) in Google Colab and run it
top to bottom. It mounts Drive, puts this repo in Drive, builds the full folder structure with
[`setup_project.py`](setup_project.py) (run `python setup_project.py --tree` to see it), and pulls each dataset
from its official source straight into Drive. Every file gets a SHA-256 entry in
`data/raw/<dataset>/MANIFEST.json`; a copy of each manifest is committed under `results/data_manifest/`,
so the exact bytes used in the thesis are on record.

The registry is [`config/datasets.yaml`](config/datasets.yaml). It follows the Finalised Dataset Plan:

| Dataset | Source | Approx. size in Drive | Needs you to |
|---|---|---|---|
| CICIDS2017, corrected labels | copied from your existing Drive folder `ids-adapt/data/raw/cicids2017`, else the DistriNet site | 1.2 GB | nothing |
| CSE-CIC-IDS2018, corrected labels | DistriNet (KU Leuven) CNS 2022 release | check with `--plan` | nothing |
| UNSW-NB15 | UNSW SharePoint | ~0.7 GB | download the CSV folder and drop it into Drive |
| ToN_IoT | UNSW SharePoint | check after download | download the processed CSV folders and drop them into Drive |
| CIC-IoT2023 | UNB CIC | ~13 GB | fill in the UNB form, paste the CSV folder link |
| LANL Cyber1 | Los Alamos | ~12 GB, kept gzipped | fill in the LANL form, paste the 5 file links |
| CICIoMT2024 | UNB CIC | check with `--plan` | fill in the UNB form, paste the folder link |
| NSL-KDD | GitHub mirror of the UNB files | 0.03 GB | nothing |
| Edge-IIoTset (copy of your existing Drive data) | `ids-adapt/data/raw/edge_iiotset` | 8.4 GB | nothing |
| ids-adapt processed parquets (copy, reference only) | `ids-adapt/data/processed` | 8.5 GB | nothing |
| CSE-CIC-IDS2018 original (off by default) | public AWS bucket | 6.4 GB | set `enabled: true` if you want the uncorrected baseline |

PCAP captures are never downloaded. The planned datasets come to roughly 40–50 GB (about 27 GB of that
is known exactly; run `--plan` for the rest), plus 17 GB of copies of data already in Drive. The downloader refuses to push raw data past 190 GB (`--budget-gb`).

Command line, from the repo root:

```bash
python -m src.acquire.fetch --plan                        # what would be downloaded, with sizes
python -m src.acquire.fetch                               # download all enabled datasets
python -m src.acquire.fetch --only ciciot2023 lanl_cyber1 # selected datasets
python -m src.acquire.fetch --verify                      # re-check SHA-256 of everything in Drive
```

## Preprocessing (protocol v1.0)

The evaluation protocol is frozen in [`docs/protocol/preregistration.md`](docs/protocol/preregistration.md), with
thresholds in [`docs/protocol/hypotheses.md`](docs/protocol/hypotheses.md). Run
[`notebooks/01_profile_and_preprocess.ipynb`](notebooks/01_profile_and_preprocess.ipynb) in Colab, or:

```bash
python scripts/preprocess.py --profile                       # schema check, samples only
python scripts/preprocess.py --dataset cicids2017_improved   # one dataset
python scripts/preprocess.py --all                           # all processed datasets
```

Each dataset gets deduplicated, identifier-free parquet in `data/processed/<id>/v1/`, frozen splits in
`data/splits/<id>/`, a report in `results/data_audit/<id>/` and a data card in `docs/data_cards/`.

## Layout

| Path | Holds |
|---|---|
| `config/` | dataset registry, Drive paths, seeds, one YAML per experiment, feature mappings |
| `src/acquire` | downloader |
| `src/data` | cleaning (identifiers, duplicates), leakage checks, splits, train-only balancing, harmonisation |
| `src/features`, `src/models` | features, baselines, deep models, proposed and predictive models, explainability |
| `src/response` | CybORG / CyberBattleSim wrappers, RL agents, playbook baseline |
| `src/threat_intel` | advisory collection, sector tagging, ATT&CK linking |
| `src/evaluation` | macro-F1, per-class recall, PR-AUC, FPR at fixed detection rate, lead time, Wilcoxon, Friedman–Nemenyi, response metrics |
| `scripts/` | run experiments, preprocess, build tables and figures |
| `notebooks/` | 00 setup, 1x data, 2x levels 1–4, 30 response, 40 explainability, 50 paper figures |
| `docs/` | protocol, hypotheses, data-quality rules, data cards, decisions, reproducibility checklist |
| `results/` | metrics, tables and figures per protocol level, dataset manifests |
| `paper/`, `thesis/` | journal paper and thesis LaTeX |

In Drive (`MyDrive/PhD-AI-Cyber/`): `data/{raw,interim,processed,splits,external}`, `models/`, `runs/`,
`simulations/`, `backups/`, plus `code/` (this repo).

## Citing the data

Each entry in `config/datasets.yaml` carries its citation where one is required. The corrected CIC
releases must be cited alongside the original CIC papers.
