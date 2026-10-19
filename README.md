# AI-driven threat detection and automated response (Ph.D. code)

Code and results for the thesis *AI-Driven Cybersecurity: Predictive Threat Detection and Automated Response*.

**Where things live**

| What | GitHub (this repo) | Google Drive (`MyDrive/PhD-AI-Cyber/`) |
|---|---|---|
| Code, notebooks, configs | yes | yes, as a clone in `code/` |
| Results (manifests, metrics, figures) | yes, in `results/` | yes, same clone |
| Raw datasets | **never** (see `.gitignore`) | `data/raw/<dataset>/` |

## Getting the data

Open [`notebooks/01_download_datasets.ipynb`](notebooks/01_download_datasets.ipynb) in Google Colab
(File → Open notebook → GitHub) and run it top to bottom. It mounts Drive, clones this repo into Drive,
and pulls each dataset from its official source straight into Drive. Every file gets a SHA-256 entry in
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
| CSE-CIC-IDS2018 original (off by default) | public AWS bucket | 6.4 GB | set `enabled: true` if you want the uncorrected baseline |

PCAP captures are never downloaded. The known sizes add up to about 27 GB; run `--plan` to see the
exact size of the others. The downloader refuses to push raw data past 190 GB (`--budget-gb`).

Command line, from the repo root:

```bash
python -m src.acquire.fetch --plan                        # what would be downloaded, with sizes
python -m src.acquire.fetch                               # download all enabled datasets
python -m src.acquire.fetch --only ciciot2023 lanl_cyber1 # selected datasets
python -m src.acquire.fetch --verify                      # re-check SHA-256 of everything in Drive
```

## Citing the data

Each entry in `config/datasets.yaml` carries its citation where one is required. The corrected CIC
releases must be cited alongside the original CIC papers.
