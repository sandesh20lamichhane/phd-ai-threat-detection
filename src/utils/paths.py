"""One place that knows where things live.

Raw and intermediate data, models and run logs stay in Google Drive. Code and small
results stay in the repo (and therefore in both GitHub and the Drive clone).
Override the Drive root with the PHD_DRIVE_ROOT environment variable.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVE = Path(os.environ.get("PHD_DRIVE_ROOT", "/content/drive/MyDrive/PhD-AI-Cyber"))

RAW = DRIVE / "data" / "raw"
INTERIM = DRIVE / "data" / "interim"          # cleaned, deduplicated, identifiers removed
PROCESSED = DRIVE / "data" / "processed"      # model-ready features (parquet)
SPLITS = DRIVE / "data" / "splits"            # saved row indices per experiment and seed
EXTERNAL = DRIVE / "data" / "external"        # threat-intelligence corpus
MODELS = DRIVE / "models"
RUNS = DRIVE / "runs"
SIMULATIONS = DRIVE / "simulations"

CONFIG = REPO / "config"
RESULTS = REPO / "results"
PAPER = REPO / "paper"


def raw(dataset):
    return RAW / dataset


def results_for(level):
    """results/<level>/{metrics,tables,figures}"""
    base = RESULTS / level
    for sub in ("metrics", "tables", "figures"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base
