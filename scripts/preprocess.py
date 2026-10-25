"""Preprocess datasets under the frozen protocol (docs/protocol/preregistration.md).

    python scripts/preprocess.py --profile                      # schema check of all datasets, samples only
    python scripts/preprocess.py --dataset cicids2017_improved  # one dataset
    python scripts/preprocess.py --dataset cicids2017_improved --max-rows 200000   # quick pilot
    python scripts/preprocess.py --all                          # every dataset that is preprocessed
    python scripts/preprocess.py --shared-features              # freeze/check the CIC 2017-2018 shared set
    python scripts/preprocess.py --cross-overlap                # source-train vs external-target overlap
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_config  # noqa: E402

PROCESSED_IN_ORDER = ["cicids2017_improved", "unsw_nb15", "nsl_kdd", "ciciomt2024", "ton_iot", "ciciot2023",
                      "csecicids2018_improved", "lanl_cyber1"]


def run(ds_id, max_rows=None):
    if ds_id == "lanl_cyber1":
        from src.data.lanl import preprocess_lanl
        return preprocess_lanl(max_rows=max_rows)
    from src.data.pipeline import preprocess
    return preprocess(ds_id, max_rows=max_rows)


def shared_features():
    import json
    from src.data.harmonise import check_or_freeze_shared
    from src.utils import paths
    cols = {d: json.loads((paths.RESULTS / "data_audit" / d / "report.json").read_text())["schema"]["feature_columns"]
            for d in ("cicids2017_improved", "csecicids2018_improved")}
    now, status = check_or_freeze_shared(cols["cicids2017_improved"], cols["csecicids2018_improved"])
    print(f"shared CIC features: {len(now['features'])} ({status}); only in 2017: {now['only_in_2017']}; "
          f"only in 2018: {now['only_in_2018']}")


def cross_overlap():
    import numpy as np
    import pandas as pd
    import yaml
    from src.data import datasets
    from src.data.clean import row_hash
    from src.data.harmonise import MAPPING_DIR, to_canonical_unsw
    from src.data.leakage import cross_dataset_overlap
    from src.utils import paths
    from src.utils.io import write_json
    out = {}
    shared = yaml.safe_load((MAPPING_DIR / "cic_shared_features.yaml").read_text())["features"]
    src, _ = datasets.load("cicids2017_improved", partitions=("train",))
    if shared:
        tgt, _ = datasets.load("csecicids2018_improved", split="external", partitions=("external",),
                               columns=None)
        out["csecicids2018_improved"] = cross_dataset_overlap(row_hash(src[shared]), row_hash(tgt[shared]))
        del tgt
    tgt, _ = datasets.load("unsw_nb15", split="external", partitions=("external",))
    out["unsw_nb15"] = cross_dataset_overlap(row_hash(to_canonical_unsw(src, "cic")),
                                             row_hash(to_canonical_unsw(tgt, "unsw")))
    write_json(paths.RESULTS / "data_audit" / "cross_dataset_overlap.json", out)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--shared-features", action="store_true")
    ap.add_argument("--cross-overlap", action="store_true")
    ap.add_argument("--max-rows", type=int)
    a = ap.parse_args()
    if a.profile:
        from src.data.profile import profile_all
        profile_all(only=a.dataset)
        return
    if a.shared_features:
        return shared_features()
    if a.cross_overlap:
        return cross_overlap()
    cfg = load_config()
    todo = PROCESSED_IN_ORDER if a.all else (a.dataset or [])
    unknown = [d for d in todo if d not in cfg["datasets"]]
    if unknown or not todo:
        ap.error(f"give --dataset from {PROCESSED_IN_ORDER} or --all (unknown: {unknown})")
    for d in todo:
        run(d, a.max_rows)


if __name__ == "__main__":
    main()
