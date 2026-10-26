"""Preprocess datasets under the frozen protocol (docs/protocol/preregistration.md).

    python scripts/preprocess.py --profile                      # schema check of all datasets, samples only
    python scripts/preprocess.py --dataset cicids2017_improved  # one dataset
    python scripts/preprocess.py --dataset cicids2017_improved --max-rows 200000   # quick pilot
    python scripts/preprocess.py --all                          # every dataset that is preprocessed
    python scripts/preprocess.py --shared-features              # freeze/check the CIC 2017-2018 shared set
    python scripts/preprocess.py --cross-overlap                # source-train vs external-target overlap
    python scripts/preprocess.py --near-duplicates              # S7 near-duplicate records (amendment 1.2)
    python scripts/preprocess.py --verify                       # rules, pilot flag and checksums of all outputs
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
    """Exact and near-duplicate overlap of each external target with CICIDS2017.

    Descriptive: exact and near shares against CICIDS2017 train. S7 record: target rows near-duplicating
    CICIDS2017 train + val, the data the model is fitted and tuned on (amendment 1.3).
    """
    import yaml
    from src.data import datasets, near
    from src.data.clean import row_hash
    from src.data.harmonise import MAPPING_DIR, to_canonical_unsw
    from src.data.leakage import cross_dataset_overlap
    from src.utils import paths
    from src.utils.io import write_json
    out = {}
    shared = yaml.safe_load((MAPPING_DIR / "cic_shared_features.yaml").read_text())["features"]
    src, smeta = datasets.load("cicids2017_improved", partitions=("train", "val"))
    is_train = (smeta["_partition"] == "train").to_numpy()
    if shared:
        tgt, meta = datasets.load("csecicids2018_improved", split="external", partitions=("external",),
                                  columns=[*shared, "_key", "_y", "_label", "_conflict", "_conflict_bin"])
        out["csecicids2018_improved"] = cross_dataset_overlap(row_hash(src.loc[is_train, shared]), row_hash(tgt[shared]))
        out["csecicids2018_improved"]["near_duplicates"] = near.external(
            "csecicids2018_improved", src[shared], src.loc[is_train, shared], tgt[shared], meta,
            meta["_block"].to_numpy())
        del tgt, meta
    tgt, meta = datasets.load("unsw_nb15", split="external", partitions=("external",))
    cs, ct = to_canonical_unsw(src, "cic"), to_canonical_unsw(tgt, "unsw")
    out["unsw_nb15"] = cross_dataset_overlap(row_hash(cs[is_train]), row_hash(ct))
    out["unsw_nb15"]["near_duplicates"] = near.external("unsw_nb15", cs, cs[is_train], ct, meta,
                                                        meta["_block"].to_numpy())
    out["scope"] = ("share_seen_in_source_train: target rows whose shared-feature vector equals a CICIDS2017 train "
                    "row (descriptive). near_duplicates: the same after rounding to 3 and 2 significant figures; "
                    "the S7 'seen' flags compare with CICIDS2017 train + val (amendment 1.3), the train-only share "
                    "is given as share_seen_in_source_train_only. None of these rules out semantically similar "
                    "traffic or shows that the datasets are independent.")
    path = paths.RESULTS / "data_audit" / "cross_dataset_overlap.json"
    write_json(path, out)
    near.record_in_manifest(path)
    for k in ("csecicids2018_improved", "unsw_nb15"):
        if k in out:
            n = out[k]["near_duplicates"]
            print(f"{k}: exact vs train {out[k]['share_seen_in_source_train']:.6%}; near vs train + val "
                  f"(3 s.f.) {1 - n['sig3']['unseen_share']:.4%}, (2 s.f.) {1 - n['sig2']['unseen_share']:.4%}; "
                  f"S7 estimable {n['sig3']['estimable']}")


def near_duplicates(ds_ids):
    from src.data import near
    for d in ds_ids or near.WITHIN:
        near.build(d)


def verify(ds_ids, checksums=True):
    from src.data.verify import CONFIRMATORY, verify_all
    rows, aux = verify_all(ds_ids or CONFIRMATORY, checksums=checksums, auxiliary=not ds_ids)
    bad = [r["dataset"] for r in rows if not r["ok"]] + [a["artifact"] for a in aux if not a["ok"]]
    if bad:
        raise SystemExit(f"verification failed for {bad}; see results/data_audit/verification.md")
    print("all outputs verified")


class Tee:
    """Copy everything printed to a log file under results/data_audit/logs/ (pushed with the results)."""

    def __init__(self, path, stream):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f, self.stream = open(path, "a", buffering=1), stream

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


def start_log(args):
    import datetime as dt
    from src.utils import paths
    what = "profile" if args.profile else "shared_features" if args.shared_features else \
        "cross_overlap" if args.cross_overlap else "near_duplicates" if args.near_duplicates else \
        "verify" if args.verify else "_".join(args.dataset or ["all"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = paths.RESULTS / "data_audit" / "logs" / f"{stamp}_{what}.log"
    sys.stdout = Tee(log, sys.stdout)
    sys.stderr = Tee(log, sys.stderr)
    print(f"command: {' '.join(sys.argv)}\nlog: {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--shared-features", action="store_true")
    ap.add_argument("--cross-overlap", action="store_true")
    ap.add_argument("--near-duplicates", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-checksums", action="store_true", help="with --verify: skip the SHA-256 of output files")
    ap.add_argument("--max-rows", type=int)
    a = ap.parse_args()
    start_log(a)
    if a.profile:
        from src.data.profile import profile_all
        profile_all(only=a.dataset)
        return
    if a.shared_features:
        return shared_features()
    if a.cross_overlap:
        return cross_overlap()
    if a.near_duplicates:
        return near_duplicates(a.dataset)
    if a.verify:
        return verify(a.dataset, checksums=not a.no_checksums)
    cfg = load_config()
    todo = PROCESSED_IN_ORDER if a.all else (a.dataset or [])
    unknown = [d for d in todo if d not in cfg["datasets"]]
    if unknown or not todo:
        ap.error(f"give --dataset from {PROCESSED_IN_ORDER} or --all (unknown: {unknown})")
    for d in todo:
        run(d, a.max_rows)


if __name__ == "__main__":
    main()
