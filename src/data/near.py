"""Near-duplicate records for sensitivity S7 (amendment 1.2, preregistration §8.1).

For every processed row: its near-duplicate hash at 3 significant figures (the preregistered definition, §4.7)
and at 2 (a stricter check of that definition), and whether an evaluation row is "seen", i.e. its hash also
occurs in data the model was fitted or tuned on:

    within a dataset   val rows vs train; test rows vs train + val
    external targets   target rows vs CICIDS2017 train + val, on the shared (or canonical UNSW) features
                       (amendment 1.3: val sets the decision threshold, so it counts as tuning data)

S7 then scores the unchanged model on the unseen evaluation rows only. Models, splits and training data do
not change. The 3-figure hashes are recomputed from the stored features, so `build` first checks that they
reproduce the counts in the preprocessing report exactly.

Every record names the outputs it was built from and the SHA-256 of its sidecar; `check_sidecar` (run by the
loader before any S7 filter and by `--verify`) refuses a sidecar that has changed or whose dataset was rebuilt
since (amendment 1.3). The JSON records themselves are hashed in results/data_audit/auxiliary_manifest.json.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..utils import paths
from ..utils.io import sha256_file, write_json, write_parquet
from .clean import round_sig, row_hash
from .split import PARTITION_NAMES, TEST, TRAIN, VAL

SIG_FIGS = (3, 2)
MIN_ROWS_PER_CLASS = 100        # below this, or below MIN_BLOCKS, an S7 estimate is reported as not estimable
MIN_BLOCKS_PER_CLASS = 30
WITHIN = ["cicids2017_improved", "ciciomt2024", "ton_iot", "ciciot2023", "nsl_kdd"]
EXTERNAL = {"csecicids2018_improved": "cicids2017_improved", "unsw_nb15": "cicids2017_improved"}
S7_DEFINITION = "1.3"           # amendment that last changed what counts as "seen"
MANIFEST = "auxiliary_manifest.json"


def _audit(*parts):
    return paths.RESULTS.joinpath("data_audit", *parts)


def record_path(ds_id, against=None):
    return _audit("cross_dataset_overlap.json") if against else _audit(ds_id, "near_duplicates.json")


def provenance(ds_id):
    """Which processed outputs a record was built from: dataset rules and the SHA-256 of every output and split."""
    rep = json.loads(_audit(ds_id, "report.json").read_text())
    files = {"outputs": rep.get("outputs", {}).get("parquet_sha256", {}),
             "splits": {k: v["sha256"] for k, v in (rep.get("splits") or {}).items()}}
    return {"dataset_config_sha256": rep.get("dataset_config_sha256"),
            "outputs_digest": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def record_in_manifest(*files):
    """Store the SHA-256 of JSON records written by this module, so --verify can tell if one was edited."""
    path = _audit(MANIFEST)
    man = json.loads(path.read_text()) if path.exists() else {}
    for f in files:
        man[f.relative_to(paths.RESULTS).as_posix()] = sha256_file(f)
    write_json(path, dict(sorted(man.items())))


def _record(ds_id, against=None):
    rec = json.loads(record_path(ds_id, against).read_text())
    return rec[ds_id]["near_duplicates"] if against else rec


_checked = {}


def check_sidecar(ds_id, against=None, use_cache=True):
    """List of problems (empty when the sidecar can be used): definition, provenance and SHA-256."""
    rp, sp = record_path(ds_id, against), sidecar_path(ds_id, against)
    if not rp.exists() or not sp.exists():
        return [f"missing {'record' if not rp.exists() else 'sidecar'} for {ds_id}; run "
                f"{'--cross-overlap' if against else '--near-duplicates'}"]
    stat = sp.stat()
    key = (str(sp), stat.st_mtime_ns, stat.st_size)
    if use_cache and _checked.get(key) == []:
        return []
    try:
        rec = _record(ds_id, against)
    except (KeyError, json.JSONDecodeError) as e:
        return [f"record for {ds_id} unreadable: {e}"]
    problems = []
    if rec.get("s7_definition") != S7_DEFINITION:
        problems.append(f"built under S7 definition {rec.get('s7_definition', '1.2')}, current is {S7_DEFINITION}")
    built = rec.get("built_from", {})
    for d in ([against, ds_id] if against else [ds_id]):
        if built.get(d) != provenance(d):
            problems.append(f"{d} outputs changed since this record was built")
    if sha256_file(sp) != rec.get("sidecar_sha256"):
        problems.append(f"sidecar {sp.name} differs from its recorded SHA-256")
    _checked[key] = problems
    return problems


def require_sidecar(ds_id, against=None):
    problems = check_sidecar(ds_id, against)
    if problems:
        raise RuntimeError(f"S7 record for {ds_id} cannot be used: {'; '.join(problems)}")


def sidecar_path(ds_id, against=None):
    return paths.SPLITS / ds_id / (f"near_vs_{against}.parquet" if against else "near.parquet")


def _hashes(X, chunk=2_000_000):
    """Row hashes after rounding, in chunks so a 45 M-row frame never needs a float64 copy in memory."""
    out = {s: [] for s in SIG_FIGS}
    for i in range(0, max(len(X), 1), chunk):
        part = X.iloc[i:i + chunk]
        for s in SIG_FIGS:
            out[s].append(row_hash(round_sig(part, s)))
    return {s: np.concatenate(v) for s, v in out.items()}


def _seen_within(h, part):
    seen = np.zeros(len(h), bool)
    tr = np.unique(h[part == TRAIN])
    trv = np.unique(h[(part == TRAIN) | (part == VAL)])
    seen[part == VAL] = np.isin(h[part == VAL], tr)
    seen[part == TEST] = np.isin(h[part == TEST], trv)
    return seen


def subset_summary(seen, y, fam, block, mask):
    """Composition of the unseen subset of the rows in mask (reported with every S7 number)."""
    keep = mask & ~seen & (y >= 0)
    out = {"rows": int((mask & (y >= 0)).sum()), "unseen_rows": int(keep.sum())}
    out["unseen_share"] = out["unseen_rows"] / out["rows"] if out["rows"] else None
    est = True
    for cls, name in ((0, "benign"), (1, "attack")):
        a = mask & (y == cls)           # whole evaluation set, for comparison with the primary analysis
        out[f"{name}_rows"], out[f"{name}_blocks"] = int(a.sum()), int(len(np.unique(block[a])))
        k = keep & (y == cls)
        nb = int(len(np.unique(block[k])))
        out[f"unseen_{name}_rows"], out[f"unseen_{name}_blocks"] = int(k.sum()), nb
        est &= k.sum() >= MIN_ROWS_PER_CLASS and nb >= MIN_BLOCKS_PER_CLASS
    out["estimable"] = bool(est)
    out["unseen_by_family"] = {f: int(v) for f, v in pd.Series(fam[keep]).value_counts().items()}
    out["all_by_family"] = {f: int(v) for f, v in pd.Series(fam[mask & (y >= 0)]).value_counts().items()}
    return out


def build(ds_id, split="primary", log=print):
    rep = json.loads((paths.RESULTS / "data_audit" / ds_id / "report.json").read_text())
    files = sorted((paths.PROCESSED / ds_id / rep.get("output_version", "v1")).glob("part-*.parquet"))
    feats = rep["schema"]["feature_columns"]
    keys, y, fam, H = [], [], [], {s: [] for s in SIG_FIGS}
    for f in files:
        df = pd.read_parquet(f, columns=[*feats, "_key", "_y", "_label"])
        for s, h in _hashes(df[feats]).items():
            H[s].append(h)
        keys.append(df["_key"].to_numpy())
        y.append(df["_y"].to_numpy())
        fam.append(df["_label"].astype(str).to_numpy())
        log(f"[{ds_id}] near-duplicate hashes: {f.name}")
    keys, y, fam = np.concatenate(keys), np.concatenate(y), np.concatenate(fam)
    H = {s: np.concatenate(v) for s, v in H.items()}
    sp = pd.read_parquet(paths.SPLITS / ds_id / f"{split}.parquet")
    if not np.array_equal(keys, sp["_key"].to_numpy()):
        raise AssertionError("processed data and split file are out of step; re-run preprocessing")
    part, block = sp["partition"].to_numpy(), sp["_block"].to_numpy()

    # the recomputed 3-figure hashes must reproduce the preprocessing report exactly
    h3 = H[3]
    _, inv, cnt = np.unique(h3, return_inverse=True, return_counts=True)
    got = int((cnt[inv] > 1).sum())
    want = rep["counts"]["rows_in_near_duplicate_groups"]
    pairs = rep["splits"][split]["integrity"]["pairs"]
    check = {"rows_in_near_duplicate_groups": {"report": want, "recomputed": got}}
    for a, b in ((TRAIN, VAL), (TRAIN, TEST), (VAL, TEST)):
        k = f"{PARTITION_NAMES[a]}->{PARTITION_NAMES[b]}"
        if k in pairs:
            mb = part == b
            share = float(np.isin(h3[mb], np.unique(h3[part == a])).mean()) if mb.any() else 0.0
            check[k] = {"report": pairs[k]["near_duplicate_share"], "recomputed": share}
    for k, v in check.items():
        if not np.isclose(v["report"], v["recomputed"], rtol=0, atol=1e-12):
            raise AssertionError(f"{ds_id}: recomputed near-duplicates differ from the report ({k}: {v}); "
                                 "the stored features do not match the preprocessing run")

    side = pd.DataFrame({"_key": keys})
    summary = {"dataset": ds_id, "split": split, "s7_definition": S7_DEFINITION,
               "built_from": {ds_id: provenance(ds_id)}, "check_against_report": check, "subsets": {}}
    for s in SIG_FIGS:
        seen = _seen_within(H[s], part)
        side[f"near_sig{s}"] = H[s]
        side[f"seen_sig{s}"] = seen
        summary["subsets"][f"sig{s}"] = {PARTITION_NAMES[p]: subset_summary(seen, y, fam, block, part == p)
                                         for p in (VAL, TEST) if (part == p).any()}
    summary["reproduces_previous_sidecar"] = _same_as_previous(side, sidecar_path(ds_id))
    summary["sidecar_sha256"] = write_parquet(side, sidecar_path(ds_id))
    write_json(record_path(ds_id), summary)
    record_in_manifest(record_path(ds_id))
    for s in SIG_FIGS:
        t = summary["subsets"][f"sig{s}"].get("test", {})
        log(f"[{ds_id}] S7 test subset at {s} s.f.: {t.get('unseen_rows', 0):,} of {t.get('rows', 0):,} rows unseen; "
            f"estimable={t.get('estimable')}")
    return summary


def _same_as_previous(side, path):
    """True / False if an earlier sidecar exists (same rows and values?), None if this is the first build."""
    if not path.exists():
        return None
    try:
        old = pd.read_parquet(path)
    except Exception:                                   # unreadable earlier file: not a reproduction
        return False
    cols = [c for c in side.columns if c in old.columns]
    return bool(len(cols) == len(side.columns) and old[cols].equals(side[cols]))


def external(target, fit_frame, train_frame, target_frame, target_meta, block, against="cicids2017_improved"):
    """S7 record for an external target (same features, same order in all three frames).

    "seen" = the target row's near-duplicate representation occurs in fit_frame (source train + val, the data
    the model is fitted and tuned on). The share seen in train alone is kept as a descriptive figure.
    Writes the sidecar and returns the summary that goes into cross_dataset_overlap.json.
    """
    side = pd.DataFrame({"_key": target_meta["_key"].to_numpy()})
    y, fam = target_meta["_y"].to_numpy(), target_meta["_label"].astype(str).to_numpy()
    out = {"s7_definition": S7_DEFINITION, "compared_with": f"{against} train + val",
           "built_from": {against: provenance(against), target: provenance(target)},
           "features_sha256": hashlib.sha256(json.dumps(list(target_frame.columns)).encode()).hexdigest()}
    hf, htr, ht = _hashes(fit_frame), _hashes(train_frame), _hashes(target_frame)
    for s in SIG_FIGS:
        seen = np.isin(ht[s], np.unique(hf[s]))
        side[f"seen_sig{s}"] = seen
        out[f"sig{s}"] = subset_summary(seen, y, fam, block, np.ones(len(y), bool))
        out[f"sig{s}"]["share_seen_in_source_train_only"] = float(np.isin(ht[s], np.unique(htr[s])).mean())
    out["reproduces_previous_sidecar"] = _same_as_previous(side, sidecar_path(target, against))
    out["sidecar_sha256"] = write_parquet(side, sidecar_path(target, against))
    return out
