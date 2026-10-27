"""Model development on CICIDS2017 train and val only (amendment 1.4, preregistration §6a).

For one candidate in one feature space:
1. grid search with the grid seed: fit on train, score val at the val-optimal threshold;
2. the best grid entry (ties within the margin go to the earlier entry) is trained with every seed;
3. per seed: threshold on val, val metrics, model file in Drive with its SHA-256.

The record goes to results/models/development/<space>/<candidate>.json after every fit, so a disconnected run
resumes where it stopped. Test partitions and external targets are never loaded here (ADR-0007).
"""
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from ..data import datasets, near
from ..evaluation.metrics import fpr_at_detection_rate, pr_auc
from ..utils import paths
from ..utils.io import sha256_file, write_json
from ..utils.logging import run_info
from .candidates import Candidate, best_threshold, make

MODELS_CONFIG = paths.CONFIG / "models.yaml"
SEEDS_CONFIG = paths.CONFIG / "seeds.yaml"
SOURCE = "cicids2017_improved"
DEV_PARTITIONS = ("train", "val")


def load_models_config(path=None):
    return yaml.safe_load(Path(path or MODELS_CONFIG).read_text())


def seeds():
    return list(yaml.safe_load(SEEDS_CONFIG.read_text())["seeds"])


def record_path(space, name):
    return paths.RESULTS / "models" / "development" / space / f"{name}.json"


def model_path(space, name, seed):
    return paths.MODELS / "development" / space / name / f"seed{seed}.pkl"


def features(space, X):
    """The frozen feature space applied to a CICIDS2017 frame."""
    from ..data import harmonise
    if space == "cic":
        shared = yaml.safe_load((harmonise.MAPPING_DIR / "cic_shared_features.yaml").read_text())["features"]
        if not shared:
            raise RuntimeError("cic_shared_features.yaml is empty; run scripts/preprocess.py --shared-features")
        return X[shared]
    if space == "unsw_canonical":
        return harmonise.to_canonical_unsw(X, "cic")
    raise ValueError(f"unknown feature space {space}")


def load_dev(space, partitions=DEV_PARTITIONS):
    bad = set(partitions) - set(DEV_PARTITIONS)
    if bad:
        raise PermissionError(f"development never loads {sorted(bad)} (sealed until freeze, ADR-0007)")
    X, meta = datasets.load(SOURCE, partitions=partitions)
    F = features(space, X)
    out = {}
    for p in partitions:
        m = (meta["_partition"] == p).to_numpy()
        out[p] = (F.loc[m].to_numpy(dtype=np.float32), meta.loc[m, "_y"].to_numpy(dtype=np.int64))
    return out, list(F.columns)


def val_metrics(y, p, thr):
    pred = (p >= thr).astype(np.int64)
    out = {}
    for cls, name in ((0, "benign"), (1, "attack")):
        tp = int(((pred == cls) & (y == cls)).sum())
        fp = int(((pred == cls) & (y != cls)).sum())
        fn = int(((pred != cls) & (y == cls)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    neg = (y == 0).sum()
    return {"macro_f1": (out["benign"]["f1"] + out["attack"]["f1"]) / 2,
            "balanced_accuracy": (out["benign"]["recall"] + out["attack"]["recall"]) / 2,
            "fpr": float(((pred == 1) & (y == 0)).sum() / neg) if neg else 0.0,
            "pr_auc": pr_auc(y, p), "fpr_at_tpr_0.95": fpr_at_detection_rate(y, p, 0.95), "per_class": out}


def _grid(spec):
    g = spec.get("grid", {})
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))] or [{}]


def _start_record(space, name, spec, cfg_path, feats, data):
    rec = {"candidate": name, "feature_space": space, "source": SOURCE, "spec": spec,
           "models_config_sha256": sha256_file(cfg_path), "seeds": seeds(),
           "source_provenance": near.provenance(SOURCE), "features": feats,
           "rows": {p: {"benign": int((y == 0).sum()), "attack": int((y == 1).sum())} for p, (_, y) in data.items()},
           "grid": [], "final": {}, "complete": False, "runs": []}
    return rec


def develop(space, name, cfg_path=MODELS_CONFIG, log=print):
    cfg = load_models_config(cfg_path)
    spec, dev = cfg["candidates"][name], cfg["development"]
    if spec["family"] == "neural" and not os.environ.get("PHD_ALLOW_CPU_NEURAL"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name} needs a GPU runtime (Runtime -> Change runtime type -> T4 GPU); on the CPU "
                               "it would take days, and shortening training is not allowed (preregistration §10)")
    data, feats = load_dev(space)
    (X, y), (Xv, yv) = data["train"], data["val"]
    rp = record_path(space, name)
    rec = json.loads(rp.read_text()) if rp.exists() else None
    if rec and rec["models_config_sha256"] != sha256_file(cfg_path):
        raise RuntimeError(f"{rp.name} was made under another config/models.yaml; it cannot be resumed. "
                           "Log an amendment and move the old record aside first.")
    if rec and rec["source_provenance"] != near.provenance(SOURCE):
        raise RuntimeError(f"{SOURCE} outputs changed since {rp.name} was started")
    rec = rec or _start_record(space, name, spec, cfg_path, feats, data)
    if rec["complete"]:
        log(f"[{space}/{name}] already complete")
        return rec
    rec["runs"].append(run_info(Path(cfg_path)))

    # 1. grid search with the grid seed
    grid, gseed = _grid(spec), dev["grid_seed"]
    done = {json.dumps(g["params"], sort_keys=True) for g in rec["grid"]}
    for params in grid:
        key = json.dumps(params, sort_keys=True)
        if key in done:
            continue
        m = make(name, spec, params, gseed).fit(X, y, Xv, yv)
        p = m.predict_proba(Xv)
        thr, mf = best_threshold(yv, p)
        rec["grid"].append({"params": params, "seed": gseed, "threshold": thr, "val_macro_f1": mf, "info": m.info})
        write_json(rp, rec)
        log(f"[{space}/{name}] grid {len(rec['grid'])}/{len(grid)} {params}: val macro-F1 {mf:.6f} "
            f"({m.info['train_seconds']} s)")
    order = {json.dumps(g, sort_keys=True): i for i, g in enumerate(grid)}
    scored = sorted(rec["grid"], key=lambda g: order[json.dumps(g["params"], sort_keys=True)])
    top = max(g["val_macro_f1"] for g in scored)
    margin = float(dev["tie_margin"])
    best = next(g for g in scored if g["val_macro_f1"] >= top - margin)
    rec["selected_params"] = best["params"]

    # 2-3. every seed with the selected entry
    for s in seeds():
        if str(s) in rec["final"]:
            continue
        m = make(name, spec, best["params"], s).fit(X, y, Xv, yv)
        p = m.predict_proba(Xv)
        thr, _ = best_threshold(yv, p)
        path = model_path(space, name, s)
        sha = m.save(path)
        rec["final"][str(s)] = {"threshold": thr, "val": val_metrics(yv, p, thr), "info": m.info,
                                "model_file": path.relative_to(paths.MODELS).as_posix(), "model_sha256": sha}
        write_json(rp, rec)
        log(f"[{space}/{name}] seed {s}: val macro-F1 {rec['final'][str(s)]['val']['macro_f1']:.6f}")
    v = [rec["final"][str(s)]["val"]["macro_f1"] for s in seeds()]
    rec["val_macro_f1_mean"], rec["val_macro_f1_std"] = float(np.mean(v)), float(np.std(v, ddof=1))
    rec["complete"] = True
    rec["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(rp, rec)
    log(f"[{space}/{name}] done: val macro-F1 {rec['val_macro_f1_mean']:.6f} ± {rec['val_macro_f1_std']:.6f}")
    return rec


def load_model(space, name, seed, expected_sha=None):
    path = model_path(space, name, seed)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"{path} differs from its recorded SHA-256")
    return Candidate.load(path)


def summary_table(cfg_path=MODELS_CONFIG):
    cfg = load_models_config(cfg_path)
    rows = []
    for space in cfg["feature_spaces"]:
        for name in cfg["candidates"]:
            rp = record_path(space, name)
            r = json.loads(rp.read_text()) if rp.exists() else {}
            rows.append({"feature_space": space, "candidate": name, "complete": r.get("complete", False),
                         "grid_done": f"{len(r.get('grid', []))}/{len(_grid(cfg['candidates'][name]))}",
                         "seeds_done": len(r.get("final", {})), "selected": r.get("selected_params"),
                         "val_macro_f1_mean": r.get("val_macro_f1_mean"), "val_macro_f1_std": r.get("val_macro_f1_std")})
    return rows
