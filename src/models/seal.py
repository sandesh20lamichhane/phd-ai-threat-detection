"""The sealed test (ADR-0007): test partitions and external targets are scored only after the freeze.

`freeze` runs once, when every candidate is fully developed in every feature space. It applies the
preregistered rule of config/models.yaml to pick the proposed detector from validation results alone and
writes results/models/freeze.json with the SHA-256 of every development record and model file.
`require_frozen` is called by any code that scores test or target data: the freeze record must exist, be
committed to git unchanged, and every model file must still match its SHA-256.
"""
import json
import subprocess
import time
from pathlib import Path

from ..utils import paths
from ..utils.io import sha256_file, write_json
from ..utils.logging import run_info
from .develop import MODELS_CONFIG, load_models_config, record_path, seeds

FREEZE = "freeze.json"


class Sealed(PermissionError):
    pass


def freeze_path():
    return paths.RESULTS / "models" / FREEZE


def freeze(cfg_path=MODELS_CONFIG, log=print):
    fp = freeze_path()
    if fp.exists():
        raise Sealed(f"{fp} already exists: the freeze happens once. Changing it needs a logged amendment.")
    cfg = load_models_config(cfg_path)
    cfg_sha = sha256_file(cfg_path)
    records, missing = {}, []
    for space in cfg["feature_spaces"]:
        for name in cfg["candidates"]:
            rp = record_path(space, name)
            r = json.loads(rp.read_text()) if rp.exists() else None
            if not r or not r.get("complete") or len(r.get("final", {})) != len(seeds()):
                missing.append(f"{space}/{name}")
                continue
            if r["models_config_sha256"] != cfg_sha:
                raise Sealed(f"{rp} was made under another config/models.yaml")
            records[f"{space}/{name}"] = r
    if missing:
        raise Sealed(f"development not complete for {missing}; finish them before the freeze")
    rule = cfg["proposed"]
    space, margin = rule["feature_space"], float(rule["tie_margin"])
    ranking = [(name, records[f"{space}/{name}"]["val_macro_f1_mean"]) for name in cfg["candidates"]]
    top = max(v for _, v in ranking)
    proposed = next(n for n, v in ranking if v >= top - margin)      # candidates in config order
    files = {}
    for key, r in records.items():
        for s, f in r["final"].items():
            path = paths.MODELS / f["model_file"]
            if sha256_file(path) != f["model_sha256"]:
                raise Sealed(f"{path} differs from its recorded SHA-256")
            files[f["model_file"]] = f["model_sha256"]
    out = {"proposed": proposed, "rule": rule, "ranking_val_macro_f1": dict(ranking),
           "models_config_sha256": cfg_sha,
           "development_records_sha256": {k: sha256_file(record_path(*k.split("/"))) for k in records},
           "model_files_sha256": files, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "run": run_info(Path(cfg_path))}
    write_json(fp, out)
    log(f"proposed detector: {proposed} (val macro-F1 {dict(ranking)[proposed]:.4f}); freeze written to {fp}")
    return out


def _git(*args):
    return subprocess.run(["git", "-C", str(paths.REPO), *args], capture_output=True, text=True)


def require_frozen(require_git=True):
    """Returns the freeze record, or raises Sealed. Called before any test or target is scored."""
    fp = freeze_path()
    if not fp.exists():
        raise Sealed("test and target data are sealed: run scripts/develop.py --freeze first (ADR-0007)")
    if require_git:
        rel = fp.relative_to(paths.REPO).as_posix()
        if _git("ls-files", "--error-unmatch", rel).returncode != 0 or _git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
            raise Sealed(f"{rel} must be committed to git unchanged before anything is scored")
    fr = json.loads(fp.read_text())
    for f, sha in fr["model_files_sha256"].items():
        if sha256_file(paths.MODELS / f) != sha:
            raise Sealed(f"model file {f} changed after the freeze")
    for k, sha in fr["development_records_sha256"].items():
        if sha256_file(record_path(*k.split("/"))) != sha:
            raise Sealed(f"development record {k} changed after the freeze")
    return fr
