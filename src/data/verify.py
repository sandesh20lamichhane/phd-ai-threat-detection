"""Check that processed outputs were built under the current rules and are the files the reports describe.

Amendment 1.2: before any model run, every dataset it uses must pass (preregistration §10a).

Primary data (processed parquet and split files), per dataset:
    config  the dataset's settings and label rules (load.dataset_fingerprint) equal the current config's
    pilot   not a --max-rows run
    files   every output and split file exists and its SHA-256 matches the report

Auxiliary sensitivity-analysis artifacts (amendment 1.3), per S7 record:
    record  the JSON record matches its SHA-256 in results/data_audit/auxiliary_manifest.json
    sidecar built under the current S7 definition, from the current outputs, SHA-256 as recorded (near.check_sidecar)
"""
import hashlib
import json
import subprocess

from ..utils import paths
from ..utils.io import sha256_file, write_json
from . import near
from .load import CONFIG_PATH, dataset_fingerprint, load_config

CONFIRMATORY = ["cicids2017_improved", "csecicids2018_improved", "unsw_nb15", "ciciomt2024", "ton_iot",
                "ciciot2023", "lanl_cyber1", "nsl_kdd"]
CONFIG_REL = "config/preprocessing.yaml"


class StaleOutput(RuntimeError):
    pass


def _report(ds_id):
    p = paths.RESULTS / "data_audit" / ds_id / "report.json"
    return json.loads(p.read_text()) if p.exists() else None


def _historical_fingerprint(ds_id, config_sha):
    """Reports written before 1.2 carry only the whole-file config SHA: find that file in git history."""
    log = subprocess.run(["git", "-C", str(paths.REPO), "log", "--format=%H", "--", CONFIG_REL],
                         capture_output=True, text=True)
    for commit in log.stdout.split():
        blob = subprocess.run(["git", "-C", str(paths.REPO), "show", f"{commit}:{CONFIG_REL}"], capture_output=True)
        if blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == config_sha:
            cfg = load_config(text=blob.stdout.decode())
            return dataset_fingerprint(cfg, ds_id) if ds_id in cfg["datasets"] else None
    return None


def config_status(ds_id, report=None, config_path=CONFIG_PATH):
    """('current' | 'stale' | 'unknown', detail)."""
    report = report or _report(ds_id)
    now = dataset_fingerprint(load_config(config_path), ds_id)
    then = report.get("dataset_config_sha256")
    how = "report"
    if then is None:
        sha = report.get("run", {}).get("config_sha256")
        then = _historical_fingerprint(ds_id, sha) if sha else None
        how = "git history of the config"
    if then is None:
        return "unknown", "config used for this output not found in git history"
    if then != now:
        return "stale", (f"built under different rules for this dataset (protocol {report.get('protocol_version')}, "
                         f"code {str(report.get('run', {}).get('git_commit'))[:7]}); re-run its preprocessing")
    return "current", f"rules identical to the current config (checked via {how})"


def require_current(ds_id, report):
    status, detail = config_status(ds_id, report)
    if status != "current":
        raise StaleOutput(f"{ds_id}: {detail}")


def _files(ds_id, report):
    out = []
    base = paths.PROCESSED / ds_id / report.get("output_version", "v1")
    for name, sha in report.get("outputs", {}).get("parquet_sha256", {}).items():
        out.append((base / name, sha))
    for name, sp in (report.get("splits") or {}).items():
        out.append((paths.SPLITS / ds_id / f"{name}.parquet", sp["sha256"]))
    return out


def verify_dataset(ds_id, checksums=True):
    r = _report(ds_id)
    if r is None:
        return {"dataset": ds_id, "ok": False, "problems": ["no report: dataset not preprocessed"]}
    problems = []
    status, detail = config_status(ds_id, r)
    if status != "current":
        problems.append(detail)
    if r.get("max_rows"):
        problems.append(f"pilot output (--max-rows {r['max_rows']})")
    files = _files(ds_id, r)
    bad = []
    if checksums:
        for p, sha in files:
            if not p.exists():
                bad.append(f"missing {p.name}")
            elif sha256_file(p) != sha:
                bad.append(f"checksum differs: {p.name}")
    problems += bad
    return {"dataset": ds_id, "ok": not problems, "config": status, "config_detail": detail,
            "protocol_version": r.get("protocol_version"), "code": r.get("run", {}).get("git_commit"),
            "built_utc": r.get("run", {}).get("utc"), "files_checked": len(files) if checksums else 0,
            "problems": problems}


def verify_auxiliary():
    """One row per S7 record: within-dataset near-duplicate records and external-target records."""
    mp = paths.RESULTS / "data_audit" / near.MANIFEST
    man = json.loads(mp.read_text()) if mp.exists() else {}
    rows = []
    items = [(d, None) for d in near.WITHIN] + list(near.EXTERNAL.items())
    for ds_id, against in items:
        rec = near.record_path(ds_id, against)
        rel = rec.relative_to(paths.RESULTS).as_posix()
        problems = []
        if not rec.exists():
            problems.append(f"missing {rel}")
        elif rel not in man:
            problems.append(f"{rel} not in {near.MANIFEST}")
        elif sha256_file(rec) != man[rel]:
            problems.append(f"{rel} differs from its SHA-256 in {near.MANIFEST}")
        if rec.exists():
            problems += near.check_sidecar(ds_id, against, use_cache=False)
        est = None
        if not problems:
            r = near._record(ds_id, against)
            sub = r["sig3"] if against else r["subsets"]["sig3"].get("test", {})
            est = sub.get("estimable")
        rows.append({"artifact": f"S7 {ds_id}" + (f" vs {against}" if against else ""), "record": rel,
                     "sidecar": near.sidecar_path(ds_id, against).name, "ok": not problems,
                     "s7_test_estimable_3sf": est, "problems": sorted(set(problems))})
    return rows


def verify_all(ds_ids=CONFIRMATORY, checksums=True, auxiliary=True, log=print):
    """Returns (primary rows, auxiliary rows) and writes results/data_audit/verification.{json,md}."""
    rows = []
    for d in ds_ids:
        log(f"verifying {d}")
        rows.append(verify_dataset(d, checksums))
        log(f"  {'OK' if rows[-1]['ok'] else 'FAIL'}: {rows[-1]['problems'] or rows[-1]['config_detail']}")
    aux = verify_auxiliary() if auxiliary else []
    for a in aux:
        log(f"  {'OK' if a['ok'] else 'FAIL'}: {a['artifact']} {a['problems'] or ''}")
    write_json(paths.RESULTS / "data_audit" / "verification.json",
               {"config_sha256": sha256_file(CONFIG_PATH), "s7_definition": near.S7_DEFINITION,
                "primary_data": rows, "auxiliary_s7_artifacts": aux})
    lines = ["# Output verification", "",
             "_Generated by `scripts/preprocess.py --verify` (amendments 1.2, 1.3)._", "",
             "## Primary data: processed parquet and split files", "",
             "| dataset | ok | rules | protocol | code | files checked | problems |", "|---|---|---|---|---|---:|---|"]
    for x in rows:
        lines.append(f"| {x['dataset']} | {'yes' if x['ok'] else 'NO'} | {x.get('config', '')} | "
                     f"{x.get('protocol_version', '')} | {str(x.get('code') or '')[:7]} | {x.get('files_checked', 0)} | "
                     f"{'; '.join(x['problems']) or 'none'} |")
    if aux:
        lines += ["", f"## Auxiliary sensitivity-analysis artifacts (S7 definition {near.S7_DEFINITION})", "",
                  "| artifact | ok | record | sidecar | S7 test estimable (3 s.f.) | problems |",
                  "|---|---|---|---|---|---|"]
        for a in aux:
            est = {True: "yes", False: "no (reported as not estimable)", None: ""}[a["s7_test_estimable_3sf"]]
            lines.append(f"| {a['artifact']} | {'yes' if a['ok'] else 'NO'} | `{a['record']}` | `{a['sidecar']}` | "
                         f"{est} | {'; '.join(a['problems']) or 'none'} |")
    (paths.RESULTS / "data_audit" / "verification.md").write_text("\n".join(lines) + "\n")
    return rows, aux
