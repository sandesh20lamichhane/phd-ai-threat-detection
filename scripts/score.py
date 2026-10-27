"""Score the frozen models once on the sealed test data, then run the confirmatory analysis (ADR-0007).

    python scripts/score.py --status                                  # which units are scored, nothing is read
    python scripts/score.py --level12                                 # every Level 1-2 unit, resumes if interrupted
    python scripts/score.py --level12 --eval unsw_nb15 --candidate random_forest
    python scripts/score.py --lanl                                    # every Level 3 unit (frozen rules 1.6)
    python scripts/score.py --analyse                                 # H1a-H1c, H2, S7, Holm -> decisions

Nothing runs unless the freeze record of its level is committed unchanged and every frozen file matches its
SHA-256 (src/models/seal.py). Logs go to results/confirmatory/logs/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def start_log(what):
    import datetime as dt
    from src.utils import paths
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocess import Tee
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = paths.RESULTS / "confirmatory" / "logs" / f"{stamp}_{what}.log"
    sys.stdout = Tee(log, sys.stdout)
    sys.stderr = Tee(log, sys.stderr)
    print(f"command: {' '.join(sys.argv)}\nlog: {log}")


def status():
    from src.models import develop as D
    from src.models import lanl_develop as L
    from src.models import score as S
    cfg = D.load_models_config()
    lines = ["| level | evaluation set | space | candidate | seeds scored |", "|---|---|---|---|---:|"]
    for e, es in S.EVAL_SETS.items():
        for sp in es["spaces"]:
            for c in cfg["candidates"]:
                n = sum(S.unit_path(e, sp, c, s).exists() for s in D.seeds())
                lines.append(f"| 1-2 | {e} | {sp} | {c} | {n}/{len(D.seeds())} |")
    for c in L.load_config()["candidates"]:
        n = sum(S.lanl_unit_path(c, s).exists() for s in D.seeds())
        lines.append(f"| 3 | lanl_cyber1 days 14-58 | host-hour | {c} | {n}/{len(D.seeds())} |")
    dec = S.out_dir() / "decisions.json"
    lines += ["", f"confirmatory analysis: {'written ' + json.loads(dec.read_text())['generated_utc'] if dec.exists() else 'not run'}"]
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--level12", action="store_true")
    ap.add_argument("--lanl", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--eval", nargs="+", help="with --level12: evaluation sets (default all, small ones first)")
    ap.add_argument("--space", nargs="+")
    ap.add_argument("--candidate", nargs="+")
    a = ap.parse_args()
    if a.status:
        return status()
    if not (a.level12 or a.lanl or a.analyse):
        ap.error("give --level12, --lanl, --analyse or --status")
    from src.models import score as S
    if a.eval and any(e not in S.EVAL_SETS for e in a.eval):
        ap.error(f"--eval must be from {list(S.EVAL_SETS)}")
    start_log("_".join(k for k in ("level12", "lanl", "analyse") if getattr(a, k)))
    if a.level12:
        S.score_level12(a.eval, a.space, a.candidate)
    if a.lanl:
        S.score_lanl(a.candidate)
    if a.analyse:
        from src.evaluation.confirmatory import analyse
        analyse()
        print((S.out_dir() / "decisions.md").read_text())
    status()


if __name__ == "__main__":
    main()
