"""Exploratory analyses planned in docs/protocol/exploratory_analyses.md (outside the confirmatory family).

    python scripts/explore.py --status                                 # which E1 results exist, nothing is read
    python scripts/explore.py --operating-point                        # E1 on every evaluation set, resumes
    python scripts/explore.py --operating-point --eval unsw_nb15

E1 runs only while the Levels 1-2 freeze is intact and only on the per-row scores of notebook 08, after checking
that they reproduce the confirmatory records. Logs go to results/exploratory/logs/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def start_log(what):
    import datetime as dt
    from src.utils import paths
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocess import Tee
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = paths.RESULTS / "exploratory" / "logs" / f"{stamp}_{what}.log"
    sys.stdout = Tee(log, sys.stdout)
    sys.stderr = Tee(log, sys.stderr)
    print(f"command: {' '.join(sys.argv)}\nlog: {log}")


def status():
    from src.evaluation import operating_point as E
    lines = ["| analysis | evaluation set | space | written |", "|---|---|---|---|"]
    for e, sp in E.SETS:
        lines.append(f"| E1 | {e} | {sp} | {'yes' if E.result_path(e, sp).exists() else 'no'} |")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--operating-point", action="store_true", help="E1: ranking versus operating point")
    ap.add_argument("--eval", nargs="+", help="with --operating-point: evaluation sets (default all)")
    a = ap.parse_args()
    if a.status:
        return status()
    if not a.operating_point:
        ap.error("give --operating-point or --status")
    from src.evaluation import operating_point as E
    sets = [s for s in E.SETS if not a.eval or s[0] in a.eval]
    if a.eval and not sets:
        ap.error(f"--eval must be from {sorted({e for e, _ in E.SETS})}")
    start_log("operating_point")
    E.run(sets)
    print((E.out_dir() / "summary.md").read_text())
    status()


if __name__ == "__main__":
    main()
