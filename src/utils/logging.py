"""Run logging: config hash, git commit and package versions saved with every result."""
import datetime as dt
import platform
import subprocess
from importlib import metadata

from .io import sha256_file
from .paths import REPO

PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "PyYAML"]


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except OSError:
        return None


def run_info(config_path=None):
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    info = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(), "python": platform.python_version(), "packages": versions}
    if config_path:
        info["config"] = str(config_path.relative_to(REPO)) if str(config_path).startswith(str(REPO)) else str(config_path)
        info["config_sha256"] = sha256_file(config_path)
    return info
