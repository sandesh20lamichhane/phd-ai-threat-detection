"""Download the thesis datasets from their official sources into a raw-data folder.

Designed for Google Colab with Google Drive mounted, so tens of GB never pass through
a laptop. Every file is streamed to local Colab disk, size-checked, SHA-256 hashed,
then moved into Drive. A MANIFEST.json per dataset records provenance, and a copy of
each manifest is written to the repo (results/data_manifest/) so it can be committed.

Re-running is safe: finished files are skipped and partial downloads resume.

    python -m src.acquire.fetch --plan                 # list files and sizes, no download
    python -m src.acquire.fetch                        # download every enabled dataset
    python -m src.acquire.fetch --only nsl_kdd ciciot2023
    python -m src.acquire.fetch --verify               # re-hash what is in Drive
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import requests
import yaml

REPO = Path(__file__).resolve().parents[2]
UA = {"User-Agent": "Mozilla/5.0 (thesis dataset fetcher; academic research use)",
      "Accept-Encoding": "identity"}  # sizes must match Content-Length byte for byte
SITE_HEADERS = {}  # host -> extra headers (e.g. a browser session cookie), from gated_urls.yaml "site_headers"
CHUNK = 8 << 20
GB = 1 << 30


# ---------------------------------------------------------------- helpers
def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def dir_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def headers_for(url):
    h = dict(UA)
    h.update(SITE_HEADERS.get(urlparse(url).hostname or "", {}))
    h["Accept-Encoding"] = "identity"
    return h


def headers_from_curl(cmd):
    """Headers worth keeping from a browser's 'Copy as cURL': cookie, user agent, referer."""
    import shlex
    words = shlex.split(cmd.replace("\\\n", " "))
    keep, url = {}, next((w for w in words if w.startswith("http")), "")
    for flag, val in zip(words, words[1:]):
        if flag in ("-H", "--header") and ":" in val:
            k, v = val.split(":", 1)
            if k.strip().lower() in ("cookie", "user-agent", "referer"):
                keep[k.strip().title()] = v.strip()
        elif flag in ("-b", "--cookie"):
            keep["Cookie"] = val
        elif flag in ("-A", "--user-agent"):
            keep["User-Agent"] = val
    return urlparse(url).hostname, keep


def remote_size(url):
    try:
        r = requests.head(url, headers=headers_for(url), allow_redirects=True, timeout=60)
        n = r.headers.get("Content-Length")
        return int(n) if r.ok and n else None
    except requests.RequestException:
        return None


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def _matches(text, include=None, exclude=None):
    if include and not re.search(include, text):
        return False
    if exclude and re.search(exclude, text):
        return False
    return True


# ---------------------------------------------------------------- listing sources
def list_http_index(url, include=None, exclude=None, max_depth=6):
    """Crawl an Apache/nginx directory listing below `url`; return [(name, url, size)].

    `name` keeps the sub-path below the start URL so folder structure is preserved.
    """
    if not url.endswith("/"):
        url += "/"
    root, out, seen = url, [], set()

    def walk(u, depth):
        if u in seen or depth > max_depth:
            return
        seen.add(u)
        r = requests.get(u, headers=headers_for(u), timeout=120)
        r.raise_for_status()
        p = _Links()
        p.feed(r.text)
        for href in p.hrefs:
            if href.startswith(("?", "#", "mailto:")):
                continue
            full = urljoin(u, href)
            if not full.startswith(root) or full == u:
                continue  # parent links, sort links, other sites
            if full.endswith("/"):
                walk(full, depth + 1)
            elif _matches(full, include, exclude):
                rel = unquote(full[len(root):])
                out.append((rel, full, None))

    walk(url, 0)
    return out


def list_s3(bucket, region, prefix, include=None, exclude=None):
    base = f"https://{bucket}.s3.{region}.amazonaws.com/"
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(base, params=params, headers=UA, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s3:Contents", ns):
            key = c.find("s3:Key", ns).text
            size = int(c.find("s3:Size", ns).text)
            if key.endswith("/") or not _matches(key, include, exclude):
                continue
            out.append((key[len(prefix):], base + quote(key), size))
        if root.findtext("s3:IsTruncated", namespaces=ns) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)


def url_name(u):
    """File name for a URL; download.php?file=CSV%2FCSV.zip style links use the file parameter."""
    q = parse_qs(urlparse(u).query)
    for key in ("file", "filename", "f"):
        if q.get(key):
            return Path(q[key][0]).name
    return unquote(Path(urlparse(u).path).name)


def list_urls(urls):
    return [(url_name(u), u, None) for u in urls]


# ---------------------------------------------------------------- the fetcher
class Fetcher:
    def __init__(self, drive_root, mydrive, tmp, gated, budget_gb):
        self.raw = Path(drive_root) / "data" / "raw"
        self.mydrive = Path(mydrive)
        self.tmp = Path(tmp)
        self.gated = gated
        self.budget = budget_gb * GB
        self.raw.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)

    # ---- config resolution
    def resolve(self, value):
        if isinstance(value, str) and value.startswith("gated:"):
            key = value.split(":", 1)[1]
            got = self.gated.get(key)
            if not got:
                raise LookupError(f"needs '{key}' in gated_urls.yaml (register with the provider first)")
            return got
        return value

    def listing(self, src):
        t = src["type"]
        if t == "urls":
            urls = self.resolve(src["urls"])
            return list_urls([urls] if isinstance(urls, str) else urls)
        if t == "s3":
            return list_s3(src["bucket"], src["region"], src["prefix"], src.get("include"), src.get("exclude"))
        if t == "http_index":
            return list_http_index(self.resolve(src["url"]), src.get("include"), src.get("exclude"))
        if t == "drive_copy":
            folder = self.mydrive / src["path"]
            if not folder.is_dir():
                return []
            return [(p.name, str(p), p.stat().st_size) for p in sorted(folder.iterdir())
                    if p.is_file() and _matches(p.name, src.get("include"), src.get("exclude"))]
        if t in ("kaggle", "manual"):
            return []  # nothing listable in advance
        raise ValueError(f"unknown source type {t}")

    # ---- manifest
    def _manifest_path(self, ds):
        return self.raw / ds["id"] / "MANIFEST.json"

    def load_manifest(self, ds):
        p = self._manifest_path(ds)
        if p.exists():
            return json.loads(p.read_text())
        return {"dataset": ds["id"], "name": ds["name"], "files": {}}

    def save_manifest(self, ds, m):
        m["updated_utc"] = now()
        m["total_bytes"] = sum(f["bytes"] for f in m["files"].values())
        text = json.dumps(m, indent=2, sort_keys=True)
        self._manifest_path(ds).write_text(text)
        repo_copy = REPO / "results" / "data_manifest" / f"{ds['id']}.json"
        repo_copy.parent.mkdir(parents=True, exist_ok=True)
        repo_copy.write_text(text)

    def record(self, ds, m, rel, dest, origin, method):
        m["files"][rel] = {"bytes": dest.stat().st_size, "sha256": sha256(dest),
                           "source": origin, "method": method, "retrieved_utc": now()}
        self.save_manifest(ds, m)

    # ---- transfers
    def download(self, url, part, expected=None):
        """Stream url to `part`, resuming if a partial file exists. Returns final size."""
        have = part.stat().st_size if part.exists() else 0
        headers = headers_for(url)
        if have:
            headers["Range"] = f"bytes={have}-"
        with requests.get(url, headers=headers, stream=True, timeout=300) as r:
            if have and r.status_code == 200:  # server ignored Range; start over
                have = 0
            elif have and r.status_code == 416:
                return have
            if r.status_code == 403:
                raise IOError(f"403 Forbidden for {url}: the site wants your browser's sign-in; "
                              "add its cookie with the 'site_headers' step in the notebook")
            r.raise_for_status()
            if "text/html" in r.headers.get("Content-Type", "") and not url_name(url).endswith((".html", ".htm")):
                raise IOError(f"got a web page instead of a file for {url}; the link may need you to "
                              "be signed in, so open it in your browser and copy the link again")
            total = expected or (int(r.headers["Content-Length"]) + have if "Content-Length" in r.headers else None)
            done, last = have, -1
            with open(part, "ab" if have else "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    pct = int(100 * done / total) if total else None
                    if pct is not None and pct // 5 != last:
                        last = pct // 5
                        print(f"\r    {human(done)} / {human(total)} ({pct}%)", end="", flush=True)
            print()
        size = part.stat().st_size
        if total and size != total:
            raise IOError(f"size mismatch for {url}: got {size}, expected {total}")
        return size

    def fetch_file(self, ds, m, rel, origin, size, method):
        dest = self.raw / ds["id"] / rel
        if rel in m["files"] and dest.exists() and dest.stat().st_size == m["files"][rel]["bytes"]:
            print(f"  = {rel} already in Drive, skipped")
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  + {rel} ({human(size)})")
        if method == "drive_copy":
            shutil.copy2(origin, dest)
        else:
            part = self.tmp / ds["id"] / (rel + ".part")
            part.parent.mkdir(parents=True, exist_ok=True)
            self.download(origin, part, size)
            shutil.move(str(part), dest)  # local disk -> Drive
        self.record(ds, m, rel, dest, origin, method)
        return dest.stat().st_size

    def fetch_kaggle(self, ds, m, src):
        if shutil.which("kaggle") is None or not (os.environ.get("KAGGLE_KEY") or Path("~/.kaggle/kaggle.json").expanduser().exists()):
            raise LookupError("Kaggle token not set (add KAGGLE_USERNAME and KAGGLE_KEY to Colab secrets)")
        work = self.tmp / ds["id"] / "kaggle"
        work.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", src["slug"], "-p", str(work)], check=True)
        n = 0
        for p in sorted(work.iterdir()):
            n += self.fetch_file(ds, m, p.name, str(p), p.stat().st_size, "drive_copy")
            m["files"][p.name]["source"] = f"kaggle:{src['slug']}"
            m["files"][p.name]["method"] = "kaggle"
            self.save_manifest(ds, m)
            p.unlink()
        return n

    def adopt_manual(self, ds, m, src):
        """Checksum files the user uploaded into the dataset folder by hand."""
        folder = self.raw / ds["id"]
        files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"]
        if not files:
            raise LookupError("manual step not done yet: " + " ".join(src["instructions"].split()))
        for p in files:
            rel = str(p.relative_to(folder))
            if rel not in m["files"] or m["files"][rel]["bytes"] != p.stat().st_size:
                print(f"  # hashing {rel}")
                self.record(ds, m, rel, p, "manual upload", "manual")
        return 0

    # ---- per dataset
    def plan(self, ds):
        for src in ds["sources"]:
            try:
                items = self.listing(src)
            except Exception as e:  # noqa: BLE001 - report and try the next source
                print(f"  [{src['type']}] unavailable: {e}")
                continue
            if src["type"] in ("kaggle", "manual"):
                print(f"  [{src['type']}] {src.get('slug') or ' '.join(src['instructions'].split())[:160]}...")
                continue
            if not items:
                print(f"  [{src['type']}] no matching files")
                continue
            total = 0
            for rel, url, size in items:
                size = size if size is not None else remote_size(url)
                total += size or 0
                print(f"    {human(size):>10}  {rel}")
            print(f"  [{src['type']}] {len(items)} files, {human(total)} -> would use this source")
            return total
        return 0

    def run(self, ds):
        m = self.load_manifest(ds)
        errors = []
        for src in ds["sources"]:
            t = src["type"]
            try:
                if t == "kaggle":
                    self.fetch_kaggle(ds, m, src)
                    return True
                if t == "manual":
                    self.adopt_manual(ds, m, src)
                    return True
                items = self.listing(src)
                if not items:
                    errors.append(f"[{t}] no matching files")
                    continue
                need = sum((s if s is not None else remote_size(u)) or 0
                           for rel, u, s in items if rel not in m["files"])
                used = dir_bytes(self.raw)
                if used + need > self.budget:
                    raise RuntimeError(f"would exceed the {human(self.budget)} budget "
                                       f"(raw already {human(used)}, this needs {human(need)})")
                for rel, url, size in items:
                    self.fetch_file(ds, m, rel, url, size, t)
                check_columns(ds, self.raw / ds["id"])
                return True
            except Exception as e:  # noqa: BLE001 - fall through to the next source
                errors.append(f"[{t}] {e}")
                print(f"  ! {t}: {e}")
        print(f"  x {ds['id']} not fetched: " + "; ".join(errors))
        return False

    def verify(self, ds):
        m = self.load_manifest(ds)
        if not m["files"]:
            print("  nothing downloaded yet")
            return False
        bad = 0
        for rel, info in m["files"].items():
            p = self.raw / ds["id"] / rel
            ok = p.exists() and p.stat().st_size == info["bytes"] and sha256(p) == info["sha256"]
            bad += not ok
            print(f"  {'ok ' if ok else 'BAD'} {rel}")
        if m["files"] and not bad:
            self.save_manifest(ds, m)  # re-export the repo copy, e.g. into a fresh clone
        return bad == 0


def check_columns(ds, folder):
    """Cheap sanity checks that the right release was fetched."""
    if ds["id"] == "cicids2017_improved":
        for p in folder.glob("*.csv"):
            header = p.open(errors="replace").readline()
            if "Attempted Category" not in header:
                print(f"  ! {p.name} has no 'Attempted Category' column: this may be the ORIGINAL "
                      f"CICIDS2017 release, not the corrected one")


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=REPO / "config" / "datasets.yaml")
    ap.add_argument("--mydrive", default="/content/drive/MyDrive")
    ap.add_argument("--drive-root", default=None, help="default: <mydrive>/PhD-AI-Cyber")
    ap.add_argument("--gated", default=None, help="default: <drive-root>/config/gated_urls.yaml")
    ap.add_argument("--tmp", default="/content/dl_tmp")
    ap.add_argument("--budget-gb", type=float, default=190, help="cap on total raw data in Drive")
    ap.add_argument("--only", nargs="*", help="dataset ids to process (default: all enabled)")
    ap.add_argument("--plan", action="store_true", help="list files and sizes only")
    ap.add_argument("--verify", action="store_true", help="re-hash files against their manifests")
    a = ap.parse_args(argv)

    drive_root = Path(a.drive_root or Path(a.mydrive) / "PhD-AI-Cyber")
    gated_path = Path(a.gated or drive_root / "config" / "gated_urls.yaml")
    gated = (yaml.safe_load(gated_path.read_text()) or {}) if gated_path.exists() else {}
    SITE_HEADERS.update(gated.get("site_headers") or {})
    cfg = yaml.safe_load(Path(a.config).read_text())
    f = Fetcher(drive_root, a.mydrive, a.tmp, gated, a.budget_gb)

    chosen = [d for d in cfg["datasets"] if (d["id"] in a.only if a.only else d.get("enabled", True))]
    if a.only and len(chosen) != len(a.only):
        known = {d["id"] for d in cfg["datasets"]}
        sys.exit(f"unknown dataset id(s): {sorted(set(a.only) - known)}")

    results = {}
    for ds in chosen:
        print(f"\n== {ds['id']}: {ds['name']}")
        if a.plan:
            f.plan(ds)
        elif a.verify:
            results[ds["id"]] = f.verify(ds)
        else:
            results[ds["id"]] = f.run(ds)

    if results:
        print("\nSummary")
        for k, ok in results.items():
            print(f"  {'done' if ok else 'TODO'}  {k}")
        print(f"Raw data in Drive: {human(dir_bytes(f.raw))} of {human(f.budget)} budget")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
