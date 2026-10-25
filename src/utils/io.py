"""File helpers that record the SHA-256 of everything read or written."""
import hashlib
import io
import json
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class HashingReader(io.RawIOBase):
    """Wrap a binary file object; hashes the bytes as they are read, so checksumming costs no extra pass."""

    def __init__(self, raw):
        self.raw, self.sha, self.bytes = raw, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.sha.update(data)
        self.bytes += n
        return n

    def finish(self):
        """Read whatever the parser left unread, then return (sha256, bytes)."""
        for block in iter(lambda: self.raw.read(CHUNK), b""):
            self.sha.update(block)
            self.bytes += len(block)
        return self.sha.hexdigest(), self.bytes

    def close(self):
        self.raw.close()
        super().close()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=_jsonable) + "\n")
    return path


def _jsonable(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_parquet(df, path):
    """Write a parquet file and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return sha256_file(path)
