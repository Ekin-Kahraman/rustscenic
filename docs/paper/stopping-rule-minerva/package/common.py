"""Small shared helpers; bootstrap can run before installing dependencies."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def verify_package():
    records = []
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_relative_to(ROOT) or ".." in Path(relative).parts:
            raise ValueError("Unsafe package manifest path")
        if digest(path) != expected:
            raise ValueError(f"Package checksum mismatch: {relative}; extract a fresh copy")
        records.append([relative, expected])
    return fingerprint(records)
