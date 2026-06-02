"""Download the real 10x PBMC3k multiome inputs used by Minerva benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("validation/real_multiome_v036")
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class DataFile:
    filename: str
    url: str
    size_bytes: int
    sha256: str


PBMC3K_FILES = (
    DataFile(
        filename="pbmc_3k_filtered_feature_bc_matrix.h5",
        url=(
            "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
            "pbmc_unsorted_3k/"
            "pbmc_unsorted_3k_filtered_feature_bc_matrix.h5"
        ),
        size_bytes=31_842_399,
        sha256="0a2e2d0b14bfe263318b56ca3f37b94ba9ae32247f0f06211bd5e616ac211825",
    ),
    DataFile(
        filename="pbmc_3k_atac_fragments.tsv.gz",
        url=(
            "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
            "pbmc_unsorted_3k/"
            "pbmc_unsorted_3k_atac_fragments.tsv.gz"
        ),
        size_bytes=537_308_082,
        sha256="0189df1de4894f10f48ffb905da4966cefc2466dc5fdd76ee2905244dd5a8dd7",
    ),
    DataFile(
        filename="pbmc_3k_atac_peaks.bed",
        url=(
            "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
            "pbmc_unsorted_3k/"
            "pbmc_unsorted_3k_atac_peaks.bed"
        ),
        size_bytes=1_939_240,
        sha256="4ecd8575ea0f7493b8177d1dafe80eb38f3b136d5eb8a8afe7dd7f4a60a832d3",
    ),
)


def file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if not path.is_file():
        return {"path": str(path), "exists": True, "type": "not_file"}
    return {
        "path": str(path),
        "exists": True,
        "type": "file",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, spec: DataFile) -> bool:
    state = file_state(path)
    return (
        state.get("type") == "file"
        and state.get("size_bytes") == spec.size_bytes
        and state.get("sha256") == spec.sha256
    )


def download_file(spec: DataFile, dest: Path, *, force: bool, timeout: float) -> dict[str, Any]:
    if dest.exists() and is_valid(dest, spec) and not force:
        return {"filename": spec.filename, "status": "present", "path": str(dest)}
    if dest.exists() and not force:
        state = file_state(dest)
        raise RuntimeError(
            f"{dest} exists but does not match expected PBMC3k hash: {state}"
        )

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    bytes_written = 0
    with urllib.request.urlopen(spec.url, timeout=timeout) as response:
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    if bytes_written != spec.size_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.filename} downloaded {bytes_written} bytes, "
            f"expected {spec.size_bytes}"
        )
    digest = sha256_file(tmp)
    if digest != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.filename} sha256 {digest}, expected {spec.sha256}"
        )
    tmp.replace(dest)
    return {
        "filename": spec.filename,
        "status": "downloaded",
        "path": str(dest),
        "size_bytes": bytes_written,
        "sha256": digest,
    }


def prepare_data(data_dir: Path, *, force: bool = False, timeout: float = 120.0) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [
        download_file(spec, data_dir / spec.filename, force=force, timeout=timeout)
        for spec in PBMC3K_FILES
    ]
    return {
        "dataset": "10x pbmc_unsorted_3k multiome",
        "source": "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_3k",
        "data_dir": str(data_dir),
        "files": files,
        "manifest": [asdict(spec) for spec in PBMC3K_FILES],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify real PBMC3k multiome inputs for HPC benchmarks."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        record = prepare_data(args.data_dir, force=args.force, timeout=args.timeout)
    except Exception as exc:
        print(f"prepare_real_pbmc3k_data failed: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
