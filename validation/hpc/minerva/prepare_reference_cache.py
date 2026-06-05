"""Warm RustScenic reference-data caches before timed Minerva benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rustscenic.data as data


def file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        return {"path": str(path), "exists": True, "type": "dir"}
    return {
        "path": str(path),
        "exists": True,
        "type": "file",
        "size_bytes": path.stat().st_size,
    }


def _reference_record(
    *,
    kind: str,
    species: str,
    before: dict[str, Any],
    after: dict[str, Any],
    elapsed_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = "present" if before.get("exists") else "cached"
    record = {
        "kind": kind,
        "species": species,
        "status": status,
        "before": before,
        "after": after,
        "elapsed_s": round(elapsed_s, 6),
    }
    if extra:
        record.update(extra)
    return record


def prepare_references(
    *,
    motif_species: str = "human",
    gene_species: str = "hs",
    motif_cache_dir: Path | None = None,
    gene_cache_dir: Path | None = None,
    skip_motif_rankings: bool = False,
    skip_gene_coords: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    references: dict[str, Any] = {}

    if not skip_motif_rankings:
        motif_path = data._motif_rankings_cache_path(
            species=motif_species,
            cache_dir=motif_cache_dir,
        )
        before = file_state(motif_path)
        start = time.perf_counter()
        cached_path = data._ensure_motif_rankings_cached(
            species=motif_species,
            cache_dir=motif_cache_dir,
            verbose=verbose,
        )
        elapsed_s = time.perf_counter() - start
        references["motif_rankings"] = _reference_record(
            kind="motif_rankings",
            species=motif_species,
            before=before,
            after=file_state(cached_path),
            elapsed_s=elapsed_s,
        )

    if not skip_gene_coords:
        gene_paths = data._gene_coords_cache_paths(
            species=gene_species,
            cache_dir=gene_cache_dir,
        )
        parquet_path = Path(gene_paths["parquet_path"])
        before = file_state(parquet_path)
        start = time.perf_counter()
        coords = data.download_gene_coords(
            species=gene_species,
            cache_dir=gene_cache_dir,
            verbose=verbose,
        )
        elapsed_s = time.perf_counter() - start
        references["gene_coords"] = _reference_record(
            kind="gene_coords",
            species=gene_species,
            before=before,
            after=file_state(parquet_path),
            elapsed_s=elapsed_s,
            extra={"rows": int(len(coords))},
        )

    return {"ok": True, "references": references}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache reference data before timed Minerva benchmarks."
    )
    parser.add_argument("--motif-species", default="human")
    parser.add_argument("--gene-species", default="hs")
    parser.add_argument("--motif-cache-dir", type=Path, default=None)
    parser.add_argument("--gene-cache-dir", type=Path, default=None)
    parser.add_argument("--skip-motif-rankings", action="store_true")
    parser.add_argument("--skip-gene-coords", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        record = prepare_references(
            motif_species=args.motif_species,
            gene_species=args.gene_species,
            motif_cache_dir=args.motif_cache_dir,
            gene_cache_dir=args.gene_cache_dir,
            skip_motif_rankings=args.skip_motif_rankings,
            skip_gene_coords=args.skip_gene_coords,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"prepare_reference_cache failed: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(record, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
