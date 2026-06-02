"""Repo cleanliness helpers for publication-grade benchmark provenance."""
from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Any


SOURCE_DIR_PREFIXES = (
    ".github/",
    "crates/",
    "python/",
    "src/",
    "tests/",
    "validation/",
)

SOURCE_SUFFIXES = {
    ".lock",
    ".lsf",
    ".md",
    ".py",
    ".pyi",
    ".rs",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}

ROOT_SOURCE_FILES = {
    ".gitignore",
    "Cargo.lock",
    "Cargo.toml",
    "CHANGELOG.md",
    "LICENSE",
    "LICENSE.md",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
}


def is_source_path(path: str) -> bool:
    """Return whether a path can affect source, tests, or benchmark logic."""
    clean = path.strip()
    if not clean:
        return False
    if clean in ROOT_SOURCE_FILES:
        return True

    suffix = PurePosixPath(clean).suffix
    if suffix not in SOURCE_SUFFIXES:
        return False
    return clean.startswith(SOURCE_DIR_PREFIXES)


def source_paths(paths: list[str]) -> list[str]:
    """Filter paths down to source-like files."""
    return [path for path in paths if is_source_path(path)]


def git_status_paths(lines: list[str]) -> list[str]:
    """Extract paths from ``git status --short`` output lines.

    For renames, include both old and new paths so source moving in or out of a
    non-source location is still detected.
    """
    paths: list[str] = []
    for line in lines:
        if len(line) < 4:
            continue
        payload = line[3:].strip()
        if not payload:
            continue
        if " -> " in payload:
            old, new = payload.split(" -> ", 1)
            paths.extend([old, new])
        else:
            paths.append(payload)
    return paths


def is_untracked_source_path(path: str) -> bool:
    """Return whether an untracked path can affect source or benchmark logic."""
    return is_source_path(path)


def untracked_source_paths(paths: list[str]) -> list[str]:
    """Filter git untracked paths down to source-like files."""
    return source_paths(paths)


def repo_state_from_git_outputs(
    *,
    commit: str | None,
    tracked_status: str,
    untracked_status: str,
    tracked_diff: str,
) -> dict[str, Any]:
    """Build benchmark provenance from git command output strings."""
    tracked_lines = tracked_status.splitlines()
    tracked_source = source_paths(git_status_paths(tracked_lines))
    untracked = [
        line[3:]
        for line in untracked_status.splitlines()
        if line.startswith("?? ")
    ]
    untracked_source = source_paths(untracked)
    tracked_dirty = bool(tracked_status.strip())
    return {
        "commit": commit,
        "tracked_dirty": tracked_dirty,
        "tracked_source_dirty": bool(tracked_source),
        "source_dirty": bool(tracked_source) or bool(untracked_source),
        "tracked_status_short": tracked_lines,
        "tracked_diff_sha256": (
            hashlib.sha256(tracked_diff.encode()).hexdigest()
            if tracked_dirty
            else None
        ),
        "tracked_diff_bytes": len(tracked_diff.encode()),
        "tracked_source_count": len(tracked_source),
        "tracked_source_sample": tracked_source[:20],
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:20],
        "untracked_source_count": len(untracked_source),
        "untracked_source_sample": untracked_source[:20],
    }
