"""Portable path labels for benchmark artefacts.

Benchmark JSON is designed to move between a workstation, Minerva, and an
archive. Absolute paths disclose personal mount layouts and make byte-for-byte
comparison needlessly host-specific, while hashes and repo-relative labels
still provide the provenance needed for validation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_EMBEDDED_WINDOWS_ABSOLUTE = re.compile(
    r"(?:^|[\s'\"=:(])[A-Za-z]:[\\/][^\s'\",;)\]}]+"
)
_EMBEDDED_POSIX_ABSOLUTE = re.compile(
    r"(?:^|[\s'\"=:(])/(?!/)[^\s'\",;)\]}]+"
)
_EMBEDDED_HOME_ABSOLUTE = re.compile(
    r"(?:^|[\s'\"=:(])~/[^\s'\",;)\]}]+"
)
_FILE_URI_ABSOLUTE = re.compile(r"file:///(?:[^\s'\",;)\]}]+)")


def portable_path(value: str | Path | None, repo_root: Path) -> str | None:
    """Return a repo-relative or external basename label, never an absolute path."""
    if value is None:
        return None
    raw = str(value)
    if not raw:
        return raw
    if _WINDOWS_ABSOLUTE.match(raw):
        name = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        return f"external:{name or 'root'}"
    path = Path(raw).expanduser()
    is_absolute = path.is_absolute()
    if not is_absolute:
        return raw
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        name = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        return f"external:{name or 'root'}"


def portable_argv(values: Iterable[str], repo_root: Path) -> list[str]:
    """Sanitise path-valued command arguments while preserving flags and values."""
    return [portable_path(value, repo_root) or "" for value in values]


def absolute_path_strings(value, *, prefix: str = "$") -> list[str]:
    """Find host-specific absolute path strings in a JSON-compatible object."""
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(absolute_path_strings(child, prefix=f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(absolute_path_strings(child, prefix=f"{prefix}[{index}]"))
    elif isinstance(value, str):
        if (
            _EMBEDDED_POSIX_ABSOLUTE.search(value)
            or _EMBEDDED_HOME_ABSOLUTE.search(value)
            or _EMBEDDED_WINDOWS_ABSOLUTE.search(value)
            or _FILE_URI_ABSOLUTE.search(value)
        ):
            failures.append(prefix)
    return failures
