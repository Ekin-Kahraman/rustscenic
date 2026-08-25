#!/usr/bin/env python3
"""Validate RustScenic release metadata and built artefact versions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
CONCEPT_DOI = "10.5281/zenodo.20246040"


class ReleaseValidationError(ValueError):
    """Raised when release metadata or artefacts disagree."""


def _toml_string(path: Path, section: str, key: str) -> str:
    """Read one quoted TOML string without adding a Python 3.10 dependency."""
    active_section = ""
    section_pattern = re.compile(r"^\[([^]]+)]$")
    value_pattern = re.compile(rf'^{re.escape(key)}\s*=\s*"([^"]+)"')

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        section_match = section_pattern.match(line)
        if section_match:
            active_section = section_match.group(1)
            continue
        if active_section == section:
            value_match = value_pattern.match(line)
            if value_match:
                return value_match.group(1)
    raise ReleaseValidationError(f"missing [{section}] {key} in {path}")


def _cff_string(path: Path, key: str) -> str:
    pattern = re.compile(rf'^{re.escape(key)}:\s*["\']?([^"\']+?)["\']?\s*$')
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    raise ReleaseValidationError(f"missing {key} in {path}")


def _require_equal(label: str, values: dict[str, str]) -> str:
    distinct = set(values.values())
    if len(distinct) != 1:
        details = ", ".join(f"{name}={value}" for name, value in values.items())
        raise ReleaseValidationError(f"{label} mismatch: {details}")
    return next(iter(distinct))


def validate_repository(root: Path = ROOT, tag: str | None = None) -> str:
    """Return the release version after validating repository state."""
    pyproject = root / "pyproject.toml"
    cargo = root / "Cargo.toml"
    citation = root / "CITATION.cff"
    zenodo_path = root / ".zenodo.json"
    zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))

    version = _require_equal(
        "release version",
        {
            "pyproject": _toml_string(pyproject, "project", "version"),
            "cargo-workspace": _toml_string(cargo, "workspace.package", "version"),
            "citation": _cff_string(citation, "version"),
            "zenodo": str(zenodo.get("version", "")),
        },
    )
    if not SEMVER.fullmatch(version):
        raise ReleaseValidationError(f"release version is not stable semver: {version}")

    expected_tag = f"v{version}"
    if tag is not None and tag != expected_tag:
        raise ReleaseValidationError(
            f"tag/version mismatch: tag={tag}, metadata={expected_tag}"
        )

    licence = _require_equal(
        "release licence",
        {
            "pyproject": _toml_string(pyproject, "project", "license"),
            "cargo-workspace": _toml_string(cargo, "workspace.package", "license"),
            "citation": _cff_string(citation, "license"),
            "zenodo": str(zenodo.get("license", "")),
        },
    )
    if licence != "Apache-2.0":
        raise ReleaseValidationError(f"unexpected release licence: {licence}")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(
        rf"^## {re.escape(version)} - [0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$",
        changelog,
        re.MULTILINE,
    ):
        raise ReleaseValidationError(f"missing dated {version} changelog section")

    required_release_text = {
        "README.md": f"Current release: `v{version}`",
        "site_docs/index.md": f"Current release `v{version}`",
        "site_docs/assets/rustscenic-evidence.svg": f">v{version}<",
    }
    for relative_path, expected in required_release_text.items():
        text = (root / relative_path).read_text(encoding="utf-8")
        if expected not in text:
            raise ReleaseValidationError(
                f"{relative_path} does not identify v{version} as the release"
            )

    citation_text = citation.read_text(encoding="utf-8")
    readme_text = (root / "README.md").read_text(encoding="utf-8")
    if CONCEPT_DOI not in citation_text or CONCEPT_DOI not in readme_text:
        raise ReleaseValidationError(
            f"README.md and CITATION.cff must use concept DOI {CONCEPT_DOI}"
        )

    lock_text = (root / "Cargo.lock").read_text(encoding="utf-8")
    workspace_crates = re.findall(
        r'name = "(rustscenic(?:-[^"]+)?)"\nversion = "([^"]+)"', lock_text
    )
    if not workspace_crates:
        raise ReleaseValidationError("Cargo.lock contains no RustScenic crates")
    wrong_crates = [
        f"{name}={crate_version}"
        for name, crate_version in workspace_crates
        if crate_version != version
    ]
    if wrong_crates:
        raise ReleaseValidationError(
            "Cargo.lock release version mismatch: " + ", ".join(wrong_crates)
        )

    return version


def validate_artifacts(dist: Path, version: str) -> None:
    """Validate that every RustScenic wheel and sdist carries the release version."""
    wheels = sorted(dist.rglob("rustscenic-*.whl"))
    sdists = sorted(dist.rglob("rustscenic-*.tar.gz"))
    if not wheels:
        raise ReleaseValidationError(f"no RustScenic wheels found under {dist}")
    if not sdists:
        raise ReleaseValidationError(f"no RustScenic sdist found under {dist}")

    wrong = [
        path.name
        for path in wheels
        if not path.name.startswith(f"rustscenic-{version}-")
    ]
    wrong.extend(
        path.name
        for path in sdists
        if path.name != f"rustscenic-{version}.tar.gz"
    )
    if wrong:
        raise ReleaseValidationError(
            f"artefact/version mismatch for {version}: {', '.join(wrong)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", help="Release tag, for example v0.5.0")
    parser.add_argument("--dist", type=Path, help="Directory of built artefacts")
    args = parser.parse_args(argv)

    try:
        version = validate_repository(args.root.resolve(), args.tag)
        if args.dist is not None:
            validate_artifacts(args.dist.resolve(), version)
    except (OSError, json.JSONDecodeError, ReleaseValidationError) as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1

    suffix = f" and artefacts under {args.dist}" if args.dist else ""
    print(f"release metadata v{version}{suffix}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
