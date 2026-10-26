#!/usr/bin/env python3
"""Validate RustScenic release metadata and built artefact versions."""

from __future__ import annotations

import argparse
from datetime import date
from email.parser import BytesParser
from email.policy import default as email_policy
import json
import re
import sys
import tarfile
import zipfile
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


def _single_version(text: str, pattern: str, label: str) -> str:
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) != 1:
        raise ReleaseValidationError(
            f"expected exactly one {label}; found {len(matches)}"
        )
    return matches[0]


def _validate_release_state(root: Path, version: str, candidate: bool) -> None:
    """Cross-check visible release labels with an explicit publication state."""
    version_pattern = r"([0-9]+\.[0-9]+\.[0-9]+)"
    readme = (root / "README.md").read_text(encoding="utf-8")
    index = (root / "site_docs/index.md").read_text(encoding="utf-8")
    svg = (root / "site_docs/assets/rustscenic-evidence.svg").read_text(
        encoding="utf-8"
    )
    published = _require_equal(
        "published release labels",
        {
            "README": _single_version(
                readme,
                rf"Current release: `v{version_pattern}`",
                "README release label",
            ),
            "README installation": _single_version(
                readme,
                rf"Expected version: `{version_pattern}`",
                "README installation version",
            ),
            "site": _single_version(
                index, rf"Current release `v{version_pattern}`", "site release label"
            ),
            "evidence": _single_version(
                svg, rf">v{version_pattern}<", "evidence release label"
            ),
        },
    )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    state_path = root / "release-candidate.json"
    candidate_pattern = rf"^Source candidate: `v{version_pattern}` \(unpublished\)\.$"
    candidate_labels = {
        "README": re.findall(candidate_pattern, readme, re.MULTILINE),
        "site": re.findall(candidate_pattern, index, re.MULTILINE),
    }
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    zenodo_description = str(
        json.loads((root / ".zenodo.json").read_text(encoding="utf-8")).get(
            "description", ""
        )
    )
    notes = (root / f"docs/releases/v{version}.md").read_text(encoding="utf-8")
    title = notes.splitlines()[0] if notes.splitlines() else ""
    if not re.match(rf"^# RustScenic v{re.escape(version)}(?:\s|:|$)", title):
        raise ReleaseValidationError("release notes title does not match metadata")

    if candidate:
        if not state_path.is_file():
            raise ReleaseValidationError(
                "candidate mode requires release-candidate.json"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or set(state) != {
            "candidate_version",
            "published_version",
            "prepared_on",
        }:
            raise ReleaseValidationError("invalid release-candidate.json fields")
        _require_equal(
            "source candidate version",
            {
                "metadata": version,
                "candidate state": str(state["candidate_version"]),
                **{
                    name: _single_version(
                        text, candidate_pattern, f"{name} source candidate label"
                    )
                    for name, text in [("README", readme), ("site", index)]
                },
            },
        )
        _require_equal(
            "published candidate baseline",
            {
                "labels": published,
                "candidate state": str(state["published_version"]),
            },
        )
        if tuple(map(int, version.split("."))) <= tuple(map(int, published.split("."))):
            raise ReleaseValidationError(
                "source candidate must be newer than the published release"
            )
        try:
            prepared_on = date.fromisoformat(str(state["prepared_on"])).isoformat()
        except ValueError as exc:
            raise ReleaseValidationError("invalid candidate preparation date") from exc
        heading = f"## {version} - Candidate prepared {prepared_on} (unpublished)"
        if changelog.splitlines().count(heading) != 1:
            raise ReleaseValidationError(
                "missing or duplicate dated candidate changelog section"
            )
        if re.search(
            rf"^## {re.escape(version)} - [0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$",
            changelog,
            re.MULTILINE,
        ):
            raise ReleaseValidationError(
                "candidate also has a published changelog section"
            )
        if not re.search(
            rf"^## {re.escape(published)} - [0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$",
            changelog,
            re.MULTILINE,
        ):
            raise ReleaseValidationError("missing published baseline changelog section")
        if re.search(r"^date-released:", citation, re.MULTILINE):
            raise ReleaseValidationError(
                "unpublished candidate must not claim a citation release date"
            )
        candidate_marker = f"Unpublished source candidate v{version}."
        if (
            candidate_marker not in _cff_string(root / "CITATION.cff", "message")
            or candidate_marker not in zenodo_description
        ):
            raise ReleaseValidationError(
                "citation and Zenodo must identify the unpublished candidate"
            )
        if f"Published release: v{published}." not in zenodo_description:
            raise ReleaseValidationError("Zenodo published baseline differs")
        if not title.startswith(f"# RustScenic v{version} candidate:"):
            raise ReleaseValidationError(
                "candidate release notes must identify the unpublished candidate"
            )
        if "**Draft for review. This candidate is not published." not in notes:
            raise ReleaseValidationError(
                "candidate release notes need an unpublished banner"
            )
        notes_published = _single_version(
            notes,
            rf"^Published release: `v{version_pattern}`\.$",
            "release notes published baseline",
        )
        if notes_published != published:
            raise ReleaseValidationError("release notes published baseline differs")
    else:
        if state_path.exists() or any(candidate_labels.values()):
            raise ReleaseValidationError(
                "candidate state cannot pass publication validation"
            )
        if published != version:
            raise ReleaseValidationError(
                f"published release labels identify v{published}, not v{version}"
            )
        release_dates = re.findall(
            rf"^## {re.escape(version)} - ([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})$",
            changelog,
            re.MULTILINE,
        )
        if len(release_dates) != 1:
            raise ReleaseValidationError(
                f"missing or duplicate dated {version} changelog section"
            )
        try:
            released_on = date.fromisoformat(release_dates[0]).isoformat()
        except ValueError as exc:
            raise ReleaseValidationError("invalid release date") from exc
        if _cff_string(root / "CITATION.cff", "date-released") != released_on:
            raise ReleaseValidationError("citation and changelog release dates differ")
        if (
            " candidate:" in title
            or "**Draft for review." in notes
            or re.search(r"^Published release:", notes, re.MULTILINE)
        ):
            raise ReleaseValidationError(
                "candidate release notes cannot pass publication validation"
            )
        if (
            "Unpublished source candidate" in citation
            or "Unpublished source candidate" in zenodo_description
        ):
            raise ReleaseValidationError(
                "candidate citation metadata cannot pass publication validation"
            )


def validate_repository(
    root: Path = ROOT, tag: str | None = None, *, candidate: bool = False
) -> str:
    """Return the release version after validating repository state."""
    if candidate and tag is not None:
        raise ReleaseValidationError(
            "candidate mode and tag validation are incompatible"
        )
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

    _validate_release_state(root, version, candidate)

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
        path.name for path in sdists if path.name != f"rustscenic-{version}.tar.gz"
    )
    if wrong:
        raise ReleaseValidationError(
            f"artefact/version mismatch for {version}: {', '.join(wrong)}"
        )

    for path in wheels + sdists:
        try:
            if path.suffix == ".whl":
                with zipfile.ZipFile(path) as archive:
                    entries = [
                        info
                        for info in archive.infolist()
                        if re.fullmatch(r"[^/]+\.dist-info/METADATA", info.filename)
                    ]
                    expected = f"rustscenic-{version}.dist-info/METADATA"
                    if len(entries) != 1 or entries[0].filename != expected:
                        raise ReleaseValidationError(
                            f"{path.name}: missing or ambiguous wheel METADATA"
                        )
                    if entries[0].file_size > 1024 * 1024:
                        raise ReleaseValidationError(
                            f"{path.name}: oversized wheel METADATA"
                        )
                    metadata = archive.read(entries[0])
            else:
                metadata_entries = []
                expected = f"rustscenic-{version}/PKG-INFO"
                with tarfile.open(path, "r|gz") as archive:
                    for entry in archive:
                        if entry.name == expected:
                            if not entry.isfile() or entry.size > 1024 * 1024:
                                raise ReleaseValidationError(
                                    f"{path.name}: invalid sdist PKG-INFO"
                                )
                            stream = archive.extractfile(entry)
                            if stream is None:
                                raise ReleaseValidationError(
                                    f"{path.name}: unreadable sdist PKG-INFO"
                                )
                            with stream:
                                metadata_entries.append(stream.read())
                if len(metadata_entries) != 1:
                    raise ReleaseValidationError(
                        f"{path.name}: missing or ambiguous sdist PKG-INFO"
                    )
                metadata = metadata_entries[0]
            headers = BytesParser(policy=email_policy).parsebytes(
                metadata, headersonly=True
            )
            names = headers.get_all("Name", [])
            versions = headers.get_all("Version", [])
            if headers.defects or len(names) != 1 or len(versions) != 1:
                raise ReleaseValidationError(f"{path.name}: malformed package metadata")
            name = re.sub(r"[-_.]+", "-", str(names[0]).strip()).lower()
            if name != "rustscenic" or str(versions[0]).strip() != version:
                raise ReleaseValidationError(
                    f"{path.name}: internal Name/Version mismatch: {names[0]}/{versions[0]}"
                )
        except (
            OSError,
            EOFError,
            tarfile.TarError,
            zipfile.BadZipFile,
            RuntimeError,
        ) as exc:
            raise ReleaseValidationError(
                f"{path.name}: cannot read package metadata: {exc}"
            ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tag", help="Release tag, for example v0.5.1")
    mode.add_argument(
        "--candidate",
        action="store_true",
        help="Validate explicit unpublished source-candidate state",
    )
    parser.add_argument("--dist", type=Path, help="Directory of built artefacts")
    args = parser.parse_args(argv)

    try:
        version = validate_repository(
            args.root.resolve(), args.tag, candidate=args.candidate
        )
        if args.dist is not None:
            validate_artifacts(args.dist.resolve(), version)
    except (OSError, json.JSONDecodeError, ReleaseValidationError) as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1

    suffix = f" and artefacts under {args.dist}" if args.dist else ""
    state = "source candidate" if args.candidate else "release"
    print(f"{state} metadata v{version}{suffix}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
