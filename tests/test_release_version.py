from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SCRIPT = ROOT / "scripts/check_release_version.py"
SPEC = importlib.util.spec_from_file_location("check_release_version", RELEASE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RELEASE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE_MODULE)

ReleaseValidationError = RELEASE_MODULE.ReleaseValidationError
validate_artifacts = RELEASE_MODULE.validate_artifacts
validate_repository = RELEASE_MODULE.validate_repository

RELEASE_FILES = [
    ".zenodo.json",
    "Cargo.lock",
    "Cargo.toml",
    "CHANGELOG.md",
    "CITATION.cff",
    "README.md",
    "pyproject.toml",
    "site_docs/assets/rustscenic-evidence.svg",
    "site_docs/index.md",
]


def _copy_release_state(destination: Path) -> Path:
    for relative_path in RELEASE_FILES:
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, target)
    return destination


def test_release_metadata_matches_v050():
    assert validate_repository(ROOT, tag="v0.5.0") == "0.5.0"


def test_every_release_upload_fails_when_its_build_output_is_missing():
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert workflow.count("if-no-files-found: error") == 4


def test_release_tag_must_match_metadata():
    with pytest.raises(ReleaseValidationError, match="tag/version mismatch"):
        validate_repository(ROOT, tag="v0.4.7")


def test_release_metadata_mismatch_is_rejected(tmp_path):
    root = _copy_release_state(tmp_path)
    zenodo = root / ".zenodo.json"
    zenodo.write_text(
        zenodo.read_text(encoding="utf-8").replace('"version": "0.5.0"', '"version": "0.4.7"'),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseValidationError, match="release version mismatch"):
        validate_repository(root)


def test_release_artifact_filenames_must_match_version(tmp_path):
    (tmp_path / "rustscenic-0.5.0-cp310-abi3-manylinux.whl").touch()
    (tmp_path / "rustscenic-0.5.0.tar.gz").touch()
    validate_artifacts(tmp_path, "0.5.0")

    (tmp_path / "rustscenic-0.4.7-cp310-abi3-win_amd64.whl").touch()
    with pytest.raises(ReleaseValidationError, match="artefact/version mismatch"):
        validate_artifacts(tmp_path, "0.5.0")
