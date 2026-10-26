from __future__ import annotations

import importlib.util
import ast
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SCRIPT = ROOT / "scripts/check_release_version.py"
SPEC = importlib.util.spec_from_file_location("check_release_version", RELEASE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RELEASE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE_MODULE)
SMOKE_SPEC = importlib.util.spec_from_file_location(
    "smoke_release_wheel", ROOT / "scripts/smoke_release_wheel.py"
)
assert SMOKE_SPEC is not None and SMOKE_SPEC.loader is not None
SMOKE_MODULE = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(SMOKE_MODULE)
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
    "release-candidate.json",
    "site_docs/assets/rustscenic-evidence.svg",
    "site_docs/index.md",
    "docs/releases/v0.5.1.md",
]


def _copy_release_state(destination: Path) -> Path:
    for relative_path in RELEASE_FILES:
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, target)
    return destination


def _replace(root: Path, path: str, old: str, new: str) -> None:
    file = root / path
    text = file.read_text(encoding="utf-8")
    assert old in text
    file.write_text(text.replace(old, new), encoding="utf-8")


def _promote_fixture(root: Path) -> Path:
    """Model the explicit later publication transition in a disposable fixture."""
    state = json.loads((root / "release-candidate.json").read_text())
    (root / "release-candidate.json").unlink()
    for name in ["README.md", "site_docs/index.md"]:
        _replace(root, name, "Source candidate: `v0.5.1` (unpublished).\n", "")
    _replace(
        root, "README.md", "Current release: `v0.5.0`", "Current release: `v0.5.1`"
    )
    _replace(
        root, "README.md", "Expected version: `0.5.0`", "Expected version: `0.5.1`"
    )
    _replace(
        root,
        "site_docs/index.md",
        "Current release `v0.5.0`",
        "Current release `v0.5.1`",
    )
    _replace(root, "site_docs/assets/rustscenic-evidence.svg", ">v0.5.0<", ">v0.5.1<")
    _replace(
        root,
        "CHANGELOG.md",
        f"## 0.5.1 - Candidate prepared {state['prepared_on']} (unpublished)",
        "## 0.5.1 - 2026-10-26",
    )
    _replace(root, "CITATION.cff", "Unpublished source candidate v0.5.1. ", "")
    with (root / "CITATION.cff").open("a") as handle:
        handle.write('date-released: "2026-10-26"\n')
    _replace(
        root,
        ".zenodo.json",
        "<p>Unpublished source candidate v0.5.1. Published release: v0.5.0.</p>",
        "",
    )
    _replace(root, "docs/releases/v0.5.1.md", "v0.5.1 candidate:", "v0.5.1:")
    _replace(root, "docs/releases/v0.5.1.md", "Published release: `v0.5.0`.\n", "")
    _replace(
        root,
        "docs/releases/v0.5.1.md",
        "**Draft for review. This candidate is not published. The current PyPI and\nGitHub release is v0.5.0.**",
        "Published 2026-10-26.",
    )
    return root


def test_candidate_metadata_and_current_published_labels_are_consistent():
    assert validate_repository(ROOT, candidate=True) == "0.5.1"
    assert "Current release: `v0.5.0`" in (ROOT / "README.md").read_text()
    assert "Current release `v0.5.0`" in (ROOT / "site_docs/index.md").read_text()


@pytest.mark.parametrize("tag", [None, "v0.5.1"])
def test_candidate_cannot_pass_default_or_tag_publication_validation(tag):
    with pytest.raises(
        ReleaseValidationError, match="candidate state cannot pass publication"
    ):
        validate_repository(ROOT, tag=tag)


def test_explicitly_promoted_state_retains_strict_default_and_tag_validation(tmp_path):
    root = _promote_fixture(_copy_release_state(tmp_path))
    assert validate_repository(root) == "0.5.1"
    assert validate_repository(root, tag="v0.5.1") == "0.5.1"
    with pytest.raises(ReleaseValidationError, match="candidate mode requires"):
        validate_repository(root, candidate=True)
    with pytest.raises(ReleaseValidationError, match="tag/version mismatch"):
        validate_repository(root, tag="v0.5.0")


def test_candidate_and_tag_are_incompatible_in_api_and_cli():
    with pytest.raises(ReleaseValidationError, match="incompatible"):
        validate_repository(ROOT, tag="v0.5.1", candidate=True)
    with pytest.raises(SystemExit) as exc:
        RELEASE_MODULE.main(["--candidate", "--tag", "v0.5.1"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "path,old,new",
    [
        (".zenodo.json", '"version": "0.5.1"', '"version": "0.4.7"'),
        (
            "Cargo.lock",
            'name = "rustscenic-py"\nversion = "0.5.1"',
            'name = "rustscenic-py"\nversion = "0.5.0"',
        ),
        ("CITATION.cff", "license: Apache-2.0", "license: MIT"),
        ("CITATION.cff", "10.5281/zenodo.20246040", "10.5281/zenodo.1"),
        ("README.md", "10.5281/zenodo.20246040", "10.5281/zenodo.1"),
        ("README.md", "Current release: `v0.5.0`", "Current release: `v0.4.7`"),
        ("README.md", "Expected version: `0.5.0`", "Expected version: `0.4.7`"),
        (
            "site_docs/index.md",
            "Source candidate: `v0.5.1`",
            "Source candidate: `v0.5.2`",
        ),
        ("site_docs/assets/rustscenic-evidence.svg", ">v0.5.0<", ">v0.4.7<"),
        (
            "CHANGELOG.md",
            "## 0.5.1 - Candidate prepared",
            "## 0.5.2 - Candidate prepared",
        ),
        (
            "docs/releases/v0.5.1.md",
            "Published release: `v0.5.0`",
            "Published release: `v0.4.7`",
        ),
    ],
)
def test_candidate_mode_does_not_exempt_mismatches(tmp_path, path, old, new):
    root = _copy_release_state(tmp_path)
    _replace(root, path, old, new)
    with pytest.raises(ReleaseValidationError):
        validate_repository(root, candidate=True)


@pytest.mark.parametrize("published", ["0.5.1", "0.6.0"])
def test_candidate_must_be_newer_than_consistent_published_baseline(
    tmp_path, published
):
    root = _copy_release_state(tmp_path)
    for path, old in [
        ("README.md", "Current release: `v0.5.0`"),
        ("README.md", "Expected version: `0.5.0`"),
        ("site_docs/index.md", "Current release `v0.5.0`"),
        ("site_docs/assets/rustscenic-evidence.svg", ">v0.5.0<"),
    ]:
        _replace(root, path, old, old.replace("0.5.0", published))
    _replace(
        root,
        "release-candidate.json",
        '"published_version": "0.5.0"',
        f'"published_version": "{published}"',
    )
    with pytest.raises(ReleaseValidationError, match="must be newer"):
        validate_repository(root, candidate=True)


@pytest.mark.parametrize(
    "change",
    [
        "missing_state",
        "missing_label",
        "duplicate_label",
        "released_date",
        "bad_preparation_date",
        "dated_release",
    ],
)
def test_candidate_state_must_be_explicit_and_unpublished(tmp_path, change):
    root = _copy_release_state(tmp_path)
    if change == "missing_state":
        (root / "release-candidate.json").unlink()
    elif change == "missing_label":
        _replace(root, "README.md", "Source candidate: `v0.5.1` (unpublished).", "")
    elif change == "duplicate_label":
        with (root / "README.md").open("a") as handle:
            handle.write("\nSource candidate: `v0.5.1` (unpublished).\n")
    elif change == "released_date":
        with (root / "CITATION.cff").open("a") as handle:
            handle.write('date-released: "2026-10-26"\n')
    elif change == "bad_preparation_date":
        state = json.loads((root / "release-candidate.json").read_text())
        state["prepared_on"] = "2026-99-99"
        (root / "release-candidate.json").write_text(json.dumps(state))
    else:
        with (root / "CHANGELOG.md").open("a") as handle:
            handle.write("\n## 0.5.1 - 2026-10-26\n")
    with pytest.raises(ReleaseValidationError):
        validate_repository(root, candidate=True)


def test_publication_rejects_lingering_candidate_citation_or_notes(tmp_path):
    root = _promote_fixture(_copy_release_state(tmp_path))
    _replace(
        root,
        "CITATION.cff",
        'message: "',
        'message: "Unpublished source candidate v0.5.1. ',
    )
    with pytest.raises(ReleaseValidationError, match="candidate citation metadata"):
        validate_repository(root, tag="v0.5.1")


def test_publication_rejects_a_draft_release_notes_banner(tmp_path):
    root = _promote_fixture(_copy_release_state(tmp_path))
    _replace(
        root,
        "docs/releases/v0.5.1.md",
        "Published 2026-10-26.",
        "**Draft for review.**",
    )
    with pytest.raises(ReleaseValidationError, match="candidate release notes"):
        validate_repository(root, tag="v0.5.1")


def test_release_notes_keep_historical_benchmark_scope_and_candidate_limits():
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    historical = (ROOT / "docs/releases/v0.5.0.md").read_text()
    candidate = (ROOT / "docs/releases/v0.5.1.md").read_text()
    assert "body_path: docs/releases/${{ github.ref_name }}.md" in workflow
    assert "generate_release_notes: false" in workflow
    assert "71.49 GB" in historical and "21.4%" in historical
    assert "does not include the unfinished CELLxGENE" in historical
    assert "candidate is not published" in candidate
    assert "No new performance or biological accuracy result" in candidate


def test_every_release_upload_fails_when_its_build_output_is_missing():
    assert (ROOT / ".github/workflows/release.yml").read_text().count(
        "if-no-files-found: error"
    ) == 4


def _write_wheel(path, name="rustscenic", version="0.5.1", raw=None):
    metadata = (
        raw
        if raw is not None
        else f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rustscenic-0.5.1.dist-info/METADATA", metadata)


def _write_sdist(path, name="rustscenic", version="0.5.1", raw=None):
    metadata = (
        raw
        if raw is not None
        else f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n"
    )
    payload = metadata.encode()
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("rustscenic-0.5.1/PKG-INFO")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))


def _write_artifacts(directory):
    wheel = directory / "rustscenic-0.5.1-cp310-abi3-manylinux.whl"
    sdist = directory / "rustscenic-0.5.1.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)
    return wheel, sdist


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("field", ["name", "version"])
def test_correct_filename_cannot_hide_wrong_internal_metadata(tmp_path, kind, field):
    wheel, sdist = _write_artifacts(tmp_path)
    writer, path = (_write_wheel, wheel) if kind == "wheel" else (_write_sdist, sdist)
    writer(path, **{field: "other-project" if field == "name" else "0.5.0"})
    with pytest.raises(ReleaseValidationError, match="internal Name/Version mismatch"):
        validate_artifacts(tmp_path, "0.5.1")


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "damage", ["not_an_archive", "empty_archive", "duplicate_header"]
)
def test_malformed_archives_or_metadata_are_rejected(tmp_path, kind, damage):
    wheel, sdist = _write_artifacts(tmp_path)
    writer, path = (_write_wheel, wheel) if kind == "wheel" else (_write_sdist, sdist)
    if damage == "not_an_archive":
        path.write_bytes(b"not an archive")
    elif damage == "empty_archive":
        if kind == "wheel":
            with zipfile.ZipFile(path, "w"):
                pass
        else:
            with tarfile.open(path, "w:gz"):
                pass
    else:
        writer(path, raw="Name: rustscenic\nName: other\nVersion: 0.5.1\n\n")
    with pytest.raises(ReleaseValidationError):
        validate_artifacts(tmp_path, "0.5.1")


def test_sdist_metadata_is_read_without_following_links_or_extracting(tmp_path):
    wheel, sdist = _write_artifacts(tmp_path)
    with tarfile.open(sdist, "w:gz") as archive:
        entry = tarfile.TarInfo("rustscenic-0.5.1/PKG-INFO")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../../outside"
        archive.addfile(entry)
    with pytest.raises(ReleaseValidationError, match="invalid sdist PKG-INFO"):
        validate_artifacts(tmp_path, "0.5.1")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [wheel.name, sdist.name]
    )


def test_candidate_artifact_filename_validation_is_not_relaxed(tmp_path):
    _write_artifacts(tmp_path)
    validate_artifacts(tmp_path, "0.5.1")
    assert (
        RELEASE_MODULE.main(
            ["--root", str(ROOT), "--candidate", "--dist", str(tmp_path)]
        )
        == 0
    )
    _write_wheel(
        tmp_path / "rustscenic-0.5.0-cp310-abi3-win_amd64.whl", version="0.5.0"
    )
    with pytest.raises(ReleaseValidationError, match="artefact/version mismatch"):
        validate_artifacts(tmp_path, "0.5.1")
    assert (
        RELEASE_MODULE.main(
            ["--root", str(ROOT), "--candidate", "--dist", str(tmp_path)]
        )
        == 1
    )


def _workflow():
    """Read only the fixed workflow fields used by these policy tests.

    The repository's tests do not otherwise require a YAML parser. These
    extracts retain the actual expressions rather than reimplementing gates.
    """
    text = (ROOT / ".github/workflows/release.yml").read_text()
    jobs = {}
    for match in re.finditer(
        r"^  ([\w-]+):\n(.*?)(?=^  [\w-]+:|\Z)",
        text.split("jobs:\n", 1)[1],
        re.MULTILINE | re.DOTALL,
    ):
        name, body = match.groups()
        job = {}
        needs = re.search(r"^    needs: \[(.+)\]$", body, re.MULTILINE)
        if needs:
            job["needs"] = [name.strip() for name in needs.group(1).split(",")]
        condition = re.search(r"^    if: (.+)$", body, re.MULTILINE)
        if condition:
            expression = condition.group(1)
            if expression == "|":
                expression = re.search(
                    r"^    if: \|\n((?:      .+\n)+)", body, re.MULTILINE
                ).group(1)
            job["if"] = expression
        permission = re.search(
            r"^    permissions:\n      ([\w-]+): (\w+)", body, re.MULTILINE
        )
        if permission:
            job["permissions"] = dict([permission.groups()])
        target = re.search(r"^        target: \[(.+)\]$", body, re.MULTILINE)
        if target:
            job["strategy"] = {
                "matrix": {"target": [x.strip() for x in target.group(1).split(",")]}
            }
        env = dict(
            re.findall(
                r"^          (RELEASE_TAG|CANDIDATE_BUILD): (.+)$", body, re.MULTILINE
            )
        )
        if env:
            run = re.search(
                r"^        run: \|\n((?:          .+\n)+)", body, re.MULTILINE
            ).group(1)
            job["steps"] = [{"env": env, "run": run}]
        jobs[name] = job
    control_block = text.split("      candidate:\n", 1)[1].split("\npermissions:", 1)[0]
    control = dict(
        re.findall(
            r"^        (type|default|required): (\w+)$", control_block, re.MULTILINE
        )
    )
    top_permission = re.search(r"^permissions:\n  (\w+): (\w+)", text, re.MULTILINE)
    return {
        "on": {"workflow_dispatch": {"inputs": {"candidate": control}}},
        "jobs": jobs,
        "permissions": dict([top_permission.groups()]),
    }


def _evaluate_workflow_expression(
    expression, event, ref, candidate=False, trusted=True, failed_build=False
):
    """Evaluate the actual workflow's small boolean-expression subset in tests."""
    expression = expression.strip().removeprefix("${{").removesuffix("}}").strip()
    expression = re.sub(r"needs\.([\w-]+)\.result", r"results['\1']", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    expression = " ".join(expression.split())
    results = {name: "success" for name in _workflow()["jobs"]}
    if failed_build:
        results["sdist"] = "failure"
    return eval(
        expression,
        {"__builtins__": {}},
        {
            "github": SimpleNamespace(
                event_name=event, ref=ref, ref_name=ref.rsplit("/", 1)[-1]
            ),
            "inputs": SimpleNamespace(candidate=candidate),
            "vars": SimpleNamespace(
                PYPI_TRUSTED_PUBLISHING="true" if trusted else "false"
            ),
            "results": results,
            "always": lambda: True,
            "startsWith": lambda value, prefix: value.startswith(prefix),
        },
    )


@pytest.mark.parametrize(
    "event,ref,candidate,expected",
    [
        ("workflow_dispatch", "refs/heads/candidate", True, True),
        ("workflow_dispatch", "refs/heads/main", False, False),
        ("workflow_dispatch", "refs/tags/v0.5.1", True, False),
        ("push", "refs/tags/v0.5.1", True, False),
        ("pull_request", "refs/heads/candidate", True, False),
    ],
)
def test_workflow_candidate_mode_is_opt_in_and_non_tag_manual_only(
    event, ref, candidate, expected
):
    workflow = _workflow()
    control = workflow["on"]["workflow_dispatch"]["inputs"]["candidate"]
    assert control["type"] == "boolean" and control["default"] == "false"
    for name in ["version-integrity", "verify-artifacts"]:
        step = next(
            s
            for s in workflow["jobs"][name]["steps"]
            if "RELEASE_TAG" in s.get("env", {})
        )
        assert (
            bool(
                _evaluate_workflow_expression(
                    step["env"]["CANDIDATE_BUILD"], event, ref, candidate
                )
            )
            == expected
        )
        assert "--candidate" in step["run"]


@pytest.mark.parametrize(
    "event,ref,expected",
    [
        ("push", "refs/tags/v0.5.1", True),
        ("push", "refs/heads/main", False),
        ("workflow_dispatch", "refs/heads/main", False),
        ("workflow_dispatch", "refs/tags/v0.5.1", False),
        ("pull_request", "refs/tags/v0.5.1", False),
    ],
)
def test_only_actual_tag_push_can_enter_either_publisher(event, ref, expected):
    for name in ["publish-pypi", "github-release"]:
        condition = _workflow()["jobs"][name]["if"]
        assert (
            bool(_evaluate_workflow_expression(condition, event, ref, candidate=True))
            == expected
        )


def test_publication_still_requires_existing_config_and_successful_builds():
    workflow = _workflow()
    pypi = workflow["jobs"]["publish-pypi"]
    release = workflow["jobs"]["github-release"]
    assert not _evaluate_workflow_expression(
        pypi["if"], "push", "refs/tags/v0.5.1", trusted=False
    )
    assert not _evaluate_workflow_expression(
        release["if"], "push", "refs/tags/v0.5.1", failed_build=True
    )
    assert workflow["permissions"] == {"contents": "read"}
    assert pypi["permissions"] == {"id-token": "write"}
    assert release["permissions"] == {"contents": "write"}
    assert workflow["jobs"]["wheels-linux"]["strategy"]["matrix"]["target"] == [
        "x86_64",
        "aarch64",
    ]
    assert workflow["jobs"]["wheels-macos"]["strategy"]["matrix"]["target"] == [
        "x86_64",
        "aarch64",
    ]
    assert workflow["jobs"]["wheels-windows"]["strategy"]["matrix"]["target"] == ["x64"]


def test_both_publishers_depend_on_the_actual_wheel_qualification_job():
    workflow = _workflow()
    for name in ["publish-pypi", "github-release"]:
        assert "verify-artifacts" in workflow["jobs"][name]["needs"]
    text = (ROOT / ".github/workflows/release.yml").read_text()
    verify = text.split("  verify-artifacts:\n", 1)[1].split("  publish-pypi:\n", 1)[0]
    assert "run: python scripts/smoke_release_wheel.py --dist dist" in verify
    assert "continue-on-error" not in verify


def test_wheel_smoke_selects_linux_x86_64_from_all_five_platforms(tmp_path):
    names = [
        "rustscenic-0.5.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "rustscenic-0.5.1-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        "rustscenic-0.5.1-cp310-abi3-macosx_10_12_x86_64.whl",
        "rustscenic-0.5.1-cp310-abi3-macosx_11_0_arm64.whl",
        "rustscenic-0.5.1-cp310-abi3-win_amd64.whl",
    ]
    for name in names:
        _write_wheel(tmp_path / name)
    selected = SMOKE_MODULE.select_linux_wheel(tmp_path)
    assert selected.name == names[0]
    assert SMOKE_MODULE.wheel_version(selected) == "0.5.1"


@pytest.mark.parametrize("count", [0, 2])
def test_wheel_smoke_rejects_absent_or_ambiguous_linux_artifacts(tmp_path, count):
    for index in range(count):
        _write_wheel(
            tmp_path / f"rustscenic-0.5.1-cp310-abi3-manylinux_{index}_x86_64.whl"
        )
    with pytest.raises(ValueError, match="expected one Linux x86_64 wheel"):
        SMOKE_MODULE.select_linux_wheel(tmp_path)


@pytest.mark.parametrize(
    "filename", ["other-0.5.1-cp310-abi3-win_amd64.whl", "rustscenic-latest.whl"]
)
def test_wheel_smoke_rejects_unversioned_or_other_project_filenames(filename):
    with pytest.raises(ValueError, match="unexpected RustScenic wheel filename"):
        SMOKE_MODULE.wheel_version(Path(filename))
