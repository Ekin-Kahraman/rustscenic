# Validating an unpublished source candidate

The release checker distinguishes an unpublished source candidate from a
publication-ready release. Both modes validate package, workspace, citation and
Zenodo versions, license, concept DOI and workspace lockfile versions. Built
archives must also have matching filenames and internal wheel `METADATA` or
sdist `PKG-INFO` Name/Version fields. The checker reads these metadata entries
without extracting archive contents.

## Candidate state

An unpublished candidate keeps the README installation version, website and
evidence card at the currently published version. The source metadata uses a
strictly newer version. `release-candidate.json` records the candidate version,
published baseline and ISO preparation date. README and website identify the
candidate separately. Its changelog section, release-notes title and banner,
citation message and Zenodo description explicitly identify it as unpublished;
the citation does not claim a release date.

```bash
python scripts/check_release_version.py --candidate
python scripts/check_release_version.py --candidate --dist target/wheels
```

`--candidate` and `--tag` cannot be combined. Without `--candidate`, candidate
state fails validation; candidate mode does not excuse inconsistent versions,
missing archives or a false publication date.

The release workflow's manual `candidate` input defaults to false. Set it to
true only to inspect a candidate on a non-tag branch. A manual run, including a
manual run whose ref is a tag, cannot publish to PyPI or create a GitHub Release.
Both publishers require a tag-push event and successful artifact verification.
The existing trusted-publisher configuration remains a separate requirement for
PyPI upload.

## Artifact qualification

The workflow builds the existing five platform wheels and source distribution.
After checking every archive's internal metadata, `verify-artifacts` selects
exactly one Linux x86_64 wheel and runs:

```bash
python scripts/smoke_release_wheel.py --dist dist
```

This installs the actual wheel and its dependencies in a fresh temporary venv.
Qualification runs in isolated Python mode from outside the checkout and checks
that package and native-extension imports resolve inside that venv. It verifies
package/native versions, malformed AUCell axis rejection, matching valid
dense/sparse AUCell scores and the documented GRN provenance fields on a tiny
synthetic input. The source-distribution build also has its separate fresh
installation/import check. These checks qualify packaging and selected behavior;
they do not establish biological accuracy or performance.

For a locally built wheel compatible with the current machine:

```bash
python scripts/smoke_release_wheel.py --wheel /absolute/path/to/rustscenic-0.5.1-cp310-abi3-macosx_11_0_arm64.whl
```

## Later publication transition

Candidate validation does not authorize publication. After reviewing the exact
commit, all platform artifacts and qualification results, a maintainer must
review a separate publication transition: remove `release-candidate.json` and
candidate labels, update the published/installation labels, replace the
candidate changelog heading with the actual release date, add the matching
citation release date, remove unpublished citation/Zenodo markers, and finalize
the release notes and source documentation without stale candidate claims.
Historical benchmark measurements retain their original version and scope.

Then run strict validation against the intended tag and artifacts:

```bash
python scripts/check_release_version.py --tag v0.5.1 --dist target/wheels
```

Tagging and publication require a separate explicit maintainer decision. No
candidate command promotes a release automatically.
