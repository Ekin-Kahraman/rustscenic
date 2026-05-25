import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_head_to_head_summary_records_scope_and_provenance():
    summary = json.loads(
        (ROOT / "validation/head_to_head/head_to_head_summary.json").read_text()
    )

    assert "correlation over the fixed search space" in summary["claim_scope"]
    assert summary["machine"]["python"]["rustscenic"] == "3.13.9"
    assert summary["machine"]["python"]["scenicplus"] == "3.11.8"
    assert "public benchmark record" in summary["source_result_files_status"]


def test_public_docs_keep_benchmark_claims_scoped():
    readme = (ROOT / "README.md").read_text()
    index = (ROOT / "site_docs/index.md").read_text()
    benchmarks = (ROOT / "site_docs/benchmarks.md").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()

    assert "RustScenic evidence snapshot" in readme
    assert "assets/rustscenic-evidence.svg" in index
    assert "Evidence Snapshot" in readme
    assert "Built" in readme
    assert "Released" in readme
    assert "Benchmarked" in readme
    assert "Lab-validated" in readme
    assert "Current release `v0.4.7`" in readme
    assert "`11x` to `52x`" in readme
    assert "`11x` to `52x`" in index
    assert "range from 11x to 52x" in benchmarks
    assert "`6.34 GB` peak RSS on a 100k-cell four-stage scale check" in readme
    assert "legacy pySCENIC reports exceed `40 GB`" in readme
    assert "Memory scaling" in index
    assert "6.34 GB RSS" in benchmarks
    assert "legacy pySCENIC reports exceed 40 GB" in benchmarks
    assert "Lower peak RSS than SCENIC+" not in readme
    normalised_benchmarks = " ".join(benchmarks.split())
    assert "algorithm-identical kernel benchmark" in normalised_benchmarks
    assert "edge-set agreement" in benchmarks
    assert "## Unreleased" in changelog
    assert "collaborator lab validation" in changelog


def test_human_brain_external_validation_is_scoped():
    artefact = json.loads(
        (ROOT / "validation/community/human_brain_10k_v0.4.6.json").read_text()
    )
    validation = (ROOT / "site_docs/validation.md").read_text()
    adoption = (ROOT / "site_docs/adoption.md").read_text()
    readme = (ROOT / "README.md").read_text()

    assert artefact["rustscenic_version"] == "0.4.6"
    assert artefact["shapes"]["rna_post_qc"][0] == 8215
    assert artefact["peak_rss_gb"] == 24.99
    assert artefact["biological_sanity"]["fraction_recovered"] == 0.9412
    assert "Huang Lab collaborator artefacts" in readme
    assert "Huang Lab collaborator artefacts" in validation
    assert "16 of 17 expected brain TFs recovered" in validation
    assert "Collaborator human brain GEM-X full monolith run recovered 16 of 17" in adoption
    assert "not a SCENIC+ head-to-head row" in validation
    assert "full monolith run recovering `16/17`" in readme
