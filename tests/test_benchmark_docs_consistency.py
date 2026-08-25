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
    scaling = json.loads(
        (ROOT / "validation/scaling/e2e_100k_synthetic.json").read_text()
    )

    assert "rustscenic-evidence.svg" not in readme
    assert "assets/rustscenic-evidence.svg" in index
    assert "## Evidence Snapshot" not in readme
    assert "## Highlights" in readme
    assert "Current release: `v0.5.0`" in readme
    assert "`11x` to `52x`" in readme
    assert "`11x` to `52x`" in index
    assert "sampled real-data inputs in a single-machine output-path benchmark" in readme
    assert "sampled real-data inputs in a single-machine output-path benchmark" in index
    assert "range from 11x to 52x" in benchmarks
    assert "Historical `v0.3.2` synthetic 100k-cell seven-stage scale check" in readme
    assert "legacy pySCENIC reports exceed `40 GB`" not in readme
    assert "Huang Lab collaborator run recovered `16/17`" in readme
    assert "Memory scaling" in index
    assert "Historical RustScenic v0.3.2 synthetic 100k cells" in benchmarks
    assert "Real human brain GEM-X monolith" in benchmarks
    assert "24.99 GB" in benchmarks
    assert "not a default-parameter or full-TF memory claim" in benchmarks
    assert "Lower peak RSS than SCENIC+" not in readme
    normalised_benchmarks = " ".join(benchmarks.split())
    assert "algorithm-identical kernel benchmark" in normalised_benchmarks
    assert "edge-set agreement" in benchmarks
    assert "## Unreleased" in changelog
    assert "## 0.5.0 - 2026-08-25" in changelog
    assert "### Migration from 0.4.x" in changelog
    assert "collaborator lab validation" in changelog
    assert scaling["benchmark_kind"] == "synthetic_scale_check"
    assert scaling["rustscenic_version"] == "0.3.2"
    assert scaling["rustscenic_sha"] == "bf1be27ef2cd4f8d3e3b2508eef3678ac64d3999"
    assert scaling["environment"]["hardware"] is None
    assert scaling["n_cells"] == 100_000
    assert scaling["n_genes"] == 15_000
    assert scaling["n_peaks"] == 50_000
    assert scaling["K"] == 30
    assert scaling["n_grn_estimators"] == 20
    assert scaling["raw_fragment_preprocessing_included"] is False
    assert "not a v0.5.0" in scaling["claim_scope"]


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
    assert "Huang Lab collaborator run recovered `16/17`" in readme
    assert "Huang Lab collaborator artefacts" in validation
    assert "16 of 17 expected brain TFs recovered" in validation
    assert "Collaborator human brain GEM-X full monolith run recovered 16 of 17" in adoption
    assert "not a SCENIC+ head-to-head row" in validation
    assert "Huang Lab collaborator run recovered `16/17`" in readme
