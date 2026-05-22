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

    assert "`11x` to `52x`" in readme
    assert "`11x` to `52x`" in index
    assert "range from 11x to 52x" in benchmarks
    assert "Comparable or lower peak RSS" in readme
    assert "Comparable or lower than SCENIC+" in index
    assert "Lower peak RSS than SCENIC+" not in readme
    normalised_benchmarks = " ".join(benchmarks.split())
    assert "algorithm-identical kernel benchmark" in normalised_benchmarks
    assert "edge-set agreement" in benchmarks
