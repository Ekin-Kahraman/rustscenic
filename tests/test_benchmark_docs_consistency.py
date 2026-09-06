import ast
import json
import re
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

    assert "rustscenic-evidence.svg" not in readme
    assert "assets/rustscenic-evidence.svg" in index
    assert "## Evidence Snapshot" not in readme
    assert "## Highlights" in readme
    assert "Current release: `v0.4.7`" in readme
    assert "`11x` to `52x`" in readme
    assert "`11x` to `52x`" in index
    assert "range from 11x to 52x" in benchmarks
    for document in (readme, index, benchmarks):
        assert "6.34 GB" not in document
        assert "reports exceed" not in document
        assert "71.49 GB" in document
        assert "v0.5.0" in document
    assert "1.3 million mouse-brain cells" in readme
    assert "2,095 selected genes" in readme
    assert "release candidate" in readme
    assert "21.4%" in readme
    assert "Icahn School of Medicine at Mount Sinai" in readme
    assert "Huang Lab collaborator run recovered `16/17`" in readme
    assert "Memory scaling" in index
    assert "24.99 GB" in benchmarks
    assert "not comparable" in benchmarks
    assert "Synthetic seven-stage" in benchmarks
    assert "not model convergence" in benchmarks
    assert "Lower peak RSS than SCENIC+" not in readme
    normalised_benchmarks = " ".join(benchmarks.split())
    assert "different methods for enhancer linking" in normalised_benchmarks
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
    assert "Huang Lab collaborator run recovered `16/17`" in readme
    assert "Huang Lab collaborator artefacts" in validation
    assert "16 of 17 expected brain TFs recovered" in validation
    assert "Collaborator human brain workflow recovered 16 of 17" in adoption
    assert "not a SCENIC+ head-to-head row" in validation
    assert "Huang Lab collaborator run recovered `16/17`" in readme


def test_quickstarts_use_published_api_and_explain_candidate_gene_sets():
    for path in (ROOT / "README.md", ROOT / "site_docs/quickstart.md"):
        document = path.read_text()
        assert "v0.4.7" in document
        assert "candidate" in document
        blocks = re.findall(r"```python\n(.*?)```", document, re.DOTALL)
        assert blocks
        for block in blocks:
            calls = {
                node.func.attr
                for node in ast.walk(ast.parse(block))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            assert {"infer", "score"} <= calls
            assert not {"add_correlation", "build_regulons"} & calls
    api = (ROOT / "site_docs/api.md").read_text()
    assert "not in the published v0.4.7 package" in api
