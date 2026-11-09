"""Public audited recipe boundaries; real-data parity is recorded separately."""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rustscenic import validated, cli, pipeline


def inputs():
    genes = ["TF"] + [f"G{i}" for i in range(59)]
    rna = ad.AnnData(sparse.csr_matrix(np.ones((8, 60), dtype=np.float32)),
                     obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
                     var=pd.DataFrame(index=genes))
    ranks = pd.DataFrame([np.arange(60)], index=pd.Index(["motif"], name="motifs"), columns=genes)
    annotations = pd.DataFrame({"motif": ["motif"], "TF": ["TF"]})
    return rna, ranks, annotations


def test_projected_rankings_preserve_original_universe(tmp_path):
    _, ranks, _ = inputs()
    ranks.attrs["rank_universe_size"] = 20000
    assert validated.load_rankings(ranks, ["TF"]).attrs["rank_universe_size"] == 20000
    path = tmp_path / "ranks.feather"
    ranks.reset_index().to_feather(path)
    path.with_suffix(".json").write_text('{"original_rank_universe":20000}')
    projected = validated.load_rankings(path, ["TF", "G0"])
    assert projected.shape == (1, 2)
    assert projected.attrs["rank_universe_size"] == 20000
    assert projected.loc["motif", "G0"] == 1


def test_standard_annotation_filter(tmp_path):
    path = tmp_path / "annotations.tbl"
    pd.DataFrame({"#motif_id": ["a", "b", "c"], "gene_name": ["TF"] * 3,
        "motif_similarity_qvalue": [0, .01, 0], "orthologous_identity": [1, 1, -1]}).to_csv(path, sep="\t", index=False)
    assert validated.load_annotations(path).motif.tolist() == ["a"]
    with pytest.raises(ValueError, match="description"):
        validated.load_annotations(path, paired=True)


@pytest.mark.parametrize("bad_rank", [60, -1, .5, np.nan])
def test_invalid_ranks_or_missing_projection_universe_rejected(bad_rank):
    _, ranks, _ = inputs()
    ranks = ranks.astype(float)
    ranks.iloc[0, 0] = bad_rank
    with pytest.raises(ValueError, match="original rank universe"):
        validated.load_rankings(ranks, ["TF"])


def test_species_alias_uses_bundled_tfs(tmp_path, monkeypatch):
    rna, ranks, annotations = inputs()
    rna.var_names = ["SPI1"] + rna.var_names[1:].tolist()
    ranks.columns = rna.var_names
    annotations["TF"] = "SPI1"
    fake_network(monkeypatch, rna)
    observed = []
    def infer(*a, **kw):
        observed.extend(kw["tf_names"])
        raise RuntimeError("reached native inference")
    monkeypatch.setattr(validated.grn, "infer", infer)
    with pytest.raises(RuntimeError, match="reached native"):
        validated.run(rna, tmp_path / "alias", tfs="hs", motif_rankings=ranks, motif_annotations=annotations)
    assert "SPI1" in observed and len(observed) > 100


@pytest.mark.parametrize("failure", ["existing", "duplicates", "negative", "nonfinite", "no_tfs", "missing_annotations", "missing_rankings"])
def test_invalid_inputs_fail_before_inference(tmp_path, monkeypatch, failure):
    rna, ranks, annotations = inputs()
    out = tmp_path / "run"
    tfs = ["TF"]
    if failure == "existing":
        out.mkdir(); (out / "important.txt").write_text("preserve")
    elif failure == "duplicates":
        rna.var_names = ["TF"] * 60
    elif failure == "negative":
        rna.X.data[0] = -1
    elif failure == "nonfinite":
        rna.X.data[0] = np.nan
    elif failure == "no_tfs":
        tfs = ["absent"]
    elif failure == "missing_annotations":
        annotations = None
    else:
        ranks = None
    monkeypatch.setattr(validated.grn, "infer", lambda *a, **k: pytest.fail("inference should not start"))
    with pytest.raises(ValueError):
        validated.run(rna, out, tfs=tfs, motif_annotations=annotations, motif_rankings=ranks)
    if failure == "existing":
        assert (out / "important.txt").read_text() == "preserve"
    else:
        assert not out.exists()


def fake_network(monkeypatch, rna):
    network = pd.DataFrame({"TF": ["TF"] * 10, "target": rna.var_names[1:11], "importance": np.ones(10)})
    monkeypatch.setattr(validated.grn, "infer", lambda *a, **kw: network)
    monkeypatch.setattr(validated.grn, "add_correlation", lambda x, *a, **kw: x)
    monkeypatch.setattr(validated.grn, "build_regulons", lambda *a, **kw: {"TF_activator": rna.var_names[1:11].tolist()})


def test_no_supported_motifs_never_falls_back(tmp_path, monkeypatch):
    rna, ranks, annotations = inputs()
    fake_network(monkeypatch, rna)
    monkeypatch.setattr(validated.cistarget, "enrich", lambda *a, **kw: pd.DataFrame(columns=["regulon", "motif", "auc", "nes"]))
    # Real native pruning handles an empty evidence table; no AUC call allowed.
    monkeypatch.setattr(validated.aucell, "score", lambda *a, **kw: pytest.fail("unsupported network scored"))
    with pytest.warns(UserWarning, match="No regulons"):
        receipt = validated.run(rna, tmp_path / "run", tfs=["TF"], motif_rankings=ranks, motif_annotations=annotations)
    assert receipt["status"] == "no_supported_regulons"
    assert receipt["gene_activity_shape"] == [8, 0]
    assert json.loads((tmp_path / "run/regulons.json").read_text()) == {}
    assert pd.read_parquet(tmp_path / "run/gene_activity.parquet").index.equals(rna.obs_names)
    report = pd.read_parquet(tmp_path / "run/regulator_coverage.parquet")
    assert report.status.tolist() == ["not_retained_after_motif_and_size_filters"]


def test_seeded_scores_are_for_pruned_signatures_only(tmp_path, monkeypatch):
    rna, ranks, annotations = inputs()
    fake_network(monkeypatch, rna)
    monkeypatch.setattr(validated.cistarget, "enrich", lambda *a, **kw: pd.DataFrame())
    signatures = {"TF_activator": rna.var_names[1:6].tolist()}
    def prune(*a, **kw):
        assert kw["min_genes"] == 5 and kw["nes_threshold"] == 3.
        return signatures
    monkeypatch.setattr(validated.cistarget, "prune_regulons", prune)
    actual_score = validated.aucell.score
    def score(expression, sets, **kw):
        assert sets == signatures and kw["seed"] == 20261106
        return actual_score(expression, sets, **kw)
    monkeypatch.setattr(validated.aucell, "score", score)
    manifest = validated.run(rna, tmp_path / "run", tfs=["TF", "missing"], motif_rankings=ranks, motif_annotations=annotations)
    assert manifest["n_regulons"] == 1 and manifest["aucell_seed"] == 20261106
    report = pd.read_parquet(tmp_path / "run/regulator_coverage.parquet")
    assert report.status.tolist() == ["retained", "not_in_expression"]
    assert json.loads((tmp_path / "run/manifest.json").read_text())["scored_signatures"] == "motif-pruned signed RNA regulons"


def test_cli_forwards_audited_recipe(tmp_path, monkeypatch):
    seen = {}
    def run(*a, **kw):
        seen.update(kw)
        return {"status": "complete", "n_regulons": 2}
    monkeypatch.setattr(validated, "run", run)
    assert cli.main(["validated", "--rna", "input.h5ad", "--tfs", "hs",
        "--output", str(tmp_path), "--motif-annotations", "motifs.tbl", "--motif-rankings", "ranks.feather",
        "--aucell-seed", "20261106", "--threads", "2", "--activity-mode", "directional"]) == 0
    assert seen["aucell_seed"] == 20261106 and seen["threads"] == 2
    assert seen["activity_mode"] == "directional"


def test_paired_directional_coverage_counts_unique_targets():
    rna, _, _ = inputs()
    result = {"regulons": {"TF_activator": ["G0", "G1"], "TF_repressor": ["G2"]}}
    row = validated._coverage(["TF"], rna, result).iloc[0]
    assert row.status == "retained" and row.retained_target_genes == 3


@pytest.mark.parametrize("mode", ["union", "unknown", ""])
def test_rna_activity_mode_rejected_before_output(tmp_path, mode):
    rna, ranks, annotations = inputs()
    out = tmp_path / "never-created"
    with pytest.raises(ValueError, match="activity_mode"):
        validated.run(rna, out, tfs=["TF"], motif_rankings=ranks,
                      motif_annotations=annotations, activity_mode=mode)
    assert not out.exists()


def test_generic_pipeline_forwards_seed_without_changing_default(tmp_path, monkeypatch):
    rna, _, _ = inputs()
    rna.X = sparse.csr_matrix(np.random.default_rng(7).lognormal(size=rna.shape).astype(np.float32))
    received = []
    score = validated.aucell.score
    def observe(*a, **kw):
        received.append(kw.get("seed"))
        return score(*a, **kw)
    monkeypatch.setattr(validated.aucell, "score", observe)
    # Unsigned modules isolate seed forwarding from correlation thresholds.
    pipeline.run(rna, tmp_path / "legacy", tfs=["TF"], grn_n_estimators=2,
                 grn_regulon_polarities="unsigned", write_integrated_adata=False, verbose=False)
    result = pipeline.run(rna, tmp_path / "seeded", tfs=["TF"], grn_n_estimators=2,
        grn_regulon_polarities="unsigned", aucell_seed=20261106, write_integrated_adata=False, verbose=False)
    assert received == [None, 20261106]
    assert result.aucell_seed == 20261106
