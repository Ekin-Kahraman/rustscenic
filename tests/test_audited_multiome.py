"""Parity and failure-boundary tests for the explicit paired recipe."""
import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from rustscenic import aucell, enhancer
from rustscenic import _audited_multiome as workflow


def matrices(cells=37, peaks=21):
    rng = np.random.default_rng(52)
    cell_topic = rng.dirichlet(np.ones(30), size=cells).astype(np.float32)
    topic_peak = rng.dirichlet(np.ones(peaks), size=30).astype(np.float32)
    topic_peak[:, -1] = 0
    names = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(peaks)]
    return (pd.DataFrame(cell_topic, index=[f"cell{i}" for i in range(cells)]),
            pd.DataFrame(topic_peak, columns=names))


def test_disk_imputation_matches_dense_recipe_and_cleans_scratch(tmp_path):
    cell_topic, topic_peak = matrices()
    expected = (cell_topic.to_numpy() @ topic_peak.to_numpy() * np.float32(1000000)).astype(np.int32)
    with workflow._imputed(cell_topic, topic_peak, tmp_path) as (values, kept):
        assert isinstance(values, np.memmap)
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(kept, np.flatnonzero(np.any(expected != 0, axis=0)))
        assert len(list(tmp_path.iterdir())) == 1
    assert list(tmp_path.iterdir()) == []


def test_disk_imputation_cleans_after_failure(tmp_path):
    cell_topic, topic_peak = matrices()
    with pytest.raises(RuntimeError):
        with workflow._imputed(cell_topic, topic_peak, tmp_path):
            raise RuntimeError("failed downstream stage")
    assert list(tmp_path.iterdir()) == []


def test_bounded_linking_and_seeded_scoring_match_dense(tmp_path):
    cell_topic, topic_peak = matrices(peaks=301)
    rng = np.random.default_rng(17)
    names = [f"Gene{i}" for i in range(40)]
    rna = workflow._ann(sparse.csr_matrix(rng.random((len(cell_topic), 40), dtype=np.float32)),
                        cell_topic.index, names)
    coords = pd.DataFrame({"gene": names, "chrom": "chr1", "tss": np.arange(40) * 100})
    expected = (cell_topic.to_numpy() @ topic_peak.to_numpy() * np.float32(1000000)).astype(np.int32)
    kept = np.flatnonzero(np.any(expected != 0, axis=0))
    atac = workflow._ann(expected[:, kept], cell_topic.index, topic_peak.columns[kept])
    full_links = enhancer.link_peaks_to_genes(rna, atac, coords, max_distance=150000, min_abs_corr=.03)
    regs = {"TF": list(atac.var_names[:13]), "TF2": list(atac.var_names[40:80])}
    expected_scores = aucell.score(atac, regs, seed=20261106)
    with workflow._imputed(cell_topic, topic_peak, tmp_path) as (values, kept):
        linked = workflow._link_imputed(rna, values, kept, topic_peak.columns, coords)
        scored = workflow._score_imputed(values, kept, cell_topic.index, topic_peak.columns, regs, 20261106)
    cols = ["peak_id", "gene"]
    pd.testing.assert_frame_equal(linked.sort_values(cols).reset_index(drop=True),
                                  full_links.sort_values(cols).reset_index(drop=True))
    pd.testing.assert_frame_equal(scored, expected_scores)


def test_region_support_uses_direct_or_orthology_only_and_no_fallback():
    peaks = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(20)]
    ranks = pd.DataFrame(np.full((60, 20), 5000, dtype=np.int32),
                         index=[f"motif{i}" for i in range(60)], columns=[f"q{i}" for i in range(20)])
    ranks.iloc[0] = np.arange(20)
    ranks.attrs["rank_universe_size"] = 10000
    mapping = pd.DataFrame({"Target": peaks, "Query": ranks.columns})
    annotation = pd.DataFrame({"motif": ["motif0"] * 3, "TF": ["Direct", "Ortholog", "Similar"],
        "description": ["gene is directly annotated", "gene is orthologous", "gene is orthologous and similar"]})
    support, scores = workflow._region_support({"Topic1": peaks}, list(annotation.TF), ranks, mapping, annotation)
    assert set(support.TF) == {"Direct", "Ortholog"}
    assert set(support.peak_id) == set(peaks)
    assert set(scores.motif) == {"motif0"}
    empty, _ = workflow._region_support({"Topic1": peaks}, ["Missing"], ranks, mapping, annotation)
    assert empty.empty
    assert list(empty.columns) == workflow._SUPPORT_COLUMNS


def test_paired_axes_reject_mismatched_cells_before_creating_outputs(tmp_path):
    rna = workflow._ann(np.ones((2, 3)), ["a", "b"], ["G1", "G2", "G3"])
    atac = workflow._ann(np.ones((2, 3)), ["b", "a"], ["p1", "p2", "p3"])
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="same order"):
        workflow.run(rna, atac, ["G1"], gene_coords=None, region_rankings=None,
                     region_mapping=None, motif_annotations=None, output_dir=output)
    assert not output.exists()


def test_cached_topics_reject_wrong_region_order():
    cell_topic, topic_peak = matrices()
    topic_peak.index = cell_topic.columns
    atac = workflow._ann(sparse.csr_matrix(np.ones((len(cell_topic), topic_peak.shape[1]))),
                         cell_topic.index, topic_peak.columns)
    with pytest.raises(ValueError, match="region order"):
        workflow._topics(atac, 777, 1, {"cell_topic": cell_topic,
                         "topic_peak": topic_peak[topic_peak.columns[::-1]]})


def test_raw_atac_rejects_fractional_counts():
    atac = workflow._ann(sparse.csr_matrix([[.1, 1], [0, 1]]), ["a", "b"], ["p1", "p2"])
    with pytest.raises(ValueError, match="integer counts"):
        workflow._topics(atac, 777, 1, {})


@pytest.mark.parametrize("activity_mode", ["union", "directional"])
def test_full_recipe_returns_explicit_empty_result_when_no_annotated_tf_survives(tmp_path, activity_mode):
    cell_topic, topic_peak = matrices()
    topic_peak.index = cell_topic.columns
    genes = [f"Gene{i}" for i in range(40)]
    rna = workflow._ann(np.random.default_rng(5).random((37, 40), dtype=np.float32), cell_topic.index, genes)
    atac = workflow._ann(sparse.csr_matrix(np.ones((37, 21))), cell_topic.index, topic_peak.columns)
    ranks = pd.DataFrame(np.full((60, 21), 5000, dtype=np.int32),
        index=[f"m{i}" for i in range(60)], columns=topic_peak.columns)
    ranks.iloc[0] = np.arange(21)
    ranks.attrs["rank_universe_size"] = 10000
    result = workflow.run(rna, atac, ["Gene0"],
        gene_coords=pd.DataFrame({"gene": genes, "chrom": "chr1", "tss": np.arange(40) * 100}),
        region_rankings=ranks,
        region_mapping=pd.DataFrame({"Target": ranks.columns, "Query": ranks.columns}),
        motif_annotations=pd.DataFrame({"motif": ["m0"], "TF": ["Unrelated"], "description": ["gene is directly annotated"]}),
        output_dir=tmp_path, activity_mode=activity_mode,
        cached_stages={"cell_topic": cell_topic, "topic_peak": topic_peak,
                       "grn": pd.DataFrame({"TF": "Gene0", "target": genes[1:], "importance": 1.})})
    assert result["metadata"]["status"] == "no_supported_eregulons"
    assert result["regulons"] == result["region_regulons"] == {}
    assert result["gene_activity"].shape == result["region_activity"].shape == (37, 0)
    assert result["eregulon_edges"].empty
    assert result["metadata"]["replayed_stages"] == ["cell_topic", "grn", "topic_peak"]
    assert result["metadata"]["activity_mode"] == activity_mode


def test_paired_activity_mode_rejected_before_output(tmp_path):
    rna = workflow._ann(np.ones((2, 3)), ["a", "b"], ["G1", "G2", "G3"])
    out = tmp_path / "never-created"
    with pytest.raises(ValueError, match="activity_mode"):
        workflow.run(rna, rna, ["G1"], gene_coords=None, region_rankings=None,
            region_mapping=None, motif_annotations=None, output_dir=out, activity_mode="invalid")
    assert not out.exists()


def test_directional_activity_preserves_opposite_modules_and_union_default(tmp_path, monkeypatch):
    cell_topic, topic_peak = matrices(cells=6, peaks=61)
    genes = [f"Gene{i}" for i in range(60)]
    rna = workflow._ann(np.random.default_rng(7).random((6, 60), dtype=np.float32), cell_topic.index, genes)
    atac = workflow._ann(np.ones((6, 61), dtype=np.float32), cell_topic.index, topic_peak.columns)
    assembled = pd.DataFrame({"tf": ["Gene0_activator"] * 5 + ["Gene0_repressor"] * 5,
        "target_gene": genes[1:11], "enhancer": topic_peak.columns[:10]})
    monkeypatch.setattr(workflow, "_topics", lambda *a: (cell_topic, topic_peak, {}))
    monkeypatch.setattr(workflow, "_region_support", lambda *a: (
        pd.DataFrame(columns=workflow._SUPPORT_COLUMNS), pd.DataFrame(columns=workflow._SCORE_COLUMNS)))
    monkeypatch.setattr(workflow, "_link_imputed", lambda *a: pd.DataFrame())
    monkeypatch.setattr(workflow.grn, "add_correlation", lambda frame, *a, **kw: frame)
    monkeypatch.setattr(workflow.eregulon, "build_eregulons_dataframe", lambda *a, **kw: assembled.copy())
    common = dict(gene_coords=None, region_rankings=None, region_mapping=None,
                  motif_annotations=None, cached_stages={"grn": pd.DataFrame()})
    union = workflow.run(rna, atac, ["Gene0"], output_dir=tmp_path / "union", **common)
    signed = workflow.run(rna, atac, ["Gene0"], output_dir=tmp_path / "signed", activity_mode="directional", **common)
    assert union["regulons"] == {"Gene0": sorted(genes[1:11])}
    assert signed["regulons"] == {"Gene0_activator": genes[1:6], "Gene0_repressor": sorted(genes[6:11])}
    assert signed["region_regulons"] == {
        "Gene0_activator": sorted(topic_peak.columns[:5]), "Gene0_repressor": sorted(topic_peak.columns[5:10])}
    pd.testing.assert_frame_equal(signed["eregulon_edges"], union["eregulon_edges"])
    for result in (union, signed):
        expected = aucell.score(rna, result["regulons"], seed=20261106)
        pd.testing.assert_frame_equal(result["gene_activity"], expected)
        assert result["region_activity"].columns.tolist() == list(result["region_regulons"])
        assert result["gene_activity"].dtypes.eq(np.float32).all()
