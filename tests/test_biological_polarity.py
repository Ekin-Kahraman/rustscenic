"""Regression checks for the cached-output biological assessment adapter."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "validation/biological_fidelity/polarity.py"
spec = importlib.util.spec_from_file_location("biological_polarity", path)
polarity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(polarity)


def test_opposite_programs_are_not_merged_and_duplicate_peaks_do_not_weight_genes():
    edges = pd.DataFrame({
        "tf": ["SPI1"] * 3,
        "target": ["LYZ", "LYZ", "CD3D"],
        "context": ["SPI1_activator; positive region-gene links"] * 2
                   + ["SPI1_repressor; positive region-gene links"],
    })
    assert polarity.signed_gene_sets(edges, "rustscenic") == {
        "SPI1_activator": ["LYZ"], "SPI1_repressor": ["CD3D"]}
    activity = pd.DataFrame({"SPI1_activator": [.8, .1], "SPI1_repressor": [.2, .7]}, index=["a", "b"])
    original = activity.copy(deep=True)
    result = polarity.signed_contrast(activity)
    np.testing.assert_allclose(result.SPI1, [.6, -.6])
    assert result.index.tolist() == ["a", "b"]
    pd.testing.assert_frame_equal(activity, original)


def test_reference_uses_tf_gene_sign_not_region_gene_sign():
    edges = pd.DataFrame({"tf": ["TF", "TF"], "target": ["A", "B"],
                          "context": ["['negative r2g', 'positive tf2g']",
                                      "['positive r2g', 'negative tf2g']"]})
    assert polarity.signed_gene_sets(edges, "reference") == {
        "TF_activator": ["A"], "TF_repressor": ["B"]}


def test_missing_one_direction_is_zero_and_tf_names_are_preserved():
    activity = pd.DataFrame({"TF_with_underscore_activator": [.7], "NEG_repressor": [.3]})
    result = polarity.signed_contrast(activity)
    assert result.to_dict("list") == {"NEG": [-.3], "TF_with_underscore": [.7]}
    assert polarity.signed_contrast(pd.DataFrame(index=["cell"])).shape == (1, 0)


@pytest.mark.parametrize("context", ["unknown", "['positive tf2g', 'negative tf2g']", None])
def test_unknown_or_ambiguous_context_fails(context):
    edges = pd.DataFrame({"tf": ["TF"], "target": ["G"], "context": [context]})
    with pytest.raises(ValueError):
        polarity.signed_gene_sets(edges, "reference")


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"TF": [.5]}),
    pd.DataFrame({"TF_activator": [np.nan]}),
    pd.DataFrame({"TF_activator": [np.inf]}),
    pd.DataFrame([[.1, .2]], columns=["TF_activator"] * 2),
    pd.DataFrame({"TF_activator": [.1, .2]}, index=["same"] * 2),
    pd.DataFrame({"_activator": [.1]}),
])
def test_invalid_activity_fails_instead_of_silent_aggregation(frame):
    with pytest.raises(ValueError):
        polarity.signed_contrast(frame)
