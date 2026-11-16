"""
Renamed arguments keep working under their old name, with a deprecation warning.
"""
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from omicspylib import PairwiseComparisonTTestFC, PeptidesDataset, ProteinsDataset
from omicspylib.plots import plot_missing_values, plot_record_frequency
from omicspylib.utils.deprecation import resolve_renamed_arg


def test_resolve_renamed_arg_returns_new_value_without_warning(recwarn) -> None:
    assert resolve_renamed_arg("a", None, "new", "old") == "a"
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_resolve_renamed_arg_warns_on_old_name() -> None:
    with pytest.deprecated_call(match="'old' is deprecated.*Use 'new'"):
        assert resolve_renamed_arg(None, "b", "new", "old") == "b"


def test_resolve_renamed_arg_keeps_falsy_old_value() -> None:
    with pytest.deprecated_call():
        assert resolve_renamed_arg(True, False, "new", "old", default=True) is False


def test_resolve_renamed_arg_rejects_both_names() -> None:
    with pytest.raises(TypeError, match="both 'new' and its deprecated alias 'old'"):
        resolve_renamed_arg("a", "b", "new", "old")


@pytest.mark.parametrize("dataset_fixture", ["proteins_dataset", "peptides_dataset"])
def test_filter_accepts_deprecated_exp_and_cond(dataset_fixture, request) -> None:
    dataset: ProteinsDataset | PeptidesDataset = request.getfixturevalue(dataset_fixture)
    expected = dataset.filter(experiments=["c1_rep1", "c2_rep1"], conditions=["c1", "c2"])

    with pytest.deprecated_call():
        result = dataset.filter(exp=["c1_rep1", "c2_rep1"], cond=["c1", "c2"])

    pd.testing.assert_frame_equal(result.to_table(), expected.to_table())


@pytest.mark.parametrize("dataset_fixture", ["proteins_dataset", "peptides_dataset"])
def test_drop_accepts_deprecated_exp_and_cond(dataset_fixture, request) -> None:
    dataset: ProteinsDataset | PeptidesDataset = request.getfixturevalue(dataset_fixture)
    expected = dataset.drop(experiments="c1_rep1", conditions="c3")

    with pytest.deprecated_call():
        result = dataset.drop(exp="c1_rep1", cond="c3")

    pd.testing.assert_frame_equal(result.to_table(), expected.to_table())


def test_filter_rejects_old_and_new_names_together(proteins_dataset: ProteinsDataset) -> None:
    with pytest.raises(TypeError):
        proteins_dataset.filter(conditions=["c1"], cond=["c2"])


def test_normalize_accepts_deprecated_ref_exp(proteins_dataset: ProteinsDataset) -> None:
    expected = proteins_dataset.normalize(method="median", ref_experiment="c1_rep2")

    with pytest.deprecated_call():
        result = proteins_dataset.normalize(method="median", ref_exp="c1_rep2")

    pd.testing.assert_frame_equal(result.to_table(), expected.to_table())


def test_ttest_comparison_accepts_deprecated_use_log_transformed(
    proteins_dataset: ProteinsDataset,
) -> None:
    comparison = PairwiseComparisonTTestFC(proteins_dataset, "c1", "c2")
    expected = comparison.eval(log_transform=False)

    with pytest.deprecated_call():
        result = comparison.eval(use_log_transformed=False)

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("plot_fn", [plot_missing_values, plot_record_frequency])
def test_plots_accept_deprecated_min_threshold(plot_fn, proteins_dataset) -> None:
    expected_ax = plot_fn(proteins_dataset, na_threshold=1e6)

    with pytest.deprecated_call():
        ax = plot_fn(proteins_dataset, min_threshold=1e6)

    assert ax.get_title() == expected_ax.get_title()
    plt.close("all")
