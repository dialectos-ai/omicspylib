import warnings

import pandas as pd
import pytest

import omicspylib as opl
from omicspylib import ProteinsDataset
from omicspylib.datasets.peptides import PeptidesDataset


def test_protein_dataset_concatenation() -> None:
    """
    You can have to datasets with different experimental conditions.
    Text that you can join them in one object.
    """
    # setup
    """Create a basic proteins dataset object."""
    data_df = pd.read_csv("tests/data/protein_dataset.tsv", sep="\t")
    d1_config = {
        "id_col": "protein_id",
        "conditions": {"c1": ["c1_rep1", "c1_rep2", "c1_rep3", "c1_rep4", "c1_rep5"],}}
    d2_config = {
        "id_col": "protein_id",
        "conditions": {"c2": ["c2_rep1", "c2_rep2", "c2_rep3", "c2_rep4", "c2_rep5"],}}
    d3_config = {
        "id_col": "protein_id",
        "conditions": {"c3": ["c3_rep1", "c3_rep2", "c3_rep3", "c3_rep4", "c3_rep5"],}}
    d1 = ProteinsDataset.from_df(data_df, **d1_config)
    d2 = ProteinsDataset.from_df(data_df, **d2_config)
    d3 = ProteinsDataset.from_df(data_df, **d3_config)

    # action
    joined_dset = opl.concat([d1, d2, d3])

    # assertion
    assert isinstance(joined_dset, ProteinsDataset)
    conditions = joined_dset.condition_names
    for exp_cond in ["c1", "c2", "c3"]:
        assert exp_cond in conditions


def test_peptide_dataset_concatenation() -> None:
    """
    You can have to datasets with different experimental conditions.
    Text that you can join them in one object.
    """
    # setup
    """Create a basic proteins dataset object."""
    data_df = pd.read_csv("tests/data/peptides_dataset.tsv", sep="\t")
    d1_config = {
        "id_col": "peptide_id",
        "protein_id_col": "protein_id",
        "conditions": {"c1": ["c1_rep1", "c1_rep2", "c1_rep3", "c1_rep4", "c1_rep5"],}}
    d2_config = {
        "id_col": "peptide_id",
        "protein_id_col": "protein_id",
        "conditions": {"c2": ["c2_rep1", "c2_rep2", "c2_rep3", "c2_rep4", "c2_rep5"],}}
    d3_config = {
        "id_col": "peptide_id",
        "protein_id_col": "protein_id",
        "conditions": {"c3": ["c3_rep1", "c3_rep2", "c3_rep3", "c3_rep4", "c3_rep5"],}}
    d1 = PeptidesDataset.from_df(data_df, **d1_config)
    d2 = PeptidesDataset.from_df(data_df, **d2_config)
    d3 = PeptidesDataset.from_df(data_df, **d3_config)

    # action
    joined_dset = opl.concat([d1, d2, d3])

    # assertion
    assert isinstance(joined_dset, PeptidesDataset)
    conditions = joined_dset.condition_names
    for exp_cond in ["c1", "c2", "c3"]:
        assert exp_cond in conditions


def _protein_datasets() -> tuple[ProteinsDataset, ProteinsDataset]:
    data_df = pd.read_csv("tests/data/protein_dataset.tsv", sep="\t")
    d1 = ProteinsDataset.from_df(
        data_df, id_col="protein_id", conditions={"c1": [f"c1_rep{i}" for i in range(1, 6)]}
    )
    d2 = ProteinsDataset.from_df(
        data_df, id_col="protein_id", conditions={"c2": [f"c2_rep{i}" for i in range(1, 6)]}
    )
    return d1, d2


def _peptide_datasets() -> tuple[PeptidesDataset, PeptidesDataset]:
    data_df = pd.read_csv("tests/data/peptides_dataset.tsv", sep="\t")
    config = {"id_col": "peptide_id", "protein_id_col": "protein_id"}
    d1 = PeptidesDataset.from_df(data_df, conditions={"c1": [f"c1_rep{i}" for i in range(1, 6)]}, **config)
    d2 = PeptidesDataset.from_df(data_df, conditions={"c2": [f"c2_rep{i}" for i in range(1, 6)]}, **config)
    return d1, d2


DATASET_FACTORIES = pytest.mark.parametrize("make_datasets", [_protein_datasets, _peptide_datasets])


@DATASET_FACTORIES
def test_append_does_not_modify_inputs(make_datasets) -> None:
    """``append`` returns a new dataset and leaves both inputs unchanged."""
    d1, d2 = make_datasets()
    d1_table, d2_table = d1.to_table(), d2.to_table()

    joined = d1.append(d2, inplace=False)

    assert joined.condition_names == ["c1", "c2"]
    assert d1.condition_names == ["c1"]
    assert d2.condition_names == ["c2"]
    pd.testing.assert_frame_equal(d1.to_table(), d1_table)
    pd.testing.assert_frame_equal(d2.to_table(), d2_table)


@DATASET_FACTORIES
def test_append_result_is_independent_of_inputs(make_datasets) -> None:
    """The appended dataset does not share condition objects with its inputs."""
    d1, d2 = make_datasets()

    joined = d1.append(d2, inplace=False)

    input_conditions = [id(c) for c in [*d1._conditions, *d2._conditions]]
    assert all(id(c) not in input_conditions for c in joined._conditions)


@DATASET_FACTORIES
def test_append_skip_duplicates_omits_existing_conditions(make_datasets) -> None:
    """With ``skip_duplicates=True``, conditions that already exist are left out."""
    d1, d2 = make_datasets()
    d12 = d1.append(d2, inplace=False)

    joined = d12.append(d1, skip_duplicates=True, inplace=False)

    assert joined.condition_names == ["c1", "c2"]
    pd.testing.assert_frame_equal(joined.to_table(), d12.to_table())


@DATASET_FACTORIES
def test_append_duplicate_condition_raises_with_name(make_datasets) -> None:
    """Without ``skip_duplicates``, a duplicate condition raises an error naming it."""
    d1, _ = make_datasets()

    with pytest.raises(ValueError, match="'c1' already exists"):
        d1.append(d1, inplace=False)


def test_append_different_dataset_type_raises() -> None:
    proteins, _ = _protein_datasets()
    peptides, _ = _peptide_datasets()

    with pytest.raises(ValueError, match="should be of type"):
        proteins.append(peptides, inplace=False)  # type: ignore[arg-type]


@DATASET_FACTORIES
def test_concat_does_not_modify_inputs(make_datasets) -> None:
    d1, d2 = make_datasets()

    joined = opl.concat([d1, d2])

    assert joined.condition_names == ["c1", "c2"]
    assert d1.condition_names == ["c1"]
    assert d2.condition_names == ["c2"]


def test_concat_single_dataset_returns_copy() -> None:
    d1, _ = _protein_datasets()

    joined = opl.concat([d1])

    assert joined is not d1
    assert joined.condition_names == ["c1"]


def test_concat_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        opl.concat([])


@DATASET_FACTORIES
def test_append_default_warns_and_modifies_in_place(make_datasets) -> None:
    """Without ``inplace``, ``append`` keeps the old behaviour and warns about the change."""
    d1, d2 = make_datasets()

    with pytest.warns(FutureWarning, match="default of `inplace`"):
        joined = d1.append(d2)

    assert joined is d1
    assert d1.condition_names == ["c1", "c2"]
    assert d2.condition_names == ["c2"]


@DATASET_FACTORIES
def test_append_inplace_true_modifies_self_without_warning(make_datasets) -> None:
    d1, d2 = make_datasets()
    d2_table = d2.to_table()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        joined = d1.append(d2, inplace=True)

    assert joined is d1
    assert d1.condition_names == ["c1", "c2"]
    pd.testing.assert_frame_equal(d2.to_table(), d2_table)
    assert all(id(c) not in [id(c2) for c2 in d2._conditions] for c in d1._conditions)


@DATASET_FACTORIES
def test_append_inplace_false_does_not_warn(make_datasets) -> None:
    d1, d2 = make_datasets()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d1.append(d2, inplace=False)


@DATASET_FACTORIES
def test_failed_inplace_append_leaves_dataset_unchanged(make_datasets) -> None:
    """A rejected duplicate must not leave a partially appended dataset behind."""
    d1, d2 = make_datasets()
    # Conditions c2, c1: c2 is new, but c1 already exists in d1.
    d21 = d2.append(d1, inplace=False)

    with pytest.raises(ValueError, match="'c1' already exists"):
        d1.append(d21, inplace=True)

    assert d1.condition_names == ["c1"]


@DATASET_FACTORIES
def test_concat_does_not_warn(make_datasets) -> None:
    d1, d2 = make_datasets()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opl.concat([d1, d2])
