import numpy as np
import pandas as pd
import pytest
from scipy.stats import trim_mean

from omicspylib import PeptidesDataset
from omicspylib.datasets.abc import NormMethod


def test_trimmed_mean_normalization_explicit() -> None:
    # Setup controlled dataset with extreme outliers
    data = {
        "peptide_id": [f"p{i}" for i in range(10)],
        "protein_id": ["prot1"] * 10,
        "ref_rep1": [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            1000.0,
        ],  # Outlier at end
        "targ_rep1": [12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 72.0, 82.0, 92.0, 102.0],
    }
    df = pd.DataFrame(data)
    config = {
        "id_col": "peptide_id",
        "conditions": {
            "c1": ["ref_rep1"],
            "c2": ["targ_rep1"],
        },
        "protein_id_col": "protein_id",
    }
    dataset = PeptidesDataset.from_df(df, **config)

    trim_fraction = 0.2  # 10% from lower, 10% from upper end
    norm_dataset = dataset.normalize(
        method="mean",
        ref_exp="ref_rep1",
        use_common_records=True,
        trim_fraction=trim_fraction,
    )

    # Merge matching rows (in order after sorting by ref)
    merged_df = df[["ref_rep1", "targ_rep1"]].sort_values("ref_rep1")
    expected_ref_mean = trim_mean(merged_df["ref_rep1"], proportiontocut=trim_fraction)
    expected_targ_mean = trim_mean(
        merged_df["targ_rep1"], proportiontocut=trim_fraction
    )
    expected_shift = expected_targ_mean - expected_ref_mean

    norm_table = norm_dataset.to_table()
    actual_shift = np.array(df["targ_rep1"].values) - np.array(
        norm_table["targ_rep1"].values
    )

    np.testing.assert_allclose(actual_shift, expected_shift)


@pytest.mark.parametrize("trim_fraction", [0.0, 0.1, 0.2])
def test_normalization_with_trim_fraction_param(trim_fraction) -> None:
    data_df = pd.read_csv("tests/data/peptides_dataset.tsv", sep="\t")
    config = {
        "id_col": "peptide_id",
        "conditions": {
            "c1": ["c1_rep1", "c1_rep2", "c1_rep3", "c1_rep4", "c1_rep5"],
            "c2": ["c2_rep1", "c2_rep2", "c2_rep3", "c2_rep4", "c2_rep5"],
            "c3": ["c3_rep1", "c3_rep2", "c3_rep3", "c3_rep4", "c3_rep5"],
        },
        "protein_id_col": "protein_id",
    }
    dataset = PeptidesDataset.from_df(data_df, **config)

    norm_dataset = dataset.normalize(
        method="mean",
        ref_exp="c1_rep1",
        use_common_records=True,
        trim_fraction=trim_fraction,
    )
    assert norm_dataset is not None


@pytest.mark.parametrize(
    "method,ref_exp,ref_condition,use_common_records,atol",
    [
        ("mean", None, None, False, 0.1),
        ("mean", "c2_rep2", None, False, 0.1),
        ("mean", None, "c3", False, 0.1),
        ("mean", None, "c3", True, 0.9),
    ],
)
def test_normalization_method(
    method: NormMethod, ref_exp, ref_condition, use_common_records, atol
) -> None:
    # setup
    data_df = pd.read_csv("tests/data/peptides_dataset.tsv", sep="\t")
    config = {
        "id_col": "peptide_id",
        "conditions": {
            "c1": ["c1_rep1", "c1_rep2", "c1_rep3", "c1_rep4", "c1_rep5"],
            "c2": ["c2_rep1", "c2_rep2", "c2_rep3", "c2_rep4", "c2_rep5"],
            "c3": ["c3_rep1", "c3_rep2", "c3_rep3", "c3_rep4", "c3_rep5"],
        },
        "protein_id_col": "protein_id",
    }
    float_columns = data_df.select_dtypes(include=["float64"]).columns
    shift = [i / 4 for i in range(len(float_columns))]
    mask = data_df[float_columns] > 0
    data_df[float_columns] += shift * mask
    dataset = PeptidesDataset.from_df(data_df, **config)

    col_means_before = dataset.mean(axis=0)

    # action
    norm_dataset = dataset.normalize(
        method=method,
        ref_exp=ref_exp,
        ref_condition=ref_condition,
        use_common_records=use_common_records,
    )
    col_means_after = norm_dataset.mean(axis=0)

    # assertion
    global_mean_before = np.mean(col_means_before)
    assert not np.any(np.isclose(global_mean_before, col_means_before, atol=0.1))
    global_mean_after = np.mean(col_means_after)
    assert np.all(np.isclose(global_mean_after, col_means_after, atol=atol))
