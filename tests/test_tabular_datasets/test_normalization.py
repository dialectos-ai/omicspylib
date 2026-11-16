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
        ref_experiment="ref_rep1",
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


def test_trimmed_mean_normalization_all_records() -> None:
    # Records are not shared, so only the all-records path applies.
    df = pd.DataFrame(
        {
            "peptide_id": [f"p{i}" for i in range(10)],
            "protein_id": ["prot1"] * 10,
            "ref_rep1": [10.0, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
            "targ_rep1": [12.0, 22, 32, 42, 52, 62, 72, 82, 92, 0],  # p9 missing
        }
    )
    config = {
        "id_col": "peptide_id",
        "conditions": {"c1": ["ref_rep1"], "c2": ["targ_rep1"]},
        "protein_id_col": "protein_id",
    }
    dataset = PeptidesDataset.from_df(df, **config)

    trim_fraction = 0.2
    norm_dataset = dataset.normalize(
        method="mean",
        ref_experiment="ref_rep1",
        use_common_records=False,
        trim_fraction=trim_fraction,
    )

    # 10 ref values -> drop 1 per end; 9 targ values -> int(0.9) = 0, no trim
    expected_ref_mean = np.mean([20.0, 30, 40, 50, 60, 70, 80, 90])
    expected_targ_mean = np.mean([12.0, 22, 32, 42, 52, 62, 72, 82, 92])
    expected_shift = expected_targ_mean - expected_ref_mean

    norm_table = norm_dataset.to_table()
    present = df["targ_rep1"] > 0
    actual_shift = (
        df.loc[present, "targ_rep1"].to_numpy()
        - norm_table.loc[df.loc[present, "peptide_id"], "targ_rep1"].to_numpy()
    )
    np.testing.assert_allclose(actual_shift, expected_shift)
    np.testing.assert_allclose(norm_table["ref_rep1"], df["ref_rep1"])


@pytest.mark.parametrize("use_common_records", [True, False])
def test_median_normalization_explicit(use_common_records) -> None:
    # p9 is missing in the target, so the two paths use different records.
    df = pd.DataFrame(
        {
            "peptide_id": [f"p{i}" for i in range(10)],
            "protein_id": ["prot1"] * 10,
            "ref_rep1": [10.0, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
            "targ_rep1": [12.0, 22, 32, 42, 52, 62, 72, 82, 500, 0],
        }
    )
    config = {
        "id_col": "peptide_id",
        "conditions": {"c1": ["ref_rep1"], "c2": ["targ_rep1"]},
        "protein_id_col": "protein_id",
    }
    dataset = PeptidesDataset.from_df(df, **config)

    norm_table = dataset.normalize(
        method="median", ref_experiment="ref_rep1", use_common_records=use_common_records
    ).to_table()

    present = df["targ_rep1"] > 0
    targ = df.loc[present, "targ_rep1"]
    ref = df.loc[present, "ref_rep1"] if use_common_records else df["ref_rep1"]
    expected_shift = np.median(targ) - np.median(ref)

    actual_shift = (
        targ.to_numpy()
        - norm_table.loc[df.loc[present, "peptide_id"], "targ_rep1"].to_numpy()
    )
    np.testing.assert_allclose(actual_shift, expected_shift)
    np.testing.assert_allclose(norm_table["ref_rep1"], df["ref_rep1"])
    # missing values stay missing
    assert norm_table.loc["p9", "targ_rep1"] == 0


def test_invalid_norm_method_raises() -> None:
    data_df = pd.read_csv("tests/data/peptides_dataset.tsv", sep="\t")
    dataset = PeptidesDataset.from_df(
        data_df,
        id_col="peptide_id",
        conditions={"c1": ["c1_rep1", "c1_rep2"], "c2": ["c2_rep1", "c2_rep2"]},
        protein_id_col="protein_id",
    )
    with pytest.raises(NotImplementedError):
        dataset.normalize(method="quantile")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["mean", "median"])
@pytest.mark.parametrize("use_common_records", [True, False])
def test_reference_condition_replicates_are_normalized(
    use_common_records, method
) -> None:
    df = pd.DataFrame(
        {
            "peptide_id": [f"p{i}" for i in range(5)],
            "protein_id": ["prot1"] * 5,
            "ref_rep1": [10.0, 20, 30, 40, 50],
            "ref_rep2": [15.0, 25, 35, 45, 55],  # same condition, +5
            "targ_rep1": [12.0, 22, 32, 42, 52],
        }
    )
    config = {
        "id_col": "peptide_id",
        "conditions": {"c1": ["ref_rep1", "ref_rep2"], "c2": ["targ_rep1"]},
        "protein_id_col": "protein_id",
    }
    dataset = PeptidesDataset.from_df(df, **config)

    norm_table = dataset.normalize(
        method=method, ref_experiment="ref_rep1", use_common_records=use_common_records
    ).to_table()

    np.testing.assert_allclose(norm_table["ref_rep2"], df["ref_rep1"])
    np.testing.assert_allclose(norm_table["targ_rep1"], df["ref_rep1"])


def test_invalid_trim_fraction_raises() -> None:
    data_df = pd.read_csv("tests/data/peptides_dataset.tsv", sep="\t")
    dataset = PeptidesDataset.from_df(
        data_df,
        id_col="peptide_id",
        conditions={"c1": ["c1_rep1", "c1_rep2"], "c2": ["c2_rep1", "c2_rep2"]},
        protein_id_col="protein_id",
    )
    with pytest.raises(ValueError, match="trim_fraction must be in the range"):
        dataset.normalize(method="mean", trim_fraction=1.0)


@pytest.mark.parametrize("use_common_records", [True, False])
@pytest.mark.parametrize("trim_fraction", [0.0, 0.1, 0.2])
def test_normalization_with_trim_fraction_param(
    trim_fraction, use_common_records
) -> None:
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
        ref_experiment="c1_rep1",
        use_common_records=use_common_records,
        trim_fraction=trim_fraction,
    )
    assert norm_dataset is not None


@pytest.mark.parametrize(
    ("method", "ref_experiment", "ref_condition", "use_common_records", "atol"),
    [
        ("mean", None, None, False, 0.1),
        ("mean", "c2_rep2", None, False, 0.1),
        ("mean", None, "c3", False, 0.1),
        ("mean", None, "c3", True, 0.9),
        ("median", None, None, False, 0.2),
        ("median", "c2_rep2", None, False, 0.2),
        ("median", None, "c3", True, 0.9),
    ],
)
def test_normalization_method(
    method: NormMethod, ref_experiment, ref_condition, use_common_records, atol
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
        ref_experiment=ref_experiment,
        ref_condition=ref_condition,
        use_common_records=use_common_records,
    )
    col_means_after = norm_dataset.mean(axis=0)

    # assertion
    global_mean_before = np.mean(col_means_before)
    assert not np.any(np.isclose(global_mean_before, col_means_before, atol=0.1))
    global_mean_after = np.mean(col_means_after)
    assert np.all(np.isclose(global_mean_after, col_means_after, atol=atol))


def _within_conditions_dataset() -> tuple[pd.DataFrame, PeptidesDataset]:
    # c1 replicates differ by +2 / +4; c2 sits ~100 above c1 (kept).
    df = pd.DataFrame(
        {
            "peptide_id": [f"p{i}" for i in range(5)],
            "protein_id": ["prot1"] * 5,
            "c1_rep1": [10.0, 20, 30, 40, 50],
            "c1_rep2": [12.0, 22, 32, 42, 52],
            "c1_rep3": [14.0, 24, 34, 44, 0],  # p4 missing
            "c2_rep1": [110.0, 120, 130, 140, 150],
            "c2_rep2": [120.0, 130, 140, 150, 160],
        }
    )
    dataset = PeptidesDataset.from_df(
        df,
        id_col="peptide_id",
        conditions={
            "c1": ["c1_rep1", "c1_rep2", "c1_rep3"],
            "c2": ["c2_rep1", "c2_rep2"],
        },
        protein_id_col="protein_id",
    )
    return df, dataset


@pytest.mark.parametrize("method", ["mean", "median"])
def test_within_conditions_normalizes_to_condition_mean(method: NormMethod) -> None:
    _, dataset = _within_conditions_dataset()

    norm = dataset.normalize(
        method=method, within_conditions=True, use_common_records=True
    ).to_table()

    # common records of c1 are p0..p3: centers 25, 27, 29 -> target 27
    np.testing.assert_allclose(norm.loc["p0":"p3", "c1_rep1"], [12.0, 22, 32, 42])
    np.testing.assert_allclose(norm.loc["p0":"p3", "c1_rep2"], [12.0, 22, 32, 42])
    np.testing.assert_allclose(norm.loc["p0":"p3", "c1_rep3"], [12.0, 22, 32, 42])
    assert norm.loc["p4", "c1_rep3"] == 0
    # c2: centers 130, 140 -> target 135; the c1/c2 offset is kept
    np.testing.assert_allclose(norm["c2_rep1"], [115.0, 125, 135, 145, 155])
    np.testing.assert_allclose(norm["c2_rep2"], [115.0, 125, 135, 145, 155])


def test_within_conditions_all_records_keeps_condition_level() -> None:
    df, dataset = _within_conditions_dataset()

    norm = dataset.normalize(method="mean", within_conditions=True).to_table()

    exps = ["c1_rep1", "c1_rep2", "c1_rep3"]
    centers_before = [df.loc[df[e] > 0, e].mean() for e in exps]
    centers_after = [norm.loc[norm[e] > 0, e].mean() for e in exps]
    np.testing.assert_allclose(centers_after, np.mean(centers_before))


@pytest.mark.parametrize("use_common_records", [True, False])
def test_within_conditions_reference_per_condition(use_common_records) -> None:
    df, dataset = _within_conditions_dataset()

    norm = dataset.normalize(
        method="median",
        within_conditions=True,
        ref_experiment=["c1_rep2", "c2_rep1"],
        use_common_records=use_common_records,
    ).to_table()

    # references are unchanged, replicates move onto them
    np.testing.assert_allclose(norm["c1_rep2"], df["c1_rep2"])
    np.testing.assert_allclose(norm["c1_rep1"], df["c1_rep2"])
    np.testing.assert_allclose(norm["c2_rep1"], df["c2_rep1"])
    np.testing.assert_allclose(norm["c2_rep2"], df["c2_rep1"])
    if use_common_records:
        np.testing.assert_allclose(
            norm.loc["p0":"p3", "c1_rep3"], df.loc[0:3, "c1_rep2"]
        )


def test_within_conditions_partial_references() -> None:
    df, dataset = _within_conditions_dataset()

    norm = dataset.normalize(
        method="mean", within_conditions=True, ref_experiment="c2_rep2"
    ).to_table()

    # c2 goes to its reference, c1 (no reference) to its mean
    np.testing.assert_allclose(norm["c2_rep1"], df["c2_rep2"])
    np.testing.assert_allclose(norm["c2_rep2"], df["c2_rep2"])
    c1_centers = [norm.loc[norm[e] > 0, e].mean() for e in ["c1_rep1", "c1_rep2"]]
    assert c1_centers[0] == pytest.approx(c1_centers[1])


def test_within_conditions_invalid_arguments_raise() -> None:
    _, dataset = _within_conditions_dataset()

    with pytest.raises(ValueError, match="ref_condition cannot be used"):
        dataset.normalize(method="mean", within_conditions=True, ref_condition="c1")
    with pytest.raises(ValueError, match="more than one reference"):
        dataset.normalize(
            method="mean", within_conditions=True, ref_experiment=["c1_rep1", "c1_rep2"]
        )
    with pytest.raises(ValueError, match="not found"):
        dataset.normalize(method="mean", within_conditions=True, ref_experiment=["x"])
    with pytest.raises(TypeError, match="within_conditions=True"):
        dataset.normalize(method="mean", ref_experiment=["c1_rep1", "c2_rep1"])
