"""Test pairwise correlation between experiments."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

from omicspylib import ProteinsDataset
from omicspylib.calculations import calc_pairwise_correlation

N_RECORDS = 200


@pytest.fixture
def replicate_dataset() -> ProteinsDataset:
    """
    Two conditions with three replicates each. Missing values are stored as 0,
    with a different pattern per experiment.
    """
    rng = np.random.default_rng(42)
    true_abundance = rng.lognormal(mean=20, sigma=2, size=N_RECORDS)
    columns = {}
    for cond in ("ctrl", "trt"):
        for rep in (1, 2, 3):
            values = true_abundance * rng.lognormal(0, 0.3, size=N_RECORDS)
            values[rng.choice(N_RECORDS, size=15 * rep, replace=False)] = 0.0
            columns[f"{cond}_rep{rep}"] = values
    data = pd.DataFrame({"protein_id": [f"P{i}" for i in range(N_RECORDS)], **columns})
    return ProteinsDataset.from_df(
        data,
        id_col="protein_id",
        conditions={
            "ctrl": ["ctrl_rep1", "ctrl_rep2", "ctrl_rep3"],
            "trt": ["trt_rep1", "trt_rep2", "trt_rep3"],
        },
    )


def _pair(result: pd.DataFrame, exp_a: str, exp_b: str) -> pd.Series:
    row = result[(result["experiment_a"] == exp_a) & (result["experiment_b"] == exp_b)]
    assert len(row) == 1
    return row.iloc[0]


def test_output_columns(replicate_dataset: ProteinsDataset) -> None:
    result = replicate_dataset.correlation()
    assert list(result.columns) == [
        "experiment_a", "experiment_b", "condition_a", "condition_b",
        "coefficient", "n_records",
    ]


@pytest.mark.parametrize(
    ("pairs", "n_pairs", "same_condition"),
    [
        ("within_conditions", 6, True),
        ("across_conditions", 9, False),
        ("all", 15, None),
    ],
)
def test_pair_selection(replicate_dataset, pairs, n_pairs, same_condition) -> None:
    result = replicate_dataset.correlation(pairs=pairs)

    assert len(result) == n_pairs
    assert not (result["experiment_a"] == result["experiment_b"]).any()
    if same_condition is not None:
        assert ((result["condition_a"] == result["condition_b"]) == same_condition).all()


def test_pairs_follow_experiment_order(replicate_dataset: ProteinsDataset) -> None:
    result = replicate_dataset.correlation()
    assert list(zip(result["experiment_a"], result["experiment_b"], strict=True))[:3] == [
        ("ctrl_rep1", "ctrl_rep2"),
        ("ctrl_rep1", "ctrl_rep3"),
        ("ctrl_rep2", "ctrl_rep3"),
    ]


def test_zeros_are_not_measurements(replicate_dataset: ProteinsDataset) -> None:
    table = replicate_dataset.to_table(mask_missing=True)
    complete = table[["ctrl_rep1", "ctrl_rep2"]].dropna()

    pair = _pair(replicate_dataset.correlation(), "ctrl_rep1", "ctrl_rep2")

    assert pair["n_records"] == len(complete)
    assert pair["n_records"] < N_RECORDS
    expected = spearmanr(complete["ctrl_rep1"].to_numpy(), complete["ctrl_rep2"].to_numpy()).statistic
    assert pair["coefficient"] == pytest.approx(expected)


def test_pearson_is_log_transformed_by_default(replicate_dataset: ProteinsDataset) -> None:
    complete = replicate_dataset.to_table(mask_missing=True)[["ctrl_rep1", "ctrl_rep2"]].dropna()

    default = _pair(replicate_dataset.correlation(method="pearson"), "ctrl_rep1", "ctrl_rep2")
    raw = _pair(
        replicate_dataset.correlation(method="pearson", log_transform=False),
        "ctrl_rep1", "ctrl_rep2",
    )

    log_values = np.log2(complete)
    assert default["coefficient"] == pytest.approx(
        pearsonr(log_values["ctrl_rep1"], log_values["ctrl_rep2"]).statistic
    )
    assert raw["coefficient"] == pytest.approx(
        pearsonr(complete["ctrl_rep1"], complete["ctrl_rep2"]).statistic
    )
    assert default["coefficient"] != pytest.approx(raw["coefficient"])


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_rank_methods_ignore_log_transform(replicate_dataset, method) -> None:
    raw = replicate_dataset.correlation(method=method, log_transform=False)
    logged = replicate_dataset.correlation(method=method, log_transform=True)
    pd.testing.assert_frame_equal(raw, logged)


def test_listwise_uses_the_same_records_for_all_pairs(replicate_dataset) -> None:
    pairwise = replicate_dataset.correlation(missing="pairwise")
    listwise = replicate_dataset.correlation(missing="listwise")

    n_complete = len(replicate_dataset.to_table(mask_missing=True).dropna())
    assert (listwise["n_records"] == n_complete).all()
    assert (pairwise["n_records"] > listwise["n_records"]).all()
    assert not np.allclose(pairwise["coefficient"], listwise["coefficient"])


def test_listwise_is_limited_to_experiments_of_the_selected_pairs() -> None:
    data = pd.DataFrame({
        "protein_id": [f"P{i}" for i in range(12)],
        "a_rep1": np.arange(1, 13, dtype=float),
        "a_rep2": np.arange(1, 13, dtype=float) * 2,
        "b_rep1": [0.0] * 6 + [1.0] * 6,  # missing in half the records
    })
    dataset = ProteinsDataset.from_df(
        data, id_col="protein_id",
        conditions={"a": ["a_rep1", "a_rep2"], "b": ["b_rep1"]},
    )

    # b has a single experiment, so it is not part of any within-condition pair
    result = dataset.correlation(missing="listwise", min_records=2)

    assert result["n_records"].tolist() == [12]


def test_pairs_below_min_records_get_nan(replicate_dataset: ProteinsDataset) -> None:
    result = replicate_dataset.correlation(min_records=N_RECORDS + 1)

    assert result["coefficient"].isna().all()
    assert (result["n_records"] > 0).all()


def test_na_threshold_masks_low_values(replicate_dataset: ProteinsDataset) -> None:
    threshold = float(np.median(replicate_dataset.to_table().to_numpy()))
    default = replicate_dataset.correlation()
    thresholded = replicate_dataset.correlation(na_threshold=threshold)
    assert (thresholded["n_records"] < default["n_records"]).all()


def test_warns_when_values_look_log_transformed(replicate_dataset) -> None:
    logged = replicate_dataset.log2_transform()

    with pytest.warns(UserWarning, match="log-transformed already"):
        logged.correlation(method="pearson")


def test_no_warning_when_log_transform_disabled(replicate_dataset, recwarn) -> None:
    replicate_dataset.log2_transform().correlation(method="pearson", log_transform=False)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_negative_na_threshold_with_log_transform_raises(replicate_dataset) -> None:
    with pytest.raises(ValueError, match="not positive"):
        replicate_dataset.correlation(method="pearson", na_threshold=-1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "cosine"}, "method"),
        ({"pairs": "some"}, "pairs"),
        ({"missing": "drop"}, "missing"),
        ({"min_records": 1}, "min_records"),
    ],
)
def test_invalid_arguments_raise(replicate_dataset, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        replicate_dataset.correlation(**kwargs)


def test_does_not_modify_the_dataset(replicate_dataset: ProteinsDataset) -> None:
    before = replicate_dataset.to_table()
    replicate_dataset.correlation(method="pearson", pairs="all", missing="listwise")
    pd.testing.assert_frame_equal(replicate_dataset.to_table(), before)


def test_single_condition_has_no_across_pairs(replicate_dataset) -> None:
    result = replicate_dataset.filter(conditions=["ctrl"]).correlation(pairs="across_conditions")
    assert result.empty
    assert "coefficient" in result.columns


def test_peptides_dataset_is_supported(peptides_dataset) -> None:
    result = peptides_dataset.correlation()
    assert len(result) == 3 * 10  # 3 conditions x C(5, 2) pairs


def test_calc_pairwise_correlation_on_a_data_frame() -> None:
    data = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, np.nan],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "z": [4.0, 3.0, 2.0, 1.0, 0.5],
    })

    result = calc_pairwise_correlation(data, pairs=[("x", "y"), ("y", "z")], min_records=3)

    assert result["coefficient"].tolist() == pytest.approx([1.0, -1.0])
    assert result["n_records"].tolist() == [4, 5]
