"""Test Bland-Altman agreement and explicit pair selection."""
import numpy as np
import pandas as pd
import pytest

from omicspylib import ProteinsDataset
from omicspylib.calculations import calc_bland_altman


@pytest.fixture
def offset_dataset() -> ProteinsDataset:
    """rep2 is exactly 2x rep1 (log2 bias of 1); rep3 is noisy; one value is missing."""
    rng = np.random.default_rng(1)
    rep1 = rng.lognormal(20, 2, size=100)
    rep3 = rep1 * rng.lognormal(0, 0.5, size=100)
    rep1_with_missing = rep1.copy()
    rep1_with_missing[0] = 0.0
    data = pd.DataFrame({
        "protein_id": [f"P{i}" for i in range(100)],
        "a_rep1": rep1_with_missing,
        "a_rep2": rep1 * 2,
        "a_rep3": rep3,
        "b_rep1": rep1,
    })
    return ProteinsDataset.from_df(
        data, id_col="protein_id",
        conditions={"a": ["a_rep1", "a_rep2", "a_rep3"], "b": ["b_rep1"]},
    )


def _pair(result: pd.DataFrame, exp_a: str, exp_b: str) -> pd.Series:
    row = result[(result["experiment_a"] == exp_a) & (result["experiment_b"] == exp_b)]
    assert len(row) == 1
    return row.iloc[0]


def test_output_columns(offset_dataset: ProteinsDataset) -> None:
    assert list(offset_dataset.bland_altman().columns) == [
        "experiment_a", "experiment_b", "condition_a", "condition_b",
        "bias", "sd", "loa_lower", "loa_upper", "n_records",
    ]


def test_bias_is_the_log2_fold_change(offset_dataset: ProteinsDataset) -> None:
    pair = _pair(offset_dataset.bland_altman(), "a_rep1", "a_rep2")

    assert pair["bias"] == pytest.approx(-1.0)
    assert pair["sd"] == pytest.approx(0.0, abs=1e-12)
    assert pair["n_records"] == 99  # the stored 0 is not a measurement


def test_limits_of_agreement(offset_dataset: ProteinsDataset) -> None:
    table = pd.DataFrame(np.log2(offset_dataset.to_table(mask_missing=True)))
    diff = (table["a_rep1"] - table["a_rep3"]).dropna()

    pair = _pair(offset_dataset.bland_altman(), "a_rep1", "a_rep3")

    assert pair["bias"] == pytest.approx(diff.mean())
    assert pair["sd"] == pytest.approx(diff.std(ddof=1))
    assert pair["loa_lower"] == pytest.approx(diff.mean() - 1.96 * diff.std(ddof=1))
    assert pair["loa_upper"] == pytest.approx(diff.mean() + 1.96 * diff.std(ddof=1))


def test_without_log_transform_uses_raw_differences(offset_dataset) -> None:
    table = offset_dataset.to_table(mask_missing=True)
    diff = (table["a_rep1"] - table["a_rep2"]).dropna()

    pair = _pair(offset_dataset.bland_altman(log_transform=False), "a_rep1", "a_rep2")

    assert pair["bias"] == pytest.approx(diff.mean())


def test_pairs_below_min_records_get_nan(offset_dataset: ProteinsDataset) -> None:
    result = offset_dataset.bland_altman(min_records=1000)
    assert result[["bias", "sd", "loa_lower", "loa_upper"]].isna().all().all()


def test_explicit_pairs(offset_dataset: ProteinsDataset) -> None:
    pairs = [("a_rep2", "b_rep1"), ("a_rep1", "a_rep3")]

    result = offset_dataset.bland_altman(pairs=pairs)
    corr = offset_dataset.correlation(pairs=pairs)

    for df in (result, corr):
        assert list(zip(df["experiment_a"], df["experiment_b"], strict=True)) == pairs
        assert df["condition_a"].tolist() == ["a", "a"]
        assert df["condition_b"].tolist() == ["b", "a"]
    assert _pair(result, "a_rep2", "b_rep1")["bias"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("pairs", "match"),
    [
        ([("a_rep1", "unknown")], "not found"),
        ([("a_rep1", "a_rep1")], "two different"),
        ([("a_rep1",)], "two experiment names"),
    ],
)
def test_invalid_explicit_pairs_raise(offset_dataset, pairs, match) -> None:
    with pytest.raises(ValueError, match=match):
        offset_dataset.bland_altman(pairs=pairs)


def test_calc_bland_altman_on_a_data_frame() -> None:
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "y": [0.0, 1.0, 2.0, 5.0]})

    result = calc_bland_altman(data, pairs=[("x", "y")], min_records=2)

    assert result.loc[0, "bias"] == pytest.approx(1.0)
    assert result.loc[0, "sd"] == pytest.approx(0.0)
    assert result.loc[0, "n_records"] == 3
