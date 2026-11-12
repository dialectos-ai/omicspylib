import numpy as np
import pandas as pd

from omicspylib.datasets.proteins import ProteinsDataset


class PairwiseUniqueEntryComparison:
    """
    Select unique and common entries between two conditions,
    based on fixed frequency thresholds.
    """
    def __init__(self,
                 dataset: ProteinsDataset,
                 condition_a: str,
                 condition_b: str) -> None:
        """
        Parameters
        ----------
        dataset: ProteinsDataset
            The dataset to analyze.
        condition_a: str
            Name of the first condition.
        condition_b: str
            Name of the second condition.
        """
        self._raw_dataset = dataset
        self._condition_a = condition_a
        self._condition_b = condition_b

    def eval(self,
             majority_grp_min_freq: int = 4,
             minority_grp_max_freq: int = 0,
             na_threshold: float = 0.0) -> pd.DataFrame:
        """
        Classify each record by the number of experiments it is found in,
        in each of the two conditions.

        Parameters
        ----------
        majority_grp_min_freq: int, optional
            Minimum number of experiments a record must be found in, in one
            condition, to be considered unique to that condition.
        minority_grp_max_freq: int, optional
            Maximum number of experiments a record can be found in, in the
            other condition, to be considered unique to the first one.
        na_threshold: float, optional
            Values equal or below this threshold are considered missing.

        Returns
        -------
        pd.DataFrame
            A data frame indexed by record id, with a single
            ``frequency_class`` column. Its value is one of:

            * ``unique <condition_a>``: found in at least
              ``majority_grp_min_freq`` experiments of condition A, and at
              most ``minority_grp_max_freq`` experiments of condition B.
            * ``unique <condition_b>``: the same, with the conditions swapped.
            * ``none``: found in at most ``minority_grp_max_freq`` experiments
              of both conditions.
            * ``common``: all other records. This includes records that do
              not pass the thresholds, e.g. found in few experiments of one
              condition and none of the other.
        """
        ftable = self._raw_dataset.frequency(
            na_threshold=na_threshold, join_method="outer")
        # Records missing from a condition (e.g., removed by ``filter``) get a
        # NaN frequency after the outer join. They were found in 0 experiments
        # of that condition, and NaN would fail every comparison below.
        ftable = ftable.fillna(0)

        unique_grp_a = (ftable[f"frequency_{self._condition_a}"] >= majority_grp_min_freq) & \
                       (ftable[f"frequency_{self._condition_b}"] <= minority_grp_max_freq)

        unique_grp_b = (ftable[f"frequency_{self._condition_a}"] <= minority_grp_max_freq) & \
                       (ftable[f"frequency_{self._condition_b}"] >= majority_grp_min_freq)

        both_missing = (ftable[f"frequency_{self._condition_a}"] <= minority_grp_max_freq) & \
                       (ftable[f"frequency_{self._condition_b}"] <= minority_grp_max_freq)

        # includes non-significant cases with low frequency
        common = ~unique_grp_a & ~unique_grp_b & ~both_missing

        freq_class = np.empty_like(common, dtype=object)
        freq_class[unique_grp_a] = f"unique {self._condition_a}"
        freq_class[unique_grp_b] = f"unique {self._condition_b}"
        freq_class[common] = "common"
        freq_class[both_missing] = "none"

        return pd.DataFrame(data={"frequency_class": freq_class}, index=ftable.index)
