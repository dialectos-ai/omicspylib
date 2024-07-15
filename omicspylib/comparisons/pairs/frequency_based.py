import numpy as np
import pandas as pd

from omicspylib import ProteinsDataset


class PairwiseUniqueEntryComparison:
    """
    Select unique and common entries between two conditions,
    based on fixed frequency thresholds.
    """
    def __init__(self,
                 dataset: ProteinsDataset,
                 condition_a: str,
                 condition_b: str):
        self._raw_dataset = dataset
        self._condition_a = condition_a
        self._condition_b = condition_b

    def compare(self,
                majority_grp_min_freq: int = 4,
                minority_grp_max_freq: int = 0,
                na_threshold: float = 0.0):
        ftable = self._raw_dataset.frequency(
            na_threshold=na_threshold, join_method='outer')

        unique_grp_a = (ftable[f'frequency_{self._condition_a}'] >= majority_grp_min_freq) & \
                       (ftable[f'frequency_{self._condition_b}'] <= minority_grp_max_freq)

        unique_grp_b = (ftable[f'frequency_{self._condition_a}'] <= minority_grp_max_freq) & \
                       (ftable[f'frequency_{self._condition_b}'] >= majority_grp_min_freq)

        both_missing = (ftable[f'frequency_{self._condition_a}'] <= minority_grp_max_freq) & \
                       (ftable[f'frequency_{self._condition_b}'] <= minority_grp_max_freq)

        # includes non-significant cases with low frequency
        common = ~unique_grp_a & ~unique_grp_b & ~both_missing

        freq_class = np.empty_like(common, dtype=object)
        freq_class[unique_grp_a] = f'unique {self._condition_a}'
        freq_class[unique_grp_b] = f'unique {self._condition_b}'
        freq_class[common] = 'common'
        freq_class[both_missing] = 'none'

        return pd.DataFrame(data={'frequency_class': freq_class}, index=ftable.index)
