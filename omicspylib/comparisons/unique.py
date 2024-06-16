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
        raise NotImplementedError()
