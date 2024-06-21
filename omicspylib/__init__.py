"""
Starting point for importing libraries main objects.
"""
from omicspylib.datasets.proteins import ProteinsDataset
from omicspylib.comparisons.pairs.statistical import PairwiseComparisonTTestFC
from omicspylib.comparisons.pairs.unique import PairwiseUniqueEntryComparison


__all__ = [
    'ProteinsDataset',
    'PairwiseComparisonTTestFC',
    'PairwiseUniqueEntryComparison'
]


__version__ = '0.0.2'
