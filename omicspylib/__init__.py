"""
Starting point for importing libraries main objects.
"""
from omicspylib.datasets.proteins import ProteinsDataset
from omicspylib.datasets.peptides import PeptidesDataset
from omicspylib.comparisons.pairs.statistical import PairwiseComparisonTTestFC
from omicspylib.comparisons.pairs.frequency_based import PairwiseUniqueEntryComparison


__all__ = [
    'PeptidesDataset',
    'ProteinsDataset',
    'PairwiseComparisonTTestFC',
    'PairwiseUniqueEntryComparison'
]


__version__ = '0.0.4'
