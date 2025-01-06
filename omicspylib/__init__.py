"""
Starting point for importing libraries main objects.
"""
from omicspylib.datasets.proteins import ProteinsDataset
from omicspylib.datasets.peptides import PeptidesDataset
from omicspylib.datasets.concat import concat
from omicspylib.analysis.pairs.statistical import PairwiseComparisonTTestFC
from omicspylib.analysis.pairs.frequency_based import PairwiseUniqueEntryComparison
from omicspylib.analysis.clusters import HierarchicallyClusteredHeatmap
from omicspylib.go.goslim import go_to_goslim

__all__ = [
    'concat',
    'HierarchicallyClusteredHeatmap',
    'PairwiseComparisonTTestFC',
    'PairwiseUniqueEntryComparison',
    'PeptidesDataset',
    'ProteinsDataset',
    'go_to_goslim'
]


__version__ = '0.1.0'
