"""
Starting point for importing libraries main objects.
"""
from omicspylib.analysis.clusters import HierarchicallyClusteredHeatmap
from omicspylib.analysis.pairs.frequency_based import PairwiseUniqueEntryComparison
from omicspylib.analysis.pairs.statistical import PairwiseComparisonTTestFC
from omicspylib.datasets.concat import concat
from omicspylib.datasets.peptides import PeptidesDataset
from omicspylib.datasets.proteins import ProteinsDataset
from omicspylib.go.goslim import go_to_goslim

__all__ = [
    'HierarchicallyClusteredHeatmap',
    'PairwiseComparisonTTestFC',
    'PairwiseUniqueEntryComparison',
    'PeptidesDataset',
    'ProteinsDataset',
    'concat',
    'go_to_goslim'
]


from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("omicspylib")
except PackageNotFoundError:
    __version__ = "0.0.0"
