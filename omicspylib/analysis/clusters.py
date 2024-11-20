from typing import Tuple, Union, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from seaborn.matrix import ClusterGrid

LinkageMethod = Literal['single', 'complete', 'average', 'weighted', 'centroid',  'median', 'ward']
FillNAMethod = Literal['min', 'mean', 'median', 'drop']


class HierarchicallyClusteredHeatmap:
    """
    Given a tabular dataset performs hierarchical clustering
    plotted on a heatmap of the data.
    """
    def __init__(self, log_transform: bool = True, fillna_method: FillNAMethod = 'min',
                 na_shift_value: float = 0.2, min_frequency: int = 1, na_threshold: float = 0.0,
                 center_scale: bool = True):
        """
        Initializer method.

        Parameters
        ----------
        log_transform: bool
            By default, values will be log transformed before clustering.
            Set to ``False`` to skip this step.
        fillna_method: FillNAMethod
            How to handle missing values. Possible options include:
            * ``min``: Use min value.
            * ``mean``: Use mean value.
            * ``median``: Use median value.
            * ``drop``: Drop rows with missing values.
        na_shift_value: float
            You can shift the na-imputed values by a fixed number.
            For example, you can set ``fillna_method`` to ``min`` by
            and decrease by ``0.2`` units. Set to ``0.0`` to skip this step.
        min_frequency: float
            In cases with a significant number of missing values,
            you might choose to first filter the dataset based on the
            number of experiments with valid values.
        na_threshold: float or None, optional
            Values below or equal to this threshold are considered missing.
            It is used in to filter records based on the number of missing values.
        center_scale: bool
            By default, data will be centered and scaled before calculating
            the distances.
        """
        self._log_transform = log_transform
        self._fillna_method = fillna_method
        self._na_shift_value = na_shift_value
        self._min_frequency = min_frequency
        self._na_threshold = na_threshold
        self._center_scale = center_scale

    def eval(self,
             data: pd.DataFrame,
             n_row_clusters: Union[int, None] = 12,
             n_col_clusters: Union[int, None] = 3,
             sorted_cols: list = None,
             linkage_method: LinkageMethod = 'average',
             figsize: Tuple[int, int] = (10, 14),
             title: str = 'Clustering groups') -> Tuple[pd.DataFrame, ClusterGrid, list, list]:
        """
        Perform hierarchical clustering and plot a heatmap with the separated groups.

        Returns a filtered version of the provided dataset, a graph object,
        and a list of row and column groups.

        Parameters
        ----------
        data: pd.DataFrame
            A Pandas data frame with the values.  Only values are expected,
            without any additional columns.The row identifier should be set
            to the data frame index.
        n_row_clusters: int or None
            Number of row clusters to create. If set to ``None`` row
            clustering is skipped.
        n_col_clusters: int or None
            Number of column clusters to create. If set to ``None`` column
            clustering is skipped.
        sorted_cols: list or None
            You might choose to skip column clustering and provide a
            list of column names as you would like to see them in the
            output.
        linkage_method: LinkageMethod
            Linkage method. See ``scipy.cluster.hierarchy.linkage`` for
            available options.
        figsize: tuple
            Tuple specifying the shape of the returned image.
            If nothing is provided, a default size will be returned.
        title: str
            Title to be placed on top of the plot.

        Returns
        -------
        pd.DataFrame:
            The provided dataset filtered to the min frequency specified in the constructor.
        """
        if n_row_clusters == 0:
            n_row_clusters = None
        if n_col_clusters == 0:
            n_col_clusters = None

        if n_col_clusters is None and n_row_clusters is None:
            raise ValueError('Both number of clusters for rows and columns are None. '
                             'At least one should be provided.')
        # if user provides column order follow it -> column names must be exact.
        if sorted_cols is not None:
            data = data[sorted_cols].copy()

        filtered_data = self._filter_based_on_nan_frequency(data)
        heatmap_inputs = self._transform(filtered_data)
        heatmap_inputs = self._fillna(heatmap_inputs)
        if self._center_scale:
            heatmap_inputs = self._apply_center_scale(heatmap_inputs)

        col_groups, col_linkage, row_groups, row_linkage = \
            self._calculate_linkage_and_groups(
                heatmap_inputs, linkage_method, n_col_clusters, n_row_clusters)

        g = self._create_heatmap(
            data=heatmap_inputs,
            col_linkage=col_linkage,
            col_groups=col_groups,
            row_linkage=row_linkage,
            row_groups=row_groups,
            figsize=figsize,
            linkage_method=linkage_method,
            title=title)

        return filtered_data, g, row_groups, col_groups

    @staticmethod
    def _calculate_linkage_and_groups(data: pd.DataFrame, linkage_method: LinkageMethod, n_col_clusters: int, n_row_clusters: int):
        # calculate linkage and cut the tree
        row_linkage = None
        row_groups = None
        col_linkage = None
        col_groups = None

        if n_row_clusters is not None:
            row_linkage = scipy.cluster \
                .hierarchy.linkage(
                scipy.spatial.distance
                .pdist(data.values),
                method=linkage_method)
            row_trees = scipy.cluster.hierarchy \
                .cut_tree(row_linkage, n_clusters=n_row_clusters)
            row_groups = [int(t) for t in row_trees.reshape(-1)]
        if n_col_clusters is not None:
            col_linkage = scipy.cluster.hierarchy.linkage(
                scipy.spatial.distance.pdist(data.values.T),
                method=linkage_method)
            col_trees = scipy.cluster.hierarchy \
                .cut_tree(col_linkage, n_clusters=n_col_clusters)
            col_groups = [int(i) for i in col_trees.reshape(-1)]
        return col_groups, col_linkage, row_groups, row_linkage

    def _transform(self, data):
        """
        Log transform protein intensities, impute missing values,
        center and scale, so that values are ready for plotting.
        """
        data = data.copy()
        if self._log_transform:
            data = np.log10(data + 1)
        return data

    def _fillna(self, data):
        """
        Log transform protein intensities, impute missing values,
        center and scale, so that values are ready for plotting.
        """
        fill_na_value = 0
        if self._fillna_method == 'min':
            fill_na_value = data.min().min() - self._na_shift_value
        elif self._fillna_method == 'mean':
            fill_na_value = data.mean().mean() - self._na_shift_value
        elif self._fillna_method == 'median':
            fill_na_value = data.median().median() - self._na_shift_value

        return data.fillna(fill_na_value)

    @staticmethod
    def _apply_center_scale(data):
        """
        Log transform protein intensities, impute missing values,
        center and scale, so that values are ready for plotting.
        """
        row_mean = data.mean(axis=1).values
        row_std = data.std(axis=1).values
        return (data - row_mean.reshape(-1, 1)) / row_std.reshape(-1, 1)

    def _filter_based_on_nan_frequency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the data based on the number of nan values they have.
        The Reason for that is that unique proteins, even if they are of low
        abundance and randomly detected, get a color of high-abundant
        protein during the plotting, due to their uniqueness in the dataset.

        Parameters
        ----------
        data: pd.DataFrame
            Working dataset.

        Returns
        -------
        pd.DataFrame
            Dataset filtered to the rows with at least ``self._min_frequency``
            valid values.
        """
        if self._na_threshold is not None:
            is_na = data <= self._na_threshold
            data[is_na] = np.nan

        if self._min_frequency > 1:
            n_cols = data.shape[1] - 1
            non_nan_freq = n_cols - data.isna().sum(axis=1).values
            keep_rows = non_nan_freq >= self._min_frequency
            data = data.loc[keep_rows, :].copy()

        if self._fillna_method == 'drop':
            data = data.dropna()

        return data

    @staticmethod
    def _create_heatmap(data,
                        row_linkage,
                        row_groups,
                        col_linkage,
                        col_groups,
                        linkage_method: LinkageMethod = 'average',
                        title: str = '',
                        **clustermap_kwargs):
        # base setup
        tree_cmap = plt.get_cmap('tab20')

        if row_linkage is not None:
            row_grp_colors = [tree_cmap(grp_idx) for grp_idx in row_groups]
            plot_row_cluster = True
        else:
            row_linkage = None
            row_grp_colors = None
            plot_row_cluster = False
        if col_linkage is not None:
            col_grp_colors = [tree_cmap(grp_idx) for grp_idx in col_groups]
            plot_col_cluster = True
        else:
            col_linkage = None
            col_grp_colors = None
            plot_col_cluster = False

        # plot heatmap using pre-calculated tree colors
        g = sns.clustermap(data,
                           row_linkage=row_linkage,
                           row_cluster=plot_row_cluster,
                           col_linkage=col_linkage,
                           col_cluster=plot_col_cluster,
                           row_colors=row_grp_colors,
                           col_colors=col_grp_colors,
                           method=linkage_method,
                           cmap="vlag",
                           # cbar_pos=(0, 2, 0.01, 0.1),  # tuple of (left, bottom, width, height)
                           **clustermap_kwargs)

        # configure legend
        if row_groups is not None:
            row_patches = [mpatches.Patch(color=tree_cmap(grp_idx), label=f'Grp {grp_idx}') for grp_idx in sorted(list(set(row_groups)))]
            g.ax_row_dendrogram.legend(handles=row_patches, loc='lower left', title='Row groups')
        if col_groups is not None:
            col_patches = [mpatches.Patch(color=tree_cmap(grp_idx), label=f'Grp {grp_idx}') for grp_idx in sorted(list(set(col_groups)))]
            g.ax_col_dendrogram.legend(handles=col_patches, loc='lower left', title='Col groups')

        g.ax_cbar.set_aspect(2)  # decrease width of colorbar
        g.fig.suptitle(title)
        plt.tight_layout()

        return g
