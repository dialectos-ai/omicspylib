import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from seaborn.matrix import ClusterGrid
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

LinkageMethod = Literal[
    "single", "complete", "average", "weighted", "centroid", "median", "ward"
]
FillNAMethod = Literal["min", "mean", "median", "drop"]


@dataclass
class HCHeatmapData:
    """
    A data class to encapsulate the results of hierarchical clustering
    and heatmap plotting.

    Attributes
    ----------
    filtered_data : pd.DataFrame
        The dataset filtered to the minimum frequency specified in the constructor.
    g : ClusterGrid
        The plot object representing the heatmap.
    row_groups : list or None
        A list of row groups. Will be ``None`` if no row clustering is performed.
    col_groups : list or None
        A list of column groups. Will be ``None`` if no column clustering is performed.
    heatmap_inputs : pd.DataFrame
        The DataFrame with the inputs passed to the heatmap function.
    row_silhouette_score: float or None
        Silhouette score for rows, if applicable and if it can be calculated.
    col_silhouette_score: float or None
        Silhouette score for columns, if applicable and if it can be calculated.
    row_linkage: np.ndarray or None
        Linkage matrix of the row tree, as returned by
        ``scipy.cluster.hierarchy.linkage``. ``None`` if no row clustering
        is performed.
    col_linkage: np.ndarray or None
        Linkage matrix of the column tree. ``None`` if no column clustering
        is performed.
    """

    filtered_data: pd.DataFrame
    g: "ClusterGrid"
    heatmap_inputs: pd.DataFrame
    row_groups: list[int] | None = None
    col_groups: list[int] | None = None
    row_silhouette_score: float | None = None
    col_silhouette_score: float | None = None
    row_linkage: np.ndarray | None = None
    col_linkage: np.ndarray | None = None


class HierarchicallyClusteredHeatmap:
    """
    Given a tabular dataset performs hierarchical clustering
    plotted on a heatmap of the data.
    """

    def __init__(
        self,
        log_transform: bool = True,
        fillna_method: FillNAMethod = "min",
        na_shift_value: float = 0.2,
        min_frequency: int = 1,
        na_threshold: float = 0.0,
        center_scale: bool = True,
        linkage_method: LinkageMethod = "average",
        n_row_clusters: int | None = 12,
        n_col_clusters: int | None = 3,
    ) -> None:
        """
        Initializer method.

        Parameters
        ----------
        log_transform: bool
            By default, values are transformed with ``log10(x + 1)`` before
            clustering. Set to ``False`` to skip this step.
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
        min_frequency: int
            Keep only rows with at least this number of valid (non-missing)
            values. Useful in datasets with many missing values.
        na_threshold: float or None, optional
            Values below or equal to this threshold are considered missing.
            It is used in to filter records based on the number of missing values.
        center_scale: bool
            By default, data will be centered and scaled before calculating
            the distances.
        linkage_method: LinkageMethod
            Linkage method. See ``scipy.cluster.hierarchy.linkage`` for
            available options.
        n_row_clusters: int or None
            Number of row clusters to create. If set to ``None`` row
            clustering is skipped.
        n_col_clusters: int or None
            Number of column clusters to create. If set to ``None`` column
            clustering is skipped.
        """
        self._log_transform = log_transform
        self._fillna_method = fillna_method
        self._na_shift_value = na_shift_value
        self._min_frequency = min_frequency
        self._na_threshold = na_threshold
        self._center_scale = center_scale
        self._linkage_method: LinkageMethod = linkage_method
        if n_row_clusters == 0:
            self._n_row_clusters = None
        else:
            self._n_row_clusters = n_row_clusters
        if n_col_clusters == 0:
            self._n_col_clusters = None
        else:
            self._n_col_clusters = n_col_clusters

        if self._n_col_clusters is None and self._n_row_clusters is None:
            raise ValueError(
                "Both number of clusters for rows and columns are None. "
                "At least one should be provided."
            )

    def eval(
        self,
        data: pd.DataFrame,
        sorted_cols: list | None = None,
        figsize: tuple[int, int] = (10, 14),
        title: str = "Clustering groups",
        row_annotations: dict[str, pd.Series] | None = None,
        row_annotation_palettes: dict[str, dict] | None = None,
    ) -> HCHeatmapData:
        """
        Perform hierarchical clustering and plot a heatmap with the separated groups.

        Parameters
        ----------
        data: pd.DataFrame
            A Pandas data frame with the values.  Only values are expected,
            without any additional columns.The row identifier should be set
            to the data frame index.
        sorted_cols: list or None
            You might choose to skip column clustering and provide a
            list of column names as you would like to see them in the
            output.
        figsize: tuple
            Tuple specifying the shape of the returned image.
            If nothing is provided, a default size will be returned.
        title: str
            Title to be placed on top of the plot.
        row_annotations: dict or None
            Extra colour tracks drawn beside the rows, one per entry. Keys
            are the track names, values are ``pd.Series`` with a
            categorical label per record, indexed like ``data``. Labels
            are matched on the index, so rows removed by filtering are
            ignored; rows without a label are left blank. The clustering
            does not use these labels, so they are a way to check whether
            the row clusters mean anything.
        row_annotation_palettes: dict or None
            Optional colours per track, as ``{track name: {label: colour}}``.
            Tracks or labels without a colour get one from a default
            categorical palette.

        Returns
        -------
        HCHeatmapData
            The filtered data, the plot, the row and column groups and
            their silhouette scores. Row and column groups are listed in
            the same order as the rows and columns of ``filtered_data``.
        """
        data = copy.deepcopy(data)
        # if user provides column order follow it -> column names must be exact.
        if sorted_cols is not None:
            data = data[sorted_cols].copy()

        filtered_data = self._filter_based_on_nan_frequency(data)
        heatmap_inputs = self._transform(filtered_data)
        heatmap_inputs = self._fillna(heatmap_inputs)
        if self._center_scale:
            heatmap_inputs = self._apply_center_scale(heatmap_inputs)

        col_groups, col_linkage, row_groups, row_linkage = (
            self._calculate_linkage_and_groups(heatmap_inputs)
        )

        g = self._create_heatmap(
            data=heatmap_inputs,
            col_linkage=col_linkage,
            col_groups=col_groups,
            row_linkage=row_linkage,
            row_groups=row_groups,
            figsize=figsize,
            title=title,
            row_annotations=row_annotations,
            row_annotation_palettes=row_annotation_palettes,
        )

        try:
            silhouette_avg_rows = float(silhouette_score(heatmap_inputs, row_groups))
        except ValueError:
            silhouette_avg_rows = None

        try:
            silhouette_avg_cols = float(silhouette_score(heatmap_inputs.T, col_groups))
        except ValueError:
            silhouette_avg_cols = None

        return HCHeatmapData(
            filtered_data=filtered_data,
            g=g,
            row_groups=row_groups,
            col_groups=col_groups,
            heatmap_inputs=heatmap_inputs,
            row_silhouette_score=silhouette_avg_rows,
            col_silhouette_score=silhouette_avg_cols,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
        )

    def _calculate_linkage_and_groups(
        self, data: pd.DataFrame
    ) -> tuple[
        list[int] | None, np.ndarray | None, list[int] | None, np.ndarray | None
    ]:
        # calculate linkage and cut the tree
        row_linkage = None
        row_groups = None
        col_linkage = None
        col_groups = None

        if self._n_row_clusters is not None:
            row_linkage = scipy.cluster.hierarchy.linkage(
                scipy.spatial.distance.pdist(data.values), method=self._linkage_method
            )
            row_trees = scipy.cluster.hierarchy.cut_tree(
                row_linkage, n_clusters=[self._n_row_clusters]
            )
            row_groups = [int(t) for t in row_trees.reshape(-1)]
        if self._n_col_clusters is not None:
            col_linkage = scipy.cluster.hierarchy.linkage(
                scipy.spatial.distance.pdist(data.to_numpy().T), method=self._linkage_method
            )
            col_trees = scipy.cluster.hierarchy.cut_tree(
                col_linkage, n_clusters=[self._n_col_clusters]
            )
            col_groups = [int(i) for i in col_trees.reshape(-1)]
        return col_groups, col_linkage, row_groups, row_linkage

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Log transform protein intensities, impute missing values,
        center and scale, so that values are ready for plotting.
        """
        data = data.copy()
        if self._log_transform:
            data = np.log10(data + 1) # pyright: ignore
        return data

    def _fillna(self, data) -> pd.DataFrame:
        """
        Log transform protein intensities, impute missing values,
        center and scale, so that values are ready for plotting.
        """
        fill_na_value = 0
        if self._fillna_method == "min":
            fill_na_value = data.min().min() - self._na_shift_value
        elif self._fillna_method == "mean":
            fill_na_value = data.mean().mean() - self._na_shift_value
        elif self._fillna_method == "median":
            fill_na_value = data.median().median() - self._na_shift_value

        return data.fillna(fill_na_value)

    @staticmethod
    def _apply_center_scale(data) -> pd.DataFrame:
        """
        Log transform protein intensities, impute missing values,
        center and scale, so that values are ready for plotting.
        """
        row_mean = data.mean(axis=1).to_numpy()
        row_std = data.std(axis=1).to_numpy()
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
            data = data.where(data > self._na_threshold)

        if self._min_frequency > 1:
            non_nan_freq = data.notna().sum(axis=1)
            keep_rows = non_nan_freq >= self._min_frequency
            data = data[keep_rows].copy()

        if self._fillna_method == "drop":
            data = data.dropna()

        return data

    def _create_heatmap(
        self,
        data,
        row_linkage,
        row_groups,
        col_linkage,
        col_groups,
        title: str = "",
        row_annotations: dict[str, pd.Series] | None = None,
        row_annotation_palettes: dict[str, dict] | None = None,
        **clustermap_kwargs,
    ) -> ClusterGrid:
        # base setup
        tree_cmap = plt.get_cmap("tab20")

        if row_linkage is not None:
            row_grp_colors = [tree_cmap(grp_idx) for grp_idx in row_groups]
            plot_row_cluster = True
        else:
            row_linkage = None
            row_grp_colors = None
            plot_row_cluster = False

        annotation_colors, annotation_palettes = self._annotation_tracks(
            data.index, row_annotations, row_annotation_palettes
        )
        if annotation_colors is not None:
            if row_grp_colors is not None:
                annotation_colors.insert(
                    0, "Row groups", [mcolors.to_hex(c) for c in row_grp_colors]
                )
            row_grp_colors = annotation_colors
        if col_linkage is not None:
            col_grp_colors = [tree_cmap(grp_idx) for grp_idx in col_groups]
            plot_col_cluster = True
        else:
            col_linkage = None
            col_grp_colors = None
            plot_col_cluster = False

        # plot heatmap using pre-calculated tree colors
        g = sns.clustermap(
            data,
            row_linkage=row_linkage,
            row_cluster=plot_row_cluster,
            col_linkage=col_linkage,
            col_cluster=plot_col_cluster,
            row_colors=row_grp_colors,
            col_colors=col_grp_colors,
            method=self._linkage_method,
            cmap="vlag",
            # cbar_pos=(0, 2, 0.01, 0.1),  # tuple of (left, bottom, width, height)
            **clustermap_kwargs,
        )

        # configure legend
        if row_groups is not None:
            row_patches = [
                mpatches.Patch(color=tree_cmap(grp_idx), label=f"Grp {grp_idx}")
                for grp_idx in sorted(set(row_groups))
            ]
            g.ax_row_dendrogram.legend(
                handles=row_patches, loc="lower left", title="Row groups"
            )
        if col_groups is not None:
            col_patches = [
                mpatches.Patch(color=tree_cmap(grp_idx), label=f"Grp {grp_idx}")
                for grp_idx in sorted(set(col_groups))
            ]
            g.ax_col_dendrogram.legend(
                handles=col_patches, loc="lower left", title="Col groups"
            )
        if annotation_palettes:
            self._add_annotation_legend(g, annotation_palettes)

        rows_only = not plot_col_cluster and col_grp_colors is None
        if g.ax_cbar is not None and not rows_only:
            g.ax_cbar.set_aspect(2)  # decrease width of colorbar
        g.figure.suptitle(title)
        plt.tight_layout()

        # positions are fixed after tight_layout, which would otherwise undo them
        if rows_only:
            self._tidy_rows_only_layout(g)
        if self._center_scale:
            self._label_cbar_low_high(g)

        return g

    @staticmethod
    def _annotation_tracks(
        index: pd.Index,
        row_annotations: dict[str, pd.Series] | None,
        row_annotation_palettes: dict[str, dict] | None,
    ) -> tuple[pd.DataFrame | None, dict[str, dict]]:
        """
        Map each per-record label to a colour, aligned to the plotted rows.

        Returns a frame of colours (one column per track, rows in the
        order of ``index``) and the ``{track: {label: colour}}`` palettes
        used, for the legend. Rows without a label are left white.
        """
        if not row_annotations:
            return None, {}
        if not index.is_unique:
            raise ValueError(
                "row_annotations are matched on the index, which must be unique."
            )
        palettes_in = row_annotation_palettes or {}
        aligned_tracks = {}
        uncoloured = []  # (track, label) pairs needing a default colour
        for name, labels in row_annotations.items():
            if not isinstance(labels, pd.Series):
                raise TypeError(
                    f"row_annotations['{name}'] should be a pd.Series indexed "
                    "like the data."
                )
            if not labels.index.is_unique:
                raise ValueError(
                    f"row_annotations['{name}'] has a duplicated index."
                )
            aligned = labels.reindex(index)
            aligned_tracks[name] = aligned
            given = palettes_in.get(name, {})
            uncoloured += [
                (name, lbl)
                for lbl in sorted(aligned.dropna().unique(), key=str)
                if lbl not in given
            ]

        # one palette across all tracks, so no two labels share a colour, and
        # distinct from the tab20 colours of the row groups
        n_colors = len(uncoloured)
        defaults = dict(
            zip(
                uncoloured,
                sns.color_palette("Dark2" if n_colors <= 8 else "husl", n_colors),
                strict=True,
            )
        )

        tracks = {}
        palettes = {}
        for name, aligned in aligned_tracks.items():
            given = palettes_in.get(name, {})
            palette = {
                lbl: mcolors.to_hex(given[lbl] if lbl in given else defaults[(name, lbl)])
                for lbl in sorted(aligned.dropna().unique(), key=str)
            }
            tracks[name] = [
                palette[lbl] if pd.notna(lbl) else "#ffffff" for lbl in aligned
            ]
            palettes[name] = palette
        return pd.DataFrame(tracks, index=index), palettes

    @staticmethod
    def _add_annotation_legend(g: ClusterGrid, palettes: dict[str, dict]) -> None:
        """Legend for the annotation tracks, in the same colours as the tracks."""
        multiple = len(palettes) > 1
        handles = [
            mpatches.Patch(
                color=color, label=f"{name}: {lbl}" if multiple else str(lbl)
            )
            for name, palette in palettes.items()
            for lbl, color in palette.items()
        ]
        legend_title = "Row annotations" if multiple else next(iter(palettes))
        # ``legend`` replaces the axes legend, so keep the row groups one
        row_groups_legend = g.ax_row_dendrogram.get_legend()
        g.ax_row_dendrogram.legend(
            handles=handles, loc="upper left", title=legend_title
        )
        if row_groups_legend is not None:
            g.ax_row_dendrogram.add_artist(row_groups_legend)

    @staticmethod
    def _tidy_rows_only_layout(g: ClusterGrid) -> None:
        """
        Fix the layout when only rows are clustered: give the empty
        column-dendrogram band to the heatmap and move the colour bar from
        the top left to a narrow strip beside the row dendrogram.
        """
        top = g.ax_col_dendrogram.get_position().y1
        g.ax_col_dendrogram.set_visible(False)
        for ax in (g.ax_heatmap, g.ax_row_dendrogram, g.ax_row_colors):
            if ax is None:
                continue
            pos = ax.get_position()
            ax.set_position((pos.x0, pos.y0, pos.width, top - pos.y0))

        if g.ax_cbar is None:
            return
        dendro = g.ax_row_dendrogram.get_position()
        cbar_width = 0.015
        pad = 0.01
        g.ax_row_dendrogram.set_position(
            (
                dendro.x0 + cbar_width + pad,
                dendro.y0,
                dendro.width - cbar_width - pad,
                dendro.height,
            )
        )
        cbar_height = dendro.height * 0.25
        g.ax_cbar.set_position(
            (dendro.x0, dendro.y1 - cbar_height, cbar_width, cbar_height)
        )

    @staticmethod
    def _label_cbar_low_high(g: ClusterGrid) -> None:
        """Values are z-scores per row: only ``low`` / ``high`` carry meaning."""
        colorbar = g.ax_heatmap.collections[0].colorbar
        if colorbar is None:
            return
        vmin, vmax = colorbar.mappable.get_clim()
        colorbar.set_ticks([vmin, vmax])
        colorbar.set_ticklabels(["low", "high"])


@dataclass
class ClusterDiagnosticsData:
    """
    Diagnostics of a hierarchical clustering. All values are descriptive:
    no thresholds or pass/fail flags are applied, the interpretation is
    left to the user.

    Attributes
    ----------
    records : pd.DataFrame or None
        One row per record (heatmap row), indexed by record id, with the
        columns ``group``, ``silhouette``, ``nearest group`` and
        ``imputed fraction``. ``None`` if no row clustering was performed.
    experiments : pd.DataFrame or None
        Same as ``records``, with one row per experiment (heatmap column).
        ``None`` if no column clustering was performed.
    row_k_scan : pd.DataFrame or None
        Metrics of the row tree cut into each number of groups ``k`` of
        ``k_range``, indexed by ``k``, with the columns ``silhouette``,
        ``calinski-harabasz``, ``davies-bouldin``, ``merge height``,
        ``height gap`` and ``min group size``. ``None`` if no row
        clustering was performed, or no value of ``k_range`` is valid.
    col_k_scan : pd.DataFrame or None
        Same as ``row_k_scan``, for the column tree.
    row_cophenetic_correlation : float or None
        Correlation between the original distances of the rows and their
        distances on the row tree.
    col_cophenetic_correlation : float or None
        Same as ``row_cophenetic_correlation``, for the column tree.
    """

    records: pd.DataFrame | None = None
    experiments: pd.DataFrame | None = None
    row_k_scan: pd.DataFrame | None = None
    col_k_scan: pd.DataFrame | None = None
    row_cophenetic_correlation: float | None = None
    col_cophenetic_correlation: float | None = None


class ClusterDiagnostics:
    """
    Describe how well-defined the groups of a
    :class:`HierarchicallyClusteredHeatmap` are: per record and experiment,
    and across different numbers of groups.

    Every metric is computed on the data the clustering used
    (``heatmap_inputs``), with Euclidean distances. Each metric has its own
    strengths and blind spots; see the documentation for guidance on
    reading them.
    """

    def __init__(self, k_range: Iterable[int] | None = range(2, 21)) -> None:
        """
        Initializer method.

        Parameters
        ----------
        k_range: Iterable of int or None
            Numbers of groups to cut each tree into, to compare them side
            by side. Values outside ``[2, n - 1]``, with ``n`` the number
            of rows or columns, are skipped for that axis. Set to ``None``
            to skip the scan.
        """
        if k_range is None:
            self._k_range: list[int] | None = None
        else:
            self._k_range = sorted({int(k) for k in k_range})

    def eval(self, hc_data: HCHeatmapData) -> ClusterDiagnosticsData:
        """
        Calculate the diagnostics of a clustering.

        Parameters
        ----------
        hc_data: HCHeatmapData
            The output of :meth:`HierarchicallyClusteredHeatmap.eval`.

        Returns
        -------
        ClusterDiagnosticsData
            Per item tables, scans across numbers of groups and cophenetic
            correlations, for each clustered axis.
        """
        result = ClusterDiagnosticsData()
        missing = hc_data.filtered_data.isna()

        if hc_data.row_groups is not None:
            result.records, result.row_k_scan, result.row_cophenetic_correlation = (
                self._diagnose_axis(
                    data=hc_data.heatmap_inputs,
                    groups=hc_data.row_groups,
                    linkage=hc_data.row_linkage,
                    imputed_fraction=missing.mean(axis=1),
                    axis_name="row",
                )
            )

        if hc_data.col_groups is not None:
            (
                result.experiments,
                result.col_k_scan,
                result.col_cophenetic_correlation,
            ) = self._diagnose_axis(
                data=hc_data.heatmap_inputs.T,
                groups=hc_data.col_groups,
                linkage=hc_data.col_linkage,
                imputed_fraction=missing.mean(axis=0),
                axis_name="col",
            )

        return result

    def _diagnose_axis(
        self,
        data: pd.DataFrame,
        groups: list[int],
        linkage: np.ndarray | None,
        imputed_fraction: pd.Series,
        axis_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, float]:
        """Diagnostics of one axis. The items clustered are the rows of ``data``."""
        if linkage is None:
            raise ValueError(
                f"{axis_name}_groups are set but {axis_name}_linkage is missing. "
                "Pass the output of HierarchicallyClusteredHeatmap.eval."
            )

        x = data.to_numpy(dtype=float)
        n_items = x.shape[0]
        groups_arr = np.asarray(groups, dtype=int)

        scan_ks: list[int] = []
        if self._k_range is not None:
            scan_ks = [k for k in self._k_range if 2 <= k <= n_items - 1]

        label_sets = [groups_arr]
        if scan_ks:
            cuts = scipy.cluster.hierarchy.cut_tree(linkage, n_clusters=scan_ks)
            label_sets += [cuts[:, j].astype(int) for j in range(len(scan_ks))]

        silhouettes = _silhouette_samples(x, label_sets)

        item_silhouette, nearest_group = silhouettes[0]
        items = pd.DataFrame(
            {
                "group": groups_arr,
                "silhouette": item_silhouette,
                "nearest group": nearest_group,
                "imputed fraction": imputed_fraction.reindex(data.index).to_numpy(),
            },
            index=data.index,
        )

        k_scan = None
        if scan_ks:
            heights = linkage[:, 2]
            rows = []
            for k, labels, (k_silhouette, _) in zip(
                scan_ks, label_sets[1:], silhouettes[1:], strict=True
            ):
                # the tree has k groups between merge n - k - 1 and merge n - k (0-based)
                upper = heights[n_items - k]
                lower = heights[n_items - k - 1]
                rows.append(
                    {
                        "k": k,
                        "silhouette": float(np.mean(k_silhouette)),
                        "calinski-harabasz": float(calinski_harabasz_score(x, labels)),
                        "davies-bouldin": float(davies_bouldin_score(x, labels)),
                        "merge height": float(upper),
                        "height gap": float(upper - lower),
                        "min group size": int(
                            np.unique(labels, return_counts=True)[1].min()
                        ),
                    }
                )
            k_scan = pd.DataFrame(rows).set_index("k")

        cophenetic_correlation = float(
            scipy.cluster.hierarchy.cophenet(
                linkage, scipy.spatial.distance.pdist(x)
            )[0]
        )

        return items, k_scan, cophenetic_correlation


def _silhouette_samples(
    x: np.ndarray, label_sets: list[np.ndarray], chunk_size: int = 1024
) -> list[tuple[np.ndarray, pd.api.extensions.ExtensionArray]]:
    """
    Per-item silhouette and nearest group, for several labelings of the
    same items. Follows the conventions of
    ``sklearn.metrics.silhouette_samples``: Euclidean distances, and ``0``
    for items alone in their group.

    Distances are computed in chunks of rows and reduced to sums per group
    right away, so the full ``n x n`` distance matrix is never stored.

    Parameters
    ----------
    x: np.ndarray
        Items to be clustered, one per row.
    label_sets: list of np.ndarray
        One array of group labels per labeling, each of length ``n``.
    chunk_size: int
        Number of rows whose distances are computed at once.

    Returns
    -------
    list of tuple
        For each labeling, the silhouette of each item and its nearest
        other group (``<NA>`` if there is only one group).
    """
    n_items = x.shape[0]
    item_idx = np.arange(n_items)
    encodings = []
    for labels in label_sets:
        uniques, inverse = np.unique(labels, return_inverse=True)
        onehot = np.zeros((n_items, len(uniques)))
        onehot[item_idx, inverse] = 1.0
        encodings.append((uniques, inverse, onehot))

    onehot_all = np.hstack([onehot for _, _, onehot in encodings])
    group_sums = np.empty((n_items, onehot_all.shape[1]))
    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        group_sums[start:stop] = (
            scipy.spatial.distance.cdist(x[start:stop], x) @ onehot_all
        )

    results: list[tuple[np.ndarray, pd.api.extensions.ExtensionArray]] = []
    offset = 0
    for uniques, inverse, onehot in encodings:
        n_groups = len(uniques)
        sums = group_sums[:, offset : offset + n_groups]
        offset += n_groups

        if n_groups < 2:
            results.append(
                (np.full(n_items, np.nan), pd.array([pd.NA] * n_items, dtype="Int64"))
            )
            continue

        counts = onehot.sum(axis=0)
        own_counts = counts[inverse]
        with np.errstate(divide="ignore", invalid="ignore"):
            # mean distance to the other members of the own group
            a = sums[item_idx, inverse] / (own_counts - 1)
            # mean distance to the members of each other group
            mean_to_groups = sums / counts
            mean_to_groups[item_idx, inverse] = np.inf
            nearest = np.argmin(mean_to_groups, axis=1)
            b = mean_to_groups[item_idx, nearest]
            denominator = np.maximum(a, b)
            silhouette = np.where(denominator > 0, (b - a) / denominator, 0.0)
        silhouette[own_counts == 1] = 0.0

        results.append((silhouette, pd.array(uniques[nearest], dtype="Int64")))

    return results
