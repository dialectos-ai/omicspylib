.. currentmodule:: omicspylib.analysis.pairs

Comparisons
===========

Comparisons answer the main question of most experiments: which records
(proteins or peptides) differ between experimental conditions? omicspylib
provides three complementary approaches:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Object
     - Question it answers
     - Input
   * - :class:`~omicspylib.analysis.pairs.statistical.PairwiseComparisonTTestFC`
     - Which records are significantly more or less abundant in one
       condition than in another?
     - A dataset and two condition names
   * - :class:`~omicspylib.analysis.pairs.frequency_based.PairwiseUniqueEntryComparison`
     - Which records are consistently found in one condition, but not in the
       other?
     - A dataset and two condition names
   * - :class:`~omicspylib.analysis.clusters.HierarchicallyClusteredHeatmap`
     - Which records and experiments behave similarly, across all
       conditions?
     - A pandas data frame, such as the output of ``to_table``

The two pairwise comparisons complement each other. A t-test needs values in
both conditions, so a protein detected only in the treated samples cannot be
tested. These are exactly the records the frequency-based comparison
finds.

All comparisons follow the same pattern: you configure the object, call
``eval``, and get the results back with the record ids as the index, so you
can join them with your data:

.. code-block:: python

   comparison = PairwiseComparisonTTestFC(dataset, condition_a="treated", condition_b="control")
   results = comparison.eval()

   output = dataset.to_table().join(results)


Statistical comparison: t-test and fold change
----------------------------------------------

:class:`~omicspylib.analysis.pairs.statistical.PairwiseComparisonTTestFC`
compares two conditions record by record. ``eval`` runs these steps:

#. Keeps only the two conditions, and within each one only the records
   with at least ``min_frequency`` values (default ``3``). Records that do
   not pass in **both** conditions are left out of the results.
#. Computes the **fold change** as the mean of condition A over the mean of
   condition B, using the values on their original scale and ignoring
   missing values.
#. Runs a two-sided Student's t-test (equal variances assumed) on
   ``log2(x + 1)`` values. Set ``log_transform=False`` to run it on the
   values as they are.
#. Adjusts the p-values for multiple testing, by default with the
   Benjamini-Hochberg method (``pval_adj_method="fdr_bh"``). Set it to
   ``None`` to skip this step.

The result has one row per record, with these columns:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``p-value``
     - p-value of the t-test.
   * - ``t-statistic``
     - Positive when values are higher in condition A.
   * - ``adj-p-value``
     - Adjusted p-value. Only present if ``pval_adj_method`` is set.
   * - ``fold change``
     - Mean of condition A divided by the mean of condition B.
   * - ``log2 fold change``
     - ``1`` means 2 times higher in A, ``-1`` means 2 times lower.

.. important::

   Pass the dataset on its **original scale**, not log-transformed. The
   comparison log-transforms the values for the t-test itself, and needs the
   original values for the fold change. If you log-transformed the dataset to
   impute or normalize it, call ``log2_backtransform`` before the comparison.

To visualize the results, use :func:`~omicspylib.plots.volcano.plot_volcano`:

.. code-block:: python

   from omicspylib.plots import plot_volcano

   plot_volcano(results, pval_col="adj-p-value", fc_col="fold change",
                condition_a="treated", condition_b="control")

See the :doc:`statistical comparison tutorial
</notebooks/proteins-dataset-statistical-based-pairwise-comparison>` for a
complete example, including missing value imputation.


Frequency-based comparison: unique records
------------------------------------------

:class:`~omicspylib.analysis.pairs.frequency_based.PairwiseUniqueEntryComparison`
counts in how many experiments of each condition a record is found, and
assigns it to one of four classes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Rule (with the default thresholds)
   * - ``unique <A>``
     - Found in at least ``majority_grp_min_freq`` (``4``) experiments of
       A, and in at most ``minority_grp_max_freq`` (``0``) experiments of B.
   * - ``unique <B>``
     - The same, with A and B swapped.
   * - ``none``
     - Found in at most ``minority_grp_max_freq`` experiments of both
       conditions.
   * - ``common``
     - Everything else.

``<A>`` and ``<B>`` are replaced by the condition names, for example
``unique treated``.

Note that ``common`` includes records that do not pass the thresholds,
such as a record found in 2 experiments of A and none of B. Adjust the two
thresholds to the number of experiments per condition in your data.

.. code-block:: python

   comparison = PairwiseUniqueEntryComparison(dataset, condition_a="treated", condition_b="control")
   results = comparison.eval(majority_grp_min_freq=4, minority_grp_max_freq=0)
   results["frequency_class"].value_counts()

See the :doc:`frequency-based comparison tutorial
</notebooks/proteins-dataset-frequency-based-pairwise-comparison>`.


Hierarchical clustering heatmap
-------------------------------

:class:`~omicspylib.analysis.clusters.HierarchicallyClusteredHeatmap`
groups records (rows) and experiments (columns) with similar profiles, and
plots them on a heatmap. Unlike the pairwise comparisons, it works on a
pandas data frame of values and looks at all conditions at once.

Settings are passed to the constructor, and the data to ``eval``. By default,
``eval``:

#. Treats values equal to or below ``na_threshold`` as missing, and keeps
   rows with at least ``min_frequency`` values.
#. Transforms the values with ``log10(x + 1)``.
#. Fills missing values with the minimum value of the table, minus
   ``na_shift_value`` (``0.2``).
#. Scales each row to a mean of 0 and a standard deviation of 1, so rows
   are clustered by profile, not by abundance.
#. Clusters rows and columns with average linkage on Euclidean distances,
   and cuts the trees into ``n_row_clusters`` (``12``) and
   ``n_col_clusters`` (``3``) groups. Set either one to ``None`` to skip
   clustering on that axis.

``eval`` returns an :class:`~omicspylib.analysis.clusters.HCHeatmapData`
object, with the plot, the group of each row and column, and silhouette
scores to judge how well separated the groups are.

.. code-block:: python

   from omicspylib import HierarchicallyClusteredHeatmap

   heatmap = HierarchicallyClusteredHeatmap(min_frequency=3, n_row_clusters=6, n_col_clusters=2)
   result = heatmap.eval(dataset.to_table())

   row_groups = result.filtered_data.assign(group=result.row_groups)

See the :doc:`clustering heatmap tutorial
</notebooks/hierrarchically-clustered-heatmap>`.


Cluster diagnostics
-------------------

Hierarchical clustering always returns the number of groups you ask for,
whether the data has such groups or not.
:class:`~omicspylib.analysis.clusters.ClusterDiagnostics` describes how
well-defined these groups are, so that you can decide how much to rely on
them.

No metric can tell whether a grouping is right. Each one measures a
different property, and each has blind spots. The diagnostics therefore
report values only, with no thresholds or pass/fail flags. Whether the
groups are meaningful depends on your data and your question, and is for
you to judge.

.. code-block:: python

   from omicspylib import ClusterDiagnostics, HierarchicallyClusteredHeatmap

   heatmap = HierarchicallyClusteredHeatmap(n_row_clusters=6, n_col_clusters=3)
   result = heatmap.eval(dataset.to_table())

   diagnostics = ClusterDiagnostics(k_range=range(2, 21)).eval(result)

   diagnostics.records.sort_values("silhouette").head(20)  # least clear-cut records
   diagnostics.row_k_scan                                   # compare numbers of groups
   diagnostics.row_cophenetic_correlation                   # how faithful the tree is

All values are computed on the data the clustering used
(``heatmap_inputs``: transformed, imputed and scaled), with Euclidean
distances, the same distances the trees are built on. Rows and columns are
diagnosed separately: records are the items of the row clustering, and
experiments the items of the column clustering.

What is reported
~~~~~~~~~~~~~~~~

``eval`` returns a :class:`~omicspylib.analysis.clusters.ClusterDiagnosticsData`
object. This table lists everything it contains, and where each value is
explained.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Value
     - Where to find it
     - Explained in
   * - Group of each item
     - ``group`` column of ``records`` and ``experiments``
     - (assigned by the clustering)
   * - Silhouette of each item
     - ``silhouette`` column of ``records`` and ``experiments``
     - `Silhouette`_
   * - Nearest other group of each item
     - ``nearest group`` column of ``records`` and ``experiments``
     - `Silhouette`_
   * - Imputed fraction of each item
     - ``imputed fraction`` column of ``records`` and ``experiments``
     - `Imputed fraction`_
   * - Mean silhouette, per number of groups
     - ``silhouette`` column of ``row_k_scan`` and ``col_k_scan``
     - `Silhouette`_
   * - Calinski-Harabasz score, per number of groups
     - ``calinski-harabasz`` column of ``row_k_scan`` and ``col_k_scan``
     - `Calinski-Harabasz score`_
   * - Davies-Bouldin score, per number of groups
     - ``davies-bouldin`` column of ``row_k_scan`` and ``col_k_scan``
     - `Davies-Bouldin score`_
   * - Merge height and height gap, per number of groups
     - ``merge height`` and ``height gap`` columns of ``row_k_scan`` and
       ``col_k_scan``
     - `Merge height and height gap`_
   * - Smallest group size, per number of groups
     - ``min group size`` column of ``row_k_scan`` and ``col_k_scan``
     - `Min group size`_
   * - Cophenetic correlation, per tree
     - ``row_cophenetic_correlation`` and ``col_cophenetic_correlation``
     - `Cophenetic correlation`_

``records`` and ``experiments`` have one row per record or experiment,
indexed like your data, so you can join them back to it.

``row_k_scan`` and ``col_k_scan`` have one row per number of groups ``k`` in
``k_range``. Each row describes the same tree, cut into ``k`` groups
instead of the number you chose. The tree is not rebuilt, so this is fast.
Values of ``k`` below ``2`` or above the number of items minus one are
skipped, so with few experiments ``col_k_scan`` has only a few rows. Set
``k_range=None`` to skip the scan.

How the values relate to the linkage method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The values in ``row_k_scan`` and ``col_k_scan`` are computed in three
different ways, which suit different ``linkage_method`` settings of
:class:`~omicspylib.analysis.clusters.HierarchicallyClusteredHeatmap`:

* ``calinski-harabasz`` and ``davies-bouldin`` summarise each group by its
  mean. They suit ``ward`` linkage best, which also builds groups around
  their means.
* ``silhouette`` uses the distances between items. It makes fewer
  assumptions, but still prefers compact, round groups.
* ``merge height`` and ``height gap``, and the cophenetic correlation, are
  read from the tree itself. They follow whatever the linkage method did.

When the values disagree, this usually tells you about the shape of the
groups, not that one of them is wrong. For example, ``single`` or
``average`` linkage can form elongated groups that are clear on the tree
but have a low silhouette.

Silhouette
~~~~~~~~~~

*Reported in:* ``silhouette`` and ``nearest group`` columns of ``records``
and ``experiments`` (per item), and ``silhouette`` column of ``row_k_scan``
and ``col_k_scan`` (mean over all items).

For each item, ``a`` is its mean distance to the other members of its
group, and ``b`` its mean distance to the members of the nearest other
group. The silhouette is ``(b - a) / max(a, b)``, from ``-1`` to ``1``:
close to ``1`` means the item is much closer to its own group, close to
``0`` that it sits between two groups, and negative that it is closer to
another group. Items alone in their group get ``0``.

``nearest group`` is the group that gives ``b``: for items with a low
silhouette, the group they could just as well belong to.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Disadvantages
   * - Available per item, so you can see which records or experiments
       are clear-cut and which sit between two groups.
     - Prefers compact, round groups. Elongated or chained groups, which
       ``single`` and ``average`` linkage can form, get low values even if
       they are clear on the tree.
   * - Bounded from ``-1`` to ``1``, so values are comparable between
       datasets and settings.
     - The mean can hide a lot: ``0.3`` can mean all items are moderately
       placed, or most are well placed and a few badly. Look at the per
       item values too.
   * - Uses distances only, no group means, so it fits any linkage method
       reasonably.
     - Needs the distance between every pair of items, which gets costly
       for very large numbers of items. The clustering itself has the same
       limit.

Calinski-Harabasz score
~~~~~~~~~~~~~~~~~~~~~~~

*Reported in:* ``calinski-harabasz`` column of ``row_k_scan`` and
``col_k_scan``.

The spread of the group means around the overall mean, divided by the
spread of the items around their group mean, each scaled by its degrees of
freedom. It is the F-statistic of a one-way ANOVA, extended to many
variables. Higher means better separated.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Disadvantages
   * - Measures what ``ward`` linkage optimises, so it is the natural
       score for ``ward`` groupings.
     - Has no upper limit: values are only comparable within the same
       data and settings, not between datasets.
   * - Fast for any number of items, as it only needs group means.
     - Assumes each group is well described by its mean. Fits groups from
       ``single`` or ``average`` linkage poorly.
   * - Familiar to many readers, as it is widely used.
     - One value per number of groups: it does not show which items or
       groups are the problem.

Davies-Bouldin score
~~~~~~~~~~~~~~~~~~~~

*Reported in:* ``davies-bouldin`` column of ``row_k_scan`` and
``col_k_scan``.

For each group, take its most similar other group, and divide the sum of
the two groups' spreads by the distance between their means. The score is
the mean of these ratios over all groups. Lower means better separated,
and ``0`` is the minimum.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Disadvantages
   * - Each group is compared with its most similar group, so a single
       pair of overlapping groups raises the score clearly.
     - Like the Calinski-Harabasz score, it relies on group means and suits
       ``ward`` linkage best.
   * - Fast for any number of items.
     - Has no upper limit, so values are only comparable within the same
       data and settings.
   * - Lower is better, with ``0`` as the ideal, which is easy to read.
     - One value per number of groups: it does not tell which pair of
       groups overlaps.

Merge height and height gap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Reported in:* ``merge height`` and ``height gap`` columns of
``row_k_scan`` and ``col_k_scan``.

The tree joins items and groups step by step, each at a height defined by
the linkage method. The ``k`` groups exist between two of these steps: the
one that formed them, and the next one, which merges them into ``k - 1``
groups. ``merge height`` is the height of that next step, and ``height
gap`` the distance between the two steps: the range of cut heights over
which the tree has exactly ``k`` groups.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Disadvantages
   * - Read from the tree itself, so they follow any linkage method on its
       own terms.
     - Describe the tree, not the data: if the tree distorts the data (see
       `Cophenetic correlation`_), they inherit the distortion.
   * - A large gap means that the ``k`` groups stay the same over a wide
       range of cut heights, so the choice of ``k`` is less arbitrary.
     - Heights are in the units of the linkage method, so they are only
       comparable within the same data and settings.
   * - Easy to relate to the dendrogram on the heatmap.
     - With ``centroid`` and ``median`` linkage, a later step can be lower
       than an earlier one. The height gap can then be negative and is hard
       to interpret.

Min group size
~~~~~~~~~~~~~~

*Reported in:* ``min group size`` column of ``row_k_scan`` and
``col_k_scan``.

The number of items in the smallest of the ``k`` groups.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Disadvantages
   * - Shows when a larger ``k`` only splits off tiny groups, often single
       outlier items, rather than dividing a real group.
     - Says nothing about how separated the groups are. Small groups can
       be real, for example a few co-regulated records.

Cophenetic correlation
~~~~~~~~~~~~~~~~~~~~~~

*Reported in:* ``row_cophenetic_correlation`` and
``col_cophenetic_correlation``, one value per tree.

The correlation between the original distances of all pairs of items, and
their distances on the tree: the height at which the two items are first
joined. Close to ``1`` means the tree represents the distances faithfully.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Disadvantages
   * - Shows how faithfully the tree represents the data, before any cut.
       A low value means the dendrogram on the heatmap is a rough
       summary, whichever number of groups you choose.
     - Describes the whole tree, not the groups: it does not depend on the
       number of groups, and does not tell which groups are well defined.
   * - Comparable between linkage methods: run the clustering with each
       method you consider, and compare their values.
     - Large distances dominate the correlation, so it can be high even if
       the fine structure of the tree is not faithful.

Imputed fraction
~~~~~~~~~~~~~~~~

*Reported in:* ``imputed fraction`` column of ``records`` and
``experiments``.

The fraction of the item's values that were missing and filled in before
clustering, from ``0`` to ``1``. It is not a cluster metric, but a check on
what drives the groups.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Disadvantages
   * - Shows groups that may be formed by the imputed values rather than
       the measured ones. With ``fillna_method="min"``, records with many
       missing values share the same low values, and can group together
       for that reason alone.
     - A high value does not make a group wrong: records missing in the
       same experiments can reflect real biology, such as records absent
       in one condition.
   * - Imputed values affect all other values on this page, the
       Calinski-Harabasz and Davies-Bouldin scores most. Check it before
       reading the others.
     - It is ``0`` everywhere with ``fillna_method="drop"``, as no values
       are imputed.


Reference
---------

.. toctree::
   :maxdepth: 1

   ttest_fc
   unique_counts
   hc_heatmap
