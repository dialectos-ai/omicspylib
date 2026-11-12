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
   ``log2(x + 1)`` values. Set ``use_log_transformed=False`` to run it on the
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


Reference
---------

.. toctree::
   :maxdepth: 1

   ttest_fc
   unique_counts
   hc_heatmap
