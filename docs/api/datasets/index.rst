Datasets
========

A dataset holds the quantitative values of your experiment and is the
starting point for almost everything in omicspylib. Exploration, filtering,
imputation, normalization, comparisons and plots all take a dataset as
input.

omicspylib has two dataset types, which share the same interface:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Use it for
   * - :class:`~omicspylib.datasets.proteins.ProteinsDataset`
     - Protein-level data, such as protein group intensities. Can be loaded
       directly from MaxQuant output.
   * - :class:`~omicspylib.datasets.peptides.PeptidesDataset`
     - Peptide-level data. If you provide the protein each peptide belongs to,
       it can be aggregated into a ``ProteinsDataset``.


Input data
----------

A dataset is created from a table in wide format: one row per record (protein
or peptide), and one column per experiment. A column with a unique id for
each record is required. Any other columns, such as annotations, are ignored.

.. list-table::
   :header-rows: 1

   * - protein_id
     - ctrl_rep1
     - ctrl_rep2
     - ctrl_rep3
     - trt_rep1
     - trt_rep2
     - trt_rep3
   * - P02768
     - 25010.3
     - 0
     - 24410.0
     - 31200.5
     - 30988.1
     - 29874.2
   * - P01024
     - 1310.8
     - 1450.2
     - 1398.7
     - 0
     - 0
     - 512.4

You also define which experiments belong to each **experimental condition**.
Conditions are the groups you will later compare, such as ``control`` and
``treated``. Many calculations, such as means, frequencies and imputation
by group, are done per condition.

**Missing values.** Missing values can be stored as ``0`` or ``NaN``. Most
methods also take an ``na_threshold`` argument (default ``0.0``), and any
value equal to or below it is treated as missing too.


Creating a dataset
------------------

From any pandas data frame, use ``from_df``:

.. code-block:: python

   import pandas as pd
   from omicspylib import ProteinsDataset

   data = pd.read_csv("protein_dataset.tsv", sep="\t")
   conditions = {
       "control": ["ctrl_rep1", "ctrl_rep2", "ctrl_rep3"],
       "treated": ["trt_rep1", "trt_rep2", "trt_rep3"],
   }
   dataset = ProteinsDataset.from_df(data, id_col="protein_id", conditions=conditions)

For MaxQuant ``proteinGroups.txt`` files, use
:meth:`~omicspylib.datasets.proteins.ProteinsDataset.from_maxquant`. It
reads the file, and by default removes reverse hits, contaminants and
only-identified-by-site records. The ids from the ``Majority protein IDs``
column are stored in a column renamed to ``protein_id``.

.. code-block:: python

   dataset = ProteinsDataset.from_maxquant("proteinGroups.txt", conditions=conditions)

For peptides, also pass the column that holds the protein id of each
peptide. You need it to aggregate peptides into proteins later on:

.. code-block:: python

   from omicspylib import PeptidesDataset

   peptides = PeptidesDataset.from_df(
       data, id_col="peptide", conditions=conditions, protein_id_col="protein_id"
   )


Working with a dataset
----------------------

Methods that change the data, such as ``filter``, ``impute`` or
``log2_transform``, return a **new dataset** and leave the original one
unchanged. You can chain them, and keep intermediate versions for
comparison:

.. code-block:: python

   l2_dataset = dataset.filter(min_frequency=2).log2_transform()
   imputed = l2_dataset.impute(method="group row mean", random_noise=True)

The most common tasks:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Task
     - Methods
   * - Inspect
     - ``describe``, ``condition_names``, ``experiment_names``,
       ``n_conditions``, ``n_experiments``, ``n_records``,
       ``unique_records``
   * - Select a subset
     - ``filter`` (keep experiments, conditions or record ids, or keep
       records present in a minimum number of experiments), ``drop`` (remove
       experiments, conditions or record ids)
   * - Transform
     - ``log2_transform``, ``log2_backtransform``, ``normalize``,
       ``impute``
   * - Summarize
     - ``mean``, ``frequency``, ``record_counts``
   * - Check reproducibility
     - ``correlation``, ``bland_altman`` (see
       :ref:`below <reproducibility_label>`)
   * - Export
     - ``to_table``, which returns a pandas data frame with one column per
       experiment, and ``detection_matrix``, which tells whether each record
       was detected in each experiment

.. note::

   ``filter(min_frequency=...)`` is applied per condition. For example,
   with ``min_frequency=2``, a protein found in 3 of the control experiments
   but only 1 of the treated experiments is kept in ``control`` and removed
   from ``treated``. In the output of ``to_table``, its ``treated`` columns
   are then empty (``NaN``).

.. tip::

   Normalization and imputation do not log-transform the data for you. If
   you want them applied on the log scale, call ``log2_transform`` first, and
   ``log2_backtransform`` afterwards if you need the original scale.


Exporting data and missing values
---------------------------------

``to_table`` returns the values as stored, so missing values can appear as
``0``. Methods of the dataset already leave them out of their calculations,
but generic tools don't know the convention: pandas ``mean`` or ``corr``,
scikit-learn or a PCA would treat a ``0`` as a value measured at zero.

If you continue the analysis outside omicspylib, ask for missing values as
``NaN``, which pandas skips by default:

.. code-block:: python

   table = dataset.to_table(mask_missing=True)
   table.corr()  # zeros no longer count as measurements

To correlate experiments, prefer ``dataset.correlation()`` (see
:ref:`below <reproducibility_label>`), which also handles the choice of
records and the log transformation for you.

Use the ``na_threshold`` argument to treat higher values as missing too,
e.g. ``dataset.to_table(mask_missing=True, na_threshold=10.0)``.

For presence/absence questions, use ``detection_matrix``. It returns a
boolean data frame with the same shape as ``to_table``, ``True`` where the
record was detected. For example, the records detected in at least one
``treated`` experiment but in none of the ``control`` experiments:

.. code-block:: python

   detected = dataset.detection_matrix()
   in_treated = detected[dataset.experiment_names("treated")].any(axis=1)
   in_control = detected[dataset.experiment_names("control")].any(axis=1)
   only_in_treated = detected.index[in_treated & ~in_control]

To count detections per record or per experiment, ``frequency`` and
``record_counts`` give the same numbers directly.


.. _reproducibility_label:

Checking reproducibility
------------------------

Replicates of the same condition should agree more closely than
experiments of different conditions. Two methods measure this agreement,
pair by pair. Both return a data frame with one row per pair of
experiments and the columns ``experiment_a``, ``experiment_b``,
``condition_a``, ``condition_b`` and ``n_records`` (the number of records
used), plus their own results.

* ``correlation`` gives a ``coefficient`` for each pair.
* ``bland_altman`` gives the ``bias`` (mean difference) and the limits of
  agreement (``loa_lower``, ``loa_upper``) for each pair. On log2 values,
  a ``bias`` of ``1`` means ``experiment_a`` is two times higher than
  ``experiment_b`` on average, which a correlation coefficient cannot show.

.. code-block:: python

   corr = dataset.correlation()  # replicates of each condition, Spearman
   corr.groupby("condition_a")["coefficient"].median()

   # replicates vs. experiments of different conditions
   within = dataset.correlation(pairs="within_conditions")["coefficient"].median()
   across = dataset.correlation(pairs="across_conditions")["coefficient"].median()

   dataset.bland_altman(pairs=[("ctrl_rep1", "ctrl_rep2")])

Three choices affect the results. The defaults are chosen to be safe:

* **Which pairs** (``pairs``): ``"within_conditions"`` (default),
  ``"across_conditions"``, ``"all"``, or a list of
  ``(experiment_a, experiment_b)`` tuples.
* **Which records** (``missing``): with ``"pairwise"`` (default), each pair
  uses the records with a value in both of its experiments. With
  ``"listwise"``, every pair uses only the records with a value in all the
  experiments of the selected pairs. The two give different results, and
  ``"listwise"`` can leave few records when there are many missing values.
  In both cases, missing values (``0``, ``NaN``, or values below
  ``na_threshold``) are never treated as measurements. Pairs with fewer
  than ``min_records`` records (default ``10``) get ``NaN``.
* **Which coefficient, on which scale** (``method``, ``log_transform``):
  Spearman (default) and Kendall are rank-based, so a log transformation
  does not change them. Pearson is not: intensities span several orders of
  magnitude, so on raw values a few dozen of the most abundant records
  dominate it. ``correlation`` therefore log2 transforms the values for
  Pearson by default. ``bland_altman`` always log2 transforms by default.
  If your dataset is already log-transformed, pass ``log_transform=False``.

To see the results, use
:func:`~omicspylib.plots.agreement.plot_correlation_heatmap` for an overview
of all pairs, and
:func:`~omicspylib.plots.agreement.plot_experiment_pair` for a scatter or
Bland-Altman plot of one pair:

.. code-block:: python

   from omicspylib.plots import plot_correlation_heatmap, plot_experiment_pair

   plot_correlation_heatmap(dataset)
   plot_experiment_pair(dataset, "ctrl_rep1", "ctrl_rep2", plot_type="bland_altman")


From peptides to proteins
-------------------------

:meth:`~omicspylib.datasets.peptides.PeptidesDataset.to_proteins` groups
peptides by their protein id and returns a ``ProteinsDataset`` with the same
conditions and experiments. Two aggregation methods are available:

* ``agg_method="sum"`` (default): protein intensity is the sum of its
  peptide intensities.
* ``agg_method="counts"``: the number of peptides of each protein found in
  each experiment. Peptides with values equal to or below ``na_threshold``
  (default ``0.0``) are not counted.

.. code-block:: python

   proteins = peptides.to_proteins(agg_method="sum")
   peptide_counts = peptides.to_proteins(agg_method="counts", add_prefix="pept_counts_")

If the peptides dataset was created without a ``protein_id_col``,
``to_proteins`` returns ``None`` and raises a warning.


Combining datasets
------------------

To combine datasets that contain different experimental conditions into
one dataset, use :func:`~omicspylib.datasets.concat.concat`. Records are
matched by their id, and records missing from a condition are treated as
missing values.
All datasets must be of the same type and use the same id column name, and
condition names must not repeat across them.

.. code-block:: python

   from omicspylib import concat

   dataset = concat([control_dataset, treated_dataset])

``concat`` does not modify the datasets you pass in.

For two datasets, you can also use ``append``. Pass ``inplace=False`` to get
a new dataset, or ``inplace=True`` to add the conditions to the existing one:

.. code-block:: python

   dataset = control_dataset.append(treated_dataset, inplace=False)

.. note::

   ``append`` currently defaults to ``inplace=True`` and emits a
   ``FutureWarning`` when ``inplace`` is not specified. The default will change
   to ``False`` in a future release, to match the other dataset methods.


Experimental condition objects
------------------------------

Internally, each condition is stored as a
:class:`~omicspylib.datasets.proteins.ProteinsDatasetExpCondition` or a
:class:`~omicspylib.datasets.peptides.PeptidesDatasetExpCondition`. They are
created for you, and you normally don't need to use them directly. Their
pages are listed below for completeness.


Reference
---------

.. toctree::
   :maxdepth: 1

   proteins
   peptides
   concat
   proteins_exp_conditions
   peptides_exp_conditions
