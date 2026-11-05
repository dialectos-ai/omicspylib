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
   * - Export
     - ``to_table``, which returns a pandas data frame with one column per
       experiment

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


From peptides to proteins
-------------------------

:meth:`~omicspylib.datasets.peptides.PeptidesDataset.to_proteins` groups
peptides by their protein id and returns a ``ProteinsDataset`` with the same
conditions and experiments. Two aggregation methods are available:

* ``agg_method="sum"`` (default): protein intensity is the sum of its
  peptide intensities.
* ``agg_method="counts"``: the number of peptides of each protein found in
  each experiment.

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
