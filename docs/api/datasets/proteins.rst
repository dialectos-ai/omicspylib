.. currentmodule:: omicspylib.datasets.proteins

ProteinsDataset
===============

.. autoclass:: ProteinsDataset
    :show-inheritance:

See the :doc:`Datasets overview <index>` for the input format and a
summary of the common tasks.


Creating a dataset
------------------
.. automethod:: ProteinsDataset.from_df
.. automethod:: ProteinsDataset.from_maxquant
.. automethod:: ProteinsDataset.__init__


Properties
----------
.. autoattribute:: ProteinsDataset.condition_names
.. autoattribute:: ProteinsDataset.n_conditions
.. autoattribute:: ProteinsDataset.n_experiments
.. autoattribute:: ProteinsDataset.n_records


Inspecting
----------
.. automethod:: ProteinsDataset.describe
.. automethod:: ProteinsDataset.experiment_names
.. automethod:: ProteinsDataset.unique_records


Selecting records, experiments and conditions
---------------------------------------------
.. automethod:: ProteinsDataset.filter
.. automethod:: ProteinsDataset.drop


Transforming values
-------------------
.. automethod:: ProteinsDataset.log2_transform
.. automethod:: ProteinsDataset.log2_backtransform
.. automethod:: ProteinsDataset.normalize
.. automethod:: ProteinsDataset.impute


Summarizing
-----------
.. automethod:: ProteinsDataset.mean
.. automethod:: ProteinsDataset.frequency
.. automethod:: ProteinsDataset.record_counts


Combining datasets
------------------
.. automethod:: ProteinsDataset.append


Exporting
---------
.. automethod:: ProteinsDataset.to_table


Deprecated
----------
.. automethod:: ProteinsDataset.missing_values
