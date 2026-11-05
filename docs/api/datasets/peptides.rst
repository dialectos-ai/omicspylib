.. currentmodule:: omicspylib.datasets.peptides

PeptidesDataset
===============

.. autoclass:: PeptidesDataset
    :show-inheritance:

See the :doc:`Datasets overview <index>` for the input format and a
summary of the common tasks.


Creating a dataset
------------------
.. automethod:: PeptidesDataset.from_df
.. automethod:: PeptidesDataset.__init__


Properties
----------
.. autoattribute:: PeptidesDataset.condition_names
.. autoattribute:: PeptidesDataset.n_conditions
.. autoattribute:: PeptidesDataset.n_experiments
.. autoattribute:: PeptidesDataset.n_records
.. autoattribute:: PeptidesDataset.protein_id_col


Inspecting
----------
.. automethod:: PeptidesDataset.describe
.. automethod:: PeptidesDataset.experiment_names
.. automethod:: PeptidesDataset.unique_records


Selecting records, experiments and conditions
---------------------------------------------
.. automethod:: PeptidesDataset.filter
.. automethod:: PeptidesDataset.drop


Transforming values
-------------------
.. automethod:: PeptidesDataset.log2_transform
.. automethod:: PeptidesDataset.log2_backtransform
.. automethod:: PeptidesDataset.normalize
.. automethod:: PeptidesDataset.impute


Summarizing
-----------
.. automethod:: PeptidesDataset.mean
.. automethod:: PeptidesDataset.frequency
.. automethod:: PeptidesDataset.record_counts


Combining datasets
------------------
.. automethod:: PeptidesDataset.append


Converting to proteins
----------------------
.. automethod:: PeptidesDataset.to_proteins


Exporting
---------
.. automethod:: PeptidesDataset.to_table


Deprecated
----------
.. automethod:: PeptidesDataset.missing_values
