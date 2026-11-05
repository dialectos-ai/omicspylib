.. currentmodule:: omicspylib.datasets.peptides

PeptidesDatasetExpCondition
===========================

.. autoclass:: PeptidesDatasetExpCondition
    :show-inheritance:


Constructor
-----------
.. automethod:: PeptidesDatasetExpCondition.__init__


Properties
----------
.. autoattribute:: PeptidesDatasetExpCondition.name
.. autoattribute:: PeptidesDatasetExpCondition.experiment_names
.. autoattribute:: PeptidesDatasetExpCondition.n_experiments
.. autoattribute:: PeptidesDatasetExpCondition.record_ids
.. autoattribute:: PeptidesDatasetExpCondition.id_col
.. autoattribute:: PeptidesDatasetExpCondition.metadata
.. autoattribute:: PeptidesDatasetExpCondition.protein_id_col


Inspecting
----------
.. automethod:: PeptidesDatasetExpCondition.describe


Selecting records and experiments
---------------------------------
These methods return a new object.

.. automethod:: PeptidesDatasetExpCondition.filter
.. automethod:: PeptidesDatasetExpCondition.drop


Transforming values
-------------------
These methods modify the object in place.

.. automethod:: PeptidesDatasetExpCondition.log2_transform
.. automethod:: PeptidesDatasetExpCondition.log2_backtransform
.. automethod:: PeptidesDatasetExpCondition.impute
.. automethod:: PeptidesDatasetExpCondition.shift


Summarizing
-----------
.. automethod:: PeptidesDatasetExpCondition.min
.. automethod:: PeptidesDatasetExpCondition.mean
.. automethod:: PeptidesDatasetExpCondition.frequency
.. automethod:: PeptidesDatasetExpCondition.missing_values


Exporting
---------
.. automethod:: PeptidesDatasetExpCondition.to_table
