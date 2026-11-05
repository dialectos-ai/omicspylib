.. currentmodule:: omicspylib.datasets.proteins

ProteinsDatasetExpCondition
===========================

.. autoclass:: ProteinsDatasetExpCondition
    :show-inheritance:


Constructor
-----------
.. automethod:: ProteinsDatasetExpCondition.__init__


Properties
----------
.. autoattribute:: ProteinsDatasetExpCondition.name
.. autoattribute:: ProteinsDatasetExpCondition.experiment_names
.. autoattribute:: ProteinsDatasetExpCondition.n_experiments
.. autoattribute:: ProteinsDatasetExpCondition.record_ids
.. autoattribute:: ProteinsDatasetExpCondition.id_col
.. autoattribute:: ProteinsDatasetExpCondition.metadata


Inspecting
----------
.. automethod:: ProteinsDatasetExpCondition.describe


Selecting records and experiments
---------------------------------
These methods return a new object.

.. automethod:: ProteinsDatasetExpCondition.filter
.. automethod:: ProteinsDatasetExpCondition.drop


Transforming values
-------------------
These methods modify the object in place.

.. automethod:: ProteinsDatasetExpCondition.log2_transform
.. automethod:: ProteinsDatasetExpCondition.log2_backtransform
.. automethod:: ProteinsDatasetExpCondition.impute
.. automethod:: ProteinsDatasetExpCondition.shift


Summarizing
-----------
.. automethod:: ProteinsDatasetExpCondition.min
.. automethod:: ProteinsDatasetExpCondition.mean
.. automethod:: ProteinsDatasetExpCondition.frequency
.. automethod:: ProteinsDatasetExpCondition.missing_values


Exporting
---------
.. automethod:: ProteinsDatasetExpCondition.to_table
