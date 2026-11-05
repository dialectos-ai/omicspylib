Utils
=====

Helper functions for preparing input data before you create a dataset.


MaxQuant output
---------------

`MaxQuant <https://www.maxquant.org/>`_ output tables include entries that
are usually excluded from the analysis. MaxQuant marks each of them with a
``+`` in a dedicated column, and each function below removes one group:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Function
     - Removes
     - MaxQuant column
   * - :func:`~omicspylib.utils.mq.mq_rm_reverse`
     - Reverse (decoy) hits
     - ``Reverse`` or ``Decoy``
   * - :func:`~omicspylib.utils.mq.mq_rm_contaminants`
     - Potential contaminants
     - ``Potential contaminant``
   * - :func:`~omicspylib.utils.mq.mq_rm_only_modified`
     - Proteins identified only by modified peptides
     - ``Only identified by site`` (``proteinGroups.txt`` only)

**For protein data**, you usually don't need to call them yourself:
:meth:`ProteinsDataset.from_maxquant
<omicspylib.datasets.proteins.ProteinsDataset.from_maxquant>` applies all
three by default. Each one can be turned off with the ``rm_reverse``,
``rm_contaminants`` and ``rm_only_modified`` arguments.

**For peptide data**, or if you load the table yourself, apply them before
creating the dataset:

.. code-block:: python

   import pandas as pd
   from omicspylib import PeptidesDataset
   from omicspylib.utils import mq_rm_contaminants, mq_rm_reverse

   data = pd.read_csv("peptides.txt", sep="\t")
   data = mq_rm_contaminants(mq_rm_reverse(data))

   peptides = PeptidesDataset.from_df(
       data, id_col="Sequence", conditions=conditions, protein_id_col="Leading razor protein"
   )

Each function returns a new data frame and leaves the input unchanged.

.. tip::

   The number of removed rows is logged at the ``INFO`` level. To see it:

   .. code-block:: python

      import logging
      logging.basicConfig(level=logging.INFO)


Reference
---------

.. toctree::
   :maxdepth: 1

   mq
