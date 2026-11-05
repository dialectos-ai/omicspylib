omicspylib
==========

**omicspylib** is a Python library for omics data analysis, focusing on
**proteomics**. It takes you from a table of protein or peptide intensities
to a list of differentially abundant proteins, with every step of the
analysis available for you to inspect and adjust.

.. code-block:: python

   from omicspylib import PairwiseComparisonTTestFC, ProteinsDataset

   dataset = ProteinsDataset.from_maxquant("proteinGroups.txt", conditions={
       "control": ["Intensity ctrl_1", "Intensity ctrl_2", "Intensity ctrl_3"],
       "treated": ["Intensity trt_1", "Intensity trt_2", "Intensity trt_3"],
   })
   results = PairwiseComparisonTTestFC(dataset, "treated", "control").eval()


What you can do
---------------

* **Load** protein and peptide tables from any pandas data frame, or protein
  groups directly from MaxQuant.
* **Explore** your data: number of identifications, missing values and
  intensity distributions, per experiment and condition.
* **Prepare** it for analysis: filtering, log transformation, normalization
  and missing value imputation.
* **Compare** experimental conditions: t-test with fold change, proteins
  unique to a condition, and hierarchical clustering.
* **Visualize** the results: volcano plots, Venn diagrams, heatmaps and
  more.
* **Annotate** proteins by mapping GO terms to GO slims.

All results are pandas data frames, so you can continue with any other
Python tool.


Where to start
--------------

* New to omicspylib? :doc:`Install it <install>`, then follow the
  :doc:`quick start <quickstart>` for a complete analysis in a few lines of
  code.
* Looking for a specific workflow? The :doc:`tutorials <tutorials>` cover
  normalization, peptide-to-protein conversion, comparisons and clustering,
  step by step.
* Looking for the details of a function or class? See the
  :doc:`API reference <apireference>`.

omicspylib is open source, under the Apache License 2.0. Report issues or
contribute on `GitHub <https://github.com/dialectos-ai/omicspylib>`_.


.. toctree::
   :hidden:
   :maxdepth: 1

   install
   quickstart
   tutorials
   apireference


Indices
-------

* :ref:`genindex`
* :ref:`search`
