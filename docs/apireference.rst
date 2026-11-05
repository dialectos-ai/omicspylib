.. _api_doc_page_label:


API Reference
=============

This section documents every public object of **omicspylib**. If you are
new to the library, read the short overview below first, then follow the
links to the part you need. For end-to-end examples, see the
:ref:`hands-on tutorials <tutorials_page_label>`.


Core concepts
-------------

Everything in omicspylib revolves around a **dataset**: a table of
quantitative values (e.g., intensities), with one row per **record** and
one column per **experiment**.

* A **record** is the measured entity, identified by a unique id, such as a
  protein group or a peptide sequence.
* An **experiment** is a single run or sample, such as one biological
  replicate.
* Experiments are grouped into **experimental conditions** (e.g.,
  ``control`` and ``treated``). You define this grouping when you create the
  dataset, and most calculations are performed per condition.

There are two dataset types, which share the same interface:

* :class:`~omicspylib.datasets.proteins.ProteinsDataset` for protein-level
  data.
* :class:`~omicspylib.datasets.peptides.PeptidesDataset` for peptide-level
  data. It can also be aggregated into a protein dataset with
  :meth:`~omicspylib.datasets.peptides.PeptidesDataset.to_proteins`.

Internally, each condition is stored as a ``...DatasetExpCondition`` object.
You normally don't need to create or use these objects directly.

.. note::

   Dataset methods such as ``filter``, ``impute``, ``normalize`` or
   ``log2_transform`` do not modify the dataset in place. They return a new
   dataset, so you can chain them and still keep the original data.


Typical workflow
----------------

#. **Load** your data with ``from_df`` (any pandas data frame) or, for
   proteins, ``from_maxquant`` (a MaxQuant output file).
#. **Explore** it with ``describe``, ``frequency`` and ``missing_values``,
   or with the :doc:`plotting functions <api/plots/index>`.
#. **Prepare** it with ``filter``, ``log2_transform``, ``impute`` and
   ``normalize``.
#. **Compare** experimental conditions using the
   :doc:`comparison objects <api/comparisons/index>`.
#. **Export** the results with ``to_table``, which returns a pandas data
   frame.

.. code-block:: python

   import pandas as pd

   from omicspylib import PairwiseComparisonTTestFC, ProteinsDataset
   from omicspylib.plots import plot_volcano

   data = pd.read_csv("protein_dataset.tsv", sep="\t")
   conditions = {
       "control": ["ctrl_rep1", "ctrl_rep2", "ctrl_rep3"],
       "treated": ["trt_rep1", "trt_rep2", "trt_rep3"],
   }
   dataset = ProteinsDataset.from_df(data, id_col="protein_id", conditions=conditions)
   dataset.describe()

   dataset = (
       dataset.filter(min_frequency=2)
       .log2_transform()
       .impute(method="group row mean", random_noise=True)
       .log2_backtransform()
   )

   comparison = PairwiseComparisonTTestFC(dataset, condition_a="control", condition_b="treated")
   results = comparison.eval(min_frequency=2)

   plot_volcano(results, pval_col="adj-p-value", fc_col="fold change", condition_a="control", condition_b="treated")


Package overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Section
     - Contents
   * - :doc:`api/datasets/index`
     - The ``ProteinsDataset`` and ``PeptidesDataset`` objects, and
       ``concat`` for combining datasets.
   * - :doc:`api/comparisons/index`
     - Pairwise comparisons between conditions (t-test with fold change,
       or unique records), and hierarchical clustering heatmaps.
   * - :doc:`api/calculations/index`
     - Lower-level statistical functions (fold change, t-test with
       multiple-testing correction) used by the comparison objects.
   * - :doc:`api/plots/index`
     - Plots for data exploration and results: density, frequency, missing
       values, Venn diagrams and volcano plots.
   * - :doc:`api/go/index`
     - Mapping Gene Ontology (GO) terms to GO slims, using the EBI QuickGO
       web service (requires an internet connection).
   * - :doc:`api/utils/index`
     - Helpers for cleaning MaxQuant output (reverse hits, contaminants,
       only-modified-site records).

The most common objects can be imported directly from the top-level
package:

.. code-block:: python

   from omicspylib import (
       HierarchicallyClusteredHeatmap,
       PairwiseComparisonTTestFC,
       PairwiseUniqueEntryComparison,
       PeptidesDataset,
       ProteinsDataset,
       concat,
       go_to_goslim,
   )


.. toctree::
   :hidden:
   :maxdepth: 1

   api/datasets/index
   api/comparisons/index
   api/calculations/index
   api/plots/index
   api/go/index
   api/utils/index
