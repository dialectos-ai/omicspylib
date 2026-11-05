Quick Start
===========

This page walks through a complete analysis: loading a protein table,
exploring it, preparing it, and finding the proteins that differ between two
conditions. If you haven't installed omicspylib yet, see the
:ref:`installation page <installation_page_label>`.

The example uses a small simulated dataset of 100 proteins, measured in 3
conditions (``c1``, ``c2`` and ``c3``) with 5 replicates each. You can run
the code as is: the data is downloaded from the omicspylib repository.


1. Load the data
----------------

omicspylib works on tables with one row per protein and one column per
experiment. Load the table with pandas, then tell omicspylib which column
holds the protein ids, and which experiments belong to each condition:

.. code-block:: python

   import matplotlib.pyplot as plt
   import pandas as pd

   from omicspylib import PairwiseComparisonTTestFC, ProteinsDataset
   from omicspylib.plots import plot_density, plot_missing_values, plot_volcano

   url = "https://raw.githubusercontent.com/dialectos-ai/omicspylib/main/docs/notebooks/data/protein_dataset.tsv"
   data = pd.read_csv(url, sep="\t")

   conditions = {
       "c1": ["c1_rep1", "c1_rep2", "c1_rep3", "c1_rep4", "c1_rep5"],
       "c2": ["c2_rep1", "c2_rep2", "c2_rep3", "c2_rep4", "c2_rep5"],
   }
   dataset = ProteinsDataset.from_df(data, id_col="protein_id", conditions=conditions)
   dataset.describe()

Only the experiments you list are used. Here, condition ``c3`` is left out.
Missing values are stored as ``0`` in this table, which omicspylib handles by
default.

.. tip::

   If your data comes from MaxQuant, use ``ProteinsDataset.from_maxquant``
   instead. It reads ``proteinGroups.txt`` directly and removes reverse
   hits and contaminants for you.


2. Explore it
-------------

Check the intensity distributions and the number of missing values in each
experiment, to spot problematic samples before the analysis:

.. code-block:: python

   fig, ax = plt.subplots(1, 2, figsize=(10, 4))
   plot_density(dataset=dataset.log2_transform(), ax=ax[0])
   plot_missing_values(dataset=dataset, ax=ax[1])
   plt.show()


3. Prepare it
-------------

Keep the proteins quantified in at least 3 of the 5 replicates of a
condition, and impute the remaining missing values with the mean of each
protein within its condition, plus some random noise. Imputation is done on
the log scale, then values are transformed back:

.. code-block:: python

   prepared = (
       dataset.filter(min_frequency=3)
       .log2_transform()
       .impute(method="group row mean", random_noise=True)
       .log2_backtransform()
   )

Each step returns a new dataset, so ``dataset`` itself is unchanged.


4. Compare the conditions
-------------------------

Run a t-test and calculate the fold change of ``c1`` over ``c2`` for every
protein, and plot the results on a volcano plot:

.. code-block:: python

   comparison = PairwiseComparisonTTestFC(prepared, condition_a="c1", condition_b="c2")
   results = comparison.eval()

   plot_volcano(results, pval_col="adj-p-value", fc_col="fold change",
                condition_a="c1", condition_b="c2")
   plt.show()

``results`` is a pandas data frame with one row per protein, with the
p-value, the adjusted p-value (Benjamini-Hochberg by default), and the fold
change.


5. Export the results
---------------------

Join the results with the values and save them:

.. code-block:: python

   output = prepared.to_table().join(results)
   output.to_csv("c1_vs_c2.tsv", sep="\t")


Next steps
----------

* The :doc:`tutorials <tutorials>` cover each step in more detail, along
  with normalization, peptide data and clustering.
* The :doc:`API reference <apireference>` explains the concepts behind
  omicspylib and documents every function and class.
