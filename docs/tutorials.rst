.. _tutorials_page_label:

Tutorials
=========

Step-by-step examples of common proteomics workflows with omicspylib. Each
tutorial is a Jupyter notebook that you can run yourself: get the notebooks
and the simulated data they use from the `notebooks folder
<https://github.com/dialectos-ai/omicspylib/tree/main/docs/notebooks>`_
of the repository.

New to omicspylib? Start with the :doc:`quick start <quickstart>`, which
covers a complete analysis in a few steps.


Preparing your data
-------------------

:doc:`Peptide-level normalization across experiments <notebooks/peptide-level-normalization-across-experiments>`
   Correct for systematic intensity differences between experiments, so
   that values can be compared.

:doc:`Converting peptides to proteins <notebooks/peptides-to-proteins-dataset-conversion>`
   Aggregate a peptide dataset into protein intensities (the sum of peptide
   intensities) or into peptide counts per protein.


Comparing conditions
--------------------

:doc:`Statistical comparison <notebooks/proteins-dataset-statistical-based-pairwise-comparison>`
   A complete differential analysis: explore the data, impute missing
   values, compare two conditions with a t-test and fold change, plot a
   volcano plot, and export the results.

:doc:`Frequency-based comparison <notebooks/proteins-dataset-frequency-based-pairwise-comparison>`
   Find the proteins found consistently in one condition and not in the
   other, and plot the overlap between conditions on a Venn diagram.


Exploring patterns
------------------

:doc:`Hierarchically clustered heatmap <notebooks/hierrarchically-clustered-heatmap>`
   Group proteins and experiments with similar profiles, and plot them on a
   heatmap.


.. toctree::
   :hidden:
   :maxdepth: 1

   notebooks/peptide-level-normalization-across-experiments
   notebooks/peptides-to-proteins-dataset-conversion
   notebooks/proteins-dataset-statistical-based-pairwise-comparison
   notebooks/proteins-dataset-frequency-based-pairwise-comparison
   notebooks/hierrarchically-clustered-heatmap
