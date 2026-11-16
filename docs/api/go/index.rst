.. currentmodule:: omicspylib.go

GO-terms
========

GO-slims
--------
.. automethod:: goslim.go_to_goslim

Local OBO files
---------------
Parse a GO release once with ``load_obo`` and pass it to ``go_to_goslim(..., obo=...)``
to map offline, without any network call.

.. autofunction:: omicspylib.go.obo.load_obo

.. autoclass:: omicspylib.go.obo.GeneOntology
   :members: primary, ancestors
