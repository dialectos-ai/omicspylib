.. _installation_page_label:

Installation
============

omicspylib requires **Python 3.10 or newer**, and is tested on Python 3.10
to 3.14. It is available on `PyPI <https://pypi.org/project/omicspylib/>`_.


Install with pip
----------------

.. code-block:: bash

   pip install omicspylib

To upgrade an existing installation:

.. code-block:: bash

   pip install --upgrade omicspylib


Install with uv
---------------

In a `uv <https://docs.astral.sh/uv/>`_ project:

.. code-block:: bash

   uv add omicspylib

Or, to install it into the active environment:

.. code-block:: bash

   uv pip install omicspylib


Use a virtual environment
-------------------------

We recommend installing omicspylib in a virtual environment, to keep its
dependencies separate from other projects.

Create and activate the environment, with the standard library:

.. code-block:: bash

   python -m venv .venv

   # macOS / Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate

Or, with uv:

.. code-block:: bash

   uv venv
   # then activate it as above

Then install omicspylib into it:

.. code-block:: bash

   pip install omicspylib


Use in Jupyter notebooks
------------------------

The :doc:`tutorials <tutorials>` are Jupyter notebooks, and the plotting
functions work well in them. omicspylib installs ``ipykernel``, so after
installing it in your environment you can select that environment as the
notebook kernel, e.g. in VS Code or PyCharm. To use it in Jupyter, install
Jupyter in the same environment:

.. code-block:: bash

   pip install jupyterlab
   jupyter lab


Check the installation
----------------------

.. code-block:: python

   import omicspylib
   print(omicspylib.__version__)

Next, follow the :doc:`quick start <quickstart>`.
