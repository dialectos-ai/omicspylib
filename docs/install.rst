.. _installation_page_label:

Installation
============

The library is tested using Python 3.10 -> 3.13.

Using pip
---------
Install the library via:

    ``pip install omicspylib``

To update an existing installation type:

    ``pip install -U omicspylib``


Using uv
--------
Install the library using `uv`:

    ``uv pip install omicspylib``

To update an existing installation type:

    ``uv pip install --upgrade omicspylib``


Using a virtual environment
---------------------------
Create a virtual environment:

Using standard tools:

    ``virtualenv venv``

Or using `uv`:

    ``uv venv``

Activate the virtual environment (Windows command prompt):

    ``venv\Scripts\activate``

Activate the virtual environment (macOS/Linux):

    ``source venv/bin/activate``

Install the library into the active environment:

    ``pip install omicspylib``

    .. code-block:: text

        # Or using uv:
        uv pip install omicspylib