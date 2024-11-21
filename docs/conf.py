# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# import os
# import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'omicspylib'
copyright = '2024, Dialectos.ai'
author = 'Dialectos.ai'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# pip install sphinx-book-theme
# pip install furo
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_book_theme'
# html_theme = 'furo'
html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']

autodoc_member_order = 'alphabetical'

autoclass_content = 'class'  # Options: 'class', 'init', 'both'
# 'class': Use the class's docstring, ignore the __init__ method (this is the default).
# 'both': Use both the class's and the __init__ method's docstring.
# 'init': Use the __init__ method's docstring, ignore the class docstring.


