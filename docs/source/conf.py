# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Add src to path for local development; on RTD the package will be installed
if os.environ.get("READTHEDOCS") != "True":
    sys.path.insert(0, os.path.abspath("../../src"))

project = "stocksimpy"
copyright = "2026, SuleymanSade"
author = "SuleymanSade"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
]
myst_heading_anchors = 3  # This makes all H1, H2, and H3 headers linkable
