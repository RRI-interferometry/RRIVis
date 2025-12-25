"""Sphinx configuration for RRIvis API documentation."""

import os
import sys
from datetime import datetime

# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "RRIvis"
copyright = f"{datetime.now().year}, Kartik Mandar"
author = "Kartik Mandar"
release = "0.2.0"
version = "0.2.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",        # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",       # Support NumPy/Google style docstrings
    "sphinx.ext.viewcode",       # Add links to source code
    "sphinx.ext.intersphinx",    # Link to other project docs
    "sphinx.ext.mathjax",        # Math rendering
    "sphinx.ext.autosummary",    # Generate summary tables
    "myst_parser",               # Markdown support
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Markdown support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Templates
templates_path = ["_templates"]

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#2980B9",
    # TOC options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Custom CSS
html_css_files = [
    "custom.css",
]

# Sidebar
html_sidebars = {
    "**": [
        "relations.html",
        "searchbox.html",
    ]
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "\\usepackage[Bjornstrup]{fncychap}",
}

latex_documents = [
    (
        "index",
        "RRIvis.tex",
        "RRIvis Documentation",
        "Kartik Mandar",
        "manual",
    ),
]

# -- Extension configuration -------------------------------------------------

# MathJax configuration
mathjax3_config = {
    "tex": {
        "macros": {
            "vec": [r"\mathbf{#1}", 1],
            "mat": [r"\mathbf{#1}", 1],
        }
    }
}
