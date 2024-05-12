import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.abspath('..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QHyper'
copyright = '2022, Cyfronet'
author = 'Cyfronet'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
]

autodoc_typehints = 'none'

pygments_style = 'sphinx'

templates_path = ['_templates']
exclude_patterns = []

# numpydoc_show_class_members = False
autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

html_theme = "pydata_sphinx_theme"

html_logo = "_static/logo.png"

html_theme_options = {
    "logo": {
        "text": "QHyper",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/qc-lab/QHyper",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_sidebars = {
    "usage": [],
    "contribution": [],
}
