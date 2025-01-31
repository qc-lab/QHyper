import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QHyper'
copyright = '2024, Cyfronet'
author = 'Cyfronet'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'numpydoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'nbsphinx',
    'sphinx_tabs.tabs',
]
autosummary_generate = True 
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints

# autodoc_typehints = 'none'

pygments_style = 'sphinx'

templates_path = ['_templates']
exclude_patterns = []

# numpydoc_show_class_members = False
autosectionlabel_prefix_document = True
html_sourcelink_suffix = ''
numpydoc_show_class_members = False 
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_logo = "_static/logo.png"
add_module_names = False
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
