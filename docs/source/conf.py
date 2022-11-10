import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, os.path.abspath('..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tmp'
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
]



templates_path = ['_templates']
exclude_patterns = []

numpydoc_show_class_members = False 

# autodoc_default_options = {"autosummary": True}

# autosummary_imported_members = True

# def _patch_autosummary():
#     from sphinx.ext import autodoc
#     from sphinx.ext import autosummary
#     from sphinx.ext.autosummary import generate

#     class ExceptionDocumenter(autodoc.ExceptionDocumenter):
#         objtype = 'class'

#     def get_documenter(app, obj, parent):
#         if isinstance(obj, type) and issubclass(obj, BaseException):
#             caller = sys._getframe().f_back.f_code.co_name
#             if caller == 'generate_autosummary_content':
#                 if obj.__module__ == 'mpi4py.MPI':
#                     if obj.__name__ == 'Exception':
#                         return ExceptionDocumenter
#         return autosummary.get_documenter(app, obj, parent)

#     generate.get_documenter = get_documenter


# _patch_autosummary()


# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
# napoleon_preprocess_types = False
# napoleon_type_aliases = None
# napoleon_attr_annotations = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

html_theme = "sphinx_rtd_theme"
# html_theme_options = {
#     "relbarbgcolor": "black"
# }