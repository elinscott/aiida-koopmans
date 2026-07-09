"""Sphinx configuration for aiida-koopmans."""

import os
import subprocess
import sys
import time

from aiida import load_profile
from aiida.storage.sqlite_temp import SqliteTempBackend

import aiida_koopmans

# -- AiiDA-related setup --------------------------------------------------

# Load a temporary AiiDA profile so autodoc can import the plugin modules.
temp_profile = SqliteTempBackend.create_profile("temp-profile")
load_profile(temp_profile, allow_switch=True)

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.contentui",
    "aiida.sphinxext",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "aiida": ("https://aiida.readthedocs.io/projects/aiida-core/en/latest", None),
}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "aiida-koopmans"
copyright_first_year = "2026"
copyright_owners = "Edward Linscott"

current_year = str(time.localtime().tm_year)
copyright_year_string = (
    current_year
    if current_year == copyright_first_year
    else f"{copyright_first_year}-{current_year}"
)
copyright = f"{copyright_year_string}, {copyright_owners}. All rights reserved"

release = aiida_koopmans.__version__
version = ".".join(release.split(".")[:2])

language = "en"
show_authors = True
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"
html_logo = "images/AiiDA_transparent_logo.png"
html_title = f"aiida-koopmans v{release}"
html_theme_options = {}
html_show_sourcelink = False
html_use_opensearch = "https://aiida-koopmans.readthedocs.io"
html_search_language = "en"

# Warnings to ignore when using the -n (nitpicky) option
nitpick_ignore = [
    ("py:class", "Logger"),
    ("py:class", "QbFields"),  # Warning started to appear with aiida 2.6
]


def run_apidoc(_):
    """Run sphinx-apidoc when building the documentation.

    Needs to be done in conf.py in order to include the APIdoc in the
    build on readthedocs. See https://github.com/rtfd/readthedocs.org/issues/1139.
    """
    source_dir = os.path.abspath(os.path.dirname(__file__))
    apidoc_dir = os.path.join(source_dir, "apidoc")
    package_dir = os.path.join(source_dir, os.pardir, os.pardir, "src", "aiida_koopmans")

    cmd_path = "sphinx-apidoc"
    if hasattr(sys, "real_prefix"):  # we are in a virtualenv: assemble the path manually
        cmd_path = os.path.abspath(os.path.join(sys.prefix, "bin", "sphinx-apidoc"))

    options = [
        "-o",
        apidoc_dir,
        package_dir,
        "--private",
        "--force",
        "--no-toc",
    ]

    env = os.environ.copy()
    env["SPHINX_APIDOC_OPTIONS"] = (
        "members,special-members,private-members,undoc-members,show-inheritance"
    )
    subprocess.check_call([cmd_path, *options], env=env)  # noqa: S603


def setup(app):
    """Register the apidoc hook."""
    app.connect("builder-inited", run_apidoc)
