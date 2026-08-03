============
Installation
============

Neither ``aiida-koopmans`` nor all of its dependencies are on PyPI.
``[tool.uv.sources]`` in ``pyproject.toml`` points at forks carrying features
that are still working their way upstream, and each has to be checked out as
a sibling of this repository::

    git clone https://github.com/elinscott/aiida-koopmans
    git clone --branch patched https://github.com/elinscott/aiida-workgraph
    git clone --branch patched https://github.com/elinscott/node-graph
    git clone --branch patched https://github.com/elinscott/aiida-wannier90-workflows
    git clone https://github.com/elinscott/aiida-wannierjl
    cd aiida-koopmans
    uv sync

Keep that list in step with ``[tool.uv.sources]``, and drop an entry as its
fork merges upstream. The continuous integration workflow clones the same
set, so it is the authority on which branch each fork is read from.

With a profile in place (``verdi presto`` creates one), the plugins should be
registered::

    verdi plugin list aiida.calculations   # the koopmans.* plugins are listed

Quantum ESPRESSO codes
======================

Every calculation runs a Quantum ESPRESSO binary, which AiiDA reaches through
a ``Code`` node. The Koopmans-specific ones come from:

``kcp.x``
    `koopmans-kcp <https://github.com/epfl-theos/koopmans-kcp>`_, a fork of
    Quantum ESPRESSO v4.1 whose ``cp.x`` implements the Koopmans functionals
    at :math:`\Gamma` only.

``wann2kcp.x``, ``merge_evc.x``
    `koopmans-qe-utils <https://github.com/epfl-theos/koopmans-qe-utils>`_.

``kcw.x``
    `Quantum ESPRESSO <https://gitlab.com/QEF/q-e>`_ itself, which
    implements the Koopmans functionals on a :math:`k`-point grid.

``pw2wannier90.x`` with ``wan_mode='decompose'``
    the ``wann-decompose`` branch of Quantum ESPRESSO. Only the
    ``power_spectrum`` machine-learning descriptor needs it.

The other codes the workflow builders drive — ``pw.x``, ``ph.x``,
``projwfc.x``, ``dos.x``, ``pw2wannier90.x``, ``wannier90.x``, and the
Wannier.jl driver behind the automated block splitting — are the stock ones
their own plugins expect.

The ``koopmans`` package's ``koopmans install`` command builds and registers
all of them, along with a pseudopotential family, and is the supported route.
