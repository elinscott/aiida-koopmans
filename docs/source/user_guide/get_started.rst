===============
Getting started
===============

``aiida-koopmans`` provides the AiiDA plugins behind the `koopmans
<https://koopmans-functionals.org>`_ package: ``CalcJob``/``Parser`` pairs for
the Koopmans-fork Quantum ESPRESSO binaries (``kcp.x``, ``kcw.x``,
``wann2kcp.x``, ``merge_evc.x``) and the ``@task.graph`` workflows that
compose them (together with upstream ``aiida-quantumespresso`` and
``aiida-wannier90-workflows`` workchains) into full Koopmans spectral
functional calculations.

Installation
++++++++++++

Use the following commands to install the plugin::

    git clone https://github.com/elinscott/aiida-koopmans
    cd aiida-koopmans
    pip install -e .
    verdi presto  # set up an AiiDA profile, if you don't have one
    verdi plugin list aiida.calculations  # should now show the koopmans.* plugins

Most users should not drive this package directly: install the ``koopmans``
package instead, whose ``koopmans install`` command sets up the AiiDA profile,
codes, and pseudopotential families, and whose ``koopmans run`` command
translates a koopmans input file into the workgraphs defined here.

Available calculations
++++++++++++++++++++++

.. aiida-calcjob:: KcpCalculation
    :module: aiida_koopmans.calculations.kcp

.. aiida-calcjob:: Wann2kcpCalculation
    :module: aiida_koopmans.calculations.wann2kcp

.. aiida-calcjob:: MergeEvcCalculation
    :module: aiida_koopmans.calculations.merge_evc

.. aiida-calcjob:: Wann2kcCalculation
    :module: aiida_koopmans.calculations.kcw

.. aiida-calcjob:: KcwScreenCalculation
    :module: aiida_koopmans.calculations.kcw

.. aiida-calcjob:: KcwHamCalculation
    :module: aiida_koopmans.calculations.kcw
