========================
Calculations and parsers
========================

The plugin defines a ``CalcJob`` only where no upstream one fits. Everything
``aiida-quantumespresso`` or ``aiida-wannier90`` already covers is used as it
comes; what follows is the remainder.

Seven ``aiida.calculations`` entry points are registered, each paired with an
``aiida.parsers`` entry point of the same name:

===================================  =====================================  ============================================
Entry point                          Class                                  Runs
===================================  =====================================  ============================================
``koopmans.kcp``                     ``KcpCalculation``                     ``kcp.x``
``koopmans.kcw_wann2kc``             ``Wann2kcCalculation``                 ``kcw.x``, ``calculation='wann2kcw'``
``koopmans.kcw_screen``              ``KcwScreenCalculation``               ``kcw.x``, ``calculation='screen'``
``koopmans.kcw_ham``                 ``KcwHamCalculation``                  ``kcw.x``, ``calculation='ham'``
``koopmans.wann2kcp``                ``Wann2kcpCalculation``                ``wann2kcp.x``
``koopmans.merge_evc``               ``MergeEvcCalculation``                ``merge_evc.x``
``koopmans.pw2wannier_decompose``    ``Pw2wannierDecomposeCalculation``     ``pw2wannier90.x``, ``wan_mode='decompose'``
===================================  =====================================  ============================================

Scratch directories and staged files
====================================

The Koopmans binaries chain through Quantum ESPRESSO's scratch directory:
each reads the previous run's ``out/`` tree, which arrives as a
``parent_folder`` ``RemoteData`` and is symlinked per file, never per
directory, so a calculation can add files to its own ``out/`` without writing
into its parent's.

Everything else travels as an enumerated ``SinglefileData`` input or output —
the ``.nnkp``, ``.chk``, and ``_hr.dat`` a fold consumes, the ``evcw`` files
it produces, the merged ``evc``. A file named on a socket is a file the graph
can rewire; a file found by convention inside a ``RemoteData`` is not. The
scratch-directory contract is the exception, not the pattern to copy.

Reference
=========

.. aiida-calcjob:: KcpCalculation
    :module: aiida_koopmans.calculations.kcp

.. aiida-calcjob:: Wann2kcCalculation
    :module: aiida_koopmans.calculations.kcw

.. aiida-calcjob:: KcwScreenCalculation
    :module: aiida_koopmans.calculations.kcw

.. aiida-calcjob:: KcwHamCalculation
    :module: aiida_koopmans.calculations.kcw

.. aiida-calcjob:: Wann2kcpCalculation
    :module: aiida_koopmans.calculations.wann2kcp

.. aiida-calcjob:: MergeEvcCalculation
    :module: aiida_koopmans.calculations.merge_evc

.. aiida-calcjob:: Pw2wannierDecomposeCalculation
    :module: aiida_koopmans.calculations.pw2wannier_decompose
