aiida-koopmans
==============

``aiida-koopmans`` is the AiiDA plugin behind Koopmans spectral functional
calculations with Quantum ESPRESSO. It holds the ``@task.graph`` workflow
builders that compose a calculation out of upstream workchains, and the
``CalcJob`` / ``Parser`` pairs for the Quantum ESPRESSO tools upstream does
not cover.

.. important::

   To *run* a Koopmans calculation, install `koopmans
   <https://github.com/elinscott/koopmans>`_ instead, and read `its
   documentation <https://koopmans.readthedocs.io>`_. That package owns the
   input file, the command line, and the profile, code, and pseudopotential
   setup, and it builds the workgraphs defined here.

These pages are for developers of that stack, and for anyone driving the
plugin from their own AiiDA code. They assume you know what a ``CalcJob``, a
``WorkChain``, an entry point, and a workgraph are.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   calculations
   workgraphs
   developer_guide/index
   API reference <apidoc/aiida_koopmans>

Citing
------

In any publication arising from the use of Koopmans functionals or this
code, please cite

.. highlights:: Edward B. Linscott, Nicola Colonna, Riccardo De Gennaro,
  Ngoc Linh Nguyen, Giovanni Borghi, Andrea Ferretti, Ismaila Dabo, and
  Nicola Marzari, *koopmans: An Open-Source Package for Accurately and
  Efficiently Predicting Spectral Properties with Koopmans Functionals*,
  J. Chem. Theory Comput. **19**, 7097-7111 (2023);
  https://doi.org/10.1021/acs.jctc.3c00652.

and, for AiiDA itself,

.. highlights:: Giovanni Pizzi, Andrea Cepellotti, Riccardo Sabatini, Nicola
  Marzari, and Boris Kozinsky, *AiiDA: automated interactive infrastructure
  and database for computational science*, Comp. Mat. Sci. **111**, 218-230
  (2016); https://doi.org/10.1016/j.commatsci.2015.09.013.

The `koopmans documentation <https://koopmans.readthedocs.io>`_ lists the
papers behind each functional and each method.

Getting in touch
----------------

``aiida-koopmans`` is released under the MIT license and developed at
http://github.com/elinscott/aiida-koopmans. Open an issue there, or write to
edwardlinscott@gmail.com. Questions about AiiDA itself belong on the `AiiDA
mailing list <http://www.aiida.net/mailing-list/>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
