=====================
Workgraph conventions
=====================

The rules a new workflow in this repository follows. They are narrow on
purpose: most of them exist because the alternative builds fine and then
fails at run time, or succeeds while quietly dropping an input.

Wrap upstream before writing anything
=====================================

Before adding a ``CalcJob``, confirm that neither ``aiida-quantumespresso``
nor ``aiida-wannier90-workflows`` already covers the step. A new ``CalcJob``
is for a binary upstream does not run, or a mode whose inputs upstream cannot
stage. Where upstream offers ``get_builder_from_protocol``, use it and pass
your changes as nested override dictionaries rather than assembling inputs by
hand.

Compose with ``@task.graph``, and type the outputs
==================================================

A workflow is a ``@task.graph`` function returning a ``TypedDict``. An
upstream ``WorkChain`` is wrapped as a task once, at module level. Nothing
here subclasses ``WorkChain``.

.. code-block:: python

    class ScfNscfOutputs(TypedDict):
        scf_remote_folder: orm.RemoteData
        nscf_remote_folder: orm.RemoteData

    PwBaseStep = task(PwBaseWorkChain)

    @task.graph
    def RunScfNscf(code, structure, overrides=None) -> ScfNscfOutputs:
        builder = PwBaseWorkChain.get_builder_from_protocol(code, structure)
        scf_inputs = get_dict_from_builder(builder)
        scf_inputs.pop("clean_workdir", None)
        scf = PwBaseStep(**scf_inputs)

        nscf_inputs = ...
        nscf_inputs["pw"]["parent_folder"] = scf["remote_folder"]
        nscf = PwBaseStep(**nscf_inputs)

        return ScfNscfOutputs(
            scf_remote_folder=scf["remote_folder"],
            nscf_remote_folder=nscf["remote_folder"],
        )

Three details in that shape are load-bearing:

- ``aiida_workgraph.utils.get_dict_from_builder`` flattens a builder into
  keyword arguments. Call it rather than passing the builder on.
- Read a task's outputs with ``outputs["name"]``. Attribute access does not
  return a socket.
- Pop ``clean_workdir`` before chaining a remote folder, or the upstream
  cleanup deletes the directory a later step reads.

Annotate graph signatures with native Python types — ``list``, ``dict``,
``bool`` — and reach for an ``orm`` class only where no native type says the
same thing. Native annotations also give the socket a tighter validator.

Every value crosses a socket
============================

Thread a parsed output; never re-parse a file a parser already read. If an
upstream parser emits the value on a socket, wire that socket through, even
when doing so means widening an interface. Reading it back out of a retrieved
folder puts a file format between two steps that had a number between them.

Named files travel the same way, as ``SinglefileData`` inputs and outputs: a
parser emits one, a ``local_copy_list`` stages it. A ``RemoteData`` plus a
filename convention is for bulk scratch that no step names — the Quantum
ESPRESSO ``out/`` tree — and nothing else.

Structure travels as data, not as a naming convention. Band order, manifold
membership, and block identity arrive as explicit lists and fields from the
caller. Deriving them from a label prefix or a key name makes a rename a
physics bug.

Things that build and then fail
===============================

A graph input arrives inside a proxy. ``==`` forwards through it and ``is``
does not, so a comparison against an enum member or an interned value takes
the wrong branch for an argument that came in as a socket. Compare with
``==``; where a downstream builder branches by identity, unwrap first.
``x is None`` is safe, because ``None`` arrives bare.

Fan out with a native ``for`` loop over the data inside the graph body, and
recurse with a nested ``@task.graph`` where a loop would need a condition.
Anything a body cannot decide until a value exists belongs in a deferred
nested graph, not in a zone.

A name that a process node will carry cannot start with an underscore. AiiDA
rejects the link label, and the graph fails when it is built.

Naming
======

Case says what a call creates. **PascalCase** for anything whose call creates
a process node: verb-first ``@task.graph`` builders (``WannierizeBlock``,
``RunScfNscf``, ``ComputeOrbitalScreeningParameters``), with the
``Workflow`` suffix reserved for the entry points a dispatcher calls, and
``Step``-suffixed ``task(WorkChain)`` or ``task(CalcJob)`` constants
(``KcpStep``, ``PwBaseStep``). **snake_case** for leaf ``@task``,
calcfunction, and workfunction computations (``compute_alpha_from_dscf``).

Modules follow the step they run: ``workgraphs/<qe_tool>.py``. A new symbol
joins its module's naming family rather than starting a second one.

Fail loudly
===========

An input that cannot take effect raises ``NotImplementedError`` or
``ValueError`` naming the gap, in the caller's vocabulary and saying what to
change. Silently ignoring a keyword produces a calculation that ran, finished,
and answered a different question.

Working around a dependency
===========================

A workaround for a dependency is a candidate bug report, most often against
``aiida-workgraph`` or ``node-graph``, which are young enough that this
repository finds their edges first. An annotation shaped for the framework
rather than for the contract, a value coerced to survive a serializer, a
socket restructured to get past a validator: say what the defect is, which
package it lives in, and what the workaround costs. A workaround that stays
says what it works around.
