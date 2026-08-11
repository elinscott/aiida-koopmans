"""Deferred access to the members of a workflow's ``codes`` namespace."""

from aiida import orm
from aiida_workgraph import task


@task.workfunction()
def get(key: str, **mapping) -> orm.AbstractCode:
    """Return one member of a mapping, resolved at run time.

    Interim for node-graph#169's ``ref()``: a build-time ``codes["member"]``
    subscript raises a bare ``KeyError`` on a missing member, killing the
    eager build before ``check_before_run`` can report the unset
    ``graph_inputs.codes.<member>`` socket. Routing the access through this
    task keeps the build alive, so a missing code surfaces as a structured
    ``MissingRequiredInputsError`` at submit.

    A ``@task.workfunction`` because the body returns an already-stored
    ``Code`` node, which calcfunction-style processes (including
    ``aiida_pythonjob.PyFunction``) reject under provenance rules —
    workfunctions may *select* existing nodes. Two side-effects of the
    process-function machinery shape the signature: ``key`` arrives as an
    ``orm.Str`` (hence ``.value``), and the mapping must be the variadic
    ``**mapping`` — a declared ``dict`` parameter would be serialized into
    an ``orm.Dict``, which rejects ``Code`` nodes. Call as
    ``get(key="pw", metadata={"call_link_label": "get_pw_code"}, **codes)``.
    """
    return mapping[key.value]
