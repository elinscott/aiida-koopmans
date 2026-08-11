"""Deferred access to the members of a workflow's ``codes`` namespace."""

import typing
from collections.abc import Iterable

from aiida import orm
from aiida_workgraph import task
from aiida_workgraph.errors import MissingInput, MissingRequiredInputsError


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
    process-function machinery shape the signature: ``key`` is annotated
    ``str`` for callers but arrives as an ``orm.Str`` at run time (hence
    the ``getattr``), and the mapping must be the variadic
    ``**mapping`` — a declared ``dict`` parameter would be serialized into
    an ``orm.Dict``, which rejects ``Code`` nodes. Call as
    ``get(key="pw", metadata={"call_link_label": "get_pw_code"}, **codes)``.
    """
    return mapping[getattr(key, "value", key)]


def missing_codes_error(codes_spec: type, members: Iterable[str]) -> MissingRequiredInputsError:
    """Build the structured error for settings-conditional codes a guard finds absent.

    A ``NotRequired`` member's need follows a setting, so
    ``check_before_run`` cannot report its absence; the workflow's own
    guard raises this instead, shaped exactly like the socket layer's
    report — one ``graph_inputs.codes.<member>`` entry per missing member,
    carrying the help declared on the ``codes_spec`` TypedDict annotation.
    """
    hints = typing.get_type_hints(codes_spec, include_extras=True)
    entries = []
    for member in members:
        hint = hints[member]
        if typing.get_origin(hint) is typing.NotRequired:
            (hint,) = typing.get_args(hint)
        help_text = next(
            (
                str(meta.help)
                for meta in getattr(hint, "__metadata__", ())
                if getattr(meta, "help", None)
            ),
            None,
        )
        entries.append(MissingInput(f"graph_inputs.codes.{member}", "workgraph.code", help_text))
    return MissingRequiredInputsError(entries)


def get_code(codes: typing.Any, member: str, *, codes_spec: type | None = None) -> orm.AbstractCode:
    """Return one member's code, named for provenance.

    Bakes in the ``get_<member>_code`` call_link_label every call site
    otherwise spelled out by hand. Pass ``codes_spec`` (the workflow's
    ``codes`` TypedDict) for a settings-conditional member: its socket is
    ``NotRequired``, so ``check_before_run`` never reports its absence, and
    the deferred :func:`get` task's own run-time ``KeyError`` fails that
    task without ``wg.run()`` itself raising — so an absent member is
    checked here instead, raising the same structured error at build time.
    """
    if codes_spec is not None and member not in codes:
        raise missing_codes_error(codes_spec, [member])
    return get(key=member, metadata={"call_link_label": f"get_{member}_code"}, **codes).result
