"""File enumeration on a calculation's remote working directory."""

from __future__ import annotations

from pathlib import PurePosixPath

from aiida import orm


def walk_remote_files(remote: orm.RemoteData, relpath: str) -> list[str]:
    """Recursively enumerate files under ``relpath`` on a ``RemoteData``.

    Returns paths relative to ``relpath``, using forward slashes. Uses the
    node's AiiDA transport (one open per call), so this works unchanged for
    ``core.local``, ``core.ssh``, or any other transport plugin. Symlinks
    pointing at directories are followed via ``transport.isdir`` -- that is
    what the kcp.x / kcw.x symlink staging wants, since a parent's ``.save``
    tree may itself already be symlinked from a further-upstream parent.
    """
    out: list[str] = []
    root = PurePosixPath(remote.get_remote_path())
    with remote.get_authinfo().get_transport() as transport:

        def _walk(sub: str) -> None:
            here = str(root / relpath / sub) if sub else str(root / relpath)
            for name in transport.listdir(here):
                child_rel = f"{sub}/{name}" if sub else name
                child_full = f"{here}/{name}"
                if transport.isdir(child_full):
                    _walk(child_rel)
                else:
                    out.append(child_rel)

        _walk("")
    return out
