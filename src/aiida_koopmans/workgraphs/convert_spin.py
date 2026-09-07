"""``@task.calcfunction`` that symmetrises an nspin=1 kcp.x save into an nspin=2 save.

The Koopmans DSCF closed-shell init is a 3-step chain:

1. A first ``KcpCalculation`` runs nspin=1 from scratch and converges the
   single-channel DFT solution.
2. A second nspin=2 ``KcpCalculation`` runs from scratch with no outer loop
   purely to lay out the disk structure of an nspin=2 ``aiida_60.save/``.
   Its wavefunction files (``evc01.dat``, ``evc02.dat``, ...) contain
   junk content but valid headers.
3. This calcfunction splices the *content* of the nspin=1 save into both
   spin channels of the nspin=2 layout, producing a spin-symmetric save
   where the down channel equals the up channel equals the nspin=1
   wavefunction. The downstream "real" nspin=2 kcp.x run then restarts
   from this save (``restart_mode='restart'``) and the symmetric spin
   state is preserved on the first iteration.

The work is pure byte substitution on a handful of small XML / binary
files — no QE binary is invoked. The substitutions are:

* ``nk="1"`` → ``nk="2"``
* ``nspin="1"`` → ``nspin="2"``
* (down channel only) ``ik="1"`` → ``ik="2"``
* (down channel only) ``ispin="1"`` → ``ispin="2"``

Implemented as ``@task.calcfunction`` (rather than a full ``CalcJob``)
because the operation never invokes an external binary — it is pure
directory restructuring and byte substitution, so there is no code to
schedule. All file access goes through the parent folders' own AiiDA
transport (``RemoteData.get_authinfo().get_transport()``), the same
mechanism a ``CalcJob`` uses to stage its inputs, so the calcfunction
works unchanged for ``core.local``, ``core.ssh``, or any other
transport plugin — it never assumes the two parents live on the
machine running the daemon.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path, PurePosixPath

from aiida import orm
from aiida_workgraph import task

from aiida_koopmans.calculations.kcp import KcpCalculation

# Files that need spin-1 → spin-2 conversion. Each entry maps a source
# filename in the nspin=1 ``K00001/`` directory to the two destination
# filenames (spin-up, spin-down) in the nspin=2 ``K00001/`` directory.
# The list is deliberately fixed — kcp.x emits exactly this set of
# per-orbital files in the closed-shell branch.
_CONVERSION_MAP: tuple[tuple[str, str, str], ...] = (
    ("evc0.dat", "evc01.dat", "evc02.dat"),
    ("evc0_empty1.dat", "evc0_empty1.dat", "evc0_empty2.dat"),
    ("evcm.dat", "evcm1.dat", "evcm2.dat"),
    ("evc.dat", "evc1.dat", "evc2.dat"),
    ("hamiltonian.xml", "hamiltonian1.xml", "hamiltonian2.xml"),
    ("eigenval.xml", "eigenval1.xml", "eigenval2.xml"),
    ("evc_empty1.dat", "evc_empty1.dat", "evc_empty2.dat"),
    ("lambda01.dat", "lambda01.dat", "lambda02.dat"),
)

# Layout constants — sourced from :class:`KcpCalculation` so the downstream
# kcp.x run picks the result up via its ``parent_folder`` symlink.
_PREFIX = KcpCalculation._PREFIX
_OUTPUT_SUBFOLDER = KcpCalculation._OUTPUT_SUBFOLDER
_NDW = KcpCalculation._NDW
_SAVE_DIRNAME = f"{_PREFIX}_{_NDW}.save"
_K_SUBDIR = KcpCalculation._K_SUBDIR


def _convert_spin1_to_spin2_bytes(content: bytes) -> tuple[bytes, bytes]:
    """Apply the nspin=1 → nspin=2 byte substitutions.

    Returns ``(spin_up_bytes, spin_down_bytes)``. Both channels start
    from the same nspin=1 content; the down channel additionally has
    ``ik="1"`` → ``ik="2"`` and ``ispin="1"`` → ``ispin="2"`` applied.
    """
    up = content.replace(b'nk="1"', b'nk="2"').replace(b'nspin="1"', b'nspin="2"')
    down = up.replace(b'ik="1"', b'ik="2"').replace(b'ispin="1"', b'ispin="2"')
    return up, down


def _remote_scratch_root(transport, authinfo) -> PurePosixPath:
    """Return the stable remote scratch root for convert-spin outputs.

    Lives under the ``AuthInfo``'s configured work directory (the same
    directory a ``CalcJob`` uploads into) so it persists for the life of
    the AiiDA install and survives whichever computer the two parent
    folders live on — unlike ``tempfile.mkdtemp`` on the local machine,
    which is meaningless once the parents are remote. Downstream
    consumers (the spin-2 restart ``KcpCalculation``) symlink into this
    directory via ``parent_folder``, so the path must remain readable
    for the lifetime of the surrounding workflow.
    """
    workdir = authinfo.get_workdir().format(username=transport.whoami())
    root = PurePosixPath(workdir) / "scratch" / "koopmans" / "convert_spin"
    transport.makedirs(str(root), ignore_existing=True)
    return root


@task.calcfunction(outputs=["remote_folder"])
def convert_spin1_to_spin2(
    spin1_parent_folder: orm.RemoteData,
    spin2_dummy_parent_folder: orm.RemoteData,
) -> dict:
    """Build a spin-symmetric nspin=2 save dir from a nspin=1 reference + nspin=2 dummy.

    :param spin1_parent_folder: ``remote_folder`` of the nspin=1 kcp.x
        run. Its ``out/aiida_60.save/K00001/`` directory is the source
        of the spin-symmetric wavefunctions.
    :param spin2_dummy_parent_folder: ``remote_folder`` of the nspin=2
        dummy kcp.x run that provides the destination save-directory
        skeleton (density, charge, XML metadata).
    :returns: Mapping ``{"remote_folder": RemoteData}`` whose
        ``out/aiida_60.save/`` contains the spin-symmetric wavefunction
        files, ready to be consumed by a downstream ``KcpCalculation``
        via its ``parent_folder`` input.
    :raises ValueError: if the two parent folders live on different
        computers.
    :raises FileNotFoundError: if the nspin=1 save directory is missing
        or contains none of the expected wavefunction files.
    """
    # Both parents must agree on the computer — the output RemoteData
    # is bound to that same computer so the downstream kcp.x can pick
    # it up with its existing parent-folder symlink machinery.
    spin1_computer = spin1_parent_folder.computer
    spin2_computer = spin2_dummy_parent_folder.computer
    if spin1_computer is None or spin2_computer is None:
        raise ValueError("both parent folders must be bound to a computer.")
    if spin1_computer.pk != spin2_computer.pk:
        raise ValueError(
            "spin1_parent_folder and spin2_dummy_parent_folder must live on the same "
            f"computer; got {spin1_computer.label} and {spin2_computer.label}."
        )

    authinfo = spin1_parent_folder.get_authinfo()
    with authinfo.get_transport() as transport:
        spin1_remote_root = PurePosixPath(spin1_parent_folder.get_remote_path())
        spin1_k = spin1_remote_root / _OUTPUT_SUBFOLDER / _SAVE_DIRNAME / _K_SUBDIR
        spin1_save = spin1_k.parent
        if not transport.isdir(str(spin1_save)):
            raise FileNotFoundError(
                f"Expected nspin=1 save directory does not exist at {spin1_save}. "
                "Did you point ``spin1_parent_folder`` at the right kcp.x run?"
            )

        dummy_remote_root = PurePosixPath(spin2_dummy_parent_folder.get_remote_path())
        dummy_save = dummy_remote_root / _OUTPUT_SUBFOLDER / _SAVE_DIRNAME
        if not transport.isdir(str(dummy_save)):
            raise FileNotFoundError(
                f"Expected nspin=2 dummy save directory does not exist at {dummy_save}. "
                "Did you point ``spin2_dummy_parent_folder`` at the right kcp.x run?"
            )

        # Stable, per-call output directory under the AuthInfo work directory.
        # ``uuid4`` here just picks a unique name; the directory is created
        # explicitly (``makedirs``) rather than relying on ``copytree`` to lay
        # down missing parents, since the ssh transport's ``copytree`` shells
        # out to ``cp -r`` which -- unlike ``shutil.copytree`` -- does not
        # create intermediate directories.
        scratch_root = _remote_scratch_root(transport, authinfo)
        new_root = scratch_root / f"convert_spin1_to_spin2_{uuid.uuid4().hex}"
        new_save = new_root / _OUTPUT_SUBFOLDER / _SAVE_DIRNAME
        new_k = new_save / _K_SUBDIR
        transport.makedirs(str(new_save.parent))

        # Start from the dummy's nspin=2 save skeleton — copies in density,
        # charge, XML metadata, plus placeholder per-channel wfc files that
        # we overwrite below with the converted content from the spin1
        # parent. ``new_save`` must not exist yet: both the local and ssh
        # transports nest the source inside an existing destination
        # directory instead of copying its contents into it.
        transport.copytree(str(dummy_save), str(new_save))

        # Now overlay the converted wavefunctions. Each file is round-tripped
        # through a local temp file (get -> substitute -> put) since neither
        # transport exposes a bytes-in-bytes-out API; this is the same
        # staging idiom a CalcJob uses to move files onto/off of a remote.
        converted_any = False
        with tempfile.TemporaryDirectory() as local_tmp:
            local_tmp_path = Path(local_tmp)
            local_src = local_tmp_path / "src"
            local_up = local_tmp_path / "up"
            local_down = local_tmp_path / "down"
            for spin1_name, up_name, down_name in _CONVERSION_MAP:
                src = spin1_k / spin1_name
                if not transport.path_exists(str(src)):
                    # Only convert files that actually exist on the parent.
                    # The map is a superset of what kcp.x emits in any given
                    # run.
                    continue
                transport.getfile(str(src), str(local_src))
                content = local_src.read_bytes()
                up_bytes, down_bytes = _convert_spin1_to_spin2_bytes(content)

                local_up.write_bytes(up_bytes)
                transport.putfile(str(local_up), str(new_k / up_name))
                local_down.write_bytes(down_bytes)
                transport.putfile(str(local_down), str(new_k / down_name))
                converted_any = True

        if not converted_any:
            raise FileNotFoundError(
                f"No known nspin=1 wavefunction files were found under {spin1_k}. "
                "Expected at least one of: " + ", ".join(name for name, _, _ in _CONVERSION_MAP)
            )

    # Build a fresh (unstored) RemoteData and hand it back. AiiDA's
    # process-function machinery calls ``self.out("remote_folder", ...)``
    # on this and handles storage + provenance linking.
    out = orm.RemoteData(
        computer=spin1_parent_folder.computer,
        remote_path=str(new_root),
    )
    return {"remote_folder": out}


__all__ = ("convert_spin1_to_spin2",)
