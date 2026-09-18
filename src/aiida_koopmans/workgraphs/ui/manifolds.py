"""Manifold-level pieces shared by the ΔSCF and DFPT band structures.

A Koopmans band structure is assembled one (filling, spin) manifold at a
time: lift that manifold's Koopmans Hamiltonian out of the producing
calculation's retrieved folder, collect its Wannier centres, interpolate
it along the k-path — with the smooth-interpolation correction when a
denser-mesh Wannierization is supplied — and concatenate the manifolds
into one ``BandsData``.

Which file holds the Koopmans Hamiltonian, and how the manifolds are
partitioned, is the route's own knowledge:
:mod:`aiida_koopmans.workgraphs.ui.dscf` reads kcp.x's supercell
``ham_*.dat`` files, :mod:`aiida_koopmans.workgraphs.ui.dfpt` kcw.x's
``*.kcw_hr_*.dat``.
"""

# No ``from __future__ import annotations``: stringified annotations hide
# ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the socket type-checker reads.

import io

import numpy as np
from aiida import orm
from aiida_workgraph import task

from aiida_koopmans.workgraphs.block_wannierize import collect_wannier_functions
from aiida_koopmans.workgraphs.ui import interpolate_bands
from aiida_koopmans.workgraphs.utils.wannier_merge import merge_wannier_hr_file_contents


@task.calcfunction
def extract_koopmans_hamiltonian(
    retrieved: orm.FolderData, filename: orm.Str
) -> orm.SinglefileData:
    """Lift one printed Koopmans Hamiltonian out of a retrieved folder.

    A calcfunction, not a plain ``@task``: it takes an AiiDA data node,
    which the PyFunction deserializer refuses.
    """
    name = filename.value
    available = retrieved.base.repository.list_object_names()
    if name not in available:
        raise ValueError(
            f"`{name}` is missing from the retrieved folder (contents: {sorted(available)}). "
            "The calculation that prints the Koopmans Hamiltonian must run with `write_hr` "
            "(kcp.x `CONTROL.write_hr`, kcw.x `HAM.write_hr`)."
        )
    content = retrieved.base.repository.get_object_content(name, mode="rb")
    return orm.SinglefileData(io.BytesIO(content), filename=name)


@task.calcfunction
def manifold_hamiltonian(**hr_files: orm.SinglefileData) -> orm.SinglefileData:
    """Combine a manifold's per-block Wannier Hamiltonians into one file.

    Keys are read in sorted order, so a caller keying them ``b00``,
    ``b01``, ... states the manifold's band order — the order
    :func:`collect_wannier_functions` concatenates the centres in. The
    combined Hamiltonian is block-diagonal: the blocks were Wannierized
    independently, so no matrix element couples them.
    """
    contents = [hr_files[key].get_content("r") for key in sorted(hr_files)]
    merged = merge_wannier_hr_file_contents(contents)
    return orm.SinglefileData(io.StringIO(merged), filename="aiida_hr.dat")


@task(outputs=["energies", "reference"])
def merge_manifold_energies(
    occupied: list[list[float]],
    empty: list[list[float]] | None = None,
    occupied_down: list[list[float]] | None = None,
    empty_down: list[list[float]] | None = None,
    offset: float = 0.0,
) -> dict:
    """Concatenate per-manifold interpolated eigenvalues into one table.

    Within a spin channel the occupied and empty energies join along the
    band axis; both ``*_down`` inputs together add a leading spin axis.
    A run with no empty manifold passes ``occupied`` alone. ``offset``
    shifts every energy, so the returned ``reference`` (the highest
    occupied energy across the channels) is shifted by it too.
    """
    if (occupied_down is None) != (empty_down is None) and empty is not None:
        raise ValueError(
            "A spin-polarized merge needs both `occupied_down` and `empty_down`; got one."
        )
    occ = np.asarray(occupied, dtype=float)

    def _channel(filled: np.ndarray, unfilled: list[list[float]] | None) -> np.ndarray:
        """Join one channel's occupied and empty eigenvalues along the band axis."""
        if unfilled is None:
            return filled
        emp = np.asarray(unfilled, dtype=float)
        if filled.shape[0] != emp.shape[0]:
            raise ValueError(
                f"The occupied and empty manifolds were interpolated along different k-paths "
                f"({filled.shape[0]} vs {emp.shape[0]} k-points); they cannot be concatenated."
            )
        return np.concatenate([filled, emp], axis=1)

    if occupied_down is None:
        energies = _channel(occ, empty)
        reference = float(occ.max())
    else:
        down_occ = np.asarray(occupied_down, dtype=float)
        up = _channel(occ, empty)
        down = _channel(down_occ, empty_down)
        if up.shape != down.shape:
            raise ValueError(
                f"The spin channels interpolated to different shapes ({up.shape} vs "
                f"{down.shape}); they cannot be stacked into one band structure."
            )
        energies = np.stack([up, down])
        reference = float(max(occ.max(), down_occ.max()))
    energies = energies + offset
    reference = reference + offset
    return {"energies": energies.tolist(), "reference": reference}


@task.calcfunction
def build_band_structure(
    kpath: orm.KpointsData, energies: orm.List, reference: orm.Float
) -> orm.BandsData:
    """Attach interpolated eigenvalues (eV) to their k-path as a ``BandsData``.

    ``reference`` is the valence-band maximum. It is an input rather than
    part of the returned node so a consumer reading the bands off
    provenance finds the energy they align to on the same calculation.
    """
    bands = orm.BandsData()
    bands.set_kpointsdata(kpath)
    bands.set_bands(np.asarray(energies.get_list(), dtype=float), units="eV")
    return bands


def manifold_dft_hamiltonian(labels: list[str], wannierizations, *, link_label: str):
    """Return the socket carrying one manifold's DFT Hamiltonian file.

    ``labels`` are the manifold's block labels in band order. A one-block
    manifold is its block's ``_hr.dat`` unchanged; several blocks are
    combined block-diagonally in that order.
    """
    hr_files = {
        f"b{index:02d}": wannierizations[label]["hr_file"] for index, label in enumerate(labels)
    }
    if len(hr_files) == 1:
        return next(iter(hr_files.values()))
    return manifold_hamiltonian(**hr_files, metadata={"call_link_label": link_label}).result


def interpolate_manifold(
    koopmans_ham_file,
    labels: list[str],
    *,
    label: str,
    block_wannierizations,
    smooth_block_wannierizations,
    structure,
    kpath,
    kgrid: list[int],
    use_ws_distance: bool,
):
    """Add the tasks interpolating one manifold; return its eigenvalue socket.

    ``labels`` are the manifold's block labels in band order. The centres
    come from each block's parsed wannier90 output, keyed so lexicographic
    key order is that band order. A denser-mesh wannierization switches on
    the smooth-interpolation correction, which needs the coarse DFT
    Hamiltonian as well as the dense one; without it neither reaches the
    interpolation.
    """
    wannier_functions = collect_wannier_functions(
        output_parameters={
            f"b{index:02d}": block_wannierizations[block_label]["output_parameters"]
            for index, block_label in enumerate(labels)
        },
        metadata={"call_link_label": f"collect_{label}_centres"},
    )

    smooth_kwargs = {}
    if smooth_block_wannierizations is not None:
        smooth_kwargs = {
            "dft_ham_file": manifold_dft_hamiltonian(
                labels, block_wannierizations, link_label=f"merge_{label}_dft_hamiltonian"
            ),
            "dft_smooth_ham_file": manifold_dft_hamiltonian(
                labels,
                smooth_block_wannierizations,
                link_label=f"merge_{label}_smooth_dft_hamiltonian",
            ),
        }

    return interpolate_bands(
        kc_ham_file=koopmans_ham_file,
        centres=wannier_functions["centres"],
        structure=structure,
        kpath=kpath,
        kgrid=[int(n) for n in kgrid],
        use_ws_distance=bool(use_ws_distance),
        metadata={"call_link_label": f"interpolate_{label}"},
        **smooth_kwargs,
    ).result
