"""Koopmans band structure assembly, shared by the ΔSCF and DFPT routes.

A Koopmans band structure is assembled one (filling, spin) manifold at a
time: lift that manifold's Koopmans Hamiltonian out of the producing
calculation's retrieved folder, collect its Wannier centres, interpolate it
along the k-path — with the smooth-interpolation correction when a
denser-mesh Wannierization is supplied — and concatenate the manifolds into
one ``BandsData``.

:func:`KoopmansBandStructureTask` runs that fan-out for whatever manifolds
its caller declares. Which file holds each manifold's Koopmans Hamiltonian,
and how the manifolds are partitioned, is the route's own knowledge: the
ΔSCF route (:mod:`aiida_koopmans.workgraphs.kcp`) reads kcp.x's supercell
``ham_*.dat`` files, keyed by the ``merge_groups`` partition its
initialisation wannierization emitted; the DFPT route
(:mod:`aiida_koopmans.workgraphs.dfpt`) reads kcw.x's ``*.kcw_hr_*.dat``
files, keyed by its own ``occ_labels`` / ``emp_labels``. Both pass that
knowledge in as a list of :class:`ManifoldSpec`.
"""

import io
from typing import Annotated, NotRequired, TypedDict

import numpy as np
from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.workgraphs.block_wannierize import (
    WannierizeBlockOutputs,
    collect_wannier_functions,
)
from aiida_koopmans.workgraphs.ui import DensityOfStates, compute_dos_from_bands, interpolate_bands
from aiida_koopmans.workgraphs.utils.wannier_merge import merge_wannier_hr_file_contents


class ManifoldSpec(TypedDict):
    """One (filling, spin) manifold's identity inside a Koopmans band structure.

    ``filled`` and ``spin`` alone decide merge order (occupied before
    empty), which manifold's top sets the valence-band-maximum reference,
    and channel stacking — never a label. ``blocks`` alone decides
    membership and band order: the caller's own key list into
    ``block_wannierizations``, never derived from how those keys are
    spelled.

    ``spin`` carries :class:`~aiida_koopmans.spin.SpinChannel`'s plain
    string ``.value`` (``"none"``/``"up"``/``"down"``), not the enum
    member: a spec built inside a stored task's own output goes through
    AiiDA's node serialization, which only a plain ``str`` is guaranteed
    to survive. Read it back with ``SpinChannel(spec["spin"])``.
    """

    filled: bool
    spin: str
    filename: str
    blocks: list[str]


def _manifold_label(*, filled: bool, spin: SpinChannel) -> str:
    """Name one manifold for task link labels: filling, spin-qualified when polarized.

    A single-channel run (``spin='none'``, or one physical channel run on
    its own, as the DFPT route's per-channel call does) never qualifies the
    label; a run stacking both spin channels in one call does.
    """
    manifold = "occ" if filled else "emp"
    return manifold if spin == SpinChannel.NONE else f"{manifold}_{spin.value}"


def _channel_manifolds(manifolds: list) -> dict:
    """Group manifold specs by spin then filling, validating the partition.

    Membership and stacking come from ``filled``/``spin`` alone; ``blocks``
    is never inspected here.

    Raises:
        ValueError: two manifolds claim the same ``(filled, spin)`` pair; a
            channel has no occupied manifold; ``spin='none'`` is mixed with
            a polarized channel; or a ``spin='down'`` channel has no
            ``spin='up'`` channel to pair it with.
    """
    by_channel: dict[SpinChannel, dict[bool, dict]] = {}
    for spec in manifolds:
        spin = SpinChannel(spec["spin"])
        filled = bool(spec["filled"])
        channel = by_channel.setdefault(spin, {})
        if filled in channel:
            raise ValueError(
                f"Two manifolds both claim filled={filled}, spin={spin.value!r}; a band "
                "structure needs at most one manifold per (filled, spin) pair."
            )
        channel[filled] = spec

    if SpinChannel.NONE in by_channel and len(by_channel) > 1:
        raise ValueError(
            "The manifolds mix spin='none' with a polarized spin channel; a band "
            "structure is either unpolarized (spin='none' only) or polarized "
            "(spin='up' / spin='down' only)."
        )
    if SpinChannel.DOWN in by_channel and SpinChannel.UP not in by_channel:
        raise ValueError(
            "The manifolds have a spin='down' channel with no spin='up' channel to pair it with."
        )
    for spin, channel in by_channel.items():
        if True not in channel:
            raise ValueError(
                f"The spin={spin.value!r} channel has an empty manifold (filled=False) but "
                "no occupied one; interpolating a band structure needs the occupied "
                "manifold in every channel it uses."
            )
    return by_channel


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
    The two channels must hold the same manifolds, so a spin-polarized
    merge takes ``occupied_down`` and ``empty_down`` together, and a merge
    with no empty manifold takes ``occupied`` alone. ``offset`` shifts
    every energy, so the returned ``reference`` (the highest occupied
    energy across the channels) is shifted by it too.
    """
    if (occupied_down is None) != (empty_down is None):
        raise ValueError(
            "A spin-polarized merge needs both `occupied_down` and `empty_down`; got one."
        )
    if empty is None and empty_down is not None:
        raise ValueError(
            "The spin channels must have the same manifolds; `empty_down` was given "
            "without `empty`."
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


class KoopmansBandStructureOutputs(TypedDict):
    """Outputs of :func:`KoopmansBandStructureTask`.

    * ``band_structure`` — the interpolated Koopmans bands along the input
      k-path, occupied then empty within each spin channel, on pw.x's
      absolute energy scale (``offset`` having been added to every
      eigenvalue; an ``offset`` of 0.0 leaves the producing route's own
      scale unchanged).
    * ``reference`` — the valence-band maximum in eV, for plot alignment.
    * ``dos`` — the bands' Gaussian-smearing total DOS, present only when
      ``do_dos``.
    """

    band_structure: orm.BandsData
    reference: float
    dos: NotRequired[DensityOfStates]


@task.graph
def KoopmansBandStructureTask(
    structure: orm.StructureData,
    koopmans_ham_retrieved: orm.FolderData,
    manifolds: list,
    block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)],
    kgrid: list[int],
    kpath: orm.KpointsData,
    smooth_block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)] | None = None,
    use_ws_distance: bool = True,
    offset: float = 0.0,
    do_dos: bool = False,
    plotting: dict | None = None,
) -> KoopmansBandStructureOutputs:
    """Interpolate a Koopmans band structure from its manifolds' Hamiltonians.

    One interpolation per manifold in ``manifolds``, off the Hamiltonian its
    own ``filename`` names inside ``koopmans_ham_retrieved``, with that
    manifold's Wannier centres; the results are concatenated
    occupied-then-empty within a channel and, when ``manifolds`` covers both
    a ``spin='up'`` and a ``spin='down'`` channel, stacked across them.
    Whether the run is polarized is read off ``manifolds`` itself — there is
    no separate flag for it.

    Args:
        structure: the primitive cell the wannierizations ran on.
        koopmans_ham_retrieved: the retrieved folder holding every
            manifold's printed Koopmans Hamiltonian.
        manifolds: one :class:`ManifoldSpec` per (filling, spin) manifold to
            interpolate. Building this list — which file names each
            manifold's Hamiltonian, and which blocks belong to it — is the
            calling route's own knowledge.
        block_wannierizations: the per-block wannierization outputs, keyed
            by block label.
        kgrid: the Monkhorst-Pack grid the Koopmans Hamiltonian lives on
            (the ΔSCF route's supercell repeat count; the DFPT route's
            kcw.x ``CONTROL.mp1-3``).
        kpath: the primitive-cell band path, in crystal coordinates.
        smooth_block_wannierizations: the same blocks Wannierized on a
            denser mesh, keyed by the same labels. Present asks for the
            smooth-interpolation correction: each manifold's DFT
            Hamiltonian is subtracted in real space and its dense
            counterpart added back in k-space. Absent interpolates the
            Koopmans Hamiltonian alone.
        use_ws_distance: whether the Wigner-Seitz distance between Wannier
            centres enters the interpolation phase, as in wannier90.
        offset: shift applied to every returned eigenvalue and to the
            reference; 0.0 leaves the producing route's own energy scale.
        do_dos: whether to also compute the bands' Gaussian-smearing DOS.
        plotting: DOS shaping — ``degauss``, ``nstep``, ``Emin``, ``Emax``.

    Raises:
        ValueError: ``manifolds`` is not a valid partition — see
            :func:`_channel_manifolds`.
    """
    channels = _channel_manifolds(manifolds)

    energies_by_manifold = {}
    for spin, by_filled in channels.items():
        for filled, spec in by_filled.items():
            label = _manifold_label(filled=filled, spin=spin)
            hamiltonian = extract_koopmans_hamiltonian(
                retrieved=koopmans_ham_retrieved,
                filename=spec["filename"],
                metadata={"call_link_label": f"extract_{label}_hamiltonian"},
            ).result
            energies_by_manifold[filled, spin] = interpolate_manifold(
                hamiltonian,
                spec["blocks"],
                label=label,
                block_wannierizations=block_wannierizations,
                smooth_block_wannierizations=smooth_block_wannierizations,
                structure=structure,
                kpath=kpath,
                kgrid=kgrid,
                use_ws_distance=use_ws_distance,
            )

    first = SpinChannel.NONE if SpinChannel.NONE in channels else SpinChannel.UP
    second = SpinChannel.DOWN if SpinChannel.DOWN in channels else None

    merge_kwargs = {
        "occupied": energies_by_manifold[True, first],
        "empty": energies_by_manifold.get((False, first)),
        "offset": offset,
    }
    if second is not None:
        merge_kwargs["occupied_down"] = energies_by_manifold[True, second]
        merge_kwargs["empty_down"] = energies_by_manifold.get((False, second))

    merged = merge_manifold_energies(
        **merge_kwargs,
        metadata={"call_link_label": "merge_manifold_energies"},
    )

    outputs = KoopmansBandStructureOutputs(
        band_structure=build_band_structure(
            kpath=kpath,
            energies=merged["energies"],
            reference=merged["reference"],
            metadata={"call_link_label": "build_band_structure"},
        ).result,
        reference=merged["reference"],
    )
    if do_dos:
        outputs["dos"] = compute_dos_from_bands(
            band_energies=merged["energies"],
            plotting=dict(plotting) if plotting is not None else {},
            metadata={"call_link_label": "interpolated_dos"},
        )
    return outputs
