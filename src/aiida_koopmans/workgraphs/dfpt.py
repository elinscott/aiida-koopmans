"""Koopmans DFPT workflow (kcw.x): wann2kc → screen → ham.

The three steps are backed by the CalcJobs in
``aiida_koopmans.calculations.kcw`` (one kcw.x binary, three
``CONTROL.calculation`` modes).

Two graphs are exposed:

* :func:`RunDFPT` -- the kcw.x chain proper. It *consumes*
  wannierization outputs (the shared nscf scratch plus the per-manifold
  wannier90 ``retrieved`` folders) and runs wann2kcw → screen → ham. When
  ``alpha_guess`` is provided the screen step is skipped and the guess is
  fed straight to ham.
* :func:`SinglepointDFPTWorkflow` -- the end-to-end workflow: one shared scf + nscf,
  one :func:`~aiida_koopmans.workgraphs.block_wannierize.WannierizeBlock`
  per manifold (occupied / empty), then :func:`RunDFPT`.

Spin handling (``SinglepointDFPTWorkflow``'s ``spin`` input, an
``aiida_quantumespresso`` ``SpinType``):

* ``NONE`` — kcw.x still requires an nspin=2 parent scratch (the DFPT
  perturbations are spin-dependent), so the PW runs are forced to
  ``nspin = 2`` + ``tot_magnetization = 0`` and pw2wannier90 to
  ``spin_component = 'up'`` (legacy ``force_nspin2``;
  ``_wannierize.py:531-532``). One kcw chain on the up channel.
* ``COLLINEAR`` — the legacy ``spin_components`` loop: per-channel
  wannierization (wannier90 ``spin``, pw2wannier90 ``spin_component``) and
  a kcw chain per channel (``CONTROL.spin_component`` 1 / 2), with the
  down-channel results in the ``_down`` output keys.
* ``NON_COLLINEAR`` / ``SPIN_ORBIT`` — spinor scratch (``noncolin``, plus
  ``lspinorb`` for SOC), ``spinors = .true.`` wannierization with doubled
  ``num_wann``, one kcw chain. No legacy equivalent; QE reference:
  ``KCW/examples/example05.1`` nspin4 variants.

Remaining scope cuts (deliberate deviations from legacy, single-manifold):

* One occupied block + at most one empty block per spin channel. Legacy
  merges multiple occupied sub-blocks (u / hr / centres merge steps) before
  kcw.x; that machinery is not ported yet, so multi-block inputs must be
  rejected upstream.
* No per-orbital screening fan-out (legacy ``i_orb`` grouping): one screen
  calculation solves all orbitals, which is legacy's own behaviour when no
  orbital grouping applies.
* No coarse-grid pre-screening (``dfpt_coarse_grid``) and no
  unfold-and-interpolate postprocessing.
"""

from __future__ import annotations

import io
from typing import Any, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType
from aiida_workgraph import task

from aiida_koopmans.calculations.kcw import (
    KcwHamCalculation,
    KcwScreenCalculation,
    Wann2kcCalculation,
)
from aiida_koopmans.types import (
    ExplicitProjectionBlock,
    ProjectionBlock,
    SpinChannel,
    SpinType,
)
from aiida_koopmans.workgraphs import Codes
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlock
from aiida_koopmans.workgraphs.pw import RunScfNscf

# kcw.x reads ``<seedname>_u.mat`` / ``<seedname>_emp_u.mat`` (etc.) from its
# working directory. The wannier90 CalcJob writes its products with the
# ``aiida`` seedname, so keeping the same seedname means the occupied-manifold
# files stage under their retrieved names unchanged.
SEEDNAME = "aiida"

# Wannier90 products each manifold must provide (suffixes appended to the
# seedname). ``_u_dis.mat`` is optional: it only exists when the manifold was
# disentangled (empty manifold with num_bands > num_wann).
_REQUIRED_SUFFIXES = ("_u.mat", "_hr.dat", "_centres.xyz")
_OPTIONAL_SUFFIXES = ("_u_dis.mat",)


Wann2kcStep = task(Wann2kcCalculation)
KcwScreenStep = task(KcwScreenCalculation)
KcwHamStep = task(KcwHamCalculation)


def projection_win_string(projection: Any) -> str:
    """Format one projection as a Wannier90 ``.win`` projections line.

    ``projection`` is duck-typed on the ``wannier90_input`` ``Projection``
    model. Element-labelled sites render as ``<element>:<ang_mtm>``;
    single-point sites use Wannier90's ``f=x,y,z`` (crystal) / ``c=x,y,z``
    (Cartesian) forms. The ``ang_mtm`` quantum numbers stringify to
    Wannier90's own syntax (``l=-3`` for sp3, ...).
    """
    if projection.site is not None:
        return f"{projection.site}:{projection.ang_mtm}"
    fractional = getattr(projection, "fractional_site", None)
    if fractional is not None:
        return f"f={','.join(str(c) for c in fractional)}:{projection.ang_mtm}"
    cartesian = getattr(projection, "cartesian_site", None)
    if cartesian is not None:
        return f"c={','.join(str(c) for c in cartesian)}:{projection.ang_mtm}"
    raise ValueError(f"Projection {projection!r} defines no site.")


def _band_range_complement(start: int, end: int, nbnd: int) -> list[int] | None:
    """Return the wannier90 ``exclude_bands`` list complementing ``[start, end]``.

    A list of band indices (not the ``.win`` range string): aiida-wannier90's
    input writer expects integers and does the range compression itself.
    """
    excluded = [*range(1, start), *range(end + 1, nbnd + 1)]
    return excluded or None


def _projection_num_wann(structure: orm.StructureData, projection: Any) -> int:
    """Count the Wannier functions of one projection: site multiplicity x (2l+1).

    ``projection`` is duck-typed on the ``wannier90_input`` ``Projection``
    model (``.site`` element label or a ``fractional_site`` /
    ``cartesian_site`` single point, ``.ang_mtm`` quantum numbers).
    """
    if projection.site is not None:
        n_sites = sum(1 for site in structure.sites if site.kind_name == projection.site)
        if n_sites == 0:
            raise ValueError(
                f"Projection site '{projection.site}' does not match any atom in the structure."
            )
    elif (
        getattr(projection, "fractional_site", None) is not None
        or getattr(projection, "cartesian_site", None) is not None
    ):
        # An explicit point hosts exactly one set of orbitals.
        n_sites = 1
    else:
        raise ValueError(f"Projection {projection!r} defines no site.")
    quantum_numbers = projection.ang_mtm
    if quantum_numbers.m_r is not None:
        multiplicity = len(quantum_numbers.m_r)
    else:
        l_value = quantum_numbers.angular.value
        # Hybrids are encoded with negative l: sp=-1 (2 orbitals), sp2=-2 (3),
        # sp3=-3 (4), sp3d=-4 (5), sp3d2=-5 (6).
        multiplicity = 2 * l_value + 1 if l_value >= 0 else 1 - l_value
    return n_sites * multiplicity


def _split_manifolds(
    blocks_with_counts: list[tuple[Any, int]], nocc: int
) -> tuple[list[tuple[Any, int]], list[tuple[Any, int]]]:
    """Split (block, num_wann) pairs at the occupied/empty boundary."""
    occupied: list[tuple[Any, int]] = []
    empty: list[tuple[Any, int]] = []
    cursor = 0
    for block, num_wann in blocks_with_counts:
        if cursor + num_wann <= nocc:
            occupied.append((block, num_wann))
        elif cursor >= nocc:
            empty.append((block, num_wann))
        else:
            raise ValueError(
                f"A projection block (bands {cursor + 1}-{cursor + num_wann}) straddles "
                f"the occupied/empty boundary at band {nocc}."
            )
        cursor += num_wann
    return occupied, empty


def _default_channel_nocc(spin_channel: SpinChannel, nelec: int) -> int:
    """Occupied-band count of a channel when the caller supplies none.

    Spinor bands are singly occupied (``nocc = nelec``); the unpolarized
    channel holds electron pairs. Collinear channels have no default — their
    occupations depend on the magnetization, which only the caller knows.
    """
    if spin_channel in (SpinChannel.UP, SpinChannel.DOWN):
        raise ValueError(
            f"spin_channel={spin_channel.value!r} needs an explicit per-channel "
            "nocc (derived from the electron count and the magnetization)."
        )
    if spin_channel == SpinChannel.SPINOR:
        return nelec
    if nelec % 2:
        raise ValueError(
            f"Odd electron count ({nelec}) requires spin='collinear', which "
            "derives per-channel occupations from the magnetization."
        )
    return nelec // 2


def derive_dfpt_manifolds(
    structure: orm.StructureData,
    projection_blocks: list,
    nelec: int,
    nbnd: int | None,
    spin_channel: SpinChannel = SpinChannel.NONE,
    nocc: int | None = None,
) -> tuple[ProjectionBlock, ProjectionBlock | None, bool, int]:
    """Turn user projection blocks into the occupied/empty DFPT manifolds.

    Ports the manifold bookkeeping of legacy ``KoopmansDFPTWorkflow.__init__``
    (nocc from the electron count, nemp from the projections, disentanglement
    when the empty manifold has more bands than Wannier functions) for one
    spin channel: exactly one occupied block, at most one empty block.

    Args:
        structure: the periodic structure (for per-site projection counting).
        projection_blocks: list of projection blocks *for this channel*, each
            a list of ``wannier90_input`` ``Projection``-like objects.
        nelec: total electron count (from the pseudopotential valences).
        nbnd: number of bands of the nscf, or None to default to nocc.
        spin_channel: which channel these blocks describe. ``NONE`` (default)
            is spin-unpolarized (``nocc = nelec / 2``); ``UP`` / ``DOWN`` are
            the collinear channels (caller must supply the per-channel
            ``nocc`` from the magnetization); ``SPINOR`` is the noncollinear
            case — every band is singly occupied (``nocc = nelec``) and each
            projection yields two spinor Wannier functions.
        nocc: per-channel occupied-band count, overriding the electron-count
            default. Required for ``UP`` / ``DOWN``.

    Returns:
        ``(occ_block, emp_block, has_disentangle, n_orbitals)`` where the
        blocks are :class:`ExplicitProjectionBlock` (``emp_block`` may be
        None) and ``n_orbitals = num_wann_occ + num_wann_emp``.
    """
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    spinor = spin_channel == SpinChannel.SPINOR
    if nocc is None:
        nocc = _default_channel_nocc(spin_channel, nelec)
    nbnd = nocc if nbnd is None else int(nbnd)

    if not projection_blocks:
        raise NotImplementedError(
            "DFPT screening requires explicit Wannier90 projections in "
            "``calculator_parameters.w90.projections``."
        )

    # With spinors (nspin=4) each projection orbital carries two spin
    # components, so a projection block spans twice as many Wannier
    # functions as its orbital count (KCW example05.1: sp3 -> num_wann 8).
    wann_per_orbital = 2 if spinor else 1
    blocks_with_counts = [
        (block, wann_per_orbital * sum(_projection_num_wann(structure, p) for p in block))
        for block in projection_blocks
    ]
    occupied, empty = _split_manifolds(blocks_with_counts, nocc)

    if len(occupied) != 1 or len(empty) > 1:
        raise NotImplementedError(
            f"DFPT screening currently supports exactly one occupied projection block "
            f"and at most one empty block (got {len(occupied)} occupied / {len(empty)} "
            "empty). Multi-block manifolds need the u/hr/centres merge machinery, "
            "which is not yet supported."
        )
    num_wann_occ = occupied[0][1]
    if num_wann_occ != nocc:
        raise ValueError(
            f"The occupied projection block spans {num_wann_occ} Wannier functions but "
            f"the system has {nocc} occupied bands."
        )

    def _projection_strings(block: list) -> list[str]:
        # Wannier90-format projection lines; aiida-wannier90 writes orm.List
        # entries verbatim into the .win projections block.
        return [projection_win_string(p) for p in block]

    label_suffix = (
        f"_{spin_channel.value}" if spin_channel in (SpinChannel.UP, SpinChannel.DOWN) else ""
    )
    occ_block = ExplicitProjectionBlock(
        label=f"occ{label_suffix}",
        spin=spin_channel,
        num_wann=num_wann_occ,
        num_bands=num_wann_occ,
        include_bands=list(range(1, nocc + 1)),
        exclude_bands=_band_range_complement(1, nocc, nbnd),
        projection_type=WannierProjectionType.ANALYTIC,
        projections=_projection_strings(occupied[0][0]),
    )

    emp_block = None
    has_disentangle = False
    num_wann_emp = 0
    if empty:
        num_wann_emp = empty[0][1]
        num_bands_emp = nbnd - nocc
        if num_bands_emp < num_wann_emp:
            raise ValueError(
                f"nbnd = {nbnd} leaves only {num_bands_emp} empty bands but the empty "
                f"projection block requires {num_wann_emp} Wannier functions."
            )
        has_disentangle = num_bands_emp != num_wann_emp
        emp_block = ExplicitProjectionBlock(
            label=f"emp{label_suffix}",
            spin=spin_channel,
            num_wann=num_wann_emp,
            num_bands=num_bands_emp,
            include_bands=list(range(nocc + 1, nbnd + 1)),
            exclude_bands=_band_range_complement(nocc + 1, nbnd, nbnd),
            projection_type=WannierProjectionType.ANALYTIC,
            projections=_projection_strings(empty[0][0]),
        )

    return occ_block, emp_block, has_disentangle, num_wann_occ + num_wann_emp


def normalize_alpha_guess(
    raw_guess: float | list,
    n_orbitals: int,
    spin_channel: SpinChannel = SpinChannel.NONE,
) -> list[float]:
    """Flatten a user ``alpha_guess`` into one alpha per orbital.

    Accepts the three shapes the input file allows: a single float (uniform
    guess), a flat list, or the nested per-spin list (``spin_channel.index``
    selects the channel: up/none/spinor take the first entry, down the
    second).
    """
    if isinstance(raw_guess, float):
        return [raw_guess] * n_orbitals
    if raw_guess and isinstance(raw_guess[0], list):
        return [float(a) for a in raw_guess[spin_channel.index]]
    return [float(a) for a in raw_guess]


@task
def alphas_from_guess(alpha_guess: list) -> list:
    """Materialise a caller-provided screening-parameter guess.

    Runs as a named ``@task`` (rather than passing the raw list around) so
    the guess becomes a provenance node and a socket that both the ham step
    and the graph outputs can consume (raw Python values are not valid graph
    return payloads).
    """
    return list(alpha_guess)


class KoopmansDFPTOutputs(TypedDict, total=False):
    """Outputs of :func:`RunDFPT` / :func:`SinglepointDFPTWorkflow`.

    * ``alphas`` -- the screening parameters fed to the ham step (computed by
      screen, or the caller's guess when screening was skipped).
    * ``screen_parameters`` -- screen-step scalars (:class:`KcwScreenParameters`;
      absent when screening was skipped).
    * ``ham_parameters`` -- ham-step scalars (:class:`KcwHamParameters`),
      including the KS / KI eigenvalues on the k-grid.
    * ``bands`` -- interpolated Koopmans band structure (present only when a
      band path was supplied).
    * ``wann2kc_remote_folder`` -- the wann2kcw scratch, for chaining further
      kcw.x runs off the same conversion.

    ``screen_parameters`` / ``ham_parameters`` carry the key sets documented
    by :class:`KcwScreenParameters` / :class:`KcwHamParameters`; they are
    annotated as plain ``dict`` here because a TypedDict annotation on a
    ``@task.graph`` output is read as a nested namespace socket rather than a
    leaf ``orm.Dict``.

    A spin-polarized (collinear) :func:`SinglepointDFPTWorkflow` runs the kcw.x
    chain once per channel: the unsuffixed keys carry the spin-up results
    and the ``_down`` variants carry spin-down. Unpolarized and spinor runs
    have a single chain and never populate the ``_down`` keys.
    """

    alphas: orm.List
    screen_parameters: dict
    ham_parameters: dict
    bands: orm.BandsData
    wann2kc_remote_folder: orm.RemoteData
    alphas_down: orm.List
    screen_parameters_down: dict
    ham_parameters_down: dict
    bands_down: orm.BandsData
    wann2kc_remote_folder_down: orm.RemoteData


@task.calcfunction(outputs=["wannier_files"])
def prepare_kcw_wannier_files(
    occ_retrieved: orm.FolderData,
    emp_retrieved: orm.FolderData = None,
) -> dict:
    """Assemble the ``wannier_files`` folder the kcw.x CalcJobs stage.

    Collects the Wannier90 products out of the per-manifold ``retrieved``
    folders (requires the wannier90 runs to have set ``write_u_matrices``
    and ``write_xyz``) and renames the empty-manifold files to kcw.x's
    hard-coded ``<seedname>_emp_*`` convention.
    """
    merged = orm.FolderData()

    def _copy(src: orm.FolderData, rename_emp: bool) -> None:
        names = set(src.base.repository.list_object_names())
        manifold = "empty" if rename_emp else "occupied"
        for suffix in _REQUIRED_SUFFIXES + _OPTIONAL_SUFFIXES:
            src_name = f"{SEEDNAME}{suffix}"
            if src_name not in names:
                if suffix in _OPTIONAL_SUFFIXES:
                    continue
                raise ValueError(
                    f"``{src_name}`` is missing from the {manifold}-manifold wannier90 "
                    "retrieved folder. The wannier90 runs feeding a DFPT chain must set "
                    "``write_u_matrices = True`` and ``write_xyz = True``."
                )
            dst_name = f"{SEEDNAME}_emp{suffix}" if rename_emp else src_name
            content = src.base.repository.get_object_content(src_name, mode="rb")
            merged.base.repository.put_object_from_filelike(io.BytesIO(content), dst_name)

    _copy(occ_retrieved, rename_emp=False)
    if emp_retrieved is not None:
        _copy(emp_retrieved, rename_emp=True)

    return {"wannier_files": merged}


@task.graph
def RunDFPT(
    codes: Codes,
    nscf_remote_folder: orm.RemoteData,
    occ_retrieved: orm.FolderData,
    num_wann_occ: int,
    num_wann_emp: int,
    kgrid: list[int],
    emp_retrieved: orm.FolderData | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    eps_inf: float | None = None,
    alpha_guess: list[float] | None = None,
    has_disentangle: bool = False,
    l_vcut: bool | None = None,
    spin_component: int = 1,
) -> KoopmansDFPTOutputs:
    """Run the kcw.x chain off provided wannierization outputs.

    Args:
        codes: code instances; only ``codes["kcw"]`` is used.
        nscf_remote_folder: scratch of the pw.x **nscf** run the Wannier
            functions were built on (kcw.x re-reads its wavefunctions). Must
            be an ``nspin = 2`` run even for closed-shell systems -- the DFPT
            perturbations are spin-dependent; the kcw chain reads the up
            channel (``CONTROL.spin_component = 1``).
        occ_retrieved: the occupied-manifold wannier90 ``retrieved`` folder
            (must hold ``aiida_u.mat`` / ``aiida_hr.dat`` /
            ``aiida_centres.xyz``).
        num_wann_occ / num_wann_emp: Wannier function counts per manifold
            (``num_wann_emp = 0`` for an occupied-only run).
        kgrid: the Monkhorst-Pack grid of the nscf, for ``CONTROL.mp1-3``.
        emp_retrieved: the empty-manifold wannier90 ``retrieved`` folder.
        bands_kpoints: explicit k-path; when given, the ham step interpolates
            the Koopmans Hamiltonian along it (``HAM.do_bands``).
        eps_inf: macroscopic dielectric constant for the screen step's
            long-range corrections.
        alpha_guess: when given, skip the screen step and feed these alphas
            straight to ham.
        has_disentangle: whether the empty manifold was disentangled
            (``num_bands != num_wann``).
        l_vcut: Gygi-Baldereschi long-range cutoff (legacy ``gb_correction``);
            None means the periodic-system default (on).
        spin_component: which collinear spin channel kcw.x reads (1 = up,
            2 = down). Spin-unpolarized runs use the default 1 (the nspin=2
            scratch's channels are identical); a spin-polarized workflow
            calls this task once per channel. Ignored by kcw.x for
            noncollinear scratches.
    """
    # ``bool()`` unwraps a possible wrapt proxy (a TaggedValue graph input)
    # to a plain bool before it lands in the stored ``control`` Dict.
    l_vcut = True if l_vcut is None else bool(l_vcut)
    control = {
        "kcw_iverbosity": 1,
        "kcw_at_ks": False,
        "read_unitary_matrix": True,
        "lrpa": False,
        "l_vcut": l_vcut,
        "spin_component": spin_component,
        "mp1": kgrid[0],
        "mp2": kgrid[1],
        "mp3": kgrid[2],
    }
    wannier = {
        "seedname": SEEDNAME,
        "check_ks": True,
        "num_wann_occ": num_wann_occ,
        "num_wann_emp": num_wann_emp,
        "have_empty": num_wann_emp > 0,
        "has_disentangle": has_disentangle,
    }

    prep_inputs: dict[str, Any] = {"occ_retrieved": occ_retrieved}
    if emp_retrieved is not None:
        prep_inputs["emp_retrieved"] = emp_retrieved
    wannier_files = prepare_kcw_wannier_files(
        **prep_inputs,
        metadata={"call_link_label": "prepare_kcw_wannier_files"},
    )["wannier_files"]

    wann2kc = Wann2kcStep(
        code=codes["kcw"],
        parameters={"CONTROL": control, "WANNIER": wannier},
        parent_folder=nscf_remote_folder,
        wannier_files=wannier_files,
        metadata={"call_link_label": "wann2kc"},
    )

    outputs = KoopmansDFPTOutputs(wann2kc_remote_folder=wann2kc["remote_folder"])

    if alpha_guess is None:
        # Screen defaults: tight tr2, spread check.
        screen_namelist: dict[str, Any] = {
            "tr2": 1.0e-18,
            "nmix": 4,
            "niter": 33,
            "check_spread": True,
        }
        if eps_inf is not None:
            screen_namelist["eps_inf"] = eps_inf
        screen = KcwScreenStep(
            code=codes["kcw"],
            parameters={"CONTROL": control, "WANNIER": wannier, "SCREEN": screen_namelist},
            parent_folder=wann2kc["remote_folder"],
            wannier_files=wannier_files,
            metadata={"call_link_label": "screen"},
        )
        alphas = screen["alphas"]
        outputs["screen_parameters"] = screen["output_parameters"]
    else:
        alphas = alphas_from_guess(
            alpha_guess=list(alpha_guess),
            metadata={"call_link_label": "alphas_from_guess"},
        ).result

    do_bands = bands_kpoints is not None
    ham_namelist = {
        "do_bands": do_bands,
        "use_ws_distance": True,
        "write_hr": True,
        "on_site_only": False,
    }
    ham_inputs: dict[str, Any] = {
        "code": codes["kcw"],
        "parameters": {"CONTROL": control, "WANNIER": wannier, "HAM": ham_namelist},
        "parent_folder": wann2kc["remote_folder"],
        "wannier_files": wannier_files,
        "alphas": alphas,
        "metadata": {"call_link_label": "ham"},
    }
    if do_bands:
        ham_inputs["kpoints"] = bands_kpoints
    ham = KcwHamStep(**ham_inputs)

    outputs["alphas"] = alphas
    outputs["ham_parameters"] = ham["output_parameters"]
    if do_bands:
        outputs["bands"] = ham["bands"]
    return outputs


def _pw_spin_system_defaults(spin: SpinType) -> dict[str, Any]:
    """Return the SYSTEM-namelist keys a DFPT chain forces on the PW runs.

    * Unpolarized: kcw.x still requires an nspin=2 scratch (the DFPT
      perturbations are spin-dependent — legacy ``force_nspin2``; the
      tutorial_3 scf/nscf.pwi carry ``nspin=2 + tot_magnetization=0``).
    * Collinear: nspin=2 without pinning the magnetization — the caller's
      overrides carry the physical ``tot_magnetization`` /
      ``starting_magnetization``.
    * Noncollinear / spin-orbit: spinor wavefunctions (``noncolin``), plus
      ``lspinorb`` for SOC (QE reference: KCW/examples/example05.1 nspin4).
    """
    if spin == SpinType.COLLINEAR:
        return {"nspin": 2}
    if spin == SpinType.NON_COLLINEAR:
        return {"noncolin": True}
    if spin == SpinType.SPIN_ORBIT:
        return {"noncolin": True, "lspinorb": True}
    return {"nspin": 2, "tot_magnetization": 0}


def _channel_w90_defaults(spin: SpinType, channel: SpinChannel) -> dict[str, Any]:
    """Return the per-channel wannierization overrides a DFPT chain forces on.

    kcw.x reads the U matrices and Wannier centres from files the wannier90
    runs only write on request (``write_u_matrices`` / ``write_xyz``). With a
    collinear scratch, pw2wannier90 must pick its channel explicitly and the
    wannier90 input selects the same channel via ``spin`` (KCW example05.1
    nspin2); a spinor scratch instead needs ``spinors = .true.`` and no
    channel selection (nspin4 variants).

    These must be explicit overrides rather than upstream's
    ``spin_type`` machinery: ``Wannier90WorkChain`` injects
    ``spin_component`` at runtime by detecting nspin=2 from its *own*
    scf/nscf inputs, which :func:`BlockWannierize` deliberately omits
    (shared-nscf pattern), so the upstream path can never fire here.
    """
    w90_params: dict[str, Any] = {"write_u_matrices": True, "write_xyz": True}
    defaults: dict[str, Any] = {"wannier90": {"wannier90": {"parameters": w90_params}}}
    if spin in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
        w90_params["spinors"] = True
        return defaults
    if spin == SpinType.COLLINEAR:
        w90_params["spin"] = channel.value
    defaults["pw2wannier90"] = {
        "pw2wannier90": {
            "parameters": {
                "INPUTPP": {"spin_component": "down" if channel == SpinChannel.DOWN else "up"}
            }
        },
    }
    return defaults


@task.graph
def SinglepointDFPTWorkflow(
    codes: Codes,
    structure: orm.StructureData,
    occ_block: ProjectionBlock,
    kpoints: orm.KpointsData,
    kgrid: list[int],
    emp_block: ProjectionBlock | None = None,
    occ_block_down: ProjectionBlock | None = None,
    emp_block_down: ProjectionBlock | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    eps_inf: float | None = None,
    alpha_guess: list[float] | None = None,
    alpha_guess_down: list[float] | None = None,
    has_disentangle: bool = False,
    has_disentangle_down: bool = False,
    l_vcut: bool | None = None,
    spin: SpinType = SpinType.NONE,
) -> KoopmansDFPTOutputs:
    """End-to-end singlepoint Koopmans DFPT: wannierize, then the kcw.x chain.

    One shared scf + nscf (:func:`RunScfNscf`, with the spin-regime SYSTEM
    keys of :func:`_pw_spin_system_defaults` forced on and ``nosym`` /
    ``noinv`` on the nscf so kcw.x sees the full k-point set), one
    :func:`WannierizeBlock` per manifold and channel, then one
    :func:`RunDFPT` per channel:

    * ``spin = NONE`` — one chain on the up channel of the closed-shell
      nspin=2 scratch (``occ_block`` / ``emp_block``; legacy behaviour).
    * ``spin = COLLINEAR`` — two chains (legacy ``spin_components`` loop):
      ``occ_block`` / ``emp_block`` / ``alpha_guess`` / ``has_disentangle``
      describe spin-up, their ``_down`` twins spin-down, and the kcw runs
      carry ``CONTROL.spin_component`` 1 / 2. The caller's ``overrides``
      must supply the magnetization (``tot_magnetization`` or
      ``starting_magnetization``). Down-channel results land in the
      ``_down`` output keys.
    * ``spin = NON_COLLINEAR`` / ``SPIN_ORBIT`` — one chain on the spinor
      scratch; the blocks must be spinor manifolds (``num_wann`` doubled,
      from ``derive_dfpt_manifolds(..., spin_channel=SPINOR)``).

    ``overrides`` namespaces: ``"scf"`` / ``"nscf"`` feed the shared PW
    steps, ``"wannier90"`` feeds every per-manifold wannier builder (its
    ``"pw2wannier90"`` sub-namespace reaches the pw2wannier90 step).
    """
    from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

    overrides = overrides or {}
    collinear = spin == SpinType.COLLINEAR

    spin_defaults: dict[str, Any] = {
        "pw": {"parameters": {"SYSTEM": _pw_spin_system_defaults(spin)}},
    }
    nscf_defaults: dict[str, Any] = recursive_merge(
        spin_defaults,
        {"pw": {"parameters": {"SYSTEM": {"nosym": True, "noinv": True}}}},
    )
    # Forced keys merge *on top of* caller overrides: the forced nspin=2
    # overwrites any user-supplied nspin, since a user nspin=1 would
    # silently break kcw.x.
    scf_nscf_overrides: dict[str, Any] = {
        "scf": recursive_merge(overrides.get("scf", {}), spin_defaults),
        "nscf": recursive_merge(overrides.get("nscf", {}), nscf_defaults),
    }

    # wannier90 / pw2wannier90 need the nscf eigenstates on the full
    # (symmetry-unreduced) user grid, listed in wannier90's own k-point
    # order — expand the mesh once and share the explicit list between the
    # nscf and every per-block wannierisation. ``mp_grid`` keeps the mesh
    # dimensions, which wannier90 cannot re-derive from an explicit list.
    from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints

    mp_grid = kpoints.get_kpoints_mesh()[0]
    explicit_kpoints = get_explicit_kpoints(kpoints)

    scf_nscf = RunScfNscf(
        code=codes["pw"],
        structure=structure,
        pseudo_family=pseudo_family,
        protocol=protocol,
        overrides=scf_nscf_overrides,
        nscf_kpoints=explicit_kpoints,
        # kcw.x refuses non-fixed occupations ("KC corrections only for
        # insulators"), so the ground state must run as an insulator.
        electronic_type=ElectronicType.INSULATOR,
        metadata={"call_link_label": "scf_nscf"},
    )
    nscf_remote_folder = scf_nscf["nscf_remote_folder"]

    channels: list[dict[str, Any]] = [
        {
            "channel": SpinChannel.UP if collinear else SpinChannel.NONE,
            "suffix": "_up" if collinear else "",
            "occ_block": occ_block,
            "emp_block": emp_block,
            "alpha_guess": alpha_guess,
            "has_disentangle": has_disentangle,
            "spin_component": 1,
            "out_key": lambda key: key,
        }
    ]
    if collinear:
        if occ_block_down is None:
            raise ValueError("spin='collinear' requires occ_block_down.")
        channels.append(
            {
                "channel": SpinChannel.DOWN,
                "suffix": "_down",
                "occ_block": occ_block_down,
                "emp_block": emp_block_down,
                "alpha_guess": alpha_guess_down,
                "has_disentangle": has_disentangle_down,
                "spin_component": 2,
                "out_key": lambda key: f"{key}_down",
            }
        )

    outputs = KoopmansDFPTOutputs()
    for ch in channels:
        # The staging files and the channel selection are requirements of the
        # kcw chain, not defaults a caller may disable: force-merge them on
        # top of the caller's wannier90 overrides.
        wannier_overrides = recursive_merge(
            overrides.get("wannier90", {}), _channel_w90_defaults(spin, ch["channel"])
        )

        ch_occ_block = ch["occ_block"]
        occ = WannierizeBlock(
            codes=codes,
            structure=structure,
            block=ch_occ_block,
            projection_type=ch_occ_block["projection_type"],
            nscf_remote_folder=nscf_remote_folder,
            kpoints=explicit_kpoints,
            mp_grid=mp_grid,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides=wannier_overrides,
            metadata={"call_link_label": f"wannierize_occ{ch['suffix']}"},
        )

        dfpt_inputs: dict[str, Any] = {
            "codes": codes,
            "nscf_remote_folder": nscf_remote_folder,
            "occ_retrieved": occ["hr_retrieved"],
            "num_wann_occ": ch_occ_block["num_wann"],
            "num_wann_emp": 0,
            "kgrid": kgrid,
            "bands_kpoints": bands_kpoints,
            "eps_inf": eps_inf,
            "alpha_guess": ch["alpha_guess"],
            "has_disentangle": ch["has_disentangle"],
            "l_vcut": l_vcut,
            "spin_component": ch["spin_component"],
            "metadata": {"call_link_label": f"dfpt{ch['suffix']}"},
        }

        ch_emp_block = ch["emp_block"]
        if ch_emp_block is not None:
            emp = WannierizeBlock(
                codes=codes,
                structure=structure,
                block=ch_emp_block,
                projection_type=ch_emp_block["projection_type"],
                nscf_remote_folder=nscf_remote_folder,
                kpoints=explicit_kpoints,
                mp_grid=mp_grid,
                pseudo_family=pseudo_family,
                protocol=protocol,
                overrides=wannier_overrides,
                metadata={"call_link_label": f"wannierize_emp{ch['suffix']}"},
            )
            dfpt_inputs["emp_retrieved"] = emp["hr_retrieved"]
            dfpt_inputs["num_wann_emp"] = ch_emp_block["num_wann"]

        dfpt = RunDFPT(**dfpt_inputs)

        out_key = ch["out_key"]
        outputs[out_key("alphas")] = dfpt["alphas"]
        outputs[out_key("ham_parameters")] = dfpt["ham_parameters"]
        outputs[out_key("wann2kc_remote_folder")] = dfpt["wann2kc_remote_folder"]
        if ch["alpha_guess"] is None:
            outputs[out_key("screen_parameters")] = dfpt["screen_parameters"]
        if bands_kpoints is not None:
            outputs[out_key("bands")] = dfpt["bands"]

    return outputs
