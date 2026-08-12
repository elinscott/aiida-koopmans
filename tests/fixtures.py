"""Shared test data, helper classes, and pytest fixtures.

Definitions live here; ``conftest.py`` just re-exports the fixtures so
pytest's collection machinery picks them up for every test module.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def sanitize(value):
    """Recursively convert numbers to ``float``/``int`` so YAML output is stable.

    Shared by the parser regression tests that snapshot ``output_parameters``
    dicts with ``data_regression``.
    """
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        # Round to 8 sig figs — legacy .cpo stdout floats have only ~6-10
        # significant digits depending on the printf format, and a tighter
        # comparison would flake on trivial last-bit differences.
        if np.isnan(value):
            return float("nan")
        return float(f"{value:.8g}")
    return value


# Ozone geometry taken from koopmans/tutorials/tutorial_1/ozone.json.
_OZONE_CELL = [[14.1738, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.66]]
_OZONE_POSITIONS = [
    ("O", [7.0869, 6.0, 5.89]),
    ("O", [8.1738, 6.0, 6.55]),
    ("O", [6.0, 6.0, 6.55]),
]


class _FakeUpf(SimpleNamespace):
    """Stand-in for an ``aiida-pseudo`` ``UpfData`` node.

    Exposes only the attributes our rendering / electron-counting helpers read
    (``filename``, ``uuid``, ``z_valence``).
    """


def _build_ozone_structure(pbc: bool):
    from aiida.orm import StructureData

    struct = StructureData(cell=_OZONE_CELL, pbc=pbc)
    for symbol, position in _OZONE_POSITIONS:
        struct.append_atom(position=position, symbols=symbol, name=symbol)
    return struct


@pytest.fixture
def ozone_structure(aiida_profile):
    """Return an ozone (O3) ``StructureData`` with the tutorial_1 geometry, non-periodic."""
    return _build_ozone_structure(pbc=False)


@pytest.fixture
def periodic_ozone_structure(aiida_profile):
    """Return the ozone geometry with ``pbc=True`` for exercising periodic scope guards."""
    return _build_ozone_structure(pbc=True)


@pytest.fixture
def fake_upf():
    """Return a factory class for stand-in UpfData objects.

    Usage in tests::

        def test_something(fake_upf):
            upf = fake_upf(filename="O.upf", uuid="abc", z_valence=6.0)
    """
    return _FakeUpf


@pytest.fixture
def ozone_pseudos(fake_upf):
    """Return the ozone pseudos dict ``{"O": FakeUpf(...)}`` with oxygen's valence."""
    return {"O": fake_upf(filename="O.upf", uuid="fake-upf-uuid", z_valence=6.0)}


@pytest.fixture
def generate_upf_data(aiida_profile):
    """Return a factory producing real (stored) ``UpfData`` nodes for parser/CalcJob tests.

    Mirrors ``aiida-quantumespresso.tests.conftest.generate_upf_data``. The
    stream content is a minimal valid UPF v2 header so the pseudo family
    loader won't reject it during import.
    """
    import io

    from aiida_pseudo.data.pseudo.upf import UpfData

    def _generate_upf_data(
        element: str, z_valence: float = 6.0, number_of_wfc: int | None = 2
    ) -> UpfData:
        # Shaped for the line-based block extractors in
        # aiida-wannier90-workflows' pseudo utilities: ``<PP_HEADER`` and its
        # ``/>`` sit on their own lines (sharing a line with the ``<UPF>``
        # root tag loses the attributes), ``has_so`` is required, and
        # ``PP_PSWFC`` provides an s+p valence so projection counting works.
        # ``number_of_wfc`` is what the projected-DOS gate reads; ``None``
        # omits the attribute and the ``PP_PSWFC`` block (a pseudo without
        # atomic wavefunctions).
        wfc_attribute = "" if number_of_wfc is None else f'number_of_wfc="{number_of_wfc}"\n'
        pswfc_block = (
            ""
            if number_of_wfc is None
            else '<PP_PSWFC>\n<PP_CHI.1 l="0"/>\n<PP_CHI.2 l="1"/>\n</PP_PSWFC>\n'
        )
        content = (
            f'<UPF version="2.0.1">\n'
            f'<PP_HEADER\nelement="{element}"\n'
            f'z_valence="{z_valence}"\nhas_so="F"\n{wfc_attribute}/>\n'
            f"{pswfc_block}"
            f"</UPF>\n"
        )
        stream = io.BytesIO(content.encode("utf-8"))
        return UpfData(stream, filename=f"{element}.upf")

    return _generate_upf_data


@pytest.fixture
def ozone_real_pseudos(generate_upf_data):
    """Return ``{"O": UpfData}`` with a real (AiiDA-storable) UpfData node for oxygen."""
    return {"O": generate_upf_data("O", z_valence=6.0)}


@pytest.fixture
def generate_full_upf_data(aiida_profile):
    """Return a factory producing a fully-parseable ``UpfData`` (real core-correction flag).

    ``generate_upf_data``'s minimal header omits fields ``upf_to_json``
    requires (``number_of_proj``, ``mesh_size``, ...), so its
    ``core_correction`` never actually parses — every caller falls into the
    "unparseable UPF" fallback. This factory fills in every section
    ``upf_to_json`` reads for a norm-conserving, non-ultrasoft, no-spin-orbit
    pseudo, so ``core_correction`` resolves to the value it declares.
    """
    import io

    from aiida_pseudo.data.pseudo.upf import UpfData

    def _generate_full_upf_data(
        element: str, *, z_valence: float = 6.0, core_correction: bool = False
    ) -> UpfData:
        nlcc = '<PP_NLCC size="3">0.0 0.0 0.0</PP_NLCC>' if core_correction else ""
        content = (
            f'<UPF version="2.0.1">\n'
            f"<PP_HEADER\n"
            f'element="{element}"\n'
            f'z_valence="{z_valence}"\nhas_so="F"\n'
            f'core_correction="{"T" if core_correction else "F"}"\n'
            f'pseudo_type="NC"\nmesh_size="3"\n'
            f'is_ultrasoft="F"\nnumber_of_proj="1"\nnumber_of_wfc="2"\n/>\n'
            f"<PP_MESH>\n"
            f'<PP_R size="3">0.0 0.1 0.2</PP_R>\n'
            f"</PP_MESH>\n"
            f"{nlcc}\n"
            f'<PP_LOCAL size="3">0.0 0.0 0.0</PP_LOCAL>\n'
            f"<PP_NONLOCAL>\n"
            f'<PP_BETA.1 angular_momentum="0">0.0 0.0 0.0</PP_BETA.1>\n'
            f"<PP_DIJ>0.0</PP_DIJ>\n"
            f"</PP_NONLOCAL>\n"
            f"<PP_PSWFC>\n"
            f'<PP_CHI.1 l="0" occupation="2.0">0.0 0.0 0.0</PP_CHI.1>\n'
            f'<PP_CHI.2 l="1" occupation="4.0">0.0 0.0 0.0</PP_CHI.2>\n'
            f"</PP_PSWFC>\n"
            f'<PP_RHOATOM size="3">0.0 0.0 0.0</PP_RHOATOM>\n'
            f"</UPF>\n"
        )
        stream = io.BytesIO(content.encode("utf-8"))
        return UpfData(stream, filename=f"{element}.upf")

    return _generate_full_upf_data


@pytest.fixture
def fake_cutoffs_family(aiida_profile, generate_upf_data):
    """Install a fake ``CutoffsPseudoPotentialFamily`` (Si and O).

    For graph builders that call ``get_builder_from_protocol`` eagerly at
    build time: the protocol machinery only accepts SSSP / PseudoDojo /
    cutoffs families — a plain ``PseudoPotentialFamily`` is not found.
    """
    return install_cutoffs_family(
        "FAKE/CUTOFFS/PBE/SR",
        [generate_upf_data(element, z_valence=z) for element, z in (("Si", 4.0), ("O", 6.0))],
    )


def install_cutoffs_family(label, pseudos):
    """Install (or fetch) a ``CutoffsPseudoPotentialFamily`` over ``pseudos``."""
    from aiida.common.exceptions import NotExistent
    from aiida_pseudo.groups.family import CutoffsPseudoPotentialFamily

    try:
        return CutoffsPseudoPotentialFamily.collection.get(label=label)
    except NotExistent:
        pass

    family = CutoffsPseudoPotentialFamily(label=label)
    family.store()
    family.add_nodes([pseudo.store() for pseudo in pseudos])
    family.set_cutoffs(
        {pseudo.element: {"cutoff_wfc": 30.0, "cutoff_rho": 240.0} for pseudo in pseudos},
        stringency="normal",
    )
    return family


@pytest.fixture
def fake_family_without_pswfc(aiida_profile, generate_upf_data):
    """Install a cutoffs family whose pseudos carry no ``PP_PSWFC`` block."""
    return install_cutoffs_family(
        "FAKE/NOPSWFC/PBE/SR",
        [
            generate_upf_data(element, z_valence=z, number_of_wfc=None)
            for element, z in (("Si", 4.0), ("O", 6.0))
        ],
    )


@pytest.fixture
def fake_family_unreadable_upf(aiida_profile):
    """Install a cutoffs family whose Si UPF header upf-tools cannot parse.

    The ``PP_HEADER`` element never closes, so the header-only reader
    raises; aiida-pseudo's own regex-based element / z_valence extraction
    still succeeds, so the node stores and the family installs.
    """
    import io

    from aiida_pseudo.data.pseudo.upf import UpfData

    content = (
        '<UPF version="2.0.1">\n<PP_HEADER\nelement="Si"\nz_valence="4.0"\nhas_so="F"\n</UPF>\n'
    )
    upf = UpfData(io.BytesIO(content.encode("utf-8")), filename="Si.upf")
    return install_cutoffs_family("FAKE/BROKEN/PBE/SR", [upf])


@pytest.fixture
def silicon_structure(aiida_profile):
    """Return a 2-atom periodic silicon ``StructureData``."""
    from aiida.orm import StructureData

    cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
    struct = StructureData(cell=cell, pbc=True)
    struct.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
    struct.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si", name="Si")
    return struct


@pytest.fixture
def kmesh(aiida_profile):
    """Return a 2x2x2 ``KpointsData`` mesh."""
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints_mesh([2, 2, 2])
    return kpts


@pytest.fixture
def denser_kmesh(aiida_profile):
    """Return a 4x4x4 ``KpointsData`` mesh, for a step that samples finer."""
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints_mesh([4, 4, 4])
    return kpts


@pytest.fixture
def kpath(aiida_profile):
    """Return a short explicit k-path ``KpointsData``."""
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return kpts


@pytest.fixture
def labelled_kpath(aiida_profile):
    """Return a short explicit k-path ``KpointsData`` carrying labels.

    The shape wannier90 band interpolation requires: the calculation's
    ``bands_kpoints`` port rejects a path without ``labels``.
    """
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
        labels=[(0, "GAMMA"), (2, "X")],
    )
    return kpts


@pytest.fixture
def auto_codes(aiida_localhost):
    """Return stand-in codes for split-mode construction-only builds.

    Covers the codes the automated block-splitting flow requires (``pw``,
    ``wannier90``, ``pw2wannier90`` and the ``wannierjl`` julia stand-in);
    the codes never execute.
    """
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    def _code(label: str, entry_point: str):
        try:
            return InstalledCode.collection.get(label=label)
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=aiida_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point,
            ).store()

    return {
        "pw": _code("aw-pw", "quantumespresso.pw"),
        "wannier90": _code("aw-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("aw-p2w", "quantumespresso.pw2wannier90"),
        "wannierjl": _code("aw-wjl", "wannierjl.check_neighbors"),
    }


def explicit_block(
    label,
    wannier_indices,
    projections=None,
    spin=None,
    filled=None,
    num_bands=None,
    exclude_bands=None,
):
    """Build a minimal explicit (ANALYTIC) projection block.

    The block maps ``num_bands`` Bloch states onto the Wannier functions
    ``wannier_indices`` names, one index per Wannier function. The
    indices are not stored, so they are expressed the way production
    blocks express them: everything below the first is excluded, which
    puts the block's own bands at the bottom of what it reads. ``filled``
    stamps the occupancy; ``None`` leaves it unstamped, the state a block
    is in before anything has classified it. ``num_bands`` beyond
    ``len(wannier_indices)`` makes the block require disentanglement,
    reading that many extra bands above its own. Pass ``exclude_bands``
    to override the exclusion outright.
    """
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    from aiida_koopmans.projections import ExplicitProjectionBlock
    from aiida_koopmans.spin import SpinChannel

    wannier_indices = list(wannier_indices)
    num_wann = len(wannier_indices)
    block = ExplicitProjectionBlock(
        label=label,
        spin=SpinChannel.NONE if spin is None else spin,
        num_wann=num_wann,
        num_bands=num_wann if num_bands is None else num_bands,
        projection_type=WannierProjectionType.ANALYTIC,
        projections=[] if projections is None else projections,
    )
    below = list(range(1, wannier_indices[0]))
    if exclude_bands is not None:
        block["exclude_bands"] = list(exclude_bands)
    elif below:
        block["exclude_bands"] = below
    if filled is not None:
        block["filled"] = filled
    return block


def bands_data(array):
    """Wrap an eigenvalue array (2D or 3D) in a ``BandsData``."""
    from aiida.orm import BandsData, KpointsData

    array = np.asarray(array, dtype=float)
    nkpts = array.shape[-2]
    kpts = KpointsData()
    kpts.set_kpoints([[i / max(nkpts, 1), 0.0, 0.0] for i in range(nkpts)])
    bands = BandsData()
    bands.set_kpointsdata(kpts)
    bands.set_bands(array)
    return bands


def count_pw_bands_runs(wg):
    """Count the graph's pw steps that declare ``calculation = 'bands'``.

    Counting tasks *named* ``bands`` is vacuous: aiida-workgraph uniquifies
    colliding task names, so a duplicated run shows up as ``bands1`` and
    the name count stays at 1. The declared ``CONTROL.calculation`` on the
    step's own ``pw`` namespace cannot be disguised that way.
    """
    count = 0
    for task_ in wg.tasks:
        try:
            parameters = task_.inputs["pw"]["parameters"].value
        except (AttributeError, KeyError, TypeError):
            continue
        if parameters is None:
            continue
        parameters = parameters.get_dict() if hasattr(parameters, "get_dict") else dict(parameters)
        if parameters.get("CONTROL", {}).get("calculation") == "bands":
            count += 1
    return count


def assert_graph_roundtrips(wg):
    """Assert a built WorkGraph survives ``to_dict`` -> ``from_dict``.

    ``wg.run()`` and the daemon reconstruct the graph through exactly this
    round-trip before executing anything, so wiring that fails it dies at
    run start even though construction succeeded (e.g. per-key links into
    the entries of a typed dynamic output namespace).
    """
    from aiida_workgraph import WorkGraph

    WorkGraph.from_dict(wg.to_dict())


def automatic_block(label, wannier_indices, spin=None, projection_type=None, filled=None):
    """Build a minimal automatic projection block.

    Defaults to pseudoatomic projectors (``ATOMIC_PROJECTORS_QE``) — the
    projection source automated wannierization always uses, and the one
    whose occupancy is unknown until the runtime split, so ``filled``
    defaults to unstamped. ``wannier_indices`` names the block's Wannier
    functions; as in :func:`explicit_block` the indices are expressed by
    excluding every band below them.
    """
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    from aiida_koopmans.projections import AutomaticProjectionBlock
    from aiida_koopmans.spin import SpinChannel

    if projection_type is None:
        projection_type = WannierProjectionType.ATOMIC_PROJECTORS_QE
    wannier_indices = list(wannier_indices)
    num_wann = len(wannier_indices)
    block = AutomaticProjectionBlock(
        label=label,
        spin=SpinChannel.NONE if spin is None else spin,
        num_wann=num_wann,
        num_bands=num_wann,
        projection_type=projection_type,
    )
    if wannier_indices[0] > 1:
        block["exclude_bands"] = list(range(1, wannier_indices[0]))
    if filled is not None:
        block["filled"] = filled
    return block


@pytest.fixture
def nscf_remote(aiida_localhost, tmp_path):
    """Return a stand-in nscf scratch ``RemoteData`` (never read; construction-only)."""
    from aiida.orm import RemoteData

    return RemoteData(computer=aiida_localhost, remote_path=str(tmp_path)).store()


@pytest.fixture
def scf_remote(aiida_localhost, tmp_path):
    """Return a stand-in scf scratch ``RemoteData`` (never read; construction-only)."""
    from aiida.orm import RemoteData

    return RemoteData(computer=aiida_localhost, remote_path=str(tmp_path / "scf")).store()


@pytest.fixture
def kcp_code(aiida_local_code_factory):
    """Return a mock ``koopmans.kcp`` code backed by the ``true`` executable."""
    return aiida_local_code_factory(executable="true", entry_point="koopmans.kcp")


@pytest.fixture
def pw_code(aiida_local_code_factory):
    """Return a mock ``quantumespresso.pw`` code backed by the ``true`` executable."""
    return aiida_local_code_factory(executable="true", entry_point="quantumespresso.pw")


def ozone_projection_blocks():
    """Return periodic-ozone projections: 9 occupied + 1 empty band, nspin=1."""
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    from aiida_koopmans.projections import ExplicitProjectionBlock
    from aiida_koopmans.spin import SpinChannel

    def _block(label, include, filled):
        include = list(include)
        n = len(include)
        return ExplicitProjectionBlock(
            label=label,
            spin=SpinChannel.NONE,
            filled=filled,
            num_wann=n,
            num_bands=n,
            exclude_bands=list(range(1, include[0])) or None,
            projection_type=WannierProjectionType.ANALYTIC,
            projections=[],
        )

    return [
        _block("block_occ", range(1, 10), filled=True),
        _block("block_emp", range(10, 11), filled=False),
    ]


@pytest.fixture
def mlwf_codes(aiida_localhost):
    """Return stand-in InstalledCode nodes for the full mlwfs-init code set."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    def _code(label: str, entry_point: str):
        try:
            return InstalledCode.collection.get(label=label)
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=aiida_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point,
            ).store()

    return {
        "pw": _code("mlwf-pw", "quantumespresso.pw"),
        "wannier90": _code("mlwf-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("mlwf-p2w", "quantumespresso.pw2wannier90"),
        "wann2kcp": _code("mlwf-w2k", "koopmans.wann2kcp"),
        "merge_evc": _code("mlwf-merge", "koopmans.merge_evc"),
    }


@pytest.fixture
def wannier_codes(aiida_localhost):
    """Return a codes dict of stand-in InstalledCode nodes for wannierisation graphs.

    The codes never execute (construction-only tests); they exist only so the
    codes input namespace is populated with real ``AbstractCode`` nodes,
    which the builder and namespace validators require.
    """
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    def _code(label: str, entry_point: str):
        try:
            return InstalledCode.collection.get(label=label)
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=aiida_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point,
            ).store()

    return {
        "pw": _code("bw-pw", "quantumespresso.pw"),
        "wannier90": _code("bw-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("bw-p2w", "quantumespresso.pw2wannier90"),
    }


@pytest.fixture
def pdos_codes(wannier_codes, aiida_localhost):
    """Extend the wannierization codes with a projwfc code for the projected-DOS flows."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    try:
        projwfc = InstalledCode.collection.get(label="bw-pjw")
    except NotExistent:
        projwfc = InstalledCode(
            label="bw-pjw",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
            default_calc_job_plugin="quantumespresso.projwfc",
        ).store()
    return {**wannier_codes, "projwfc": projwfc}


@pytest.fixture
def dfpt_codes(aiida_localhost):
    """Return a codes dict of stand-in nodes for the kcw.x graphs.

    Like ``wannier_codes`` but with kcw.x alongside; the codes
    never execute.
    """
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    def _code(label: str, entry_point: str):
        try:
            return InstalledCode.collection.get(label=label)
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=aiida_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point,
            ).store()

    return {
        "pw": _code("dfpt-pw", "quantumespresso.pw"),
        "wannier90": _code("dfpt-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("dfpt-p2w", "quantumespresso.pw2wannier90"),
        "kcw": _code("dfpt-kcw", "koopmans.kcw_wann2kc"),
    }


@pytest.fixture
def dfpt_pdos_codes(dfpt_codes, aiida_localhost):
    """Extend ``dfpt_codes`` with a projwfc code for the projected-DOS flows."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    try:
        projwfc = InstalledCode.collection.get(label="dfpt-pjw")
    except NotExistent:
        projwfc = InstalledCode(
            label="dfpt-pjw",
            computer=aiida_localhost,
            filepath_executable="/bin/true",
            default_calc_job_plugin="quantumespresso.projwfc",
        ).store()
    return {**dfpt_codes, "projwfc": projwfc}


@pytest.fixture
def ph_codes(aiida_localhost):
    """Return a codes dict of stand-in nodes for the dielectric chain.

    Like ``dfpt_codes`` but with ph.x alongside; the codes never execute.
    """
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    def _code(label: str, entry_point: str):
        try:
            return InstalledCode.collection.get(label=label)
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=aiida_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point,
            ).store()

    return {
        "pw": _code("eps-pw", "quantumespresso.pw"),
        "ph": _code("eps-ph", "quantumespresso.ph"),
        "wannier90": _code("eps-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("eps-p2w", "quantumespresso.pw2wannier90"),
        "kcw": _code("eps-kcw", "koopmans.kcw_wann2kc"),
    }


@pytest.fixture
def ozone_pseudo_family(ozone_real_pseudos):
    """Register (or fetch) a one-pseudo family covering ozone's O kind."""
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    family, _ = PseudoPotentialFamily.collection.get_or_create(label="test-ozone-family")
    if family.count() == 0:
        pseudo = ozone_real_pseudos["O"]
        if not pseudo.is_stored:
            pseudo.store()
        family.add_nodes([pseudo])
    return family.label


@pytest.fixture(scope="module")
def si_reference() -> dict:
    """Load the silicon reference data."""
    with open(Path(__file__).parent / "data" / "ui" / "si_ui_reference.json") as handle:
        return json.load(handle)


def block_wannierization(label: str, *, with_u_dis: bool = False) -> dict:
    """Build a stored per-block ``WannierizeBlockOutputs``-shaped entry.

    The ``retrieved`` folder carries the wannier90 read-back files a
    ``wan_mode='decompose'`` pass stages (``aiida_u.mat``,
    ``aiida_centres.xyz``, and ``aiida_u_dis.mat`` for a disentangling
    manifold), plus the ``nnkp_file`` the pass reads first. Contents are
    placeholders: every consumer of this fixture inspects graph structure
    rather than running the pass.
    """
    import io

    from aiida import orm

    folder = orm.FolderData()
    folder.base.repository.put_object_from_filelike(io.BytesIO(b"u"), "aiida_u.mat")
    if with_u_dis:
        folder.base.repository.put_object_from_filelike(io.BytesIO(b"ud"), "aiida_u_dis.mat")
    folder.base.repository.put_object_from_filelike(
        io.BytesIO(b"1\n\nX 0 0 0\n"), "aiida_centres.xyz"
    )
    folder.store()
    return {
        "retrieved": folder,
        "nnkp_file": orm.SinglefileData(io.BytesIO(b"n"), filename=f"{label}.nnkp").store(),
    }


def occ_emp_merge_groups(spin: str = "none") -> list[dict]:
    """Return a one-block-per-filling ``merge_groups`` partition for one spin channel."""
    return [
        {"filled": True, "spin": spin, "blocks": [{"label": "occ"}]},
        {"filled": False, "spin": spin, "blocks": [{"label": "emp"}]},
    ]


def si_external_projector_tables() -> dict:
    """Silicon external-projector orbital tables: s + p per atom, 8 in total.

    The caller-synthesized shape the upstream builder consumes: one entry
    per orbital with its ``l``, explicitly unfrozen — the upstream builder
    stages every entry not marked ``frozen: False`` in ``atom_proj_frozen``.
    """
    return {
        "Si": [
            {"l": 0, "frozen": False},
            {"l": 1, "frozen": False},
        ]
    }
