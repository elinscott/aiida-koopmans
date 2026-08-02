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

    def _generate_upf_data(element: str, z_valence: float = 6.0) -> UpfData:
        # Shaped for the line-based block extractors in
        # aiida-wannier90-workflows' pseudo utilities: ``<PP_HEADER`` and its
        # ``/>`` sit on their own lines (sharing a line with the ``<UPF>``
        # root tag loses the attributes), ``has_so`` is required, and
        # ``PP_PSWFC`` provides an s+p valence so projection counting works.
        content = (
            f'<UPF version="2.0.1">\n'
            f'<PP_HEADER\nelement="{element}"\n'
            f'z_valence="{z_valence}"\nhas_so="F"\n/>\n'
            f"<PP_PSWFC>\n"
            f'<PP_CHI.1 l="0"/>\n<PP_CHI.2 l="1"/>\n'
            f"</PP_PSWFC>\n"
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
def fake_cutoffs_family(aiida_profile, generate_upf_data):
    """Install a fake ``CutoffsPseudoPotentialFamily`` (Si and O).

    For graph builders that call ``get_builder_from_protocol`` eagerly at
    build time: the protocol machinery only accepts SSSP / PseudoDojo /
    cutoffs families — a plain ``PseudoPotentialFamily`` is not found.
    """
    from aiida.common.exceptions import NotExistent
    from aiida_pseudo.groups.family import CutoffsPseudoPotentialFamily

    label = "FAKE/CUTOFFS/PBE/SR"
    try:
        return CutoffsPseudoPotentialFamily.collection.get(label=label)
    except NotExistent:
        pass

    family = CutoffsPseudoPotentialFamily(label=label)
    family.store()
    pseudos = [
        generate_upf_data(element, z_valence=z_valence).store()
        for element, z_valence in (("Si", 4.0), ("O", 6.0))
    ]
    family.add_nodes(pseudos)
    family.set_cutoffs(
        {element: {"cutoff_wfc": 30.0, "cutoff_rho": 240.0} for element in ("Si", "O")},
        stringency="normal",
    )
    return family


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
def kpath(aiida_profile):
    """Return a short explicit k-path ``KpointsData``."""
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
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
    include,
    projections=None,
    spin=None,
    filled=None,
    num_bands=None,
    exclude_bands=None,
):
    """Build a minimal explicit (ANALYTIC) projection block over ``include`` bands.

    ``include`` names the bands the block's Wannier functions are to
    occupy, one per function. Those bands are not stored, so they are
    expressed the way production blocks express them: everything below the
    first of them is excluded, which puts the block's own bands at the
    bottom of what it reads. ``filled`` stamps the occupancy; ``None``
    leaves it unstamped, the state a block is in before anything has
    classified it. ``num_bands`` beyond ``len(include)`` gives the block a
    disentanglement pool, which sits above those bands. Pass
    ``exclude_bands`` to override the exclusion outright.
    """
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel

    include = list(include)
    n = len(include)
    block = ExplicitProjectionBlock(
        label=label,
        spin=SpinChannel.NONE if spin is None else spin,
        num_wann=n,
        num_bands=n if num_bands is None else num_bands,
        projection_type=WannierProjectionType.ANALYTIC,
        projections=[] if projections is None else projections,
    )
    below = list(range(1, include[0]))
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


def assert_graph_roundtrips(wg):
    """Assert a built WorkGraph survives ``to_dict`` -> ``from_dict``.

    ``wg.run()`` and the daemon reconstruct the graph through exactly this
    round-trip before executing anything, so wiring that fails it dies at
    run start even though construction succeeded (e.g. per-key links into
    the entries of a typed dynamic output namespace).
    """
    from aiida_workgraph import WorkGraph

    WorkGraph.from_dict(wg.to_dict())


def automatic_block(label, include, spin=None, projection_type=None, filled=None):
    """Build a minimal automatic projection block over ``include`` bands.

    Defaults to pseudoatomic projectors (``ATOMIC_PROJECTORS_QE``) — the
    projection source automated wannierization always uses, and the one
    whose occupancy is unknown until the runtime split, so ``filled``
    defaults to unstamped. ``include`` names the bands the block's Wannier
    functions are to occupy; as in :func:`explicit_block` they are
    expressed by excluding everything below them.
    """
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    from aiida_koopmans.types import AutomaticProjectionBlock, SpinChannel

    if projection_type is None:
        projection_type = WannierProjectionType.ATOMIC_PROJECTORS_QE
    include = list(include)
    n = len(include)
    block = AutomaticProjectionBlock(
        label=label,
        spin=SpinChannel.NONE if spin is None else spin,
        num_wann=n,
        num_bands=n,
        projection_type=projection_type,
    )
    if include[0] > 1:
        block["exclude_bands"] = list(range(1, include[0]))
    if filled is not None:
        block["filled"] = filled
    return block


@pytest.fixture
def nscf_remote(aiida_localhost, tmp_path):
    """Return a stand-in nscf scratch ``RemoteData`` (never read; construction-only)."""
    from aiida.orm import RemoteData

    return RemoteData(computer=aiida_localhost, remote_path=str(tmp_path)).store()


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

    from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel

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
        "projwfc": _code("mlwf-pjw", "quantumespresso.projwfc"),
        "wann2kcp": _code("mlwf-w2k", "koopmans.wann2kcp"),
        "merge_evc": _code("mlwf-merge", "koopmans.merge_evc"),
    }


@pytest.fixture
def wannier_codes(aiida_localhost):
    """Return a ``Codes`` dict of stand-in InstalledCode nodes for wannierisation graphs.

    The codes never execute (construction-only tests); they exist only so the
    ``Codes`` input namespace is populated with real ``AbstractCode`` nodes,
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
        "projwfc": _code("bw-pjw", "quantumespresso.projwfc"),
    }


@pytest.fixture
def dfpt_codes(aiida_localhost):
    """Return a ``Codes`` dict of stand-in nodes for the kcw.x graphs.

    Like ``wannier_codes`` but with kcw.x in place of projwfc.x; the codes
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
def ph_codes(aiida_localhost):
    """Return a ``Codes`` dict of stand-in nodes for the dielectric chain.

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
