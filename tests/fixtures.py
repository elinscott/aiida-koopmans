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
def kcp_code(aiida_local_code_factory):
    """Return a mock ``koopmans.kcp`` code backed by the ``true`` executable."""
    return aiida_local_code_factory(executable="true", entry_point="koopmans.kcp")


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
