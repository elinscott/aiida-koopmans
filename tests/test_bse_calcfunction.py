"""Tests for :func:`generate_qp_database` (``@task.calcfunction``).

k2y is an optional dependency of this PR's scope (see ``pyproject.toml``):
``pytest.importorskip`` keeps this module a clean skip -- not a collection
error -- everywhere it is not installed, so the rest of the suite (run
against the plain ``koopmans2`` venv, which does not carry k2y) stays green.

The discriminating check is :func:`test_matches_a_direct_k2y_call`: it
builds the same ``ndb.QP`` two ways from the same inputs -- once through
the calcfunction, once by calling ``KcwQpDatabaseGenerator`` directly on
the same arrays -- and diffs every netCDF variable. That isolates the
calcfunction's own data threading (unpacking ``yambo_save`` to a temp
SAVE directory, reading the KI/KS grids out of ``output_parameters``,
reading the k-points out of ``output_band``) from k2y's own mapping
math, which is k2y's own test suite's job, not ours.

The fixture data (``tests/data/bse/``, ``tests/parsers/fixtures/kcw_ham/
bse_si/``) is copied from k2y's own ``tests/data/`` (a real 2x2x2-grid,
20-band kcw.x ``ham`` run on silicon): ``ns.db1`` (284 KiB), ``ndb.kindx``
(65 KiB -- one of the two SAVE siblings ``generate_qp_database`` reads
the SERIAL_NUMBER match from; the other, ``ndb.gops``, is 3 MiB and
skipped to keep this fixture small -- ``ndb.kindx`` alone carries the
same HEAD_VERSION/HEAD_REVISION/SERIAL_NUMBER variables), ``Si.scf.in``
(1 KiB, its K_POINTS crystal card copied into ``IBZ_KPOINTS_CRYSTAL``
below) and the kcw.x ``ham`` stdout (20 KiB, reused as a
``koopmans.kcw_ham`` parser fixture so ``ham_output_parameters`` below
is genuinely parsed, not hand-built).

k2y's own ``tests/data/ref_ndb.QP`` is deliberately not used as a
reference here: reproducing it locally (feeding the same fixture through
k2y's ``generate_mappings()``) gives a ``QP_kpts`` of shape ``(3, 8)`` --
the *un*-expanded IBZ grid -- while the installed k2y (legacy-extra,
b284fca) always expands to the full 64-point BZ per its own
``generate_mappings`` docstring and the k2y-core test suite's
``N_FULL_KPTS = 64`` constant. ``ref_ndb.QP`` predates that full-BZ-
expansion fix and is not representative of current k2y output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aiida import orm

pytest.importorskip("k2y")

import netCDF4
from k2y.k2y import KcwQpDatabaseGenerator

from aiida_koopmans.workgraphs.bse import generate_qp_database

DATA_DIR = Path(__file__).parent / "data" / "bse"

#: The K_POINTS crystal card from ``tests/data/bse/Si.scf.in`` -- the 2x2x2
#: IBZ grid the ``bse_si`` kcw.x ``ham`` fixture interpolated on. Stands in
#: for a real nscf run's ``output_band``: ``BandsData.get_array('kpoints')``
#: reports crystal fractional coordinates by default, the same coordinates
#: this K_POINTS card already lists.
IBZ_KPOINTS_CRYSTAL = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5],
    [0.0, 0.5, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.0],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.5, 0.5, 0.5],
]


def _folder_with(names: list[str]) -> orm.FolderData:
    folder = orm.FolderData()
    for name in names:
        with open(DATA_DIR / name, "rb") as handle:
            folder.base.repository.put_object_from_filelike(handle, name)
    return folder


@pytest.fixture
def yambo_save() -> orm.FolderData:
    """Build a yambo p2y ``retrieved`` folder: ``ns.db1`` plus its SAVE siblings."""
    return _folder_with(["ns.db1", "ndb.kindx"]).store()


@pytest.fixture
def nscf_output_band() -> orm.BandsData:
    bands = orm.BandsData()
    bands.set_kpoints(np.array(IBZ_KPOINTS_CRYSTAL))
    return bands.store()


@pytest.fixture
def ham_output_parameters(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    """Build a real, parsed ``output_parameters`` from the ``bse_si`` kcw.x ``ham`` fixture."""
    node = generate_calc_job_node(
        entry_point_name="koopmans.kcw_ham",
        computer=fixture_localhost,
        test_name="bse_si",
        input_filename="aiida.khi",
        output_filename="aiida.kho",
        inputs={
            "parameters": orm.Dict({"CONTROL": {}, "WANNIER": {}, "HAM": {"do_bands": False}}),
        },
    )
    parser = generate_parser("koopmans.kcw_ham")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    return results["output_parameters"]


def _direct_k2y_qp_db(save_dir: Path, params: dict, output_path: Path) -> None:
    """Build an ``ndb.QP`` by calling k2y directly on the fixture's own values.

    Mirrors what :func:`generate_qp_database` does with the same inputs,
    without going through AiiDA nodes or a calcfunction -- the reference
    the calcfunction's output is checked against.
    """
    generator = KcwQpDatabaseGenerator(ns_db1=str(save_dir / "ns.db1"))
    generator.eigenvalues_KI = np.array(params["ki_eigenvalues_on_grid"])
    generator.eigenvalues_KS = np.array(params["ks_eigenvalues_on_grid"])
    generator.kpoints_grid_kcw = np.array(IBZ_KPOINTS_CRYSTAL)
    generator.kpoints_type = "crystal"
    generator.generate_mappings()
    generator.generate_QP_db(str(output_path))


class TestGenerateQpDatabase:
    def test_matches_a_direct_k2y_call(
        self, aiida_profile, yambo_save, ham_output_parameters, nscf_output_band, tmp_path
    ):
        produced = generate_qp_database._callable(
            yambo_save=yambo_save,
            ham_output_parameters=ham_output_parameters,
            nscf_output_band=nscf_output_band,
        )
        produced_path = tmp_path / "produced_ndb.QP"
        produced_path.write_bytes(produced.get_content(mode="rb"))

        reference_dir = tmp_path / "reference_save"
        reference_dir.mkdir()
        for name in ("ns.db1", "ndb.kindx"):
            (reference_dir / name).write_bytes((DATA_DIR / name).read_bytes())
        reference_path = tmp_path / "reference_ndb.QP"
        _direct_k2y_qp_db(reference_dir, ham_output_parameters.get_dict(), reference_path)

        with (
            netCDF4.Dataset(produced_path) as produced_ds,
            netCDF4.Dataset(reference_path) as reference_ds,
        ):
            assert set(produced_ds.variables) == set(reference_ds.variables)
            for name in produced_ds.variables:
                produced_values = np.asarray(produced_ds.variables[name][:])
                reference_values = np.asarray(reference_ds.variables[name][:])
                assert produced_values.shape == reference_values.shape, name
                if produced_values.dtype.kind in "fc":
                    assert np.allclose(produced_values, reference_values, atol=1e-8, rtol=1e-8), (
                        name
                    )
                else:
                    assert np.array_equal(produced_values, reference_values), name

        # The comparison above is value-wise (netCDF4/HDF5 do not guarantee
        # byte-identical files for identical variable data across separate
        # writes -- e.g. internal library-version attributes). Confirmed by
        # running this exact check by hand before this test was written.
        assert produced_path.read_bytes() != b""

    def test_shape_and_scale_are_physically_sane(
        self, aiida_profile, yambo_save, ham_output_parameters, nscf_output_band
    ):
        """Sanity check independent of the direct-call reference above."""
        produced = generate_qp_database._callable(
            yambo_save=yambo_save,
            ham_output_parameters=ham_output_parameters,
            nscf_output_band=nscf_output_band,
        )
        with netCDF4.Dataset("in-memory", memory=produced.get_content(mode="rb")) as ds:
            n_bands = 20  # tests/data/bse fixture: 4 occ + 4 emp Wannier + 12 unscreened KS
            n_kpoints_full_bz = 64  # 2x2x2 MP grid, 8 IBZ points expanded
            n_states = n_bands * n_kpoints_full_bz
            pars = np.asarray(ds.variables["PARS"][:]).flatten()
            assert int(pars[0]) == n_bands
            assert int(pars[1]) == n_kpoints_full_bz
            assert int(pars[2]) == n_states
            qp_e = np.asarray(ds.variables["QP_E"][:, 0])
            # Ha, silicon KI valence/conduction states: a few eV either side of
            # the gap, well inside +/-3 Ha.
            assert np.all(np.abs(qp_e) < 3.0)


class TestGenerateQpDatabaseRefusals:
    def test_missing_eigenvalue_keys_names_them(self, aiida_profile, yambo_save, nscf_output_band):
        bad_params = orm.Dict({"job_done": True}).store()
        with pytest.raises(ValueError, match="ki_eigenvalues_on_grid"):
            generate_qp_database._callable(
                yambo_save=yambo_save,
                ham_output_parameters=bad_params,
                nscf_output_band=nscf_output_band,
            )

    def test_missing_ns_db1_is_refused(
        self, aiida_profile, ham_output_parameters, nscf_output_band
    ):
        empty_folder = _folder_with([]).store()
        with pytest.raises(ValueError, match=r"ns\.db1"):
            generate_qp_database._callable(
                yambo_save=empty_folder,
                ham_output_parameters=ham_output_parameters,
                nscf_output_band=nscf_output_band,
            )


def test_generate_qp_database_builds_inside_a_graph(
    aiida_profile, yambo_save, ham_output_parameters, nscf_output_band
):
    """Construction-level check: the calcfunction wires into a WorkGraph."""
    from aiida_workgraph import WorkGraph

    wg = WorkGraph("test-bse-qp-database")
    wg.add_task(
        generate_qp_database,
        name="qp_database",
        yambo_save=yambo_save,
        ham_output_parameters=ham_output_parameters,
        nscf_output_band=nscf_output_band,
    )
    assert "qp_database" in wg.tasks
