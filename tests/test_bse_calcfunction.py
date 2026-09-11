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
SAVE directory, reading the requested eigenvalue grid out of
``output_parameters``, reading the k-points out of ``output_band``) from
k2y's own mapping math, which is k2y's own test suite's job, not ours.
It is parametrized over ``eigenvalues`` ('ki'/'pki'): the synthetic
``ham_output_parameters`` fixture gives the two flavors distinct shifts,
so reading the wrong ``output_parameters`` key would fail the parity
comparison.

``tests/data/bse/ns.db1`` (284 KiB) and ``ndb.kindx`` (65 KiB -- one of
the two SAVE siblings ``generate_qp_database`` reads the SERIAL_NUMBER
match from; the other, ``ndb.gops``, is 3 MiB and skipped to keep this
fixture small -- ``ndb.kindx`` alone carries the same HEAD_VERSION/
HEAD_REVISION/SERIAL_NUMBER variables) are copied from k2y's own
``tests/data/``: a real yambo p2y run on silicon, on a 4x4x4 Monkhorst-Pack
grid (8 IBZ k-points expanding to the full 64-point BZ).

The kcw.x side of k2y's own fixture pair (``Si.scf.in``, the ``ham``
stdout reused below as ``ham_output_parameters``) ran a *different*,
coarser 2x2x2 grid (8 k-points total) -- only 8 of ns.db1's 64 points
coincide with it, so pairing the two directly would leave k2y's
k-point matching to silently fall back to a stale index for the other
56 (see :class:`TestGenerateQpDatabaseRefusals`, which now catches
exactly this mismatch). ``ham_output_parameters`` below is kept only
for :func:`test_ham_output_parameters_has_the_expected_shape`, a
Dict-shape contract check -- the discriminating tests instead build a
``synthetic_ham_output_parameters`` / ``synthetic_nscf_output_band``
pair on ns.db1's *own* 64-point grid (derived the same way k2y's
``generate_mappings`` derives it, via yambopy's
``YamboElectronsDB.expand_kpoints`` + ``car_red``), so every k-point
genuinely matches.

k2y's own ``tests/data/ref_ndb.QP`` is deliberately not used as a
reference here. Reproducing it locally -- feeding the *same* ``bse_si``
fixture pair through k2y's ``generate_mappings()`` -- gives a
``QP_kpts`` of shape ``(3, 64)``, the full-BZ expansion, matching the
installed k2y (legacy-extra, b284fca)'s own ``generate_mappings``
docstring and the k2y-core test suite's ``N_FULL_KPTS = 64`` constant.
``ref_ndb.QP`` itself reports ``QP_kpts`` of shape ``(3, 8)`` -- the
un-expanded IBZ grid -- while its own ``QP_E`` still carries 1,280 rows
(20 bands x 64 k-points): internally inconsistent, and not reproducible
from the fixture it ships beside. It predates the full-BZ-expansion fix
and is not representative of current k2y output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aiida import orm

pytest.importorskip("k2y")

import netCDF4
from k2y.k2y import KcwQpDatabaseGenerator
from yambopy import YamboElectronsDB
from yambopy.lattice import car_red

from aiida_koopmans.workgraphs.bse import generate_qp_database

DATA_DIR = Path(__file__).parent / "data" / "bse"

#: The K_POINTS crystal card from ``tests/data/bse/Si.scf.in`` -- the 2x2x2
#: grid the ``bse_si`` kcw.x ``ham`` fixture ran on. Used only by
#: :func:`test_ham_output_parameters_has_the_expected_shape`; the
#: discriminating tests below use ``synthetic_nscf_output_band`` instead
#: (see the module docstring for why).
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
    """Build the real ``bse_si`` kcw.x ``ham`` fixture's own 2x2x2 grid."""
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


@pytest.fixture
def yambo_kpoints_crystal() -> np.ndarray:
    """Return the 64 k-points ``ns.db1``'s own grid expands to, in crystal coordinates.

    Derived the same way k2y's own ``generate_mappings`` derives them (via
    yambopy's ``YamboElectronsDB.expand_kpoints`` + ``car_red``), so a
    synthetic kcw grid built from this array genuinely matches every one
    of ns.db1's 64 yambo k-points -- unlike the real ``bse_si`` kcw.x
    ``ham`` fixture's 2x2x2 grid (see the module docstring).
    """
    db = YamboElectronsDB.from_db_file(folder=str(DATA_DIR), Expand=True)
    expanded, _, _ = db.expand_kpoints()
    return car_red(expanded, db.rlat).astype(float)


@pytest.fixture
def synthetic_nscf_output_band(yambo_kpoints_crystal) -> orm.BandsData:
    """Build a 64-point ``output_band`` matching ``ns.db1``'s own expanded grid."""
    bands = orm.BandsData()
    bands.set_kpoints(yambo_kpoints_crystal)
    return bands.store()


@pytest.fixture
def synthetic_ham_output_parameters(yambo_kpoints_crystal) -> orm.Dict:
    """Synthetic KI/pKI/KS eigenvalues on the 64-point grid ``ns.db1`` expects.

    Not a real kcw.x output: no fixture pairs a real ``ham`` run with
    ``ns.db1``'s own 4x4x4 grid (see the module docstring), so this
    hand-picks 4 occupied + 4 empty bands, KS a few eV either side of a
    gap, KI shifted by a small Koopmans correction and pKI shifted by a
    *different* one -- both varying slightly across k-points -- so the
    two grids are not degenerate and picking the wrong key is
    detectable.
    """
    n_kpoints = len(yambo_kpoints_crystal)
    k = np.arange(n_kpoints)[:, None]
    band = np.arange(4)[None, :]
    ks = np.concatenate([-5.0 + 0.01 * k + 0.1 * band, 1.0 + 0.01 * k + 0.1 * band], axis=1)
    ki = ks + np.concatenate([np.full((n_kpoints, 4), -0.3), np.full((n_kpoints, 4), 0.3)], axis=1)
    pki = ks + np.concatenate([np.full((n_kpoints, 4), -0.5), np.full((n_kpoints, 4), 0.5)], axis=1)
    return orm.Dict(
        {
            "ki_eigenvalues_on_grid": ki.tolist(),
            "pki_eigenvalues_on_grid": pki.tolist(),
            "ks_eigenvalues_on_grid": ks.tolist(),
        }
    ).store()


def _direct_k2y_qp_db(
    save_dir: Path, params: dict, kpoints_grid: np.ndarray, output_path: Path, eigenvalue_key: str
) -> None:
    """Build an ``ndb.QP`` by calling k2y directly on the fixture's own values.

    Mirrors what :func:`generate_qp_database` does with the same inputs,
    without going through AiiDA nodes or a calcfunction -- the reference
    the calcfunction's output is checked against.
    """
    generator = KcwQpDatabaseGenerator(ns_db1=str(save_dir / "ns.db1"))
    generator.eigenvalues_KI = np.array(params[eigenvalue_key])
    generator.eigenvalues_KS = np.array(params["ks_eigenvalues_on_grid"])
    generator.kpoints_grid_kcw = np.array(kpoints_grid)
    generator.kpoints_type = "crystal"
    generator.generate_mappings()
    generator.generate_QP_db(str(output_path))


class TestGenerateQpDatabase:
    @pytest.mark.parametrize("flavor", ["ki", "pki"])
    def test_matches_a_direct_k2y_call(
        self,
        flavor,
        aiida_profile,
        yambo_save,
        synthetic_ham_output_parameters,
        synthetic_nscf_output_band,
        yambo_kpoints_crystal,
        tmp_path,
    ):
        """Parametrized over both eigenvalue flavors: picking the wrong key fails.

        ``synthetic_ham_output_parameters`` gives ``ki`` and ``pki`` distinct
        shifts, so a calcfunction that read the wrong ``output_parameters``
        key would diverge from the reference built with the requested one.
        """
        produced = generate_qp_database._callable(
            yambo_save=yambo_save,
            ham_output_parameters=synthetic_ham_output_parameters,
            nscf_output_band=synthetic_nscf_output_band,
            eigenvalues=orm.Str(flavor),
        )
        produced_path = tmp_path / f"produced_{flavor}_ndb.QP"
        produced_path.write_bytes(produced.get_content(mode="rb"))

        reference_dir = tmp_path / f"reference_save_{flavor}"
        reference_dir.mkdir()
        for name in ("ns.db1", "ndb.kindx"):
            (reference_dir / name).write_bytes((DATA_DIR / name).read_bytes())
        reference_path = tmp_path / f"reference_{flavor}_ndb.QP"
        _direct_k2y_qp_db(
            reference_dir,
            synthetic_ham_output_parameters.get_dict(),
            yambo_kpoints_crystal,
            reference_path,
            f"{flavor}_eigenvalues_on_grid",
        )

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
        self,
        aiida_profile,
        yambo_save,
        synthetic_ham_output_parameters,
        synthetic_nscf_output_band,
    ):
        """Sanity check independent of the direct-call reference above."""
        produced = generate_qp_database._callable(
            yambo_save=yambo_save,
            ham_output_parameters=synthetic_ham_output_parameters,
            nscf_output_band=synthetic_nscf_output_band,
            eigenvalues=orm.Str("ki"),
        )
        with netCDF4.Dataset("in-memory", memory=produced.get_content(mode="rb")) as ds:
            n_bands = 8  # synthetic_ham_output_parameters: 4 occ + 4 empty
            n_kpoints_full_bz = 64  # ns.db1's own 4x4x4 grid, 8 IBZ points expanded
            n_states = n_bands * n_kpoints_full_bz
            pars = np.asarray(ds.variables["PARS"][:]).flatten()
            assert int(pars[0]) == n_bands
            assert int(pars[1]) == n_kpoints_full_bz
            assert int(pars[2]) == n_states
            qp_e = np.asarray(ds.variables["QP_E"][:, 0])
            # Ha; synthetic_ham_output_parameters picks eV values a few eV
            # either side of a gap, well inside +/-3 Ha once converted.
            assert np.all(np.abs(qp_e) < 3.0)


def test_ham_output_parameters_has_the_expected_shape(aiida_profile, ham_output_parameters):
    """Dict-shape contract check for the real ``bse_si`` kcw.x ``ham`` fixture.

    Independent of :class:`TestGenerateQpDatabase` above: this fixture's
    2x2x2 grid does not match ``ns.db1``'s 4x4x4 grid (see the module
    docstring), so it is not used to build an ``ndb.QP`` here -- only to
    confirm the ``koopmans.kcw_ham`` parser produces the KI/KS shape
    :func:`generate_qp_database` expects.
    """
    params = ham_output_parameters.get_dict()
    # bse_si fixture: 2x2x2 grid, 4 occ + 4 emp Wannier + 12 unscreened KS.
    n_kpoints, n_bands = 8, 20
    assert np.asarray(params["ki_eigenvalues_on_grid"]).shape == (n_kpoints, n_bands)
    assert np.asarray(params["ks_eigenvalues_on_grid"]).shape == (n_kpoints, n_bands)


class TestGenerateQpDatabaseRefusals:
    def test_missing_eigenvalue_keys_names_them(self, aiida_profile, yambo_save, nscf_output_band):
        bad_params = orm.Dict({"job_done": True}).store()
        with pytest.raises(ValueError, match="ki_eigenvalues_on_grid"):
            generate_qp_database._callable(
                yambo_save=yambo_save,
                ham_output_parameters=bad_params,
                nscf_output_band=nscf_output_band,
                eigenvalues=orm.Str("ki"),
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
                eigenvalues=orm.Str("ki"),
            )

    def test_mismatched_grid_sizes_are_refused(
        self, aiida_profile, yambo_save, ham_output_parameters, synthetic_nscf_output_band
    ):
        """The real 2x2x2 ``ham`` fixture paired with ns.db1's 4x4x4 grid is refused.

        Pairing these two (8 eigenvalue rows against a 64-point
        ``nscf_output_band``) is exactly the mismatch the module docstring
        describes: without this guard, k2y's own k-point matching would
        silently fall back to a stale index for the 56 unmatched rows
        instead of failing.
        """
        with pytest.raises(ValueError, match="k-points"):
            generate_qp_database._callable(
                yambo_save=yambo_save,
                ham_output_parameters=ham_output_parameters,
                nscf_output_band=synthetic_nscf_output_band,
                eigenvalues=orm.Str("ki"),
            )

    def test_bad_eigenvalues_value_is_refused(
        self, aiida_profile, yambo_save, ham_output_parameters, nscf_output_band
    ):
        with pytest.raises(ValueError, match="kipz"):
            generate_qp_database._callable(
                yambo_save=yambo_save,
                ham_output_parameters=ham_output_parameters,
                nscf_output_band=nscf_output_band,
                eigenvalues=orm.Str("kipz"),
            )

    def test_pki_request_against_a_ki_only_run_names_the_keys_present(
        self, aiida_profile, yambo_save, ham_output_parameters, nscf_output_band
    ):
        """``ham_output_parameters`` is the real ``bse_si`` fixture: KI/KS only.

        Requesting ``pki`` against it is exactly the scenario the missing-key
        refusal exists for -- a run that computed KI but not pKI.
        """
        with pytest.raises(ValueError, match="pki_eigenvalues_on_grid") as excinfo:
            generate_qp_database._callable(
                yambo_save=yambo_save,
                ham_output_parameters=ham_output_parameters,
                nscf_output_band=nscf_output_band,
                eigenvalues=orm.Str("pki"),
            )
        assert "ki_eigenvalues_on_grid" in str(excinfo.value)
        assert "ks_eigenvalues_on_grid" in str(excinfo.value)


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
        eigenvalues=orm.Str("ki"),
    )
    assert "qp_database" in wg.tasks
