"""Yambo BSE with Koopmans eigenvalues: the quasiparticle-database step.

Yambo's BSE solver reads its quasiparticle corrections from an ``ndb.QP``
database in the SAVE directory, rather than computing a GW correction
itself. Feeding it Koopmans (KI or pKI) eigenvalues in place of a GW run
lets a BSE spectrum be built directly from a kcw.x ``ham`` calculation.

:func:`generate_qp_database` builds that ``ndb.QP`` by driving k2y's
``KcwQpDatabaseGenerator`` directly on parsed AiiDA data: the kcw.x
``ham`` output's chosen Koopmans/KS eigenvalue grids and the k-points of
the nscf run that seeded it, mapped onto the yambo p2y run's own k-point grid
(``ns.db1``). This is deliberately not k2y's ``from_aiida`` classmethod
(walks PKs, bypassing AiiDA provenance) nor its ``set_koopmans_eval``
file route (re-parses a kcw.x stdout file and needs the optional
``ase-koopmans`` dependency this package does not carry) nor
``generate_kcw_qp_database``/``generate_QP_db_SinglefileData`` (the
former only wraps ``from_aiida`` for provenance; the latter writes its
output to the current working directory).

Wiring pw.x -> p2y -> yambo init -> this step -> yambo BSE into a
``@task.graph`` route is a follow-up; this module scaffolds the codes
that route will need (:class:`BseCodes`) and the database-generation
step it will call.
"""

# No ``from __future__ import annotations`` in this module: stringified
# annotations hide ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the dispatcher reads off the Codes
# TypedDicts.

from pathlib import Path
from typing import Annotated, TypedDict

import numpy as np
from aiida import orm
from aiida.common.folders import SandboxFolder
from aiida_workgraph import task
from aiida_workgraph.socket_spec import SocketMeta

# k2y.k2y imports only netCDF4/ase/xarray/numpy/yambopy -- no AiiDA import,
# so unlike aiida_yambo.workflows it never touches AiiDA's profile config at
# import time and is safe to import at module scope.
from k2y.k2y import KcwQpDatabaseGenerator

from aiida_koopmans.workgraphs.pw import PwCode

P2yCode = Annotated[
    orm.AbstractCode,
    SocketMeta(help="Needed to convert the pw.x ground state into a yambo SAVE directory."),
]
YamboCode = Annotated[
    orm.AbstractCode,
    SocketMeta(help="Needed to run the BSE calculation."),
]


class BseCodes(TypedDict):
    """Codes the Koopmans-eigenvalues BSE route will need."""

    pw: PwCode
    p2y: P2yCode
    yambo: YamboCode


#: ``eigenvalues`` values accepted by :func:`generate_qp_database`, and the
#: ``output_parameters`` key each reads (see
#: ``aiida_koopmans.parsers.kcw.KcwHamParameters``): one row per grid
#: k-point, one column per Wannier-Hamiltonian band, in eV.
_EIGENVALUE_KEYS = {"ki": "ki_eigenvalues_on_grid", "pki": "pki_eigenvalues_on_grid"}
_KS_KEY = "ks_eigenvalues_on_grid"


@task.calcfunction
def generate_qp_database(
    yambo_save: orm.FolderData,
    ham_output_parameters: orm.Dict,
    nscf_output_band: orm.BandsData,
    eigenvalues: orm.Str,
) -> orm.SinglefileData:
    """Build a yambo ``ndb.QP`` from a kcw.x ``ham`` run's Koopmans eigenvalues.

    ``yambo_save`` is a yambo p2y run's ``retrieved`` folder: every file
    it holds is staged into one directory, so ``ns.db1`` sits alongside
    ``ndb.gops`` / ``ndb.kindx`` when those are present (needed for the
    SERIAL_NUMBER that lets yambo match the database to its own SAVE).

    ``eigenvalues`` selects which Koopmans eigenvalue flavor seeds the QP
    corrections: ``'ki'`` reads the ham run's ``ki_eigenvalues_on_grid``,
    ``'pki'`` its ``pki_eigenvalues_on_grid``. k2y's own file-based route
    (``KcwQpDatabaseGenerator.set_koopmans_eval``) defaults to
    ``pki_eigenvalues_on_grid``; this input makes that choice explicit
    instead of hardcoding either flavor.

    ``ham_output_parameters`` is the kcw.x ``ham`` run's parsed
    ``output_parameters`` and must carry the requested eigenvalue grid
    alongside ``ks_eigenvalues_on_grid``, each shaped
    ``(n_grid_kpoints, n_bands)``.

    ``nscf_output_band`` is the ``output_band`` of the nscf run that
    produced the kcw.x k-point grid, and must carry the same number of
    k-points as the requested eigenvalue grid and
    ``ks_eigenvalues_on_grid`` -- a mismatch means the two inputs come
    from different runs, and k2y's own k-point matching would otherwise
    mis-associate the surplus rows against a stale index rather than fail
    loudly. ``BandsData.get_array('kpoints')`` returns crystal
    (reciprocal-lattice-fraction) coordinates by default
    (``KpointsData.get_kpoints(cartesian=False)``), which is the
    coordinate system k2y's k-point matching expects.
    """
    flavor = eigenvalues.value
    if flavor not in _EIGENVALUE_KEYS:
        raise ValueError(
            f"`eigenvalues` is {flavor!r} -- must be one of {tuple(_EIGENVALUE_KEYS)}."
        )
    eigenvalue_key = _EIGENVALUE_KEYS[flavor]
    required_keys = (eigenvalue_key, _KS_KEY)

    params = ham_output_parameters.get_dict()
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise ValueError(
            f"`ham_output_parameters` is missing {missing} -- pass the "
            f"`output_parameters` of a `ham`-mode kcw.x run that computed "
            f"`eigenvalues={flavor!r}` eigenvalues. Keys present: {sorted(params)}."
        )

    n_nscf_kpoints = nscf_output_band.get_array("kpoints").shape[0]
    for key in required_keys:
        n_grid_kpoints = np.asarray(params[key]).shape[0]
        if n_grid_kpoints != n_nscf_kpoints:
            raise ValueError(
                f"`{key}` has {n_grid_kpoints} k-points but `nscf_output_band` has "
                f"{n_nscf_kpoints} -- pass the `output_band` of the same kcw.x `ham` "
                "run's own seeding nscf, not some other run's grid."
            )

    # SandboxFolder, not ``tempfile`` directly: AiiDA's own local-scratch
    # abstraction for this get-then-consume shape (see ``convert_spin``'s
    # get/put staging), auto-erased on exit like ``TemporaryDirectory``.
    with SandboxFolder() as sandbox:
        for filename in yambo_save.base.repository.list_object_names():
            with yambo_save.base.repository.open(filename, "rb") as handle:
                Path(sandbox.get_abs_path(filename)).write_bytes(handle.read())
        if "ns.db1" not in yambo_save.base.repository.list_object_names():
            raise ValueError(
                "`yambo_save` has no `ns.db1` -- pass the p2y (yambo init) run's "
                "retrieved folder, not some other calculation's."
            )
        ns_db1 = sandbox.get_abs_path("ns.db1")

        generator = KcwQpDatabaseGenerator(ns_db1=str(ns_db1))
        generator.eigenvalues_KI = np.array(params[eigenvalue_key])
        generator.eigenvalues_KS = np.array(params[_KS_KEY])
        generator.kpoints_grid_kcw = nscf_output_band.get_array("kpoints")
        generator.kpoints_type = "crystal"
        generator.generate_mappings()

        qp_path = sandbox.get_abs_path("ndb.QP")
        generator.generate_QP_db(str(qp_path))
        return orm.SinglefileData(file=str(qp_path), filename="ndb.QP")
