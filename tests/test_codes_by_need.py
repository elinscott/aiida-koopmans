"""Socket-level contract of the per-chain ``Codes`` TypedDicts.

Every chain declares its codes as a TypedDict graph input: required members
become required ``workgraph.code`` sockets, ``NotRequired`` members stay
optional and carry a "Needed for ..." help string. aiida-workgraph's input
check is the error contract — ``check_before_run`` raises
``MissingRequiredInputsError`` whose entries name the socket path, the
socket identifier, and the declared help. These tests pin that contract per
chain (deferred-task shape, where no graph body runs before the check), the
metadata on the sockets themselves, the serialization round-trip, and the
build-time guards that cover the settings-conditional needs the socket
layer cannot express.
"""

from importlib import import_module

import pytest

CASES = {
    "RunPwBands": (
        "aiida_koopmans.workgraphs.pw.RunPwBands",
        {"pw"},
        set(),
    ),
    "DielectricTask": (
        "aiida_koopmans.workgraphs.ph.DielectricTask",
        {"pw", "ph"},
        set(),
    ),
    "Wannierize": (
        "aiida_koopmans.workgraphs.wannier90.Wannierize",
        {"pw", "pw2wannier90", "wannier90"},
        {"projwfc"},
    ),
    "OptimizeWannierization": (
        "aiida_koopmans.workgraphs.wannier90.OptimizeWannierization",
        {"pw", "pw2wannier90", "wannier90"},
        {"projwfc"},
    ),
    "WannierizeBlocks": (
        "aiida_koopmans.workgraphs.block_wannierize.WannierizeBlocks",
        {"pw", "pw2wannier90", "wannier90"},
        {"wannierjl"},
    ),
    "WannierizeBlock": (
        "aiida_koopmans.workgraphs.block_wannierize.WannierizeBlock",
        {"pw", "pw2wannier90", "wannier90"},
        set(),
    ),
    "WannierizeAndSplitBlock": (
        "aiida_koopmans.workgraphs.auto_wannierize.WannierizeAndSplitBlock",
        {"pw", "pw2wannier90", "wannier90", "wannierjl"},
        set(),
    ),
    "MlwfInitialization": (
        "aiida_koopmans.workgraphs.mlwf_init.MlwfInitialization",
        {"pw", "pw2wannier90", "wannier90", "wann2kcp", "merge_evc", "kcp"},
        set(),
    ),
    "FoldToSupercell": (
        "aiida_koopmans.workgraphs.folding.FoldToSupercell",
        {"wann2kcp", "merge_evc"},
        set(),
    ),
    "KoopmansDSCFWorkflow": (
        "aiida_koopmans.workgraphs.kcp.KoopmansDSCFWorkflow",
        {"kcp"},
        {"pw", "pw2wannier90", "wannier90", "wann2kcp", "merge_evc"},
    ),
    "TrajectoryWorkflow": (
        "aiida_koopmans.workgraphs.ml.TrajectoryWorkflow",
        {"kcp"},
        {"pw", "pw2wannier90", "wannier90", "wann2kcp", "merge_evc"},
    ),
    "SinglepointDFPTWorkflow": (
        "aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow",
        {"pw", "pw2wannier90", "wannier90", "kcw"},
        {"ph"},
    ),
    "RunPdos": (
        "aiida_koopmans.workgraphs.pdos.RunPdos",
        {"pw", "dos", "projwfc"},
        set(),
    ),
}


def _graph(dotted: str):
    """Import a graph lazily (workgraph imports stay function-local in tests)."""
    module_name, attr = dotted.rsplit(".", 1)
    return getattr(import_module(module_name), attr)


def _deferred(dotted: str, **inputs):
    """Add the chain as a deferred task and return ``(workgraph, task)``."""
    from aiida_workgraph import WorkGraph

    wg = WorkGraph()
    task = wg.add_task(_graph(dotted), **inputs)
    return wg, task


def _missing_code_entries(wg):
    """Run the input check and return the entries under a ``codes`` namespace."""
    from aiida_workgraph.errors import MissingRequiredInputsError

    try:
        wg.check_before_run()
    except MissingRequiredInputsError as exc:
        return [entry for entry in exc.missing if ".codes." in entry.socket_path]
    return []


@pytest.mark.parametrize("case", CASES, ids=CASES)
class TestMissingRequiredCodes:
    def test_absent_required_members_are_reported(self, case, aiida_profile):
        dotted, required, _ = CASES[case]
        wg, task = _deferred(dotted)
        entries = _missing_code_entries(wg)
        assert {entry.socket_path for entry in entries} == {
            f"{task.name}.codes.{member}" for member in required
        }
        assert {entry.identifier for entry in entries} == {"workgraph.code"}

    def test_socket_metadata_matches_the_typeddict(self, case, aiida_profile):
        dotted, required, conditional = CASES[case]
        _, task = _deferred(dotted)
        members = {socket._name for socket in task.inputs.codes}
        assert members == required | conditional
        for member in required:
            assert task.inputs.codes[member]._metadata.required
        for member in conditional:
            socket = task.inputs.codes[member]
            assert not socket._metadata.required
            assert (socket._metadata.help or "").startswith("Needed for")

    def test_codes_socket_shape_survives_a_roundtrip(self, case, aiida_profile):
        from aiida_workgraph import WorkGraph

        dotted, required, conditional = CASES[case]
        wg, task = _deferred(dotted)
        restored = WorkGraph.from_dict(wg.to_dict())
        restored_task = restored.tasks[task.name]
        for member in required | conditional:
            before = task.inputs.codes[member]
            after = restored_task.inputs.codes[member]
            assert after._identifier == before._identifier == "workgraph.code"
            assert after._metadata.required == before._metadata.required
            assert after._metadata.help == before._metadata.help
        assert {entry.socket_path for entry in _missing_code_entries(restored)} == {
            f"{task.name}.codes.{member}" for member in required
        }


class TestProvidedRequiredCodesSatisfyTheCheck:
    """With the required members provided, the codes namespace reports nothing.

    The conditional members stay absent, so these are also the
    condition-off half of the ``NotRequired`` contract: no split trigger,
    molecular DSCF, numeric ``eps_inf`` — each builds without its
    conditional code.
    """

    def test_wannierize_blocks_without_wannierjl(self, wannier_codes, aiida_profile):
        wg, _ = _deferred(CASES["WannierizeBlocks"][0], codes=wannier_codes)
        assert _missing_code_entries(wg) == []

    def test_dscf_without_the_wannier_route_codes(self, kcp_code, aiida_profile):
        wg, _ = _deferred(CASES["KoopmansDSCFWorkflow"][0], codes={"kcp": kcp_code})
        assert _missing_code_entries(wg) == []

    def test_dfpt_without_ph(self, dfpt_codes, aiida_profile):
        wg, _ = _deferred(CASES["SinglepointDFPTWorkflow"][0], codes=dfpt_codes)
        assert _missing_code_entries(wg) == []

    def test_dielectric_with_both_codes(self, ph_codes, aiida_profile):
        wg, _ = _deferred(
            CASES["DielectricTask"][0],
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
        )
        assert _missing_code_entries(wg) == []


class TestUndeclaredCodesAreRejected:
    """A typed codes namespace refuses keys its chain does not declare."""

    def test_extra_key_raises_at_assignment(self, wannier_codes, kcp_code, aiida_profile):
        with pytest.raises(ValueError, match="not defined"):
            _deferred(
                CASES["WannierizeBlocks"][0],
                codes={**wannier_codes, "kcp": kcp_code},
            )


class TestConditionOnGuards:
    """The build-time guards own the settings-conditional needs.

    A conditional code's socket is optional, so when the setting that needs
    it is on, the chain's own guard raises at build — the socket layer
    cannot see the setting.
    """

    def test_eps_auto_without_ph_raises(
        self, dfpt_codes, silicon_structure, kmesh, kpath, aiida_profile
    ):
        from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow
        from tests.fixtures import explicit_block

        block = explicit_block("occ", range(1, 5), projections=["Si:sp3"])
        with pytest.raises(ValueError, match=r"eps_inf='auto' requires a ph\.x code"):
            SinglepointDFPTWorkflow.build(
                codes=dfpt_codes,
                structure=silicon_structure,
                manifolds={"none": {"occ": [block]}},
                kpoints=kmesh,
                bands_kpoints=kpath,
                pseudo_family="SSSP/1.3/PBE/efficiency",
                eps_inf="auto",
            )
