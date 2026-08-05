"""Tests for the orbital-density decompose workgraph pieces in ``ml.py``."""

from __future__ import annotations

import io

import pytest


def _wannierize_folder():
    """Build a stored ``FolderData`` mimicking a per-block wannier90 retrieved folder."""
    from aiida import orm

    folder = orm.FolderData()
    folder.base.repository.put_object_from_filelike(io.BytesIO(b"u matrix bytes"), "aiida_u.mat")
    xyz = "3\n\nX 0.10 0.20 0.30\nX 0.50 0.50 0.50\nSi 1.00 1.00 1.00\n"
    folder.base.repository.put_object_from_filelike(io.BytesIO(xyz.encode()), "aiida_centres.xyz")
    folder.store()
    return folder


def test_extract_decompose_inputs_emits_files_and_group_centres(aiida_profile):
    """The calcfunction lifts u.mat / centres.xyz and synthesises gc centres."""
    from aiida_koopmans.workgraphs.ml import extract_decompose_inputs

    folder = _wannierize_folder()
    outputs, _ = extract_decompose_inputs._callable.run_get_node(retrieved=folder)

    assert outputs["u_mat"].filename == "aiida_u.mat"
    assert outputs["centres_xyz"].filename == "aiida_centres.xyz"

    gc = outputs["centres_file"].get_content()
    # Only the two ``X`` (Wannier) rows become group-density centres.
    body = [line for line in gc.splitlines() if not line.startswith("#") and line.strip()]
    assert len(body) == 2
    assert body[0].split() == ["0.1000000000", "0.2000000000", "0.3000000000"]
    assert body[1].split() == ["0.5000000000", "0.5000000000", "0.5000000000"]


def test_extract_decompose_inputs_missing_file_raises(aiida_profile):
    """A folder without ``aiida_u.mat`` is a clear error."""
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import extract_decompose_inputs

    folder = orm.FolderData()
    folder.base.repository.put_object_from_filelike(io.BytesIO(b"x"), "aiida_centres.xyz")
    folder.store()

    with pytest.raises(FileNotFoundError, match=r"aiida_u\.mat"):
        extract_decompose_inputs._callable.run_get_node(retrieved=folder)


def test_extract_u_dis_mat_emits_singlefile(aiida_profile):
    """The calcfunction lifts ``aiida_u_dis.mat`` out of a disentangling block's folder."""
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import extract_u_dis_mat

    folder = orm.FolderData()
    folder.base.repository.put_object_from_filelike(io.BytesIO(b"u dis bytes"), "aiida_u_dis.mat")
    folder.store()

    result, _ = extract_u_dis_mat._callable.run_get_node(retrieved=folder)

    assert result.filename == "aiida_u_dis.mat"
    assert result.get_content() == "u dis bytes"


def test_extract_u_dis_mat_missing_file_raises(aiida_profile):
    """A folder without ``aiida_u_dis.mat`` is a hard error, not a silent skip."""
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import extract_u_dis_mat

    folder = orm.FolderData()
    folder.base.repository.put_object_from_filelike(io.BytesIO(b"u"), "aiida_u.mat")
    folder.store()

    with pytest.raises(FileNotFoundError, match=r"aiida_u_dis\.mat"):
        extract_u_dis_mat._callable.run_get_node(retrieved=folder)


def test_power_spectrum_dataset_workflow_fans_out_per_block(
    aiida_profile, aiida_local_code_factory, tmp_path
):
    """The multi-block segment builds a decompose pass per block plus a gather.

    Construction-level (nothing runs): mirrors the ``self_hartree`` route's
    graph-build tests. The end-to-end WF-to-alpha alignment is exercised by
    the pure-python `assemble_power_spectrum_dataset` discriminator tests in
    `test_ml_helpers.py`; running the graph awaits a daemon regression.
    """
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import PowerSpectrumDatasetWorkflow

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.pw2wannier_decompose")
    nscf = orm.RemoteData(computer=code.computer, remote_path=str(tmp_path)).store()
    blocks = {}
    for label in ("occ", "emp"):
        folder = orm.FolderData()
        folder.base.repository.put_object_from_filelike(io.BytesIO(b"u"), "aiida_u.mat")
        folder.base.repository.put_object_from_filelike(
            io.BytesIO(b"1\n\nX 0 0 0\n"), "aiida_centres.xyz"
        )
        folder.store()
        blocks[label] = {
            "retrieved": folder,
            "remote_folder": nscf,
            "nnkp_file": orm.SinglefileData(io.BytesIO(b"n"), filename=f"{label}.nnkp").store(),
        }
    merge_groups = [
        {"filled": True, "spin": "none", "blocks": [{"label": "occ"}]},
        {"filled": False, "spin": "none", "blocks": [{"label": "emp"}]},
    ]
    alphas = {"filled": {"none": [0.1]}, "empty": {"none": [0.5]}}

    wg = PowerSpectrumDatasetWorkflow.build(
        code=code,
        nscf_remote_folder=nscf,
        block_wannierizations=blocks,
        merge_groups=merge_groups,
        alphas=alphas,
    )
    names = [t.name for t in wg.tasks]
    # One decompose pass per block, plus the gather/align step.
    assert "decompose_occ" in names
    assert "decompose_emp" in names
    assert any("align_block_descriptors" in n for n in names)


def _spin_block(label, filled, spin, num_bands, num_wann):
    """Build a merge-group carrying the band counts the u_dis decision reads."""
    return {
        "filled": filled,
        "spin": spin,
        "blocks": [{"label": label, "num_bands": num_bands, "num_wann": num_wann}],
    }


def _block_wannierization(label, *, with_u_dis):
    """Build a stored per-block WannierizeBlockOutputs-shaped namespace entry."""
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


def test_power_spectrum_dataset_workflow_threads_nnkp_udis_spin(
    aiida_profile, aiida_local_code_factory, tmp_path
):
    """The fan-out wires nnkp always, u_dis only for disentangling blocks, spin per channel.

    Construction-level: an nspin=2 layout with a disentangling empty manifold
    (num_bands > num_wann) per channel. Asserts (a) every decompose task takes
    its block's nnkp, (b) ``spin_component`` is set from the manifold's spin
    channel, and (c) ``extract_u_dis_mat`` fires exactly for the disentangling
    blocks and feeds their ``u_dis_mat`` input.
    """
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import PowerSpectrumDatasetWorkflow

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.pw2wannier_decompose")
    nscf = orm.RemoteData(computer=code.computer, remote_path=str(tmp_path)).store()

    labels = ("occ_up", "emp_up", "occ_down", "emp_down")
    # Empty manifolds disentangle (num_bands 6 > num_wann 2); occupied do not.
    disentangling = {"emp_up", "emp_down"}
    block_wannierizations = {
        label: _block_wannierization(label, with_u_dis=label in disentangling) for label in labels
    }
    merge_groups = [
        _spin_block("occ_up", True, "up", 4, 4),
        _spin_block("emp_up", False, "up", 6, 2),
        _spin_block("occ_down", True, "down", 4, 4),
        _spin_block("emp_down", False, "down", 6, 2),
    ]
    alphas = {
        "filled": {"up": [0.1], "down": [0.1]},
        "empty": {"up": [0.5], "down": [0.5]},
    }

    wg = PowerSpectrumDatasetWorkflow.build(
        code=code,
        nscf_remote_folder=nscf,
        block_wannierizations=block_wannierizations,
        merge_groups=merge_groups,
        alphas=alphas,
    )
    names = [t.name for t in wg.tasks]

    # (a) nnkp threaded into every decompose pass (linked from the block input).
    for label in labels:
        nnkp_socket = wg.tasks[f"decompose_{label}"].inputs["nnkp"]
        assert nnkp_socket.value is not None or nnkp_socket._links, (
            f"decompose_{label} has no nnkp wired"
        )

    # (b) spin_component set per manifold channel.
    assert wg.tasks["decompose_occ_up"].inputs["parameters"].value["spin_component"] == "up"
    assert wg.tasks["decompose_emp_down"].inputs["parameters"].value["spin_component"] == "down"

    # (c) u_dis lifted for the disentangling manifolds only.
    u_dis_tasks = [n for n in names if "extract_u_dis_mat" in n]
    assert len(u_dis_tasks) == len(disentangling)
    for label in disentangling:
        u_dis_socket = wg.tasks[f"decompose_{label}"].inputs["u_dis_mat"]
        assert u_dis_socket.value is not None or u_dis_socket._links, (
            f"decompose_{label} disentangles but has no u_dis_mat wired"
        )
    for label in ("occ_up", "occ_down"):
        u_dis_socket = wg.tasks[f"decompose_{label}"].inputs["u_dis_mat"]
        assert u_dis_socket.value is None and not u_dis_socket._links, (
            f"decompose_{label} does not disentangle but has u_dis_mat wired"
        )


def test_power_spectrum_dataset_builds_without_parallelization(
    aiida_profile, aiida_local_code_factory, tmp_path
):
    """Every decompose pass carries resources when no parallelization is given.

    A CalcJob is rejected at creation without ``metadata.options.resources``,
    and that rejection only surfaces once the task runs, so pin it here.
    """
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import PowerSpectrumDatasetWorkflow

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.pw2wannier_decompose")
    nscf = orm.RemoteData(computer=code.computer, remote_path=str(tmp_path)).store()
    block_wannierizations = {"occ": _block_wannierization("occ", with_u_dis=False)}
    merge_groups = [_spin_block("occ", True, "none", 4, 4)]
    alphas = {"filled": {"none": [0.1]}, "empty": {"none": []}}

    wg = PowerSpectrumDatasetWorkflow.build(
        code=code,
        nscf_remote_folder=nscf,
        block_wannierizations=block_wannierizations,
        merge_groups=merge_groups,
        alphas=alphas,
    )

    resources = wg.tasks["decompose_occ"].inputs["metadata"]["options"]["resources"].value
    assert resources == {"num_machines": 1}, resources


def test_power_spectrum_dataset_nspin1_omits_spin_component(
    aiida_profile, aiida_local_code_factory, tmp_path
):
    """On an nspin=1 (spin=none) scratch no ``spin_component`` is injected."""
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import PowerSpectrumDatasetWorkflow

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.pw2wannier_decompose")
    nscf = orm.RemoteData(computer=code.computer, remote_path=str(tmp_path)).store()
    block_wannierizations = {"occ": _block_wannierization("occ", with_u_dis=False)}
    merge_groups = [_spin_block("occ", True, "none", 4, 4)]
    alphas = {"filled": {"none": [0.1]}, "empty": {"none": []}}

    wg = PowerSpectrumDatasetWorkflow.build(
        code=code,
        nscf_remote_folder=nscf,
        block_wannierizations=block_wannierizations,
        merge_groups=merge_groups,
        alphas=alphas,
    )
    params = wg.tasks["decompose_occ"].inputs["parameters"].value
    # No decompose_parameters and spin=none -> no parameters injected at all.
    assert params is None or "spin_component" not in params


def test_power_spectrum_dataset_survives_a_replayed_worker(
    aiida_profile, aiida_local_code_factory, tmp_path
):
    """The ``dataset`` graph output is wired socket by socket, so phantom sockets pass.

    A worker that has replayed a cached python task's namespaced outputs
    builds every later python task with those outputs attached. Returning
    ``align_block_descriptors``' whole output namespace as the graph's
    ``dataset`` would then be rejected: the namespaces no longer have the
    same children.
    """
    from aiida import orm

    from aiida_koopmans.workgraphs import ml
    from tests.fixtures import assert_graph_roundtrips, replayed_namespace_outputs

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.pw2wannier_decompose")
    nscf = orm.RemoteData(computer=code.computer, remote_path=str(tmp_path)).store()
    block_wannierizations = {"occ": _block_wannierization("occ", with_u_dis=False)}
    merge_groups = [_spin_block("occ", True, "none", 4, 4)]
    alphas = {"filled": {"none": [0.1]}, "empty": {"none": []}}

    with replayed_namespace_outputs(
        "alphas.filled",
        "errors.filled",
        handles=[(ml, "align_block_descriptors")],
    ):
        wg = ml.PowerSpectrumDatasetWorkflow.build(
            code=code,
            nscf_remote_folder=nscf,
            block_wannierizations=block_wannierizations,
            merge_groups=merge_groups,
            alphas=alphas,
        )
        align = next(t for t in wg.tasks if "align_block_descriptors" in t.name)
        assert {"alphas", "errors"} <= {socket._name for socket in align.outputs}
        assert_graph_roundtrips(wg)


def test_compute_block_descriptors_returns_cross_power(aiida_profile):
    """`compute_block_descriptors` cross-powers a block's decompose arrays."""
    import numpy as np
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import compute_block_descriptors
    from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

    n_max, l_max = 2, 1
    n_coeff = n_max * (l_max + 1) ** 2
    rng = np.random.default_rng(7)
    coeff = rng.standard_normal((2, n_coeff))
    group = rng.standard_normal((2, n_coeff))
    coeff_node = orm.ArrayData()
    coeff_node.set_array("coefficients", coeff)
    group_node = orm.ArrayData()
    group_node.set_array("group_coefficients", group)

    out = compute_block_descriptors._callable(
        coefficients=coeff_node,
        group_coefficients=group_node,
        output_parameters={"n_max": n_max, "l_max": l_max},
    )
    descriptors = out.get_array("descriptors")
    expected = ml_helpers.cross_power_spectra(coeff, group, n_max, l_max)
    assert descriptors.shape == expected.shape
    assert np.allclose(descriptors, expected)


def test_align_block_descriptors_orders_by_alphascreening(aiida_profile):
    """`align_block_descriptors` gathers block arrays into an aligned dataset."""
    import numpy as np
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import align_block_descriptors

    occ = orm.ArrayData()
    occ.set_array("descriptors", np.array([[1.0], [2.0]]))
    emp = orm.ArrayData()
    emp.set_array("descriptors", np.array([[10.0]]))
    merge_groups = [
        {"filled": True, "spin": "none", "blocks": [{"label": "occ"}]},
        {"filled": False, "spin": "none", "blocks": [{"label": "emp"}]},
    ]
    alphas = {"filled": {"none": [0.1, 0.2]}, "empty": {"none": [0.5]}}

    ds = align_block_descriptors._callable(
        block_descriptors={"occ": occ, "emp": emp},
        merge_groups=merge_groups,
        alphas=alphas,
    )
    assert ds["descriptors"] == [[1.0], [2.0], [10.0]]
    assert ds["alpha_targets"] == [0.1, 0.2, 0.5]
    assert ds["filled"] == [True, True, False]


def test_require_wannier_route_inputs_missing_scratch_raises():
    """The power_spectrum route names the requirement when the nscf scratch is absent."""
    from aiida_koopmans.workgraphs.ml import require_wannier_route_inputs

    # Molecular (KS-init) route: KoopmansDSCFOutputs omits nscf_remote_folder.
    with pytest.raises(ValueError, match=r"requires `nscf_remote_folder`"):
        require_wannier_route_inputs(None, {}, [])


def test_require_wannier_route_inputs_missing_block_raises():
    """A merge-group block with no wannierization is named, not a bare KeyError."""
    from aiida_koopmans.workgraphs.ml import require_wannier_route_inputs

    merge_groups = [{"filled": True, "spin": "none", "blocks": [{"label": "occ"}]}]
    with pytest.raises(ValueError, match="occ"):
        # Non-None scratch clears the first guard; the empty namespace trips the block guard.
        require_wannier_route_inputs(object(), {}, merge_groups)


def test_require_wannier_route_inputs_accepts_complete_inputs():
    """With scratch and every block present the guard is a no-op (returns None)."""
    from aiida_koopmans.workgraphs.ml import require_wannier_route_inputs

    merge_groups = [{"filled": True, "spin": "none", "blocks": [{"label": "occ"}]}]
    assert require_wannier_route_inputs(object(), {"occ": object()}, merge_groups) is None


class TestArrayInputShapes:
    """The descriptor tasks take an array socket whichever shape it arrives in.

    ``aiida-pythonjob`` deserializes a single-array ``ArrayData`` input to a
    bare ``numpy`` array before the body runs, so a task that only knew how
    to unwrap a node died on the live decompose route.
    """

    @staticmethod
    def _output_parameters():
        return {"n_max": 2, "l_max": 1}

    def test_compute_block_descriptors_accepts_bare_arrays(self):
        import numpy as np

        from aiida_koopmans.workgraphs.ml import compute_block_descriptors

        coefficients = np.arange(16, dtype=float).reshape(2, 8)
        group = np.arange(16, dtype=float).reshape(2, 8) + 0.5
        out = compute_block_descriptors._callable(
            coefficients=coefficients,
            group_coefficients=group,
            output_parameters=self._output_parameters(),
        )
        assert out.get_array("descriptors").shape[0] == 2

    def test_compute_block_descriptors_accepts_nodes(self, aiida_profile):
        import numpy as np
        from aiida import orm

        from aiida_koopmans.workgraphs.ml import compute_block_descriptors

        coefficients = orm.ArrayData()
        coefficients.set_array("coefficients", np.arange(16, dtype=float).reshape(2, 8))
        group = orm.ArrayData()
        group.set_array("group_coefficients", np.arange(16, dtype=float).reshape(2, 8) + 0.5)
        out = compute_block_descriptors._callable(
            coefficients=coefficients,
            group_coefficients=group,
            output_parameters=self._output_parameters(),
        )
        assert out.get_array("descriptors").shape[0] == 2

    def test_align_block_descriptors_accepts_bare_arrays(self):
        import numpy as np

        from aiida_koopmans.workgraphs.ml import align_block_descriptors

        dataset = align_block_descriptors._callable(
            block_descriptors={"occ": np.array([[1.0, 2.0]])},
            merge_groups=[{"filled": True, "spin": "none", "blocks": [{"label": "occ"}]}],
            alphas={"filled": {"none": [0.6]}, "empty": {}},
        )
        assert dataset["descriptors"] == [[1.0, 2.0]]
        assert dataset["alpha_targets"] == [0.6]
