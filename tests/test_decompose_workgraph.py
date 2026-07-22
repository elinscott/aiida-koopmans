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
    outputs, _ = extract_decompose_inputs._callable.run_get_node(hr_retrieved=folder)

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
        extract_decompose_inputs._callable.run_get_node(hr_retrieved=folder)


def test_orbital_density_dataset_workflow_fans_out_per_block(
    aiida_profile, aiida_local_code_factory, tmp_path
):
    """The multi-block segment builds a decompose pass per block plus a gather.

    Construction-level (nothing runs): mirrors the ``self_hartree`` route's
    graph-build tests. The end-to-end WF-to-alpha alignment is exercised by
    the pure-python `assemble_orbital_density_dataset` discriminator tests in
    `test_ml_helpers.py`; running the graph awaits a daemon regression.
    """
    from aiida import orm

    from aiida_koopmans.workgraphs.ml import OrbitalDensityDatasetWorkflow

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
            "hr_retrieved": folder,
            "remote_folder": nscf,
            "nnkp_file": orm.SinglefileData(io.BytesIO(b"n"), filename=f"{label}.nnkp").store(),
        }
    merge_groups = [
        {"filled": True, "spin": "none", "blocks": [{"label": "occ"}]},
        {"filled": False, "spin": "none", "blocks": [{"label": "emp"}]},
    ]
    alphas = {"filled": {"none": [0.1]}, "empty": {"none": [0.5]}}

    wg = OrbitalDensityDatasetWorkflow.build(
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


def test_compute_block_descriptors_returns_cross_power(aiida_profile):
    """`compute_block_descriptors` cross-powers a block's decompose arrays."""
    import numpy as np
    from aiida import orm

    from aiida_koopmans import ml_helpers
    from aiida_koopmans.workgraphs.ml import compute_block_descriptors

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
    assert ds["alphas"] == [0.1, 0.2, 0.5]
    assert ds["filled"] == [True, True, False]


def test_require_wannier_route_inputs_missing_scratch_raises():
    """The orbital_density route names the requirement when the nscf scratch is absent."""
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
