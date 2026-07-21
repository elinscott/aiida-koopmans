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
