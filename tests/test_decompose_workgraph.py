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
