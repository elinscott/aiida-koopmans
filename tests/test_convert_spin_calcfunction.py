"""Unit tests for :func:`convert_spin1_to_spin2` (``@task.calcfunction``).

The calcfunction performs no external work — it just byte-substitutes the
contents of a handful of nspin=1 wavefunction / Hamiltonian files into
their spin-up / spin-down nspin=2 counterparts. The tests exercise both:

* the pure byte-substitution helper (no AiiDA profile); and
* the calcfunction end-to-end against trivial ``RemoteData`` inputs
  pointing at on-disk fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiida_koopmans.workgraphs.convert_spin import (
    _CONVERSION_MAP,
    _convert_spin1_to_spin2_bytes,
    convert_spin1_to_spin2,
)

# ---------------------------------------------------------------------------
# Pure substitution helper (no AiiDA profile)
# ---------------------------------------------------------------------------


def test_convert_bytes_substitutes_nk_and_nspin_in_up_channel():
    src = b'<HEADER nk="1" nspin="1"/>'
    up, _down = _convert_spin1_to_spin2_bytes(src)
    assert b'nk="2"' in up
    assert b'nspin="2"' in up
    assert b'nk="1"' not in up
    assert b'nspin="1"' not in up


def test_convert_bytes_down_channel_also_substitutes_ik_and_ispin():
    src = b'<HEADER nk="1" nspin="1" ik="1" ispin="1"/>'
    up, down = _convert_spin1_to_spin2_bytes(src)
    # Up channel: nk + nspin only.
    assert b'ik="1"' in up
    assert b'ispin="1"' in up
    # Down channel: nk + nspin + ik + ispin.
    assert b'nk="2"' in down
    assert b'nspin="2"' in down
    assert b'ik="2"' in down
    assert b'ispin="2"' in down
    assert b'ik="1"' not in down
    assert b'ispin="1"' not in down


def test_convert_bytes_preserves_binary_payload():
    # Real evc files have a small XML header followed by binary float data.
    payload = bytes(range(256))
    src = b'<HEADER nk="1" nspin="1"/>\n' + payload
    up, down = _convert_spin1_to_spin2_bytes(src)
    assert up.endswith(payload)
    assert down.endswith(payload)


def test_conversion_map_matches_legacy_filename_set():
    # Mirror of the (manually-curated) legacy list at
    # koopmans/src/koopmans/calculators/_koopmans_cp.py:597-598.
    expected_sources = {
        "evc0.dat",
        "evc0_empty1.dat",
        "evcm.dat",
        "evc.dat",
        "hamiltonian.xml",
        "eigenval.xml",
        "evc_empty1.dat",
        "lambda01.dat",
    }
    assert {src for src, _, _ in _CONVERSION_MAP} == expected_sources


# ---------------------------------------------------------------------------
# End-to-end calcfunction execution against on-disk fixtures
# ---------------------------------------------------------------------------


def _populate_spin1_save(root: Path) -> dict[str, bytes]:
    """Drop a representative set of nspin=1 files into a fake remote root."""
    k_dir = root / "out" / "aiida_60.save" / "K00001"
    k_dir.mkdir(parents=True)

    sources = {
        "evc0.dat": b'<INFO nk="1" nspin="1" ik="1" ispin="1"/>\nBYTES0',
        "evc.dat": b'<INFO nk="1" nspin="1"/>\nBYTES1',
        "hamiltonian.xml": b'<HAM nk="1" nspin="1" ispin="1"/>',
    }
    for name, content in sources.items():
        (k_dir / name).write_bytes(content)
    return sources


def _populate_spin2_dummy_save(root: Path) -> None:
    """Drop a tiny nspin=2 skeleton onto a fake remote root."""
    save_dir = root / "out" / "aiida_60.save"
    k_dir = save_dir / "K00001"
    k_dir.mkdir(parents=True)
    # Files unrelated to the conversion map — these should survive in the
    # output via the dummy-skeleton copy step.
    (save_dir / "charge-density.dat").write_bytes(b"DENSITY")
    (save_dir / "data-file-schema.xml").write_bytes(b"<XML/>")
    # Placeholder wfc files with garbage; should be overwritten by the
    # converted versions when the calcfunction overlays the new content.
    (k_dir / "evc1.dat").write_bytes(b"GARBAGE")
    (k_dir / "evc2.dat").write_bytes(b"GARBAGE")


@pytest.fixture
def convert_spin_remote_inputs(tmp_path, fixture_localhost):
    """Build two ``RemoteData`` inputs with realistic on-disk fixtures."""
    from aiida import orm

    spin1_root = tmp_path / "spin1_parent"
    spin2_root = tmp_path / "spin2_dummy_parent"
    spin1_root.mkdir()
    spin2_root.mkdir()

    spin1_sources = _populate_spin1_save(spin1_root)
    _populate_spin2_dummy_save(spin2_root)

    spin1_remote = orm.RemoteData(computer=fixture_localhost, remote_path=str(spin1_root))
    spin1_remote.store()
    spin2_remote = orm.RemoteData(computer=fixture_localhost, remote_path=str(spin2_root))
    spin2_remote.store()

    return {
        "spin1_parent_folder": spin1_remote,
        "spin2_dummy_parent_folder": spin2_remote,
        "spin1_sources": spin1_sources,
    }


def test_convert_spin1_to_spin2_writes_converted_wfc_files(
    aiida_profile,
    convert_spin_remote_inputs,
):
    """Run the calcfunction and verify the substitution + file layout."""
    # ``_callable`` is the underlying aiida-core ``calcfunction``-wrapped
    # function — calling it executes the process and returns the output
    # dict (with stored Data nodes).
    outputs = convert_spin1_to_spin2._callable(
        spin1_parent_folder=convert_spin_remote_inputs["spin1_parent_folder"],
        spin2_dummy_parent_folder=convert_spin_remote_inputs["spin2_dummy_parent_folder"],
    )

    remote = outputs["remote_folder"]
    out_path = Path(remote.get_remote_path())
    assert out_path.is_dir()

    k_dir = out_path / "out" / "aiida_60.save" / "K00001"
    assert k_dir.is_dir()

    # The dummy skeleton (density, xml metadata) must have been carried over.
    assert (out_path / "out" / "aiida_60.save" / "charge-density.dat").read_bytes() == b"DENSITY"

    # evc0.dat (the file with all four matchable attributes) — strictest test.
    src = convert_spin_remote_inputs["spin1_sources"]["evc0.dat"]
    up_expected = src.replace(b'nk="1"', b'nk="2"').replace(b'nspin="1"', b'nspin="2"')
    down_expected = up_expected.replace(b'ik="1"', b'ik="2"').replace(b'ispin="1"', b'ispin="2"')
    assert (k_dir / "evc01.dat").read_bytes() == up_expected
    assert (k_dir / "evc02.dat").read_bytes() == down_expected
    # Verify the substitutions landed (per the spec call-out).
    assert b'nk="2"' in (k_dir / "evc01.dat").read_bytes()
    assert b'ispin="1"' in (k_dir / "evc01.dat").read_bytes()
    assert b'nk="2"' in (k_dir / "evc02.dat").read_bytes()
    assert b'ispin="2"' in (k_dir / "evc02.dat").read_bytes()

    # evc.dat → evc1.dat / evc2.dat. These names collide with the dummy's
    # placeholder wfc files; the calcfunction must overwrite GARBAGE with
    # the converted content.
    src = convert_spin_remote_inputs["spin1_sources"]["evc.dat"]
    up_expected = src.replace(b'nk="1"', b'nk="2"').replace(b'nspin="1"', b'nspin="2"')
    assert (k_dir / "evc1.dat").read_bytes() == up_expected
    # No ik/ispin in evc.dat, so down content equals up content.
    assert (k_dir / "evc2.dat").read_bytes() == up_expected
    # And the placeholder GARBAGE must be gone.
    assert b"GARBAGE" not in (k_dir / "evc1.dat").read_bytes()

    # hamiltonian.xml → hamiltonian1.xml / hamiltonian2.xml.
    src = convert_spin_remote_inputs["spin1_sources"]["hamiltonian.xml"]
    up_expected = src.replace(b'nk="1"', b'nk="2"').replace(b'nspin="1"', b'nspin="2"')
    down_expected = up_expected.replace(b'ispin="1"', b'ispin="2"')
    assert (k_dir / "hamiltonian1.xml").read_bytes() == up_expected
    assert (k_dir / "hamiltonian2.xml").read_bytes() == down_expected


def test_convert_spin1_to_spin2_returns_stored_remote_data(
    aiida_profile,
    convert_spin_remote_inputs,
    fixture_localhost,
):
    """The calcfunction's output must be a stored ``RemoteData`` on the right computer."""
    from aiida import orm

    outputs = convert_spin1_to_spin2._callable(
        spin1_parent_folder=convert_spin_remote_inputs["spin1_parent_folder"],
        spin2_dummy_parent_folder=convert_spin_remote_inputs["spin2_dummy_parent_folder"],
    )
    remote = outputs["remote_folder"]
    assert isinstance(remote, orm.RemoteData)
    assert remote.is_stored
    assert remote.computer.pk == fixture_localhost.pk


def test_convert_spin1_to_spin2_rejects_mismatched_computers(
    aiida_profile,
    tmp_path,
    fixture_localhost,
):
    """If the two parents live on different computers, fail loudly."""
    from aiida import orm
    from aiida.orm import Computer

    # Build a second computer to deliberately mismatch.
    other = Computer(
        label="other_localhost",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.direct",
        workdir=str(tmp_path / "other_workdir"),
    ).store()
    other.configure()

    spin1_root = tmp_path / "spin1"
    spin2_root = tmp_path / "spin2"
    _populate_spin1_save(spin1_root)
    _populate_spin2_dummy_save(spin2_root)

    spin1_remote = orm.RemoteData(computer=fixture_localhost, remote_path=str(spin1_root))
    spin1_remote.store()
    spin2_remote = orm.RemoteData(computer=other, remote_path=str(spin2_root))
    spin2_remote.store()

    # Direct call: the calcfunction raises during execution; aiida-core
    # surfaces that as the original exception.
    with pytest.raises(ValueError, match="same computer"):
        convert_spin1_to_spin2._callable(
            spin1_parent_folder=spin1_remote,
            spin2_dummy_parent_folder=spin2_remote,
        )


def test_convert_spin1_to_spin2_raises_when_no_spin1_files_present(
    aiida_profile,
    tmp_path,
    fixture_localhost,
):
    """If the spin1 parent has no recognisable wfc files, we fail loudly."""
    from aiida import orm

    spin1_root = tmp_path / "spin1_empty"
    (spin1_root / "out" / "aiida_60.save" / "K00001").mkdir(parents=True)
    spin1_remote = orm.RemoteData(computer=fixture_localhost, remote_path=str(spin1_root))
    spin1_remote.store()

    spin2_root = tmp_path / "spin2_dummy"
    _populate_spin2_dummy_save(spin2_root)
    spin2_remote = orm.RemoteData(computer=fixture_localhost, remote_path=str(spin2_root))
    spin2_remote.store()

    with pytest.raises(FileNotFoundError, match="No known nspin=1 wavefunction files"):
        convert_spin1_to_spin2._callable(
            spin1_parent_folder=spin1_remote,
            spin2_dummy_parent_folder=spin2_remote,
        )
