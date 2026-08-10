"""Tier 7D: the Jones record in the fingerprint and on disk.

``Tier7JonesSciencePlan.md`` Section 25.  Invariant **I13**: changing any single
Jones parameter changes ``scientific_sha256``; changing no Jones parameter
leaves it unchanged; ``instrument_sha256`` is unchanged by every Jones
configuration.

The absence half of **I1** is here too, at the fingerprint level: equivalent
current runs with no ``jones:`` section add no optional Jones snapshot to the
digest.  Always-present factors, including the canonical receptor convention,
are fingerprinted independently; this test does not promise pre-SCI-006 digest
compatibility.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.core.jones_terms import EMPTY_JONES_TERMS
from radiosim.io.hdf5 import SCHEMA_VERSION, load_result_hdf5, write_result_hdf5
from radiosim.io.summary_json import write_result_summary_json
from tests.fixtures.configs import valid_config_mapping

_GAIN: dict[str, Any] = {"G": {"amplitude_error": 0.02}}
_GAIN_PERTURBED: dict[str, Any] = {"G": {"amplitude_error": 0.020000001}}
_BANDPASS: dict[str, Any] = {
    "B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.0, -0.05]}}
}


def _run(tmp_path, jones: dict[str, Any] | None, name: str = "run"):
    """Run one complete simulation, with or without a ``jones:`` section."""
    work = tmp_path / name
    work.mkdir(parents=True, exist_ok=True)
    data = valid_config_mapping(work)
    if jones is not None:
        data["jones"] = jones
    simulator = Simulator.from_mapping(data, base_dir=work)
    return simulator.run(progress=False)


# ---------------------------------------------------------------------------
# I13 -- fingerprint sensitivity
# ---------------------------------------------------------------------------


def test_absence_leaves_the_scientific_fingerprint_unchanged(tmp_path) -> None:
    """The I1 property at the digest level, and the cube with it.

    Two current runs of the same configuration and empty optional-term
    inventory must be the same run.  Byte equality of the cube is asserted
    alongside the digest, because a digest can agree for the wrong reason and a
    cube cannot.
    """
    first = _run(tmp_path, None, "a")
    second = _run(tmp_path, None, "b")

    assert first.scientific_sha256 == second.scientific_sha256
    np.testing.assert_array_equal(first.visibilities, second.visibilities)
    assert dict(first.jones) == {}
    assert first.jones == second.jones


def test_enabling_a_term_changes_the_scientific_fingerprint(tmp_path) -> None:
    """I13, first half."""
    without = _run(tmp_path, None, "a")
    with_gain = _run(tmp_path, _GAIN, "b")

    assert with_gain.scientific_sha256 != without.scientific_sha256
    assert tuple(dict(with_gain.jones)["enabled_terms"]) == ("H", "G", "C", "E")


def test_changing_one_jones_parameter_changes_the_fingerprint(tmp_path) -> None:
    """I13: *any* single parameter, including one too small to see in the cube.

    The perturbation here is one part in twenty million.  The cube moves by
    about that much, and the digest must move regardless: a fingerprint that
    only noticed large changes would let two scientifically different runs be
    filed as the same run.
    """
    baseline = _run(tmp_path, _GAIN, "a")
    perturbed = _run(tmp_path, _GAIN_PERTURBED, "b")

    assert baseline.scientific_sha256 != perturbed.scientific_sha256
    assert dict(baseline.jones)["jones_sha256"] != dict(perturbed.jones)["jones_sha256"]


def test_two_runs_of_the_same_jones_configuration_agree(tmp_path) -> None:
    """I13's converse: changing no Jones parameter leaves the digest alone."""
    first = _run(tmp_path, _GAIN, "a")
    second = _run(tmp_path, _GAIN, "b")

    assert first.scientific_sha256 == second.scientific_sha256
    np.testing.assert_array_equal(first.visibilities, second.visibilities)


def test_the_instrument_fingerprint_is_untouched_by_every_jones_configuration(
    tmp_path,
) -> None:
    """I13's third clause.

    The instrument is what the array *is*; the Jones terms are what its
    electronics do to the signal.  A gain that changed the instrument
    fingerprint would make two runs on the same telescope look like runs on two
    telescopes.
    """
    without = _run(tmp_path, None, "a")
    with_gain = _run(tmp_path, _GAIN, "b")
    with_bandpass = _run(tmp_path, _BANDPASS, "c")

    digests = {
        result.instrument.provenance.instrument_sha256
        for result in (without, with_gain, with_bandpass)
    }
    assert len(digests) == 1

    receptor_digests = {
        result.receptors.provenance.receptor_sha256
        for result in (without, with_gain, with_bandpass)
    }
    assert len(receptor_digests) == 1


def test_the_jones_digest_covers_configuration_and_not_the_filesystem(
    tmp_path,
) -> None:
    """The ``RUN-005``/``RUN-006`` lesson applied to a new section.

    Two runs of the same Jones configuration from two different working
    directories must produce the same ``jones_sha256``.  Nothing
    filesystem-path-derived enters the Jones snapshot, and the two runs above
    already live in different directories, so this is a direct check rather than
    an inspection.
    """
    first = _run(tmp_path, _BANDPASS, "somewhere")
    second = _run(tmp_path, _BANDPASS, "somewhere-else")

    assert dict(first.jones)["jones_sha256"] == dict(second.jones)["jones_sha256"]
    assert str(tmp_path) not in repr(dict(first.jones))


# ---------------------------------------------------------------------------
# Serialization (Section 25.2)
# ---------------------------------------------------------------------------


def test_the_hdf5_group_round_trips_the_jones_record(tmp_path) -> None:
    """Write, read back, and recompute both fingerprints."""
    result = _run(tmp_path, {**_GAIN, **_BANDPASS}, "run")
    path = write_result_hdf5(result, tmp_path / "with-jones.h5")

    loaded = load_result_hdf5(path)

    assert loaded.scientific_sha256 == result.scientific_sha256
    assert dict(loaded.jones) == dict(result.jones)
    assert tuple(dict(loaded.jones)["enabled_terms"]) == ("H", "G", "B", "C", "E")
    assert set(dict(loaded.jones)["term_snapshots"]) == {"G", "B"}


def test_a_file_from_a_run_with_no_terms_carries_no_jones_group(tmp_path) -> None:
    """Section 25.2: the group exists only when there is something to record.

    Checked with h5py directly rather than through the reader, because "the
    reader is happy" would also be true of a group full of empty strings.
    """
    h5py = pytest.importorskip("h5py")
    result = _run(tmp_path, None, "run")
    path = write_result_hdf5(result, tmp_path / "no-jones.h5")

    with h5py.File(path, "r") as handle:
        assert "jones" not in handle
        assert bytes(handle.attrs["schema_version"]).decode() == SCHEMA_VERSION

    loaded = load_result_hdf5(path)
    assert dict(loaded.jones) == {}
    assert loaded.scientific_sha256 == result.scientific_sha256


def test_a_partially_present_jones_group_is_rejected(tmp_path) -> None:
    """All-or-nothing.

    The group is optional; a *fragment* of it is a corrupted file, and the
    allowlist says so rather than reading whatever survived.
    """
    h5py = pytest.importorskip("h5py")
    from radiosim.io.result_errors import UnsafeResultInputError

    result = _run(tmp_path, _GAIN, "run")
    path = write_result_hdf5(result, tmp_path / "partial.h5")

    with h5py.File(path, "r+") as handle:
        del handle["jones/mount_types_json"]

    with pytest.raises(UnsafeResultInputError):
        load_result_hdf5(path)


def test_the_summary_json_carries_the_bounded_jones_block(tmp_path) -> None:
    """Section 25.2's summary row."""
    result = _run(tmp_path, _GAIN, "run")
    path = write_result_summary_json(result, tmp_path / "run.summary.json")

    payload = json.loads(path.read_text(encoding="utf-8"))

    block = payload["jones"]
    assert block["enabled_terms"] == ["H", "G", "C", "E"]
    assert block["chain_order"] == ["H", "G", "C", "E"]
    assert block["jones_sha256"] == dict(result.jones)["jones_sha256"]
    assert block["terms"]["G"]["amplitude_error"] == pytest.approx(0.02)


def test_a_summary_from_a_run_with_no_optional_terms_says_so_explicitly(
    tmp_path,
) -> None:
    """Empty lists and a ``null`` digest, not an absent key.

    A reader must be able to tell "this run enabled no optional term" from
    "this summary predates the optional Jones record", and only one of those
    two is expressible by omitting the block.
    """
    result = _run(tmp_path, None, "run")
    path = write_result_summary_json(result, tmp_path / "run.summary.json")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["jones"] == {
        "enabled_terms": [],
        "chain_order": [],
        "jones_sha256": None,
        "terms": {},
    }


def test_the_measurement_set_and_uvfits_writers_are_unchanged(tmp_path) -> None:
    """Section 25.2: RadioSim does not write calibration tables, and 7D does not start.

    A corrupted visibility is still a visibility.  Asserted as a source scan
    rather than by writing a Measurement Set, because the claim is that no
    ``CALDEVICE`` or ``BANDPASS`` subtable was invented -- and the absence of a
    thing is not observable in a file that never had it.
    """
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[3] / "src" / "radiosim"
    for relative in ("io/measurement_set.py", "io/standard_visibility.py"):
        text = (source_root / relative).read_text(encoding="utf-8")
        assert "jones" not in text.lower()
        assert "CALDEVICE" not in text
        assert "BANDPASS" not in text


# ---------------------------------------------------------------------------
# The resolved carrier itself
# ---------------------------------------------------------------------------


def test_the_empty_inventory_snapshots_to_nothing() -> None:
    """The one property every bit-identity claim in this slice rests on."""
    assert EMPTY_JONES_TERMS.to_snapshot() == {}
    assert EMPTY_JONES_TERMS.is_empty


def test_a_provenance_record_verifies_its_own_digest() -> None:
    """A tampered record cannot be constructed, as for the receptor set.

    The container recomputing its own fingerprint is what makes the digest a
    property of the content rather than a field somebody remembered to update.
    """
    from dataclasses import replace

    honest = EMPTY_JONES_TERMS.provenance
    with pytest.raises(ValueError):
        replace(honest, enabled_terms=("H", "G", "C", "E"))
