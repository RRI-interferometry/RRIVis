"""Characterize pinned pyuvdata 3.2.1 receptor and polarization behavior.

This module resolves `Tier5ReceptorFeedPlan.md` §43 open question **Q3** by
executing the circular-feed writer round trips the design gate did not run.
Every probe is offline: no network, no registry, no bundled data file.  All
artifacts are written under pytest's ``tmp_path``, which is outside the
repository and is removed by pytest.

Q3 verdict
==========

The plan's Section 14.2 table and Section 22.1 assumptions **hold**, with one
construction-form correction:

* ``Telescope.new(feeds=["r", "l"], feed_angle=..., mount_type="fixed")`` does
  **not** configure circular feeds.  ``feeds`` is consumed only by
  ``set_feeds_from_x_orientation``, which ``Telescope.new`` invokes solely when
  ``x_orientation`` is supplied, so without ``x_orientation`` the argument is
  silently ignored and ``feed_array`` stays ``None``
  (``pyuvdata/telescopes.py:884-950``).  The working form passes ``feed_array``
  directly, and ``Telescope.new`` requires it with the full ``(Nants, Nfeeds)``
  shape even though ``update_params_from_known_telescopes`` documents a
  ``(Nfeeds,)`` shorthand.  Tier 5F must therefore write ``feed_array`` and
  ``feed_angle``, not ``feeds``; the existing linear writer
  (``src/radiosim/io/standard_visibility.py:887``) only works because it also
  passes ``x_orientation="east"``.
* With ``feed_array=[["r","l"], ...]``, ``feed_angle=0``,
  ``mount_type="fixed"``, no ``x_orientation``, and
  ``polarization_array=["rr","rl","lr","ll"]``, ``UVData.check()`` returns
  ``True`` both before and after phasing, and both writers succeed with no
  warning beyond the ones Tier 4 already classified.
* In-memory codes are ``(-1, -3, -4, -2)`` and the read-back order from both a
  Measurement Set and a UVFITS file is ``(-1, -2, -3, -4)`` — exactly the
  Section 14.2 circular row.
* Measurement Set ``CORR_TYPE`` preserves the **in-memory** order in both bases
  (``[5, 6, 7, 8]`` = RR, RL, LR, LL for circular; ``[9, 10, 11, 12]`` = XX, XY,
  YX, YY for linear, casacore ``Stokes`` enumeration).  The descending on-disk
  order the plan tabulates is produced by pyuvdata's *reader*, which
  canonicalizes the axis, not by the MS layout.  Section 14.3's delegation
  decision is unaffected; a reviewer inspecting ``CORR_TYPE`` directly must not
  expect ``(-1, -2, -3, -4)`` there.
* ``x_orientation`` is *not* required for circular feeds, and is not rejected
  alongside them either.  The deprecated ``Telescope.x_orientation`` accessor
  returns ``"east"`` for ``r``/``l`` feeds at zero feed angle, which is
  meaningless; Tier 5F must not derive basis information from it.
* pyuvdata does **not** cross-validate ``feed_array`` against
  ``polarization_array``: linear polarization codes on circular feeds pass
  ``check()``.  RadioSim must enforce that coupling itself.
"""

from __future__ import annotations

import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import pyuvdata
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from casacore.tables import table
from pyuvdata import Telescope, UVData
from pyuvdata.utils import ECEF_from_ENU
from pyuvdata.utils import pol as pol_utils

FREQUENCIES_HZ = np.array([100_000_000.0, 101_500_000.0], dtype=np.float64)
CHANNEL_WIDTHS_HZ = np.array([1_500_000.0, 1_500_000.0], dtype=np.float64)
TIMES = Time(["2025-01-01T00:00:00", "2025-01-01T00:00:02"])
ANTENNA_PAIRS = [(0, 0), (0, 2)]
ANTENNA_COUNT = 3

LINEAR_LABELS = ["xx", "xy", "yx", "yy"]
LINEAR_MEMORY_CODES = [-5, -7, -8, -6]
LINEAR_FILE_CODES = [-5, -6, -7, -8]
LINEAR_CORR_TYPE = [9, 10, 11, 12]

CIRCULAR_LABELS = ["rr", "rl", "lr", "ll"]
CIRCULAR_MEMORY_CODES = [-1, -3, -4, -2]
CIRCULAR_FILE_CODES = [-1, -2, -3, -4]
CIRCULAR_CORR_TYPE = [5, 6, 7, 8]

NUMPY_WHERE_WITHOUT_OUT_WARNING = (
    "'where' used without 'out', expect unitialized memory in output. "
    "If this is intentional, use out=None."
)


def _location_and_relative_ecef() -> tuple[EarthLocation, np.ndarray]:
    location = EarthLocation.from_geodetic(
        21.4 * u.deg,
        -30.7 * u.deg,
        1000.0 * u.m,
    )
    positions_enu = np.array(
        [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 30.0, 0.0]],
        dtype=np.float64,
    )
    center = np.array(
        [coordinate.to_value(u.m) for coordinate in location.geocentric],
        dtype=np.float64,
    )
    relative_ecef = ECEF_from_ENU(positions_enu, center_loc=location) - center
    return location, relative_ecef


def _telescope(
    feeds: list[str],
    feed_angle: list[float],
    *,
    use_feed_array: bool = True,
    x_orientation: str | None = None,
) -> Telescope:
    location, relative_ecef = _location_and_relative_ecef()
    extra: dict[str, object] = {}
    if x_orientation is not None:
        extra["x_orientation"] = x_orientation
    if use_feed_array:
        extra["feed_array"] = np.tile(
            np.asarray(feeds, dtype="<U1"),
            (ANTENNA_COUNT, 1),
        )
    else:
        extra["feeds"] = feeds
    return Telescope.new(
        name="Tier5ReceptorArray",
        instrument="Tier5ReceptorArray",
        location=location,
        antenna_positions={
            number: relative_ecef[number] for number in range(ANTENNA_COUNT)
        },
        antenna_names=["A0", "A1", "A2"],
        antenna_numbers=list(range(ANTENNA_COUNT)),
        antenna_diameters=[14.0, 15.0, 16.0],
        feed_angle=np.tile(
            np.asarray(feed_angle, dtype=np.float64),
            (ANTENNA_COUNT, 1),
        ),
        mount_type="fixed",
        update_from_known=False,
        **extra,
    )


def _deterministic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (4, 2, 4)
    real = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)
    data = (real + 1j * ((real * 0.123456789) + 0.03125)).astype(np.complex128)
    for row in (0, 2):
        data[row, :, 0] = data[row, :, 0].real
        data[row, :, 3] = data[row, :, 3].real
        data[row, :, 2] = np.conj(data[row, :, 1])
    flags = np.zeros(shape, dtype=bool)
    flags[3, 1, 2] = True
    nsamples = np.full(shape, 2.5, dtype=np.float64)
    nsamples[1, 0, 0] = 0.5
    return data, flags, nsamples


def _new_uvdata(labels: list[str], telescope: Telescope) -> UVData:
    data, flags, nsamples = _deterministic_data()
    return UVData.new(
        freq_array=FREQUENCIES_HZ,
        polarization_array=labels,
        times=TIMES.jd,
        antpairs=ANTENNA_PAIRS,
        telescope=telescope,
        do_blt_outer=True,
        time_axis_faster_than_bls=False,
        update_telescope_from_known=False,
        integration_time=2.0,
        channel_width=CHANNEL_WIDTHS_HZ,
        history="Offline deterministic Tier 5 receptor characterization.",
        vis_units="Jy",
        data_array=data,
        flag_array=flags,
        nsample_array=nsamples,
    )


def _classify_warnings(captured: list[warnings.WarningMessage]) -> set[str]:
    known = {
        (UserWarning, NUMPY_WHERE_WITHOUT_OUT_WARNING): "numpy-where-without-out",
    }
    categories: set[str] = set()
    unknown: list[str] = []
    for warning in captured:
        category = known.get((warning.category, str(warning.message)))
        if category is None:
            unknown.append(f"{warning.category.__name__}: {warning.message}")
        else:
            categories.add(category)
    assert unknown == [], f"Unclassified warnings: {unknown}"
    return categories


def _expected_ms_warning_categories() -> set[str]:
    if sys.version_info[:2] == (3, 12):
        return {"numpy-where-without-out"}
    return set()


def _projected(labels: list[str], telescope: Telescope) -> UVData:
    uvdata = _new_uvdata(labels, telescope)
    uvdata.polarization_array = np.asarray(
        uvdata.polarization_array,
        dtype=np.int64,
    )
    uvdata.phase_to_time(TIMES[0])
    return uvdata


def _circular_telescope() -> Telescope:
    return _telescope(["r", "l"], [0.0, 0.0])


def _linear_telescope() -> Telescope:
    return _telescope(["x", "y"], [np.pi / 2.0, 0.0])


# ---------------------------------------------------------------------------
# Static dependency tables
# ---------------------------------------------------------------------------


def test_dependency_polarization_code_tables_match_aips_memo_117() -> None:
    """Characterizes the installed AIPS code tables for both bases."""
    assert pyuvdata.__version__ == "3.2.1"
    for label, code in (
        ("rr", -1),
        ("ll", -2),
        ("rl", -3),
        ("lr", -4),
        ("xx", -5),
        ("yy", -6),
        ("xy", -7),
        ("yx", -8),
    ):
        assert pol_utils.POL_STR2NUM_DICT[label] == code
        assert pol_utils.POL_NUM2STR_DICT[code] == label

    assert pol_utils.POL_TO_FEED_DICT["rr"] == ["r", "r"]
    assert pol_utils.POL_TO_FEED_DICT["rl"] == ["r", "l"]
    assert pol_utils.POL_TO_FEED_DICT["lr"] == ["l", "r"]
    assert pol_utils.POL_TO_FEED_DICT["ll"] == ["l", "l"]


def test_dependency_feed_parameters_accept_both_bases() -> None:
    """Characterizes feed_array, feed_angle, and Nfeeds acceptance."""
    telescope = Telescope()
    assert telescope._feed_array.acceptable_vals == ["x", "y", "r", "l"]
    assert telescope._feed_array.form == ("Nants", "Nfeeds")
    assert telescope._feed_angle.form == ("Nants", "Nfeeds")
    assert telescope._Nfeeds.acceptable_vals == [1, 2]

    new_parameters = inspect.signature(Telescope.new).parameters
    assert {"feeds", "feed_array", "feed_angle", "mount_type"} <= set(new_parameters)


def test_dependency_feeds_argument_is_ignored_without_x_orientation() -> None:
    """Characterizes the Q3 construction-form correction.

    The Section 43 Q3 form leaves ``feed_array`` unset, so a Tier 5F writer built
    on it would emit files with no receptor metadata at all.
    """
    ignored = _telescope(["r", "l"], [0.0, 0.0], use_feed_array=False)
    assert ignored.feed_array is None
    assert ignored.feed_angle is None
    assert ignored.Nfeeds is None

    # Supplying x_orientation is what makes the `feeds` argument take effect,
    # which is why the existing linear writer works.
    honoured = _telescope(
        ["x", "y"],
        [np.pi / 2.0, 0.0],
        use_feed_array=False,
        x_orientation="east",
    )
    assert honoured.feed_array.tolist() == [["x", "y"]] * ANTENNA_COUNT

    # A one-dimensional feed_array is rejected by Telescope.new.
    location, relative_ecef = _location_and_relative_ecef()
    with pytest.raises(ValueError, match="not expected shape"):
        Telescope.new(
            name="Tier5ReceptorArray",
            location=location,
            antenna_positions={
                number: relative_ecef[number] for number in range(ANTENNA_COUNT)
            },
            antenna_names=["A0", "A1", "A2"],
            antenna_numbers=list(range(ANTENNA_COUNT)),
            feed_array=["r", "l"],
            feed_angle=np.zeros((ANTENNA_COUNT, 2), dtype=np.float64),
            mount_type="fixed",
            update_from_known=False,
        )


def test_dependency_circular_telescope_needs_no_x_orientation() -> None:
    """Characterizes circular feed acceptance and the useless legacy accessor."""
    telescope = _circular_telescope()
    assert telescope.feed_array.tolist() == [["r", "l"]] * ANTENNA_COUNT
    assert telescope.feed_angle.tolist() == [[0.0, 0.0]] * ANTENNA_COUNT
    assert telescope.Nfeeds == 2
    assert list(telescope.mount_type) == ["fixed"] * ANTENNA_COUNT
    assert telescope.check() is True

    # The deprecated derivation cannot distinguish the two bases.
    assert (
        pol_utils.get_x_orientation_from_feeds(
            feed_array=telescope.feed_array,
            feed_angle=telescope.feed_angle,
            tols=(0.0, 1e-6),
        )
        == "east"
    )
    linear = _linear_telescope()
    assert (
        pol_utils.get_x_orientation_from_feeds(
            feed_array=linear.feed_array,
            feed_angle=linear.feed_angle,
            tols=(0.0, 1e-6),
        )
        == "east"
    )

    # x_orientation alongside circular feeds is accepted, not rejected.
    assert _telescope(["r", "l"], [0.0, 0.0], x_orientation="east").check() is True


# ---------------------------------------------------------------------------
# In-memory ordering and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "memory_codes", "telescope_factory"),
    [
        (LINEAR_LABELS, LINEAR_MEMORY_CODES, _linear_telescope),
        (CIRCULAR_LABELS, CIRCULAR_MEMORY_CODES, _circular_telescope),
    ],
    ids=["linear", "circular"],
)
def test_dependency_uvdata_new_preserves_requested_polarization_order(
    labels: list[str],
    memory_codes: list[int],
    telescope_factory,
) -> None:
    """Characterizes UVData.new ordering and check() for both bases."""
    uvdata = _new_uvdata(labels, telescope_factory())
    assert uvdata.polarization_array == memory_codes
    assert uvdata.Npols == 4
    assert uvdata.telescope.Nfeeds == 2
    assert uvdata.check() is True

    uvdata.polarization_array = np.asarray(
        uvdata.polarization_array,
        dtype=np.int64,
    )
    uvdata.phase_to_time(TIMES[0])
    assert uvdata.check() is True


def test_dependency_does_not_cross_validate_feeds_against_polarizations() -> None:
    """Characterizes the absent basis coupling that RadioSim must supply."""
    mismatched = _new_uvdata(LINEAR_LABELS, _circular_telescope())
    assert mismatched.check() is True
    assert mismatched.polarization_array == LINEAR_MEMORY_CODES
    assert mismatched.telescope.feed_array.tolist() == [["r", "l"]] * ANTENNA_COUNT


# ---------------------------------------------------------------------------
# Writer round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "labels",
        "file_codes",
        "corr_type",
        "feed_letters",
        "receptor_angle",
        "telescope_factory",
    ),
    [
        (
            LINEAR_LABELS,
            LINEAR_FILE_CODES,
            LINEAR_CORR_TYPE,
            ["X", "Y"],
            np.pi / 2.0,
            _linear_telescope,
        ),
        (
            CIRCULAR_LABELS,
            CIRCULAR_FILE_CODES,
            CIRCULAR_CORR_TYPE,
            ["R", "L"],
            0.0,
            _circular_telescope,
        ),
    ],
    ids=["linear", "circular"],
)
def test_dependency_measurement_set_round_trips_both_bases(
    tmp_path: Path,
    labels: list[str],
    file_codes: list[int],
    corr_type: list[int],
    feed_letters: list[str],
    receptor_angle: float,
    telescope_factory,
) -> None:
    """Characterizes MS write, read-back order, CORR_TYPE, and the FEED table."""
    uvdata = _projected(labels, telescope_factory())
    path = tmp_path / "receptor.ms"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        uvdata.write_ms(str(path), clobber=False, force_phase=False)
    assert _classify_warnings(captured) == _expected_ms_warning_categories()

    readback = UVData()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        readback.read_ms(str(path))
    assert _classify_warnings(captured) <= _expected_ms_warning_categories()

    assert np.asarray(readback.polarization_array).tolist() == file_codes
    assert (
        readback.telescope.feed_array.tolist()
        == [[letter.lower() for letter in feed_letters]] * ANTENNA_COUNT
    )
    np.testing.assert_allclose(
        np.asarray(readback.telescope.feed_angle),
        np.tile([receptor_angle, 0.0], (ANTENNA_COUNT, 1)),
        rtol=0.0,
        atol=1e-9,
    )
    assert list(readback.telescope.mount_type) == ["fixed"] * ANTENNA_COUNT

    with table(str(path / "POLARIZATION"), ack=False) as polarization_table:
        assert polarization_table.getcol("NUM_CORR").tolist() == [4]
        # CORR_TYPE keeps the in-memory order; pyuvdata's reader sorts.
        assert polarization_table.getcol("CORR_TYPE").tolist() == [corr_type]

    with table(str(path / "FEED"), ack=False) as feed_table:
        polarization_type = feed_table.getcol("POLARIZATION_TYPE")
        assert polarization_type["shape"] == [ANTENNA_COUNT, 2]
        assert polarization_type["array"] == feed_letters * ANTENNA_COUNT
        np.testing.assert_allclose(
            np.asarray(feed_table.getcol("RECEPTOR_ANGLE")),
            np.tile([receptor_angle, 0.0], (ANTENNA_COUNT, 1)),
            rtol=0.0,
            atol=1e-9,
        )


@pytest.mark.parametrize(
    ("labels", "file_codes", "feed_letters", "telescope_factory"),
    [
        (LINEAR_LABELS, LINEAR_FILE_CODES, ["x", "y"], _linear_telescope),
        (CIRCULAR_LABELS, CIRCULAR_FILE_CODES, ["r", "l"], _circular_telescope),
    ],
    ids=["linear", "circular"],
)
def test_dependency_uvfits_round_trips_both_bases(
    tmp_path: Path,
    labels: list[str],
    file_codes: list[int],
    feed_letters: list[str],
    telescope_factory,
) -> None:
    """Characterizes UVFITS write and the descending on-disk code order."""
    uvdata = _projected(labels, telescope_factory())
    path = tmp_path / "receptor.uvfits"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        uvdata.write_uvfits(str(path), force_phase=False)
    assert _classify_warnings(captured) == set()

    readback = UVData()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        readback.read_uvfits(str(path))
    assert _classify_warnings(captured) == set()

    assert np.asarray(readback.polarization_array).tolist() == file_codes
    assert readback.telescope.feed_array.tolist() == [feed_letters] * ANTENNA_COUNT
    assert list(readback.telescope.mount_type) == ["fixed"] * ANTENNA_COUNT

    # Data survives the reordering when matched by AIPS code.
    original_codes = np.asarray(uvdata.polarization_array)
    actual_codes = np.asarray(readback.polarization_array)
    for code in file_codes:
        original_index = int(np.flatnonzero(original_codes == code)[0])
        actual_index = int(np.flatnonzero(actual_codes == code)[0])
        np.testing.assert_allclose(
            readback.data_array[:, :, actual_index],
            uvdata.data_array[:, :, original_index],
            rtol=2e-6,
            atol=1e-7,
        )
