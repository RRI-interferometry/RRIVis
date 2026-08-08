"""Tier-2 cross-implementation validation of the RIME against ``pyuvsim``.

``Tier7JonesSciencePlan.md`` Section 29 splits cross-validation evidence in two.
Tier-1 evidence -- comparisons against a closed-form expression written out in
the test body, or against a library already in the gating environment -- lives
in the ordinary unit suite and gates every slice.  Tier-2 evidence is a
comparison against an independent *simulator*, and Section 41's question Q1
established that the only such simulator resolvable against this repository's
``pyuvdata ==3.2.1`` pin is ``pyuvsim 1.4.0``, from PyPI, in an optional pixi
feature.  This module is that comparison.

**It never gates.**  Every test here is marked ``crossval`` *and* ``slow``, so
``pixi run test -- -m "not slow"`` -- the standard gate, and the two CI suites
-- deselect the module entirely.  The full-suite run in the ``default``
environment skips it at import, because ``pyuvsim`` is not installed there.  It
runs only under::

    pixi run --environment crossval -- python -m pytest tests/crossvalidation/ -m crossval

The current measured result on the authoring host is committed as
``output/crossvalidation/2026-08-08-pyuvsim-1.4.0.json``, which is the evidence
artifact Section 29 requires; the numbers asserted below are the ones in that
file.

What is compared, and what is not
=================================

Both simulators evaluate the same RIME on the same inputs -- the same antenna
ENU positions, the same UTC instants (as exact two-part Julian dates), the same
frequencies, the same point-source catalogue read from one ``skyh5`` file, and
the same primary beam read from one ``beamfits`` file whose response is
identically ``1`` over the whole visible hemisphere so that beam interpolation
cannot contribute a difference.  What is left is the part worth cross-checking:
the geometry (ENU baseline vectors, direction cosines, the ``ICRS -> AltAz``
transform, sidereal time), the fringe, the flux and coherency normalisation,
and -- in the polarized case -- the field rotation.

``pyuvsim``'s MPI driver (``run_uvdata_uvsim``) needs ``mpi4py``, which is not
in the ``crossval`` feature.  The comparison therefore drives ``pyuvsim``'s own
:class:`pyuvsim.UVEngine` over :class:`pyuvsim.uvsim.UVTask` directly.  That is
the class where every visibility in ``pyuvsim`` is actually computed -- the
driver only distributes tasks to it -- and it is a documented export of the
package.  No ``pyuvsim`` code is reimplemented here.

Two explicit convention mappings, each derived rather than fitted
===============================================================

1. **Fringe sign and the phase reference.**  RadioSim evaluates
   ``exp(-2j*pi*(u*l + v*m + w*(n-1)))`` (``core/jones/geometric.py``);
   ``pyuvsim`` evaluates ``exp(+2j*pi*(u*l + v*m + w*n))``
   (``pyuvsim/uvsim.py``, ``UVEngine.make_visibility``).  Both build the
   baseline vector as ``antenna2 - antenna1`` in ENU.  For an array whose
   antennas are coplanar in ``Up`` the ``w`` component is zero, so the two
   differ by conjugation of the exponent alone, and because
   ``A_s = J_1 C_s J_2^H`` is Hermitian when both antennas carry the same
   Jones matrix, summing ``A_s`` against conjugated fringes gives exactly the
   Hermitian conjugate of the other sum -- ``V_radiosim(1, 2)`` equals
   ``V_pyuvsim(1, 2)`` conjugate-transposed --
   which in correlation order is ``[XX, XY, YX, YY] -> conj([XX, YX, XY, YY])``.
   The comparison array is coplanar for that reason.

2. **The coherency's Stokes-V sign.**  RadioSim uses
   ``B = (1/2) [[I+Q, U+iV], [U-iV, I-Q]]`` (``core/polarization.py``, the
   convention Tier 5C adopted); ``pyradiosky 1.1.0`` uses the mirror image,
   ``0.5 * [[I+Q, U-iV], [U+iV, I-Q]]``
   (``pyradiosky/utils.py::stokes_to_coherency``).  This flips the sign of
   Stokes ``V`` between the two forward models.

SCI-006 removed the former third compensation: both simulators now report the
same ``(X=east, Y=north)`` frame, so ``Q`` and ``U`` are compared directly.
Only the independently documented Stokes-V sign mapping remains explicit.  The
small residual polarization-frame rotation is recorded as SCI-007; it is not
hidden by a Q-axis swap or a refitted convention.  Section 29.2's forbidden
claims stay forbidden: nothing in this module licenses the sentence "validated
against pyuvsim" without naming exactly which quantity, at which tolerance.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.crossval, pytest.mark.slow]

pyuvsim = pytest.importorskip(
    "pyuvsim",
    reason=(
        "pyuvsim is only present in the optional `crossval` pixi environment; "
        "run `pixi run --environment crossval -- python -m pytest "
        "tests/crossvalidation/ -m crossval`"
    ),
)

REFERENCE_PYUVSIM_VERSION = "1.4.0"

LATITUDE_DEG = -30.72152
LONGITUDE_DEG = 21.42830
HEIGHT_M = 1073.0
TELESCOPE_NAME = "CROSSVAL"

# Coplanar in Up, so the `w` component of every baseline is exactly zero and
# mapping 1 above reduces to a Hermitian conjugate with no residual w phase.
ANTENNA_ENU_M = ((0.0, 0.0, 0.0), (50.0, 0.0, 0.0), (0.0, 70.0, 0.0))
ANTENNA_DIAMETER_M = 10.0

FREQUENCIES_HZ = (1.20e8, 1.30e8, 1.40e8)
CHANNEL_WIDTH_HZ = 1.0e6
START_TIME_ISO = "2025-01-01T00:00:00"
CADENCE_SECONDS = 120.0
TIME_SAMPLES = 3


def _emit_metrics(case: str, metrics: dict[str, float]) -> None:
    """Print machine-readable evidence only when explicitly requested."""
    if os.environ.get("RADIOSIM_CROSSVAL_METRICS") == "1":
        print(json.dumps({"case": case, **metrics}, sort_keys=True))


def _frequencies() -> np.ndarray:
    return np.array(FREQUENCIES_HZ, dtype=np.float64)


def _earth_location():
    from astropy import units
    from astropy.coordinates import EarthLocation

    return EarthLocation.from_geodetic(
        lon=LONGITUDE_DEG * units.deg,
        lat=LATITUDE_DEG * units.deg,
        height=HEIGHT_M * units.m,
    )


def _write_unit_beamfits(path: Path) -> Path:
    """Write a BeamFITS whose E-field response is exactly 1 everywhere.

    A constant is the only beam that survives both codes' angular and spectral
    interpolation without contributing a difference of its own: bilinear and
    bicubic interpolation of a constant grid return the constant.  Four
    intrinsic frequencies are the minimum RadioSim's default cubic frequency
    interpolation accepts.
    """
    from pyuvdata import UVBeam

    azimuth = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float64)
    zenith_angle = np.linspace(0.0, np.pi / 2.0, 5, dtype=np.float64)
    frequencies = np.array([1.0e8, 1.2e8, 1.4e8, 1.6e8], dtype=np.float64)
    data = np.zeros(
        (2, 2, frequencies.size, zenith_angle.size, azimuth.size),
        dtype=np.complex128,
    )
    data[0, 0] = 1.0
    data[1, 1] = 1.0
    beam = UVBeam.new(
        telescope_name="RadioSim crossvalidation unit beam",
        data_normalization="peak",
        freq_array=frequencies,
        feed_name="crossval",
        feed_version="tier7j",
        model_name="unit response",
        model_version="tier7j",
        feed_array=np.array(["x", "y"]),
        x_orientation="east",
        mount_type="fixed",
        axis1_array=azimuth,
        axis2_array=zenith_angle,
        bandpass_array=np.ones(frequencies.size, dtype=np.float64),
        data_array=data,
        history="RadioSim Tier 7J cross-validation unit beam. ",
    )
    beam.write_beamfits(str(path), clobber=True)
    return path


def _write_sky(path: Path, iquv_by_source, ra_deg, dec_deg) -> object:
    from astropy import units
    from astropy.coordinates import SkyCoord
    from pyradiosky import SkyModel

    frequencies = _frequencies()
    stokes = np.zeros((4, frequencies.size, len(ra_deg))) * units.Jy
    for index, iquv in enumerate(iquv_by_source):
        for pol, value in enumerate(iquv):
            stokes[pol, :, index] = value * units.Jy
    sky = SkyModel(
        name=np.array([f"S{index}" for index in range(len(ra_deg))]),
        skycoord=SkyCoord(
            ra=np.asarray(ra_deg) * units.deg,
            dec=np.asarray(dec_deg) * units.deg,
            frame="icrs",
        ),
        stokes=stokes,
        spectral_type="full",
        freq_array=frequencies * units.Hz,
    )
    sky.write_skyh5(str(path), clobber=True)
    return sky


def _write_layout(path: Path) -> Path:
    """Write the array as a RadioSim ENU layout file.

    The layout format stores ENU metres verbatim, so the antennas are exactly
    coplanar in ``Up`` and every baseline's ``w`` is exactly zero -- which is
    what makes convention mapping 1 exact rather than approximate.  The UVFITS
    path below cannot offer that: it stores ECEF, and the round trip leaves
    about 1e-10 m of residual ``Up``, worth a 3e-10 rad phase at 140 MHz.
    """
    lines = ["Name    Number  BeamID  E       N       U       Diameter"]
    for index, (east, north, up) in enumerate(ANTENNA_ENU_M):
        lines.append(
            f"A{index:03d}   {index}   0   {east}   {north}   {up}   "
            f"{ANTENNA_DIAMETER_M}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_uvfits_array(path: Path, mount_type: str) -> Path:
    """Write a metadata-only UVFITS carrying the array and its mount type.

    ``mount_type`` reaches a RadioSim ``instrument:`` section only through a
    pyuvdata dataset source -- a layout file has no column for it -- so the
    ``alt-az`` case that ``jones.P`` requires is expressed as a UVFITS file.
    """
    from pyuvdata import UVData
    from pyuvdata.telescopes import Telescope
    from pyuvdata.utils import ECEF_from_ENU

    location = _earth_location()
    geocentric = np.array([value.to_value("m") for value in location.geocentric])
    positions = (
        ECEF_from_ENU(np.array(ANTENNA_ENU_M, dtype=np.float64), center_loc=location)
        - geocentric
    )
    count = len(ANTENNA_ENU_M)
    telescope = Telescope.new(
        name=TELESCOPE_NAME,
        location=location,
        antenna_names=[f"A{index:03d}" for index in range(count)],
        antenna_numbers=np.arange(count),
        antenna_positions=positions,
        instrument=TELESCOPE_NAME,
        mount_type=mount_type,
        feed_array=["x", "y"],
        feed_angle=[np.pi / 2, 0.0],
        antenna_diameters=np.full(count, ANTENNA_DIAMETER_M),
    )
    frequencies = _frequencies()
    uvdata = UVData.new(
        freq_array=frequencies,
        # UVFITS rejects channels spaced more widely than their declared width;
        # only the telescope table of this file is ever read back.
        channel_width=np.full(frequencies.size, 1.0e7),
        polarization_array=np.array([-5, -6, -7, -8]),
        telescope=telescope,
        times=np.array([2460676.5, 2460676.50138889]),
        antpairs=[(0, 1), (0, 2), (1, 2)],
        do_blt_outer=True,
        integration_time=CADENCE_SECONDS,
        empty=True,
        phase_center_catalog={
            0: {"cat_name": "unprojected", "cat_type": "unprojected"}
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uvdata.write_uvfits(str(path), force_phase=True)
    return path


def _run_radiosim(*, array: Path, beamfits: Path, skyh5: Path, jones, output: Path):
    from radiosim import Simulator

    if array.suffix == ".uvfits":
        source = {"kind": "layout_file", "path": str(array), "format": "uvfits"}
        instrument: dict = {"source": source}
    else:
        source = {
            "kind": "layout_file",
            "path": str(array),
            "format": "radiosim",
            "telescope_name": TELESCOPE_NAME,
        }
        instrument = {
            "source": source,
            "location": {
                "longitude_deg": LONGITUDE_DEG,
                "latitude_deg": LATITUDE_DEG,
                "height_m": HEIGHT_M,
            },
        }
    mapping = {
        "instrument": {
            **instrument,
            "default_diameter_m": ANTENNA_DIAMETER_M,
        },
        "beams": {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": str(beamfits)},
        },
        "baseline_selection": {"correlations": "cross"},
        "sky_model": {
            "flux_unit": "Jy",
            "sources": [{"kind": "pyradiosky_file", "filename": str(skyh5)}],
        },
        "obs_time": {
            "start_time": START_TIME_ISO,
            "duration_seconds": TIME_SAMPLES * CADENCE_SECONDS,
            "time_step_seconds": CADENCE_SECONDS,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": list(FREQUENCIES_HZ),
            "channel_widths_hz": [CHANNEL_WIDTH_HZ] * len(FREQUENCIES_HZ),
        },
        "visibility": {"sky_representation": "point_sources"},
        "execution": {"backend": "numpy", "offline": True},
        "workflow": {"save_results": False, "output_dir": str(output)},
    }
    if jones is not None:
        mapping["jones"] = jones
    simulator = Simulator.from_mapping(mapping)
    simulator.setup()
    return simulator.run(progress=False)


def _pyuvsim_cube(result, beamfits: Path, sky) -> np.ndarray:
    """Evaluate the same cube with ``pyuvsim``'s own engine.

    The antenna positions, the two-part UTC instants, the frequencies and the
    baseline ordering are taken from RadioSim's *resolved* result rather than
    from the inputs, so neither code can be compared against a differently
    rounded version of the other's geometry or time.
    """
    from astropy.time import Time
    from pyuvdata import UVBeam
    from pyuvsim import Antenna, Baseline, BeamList, SkyModelData, Telescope, UVEngine
    from pyuvsim.uvsim import UVTask

    beam_list = BeamList([UVBeam.from_file(str(beamfits))])
    telescope = Telescope(TELESCOPE_NAME, _earth_location(), beam_list)
    antennas = {
        antenna.id.number: Antenna(
            f"A{antenna.id.number:03d}",
            int(antenna.id.number),
            np.asarray(antenna.position_enu_m, dtype=np.float64),
            0,
        )
        for antenna in result.instrument.antennas
    }
    grid = result.time_grid
    jd1 = np.asarray(grid.utc_jd1)
    jd2 = np.asarray(grid.utc_jd2)
    frequencies = np.asarray(result.frequencies_hz, dtype=np.float64)
    pairs = [
        (baseline.ant1.number, baseline.ant2.number)
        for baseline in result.selection.baselines
    ]
    cube = np.zeros((jd1.size, len(pairs), frequencies.size, 4), dtype=np.complex128)
    source_indices = np.arange(sky.Ncomponents)
    for time_index in range(jd1.size):
        time = Time(jd1[time_index], jd2[time_index], format="jd", scale="utc")
        for baseline_index, (first, second) in enumerate(pairs):
            baseline = Baseline(antennas[first], antennas[second])
            for freq_index, frequency in enumerate(frequencies):
                sources = SkyModelData(sky).get_skymodel(source_indices)
                task = UVTask(
                    sources,
                    time,
                    frequency,
                    baseline,
                    telescope,
                    freq_i=freq_index,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    visibility = UVEngine(task).make_visibility()
                # pyuvsim returns [xx, yy, xy, yx]; RadioSim reports
                # [XX, XY, YX, YY].
                cube[time_index, baseline_index, freq_index] = np.asarray(
                    [
                        visibility[0],
                        visibility[2],
                        visibility[3],
                        visibility[1],
                    ]
                )
    return cube


def _apply_fringe_hermitian_mapping(cube: np.ndarray) -> np.ndarray:
    """Apply convention mapping 1: the per-baseline Hermitian conjugate."""
    return np.conj(cube[..., [0, 2, 1, 3]])


def _local_stokes(cube: np.ndarray) -> np.ndarray:
    xx, xy, yx, yy = (cube[..., index] for index in range(4))
    return np.stack([xx + yy, xx - yy, xy + yx, -1j * (xy - yx)], axis=-1)


@pytest.fixture(scope="module")
def unit_beamfits(tmp_path_factory) -> Path:
    return _write_unit_beamfits(tmp_path_factory.mktemp("beam") / "unit.beamfits")


def test_installed_pyuvsim_is_the_recorded_reference_version():
    """The artifact records ``pyuvsim 1.4.0``, never "latest" (Section 41 Q1)."""
    assert pyuvsim.__version__ == REFERENCE_PYUVSIM_VERSION


def test_pyuvsim_east_x_unit_beam_matches_the_sci006_closed_form(unit_beamfits):
    """Independently pin pyuvsim's feed-by-sky binding for east-oriented X.

    This is the WP-4 analytic example, evaluated through pyuvsim's own
    ``Antenna.get_beam_jones`` and pyradiosky's own coherency constructor.  It
    does not compare against RadioSim and therefore cannot inherit RadioSim's
    axis-order assumption.
    """
    from astropy import units
    from pyradiosky.utils import stokes_to_coherency
    from pyuvdata import UVBeam
    from pyuvsim import Antenna, BeamList, Telescope

    telescope = Telescope(
        "SCI-006 analytic probe",
        _earth_location(),
        BeamList([UVBeam.from_file(str(unit_beamfits))]),
    )
    antenna = Antenna("A000", 0, np.zeros(3), 0)
    jones = antenna.get_beam_jones(
        telescope,
        # Both angles land on the unit beam's regular grid, minimizing the
        # interpolation round-off that is tolerated explicitly below.
        np.array([[np.pi / 4.0], [0.0]]),
        1.2e8,
        reuse_spline=False,
    )[..., 0]

    stokes = np.array([1.0, 0.6, 0.0, 0.0]) * units.Jy
    brightness = np.asarray(stokes_to_coherency(stokes).value)
    visibility = jones @ brightness @ jones.conj().T

    np.testing.assert_allclose(
        jones,
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        rtol=0.0,
        # UVBeam's response path leaves one double-precision interpolation ulp
        # even when frequency and direction land on the regular grid.
        atol=5e-16,
    )
    np.testing.assert_allclose(
        visibility,
        np.array([[0.2, 0.0], [0.0, 0.8]], dtype=np.complex128),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        visibility[0, 0] - visibility[1, 1],
        -0.6 + 0.0j,
        rtol=0.0,
        atol=1e-15,
    )


def test_unpolarized_point_sources_match_pyuvsim(tmp_path, unit_beamfits):
    """The geometric half of the RIME agrees to double-precision round-off.

    Four unpolarized sources, three coplanar antennas, three times and three
    frequencies, a unit beam, no ``jones:`` section and a ``fixed`` mount, so
    the only physics in play is ``K`` (fringe), the coordinate chain, and the
    flux-to-coherency normalisation.  Under convention mapping 1 the two cubes
    must agree to round-off; the committed artifact records the measured value.
    """
    skyh5 = tmp_path / "unpolarized.skyh5"
    sky = _write_sky(
        skyh5,
        iquv_by_source=(
            (3.0, 0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0, 0.0),
            (2.25, 0.0, 0.0, 0.0),
            (0.75, 0.0, 0.0, 0.0),
        ),
        ra_deg=(20.0, 25.0, 15.0, 22.0),
        dec_deg=(-30.72, -26.0, -35.0, -31.5),
    )
    layout = _write_layout(tmp_path / "array.txt")
    result = _run_radiosim(
        array=layout,
        beamfits=unit_beamfits,
        skyh5=skyh5,
        jones=None,
        output=tmp_path / "out",
    )
    ours = np.asarray(result.visibilities)
    theirs = _apply_fringe_hermitian_mapping(_pyuvsim_cube(result, unit_beamfits, sky))

    assert ours.shape == theirs.shape == (3, 3, 3, 4)
    scale = float(np.max(np.abs(theirs)))
    assert scale > 1.0
    relative = float(np.max(np.abs(ours - theirs))) / scale
    assert relative < 1e-11, relative

    # The comparison is not vacuous: undo convention mapping 1 and the two
    # cubes disagree at order unity, so the assertion above is testing the
    # fringe and not an accidental agreement of two near-zero arrays.
    without_mapping = _apply_fringe_hermitian_mapping(theirs)
    control_relative = float(np.max(np.abs(ours - without_mapping))) / scale
    assert control_relative > 0.1

    # The unpolarized invariant both codes must satisfy independently.
    stokes = _local_stokes(ours)
    assert float(np.max(np.abs(stokes[..., 1]))) / scale < 1e-11
    assert float(np.max(np.abs(stokes[..., 2]))) / scale < 1e-11
    assert float(np.max(np.abs(stokes[..., 3]))) / scale < 1e-11
    _emit_metrics(
        "unpolarized",
        {
            "cube_scale_jy": scale,
            "measured_absolute_jy": float(np.max(np.abs(ours - theirs))),
            "measured_relative": relative,
            "control_relative_without_fringe_mapping": control_relative,
        },
    )


def test_polarized_sources_with_jones_p_match_pyuvsim_in_common_east_x_frame(
    tmp_path, unit_beamfits
):
    """``P`` reproduces ``pyuvsim`` in the common east-X frame.

    A full-Stokes sky on an ``alt-az`` array with ``jones.P`` enabled.  Total
    intensity and ``Q + iU`` are compared directly after the named fringe
    Hermitian mapping.  Stokes V uses the one explicit sign conversion derived
    from the two coherency definitions.
    """
    skyh5 = tmp_path / "polarized.skyh5"
    sky = _write_sky(
        skyh5,
        iquv_by_source=(
            (3.0, 0.6, -0.4, 0.2),
            (1.5, -0.3, 0.5, -0.1),
            (2.25, 0.0, 0.0, 0.9),
        ),
        ra_deg=(20.0, 25.0, 15.0),
        dec_deg=(-30.72, -26.0, -35.0),
    )
    uvfits = _write_uvfits_array(tmp_path / "altaz.uvfits", "alt-az")
    result = _run_radiosim(
        array=uvfits,
        beamfits=unit_beamfits,
        skyh5=skyh5,
        jones={"P": {"enabled": True}},
        output=tmp_path / "out",
    )
    ours = np.asarray(result.visibilities)
    theirs = _apply_fringe_hermitian_mapping(_pyuvsim_cube(result, unit_beamfits, sky))

    ours_stokes = _local_stokes(ours)
    theirs_stokes = _local_stokes(theirs)
    scale = float(np.max(np.abs(theirs_stokes[..., 0])))

    # Total intensity is direct.  Circular polarization uses mapping 2
    # explicitly: RadioSim and pyradiosky define the sign of V oppositely.
    intensity = float(np.max(np.abs(ours_stokes[..., 0] - theirs_stokes[..., 0])))
    assert intensity / scale < 1e-8, intensity / scale
    circular = float(np.max(np.abs(ours_stokes[..., 3] + theirs_stokes[..., 3])))
    assert circular / scale < 1e-8, circular / scale

    # SCI-006: direct Q/U in the shared east-X frame, with no Q compensation.
    ours_linear = ours_stokes[..., 1] + 1j * ours_stokes[..., 2]
    theirs_linear = theirs_stokes[..., 1] + 1j * theirs_stokes[..., 2]
    linear_scale = float(np.max(np.abs(theirs_linear)))
    assert linear_scale / scale > 0.1
    residual = float(np.max(np.abs(ours_linear - theirs_linear))) / linear_scale
    assert residual < 5e-3, residual
    # The retired pre-SCI-006 Q compensation is now a loud control failure.
    old_q_compensation = -theirs_stokes[..., 1] + 1j * theirs_stokes[..., 2]
    old_q_control = (
        float(np.max(np.abs(ours_linear - old_q_compensation))) / linear_scale
    )
    assert old_q_control > 0.5

    valid = np.abs(theirs_linear) > linear_scale * 1e-12
    reference = theirs_linear[valid]
    measured = ours_linear[valid]
    fitted_ratio = np.vdot(reference, measured) / np.vdot(reference, reference)
    fitted_rotation_deg = 0.5 * float(np.degrees(np.angle(fitted_ratio)))
    _emit_metrics(
        "polarized",
        {
            "measured_intensity_relative": intensity / scale,
            "measured_circular_relative_after_explicit_v_mapping": circular / scale,
            "measured_linear_relative_direct_q_u": residual,
            "control_relative_with_retired_q_compensation": old_q_control,
            "fitted_residual_frame_rotation_deg": fitted_rotation_deg,
            "fitted_linear_ratio_modulus": float(abs(fitted_ratio)),
            "linear_scale_over_intensity_scale": linear_scale / scale,
        },
    )


def test_the_committed_artifact_describes_this_comparison():
    """Section 29 requires the Tier-2 comparison to exist as an artifact."""
    import json

    root = Path(__file__).resolve().parents[2]
    artifact = root / "output" / "crossvalidation" / "2026-08-08-pyuvsim-1.4.0.json"
    assert artifact.is_file(), artifact
    record = json.loads(artifact.read_text(encoding="utf-8"))
    assert record["reference"]["package"] == "pyuvsim"
    assert record["reference"]["version"] == REFERENCE_PYUVSIM_VERSION
    assert record["gating"] is False
    names = {case["test"] for case in record["cases"]}
    assert "test_unpolarized_point_sources_match_pyuvsim" in names
    assert (
        "test_polarized_sources_with_jones_p_match_pyuvsim_in_common_east_x_frame"
        in names
    )
