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
same ``(X=east, Y=north)`` frame, so raw ``Q`` and ``U`` are compared directly.
Only the independently documented Stokes-V sign mapping remains explicit.
SCI-007 additionally decomposes the retained fixture by source and time and
reconciles the raw milli-radian frame residual with the exact pinned
``pyradiosky`` tangent-basis rotation, without changing RadioSim production
code.  Section 29.2's forbidden claims stay forbidden: nothing in this module
licenses the sentence "validated against pyuvsim" without naming exactly which
quantity, at which tolerance.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
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
REFERENCE_PYRADIOSKY_VERSION = "1.1.0"
REFERENCE_ASTROPY_VERSION = "7.1.0"

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

POLARIZED_IQUV_BY_SOURCE = (
    (3.0, 0.6, -0.4, 0.2),
    (1.5, -0.3, 0.5, -0.1),
    (2.25, 0.0, 0.0, 0.9),
)
POLARIZED_RA_DEG = (20.0, 25.0, 15.0)
POLARIZED_DEC_DEG = (-30.72, -26.0, -35.0)


@dataclass(frozen=True)
class Sci007Comparison:
    """Structured optional evidence shared by pytest and the artifact generator."""

    result: object
    sky: object
    iers_table: object
    ours: np.ndarray
    ours_by_source: np.ndarray
    theirs: np.ndarray
    theirs_by_source: np.ndarray
    exact_rotations: np.ndarray
    public_angles: np.ndarray
    exact_angles: np.ndarray
    ours_linear: np.ndarray
    theirs_linear: np.ndarray
    theirs_source_linear: np.ndarray
    exact_corrected_linear: np.ndarray
    valid_linear: np.ndarray
    metrics: dict[str, float]


@dataclass(frozen=True)
class UnpolarizedComparison:
    """Structured four-source fixed-mount control, separate from SCI-007."""

    result: object
    sky: object
    iers_table: object
    ours: np.ndarray
    theirs: np.ndarray
    metrics: dict[str, float]


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


def _pyuvsim_source_cubes_and_rotations(
    result, beamfits: Path, sky
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-resolved cubes and pyradiosky's exact frame rotations.

    The private call is deliberately confined to this pinned optional
    environment.  It follows the path used by ``pyuvsim`` itself: after
    ``update_positions``, ``UVEngine.apply_beam`` calls ``coherency_calc``,
    whose private rotation chain is ``_calc_coherency_rotation`` ->
    ``_calc_rotation_matrix`` -> ``_calc_average_rotation_matrix`` before
    ``UVEngine.make_visibility`` sums the source contributions.  The returned
    rotations have axes ``[time, source, receptor, tangent_basis]`` and the
    visibility cubes have axes ``[time, source, baseline, frequency, pol]``.
    """
    import pyradiosky
    from astropy.time import Time
    from pyuvdata import UVBeam
    from pyuvsim import Antenna, Baseline, BeamList, SkyModelData, Telescope, UVEngine
    from pyuvsim.uvsim import UVTask

    assert pyradiosky.__version__ == REFERENCE_PYRADIOSKY_VERSION
    assert callable(getattr(type(sky), "_calc_average_rotation_matrix", None))
    assert callable(getattr(type(sky), "_calc_rotation_matrix", None))
    assert callable(getattr(type(sky), "_calc_coherency_rotation", None))

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
    source_count = int(sky.Ncomponents)
    cubes = np.zeros(
        (jd1.size, source_count, len(pairs), frequencies.size, 4),
        dtype=np.complex128,
    )
    rotations = np.zeros((jd1.size, source_count, 2, 2), dtype=np.float64)
    sky_data = SkyModelData(sky)

    for time_index in range(jd1.size):
        time = Time(jd1[time_index], jd2[time_index], format="jd", scale="utc")
        all_sources = sky_data.get_skymodel(np.arange(source_count))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pyradiosky 1.1.0 exposes no pressure argument here.  Its AltAz
            # construction therefore uses Astropy's effective 0 hPa default;
            # the separate public oracle below spells pressure=0 out.
            all_sources.update_positions(time, telescope.location)
        assert np.all(all_sources.above_horizon)
        exact_rotation = all_sources._calc_coherency_rotation()
        assert exact_rotation.shape == (2, 2, source_count)
        rotations[time_index] = np.moveaxis(exact_rotation, -1, 0)

        source_models = [
            sky_data.get_skymodel(np.array([source_index]))
            for source_index in range(source_count)
        ]
        for source_index, sources in enumerate(source_models):
            for baseline_index, (first, second) in enumerate(pairs):
                baseline = Baseline(antennas[first], antennas[second])
                for freq_index, frequency in enumerate(frequencies):
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
                    cubes[time_index, source_index, baseline_index, freq_index] = (
                        np.asarray(
                            [
                                visibility[0],
                                visibility[2],
                                visibility[3],
                                visibility[1],
                            ]
                        )
                    )
    return cubes, rotations


def _radiosim_source_cubes(
    *, tmp_path: Path, uvfits: Path, beamfits: Path
) -> np.ndarray:
    """Run the retained fixture one source at a time without changing its RIME."""
    cubes = []
    for source_index, iquv in enumerate(POLARIZED_IQUV_BY_SOURCE):
        skyh5 = tmp_path / f"polarized-source-{source_index}.skyh5"
        _write_sky(
            skyh5,
            iquv_by_source=(iquv,),
            ra_deg=(POLARIZED_RA_DEG[source_index],),
            dec_deg=(POLARIZED_DEC_DEG[source_index],),
        )
        source_result = _run_radiosim(
            array=uvfits,
            beamfits=beamfits,
            skyh5=skyh5,
            jones={"P": {"enabled": True}},
            output=tmp_path / f"source-{source_index}-out",
        )
        cubes.append(np.asarray(source_result.visibilities))
    return np.stack(cubes, axis=1)


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def _sci007_frame_angles(
    result, sky, exact_rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derive public and exact RadioSim-minus-pyradiosky frame angles."""
    from astropy import units
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time

    from radiosim.core.jones.directions import DirectionBatch
    from radiosim.core.jones.parallactic import parallactic_angle

    location = _earth_location()
    grid = result.time_grid
    jd1 = np.asarray(grid.utc_jd1)
    jd2 = np.asarray(grid.utc_jd2)
    public = np.zeros((jd1.size, sky.Ncomponents), dtype=np.float64)
    exact = np.zeros_like(public)

    for time_index in range(jd1.size):
        time = Time(jd1[time_index], jd2[time_index], format="jd", scale="utc")
        altaz_frame = AltAz(
            obstime=time,
            location=location,
            pressure=0.0 * units.hPa,
        )
        horizontal = sky.skycoord.transform_to(altaz_frame)
        altitude = np.asarray(horizontal.alt.rad, dtype=np.float64)
        azimuth = np.asarray(horizontal.az.rad, dtype=np.float64)
        directions = DirectionBatch.from_horizontal(
            alt_rad=altitude,
            az_rad=azimuth,
            dir_l=np.cos(altitude) * np.sin(azimuth),
            dir_m=np.cos(altitude) * np.cos(azimuth),
            dir_n=np.sin(altitude),
            latitude_rad=location.lat.rad,
            local_sidereal_time_rad=time.sidereal_time(
                "apparent", longitude=location.lon
            ).rad,
        )
        psi_rs = parallactic_angle(
            hour_angle_rad=directions.hour_angle_rad,
            dec_rad=directions.dec_rad,
            latitude_rad=location.lat.rad,
        )

        zenith_icrs = SkyCoord(
            az=0.0 * units.deg,
            alt=90.0 * units.deg,
            frame=altaz_frame,
        ).transform_to("icrs")
        public_basis_angle = sky.skycoord.position_angle(zenith_icrs).rad
        public[time_index] = _wrap_pi(psi_rs - public_basis_angle)

        # K.T = R(alpha_PY), so atan2(K[0, 1], K[0, 0]) = -alpha_PY.
        rotation = exact_rotations[time_index]
        exact[time_index] = _wrap_pi(
            psi_rs + np.arctan2(rotation[:, 0, 1], rotation[:, 0, 0])
        )

    return public, exact


def _apply_fringe_hermitian_mapping(cube: np.ndarray) -> np.ndarray:
    """Apply convention mapping 1: the per-baseline Hermitian conjugate."""
    return np.conj(cube[..., [0, 2, 1, 3]])


def _local_stokes(cube: np.ndarray) -> np.ndarray:
    xx, xy, yx, yy = (cube[..., index] for index in range(4))
    return np.stack([xx + yy, xx - yy, xy + yx, -1j * (xy - yx)], axis=-1)


def _run_sci007_comparison(tmp_path: Path, unit_beamfits: Path) -> Sci007Comparison:
    """Run the hermetic SCI-007 fixture once and return all evidence axes."""
    import astropy
    import pyradiosky
    from astropy.utils import iers

    assert astropy.__version__ == REFERENCE_ASTROPY_VERSION
    assert pyradiosky.__version__ == REFERENCE_PYRADIOSKY_VERSION
    assert pyuvsim.__version__ == REFERENCE_PYUVSIM_VERSION

    iers_table = iers.IERS_A.open(iers.IERS_A_FILE)
    assert type(iers_table).__name__ == "IERS_A"
    with (
        iers.conf.set_temp("auto_download", False),
        iers.earth_orientation_table.set(iers_table),
    ):
        skyh5 = tmp_path / "polarized.skyh5"
        sky = _write_sky(
            skyh5,
            iquv_by_source=POLARIZED_IQUV_BY_SOURCE,
            ra_deg=POLARIZED_RA_DEG,
            dec_deg=POLARIZED_DEC_DEG,
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
        ours_by_source = _radiosim_source_cubes(
            tmp_path=tmp_path,
            uvfits=uvfits,
            beamfits=unit_beamfits,
        )
        theirs_by_source, exact_rotations = _pyuvsim_source_cubes_and_rotations(
            result, unit_beamfits, sky
        )
        theirs_by_source = _apply_fringe_hermitian_mapping(theirs_by_source)
        theirs = _apply_fringe_hermitian_mapping(
            _pyuvsim_cube(result, unit_beamfits, sky)
        )
        public_angles, exact_angles = _sci007_frame_angles(result, sky, exact_rotations)

    ours_stokes = _local_stokes(ours)
    theirs_stokes = _local_stokes(theirs)
    intensity_scale = float(np.max(np.abs(theirs_stokes[..., 0])))
    intensity = float(np.max(np.abs(ours_stokes[..., 0] - theirs_stokes[..., 0])))
    circular = float(np.max(np.abs(ours_stokes[..., 3] + theirs_stokes[..., 3])))

    ours_linear = ours_stokes[..., 1] + 1j * ours_stokes[..., 2]
    theirs_linear = theirs_stokes[..., 1] + 1j * theirs_stokes[..., 2]
    linear_scale = float(np.max(np.abs(theirs_linear)))
    valid = np.abs(theirs_linear) > linear_scale * 1e-12
    raw_residual = (
        float(np.max(np.abs(ours_linear[valid] - theirs_linear[valid]))) / linear_scale
    )

    reference = theirs_linear[valid]
    measured = ours_linear[valid]
    fitted_ratio = np.vdot(reference, measured) / np.vdot(reference, reference)
    global_corrected = ours_linear * np.exp(-1j * np.angle(fitted_ratio))
    global_residual = (
        float(np.max(np.abs(global_corrected[valid] - theirs_linear[valid])))
        / linear_scale
    )

    ours_source_stokes = _local_stokes(ours_by_source)
    theirs_source_stokes = _local_stokes(theirs_by_source)
    ours_source_linear = ours_source_stokes[..., 1] + 1j * ours_source_stokes[..., 2]
    theirs_source_linear = (
        theirs_source_stokes[..., 1] + 1j * theirs_source_stokes[..., 2]
    )
    exact_corrected = np.sum(
        ours_source_linear * np.exp(-2j * exact_angles[:, :, None, None]),
        axis=1,
    )
    exact_residual = (
        float(np.max(np.abs(exact_corrected[valid] - theirs_linear[valid])))
        / linear_scale
    )
    wrong_sign = np.sum(
        ours_source_linear * np.exp(2j * exact_angles[:, :, None, None]),
        axis=1,
    )
    wrong_sign_residual = (
        float(np.max(np.abs(wrong_sign[valid] - theirs_linear[valid]))) / linear_scale
    )
    old_q_compensation = -theirs_stokes[..., 1] + 1j * theirs_stokes[..., 2]
    old_q_control = (
        float(np.max(np.abs(ours_linear - old_q_compensation))) / linear_scale
    )
    public_exact_relative = float(
        np.max(np.abs(public_angles - exact_angles) / np.abs(exact_angles))
    )
    metrics = {
        "intensity_scale_jy": intensity_scale,
        "linear_scale_jy": linear_scale,
        "measured_intensity_absolute_jy": intensity,
        "measured_intensity_relative": intensity / intensity_scale,
        "measured_circular_absolute_jy": circular,
        "measured_circular_relative_after_explicit_v_mapping": circular
        / intensity_scale,
        "measured_linear_relative_direct_q_u": raw_residual,
        "measured_linear_relative_single_global_angle": global_residual,
        "measured_linear_relative_exact_source_time": exact_residual,
        "control_relative_with_wrong_exact_sign": wrong_sign_residual,
        "control_relative_with_retired_q_compensation": old_q_control,
        "fitted_residual_frame_rotation_deg": 0.5
        * float(np.degrees(np.angle(fitted_ratio))),
        "fitted_linear_ratio_modulus": float(abs(fitted_ratio)),
        "linear_scale_over_intensity_scale": linear_scale / intensity_scale,
        "public_exact_angle_max_relative": public_exact_relative,
    }
    return Sci007Comparison(
        result=result,
        sky=sky,
        iers_table=iers_table,
        ours=ours,
        ours_by_source=ours_by_source,
        theirs=theirs,
        theirs_by_source=theirs_by_source,
        exact_rotations=exact_rotations,
        public_angles=public_angles,
        exact_angles=exact_angles,
        ours_linear=ours_linear,
        theirs_linear=theirs_linear,
        theirs_source_linear=theirs_source_linear,
        exact_corrected_linear=exact_corrected,
        valid_linear=valid,
        metrics=metrics,
    )


def _run_unpolarized_comparison(
    tmp_path: Path, unit_beamfits: Path
) -> UnpolarizedComparison:
    """Run the separate four-source fixed-mount control hermetically."""
    import astropy
    import pyradiosky
    from astropy.utils import iers

    assert astropy.__version__ == REFERENCE_ASTROPY_VERSION
    assert pyradiosky.__version__ == REFERENCE_PYRADIOSKY_VERSION
    assert pyuvsim.__version__ == REFERENCE_PYUVSIM_VERSION

    iers_table = iers.IERS_A.open(iers.IERS_A_FILE)
    assert type(iers_table).__name__ == "IERS_A"
    with (
        iers.conf.set_temp("auto_download", False),
        iers.earth_orientation_table.set(iers_table),
    ):
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
        theirs = _apply_fringe_hermitian_mapping(
            _pyuvsim_cube(result, unit_beamfits, sky)
        )

    scale = float(np.max(np.abs(theirs)))
    absolute = float(np.max(np.abs(ours - theirs)))
    without_mapping = _apply_fringe_hermitian_mapping(theirs)
    stokes = _local_stokes(ours)
    metrics = {
        "cube_scale_jy": scale,
        "measured_absolute_jy": absolute,
        "measured_relative": absolute / scale,
        "control_relative_without_fringe_mapping": float(
            np.max(np.abs(ours - without_mapping))
        )
        / scale,
        "stokes_q_relative_max": float(np.max(np.abs(stokes[..., 1]))) / scale,
        "stokes_u_relative_max": float(np.max(np.abs(stokes[..., 2]))) / scale,
        "stokes_v_relative_max": float(np.max(np.abs(stokes[..., 3]))) / scale,
    }
    return UnpolarizedComparison(
        result=result,
        sky=sky,
        iers_table=iers_table,
        ours=ours,
        theirs=theirs,
        metrics=metrics,
    )


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
    frequencies, a unit beam, no optional ``jones:`` terms and a ``fixed``
    mount.  The current chain still applies unit ``E``, east-X ``C=P``, and
    matching-output ``H=I2`` alongside ``K`` (fringe), the coordinate chain,
    and flux-to-coherency normalization.  For this unpolarized comparison the
    unitary receptor factor cancels between the two sides of the coherency.
    Under convention mapping 1 the cubes must agree to round-off; the committed
    artifact records the measured value.
    """
    comparison = _run_unpolarized_comparison(tmp_path, unit_beamfits)
    metrics = comparison.metrics

    assert comparison.ours.shape == comparison.theirs.shape == (3, 3, 3, 4)
    assert metrics["cube_scale_jy"] > 1.0
    assert metrics["measured_relative"] < 1e-11, metrics["measured_relative"]

    # The comparison is not vacuous: undo convention mapping 1 and the two
    # cubes disagree at order unity, so the assertion above is testing the
    # fringe and not an accidental agreement of two near-zero arrays.
    assert metrics["control_relative_without_fringe_mapping"] > 0.1

    # The unpolarized invariant both codes must satisfy independently.
    assert metrics["stokes_q_relative_max"] < 1e-11
    assert metrics["stokes_u_relative_max"] < 1e-11
    assert metrics["stokes_v_relative_max"] < 1e-11
    _emit_metrics("unpolarized", metrics)


def test_polarized_sources_with_jones_p_match_pyuvsim_in_common_east_x_frame(
    tmp_path, unit_beamfits
):
    """Bound and explain ``P``'s residual in the common east-X frame.

    A full-Stokes sky on an ``alt-az`` array with ``jones.P`` enabled.  Total
    intensity and raw ``Q + iU`` are compared after the named fringe Hermitian
    mapping.  Stokes V uses the one explicit sign conversion derived from the
    two coherency definitions.  The SCI-007 correction moves each RadioSim
    source-time contribution through ``exp(-2j * Delta)`` before summation.
    """
    comparison = _run_sci007_comparison(tmp_path, unit_beamfits)
    metrics = comparison.metrics

    assert (
        comparison.ours_by_source.shape
        == comparison.theirs_by_source.shape
        == (
            3,
            3,
            3,
            3,
            4,
        )
    )
    np.testing.assert_allclose(
        np.sum(comparison.ours_by_source, axis=1),
        comparison.ours,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        np.sum(comparison.theirs_by_source, axis=1),
        comparison.theirs,
        rtol=0.0,
        atol=2e-15,
    )

    expected_exact_degrees = np.array(
        [
            [0.054345212925136, 0.064500055921101, 0.042970437265188],
            [0.054345990627089, 0.064500756258174, 0.042971289516061],
            [0.054346765297796, 0.064501454583101, 0.042972137541478],
        ]
    )
    np.testing.assert_allclose(
        np.degrees(comparison.exact_angles),
        expected_exact_degrees,
        rtol=0.0,
        atol=5e-13,
    )
    assert metrics["public_exact_angle_max_relative"] < 0.10

    # Total intensity is direct.  Circular polarization uses mapping 2
    # explicitly: RadioSim and pyradiosky define the sign of V oppositely.
    assert metrics["measured_intensity_relative"] < 1e-8
    assert metrics["measured_circular_relative_after_explicit_v_mapping"] < 1e-8

    # SCI-006: direct Q/U in the shared east-X frame, with no Q compensation.
    assert metrics["linear_scale_over_intensity_scale"] > 0.1
    assert np.any(comparison.valid_linear)
    raw_residual = metrics["measured_linear_relative_direct_q_u"]
    assert 1e-3 < raw_residual < 5e-3, raw_residual
    global_residual = metrics["measured_linear_relative_single_global_angle"]
    assert 1e-4 < global_residual < raw_residual, global_residual
    exact_residual = metrics["measured_linear_relative_exact_source_time"]
    assert exact_residual < 5e-10, exact_residual

    # Source decomposition adds an evidence axis, not a different reference.
    np.testing.assert_allclose(
        np.sum(comparison.theirs_source_linear, axis=1),
        comparison.theirs_linear,
        rtol=0.0,
        atol=2e-15,
    )
    wrong_sign_residual = metrics["control_relative_with_wrong_exact_sign"]
    assert wrong_sign_residual > raw_residual, wrong_sign_residual

    # The retired pre-SCI-006 Q compensation is now a loud control failure.
    old_q_control = metrics["control_relative_with_retired_q_compensation"]
    assert old_q_control > 0.5
    _emit_metrics("polarized", metrics)


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
