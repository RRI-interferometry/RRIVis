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

The measured result of a real run on the authoring host is committed as
``output/crossvalidation/2026-08-02-pyuvsim-1.4.0.json``, which is the evidence
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

Three convention mappings, each derived rather than fitted
==========================================================

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

3. **The local polarization basis axis order.**  RadioSim's receptor frame
   places feed 0 along the celestial-north direction rotated by the parallactic
   angle; the accepted scalar BeamFITS subset binds feed 0 to
   ``data_array[0, 0]``, which is ``pyuvdata``'s first sky-vector component.
   The two conventions turn out to be each other's axis swap, which flips
   Stokes ``Q`` and Stokes ``V``.

Mappings 2 and 3 both flip ``V``, so ``V`` agrees between the two codes while
``Q`` does not.  That is a **characterization, not an endorsement**: this module
does not claim RadioSim's local ``Q`` sign is right, and the residual
polarization-frame rotation of about 0.05 degrees left after the swap is not
fully explained here either.  Both are recorded in the artifact and routed to
the Tier 7 whole-tier reviewer.  Section 29.2's forbidden claims stay
forbidden: nothing in this module licenses the sentence "validated against
pyuvsim" without naming exactly which quantity, at which tolerance.
"""

from __future__ import annotations

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


def _hermitian(cube: np.ndarray) -> np.ndarray:
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
    theirs = _hermitian(_pyuvsim_cube(result, unit_beamfits, sky))

    assert ours.shape == theirs.shape == (3, 3, 3, 4)
    scale = float(np.max(np.abs(theirs)))
    assert scale > 1.0
    relative = float(np.max(np.abs(ours - theirs))) / scale
    assert relative < 1e-11, relative

    # The comparison is not vacuous: undo convention mapping 1 and the two
    # cubes disagree at order unity, so the assertion above is testing the
    # fringe and not an accidental agreement of two near-zero arrays.
    without_mapping = _hermitian(theirs)
    assert float(np.max(np.abs(ours - without_mapping))) / scale > 0.1

    # The unpolarized invariant both codes must satisfy independently.
    stokes = _local_stokes(ours)
    assert float(np.max(np.abs(stokes[..., 1]))) / scale < 1e-11
    assert float(np.max(np.abs(stokes[..., 2]))) / scale < 1e-11
    assert float(np.max(np.abs(stokes[..., 3]))) / scale < 1e-11


def test_polarized_sources_with_jones_p_match_pyuvsim_up_to_the_basis_swap(
    tmp_path, unit_beamfits
):
    """``P`` reproduces ``pyuvsim``'s field rotation, modulo mappings 2 and 3.

    A full-Stokes sky on an ``alt-az`` array with ``jones.P`` enabled.  Total
    intensity and the polarized intensity ``|Q + iU|`` are frame-independent
    and must agree closely; the sign of local ``Q`` does not, for the reasons
    the module docstring derives.  This test pins the measured relation rather
    than asserting that RadioSim's local ``Q`` sign is the correct one.
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
    theirs = _hermitian(_pyuvsim_cube(result, unit_beamfits, sky))

    ours_stokes = _local_stokes(ours)
    theirs_stokes = _local_stokes(theirs)
    scale = float(np.max(np.abs(theirs_stokes[..., 0])))

    # Frame-independent quantities: total intensity and circular polarization
    # (the latter agrees because mappings 2 and 3 flip its sign twice).
    intensity = float(np.max(np.abs(ours_stokes[..., 0] - theirs_stokes[..., 0])))
    assert intensity / scale < 1e-8, intensity / scale
    circular = float(np.max(np.abs(ours_stokes[..., 3] - theirs_stokes[..., 3])))
    assert circular / scale < 1e-8, circular / scale

    # Linear polarization: the magnitude survives the basis difference, the
    # sign of Q does not.
    ours_linear = ours_stokes[..., 1] + 1j * ours_stokes[..., 2]
    theirs_linear = -theirs_stokes[..., 1] + 1j * theirs_stokes[..., 2]
    linear_scale = float(np.max(np.abs(theirs_linear)))
    assert linear_scale / scale > 0.1
    residual = float(np.max(np.abs(ours_linear - theirs_linear))) / linear_scale
    assert residual < 5e-3, residual
    # And it really is a swap, not an agreement: the unswapped comparison is
    # two orders of magnitude worse.
    unswapped = theirs_stokes[..., 1] + 1j * theirs_stokes[..., 2]
    assert float(np.max(np.abs(ours_linear - unswapped))) / linear_scale > 0.5


def test_the_committed_artifact_describes_this_comparison():
    """Section 29 requires the Tier-2 comparison to exist as an artifact."""
    import json

    root = Path(__file__).resolve().parents[2]
    artifact = root / "output" / "crossvalidation" / "2026-08-02-pyuvsim-1.4.0.json"
    assert artifact.is_file(), artifact
    record = json.loads(artifact.read_text(encoding="utf-8"))
    assert record["reference"]["package"] == "pyuvsim"
    assert record["reference"]["version"] == REFERENCE_PYUVSIM_VERSION
    assert record["gating"] is False
    names = {case["test"] for case in record["cases"]}
    assert "test_unpolarized_point_sources_match_pyuvsim" in names
    assert (
        "test_polarized_sources_with_jones_p_match_pyuvsim_up_to_the_basis_swap"
        in names
    )
