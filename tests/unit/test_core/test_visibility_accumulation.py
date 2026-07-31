"""The Tier 6D accumulation shape: block assembly, not per-cell ``set_at``.

``Tier6HybridRuntimePlan.md`` Section 13.3 replaces the ``O(T*B*F)`` per-cell
``set_at`` accumulation (defect D11) with one block assembly per stage::

    for each time index t:
        host preprocessing (astropy, horizon mask, direction cosines)  # named
        for each frequency index f:
            assemble one (B, 2, 2) block for all baselines at (t, f)
        assemble one (B, F, 2, 2) block for time t
    collect T time blocks -> one (T, B, F, 2, 2) cube in one operation

This module holds test R2 of Section 27 ("the number of whole-cube assembly
operations per solver call is 1, asserted through a counting backend wrapper")
for both solvers, plus the exact call-count shape of the intermediate stages, so
a regression to per-cell writes cannot pass silently.  Bit-identity of the
numbers themselves (test R1 / invariant S8) is held by the Tier 6A fingerprints
in ``tests/characterization/test_tier6_current_behavior.py``, which this slice
leaves untouched.

Section 13.3 axis-order note
============================

Section 13.3's sketch names the per-time block ``(F, B, 2, 2)`` while requiring
that ``T`` such blocks collect into a ``(T, B, F, 2, 2)`` cube "in one
operation".  Those two statements cannot both hold: no single ``stack`` of
``(F, B, 2, 2)`` blocks produces ``(T, B, F, 2, 2)`` -- it yields
``(T, F, B, 2, 2)``, which needs a further transpose and leaves the returned
cube non-contiguous.  The implementation therefore assembles the per-time block
as ``(B, F, 2, 2)`` (``stack`` of the ``F`` baseline blocks on ``axis=1``), which
makes the final ``stack`` on ``axis=0`` produce the canonical
``(T, B, F, 2, 2)`` cube exactly, contiguously, and in one operation.  The
binding properties -- one block per ``(t, f)``, one block per ``t``, one
whole-cube assembly, and the canonical cube shape -- are all preserved; only the
intermediate axis order differs from the sketch.  See the matching correction
recorded in Section 13.3.

Q3 evidence -- peak host memory across the restructure
======================================================

Section 41's Q3 asks whether holding the per-time blocks before the single
assembly changes peak memory materially, and requires a ``tracemalloc`` peak for
the largest shipped configuration, before and after, in 6D's record.

Measured 2026-07-30 on macOS 26.5.2, Apple M1 Max, 64 GB RAM, ``pixi``
``default``/py311 (Python 3.11.13, numpy 2.3.2, astropy 7.1.0), by running
``configs/config.yaml`` end to end under ``tracemalloc`` -- the largest shipped
configuration, a ``(60, 15, 101, 2, 2)`` receptor cube, ``complex128``,
5.548 MiB -- three times in each of two separate processes: one importing a
detached worktree of the pre-restructure commit ``c5d79aa``, one importing this
slice's tree, both reading the same ``configs/`` directory.  Run-to-run spread
was below 0.005 MiB, and both trees produced the identical
``scientific_sha256`` ``302deb27...``, which is 6A's py311 pin.

=========================================  ==========  ==========  ==========
Stage                                      before      after       delta
=========================================  ==========  ==========  ==========
whole run, tracemalloc peak (MiB)          94.359      97.718      +3.359
solver call, peak above entry (MiB)        91.588      94.951      +3.363
solver call, retained at return (MiB)      17.628      17.629      +0.001
=========================================  ==========  ==========  ==========

The whole-run peak *is* the solver call in both trees, so the two rows move
together.  The transient the question predicted is real and is
``+3.36 MiB`` -- less than one cube (5.548 MiB), because the blocks accumulate
over the time loop rather than existing all at once alongside a pre-allocated
cube, and only coexist with the assembled cube during the final ``stack``.
Against a 94.4 MiB peak that is **+3.6%, and nowhere near a doubling**; nothing
is retained after the call, because the blocks are freed as soon as the
assembly returns.  Q3's conditional therefore does not fire: no per-backend
assembly strategy is needed, and the single block-structured path is kept for
every backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from radiosim.api import Simulator
from radiosim.backends.numpy_backend import NumPyBackend
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, SkyModel
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from tests.fixtures.configs import valid_config_mapping

LOCATION = EarthLocation.from_geodetic(21.4283 * u.deg, -30.72152 * u.deg, 1073.0 * u.m)
OBSTIME = Time("2025-01-01T00:00:00")
TIME_GRID = build_observation_time_grid(
    start_time=OBSTIME.isot,
    duration_seconds=3.0,
    cadence_seconds=1.0,
)
FREQUENCIES = np.array([100e6, 101e6], dtype=np.float64)


class _AssemblyCountingBackend(NumPyBackend):
    """A NumPy backend that records every accumulation operation it is asked for."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.set_at_calls = 0
        self.stack_shapes: list[tuple[int, ...]] = []

    def set_at(self, arr: Any, index: Any, value: Any) -> Any:
        self.set_at_calls += 1
        return super().set_at(arr, index, value)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        result = super().stack(arrays, axis=axis)
        self.stack_shapes.append(tuple(np.asarray(result).shape))
        return result

    def assemblies_of_rank(self, rank: int) -> list[tuple[int, ...]]:
        return [shape for shape in self.stack_shapes if len(shape) == rank]


def _solver_components(tmp_path, **overrides: Any):
    from radiosim.core.instrument_adapters import SolverInstrumentView

    data = valid_config_mapping(tmp_path, **overrides)
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
    )


def _point_sources(*, visible: bool) -> dict[str, Any]:
    lst_rad = OBSTIME.sidereal_time("apparent", longitude=LOCATION.lon).rad
    n = 2
    zeros = np.zeros(n, dtype=np.float64)
    # A declination of +85 deg is permanently below the horizon at -30.7 deg.
    dec = np.array([-0.536, -0.526]) if visible else np.array([1.4835, 1.4835])
    return {
        "ra_rad": np.array([lst_rad, lst_rad + 0.01], dtype=np.float64),
        "dec_rad": dec.astype(np.float64),
        "flux": np.array([2.0, 1.0], dtype=np.float64),
        "spectral_index": np.array([-0.7, -0.8], dtype=np.float64),
        "stokes_q": np.array([0.2, 0.0], dtype=np.float64),
        "stokes_u": np.array([0.0, 0.1], dtype=np.float64),
        "stokes_v": np.array([0.05, 0.0], dtype=np.float64),
        "ref_freq": np.full(n, 100e6, dtype=np.float64),
        "rotation_measure": zeros.copy(),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": zeros.copy(),
        "minor_arcsec": zeros.copy(),
        "pa_deg": zeros.copy(),
    }


def _healpix_model(*, polarized: bool) -> SkyModel:
    npix = 12
    maps = np.linspace(1.0, 2.0, npix, dtype=np.float64)
    maps = np.vstack([maps, maps * 1.1])
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=1,
            frequencies=FREQUENCIES,
            coordinate_frame="icrs",
            q_maps=np.full_like(maps, 0.1) if polarized else None,
            u_maps=np.full_like(maps, 0.05) if polarized else None,
            v_maps=np.full_like(maps, 0.02) if polarized else None,
        ),
        model_name="tier6d-accumulation",
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )


def _assert_block_assembly_shape(backend: _AssemblyCountingBackend, cube: Any) -> None:
    """Assert the exact Section 13.3 assembly shape for one solver call.

    Narrowed by Tier 6H, deliberately and in exactly one respect. Tier 6D
    assembled the per-(time, frequency) ``(B, 2, 2)`` block with
    ``backend.stack`` over ``B`` separately computed ``(2, 2)`` matrices,
    because the contraction ran one baseline at a time. Section 13.6's compiled
    kernel is *baseline-batched*: it returns that whole ``(B, 2, 2)`` block from
    one call, so there is no longer anything to assemble at that level. That is
    strictly fewer assemblies, not more, and every binding property of
    Section 13.3 is unchanged and still asserted below -- one ``(B, F, 2, 2)``
    block per time, exactly one whole-cube assembly per call, and zero
    ``set_at`` calls.

    The kernel's own input batching (the two ``(B, S, 2, 2)`` antenna-Jones
    batches per step) deliberately does **not** go through ``ArrayBackend.stack``
    and therefore does not appear in these counts: ``stack`` is documented as the
    solvers' one *accumulation* primitive, and conflating input batching with
    output accumulation is what would make these counts unreadable.
    """
    array = np.asarray(cube)
    n_times, n_baselines, n_freqs = array.shape[:3]

    # R2: exactly one whole-cube assembly per solver call.
    assert backend.assemblies_of_rank(5) == [array.shape]

    # No per-cell functional copies survive anywhere in the hot path (D11).
    assert backend.set_at_calls == 0

    # The per-(time, frequency) block now comes straight out of the kernel.
    assert backend.assemblies_of_rank(3) == []
    # One (B, F, 2, 2) block per time.
    assert backend.assemblies_of_rank(4) == [(n_baselines, n_freqs, 2, 2)] * n_times
    assert len(backend.stack_shapes) == n_times + 1


def test_point_solver_assembles_one_cube_from_per_time_blocks(tmp_path) -> None:
    """R2 for the point solver: one whole-cube assembly, zero ``set_at`` calls."""
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = _AssemblyCountingBackend()

    cube = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=_point_sources(visible=True),
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=receptors,
    )

    assert np.asarray(cube).shape[:3] == (3, len(instrument.selected_pairs), 2)
    assert float(np.max(np.abs(np.asarray(cube)))) > 0.0
    _assert_block_assembly_shape(backend, cube)


@pytest.mark.parametrize("polarized", [False, True])
def test_healpix_solver_assembles_one_cube_from_per_time_blocks(
    tmp_path,
    polarized: bool,
) -> None:
    """R2 for both HEALPix paths: one whole-cube assembly, zero ``set_at`` calls."""
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = _AssemblyCountingBackend()

    cube = calculate_visibility_healpix(
        _healpix_model(polarized=polarized),
        instrument=instrument,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=receptors,
        include_polarization=polarized,
    )

    assert float(np.max(np.abs(np.asarray(cube)))) > 0.0
    _assert_block_assembly_shape(backend, cube)


def test_a_time_with_no_visible_sources_still_contributes_one_block(tmp_path) -> None:
    """The skipped time step must still occupy its slot, and stay exactly zero."""
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = _AssemblyCountingBackend()

    cube = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=_point_sources(visible=False),
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=receptors,
    )

    array = np.asarray(cube)
    n_times, n_baselines, n_freqs = array.shape[:3]
    np.testing.assert_array_equal(array, np.zeros_like(array))
    assert backend.set_at_calls == 0
    # Every time step short-circuits before the frequency loop and contributes a
    # pre-zeroed (B, F, 2, 2) block, so the only assembly is the final cube.
    assert backend.stack_shapes == [array.shape]
    assert array.shape == (n_times, n_baselines, n_freqs, 2, 2)


def test_empty_point_source_batch_assembles_nothing(tmp_path) -> None:
    """The degenerate batch returns the zero cube without any assembly at all."""
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = _AssemblyCountingBackend()
    source_arrays = _point_sources(visible=True)
    for key, value in tuple(source_arrays.items()):
        if isinstance(value, np.ndarray):
            source_arrays[key] = value[:0]

    cube = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=source_arrays,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=receptors,
    )

    array = np.asarray(cube)
    assert array.shape == (3, len(instrument.selected_pairs), 2, 2, 2)
    np.testing.assert_array_equal(array, np.zeros_like(array))
    assert backend.stack_shapes == []
    assert backend.set_at_calls == 0


def test_assembled_cube_is_contiguous_in_canonical_axis_order(tmp_path) -> None:
    """The assembly produces the canonical cube directly, with no transpose."""
    instrument, beam_system, receptors = _solver_components(tmp_path)

    cube = np.asarray(
        calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=_point_sources(visible=True),
            location=LOCATION,
            time_grid=TIME_GRID,
            frequencies=FREQUENCIES,
            backend=NumPyBackend(),
            receptors=receptors,
        )
    )

    assert cube.shape == (3, len(instrument.selected_pairs), 2, 2, 2)
    assert cube.flags["C_CONTIGUOUS"]
    assert cube.dtype == np.dtype(np.complex128)


@pytest.mark.parametrize("preset", ["standard", "fast"])
def test_block_assembly_preserves_the_output_dtype_for_every_precision(
    tmp_path,
    preset: str,
) -> None:
    """Assembly never widens or narrows the declared output complex dtype."""
    precision = getattr(PrecisionConfig, preset)()
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = _AssemblyCountingBackend(precision=precision)

    cube = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=_point_sources(visible=True),
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=receptors,
    )

    expected = np.dtype(backend.get_complex_dtype("output"))
    assert np.asarray(cube).dtype == expected
    assert np.all(np.isfinite(np.asarray(cube)))
    _assert_block_assembly_shape(backend, cube)


def test_solver_sources_contain_no_per_cell_set_at_write() -> None:
    """Source truth for D11's closure, independent of any counting wrapper."""
    import inspect

    from radiosim.core import visibility as point_module
    from radiosim.core import visibility_healpix as healpix_module

    for module in (point_module, healpix_module):
        source = inspect.getsource(module)
        assert "backend.set_at(" not in source, module.__name__
        assert "backend.stack(" in source, module.__name__
