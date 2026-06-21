"""Characterization + behavior tests for the BBS parser decomposition (spec A3).

These pin the observable behavior of ``_parse_bbs_lines`` and its helper
pieces so the god-function decomposition (bracket-aware tokenizer + per-row
field parser + explicit coordinate-format detection) stays behavior-preserving.

They are invariant / known-analytic-case assertions (counts, dtypes, parsed
values of a known line), not golden snapshots of an entire file.
"""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.sky.loaders.bbs import (
    _parse_bbs_dec,
    _parse_bbs_lines,
    _parse_bbs_ra,
)

# A representative WSClean-style header + mixed body:
#   - decimal-degree coordinates with explicit "deg" suffix
#   - sexagesimal RA (colon) + sexagesimal Dec (dotted)
#   - a POINT source and a GAUSSIAN source
#   - a polarized source (Q/U/V set)
#   - a bracket-array spectral index [-0.8,-0.1]
_FIXTURE_LINES = [
    "Format = Name, Type, Ra, Dec, I, Q, U, V, "
    "MajorAxis, MinorAxis, Orientation, SpectralIndex, "
    "ReferenceFrequency='150e6'",
    # decimal-deg point source, polarized, multi-term spectral index
    "srcA, POINT, 123.5deg, -30.25deg, 4.0, 0.4, 0.2, 0.05, 0, 0, 0, [-0.8,-0.1]",
    # sexagesimal RA (hours) + dotted-sexagesimal Dec, gaussian
    "srcB, GAUSSIAN, 08:13:36.0, +48.13.02.25, 2.0, 0, 0, 0, 120, 60, 30, [-0.7]",
]


def test_parse_counts_and_shapes():
    parsed = _parse_bbs_lines(_FIXTURE_LINES, filename="fixture.skymodel")

    assert len(parsed.ra_deg) == 2
    assert len(parsed.dec_deg) == 2
    assert len(parsed.flux) == 2
    # Header reference frequency is parsed.
    assert parsed.ref_freq_from_header == pytest.approx(150e6)
    # One gaussian present -> has_gaussian.
    assert parsed.has_gaussian is True
    # One multi-term spectral index present -> has_spectral_coeffs.
    assert parsed.has_spectral_coeffs is True
    # All parsed coordinate / flux arrays are float64.
    for arr in (
        parsed.ra_deg,
        parsed.dec_deg,
        parsed.flux,
        parsed.spectral_index,
        parsed.stokes_q,
        parsed.stokes_u,
        parsed.stokes_v,
    ):
        assert arr.dtype == np.float64


def test_parse_known_line_values():
    parsed = _parse_bbs_lines(_FIXTURE_LINES, filename="fixture.skymodel")

    # srcA (decimal-deg).
    assert parsed.ra_deg[0] == pytest.approx(123.5)
    assert parsed.dec_deg[0] == pytest.approx(-30.25)
    assert parsed.flux[0] == pytest.approx(4.0)
    assert parsed.stokes_q[0] == pytest.approx(0.4)
    assert parsed.stokes_u[0] == pytest.approx(0.2)
    assert parsed.stokes_v[0] == pytest.approx(0.05)
    # First spectral-index term is the stored scalar spectral_index.
    assert parsed.spectral_index[0] == pytest.approx(-0.8)
    assert parsed.spectral_coeffs[0] == [-0.8, -0.1]

    # srcB sexagesimal RA: 08:13:36.0 hours -> degrees.
    expected_ra_b = (8 + 13 / 60.0 + 36.0 / 3600.0) * 15.0
    assert parsed.ra_deg[1] == pytest.approx(expected_ra_b)
    # srcB dotted-sexagesimal Dec: +48.13.02.25 deg.
    expected_dec_b = 48 + 13 / 60.0 + 2.25 / 3600.0
    assert parsed.dec_deg[1] == pytest.approx(expected_dec_b)
    # srcB gaussian morphology.
    assert parsed.major_arcsec[1] == pytest.approx(120.0)
    assert parsed.minor_arcsec[1] == pytest.approx(60.0)
    assert parsed.pa_deg[1] == pytest.approx(30.0)


def test_bracket_array_not_split_on_inner_comma():
    """The bracket-aware tokenizer must keep ``[-0.8,-0.1]`` as one field."""
    parsed = _parse_bbs_lines(_FIXTURE_LINES, filename="fixture.skymodel")
    # If the inner comma had split the field, the row would be misaligned and
    # the spectral coeffs would not recover both terms.
    assert parsed.spectral_coeffs[0] == [-0.8, -0.1]


def test_nonpositive_stokes_i_dropped():
    lines = [
        "Format = Name, Type, Ra, Dec, I",
        "good, POINT, 10.0deg, -20.0deg, 1.5",
        "neg, POINT, 11.0deg, -21.0deg, -2.0",
        "zero, POINT, 12.0deg, -22.0deg, 0.0",
    ]
    parsed = _parse_bbs_lines(lines, filename="drop.skymodel")
    assert len(parsed.flux) == 1
    assert parsed.flux[0] == pytest.approx(1.5)


# --- Explicit coordinate-format detection (A3 behavior assertion) ----------


def test_decimal_with_deg_suffix_is_decimal():
    assert _parse_bbs_ra("123.4deg") == pytest.approx(123.4)
    assert _parse_bbs_dec("-30.5deg") == pytest.approx(-30.5)


def test_dotted_dec_three_or_more_dots_is_sexagesimal():
    # +48.13.02.25 -> 48 deg 13 arcmin 02.25 arcsec.
    val = _parse_bbs_dec("+48.13.02.25")
    assert val == pytest.approx(48 + 13 / 60.0 + 2.25 / 3600.0)


def test_plain_decimal_dec_single_dot_is_decimal():
    """A single-dot value like ``-30.25`` is an unambiguous decimal degree."""
    assert _parse_bbs_dec("-30.25") == pytest.approx(-30.25)


def test_colon_ra_is_sexagesimal_hours():
    val = _parse_bbs_ra("08:13:36.0")
    assert val == pytest.approx((8 + 13 / 60.0 + 36.0 / 3600.0) * 15.0)


def test_ambiguous_two_dot_dec_uses_explicit_sexagesimal_rule():
    """A two-dot dotted-sexagesimal Dec (dd.mm.ss) parses by the explicit rule.

    ``-12.30.45`` is ambiguous to a naive reader but the explicit
    dotted-sexagesimal rule reads it as 12 deg 30 arcmin 45 arcsec (negative).
    """
    val = _parse_bbs_dec("-12.30.45")
    assert val == pytest.approx(-(12 + 30 / 60.0 + 45 / 3600.0))
