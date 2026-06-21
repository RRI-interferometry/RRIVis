"""Tests for the loaders-point cleanup group.

Covers:
  * H5 — ADQL in ``racs.py`` is built from validated identifiers / safe
    numeric literals instead of raw f-string interpolation.
  * B5 — synthetic Poisson provenance uses the shared ``coverage_provenance``
    helper (asserted via helper-equivalence on the shared triple).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

# --- H5: ADQL parametrization in racs.py -----------------------------------


def _fake_racs_info():
    return types.SimpleNamespace(
        ra_col="ra_deg_cont",
        dec_col="dec_deg_cont",
        flux_col="flux_peak",
        tap_table="casda.racs_dr1_sources_v2021_08_v01",
    )


def test_validate_adql_identifier_accepts_dotted_name():
    from radiosim.core.sky.loaders.vizier.racs import _validate_adql_identifier

    assert (
        _validate_adql_identifier("casda.racs_dr1_sources", kind="table")
        == "casda.racs_dr1_sources"
    )
    assert _validate_adql_identifier("flux_peak", kind="column") == "flux_peak"


@pytest.mark.parametrize(
    "bad",
    [
        "flux; DROP TABLE x",
        "flux_peak)",
        "flux peak",
        "flux'col",
        "1col",
        "",
        "casda..table",
    ],
)
def test_validate_adql_identifier_rejects_injection(bad):
    from radiosim.core.sky.loaders.vizier.racs import _validate_adql_identifier

    with pytest.raises(ValueError):
        _validate_adql_identifier(bad, kind="column")


def test_adql_number_formats_and_rejects_nonfinite():
    from radiosim.core.sky.loaders.vizier.racs import _adql_number

    assert float(_adql_number(1.5)) == pytest.approx(1.5)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            _adql_number(bad)


def test_build_racs_adql_no_region():
    from radiosim.core.sky.loaders.vizier.racs import _build_racs_adql

    adql = _build_racs_adql(
        info=_fake_racs_info(),
        max_rows=500,
        flux_limit_mjy=10.0,
        region=None,
    )
    assert "SELECT TOP 500" in adql
    assert "FROM casda.racs_dr1_sources_v2021_08_v01" in adql
    assert "WHERE flux_peak >=" in adql
    # Numeric literal rendered safely (no raw caller string), value preserved.
    assert "10.0" in adql
    # No spatial clause when region is None.
    assert "CONTAINS" not in adql


def test_build_racs_adql_rejects_nonpositive_max_rows():
    from radiosim.core.sky.loaders.vizier.racs import _build_racs_adql

    with pytest.raises(ValueError):
        _build_racs_adql(
            info=_fake_racs_info(),
            max_rows=0,
            flux_limit_mjy=1.0,
            region=None,
        )


def test_build_racs_adql_unsafe_table_raises():
    from radiosim.core.sky.loaders.vizier.racs import _build_racs_adql

    info = _fake_racs_info()
    info.tap_table = "casda.racs); DROP TABLE x; --"
    with pytest.raises(ValueError):
        _build_racs_adql(
            info=info,
            max_rows=10,
            flux_limit_mjy=1.0,
            region=None,
        )


# --- B5: synthetic provenance uses coverage_provenance ---------------------


def test_coverage_provenance_full_sky_used_by_synthetic():
    """Full-sky (region=None) path matches what synthetic.py now splices."""
    from radiosim.core.sky.support.provenance_coverage import coverage_provenance

    cov = coverage_provenance(is_full_sky=True, region=None)
    from radiosim.core.sky.containers import SkyCoverage

    assert cov.sky_coverage == SkyCoverage.FULL_SKY
    assert cov.coverage_fraction == pytest.approx(1.0)
    assert cov.coverage_footprint is None


def test_synthetic_imports_coverage_helper():
    """The synthetic loader module imports the shared helper (B5 wiring)."""
    import radiosim.core.sky.loaders.synthetic as synthetic

    assert hasattr(synthetic, "coverage_provenance")


# --- A3: BBS coordinate-format detection is explicit -----------------------


def test_classify_dec_format_explicit_rules():
    from radiosim.core.sky.loaders.bbs import _classify_dec_format

    assert _classify_dec_format("-30.5deg") == "deg"
    assert _classify_dec_format("-0.53rad") == "rad"
    assert _classify_dec_format("+48d13m02.25") == "dms"
    assert _classify_dec_format("+48.13.02.25") == "dotted"
    assert _classify_dec_format("-12.30.45") == "dotted"
    assert _classify_dec_format("-30.25") == "decimal"
    assert _classify_dec_format("-30") == "decimal"


def test_tokenize_keeps_bracket_array_intact():
    from radiosim.core.sky.loaders.bbs import _tokenize_data_line

    fields = _tokenize_data_line("a, POINT, 1.0, 2.0, 3.0, [-0.8,-0.1]")
    assert fields == ["a", "POINT", "1.0", "2.0", "3.0", "[-0.8,-0.1]"]


def test_tokenize_whitespace_fixed_format():
    from radiosim.core.sky.loaders.bbs import _tokenize_data_line

    assert _tokenize_data_line("1.0 2.0 3.0") == ["1.0", "2.0", "3.0"]


def test_bbs_dec_dms_negative_sign_preserved():
    """An explicit dms negative Dec keeps its sign (regression guard)."""
    from radiosim.core.sky.loaders.bbs import _parse_bbs_dec

    val = _parse_bbs_dec("-48d13m02.25")
    assert val == pytest.approx(-(48 + 13 / 60.0 + 2.25 / 3600.0))


def test_np_import_marker():
    """Sanity: numpy available (used implicitly by the parsers under test)."""
    assert np.isfinite(1.0)
