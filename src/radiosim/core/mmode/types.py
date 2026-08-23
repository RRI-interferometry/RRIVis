"""Canonical SCI-004 value types and the Section 14 digest vocabulary.

``docs/development/sci004_mmode_design.md`` Section 14.0 fixes one digest
vocabulary for the whole m-mode programme, and Sections 3.1, 5.3 and 7.3 fix the
exact literals, key orders and derivations every record and every runtime object
must reproduce.  This module owns those primitives so that the runtime, the
evidence generator and the strict validators all read them from one place
instead of each spelling a rule that only looks the same.

The primitives are deliberately narrow:

``canonical_rational`` / ``parse_rational``
    Section 3.1's normalized ``p/q`` serialization: positive denominator,
    ``gcd(abs(p), q) == 1``, zero only as ``0/1``, shortest base-10 ASCII with no
    leading ``+``, whitespace or redundant zero.
``f64be``
    Section 14.0's ``F64(x)``, the lowercase 16-hex-character encoding of
    ``struct.pack(">d", x)``.  Scientific binary64 values inside identity
    manifests are ``F64`` strings, never JSON decimals.
``canonical_json``
    Section 14's ``J(x)``: UTF-8, lexicographically sorted object keys,
    separators ``(',', ':')``, ``ensure_ascii=true``, no whitespace and no
    trailing newline, with RFC 8785 / ECMAScript shortest-round-trip numbers.
``domain_digest`` / ``array_digest``
    Section 14.0's ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)`` and the
    numeric-array primitive ``A(d, role, axes, units, array)`` that authenticates
    role, rank, shape, axis names, units, endianness and dtype before hashing.

Nothing here evaluates physics.  The frozen containers below (``MModeDimensions``,
``ScalarPackedTable``, ``ScalarHarmonicCoefficients``, ``ScalarPackedCube``) carry
resolved values whose invariants Sections 5.3 and 7.3 state, so a consumer can
never pair a packed value buffer with a table that does not describe it.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final

import numpy as np

__all__ = [
    "CONVENTION_IDENTITY",
    "FIELD_ORDER",
    "MMODE_CONVENTION",
    "MMODE_EXECUTION_POLICY",
    "MMODE_FRAME_MODEL",
    "MMODE_HARMONIC_CONVENTION",
    "MMODE_QUADRATURE_POLICY",
    "MMODE_STOKES_BRIDGE",
    "MMODE_TANGENT_FRAME_M1",
    "MMODE_TIME_GRID_CONVENTION",
    "MMODE_TRUNCATION_POLICY",
    "SCALAR_BLOCK_TABLE_DOMAIN",
    "SPIN_ORDER",
    "TAU",
    "MModeDimensions",
    "ScalarHarmonicCoefficients",
    "ScalarPackedCube",
    "ScalarPackedTable",
    "array_digest",
    "canonical_json",
    "canonical_rational",
    "derive_mmode_dimensions",
    "domain_digest",
    "f64be",
    "next_power_of_two",
    "parse_rational",
    "round_to_nearest_turn_radians",
]

# ---------------------------------------------------------------------------
# Section 3.1 / 8 / 10 / 14.0 exact literals
# ---------------------------------------------------------------------------

#: Section 3.1's exact binary64 turn-to-radian constant, spelled as the hex
#: literal the design uses precisely so no decimal transcription perturbs it.
TAU: Final[float] = float.fromhex("0x1.921fb54442d18p+2")

MMODE_CONVENTION: Final = "radiosim.mmode-forward.v1"
MMODE_FRAME_MODEL: Final = "radiosim.frozen-cirs-rigid-era.v1"
MMODE_HARMONIC_CONVENTION: Final = "radiosim.shaw-polarized-harmonics.v1"
MMODE_TIME_GRID_CONVENTION: Final = "radiosim.mmode-era-turn-grid.v1"
MMODE_RADIAN_GRID_CONVENTION: Final = "radiosim.mmode-era-grid.v2"
MMODE_TANGENT_FRAME_V1: Final = "radiosim.sky-tangent-polarization.v1"
MMODE_TANGENT_FRAME_M1: Final = "not_applicable_scalar_m1"
MMODE_STOKES_BRIDGE: Final = "radiosim.stokes-ne-theta-phi.v1"
MMODE_QUADRATURE_POLICY: Final = "iso-gauss-ring-production-plus-qcheck.v1"
MMODE_TRUNCATION_POLICY: Final = "complete-frozen-direct-plus-local-shells.v1"
MMODE_EXECUTION_POLICY: Final = "host_harmonics_backend_native_dense_v1"

SCALAR_BLOCK_TABLE_DOMAIN: Final = "radiosim.mmode-packed-block-table.v1"

#: Section 5.3's science field order and its spin labels.
FIELD_ORDER: Final[tuple[str, ...]] = ("I", "+2", "-2", "V")
SPIN_ORDER: Final[tuple[int, ...]] = (0, 2, -2, 0)

#: Section 14.0's ``convention_identity`` object, verbatim.  No listed
#: convention is implementation-selected, so the mapping is a literal.
CONVENTION_IDENTITY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema_version": "radiosim.mmode-conventions.v1",
        "execution": MMODE_CONVENTION,
        "era_turn_grid": MMODE_TIME_GRID_CONVENTION,
        "era_radian_grid": MMODE_RADIAN_GRID_CONVENTION,
        "frozen_frame": MMODE_FRAME_MODEL,
        "harmonics": MMODE_HARMONIC_CONVENTION,
        "tangent_frame": MMODE_TANGENT_FRAME_V1,
        "stokes_bridge": MMODE_STOKES_BRIDGE,
        "transfer_catalog": "radiosim.mmode-transfer-grid-catalog.v1",
        "quadrature_policy": MMODE_QUADRATURE_POLICY,
        "truncation_policy": MMODE_TRUNCATION_POLICY,
        "execution_policy": MMODE_EXECUTION_POLICY,
        "field_order": ["I", "+2", "-2", "V"],
        "spin_order": [0, 2, -2, 0],
        "dft_forward": "bar_v_m=(1/N)*sum_k(bar_V_k*exp(-i*2*pi*m*u_k))",
        "dft_inverse": "bar_V_k=sum_m(bar_v_m*exp(+i*2*pi*m*u_k))",
        "exposure_window": "exact-turn-top-hat-sinc.v1",
        "horizon_predicate": "sin_altitude_strictly_greater_than_zero",
    }
)


# ---------------------------------------------------------------------------
# Section 3.1 exact rational serialization
# ---------------------------------------------------------------------------


def canonical_rational(value: Fraction | int) -> str:
    """Return Section 3.1's normalized ``p/q`` spelling of an exact rational.

    Parameters
    ----------
    value : Fraction or int
        The exact value.  ``Fraction`` already normalizes to a positive
        denominator with ``gcd(abs(p), q) == 1``, which is exactly the required
        form; zero is therefore represented only as ``0/1``.

    Returns
    -------
    str
        Shortest base-10 ASCII numerator, ``/``, shortest base-10 ASCII
        denominator.  No leading ``+``, whitespace, or redundant leading zero.
    """
    exact = Fraction(value)
    return f"{exact.numerator}/{exact.denominator}"


def parse_rational(text: str) -> Fraction:
    """Parse a Section 3.1 canonical ``p/q`` string back to its exact value.

    Raises
    ------
    ValueError
        If ``text`` is not the canonical spelling of its own value.  A consumer
        that accepted ``2/4`` or ``+1/2`` would let two spellings of one number
        produce two digests, which Section 14.0 forbids.
    """
    if not isinstance(text, str) or "/" not in text:
        raise ValueError(f"not a canonical rational: {text!r}")
    numerator_text, _, denominator_text = text.partition("/")
    try:
        value = Fraction(int(numerator_text), int(denominator_text))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"not a canonical rational: {text!r}") from exc
    if canonical_rational(value) != text:
        raise ValueError(f"not a canonical rational: {text!r}")
    return value


def round_to_nearest_turn_radians(exact_turn: Fraction) -> float:
    """Return ``RN(exact(tau) * u)`` with one final round-to-nearest.

    Section 3.1 requires the derived radian view to be a single correctly
    rounded conversion of ``exact(tau) * <exact rational>`` with **no**
    intermediate binary64 arithmetic.  ``Fraction`` multiplication is exact and
    ``float(Fraction)`` is the unique round-to-nearest-ties-to-even binary64,
    so this expression is the definition rather than an approximation of it.
    """
    return float(Fraction(*TAU.as_integer_ratio()) * Fraction(exact_turn))


# ---------------------------------------------------------------------------
# Section 14.0 canonical JSON and digest vocabulary
# ---------------------------------------------------------------------------


def _ecmascript_number(value: float) -> str:
    """Render a finite binary64 with ECMAScript ``Number::toString`` spelling.

    Section 14 requires RFC 8785 / ECMAScript shortest-round-trip numbers, so
    ``1.0`` and ``1e0`` are not canonical bytes for the integer one.  Python's
    ``repr`` already produces the shortest round-tripping decimal; only the
    integral-value and exponent spellings differ, and both are corrected here.
    """
    if not math.isfinite(value):
        raise ValueError("canonical JSON forbids NaN and Infinity")
    if value == 0.0:
        return "0"
    text = repr(float(value))
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    mantissa, _, exponent_text = text.partition("e")
    if exponent_text:
        exponent = int(exponent_text)
    else:
        exponent = 0
    integer_part, _, fraction_part = mantissa.partition(".")
    digits = (integer_part + fraction_part).lstrip("0")
    # ``point`` is the decimal exponent of the first retained significant digit.
    point = len(integer_part) - (len(integer_part + fraction_part) - len(digits))
    point += exponent
    digits = digits.rstrip("0") or "0"
    count = len(digits)
    if count <= point <= 21:
        rendered = digits + "0" * (point - count)
    elif 0 < point <= 21:
        rendered = digits[:point] + "." + digits[point:]
    elif -6 < point <= 0:
        rendered = "0." + "0" * (-point) + digits
    else:
        exponent_value = point - 1
        sign = "+" if exponent_value >= 0 else "-"
        head = digits[0] if count == 1 else digits[0] + "." + digits[1:]
        rendered = f"{head}e{sign}{abs(exponent_value)}"
    return "-" + rendered if negative else rendered


def _render(value: Any) -> str:
    """Render one JSON value with Section 14's exact serialization.

    ``json.dumps`` is deliberately not used for the numbers: its encoder is
    hard-wired to ``float.__repr__``, which spells the integer one as ``1.0``
    and the exponent of ``1e-7`` as ``1e-07``.  Section 14 accepts neither, so
    the number path is rendered here and only strings are delegated to the
    standard escaper.  Object keys are sorted by code point, which is exactly
    what ``sort_keys`` would have done.
    """
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _ecmascript_number(float(value))
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, Mapping):
        entries = sorted((str(key), item) for key, item in value.items())
        return (
            "{"
            + ",".join(
                json.dumps(key, ensure_ascii=True) + ":" + _render(item)
                for key, item in entries
            )
            + "}"
        )
    if isinstance(value, np.ndarray):
        return "[" + ",".join(_render(item) for item in value.tolist()) + "]"
    if isinstance(value, Sequence):
        return "[" + ",".join(_render(item) for item in value) + "]"
    raise TypeError(f"canonical JSON cannot encode {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Return Section 14's ``J(x)`` bytes for a JSON-primitive tree."""
    return _render(value).encode("utf-8")


def domain_digest(domain: str, payload: bytes) -> str:
    """Return Section 14.0's ``D(d, p) = SHA256(d || NUL || U64(len(p)) || p)``."""
    if not domain or not domain.isascii() or "\x00" in domain:
        raise ValueError("digest domain must be a non-empty NUL-free ASCII string")
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)
    return digest.hexdigest()


def streamed_domain_digest(domain: str, chunks: Callable[[], Iterator[bytes]]) -> str:
    """Return ``D(domain, payload)`` for a payload that is never materialized.

    Section 14.0's ``D`` prefixes the payload with its own length, so a single
    streaming pass cannot produce it.  ``chunks`` is therefore a factory that
    yields the payload's bytes on demand and is called twice: once to measure
    the length and once to feed the hash.  Nothing but the running length and
    the hash state is retained, which is what Sections 12.1 and 14.2 require of
    the ledgers whose expanded arrays are far larger than the artifact.
    """
    if not domain or not domain.isascii() or "\x00" in domain:
        raise ValueError("digest domain must be a non-empty NUL-free ASCII string")
    length = 0
    for chunk in chunks():
        length += len(chunk)
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(struct.pack(">Q", length))
    written = 0
    for chunk in chunks():
        digest.update(chunk)
        written += len(chunk)
    if written != length:
        raise ValueError("a streamed digest payload was not reproducible")
    return digest.hexdigest()


def object_digest(domain: str, value: Any) -> str:
    """Return ``D(domain, J(value))`` for a canonical JSON object."""
    return domain_digest(domain, canonical_json(value))


_DTYPE_NAMES: Final[Mapping[str, tuple[str, int]]] = MappingProxyType(
    {
        "float32-be": (">f4", 4),
        "float64-be": (">f8", 8),
        "complex64-be": (">c8", 8),
        "complex128-be": (">c16", 16),
        "int64-be": (">i8", 8),
        "uint64-be": (">u8", 8),
        "bool-u8": ("|u1", 1),
    }
)


def array_digest(
    domain: str,
    role: str,
    axes: Sequence[str],
    units: str,
    array: np.ndarray,
    *,
    dtype: str,
) -> str:
    """Return Section 14.0's ``A(d, role, axes, units, array)`` identity.

    Role, rank, shape, axis names, units, endianness and dtype are all
    authenticated before hashing, so two buffers with the same bytes but a
    different declared meaning can never share an identity.
    """
    if dtype not in _DTYPE_NAMES:
        raise ValueError(f"unsupported identity dtype {dtype!r}")
    numpy_dtype, itemsize = _DTYPE_NAMES[dtype]
    for text, label in ((domain, "domain"), (role, "role"), (units, "units")):
        if not text or not text.isascii() or "\x00" in text:
            raise ValueError(f"{label} must be a non-empty NUL-free ASCII string")
    axis_order = [str(axis) for axis in axes]
    if len(set(axis_order)) != len(axis_order):
        raise ValueError("axis names must be unique")
    for axis in axis_order:
        if not axis or not axis.isascii() or "\x00" in axis:
            raise ValueError("axis names must be non-empty NUL-free ASCII strings")
    values = np.asarray(array)
    if len(axis_order) != values.ndim:
        raise ValueError("axis_order length must equal the array rank")
    if not np.all(np.isfinite(values)):
        raise ValueError("identity arrays must be finite")
    shape = [int(extent) for extent in values.shape]
    if any(extent < 0 for extent in shape):
        raise ValueError("array shape entries must be non-negative")
    if dtype == "bool-u8":
        data = np.ascontiguousarray(values.astype(np.uint8)).tobytes(order="C")
    else:
        data = np.ascontiguousarray(values.astype(numpy_dtype)).tobytes(order="C")
    if math.prod(shape) * itemsize != len(data):
        raise ValueError("array payload length does not match its declared shape")
    header = canonical_json(
        {
            "axis_order": axis_order,
            "dtype": dtype,
            "role": role,
            "shape": shape,
            "units": units,
        }
    )
    payload = (
        struct.pack(">Q", len(header)) + header + struct.pack(">Q", len(data)) + data
    )
    return domain_digest(domain, payload)


def f64be(value: float) -> str:
    """Return Section 14.0's ``F64(x)``: 16 lowercase hex characters."""
    return struct.pack(">d", float(value)).hex()


def decode_f64be(text: str) -> float:
    """Return the binary64 a Section 14.0 ``F64`` string encodes."""
    if not isinstance(text, str) or len(text) != 16:
        raise ValueError(f"not an F64 string: {text!r}")
    if text != text.lower():
        raise ValueError(f"F64 strings are lowercase: {text!r}")
    return float(struct.unpack(">d", bytes.fromhex(text))[0])


# ---------------------------------------------------------------------------
# Section 7.3 truncation dimensions
# ---------------------------------------------------------------------------


def next_power_of_two(value: int) -> int:
    """Return the least power of two not smaller than a positive integer."""
    if value < 1:
        raise ValueError("next_power_of_two requires a positive integer")
    result = 1
    while result < value:
        result *= 2
    return result


@dataclass(frozen=True, slots=True)
class MModeDimensions:
    """Section 7.3's declared and derived truncation dimensions.

    ``lcheck``, ``mcheck`` and ``qcheck`` have no configuration knobs: every
    production run derives them from the three declared values by the frozen
    formulas, so a validation grid that could not represent the omitted physical
    tail is structurally impossible.
    """

    lmax: int
    mmax: int
    quadrature_nside: int
    lcheck: int
    mcheck: int
    qcheck: int

    @property
    def diagnostic_nsides(self) -> tuple[int, ...]:
        """Return Section 7.3's ``Q_diag``, which the frozen formulas fix."""
        return (self.qcheck,)

    @property
    def production_grid_id(self) -> str:
        """Return the exact ``production:<nside>`` catalogue identifier."""
        return f"production:{self.quadrature_nside}"

    @property
    def diagnostic_grid_ids(self) -> tuple[str, ...]:
        """Return the exact ``diagnostic:<nside>`` catalogue identifiers."""
        return tuple(f"diagnostic:{nside}" for nside in self.diagnostic_nsides)


def derive_mmode_dimensions(
    *, lmax: int, mmax: int, quadrature_nside: int
) -> MModeDimensions:
    """Derive Section 7.3's ``lcheck``, ``mcheck`` and ``qcheck``.

    ``lcheck = min(lmax + max(8, lmax // 8), 4096)``,
    ``mcheck = min(lcheck, mmax + max(8, max(1, mmax // 8)))`` and
    ``qcheck = next_power_of_two(max(2 * quadrature_nside, ceil(lcheck / 2)))``.
    """
    lcheck = min(lmax + max(8, lmax // 8), 4096)
    mcheck = min(lcheck, mmax + max(8, max(1, mmax // 8)))
    qcheck = next_power_of_two(max(2 * quadrature_nside, -(-lcheck // 2)))
    return MModeDimensions(
        lmax=lmax,
        mmax=mmax,
        quadrature_nside=quadrature_nside,
        lcheck=lcheck,
        mcheck=mcheck,
        qcheck=qcheck,
    )


# ---------------------------------------------------------------------------
# Section 5.3 packed scalar representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScalarPackedTable:
    """Section 5.3's scalar packed block table.

    Rows are ``(m, l_start, l_stop, value_start, value_stop)``, signed-m-major
    over the inclusive ascending range ``-mmax..mmax`` with ascending ``l``
    inside each row.  ``l_start = max(abs(m), abs(spin))`` -- ``abs(m)`` for the
    scalar spin-zero subset M1 binds -- and ``l_stop = lmax + 1``; each row
    starts at the preceding row's ``value_stop``, beginning at zero.  Invalid
    ``(l, m, s)`` cells **do not exist**: no padding is allocated, so no padded
    value can enter a digest.
    """

    lmax: int
    mmax: int
    block_rows: tuple[Mapping[str, int], ...]
    packed_value_count: int
    block_table_sha256: str

    @property
    def field_order(self) -> tuple[str, ...]:
        """Return Section 5.3's fixed science field order."""
        return FIELD_ORDER

    @property
    def spin_order(self) -> tuple[int, ...]:
        """Return the spin label of each field, in the same fixed order."""
        return SPIN_ORDER

    @property
    def block_table_domain(self) -> str:
        """Return the Section 14.0 domain the table digest is taken under."""
        return SCALAR_BLOCK_TABLE_DOMAIN

    @property
    def invalid_cell_count(self) -> int:
        """Return zero: an unpadded table represents no invalid cell."""
        return 0

    @property
    def block_count(self) -> int:
        """Return the number of signed-``m`` blocks, ``2 * mmax + 1``."""
        return len(self.block_rows)

    def index(self, degree: int, order: int) -> int:
        """Return the packed offset of one ``(l, m)`` cell.

        Raises
        ------
        IndexError
            If the cell is outside the retained ``(lmax, mmax)`` triangle.  An
            absent cell is absent, never silently zero.
        """
        if abs(order) > self.mmax or degree > self.lmax or degree < abs(order):
            raise IndexError(f"cell (l={degree}, m={order}) is not represented")
        row = self.block_rows[order + self.mmax]
        return int(row["value_start"]) + degree - int(row["l_start"])


@dataclass(frozen=True, slots=True)
class ScalarHarmonicCoefficients:
    """A packed scalar coefficient buffer bound to the table describing it.

    Section 5.3 makes the block table and the packed value buffer inseparable.
    ``__array__`` publishes the values so ordinary NumPy expressions work, while
    the table travels with them so ``scalar_coefficient`` never has to guess a
    layout from a buffer length.
    """

    table: ScalarPackedTable
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.complex128)
        if values.ndim != 1 or values.shape[0] != self.table.packed_value_count:
            raise ValueError("packed values do not match the block table")
        frozen = np.ndarray(
            values.shape, dtype=values.dtype, buffer=values.tobytes(order="C")
        )
        object.__setattr__(self, "values", frozen)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        array = np.array(self.values, copy=True)
        return array if dtype is None else array.astype(dtype)

    def __len__(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True, slots=True)
class ScalarPackedCube:
    """A ``(baseline, frequency, correlation, packed_value)`` transfer cube.

    Indexing with the leading three axes yields a
    :class:`ScalarHarmonicCoefficients` still carrying its table, while
    ``__array__`` publishes the dense complex cube for ordinary comparisons.
    """

    table: ScalarPackedTable
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.complex128)
        if values.ndim != 4 or values.shape[3] != self.table.packed_value_count:
            raise ValueError("packed cube does not match the block table")
        frozen = np.ndarray(
            values.shape, dtype=values.dtype, buffer=values.tobytes(order="C")
        )
        object.__setattr__(self, "values", frozen)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dense cube shape ``(B, F, C, packed_value)``."""
        return tuple(int(extent) for extent in self.values.shape)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        array = np.array(self.values, copy=True)
        return array if dtype is None else array.astype(dtype)

    def __getitem__(self, key: tuple[int, int, int]) -> ScalarHarmonicCoefficients:
        baseline, frequency, correlation = key
        return ScalarHarmonicCoefficients(
            table=self.table,
            values=self.values[baseline, frequency, correlation],
        )
