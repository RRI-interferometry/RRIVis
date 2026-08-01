"""Strict, frozen ``jones:`` configuration input models.

``Tier7JonesSciencePlan.md`` Section 21.  A new top-level section on
:class:`~radiosim.io.config.RadioSimConfig`, placed in its own module and
imported into ``io/config.py`` -- the same placement Tier 5 used for
``receptors:`` (``io/receptor_config.py``).

Three properties the section must have, and how each is obtained here
--------------------------------------------------------------------
1. **Every term is absent by default.**  Each field defaults to ``None`` and an
   absent term is not in the chain at all, so a configuration that does not
   mention ``jones:`` is bit-identical to one written before the section
   existed (invariant I1).
2. **A present term must do something.**  There is no ``enabled: false``:
   configuring a term whose resolved parameters make it exactly ``I2`` is
   rejected (R7), because a term that cannot change the visibilities is
   indistinguishable from no term, which is the ``SCI-001`` defect this tier
   removes.  That check needs *resolved* values, so it lives in
   :func:`~radiosim.core.jones_terms.resolve_jones_terms` and not here.
3. **Unknown input is rejected, not ignored.**  Every model is a
   :class:`~radiosim.io.model_base.StrictFrozenModel`: extra keys forbidden,
   instances frozen.  Model variants are discriminated unions on ``kind``, so an
   unknown ``kind`` is rejected by Pydantic itself rather than by a hand-written
   branch that a later variant could forget to update.

Which terms are here
--------------------
``G`` and ``B`` only.  Tier 7D implements exactly those two, and a schema field
for a term whose ``compute_jones_batch`` raises would be a configuration surface
that cannot be honoured -- the defect ``D2`` shape that Tier 7C stripped from the
planned terms' constructors.  Tier 7E-7H each add their own term's block to this
module together with its physics.

Units and complex numbers (Section 21.3)
----------------------------------------
Every angle field carries ``_rad`` or ``_deg``, every frequency ``_hz``; no unit
is implicit.  A complex number is a two-element ``[re, im]`` sequence, never a
Python complex literal and never a string, so the YAML stays round-trippable.  A
field that accepts a complex value also accepts a bare real number, which is
what ``coefficients: [1.0, 0.0, -0.05]`` in the plan's own example is.

The one departure from the ``_s`` suffix rule is deliberate and follows the
plan's own accepted YAML: the gain time model is parameterized in **hours**
(``rate_per_hour``, ``period_hours``), because an observation is hours long and
a per-second drift rate would be written as ``2.8e-6`` in every real
configuration.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from radiosim.io.model_base import StrictFrozenModel

__all__ = [
    "BandpassOverrideConfig",
    "BandpassPolynomialModel",
    "BandpassTabulatedModel",
    "BandpassTermConfig",
    "ComplexInput",
    "GainOverrideConfig",
    "GainTermConfig",
    "GainTimeModelConfig",
    "JONES_TERM_LETTERS",
    "JonesConfig",
    "LinearDriftTimeModel",
    "SinusoidalTimeModel",
    "StaticTimeModel",
    "as_complex",
]

_StrictFiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_StrictPositiveFloat = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0.0)]

#: A complex configuration value: ``[re, im]``, or a bare real number.
ComplexInput: TypeAlias = (
    tuple[_StrictFiniteFloat, _StrictFiniteFloat] | _StrictFiniteFloat
)

#: The term letters this schema accepts, in canonical chain order.  Tier 7E-7H
#: extend it as each term becomes real.
JONES_TERM_LETTERS: tuple[str, ...] = ("G", "B")


def as_complex(value: ComplexInput) -> complex:
    """Return one Python ``complex`` from a configured complex value.

    ``[re, im]`` becomes ``re + im j``; a bare real number becomes a real
    ``complex``.  The one conversion, so no consumer invents a second.
    """
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    real, imaginary = value
    return complex(float(real), float(imaginary))


# --------------------------------------------------------------------------- G


class StaticTimeModel(StrictFrozenModel):
    """``s(t) = 1``: the gain does not vary over the observation."""

    kind: Literal["constant"] = "constant"


class LinearDriftTimeModel(StrictFrozenModel):
    """``s(t) = 1 + rate * (t - t0)``, with ``t`` in hours from the first sample.

    Parameters
    ----------
    rate_per_hour
        Fractional gain drift per hour.  Positive rises, negative falls.
    """

    kind: Literal["linear_drift"]
    rate_per_hour: _StrictFiniteFloat


class SinusoidalTimeModel(StrictFrozenModel):
    """``s(t) = 1 + depth * sin(2 pi (t - t0) / period + phase)``.

    Parameters
    ----------
    depth
        Fractional peak amplitude of the oscillation.
    period_hours
        Oscillation period, in hours; must be positive.
    phase_rad
        Phase at the first time sample, in radians.
    """

    kind: Literal["sinusoidal"]
    depth: _StrictFiniteFloat
    period_hours: _StrictPositiveFloat
    phase_rad: _StrictFiniteFloat = 0.0


GainTimeModelConfig = Annotated[
    StaticTimeModel | LinearDriftTimeModel | SinusoidalTimeModel,
    Field(discriminator="kind"),
]


class GainOverrideConfig(StrictFrozenModel):
    """One per-``(antenna, feed)`` gain override.

    Parameters
    ----------
    antenna
        The antenna **number** in the resolved instrument.  Not a Tier 2 tagged
        reference: Section 21.3 keys every ``per_antenna`` entry by number, and
        the resolution step rejects a number the instrument does not have (R4).
    feed
        ``0`` or ``1``, in the antenna's own receptor basis -- which is why the
        gain is defined per feed *index* and not per ``x``/``y`` (Section 20.0).
        The value is validated at resolution (R6) rather than here, so that the
        message names the term.
    amplitude_error, phase_error_rad
        Optional replacements for the array-wide defaults.  ``None`` keeps the
        default, so an override may change only one of the two.
    """

    antenna: int
    feed: int
    amplitude_error: _StrictFiniteFloat | None = None
    phase_error_rad: _StrictFiniteFloat | None = None

    @model_validator(mode="after")
    def require_override_content(self) -> Self:
        if self.amplitude_error is None and self.phase_error_rad is None:
            raise ValueError(
                "must set at least one of 'amplitude_error' or 'phase_error_rad'"
            )
        return self


class GainTermConfig(StrictFrozenModel):
    """The ``G`` term: complex electronic gain (Section 20.1).

    ``g_pf(t) = (1 + a_pf) * exp(i phi_pf) * s_pf(t)``, times an optional
    elevation gain curve shared by both feeds.

    Parameters
    ----------
    amplitude_error
        The array-wide fractional amplitude error ``a``.
    phase_error_rad
        The array-wide phase error ``phi``, in radians.
    per_antenna
        Ordered per-``(antenna, feed)`` overrides.  An explicit entry beats the
        array-wide default; there is no third source and no environment or CLI
        override for a Jones parameter (Section 22 rule 5).
    elevation_curve
        Optional polynomial coefficients ``c_k`` of the elevation gain curve
        ``g_el(el) = sum_k c_k el^k``, with ``el`` in **degrees**, lowest order
        first.  ``None`` disables the curve entirely.
    time_model
        The reproducible time variation ``s(t)``.  None of the three kinds draws
        a random number: every one is exactly reproducible from configuration.
    """

    amplitude_error: _StrictFiniteFloat = 0.0
    phase_error_rad: _StrictFiniteFloat = 0.0
    per_antenna: tuple[GainOverrideConfig, ...] = ()
    elevation_curve: tuple[_StrictFiniteFloat, ...] | None = None
    time_model: GainTimeModelConfig = Field(default_factory=StaticTimeModel)

    @model_validator(mode="after")
    def require_nonempty_elevation_curve(self) -> Self:
        if self.elevation_curve is not None and not self.elevation_curve:
            raise ValueError(
                "jones.G.elevation_curve must have at least one coefficient; "
                "omit the field to disable the elevation gain curve"
            )
        return self


# --------------------------------------------------------------------------- B


class BandpassPolynomialModel(StrictFrozenModel):
    """``b(nu) = sum_k c_k x^k`` with ``x = (nu - nu_ref) / nu_scale``.

    Parameters
    ----------
    coefficients
        Complex polynomial coefficients, lowest order first.  A bare real
        number is accepted for a real coefficient.
    reference_frequency_hz
        ``nu_ref``.  ``None`` resolves to the band centre.
    scale_frequency_hz
        ``nu_scale``.  ``None`` resolves to the half-bandwidth, so that ``x``
        spans ``[-1, 1]`` across the observed band and a low-order polynomial is
        well conditioned.  A single-channel observation has no half-bandwidth,
        and resolution rejects a scale it cannot derive.
    """

    kind: Literal["polynomial"]
    coefficients: tuple[ComplexInput, ...]
    reference_frequency_hz: _StrictPositiveFloat | None = None
    scale_frequency_hz: _StrictPositiveFloat | None = None

    @model_validator(mode="after")
    def require_coefficients(self) -> Self:
        if not self.coefficients:
            raise ValueError("must have at least one coefficient")
        return self


class BandpassTabulatedModel(StrictFrozenModel):
    """Complex gains at explicit node frequencies, cubic-spline interpolated.

    Real and imaginary parts are splined separately.  Frequencies outside the
    node range are **rejected** at resolution and never extrapolated (R11): a
    bandpass extrapolated past its measurement is a fabricated number, and
    RadioSim does not fabricate one silently.

    Parameters
    ----------
    node_frequencies_hz
        Strictly increasing node frequencies, at least four of them, which is
        the minimum a cubic spline is determined by.
    gains
        One complex gain per node, in the same order.
    """

    kind: Literal["tabulated"]
    node_frequencies_hz: tuple[_StrictPositiveFloat, ...]
    gains: tuple[ComplexInput, ...]

    @model_validator(mode="after")
    def require_consistent_nodes(self) -> Self:
        nodes = self.node_frequencies_hz
        if len(nodes) < 4:
            raise ValueError(
                "must have at least 4 node frequencies for a cubic spline, got "
                f"{len(nodes)}"
            )
        if any(
            later <= earlier for earlier, later in zip(nodes, nodes[1:], strict=False)
        ):
            raise ValueError("node_frequencies_hz must be strictly increasing")
        if len(self.gains) != len(nodes):
            raise ValueError(
                f"gains has {len(self.gains)} entries but node_frequencies_hz has "
                f"{len(nodes)}; there must be exactly one gain per node"
            )
        return self


BandpassModelConfig = Annotated[
    BandpassPolynomialModel | BandpassTabulatedModel,
    Field(discriminator="kind"),
]


class BandpassOverrideConfig(StrictFrozenModel):
    """One per-``(antenna, feed)`` bandpass override.

    Parameters
    ----------
    antenna, feed
        The same antenna-number and feed-index keying as
        :class:`GainOverrideConfig`, validated at resolution (R4, R5, R6).
    model
        The replacement frequency-response model for this feed.  Required: a
        bandpass has exactly one overridable quantity, so an override with no
        model would be an empty entry.
    """

    antenna: int
    feed: int
    model: BandpassModelConfig


class BandpassTermConfig(StrictFrozenModel):
    """The ``B`` term: frequency-dependent bandpass (Section 20.2).

    Parameters
    ----------
    model
        The array-wide frequency response applied to every feed without an
        override.
    per_antenna
        Ordered per-``(antenna, feed)`` overrides, same shape as ``G``'s.
    """

    model: BandpassModelConfig
    per_antenna: tuple[BandpassOverrideConfig, ...] = ()


# ------------------------------------------------------------------- the section


class JonesConfig(StrictFrozenModel):
    """The ``jones:`` section: which Jones terms this run enables.

    Every field is optional and defaults to ``None``.  A ``None`` term is not
    resolved, not constructed, and not in the chain, and contributes nothing to
    the scientific fingerprint -- so ``jones:`` absent and a ``jones:`` with a
    term the user later removes are the same run, bit for bit.

    A present-but-empty section (``jones: {}``) is rejected at resolution (R2)
    rather than treated as absent, because writing an empty section is a
    statement of intent that the file does not carry out, and silently accepting
    it would hide a deleted term or a mis-indented key.
    """

    G: GainTermConfig | None = None
    B: BandpassTermConfig | None = None

    @property
    def configured_terms(self) -> tuple[str, ...]:
        """Return the configured term letters, in canonical chain order."""
        return tuple(
            letter
            for letter in JONES_TERM_LETTERS
            if getattr(self, letter, None) is not None
        )
