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
All eleven: ``G``, ``B``, ``Rc``, ``Kd``, ``X``, ``D``, ``P``, ``T`` and ``Z``,
the per-antenna chain, plus ``M`` and ``Q``, the two baseline-dependent Hadamard
terms Tier 7H added together with their physics.  Every letter this schema
accepts names a term that runs: a field for a term whose evaluation raised would
be a configuration surface that cannot be honoured, which is the defect ``D2``
shape one level up.

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
    "BaselineErrorOverrideConfig",
    "BaselineErrorTermConfig",
    "CableReflectionOverrideConfig",
    "CableReflectionTermConfig",
    "ComplexInput",
    "ComplexMatrix2x2",
    "ConstantTecModel",
    "CrosshandOverrideConfig",
    "CrosshandTermConfig",
    "DelayOverrideConfig",
    "DelayTermConfig",
    "ExplicitFeedLeakage",
    "ExplicitLeakageModel",
    "ExplicitZenithDelay",
    "FrequencyPolynomialFeedLeakage",
    "FrequencyPolynomialLeakageModel",
    "GainOverrideConfig",
    "GainTermConfig",
    "GainTimeModelConfig",
    "GradientTecModel",
    "IXRFeedLeakage",
    "IXRLeakageModel",
    "IonosphereTermConfig",
    "IonosphericFaradayConfig",
    "IonosphericFaradayOverrideConfig",
    "JONES_TERM_LETTERS",
    "JonesConfig",
    "LeakageOverrideConfig",
    "LeakageTermConfig",
    "LinearDriftTimeModel",
    "ParallacticTermConfig",
    "SaastamoinenZenithDelay",
    "SinusoidalTimeModel",
    "SmearingTermConfig",
    "StaticTimeModel",
    "TroposphereTermConfig",
    "TroposphericOpacityConfig",
    "as_complex",
]

_StrictFiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_StrictPositiveFloat = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0.0)]

#: A complex configuration value: ``[re, im]``, or a bare real number.
ComplexInput: TypeAlias = (
    tuple[_StrictFiniteFloat, _StrictFiniteFloat] | _StrictFiniteFloat
)

#: A ``2x2`` complex configuration value, row by row: the shape ``M``'s
#: per-baseline error takes.  Written as nested fixed-length tuples rather than
#: as a free sequence so that a mis-shaped matrix is a parse error naming the
#: position, not a runtime surprise.
ComplexMatrix2x2: TypeAlias = tuple[
    tuple[ComplexInput, ComplexInput],
    tuple[ComplexInput, ComplexInput],
]

#: The term letters this schema accepts: the nine chain letters in canonical
#: chain order (Section 12.2), then the two baseline-dependent ones Tier 7H
#: added.  Every letter here names a term whose physics exists.
JONES_TERM_LETTERS: tuple[str, ...] = (
    "G",
    "B",
    "Rc",
    "Kd",
    "X",
    "D",
    "P",
    "T",
    "Z",
    "M",
    "Q",
)


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


# -------------------------------------------------------------------------- Rc


class CableReflectionOverrideConfig(StrictFrozenModel):
    """One per-``(antenna, feed)`` cable-reflection override.

    Parameters
    ----------
    antenna, feed
        The same antenna-number and feed-index keying as
        :class:`GainOverrideConfig`, validated at resolution (R4, R5, R6).
    amplitude, cable_delay_s, phase_rad
        Optional replacements for the array-wide defaults.  ``None`` keeps the
        default, so an override may re-cable one feed without restating the
        other two numbers.
    """

    antenna: int
    feed: int
    amplitude: _StrictFiniteFloat | None = None
    cable_delay_s: _StrictFiniteFloat | None = None
    phase_rad: _StrictFiniteFloat | None = None

    @model_validator(mode="after")
    def require_override_content(self) -> Self:
        if (
            self.amplitude is None
            and self.cable_delay_s is None
            and self.phase_rad is None
        ):
            raise ValueError(
                "must set at least one of 'amplitude', 'cable_delay_s' or 'phase_rad'"
            )
        return self


class CableReflectionTermConfig(StrictFrozenModel):
    """The ``Rc`` term: single-bounce cable reflection (Section 20.6).

    ``r_pf(nu) = 1 + A_pf exp(-2 pi i nu tau_cable,pf + i phi_pf)``.

    Parameters
    ----------
    amplitude
        The dimensionless reflection amplitude ``A``.  The physical range
        ``0 < |A| < 1`` is enforced at resolution (R8) rather than here, so that
        the message names the physics -- a reflection cannot return more power
        than it receives -- instead of rendering a bounds error.
    cable_delay_s
        The round-trip cable delay, in seconds.  This is the delay at which the
        term's ripple appears in the delay domain, which is the observable that
        distinguishes ``Rc`` from a bandpass shape.
    phase_rad
        A phase offset, in radians.
    per_antenna
        Ordered per-``(antenna, feed)`` overrides.
    """

    amplitude: _StrictFiniteFloat
    cable_delay_s: _StrictFiniteFloat
    phase_rad: _StrictFiniteFloat = 0.0
    per_antenna: tuple[CableReflectionOverrideConfig, ...] = ()


# -------------------------------------------------------------------------- Kd


class DelayOverrideConfig(StrictFrozenModel):
    """One per-``(antenna, feed)`` instrumental-delay override.

    Parameters
    ----------
    antenna, feed
        The same antenna-number and feed-index keying as
        :class:`GainOverrideConfig`, validated at resolution (R4, R5, R6).
    delay_s
        The replacement delay for that feed, in seconds.  Required: a delay has
        exactly one overridable quantity, so an override with no delay would be
        an empty entry.
    """

    antenna: int
    feed: int
    delay_s: _StrictFiniteFloat


class DelayTermConfig(StrictFrozenModel):
    """The ``Kd`` term: instrumental delay offset (Section 20.5).

    ``Kd_p(nu) = diag(exp(-2 pi i nu tau_p0), exp(-2 pi i nu tau_p1))``.

    Parameters
    ----------
    delay_s
        The array-wide delay ``tau``, in seconds, applied to every feed without
        an override.  Defaults to zero so that a block written with overrides
        alone means "these antennas only"; a block that resolves to zero
        everywhere is rejected as the identity (R7).
    per_antenna
        Ordered per-``(antenna, feed)`` overrides.
    """

    delay_s: _StrictFiniteFloat = 0.0
    per_antenna: tuple[DelayOverrideConfig, ...] = ()


# --------------------------------------------------------------------------- X


class CrosshandOverrideConfig(StrictFrozenModel):
    """One per-antenna cross-hand override.

    Parameters
    ----------
    antenna
        The antenna **number** in the resolved instrument.  There is
        deliberately **no** ``feed`` field: ``X`` carries the *relative* phase
        between an antenna's two feeds, which is one number per antenna, and a
        feed index here would have to name the feed the phase is not on.  A
        second per-feed parameter would be exactly degenerate with ``G``
        (Section 20.4).
    phase_rad, delay_s
        Optional replacements for the array-wide defaults.  ``None`` keeps the
        default, so an override may change only one of the two.
    """

    antenna: int
    phase_rad: _StrictFiniteFloat | None = None
    delay_s: _StrictFiniteFloat | None = None

    @model_validator(mode="after")
    def require_override_content(self) -> Self:
        if self.phase_rad is None and self.delay_s is None:
            raise ValueError("must set at least one of 'phase_rad' or 'delay_s'")
        return self


class CrosshandTermConfig(StrictFrozenModel):
    """The ``X`` term: cross-hand phase and delay (Section 20.4).

    ``X_p(nu) = diag(1, exp(i (phi_x + 2 pi nu tau_x)))``.

    Parameters
    ----------
    phase_rad
        The frequency-constant relative phase between the two feeds.
    delay_s
        The cross-hand delay, in seconds: the frequency slope of that same
        relative phase.  Cross-hand phase and cross-hand delay are one term,
        because they are one matrix.
    per_antenna
        Ordered per-antenna overrides, keyed by antenna number alone.
    """

    phase_rad: _StrictFiniteFloat = 0.0
    delay_s: _StrictFiniteFloat = 0.0
    per_antenna: tuple[CrosshandOverrideConfig, ...] = ()


# --------------------------------------------------------------------------- D


class ExplicitLeakageModel(StrictFrozenModel):
    """Both feeds' leakage written out as complex numbers.

    Parameters
    ----------
    d0, d1
        The leakage of feed 1 into feed 0's chain and the converse.  Both
        default to zero so that a block naming only one feed is legal; a block
        that resolves to zero everywhere is rejected as the identity (R7).
    """

    kind: Literal["explicit"]
    d0: ComplexInput = 0.0
    d1: ComplexInput = 0.0


class IXRLeakageModel(StrictFrozenModel):
    """Both feeds' leakage from an intrinsic cross-polarization ratio.

    ``|d| = 1 / sqrt(IXR_lin)`` with ``IXR_lin = 10^(IXR_dB/10)``, equivalently
    ``IXR_dB = -20 log10 |d|`` -- so a *larger* IXR is a *smaller* leakage, and
    30 dB is about 3 per cent.  See
    :mod:`radiosim.core.jones.polarization_leakage` for the derivation from
    Carozzi & Woan (2011).

    Parameters
    ----------
    ixr_db
        The ratio in decibels.  Must be positive: ``IXR_dB = 0`` is a completely
        depolarizing receptor (``|d| = 1``) and a negative value would be a
        leakage larger than the direct path, which the first-order form does not
        describe.
    phase_rad
        The common phase of both feeds' leakage.  A per-``(antenna, feed)``
        override is the way to give the two feeds different phases.
    """

    kind: Literal["ixr"]
    ixr_db: Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0.0)]
    phase_rad: _StrictFiniteFloat = 0.0


class FrequencyPolynomialLeakageModel(StrictFrozenModel):
    """Both feeds' leakage as complex polynomials in normalized frequency.

    ``d(nu) = sum_k c_k x^k`` with ``x = (nu - nu_ref) / nu_scale``.  This is
    what the deleted frequency-dependent leakage class was; see
    ``docs/migration_guide.md``.

    Parameters
    ----------
    coefficients0, coefficients1
        Complex coefficients for each feed, lowest order first.
    reference_frequency_hz, scale_frequency_hz
        ``nu_ref`` and ``nu_scale``.  ``None`` resolves to the band centre and
        the half-bandwidth respectively, exactly as for ``B``, so that ``x``
        spans ``[-1, 1]`` across the observed band.
    """

    kind: Literal["frequency_polynomial"]
    coefficients0: tuple[ComplexInput, ...]
    coefficients1: tuple[ComplexInput, ...]
    reference_frequency_hz: _StrictPositiveFloat | None = None
    scale_frequency_hz: _StrictPositiveFloat | None = None

    @model_validator(mode="after")
    def require_coefficients(self) -> Self:
        if not self.coefficients0 or not self.coefficients1:
            raise ValueError(
                "coefficients0 and coefficients1 must each have at least one "
                "coefficient"
            )
        return self


LeakageModelConfig = Annotated[
    ExplicitLeakageModel | IXRLeakageModel | FrequencyPolynomialLeakageModel,
    Field(discriminator="kind"),
]


class ExplicitFeedLeakage(StrictFrozenModel):
    """One feed's leakage written out as a complex number."""

    kind: Literal["explicit"]
    d: ComplexInput


class IXRFeedLeakage(StrictFrozenModel):
    """One feed's leakage from an intrinsic cross-polarization ratio."""

    kind: Literal["ixr"]
    ixr_db: Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0.0)]
    phase_rad: _StrictFiniteFloat = 0.0


class FrequencyPolynomialFeedLeakage(StrictFrozenModel):
    """One feed's leakage as a complex polynomial in normalized frequency."""

    kind: Literal["frequency_polynomial"]
    coefficients: tuple[ComplexInput, ...]
    reference_frequency_hz: _StrictPositiveFloat | None = None
    scale_frequency_hz: _StrictPositiveFloat | None = None

    @model_validator(mode="after")
    def require_coefficients(self) -> Self:
        if not self.coefficients:
            raise ValueError("must have at least one coefficient")
        return self


FeedLeakageConfig = Annotated[
    ExplicitFeedLeakage | IXRFeedLeakage | FrequencyPolynomialFeedLeakage,
    Field(discriminator="kind"),
]


class LeakageOverrideConfig(StrictFrozenModel):
    """One per-``(antenna, feed)`` leakage override.

    Parameters
    ----------
    antenna, feed
        The same antenna-number and feed-index keying as
        :class:`GainOverrideConfig`, validated at resolution (R4, R5, R6).
    d_term
        The replacement leakage for **that one feed**, in the same three kinds
        the array-wide block offers.  The field is ``d_term`` and not
        ``d_terms`` because the shapes differ by exactly that: the array-wide
        block names both feeds (``d0``, ``d1``), while an override is keyed by a
        feed index and therefore names one.  An override that had to restate
        both feeds would make the index it is keyed by meaningless.
    """

    antenna: int
    feed: int
    d_term: FeedLeakageConfig


class LeakageTermConfig(StrictFrozenModel):
    """The ``D`` term: polarization leakage (Section 20.3).

    ``D_p(nu) = [[1, d_p0(nu)], [-d_p1(nu)^*, 1]]``.

    Parameters
    ----------
    d_terms
        The array-wide leakage model, naming both feeds.
    per_antenna
        Ordered per-``(antenna, feed)`` overrides, each naming one feed.
    """

    d_terms: LeakageModelConfig
    per_antenna: tuple[LeakageOverrideConfig, ...] = ()


# --------------------------------------------------------------------------- P


class ParallacticTermConfig(StrictFrozenModel):
    """The ``P`` term: parallactic angle and field rotation (Section 20.7).

    ``P_p(s, t) = R(eta_p psi_p(s, t) + nasmyth_p el)``.

    Parameters
    ----------
    enabled
        The whole configuration.  ``P`` is the only block that is a bare flag,
        because the parallactic angle has no free parameter: it is fully
        determined by the resolved instrument's latitude and mount types, the
        time grid, and the directions (Section 21.3).  The field is required
        rather than defaulted, because ``jones.P: {}`` would be a block that
        says nothing while looking like a decision.

        ``enabled: false`` parses and is then rejected at resolution as an
        identity (R7), with a message telling the reader to remove the block.
        Refusing it here instead would replace that sentence with a type error,
        and Section 21's "there is no ``enabled: false``" is a rule about runs
        rather than about parsing.

    Notes
    -----
    Section 21.2's accepted YAML also carried ``minimum_elevation_deg``, whose
    own comment said the directions it named were "already masked".  The 7F
    correction removed it: a field documented as having no effect is exactly the
    surface this tier exists to remove.  ``minimum_elevation_deg`` survives on
    ``T`` and ``Z``, where the mapping function genuinely diverges.
    """

    enabled: bool


# --------------------------------------------------------------------------- T


class ExplicitZenithDelay(StrictFrozenModel):
    """Both zenith delays written out in metres.

    Parameters
    ----------
    zenith_hydrostatic_delay_m
        ``ZHD``, the dry excess path at zenith.  About 2.31 m at sea level,
        mid latitude.
    zenith_wet_delay_m
        ``ZWD``, the wet excess path at zenith.  Typically 0.05-0.4 m.  There is
        no wet *model* on either variant, because every credible one needs
        humidity and temperature profiles RadioSim does not have (Section 4), and
        a model field with nothing behind it is the surface this tier removes.

    Both default to zero so a block naming one is legal; a block that resolves to
    zero everywhere with no opacity is rejected as the identity (R7).
    """

    kind: Literal["explicit"]
    zenith_hydrostatic_delay_m: _StrictFiniteFloat = 0.0
    zenith_wet_delay_m: _StrictFiniteFloat = 0.0


class SaastamoinenZenithDelay(StrictFrozenModel):
    """``ZHD`` from the Saastamoinen (1972) formula, ``ZWD`` written out.

    ``ZHD = 0.0022768 P_0 / (1 - 0.00266 cos(2 lat) - 0.00028 h_km)``: the
    surface pressure is configured, while the latitude and each antenna's height
    come from the resolved instrument, which is why they are not fields here.

    Parameters
    ----------
    surface_pressure_hpa
        ``P_0``, the surface pressure in hectopascals.  Positive.
    zenith_wet_delay_m
        ``ZWD`` in metres, exactly as on the explicit variant.
    """

    kind: Literal["saastamoinen"]
    surface_pressure_hpa: _StrictPositiveFloat = 1013.25
    zenith_wet_delay_m: _StrictFiniteFloat = 0.0


ZenithDelayConfig = Annotated[
    ExplicitZenithDelay | SaastamoinenZenithDelay,
    Field(discriminator="kind"),
]


class TroposphericOpacityConfig(StrictFrozenModel):
    """The optional opacity sub-block of ``jones.T``.

    Parameters
    ----------
    zenith_opacity
        The dimensionless zenith opacity ``tau_0``, defined on **power**.  The
        voltage attenuation is ``exp(-tau_0 / (2 sin el))``, so a baseline of two
        identical antennas at zenith is scaled by ``exp(-tau_0)`` in visibility
        amplitude (invariant I10).  A negative value is rejected at resolution
        (R10) rather than here, so the message names the physics: a negative
        opacity would amplify.

        RadioSim applies one number at every channel.  A frequency-dependent
        ``tau_0(nu)`` would need an atmospheric absorption model, which is the
        data ingestion Section 4 excludes; the field says what it does.
    """

    zenith_opacity: _StrictFiniteFloat


class TroposphereTermConfig(StrictFrozenModel):
    """The ``T`` term: tropospheric delay and opacity (Section 20.9).

    ``T_p(s, nu) = exp(-tau_0 / (2 sin el)) exp(-2 pi i nu tau_trop(s)) I2``.

    Parameters
    ----------
    zenith_delay
        The zenith delay pair, either written out or from Saastamoinen.
    mapping_function
        ``simple`` is the flat-atmosphere ``1 / sin(el)``; ``niell`` is the
        Niell (1996) three-term continued fraction with its published
        latitude-, season- and height-dependent coefficients.
    minimum_elevation_deg
        The elevation below which this run declines to evaluate the mapping
        function, which diverges at the horizon (R13).  Required, and not
        defaulted: where a model stops being trusted is a scientific decision,
        and a default would make it RadioSim's rather than the user's.  ``0``
        accepts every direction the horizon mask passes and is the explicit way
        to say "I accept the divergence".
    opacity
        The optional opacity sub-block.  Absent means a transparent atmosphere,
        which is the case in which ``T`` is unitary.
    """

    zenith_delay: ZenithDelayConfig
    mapping_function: Literal["simple", "niell"] = "niell"
    minimum_elevation_deg: Annotated[
        float, Field(strict=True, allow_inf_nan=False, ge=0.0, lt=90.0)
    ]
    opacity: TroposphericOpacityConfig | None = None


# --------------------------------------------------------------------------- Z


class ConstantTecModel(StrictFrozenModel):
    """One vertical electron column for the whole array.

    Produces an antenna-common but direction-varying phase: a single source at
    zenith changes no visibility, while a wide field does -- which is the
    discriminating test Section 20.8 names.

    Parameters
    ----------
    vertical_tec_tecu
        The vertical column in TECU (``1 TECU = 1e16`` electrons m^-2).  A
        negative column is rejected at resolution (R9).
    """

    kind: Literal["constant"]
    vertical_tec_tecu: _StrictFiniteFloat


class GradientTecModel(StrictFrozenModel):
    """A vertical column with a linear gradient across the array's sky.

    The column is evaluated at each antenna's own ionospheric **pierce point**,
    so two antennas looking at the same direction see different columns.  That is
    the minimal model with a closure-visible effect, and it is why this variant
    exists at all rather than being expressible as a constant.

    Parameters
    ----------
    vertical_tec_tecu
        The column at the array reference position, in TECU.
    gradient_east_tecu_per_km, gradient_north_tecu_per_km
        The gradient with the pierce point's topocentric East and North offset,
        in TECU per kilometre.  Signed.  At least one must be non-zero: a
        gradient model with no gradient is the constant model written the long
        way, and accepting it would let a document say ``gradient`` while meaning
        something else.
    """

    kind: Literal["gradient"]
    vertical_tec_tecu: _StrictFiniteFloat
    gradient_east_tecu_per_km: _StrictFiniteFloat = 0.0
    gradient_north_tecu_per_km: _StrictFiniteFloat = 0.0

    @model_validator(mode="after")
    def require_a_gradient(self) -> Self:
        if (
            self.gradient_east_tecu_per_km == 0.0
            and self.gradient_north_tecu_per_km == 0.0
        ):
            raise ValueError(
                "must set a non-zero 'gradient_east_tecu_per_km' or "
                "'gradient_north_tecu_per_km'; use kind: constant for a uniform "
                "screen"
            )
        return self


TecModelConfig = Annotated[
    ConstantTecModel | GradientTecModel,
    Field(discriminator="kind"),
]


class IonosphericFaradayOverrideConfig(StrictFrozenModel):
    """One per-antenna ionospheric rotation-measure override.

    Parameters
    ----------
    antenna
        The antenna **number** in the resolved instrument.  There is
        deliberately **no** ``feed`` field: an ionospheric rotation measure is a
        property of the line of sight, not of a receptor, and a feed index here
        would name something the physics does not have.
    rotation_measure_rad_m2
        The replacement rotation measure for that antenna, in rad m^-2.
    """

    antenna: int
    rotation_measure_rad_m2: _StrictFiniteFloat


class IonosphericFaradayConfig(StrictFrozenModel):
    """The optional Faraday sub-block of ``jones.Z``.

    Parameters
    ----------
    rotation_measure_rad_m2
        The array-wide ionospheric rotation measure ``RM_ion``, in rad m^-2.
        Configured directly and **not** derived from the electron column and a
        geomagnetic field model: RadioSim has no magnetic-field model and
        Section 4 forbids adding the data ingestion for one.  ``RMextract`` is
        the tool that produces the number to write here.
    per_antenna
        Ordered per-antenna overrides, keyed by antenna number alone.

    Notes
    -----
    This is *ionospheric* rotation only.  A source's intrinsic rotation measure
    belongs to the sky model, is applied there, and composes with this one; the
    two live in different objects, so they cannot be configured twice by accident
    (Section 11, defect D18, invariant I8).
    """

    rotation_measure_rad_m2: _StrictFiniteFloat = 0.0
    per_antenna: tuple[IonosphericFaradayOverrideConfig, ...] = ()


class IonosphereTermConfig(StrictFrozenModel):
    """The ``Z`` term: ionospheric dispersive phase and Faraday rotation.

    ``Z_p(s, nu) = exp(-2 pi i k_TEC sTEC / nu) F(RM_ion lambda^2)``
    (Section 20.8).

    Parameters
    ----------
    tec
        The vertical-TEC model, ``constant`` or ``gradient``.
    shell_height_km
        The thin shell's height above the surface, in kilometres.  350 km is the
        conventional value and the default.
    minimum_elevation_deg
        The elevation below which this run declines to evaluate the thin-shell
        mapping (R13).  Required for the same reason as on ``T``.  Unlike ``T``'s
        ``1 / sin(el)``, the slant factor is *bounded* at the horizon -- about
        3.13 for a 350 km shell -- so what fails at low elevation is the
        thin-shell approximation rather than the arithmetic.
    faraday
        The optional Faraday sub-block.  Absent means dispersive phase only, in
        which case ``Z`` is a scalar phase and commutes with everything.
    """

    tec: TecModelConfig
    shell_height_km: _StrictPositiveFloat = 350.0
    minimum_elevation_deg: Annotated[
        float, Field(strict=True, allow_inf_nan=False, ge=0.0, lt=90.0)
    ]
    faraday: IonosphericFaradayConfig | None = None


# ------------------------------------------------------------------------- M


class BaselineErrorOverrideConfig(StrictFrozenModel):
    """One baseline's closure error, keyed by its ordered antenna-number pair.

    Parameters
    ----------
    antennas
        The ordered pair, exactly as the resolved baseline selection carries it
        (``ant1 <= ant2``).  The reversed pair is a *different* key and is
        rejected by R14 rather than silently read as the conjugate baseline: a
        configuration that names ``[1, 0]`` has not said whether it means the
        conjugate error, and inventing an answer is what this tier removes.
    matrix
        The ``2x2`` complex error, as ``[[m00, m01], [m10, m11]]`` with each
        entry an ``[re, im]`` pair or a bare real number.  Each entry multiplies
        the correlation of the same name, so the value that changes nothing is
        ``1`` in **every** entry -- not the identity matrix, whose off-diagonal
        zeros would null both cross-hands.
    """

    antennas: tuple[
        Annotated[int, Field(strict=True, ge=0)],
        Annotated[int, Field(strict=True, ge=0)],
    ]
    matrix: ComplexMatrix2x2


class BaselineErrorTermConfig(StrictFrozenModel):
    """The ``M`` term: per-baseline multiplicative closure error (Section 20.10).

    ``V_pq -> M_pq (*) V_pq``, a Hadamard product on the finished ``2x2``
    correlation matrix rather than a factor in the antenna chain -- which is why
    ``M`` breaks closure and every per-antenna term does not (invariant I11).

    Parameters
    ----------
    matrix
        The optional array-wide default, applied to every selected baseline.
    per_baseline
        Optional per-baseline overrides, each beating the default for the pair
        it names.  A baseline named by neither carries ``[[1, 1], [1, 1]]`` and
        is untouched.

    Notes
    -----
    There is no "enabled" flag and no way to write an ``M`` that does nothing: a
    block whose every resolved entry is ``1`` -- including a block that
    configures neither field -- is rejected at resolution as an identity (R7).
    """

    matrix: ComplexMatrix2x2 | None = None
    per_baseline: tuple[BaselineErrorOverrideConfig, ...] = ()


# ------------------------------------------------------------------------- Q


class SmearingTermConfig(StrictFrozenModel):
    """The ``Q`` term: time and bandwidth smearing (Section 20.11).

    ``Q_pqs = sinc(pi dnu tau_res) sinc(pi dt nu_f)``.

    Parameters
    ----------
    bandwidth_smearing, time_smearing
        Which envelopes are active.  Both are **required** and neither has a
        default, for the same reason ``P.enabled`` is required: which mechanism
        a run models is a scientific decision, and a default would silently make
        it RadioSim's.  Both ``false`` parses and is then rejected at resolution
        (R16).

    Notes
    -----
    There is deliberately no ``channel_width_hz`` and no ``integration_time_s``
    here.  ``dnu`` is the run's own resolved per-channel width and ``dt`` its
    resolved per-sample integration time; a field for either would be a second,
    contradictable statement of something the observation configuration already
    fixes, and a simulation that smeared over a bandwidth different from the one
    it reports in its own result would be exactly the fabrication this tier
    exists to remove (Section 20.11, and Section 41's Q6 resolution).
    """

    bandwidth_smearing: bool
    time_smearing: bool


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
    Rc: CableReflectionTermConfig | None = None
    Kd: DelayTermConfig | None = None
    X: CrosshandTermConfig | None = None
    D: LeakageTermConfig | None = None
    P: ParallacticTermConfig | None = None
    T: TroposphereTermConfig | None = None
    Z: IonosphereTermConfig | None = None
    M: BaselineErrorTermConfig | None = None
    Q: SmearingTermConfig | None = None

    @property
    def configured_terms(self) -> tuple[str, ...]:
        """Return the configured term letters, in canonical chain order.

        The nine chain letters first, in the order they compose, then the two
        baseline-dependent ones, which are not in the chain at all.
        """
        return tuple(
            letter
            for letter in JONES_TERM_LETTERS
            if getattr(self, letter, None) is not None
        )
