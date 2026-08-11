"""Private host-side source-axis bucketing for compiled visibility solvers.

The public solvers always use :data:`PRODUCTION_SOURCE_BUCKET_POLICY`.  The
identity policy exists only so tests and retained performance evidence can run
the same complete private solvers without source padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

PRODUCTION_SOURCE_BUCKET_POLICY = "pow2_compiled_v1"
IDENTITY_SOURCE_BUCKET_POLICY = "identity_reference_v1"

_SourceBucketPolicy = Literal["pow2_compiled_v1", "identity_reference_v1"]
_SOURCE_BUCKET_POLICIES = frozenset(
    {PRODUCTION_SOURCE_BUCKET_POLICY, IDENTITY_SOURCE_BUCKET_POLICY}
)


@dataclass(frozen=True)
class _SourceBucketPlan:
    """Observable logical and scheduled source counts for one time step."""

    logical_count: int
    kernel_count: int

    @property
    def padding_count(self) -> int:
        """Number of exact-zero dummy sources appended to the logical batch."""
        return self.kernel_count - self.logical_count


def _require_source_bucket_policy(policy: str) -> _SourceBucketPolicy:
    """Validate and narrow the private solver/evidence policy."""
    if type(policy) is not str or policy not in _SOURCE_BUCKET_POLICIES:
        choices = ", ".join(sorted(_SOURCE_BUCKET_POLICIES))
        raise ValueError(f"source bucket policy must be one of: {choices}")
    return cast(_SourceBucketPolicy, policy)


def _resolve_source_bucket(
    logical_count: int,
    *,
    supports_compilation: bool,
    policy: str,
) -> _SourceBucketPlan:
    """Resolve the private P-b source bucket for one nonempty logical batch."""
    if type(logical_count) is not int or logical_count <= 0:
        raise ValueError("logical_count must be positive")
    if type(supports_compilation) is not bool:
        raise TypeError("supports_compilation must be an exact bool")
    resolved_policy = _require_source_bucket_policy(policy)

    kernel_count = logical_count
    if supports_compilation and resolved_policy == PRODUCTION_SOURCE_BUCKET_POLICY:
        kernel_count = 1 << (logical_count - 1).bit_length()
    return _SourceBucketPlan(
        logical_count=logical_count,
        kernel_count=kernel_count,
    )


def _require_source_axis(
    values: np.ndarray,
    plan: _SourceBucketPlan,
    *,
    axis: int,
) -> int:
    """Return a normalized source axis after validating its logical extent."""
    if values.ndim == 0:
        raise ValueError("source-bearing arrays must have at least one dimension")
    normalized_axis = axis + values.ndim if axis < 0 else axis
    if normalized_axis < 0 or normalized_axis >= values.ndim:
        raise ValueError(f"axis {axis} is out of bounds for {values.ndim} dimensions")
    if values.shape[normalized_axis] != plan.logical_count:
        raise ValueError(
            "source axis has "
            f"{values.shape[normalized_axis]} entries but the bucket plan has "
            f"{plan.logical_count} logical sources"
        )
    return normalized_axis


def _pad_host_zeros(
    values: np.ndarray,
    plan: _SourceBucketPlan,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Append exact-zero dummy signal along one host array's source axis."""
    array = np.asarray(values)
    source_axis = _require_source_axis(array, plan, axis=axis)
    if plan.padding_count == 0:
        return array

    padding_shape = list(array.shape)
    padding_shape[source_axis] = plan.padding_count
    padding = np.zeros(tuple(padding_shape), dtype=array.dtype)
    return np.concatenate((array, padding), axis=source_axis)


def _pad_host_repeated(
    values: np.ndarray,
    plan: _SourceBucketPlan,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Append copies of the first finite, domain-safe logical metadata row."""
    array = np.asarray(values)
    source_axis = _require_source_axis(array, plan, axis=axis)
    if plan.padding_count == 0:
        return array

    first = np.take(array, [0], axis=source_axis)
    if np.issubdtype(first.dtype, np.number) and not np.all(np.isfinite(first)):
        raise ValueError("repeated dummy metadata must be finite")
    repeated = np.repeat(first, plan.padding_count, axis=source_axis)
    return np.concatenate((array, repeated), axis=source_axis)


def _pad_reference_frequencies(
    values: np.ndarray,
    plan: _SourceBucketPlan,
    *,
    fallback_hz: float,
) -> np.ndarray:
    """Append one positive finite reference frequency for every dummy source."""
    array = np.asarray(values)
    _require_source_axis(array, plan, axis=0)
    if array.ndim != 1:
        raise ValueError("reference frequencies must be one-dimensional")
    if plan.padding_count == 0:
        return array

    valid = np.flatnonzero(np.isfinite(array) & (array > 0.0))
    if valid.size:
        dummy_frequency = array[int(valid[0])]
    else:
        if not np.isfinite(fallback_hz) or fallback_hz <= 0.0:
            raise ValueError("fallback_hz must be positive and finite")
        dummy_frequency = np.asarray(fallback_hz, dtype=array.dtype)

    padding = np.full(
        plan.padding_count,
        dummy_frequency,
        dtype=array.dtype,
    )
    return np.concatenate((array, padding))
