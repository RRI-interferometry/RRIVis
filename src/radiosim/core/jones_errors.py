"""The typed failures of Jones configuration, resolution, and evaluation.

``Tier7JonesSciencePlan.md`` Section 26.  The shape follows
``core/receptor.py``'s Tier 5 taxonomy exactly: one package-level root that a
caller can catch wholesale, and one leaf per *kind* of mistake, so a message can
be matched by type instead of by substring.

::

    JonesError(RuntimeError)
    |-- InvalidJonesConfigError        malformed or physically invalid values
    |   |-- IdentityJonesTermError     R7: a term configured to be exactly I2
    |   `-- UnsupportedMountTypeError  R12, R15: a mount P cannot model
    |-- JonesAssignmentError           R4, R14: an unknown antenna or baseline
    `-- JonesEvaluationError           a term produced a bad array

Why ``JonesEvaluationError`` is not decorative
----------------------------------------------
Every term's batch evaluation is shape- and finiteness-checked once per
``(time, frequency)`` step (Section 26).  The check is a full
``isfinite().all()`` on the block, not a spot check of the first and last
element: one reduction per antenna per step is negligible beside the
contraction, and a term that produces ``nan`` from a *legal* configuration is a
defect that must surface at the term rather than as a silent ``nan`` in the
output cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "IdentityJonesTermError",
    "InvalidJonesConfigError",
    "JonesAssignmentError",
    "JonesError",
    "JonesEvaluationError",
    "UnsupportedMountTypeError",
    "require_finite_jones_block",
]


class JonesError(RuntimeError):
    """Base class for every Jones configuration, resolution, or evaluation failure."""


class InvalidJonesConfigError(JonesError):
    """A ``jones:`` block is malformed, or its values are physically invalid.

    Raised for the rejections that need resolved values rather than a schema:
    R2 (an empty section), R5/R6 (duplicate or out-of-range feed indices),
    R8-R11, R13 and R16.  Rejections a strict Pydantic model can make on its own
    (an unknown key, a wrong type, an unknown ``kind``) are raised by Pydantic
    itself and rendered by the Tier 1 renderer -- this class is for what the
    schema cannot see.
    """


class IdentityJonesTermError(InvalidJonesConfigError):
    """R7: a configured term resolves to exactly the identity for every input.

    A term that cannot change the visibilities is indistinguishable from no term
    at all, which is the ``SCI-001`` defect Tier 7 exists to remove.  Configuring
    one is therefore an error and not a silent no-op: the fix is to remove the
    section, and the message says so.
    """


class UnsupportedMountTypeError(InvalidJonesConfigError):
    """R12/R15: an antenna's mount type and the ``P`` configuration disagree.

    Owned by Tier 7F, which implements ``P``; declared here so the taxonomy is
    complete in one place and later slices add messages rather than classes.
    """


class JonesAssignmentError(JonesError):
    """R4/R14: a ``jones:`` block names an antenna or baseline that does not exist.

    Distinct from :class:`InvalidJonesConfigError` because the value itself is
    well formed -- antenna 12 is a perfectly good antenna number -- and what is
    wrong is that the *resolved instrument* has no such antenna.  The two are
    fixed differently, so they are caught differently.
    """


class JonesEvaluationError(JonesError):
    """A term returned a non-finite or wrong-shaped block at evaluation time.

    This is an internal-consistency failure, not a user mistake: a legal
    configuration that produces ``nan`` is a defect in the term.  It is raised
    with the term's name so that the defect is attributed at the term rather
    than discovered later as a ``nan`` in the output cube.
    """


def require_finite_jones_block(term_name: str, block: Any) -> Any:
    """Return ``block`` after checking its shape and finiteness (Section 26).

    The check is a full ``isfinite().all()`` over the block, not a spot check of
    its first and last element: that is one reduction per antenna per
    ``(time, frequency)`` step, which is negligible beside the contraction, and
    a partial check is exactly how a ``nan`` reaches the output cube unnoticed.

    The block is a host ``numpy`` array here by construction -- every term that
    calls this builds its matrices on the host before handing them to the
    backend -- so the reduction never forces a device synchronization and never
    branches on a traced value (Section 17.2).

    Parameters
    ----------
    term_name : str
        The term's letter, used verbatim in the failure message so the defect is
        attributed at the term.
    block : ndarray
        A host complex array of shape ``(n, 2, 2)``.

    Returns
    -------
    ndarray
        ``block`` unchanged, so the check composes into a return statement.

    Raises
    ------
    JonesEvaluationError
        The block has the wrong shape or contains a non-finite entry.
    """
    array = np.asarray(block)
    if array.ndim != 3 or array.shape[1:] != (2, 2):
        raise JonesEvaluationError(
            f"Jones term {term_name!r} produced a block of shape {array.shape}; "
            "every term must return (n_dir, 2, 2) or (1, 2, 2)."
        )
    if not bool(np.isfinite(array).all()):
        raise JonesEvaluationError(
            f"Jones term {term_name!r} produced a non-finite Jones matrix from a "
            "legal configuration; this is a defect in the term, not in the "
            "configuration."
        )
    return block
