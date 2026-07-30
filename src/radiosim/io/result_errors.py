"""Typed failures shared by canonical result I/O implementations."""

from __future__ import annotations

from pathlib import Path


class ResultIOError(RuntimeError):
    """Base class for canonical result I/O failures."""


class UnsupportedResultFormatError(ResultIOError):
    """The requested result format is not supported."""


class OptionalResultDependencyError(ResultIOError):
    """A requested result operation is missing an optional dependency."""


class OutputPathError(ResultIOError):
    """An output path violates the requested format contract."""


class OutputCollisionError(ResultIOError):
    """An existing or concurrently created output prevents publication."""


class UnsafeOutputDirectoryError(OutputCollisionError):
    """An output directory or ancestor is unsafe for publication."""


class OverwriteRefusedError(OutputCollisionError):
    """Publication would overwrite an existing target without permission."""


class AtomicWriteError(ResultIOError):
    """An atomic output transaction failed."""


class AtomicWriteUnsupportedError(AtomicWriteError):
    """The host cannot provide the required atomic primitive."""


class PartialCleanupError(AtomicWriteError):
    """A failed transaction left one exact temporary artifact behind."""

    def __init__(self, residual_path: str | Path) -> None:
        self.residual_path = Path(residual_path)
        super().__init__(
            "atomic result cleanup failed; residual temporary path: "
            f"{self.residual_path}"
        )


class SummaryContractError(ResultIOError):
    """A result-summary request violates its bounded schema contract."""


class FormatRepresentationError(ResultIOError):
    """A canonical result cannot be represented by the requested format."""


class UnsafeResultInputError(ResultIOError):
    """An input file violates the safe canonical result contract."""


class UnsupportedSchemaVersionError(UnsafeResultInputError):
    """A versioned result uses an unsupported schema version."""

    GUIDANCE = (
        "Tier 5 replaced radiosim.visibility 1.0.0 with 2.0.0, which records the "
        "polarization basis and the resolved receptor set. There is no upgrade "
        "path by design: re-run the simulation to write a 2.0.0 file."
    )

    def __init__(self, version: object) -> None:
        if type(version) is str:
            safe = "".join(
                character if character.isprintable() and character != "\x00" else "?"
                for character in version
            )[:128]
        else:
            safe = f"<{type(version).__name__}>"
        self.version = safe
        super().__init__(
            f"unsupported radiosim.visibility schema version: {safe}. {self.GUIDANCE}"
        )


class LegacyHDF5Error(UnsafeResultInputError):
    """An unversioned pre-Tier-4 HDF5 file was supplied."""

    GUIDANCE = (
        "Legacy unversioned RadioSim HDF5 is not accepted because baseline names "
        "were parsed unsafely and scientific fields were incomplete. Re-run the "
        "simulation with Tier 4 or convert a trusted file in an isolated pre-Tier-4 "
        "environment."
    )

    def __init__(self) -> None:
        super().__init__(self.GUIDANCE)


__all__ = [
    "AtomicWriteError",
    "AtomicWriteUnsupportedError",
    "FormatRepresentationError",
    "LegacyHDF5Error",
    "OptionalResultDependencyError",
    "OutputCollisionError",
    "OutputPathError",
    "OverwriteRefusedError",
    "PartialCleanupError",
    "ResultIOError",
    "SummaryContractError",
    "UnsafeOutputDirectoryError",
    "UnsafeResultInputError",
    "UnsupportedResultFormatError",
    "UnsupportedSchemaVersionError",
]
