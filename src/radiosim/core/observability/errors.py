"""Public errors for observability planning, rendering, and persistence."""


class ObservabilityError(RuntimeError):
    """Base class for every public observability failure."""


class InvalidObservabilityReferenceError(ObservabilityError):
    """A requested observability reference antenna is invalid."""


class InvalidObservabilityContextError(ObservabilityError):
    """An observability time, channel, location, or option is invalid."""


class ObservabilitySkyUnavailableError(ObservabilityError):
    """A requested prepared sky payload is unavailable."""


class UnsupportedObservabilitySemanticsError(ObservabilityError):
    """A removed or unsupported multi-beam observability mode was requested."""


class ObservabilityRenderError(ObservabilityError):
    """Renderer input validation or layout construction failed."""


class ObservabilityOutputError(ObservabilityError):
    """An output target or atomic publication operation failed."""


class ObservabilityOutputCollisionError(ObservabilityOutputError):
    """The output target already exists and overwrite was not requested."""


class ObservabilityBrowserError(ObservabilityOutputError):
    """An explicit browser action failed after successful publication."""


__all__ = [
    "ObservabilityError",
    "InvalidObservabilityReferenceError",
    "InvalidObservabilityContextError",
    "ObservabilitySkyUnavailableError",
    "UnsupportedObservabilitySemanticsError",
    "ObservabilityRenderError",
    "ObservabilityOutputError",
    "ObservabilityOutputCollisionError",
    "ObservabilityBrowserError",
]
