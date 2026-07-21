"""Public beam error hierarchy for canonical Tier 3 beam processing."""


class BeamError(RuntimeError):
    """Base class for beam assignment, loading, and evaluation failures."""


class BeamAssignmentError(BeamError):
    """Base class for canonical beam-assignment failures."""


class UnknownBeamAntennaError(BeamAssignmentError):
    """An authored assignment does not match a canonical antenna."""


class DuplicateBeamAssignmentError(BeamAssignmentError):
    """Two authored entries resolve to the same canonical antenna."""


class IncompleteBeamAssignmentError(BeamAssignmentError):
    """Explicit assignments do not cover every canonical antenna."""


class InconsistentBeamAssignmentError(BeamAssignmentError):
    """Resolved assignment state and a later runtime lookup disagree."""


class BeamLoadError(BeamError):
    """Base class for beam dependency, file, and load-validation failures."""


class BeamDependencyError(BeamLoadError):
    """A dependency required to load a beam is unavailable."""


class BeamFileReadError(BeamLoadError):
    """A beam file cannot be read as required."""


class BeamFileChangedError(BeamLoadError):
    """A beam file changed during an atomic load."""


class UnsupportedBeamMetadataError(BeamLoadError):
    """Base class for unsupported BeamFITS metadata."""


class UnsupportedBeamTypeError(UnsupportedBeamMetadataError):
    """The beam or antenna type is unsupported."""


class UnsupportedBeamFeedError(UnsupportedBeamMetadataError):
    """The beam feed, order, orientation, or mount is unsupported."""


class UnsupportedBeamBasisError(UnsupportedBeamMetadataError):
    """The beam basis or Jones structure is unsupported."""


class UnsupportedBeamCoordinateError(UnsupportedBeamMetadataError):
    """The beam coordinate system or native grid is unsupported."""


class BeamNormalizationError(BeamLoadError):
    """Beam normalization does not satisfy the accepted contract."""


class UnsupportedBeamPrecisionError(BeamLoadError):
    """The requested precision exceeds accepted beam information width."""


class BeamSamplingDerivationError(BeamLoadError):
    """A beam sampling requirement cannot be derived."""


class BeamEvaluationError(BeamError):
    """Base class for beam evaluation failures."""


class BeamFrequencyDomainError(BeamEvaluationError):
    """A requested frequency is outside the beam domain."""


class BeamAngularDomainError(BeamEvaluationError):
    """A requested direction is outside the beam domain."""


class NonFiniteBeamResponseError(BeamEvaluationError):
    """A native or evaluated beam response is non-finite."""


class BeamDisplayNormalizationError(BeamEvaluationError):
    """A display beam cannot be normalized over its requested domain."""


__all__ = [
    "BeamError",
    "BeamAssignmentError",
    "UnknownBeamAntennaError",
    "DuplicateBeamAssignmentError",
    "IncompleteBeamAssignmentError",
    "InconsistentBeamAssignmentError",
    "BeamLoadError",
    "BeamDependencyError",
    "BeamFileReadError",
    "BeamFileChangedError",
    "UnsupportedBeamMetadataError",
    "UnsupportedBeamTypeError",
    "UnsupportedBeamFeedError",
    "UnsupportedBeamBasisError",
    "UnsupportedBeamCoordinateError",
    "BeamNormalizationError",
    "UnsupportedBeamPrecisionError",
    "BeamSamplingDerivationError",
    "BeamEvaluationError",
    "BeamFrequencyDomainError",
    "BeamAngularDomainError",
    "NonFiniteBeamResponseError",
    "BeamDisplayNormalizationError",
]
