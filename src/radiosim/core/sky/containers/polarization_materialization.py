"""Typed polarization records and retained evidence; consumers validate their truth.

Constructing these values alone does not complete a normalization operation.
The canonical identity factory derives them from actual stored owner values.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import BrightnessConversion
from .point import TangentPolarizationFrame

__all__ = [
    "PolarizationOperation",
    "PolarizationMaterialization",
    "PolarizationMaterializationEvidence",
]


@dataclass(frozen=True, slots=True)
class PolarizationOperation:
    """Unvalidated operation values; construction does not execute an operation.

    Parameters
    ----------
    kind : str
        Operation identifier, such as ``identity``.
    input_sha256, output_sha256 : str
        Ordered input and output payload digests.
    parameters_sha256 : str
        Digest resolving to retained operation-parameter bytes.
    """

    kind: str
    input_sha256: str
    output_sha256: str
    parameters_sha256: str

    def as_mapping(self) -> dict[str, str]:
        """Present fields without validation or operation execution.

        Returns
        -------
        dict of str to str
            Fresh four-field mapping of the retained operation values.
        """
        return {
            "kind": self.kind,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "parameters_sha256": self.parameters_sha256,
        }


@dataclass(frozen=True, slots=True)
class PolarizationMaterialization:
    """Twelve scientific fields, validated by an actual-value consumer.

    Frozen construction alone does not attest normalization or provenance.

    Parameters
    ----------
    schema_version : str
        Materialization record schema identifier.
    component_kind : str
        Payload representation, such as ``healpix``.
    source_profile, declaration_origin : str
        Declared polarization convention and its origin.
    declaration_digest : str
        Digest resolving to the retained declaration bytes.
    source_frame, output_frame : str
        Input and output celestial coordinate frames.
    input_payload_sha256, output_payload_sha256 : str
        Actual operation-chain endpoint payload digests.
    operations : tuple of PolarizationOperation
        Operations in execution order; identity is an explicit operation.
    parent_materialization_ids : tuple of str
        Ordered parent record identifiers; empty for a fresh input.
    materialization_id : str
        Domain-separated digest of the preceding eleven fields.
    """

    schema_version: str
    component_kind: str
    source_profile: str
    declaration_origin: str
    declaration_digest: str
    source_frame: str
    output_frame: str
    input_payload_sha256: str
    output_payload_sha256: str
    operations: tuple[PolarizationOperation, ...]
    parent_materialization_ids: tuple[str, ...]
    materialization_id: str

    def as_mapping(self) -> dict[str, object]:
        """Present fields without validation or operation execution.

        Returns
        -------
        dict of str to object
            Fresh twelve-field mapping, operation mappings and parent list.
        """
        return {
            "schema_version": self.schema_version,
            "component_kind": self.component_kind,
            "source_profile": self.source_profile,
            "declaration_origin": self.declaration_origin,
            "declaration_digest": self.declaration_digest,
            "source_frame": self.source_frame,
            "output_frame": self.output_frame,
            "input_payload_sha256": self.input_payload_sha256,
            "output_payload_sha256": self.output_payload_sha256,
            "operations": [operation.as_mapping() for operation in self.operations],
            "parent_materialization_ids": list(self.parent_materialization_ids),
            "materialization_id": self.materialization_id,
        }


@dataclass(frozen=True, slots=True)
class PolarizationMaterializationEvidence:
    """Unvalidated record, immutable sidecars and explicit brightness context.

    The factory derives evidence from actual values; the consumer checks it.

    Parameters
    ----------
    record : PolarizationMaterialization
        Twelve-field scientific record, separate from these evidence fields.
    tangent_frame : TangentPolarizationFrame or None
        Canonical declared frame; null is allowed only without nonzero Q/U.
    declaration_json : bytes
        Exact canonical declaration bytes resolving the declaration digest.
    identity_parameters_json : bytes
        Exact identity-operation parameter bytes resolving its digest.
    payload_metadata_json : bytes
        Retained payload metadata resolving the identity parameters.
    brightness_conversion : BrightnessConversion
        Exact enclosing enum context, joined to the actual owner by consumers.
        It adds no field to the serialized scientific record.
    """

    record: PolarizationMaterialization
    tangent_frame: TangentPolarizationFrame | None
    declaration_json: bytes
    identity_parameters_json: bytes
    payload_metadata_json: bytes
    brightness_conversion: BrightnessConversion
