"""Strict Tier 2D instrument source loaders and dependency adapters.

Returned values contain only immutable built-in scalar and tuple facts.  No
caller-owned array, FITS table, pyuvdata Telescope, or EarthLocation crosses the
source boundary.

Offline known-telescope calls are serialized by a RadioSim-owned re-entrant lock.
Unrelated third-party Astropy calls do not share this lock and may observe the
temporary setting; this is the narrow process-global limitation of Astropy's
configuration API.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, cast

from radiosim.core.instrument_resolution import (
    AntennaIdentifierError,
    DiameterResolutionError,
    EmptyInstrumentError,
    InstrumentFormatError,
    InstrumentResolutionError,
    InstrumentSourceError,
    OptionalInstrumentDependencyError,
    TelescopeNotFoundError,
    earth_location_facts,
    normalize_antenna_number,
    normalize_beam_id,
    normalize_identity,
    normalize_mount,
    normalize_position,
    normalize_source_diameter,
    validate_unique_antennas,
)
from radiosim.io.instrument_config import (
    KnownTelescopeSourceConfig,
    LayoutFileSourceConfig,
)

SourceFormat = Literal[
    "radiosim",
    "casa_loc",
    "measurement_set",
    "uvfits",
    "mwa_metafits",
]
PositionFrame = Literal["enu", "relative_ecef"]
_KNOWN_ABSENCE_TEXT = "not in astropy_sites or known_telescopes_dict"
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+\Z")
_KNOWN_TELESCOPE_LOCK = threading.RLock()


class ModuleImporter(Protocol):
    """Typed lazy-import seam."""

    def __call__(self, name: str) -> object: ...


class DatasetTelescopeLoader(Protocol):
    """Return a dataset's extracted telescope metadata only."""

    def __call__(self, path: Path, source_format: str) -> object: ...


class KnownTelescopeLoader(Protocol):
    """Attempt one known-telescope lookup without prior enumeration."""

    def __call__(self, name: str) -> object: ...


class InternetConfiguration(Protocol):
    """Narrow Astropy temporary-configuration surface."""

    allow_internet: object

    def set_temp(
        self, attribute: str, value: object
    ) -> AbstractContextManager[None]: ...


class _TableData(Protocol):
    names: Sequence[str] | None

    def __len__(self) -> int: ...

    def __getitem__(self, key: str) -> Sequence[object]: ...


class _TableHdu(Protocol):
    data: _TableData | None


class _HduList(Protocol):
    def __enter__(self) -> _HduList: ...

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> object: ...

    def __contains__(self, name: object) -> bool: ...

    def __getitem__(self, name: str) -> _TableHdu: ...


class _FitsModule(Protocol):
    def open(self, path: Path, *, memmap: bool) -> _HduList: ...


class _UVData(Protocol):
    telescope: object

    def read(self, path: Path, *, read_data: bool) -> object: ...


class _UVDataFactory(Protocol):
    def __call__(self) -> _UVData: ...


class _PyuvdataModule(Protocol):
    UVData: _UVDataFactory
    Telescope: object
    __version__: object


class _TelescopeType(Protocol):
    def from_known_telescopes(self, name: str) -> object: ...


class _TelescopeMetadata(Protocol):
    antenna_names: object
    antenna_numbers: object
    antenna_positions: object
    location: object
    antenna_diameters: object
    mount_type: object
    name: object


class _AstropyDataModule(Protocol):
    conf: InternetConfiguration


@dataclass(frozen=True, slots=True)
class SourceLocationFacts:
    """Copied Earth-location metadata with exact source ITRS coordinates."""

    longitude_deg: float
    latitude_deg: float
    height_m: float
    itrs_xyz_m: tuple[float, float, float]
    reference: str


@dataclass(frozen=True, slots=True)
class SourceAntenna:
    """One validated source record in its declared source frame."""

    number: int
    name: str
    position_m: tuple[float, float, float]
    source_diameter_m: float | None
    mount_type: str | None
    beam_id: int | str | None
    number_generated: bool
    name_generated: bool
    source_record: str


@dataclass(frozen=True, slots=True)
class LoadedInstrumentSource:
    """Owned source facts ready for source-independent coordinate resolution."""

    source_kind: Literal["layout_file", "known_telescope"]
    source_reference: str
    source_format: SourceFormat | None
    position_frame: PositionFrame
    explicit_telescope_name: str | None
    embedded_telescope_name: str | None
    embedded_location: SourceLocationFacts | None
    antennas: tuple[SourceAntenna, ...]
    registry_policy: Literal["offline", "allow_network"] | None
    pyuvdata_version: str | None
    source_sha256: str | None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise InstrumentSourceError(
            f"{path}: selected instrument file could not be read"
        ) from error
    return digest.hexdigest()


def _require_path(path: Path, *, source_format: SourceFormat) -> Path:
    resolved = path.resolve(strict=False)
    expected_directory = source_format == "measurement_set"
    if expected_directory and not resolved.is_dir():
        raise InstrumentSourceError(
            f"{resolved}: measurement_set source must be an existing directory"
        )
    if not expected_directory and not resolved.is_file():
        raise InstrumentSourceError(
            f"{resolved}: {source_format} source must be an existing regular file"
        )
    return resolved


def _parse_integer_text(value: str, *, reference: str, label: str) -> int:
    if _INTEGER_TEXT.fullmatch(value) is None:
        raise AntennaIdentifierError(f"{reference}: {label} must be an integer")
    try:
        parsed = int(value)
    except ValueError as error:  # defensive against interpreter conversion limits
        raise AntennaIdentifierError(
            f"{reference}: {label} is not a supported integer"
        ) from error
    return normalize_antenna_number(parsed, reference=reference)


def _parse_float_text(
    value: str,
    *,
    reference: str,
    label: str,
) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise InstrumentFormatError(
            f"{reference}: {label} must be a finite number"
        ) from error
    try:
        return normalize_position((parsed, 0.0, 0.0), reference=reference)[0]
    except Exception as error:
        if isinstance(error, InstrumentFormatError):
            raise
        raise InstrumentFormatError(
            f"{reference}: {label} must be a finite number"
        ) from error


def _text_lines(path: Path, *, source_format: SourceFormat) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise InstrumentSourceError(
            f"{path}: {source_format} source must be readable UTF-8 text"
        ) from error


def _sorted_validated(
    antennas: Iterable[SourceAntenna], *, source_reference: str
) -> tuple[SourceAntenna, ...]:
    copied = tuple(antennas)
    validate_unique_antennas(copied, source_reference=source_reference)
    return tuple(sorted(copied, key=lambda antenna: antenna.number))


def _load_radiosim(path: Path, telescope_name: str) -> LoadedInstrumentSource:
    lines = _text_lines(path, source_format="radiosim")
    header_index: int | None = None
    header: tuple[str, ...] | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header_index = index
        header = tuple(stripped.split())
        break
    if header is None or header_index is None:
        raise EmptyInstrumentError(f"{path}: radiosim source contains zero antennas")
    accepted = {
        ("Name", "Number", "E", "N", "U"),
        ("Name", "Number", "BeamID", "E", "N", "U"),
        ("Name", "Number", "E", "N", "U", "Diameter"),
        ("Name", "Number", "BeamID", "E", "N", "U", "Diameter"),
    }
    if header not in accepted:
        raise InstrumentFormatError(
            f"{path}: radiosim header is not one of the strict retained schemas"
        )
    beam_index = header.index("BeamID") if "BeamID" in header else None
    diameter_index = header.index("Diameter") if "Diameter" in header else None
    east_index = header.index("E")
    antennas: list[SourceAntenna] = []
    for index in range(header_index + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        reference = f"line {index + 1}"
        fields = stripped.split()
        if len(fields) != len(header):
            raise InstrumentFormatError(
                f"{path} {reference}: expected {len(header)} columns, got {len(fields)}"
            )
        name = normalize_identity(fields[0], reference=reference, label="antenna name")
        number = _parse_integer_text(
            fields[1], reference=reference, label="antenna number"
        )
        position = tuple(
            _parse_float_text(
                fields[east_index + component],
                reference=reference,
                label=("East", "North", "Up")[component],
            )
            for component in range(3)
        )
        beam_id: int | str | None = None
        if beam_index is not None:
            raw_beam = fields[beam_index]
            if _INTEGER_TEXT.fullmatch(raw_beam) is not None:
                beam_id = normalize_beam_id(int(raw_beam), reference=reference)
            else:
                beam_id = normalize_beam_id(raw_beam, reference=reference)
        diameter = None
        if diameter_index is not None:
            raw_diameter = _parse_float_text(
                fields[diameter_index],
                reference=reference,
                label="Diameter",
            )
            diameter = normalize_source_diameter(raw_diameter, reference=reference)
        antennas.append(
            SourceAntenna(
                number=number,
                name=name,
                position_m=(position[0], position[1], position[2]),
                source_diameter_m=diameter,
                mount_type=None,
                beam_id=beam_id,
                number_generated=False,
                name_generated=False,
                source_record=reference,
            )
        )
    return LoadedInstrumentSource(
        source_kind="layout_file",
        source_reference=str(path),
        source_format="radiosim",
        position_frame="enu",
        explicit_telescope_name=telescope_name,
        embedded_telescope_name=None,
        embedded_location=None,
        antennas=_sorted_validated(antennas, source_reference=str(path)),
        registry_policy=None,
        pyuvdata_version=None,
        source_sha256=_hash_file(path),
    )


def _load_casa_loc(path: Path, telescope_name: str) -> LoadedInstrumentSource:
    lines = _text_lines(path, source_format="casa_loc")
    frame: str | None = None
    raw_rows: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#coordsys="):
            candidate = stripped.removeprefix("#coordsys=").strip().upper()
            if frame is not None and candidate != frame:
                raise InstrumentFormatError(
                    f"{path} line {line_number}: conflicting coordinate headers"
                )
            frame = candidate
            continue
        if stripped.startswith("#"):
            continue
        raw_rows.append((line_number, stripped))
    if frame not in {"LOC", "ENU"}:
        shown = "missing" if frame is None else frame
        raise InstrumentFormatError(
            f"{path}: casa_loc coordinate header must be exactly LOC or ENU; got {shown}"
        )
    antennas: list[SourceAntenna] = []
    for row_number, (line_number, row) in enumerate(raw_rows):
        reference = f"data row {row_number} (line {line_number})"
        fields = row.split()
        if not 3 <= len(fields) <= 6:
            raise InstrumentFormatError(
                f"{path} {reference}: CASA LOC rows require 3 through 6 fields"
            )
        position_values = tuple(
            _parse_float_text(
                fields[index],
                reference=reference,
                label=("East", "North", "Up")[index],
            )
            for index in range(3)
        )
        diameter = None
        if len(fields) >= 4:
            raw_diameter = _parse_float_text(
                fields[3], reference=reference, label="Diameter"
            )
            diameter = normalize_source_diameter(raw_diameter, reference=reference)
        station = (
            normalize_identity(fields[4], reference=reference, label="station name")
            if len(fields) >= 5
            else None
        )
        antenna_name = (
            normalize_identity(fields[5], reference=reference, label="antenna name")
            if len(fields) == 6
            else None
        )
        name = antenna_name or station or f"ANT{row_number:03d}"
        antennas.append(
            SourceAntenna(
                number=row_number,
                name=name,
                position_m=(position_values[0], position_values[1], position_values[2]),
                source_diameter_m=diameter,
                mount_type=None,
                beam_id=None,
                number_generated=True,
                name_generated=antenna_name is None and station is None,
                source_record=reference,
            )
        )
    return LoadedInstrumentSource(
        source_kind="layout_file",
        source_reference=str(path),
        source_format="casa_loc",
        position_frame="enu",
        explicit_telescope_name=telescope_name,
        embedded_telescope_name=None,
        embedded_location=None,
        antennas=_sorted_validated(antennas, source_reference=str(path)),
        registry_policy=None,
        pyuvdata_version=None,
        source_sha256=_hash_file(path),
    )


def _optional_dependency_error(
    *, source_format: str, dependency: str
) -> OptionalInstrumentDependencyError:
    extra = " Install radiosim[ms]." if source_format == "measurement_set" else ""
    return OptionalInstrumentDependencyError(
        f"{source_format}: missing optional dependency {dependency}."
        f" Install {dependency} for this selected source.{extra}"
    )


def _load_mwa(
    path: Path,
    telescope_name: str,
    *,
    module_importer: ModuleImporter,
) -> LoadedInstrumentSource:
    try:
        fits_module = cast(_FitsModule, module_importer("astropy.io.fits"))
    except (ImportError, ModuleNotFoundError) as error:
        raise _optional_dependency_error(
            source_format="mwa_metafits", dependency="astropy"
        ) from error
    try:
        with fits_module.open(path, memmap=False) as hdus:
            if "TILEDATA" not in hdus:
                raise InstrumentFormatError(
                    f"{path}: mwa_metafits source is missing TILEDATA"
                )
            table = hdus["TILEDATA"].data
            if table is None:
                raise EmptyInstrumentError(
                    f"{path}: TILEDATA contains zero antenna records"
                )
            required = ("TileName", "Antenna", "East", "North", "Height")
            names = tuple(table.names or ())
            missing = tuple(column for column in required if column not in names)
            if missing:
                raise InstrumentFormatError(
                    f"{path}: TILEDATA is missing required column(s): "
                    + ", ".join(missing)
                )
            row_count = len(table)
            if row_count == 0:
                raise EmptyInstrumentError(
                    f"{path}: TILEDATA contains zero antenna records"
                )
            column_values = {
                column: tuple(table[column][index] for index in range(row_count))
                for column in required
            }
    except (InstrumentFormatError, EmptyInstrumentError):
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise InstrumentSourceError(
            f"{path}: mwa_metafits source could not be read"
        ) from error

    by_identity: dict[tuple[int, str], SourceAntenna] = {}
    row_indices: dict[tuple[int, str], list[int]] = {}
    for index in range(row_count):
        reference = f"TILEDATA row {index}"
        raw_name = column_values["TileName"][index]
        if isinstance(raw_name, bytes):
            try:
                raw_name = raw_name.decode("utf-8")
            except UnicodeError as error:
                raise AntennaIdentifierError(
                    f"{path} {reference}: TileName must be UTF-8"
                ) from error
        name = normalize_identity(raw_name, reference=reference, label="antenna name")
        number = normalize_antenna_number(
            column_values["Antenna"][index], reference=reference
        )
        position = normalize_position(
            (
                column_values["East"][index],
                column_values["North"][index],
                column_values["Height"][index],
            ),
            reference=reference,
        )
        key = (number, name)
        candidate = SourceAntenna(
            number=number,
            name=name,
            position_m=position,
            source_diameter_m=None,
            mount_type=None,
            beam_id=None,
            number_generated=False,
            name_generated=False,
            source_record=reference,
        )
        previous = by_identity.get(key)
        if previous is None:
            by_identity[key] = candidate
            row_indices[key] = [index]
        elif replace(previous, source_record=reference) != candidate:
            raise AntennaIdentifierError(
                f"{path} {reference}: repeated MWA polarization records conflict "
                f"for antenna {number}/{name}"
            )
        else:
            row_indices[key].append(index)
    collapsed = tuple(
        replace(
            antenna,
            source_record=(
                "TILEDATA row " + str(row_indices[key][0])
                if len(row_indices[key]) == 1
                else "TILEDATA rows " + ",".join(map(str, row_indices[key]))
            ),
        )
        for key, antenna in by_identity.items()
    )
    return LoadedInstrumentSource(
        source_kind="layout_file",
        source_reference=str(path),
        source_format="mwa_metafits",
        position_frame="enu",
        explicit_telescope_name=telescope_name,
        embedded_telescope_name=None,
        embedded_location=None,
        antennas=_sorted_validated(collapsed, source_reference=str(path)),
        registry_policy=None,
        pyuvdata_version=None,
        source_sha256=_hash_file(path),
    )


def _as_sequence(value: object, *, reference: str, label: str) -> tuple[object, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise InstrumentFormatError(f"{reference}: {label} must be an array")
    try:
        return tuple(cast(Iterable[object], value))
    except TypeError as error:
        raise InstrumentFormatError(f"{reference}: {label} must be an array") from error


def _source_location(
    location: object,
    *,
    reference: str,
    module_importer: ModuleImporter,
) -> SourceLocationFacts:
    longitude, latitude, height, xyz = earth_location_facts(
        location,
        reference=reference,
        module_importer=module_importer,
    )
    return SourceLocationFacts(
        longitude_deg=longitude,
        latitude_deg=latitude,
        height_m=height,
        itrs_xyz_m=xyz,
        reference=reference,
    )


def _normalized_telescope(
    telescope: object,
    *,
    source_reference: str,
    source_kind: Literal["layout_file", "known_telescope"],
    source_format: SourceFormat | None,
    explicit_telescope_name: str | None,
    registry_policy: Literal["offline", "allow_network"] | None,
    pyuvdata_version: str,
    source_sha256: str | None,
    module_importer: ModuleImporter,
) -> LoadedInstrumentSource:
    required = ("antenna_names", "antenna_numbers", "antenna_positions", "location")
    missing = tuple(name for name in required if not hasattr(telescope, name))
    if missing:
        raise InstrumentFormatError(
            f"{source_reference}: telescope metadata is missing " + ", ".join(missing)
        )
    metadata = cast(_TelescopeMetadata, telescope)
    names = _as_sequence(
        metadata.antenna_names,
        reference=source_reference,
        label="antenna_names",
    )
    numbers = _as_sequence(
        metadata.antenna_numbers,
        reference=source_reference,
        label="antenna_numbers",
    )
    positions = _as_sequence(
        metadata.antenna_positions,
        reference=source_reference,
        label="antenna_positions",
    )
    count = len(names)
    if len(numbers) != count or len(positions) != count:
        raise InstrumentFormatError(
            f"{source_reference}: telescope antenna arrays must have identical lengths"
        )
    if count == 0:
        raise EmptyInstrumentError(
            f"{source_reference}: telescope metadata contains zero antennas"
        )
    try:
        raw_diameters = metadata.antenna_diameters
    except AttributeError:
        raw_diameters = None
    diameters: tuple[object | None, ...]
    if raw_diameters is None:
        diameters = (None,) * count
    else:
        dense = _as_sequence(
            raw_diameters,
            reference=source_reference,
            label="antenna_diameters",
        )
        if len(dense) != count:
            raise DiameterResolutionError(
                f"{source_reference}: dense antenna_diameters must contain exactly "
                f"{count} values"
            )
        diameters = dense
    try:
        raw_mounts = metadata.mount_type
    except AttributeError:
        raw_mounts = None
    if raw_mounts is None:
        mounts: tuple[object | None, ...] = (None,) * count
    elif isinstance(raw_mounts, str):
        mounts = (raw_mounts,) * count
    else:
        mounts = _as_sequence(
            raw_mounts, reference=source_reference, label="mount_type"
        )
        if len(mounts) != count:
            raise InstrumentFormatError(
                f"{source_reference}: mount_type array must contain exactly {count} values"
            )
    antennas: list[SourceAntenna] = []
    for index in range(count):
        record = f"antenna metadata index {index}"
        position_values = _as_sequence(
            positions[index], reference=record, label="antenna position"
        )
        if len(position_values) != 3:
            raise InstrumentFormatError(
                f"{source_reference} {record}: antenna position must contain "
                "exactly three values"
            )
        position = normalize_position(position_values, reference=record)
        antennas.append(
            SourceAntenna(
                number=normalize_antenna_number(numbers[index], reference=record),
                name=normalize_identity(
                    names[index], reference=record, label="antenna name"
                ),
                position_m=position,
                source_diameter_m=normalize_source_diameter(
                    diameters[index], reference=record
                ),
                mount_type=normalize_mount(mounts[index], reference=record),
                beam_id=None,
                number_generated=False,
                name_generated=False,
                source_record=record,
            )
        )
    try:
        embedded_name_raw = metadata.name
    except AttributeError:
        embedded_name_raw = None
    embedded_name = (
        normalize_identity(
            embedded_name_raw,
            reference=source_reference,
            label="embedded telescope name",
        )
        if embedded_name_raw is not None
        else None
    )
    location = _source_location(
        metadata.location,
        reference=f"{source_reference} embedded location",
        module_importer=module_importer,
    )
    return LoadedInstrumentSource(
        source_kind=source_kind,
        source_reference=source_reference,
        source_format=source_format,
        position_frame="relative_ecef",
        explicit_telescope_name=explicit_telescope_name,
        embedded_telescope_name=embedded_name,
        embedded_location=location,
        antennas=_sorted_validated(antennas, source_reference=source_reference),
        registry_policy=registry_policy,
        pyuvdata_version=pyuvdata_version,
        source_sha256=source_sha256,
    )


def _normalize_dependency_telescope(
    telescope: object,
    *,
    source_reference: str,
    source_kind: Literal["layout_file", "known_telescope"],
    source_format: SourceFormat | None,
    explicit_telescope_name: str | None,
    registry_policy: Literal["offline", "allow_network"] | None,
    pyuvdata_version: str,
    source_sha256: str | None,
    module_importer: ModuleImporter,
) -> LoadedInstrumentSource:
    """Map unexpected dependency-object failures without hiding their cause."""
    try:
        return _normalized_telescope(
            telescope,
            source_reference=source_reference,
            source_kind=source_kind,
            source_format=source_format,
            explicit_telescope_name=explicit_telescope_name,
            registry_policy=registry_policy,
            pyuvdata_version=pyuvdata_version,
            source_sha256=source_sha256,
            module_importer=module_importer,
        )
    except InstrumentResolutionError:
        raise
    except Exception as error:
        selected = source_format or source_kind
        raise InstrumentSourceError(
            f"{source_reference}: {selected} telescope metadata normalization failed"
        ) from error


def _pyuvdata_module(
    module_importer: ModuleImporter, *, source_format: str
) -> _PyuvdataModule:
    try:
        return cast(_PyuvdataModule, module_importer("pyuvdata"))
    except (ImportError, ModuleNotFoundError) as error:
        raise _optional_dependency_error(
            source_format=source_format, dependency="pyuvdata"
        ) from error


def _version(
    supplied: str | None,
    *,
    module_importer: ModuleImporter,
    source_format: str,
) -> str:
    if supplied is not None:
        return normalize_identity(
            supplied, reference=source_format, label="pyuvdata version"
        )
    module = _pyuvdata_module(module_importer, source_format=source_format)
    try:
        return normalize_identity(
            module.__version__, reference=source_format, label="pyuvdata version"
        )
    except AntennaIdentifierError as error:
        raise InstrumentSourceError(
            f"{source_format}: pyuvdata version metadata is invalid"
        ) from error


def _production_dataset_loader(
    path: Path,
    *,
    source_format: SourceFormat,
    module_importer: ModuleImporter,
) -> object:
    module = _pyuvdata_module(module_importer, source_format=source_format)
    try:
        uvdata = module.UVData()
        _ = uvdata.read(path, read_data=False)
        telescope = uvdata.telescope
    except (ImportError, ModuleNotFoundError) as error:
        raise _optional_dependency_error(
            source_format=source_format,
            dependency="radiosim[ms]"
            if source_format == "measurement_set"
            else "pyuvdata",
        ) from error
    except Exception as error:
        raise InstrumentSourceError(
            f"{path}: {source_format} metadata-only telescope loading failed"
        ) from error
    if telescope is None:
        raise InstrumentFormatError(
            f"{path}: {source_format} metadata contains no telescope"
        )
    return telescope


def _load_dataset(
    path: Path,
    source: LayoutFileSourceConfig,
    *,
    dataset_loader: DatasetTelescopeLoader | None,
    module_importer: ModuleImporter,
    pyuvdata_version: str | None,
) -> LoadedInstrumentSource:
    source_format = source.format
    version = _version(
        pyuvdata_version,
        module_importer=module_importer,
        source_format=source_format,
    )
    try:
        telescope = (
            dataset_loader(path, source_format)
            if dataset_loader is not None
            else _production_dataset_loader(
                path,
                source_format=source_format,
                module_importer=module_importer,
            )
        )
    except OptionalInstrumentDependencyError:
        raise
    except InstrumentSourceError:
        raise
    except Exception as error:
        raise InstrumentSourceError(
            f"{path}: {source_format} metadata-only telescope loading failed"
        ) from error
    return _normalize_dependency_telescope(
        telescope,
        source_reference=str(path),
        source_kind="layout_file",
        source_format=source_format,
        explicit_telescope_name=source.telescope_name,
        registry_policy=None,
        pyuvdata_version=version,
        source_sha256=None if source_format == "measurement_set" else _hash_file(path),
        module_importer=module_importer,
    )


def _production_known_loader(
    name: str,
    *,
    module_importer: ModuleImporter,
) -> object:
    module = _pyuvdata_module(module_importer, source_format="known_telescope")
    telescope_type = cast(_TelescopeType, module.Telescope)
    return telescope_type.from_known_telescopes(name)


def _internet_configuration(
    *, module_importer: ModuleImporter
) -> InternetConfiguration:
    try:
        module = cast(_AstropyDataModule, module_importer("astropy.utils.data"))
    except (ImportError, ModuleNotFoundError) as error:
        raise _optional_dependency_error(
            source_format="known_telescope", dependency="astropy"
        ) from error
    return module.conf


def _load_known(
    source: KnownTelescopeSourceConfig,
    *,
    known_telescope_loader: KnownTelescopeLoader | None,
    module_importer: ModuleImporter,
    internet_config: InternetConfiguration | None,
    pyuvdata_version: str | None,
) -> LoadedInstrumentSource:
    version = _version(
        pyuvdata_version,
        module_importer=module_importer,
        source_format="known_telescope",
    )
    if known_telescope_loader is None:

        def production_loader(name: str) -> object:
            return _production_known_loader(name, module_importer=module_importer)

        loader: KnownTelescopeLoader = production_loader
    else:
        loader = known_telescope_loader
    try:
        if source.registry_policy == "offline":
            config = internet_config or _internet_configuration(
                module_importer=module_importer
            )
            with _KNOWN_TELESCOPE_LOCK:
                with config.set_temp("allow_internet", False):
                    telescope = loader(source.name)
        else:
            telescope = loader(source.name)
    except OptionalInstrumentDependencyError:
        raise
    except ValueError as error:
        if _KNOWN_ABSENCE_TEXT in str(error):
            raise TelescopeNotFoundError(
                f"known_telescope {source.name!r}: requested telescope was not found"
            ) from error
        raise InstrumentSourceError(
            f"known_telescope {source.name!r}: telescope metadata validation failed"
        ) from error
    except Exception as error:
        raise InstrumentSourceError(
            f"known_telescope {source.name!r}: telescope loading failed"
        ) from error
    return _normalize_dependency_telescope(
        telescope,
        source_reference=source.name,
        source_kind="known_telescope",
        source_format=None,
        explicit_telescope_name=source.name,
        registry_policy=source.registry_policy,
        pyuvdata_version=version,
        source_sha256=None,
        module_importer=module_importer,
    )


def load_instrument_source(
    source: LayoutFileSourceConfig | KnownTelescopeSourceConfig,
    *,
    dataset_loader: DatasetTelescopeLoader | None = None,
    known_telescope_loader: KnownTelescopeLoader | None = None,
    module_importer: ModuleImporter = import_module,
    internet_config: InternetConfiguration | None = None,
    pyuvdata_version: str | None = None,
) -> LoadedInstrumentSource:
    """Load exactly one frozen Tier 2 source into copied source-frame facts."""
    if isinstance(source, KnownTelescopeSourceConfig):
        return _load_known(
            source,
            known_telescope_loader=known_telescope_loader,
            module_importer=module_importer,
            internet_config=internet_config,
            pyuvdata_version=pyuvdata_version,
        )
    source_format = source.format
    path = _require_path(source.path, source_format=source_format)
    if source_format == "radiosim":
        assert source.telescope_name is not None
        return _load_radiosim(path, source.telescope_name)
    if source_format == "casa_loc":
        assert source.telescope_name is not None
        return _load_casa_loc(path, source.telescope_name)
    if source_format == "mwa_metafits":
        assert source.telescope_name is not None
        return _load_mwa(
            path,
            source.telescope_name,
            module_importer=module_importer,
        )
    return _load_dataset(
        path,
        source,
        dataset_loader=dataset_loader,
        module_importer=module_importer,
        pyuvdata_version=pyuvdata_version,
    )


__all__ = [
    "DatasetTelescopeLoader",
    "InternetConfiguration",
    "KnownTelescopeLoader",
    "LoadedInstrumentSource",
    "ModuleImporter",
    "SourceAntenna",
    "SourceLocationFacts",
    "load_instrument_source",
]
