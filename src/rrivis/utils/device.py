"""Device resource detection for RRIVis.

Provides utilities to detect CPU, GPU, memory, storage, and OS
information for the current system. Used by the Simulator to report
available hardware and by the CLI ``rrivis status`` command.
"""

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rrivis.utils.logging import get_logger

logger = get_logger("rrivis.utils.device")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUBPROCESS_TIMEOUT = 5  # seconds

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryInfo:
    """System RAM information."""

    total_gb: float | None = None
    available_gb: float | None = None


@dataclass(frozen=True)
class StorageInfo:
    """Disk storage information for a given path."""

    total_gb: float
    free_gb: float
    path: str = "."


@dataclass(frozen=True)
class CPUInfo:
    """CPU information."""

    model: str | None = None
    architecture: str = ""
    physical_cores: int | None = None
    logical_cores: int = 0


@dataclass(frozen=True)
class GPUInfo:
    """Information about a single GPU."""

    name: str = ""
    vendor: str = ""
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    cores: int | None = None
    metal_support: str | None = None
    cuda_driver: str | None = None
    rocm_version: str | None = None
    backend: str | None = None


@dataclass(frozen=True)
class OSInfo:
    """Operating system information."""

    name: str = ""
    version: str = ""
    architecture: str = ""
    bits: str = ""


@dataclass(frozen=True)
class DeviceResources:
    """Aggregated device resource information.

    Attributes
    ----------
    memory : MemoryInfo
        System RAM.
    storage : StorageInfo
        Disk storage at the working directory.
    cpu : CPUInfo
        CPU details.
    gpus : list[GPUInfo]
        Detected GPUs (may be empty).
    os_info : OSInfo
        Operating system details.
    timestamp : float
        Monotonic time when detection was performed.
    """

    memory: MemoryInfo
    storage: StorageInfo
    cpu: CPUInfo
    gpus: list[GPUInfo] = field(default_factory=list)
    os_info: OSInfo = field(default_factory=OSInfo)
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def has_gpu(self) -> bool:
        """Whether at least one GPU was detected."""
        return len(self.gpus) > 0

    def summary(self) -> str:
        """Compact one-liner for display during simulation setup."""
        parts: list[str] = []

        # CPU
        cpu_label = self.cpu.model or self.cpu.architecture
        cores = self.cpu.physical_cores or self.cpu.logical_cores
        if cpu_label:
            parts.append(f"{cpu_label} ({self.cpu.architecture}, {cores} cores)")

        # RAM
        if self.memory.total_gb is not None:
            parts.append(f"{self.memory.total_gb:.1f} GB RAM")

        # GPU(s)
        for gpu in self.gpus:
            gpu_parts = [gpu.name or gpu.vendor]
            details: list[str] = []
            if gpu.vram_total_gb is not None:
                details.append(f"{gpu.vram_total_gb:.1f} GB VRAM")
            if gpu.cores is not None:
                details.append(f"{gpu.cores} cores")
            if gpu.metal_support:
                details.append(gpu.metal_support)
            if details:
                gpu_parts.append(f"({', '.join(details)})")
            parts.append(" ".join(gpu_parts))

        return " · ".join(parts) if parts else "unknown device"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "memory": {
                "total_gb": self.memory.total_gb,
                "available_gb": self.memory.available_gb,
            },
            "storage": {
                "total_gb": self.storage.total_gb,
                "free_gb": self.storage.free_gb,
                "path": self.storage.path,
            },
            "cpu": {
                "model": self.cpu.model,
                "architecture": self.cpu.architecture,
                "physical_cores": self.cpu.physical_cores,
                "logical_cores": self.cpu.logical_cores,
            },
            "gpus": [
                {
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_total_gb": g.vram_total_gb,
                    "vram_free_gb": g.vram_free_gb,
                    "cores": g.cores,
                    "metal_support": g.metal_support,
                    "cuda_driver": g.cuda_driver,
                    "rocm_version": g.rocm_version,
                    "backend": g.backend,
                }
                for g in self.gpus
            ],
            "os_info": {
                "name": self.os_info.name,
                "version": self.os_info.version,
                "architecture": self.os_info.architecture,
                "bits": self.os_info.bits,
            },
            "has_gpu": self.has_gpu,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Module-level cache (hardware doesn't change during a process)
# ---------------------------------------------------------------------------

_cached_resources: DeviceResources | None = None


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


def _run_cmd(args: list[str], timeout: int = _SUBPROCESS_TIMEOUT) -> str | None:
    """Run a subprocess and return stdout, or None on any failure."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# ---------------------------------------------------------------------------
# Memory detection
# ---------------------------------------------------------------------------


def _detect_memory() -> MemoryInfo:
    """Detect system RAM.

    Tries psutil first, then platform-specific stdlib fallbacks.
    """
    # 1. psutil (cross-platform, best)
    try:
        import psutil

        vm = psutil.virtual_memory()
        return MemoryInfo(
            total_gb=vm.total / (1024**3),
            available_gb=vm.available / (1024**3),
        )
    except (ImportError, Exception):
        logger.debug("psutil not available for memory detection, using fallback")

    system = platform.system()

    # 2. macOS: sysctl
    if system == "Darwin":
        raw = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if raw:
            try:
                total = int(raw) / (1024**3)
                return MemoryInfo(total_gb=total)
            except ValueError:
                pass

    # 3. Linux: /proc/meminfo
    if system == "Linux":
        try:
            meminfo = Path("/proc/meminfo").read_text()
            total = None
            available = None
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    # Values labeled "kB" are actually KiB (x1024)
                    total = int(line.split()[1]) * 1024 / (1024**3)
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024 / (1024**3)
            if total is not None:
                return MemoryInfo(total_gb=total, available_gb=available)
        except (OSError, ValueError):
            pass

    # 4. Windows: ctypes + GlobalMemoryStatusEx
    if system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return MemoryInfo(
                total_gb=stat.ullTotalPhys / (1024**3),
                available_gb=stat.ullAvailPhys / (1024**3),
            )
        except (OSError, AttributeError, Exception):
            pass

    return MemoryInfo()


# ---------------------------------------------------------------------------
# Storage detection
# ---------------------------------------------------------------------------


def _detect_storage(path: str = ".") -> StorageInfo:
    """Detect disk storage at *path* using shutil (stdlib)."""
    try:
        usage = shutil.disk_usage(path)
        return StorageInfo(
            total_gb=usage.total / (1024**3),
            free_gb=usage.free / (1024**3),
            path=path,
        )
    except OSError:
        return StorageInfo(total_gb=0.0, free_gb=0.0, path=path)


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------


def _detect_cpu() -> CPUInfo:
    """Detect CPU information."""
    architecture = platform.machine()
    logical_cores = os.cpu_count() or 0
    physical_cores: int | None = None
    model: str | None = None

    # Physical cores — try psutil first
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False)
    except (ImportError, Exception):
        pass

    system = platform.system()

    # Physical cores fallback (without psutil)
    if physical_cores is None:
        if system == "Darwin":
            raw = _run_cmd(["sysctl", "-n", "hw.physicalcpu"])
            if raw:
                try:
                    physical_cores = int(raw)
                except ValueError:
                    pass
        elif system == "Linux":
            # Try lscpu: Socket(s) x Core(s) per socket
            raw = _run_cmd(["lscpu"])
            if raw:
                sockets = cores_per = None
                for line in raw.splitlines():
                    if line.startswith("Socket(s):"):
                        try:
                            sockets = int(line.split(":")[1].strip())
                        except ValueError:
                            pass
                    elif line.startswith("Core(s) per socket:"):
                        try:
                            cores_per = int(line.split(":")[1].strip())
                        except ValueError:
                            pass
                if sockets is not None and cores_per is not None:
                    physical_cores = sockets * cores_per

    # CPU model name
    if system == "Darwin":
        model = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif system == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            for line in cpuinfo.splitlines():
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif system == "Windows":
        # Registry gives the human-readable brand string
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"Hardware\Description\System\CentralProcessor\0",
            )
            model = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
        except (OSError, ImportError, Exception):
            # Fall back to platform.processor() (coded string on Windows)
            model = platform.processor() or None

    return CPUInfo(
        model=model,
        architecture=architecture,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
    )


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def _parse_nvidia_smi(output: str) -> list[GPUInfo]:
    """Parse nvidia-smi CSV output. Memory values are in MiB."""
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            vram_total = float(parts[1]) / 1024  # MiB → GiB
        except ValueError:
            vram_total = None
        try:
            vram_free = float(parts[2]) / 1024  # MiB → GiB
        except ValueError:
            vram_free = None
        driver = parts[3] if len(parts) > 3 else None
        gpus.append(
            GPUInfo(
                name=name,
                vendor="NVIDIA",
                vram_total_gb=vram_total,
                vram_free_gb=vram_free,
                cuda_driver=driver,
                backend="cuda",
            )
        )
    return gpus


def _parse_amd_smi(output: str) -> list[GPUInfo]:
    """Parse amd-smi static --csv output."""
    gpus = []
    lines = output.strip().splitlines()
    if len(lines) < 2:
        return gpus
    # CSV with header row
    header = [h.strip().lower() for h in lines[0].split(",")]
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        row = dict(zip(header, parts, strict=False))
        name = row.get("gpu", "") or row.get("name", "") or row.get("asic", "")
        vram_total = None
        # amd-smi reports in various formats depending on version
        for key in ("vram total", "vram_total", "vram total memory"):
            if key in row:
                try:
                    vram_total = float(row[key]) / (1024**3)  # bytes → GB
                except ValueError:
                    pass
        if name:
            gpus.append(
                GPUInfo(
                    name=name, vendor="AMD", vram_total_gb=vram_total, backend="rocm"
                )
            )
    return gpus


def _parse_rocm_smi(output: str) -> list[GPUInfo]:
    """Parse legacy rocm-smi --csv output. VRAM values are in bytes."""
    gpus = []
    lines = output.strip().splitlines()
    if len(lines) < 2:
        return gpus
    header = [h.strip().lower() for h in lines[0].split(",")]
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        row = dict(zip(header, parts, strict=False))
        name = row.get("card series", "") or row.get("card model", "")
        vram_total = None
        for key in ("vram total memory (b)", "vram total"):
            if key in row:
                try:
                    vram_total = float(row[key]) / (1024**3)  # bytes → GB
                except ValueError:
                    pass
        if name:
            gpus.append(
                GPUInfo(
                    name=name, vendor="AMD", vram_total_gb=vram_total, backend="rocm"
                )
            )
    return gpus


def _parse_xpu_smi(output: str) -> list[GPUInfo]:
    """Parse xpu-smi discovery --dump 1,2,16 output."""
    gpus = []
    lines = output.strip().splitlines()
    if len(lines) < 2:
        return gpus
    for line in lines[1:]:
        # Columns: Device ID, Device Name, Memory Physical Size
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[1]
        vram_total = None
        mem_str = parts[2]
        # Memory may be like "12800 MiB" or a raw number
        match = re.match(r"([\d.]+)\s*(MiB|MB|GiB|GB)?", mem_str)
        if match:
            val = float(match.group(1))
            unit = (match.group(2) or "MiB").lower()
            if unit in ("mib", "mb"):
                vram_total = val / 1024
            elif unit in ("gib", "gb"):
                vram_total = val
        if name:
            gpus.append(
                GPUInfo(
                    name=name, vendor="Intel", vram_total_gb=vram_total, backend="xpu"
                )
            )
    return gpus


def _parse_system_profiler(output: str) -> list[GPUInfo]:
    """Parse macOS system_profiler SPDisplaysDataType output."""
    gpus = []
    # Split into GPU blocks (each starts with an indented name ending with ":")
    blocks = re.split(r"\n\s{4}(\S[^:]+):", output)
    # blocks[0] is the header, then alternating (name, body) pairs
    for i in range(1, len(blocks) - 1, 2):
        name = blocks[i].strip()
        body = blocks[i + 1]

        chipset = None
        cores = None
        vendor = None
        metal = None

        for line in body.splitlines():
            line = line.strip()
            if line.startswith("Chipset Model:"):
                chipset = line.split(":", 1)[1].strip()
            elif line.startswith("Total Number of Cores:"):
                try:
                    cores = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Vendor:"):
                vendor = line.split(":", 1)[1].strip()
                # Clean up vendor id like "Apple (0x106b)"
                if "(" in vendor:
                    vendor = vendor.split("(")[0].strip()
            elif line.startswith("Metal Support:") or line.startswith("Metal Family:"):
                metal = line.split(":", 1)[1].strip()

        gpus.append(
            GPUInfo(
                name=chipset or name,
                vendor=vendor or "Apple",
                cores=cores,
                metal_support=metal,
                backend="metal",
            )
        )
    return gpus


def _detect_jax_gpus() -> list[GPUInfo]:
    """Detect GPUs via JAX (fallback for any vendor)."""
    gpus = []
    try:
        import jax

        devices = jax.devices()
        for dev in devices:
            if dev.platform != "cpu":
                gpus.append(
                    GPUInfo(
                        name=str(dev.device_kind)
                        if hasattr(dev, "device_kind")
                        else str(dev),
                        vendor=dev.platform,
                        backend=f"jax:{dev.platform}",
                    )
                )
    except (ImportError, Exception):
        pass
    return gpus


def _detect_gpus() -> list[GPUInfo]:
    """Detect all available GPUs.

    Tries vendor-specific tools in order, then falls back to JAX.
    """
    gpus: list[GPUInfo] = []
    system = platform.system()

    # 1. NVIDIA (Linux/Windows)
    if system != "Darwin":
        output = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if output:
            gpus.extend(_parse_nvidia_smi(output))
            logger.debug("Detected %d NVIDIA GPU(s)", len(gpus))

    # 2. AMD — try amd-smi first, then legacy rocm-smi
    if system == "Linux":
        output = _run_cmd(["amd-smi", "static", "--csv"])
        if output:
            amd = _parse_amd_smi(output)
            gpus.extend(amd)
            logger.debug("Detected %d AMD GPU(s) via amd-smi", len(amd))
        else:
            output = _run_cmd(
                [
                    "rocm-smi",
                    "--showmeminfo",
                    "vram",
                    "--showproductname",
                    "--csv",
                ]
            )
            if output:
                amd = _parse_rocm_smi(output)
                gpus.extend(amd)
                logger.debug("Detected %d AMD GPU(s) via rocm-smi", len(amd))

    # 3. Intel discrete (Linux)
    if system == "Linux":
        output = _run_cmd(["xpu-smi", "discovery", "--dump", "1,2,16"])
        if output:
            intel = _parse_xpu_smi(output)
            gpus.extend(intel)
            logger.debug("Detected %d Intel GPU(s)", len(intel))

    # 4. Apple (macOS)
    if system == "Darwin":
        output = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        if output:
            apple = _parse_system_profiler(output)
            gpus.extend(apple)
            logger.debug("Detected %d Apple GPU(s)", len(apple))

    # 5. JAX fallback (any vendor)
    if not gpus:
        jax_gpus = _detect_jax_gpus()
        if jax_gpus:
            gpus.extend(jax_gpus)
            logger.debug("Detected %d GPU(s) via JAX", len(jax_gpus))

    return gpus


# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------


def _detect_os() -> OSInfo:
    """Detect operating system information."""
    bits = platform.architecture()[0]  # "64bit" or "32bit"
    return OSInfo(
        name=platform.system(),
        version=platform.release(),
        architecture=platform.machine(),
        bits=bits,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_device_resources() -> DeviceResources:
    """Detect and return current device resources.

    Results are cached for the lifetime of the process since hardware
    does not change during a run.

    Returns
    -------
    DeviceResources
        Aggregated hardware information.
    """
    global _cached_resources

    if _cached_resources is not None:
        return _cached_resources

    logger.debug("Detecting device resources...")

    resources = DeviceResources(
        memory=_detect_memory(),
        storage=_detect_storage(),
        cpu=_detect_cpu(),
        gpus=_detect_gpus(),
        os_info=_detect_os(),
        timestamp=time.monotonic(),
    )

    _cached_resources = resources
    logger.debug("Device detection complete: %s", resources.summary())
    return resources


def clear_cache() -> None:
    """Reset the cached device resources.

    Intended for use in tests.
    """
    global _cached_resources
    _cached_resources = None
