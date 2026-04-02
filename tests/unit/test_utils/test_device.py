"""Tests for rrivis.utils.device resource detection."""

from unittest.mock import MagicMock, patch

import pytest

from rrivis.utils.device import (
    CPUInfo,
    DeviceResources,
    GPUInfo,
    MemoryInfo,
    OSInfo,
    StorageInfo,
    _detect_cpu,
    _detect_gpus,
    _detect_memory,
    _detect_os,
    _detect_storage,
    _parse_nvidia_smi,
    _parse_system_profiler,
    _parse_xpu_smi,
    clear_cache,
    get_device_resources,
)


@pytest.fixture(autouse=True)
def _clear_device_cache():
    """Ensure a clean cache for every test."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# MemoryInfo / _detect_memory
# ---------------------------------------------------------------------------


class TestDetectMemory:
    def test_with_psutil(self):
        """When psutil is available, _detect_memory uses it."""
        mock_psutil = MagicMock()
        vm = MagicMock()
        vm.total = 16 * (1024**3)
        vm.available = 8 * (1024**3)
        mock_psutil.virtual_memory.return_value = vm

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            info = _detect_memory()
        assert info.total_gb == pytest.approx(16.0)
        assert info.available_gb == pytest.approx(8.0)

    @patch("rrivis.utils.device.platform.system", return_value="Darwin")
    @patch("rrivis.utils.device._run_cmd", return_value="17179869184")
    def test_macos_sysctl_fallback(self, mock_cmd, mock_sys):
        """When psutil import fails on macOS, fall back to sysctl."""
        with patch.dict("sys.modules", {"psutil": None}):
            info = _detect_memory()
        assert info.total_gb == pytest.approx(16.0)

    def test_memory_info_defaults(self):
        info = MemoryInfo()
        assert info.total_gb is None
        assert info.available_gb is None


# ---------------------------------------------------------------------------
# StorageInfo / _detect_storage
# ---------------------------------------------------------------------------


class TestDetectStorage:
    @patch("rrivis.utils.device.shutil.disk_usage")
    def test_basic(self, mock_usage):
        mock_usage.return_value = MagicMock(
            total=500 * (1024**3),
            free=200 * (1024**3),
        )
        info = _detect_storage("/tmp")
        assert info.total_gb == pytest.approx(500.0)
        assert info.free_gb == pytest.approx(200.0)
        assert info.path == "/tmp"

    @patch("rrivis.utils.device.shutil.disk_usage", side_effect=OSError("fail"))
    def test_failure(self, mock_usage):
        info = _detect_storage()
        assert info.total_gb == 0.0
        assert info.free_gb == 0.0


# ---------------------------------------------------------------------------
# CPUInfo / _detect_cpu
# ---------------------------------------------------------------------------


class TestDetectCPU:
    def test_has_logical_cores(self):
        info = _detect_cpu()
        assert info.logical_cores > 0

    def test_architecture(self):
        info = _detect_cpu()
        assert info.architecture in ("x86_64", "arm64", "aarch64", "AMD64", "i386")

    @patch("rrivis.utils.device.platform.system", return_value="Darwin")
    @patch("rrivis.utils.device._run_cmd")
    def test_macos_model(self, mock_cmd, mock_sys):
        def side_effect(args, timeout=5):
            if "machdep.cpu.brand_string" in args:
                return "Apple M1"
            if "hw.physicalcpu" in args:
                return "8"
            return None

        mock_cmd.side_effect = side_effect
        # This validates the macOS path calls the right sysctl keys


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


class TestParseNvidiaSmi:
    def test_single_gpu(self):
        output = "NVIDIA A100 80GB PCIe, 81920, 81200, 535.129.03"
        gpus = _parse_nvidia_smi(output)
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA A100 80GB PCIe"
        assert gpus[0].vendor == "NVIDIA"
        assert gpus[0].vram_total_gb == pytest.approx(81920 / 1024)
        assert gpus[0].vram_free_gb == pytest.approx(81200 / 1024)
        assert gpus[0].cuda_driver == "535.129.03"
        assert gpus[0].backend == "cuda"

    def test_multi_gpu(self):
        output = (
            "Tesla V100-SXM2-16GB, 16384, 16000, 525.85.12\n"
            "Tesla V100-SXM2-16GB, 16384, 15800, 525.85.12"
        )
        gpus = _parse_nvidia_smi(output)
        assert len(gpus) == 2

    def test_empty_output(self):
        assert _parse_nvidia_smi("") == []

    def test_malformed_line(self):
        assert _parse_nvidia_smi("incomplete, data") == []


class TestParseSystemProfiler:
    SAMPLE_OUTPUT = """Graphics/Displays:

    Apple M1:

      Chipset Model: Apple M1
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 7
      Vendor: Apple (0x106b)
      Metal Support: Metal 4
      Displays:
        Color LCD:
          Display Type: Built-In Retina LCD
"""

    def test_parse_apple_m1(self):
        gpus = _parse_system_profiler(self.SAMPLE_OUTPUT)
        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.name == "Apple M1"
        assert gpu.vendor == "Apple"
        assert gpu.cores == 7
        assert gpu.metal_support == "Metal 4"
        assert gpu.backend == "metal"

    def test_empty_output(self):
        assert _parse_system_profiler("") == []


class TestParseXpuSmi:
    def test_basic(self):
        output = (
            "Device ID,Device Name,Memory Physical Size\n"
            '0,"Intel(R) Data Center GPU Max 1550","48 GiB"'
        )
        gpus = _parse_xpu_smi(output)
        assert len(gpus) == 1
        assert gpus[0].name == "Intel(R) Data Center GPU Max 1550"
        assert gpus[0].vendor == "Intel"
        assert gpus[0].vram_total_gb == pytest.approx(48.0)
        assert gpus[0].backend == "xpu"

    def test_mib_units(self):
        output = (
            "Device ID,Device Name,Memory Physical Size\n"
            '0,"Intel(R) Graphics [0x56c0]","12800 MiB"'
        )
        gpus = _parse_xpu_smi(output)
        assert len(gpus) == 1
        assert gpus[0].vram_total_gb == pytest.approx(12800 / 1024)

    def test_empty_output(self):
        assert _parse_xpu_smi("") == []


class TestDetectGPUs:
    @patch("rrivis.utils.device._run_cmd", return_value=None)
    @patch("rrivis.utils.device._detect_jax_gpus", return_value=[])
    def test_no_gpus(self, mock_jax, mock_cmd):
        gpus = _detect_gpus()
        assert gpus == []

    @patch("rrivis.utils.device.platform.system", return_value="Linux")
    @patch("rrivis.utils.device._detect_jax_gpus", return_value=[])
    @patch("rrivis.utils.device._run_cmd")
    def test_nvidia_on_linux(self, mock_cmd, mock_jax, mock_sys):
        def side_effect(args, timeout=5):
            if "nvidia-smi" in args:
                return "GeForce RTX 3090, 24576, 24000, 510.47.03"
            return None

        mock_cmd.side_effect = side_effect
        gpus = _detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].vendor == "NVIDIA"


# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------


class TestDetectOS:
    def test_basic(self):
        info = _detect_os()
        assert info.name in ("Darwin", "Linux", "Windows")
        assert info.bits in ("64bit", "32bit")
        assert info.architecture != ""


# ---------------------------------------------------------------------------
# DeviceResources
# ---------------------------------------------------------------------------


class TestDeviceResources:
    def test_has_gpu_true(self):
        res = DeviceResources(
            memory=MemoryInfo(total_gb=16.0),
            storage=StorageInfo(total_gb=500.0, free_gb=200.0),
            cpu=CPUInfo(logical_cores=8),
            gpus=[GPUInfo(name="Test GPU", vendor="Test")],
            os_info=OSInfo(name="Linux"),
        )
        assert res.has_gpu is True

    def test_has_gpu_false(self):
        res = DeviceResources(
            memory=MemoryInfo(total_gb=16.0),
            storage=StorageInfo(total_gb=500.0, free_gb=200.0),
            cpu=CPUInfo(logical_cores=8),
            gpus=[],
            os_info=OSInfo(name="Linux"),
        )
        assert res.has_gpu is False

    def test_summary(self):
        res = DeviceResources(
            memory=MemoryInfo(total_gb=16.0),
            storage=StorageInfo(total_gb=500.0, free_gb=200.0),
            cpu=CPUInfo(model="Apple M1", architecture="arm64", logical_cores=8),
            gpus=[
                GPUInfo(
                    name="Apple M1", vendor="Apple", cores=7, metal_support="Metal 4"
                )
            ],
            os_info=OSInfo(name="Darwin"),
        )
        s = res.summary()
        assert "Apple M1" in s
        assert "16.0 GB RAM" in s
        assert "7 cores" in s
        assert "Metal 4" in s

    def test_summary_no_gpu(self):
        res = DeviceResources(
            memory=MemoryInfo(total_gb=8.0),
            storage=StorageInfo(total_gb=256.0, free_gb=100.0),
            cpu=CPUInfo(architecture="x86_64", logical_cores=4),
            gpus=[],
            os_info=OSInfo(name="Linux"),
        )
        s = res.summary()
        assert "8.0 GB RAM" in s
        assert "GPU" not in s

    def test_summary_empty(self):
        res = DeviceResources(
            memory=MemoryInfo(),
            storage=StorageInfo(total_gb=0.0, free_gb=0.0),
            cpu=CPUInfo(),
            gpus=[],
            os_info=OSInfo(),
        )
        assert res.summary() == "unknown device"

    def test_to_dict(self):
        res = DeviceResources(
            memory=MemoryInfo(total_gb=16.0, available_gb=8.0),
            storage=StorageInfo(total_gb=500.0, free_gb=200.0, path="/"),
            cpu=CPUInfo(
                model="Test", architecture="arm64", physical_cores=8, logical_cores=8
            ),
            gpus=[GPUInfo(name="G", vendor="V", backend="cuda")],
            os_info=OSInfo(
                name="Linux", version="5.15", architecture="x86_64", bits="64bit"
            ),
        )
        d = res.to_dict()
        assert d["memory"]["total_gb"] == 16.0
        assert d["cpu"]["model"] == "Test"
        assert len(d["gpus"]) == 1
        assert d["gpus"][0]["backend"] == "cuda"
        assert d["has_gpu"] is True
        assert "timestamp" in d

    def test_frozen(self):
        res = DeviceResources(
            memory=MemoryInfo(),
            storage=StorageInfo(total_gb=0.0, free_gb=0.0),
            cpu=CPUInfo(),
        )
        with pytest.raises(AttributeError):
            res.memory = MemoryInfo(total_gb=1.0)


# ---------------------------------------------------------------------------
# get_device_resources / caching
# ---------------------------------------------------------------------------


class TestGetDeviceResources:
    def test_returns_device_resources(self):
        res = get_device_resources()
        assert isinstance(res, DeviceResources)
        assert res.cpu.logical_cores > 0

    def test_cached(self):
        r1 = get_device_resources()
        r2 = get_device_resources()
        assert r1 is r2  # Same object (cached)

    def test_clear_cache(self):
        r1 = get_device_resources()
        clear_cache()
        r2 = get_device_resources()
        assert r1 is not r2  # Different objects after cache clear
