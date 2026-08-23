"""PERF-001 P-c/P-d simulator acceptance tests."""

from __future__ import annotations

import subprocess
import sys

from radiosim.simulator import _SIMULATORS, RIMESimulator, VisibilitySimulator
from tests.fixtures.configs import valid_config_mapping, write_config_yaml


class _UnmeasuredSimulator(VisibilitySimulator):
    @property
    def name(self) -> str:
        return "unmeasured"

    @property
    def description(self) -> str:
        return "Test-only simulator with no accelerator evidence"

    def calculate_visibilities(self, *args, **kwargs):
        raise NotImplementedError


def test_abstract_simulator_capability_defaults_to_false() -> None:
    assert _UnmeasuredSimulator().supports_gpu is False


def test_every_registered_simulator_has_an_explicit_unbacked_false() -> None:
    assert RIMESimulator.__dict__["supports_gpu"].fget is not None
    # WIDENED BY: SCI-004 phase M1.  ``docs/development/
    # sci004_mmode_design.md`` Section 2 registers the m-mode forward model, and
    # Section 9 keeps its ``supports_gpu`` explicitly ``False`` -- no
    # independently accepted end-to-end accelerator record names it either.
    # Section 13.3 authorizes exactly this inventory widening.
    assert {name: cls().supports_gpu for name, cls in _SIMULATORS.items()} == {
        "mmode": False,
        "rime": False,
    }


def test_minimal_auto_setup_does_not_import_jax(tmp_path) -> None:
    config = valid_config_mapping(tmp_path, execution={"backend": "auto"})
    config_path = write_config_yaml(tmp_path, config)
    code = f"""
import sys
from radiosim.api import Simulator

assert "jax" not in sys.modules
assert "jaxlib" not in sys.modules
simulator = Simulator.from_yaml({str(config_path)!r})
simulator.setup()
assert simulator._backend.name == "numpy-cpu"
assert "jax" not in sys.modules
assert "jaxlib" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
