"""Reusable CLI test doubles that preserve the real resolver boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from radiosim.api.simulator import Simulator


@pytest.fixture
def recording_simulator(monkeypatch):
    """Record resolved Simulator use while replacing only runtime side effects."""

    original_init = Simulator.__init__
    instances: list[Simulator] = []

    def recording_init(self, resolved, /):
        original_init(self, resolved)
        self.ran = False
        self.save_calls = []
        self.plot_calls = []
        instances.append(self)

    def run(self, *args, **kwargs):
        self.ran = True
        self._result = object()
        return self._result

    def save(self, *args, **kwargs):
        self.save_calls.append((args, kwargs))
        target = Path(args[0])
        result_format = kwargs["format"]
        if not target.name.endswith(result_format.extension):
            target = target.with_name(target.name + result_format.extension)
        if result_format.value == "ms":
            target.mkdir()
            (target / "table.dat").write_bytes(b"test measurement set")
        else:
            target.write_bytes(b"test result")
        return target

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name in ("antenna_layout.html", "visibility-phase-lsts.html"):
            path = output_dir / name
            path.write_text("<html></html>", encoding="utf-8")
            written.append(path)
        return tuple(written)

    monkeypatch.setattr(Simulator, "__init__", recording_init)
    monkeypatch.setattr(Simulator, "run", run)
    monkeypatch.setattr(Simulator, "save", save)
    monkeypatch.setattr(Simulator, "plot", plot)
    Simulator.instances = instances
    return Simulator
