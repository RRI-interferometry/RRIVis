# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `src/radiosim/`. Keep domain logic grouped by area: `core/` for simulation primitives, `backends/` for NumPy/JAX/Dask execution, `io/` for config and file formats, `api/` for high-level entry points, `cli/` for the `radiosim` command, and `utils/` for shared helpers. Tests are under `tests/` with `unit/`, `integration/`, `characterization/` (golden fingerprint pins), `performance/` (benchmark records, never gating), `crossvalidation/` (the optional `crossval` environment), plus the shared `fixtures/` and `support/` helper packages. Documentation lives in `docs/`; runnable examples are in `examples/`; sample configs and antenna layouts live in `configs/` and `antenna_layout_examples/`.

`simulators/` holds 41 third-party simulator checkouts as git submodules, for reference reading only. They are not part of the package, are excluded from the wheel and from Ruff, and a plain `git clone` does not fetch them: they arrive only with `--recursive` or an explicit `git submodule update --init`, and cost roughly 3.9 GB when checked out. Do not add one to a dependency, an import, or a test.

## Build, Test, and Development Commands
Use Pixi for local setup and repeatable tooling:

- `pixi install` installs the Python 3.11 environment and editable package.
- `pixi run test` runs the full pytest suite.
- `pixi run lint` checks Ruff rules; `pixi run fix` applies safe lint fixes.
- `pixi run format` formats code with Ruff; `pixi run check-format` verifies formatting in CI-style mode.
- `pixi run typecheck` runs Pyright using the repo config in `pyproject.toml`.
- `pixi run radiosim --config config.yaml` runs the CLI against a config file.
- `make -C docs html` builds the Sphinx docs into `docs/_build/html/`.

## Coding Style & Naming Conventions
Target Python 3.11+, 4-space indentation, and a max line length of 88 characters. Follow existing `snake_case` module/function names, `PascalCase` classes, and typed public APIs. Use NumPy-style docstrings for user-facing functions and classes. Ruff is the formatter and primary linter; pre-commit currently runs `ruff --fix` and `ruff-format`.

## Testing Guidelines
Pytest is configured in `pyproject.toml` with `test_*.py` and `Test*` collection. Docstring doctests are a separate invocation, `pixi run doctest`, scoped to `src/radiosim`: `pixi run test` does not collect them. Add unit tests beside the affected area, for example `tests/unit/test_core/` for `src/radiosim/core/` changes. Use markers when relevant: `slow`, `gpu`, `integration`, `performance`, and `crossval`. Helpful commands: `pixi run test -- tests/unit/`, `pixi run test -- -m "not slow"`, and `pixi run test -- --cov=radiosim --cov-report=html`.

## Commit & Pull Request Guidelines
Recent history uses conventional prefixes such as `feat:`, `refactor:`, `test:`, and `perf:`. Keep messages imperative and scoped to one change, for example `feat: add ionosphere Jones term`. PRs should explain the behavioral change, note any config or data impacts, link related issues, and include screenshots or plots when visualization output changes.

After completing and verifying any small, coherent task that changes repository
files, create a local commit automatically before handing the work back. Do not
wait for a separate request to commit. Keep each commit narrowly scoped and use
the conventional message format above. Never push, create a pull request, or
otherwise publish commits without first asking the user and receiving explicit
approval.

## API Evolution Policy
Until RadioSim reaches a major stable release such as `v1.0`, prefer architectural clarity and coherent public APIs over backward compatibility. Breaking public API and config changes are acceptable when they remove transitional shims, private-surface leakage, or misleading abstractions. Do not preserve a bad API just to avoid breakage before `v1.0` unless the user explicitly asks for a deprecation path. The contributor-facing statement of this policy is ["Pre-v1 API Evolution Policy"](docs/contributing.rst); breaking changes still have to reach the changelog or the [migration guide](docs/migration_guide.md).

## Configuration & Data Hygiene
Do not commit generated outputs from `output/`, `plots/`, `results/`, coverage artifacts, or large transient FITS/HDF5 products. Keep reproducible examples in `configs/` or `examples/`, and document any new external data requirement in `README.md` or `docs/`. Two output paths are the exception and are committed on purpose: `output/benchmarks/reference/` (the backend benchmark records every performance sentence must cite) and `output/crossvalidation/` (the `pyuvsim` comparison).
