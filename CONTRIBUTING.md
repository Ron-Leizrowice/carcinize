# Contributing to this project

Ensure you read this document **in full** before planning, or making any edits to this project.

## Getting started

**Requirements:** Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # install all dependencies (including dev)
uv run prek install  # install pre-commit hooks
uv run pytest        # verify everything works
```

## `uv` (dependency and package management)

This project uses [uv](https://docs.astral.sh/uv/) for dependency and package management exclusively.

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`
- Upgrade dependencies: `uv sync --upgrade`
- Update lock file: `uv lock`
- Run a tool: `uv run pytest`, `uv run ruff`, `uv run ty check`
- Launch a REPL: `uv run python`

## `prek` (pre-commit hooks)

This project uses [prek](https://prek.j178.dev/) for pre-commit hooks. It enforces a set of checks defined in `.pre-commit-config.yaml` automatically on every commit. The hooks run in priority order:

| Priority | Hooks                                                            |
| -------- | ---------------------------------------------------------------- |
| 0        | Formatting and validation: e.g. `pyproject-fmt`, `uv-lock`, etc. |
| 1        | Dependency analysis: e.g. `deptry`, `pysentry`                   |
| 2        | Python formatting: e.g. `ruff-format`                            |
| 3        | Linting: e.g. `ruff`                                             |
| 4        | Type checking: e.g. `pyrefly`, `ty`                              |

You can bypass hooks with `git commit -m "..." --no-verify`, but this should be a last resort. CI runs the same checks and will reject the PR if they fail.

## `ruff` (linting and formatting)

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting exclusively. Ruff runs automatically via pre-commit on every commit. To run manually:

- Format code: `uv run ruff format`
- Lint and auto-fix safe issues: `uv run ruff check --fix`
- Apply "unsafe" fixes: `uv run ruff check --fix --unsafe-fixes`

## `ty` (static type checking)

This project uses [ty](https://docs.astral.sh/ty/) for static type checking. It runs automatically via pre-commit (priority 4) and in CI.

To run manually: `uv run ty check`

- Always run `ty` from the repo root.
- Run `ty` after any significant change set.
- See `pyproject.toml` under `[tool.ty.analysis]` for configuration.

## `pyrefly` (type checking)

This project also runs [Pyrefly](https://github.com/facebook/pyrefly) as a secondary type checker via pre-commit (priority 4). There is no manual invocation needed; it runs automatically alongside `ty`.

## `pytest` (testing)

This project uses [pytest](https://docs.pytest.org/) for testing. Tests live in the `tests/` directory and mirror the source module structure.

- Run all tests: `uv run pytest`
- Run a specific file: `uv run pytest tests/test_result.py`
- Run a specific test: `uv run pytest tests/test_result.py::test_ok_unwrap -v`
- Run with verbose output: `uv run pytest -v --tb=short`

Async tests are supported out of the box (`asyncio_mode = "auto"` is configured in `pyproject.toml`).

It is strongly encouraged to add tests for new features and bug fixes.

## Repository structure

```text
src/carcinize/
  __init__.py        # public API: re-exports from internal modules
  _base.py           # RustType mixin (clone)
  _exceptions.py     # CarcinizeError, UnwrapError
  _result.py         # Ok, Err, Result, try_except
  _option.py         # Some, Nothing, Option
  _struct.py         # Struct (Pydantic-based)
  _iter.py           # Iter (fluent iterator)
  _lazy.py           # Lazy, OnceCell
tests/               # mirrors src/ -- one test_<name>.py per _<name>.py
```

### Where to put new code

- **New type or module:** Create `src/carcinize/_thing.py`. The `_` prefix marks it as internal.
- **Shared helpers:** If multiple modules need the same utility, put it in `_base.py` or a new `_<name>.py` internal module.
- **Tests:** Create `tests/test_thing.py` to mirror the source module. Cross-module integration tests go in a separate file (e.g., `test_integration.py`).

### Exporting

Only `__init__.py` defines the public API. To make something importable as `from carcinize import Foo`:

1. Add the import to `__init__.py`
2. Add the name to the `__all__` list (kept in alphabetical order)

Anything not in `__all__` is considered internal. Users should never import from `carcinize._module` directly.

## Python Style Guide

Please read the [Python Style Guide](STYLEGUIDE.md) to ensure your code is consistent with the project's style and conventions.
