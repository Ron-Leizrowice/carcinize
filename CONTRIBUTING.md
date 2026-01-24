# Contributing to this project

## `uv` (dependency and package management)

This project uses [uv](https://docs.astral.sh/uv/) for dependency and package management exclusively.

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`
- Upgrade dependencies: `uv sync --upgrade`
- Update lock file: `uv lock`
- Run a tool: `uv run pytest`, `uv run ruff`, `uv run ty check`
- Launch a REPL: `uv run python`

Pre-commit hooks (see below) will automatically ensure:

1. Ensure `uv.lock` file is up to date to ensure any local dependency changes you have made are pushed to every developer.
2. Ensure a single `uv.lock` file is present in the root of the project. There should be no other `uv.lock` files in the repo.

## `ruff` (linting and formatting)

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting exclusively.

- Format code: `uv run ruff format`
- Lint and auto-fix safe issues: `uv run ruff check --fix`
- Apply "unsafe" fixes: `uv run ruff check --fix --unsafe-fixes`

## `ty` (static type checking)

This project uses [ty](https://docs.astral.sh/ty/) for static type checking. This is not optional, it will be enforced by pre-commit hooks and CI checks. You can check for type errors manually by running:

```bash
uv run ty check
```

- Do not use `# type: ignore` to suppress errors, fix the underlying issue instead.
- You may use `cast(...)` as an absolute last resort when there is no rigorous alternative.
- If you genuinely need to violate type-safety in a specific part of the code because of unavoidable deficiencies of python's lack of comprehensive compile-time type checking, you may use `# ty: ignore[specific-rule]` to suppress the error, but it must be documented with a concrete rationale.
- Tests which confirm error behaviour are exempt, you may use `cast(...)` or `# ty: ignore` to suppress the error, but it must be clearly documented.

## `prek` (pre-commit hooks)

This project uses [prek](https://prek.j178.dev/) for pre-commit hooks.

You will need to install the prek CLI tool to your system:

```bash
uv run prek install
```

It will enforce a set of rules defined in `.pre-commit-config.yaml` automatically on every commit.
If necessary, you can bypass the hooks by running `git commit -m "..." --no-verify`.
Note that CI checks will still run and fail if the hooks are not passed.

## `pytest` (testing)

This project uses [pytest](https://docs.pytest.org/) for testing.
It is strongly encouraged to add tests for new features and bug fixes.

You can run tests manually using the text explorer in your IDE or by running `uv run pytest` in the terminal.

## Python Style Guide

Please read the [Python Style Guide](STYLEGUIDE.md) to ensure your code is consistent with the project's style and conventions.
