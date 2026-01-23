# Python Development Guide

This project targets **Python 3.14**. Write code that uses modern syntax, modern typing, and the modern standard library. Avoid legacy patterns unless there's a clear, documented reason.

This guide exists to reduce friction in reviews and production incidents: consistent patterns, predictable failure modes, and a strong bias toward clarity.

---

## Engineering Philosophy

### Clean code (clarity wins)

- Optimize for the next reader: code is read more than written.
- Make intent obvious: descriptive names, straightforward control flow, explicit data shapes.
- Avoid cleverness: if it needs a comment to explain, it likely needs a refactor.

### DRY (but don’t worship it)

- Avoid duplication of *knowledge*, not duplication of *syntax*.
- Local duplication is acceptable if it improves readability.
- Deduplicate when:
  - the same bug was fixed twice, or
  - the same invariant/logic appears in multiple places, or
  - behavior must stay consistent across features.

### Fast-fail and fail loudly

- Validate early at boundaries (config, API input, file contents, env vars).
- Fail on invalid states with clear exceptions and messages.
- Use assertions only for programmer errors/invariants (not user input).

### Exceptions are signal, not noise

- Do not swallow exceptions.
  - Never use bare `except:`.
  - Never “log and continue” unless the component is explicitly designed to degrade gracefully.
- Don’t hide root causes.
  - If translating exceptions, use `raise NewError(...) from e` to preserve causality.
- Catch only what you can meaningfully handle, usually near boundaries.

### Refactor over tech debt

- Don’t stack hacks: if it feels brittle, refactor while context is fresh.
- Refactor when:
  - a function grows beyond easy comprehension,
  - you add conditionals to conditionals,
  - behavior can’t be described in one sentence,
  - you’re tempted to add “one more special case”.
- Leave code better than you found it (naming, structure, types).

### Use libraries; don’t reinvent solved problems

- Prefer well-maintained libraries for common domains (CLI, HTTP, retries, validation, rendering, parsing).
- Before implementing something non-trivial: check existing libraries and evaluate
  - maintenance activity, tests, API quality, license, adoption.
- If used broadly, wrap third-party libraries behind small interfaces.

### Backwards compatibility is optional (unless requested)

- Default stance: do not maintain backwards compatibility “just in case”.
- If compatibility is required, state it explicitly (supported versions + migration plan).
- Prefer breaking changes with clear migration over accumulating legacy flags/paths.

### Tests protect behavior

- Write tests that protect behavior, not implementation details.
- Bugs fixed → test added.

---

## General Principles

- **Strict typing:** No `Any`. Prefer precise types and make invalid states unrepresentable where practical.
  - If something is unknown, model it as `object` and narrow via validation/parsing.
- **Small surfaces:** Prefer small, well-named units of behavior over monoliths, but avoid thin wrappers that add indirection without meaning.
- **Paths:** Always use `pathlib.Path` instead of `os.path`.
- **Async:** Use async I/O for I/O-bound work. Prefer structured concurrency (`asyncio.TaskGroup`).
- **Boundaries:** Convert untyped input into typed models as early as possible; keep the interior of the system strongly typed.

---

## Code-Style Preferences (Project Defaults)

These are defaults for readability and debuggability. Deviations are fine when they improve clarity or match an external interface, but should be intentional.

### Prefer named return types over tuples

- Avoid returning tuples in most cases.
- Return a `@dataclass(slots=True)` or `BaseModel` when multiple values belong together.
- Tuples are acceptable for:
  - very small algorithmic helpers in tight local scope where meaning is obvious, or
  - Python/stdlib APIs that naturally return tuples when the result is used immediately.

### Don’t pass raw dictionaries around internally

- Dictionaries are allowed at boundaries where libraries force them (JSON/YAML/etc.).
- Inside the codebase, prefer:
  - `@dataclass(slots=True, kw_only=True)` for internal domain/data flow
  - `BaseModel` for validated boundary inputs/outputs
- Raw dicts are acceptable only as *arbitrary mappings* (schema not owned by us), and should be typed as `Mapping[...]`, not `dict[...]`.

### Avoid silent fallbacks like `d.get("key", "")`

- Default values in `.get()` can hide missing data and make bugs harder to trace.
- Prefer:
  - `d["key"]` for required keys (fail fast with a clear error)
  - `d.get("key")` returning `None` for optional keys, handled explicitly
  - boundary parsing into a model that enforces required/optional fields
- If a default is genuinely desired, apply it explicitly after validation and with a clear name.

### Avoid trivial one-liner functions

A function should either reduce cognitive load or enforce a boundary.

- Don’t create wrappers that only rename or forward a single expression.
- A function is justified when it:
  - names a concept that improves understanding, **or**
  - enforces an invariant, **or**
  - isolates boundary/I/O behavior, **or**
  - is reused across call sites (or is very likely to be)
- Inline simple logic when a function would add indirection without meaning.

### Enum mapping: prefer behavior on the enum

- Avoid separate “mapping dicts” from enum variants to values/handlers.
- Prefer `@property` or methods on the enum to keep mapping logic co-located and type-checked.
- Exception: use an external mapping only for plugin/registry-style extensibility or configuration-driven dispatch.

### Keyword-only booleans

- If a function takes boolean flags, make them keyword-only:
  - `def f(x: int, *, strict: bool = True) -> int: ...`
- Positional booleans harm readability at call sites.

### Prefer explicit `None` over sentinel-y defaults

- If “missing” is meaningful, represent it with `None` (or a dedicated type), not `""`, `0`, `[]`, etc.
- Keep “absence” distinct from “empty”.

---

## Package Management (uv)

Use **uv** exclusively for dependency and package management. See [uv](https://docs.astral.sh/uv/) for more information.

### Commands

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`
- Upgrade dependencies: `uv sync --upgrade`
- Update lock file: `uv lock`

### Running code and tools

- Run a tool: `uv run pytest`, `uv run ruff`, `uv run ty check`
- Launch a REPL: `uv run python`

> Prefer running project CLIs via `[project.scripts]` entry points (see below), instead of targeting `.py` files directly.

---

## Linting & Formatting (Ruff)

Use **Ruff** exclusively for linting and formatting.

### Ruff instructions

- All Python code must be formatted and linted with Ruff.
- Never use `# noqa` to suppress errors; fix the underlying issue instead.
- In tests, it is acceptable to suppress errors with `# noqa: <rule_name>` if the warning is intentional and the test stays readable.

### Ruff commands

- Format code: `uv run ruff format`
- Lint and auto-fix safe issues: `uv run ruff check --fix`

---

## Static Type Checking (ty)

This project uses **ty** for static type checking. See [ty](https://docs.astral.sh/ty/) for more information.

### Ty commands

- Run static type checker: `uv run ty check`

### Ty instructions

- Run `ty` after any significant change set.
- Always run `ty` at the repo root (changes may affect multiple packages).
- Not all packages share the same rules; see root `pyproject.toml` for overrides.
- Never use `# ty: ignore` except in rare, justified cases (e.g., certain tests).
- You may use `cast(...)` as an absolute last resort when there is no rigorous alternative.

---

## Data Modeling (Strict)

Avoid dictionaries for internal data transfer, domain objects, and config.

### Decision matrix

1. **Internal logic:** `@dataclass(slots=True)`
2. **Validation/serialization/config (boundaries):** Pydantic `BaseModel` with strict validation.

### Defaults and invariants

- Prefer constructors that enforce invariants.
- Prefer `kw_only=True` for dataclasses when argument order would be unclear.
- Prefer immutability for value objects: `@dataclass(frozen=True, slots=True, kw_only=True)`.

### Boundary rule (dicts only at the edge)

Some libraries return dicts (e.g., `json.load`, `yaml.load`). That's fine at the boundary:

- Parse → validate → convert immediately into a pydantic model.
- Do not pass raw dicts around internally.

### Arbitrary mappings

Sometimes you do not own the schema (e.g., labels, headers, tags, opaque metadata). In these cases:

- Use a typed mapping interface such as `Mapping[str, str]` (or a project-specific JSON type alias).
- Keep it at the edge where possible; avoid letting “bag of fields” data seep into domain logic.

---

## Type System (PEP 695 + Modern Typing)

### Core conventions

- Use built-in collections and unions:
  - Good: `list[int]`, `dict[str, str]`, `T | None`
  - Avoid: `List[int]`, `Dict[str, str]`, `Optional[T]`

### Type parameters (PEP 695)

- Type variables: `def identity[T](x: T) -> T: ...`
- Constrained: `def clamp[T: (int, float)](x: T, lo: T, hi: T) -> T: ...`
- ParamSpecs: `def timed[**P, R](f: Callable[P, R]) -> Callable[P, R]: ...`

### Type aliases

- Use `type`: `type UserId = int`

### Narrowing

- Prefer `TypeIs[T]` for narrowing helpers:
  - `def is_user_id(x: object) -> TypeIs[UserId]: ...`

### Overrides

- Use `@override` from `typing` on overridden methods.

---

## Control Flow & Syntax

### Pattern matching vs. isinstance

- Use `match/case` for shape/variant dispatch.
- Use `isinstance(...)` for simple type checks.
- Never use `type(x) == MyType`.

### Concurrency errors

- Use `except*` when handling exception groups from concurrent execution.

---

## Async Concurrency & I/O

- Prefer `asyncio.TaskGroup()` over `asyncio.gather()`.
- Prefer scoped timeouts: `asyncio.timeout(...)`.
- For blocking work, use `await asyncio.to_thread(...)`.
- Treat cancellation as normal control flow; don’t swallow cancellation errors.

---

## Testing Guidelines (Behavior-Focused)

We already use a static type checker. Tests should be used conservatively and focus on **behavior** and **contracts**, not type trivia.

### What to test

Write tests when they meaningfully reduce risk:

- **Business rules and invariants**
  - Any logic that encodes product/engineering rules belongs in tests.
- **Boundary behavior**
  - Parsing/validation of external input (API payloads, config, env vars, file formats).
  - Error types and messages that callers depend on.
- **Bug fixes**
  - Every bug fix should include a regression test.
- **Tricky logic / edge cases**
  - Date math, rounding, pagination, retries, normalization, escaping, state machines.
- **Integration seams**
  - Interactions with external systems (HTTP, filesystem, subprocess), using fakes/stubs where appropriate.

### What not to test

Avoid low-value tests that add maintenance cost:

- Trivial tests for “it accepts these types” or “returns this type”.
- Tests that mirror the implementation (asserting internal helper calls, exact intermediate steps).
- Tests that lock down incidental details (ordering of dict keys, internal log lines) unless it is a contract.

### A good test is a contract

A test should answer: *what must remain true as the implementation evolves?*

- Prefer asserting on externally observable outcomes: returned values, state changes, emitted events, persisted records, raised exceptions.
- Don’t over-mock. If mocking becomes the test, it’s usually a smell.
- If you need heavy mocking, consider extracting an interface and testing through that boundary.

### Test levels: choose the cheapest that still proves behavior

- **Unit tests:** pure logic and invariants (fast, cheap).
- **Contract tests:** boundaries (input → validated model or specific error).
- **Integration tests:** critical seams (use test doubles or local containers where available).
- Avoid over-indexing on integration tests; use them for a small number of high-value paths.

### Determinism and stability

- Freeze time when time matters.
- Control randomness (seed or property-based testing).
- Avoid relying on global state; prefer fixtures that build explicit state.
- Don’t hit the network in unit tests.

### Property-based testing (Hypothesis)

Use Hypothesis when the input space is large or edge cases matter:

- parsers/serializers
- normalization/escaping
- merge/diff logic

### Async tests

- Use `pytest-asyncio` for async code.
- Prefer testing async functions directly.
- For concurrency, test observable behavior (cancellation propagation, error grouping), not scheduling minutiae.

---

## Opinionated Library Choices (2025 baseline)

The goal is consistency and leverage: pick a small set of excellent, well-supported libraries and standardize on them.

### CLI

- **click** for CLI structure and ergonomics.  [click.palletsprojects.com](https://click.palletsprojects.com/)
- **rich** for terminal rendering (tables, progress, tracebacks, formatting).  [rich.readthedocs.io](https://rich.readthedocs.io/en/stable/reference/console.html)
- Optional: **rich-click** to render Click help output via Rich.  [PyPI](https://pypi.org/project/rich-click/)

### Validation / configuration

- **pydantic** for validation and serialization at boundaries, with strict mode when appropriate.  [Pydantic](https://docs.pydantic.dev/latest/concepts/strict_mode/)
- **pydantic-settings** for environment-driven configuration (12-factor style).  [Pydantic](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

### HTTP + resilience

- **httpx** for HTTP clients (sync and async, prefer async in async services).  [python-httpx.org](https://www.python-httpx.org/async/)
- **tenacity** for retries/backoff (explicit retry policies, never “magic”).  [tenacity.readthedocs.io](https://tenacity.readthedocs.io/en/latest/)

### Logging

- Prefer stdlib `logging` semantics; avoid print statements.
- In exception handlers, use `logger.exception("Helpful message with context", extra={...})` to preserve tracebacks.

### Testing

- **pytest** as the default test runner.  [docs.pytest.org](https://docs.pytest.org/)
- **pytest-asyncio** for async tests.  [pytest-asyncio.readthedocs.io](https://pytest-asyncio.readthedocs.io/en/latest/)
- **hypothesis** for property-based tests when input spaces are large or edge cases matter.  [hypothesis.readthedocs.io](https://hypothesis.readthedocs.io/en/latest/)

> Adding a new dependency is fine when it’s clearly better than building/maintaining our own. Wrap widely-used third-party libraries behind small interfaces when appropriate. Deviations from the baseline should include a short justification in the PR description.

---

## Entry Points: Always use `[project.scripts]`

### Rule

- Do not treat `.py` files as “executables” in the repo.
- If something is meant to be run by humans/CI as a command, it must be exposed as an entry point via:
  - `pyproject.toml` → `[project.scripts]`  [packaging.python.org](https://packaging.python.org/en/latest/specifications/pyproject-toml/)

This aligns with modern packaging standards and ensures cross-platform command wrappers are generated (especially important on Windows). Click explicitly recommends packaging CLIs with entry points rather than relying on `if __name__ == "__main__": ...`.  [click.palletsprojects.com](https://click.palletsprojects.com/en/stable/quickstart/)

### Example

```toml
[project.scripts]
my-tool = "my_package.cli:main"
```
