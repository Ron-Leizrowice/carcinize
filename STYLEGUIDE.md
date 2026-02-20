# Style Guide

Read this before writing, editing, or reviewing any code in this project.

This guide covers decisions that **aren't** already enforced by tooling. Formatting, import ordering, unused imports, and basic lint rules are handled by `ruff`. Type correctness is handled by `ty` and `pyrefly`. If a tool can catch it, it isn't repeated here.

For tooling setup and commands, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Design Principles

### Reuse within the codebase

Before writing new code, check if equivalent functionality exists elsewhere. Don't duplicate patterns across modules; if `_result.py` and `_option.py` need the same helper, extract it into `_base.py` or a shared internal module.

### Use libraries for infrastructure; reimplement with intent

The core types (`Result`, `Option`, `Struct`, `Iter`, etc.) are intentionally reimplemented from Rust. For everything around them -- validation, serialization, concurrency primitives -- prefer established libraries (Pydantic for model validation, `itertools` for iterator plumbing, `threading.Lock` for synchronization).

### Stability

This is a published PyPI package. Breaking changes to the public API (anything in `__all__`) require a semver version bump. Internal modules (prefixed with `_`) can change freely. Don't accumulate backwards-compatibility shims for internal code.

### Refactor early

Don't stack hacks. Refactor when a function grows beyond easy comprehension, when you're adding conditionals to conditionals, or when behavior can't be described in one sentence.

### Exceptions are signal

Don't swallow exceptions. If translating exceptions, use `raise NewError(...) from e` to preserve the cause chain. Catch only what you can meaningfully handle.

---

## Architecture

### Variant types (Ok/Err, Some/Nothing)

Each variant type is a separate class. Both variants of a pair implement the same set of methods with the same signatures. The "wrong" variant's methods either return `self` unchanged, return a default, or raise. This means callers never need to check which variant they have before calling a method.

The pattern for a method that the "wrong" variant ignores:

```python
# On Ok: actually uses the function
def map[U](self, f: Callable[[T_co], U]) -> Ok[U]:
    return Ok(f(self.value))

# On Err: ignores the function, returns self
def map[U](self, f: Callable[[Never], U]) -> Err[E_co]:  # noqa: ARG002
    return self
```

Key details:

- Use `Never` as the callable's input type on the "wrong" variant. This makes the type checker prove the function is never actually called.
- Suppress `ARG002` (unused argument) with `# noqa: ARG002` on the "wrong" variant. No explanation needed; the pattern is understood project-wide.
- Both variants must have identical method names and positional parameter names.

### Immutability

Variant types (`Ok`, `Err`, `Some`, `Nothing`) are frozen dataclasses:

```python
@final
@dataclass(frozen=True, slots=True)
class Ok(Generic[T_co]):
    value: T_co
```

- `@final` prevents subclassing. All variant types are final.
- `frozen=True` prevents mutation after construction.
- `slots=True` for memory efficiency.

To mutate fields on a frozen dataclass during `__post_init__` (as `Err` does for origin tracking), use `object.__setattr__`:

```python
def __post_init__(self) -> None:
    object.__setattr__(self, "_origin", _capture_origin())
```

### `__slots__` everywhere

All classes define `__slots__`. For dataclasses, `slots=True` handles it. For regular classes, declare explicitly:

```python
class Iter[T](RustType):
    __slots__ = ("_iter",)
```

### Covariance

PEP 695 inline type parameters (`class Foo[T]`) don't support variance annotations. When a class needs covariant type parameters, use old-style `TypeVar` with `Generic`:

```python
T_co = TypeVar("T_co", covariant=True)

class Ok(Generic[T_co]):  # noqa: UP046 - Generic required for covariance
    value: T_co
```

Suppress `UP046` with a `# noqa` comment. New types that hold immutable data and only produce (never consume) their type parameter should be covariant.

### Circular imports

`_result.py` and `_option.py` depend on each other (e.g., `Ok.ok()` returns `Some`). This is resolved with:

1. `if TYPE_CHECKING:` imports for annotations
2. Local imports inside method bodies for runtime

```python
if TYPE_CHECKING:
    from carcinize._option import Some

# Then inside the method:
def ok(self) -> Some[T_co]:
    from carcinize._option import Some  # noqa: PLC0415
    return Some(self.value)
```

Follow this same pattern if adding new cross-module dependencies.

### Type system workarounds: `cast()` vs. `ty:ignore` / `noqa`

These solve different problems. Use the right one.

**`cast()` -- "I know the type; the checker can't prove it."** Use when a value's type is narrower than what the checker infers, and you can state an invariant that guarantees it. Always comment the invariant:

```python
# cast: _value is T (not None) when _initialized is True
return Some(cast(T, self._value))
```

`cast()` is preferred over type-ignore comments for value types because it documents the intended type explicitly, is narrowly scoped to one expression, and won't accidentally suppress unrelated errors on the same line.

**`ty:ignore` / `noqa` -- "the operation is valid; the checker can't model it."** Use when the checker is wrong about whether something is *permitted*, not about what type something *is*. Typical cases: metaclass dynamics, `object.__setattr__` on frozen dataclasses, or linter rules that conflict with a deliberate pattern. Always include the rule code:

```python
cls.__match_args__ = tuple(cls.model_fields.keys())  # ty:ignore[unresolved-attribute]
class Ok(Generic[T_co]):  # noqa: UP046 - Generic required for covariance
```

Brief reasons are encouraged. For project-wide patterns like `ARG002` on variant methods, the rule code alone is sufficient.

---

## Error Handling

### When to return `Result` vs. raise

- **Return `Result[T, E]`** from public API methods that represent operations which can fail as part of normal usage (parsing, validation, fallible conversions like `Struct.try_from()`).
- **Raise exceptions** for programming errors that indicate a bug: invalid arguments (`ValueError` for a negative step in `Iter.step_by`), violated contracts (`UnwrapError` from `unwrap()`), and states that should be impossible.

### Error context

`Err` automatically captures its creation site. Contributors can add richer context:

- `.context("while doing X")` -- high-level description of the operation
- `.note(f"key={value}")` -- debugging details (variable values, state)

Both are no-ops on `Ok`, so they can be chained unconditionally.

---

## Docstrings

Use Google-style docstrings. Include `Args:` and `Returns:` sections when they add information beyond what the signature already says. Omit them on simple methods like `is_ok()` or `unwrap()`.

```python
def try_except(
    f: Callable[[], T],
    *exception_types: type[Exception],
) -> Result[T, Exception]:
    """Execute a function and capture any exceptions as an Err.

    Args:
        f: The function to execute.
        *exception_types: Exception types to catch. Catches all if omitted.

    Returns:
        Ok(result) if the function succeeds, Err(exception) if it raises.
    """
```

---

## Testing

- Test files mirror source modules: `_result.py` -> `test_result.py`.
- Cross-module integration tests go in separate files (`test_integration.py`).
- Prefer flat test functions. Use `Test*` classes only to group tests that share fixtures.
- Name tests after the behavior: `test_ok_map_transforms_value`, not `test_map_1`.
- `xfail_strict = true` is configured globally. An xfail test that passes will fail the suite. Only mark tests as xfail for known, tracked issues.
