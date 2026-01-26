"""A crude attempt at replicating Rust's `Result` type in Python.

Provides `Ok[T]` and `Err[E]` variants with common Rust-like methods:

- Inspection: `is_ok()`, `is_err()`
- Extraction: `ok()`, `err()`, `unwrap()`, `unwrap_err()`, `expect()`, `expect_err()`
- Fallbacks: `unwrap_or()`, `unwrap_or_else()`
- Transformation: `map()`, `map_err()`, `and_then()`, `or_else()`

Pattern matching is supported via `match/case`:

    match result:
        case Ok(value):
            print(f"Success: {value}")
        case Err(error):
            print(f"Error: {error}")

Type Variance:
    Both `Ok[T]` and `Err[E]` are covariant in their type parameters:
    - `Ok[Subclass]` is a subtype of `Ok[Superclass]`
    - `Err[SubException]` is a subtype of `Err[SuperException]`

    This is safe because both types are immutable (frozen dataclasses) and matches
    Rust's behavior where `Result<T, E>` is covariant in both type parameters.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never, NoReturn, TypeVar, final

from carcinize._exceptions import UnwrapError

if TYPE_CHECKING:
    from carcinize._option import Nothing, Some

T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", bound=Exception, covariant=True)

type Result[T_co, E_co: Exception] = Ok[T_co] | Err[E_co]
"""Type alias for Result: either Ok[T] or Err[E].

Both type parameters are covariant:
- Result[Subclass, E] is a subtype of Result[Superclass, E]
- Result[T, SubException] is a subtype of Result[T, SuperException]

The error type E must be an Exception subclass, allowing unwrap() to raise it directly.
"""


@final
@dataclass(frozen=True, slots=True)
class Ok[T_co]:
    """Success variant of Result, containing a value of type T (covariant)."""

    value: T_co

    def is_ok(self) -> bool:
        """Check if this is an Ok variant."""
        return True

    def is_err(self) -> bool:
        """Check if this is an Err variant."""
        return False

    def ok(self) -> Some[T_co]:
        """Convert to Option, returning Some(value).

        This matches Rust's Result::ok() which returns Option<T>.
        """
        from carcinize._option import Some  # noqa: PLC0415

        return Some(self.value)

    def err(self) -> Nothing:
        """Convert to Option, returning Nothing since this is Ok.

        This matches Rust's Result::err() which returns Option<E>.
        """
        from carcinize._option import Nothing  # noqa: PLC0415

        return Nothing()

    def unwrap(self) -> T_co:
        """Return the contained value.

        Unlike Err.unwrap(), this never raises.
        """
        return self.value

    def unwrap_err(self) -> NoReturn:
        """Raise UnwrapError since this is not an Err variant."""
        raise UnwrapError(f"called `unwrap_err()` on an `Ok` value: {self.value!r}")

    def expect(self, msg: str) -> T_co:  # noqa: ARG002
        """Return the contained value.

        The message is ignored since this is an Ok variant.
        """
        return self.value

    def expect_err(self, msg: str) -> NoReturn:
        """Raise UnwrapError with the provided message."""
        raise UnwrapError(msg)

    def unwrap_or[D](self, default: D) -> T_co:  # noqa: ARG002
        """Return the contained value, ignoring the default."""
        return self.value

    def unwrap_or_else[D](self, f: Callable[[], D]) -> T_co:  # noqa: ARG002
        """Return the contained value, ignoring the fallback function."""
        return self.value

    def map[U](self, f: Callable[[T_co], U]) -> Ok[U]:
        """Transform the contained value using the provided function."""
        return Ok(f(self.value))

    def map_err[F](self, f: Callable[[Never], F]) -> Ok[T_co]:  # noqa: ARG002
        """Return self unchanged since this is an Ok variant."""
        return self

    def map_or[U](self, default: U, f: Callable[[T_co], U]) -> U:  # noqa: ARG002
        """Apply the function to the contained value."""
        return f(self.value)

    def map_or_else[U](self, default_f: Callable[[], U], f: Callable[[T_co], U]) -> U:  # noqa: ARG002
        """Apply the function to the contained value."""
        return f(self.value)

    def and_then[U, E: Exception](self, f: Callable[[T_co], Result[U, E]]) -> Result[U, E]:
        """Call the function with the contained value and return its result.

        This is useful for chaining operations that may fail.
        """
        return f(self.value)

    def or_else[F: Exception](self, f: Callable[[Never], Result[T_co, F]]) -> Ok[T_co]:  # noqa: ARG002
        """Return self unchanged since this is an Ok variant."""
        return self


@final
@dataclass(frozen=True, slots=True)
class Err[E_co: Exception]:
    """Error variant of Result, containing an error of type E (covariant).

    The error type E must be an Exception subclass, allowing unwrap() to raise it directly.
    """

    error: E_co

    def is_ok(self) -> bool:
        """Check if this is an Ok variant."""
        return False

    def is_err(self) -> bool:
        """Check if this is an Err variant."""
        return True

    def ok(self) -> Nothing:
        """Convert to Option, returning Nothing since this is Err.

        This matches Rust's Result::ok() which returns Option<T>.
        """
        from carcinize._option import Nothing  # noqa: PLC0415

        return Nothing()

    def err(self) -> Some[E_co]:
        """Convert to Option, returning Some(error).

        This matches Rust's Result::err() which returns Option<E>.
        """
        from carcinize._option import Some  # noqa: PLC0415

        return Some(self.error)

    def unwrap(self) -> NoReturn:
        """Raise the contained error.

        Since E is bounded by Exception, this directly raises self.error.
        """
        raise self.error

    def unwrap_err(self) -> E_co:
        """Return the contained error.

        Unlike Ok.unwrap_err(), this never raises.
        """
        return self.error

    def expect(self, msg: str) -> NoReturn:
        """Raise the contained error, chained from an UnwrapError with the provided message."""
        raise self.error from UnwrapError(msg)

    def expect_err(self, msg: str) -> E_co:  # noqa: ARG002
        """Return the contained error.

        The message is ignored since this is an Err variant.
        """
        return self.error

    def unwrap_or[D](self, default: D) -> D:
        """Return the provided default value."""
        return default

    def unwrap_or_else[D](self, f: Callable[[], D]) -> D:
        """Call the provided function and return its result."""
        return f()

    def map[U](self, f: Callable[[Never], U]) -> Err[E_co]:  # noqa: ARG002
        """Return self unchanged since this is an Err variant."""
        return self

    def map_err[F: Exception](self, f: Callable[[E_co], F]) -> Err[F]:
        """Transform the contained error using the provided function."""
        return Err(f(self.error))

    def map_or[U](self, default: U, f: Callable[[Never], U]) -> U:  # noqa: ARG002
        """Return the default value since this is an Err variant."""
        return default

    def map_or_else[U](self, default_f: Callable[[], U], f: Callable[[Never], U]) -> U:  # noqa: ARG002
        """Call the default function since this is an Err variant."""
        return default_f()

    def and_then[U, F: Exception](self, f: Callable[[Never], Result[U, F]]) -> Err[E_co]:  # noqa: ARG002
        """Return self unchanged since this is an Err variant."""
        return self

    def or_else[F: Exception](self, f: Callable[[E_co], Result[object, F]]) -> Result[object, F]:
        """Call the function with the contained error and return its result.

        This is useful for handling errors and potentially recovering.
        """
        return f(self.error)
