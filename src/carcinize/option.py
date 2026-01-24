"""A crude attempt at replicating Rust's `Option` type in Python.

Provides `Some[T]` and `Nothing` variants with common Rust-like methods:

- Inspection: `is_some()`, `is_nothing()`
- Extraction: `unwrap()`, `expect()`, `unwrap_or()`, `unwrap_or_else()`
- Transformation: `map()`, `map_or()`, `map_or_else()`, `and_then()`, `or_else()`
- Filtering: `filter()`
- Conversion: `ok_or()`, `ok_or_else()`

Note: We use `Nothing` instead of `None` to avoid conflict with Python's `None`.

Pattern matching is supported via `match/case`:
```python
match option:
    case Some(value):
        print(f"Got: {value}")
    case Nothing():
        print("Nothing here")
```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Never, NoReturn, cast, final

from carcinize.exceptions import UnwrapError
from carcinize.result import Err, Ok

type Option[T] = Some[T] | Nothing
"""Type alias for Option: either Some[T] or Nothing."""


@final
@dataclass(frozen=True, slots=True)
class Some[T]:
    """Some variant of Option, containing a value of type T."""

    value: T

    def is_some(self) -> bool:
        """Check if this is a Some variant."""
        return True

    def is_nothing(self) -> bool:
        """Check if this is a Nothing variant."""
        return False

    def unwrap(self) -> T:
        """Return the contained value.

        Unlike Nothing.unwrap(), this never raises.
        """
        return self.value

    def expect(self, msg: str) -> T:  # noqa: ARG002
        """Return the contained value.

        The message is ignored since this is a Some variant.
        """
        return self.value

    def unwrap_or[D](self, default: D) -> T:  # noqa: ARG002
        """Return the contained value, ignoring the default."""
        return self.value

    def unwrap_or_else[D](self, f: Callable[[], D]) -> T:  # noqa: ARG002
        """Return the contained value, ignoring the fallback function."""
        return self.value

    def map[U](self, f: Callable[[T], U]) -> Some[U]:
        """Transform the contained value using the provided function."""
        return Some(f(self.value))

    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:  # noqa: ARG002
        """Apply the function to the contained value."""
        return f(self.value)

    def map_or_else[U](self, default_f: Callable[[], U], f: Callable[[T], U]) -> U:  # noqa: ARG002
        """Apply the function to the contained value."""
        return f(self.value)

    def and_then[U](self, f: Callable[[T], Option[U]]) -> Option[U]:
        """Call the function with the contained value and return its result.

        This is useful for chaining operations that may return Nothing.
        """
        return f(self.value)

    def or_else(self, f: Callable[[], Option[T]]) -> Some[T]:  # noqa: ARG002
        """Return self unchanged since this is a Some variant."""
        return self

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        """Return Some if predicate returns True, otherwise Nothing."""
        if predicate(self.value):
            return self
        return Nothing()

    def ok_or[E: Exception](self, err: E) -> Ok[T]:  # noqa: ARG002
        """Convert to Ok since this is a Some variant."""
        return Ok(self.value)

    def ok_or_else[E: Exception](self, err_f: Callable[[], E]) -> Ok[T]:  # noqa: ARG002
        """Convert to Ok since this is a Some variant."""
        return Ok(self.value)

    def zip[U](self, other: Option[U]) -> Option[tuple[T, U]]:
        """Combine with another Option into a tuple if both are Some."""
        if isinstance(other, Some):
            # cast needed due to ty limitation with generic dataclass inference
            return cast(Option[tuple[T, U]], Some((self.value, other.value)))
        return Nothing()


@final
@dataclass(frozen=True, slots=True)
class Nothing:
    """Nothing variant of Option, representing absence of a value."""

    def is_some(self) -> bool:
        """Check if this is a Some variant."""
        return False

    def is_nothing(self) -> bool:
        """Check if this is a Nothing variant."""
        return True

    def unwrap(self) -> NoReturn:
        """Raise UnwrapError since this is a Nothing variant."""
        raise UnwrapError("called `unwrap()` on a `Nothing` value")

    def expect(self, msg: str) -> NoReturn:
        """Raise UnwrapError with the provided message."""
        raise UnwrapError(msg)

    def unwrap_or[D](self, default: D) -> D:
        """Return the provided default value."""
        return default

    def unwrap_or_else[D](self, f: Callable[[], D]) -> D:
        """Call the provided function and return its result."""
        return f()

    def map[U](self, f: Callable[[Never], U]) -> Nothing:  # noqa: ARG002
        """Return self unchanged since this is a Nothing variant."""
        return self

    def map_or[U](self, default: U, f: Callable[[Never], U]) -> U:  # noqa: ARG002
        """Return the default value since this is a Nothing variant."""
        return default

    def map_or_else[U](self, default_f: Callable[[], U], f: Callable[[Never], U]) -> U:  # noqa: ARG002
        """Call the default function since this is a Nothing variant."""
        return default_f()

    def and_then[U](self, f: Callable[[Never], Option[U]]) -> Nothing:  # noqa: ARG002
        """Return self unchanged since this is a Nothing variant."""
        return self

    def or_else[T](self, f: Callable[[], Option[T]]) -> Option[T]:
        """Call the function and return its result.

        This is useful for providing an alternative when the Option is Nothing.
        """
        return f()

    def filter(self, predicate: Callable[[Never], bool]) -> Nothing:  # noqa: ARG002
        """Return Nothing since there is no value to filter."""
        return self

    def ok_or[E: Exception](self, err: E) -> Err[E]:
        """Convert to Err with the provided error."""
        return Err(err)

    def ok_or_else[E: Exception](self, err_f: Callable[[], E]) -> Err[E]:
        """Convert to Err by calling the provided function."""
        return Err(err_f())

    def zip[U](self, other: Option[U]) -> Nothing:  # noqa: ARG002
        """Return Nothing since this variant has no value to combine."""
        return self
