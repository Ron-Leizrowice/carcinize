"""A crude attempt at replicating Rust's `Result` type in Python.

Provides `Ok[T]` and `Err[E]` variants with common Rust-like methods:

- Inspection: `is_ok()`, `is_err()`
- Extraction: `ok()`, `err()`, `unwrap()`, `unwrap_err()`, `expect()`, `expect_err()`
- Fallbacks: `unwrap_or()`, `unwrap_or_else()`
- Transformation: `map()`, `map_err()`, `and_then()`, `or_else()`
- Context: `context()`, `note()` for adding debugging context

Pattern matching is supported via `match/case`:

    match result:
        case Ok(value):
            print(f"Success: {value}")
        case Err(error):
            print(f"Error: {error}")

Enhanced Error Context:
    The `Err` type automatically captures where it was created, providing rich
    debugging information when errors are unwrapped. When `unwrap()` is called,
    the traceback shows both where the error originated and where it was unwrapped.

    Additional context can be added via `.context()` and `.note()` methods:

        result.context("while fetching user data").unwrap()
        result.note("user_id was 42").unwrap()

Type Variance:
    Both `Ok[T]` and `Err[E]` are covariant in their type parameters:
    - `Ok[Subclass]` is a subtype of `Ok[Superclass]`
    - `Err[SubException]` is a subtype of `Err[SuperException]`

    This is safe because both types are immutable (frozen dataclasses) and matches
    Rust's behavior where `Result<T, E>` is covariant in both type parameters.

"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Generic, Never, NoReturn, TypeVar, final

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

# =============================================================================
# Error Context Tracking
# =============================================================================


def _capture_origin(skip_frames: int = 2) -> str:
    """Capture the current call location for error origin tracking.

    Args:
        skip_frames: Number of frames to skip (to exclude internal calls).

    Returns:
        A formatted string showing file, line, and code context.
    """
    # Get the stack, excluding this function and the specified frames
    stack = traceback.extract_stack()[: -skip_frames - 1]
    if not stack:
        return "<unknown origin>"

    # Get the most recent frame (where Err was created)
    frame = stack[-1]
    return f"{frame.filename}:{frame.lineno} in {frame.name}\n    {frame.line or '<no source>'}"


def _format_traceback_location(tb: TracebackType) -> str | None:
    """Format a traceback's last frame as a location string."""
    tb_list = traceback.extract_tb(tb)
    if not tb_list:
        return None
    last_frame = tb_list[-1]
    return (
        f"Exception was originally raised at:\n"
        f"  {last_frame.filename}:{last_frame.lineno} in {last_frame.name}\n"
        f"    {last_frame.line or '<no source>'}"
    )


# =============================================================================
# Ok Type
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class Ok(Generic[T_co]):  # noqa: UP046 - Generic required for covariance
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

    def context(self, msg: str) -> Ok[T_co]:  # noqa: ARG002
        """Return self unchanged since this is an Ok variant.

        This method exists for API consistency with Err, allowing you to add
        context to a Result without checking if it's Ok or Err first.

        Example:
            result.context("while processing user").unwrap()
        """
        return self

    def note(self, msg: str) -> Ok[T_co]:  # noqa: ARG002
        """Return self unchanged since this is an Ok variant.

        This method exists for API consistency with Err, allowing you to add
        notes to a Result without checking if it's Ok or Err first.

        Example:
            result.note(f"user_id={user_id}").unwrap()
        """
        return self


# =============================================================================
# Err Type
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class Err(Generic[E_co]):  # noqa: UP046 - Generic required for covariance
    """Error variant of Result, containing an error of type E (covariant).

    The error type E must be an Exception subclass, allowing unwrap() to raise it directly.

    Enhanced Error Context:
        When an Err is created, it automatically captures where in the code it was
        created (the "origin"). When the error is later unwrapped, this origin
        information is added to the exception as a note, making debugging much easier.

        If the exception was caught from a try/except block (has an existing traceback),
        that information is also preserved and shown.

    Attributes:
        error: The contained exception.
        origin: Where this Err was created (file, line, function).
        original_traceback: The exception's traceback if it was caught (not newly created).

    Example:
        def fetch_user(id: int) -> Result[User, ValueError]:
            if id < 0:
                return Err(ValueError("invalid user id"))  # Origin captured here
            return Ok(User(id))

        # Later, when unwrapped, the traceback will show:
        # 1. Where unwrap() was called
        # 2. A note showing where the Err was originally created
    """

    error: E_co
    # These fields store context but don't affect equality/hashing (only error matters)
    _origin: str = field(default="", init=False, repr=False, compare=False, hash=False)
    _original_tb: TracebackType | None = field(default=None, init=False, repr=False, compare=False, hash=False)
    _notes_applied: bool = field(default=False, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Capture origin and preserve traceback on Err creation."""
        # Capture where this Err was created
        # Skip 2 frames: __post_init__ and __init__
        object.__setattr__(self, "_origin", _capture_origin(skip_frames=2))

        # If the exception already has a traceback (was caught), preserve it
        if self.error.__traceback__ is not None:
            object.__setattr__(self, "_original_tb", self.error.__traceback__)

    @property
    def origin(self) -> str:
        """The code location where this Err was created."""
        return self._origin

    @property
    def original_traceback(self) -> TracebackType | None:
        """The original traceback if the exception was caught from a try/except."""
        return self._original_tb

    def _apply_notes_to_exception(self) -> None:
        """Add context notes to the exception before raising (idempotent)."""
        if self._notes_applied:
            return
        object.__setattr__(self, "_notes_applied", True)

        # Add origin note
        if self._origin:
            origin_note = f"Error originated at:\n  {self._origin}"
            existing_notes = getattr(self.error, "__notes__", None) or []
            if origin_note not in existing_notes:
                self.error.add_note(origin_note)

        # Add original traceback note if we have one
        if self._original_tb:
            tb_note = _format_traceback_location(self._original_tb)
            if tb_note:
                existing_notes = getattr(self.error, "__notes__", None) or []
                if tb_note not in existing_notes:
                    self.error.add_note(tb_note)

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
        """Raise the contained error with enhanced context.

        The raised exception will include notes showing:
        - Where the Err was originally created
        - Where the exception was originally raised (if caught from try/except)

        Since E is bounded by Exception, this directly raises self.error.
        """
        self._apply_notes_to_exception()
        raise self.error

    def unwrap_err(self) -> E_co:
        """Return the contained error.

        Unlike Ok.unwrap_err(), this never raises.
        """
        return self.error

    def expect(self, msg: str) -> NoReturn:
        """Raise the contained error, chained from an UnwrapError with the provided message.

        The raised exception will include context notes showing where the Err originated.
        """
        self._apply_notes_to_exception()
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
        """Transform the contained error using the provided function.

        Note: The new error will get fresh origin tracking from where map_err is called.
        """
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

    def context(self, msg: str) -> Err[E_co]:
        """Add context to the error, similar to Rust's anyhow::Context.

        This adds a note to the exception that will be shown in the traceback
        when the error is unwrapped. Useful for adding high-level context about
        what operation was being performed when the error occurred.

        Example:
            def fetch_user(id: int) -> Result[User, IOError]:
                return (
                    read_from_database(id)
                    .context(f"while fetching user {id}")
                )

        Args:
            msg: A message describing the context (what was being done).

        Returns:
            Self, for method chaining.
        """
        self.error.add_note(f"Context: {msg}")
        return self

    def note(self, msg: str) -> Err[E_co]:
        """Add a debugging note to the error.

        This adds a note to the exception that will be shown in the traceback
        when the error is unwrapped. Useful for adding variable values or
        other debugging information.

        Example:
            def process_item(item: Item) -> Result[Output, ProcessError]:
                return (
                    do_processing(item)
                    .note(f"item_id={item.id}")
                    .note(f"item_type={item.type}")
                )

        Args:
            msg: A debugging note to add.

        Returns:
            Self, for method chaining.
        """
        self.error.add_note(msg)
        return self


# =============================================================================
# Utility Functions
# =============================================================================


def try_except[T](
    f: Callable[[], T],
    *exception_types: type[Exception],
) -> Result[T, Exception]:
    """Execute a function and capture any exceptions as an Err.

    This is a convenience function for converting exception-throwing code
    into Result-returning code.

    Note:
        The return type is `Result[T, Exception]` because Python's type system
        cannot express "the union of all types in *exception_types". Use pattern
        matching on the error to narrow to specific exception types:

            match try_except(risky_op, ValueError, TypeError):
                case Ok(value): ...
                case Err(ValueError() as e): ...
                case Err(TypeError() as e): ...

    Example:
        result = try_except(lambda: int("not a number"), ValueError)
        # Returns Err(ValueError("invalid literal for int()..."))

        result = try_except(lambda: 42, ValueError)
        # Returns Ok(42)

    Args:
        f: The function to execute.
        *exception_types: The exception types to catch. If none specified,
                         catches all Exceptions.

    Returns:
        Ok(result) if the function succeeds, Err(exception) if it raises.
    """
    catch = exception_types if exception_types else (Exception,)
    try:
        return Ok(f())
    except catch as e:
        # Type narrowing: catch tuple only contains Exception subclasses
        # The type checker sees BaseException, but we know it's Exception
        if not isinstance(e, Exception):
            raise  # Impossible branch, satisfies type checker
        return Err(e)
