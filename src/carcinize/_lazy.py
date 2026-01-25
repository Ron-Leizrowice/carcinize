"""Thread-safe lazy initialization primitives.

Provides `OnceCell` and `Lazy` for deferred computation:

- `OnceCell[T]`: A cell that can be written to exactly once (thread-safe)
- `Lazy[T]`: A value computed on first access (thread-safe)

These are useful for expensive computations that should only happen once,
or for breaking initialization cycles.

```python
from carcinize.lazy import Lazy, OnceCell

# Lazy - computed on first access
config = Lazy(lambda: load_expensive_config())
# ... later, computed only once ...
value = config.get()

# OnceCell - set exactly once
cell: OnceCell[int] = OnceCell()
cell.set(42)  # Ok(None)
cell.set(100)  # Err - already set
cell.get()  # Some(42)
```
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import cast

from carcinize._option import Nothing, Option, Some
from carcinize._result import Err, Ok, Result


class OnceCellAlreadyInitializedError(Exception):
    """Raised when attempting to set an already-initialized OnceCell."""


class OnceCell[T]:
    """A thread-safe cell that can be written to exactly once.

    Once a value is set, it cannot be changed. All reads after initialization
    return the same value.
    """

    __slots__ = ("_initialized", "_lock", "_value")

    def __init__(self) -> None:
        """Create an empty OnceCell."""
        self._value: T | None = None
        self._initialized: bool = False
        self._lock: threading.Lock = threading.Lock()

    def get(self) -> Option[T]:
        """Get the value if initialized, otherwise Nothing.

        Thread-safe: can be called concurrently.
        """
        # Fast path: no lock needed for reads after initialization
        if self._initialized:
            # cast: we know _value is T when _initialized is True
            return Some(cast(T, self._value))
        return Nothing()

    def get_or_init(self, f: Callable[[], T]) -> T:
        """Get the value, initializing with f() if not yet set.

        Thread-safe: if multiple threads call this concurrently,
        only one will execute `f()`, and all will receive the same value.
        """
        # Fast path: already initialized
        if self._initialized:
            # cast: we know _value is T when _initialized is True
            return cast(T, self._value)

        with self._lock:
            # Double-check after acquiring lock
            if not self._initialized:
                self._value = f()
                self._initialized = True
            return cast(T, self._value)

    def set(self, value: T) -> Result[None, OnceCellAlreadyInitializedError]:
        """Set the value. Returns Err if already initialized.

        Thread-safe: only one thread can successfully set the value.
        """
        with self._lock:
            if self._initialized:
                return Err(OnceCellAlreadyInitializedError("OnceCell is already initialized"))
            self._value = value
            self._initialized = True
            return Ok(None)

    def is_initialized(self) -> bool:
        """Check if the cell has been initialized.

        Thread-safe.
        """
        return self._initialized

    def take(self) -> Option[T]:
        """Take the value out of the cell, leaving it uninitialized.

        Returns Nothing if the cell was not initialized.
        Thread-safe.
        """
        with self._lock:
            if not self._initialized:
                return Nothing()
            # cast: we know _value is T when _initialized is True
            value = cast(T, self._value)
            self._value = None
            self._initialized = False
            return Some(value)

    def __repr__(self) -> str:
        """Return a string representation."""
        if self._initialized:
            return f"OnceCell({self._value!r})"
        return "OnceCell(<uninitialized>)"


class Lazy[T]:
    """A thread-safe lazily-initialized value.

    The initialization function is called at most once, on first access.
    All subsequent accesses return the cached value.
    """

    __slots__ = ("_computed", "_init", "_lock", "_value")

    def __init__(self, init: Callable[[], T]) -> None:
        """Create a Lazy value with the given initialization function.

        The function will not be called until `get()` is invoked.
        """
        self._init: Callable[[], T] = init
        self._value: T | None = None
        self._computed: bool = False
        self._lock: threading.Lock = threading.Lock()

    def get(self) -> T:
        """Get the value, computing it on first access.

        Thread-safe: if multiple threads call this concurrently,
        only one will execute the init function, and all will receive
        the same value.
        """
        # Fast path: already computed
        if self._computed:
            # cast: we know _value is T when _computed is True
            return cast(T, self._value)

        with self._lock:
            # Double-check after acquiring lock
            if not self._computed:
                self._value = self._init()
                self._computed = True
            return cast(T, self._value)

    def is_computed(self) -> bool:
        """Check if the value has been computed.

        Thread-safe.
        """
        return self._computed

    def get_if_computed(self) -> Option[T]:
        """Get the value only if already computed, without triggering computation."""
        if self._computed:
            # cast: we know _value is T when _computed is True
            return Some(cast(T, self._value))
        return Nothing()

    def __repr__(self) -> str:
        """Return a string representation."""
        if self._computed:
            return f"Lazy({self._value!r})"
        return "Lazy(<not computed>)"
