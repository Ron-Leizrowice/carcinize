"""Fluent iterator combinators inspired by Rust's Iterator trait.

Provides chainable methods for transforming and consuming iterables:

- Transformations: `map`, `filter`, `filter_map`, `flat_map`, `flatten`
- Slicing: `take`, `skip`, `take_while`, `skip_while`, `step_by`
- Combining: `chain`, `zip`, `enumerate`
- Folding: `fold`, `reduce`, `sum`, `product`
- Searching: `find`, `find_map`, `position`, `any`, `all`, `count`
- Collecting: `collect_list`, `collect_set`, `collect_dict`
- Accessing: `first`, `last`, `nth`

All transformation methods return a new `Iter`, allowing method chaining.
Terminal methods consume the iterator and return a value.

```python
from carcinize.iter import Iter

result = (
    Iter([1, 2, 3, 4, 5])
    .filter(lambda x: x > 2)
    .map(lambda x: x * 2)
    .collect_list()
)  # [6, 8, 10]
```
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from typing import cast

from carcinize._base import RustType
from carcinize._option import Nothing, Option, Some


class Iter[T](RustType):
    """A fluent iterator wrapper with chainable combinators."""

    __slots__ = ("_iter",)

    def __init__(self, iterable: Iterable[T]) -> None:
        """Create an Iter from any iterable."""
        self._iter: Iterator[T] = iter(iterable)

    def __iter__(self) -> Iterator[T]:
        """Return the underlying iterator."""
        return self._iter

    def __next__(self) -> T:
        """Get the next item."""
        return next(self._iter)

    # =========================================================================
    # Transformations (return new Iter)
    # =========================================================================

    def map[U](self, f: Callable[[T], U]) -> Iter[U]:
        """Transform each element using the provided function."""
        return Iter(f(x) for x in self._iter)

    def filter(self, predicate: Callable[[T], bool]) -> Iter[T]:
        """Keep only elements that satisfy the predicate."""
        return Iter(x for x in self._iter if predicate(x))

    def filter_map[U](self, f: Callable[[T], Option[U]]) -> Iter[U]:
        """Apply function and keep only Some values, extracting the inner value."""

        def gen() -> Iterator[U]:
            for x in self._iter:
                match f(x):
                    case Some(value):
                        yield value
                    case Nothing():
                        pass

        return Iter(gen())

    def flat_map[U](self, f: Callable[[T], Iterable[U]]) -> Iter[U]:
        """Apply function and flatten the results."""
        return Iter(item for x in self._iter for item in f(x))

    def flatten[U](self: Iter[Iterable[U]]) -> Iter[U]:
        """Flatten nested iterables by one level."""
        return Iter(item for inner in self._iter for item in inner)

    def inspect(self, f: Callable[[T], object]) -> Iter[T]:
        """Call a function on each element, passing through unchanged. Useful for debugging."""

        def gen() -> Iterator[T]:
            for x in self._iter:
                f(x)
                yield x

        return Iter(gen())

    # =========================================================================
    # Slicing
    # =========================================================================

    def take(self, n: int) -> Iter[T]:
        """Take the first n elements."""
        return Iter(itertools.islice(self._iter, n))

    def skip(self, n: int) -> Iter[T]:
        """Skip the first n elements."""
        return Iter(itertools.islice(self._iter, n, None))

    def take_while(self, predicate: Callable[[T], bool]) -> Iter[T]:
        """Take elements while predicate is true."""
        return Iter(itertools.takewhile(predicate, self._iter))

    def skip_while(self, predicate: Callable[[T], bool]) -> Iter[T]:
        """Skip elements while predicate is true, then yield the rest."""
        return Iter(itertools.dropwhile(predicate, self._iter))

    def step_by(self, step: int) -> Iter[T]:
        """Yield every nth element, starting from the first."""
        if step < 1:
            raise ValueError("step must be positive")

        def gen() -> Iterator[T]:
            for i, x in enumerate(self._iter):
                if i % step == 0:
                    yield x

        return Iter(gen())

    # =========================================================================
    # Combining
    # =========================================================================

    def chain(self, other: Iterable[T]) -> Iter[T]:
        """Chain another iterable after this one."""
        return Iter(itertools.chain(self._iter, other))

    def zip[U](self, other: Iterable[U]) -> Iter[tuple[T, U]]:
        """Zip with another iterable into pairs."""
        return cast(Iter[tuple[T, U]], Iter(zip(self._iter, other, strict=False)))

    def zip_longest[U, D](self, other: Iterable[U], *, fillvalue: D) -> Iter[tuple[T | D, U | D]]:
        """Zip with another iterable, filling missing values."""
        return cast(Iter[tuple[T | D, U | D]], Iter(itertools.zip_longest(self._iter, other, fillvalue=fillvalue)))

    def enumerate(self, start: int = 0) -> Iter[tuple[int, T]]:
        """Pair each element with its index."""
        return cast(Iter[tuple[int, T]], Iter(enumerate(self._iter, start)))

    def interleave(self, other: Iterable[T]) -> Iter[T]:
        """Interleave elements from both iterables."""

        def gen() -> Iterator[T]:
            other_iter = iter(other)
            for a, b in zip(self._iter, other_iter, strict=False):
                yield a
                yield b

        return Iter(gen())

    def fold[U](self, init: U, f: Callable[[U, T], U]) -> U:
        """Fold elements into a single value using an accumulator."""
        acc = init
        for x in self._iter:
            acc = f(acc, x)
        return acc

    def reduce(self, f: Callable[[T, T], T]) -> Option[T]:
        """Reduce elements using a binary function. Returns Nothing if empty."""
        try:
            first = next(self._iter)
        except StopIteration:
            return Nothing()
        return Some(self.fold(first, f))

    def sum(self: Iter[int]) -> int:
        """Sum all elements (for numeric iterators)."""
        return sum(self._iter)

    def product(self: Iter[int]) -> int:
        """Multiply all elements (for numeric iterators)."""
        result = 1
        for x in self._iter:
            result *= x
        return result

    def find(self, predicate: Callable[[T], bool]) -> Option[T]:
        """Find the first element satisfying the predicate."""
        for x in self._iter:
            if predicate(x):
                return Some(x)
        return Nothing()

    def find_map[U](self, f: Callable[[T], Option[U]]) -> Option[U]:
        """Apply function and return the first Some value."""
        for x in self._iter:
            result = f(x)
            if isinstance(result, Some):
                return result
        return Nothing()

    def position(self, predicate: Callable[[T], bool]) -> Option[int]:
        """Find the index of the first element satisfying the predicate."""
        for i, x in enumerate(self._iter):
            if predicate(x):
                return Some(i)
        return Nothing()

    def any(self, predicate: Callable[[T], bool]) -> bool:
        """Check if any element satisfies the predicate."""
        return any(predicate(x) for x in self._iter)

    def all(self, predicate: Callable[[T], bool]) -> bool:
        """Check if all elements satisfy the predicate."""
        return all(predicate(x) for x in self._iter)

    def count(self) -> int:
        """Count the number of elements."""
        return sum(1 for _ in self._iter)

    def min(self) -> Option[T]:
        """Find the minimum element. Requires elements to be comparable."""
        items = self.collect_list()
        if not items:
            return Nothing()
        # Use key function to satisfy type checker; identity preserves ordering
        return Some(min(items, key=lambda x: cast(object, x)))  # type: ignore[type-var]

    def max(self) -> Option[T]:
        """Find the maximum element. Requires elements to be comparable."""
        items = self.collect_list()
        if not items:
            return Nothing()
        return Some(max(items, key=lambda x: cast(object, x)))  # type: ignore[type-var]

    def collect_list(self) -> list[T]:
        """Collect all elements into a list."""
        return list(self._iter)

    def collect_set(self) -> set[T]:
        """Collect all elements into a set."""
        return set(self._iter)

    def collect_dict[K, V](self: Iter[tuple[K, V]]) -> dict[K, V]:
        """Collect key-value pairs into a dict."""
        return dict(self._iter)  # type: ignore[arg-type]

    def collect_string(self: Iter[str], sep: str = "") -> str:
        """Join string elements with a separator."""
        return sep.join(self._iter)

    def first(self) -> Option[T]:
        """Get the first element."""
        try:
            return Some(next(self._iter))
        except StopIteration:
            return Nothing()

    def last(self) -> Option[T]:
        """Get the last element. Consumes the entire iterator."""
        last_item: Option[T] = Nothing()
        for x in self._iter:
            last_item = Some(x)
        return last_item

    def nth(self, n: int) -> Option[T]:
        """Get the nth element (0-indexed)."""
        try:
            return Some(next(itertools.islice(self._iter, n, n + 1)))
        except StopIteration:
            return Nothing()

    def partition(self, predicate: Callable[[T], bool]) -> tuple[list[T], list[T]]:
        """Split into two lists: (matches, non-matches)."""
        matches: list[T] = []
        non_matches: list[T] = []
        for x in self._iter:
            if predicate(x):
                matches.append(x)
            else:
                non_matches.append(x)
        return matches, non_matches

    def group_by[K](self, key: Callable[[T], K]) -> dict[K, list[T]]:
        """Group elements by a key function."""
        groups: dict[K, list[T]] = {}
        for x in self._iter:
            k = key(x)
            if k not in groups:
                groups[k] = []
            groups[k].append(x)
        return groups

    # =========================================================================
    # Sorting (terminal, returns list)
    # =========================================================================

    def sorted(self, *, reverse: bool = False) -> list[T]:
        """Collect and sort elements. Requires elements to be comparable."""
        items = self.collect_list()
        items.sort(key=lambda x: cast(object, x), reverse=reverse)  # type: ignore[misc]
        return items

    def sorted_by[K](self, key: Callable[[T], K], *, reverse: bool = False) -> list[T]:
        """Collect and sort elements by a key function."""
        items = self.collect_list()
        items.sort(key=lambda x: cast(object, key(x)), reverse=reverse)  # type: ignore[misc]
        return items

    # =========================================================================
    # Deduplication
    # =========================================================================

    def unique(self) -> Iter[T]:
        """Yield only unique elements, preserving order."""
        seen: set[T] = set()

        def gen() -> Iterator[T]:
            for x in self._iter:
                if x not in seen:
                    seen.add(x)
                    yield x

        return Iter(gen())

    def unique_by[K](self, key: Callable[[T], K]) -> Iter[T]:
        """Yield elements with unique keys, preserving order."""
        seen: set[K] = set()

        def gen() -> Iterator[T]:
            for x in self._iter:
                k = key(x)
                if k not in seen:
                    seen.add(k)
                    yield x

        return Iter(gen())
