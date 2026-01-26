# 🦀 Carcinize 🦀

Born to write Rust but forced to use Python to survive?
Learned Rust for fun and now all the missing language features are driving you crazy?
Want to look really p̶r̶e̶t̶e̶n̶t̶i̶o̶u̶s̶ cool in front of your coworkers?

Try Carcinization! 🦀 🦀 🦀

## What's up with the name?

[Carcinization](https://en.wikipedia.org/wiki/Carcinisation) is the tendency for convergent evolution of many species to eventually become crabs, or as Lancelot Alexander Borradaile put it: "the many attempts of Nature to evolve a crab".

As I have fully drunk the Rust kool-aid, I am now fully committed to the idea that everything should be a crab. Including Python.

Another equally valid interpretation is it's a verb form of "carcinogen", because using this library will give your Python projects cancer. Who's to say which is correct?

## Installation

Install with uv:

```bash
uv add carcinize
```

No I won't add examples for `pip`, `poetry`, or god forbid `conda`. It's 2026, grow up and use `uv`.

Requires Python 3.12+.

## Features

### Result

A type representing either success (`Ok`) or failure (`Err`). Errors must be `Exception` subclasses, so `unwrap()` raises the actual error.

```python
from carcinize import Ok, Err, Result

def divide(a: int, b: int) -> Result[float, ZeroDivisionError]:
    if b == 0:
        return Err(ZeroDivisionError("cannot divide by zero"))
    return Ok(a / b)

# Pattern matching
match divide(10, 2):
    case Ok(value):
        print(f"Result: {value}")
    case Err(error):
        print(f"Error: {error}")

# Method chaining
result = (
    divide(10, 2)
    .map(lambda x: x * 2)
    .unwrap_or(0.0)
)
```

**Methods:** `is_ok()`, `is_err()`, `ok()`, `err()`, `unwrap()`, `unwrap_err()`, `expect()`, `expect_err()`, `unwrap_or()`, `unwrap_or_else()`, `map()`, `map_err()`, `map_or()`, `map_or_else()`, `and_then()`, `or_else()`

**Converting to Option:** The `ok()` and `err()` methods return `Option` types, matching Rust's API:

- `Ok(v).ok()` returns `Some(v)`, `Ok(v).err()` returns `Nothing()`
- `Err(e).ok()` returns `Nothing()`, `Err(e).err()` returns `Some(e)`

**Type variance:** Both `Ok[T]` and `Err[E]` are covariant in their type parameters, meaning `Ok[Subclass]` is a subtype of `Ok[Superclass]` and `Err[SubException]` is a subtype of `Err[SuperException]`. This is safe because both types are immutable, and matches Rust's `Result<T, E>` which is covariant in both type parameters.

### Option

A type representing an optional value (`Some` or `Nothing`). We use `Nothing` instead of `None` to avoid confusion with Python's `None`.

```python
from carcinize import Some, Nothing, Option

def find_user(id: int) -> Option[str]:
    users = {1: "alice", 2: "bob"}
    if id in users:
        return Some(users[id])
    return Nothing()

# Pattern matching
match find_user(1):
    case Some(name):
        print(f"Found: {name}")
    case Nothing():
        print("User not found")

# Method chaining
name = find_user(1).map(str.upper).unwrap_or("anonymous")
```

**Methods:** `is_some()`, `is_nothing()`, `unwrap()`, `expect()`, `unwrap_or()`, `unwrap_or_else()`, `map()`, `map_or()`, `map_or_else()`, `and_then()`, `or_else()`, `filter()`, `ok_or()`, `ok_or_else()`, `zip()`

**Type variance:** `Some[T]` is covariant in `T`, meaning `Some[Subclass]` is a subtype of `Some[Superclass]`. This is safe because `Some` is immutable, and matches Rust's `Option<T>` which is covariant in T.

### Struct

Pydantic-based structs with Rust-like semantics. Immutable by default, use `mut=True` for mutable structs.

```python
from carcinize import Struct

# Immutable by default (like Rust's default)
class User(Struct):
    name: str
    age: int

user = User(name="Alice", age=30)
# user.age = 31  # ValidationError: frozen instance

# Mutable when you need it
class Config(Struct, mut=True):
    host: str
    port: int = 8080

config = Config(host="localhost")
config.port = 9000  # OK - mutable

# Pattern matching (via __match_args__)
match user:
    case User(name, age) if age >= 18:
        print(f"{name} is an adult")
    case User(name, _):
        print(f"{name} is a minor")

# Functional updates (like Rust's struct update syntax)
updated_user = user.replace(age=31)  # Returns new instance

# Safe parsing with Result
match User.try_from({"name": "Bob", "age": 25}):
    case Ok(u):
        print(u.name)
    case Err(validation_error):
        print(validation_error)

# Also accepts JSON strings
User.try_from('{"name": "Charlie", "age": 35}')
```

**Features:**

- Extra fields forbidden
- Strict type validation (no coercion)
- Pattern matching support via `__match_args__`
- Functional updates via `replace()`
- `__mutable__` class variable for runtime checks
- Immutable structs are hashable (can be used in sets/dicts)

**Methods:** `try_from()`, `replace()`, `clone()`, `as_dict()`, `as_json()`

**Rust-like immutability:** Like Rust, immutability applies to the *binding*, not the type. An immutable struct can contain fields of any type - you just can't mutate them through the immutable binding:

```python
class Inner(Struct, mut=True):
    value: int

class Outer(Struct):  # Immutable
    inner: Inner

outer = Outer(inner=Inner(value=42))
# outer.inner = Inner(value=100)  # Error: can't mutate through outer
# But inner itself is a mutable type - this is fine, just like Rust
```

### Iter

Fluent iterator with chainable combinators, inspired by Rust's `Iterator` trait.

```python
from carcinize import Iter

# Chain transformations
result = (
    Iter([1, 2, 3, 4, 5])
    .filter(lambda x: x > 2)
    .map(lambda x: x * 2)
    .collect_list()
)  # [6, 8, 10]

# Find with Option
first_even = Iter([1, 3, 4, 5]).find(lambda x: x % 2 == 0)  # Some(4)

# Fold/reduce
total = Iter([1, 2, 3]).fold(0, lambda acc, x: acc + x)  # 6

# Clone an iterator
it = Iter([1, 2, 3])
cloned = it.clone()  # Independent copy
```

**Transformations:** `map()`, `filter()`, `filter_map()`, `flat_map()`, `flatten()`, `inspect()`

**Slicing:** `take()`, `skip()`, `take_while()`, `skip_while()`, `step_by()`

**Combining:** `chain()`, `zip()`, `zip_longest()`, `enumerate()`, `interleave()`

**Folding:** `fold()`, `reduce()`, `sum()`, `product()`

**Searching:** `find()`, `find_map()`, `position()`, `any()`, `all()`, `count()`, `min()`, `max()`

**Collecting:** `collect_list()`, `collect_set()`, `collect_dict()`, `collect_string()`, `partition()`, `group_by()`

**Accessing:** `first()`, `last()`, `nth()`

**Sorting:** `sorted()`, `sorted_by()`

**Deduplication:** `unique()`, `unique_by()`

### Lazy / OnceCell

Thread-safe lazy initialization primitives.

```python
from carcinize import Lazy, OnceCell

# Lazy - computed on first access
expensive_config = Lazy(lambda: load_config_from_disk())
# ... nothing computed yet ...
config = expensive_config.get()  # computed once, cached forever

# Check without triggering computation
expensive_config.is_computed()  # True (after first get())
expensive_config.get_if_computed()  # Some(config) or Nothing

# OnceCell - write exactly once
cell: OnceCell[int] = OnceCell()
cell.get()      # Nothing
cell.set(42)    # Ok(None)
cell.get()      # Some(42)
cell.set(100)   # Err(OnceCellAlreadyInitializedError)

# Initialize with a function if not already set
value = cell.get_or_init(lambda: compute_default())

# Take the value out, resetting the cell
taken = cell.take()  # Some(42), cell is now empty
```

Both are thread-safe and use double-checked locking for performance.

## Error Handling

All unwrap operations raise `UnwrapError` (importable from `carcinize`) when they fail on the wrong variant:

```python
from carcinize import Nothing, Err, UnwrapError

try:
    Nothing().unwrap()
except UnwrapError as e:
    print(e)  # "called `unwrap()` on a `Nothing` value"

# Err.unwrap() raises the contained error directly
try:
    Err(ValueError("oops")).unwrap()
except ValueError as e:
    print(e)  # "oops"
```

## Type Checking

This library is fully typed and works with type checkers like `mypy`, `pyright`, and `ty`. The `Struct` class uses `@dataclass_transform` to ensure proper type inference for fields.

## License

WTFPL - Do What The F*ck You Want To Public License. Because life's too short for licensing drama.
