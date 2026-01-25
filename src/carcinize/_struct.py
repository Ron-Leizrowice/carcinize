"""Rust-like structs for Python, built on Pydantic.

Use the `mut` parameter to control mutability:

    class User(Struct):  # Immutable by default (like Rust)
        name: str
        age: int

    class MutableUser(Struct, mut=True):  # Mutable
        name: str
        age: int

Both include:
    - Extra fields forbidden
    - Strict type validation (no coercion)
    - Pattern matching support via __match_args__
    - Functional updates via replace()
    - Serialization via as_dict() and as_json()
    - Parsing via try_from()

"""

from __future__ import annotations

from typing import ClassVar, Final, NoReturn, Self, dataclass_transform

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
from pydantic._internal._model_construction import ModelMetaclass

from carcinize._base import RustType
from carcinize._result import Err, Ok, Result

# =============================================================================
# Base Configuration
# =============================================================================

_BASE_CONFIG: dict[str, object] = {
    "extra": "forbid",
    "strict": True,
    "validate_default": True,
    "validate_assignment": True,
    "use_enum_values": True,
}


# =============================================================================
# Internal Base Classes
# =============================================================================


class _StructBase(BaseModel, RustType):
    """Internal base class with shared methods for all structs."""

    __mutable__: ClassVar[bool]  # Declared here, defined in subclasses

    def replace(self, **changes: object) -> Self:
        """Return a new instance with the specified fields replaced.

        Similar to Rust's struct update syntax: `Point { x: 5, ..point }`

        Example:
            updated = user.replace(age=user.age + 1)

        """
        current = self.model_dump()
        current.update(changes)
        return self.model_validate(current)

    @classmethod
    def try_from[R: JsonValue](cls, data: R) -> Result[Self, ValidationError | TypeError]:
        """Try to validate the data and return a Result.

        Accepts either a dict or a JSON string. Returns Err(TypeError) for other input types.
        """
        try:
            match data:
                case dict(d):
                    return Ok(cls.model_validate(d, strict=False))
                case str(s):
                    return Ok(cls.model_validate_json(s, strict=False))
                case _:
                    return Err(TypeError(f"Expected dict or JSON string, got {type(data).__name__}"))
        except ValidationError as e:
            return Err(e)

    def as_dict(self) -> dict[str, JsonValue]:
        """Return the struct as a dictionary."""
        return self.model_dump()

    def as_json(self) -> str:
        """Return the struct as a JSON string."""
        return self.model_dump_json()


class MutStruct(_StructBase):
    """Mutable struct base class (internal, use `Struct` with `mut=True`)."""

    __mutable__: ClassVar[Final[bool]] = True

    model_config = ConfigDict(
        **_BASE_CONFIG,
        frozen=False,
    )


def _frozen_setattr(self: BaseModel, name: str, value: object) -> NoReturn:  # ty: ignore[invalid-return-type]
    """Prevent attribute assignment on frozen struct.

    The NoReturn annotation tells type checkers that assignment always fails.
    We call BaseModel.__setattr__ which will raise ValidationError for frozen models.
    """
    BaseModel.__setattr__(self, name, value)


class FrozenStruct(_StructBase):
    """Immutable struct base class (internal, use `Struct` without `mut`).

    Like Rust, immutability applies to the binding - nested fields of any type
    are allowed, but cannot be mutated through this immutable struct.
    """

    __mutable__: ClassVar[Final[bool]] = False
    __setattr__ = _frozen_setattr  # type: ignore[method-assign,assignment]

    model_config = ConfigDict(
        **_BASE_CONFIG,
        frozen=True,
    )


# =============================================================================
# User-Facing Struct with Metaclass
# =============================================================================


@dataclass_transform(kw_only_default=True)
class _StructMeta(ModelMetaclass):
    """Metaclass that routes Struct subclasses based on `mut` parameter.

    - `class Foo(Struct)` → inherits from FrozenStruct (immutable)
    - `class Foo(Struct, mut=True)` → inherits from MutStruct (mutable)
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        *,
        mut: bool = False,
        **kwargs: object,
    ) -> type:
        """Create struct class, routing to appropriate base."""

        # Swap Struct for the appropriate internal base
        # Use name check to avoid forward reference issues
        def swap_base(base: type) -> type:
            if base.__name__ == "Struct" and base.__module__ == "carcinize._struct":
                return MutStruct if mut else FrozenStruct
            return base

        new_bases = tuple(swap_base(base) for base in bases)
        cls = super().__new__(mcs, name, new_bases, namespace, **kwargs)  # ty:ignore[invalid-argument-type]

        # Set up pattern matching
        cls.__match_args__ = tuple(cls.model_fields.keys())  # ty:ignore[unresolved-attribute]

        return cls


class Struct(_StructBase, metaclass=_StructMeta):
    """Struct with Rust-like semantics.

    Immutable by default. Use `mut=True` for a mutable struct:

        class User(Struct):  # Immutable (default)
            name: str
            age: int

        class MutableUser(Struct, mut=True):  # Mutable
            name: str
            age: int

    Features:
        - Extra fields forbidden
        - Strict type validation (no coercion)
        - Pattern matching: `match user: case User(name, age):`
        - Functional updates: `user.replace(age=31)`
        - Serialization: `as_dict()`, `as_json()`
        - Parsing: `try_from(data)`

    """

    __mutable__: ClassVar[Final[bool]] = False

    model_config = ConfigDict(
        **_BASE_CONFIG,
        frozen=True,
    )
