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

from typing import ClassVar, NoReturn, Self, dataclass_transform

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
from pydantic._internal._model_construction import ModelMetaclass
from pydantic_core import PydanticCustomError

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


def _frozen_setattr(self: BaseModel, name: str, value: object) -> NoReturn:
    """Prevent attribute assignment on frozen struct.

    Explicitly raises ValidationError with the same format as Pydantic's frozen instance error.
    The NoReturn annotation correctly indicates this function always raises.
    """
    raise ValidationError.from_exception_data(
        title=self.__class__.__name__,
        line_errors=[
            {
                "type": PydanticCustomError("frozen_instance", "Mutation of immutable struct"),
                "loc": (name,),
                "input": value,
            }
        ],
    )


# =============================================================================
# Metaclass
# =============================================================================


@dataclass_transform(kw_only_default=True)
class _StructMeta(ModelMetaclass):
    """Metaclass that configures Struct subclasses based on `mut` parameter.

    - `class Foo(Struct)` → immutable (frozen=True)
    - `class Foo(Struct, mut=True)` → mutable (frozen=False)

    All user classes inherit directly from Struct, so isinstance(obj, Struct) works.
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
        """Create struct class with appropriate mutability configuration."""
        # Check if this is the Struct base class itself (inherits from _StructBase)
        is_struct_base = name == "Struct" and any(getattr(b, "__name__", None) == "_StructBase" for b in bases)

        if not is_struct_base:
            # Configure mutability for user-defined classes
            namespace["__mutable__"] = mut
            namespace["model_config"] = ConfigDict(**_BASE_CONFIG, frozen=not mut)

            if mut:
                # Mutable structs use standard Pydantic setattr (allows assignment)
                namespace["__setattr__"] = BaseModel.__setattr__
            else:
                # Immutable structs use our explicit-raise setattr
                namespace["__setattr__"] = _frozen_setattr

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)  # ty:ignore[invalid-argument-type]

        # Set up pattern matching
        cls.__match_args__ = tuple(cls.model_fields.keys())  # ty:ignore[unresolved-attribute]

        return cls


# =============================================================================
# Base Class with Shared Methods
# =============================================================================


class _StructBase(BaseModel, RustType):
    """Internal base class with shared methods for all structs."""

    __mutable__: ClassVar[bool]  # Declared here, defined by metaclass

    @classmethod
    def is_mut(cls) -> bool:
        """Check if the struct is mutable."""
        return cls.__mutable__

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


# =============================================================================
# User-Facing Struct
# =============================================================================


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

    __mutable__: ClassVar[bool] = False
    __setattr__ = _frozen_setattr  # type: ignore[method-assign,assignment]

    model_config = ConfigDict(
        **_BASE_CONFIG,
        frozen=True,
    )
