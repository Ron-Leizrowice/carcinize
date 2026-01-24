"""A crude attempt at replicating some of Rust's `struct` features in python.

Pydantic did 99% of the work for us, but we add some opinionated defaults and features:

- Extra fields are forbidden by default.
- Strict mode for validation.
- Default values are validated against the schema.
- Fields are validated after assignment.
- Enum values are used for enum fields.

There is a mutable and an immutable version. Use them accordingly.

"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError

from carcinize.result import Err, Ok, Result


class MutStruct(BaseModel):
    """Struct with mutable fields."""

    model_config = ConfigDict(
        extra="forbid",  # Any extra fields passed to the constructor will trigger validation error
        strict=True,  # Strict mode for validation
        validate_default=True,  # Validate default values against the schema
        validate_assignment=True,  # Re-validate fields after assignment
        use_enum_values=True,  # Use enum values for enum fields
    )

    @classmethod
    def try_from[R: JsonValue](cls, data: R) -> Result[Self, ValidationError | TypeError]:
        """Try to validate the data and return a Result.

        Accepts either a dict or a JSON string. Returns Err(TypeError) for other input types.
        """
        try:
            match data:
                case dict(d):
                    return Ok(cls.model_validate(d))
                case str(s):
                    return Ok(cls.model_validate_json(s))
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

    def clone(self) -> Self:
        """Create a deep copy of this struct."""
        return self.model_copy(deep=True)


class Struct(MutStruct):
    """A immutable struct."""

    model_config = ConfigDict(
        **MutStruct.model_config,
        frozen=True,  # Make the model immutable and hashable
    )

    @model_validator(mode="after")
    def _validate_immutability(self) -> Self:
        """Validate that nested BaseModel or dataclass fields are also immutable."""
        errors: list[MutabilityError] = []

        for name, value in self:
            if value is None:
                continue

            if isinstance(value, MutStruct) and not value.model_config.get("frozen", False):
                errors.append(MutableStructFieldError(name, value))
            elif isinstance(value, BaseModel) and not value.model_config.get("frozen", False):
                errors.append(MutableBaseModelFieldError(name, value))
            elif is_dataclass(value) and not isinstance(value, type) and not type(value).__dataclass_params__.frozen:
                errors.append(MutableDataclassFieldError(name, value))

        if errors:
            raise ValidationError.from_exception_data(
                title=type(self).__name__,
                line_errors=[e.as_init_error_details() for e in errors],
            )

        return self


# =============================================================================
# Errors
# =============================================================================


class MutabilityError:
    """Base class for immutability validation errors.

    Subclasses override `as_init_error_details()` to provide their specific error.
    """

    def __init__(self, field: str, value: object) -> None:
        self.field = field
        self.value = value

    def as_init_error_details(self) -> InitErrorDetails:
        """Convert to Pydantic's InitErrorDetails format."""
        raise NotImplementedError


class MutableStructFieldError(MutabilityError):
    """Field is a MutStruct but is not frozen."""

    def as_init_error_details(self) -> InitErrorDetails:
        """Return error details for a mutable MutStruct field."""
        return {
            "type": PydanticCustomError(
                "mutable_struct_field",
                "Field '{field}' is a MutStruct but must be frozen. "
                "Change it to inherit from Struct instead of MutStruct.",
                {"field": self.field},
            ),
            "loc": (self.field,),
            "input": self.value,
        }


class MutableBaseModelFieldError(MutabilityError):
    """Field is a BaseModel but is not frozen."""

    def as_init_error_details(self) -> InitErrorDetails:
        """Return error details for a mutable BaseModel field."""
        return {
            "type": PydanticCustomError(
                "mutable_basemodel_field",
                "Field '{field}' is a BaseModel but is not frozen. Set `frozen=True` in its ConfigDict, or use Struct.",
                {"field": self.field},
            ),
            "loc": (self.field,),
            "input": self.value,
        }


class MutableDataclassFieldError(MutabilityError):
    """Field is a dataclass but is not frozen."""

    def as_init_error_details(self) -> InitErrorDetails:
        """Return error details for a mutable dataclass field."""
        return {
            "type": PydanticCustomError(
                "mutable_dataclass_field",
                "Field '{field}' is a dataclass but is not frozen. Add `frozen=True` to the @dataclass decorator.",
                {"field": self.field},
            ),
            "loc": (self.field,),
            "input": self.value,
        }
