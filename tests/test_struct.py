"""Tests for the struct module."""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from carcinize.result import Err, Ok
from carcinize.struct import MutStruct, Struct

# =============================================================================
# Test Fixtures (helper classes)
# =============================================================================


class SimpleMutStruct(MutStruct):
    """A simple mutable struct for testing."""

    name: str
    age: int


class SimpleStruct(Struct):
    """A simple immutable struct for testing."""

    name: str
    age: int


class NestedStruct(Struct):
    """A struct containing another struct."""

    inner: SimpleStruct
    label: str


class MutableInner(MutStruct):
    """A mutable inner struct (not frozen)."""

    value: int


class MutableBaseModel(BaseModel):
    """A mutable BaseModel (not frozen)."""

    value: int


class FrozenBaseModel(BaseModel):
    """A frozen BaseModel."""

    model_config = ConfigDict(frozen=True)
    value: int


@dataclass
class MutableDataclass:
    """A mutable dataclass."""

    value: int


@dataclass(frozen=True)
class FrozenDataclass:
    """A frozen dataclass."""

    value: int


# =============================================================================
# MutStruct Tests
# =============================================================================


class TestMutStructConstruction:
    """Test MutStruct construction and validation."""

    def test_valid_construction(self) -> None:
        """Construct a MutStruct with valid data."""
        s = SimpleMutStruct(name="Alice", age=30)
        assert s.name == "Alice"
        assert s.age == 30

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMutStruct(name="Alice", age=30, extra="not allowed")  # ty: ignore[unknown-argument]

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_strict_mode_rejects_coercion(self) -> None:
        """Strict mode should reject type coercion (e.g., str to int)."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMutStruct(name="Alice", age="30")  # ty: ignore[invalid-argument-type]

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "int_type"

    def test_validate_assignment(self) -> None:
        """Assignment should be validated."""
        s = SimpleMutStruct(name="Alice", age=30)
        s.age = 31  # Valid assignment
        assert s.age == 31

        with pytest.raises(ValidationError):
            s.age = "not an int"  # ty: ignore[invalid-assignment]


class TestMutStructMethods:
    """Test MutStruct helper methods."""

    def test_try_from_dict_success(self) -> None:
        """try_from() with valid dict should return Ok."""
        result = SimpleMutStruct.try_from({"name": "Bob", "age": 25})
        assert isinstance(result, Ok)
        assert result.value.name == "Bob"
        assert result.value.age == 25

    def test_try_from_dict_failure(self) -> None:
        """try_from() with invalid dict should return Err."""
        result = SimpleMutStruct.try_from({"name": "Bob"})  # Missing age
        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)

    def test_try_from_json_success(self) -> None:
        """try_from() with valid JSON string should return Ok."""
        result = SimpleMutStruct.try_from('{"name": "Charlie", "age": 35}')
        assert isinstance(result, Ok)
        assert result.value.name == "Charlie"
        assert result.value.age == 35

    def test_try_from_json_failure(self) -> None:
        """try_from() with invalid JSON should return Err."""
        result = SimpleMutStruct.try_from('{"name": "Charlie"}')  # Missing age
        assert isinstance(result, Err)

    def test_try_from_invalid_type(self) -> None:
        """try_from() with invalid type should return Err(TypeError)."""
        result = SimpleMutStruct.try_from(123)
        assert isinstance(result, Err)
        assert isinstance(result.error, TypeError)
        assert "int" in str(result.error)

    def test_as_dict(self) -> None:
        """as_dict() should return a dictionary representation."""
        s = SimpleMutStruct(name="Dave", age=40)
        d = s.as_dict()
        assert d == {"name": "Dave", "age": 40}

    def test_as_json(self) -> None:
        """as_json() should return a JSON string."""
        s = SimpleMutStruct(name="Eve", age=45)
        json_str = s.as_json()
        assert json_str == '{"name":"Eve","age":45}'

    def test_clone_creates_deep_copy(self) -> None:
        """clone() should create a deep copy."""
        original = SimpleMutStruct(name="Frank", age=50)
        cloned = original.clone()

        assert cloned.name == original.name
        assert cloned.age == original.age
        assert cloned is not original

        # Modifying clone should not affect original
        cloned.name = "Modified"
        assert original.name == "Frank"


# =============================================================================
# Struct Tests (Immutability)
# =============================================================================


class TestStructImmutability:
    """Test Struct immutability features."""

    def test_valid_construction(self) -> None:
        """Construct a Struct with valid data."""
        s = SimpleStruct(name="Grace", age=55)
        assert s.name == "Grace"
        assert s.age == 55

    def test_frozen_cannot_modify(self) -> None:
        """Frozen struct should not allow field modification."""
        s = SimpleStruct(name="Grace", age=55)
        with pytest.raises(ValidationError) as exc_info:
            s.name = "Modified"

        errors = exc_info.value.errors()
        assert errors[0]["type"] == "frozen_instance"

    def test_is_hashable(self) -> None:
        """Frozen struct should be hashable."""
        s = SimpleStruct(name="Henry", age=60)
        # Should not raise
        h = hash(s)
        assert isinstance(h, int)

        # Equal structs should have equal hashes
        s2 = SimpleStruct(name="Henry", age=60)
        assert hash(s) == hash(s2)

    def test_can_be_used_in_set(self) -> None:
        """Frozen struct should be usable in sets."""
        s1 = SimpleStruct(name="Ivy", age=65)
        s2 = SimpleStruct(name="Ivy", age=65)
        s3 = SimpleStruct(name="Jack", age=70)

        my_set = {s1, s2, s3}
        assert len(my_set) == 2  # s1 and s2 are equal


class TestStructNestedImmutability:
    """Test that Struct validates nested field immutability."""

    def test_nested_frozen_struct_allowed(self) -> None:
        """Nested frozen Struct should be allowed."""
        inner = SimpleStruct(name="Inner", age=10)
        outer = NestedStruct(inner=inner, label="outer")
        assert outer.inner.name == "Inner"

    def test_nested_mutable_mutstruct_rejected(self) -> None:
        """Nested mutable MutStruct should be rejected."""

        class OuterWithMutableInner(Struct):
            inner: MutableInner

        with pytest.raises(ValidationError) as exc_info:
            OuterWithMutableInner(inner=MutableInner(value=42))

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "mutable_struct_field"
        assert errors[0]["loc"] == ("inner",)

    def test_nested_mutable_basemodel_rejected(self) -> None:
        """Nested mutable BaseModel should be rejected."""

        class OuterWithMutableBaseModel(Struct):
            inner: MutableBaseModel

        with pytest.raises(ValidationError) as exc_info:
            OuterWithMutableBaseModel(inner=MutableBaseModel(value=42))

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "mutable_basemodel_field"
        assert errors[0]["loc"] == ("inner",)

    def test_nested_frozen_basemodel_allowed(self) -> None:
        """Nested frozen BaseModel should be allowed."""

        class OuterWithFrozenBaseModel(Struct):
            inner: FrozenBaseModel

        outer = OuterWithFrozenBaseModel(inner=FrozenBaseModel(value=42))
        assert outer.inner.value == 42

    def test_nested_mutable_dataclass_rejected(self) -> None:
        """Nested mutable dataclass should be rejected."""

        class OuterWithMutableDataclass(Struct):
            inner: MutableDataclass

        with pytest.raises(ValidationError) as exc_info:
            OuterWithMutableDataclass(inner=MutableDataclass(value=42))

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "mutable_dataclass_field"
        assert errors[0]["loc"] == ("inner",)

    def test_nested_frozen_dataclass_allowed(self) -> None:
        """Nested frozen dataclass should be allowed."""

        class OuterWithFrozenDataclass(Struct):
            inner: FrozenDataclass

        outer = OuterWithFrozenDataclass(inner=FrozenDataclass(value=42))
        assert outer.inner.value == 42

    def test_multiple_mutable_fields_collect_all_errors(self) -> None:
        """Multiple mutable fields should collect all errors."""

        class MultipleNested(Struct):
            mut_struct: MutableInner
            mut_basemodel: MutableBaseModel
            mut_dataclass: MutableDataclass

        with pytest.raises(ValidationError) as exc_info:
            MultipleNested(
                mut_struct=MutableInner(value=1),
                mut_basemodel=MutableBaseModel(value=2),
                mut_dataclass=MutableDataclass(value=3),
            )

        errors = exc_info.value.errors()
        assert len(errors) == 3

        error_types = {e["type"] for e in errors}
        assert error_types == {
            "mutable_struct_field",
            "mutable_basemodel_field",
            "mutable_dataclass_field",
        }

    def test_none_values_are_skipped(self) -> None:
        """None values should not cause validation errors."""

        class WithOptional(Struct):
            inner: MutableInner | None = None

        # Should not raise - None is skipped
        s = WithOptional()
        assert s.inner is None


class TestStructInheritsMutStructBehavior:
    """Test that Struct inherits MutStruct behavior."""

    def test_extra_fields_forbidden(self) -> None:
        """Struct should also forbid extra fields."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleStruct(name="Test", age=1, extra="bad")  # ty: ignore[unknown-argument]

        errors = exc_info.value.errors()
        assert errors[0]["type"] == "extra_forbidden"

    def test_strict_mode(self) -> None:
        """Struct should also use strict mode."""
        with pytest.raises(ValidationError):
            SimpleStruct(name="Test", age="1")  # ty: ignore[invalid-argument-type]

    def test_try_from_works(self) -> None:
        """try_from should work on Struct."""
        result = SimpleStruct.try_from({"name": "Test", "age": 1})
        assert isinstance(result, Ok)
        assert result.value.name == "Test"

    def test_clone_works(self) -> None:
        """Clone should work on Struct."""
        s = SimpleStruct(name="Test", age=1)
        cloned = s.clone()
        assert cloned == s
        assert cloned is not s
