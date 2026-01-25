"""Tests for the struct module."""

from dataclasses import dataclass
from enum import StrEnum

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from carcinize._result import Err, Ok
from carcinize._struct import Struct

# =============================================================================
# Test Fixtures (helper classes)
# =============================================================================


class SimpleMutableStruct(Struct, mut=True):
    """A simple mutable struct for testing."""

    name: str
    age: int


class SimpleImmutableStruct(Struct):
    """A simple immutable struct for testing."""

    name: str
    age: int


class NestedImmutableStruct(Struct):
    """An immutable struct containing another immutable struct."""

    inner: SimpleImmutableStruct
    label: str


class MutableInner(Struct, mut=True):
    """A mutable inner struct."""

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
# Mutable Struct Tests
# =============================================================================


class TestMutableStructConstruction:
    """Test mutable Struct construction and validation."""

    def test_valid_construction(self) -> None:
        """Construct a mutable Struct with valid data."""
        s = SimpleMutableStruct(name="Alice", age=30)
        assert s.name == "Alice"
        assert s.age == 30

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMutableStruct(name="Alice", age=30, extra="not allowed")  # ty: ignore[unknown-argument]

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_strict_mode_rejects_coercion(self) -> None:
        """Strict mode should reject type coercion (e.g., str to int)."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleMutableStruct(name="Alice", age="30")  # ty: ignore[invalid-argument-type]

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "int_type"

    def test_validate_assignment(self) -> None:
        """Assignment should be validated."""
        s = SimpleMutableStruct(name="Alice", age=30)
        s.age = 31  # Valid assignment
        assert s.age == 31

        with pytest.raises(ValidationError):
            s.age = "not an int"  # ty: ignore[invalid-assignment]

    def test_mutable_flag(self) -> None:
        """Mutable struct should have __mutable__ = True."""
        s = SimpleMutableStruct(name="Alice", age=30)
        assert s.__mutable__ is True


class TestMutableStructMethods:
    """Test mutable Struct helper methods."""

    def test_try_from_dict_success(self) -> None:
        """try_from() with valid dict should return Ok."""
        result = SimpleMutableStruct.try_from({"name": "Bob", "age": 25})
        assert isinstance(result, Ok)
        assert result.value.name == "Bob"
        assert result.value.age == 25

    def test_try_from_dict_failure(self) -> None:
        """try_from() with invalid dict should return Err."""
        result = SimpleMutableStruct.try_from({"name": "Bob"})  # Missing age
        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)

    def test_try_from_json_success(self) -> None:
        """try_from() with valid JSON string should return Ok."""
        result = SimpleMutableStruct.try_from('{"name": "Charlie", "age": 35}')
        assert isinstance(result, Ok)
        assert result.value.name == "Charlie"
        assert result.value.age == 35

    def test_try_from_json_failure(self) -> None:
        """try_from() with invalid JSON should return Err."""
        result = SimpleMutableStruct.try_from('{"name": "Charlie"}')  # Missing age
        assert isinstance(result, Err)

    def test_try_from_invalid_type(self) -> None:
        """try_from() with invalid type should return Err(TypeError)."""
        result = SimpleMutableStruct.try_from(123)
        assert isinstance(result, Err)
        assert isinstance(result.error, TypeError)
        assert "int" in str(result.error)

    def test_as_dict(self) -> None:
        """as_dict() should return a dictionary representation."""
        s = SimpleMutableStruct(name="Dave", age=40)
        d = s.as_dict()
        assert d == {"name": "Dave", "age": 40}

    def test_as_json(self) -> None:
        """as_json() should return a JSON string."""
        s = SimpleMutableStruct(name="Eve", age=45)
        json_str = s.as_json()
        assert json_str == '{"name":"Eve","age":45}'

    def test_clone_creates_deep_copy(self) -> None:
        """clone() should create a deep copy."""
        original = SimpleMutableStruct(name="Frank", age=50)
        cloned = original.clone()

        assert cloned.name == original.name
        assert cloned.age == original.age
        assert cloned is not original

        # Modifying clone should not affect original
        cloned.name = "Modified"
        assert original.name == "Frank"


# =============================================================================
# Immutable Struct Tests
# =============================================================================


class TestImmutableStructConstruction:
    """Test immutable Struct construction and features."""

    def test_valid_construction(self) -> None:
        """Construct an immutable Struct with valid data."""
        s = SimpleImmutableStruct(name="Grace", age=55)
        assert s.name == "Grace"
        assert s.age == 55

    def test_frozen_cannot_modify(self) -> None:
        """Frozen struct should not allow field modification."""
        s = SimpleImmutableStruct(name="Grace", age=55)
        with pytest.raises(ValidationError) as exc_info:
            s.name = "Modified"

        errors = exc_info.value.errors()
        assert errors[0]["type"] == "frozen_instance"

    def test_is_hashable(self) -> None:
        """Frozen struct should be hashable."""
        s = SimpleImmutableStruct(name="Henry", age=60)
        h = hash(s)
        assert isinstance(h, int)

        # Equal structs should have equal hashes
        s2 = SimpleImmutableStruct(name="Henry", age=60)
        assert hash(s) == hash(s2)

    def test_can_be_used_in_set(self) -> None:
        """Frozen struct should be usable in sets."""
        s1 = SimpleImmutableStruct(name="Ivy", age=65)
        s2 = SimpleImmutableStruct(name="Ivy", age=65)
        s3 = SimpleImmutableStruct(name="Jack", age=70)

        my_set = {s1, s2, s3}
        assert len(my_set) == 2  # s1 and s2 are equal

    def test_immutable_flag(self) -> None:
        """Immutable struct should have __mutable__ = False."""
        s = SimpleImmutableStruct(name="Grace", age=55)
        assert s.__mutable__ is False


class TestNestedImmutability:
    """Test that immutable Struct validates nested field immutability."""

    def test_nested_immutable_struct_allowed(self) -> None:
        """Nested immutable Struct should be allowed."""
        inner = SimpleImmutableStruct(name="Inner", age=10)
        outer = NestedImmutableStruct(inner=inner, label="outer")
        assert outer.inner.name == "Inner"

    def test_nested_mutable_struct_rejected(self) -> None:
        """Nested mutable Struct should be rejected."""

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


class TestImmutableStructInheritsBehavior:
    """Test that immutable Struct inherits shared behavior."""

    def test_extra_fields_forbidden(self) -> None:
        """Immutable Struct should also forbid extra fields."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleImmutableStruct(name="Test", age=1, extra="bad")  # ty: ignore[unknown-argument]

        errors = exc_info.value.errors()
        assert errors[0]["type"] == "extra_forbidden"

    def test_strict_mode(self) -> None:
        """Immutable Struct should also use strict mode."""
        with pytest.raises(ValidationError):
            SimpleImmutableStruct(name="Test", age="1")  # ty: ignore[invalid-argument-type]

    def test_try_from_works(self) -> None:
        """try_from should work on immutable Struct."""
        result = SimpleImmutableStruct.try_from({"name": "Test", "age": 1})
        assert isinstance(result, Ok)
        assert result.value.name == "Test"

    def test_clone_works(self) -> None:
        """Clone should work on immutable Struct."""
        s = SimpleImmutableStruct(name="Test", age=1)
        cloned = s.clone()
        assert cloned == s
        assert cloned is not s


# =============================================================================
# StrEnum Field Tests
# =============================================================================


class Status(StrEnum):
    """Example StrEnum for testing."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


class Priority(StrEnum):
    """Another StrEnum for testing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskImmutable(Struct):
    """An immutable struct with StrEnum fields."""

    title: str
    status: Status
    priority: Priority


class TaskMutable(Struct, mut=True):
    """A mutable struct with StrEnum fields."""

    title: str
    status: Status
    priority: Priority


class TestStructWithStrEnum:
    """Test Struct with StrEnum fields."""

    def test_construct_with_enum_values(self) -> None:
        """Construct a Struct using actual enum members."""
        task = TaskImmutable(title="Test", status=Status.ACTIVE, priority=Priority.HIGH)
        assert task.title == "Test"
        assert task.status == "active"  # use_enum_values=True returns the string value
        assert task.priority == "high"

    def test_try_from_dict_with_enum_instances(self) -> None:
        """try_from() with dict accepts actual enum instances."""
        result = TaskImmutable.try_from(
            {
                "title": "My Task",
                "status": Status.PENDING,
                "priority": Priority.MEDIUM,
            }
        )
        assert isinstance(result, Ok)
        task = result.unwrap()
        assert task.title == "My Task"
        assert task.status == "pending"
        assert task.priority == "medium"

    def test_try_from_dict_with_string_values(self) -> None:
        """try_from() with dict accepts string values for enum fields."""
        result = TaskImmutable.try_from(
            {
                "title": "My Task",
                "status": "pending",
                "priority": "medium",
            }
        )
        assert isinstance(result, Ok)
        task = result.unwrap()
        assert task.status == "pending"
        assert task.priority == "medium"

    def test_try_from_json_with_string_values(self) -> None:
        """try_from() should parse JSON with string values for StrEnum fields."""
        json_str = '{"title": "JSON Task", "status": "completed", "priority": "low"}'
        result = TaskImmutable.try_from(json_str)
        assert isinstance(result, Ok)
        task = result.unwrap()
        assert task.status == "completed"
        assert task.priority == "low"

    def test_try_from_json_invalid_enum_value(self) -> None:
        """try_from() should return Err for invalid enum values."""
        json_str = '{"title": "Bad Task", "status": "invalid_status", "priority": "low"}'
        result = TaskImmutable.try_from(json_str)
        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)

    def test_as_dict_returns_string_values(self) -> None:
        """as_dict() should return string values for StrEnum fields."""
        task = TaskImmutable(title="Test", status=Status.ACTIVE, priority=Priority.HIGH)
        d = task.as_dict()
        assert d == {"title": "Test", "status": "active", "priority": "high"}

    def test_as_json_returns_string_values(self) -> None:
        """as_json() should serialize StrEnum fields as strings."""
        task = TaskImmutable(title="Test", status=Status.PENDING, priority=Priority.LOW)
        json_str = task.as_json()
        assert '"status":"pending"' in json_str
        assert '"priority":"low"' in json_str

    def test_mutable_struct_enum_assignment(self) -> None:
        """Mutable Struct should allow reassigning StrEnum fields."""
        task = TaskMutable(title="Test", status=Status.PENDING, priority=Priority.LOW)
        assert task.status == "pending"

        task.status = Status.ACTIVE
        assert task.status == "active"

    def test_immutable_struct_hashable_with_enum(self) -> None:
        """Immutable Struct with StrEnum fields should be hashable."""
        task1 = TaskImmutable(title="Test", status=Status.ACTIVE, priority=Priority.HIGH)
        task2 = TaskImmutable(title="Test", status=Status.ACTIVE, priority=Priority.HIGH)

        assert hash(task1) == hash(task2)
        assert {task1, task2} == {task1}

    def test_struct_equality_with_enum(self) -> None:
        """Struct equality should work with StrEnum fields."""
        task1 = TaskImmutable(title="Test", status=Status.ACTIVE, priority=Priority.HIGH)
        task2 = TaskImmutable(title="Test", status=Status.ACTIVE, priority=Priority.HIGH)
        task3 = TaskImmutable(title="Test", status=Status.PENDING, priority=Priority.HIGH)

        assert task1 == task2
        assert task1 != task3

    def test_roundtrip_json_serialization(self) -> None:
        """Struct should survive JSON roundtrip with StrEnum fields."""
        original = TaskImmutable(title="Roundtrip", status=Status.COMPLETED, priority=Priority.MEDIUM)

        json_str = original.as_json()
        result = TaskImmutable.try_from(json_str)
        assert isinstance(result, Ok)
        restored = result.unwrap()

        assert restored.title == original.title
        assert restored.status == original.status
        assert restored.priority == original.priority


# =============================================================================
# Pattern Matching Tests
# =============================================================================


class TestPatternMatching:
    """Test Rust-like pattern matching support."""

    def test_match_args_set_on_mutable(self) -> None:
        """Mutable Struct should have __match_args__ set."""
        assert hasattr(SimpleMutableStruct, "__match_args__")
        assert SimpleMutableStruct.__match_args__ == ("name", "age")

    def test_match_args_set_on_immutable(self) -> None:
        """Immutable Struct should have __match_args__ set."""
        assert hasattr(SimpleImmutableStruct, "__match_args__")
        assert SimpleImmutableStruct.__match_args__ == ("name", "age")

    def test_positional_pattern_matching_mutable(self) -> None:
        """Mutable Struct should support positional pattern matching."""
        user = SimpleMutableStruct(name="Alice", age=30)

        match user:
            case SimpleMutableStruct(name, age):
                assert name == "Alice"
                assert age == 30
            case _:
                pytest.fail("Pattern should have matched")

    def test_positional_pattern_matching_immutable(self) -> None:
        """Immutable Struct should support positional pattern matching."""
        user = SimpleImmutableStruct(name="Bob", age=25)

        match user:
            case SimpleImmutableStruct(name, age):
                assert name == "Bob"
                assert age == 25
            case _:
                pytest.fail("Pattern should have matched")

    def test_pattern_matching_with_guards(self) -> None:
        """Pattern matching should work with guards."""
        user = SimpleImmutableStruct(name="Charlie", age=17)

        match user:
            case SimpleImmutableStruct(name, age) if age >= 18:
                pytest.fail("Guard should have prevented match")
            case SimpleImmutableStruct(name, age):
                assert name == "Charlie"
                assert age == 17

    def test_pattern_matching_nested_struct(self) -> None:
        """Pattern matching should work with nested structs."""
        inner = SimpleImmutableStruct(name="Inner", age=10)
        outer = NestedImmutableStruct(inner=inner, label="outer")

        match outer:
            case NestedImmutableStruct(SimpleImmutableStruct(name, age), label):
                assert name == "Inner"
                assert age == 10
                assert label == "outer"
            case _:
                pytest.fail("Pattern should have matched")


# =============================================================================
# Replace Method Tests
# =============================================================================


class TestReplaceMethod:
    """Test Rust-like struct update syntax via replace()."""

    def test_replace_single_field_mutable(self) -> None:
        """replace() should return new mutable Struct with field updated."""
        original = SimpleMutableStruct(name="Alice", age=30)
        updated = original.replace(age=31)

        assert updated.name == "Alice"
        assert updated.age == 31
        assert original.age == 30  # Original unchanged

    def test_replace_single_field_immutable(self) -> None:
        """replace() should return new immutable Struct with field updated."""
        original = SimpleImmutableStruct(name="Bob", age=25)
        updated = original.replace(age=26)

        assert updated.name == "Bob"
        assert updated.age == 26
        assert original.age == 25

    def test_replace_multiple_fields(self) -> None:
        """replace() should handle multiple field updates."""
        original = SimpleImmutableStruct(name="Charlie", age=20)
        updated = original.replace(name="Charles", age=21)

        assert updated.name == "Charles"
        assert updated.age == 21

    def test_replace_returns_new_instance(self) -> None:
        """replace() should always return a new instance."""
        original = SimpleImmutableStruct(name="Dave", age=35)
        updated = original.replace(age=36)

        assert updated is not original
        assert updated == SimpleImmutableStruct(name="Dave", age=36)

    def test_replace_with_no_changes(self) -> None:
        """replace() with no arguments should return equal copy."""
        original = SimpleImmutableStruct(name="Eve", age=40)
        copy = original.replace()

        assert copy is not original
        assert copy == original

    def test_replace_validates_new_values(self) -> None:
        """replace() should validate the new field values."""
        original = SimpleMutableStruct(name="Frank", age=45)

        with pytest.raises(ValidationError):
            original.replace(age="not an int")

    def test_replace_nested_struct(self) -> None:
        """replace() should work with nested structs."""
        inner = SimpleImmutableStruct(name="Inner", age=10)
        outer = NestedImmutableStruct(inner=inner, label="original")

        new_inner = SimpleImmutableStruct(name="NewInner", age=20)
        updated = outer.replace(inner=new_inner)

        assert updated.inner.name == "NewInner"
        assert updated.label == "original"
        assert outer.inner.name == "Inner"

    def test_replace_preserves_type(self) -> None:
        """replace() should preserve the exact subclass type."""
        original = SimpleImmutableStruct(name="Grace", age=50)
        updated = original.replace(age=51)

        assert type(updated) is SimpleImmutableStruct


# =============================================================================
# Mutability API Tests
# =============================================================================


class TestMutabilityAPI:
    """Test the mut=True/False API."""

    def test_struct_immutable_by_default(self) -> None:
        """Struct should be immutable by default."""

        class DefaultStruct(Struct):
            value: int

        s = DefaultStruct(value=42)
        assert s.__mutable__ is False
        assert s.model_config.get("frozen") is True

        with pytest.raises(ValidationError):
            s.value = 100

    def test_struct_with_mut_true_is_mutable(self) -> None:
        """Struct with mut=True should be mutable."""

        class MutableStruct(Struct, mut=True):
            value: int

        s = MutableStruct(value=42)
        assert s.__mutable__ is True
        assert s.model_config.get("frozen") is False

        s.value = 100
        assert s.value == 100

    def test_struct_with_mut_false_is_immutable(self) -> None:
        """Struct with explicit mut=False should be immutable."""

        class ExplicitImmutable(Struct, mut=False):
            value: int

        s = ExplicitImmutable(value=42)
        assert s.__mutable__ is False

        with pytest.raises(ValidationError):
            s.value = 100

    def test_mutable_struct_allows_mutable_nested_fields(self) -> None:
        """Mutable struct should allow mutable nested fields."""

        class Inner(Struct, mut=True):
            value: int

        class Outer(Struct, mut=True):
            inner: Inner

        outer = Outer(inner=Inner(value=42))
        assert outer.inner.value == 42

    def test_immutable_struct_rejects_mutable_nested_fields(self) -> None:
        """Immutable struct should reject mutable nested fields."""

        class MutableInnerNew(Struct, mut=True):
            value: int

        class ImmutableOuter(Struct):
            inner: MutableInnerNew

        with pytest.raises(ValidationError) as exc_info:
            ImmutableOuter(inner=MutableInnerNew(value=42))

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "mutable_struct_field"

    def test_immutable_struct_uses_content_hash(self) -> None:
        """Immutable struct should use content-based hashing."""

        class ImmutableHashTest(Struct):
            value: int

        s1 = ImmutableHashTest(value=42)
        s2 = ImmutableHashTest(value=42)

        assert s1 == s2
        assert hash(s1) == hash(s2)
