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


class TestNestedFields:
    """Test Rust-like nested field behavior.

    Like Rust, immutability applies to the binding - nested fields of any type
    are allowed, but cannot be mutated through the immutable outer struct.
    """

    def test_nested_immutable_struct_allowed(self) -> None:
        """Nested immutable Struct should be allowed."""
        inner = SimpleImmutableStruct(name="Inner", age=10)
        outer = NestedImmutableStruct(inner=inner, label="outer")
        assert outer.inner.name == "Inner"

    def test_nested_mutable_struct_allowed(self) -> None:
        """Nested mutable Struct is allowed (Rust-like behavior).

        Like Rust, the outer immutability prevents mutation through
        the outer binding, even if the inner type is mutable.
        """

        class OuterWithMutableInner(Struct):
            inner: MutableInner

        outer = OuterWithMutableInner(inner=MutableInner(value=42))
        assert outer.inner.value == 42

        # Cannot mutate through the outer immutable struct
        with pytest.raises(ValidationError):
            outer.inner = MutableInner(value=100)

    def test_nested_mutable_basemodel_allowed(self) -> None:
        """Nested mutable BaseModel is allowed (Rust-like behavior)."""

        class OuterWithMutableBaseModel(Struct):
            inner: MutableBaseModel

        outer = OuterWithMutableBaseModel(inner=MutableBaseModel(value=42))
        assert outer.inner.value == 42

    def test_nested_frozen_basemodel_allowed(self) -> None:
        """Nested frozen BaseModel should be allowed."""

        class OuterWithFrozenBaseModel(Struct):
            inner: FrozenBaseModel

        outer = OuterWithFrozenBaseModel(inner=FrozenBaseModel(value=42))
        assert outer.inner.value == 42

    def test_nested_mutable_dataclass_allowed(self) -> None:
        """Nested mutable dataclass is allowed (Rust-like behavior)."""

        class OuterWithMutableDataclass(Struct):
            inner: MutableDataclass

        outer = OuterWithMutableDataclass(inner=MutableDataclass(value=42))
        assert outer.inner.value == 42

    def test_nested_frozen_dataclass_allowed(self) -> None:
        """Nested frozen dataclass should be allowed."""

        class OuterWithFrozenDataclass(Struct):
            inner: FrozenDataclass

        outer = OuterWithFrozenDataclass(inner=FrozenDataclass(value=42))
        assert outer.inner.value == 42

    def test_multiple_nested_types_allowed(self) -> None:
        """Multiple nested types of any mutability are allowed."""

        class MultipleNested(Struct):
            mut_struct: MutableInner
            mut_basemodel: MutableBaseModel
            mut_dataclass: MutableDataclass

        outer = MultipleNested(
            mut_struct=MutableInner(value=1),
            mut_basemodel=MutableBaseModel(value=2),
            mut_dataclass=MutableDataclass(value=3),
        )
        assert outer.mut_struct.value == 1
        assert outer.mut_basemodel.value == 2
        assert outer.mut_dataclass.value == 3

    def test_none_values_work(self) -> None:
        """None values should work for optional fields."""

        class WithOptional(Struct):
            inner: MutableInner | None = None

        s = WithOptional()
        assert s.inner is None

        s2 = WithOptional(inner=MutableInner(value=42))
        assert s2.inner is not None
        assert s2.inner.value == 42


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

    def test_immutable_struct_allows_mutable_nested_fields(self) -> None:
        """Immutable struct allows mutable nested fields (Rust-like).

        Like Rust, immutability is a property of the binding, not the type.
        The outer frozen struct prevents mutation of its fields.
        """

        class MutableInnerNew(Struct, mut=True):
            value: int

        class ImmutableOuter(Struct):
            inner: MutableInnerNew

        outer = ImmutableOuter(inner=MutableInnerNew(value=42))
        assert outer.inner.value == 42

        # Cannot replace the inner field through the frozen outer
        with pytest.raises(ValidationError):
            outer.inner = MutableInnerNew(value=100)

    def test_immutable_struct_uses_content_hash(self) -> None:
        """Immutable struct should use content-based hashing."""

        class ImmutableHashTest(Struct):
            value: int

        s1 = ImmutableHashTest(value=42)
        s2 = ImmutableHashTest(value=42)

        assert s1 == s2
        assert hash(s1) == hash(s2)


# =============================================================================
# is_mut() Classmethod Tests
# =============================================================================


class TestIsMutClassmethod:
    """Test the is_mut() classmethod for runtime mutability checks."""

    def test_is_mut_on_immutable_struct(self) -> None:
        """is_mut() should return False for immutable structs."""
        assert SimpleImmutableStruct.is_mut() is False

    def test_is_mut_on_mutable_struct(self) -> None:
        """is_mut() should return True for mutable structs."""
        assert SimpleMutableStruct.is_mut() is True

    def test_is_mut_on_explicit_mut_false(self) -> None:
        """is_mut() should return False for explicit mut=False."""

        class ExplicitImmutable(Struct, mut=False):
            value: int

        assert ExplicitImmutable.is_mut() is False

    def test_is_mut_via_instance(self) -> None:
        """is_mut() should be callable via instance as well."""
        mutable = SimpleMutableStruct(name="test", age=1)
        immutable = SimpleImmutableStruct(name="test", age=1)

        assert mutable.is_mut() is True
        assert immutable.is_mut() is False

    def test_is_mut_matches_mutable_flag(self) -> None:
        """is_mut() should always match __mutable__ class variable."""

        class TestMutable(Struct, mut=True):
            x: int

        class TestImmutable(Struct):
            x: int

        assert TestMutable.is_mut() == TestMutable.__mutable__
        assert TestImmutable.is_mut() == TestImmutable.__mutable__


# =============================================================================
# isinstance() Behavior Tests
# =============================================================================


class TestIsInstanceBehavior:
    """Test that isinstance() works correctly with the metaclass."""

    def test_mutable_struct_is_instance_of_struct(self) -> None:
        """Mutable struct instances should be isinstance of Struct."""
        s = SimpleMutableStruct(name="test", age=1)
        assert isinstance(s, Struct)

    def test_immutable_struct_is_instance_of_struct(self) -> None:
        """Immutable struct instances should be isinstance of Struct."""
        s = SimpleImmutableStruct(name="test", age=1)
        assert isinstance(s, Struct)

    def test_subclass_check_for_mutable(self) -> None:
        """Mutable struct class should be subclass of Struct."""
        assert issubclass(SimpleMutableStruct, Struct)

    def test_subclass_check_for_immutable(self) -> None:
        """Immutable struct class should be subclass of Struct."""
        assert issubclass(SimpleImmutableStruct, Struct)

    def test_isinstance_with_nested_struct(self) -> None:
        """Nested struct should also be isinstance of Struct."""
        inner = SimpleImmutableStruct(name="inner", age=1)
        outer = NestedImmutableStruct(inner=inner, label="outer")

        assert isinstance(outer, Struct)
        assert isinstance(outer.inner, Struct)


# =============================================================================
# Base Struct Class Immutability Tests
# =============================================================================


class TestBaseStructImmutability:
    """Test that the base Struct class itself is properly frozen.

    This verifies the fix where the metaclass was skipping _frozen_setattr
    assignment for the base Struct class.
    """

    def test_base_struct_has_frozen_config(self) -> None:
        """Base Struct class should have frozen=True in model_config."""
        assert Struct.model_config.get("frozen") is True

    def test_base_struct_is_not_mutable(self) -> None:
        """Base Struct class should have __mutable__ = False."""
        assert Struct.__mutable__ is False

    def test_base_struct_is_mut_returns_false(self) -> None:
        """Base Struct class is_mut() should return False."""
        assert Struct.is_mut() is False


# =============================================================================
# Inheritance Hierarchy Tests
# =============================================================================


class TestInheritanceHierarchy:
    """Test that the inheritance hierarchy is properly maintained."""

    def test_user_class_inherits_from_struct(self) -> None:
        """User-defined classes should inherit from Struct."""
        assert Struct in SimpleImmutableStruct.__mro__
        assert Struct in SimpleMutableStruct.__mro__

    def test_mro_includes_struct_base(self) -> None:
        """Method Resolution Order should include Struct."""

        class TestStruct(Struct):
            x: int

        mro_names = [cls.__name__ for cls in TestStruct.__mro__]
        assert "Struct" in mro_names
        assert "_StructBase" in mro_names

    def test_multi_level_inheritance_immutable(self) -> None:
        """Multi-level inheritance should preserve immutability."""

        class Base(Struct):
            x: int

        class Derived(Base):
            y: int

        d = Derived(x=1, y=2)
        assert d.__mutable__ is False
        assert d.is_mut() is False

        with pytest.raises(ValidationError):
            d.x = 10

    def test_multi_level_inheritance_mutable(self) -> None:
        """Multi-level inheritance requires explicit mut=True on each level.

        Mutability does NOT inherit - each derived class must explicitly opt in.
        This is the safe default (immutability by default, like Rust).
        """

        class MutableBase(Struct, mut=True):
            x: int

        # Derived without mut=True becomes immutable (the default)
        class ImmutableDerived(MutableBase):
            y: int

        d = ImmutableDerived(x=1, y=2)
        assert d.__mutable__ is False  # Immutable by default!
        assert d.model_config.get("frozen") is True

        with pytest.raises(ValidationError):
            d.x = 10

        # Must explicitly specify mut=True for mutable derived class
        class ExplicitMutableDerived(MutableBase, mut=True):
            y: int

        d2 = ExplicitMutableDerived(x=1, y=2)
        assert d2.__mutable__ is True
        assert d2.model_config.get("frozen") is False

        d2.x = 10
        assert d2.x == 10

    def test_mixed_inheritance_immutable_from_mutable(self) -> None:
        """Derived immutable struct from mutable base should be immutable."""

        class MutableParent(Struct, mut=True):
            x: int

        class ImmutableChild(MutableParent, mut=False):
            y: int

        child = ImmutableChild(x=1, y=2)
        assert child.__mutable__ is False
        assert child.is_mut() is False
        assert child.model_config.get("frozen") is True

        with pytest.raises(ValidationError):
            child.x = 10

    def test_mixed_inheritance_mutable_from_immutable(self) -> None:
        """Derived mutable struct from immutable base should be mutable."""

        class ImmutableParent(Struct):
            x: int

        class MutableChild(ImmutableParent, mut=True):
            y: int

        child = MutableChild(x=1, y=2)
        assert child.__mutable__ is True
        assert child.is_mut() is True
        assert child.model_config.get("frozen") is False

        child.x = 10
        assert child.x == 10


# =============================================================================
# Edge Cases and Contract Tests
# =============================================================================


class TestEdgeCasesAndContracts:
    """Test edge cases and contractual behavior."""

    def test_empty_struct(self) -> None:
        """Empty struct (no fields) should work."""

        class EmptyStruct(Struct):
            pass

        s = EmptyStruct()
        assert s.__mutable__ is False
        assert s.is_mut() is False
        assert s.as_dict() == {}

    def test_empty_mutable_struct(self) -> None:
        """Empty mutable struct should work."""

        class EmptyMutableStruct(Struct, mut=True):
            pass

        s = EmptyMutableStruct()
        assert s.__mutable__ is True
        assert s.is_mut() is True

    def test_struct_with_defaults(self) -> None:
        """Struct with default values should work correctly."""

        class WithDefaults(Struct):
            name: str = "default"
            count: int = 0

        s = WithDefaults()
        assert s.name == "default"
        assert s.count == 0
        assert s.__mutable__ is False

    def test_struct_with_optional_fields(self) -> None:
        """Struct with optional fields should work correctly."""

        class WithOptional(Struct):
            required: str
            optional: str | None = None

        s = WithOptional(required="test")
        assert s.required == "test"
        assert s.optional is None
        assert s.__mutable__ is False

    def test_replace_preserves_mutability(self) -> None:
        """replace() should return instance with same mutability."""
        mutable = SimpleMutableStruct(name="test", age=1)
        replaced_mutable = mutable.replace(age=2)
        assert replaced_mutable.__mutable__ is True
        assert replaced_mutable.is_mut() is True

        immutable = SimpleImmutableStruct(name="test", age=1)
        replaced_immutable = immutable.replace(age=2)
        assert replaced_immutable.__mutable__ is False
        assert replaced_immutable.is_mut() is False

    def test_clone_preserves_mutability(self) -> None:
        """clone() should return instance with same mutability."""
        mutable = SimpleMutableStruct(name="test", age=1)
        cloned_mutable = mutable.clone()
        assert cloned_mutable.__mutable__ is True

        immutable = SimpleImmutableStruct(name="test", age=1)
        cloned_immutable = immutable.clone()
        assert cloned_immutable.__mutable__ is False

    def test_try_from_preserves_mutability(self) -> None:
        """try_from() should return instance with correct mutability."""
        mutable_result = SimpleMutableStruct.try_from({"name": "test", "age": 1})
        assert mutable_result.is_ok()
        assert mutable_result.unwrap().__mutable__ is True

        immutable_result = SimpleImmutableStruct.try_from({"name": "test", "age": 1})
        assert immutable_result.is_ok()
        assert immutable_result.unwrap().__mutable__ is False

    def test_match_args_on_derived_class(self) -> None:
        """__match_args__ should be set correctly on derived classes."""

        class Parent(Struct):
            x: int

        class Child(Parent):
            y: str

        assert Child.__match_args__ == ("x", "y")

        child = Child(x=1, y="test")
        match child:
            case Child(x, y):
                assert x == 1
                assert y == "test"
            case _:
                pytest.fail("Pattern should have matched")

    def test_multiple_inheritance_not_supported_with_other_pydantic(self) -> None:
        """Document that multiple inheritance with other BaseModel subclasses may be complex."""
        # This is more of a documentation test - complex multiple inheritance
        # scenarios should be tested individually if needed

        class OtherModel(BaseModel):
            z: int

        # This should work but the config merging behavior depends on Pydantic
        class Combined(Struct, OtherModel):
            x: int

        c = Combined(x=1, z=2)
        assert c.x == 1
        assert c.z == 2
