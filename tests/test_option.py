"""Tests for the Option type (Some and Nothing)."""

import pytest

from carcinize.option import Nothing, Some, UnwrapError
from carcinize.result import Err, Ok

# =============================================================================
# Some Tests
# =============================================================================


class TestSomeInspection:
    """Test Some inspection methods."""

    def test_is_some_returns_true(self) -> None:
        """is_some() should return True for Some."""
        assert Some(42).is_some() is True

    def test_is_nothing_returns_false(self) -> None:
        """is_nothing() should return False for Some."""
        assert Some(42).is_nothing() is False


class TestSomeExtraction:
    """Test Some extraction methods."""

    def test_unwrap_returns_value(self) -> None:
        """unwrap() should return the contained value."""
        assert Some("hello").unwrap() == "hello"

    def test_expect_returns_value(self) -> None:
        """expect() should return the contained value."""
        assert Some(42).expect("should not see this") == 42


class TestSomeFallbacks:
    """Test Some fallback methods."""

    def test_unwrap_or_returns_value(self) -> None:
        """unwrap_or() should return contained value, ignoring default."""
        assert Some(42).unwrap_or(100) == 42

    def test_unwrap_or_else_returns_value(self) -> None:
        """unwrap_or_else() should return contained value, ignoring function."""
        assert Some(42).unwrap_or_else(lambda: 100) == 42


class TestSomeTransformation:
    """Test Some transformation methods."""

    def test_map_transforms_value(self) -> None:
        """map() should transform the contained value."""
        result = Some(5).map(lambda x: x * 2)
        assert isinstance(result, Some)
        assert result.value == 10

    def test_map_or_applies_function(self) -> None:
        """map_or() should apply function to value."""
        assert Some(5).map_or(0, lambda x: x * 2) == 10

    def test_map_or_else_applies_function(self) -> None:
        """map_or_else() should apply function to value."""
        assert Some(5).map_or_else(lambda: 0, lambda x: x * 2) == 10

    def test_and_then_chains_operations(self) -> None:
        """and_then() should chain operations that return Option."""

        def double_if_positive(x: int) -> Some[int] | Nothing:
            if x > 0:
                return Some(x * 2)
            return Nothing()

        result = Some(5).and_then(double_if_positive)
        assert isinstance(result, Some)
        assert result.value == 10

    def test_or_else_returns_self(self) -> None:
        """or_else() should return self unchanged for Some."""
        some = Some(42)
        result = some.or_else(lambda: Some(0))
        assert result is some


class TestSomeFilter:
    """Test Some filter method."""

    def test_filter_keeps_value_when_predicate_true(self) -> None:
        """filter() should return Some when predicate is True."""
        result = Some(10).filter(lambda x: x > 5)
        assert isinstance(result, Some)
        assert result.value == 10

    def test_filter_returns_nothing_when_predicate_false(self) -> None:
        """filter() should return Nothing when predicate is False."""
        result = Some(3).filter(lambda x: x > 5)
        assert isinstance(result, Nothing)


class TestSomeConversion:
    """Test Some conversion methods."""

    def test_ok_or_returns_ok(self) -> None:
        """ok_or() should return Ok with the value."""
        result = Some(42).ok_or(ValueError("error"))
        assert isinstance(result, Ok)
        assert result.value == 42

    def test_ok_or_else_returns_ok(self) -> None:
        """ok_or_else() should return Ok with the value."""
        result = Some(42).ok_or_else(lambda: ValueError("error"))
        assert isinstance(result, Ok)
        assert result.value == 42


class TestSomeZip:
    """Test Some zip method."""

    def test_zip_with_some_returns_tuple(self) -> None:
        """zip() with another Some should return Some with tuple."""
        result = Some(1).zip(Some("a"))
        assert isinstance(result, Some)
        assert result.value == (1, "a")

    def test_zip_with_nothing_returns_nothing(self) -> None:
        """zip() with Nothing should return Nothing."""
        result = Some(1).zip(Nothing())
        assert isinstance(result, Nothing)


# =============================================================================
# Nothing Tests
# =============================================================================


class TestNothingInspection:
    """Test Nothing inspection methods."""

    def test_is_some_returns_false(self) -> None:
        """is_some() should return False for Nothing."""
        assert Nothing().is_some() is False

    def test_is_nothing_returns_true(self) -> None:
        """is_nothing() should return True for Nothing."""
        assert Nothing().is_nothing() is True


class TestNothingExtraction:
    """Test Nothing extraction methods."""

    def test_unwrap_raises(self) -> None:
        """unwrap() should raise UnwrapError for Nothing."""
        with pytest.raises(UnwrapError) as exc_info:
            Nothing().unwrap()
        assert "Nothing" in str(exc_info.value)

    def test_expect_raises_with_message(self) -> None:
        """expect() should raise UnwrapError with custom message."""
        with pytest.raises(UnwrapError) as exc_info:
            Nothing().expect("custom error message")
        assert str(exc_info.value) == "custom error message"


class TestNothingFallbacks:
    """Test Nothing fallback methods."""

    def test_unwrap_or_returns_default(self) -> None:
        """unwrap_or() should return the default value."""
        assert Nothing().unwrap_or(42) == 42

    def test_unwrap_or_else_calls_function(self) -> None:
        """unwrap_or_else() should call the fallback function."""
        assert Nothing().unwrap_or_else(lambda: 42) == 42


class TestNothingTransformation:
    """Test Nothing transformation methods."""

    def test_map_returns_self(self) -> None:
        """map() should return self unchanged for Nothing."""
        nothing = Nothing()
        result = nothing.map(lambda x: x * 2)
        assert result is nothing

    def test_map_or_returns_default(self) -> None:
        """map_or() should return default for Nothing."""
        assert Nothing().map_or(42, lambda x: x * 2) == 42

    def test_map_or_else_calls_default_function(self) -> None:
        """map_or_else() should call default function for Nothing."""
        assert Nothing().map_or_else(lambda: 42, lambda x: x * 2) == 42

    def test_and_then_returns_self(self) -> None:
        """and_then() should return self unchanged for Nothing."""
        nothing = Nothing()
        result = nothing.and_then(lambda x: Some(x * 2))
        assert result is nothing

    def test_or_else_calls_function(self) -> None:
        """or_else() should call function and return its result."""
        result = Nothing().or_else(lambda: Some(42))
        assert isinstance(result, Some)
        assert result.value == 42


class TestNothingFilter:
    """Test Nothing filter method."""

    def test_filter_returns_self(self) -> None:
        """filter() should return self for Nothing."""
        nothing = Nothing()
        result = nothing.filter(lambda _x: True)
        assert result is nothing


class TestNothingConversion:
    """Test Nothing conversion methods."""

    def test_ok_or_returns_err(self) -> None:
        """ok_or() should return Err with the provided error."""
        err = ValueError("error")
        result = Nothing().ok_or(err)
        assert isinstance(result, Err)
        assert result.error is err

    def test_ok_or_else_returns_err(self) -> None:
        """ok_or_else() should return Err from the function."""
        result = Nothing().ok_or_else(lambda: ValueError("computed error"))
        assert isinstance(result, Err)
        assert isinstance(result.error, ValueError)


class TestNothingZip:
    """Test Nothing zip method."""

    def test_zip_returns_self(self) -> None:
        """zip() should return self for Nothing."""
        nothing = Nothing()
        result = nothing.zip(Some(42))
        assert result is nothing


# =============================================================================
# General Option Tests
# =============================================================================


class TestOptionPatternMatching:
    """Test pattern matching with Option."""

    def test_match_some(self) -> None:
        """Pattern matching should work with Some."""
        option: Some[int] | Nothing = Some(42)
        match option:
            case Some(value):
                assert value == 42
            case Nothing():
                pytest.fail("Should not match Nothing")

    def test_match_nothing(self) -> None:
        """Pattern matching should work with Nothing."""
        option: Some[int] | Nothing = Nothing()
        match option:
            case Some():
                pytest.fail("Should not match Some")
            case Nothing():
                pass  # Expected


class TestOptionImmutability:
    """Test that Option types are immutable."""

    def test_some_is_frozen(self) -> None:
        """Some should be immutable (frozen dataclass)."""
        some = Some(42)
        with pytest.raises(AttributeError):
            some.value = 100  # ty: ignore[invalid-assignment]

    def test_nothing_is_frozen(self) -> None:
        """Nothing should be immutable (frozen dataclass)."""
        # Nothing has no fields, so we verify it's frozen via dataclass metadata
        import dataclasses

        assert dataclasses.is_dataclass(Nothing)
        # Access the frozen parameter from the class
        assert Nothing.__dataclass_params__.frozen is True


class TestOptionHashable:
    """Test that Option types are hashable."""

    def test_some_is_hashable(self) -> None:
        """Some should be hashable."""
        some1 = Some(42)
        some2 = Some(42)
        assert hash(some1) == hash(some2)
        assert {some1, some2} == {Some(42)}

    def test_nothing_is_hashable(self) -> None:
        """Nothing should be hashable."""
        n1 = Nothing()
        n2 = Nothing()
        assert hash(n1) == hash(n2)
        assert {n1, n2} == {Nothing()}


class TestOptionEquality:
    """Test Option equality."""

    def test_some_equality(self) -> None:
        """Some values with same content should be equal."""
        assert Some(42) == Some(42)
        assert Some(42) != Some(43)
        assert Some(42) != Nothing()

    def test_nothing_equality(self) -> None:
        """Nothing instances should be equal."""
        assert Nothing() == Nothing()
        assert Nothing() != Some(42)
