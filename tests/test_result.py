"""Tests for the Result type (Ok and Err)."""

import pytest

from carcinize._exceptions import UnwrapError
from carcinize._result import Err, Ok

# =============================================================================
# Ok Tests
# =============================================================================


class TestOkInspection:
    """Test Ok inspection methods."""

    def test_is_ok_returns_true(self) -> None:
        """is_ok() should return True for Ok."""
        assert Ok(42).is_ok() is True

    def test_is_err_returns_false(self) -> None:
        """is_err() should return False for Ok."""
        assert Ok(42).is_err() is False


class TestOkExtraction:
    """Test Ok extraction methods."""

    def test_ok_returns_value(self) -> None:
        """ok() should return the contained value."""
        assert Ok(42).ok() == 42

    def test_err_returns_none(self) -> None:
        """err() should return None for Ok."""
        assert Ok(42).err() is None

    def test_unwrap_returns_value(self) -> None:
        """unwrap() should return the contained value."""
        assert Ok("hello").unwrap() == "hello"

    def test_unwrap_err_raises(self) -> None:
        """unwrap_err() should raise UnwrapError for Ok."""
        with pytest.raises(UnwrapError) as exc_info:
            Ok(42).unwrap_err()
        assert "Ok" in str(exc_info.value)

    def test_expect_returns_value(self) -> None:
        """expect() should return the contained value."""
        assert Ok(42).expect("should not see this") == 42

    def test_expect_err_raises_with_message(self) -> None:
        """expect_err() should raise UnwrapError with custom message."""
        with pytest.raises(UnwrapError) as exc_info:
            Ok(42).expect_err("custom error message")
        assert str(exc_info.value) == "custom error message"


class TestOkFallbacks:
    """Test Ok fallback methods."""

    def test_unwrap_or_returns_value(self) -> None:
        """unwrap_or() should return contained value, ignoring default."""
        assert Ok(42).unwrap_or(100) == 42

    def test_unwrap_or_else_returns_value(self) -> None:
        """unwrap_or_else() should return contained value, ignoring function."""
        assert Ok(42).unwrap_or_else(lambda: 100) == 42


class TestOkTransformation:
    """Test Ok transformation methods."""

    def test_map_transforms_value(self) -> None:
        """map() should transform the contained value."""
        result = Ok(5).map(lambda x: x * 2)
        assert isinstance(result, Ok)
        assert result.value == 10

    def test_map_err_returns_self(self) -> None:
        """map_err() should return self unchanged for Ok."""
        ok = Ok(42)
        result = ok.map_err(lambda e: str(e))
        assert result is ok

    def test_map_or_applies_function(self) -> None:
        """map_or() should apply function to value."""
        assert Ok(5).map_or(0, lambda x: x * 2) == 10

    def test_map_or_else_applies_function(self) -> None:
        """map_or_else() should apply function to value."""
        assert Ok(5).map_or_else(lambda: 0, lambda x: x * 2) == 10

    def test_and_then_chains_operations(self) -> None:
        """and_then() should chain operations that return Result."""

        def double_if_positive(x: int) -> Ok[int] | Err[ValueError]:
            if x > 0:
                return Ok(x * 2)
            return Err(ValueError("not positive"))

        result = Ok(5).and_then(double_if_positive)
        assert isinstance(result, Ok)
        assert result.value == 10

    def test_or_else_returns_self(self) -> None:
        """or_else() should return self unchanged for Ok."""
        ok = Ok(42)
        result = ok.or_else(lambda _e: Ok(0))
        assert result is ok


# =============================================================================
# Err Tests
# =============================================================================


class TestErrInspection:
    """Test Err inspection methods."""

    def test_is_ok_returns_false(self) -> None:
        """is_ok() should return False for Err."""
        assert Err(ValueError("error")).is_ok() is False

    def test_is_err_returns_true(self) -> None:
        """is_err() should return True for Err."""
        assert Err(ValueError("error")).is_err() is True


class TestErrExtraction:
    """Test Err extraction methods."""

    def test_ok_returns_none(self) -> None:
        """ok() should return None for Err."""
        assert Err(ValueError("error")).ok() is None

    def test_err_returns_error(self) -> None:
        """err() should return the contained error."""
        err = ValueError("oops")
        assert Err(err).err() is err

    def test_unwrap_raises_contained_error(self) -> None:
        """unwrap() should raise the contained error directly."""
        err = ValueError("oops")
        with pytest.raises(ValueError, match="oops") as exc_info:
            Err(err).unwrap()
        assert exc_info.value is err

    def test_unwrap_err_returns_error(self) -> None:
        """unwrap_err() should return the contained error."""
        err = ValueError("oops")
        assert Err(err).unwrap_err() is err

    def test_expect_raises_contained_error_with_cause(self) -> None:
        """expect() should raise the contained error with UnwrapError as cause."""
        err = ValueError("oops")
        with pytest.raises(ValueError, match="oops") as exc_info:
            Err(err).expect("custom error message")
        assert exc_info.value is err
        assert isinstance(exc_info.value.__cause__, UnwrapError)
        assert str(exc_info.value.__cause__) == "custom error message"

    def test_expect_err_returns_error(self) -> None:
        """expect_err() should return the contained error."""
        err = ValueError("oops")
        assert Err(err).expect_err("should not see this") is err


class TestErrFallbacks:
    """Test Err fallback methods."""

    def test_unwrap_or_returns_default(self) -> None:
        """unwrap_or() should return the default value."""
        assert Err(ValueError("error")).unwrap_or(42) == 42

    def test_unwrap_or_else_calls_function(self) -> None:
        """unwrap_or_else() should call the fallback function."""
        assert Err(ValueError("error")).unwrap_or_else(lambda: 42) == 42


class TestErrTransformation:
    """Test Err transformation methods."""

    def test_map_returns_self(self) -> None:
        """map() should return self unchanged for Err."""
        err = Err(ValueError("oops"))
        result = err.map(lambda x: x * 2)
        assert result is err

    def test_map_err_transforms_error(self) -> None:
        """map_err() should transform the contained error."""
        result = Err(ValueError("oops")).map_err(lambda e: TypeError(f"converted: {e}"))
        assert isinstance(result, Err)
        assert isinstance(result.error, TypeError)

    def test_map_or_returns_default(self) -> None:
        """map_or() should return default for Err."""
        assert Err(ValueError("error")).map_or(42, lambda x: x * 2) == 42

    def test_map_or_else_calls_default_function(self) -> None:
        """map_or_else() should call default function for Err."""
        assert Err(ValueError("error")).map_or_else(lambda: 42, lambda x: x * 2) == 42

    def test_and_then_returns_self(self) -> None:
        """and_then() should return self unchanged for Err."""
        err = Err(ValueError("oops"))
        result = err.and_then(lambda x: Ok(x * 2))
        assert result is err

    def test_or_else_calls_function(self) -> None:
        """or_else() should call function with error."""
        result = Err(ValueError("oops")).or_else(lambda e: Ok(len(str(e))))
        assert isinstance(result, Ok)
        assert result.value == 4


# =============================================================================
# General Result Tests
# =============================================================================


class TestResultPatternMatching:
    """Test pattern matching with Result."""

    def test_match_ok(self) -> None:
        """Pattern matching should work with Ok."""
        result: Ok[int] | Err[ValueError] = Ok(42)
        match result:
            case Ok(value):
                assert value == 42
            case Err():
                pytest.fail("Should not match Err")

    def test_match_err(self) -> None:
        """Pattern matching should work with Err."""
        err = ValueError("oops")
        result: Ok[int] | Err[ValueError] = Err(err)
        match result:
            case Ok():
                pytest.fail("Should not match Ok")
            case Err(error):
                assert error is err


class TestResultImmutability:
    """Test that Result types are immutable."""

    def test_ok_is_frozen(self) -> None:
        """Ok should be immutable (frozen dataclass)."""
        ok = Ok(42)
        with pytest.raises(AttributeError):
            ok.value = 100  # ty: ignore[invalid-assignment]

    def test_err_is_frozen(self) -> None:
        """Err should be immutable (frozen dataclass)."""
        err = Err(ValueError("oops"))
        with pytest.raises(AttributeError):
            err.error = ValueError("changed")  # ty: ignore[invalid-assignment]


class TestResultHashable:
    """Test that Result types are hashable."""

    def test_ok_is_hashable(self) -> None:
        """Ok should be hashable."""
        ok1 = Ok(42)
        ok2 = Ok(42)
        assert hash(ok1) == hash(ok2)
        assert {ok1, ok2} == {Ok(42)}

    def test_err_is_hashable(self) -> None:
        """Err should be hashable."""
        # Note: We use the same exception instance because different instances
        # of the same exception are not equal
        err = ValueError("oops")
        err1 = Err(err)
        err2 = Err(err)
        assert hash(err1) == hash(err2)
        assert {err1, err2} == {Err(err)}


class TestResultEquality:
    """Test Result equality."""

    def test_ok_equality(self) -> None:
        """Ok values with same content should be equal."""
        assert Ok(42) == Ok(42)
        assert Ok(42) != Ok(43)
        assert Ok(42) != Err(ValueError("42"))

    def test_err_equality(self) -> None:
        """Err values with same error instance should be equal."""
        err = ValueError("oops")
        assert Err(err) == Err(err)
        assert Err(ValueError("oops")) != Err(ValueError("oops"))  # Different instances
        assert Err(ValueError("oops")) != Err(ValueError("other"))
        assert Err(ValueError("42")) != Ok(42)
