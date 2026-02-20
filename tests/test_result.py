"""Tests for the Result type (Ok and Err)."""

import pytest

from carcinize._exceptions import UnwrapError
from carcinize._option import Nothing, Some
from carcinize._result import Err, Ok, try_except

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

    def test_ok_returns_some(self) -> None:
        """ok() should return Some(value), matching Rust's Result::ok()."""
        result = Ok(42).ok()
        assert isinstance(result, Some)
        assert result.unwrap() == 42

    def test_err_returns_nothing(self) -> None:
        """err() should return Nothing for Ok, matching Rust's Result::err()."""
        result = Ok(42).err()
        assert isinstance(result, Nothing)

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
        result = ok.map_err(str)
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

    def test_ok_returns_nothing(self) -> None:
        """ok() should return Nothing for Err, matching Rust's Result::ok()."""
        result = Err(ValueError("error")).ok()
        assert isinstance(result, Nothing)

    def test_err_returns_some(self) -> None:
        """err() should return Some(error), matching Rust's Result::err()."""
        err = ValueError("oops")
        result = Err(err).err()
        assert isinstance(result, Some)
        assert result.unwrap() is err

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


class TestResultCovariance:
    """Test that Ok[T] and Err[E] are covariant in their type parameters.

    Ok[Subclass] should be assignable to Ok[Superclass] and
    Err[SubException] should be assignable to Err[SuperException]
    since both types are immutable.
    This matches Rust's Result<T, E> which is covariant in both T and E.
    """

    def test_ok_covariance_with_subclass(self) -> None:
        """Ok containing a subclass instance should work where superclass is expected."""

        class Animal:
            pass

        class Dog(Animal):
            def bark(self) -> str:
                return "woof"

        def process_animal_result(result: Ok[Animal]) -> Animal:
            return result.unwrap()

        dog_ok: Ok[Dog] = Ok(Dog())
        # This assignment is valid because Ok is covariant in T
        animal: Animal = process_animal_result(dog_ok)
        assert isinstance(animal, Dog)

    def test_err_covariance_with_subexception(self) -> None:
        """Err containing a subexception should work where superexception is expected."""

        class CustomError(ValueError):
            pass

        def process_error_result(result: Err[ValueError]) -> ValueError:
            return result.unwrap_err()

        custom_err: Err[CustomError] = Err(CustomError("custom"))
        # This assignment is valid because Err is covariant in E
        error: ValueError = process_error_result(custom_err)
        assert isinstance(error, CustomError)

    def test_result_covariance_in_collections(self) -> None:
        """List of Ok[Subclass] should work in contexts expecting Ok[Superclass]."""

        class Base:
            value: int = 0

        class Derived(Base):
            value: int = 42

        results: list[Ok[Derived]] = [Ok(Derived()), Ok(Derived())]
        # Covariance allows this to work
        first_value = results[0].unwrap().value
        assert first_value == 42


# =============================================================================
# Error Context and Origin Tracking Tests
# =============================================================================


class TestErrOriginCapture:
    """Test that Err captures origin information when created."""

    def test_err_captures_origin_on_creation(self) -> None:
        """Err should capture where it was created."""
        err = Err(ValueError("test error"))

        # Origin is now a typed property on Err itself
        assert err.origin  # Non-empty
        assert "test_result.py" in err.origin
        assert "test_err_captures_origin_on_creation" in err.origin

    def test_origin_shows_in_unwrap_traceback(self) -> None:
        """When unwrap() is called, origin should appear in exception notes."""
        err = Err(ValueError("test error"))

        with pytest.raises(ValueError, match="test error") as exc_info:
            err.unwrap()

        # Check that notes were added
        notes = getattr(exc_info.value, "__notes__", [])
        assert len(notes) >= 1

        # The origin note should mention where Err was created
        origin_note = notes[0]
        assert "Error originated at:" in origin_note
        assert "test_result.py" in origin_note

    def test_origin_shows_in_expect_traceback(self) -> None:
        """When expect() is called, origin should appear in exception notes."""
        err = Err(ValueError("test error"))

        with pytest.raises(ValueError, match="test error") as exc_info:
            err.expect("custom message")

        # Check that notes were added
        notes = getattr(exc_info.value, "__notes__", [])
        assert len(notes) >= 1
        assert "Error originated at:" in notes[0]

        # Also check the cause chain
        assert isinstance(exc_info.value.__cause__, UnwrapError)
        assert str(exc_info.value.__cause__) == "custom message"

    def test_origin_not_duplicated_on_multiple_unwraps(self) -> None:
        """Multiple unwrap attempts should not duplicate the origin note."""
        err = Err(ValueError("test error"))

        # First unwrap attempt
        with pytest.raises(ValueError, match="test error"):
            err.unwrap()

        # Second unwrap attempt
        with pytest.raises(ValueError, match="test error") as exc_info:
            err.unwrap()

        # Should still only have one origin note
        notes = getattr(exc_info.value, "__notes__", [])
        origin_notes = [n for n in notes if "Error originated at:" in n]
        assert len(origin_notes) == 1


class TestErrOriginalTracebackPreservation:
    """Test that Err preserves the original traceback when exception was caught."""

    def test_caught_exception_preserves_traceback(self) -> None:
        """When Err wraps a caught exception, original traceback should be preserved."""

        class CustomError(ValueError):
            pass

        def function_that_raises() -> None:
            raise CustomError("original error")

        with pytest.raises(CustomError) as exc_info:
            function_that_raises()

        err = Err(exc_info.value)

        # original_traceback is now a typed property on Err itself
        assert err.original_traceback is not None

        with pytest.raises(ValueError, match="original error"):
            err.unwrap()

        # Check that notes include original raise location
        notes = getattr(exc_info.value, "__notes__", [])
        note_text = "\n".join(notes)
        assert "Exception was originally raised at:" in note_text
        assert "function_that_raises" in note_text

    def test_fresh_exception_has_no_original_traceback(self) -> None:
        """Fresh exceptions (never raised) should not have original traceback."""
        err = Err(ValueError("fresh error"))

        # No original traceback for fresh exceptions
        assert err.original_traceback is None

        with pytest.raises(ValueError, match="fresh error") as exc_info:
            err.unwrap()

        # Should not have the "originally raised at" note
        notes = getattr(exc_info.value, "__notes__", [])
        note_text = "\n".join(notes)
        assert "Exception was originally raised at:" not in note_text


class TestErrContextMethod:
    """Test the context() method for adding context to errors."""

    def test_context_adds_note_to_error(self) -> None:
        """context() should add a context note to the error."""
        err = Err(ValueError("base error"))
        err.context("while processing user data")

        with pytest.raises(ValueError, match="base error") as exc_info:
            err.unwrap()

        notes = getattr(exc_info.value, "__notes__", [])
        assert any("Context: while processing user data" in n for n in notes)

    def test_context_is_chainable(self) -> None:
        """context() should return self for method chaining."""
        err = Err(ValueError("base error"))
        result = err.context("first context").context("second context")
        assert result is err

        with pytest.raises(ValueError, match="base error") as exc_info:
            err.unwrap()

        notes = getattr(exc_info.value, "__notes__", [])
        note_text = "\n".join(notes)
        assert "first context" in note_text
        assert "second context" in note_text

    def test_ok_context_returns_self(self) -> None:
        """Ok.context() should return self unchanged for API consistency."""
        ok = Ok(42)
        result = ok.context("ignored context")
        assert result is ok
        assert result.unwrap() == 42


class TestErrNoteMethod:
    """Test the note() method for adding debugging notes."""

    def test_note_adds_note_to_error(self) -> None:
        """note() should add a note to the error."""
        err = Err(ValueError("base error"))
        err.note("user_id=42")

        with pytest.raises(ValueError, match="base error") as exc_info:
            err.unwrap()

        notes = getattr(exc_info.value, "__notes__", [])
        assert any("user_id=42" in n for n in notes)

    def test_note_is_chainable(self) -> None:
        """note() should return self for method chaining."""
        err = Err(ValueError("base error"))
        result = err.note("first note").note("second note")
        assert result is err

    def test_multiple_notes_preserved(self) -> None:
        """Multiple notes should all be preserved."""
        err = Err(ValueError("base error"))
        err.note("note1").note("note2").note("note3")

        with pytest.raises(ValueError, match="base error") as exc_info:
            err.unwrap()

        notes = getattr(exc_info.value, "__notes__", [])
        note_text = "\n".join(notes)
        assert "note1" in note_text
        assert "note2" in note_text
        assert "note3" in note_text

    def test_ok_note_returns_self(self) -> None:
        """Ok.note() should return self unchanged for API consistency."""
        ok = Ok(42)
        result = ok.note("ignored note")
        assert result is ok


class TestErrContextAndNotesCombined:
    """Test combining context and notes for rich error information."""

    def test_context_and_notes_together(self) -> None:
        """context() and note() should work together."""
        err = Err(ValueError("connection failed"))
        err.context("while fetching user profile").note("user_id=123").note("retry_count=3")

        with pytest.raises(ValueError, match="connection failed") as exc_info:
            err.unwrap()

        notes = getattr(exc_info.value, "__notes__", [])
        note_text = "\n".join(notes)

        assert "Error originated at:" in note_text
        assert "Context: while fetching user profile" in note_text
        assert "user_id=123" in note_text
        assert "retry_count=3" in note_text

    def test_pipeline_with_context(self) -> None:
        """Test realistic pipeline with context at each step."""

        def fetch_config() -> Ok[dict[str, str]] | Err[IOError]:
            return Err(OSError("config file not found"))

        def process_config(config: dict[str, str]) -> Ok[str] | Err[ValueError]:
            return Ok(config.get("key", "default"))

        result = fetch_config().context("while loading application config")

        with pytest.raises(IOError, match="config file not found") as exc_info:
            result.unwrap()

        notes = getattr(exc_info.value, "__notes__", [])
        assert any("while loading application config" in n for n in notes)


# =============================================================================
# try_except Utility Function Tests
# =============================================================================


class TestTryExcept:
    """Test the try_except utility function."""

    def test_try_except_returns_ok_on_success(self) -> None:
        """try_except should return Ok when function succeeds."""
        result = try_except(lambda: 42, ValueError)
        assert isinstance(result, Ok)
        assert result.unwrap() == 42

    def test_try_except_returns_err_on_exception(self) -> None:
        """try_except should return Err when function raises."""
        result = try_except(lambda: int("not a number"), ValueError)
        assert isinstance(result, Err)
        assert isinstance(result.error, ValueError)

    def test_try_except_catches_specified_exception(self) -> None:
        """try_except should only catch specified exception types."""
        result = try_except(lambda: int("not a number"), ValueError)
        assert isinstance(result, Err)

    def test_try_except_propagates_unspecified_exception(self) -> None:
        """try_except should not catch exceptions not in the list."""

        def raise_type_error() -> None:
            raise TypeError("intentional type error")

        with pytest.raises(TypeError):
            try_except(raise_type_error, ValueError)

    def test_try_except_catches_multiple_exception_types(self) -> None:
        """try_except should catch any of the specified exception types."""
        result1 = try_except(lambda: int("x"), ValueError, TypeError)
        assert isinstance(result1, Err)

        def raise_index_error() -> None:
            raise IndexError("intentional")

        result2 = try_except(raise_index_error, ValueError, IndexError)
        assert isinstance(result2, Err)
        assert isinstance(result2.error, IndexError)

    def test_try_except_captures_origin(self) -> None:
        """try_except should capture origin information."""
        result = try_except(lambda: int("x"), ValueError)
        assert isinstance(result, Err)
        assert result.origin  # Non-empty origin string

    def test_try_except_preserves_original_traceback(self) -> None:
        """try_except should preserve the original exception traceback."""
        result = try_except(lambda: int("x"), ValueError)
        assert isinstance(result, Err)
        # The exception was raised inside the lambda, so it has a traceback
        assert result.original_traceback is not None

    def test_try_except_with_no_exception_types_catches_all(self) -> None:
        """try_except with no types should catch all Exceptions."""
        result = try_except(lambda: int("x"))
        assert isinstance(result, Err)


# =============================================================================
# map_err and Error Transformation Tests
# =============================================================================


class TestMapErrWithContext:
    """Test that map_err works correctly with the new context system."""

    def test_map_err_creates_new_err_with_fresh_origin(self) -> None:
        """map_err should create a new Err with fresh origin tracking."""

        def transform_error(e: ValueError) -> TypeError:
            return TypeError(f"converted: {e}")

        original = Err(ValueError("original"))
        transformed = original.map_err(transform_error)

        assert isinstance(transformed, Err)
        assert isinstance(transformed.error, TypeError)

        # The new Err has its own origin (the Err is created inside map_err)
        assert transformed.origin  # Non-empty
        assert "map_err" in transformed.origin

        # The origins are different (transformed has a different origin than original)
        assert transformed.origin != original.origin

    def test_original_err_context_not_lost(self) -> None:
        """Original Err's context should still be accessible."""
        original = Err(ValueError("original"))
        original.note("original note")

        # map_err creates a new Err, original is unchanged
        _ = original.map_err(lambda e: TypeError(str(e)))

        # Original still has its note
        with pytest.raises(ValueError, match="original") as exc_info:
            original.unwrap()
        notes = getattr(exc_info.value, "__notes__", [])
        assert any("original note" in n for n in notes)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases for error context tracking."""

    def test_nested_err_creation(self) -> None:
        """Nested Err creation should track innermost origin."""

        def inner() -> Err[ValueError]:
            return Err(ValueError("inner error"))

        def outer() -> Err[ValueError]:
            return inner()

        err = outer()

        # Origin should point to inner() where Err was actually created
        assert "inner" in err.origin

        # Also verify it shows up in the notes when unwrapped
        with pytest.raises(ValueError, match="inner error") as exc_info:
            err.unwrap()

        notes = getattr(exc_info.value, "__notes__", [])
        origin_note = next(n for n in notes if "Error originated at:" in n)
        assert "inner" in origin_note

    def test_err_with_exception_subclass(self) -> None:
        """Err should work with custom exception subclasses."""

        class CustomError(Exception):
            def __init__(self, code: int, message: str) -> None:
                super().__init__(message)
                self.code = code

        err = Err(CustomError(404, "not found"))
        err.note(f"error_code={err.error.code}")

        with pytest.raises(CustomError) as exc_info:
            err.unwrap()

        assert exc_info.value.code == 404
        notes = getattr(exc_info.value, "__notes__", [])
        assert any("error_code=404" in n for n in notes)

    def test_and_then_preserves_err_context(self) -> None:
        """and_then should preserve Err context when returning self."""
        err: Err[ValueError] = Err(ValueError("original"))
        err.note("test note")

        result = err.and_then(lambda x: Ok(x * 2))
        assert result is err

    def test_or_else_can_add_context_to_recovered_err(self) -> None:
        """or_else can be used to add context before recovery."""

        def recover(_e: ValueError) -> Ok[int] | Err[ValueError]:
            return Ok(0)  # Recovery successful

        err: Err[ValueError] = Err(ValueError("recoverable"))
        result = err.or_else(recover)

        assert isinstance(result, Ok)
        assert result.unwrap() == 0

    def test_chained_operations_with_context(self) -> None:
        """Test realistic chained operations with context."""

        def parse_int(s: str) -> Ok[int] | Err[ValueError]:
            try:
                return Ok(int(s))
            except ValueError as e:
                return Err(e)

        def validate_positive(n: int) -> Ok[int] | Err[ValueError]:
            if n <= 0:
                return Err(ValueError("must be positive"))
            return Ok(n)

        # Chain with context
        result = parse_int("abc").context("while parsing user input").and_then(validate_positive)

        assert isinstance(result, Err)
        with pytest.raises(ValueError, match="invalid literal") as exc_info:
            result.unwrap()

        notes = getattr(exc_info.value, "__notes__", [])
        assert any("while parsing user input" in n for n in notes)
