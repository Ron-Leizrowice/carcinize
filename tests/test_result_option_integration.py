"""Integration tests for Result and Option type interactions."""

import pytest

from carcinize.option import Nothing, Some
from carcinize.result import Err, Ok

# =============================================================================
# Option to Result Conversion
# =============================================================================


class TestOptionToResultConversion:
    """Test converting Option to Result."""

    def test_some_ok_or_returns_ok(self) -> None:
        """Some.ok_or() should return Ok with the value."""
        option = Some(42)
        result = option.ok_or(ValueError("missing"))

        assert isinstance(result, Ok)
        assert result.value == 42

    def test_nothing_ok_or_returns_err(self) -> None:
        """Nothing.ok_or() should return Err with the provided error."""
        option = Nothing()
        err = ValueError("missing")
        result = option.ok_or(err)

        assert isinstance(result, Err)
        assert result.error is err

    def test_some_ok_or_else_returns_ok(self) -> None:
        """Some.ok_or_else() should return Ok without calling the function."""
        called = False

        def make_error() -> ValueError:
            nonlocal called
            called = True
            return ValueError("error")

        option = Some(42)
        result = option.ok_or_else(make_error)

        assert isinstance(result, Ok)
        assert result.value == 42
        assert not called  # Function should not be called

    def test_nothing_ok_or_else_calls_function(self) -> None:
        """Nothing.ok_or_else() should call the function to create error."""
        option = Nothing()
        result = option.ok_or_else(lambda: ValueError("computed error"))

        assert isinstance(result, Err)
        assert isinstance(result.error, ValueError)


# =============================================================================
# Nested Types
# =============================================================================


class TestNestedTypes:
    """Test nested Result and Option types."""

    def test_option_containing_result(self) -> None:
        """Option can contain a Result."""
        opt = Some(Ok(42))

        assert opt.is_some()
        inner = opt.unwrap()
        assert isinstance(inner, Ok)
        assert inner.value == 42

    def test_result_containing_option(self) -> None:
        """Result can contain an Option."""
        res = Ok(Some(42))

        assert res.is_ok()
        inner = res.unwrap()
        assert isinstance(inner, Some)
        assert inner.value == 42

    def test_deeply_nested_unwrap(self) -> None:
        """Can unwrap deeply nested types."""
        nested = Some(Ok(Some(42)))

        # Unwrap step by step
        result = nested.unwrap()  # Ok[Some[int]]
        option = result.unwrap()  # Some[int]
        value = option.unwrap()  # int

        assert value == 42

    def test_option_of_err(self) -> None:
        """Option can contain an Err."""
        err = ValueError("oops")
        opt = Some(Err(err))

        assert opt.is_some()
        inner = opt.unwrap()
        assert isinstance(inner, Err)
        assert inner.error is err

    def test_result_of_nothing(self) -> None:
        """Result can contain Nothing."""
        res = Ok(Nothing())

        assert res.is_ok()
        inner = res.unwrap()
        assert isinstance(inner, Nothing)


# =============================================================================
# Chaining Across Types
# =============================================================================


class TestChainingAcrossTypes:
    """Test chaining operations across Result and Option."""

    def test_option_to_result_to_map(self) -> None:
        """Convert Option to Result and map the value."""
        option = Some("42")
        result = option.ok_or(ValueError("no input")).map(int)

        assert isinstance(result, Ok)
        assert result.value == 42

    def test_nothing_to_result_short_circuits(self) -> None:
        """Nothing converted to Result should carry the error through."""
        option = Nothing()
        err = ValueError("no input")
        result = option.ok_or(err).map(int)

        assert isinstance(result, Err)
        assert result.error is err

    def test_result_map_to_option(self) -> None:
        """Map Result value to Option."""

        def find_first_even(nums: list[int]) -> Some[int] | Nothing:
            for n in nums:
                if n % 2 == 0:
                    return Some(n)
            return Nothing()

        result = Ok([1, 3, 4, 5])
        mapped = result.map(find_first_even)

        assert isinstance(mapped, Ok)
        assert isinstance(mapped.value, Some)
        assert mapped.value.value == 4

    def test_and_then_with_ok(self) -> None:
        """and_then on Ok should call the function."""

        def double_ok(x: int) -> Ok[int] | Err[ValueError]:
            return Ok(x * 2)

        result = Ok(5).and_then(double_ok)
        assert isinstance(result, Ok)
        assert result.value == 10

    def test_and_then_with_err(self) -> None:
        """and_then on Err should return self."""
        err = Err(ValueError("oops"))
        result = err.and_then(lambda x: Ok(x * 2))
        assert result is err

    def test_option_and_then_to_option(self) -> None:
        """Option and_then can chain to another Option."""

        def get_length_if_long(s: str) -> Some[int] | Nothing:
            if len(s) > 3:
                return Some(len(s))
            return Nothing()

        # Long string
        result = Some("hello").and_then(get_length_if_long)
        assert isinstance(result, Some)
        assert result.value == 5

        # Short string
        result = Some("hi").and_then(get_length_if_long)
        assert isinstance(result, Nothing)

        # Nothing
        result = Nothing().and_then(get_length_if_long)
        assert isinstance(result, Nothing)


# =============================================================================
# Pattern Matching with Nested Types
# =============================================================================


class TestPatternMatchingNested:
    """Test pattern matching with nested types."""

    def test_match_option_of_ok(self) -> None:
        """Pattern match Option containing Ok."""
        opt = Some(Ok(42))

        match opt:
            case Some(Ok(value)):
                assert value == 42
            case _:
                pytest.fail("Should match Some(Ok(value))")

    def test_match_option_of_err(self) -> None:
        """Pattern match Option containing Err."""
        err = ValueError("oops")
        opt = Some(Err(err))

        match opt:
            case Some(Err(error)):
                assert error is err
            case _:
                pytest.fail("Should match Some(Err(error))")

    def test_match_result_of_some(self) -> None:
        """Pattern match Result containing Some."""
        res = Ok(Some(42))

        match res:
            case Ok(Some(value)):
                assert value == 42
            case _:
                pytest.fail("Should match Ok(Some(value))")

    def test_match_result_of_nothing(self) -> None:
        """Pattern match Result containing Nothing."""
        res = Ok(Nothing())

        match res:
            case Ok(Nothing()):
                pass  # Expected
            case _:
                pytest.fail("Should match Ok(Nothing())")

    def test_match_nothing_outer(self) -> None:
        """Pattern match Nothing at outer level."""
        opt = Nothing()

        match opt:
            case Nothing():
                pass  # Expected
            case _:
                pytest.fail("Should match Nothing()")


# =============================================================================
# Practical Use Cases
# =============================================================================


class TestPracticalUseCases:
    """Test practical use cases combining Result and Option."""

    def test_database_lookup_pattern(self) -> None:
        """Simulate database lookup that may fail or return nothing."""

        class DatabaseError(Exception):
            pass

        def db_query(query: str) -> Ok[Some[dict[str, str]] | Nothing] | Err[DatabaseError]:
            """Simulate a database query that can fail or return no results."""
            if query == "error":
                return Err(DatabaseError("Database connection failed"))
            if query == "alice":
                return Ok(Some({"name": "Alice", "email": "alice@example.com"}))
            return Ok(Nothing())

        # Successful query with result
        result = db_query("alice")
        assert isinstance(result, Ok)
        assert isinstance(result.value, Some)
        assert result.value.value["name"] == "Alice"

        # Successful query with no result
        result = db_query("unknown")
        assert isinstance(result, Ok)
        assert isinstance(result.value, Nothing)

        # Failed query
        result = db_query("error")
        assert isinstance(result, Err)
        assert "Database" in str(result.error)

    def test_optional_field_extraction(self) -> None:
        """Extract optional fields from a dict-like structure."""

        def get_field(data: dict[str, str], key: str) -> Some[str] | Nothing:
            if key in data:
                return Some(data[key])
            return Nothing()

        data = {"name": "Alice", "email": "alice@example.com"}

        # Existing field
        result = get_field(data, "name")
        assert isinstance(result, Some)
        assert result.value == "Alice"

        # Missing field
        result = get_field(data, "phone")
        assert isinstance(result, Nothing)

        # Convert to Result for error handling
        result = get_field(data, "phone").ok_or(KeyError("phone is required"))
        assert isinstance(result, Err)
        assert isinstance(result.error, KeyError)

    def test_validation_with_option_result(self) -> None:
        """Validate values using both Option and Result."""

        def parse_positive(s: str) -> Ok[int] | Err[ValueError]:
            try:
                n = int(s)
                if n > 0:
                    return Ok(n)
                return Err(ValueError("must be positive"))
            except ValueError as e:
                return Err(e)

        def maybe_halve(n: int) -> Some[int] | Nothing:
            """Halve if even, otherwise Nothing."""
            if n % 2 == 0:
                return Some(n // 2)
            return Nothing()

        # Valid even number
        result = parse_positive("10").map(maybe_halve)
        assert isinstance(result, Ok)
        assert isinstance(result.value, Some)
        assert result.value.value == 5

        # Valid odd number
        result = parse_positive("7").map(maybe_halve)
        assert isinstance(result, Ok)
        assert isinstance(result.value, Nothing)

        # Invalid input
        result = parse_positive("abc").map(maybe_halve)
        assert isinstance(result, Err)
        assert isinstance(result.error, ValueError)


# =============================================================================
# Equality and Hashing with Nested Types
# =============================================================================


class TestNestedEquality:
    """Test equality with nested types."""

    def test_nested_some_ok_equality(self) -> None:
        """Nested Some[Ok] should have value equality."""
        opt1 = Some(Ok(42))
        opt2 = Some(Ok(42))
        opt3 = Some(Ok(43))

        assert opt1 == opt2
        assert opt1 != opt3

    def test_nested_ok_some_equality(self) -> None:
        """Nested Ok[Some] should have value equality."""
        res1 = Ok(Some(42))
        res2 = Ok(Some(42))
        res3 = Ok(Nothing())

        assert res1 == res2
        assert res1 != res3

    def test_nested_types_hashable(self) -> None:
        """Nested types should be hashable."""
        opt1 = Some(Ok(42))
        opt2 = Some(Ok(42))

        # Should be usable in sets
        s = {opt1, opt2}
        assert len(s) == 1

        # Should be usable as dict keys
        d = {opt1: "value"}
        assert d[opt2] == "value"

    def test_mixed_nesting_not_equal(self) -> None:
        """Different nesting structures should not be equal."""
        opt_ok = Some(Ok(42))
        opt_err = Some(Err(ValueError("42")))
        ok_some = Ok(Some(42))

        assert opt_ok != opt_err
        assert opt_ok != ok_some  # Different structure entirely
