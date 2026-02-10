"""Integration tests for combinations of carcinize types.

Tests the intersection of multiple functionalities:
- Iter + Result/Option
- Struct + Result/Option
- Lazy/OnceCell + other types
- Full pipeline scenarios
- try_except integration
- Complex method chaining
"""

import threading

import pytest
from pydantic import ValidationError

from carcinize import (
    Err,
    Iter,
    Lazy,
    Nothing,
    Ok,
    OnceCell,
    Some,
    Struct,
    try_except,
)

# =============================================================================
# Iter + Result Integration
# =============================================================================


class TestIterWithResult:
    """Test Iter operations with Result types."""

    def test_filter_map_with_result_to_option(self) -> None:
        """Use filter_map to extract Ok values from Results."""

        def result_to_option(r: Ok[int] | Err[ValueError]) -> Some[int] | Nothing:
            match r:
                case Ok(value):
                    return Some(value)
                case Err():
                    return Nothing()

        results: list[Ok[int] | Err[ValueError]] = [
            Ok(1),
            Err(ValueError("bad")),
            Ok(2),
            Err(ValueError("bad")),
            Ok(3),
        ]

        values = Iter(results).filter_map(result_to_option).collect_list()
        assert values == [1, 2, 3]

    def test_partition_ok_and_err(self) -> None:
        """Partition Results into Ok and Err values."""
        results: list[Ok[int] | Err[ValueError]] = [
            Ok(1),
            Err(ValueError("a")),
            Ok(2),
            Err(ValueError("b")),
        ]

        oks, errs = Iter(results).partition(lambda r: r.is_ok())
        assert len(oks) == 2
        assert len(errs) == 2
        assert all(r.is_ok() for r in oks)
        assert all(r.is_err() for r in errs)

    def test_fold_results(self) -> None:
        """Fold over Results, short-circuiting on first Err."""

        def try_add(acc: Ok[int] | Err[ValueError], x: int) -> Ok[int] | Err[ValueError]:
            if x < 0:
                return Err(ValueError(f"negative: {x}"))
            match acc:
                case Ok(total):
                    return Ok(total + x)
                case err:
                    return err

        # All positive - success
        result = Iter([1, 2, 3]).fold(Ok(0), try_add)
        assert result == Ok(6)

        # Contains negative - error
        result = Iter([1, -2, 3]).fold(Ok(0), try_add)
        assert isinstance(result, Err)

    def test_find_first_ok(self) -> None:
        """Find the first Ok value in a sequence."""
        results: list[Ok[int] | Err[ValueError]] = [
            Err(ValueError("1")),
            Err(ValueError("2")),
            Ok(42),
            Ok(100),
        ]

        first_ok = Iter(results).find(lambda r: r.is_ok())
        assert first_ok == Some(Ok(42))

    def test_all_ok_check(self) -> None:
        """Check if all Results are Ok."""
        all_ok: list[Ok[int] | Err[ValueError]] = [Ok(1), Ok(2), Ok(3)]
        some_err: list[Ok[int] | Err[ValueError]] = [Ok(1), Err(ValueError("x")), Ok(3)]

        assert Iter(all_ok).all(lambda r: r.is_ok()) is True
        assert Iter(some_err).all(lambda r: r.is_ok()) is False

    def test_collect_results_pattern(self) -> None:
        """Collect all Ok values or return first Err (like Rust's collect::<Result<Vec<_>, _>>)."""

        def collect_results(
            results: list[Ok[int] | Err[ValueError]],
        ) -> Ok[list[int]] | Err[ValueError]:
            values: list[int] = []
            for r in results:
                match r:
                    case Ok(v):
                        values.append(v)
                    case Err() as e:
                        return e
            return Ok(values)

        # All Ok
        result = collect_results([Ok(1), Ok(2), Ok(3)])
        assert result == Ok([1, 2, 3])

        # Has Err
        err = ValueError("oops")
        result = collect_results([Ok(1), Err(err), Ok(3)])
        assert isinstance(result, Err)
        assert result.error is err


# =============================================================================
# Iter + Option Integration
# =============================================================================


class TestIterWithOption:
    """Test Iter operations with Option types."""

    def test_flatten_options(self) -> None:
        """Extract values from a list of Options using filter_map."""
        options: list[Some[int] | Nothing] = [Some(1), Nothing(), Some(2), Nothing(), Some(3)]

        values = Iter(options).filter_map(lambda x: x).collect_list()
        assert values == [1, 2, 3]

    def test_find_map_with_option_returning_function(self) -> None:
        """Use find_map to find first matching transformation."""

        def parse_if_digit(s: str) -> Some[int] | Nothing:
            if s.isdigit():
                return Some(int(s))
            return Nothing()

        result = Iter(["a", "b", "42", "c"]).find_map(parse_if_digit)
        assert result == Some(42)

        result = Iter(["a", "b", "c"]).find_map(parse_if_digit)
        assert isinstance(result, Nothing)

    def test_reduce_options(self) -> None:
        """Reduce Option values, combining Some values."""

        def combine(a: Some[int] | Nothing, b: Some[int] | Nothing) -> Some[int] | Nothing:
            match (a, b):
                case (Some(x), Some(y)):
                    return Some(x + y)
                case (Some(_), Nothing()):
                    return a
                case (Nothing(), Some(_)):
                    return b
                case _:
                    return Nothing()

        options: list[Some[int] | Nothing] = [Some(1), Nothing(), Some(2), Some(3)]
        result = Iter(options).reduce(combine)
        assert result == Some(Some(6))

    def test_zip_with_options(self) -> None:
        """Zip iterators and combine with Option.zip."""
        a_opts: list[Some[int] | Nothing] = [Some(1), Some(2), Nothing()]
        b_opts: list[Some[str] | Nothing] = [Some("a"), Nothing(), Some("c")]

        pairs = Iter(a_opts).zip(b_opts).collect_list()
        assert len(pairs) == 3

        # Combine each pair with Option.zip
        combined = [a.zip(b) for a, b in pairs]
        assert combined[0] == Some((1, "a"))
        assert isinstance(combined[1], Nothing)
        assert isinstance(combined[2], Nothing)


# =============================================================================
# Struct + Result/Option Integration
# =============================================================================


class TestStructWithResultOption:
    """Test Struct with Result and Option types."""

    def test_struct_with_optional_fields(self) -> None:
        """Struct can have Option-typed fields."""

        class UserProfile(Struct, mut=True):
            name: str
            email: Some[str] | Nothing

        # With email
        user = UserProfile(name="Alice", email=Some("alice@example.com"))
        assert user.email == Some("alice@example.com")

        # Without email
        user = UserProfile(name="Bob", email=Nothing())
        assert isinstance(user.email, Nothing)

    def test_try_from_returns_result(self) -> None:
        """try_from integrates with Result type."""

        class Config(Struct, mut=True):
            host: str
            port: int

        # Valid
        result = Config.try_from({"host": "localhost", "port": 8080})
        assert isinstance(result, Ok)
        assert result.unwrap().host == "localhost"

        # Invalid - can chain with Result methods
        result = Config.try_from({"host": "localhost"})
        default = result.unwrap_or(Config(host="default", port=80))
        assert default.port == 80

    def test_parse_list_of_structs(self) -> None:
        """Parse a list of dicts into Structs, collecting errors."""

        class Item(Struct, mut=True):
            id: int
            name: str

        data = [
            {"id": 1, "name": "one"},
            {"id": "bad", "name": "two"},  # Invalid
            {"id": 3, "name": "three"},
        ]

        results = [Item.try_from(d) for d in data]
        oks, errs = Iter(results).partition(lambda r: r.is_ok())

        assert len(oks) == 2
        assert len(errs) == 1

    def test_struct_as_iter_element(self) -> None:
        """Structs can be elements in Iter operations."""

        class Point(Struct):
            x: int
            y: int

        points = [Point(x=1, y=2), Point(x=3, y=4), Point(x=5, y=6)]

        # Map over structs
        x_values = Iter(points).map(lambda p: p.x).collect_list()
        assert x_values == [1, 3, 5]

        # Filter structs
        filtered = Iter(points).filter(lambda p: p.x > 2).collect_list()
        assert len(filtered) == 2

        # Find struct
        found = Iter(points).find(lambda p: p.x == 3)
        assert found == Some(Point(x=3, y=4))

    def test_struct_in_option(self) -> None:
        """Option can contain Struct values."""

        class User(Struct):
            id: int
            name: str

        opt: Some[User] | Nothing = Some(User(id=1, name="Alice"))

        # Map over the struct
        name = opt.map(lambda u: u.name)
        assert name == Some("Alice")

        # Chain operations
        upper_name = opt.map(lambda u: u.name).map(str.upper)
        assert upper_name == Some("ALICE")


# =============================================================================
# Lazy/OnceCell + Other Types Integration
# =============================================================================


class TestLazyWithOtherTypes:
    """Test Lazy and OnceCell with other carcinize types."""

    def test_lazy_returning_result(self) -> None:
        """Lazy can wrap a function returning Result."""

        def fetch_config() -> Ok[dict[str, int]] | Err[IOError]:
            return Ok({"timeout": 30, "retries": 3})

        lazy_config: Lazy[Ok[dict[str, int]] | Err[IOError]] = Lazy(fetch_config)

        # Not computed yet
        assert not lazy_config.is_computed()

        # Get and unwrap
        config = lazy_config.get()
        assert isinstance(config, Ok)
        assert config.unwrap()["timeout"] == 30

    def test_lazy_returning_option(self) -> None:
        """Lazy can wrap a function returning Option."""
        call_count = 0

        def maybe_load() -> Some[str] | Nothing:
            nonlocal call_count
            call_count += 1
            return Some("loaded data")

        lazy_data: Lazy[Some[str] | Nothing] = Lazy(maybe_load)

        # Access multiple times
        result1 = lazy_data.get()
        result2 = lazy_data.get()

        assert result1 == result2 == Some("loaded data")
        assert call_count == 1  # Only called once

    def test_oncecell_with_struct(self) -> None:
        """OnceCell can store Struct values."""

        class Settings(Struct):
            debug: bool
            log_level: str

        cell: OnceCell[Settings] = OnceCell()

        # Set once
        result = cell.set(Settings(debug=True, log_level="INFO"))
        assert isinstance(result, Ok)

        # Retrieve
        settings = cell.get()
        assert settings == Some(Settings(debug=True, log_level="INFO"))

        # Can't set again
        result = cell.set(Settings(debug=False, log_level="ERROR"))
        assert isinstance(result, Err)

    def test_lazy_struct_initialization(self) -> None:
        """Lazy can defer expensive Struct construction."""

        class ExpensiveConfig(Struct, mut=True):
            values: list[int]

        def create_config() -> ExpensiveConfig:
            # Simulate expensive computation
            return ExpensiveConfig(values=list(range(100)))

        lazy_config = Lazy(create_config)

        # Not created yet
        assert not lazy_config.is_computed()

        # Access triggers creation
        config = lazy_config.get()
        assert len(config.values) == 100
        assert lazy_config.is_computed()

    def test_oncecell_get_or_init_with_result(self) -> None:
        """get_or_init can initialize with Result-returning logic."""
        cell: OnceCell[int] = OnceCell()

        def fallible_init() -> int:
            # Could fail in real code, but here just returns value
            return 42

        # First call initializes
        value = cell.get_or_init(fallible_init)
        assert value == 42

        # Subsequent calls return cached value
        value = cell.get_or_init(lambda: 100)
        assert value == 42


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestFullPipelines:
    """Test realistic pipelines combining multiple types."""

    def test_parse_validate_transform_pipeline(self) -> None:
        """Full pipeline: parse JSON -> validate -> transform -> collect."""

        class UserInput(Struct, mut=True):
            name: str
            age: int

        raw_data = [
            '{"name": "Alice", "age": 30}',
            '{"name": "Bob"}',  # Missing age
            '{"name": "Charlie", "age": 25}',
            "not json",  # Invalid JSON
        ]

        # Parse all
        parsed = [UserInput.try_from(d) for d in raw_data]

        # Extract valid users and filter by age
        adults = (
            Iter(parsed)
            .filter(lambda r: r.is_ok())
            .map(lambda r: r.unwrap())
            .filter(lambda u: u.age >= 18)
            .collect_list()
        )

        assert len(adults) == 2
        assert adults[0].name == "Alice"
        assert adults[1].name == "Charlie"

    def test_cached_computation_with_fallback(self) -> None:
        """Use OnceCell for caching with Option fallback."""
        cache: OnceCell[str] = OnceCell()

        def get_or_compute(compute: str) -> str:
            # Try cache first
            cached = cache.get()
            match cached:
                case Some(value):
                    return value
                case Nothing():
                    # Compute and cache
                    cache.set(compute)
                    return compute

        # First call computes
        result1 = get_or_compute("computed")
        assert result1 == "computed"

        # Second call returns cached (ignores new value)
        result2 = get_or_compute("ignored")
        assert result2 == "computed"

    def test_iterator_with_lazy_evaluation(self) -> None:
        """Iter operations are lazy until collected."""
        side_effects: list[int] = []

        def track(x: int) -> int:
            side_effects.append(x)
            return x * 2

        # Create iterator chain - no side effects yet
        it = Iter([1, 2, 3, 4, 5]).map(track).filter(lambda x: x > 4)
        assert side_effects == []

        # Collect triggers evaluation
        result = it.collect_list()
        assert result == [6, 8, 10]
        assert side_effects == [1, 2, 3, 4, 5]  # All were mapped

    def test_error_handling_pipeline(self) -> None:
        """Pipeline with error handling at each stage."""

        class ParseError(Exception):
            pass

        class NegativeValueError(Exception):
            pass

        def parse(s: str) -> Ok[int] | Err[ParseError]:
            try:
                return Ok(int(s))
            except ValueError:
                return Err(ParseError(f"Cannot parse: {s}"))

        def validate(n: int) -> Ok[int] | Err[NegativeValueError]:
            if n < 0:
                return Err(NegativeValueError(f"Negative not allowed: {n}"))
            return Ok(n)

        def process(s: str) -> Ok[int] | Err[ParseError | NegativeValueError]:
            return parse(s).and_then(validate)

        # Success path
        assert process("42") == Ok(42)

        # Parse error
        result = process("abc")
        assert isinstance(result, Err)
        assert isinstance(result.error, ParseError)

        # Validation error
        result = process("-5")
        assert isinstance(result, Err)
        assert isinstance(result.error, NegativeValueError)

    def test_struct_collection_with_deduplication(self) -> None:
        """Collect unique structs using Iter."""

        class Tag(Struct):
            name: str

        tags = [
            Tag(name="python"),
            Tag(name="rust"),
            Tag(name="python"),  # Duplicate
            Tag(name="go"),
            Tag(name="rust"),  # Duplicate
        ]

        unique_tags = Iter(tags).unique().collect_list()
        assert len(unique_tags) == 3
        assert Tag(name="python") in unique_tags
        assert Tag(name="rust") in unique_tags
        assert Tag(name="go") in unique_tags

    def test_grouped_aggregation(self) -> None:
        """Group structs and aggregate."""

        class Sale(Struct, mut=True):
            product: str
            amount: int

        sales = [
            Sale(product="A", amount=100),
            Sale(product="B", amount=200),
            Sale(product="A", amount=150),
            Sale(product="B", amount=50),
            Sale(product="A", amount=75),
        ]

        # Group by product
        grouped = Iter(sales).group_by(lambda s: s.product)
        assert len(grouped) == 2

        # Calculate totals
        totals = {product: sum(s.amount for s in items) for product, items in grouped.items()}
        assert totals["A"] == 325
        assert totals["B"] == 250


# =============================================================================
# Thread Safety Integration
# =============================================================================


class TestThreadSafetyIntegration:
    """Test thread safety across type combinations."""

    def test_oncecell_with_struct_concurrent(self) -> None:
        """OnceCell with Struct is thread-safe."""

        class Config(Struct):
            value: int

        cell: OnceCell[Config] = OnceCell()
        results: list[Config] = []
        lock = threading.Lock()

        def worker(n: int) -> None:
            config = cell.get_or_init(lambda: Config(value=n))
            with lock:
                results.append(config)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads got the same config
        assert len(results) == 10
        assert all(r == results[0] for r in results)

    def test_lazy_result_concurrent(self) -> None:
        """Lazy returning Result is thread-safe."""
        call_count = 0
        call_lock = threading.Lock()

        def expensive() -> Ok[int] | Err[ValueError]:
            nonlocal call_count
            with call_lock:
                call_count += 1
            return Ok(42)

        lazy: Lazy[Ok[int] | Err[ValueError]] = Lazy(expensive)
        results: list[Ok[int] | Err[ValueError]] = []
        results_lock = threading.Lock()

        def worker() -> None:
            result = lazy.get()
            with results_lock:
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count == 1
        assert all(r == Ok(42) for r in results)


# =============================================================================
# try_except Integration
# =============================================================================


class TestTryExceptIntegration:
    """Test try_except with other carcinize types."""

    def test_try_except_with_struct_validation(self) -> None:
        """try_except captures Struct validation errors."""

        class User(Struct):
            name: str
            age: int

        # Valid creation
        result = try_except(lambda: User(name="Alice", age=30), ValidationError)
        assert isinstance(result, Ok)
        assert result.unwrap().name == "Alice"

        # Invalid creation - wrong type (intentionally passing wrong type to test validation)
        result = try_except(lambda: User(name="Bob", age="not a number"), ValidationError)  # ty: ignore[invalid-argument-type]
        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)

    def test_try_except_in_iter_pipeline(self) -> None:
        """try_except can be used within Iter operations."""

        def parse_int(s: str) -> Ok[int] | Err[Exception]:
            return try_except(lambda: int(s), ValueError)

        data = ["1", "2", "bad", "4", "oops"]

        # Parse all, keeping results
        results = Iter(data).map(parse_int).collect_list()
        assert len(results) == 5
        assert results[0] == Ok(1)
        assert results[1] == Ok(2)
        assert isinstance(results[2], Err)
        assert results[3] == Ok(4)
        assert isinstance(results[4], Err)

    def test_try_except_with_multiple_exceptions(self) -> None:
        """try_except catches multiple exception types."""

        def risky_division(a: int, b: int) -> float:
            if a < 0:
                raise ValueError("negative not allowed")
            return a / b

        # Success
        result = try_except(lambda: risky_division(10, 2), ValueError, ZeroDivisionError)
        assert result == Ok(5.0)

        # ValueError
        result = try_except(lambda: risky_division(-1, 2), ValueError, ZeroDivisionError)
        assert isinstance(result, Err)
        assert isinstance(result.error, ValueError)

        # ZeroDivisionError
        result = try_except(lambda: risky_division(10, 0), ValueError, ZeroDivisionError)
        assert isinstance(result, Err)
        assert isinstance(result.error, ZeroDivisionError)

    def test_try_except_uncaught_propagates(self) -> None:
        """Uncaught exceptions propagate through try_except."""

        def raises_type_error() -> int:
            raise TypeError("unexpected")

        # Only catching ValueError - TypeError should propagate
        with pytest.raises(TypeError, match="unexpected"):
            try_except(raises_type_error, ValueError)

    def test_try_except_with_lazy(self) -> None:
        """try_except works with Lazy initialization."""

        def fallible_init() -> int:
            return int("42")

        lazy = Lazy(lambda: try_except(fallible_init, ValueError))

        assert not lazy.is_computed()
        result = lazy.get()
        assert result == Ok(42)
        assert lazy.is_computed()


# =============================================================================
# Iter + Struct Sorting and Grouping
# =============================================================================


class TestIterStructSortingGrouping:
    """Test Iter sorting and grouping operations with Structs."""

    def test_sort_structs_by_field(self) -> None:
        """Sort structs by a specific field."""

        class Person(Struct):
            name: str
            age: int

        people = [
            Person(name="Charlie", age=30),
            Person(name="Alice", age=25),
            Person(name="Bob", age=35),
        ]

        # Sort by age
        by_age = Iter(people).sorted_by(lambda p: p.age)
        assert [p.name for p in by_age] == ["Alice", "Charlie", "Bob"]

        # Sort by name
        by_name = Iter(people).sorted_by(lambda p: p.name)
        assert [p.name for p in by_name] == ["Alice", "Bob", "Charlie"]

    def test_sort_structs_reverse(self) -> None:
        """Sort structs in reverse order."""

        class Score(Struct):
            player: str
            points: int

        scores = [
            Score(player="A", points=100),
            Score(player="B", points=300),
            Score(player="C", points=200),
        ]

        # Sort by points descending (highest first)
        leaderboard = Iter(scores).sorted_by(lambda s: s.points, reverse=True)
        assert [s.player for s in leaderboard] == ["B", "C", "A"]

    def test_group_structs_and_aggregate(self) -> None:
        """Group structs by field and perform aggregation."""

        class Transaction(Struct, mut=True):
            category: str
            amount: float

        txns = [
            Transaction(category="food", amount=25.50),
            Transaction(category="transport", amount=15.00),
            Transaction(category="food", amount=30.00),
            Transaction(category="entertainment", amount=50.00),
            Transaction(category="food", amount=12.50),
        ]

        grouped = Iter(txns).group_by(lambda t: t.category)

        # Check grouping
        assert len(grouped["food"]) == 3
        assert len(grouped["transport"]) == 1
        assert len(grouped["entertainment"]) == 1

        # Aggregate totals
        totals = {cat: sum(t.amount for t in items) for cat, items in grouped.items()}
        assert totals["food"] == 68.00
        assert totals["transport"] == 15.00

    def test_unique_structs_by_field(self) -> None:
        """Deduplicate structs by a specific field."""

        class Event(Struct, mut=True):
            event_id: int
            name: str
            timestamp: int

        # Same event_id but different timestamps (duplicates)
        events = [
            Event(event_id=1, name="click", timestamp=100),
            Event(event_id=2, name="view", timestamp=101),
            Event(event_id=1, name="click", timestamp=102),  # Duplicate event_id
            Event(event_id=3, name="purchase", timestamp=103),
            Event(event_id=2, name="view", timestamp=104),  # Duplicate event_id
        ]

        unique = Iter(events).unique_by(lambda e: e.event_id).collect_list()
        assert len(unique) == 3
        assert [e.event_id for e in unique] == [1, 2, 3]


# =============================================================================
# Option + Result Chaining Edge Cases
# =============================================================================


class TestOptionResultChainingEdgeCases:
    """Test edge cases in Option and Result chaining."""

    def test_option_filter_then_ok_or(self) -> None:
        """Chain Option.filter with ok_or conversion."""
        opt = Some(10)

        # Filter passes -> Ok
        result = opt.filter(lambda x: x > 5).ok_or(ValueError("too small"))
        assert result == Ok(10)

        # Filter fails -> Err
        result = opt.filter(lambda x: x > 20).ok_or(ValueError("too small"))
        assert isinstance(result, Err)

    def test_result_and_then_returning_err(self) -> None:
        """and_then can transform Ok to Err."""

        def validate_positive(n: int) -> Ok[int] | Err[ValueError]:
            if n > 0:
                return Ok(n)
            return Err(ValueError("must be positive"))

        # Ok -> Ok
        result = Ok(5).and_then(validate_positive)
        assert result == Ok(5)

        # Ok -> Err
        result = Ok(-5).and_then(validate_positive)
        assert isinstance(result, Err)

    def test_option_or_else_chain(self) -> None:
        """Chain multiple or_else for fallback logic."""

        def try_primary() -> Some[str] | Nothing:
            return Nothing()

        def try_secondary() -> Some[str] | Nothing:
            return Nothing()

        def try_tertiary() -> Some[str] | Nothing:
            return Some("tertiary")

        result = try_primary().or_else(try_secondary).or_else(try_tertiary)
        assert result == Some("tertiary")

    def test_result_map_err_chain(self) -> None:
        """Chain map_err to transform error types."""

        class LowLevelError(Exception):
            pass

        class HighLevelError(Exception):
            def __init__(self, cause: Exception) -> None:
                self.cause = cause
                super().__init__(f"High level error: {cause}")

        err: Ok[int] | Err[LowLevelError] = Err(LowLevelError("disk full"))
        transformed = err.map_err(HighLevelError)

        assert isinstance(transformed, Err)
        assert isinstance(transformed.error, HighLevelError)
        assert isinstance(transformed.error.cause, LowLevelError)


# =============================================================================
# Iter Window and Chunk Operations with Structs
# =============================================================================


class TestIterAdvancedOperations:
    """Test advanced Iter operations with Struct elements."""

    def test_iter_batched_processing(self) -> None:
        """Process structs in batches."""

        class Item(Struct, mut=True):
            id: int
            processed: bool = False

        items = [Item(id=i) for i in range(10)]

        # Process in batches of 3
        batches = Iter(items).batched(3).collect_list()

        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[3]) == 1

    def test_iter_window_pairs(self) -> None:
        """Use window to create pairs for comparison."""

        class Measurement(Struct):
            timestamp: int
            value: float

        measurements = [
            Measurement(timestamp=1, value=10.0),
            Measurement(timestamp=2, value=15.0),
            Measurement(timestamp=3, value=12.0),
            Measurement(timestamp=4, value=18.0),
        ]

        # Calculate deltas using windows
        windows = Iter(measurements).window(2).collect_list()
        deltas = [(w[1].value - w[0].value) for w in windows]

        assert deltas == [5.0, -3.0, 6.0]

    def test_iter_take_and_skip(self) -> None:
        """Use take and skip for pagination-like behavior."""

        class Item(Struct, mut=True):
            id: int
            name: str

        items = [Item(id=i, name=f"item_{i}") for i in range(10)]

        # Take first 3
        first_page = Iter(items).take(3).collect_list()
        assert len(first_page) == 3
        assert [i.id for i in first_page] == [0, 1, 2]

        # Skip first 3, take next 3
        second_page = Iter(items).skip(3).take(3).collect_list()
        assert len(second_page) == 3
        assert [i.id for i in second_page] == [3, 4, 5]

    def test_iter_zip_structs(self) -> None:
        """Zip two struct iterators together."""

        class Key(Struct):
            id: int

        class Value(Struct):
            data: str

        keys = [Key(id=1), Key(id=2), Key(id=3)]
        values = [Value(data="a"), Value(data="b"), Value(data="c")]

        pairs = Iter(keys).zip(values).collect_list()
        assert len(pairs) == 3
        assert pairs[0] == (Key(id=1), Value(data="a"))
        assert pairs[2] == (Key(id=3), Value(data="c"))

    def test_iter_inspect_side_effects(self) -> None:
        """Use inspect for logging/debugging in pipeline."""

        class Order(Struct):
            order_id: int
            total: float

        orders = [
            Order(order_id=1, total=100.0),
            Order(order_id=2, total=50.0),
            Order(order_id=3, total=200.0),
        ]

        inspected: list[int] = []

        result = (
            Iter(orders)
            .filter(lambda o: o.total > 75)
            .inspect(lambda o: inspected.append(o.order_id))
            .map(lambda o: o.total)
            .collect_list()
        )

        assert result == [100.0, 200.0]
        assert inspected == [1, 3]


# =============================================================================
# Struct.replace with Validation
# =============================================================================


class TestStructReplaceValidation:
    """Test Struct.replace with validation and Result."""

    def test_replace_creates_new_instance(self) -> None:
        """Replace returns a new instance, original unchanged."""

        class Config(Struct):
            host: str
            port: int
            debug: bool = False

        original = Config(host="localhost", port=8080)
        updated = original.replace(port=9090, debug=True)

        # Original unchanged
        assert original.port == 8080
        assert original.debug is False

        # Updated has new values
        assert updated.port == 9090
        assert updated.debug is True
        assert updated.host == "localhost"

    def test_replace_validates_new_values(self) -> None:
        """Replace validates the new field values."""

        class PositiveInt(Struct, mut=True):
            value: int

            def model_post_init(self, _context: object, /) -> None:
                if self.value < 0:
                    msg = "value must be positive"
                    raise ValueError(msg)

        original = PositiveInt(value=10)

        # Valid replacement
        updated = original.replace(value=20)
        assert updated.value == 20

        # Invalid replacement raises
        with pytest.raises(ValueError, match="value must be positive"):
            original.replace(value=-5)

    def test_try_from_with_replace_pattern(self) -> None:
        """Combine try_from and replace for update patterns."""

        class User(Struct, mut=True):
            id: int
            name: str
            email: str

        # Parse initial data
        result = User.try_from({"id": 1, "name": "Alice", "email": "alice@example.com"})
        assert isinstance(result, Ok)

        # Update email
        user = result.unwrap()
        updated = user.replace(email="alice.new@example.com")
        assert updated.email == "alice.new@example.com"


# =============================================================================
# OnceCell + Error Recovery
# =============================================================================


class TestOnceCellErrorRecovery:
    """Test OnceCell with error handling patterns."""

    def test_oncecell_take_and_reinit(self) -> None:
        """Take value from OnceCell and reinitialize."""
        cell: OnceCell[int] = OnceCell()

        # Initial set
        cell.set(42)
        assert cell.get() == Some(42)

        # Take the value
        taken = cell.take()
        assert taken == Some(42)
        assert cell.get() == Nothing()

        # Can set again after take
        cell.set(100)
        assert cell.get() == Some(100)

    def test_oncecell_get_or_init_with_fallible_init(self) -> None:
        """get_or_init with initialization that may fail."""
        cell: OnceCell[int] = OnceCell()
        call_count = 0

        def init() -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("transient error")
            return 42

        # First call fails - exception propagates
        with pytest.raises(ValueError, match="transient error"):
            cell.get_or_init(init)

        # Cell should still be uninitialized
        assert cell.get() == Nothing()

        # Second call succeeds
        value = cell.get_or_init(init)
        assert value == 42
        assert call_count == 2

    def test_oncecell_with_result_value(self) -> None:
        """OnceCell storing Result values."""
        cache: OnceCell[Ok[dict[str, int]] | Err[IOError]] = OnceCell()

        # Cache a successful result
        cache.set(Ok({"key": 42}))

        # Retrieve and use
        cached = cache.get()
        assert isinstance(cached, Some)

        match cached.unwrap():
            case Ok(data):
                assert data["key"] == 42
            case Err():
                pytest.fail("Expected Ok")


# =============================================================================
# Lazy + Iter Integration
# =============================================================================


class TestLazyIterIntegration:
    """Test Lazy with Iter operations."""

    def test_lazy_iter_deferred(self) -> None:
        """Lazy Iter is not evaluated until consumed."""
        evaluated = False

        def create_iter() -> Iter[int]:
            nonlocal evaluated
            evaluated = True
            return Iter([1, 2, 3])

        lazy_iter = Lazy(create_iter)

        assert not evaluated
        assert not lazy_iter.is_computed()

        # Accessing the iter triggers computation
        it = lazy_iter.get()
        assert evaluated
        assert it.collect_list() == [1, 2, 3]

    def test_lazy_expensive_computation_cached(self) -> None:
        """Expensive Iter computation is cached."""
        compute_count = 0

        def expensive_computation() -> list[int]:
            nonlocal compute_count
            compute_count += 1
            return Iter(range(1000)).map(lambda x: x * 2).filter(lambda x: x % 3 == 0).collect_list()

        lazy_result = Lazy(expensive_computation)

        # First access computes
        result1 = lazy_result.get()
        assert compute_count == 1

        # Second access returns cached
        result2 = lazy_result.get()
        assert compute_count == 1
        assert result1 == result2

    def test_lazy_struct_collection(self) -> None:
        """Lazy loading of a Struct collection."""

        class Product(Struct, mut=True):
            id: int
            name: str
            price: float

        def load_products() -> list[Product]:
            return [
                Product(id=1, name="Widget", price=9.99),
                Product(id=2, name="Gadget", price=19.99),
                Product(id=3, name="Gizmo", price=29.99),
            ]

        products = Lazy(load_products)

        # Filter and sort without recomputing
        result = Iter(products.get()).filter(lambda p: p.price > 10).sorted_by(lambda p: p.price, reverse=True)

        assert [p.name for p in result] == ["Gizmo", "Gadget"]


# =============================================================================
# Complex Pipeline Scenarios
# =============================================================================


class TestComplexPipelines:
    """Test complex real-world pipeline scenarios."""

    def test_etl_pipeline(self) -> None:
        """Extract-Transform-Load pipeline with full error handling."""

        class RawRecord(Struct, mut=True):
            id: str
            value: str
            timestamp: str

        class ProcessedRecord(Struct, mut=True):
            id: int
            value: float
            timestamp: int

        raw_data = [
            {"id": "1", "value": "10.5", "timestamp": "1000"},
            {"id": "bad", "value": "20.0", "timestamp": "1001"},  # Bad id
            {"id": "3", "value": "invalid", "timestamp": "1002"},  # Bad value
            {"id": "4", "value": "40.0", "timestamp": "1003"},
        ]

        def extract(data: dict[str, str]) -> Ok[RawRecord] | Err[Exception]:
            return RawRecord.try_from(data)

        def transform(raw: RawRecord) -> Ok[ProcessedRecord] | Err[Exception]:
            return try_except(
                lambda: ProcessedRecord(id=int(raw.id), value=float(raw.value), timestamp=int(raw.timestamp))
            )

        # Full pipeline
        results = [extract(d).and_then(transform) for d in raw_data]

        successes = [r.unwrap() for r in results if r.is_ok()]
        failures = [r for r in results if r.is_err()]

        assert len(successes) == 2
        assert len(failures) == 2
        assert successes[0].id == 1
        assert successes[1].id == 4

    def test_validation_pipeline_with_accumulation(self) -> None:
        """Validate multiple fields, accumulating all errors."""

        class ValidationError(Exception):
            """Exception that holds multiple field-level errors."""

            def __init__(self, errors: list[tuple[str, str]]) -> None:
                self.errors = errors
                super().__init__(f"Validation failed: {errors}")

        def validate_name(name: str) -> list[tuple[str, str]]:
            errors: list[tuple[str, str]] = []
            if len(name) < 2:
                errors.append(("name", "too short"))
            if not name.isalpha():
                errors.append(("name", "must be alphabetic"))
            return errors

        def validate_age(age: int) -> list[tuple[str, str]]:
            errors: list[tuple[str, str]] = []
            if age < 0:
                errors.append(("age", "cannot be negative"))
            if age > 150:
                errors.append(("age", "unrealistic age"))
            return errors

        def validate_all(name: str, age: int) -> Ok[tuple[str, int]] | Err[ValidationError]:
            errors = validate_name(name) + validate_age(age)
            if errors:
                return Err(ValidationError(errors))
            return Ok((name, age))

        # Valid input
        result = validate_all("Alice", 30)
        assert result == Ok(("Alice", 30))

        # Multiple errors: "A1" triggers "must be alphabetic", -5 triggers "cannot be negative"
        result = validate_all("A1", -5)
        assert isinstance(result, Err)
        assert len(result.error.errors) == 2

        # Even more errors with shorter invalid name
        result = validate_all("1", -5)
        assert isinstance(result, Err)
        assert len(result.error.errors) == 3  # too short, not alphabetic, negative

    def test_state_machine_with_result(self) -> None:
        """State machine transitions using Result."""

        class State(Struct):
            name: str
            data: dict[str, object]

        class InvalidTransitionError(Exception):
            pass

        def transition(state: State, action: str) -> Ok[State] | Err[InvalidTransitionError]:
            match (state.name, action):
                case ("idle", "start"):
                    return Ok(State(name="running", data={"started_at": 0}))
                case ("running", "pause"):
                    return Ok(State(name="paused", data=state.data))
                case ("paused", "resume"):
                    return Ok(State(name="running", data=state.data))
                case ("running", "stop") | ("paused", "stop"):
                    return Ok(State(name="stopped", data=state.data))
                case _:
                    return Err(InvalidTransitionError(f"Cannot {action} from {state.name}"))

        # Valid sequence
        state = State(name="idle", data={})
        state = transition(state, "start").unwrap()
        state = transition(state, "pause").unwrap()
        state = transition(state, "resume").unwrap()
        state = transition(state, "stop").unwrap()
        assert state.name == "stopped"

        # Invalid transition
        state = State(name="idle", data={})
        result = transition(state, "pause")
        assert isinstance(result, Err)
        assert "Cannot pause from idle" in str(result.error)
