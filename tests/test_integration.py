"""Integration tests for combinations of carcinize types.

Tests the intersection of multiple functionalities:
- Iter + Result/Option
- Struct + Result/Option
- Lazy/OnceCell + other types
- Full pipeline scenarios
"""

import threading

from carcinize import (
    Err,
    Iter,
    Lazy,
    MutStruct,
    Nothing,
    Ok,
    OnceCell,
    Some,
    Struct,
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

        class UserProfile(MutStruct):
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

        class Config(MutStruct):
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

        class Item(MutStruct):
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

        class ExpensiveConfig(MutStruct):
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

        class UserInput(MutStruct):
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

        class Sale(MutStruct):
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
