"""Tests for the Iter fluent iterator."""

from carcinize._iter import Iter
from carcinize._option import Nothing, Some

# =============================================================================
# Transformation Tests
# =============================================================================


class TestIterTransformations:
    """Test Iter transformation methods."""

    def test_map_transforms_elements(self) -> None:
        """map() should transform each element."""
        result = Iter([1, 2, 3]).map(lambda x: x * 2).collect_list()
        assert result == [2, 4, 6]

    def test_filter_keeps_matching(self) -> None:
        """filter() should keep only matching elements."""
        result = Iter([1, 2, 3, 4, 5]).filter(lambda x: x > 2).collect_list()
        assert result == [3, 4, 5]

    def test_filter_map_applies_and_filters(self) -> None:
        """filter_map() should apply function and keep Some values."""

        def to_even(x: int) -> Some[int] | Nothing:
            if x % 2 == 0:
                return Some(x * 10)
            return Nothing()

        result = Iter([1, 2, 3, 4]).filter_map(to_even).collect_list()
        assert result == [20, 40]

    def test_flat_map_flattens(self) -> None:
        """flat_map() should apply and flatten."""
        result = Iter([1, 2, 3]).flat_map(lambda x: [x, x]).collect_list()
        assert result == [1, 1, 2, 2, 3, 3]

    def test_flatten_nested_iterables(self) -> None:
        """flatten() should flatten one level."""
        result = Iter([[1, 2], [3, 4], [5]]).flatten().collect_list()
        assert result == [1, 2, 3, 4, 5]

    def test_inspect_sees_elements(self) -> None:
        """inspect() should call function on each element."""
        seen: list[int] = []
        result = Iter([1, 2, 3]).inspect(lambda x: seen.append(x)).collect_list()
        assert result == [1, 2, 3]
        assert seen == [1, 2, 3]


# =============================================================================
# Slicing Tests
# =============================================================================


class TestIterSlicing:
    """Test Iter slicing methods."""

    def test_take_first_n(self) -> None:
        """take() should return first n elements."""
        result = Iter([1, 2, 3, 4, 5]).take(3).collect_list()
        assert result == [1, 2, 3]

    def test_take_more_than_available(self) -> None:
        """take() with n > length should return all elements."""
        result = Iter([1, 2]).take(10).collect_list()
        assert result == [1, 2]

    def test_skip_first_n(self) -> None:
        """skip() should skip first n elements."""
        result = Iter([1, 2, 3, 4, 5]).skip(2).collect_list()
        assert result == [3, 4, 5]

    def test_take_while_predicate(self) -> None:
        """take_while() should take while predicate is true."""
        result = Iter([1, 2, 3, 4, 1, 2]).take_while(lambda x: x < 4).collect_list()
        assert result == [1, 2, 3]

    def test_skip_while_predicate(self) -> None:
        """skip_while() should skip while predicate is true."""
        result = Iter([1, 2, 3, 4, 1, 2]).skip_while(lambda x: x < 3).collect_list()
        assert result == [3, 4, 1, 2]

    def test_step_by(self) -> None:
        """step_by() should yield every nth element."""
        result = Iter([1, 2, 3, 4, 5, 6]).step_by(2).collect_list()
        assert result == [1, 3, 5]

    def test_batched_exact_division(self) -> None:
        """batched() should group elements into equal-sized batches."""
        result = Iter([1, 2, 3, 4, 5, 6]).batched(2).collect_list()
        assert result == [(1, 2), (3, 4), (5, 6)]

    def test_batched_with_remainder(self) -> None:
        """batched() should handle incomplete final batch."""
        result = Iter([1, 2, 3, 4, 5]).batched(2).collect_list()
        assert result == [(1, 2), (3, 4), (5,)]

    def test_batched_single_element_batches(self) -> None:
        """batched(1) should create single-element tuples."""
        result = Iter([1, 2, 3]).batched(1).collect_list()
        assert result == [(1,), (2,), (3,)]

    def test_batched_larger_than_input(self) -> None:
        """batched() with n > len should return single batch."""
        result = Iter([1, 2, 3]).batched(10).collect_list()
        assert result == [(1, 2, 3)]

    def test_batched_empty_iterator(self) -> None:
        """batched() on empty iterator should yield nothing."""
        result = Iter([]).batched(3).collect_list()
        assert result == []

    def test_batched_invalid_size_raises(self) -> None:
        """batched() with n < 1 should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="batch size must be at least 1"):
            Iter([1, 2, 3]).batched(0).collect_list()

    def test_window_sliding(self) -> None:
        """window() should create sliding windows."""
        result = Iter([1, 2, 3, 4]).window(2).collect_list()
        assert result == [(1, 2), (2, 3), (3, 4)]

    def test_window_size_three(self) -> None:
        """window() with size 3 should create overlapping triplets."""
        result = Iter([1, 2, 3, 4, 5]).window(3).collect_list()
        assert result == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    def test_window_size_one(self) -> None:
        """window(1) should wrap each element in a tuple."""
        result = Iter([1, 2, 3]).window(1).collect_list()
        assert result == [(1,), (2,), (3,)]

    def test_window_equal_to_length(self) -> None:
        """window() with n == len should return single window."""
        result = Iter([1, 2, 3]).window(3).collect_list()
        assert result == [(1, 2, 3)]

    def test_window_larger_than_input(self) -> None:
        """window() with n > len should yield nothing."""
        result = Iter([1, 2]).window(5).collect_list()
        assert result == []

    def test_window_empty_iterator(self) -> None:
        """window() on empty iterator should yield nothing."""
        result = Iter([]).window(2).collect_list()
        assert result == []

    def test_window_invalid_size_raises(self) -> None:
        """window() with n < 1 should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="window size must be at least 1"):
            Iter([1, 2, 3]).window(0).collect_list()


# =============================================================================
# Combining Tests
# =============================================================================


class TestIterCombining:
    """Test Iter combining methods."""

    def test_chain_iterables(self) -> None:
        """chain() should concatenate iterables."""
        result = Iter([1, 2]).chain([3, 4]).collect_list()
        assert result == [1, 2, 3, 4]

    def test_zip_pairs(self) -> None:
        """zip() should pair elements."""
        result = Iter([1, 2, 3]).zip(["a", "b", "c"]).collect_list()
        assert result == [(1, "a"), (2, "b"), (3, "c")]

    def test_zip_different_lengths(self) -> None:
        """zip() should stop at shorter iterable."""
        result = Iter([1, 2, 3]).zip(["a", "b"]).collect_list()
        assert result == [(1, "a"), (2, "b")]

    def test_enumerate_with_index(self) -> None:
        """enumerate() should pair with indices."""
        result = Iter(["a", "b", "c"]).enumerate().collect_list()
        assert result == [(0, "a"), (1, "b"), (2, "c")]

    def test_enumerate_custom_start(self) -> None:
        """enumerate() should support custom start index."""
        result = Iter(["a", "b"]).enumerate(start=10).collect_list()
        assert result == [(10, "a"), (11, "b")]

    def test_interleave(self) -> None:
        """interleave() should alternate elements."""
        result = Iter([1, 2, 3]).interleave([10, 20, 30]).collect_list()
        assert result == [1, 10, 2, 20, 3, 30]


# =============================================================================
# Folding Tests
# =============================================================================


class TestIterFolding:
    """Test Iter folding methods."""

    def test_fold_accumulates(self) -> None:
        """fold() should accumulate with initial value."""
        result = Iter([1, 2, 3]).fold(10, lambda acc, x: acc + x)
        assert result == 16

    def test_reduce_no_initial(self) -> None:
        """reduce() should fold without initial value."""
        result = Iter([1, 2, 3]).reduce(lambda a, b: a + b)
        assert result == Some(6)

    def test_reduce_empty_returns_nothing(self) -> None:
        """reduce() on empty iterator should return Nothing."""
        result = Iter([]).reduce(lambda a, b: a + b)
        assert isinstance(result, Nothing)

    def test_sum(self) -> None:
        """sum() should add all elements."""
        result = Iter([1, 2, 3, 4]).sum()
        assert result == 10

    def test_product(self) -> None:
        """product() should multiply all elements."""
        result = Iter([1, 2, 3, 4]).product()
        assert result == 24


# =============================================================================
# Searching Tests
# =============================================================================


class TestIterSearching:
    """Test Iter searching methods."""

    def test_find_returns_first_match(self) -> None:
        """find() should return first matching element."""
        result = Iter([1, 2, 3, 4]).find(lambda x: x > 2)
        assert result == Some(3)

    def test_find_no_match_returns_nothing(self) -> None:
        """find() with no match should return Nothing."""
        result = Iter([1, 2, 3]).find(lambda x: x > 10)
        assert isinstance(result, Nothing)

    def test_find_map(self) -> None:
        """find_map() should return first Some value."""

        def parse_even(x: int) -> Some[str] | Nothing:
            if x % 2 == 0:
                return Some(f"even:{x}")
            return Nothing()

        result = Iter([1, 3, 4, 6]).find_map(parse_even)
        assert result == Some("even:4")

    def test_position_finds_index(self) -> None:
        """position() should return index of first match."""
        result = Iter(["a", "b", "c"]).position(lambda x: x == "b")
        assert result == Some(1)

    def test_any_true(self) -> None:
        """any() should return True if any match."""
        result = Iter([1, 2, 3]).any(lambda x: x > 2)
        assert result is True

    def test_any_false(self) -> None:
        """any() should return False if none match."""
        result = Iter([1, 2, 3]).any(lambda x: x > 10)
        assert result is False

    def test_all_true(self) -> None:
        """all() should return True if all match."""
        result = Iter([2, 4, 6]).all(lambda x: x % 2 == 0)
        assert result is True

    def test_all_false(self) -> None:
        """all() should return False if any don't match."""
        result = Iter([2, 3, 4]).all(lambda x: x % 2 == 0)
        assert result is False

    def test_count(self) -> None:
        """count() should return number of elements."""
        result = Iter([1, 2, 3, 4, 5]).count()
        assert result == 5

    def test_min_max(self) -> None:
        """min() and max() should return extremes."""
        items = [3, 1, 4, 1, 5]
        assert Iter(items).min() == Some(1)
        assert Iter(items.copy()).max() == Some(5)

    def test_min_max_empty(self) -> None:
        """min() and max() on empty should return Nothing."""
        assert Iter([]).min() == Nothing()
        assert Iter([]).max() == Nothing()


# =============================================================================
# Collecting Tests
# =============================================================================


class TestIterCollecting:
    """Test Iter collecting methods."""

    def test_collect_list(self) -> None:
        """collect_list() should return a list."""
        result = Iter([1, 2, 3]).collect_list()
        assert result == [1, 2, 3]

    def test_collect_set(self) -> None:
        """collect_set() should return a set."""
        result = Iter([1, 2, 2, 3, 3, 3]).collect_set()
        assert result == {1, 2, 3}

    def test_collect_dict(self) -> None:
        """collect_dict() should return a dict from pairs."""
        result = Iter([("a", 1), ("b", 2)]).collect_dict()
        assert result == {"a": 1, "b": 2}

    def test_collect_string(self) -> None:
        """collect_string() should join strings."""
        result = Iter(["a", "b", "c"]).collect_string(sep="-")
        assert result == "a-b-c"


# =============================================================================
# Accessing Tests
# =============================================================================


class TestIterAccessing:
    """Test Iter accessing methods."""

    def test_first_returns_first(self) -> None:
        """first() should return first element."""
        result = Iter([1, 2, 3]).first()
        assert result == Some(1)

    def test_first_empty_returns_nothing(self) -> None:
        """first() on empty should return Nothing."""
        result = Iter([]).first()
        assert isinstance(result, Nothing)

    def test_last_returns_last(self) -> None:
        """last() should return last element."""
        result = Iter([1, 2, 3]).last()
        assert result == Some(3)

    def test_last_empty_returns_nothing(self) -> None:
        """last() on empty should return Nothing."""
        result = Iter([]).last()
        assert isinstance(result, Nothing)

    def test_nth_returns_element(self) -> None:
        """nth() should return nth element (0-indexed)."""
        result = Iter([10, 20, 30, 40]).nth(2)
        assert result == Some(30)

    def test_nth_out_of_bounds_returns_nothing(self) -> None:
        """nth() out of bounds should return Nothing."""
        result = Iter([1, 2]).nth(10)
        assert isinstance(result, Nothing)


# =============================================================================
# Partitioning Tests
# =============================================================================


class TestIterPartitioning:
    """Test Iter partitioning methods."""

    def test_partition_splits(self) -> None:
        """partition() should split by predicate."""
        evens, odds = Iter([1, 2, 3, 4, 5]).partition(lambda x: x % 2 == 0)
        assert evens == [2, 4]
        assert odds == [1, 3, 5]

    def test_group_by(self) -> None:
        """group_by() should group by key."""
        result = Iter([1, 2, 3, 4, 5, 6]).group_by(lambda x: x % 3)
        assert result == {1: [1, 4], 2: [2, 5], 0: [3, 6]}


# =============================================================================
# Sorting Tests
# =============================================================================


class TestIterSorting:
    """Test Iter sorting methods."""

    def test_sorted_returns_sorted_list(self) -> None:
        """sorted() should return a sorted list."""
        result = Iter([3, 1, 4, 1, 5, 9, 2, 6]).sorted()
        assert result == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_sorted_reverse(self) -> None:
        """sorted() with reverse=True should return descending order."""
        result = Iter([3, 1, 4, 1, 5]).sorted(reverse=True)
        assert result == [5, 4, 3, 1, 1]

    def test_sorted_empty(self) -> None:
        """sorted() on empty iterator should return empty list."""
        result = Iter([]).sorted()
        assert result == []

    def test_sorted_strings(self) -> None:
        """sorted() should work with strings (lexicographic order)."""
        result = Iter(["banana", "apple", "cherry"]).sorted()
        assert result == ["apple", "banana", "cherry"]

    def test_sorted_by_key(self) -> None:
        """sorted_by() should sort by key function."""
        result = Iter(["bb", "a", "ccc"]).sorted_by(len)
        assert result == ["a", "bb", "ccc"]

    def test_sorted_by_reverse(self) -> None:
        """sorted_by() with reverse=True should sort descending by key."""
        result = Iter(["bb", "a", "ccc"]).sorted_by(len, reverse=True)
        assert result == ["ccc", "bb", "a"]

    def test_sorted_by_with_complex_key(self) -> None:
        """sorted_by() should work with complex key functions."""
        items = [{"name": "bob", "age": 30}, {"name": "alice", "age": 25}, {"name": "charlie", "age": 35}]
        result = Iter(items).sorted_by(lambda x: x["age"])
        assert result[0]["name"] == "alice"
        assert result[1]["name"] == "bob"
        assert result[2]["name"] == "charlie"

    def test_sorted_preserves_equal_elements_stability(self) -> None:
        """sorted() should be stable (preserve relative order of equal elements)."""
        # Python's sort is stable, so equal elements keep their relative order
        items = [(1, "a"), (2, "b"), (1, "c"), (2, "d")]
        result = Iter(items).sorted_by(lambda x: x[0])
        # Elements with key 1 should appear in original order: (1, "a"), (1, "c")
        # Elements with key 2 should appear in original order: (2, "b"), (2, "d")
        assert result == [(1, "a"), (1, "c"), (2, "b"), (2, "d")]


# =============================================================================
# Deduplication Tests
# =============================================================================


class TestIterDeduplication:
    """Test Iter deduplication methods."""

    def test_unique_removes_duplicates(self) -> None:
        """unique() should remove duplicates preserving order."""
        result = Iter([1, 2, 2, 3, 1, 4, 3]).unique().collect_list()
        assert result == [1, 2, 3, 4]

    def test_unique_by_key(self) -> None:
        """unique_by() should deduplicate by key function."""
        result = Iter(["a", "bb", "c", "dd", "e"]).unique_by(len).collect_list()
        assert result == ["a", "bb"]


# =============================================================================
# Chaining Tests
# =============================================================================


class TestIterChaining:
    """Test method chaining."""

    def test_complex_chain(self) -> None:
        """Multiple operations should chain correctly."""
        result = Iter(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * 3).skip(1).take(3).collect_list()
        assert result == [6, 12, 18]

    def test_chain_with_option_integration(self) -> None:
        """Iter should integrate with Option types."""

        def safe_div(x: int) -> Some[int] | Nothing:
            if x == 0:
                return Nothing()
            return Some(100 // x)

        result = Iter([5, 0, 2, 0, 4]).filter_map(safe_div).collect_list()
        assert result == [20, 50, 25]


# =============================================================================
# Iterator Protocol Tests
# =============================================================================


class TestIterProtocol:
    """Test iterator protocol compliance."""

    def test_iter_returns_iterator(self) -> None:
        """__iter__ should return an iterator."""
        it = Iter([1, 2, 3])
        assert iter(it) is it._iter

    def test_next_yields_elements(self) -> None:
        """__next__ should yield elements."""
        it = Iter([1, 2])
        assert next(it) == 1
        assert next(it) == 2

    def test_works_in_for_loop(self) -> None:
        """Iter should work in for loops."""
        result = list(Iter([1, 2, 3]).map(lambda x: x * 2))
        assert result == [2, 4, 6]
