"""Tests for OnceCell and Lazy types."""

import threading
import time

import pytest

from carcinize._lazy import Lazy, OnceCell, OnceCellAlreadyInitializedError
from carcinize._option import Nothing, Some
from carcinize._result import Err, Ok

# =============================================================================
# OnceCell Tests
# =============================================================================


class TestOnceCellBasics:
    """Test OnceCell basic operations."""

    def test_new_cell_is_empty(self) -> None:
        """New OnceCell should be uninitialized."""
        cell: OnceCell[int] = OnceCell()
        assert cell.is_initialized() is False
        assert cell.get() == Nothing()

    def test_set_initializes_cell(self) -> None:
        """set() should initialize the cell."""
        cell: OnceCell[int] = OnceCell()
        result = cell.set(42)

        assert isinstance(result, Ok)
        assert cell.is_initialized() is True
        assert cell.get() == Some(42)

    def test_set_twice_returns_err(self) -> None:
        """set() on initialized cell should return Err."""
        cell: OnceCell[int] = OnceCell()
        cell.set(42)
        result = cell.set(100)

        assert isinstance(result, Err)
        assert isinstance(result.error, OnceCellAlreadyInitializedError)
        assert cell.get() == Some(42)  # Original value unchanged

    def test_get_or_init_initializes(self) -> None:
        """get_or_init() should initialize on first call."""
        cell: OnceCell[int] = OnceCell()
        value = cell.get_or_init(lambda: 42)

        assert value == 42
        assert cell.is_initialized() is True

    def test_get_or_init_returns_existing(self) -> None:
        """get_or_init() should return existing value without calling f."""
        cell: OnceCell[int] = OnceCell()
        cell.set(42)

        call_count = 0

        def expensive() -> int:
            nonlocal call_count
            call_count += 1
            return 100

        value = cell.get_or_init(expensive)

        assert value == 42
        assert call_count == 0

    def test_take_removes_value(self) -> None:
        """take() should remove and return the value."""
        cell: OnceCell[int] = OnceCell()
        cell.set(42)

        taken = cell.take()
        assert taken == Some(42)
        assert cell.is_initialized() is False
        assert cell.get() == Nothing()

    def test_take_empty_returns_nothing(self) -> None:
        """take() on empty cell should return Nothing."""
        cell: OnceCell[int] = OnceCell()
        assert cell.take() == Nothing()

    def test_repr(self) -> None:
        """repr() should show state."""
        cell: OnceCell[int] = OnceCell()
        assert "uninitialized" in repr(cell)

        cell.set(42)
        assert "42" in repr(cell)


class TestOnceCellThreadSafety:
    """Test OnceCell thread safety."""

    def test_concurrent_get_or_init_calls_once(self) -> None:
        """Multiple threads calling get_or_init should only initialize once."""
        cell: OnceCell[int] = OnceCell()
        call_count = 0
        call_count_lock = threading.Lock()

        def init() -> int:
            nonlocal call_count
            with call_count_lock:
                call_count += 1
            time.sleep(0.01)  # Simulate slow initialization
            return 42

        results: list[int] = []
        results_lock = threading.Lock()

        def worker() -> None:
            value = cell.get_or_init(init)
            with results_lock:
                results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count == 1
        assert all(r == 42 for r in results)

    def test_concurrent_set_only_one_succeeds(self) -> None:
        """Multiple threads calling set should have only one succeed."""
        cell: OnceCell[int] = OnceCell()
        success_count = 0
        success_lock = threading.Lock()

        def worker(value: int) -> None:
            nonlocal success_count
            result = cell.set(value)
            if isinstance(result, Ok):
                with success_lock:
                    success_count += 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count == 1
        assert cell.is_initialized() is True


# =============================================================================
# Lazy Tests
# =============================================================================


class TestLazyBasics:
    """Test Lazy basic operations."""

    def test_not_computed_initially(self) -> None:
        """Lazy should not compute until accessed."""
        computed = False

        def init() -> int:
            nonlocal computed
            computed = True
            return 42

        lazy = Lazy(init)
        assert computed is False
        assert lazy.is_computed() is False

    def test_get_computes_value(self) -> None:
        """get() should compute and return the value."""
        lazy = Lazy(lambda: 42)
        value = lazy.get()

        assert value == 42
        assert lazy.is_computed() is True

    def test_get_caches_value(self) -> None:
        """get() should cache the value after first call."""
        call_count = 0

        def init() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        lazy = Lazy(init)

        assert lazy.get() == 42
        assert lazy.get() == 42
        assert lazy.get() == 42

        assert call_count == 1

    def test_get_if_computed_before_access(self) -> None:
        """get_if_computed() should return Nothing before computation."""
        lazy = Lazy(lambda: 42)
        assert lazy.get_if_computed() == Nothing()

    def test_get_if_computed_after_access(self) -> None:
        """get_if_computed() should return Some after computation."""
        lazy = Lazy(lambda: 42)
        lazy.get()
        assert lazy.get_if_computed() == Some(42)

    def test_repr(self) -> None:
        """repr() should show state."""
        lazy = Lazy(lambda: 42)
        assert "not computed" in repr(lazy)

        lazy.get()
        assert "42" in repr(lazy)


class TestLazyThreadSafety:
    """Test Lazy thread safety."""

    def test_concurrent_get_computes_once(self) -> None:
        """Multiple threads calling get should only compute once."""
        call_count = 0
        call_count_lock = threading.Lock()

        def init() -> int:
            nonlocal call_count
            with call_count_lock:
                call_count += 1
            time.sleep(0.01)  # Simulate slow computation
            return 42

        lazy = Lazy(init)
        results: list[int] = []
        results_lock = threading.Lock()

        def worker() -> None:
            value = lazy.get()
            with results_lock:
                results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count == 1
        assert all(r == 42 for r in results)


# =============================================================================
# Exception Handling Tests
# =============================================================================


class TestLazyExceptions:
    """Test exception handling in Lazy."""

    def test_exception_propagates(self) -> None:
        """Exceptions during init should propagate."""

        def failing_init() -> int:
            raise ValueError("init failed")

        lazy = Lazy(failing_init)

        with pytest.raises(ValueError, match="init failed"):
            lazy.get()

    def test_exception_not_cached(self) -> None:
        """Failed init should not mark as computed, allowing retry."""
        call_count = 0

        def sometimes_fail() -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first call fails")
            return 42

        lazy = Lazy(sometimes_fail)

        # First call fails
        with pytest.raises(ValueError, match="first call fails"):
            lazy.get()
        assert call_count == 1
        assert not lazy.is_computed()

        # Second call retries and succeeds
        result = lazy.get()
        assert result == 42
        assert call_count == 2
        assert lazy.is_computed()
