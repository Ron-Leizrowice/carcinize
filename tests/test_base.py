"""Tests for the RustType base mixin."""

from carcinize._base import RustType
from carcinize._iter import Iter


class TestRustTypeMixin:
    """Test the RustType mixin behavior."""

    def test_clone_creates_deep_copy(self) -> None:
        """clone() should create a deep copy via deepcopy."""

        class Container(RustType):
            def __init__(self, items: list[int]) -> None:
                self.items = items

        original = Container([1, 2, 3])
        cloned = original.clone()

        # Should be equal but not the same object
        assert cloned is not original
        assert cloned.items == original.items
        assert cloned.items is not original.items

        # Modifying clone should not affect original
        cloned.items.append(4)
        assert original.items == [1, 2, 3]
        assert cloned.items == [1, 2, 3, 4]

    def test_clone_with_nested_objects(self) -> None:
        """clone() should deep copy nested objects."""

        class Nested(RustType):
            def __init__(self, inner: dict[str, list[int]]) -> None:
                self.inner = inner

        original = Nested({"a": [1, 2], "b": [3, 4]})
        cloned = original.clone()

        # Nested structures should also be copied
        assert cloned.inner is not original.inner
        assert cloned.inner["a"] is not original.inner["a"]

        cloned.inner["a"].append(99)
        assert original.inner["a"] == [1, 2]

    def test_clone_preserves_type(self) -> None:
        """clone() should return the same type."""

        class MyType(RustType):
            def __init__(self, value: int) -> None:
                self.value = value

        original = MyType(42)
        cloned = original.clone()

        assert type(cloned) is MyType
        assert cloned.value == 42


class TestIterClone:
    """Test that Iter inherits clone() from RustType."""

    def test_iter_has_clone(self) -> None:
        """Iter should have clone() method from RustType."""
        it = Iter([1, 2, 3])
        assert hasattr(it, "clone")

    def test_iter_clone_creates_independent_iterator(self) -> None:
        """Cloning an Iter should create an independent copy.

        Due to deepcopy semantics, the clone gets its own copy of the
        underlying iterator state.
        """
        # Create and collect to list for comparison
        original = Iter([1, 2, 3])
        original_list = original.collect_list()

        # Clone a fresh iterator
        fresh = Iter([1, 2, 3])
        cloned = fresh.clone()

        # Clone should yield results
        cloned_list = cloned.collect_list()
        assert cloned_list == original_list

        # Fresh should also yield results (deepcopy creates independent copy)
        fresh_list = fresh.collect_list()
        assert fresh_list == original_list

        # After consumption, both should be empty
        assert fresh.collect_list() == []
        assert cloned.collect_list() == []
