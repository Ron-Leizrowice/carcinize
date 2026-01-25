"""Tests for the String type (work in progress)."""

from carcinize._string import String


class TestString:
    """Test the String type that inherits from str and RustType."""

    def test_string_is_str(self) -> None:
        """String should be a subclass of str."""
        s = String("hello")
        assert isinstance(s, str)
        assert s == "hello"

    def test_string_has_clone(self) -> None:
        """String should inherit clone() from RustType."""
        s = String("hello")
        cloned = s.clone()

        assert cloned == s
        assert cloned == "hello"
        assert isinstance(cloned, String)

    def test_string_clone_is_independent(self) -> None:
        """Cloned String should be a separate object."""
        s = String("hello")
        cloned = s.clone()

        # Strings are immutable, but they should still be separate objects
        # (though Python may intern identical strings)
        assert cloned == s

    def test_string_str_methods_work(self) -> None:
        """String should support all str methods."""
        s = String("hello world")

        assert s.upper() == "HELLO WORLD"
        assert s.split() == ["hello", "world"]
        assert s.startswith("hello")
        assert len(s) == 11

    def test_string_concatenation(self) -> None:
        """String should support concatenation."""
        s1 = String("hello")
        s2 = String(" world")

        # Note: concatenation returns str, not String
        result = s1 + s2
        assert result == "hello world"

    def test_string_formatting(self) -> None:
        """String should support formatting."""
        s = String("Hello, {}!")
        result = s.format("world")
        assert result == "Hello, world!"

    def test_string_slicing(self) -> None:
        """String should support slicing."""
        s = String("hello")
        assert s[0] == "h"
        assert s[1:4] == "ell"
        assert s[::-1] == "olleh"

    def test_string_iteration(self) -> None:
        """String should be iterable."""
        s = String("abc")
        chars = list(s)
        assert chars == ["a", "b", "c"]

    def test_string_in_collection(self) -> None:
        """String should work in collections."""
        s = String("hello")

        # In set
        string_set = {s, String("world")}
        assert len(string_set) == 2
        assert s in string_set

        # In dict
        string_dict = {s: 1}
        assert string_dict[s] == 1

    def test_string_comparison(self) -> None:
        """String should support comparison operators."""
        s1 = String("apple")
        s2 = String("banana")
        s3 = String("apple")  # Equal to s1

        assert s1 < s2
        assert s2 > s1
        assert s1 <= s3
        assert s1 >= s3
        assert s1 != s2
        assert s1 == s3

    def test_empty_string(self) -> None:
        """Empty String should work correctly."""
        s = String("")

        assert s == ""
        assert len(s) == 0
        assert not s  # Empty string is falsy
        assert s.clone() == ""
