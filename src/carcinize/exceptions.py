"""Shared exceptions for carcinize types."""


class UnwrapError(Exception):
    """Raised when unwrapping a Result or Option fails.

    This is analogous to a panic in Rust when calling `unwrap()` on the wrong variant.
    """
