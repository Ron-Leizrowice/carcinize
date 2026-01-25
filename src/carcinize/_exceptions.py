"""Shared exceptions for carcinize types."""


class CarcinizeError(Exception):
    """Base class for all Carcinize exceptions."""


class UnwrapError(CarcinizeError):
    """Raised when unwrapping a Result or Option fails.

    This is analogous to a panic in Rust when calling `unwrap()` on the wrong variant.
    """
