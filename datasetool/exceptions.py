from __future__ import annotations


class DatasetoolError(Exception):
    """Base exception for datasetool."""


class MismatchError(DatasetoolError):
    """Raised when images and labels don't match in strict mode."""
