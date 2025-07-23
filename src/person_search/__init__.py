"""
person_search package
---------------------

Importing this package will not auto-run the heavy pipeline; it only
exposes convenience symbols.

Example
-------
>>> from person_search import main_complete
>>> main_complete()
"""
from .system import main_complete, CompleteVideoProcessor

__all__ = ["main_complete", "CompleteVideoProcessor"]
