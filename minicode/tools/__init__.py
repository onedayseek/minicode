from pathlib import Path

from . import files, search, shell
from .base import Registry, Tool


def build_registry(root: Path, seen: set) -> Registry:
    return Registry(
        files.make_tools(root, seen) + search.make_tools(root) + shell.make_tools(root)
    )


__all__ = ["Registry", "Tool", "build_registry"]
