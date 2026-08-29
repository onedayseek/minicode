from pathlib import Path

from . import files, search, shell
from .base import Registry, Tool
from .shell import Shell, resolve_shell


def build_registry(root: Path, seen: set, sh: Shell | None = None) -> Registry:
    sh = sh or resolve_shell()
    return Registry(
        files.make_tools(root, seen) + search.make_tools(root) + shell.make_tools(root, sh)
    )


__all__ = ["Registry", "Shell", "Tool", "build_registry", "resolve_shell"]
