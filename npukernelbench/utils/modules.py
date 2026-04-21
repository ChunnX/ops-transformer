"""Dynamic Python module loading utilities."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


def load_module_from_path(module_name: str, module_path: str) -> ModuleType:
    """Load a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot load module '{module_name}' from '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
