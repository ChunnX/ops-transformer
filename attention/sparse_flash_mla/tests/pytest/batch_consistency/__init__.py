"""Reusable SMLA/MQSMLA batch-consistency test primitives.

Only the dependency-free registry is re-exported here so CLI validation does
not import PyTorch or initialize an NPU backend. Other APIs are imported from
their concrete submodules.
"""

from .registry import ConsistencyMode, ModeRegistry

__all__ = ["ConsistencyMode", "ModeRegistry"]
