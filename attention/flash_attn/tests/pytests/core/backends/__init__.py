# core/backends/__init__.py
from .base import Backend
from .cpu import CPUBackend
try:
    from .gpu import GPUBackend
except ImportError:
    GPUBackend = None
try:
    from .npu import NPUBackend
except ImportError:
    NPUBackend = None
