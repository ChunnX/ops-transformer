# Auto-generated script for creating __init__.py
file(WRITE "/home/developer/codes/ops-transformer/build/prepare_build/common/stub/op_graph/es_math_build/python_package/es_math/__init__.py" "# Auto-generated ES API for es_math
from .es_math_ops import *

__all__ = [name for name in dir() if not name.startswith('_')]

def get_module():
    import sys
    return sys.modules[__name__]
")
