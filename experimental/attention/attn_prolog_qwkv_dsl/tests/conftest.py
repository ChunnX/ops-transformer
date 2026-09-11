# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Make the sibling torch_extension modules importable as top-level names.

setup.py only packages top-level ``.py`` files from ``torch_extension/``, so the
DSL host lives there as ``golden`` / ``op`` / ``kernel`` rather than a nested
package.  Tests import those modules directly and do not load ``__init__.py``,
which would pull in torch_npu and OpBuilder.
"""

import os
import sys

_TE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "torch_extension"))
if _TE not in sys.path:
    sys.path.insert(0, _TE)
