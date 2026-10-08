# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Import the standalone install's constituent source roots without an NPU."""

from pathlib import Path
import sys

MC2 = Path(__file__).resolve().parents[2]
for source in (
    "common/deepep/python",
    "low_latency_dispatch/python",
    "mega_moe/python",
):
    sys.path.insert(0, str(MC2 / source))

# When run together, Dispatch tests may already have imported the namespace.
if "ops_transformer_mc2" in sys.modules:
    from pkgutil import extend_path

    package = sys.modules["ops_transformer_mc2"]
    package.__path__ = extend_path(package.__path__, package.__name__)
