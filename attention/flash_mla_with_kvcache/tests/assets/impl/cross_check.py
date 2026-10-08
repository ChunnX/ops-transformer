# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Select TTK's built-in three-way comparison without changing normal runs."""


def enabled():
    try:
        from ttk.utilities.container_utils import get_global_storage
    except ImportError:
        # The CUDA server can load this plugin without the TTK worker package.
        return False
    return getattr(get_global_storage(), "compare_method", None) == "cross_check"


class CompareForMode:
    """TTK custom compare cannot consume third_party outputs or return None.

    Expose no custom comparator in cross_check mode so TTK routes all three
    outputs through its built-in comparator. Resolve on access, not at import,
    because worker switches may be initialized after plugin discovery.
    """

    def __init__(self, compare):
        self.compare = compare

    def __get__(self, instance, owner):
        return None if enabled() else self.compare
