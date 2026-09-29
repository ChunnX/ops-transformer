# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SageAttention2 Ascend 950: INT8 QK, FP8 PV."""

from . import core
from ._ops import replace_sdpa, run, sageattn, sageattn_qk_int8_pv_fp8_asc

__all__ = ["core", "sageattn", "sageattn_qk_int8_pv_fp8_asc", "replace_sdpa", "run"]
