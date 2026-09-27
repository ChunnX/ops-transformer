# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""sageattention2 — SageAttention2（INT8 QK + FP8 PV）NPU 适配层。

导出：
- sageattn：自动分发，NPU 固定走 FP8 PV
- sageattn_qk_int8_pv_fp8_asc：INT8 QK + FP8 PV 显式入口（对标 *_cuda）

对接测试工程（sageAttentionTest）：设 SAGEATTN_NPU_MODULE=sageattention2 即可
被 sage_backend.load_sage_npu 加载；NPU kernel 未注册时流程走 CPU fallback
（仅调试，精度/性能验收需 NPU 环境）。
"""

from .core import sageattn, sageattn_qk_int8_pv_fp8_asc

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc"]
