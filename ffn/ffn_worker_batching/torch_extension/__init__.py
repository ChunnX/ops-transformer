# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FFN worker batching frontend and optional TorchAir GE registration."""

__all__ = ["ffn_worker_batching", "convert_ffn_worker_batching"]

# Register the dispatcher schema before the converter decorator looks it up.
# 导入顺序有依赖：前一模块的 builder 先建立 torch.ops 中的 schema，后一模块的
# 注册装饰器才有可引用的 .default overload。TorchAir 是可选依赖，由转换模块内部
# 捕获 ImportError；没有安装它也应能使用 eager/Meta，不能在包入口强制导入 TorchAir。
from .ffn_worker_batching import ffn_worker_batching
from .graph_convert_ffn_worker_batching import convert_ffn_worker_batching
