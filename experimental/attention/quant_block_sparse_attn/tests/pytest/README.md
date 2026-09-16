<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# QuantBlockSparseAttn pytest

本目录按量化模式组织 QBSA pytest 与 golden：

- `common_utils/`：FP8 与 MXFP8 共用的自定义算子加载实现。
- `qbsa_fp8/`：FP8 pytest、golden、执行脚本、CSV 和模式专用辅助实现。
- `qbsa_mxfp8/`：MXFP8 golden 主入口、Python testcase、CSV 和模式专用辅助实现。
- `conftest.py`、`pytest.ini`：两个模式共用的 pytest 配置。

FP8 从仓库任意目录执行：

```bash
bash experimental/attention/quant_block_sparse_attn/tests/pytest/qbsa_fp8/test_run.sh single
bash experimental/attention/quant_block_sparse_attn/tests/pytest/qbsa_fp8/test_run.sh batch
bash experimental/attention/quant_block_sparse_attn/tests/pytest/qbsa_fp8/test_run.sh graph
```

MXFP8 从仓库根目录执行：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/qbsa_mxfp8/qbsa_mxfp8_golden.py --list_cases
```

两个入口的详细参数分别参见各自目录内的 `README.md`。
