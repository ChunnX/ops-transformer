<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software: you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You should not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# QFA MXFP8 Softmax FP16 — S2 基本块 512 测试用例执行指南

## 1. 概述

本目录承载 `quant_flash_attn` 的 MxFP8 **Softmax FP16**（`quant_mode=3`）**S2 基本块 512
几何**的 pytest 测试框架，由 `qfa_mxfp8_softmax_fp16_test`（256 基本块几何）派生。

**与 256 几何的唯一差异**是 CPU golden 仿真的分块几何，对齐新 kernel 流水
**C1V1-C1V1-C2V2**（基本块 128×512）：

| 概念 | 256 几何（参考） | 512 几何（本目录） |
| --- | --- | --- |
| Q 块（S1） | 128 | 128 |
| 基本块（singleLoop S2） | 256 | **512**（C2V2 处理一整块） |
| C1V1 subloop | 块内 2×128 | **块内 2×256**（单轮一整段） |
| C1 mm1 粒度 | 128×256 | 128×256（每轮 subloop） |
| k 链 / exp / P 生成粒度 | 128 | **256** |
| C2 mmad 分段 | 无细分 | **4×128**（512/128=4 次） |
| rowsum 累积粒度 | 128 段 | 256 段 |
| 跨块 rescale | 256 块边界 | **512 基本块边界** |

数值影响点：k 链（accMax）更新粒度 128→256；P 量化 headroom（ps_u8）按 256 subloop
的 k_i 与 512 块尾 k_final 差值编码；C2 分子按 4×128 段 mmad 累积（fp32 累加顺序对齐
kernel Mmad）；rescale factor 在 512 边界。

NPU 调用、数据生成（BNSD fp8 + 5D e8m0 descale）、精度对比、缓存与性能解析等全部
沿用参考目录实现。

## 2. 文件结构

```
qfa_mxfp8_softmax_fp16_s2_512_test/
├── pytest.ini / conftest.py / .gitignore          # 与参考目录一致
├── quant_flash_attn_paramset_common.py            # 参数展开（quant_mode=3）
├── quant_flash_attn_paramset_debug.py             # 6 个 512 几何核心用例
├── quant_flash_attn_paramset_func_rdv.py          # 512 几何全维度矩阵（LSE 孪生）
├── quant_flash_attn_paramset_perf_rdv.py          # serving 大序列（同参考，与块几何无关）
├── test_quant_flash_attn_debug.py                 # debug 入口
├── test_quant_flash_attn_func_rdv.py              # 功能正确性入口
├── test_quant_flash_attn_perf_rdv.py              # 性能/压力入口
└── common/
    ├── quant_flash_attn_golden.py                 # 512 几何 CPU golden + NPU 调用
    ├── golden_cache.py / result_compare_method.py / perf_parser.py / test_runner.py
    └── __init__.py
```

## 3. 执行测试

```bash
pytest -v -m debug        # 6 个核心用例（单块/两块/513 尾块/257 边界/GQA+LSE/变长空批）
pytest -v -m func_rdv     # 全维度矩阵（S1/S2 三层边界 ±1、GQA、B、变长、交叉；LSE 孪生）
pytest -v -m perf_rdv     # serving 大序列（首次全流程，后续 --golden-mode=npu --msprof）
pytest -v -m ci
```

## 4. 环境准备

参考仓库根目录 README 安装 CANN + PyTorch + TorchNPU，每次运行前加载环境：

```bash
source <ascend_path>/set_env.sh
pip install pytest
cd experimental/attention/quant_flash_attn/tests/pytest/qfa_mxfp8_softmax_fp16_s2_512_test
```

> 注意：本目录 golden 仿真的是 512 基本块 kernel 行为，需配套实现了该几何的
> `cann_ops_transformer` 包；与 256 几何包混用时，差异会体现为 subloop 边界
> （256±1）与 mmad 分段相关的精度失配。
