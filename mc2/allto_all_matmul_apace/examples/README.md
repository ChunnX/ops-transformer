<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->
# AlltoAllMatmulApace aclnn 直调测试

## 概述

对 `AlltoAllMatmulApace` 算子的两段式 aclnn API 进行精度测试。

- **算子**: All-to-All + MX 量化 FP8 矩阵乘法融合算子（apace UDMA 路径）
- **输入**: x1(FP8_E4M3FN/E5M2), x2(FP8_E4M3FN/E5M2), x1_scale/x2_scale(FP8_E8M0)
- **输出**: y(BF16)
- **通信**: AIV 负责多卡 all-to-all，AIC 负责矩阵乘法

## 前置条件

1. CANN 开发环境已安装并 source：
   ```bash
   source /path/to/cann/set_env.sh
   ```

2. 算子已完成 opbuild 并安装到 CANN：
   ```bash
   cd ops-transformer
   bash build.sh --soc=ascend950 --module=mc2 --ops=allto_all_matmul_apace --pkg -j8
   ./build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=${ASCEND_HOME_PATH} --quiet --force
   ```

3. 自定义算子环境已 source：
   ```bash
   source ${ASCEND_HOME_PATH}/vendors/custom_transformer/bin/set_env.bash
   ```

4. NPU 卡数 ≥ `rank` 参数（默认 4 卡）

## 快速开始

```bash
cd mc2/allto_all_matmul_apace/examples
bash run.sh
```

默认参数：`M=2048, K_total=8192, N=3584, rank=4`

### 自定义参数

```bash
bash run.sh --m 1024 --k 3584 --n 4096 --rank 2
```

### 分步执行

```bash
bash run.sh --gen-only                          # 仅生成数据
bash run.sh --skip-build                        # 跳过编译（复用已有二进制）
```

## 参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--m`     | 单卡 M（总 M = m × rank） | 2048 |
| `--k`     | 总 K（单卡 Ka = k / rank） | 8192 |
| `--n`     | N           | 3584 |
| `--rank`  | 卡数        | 4    |

### 约束条件

- `k_total` 能被 `rank` 整除
- `CeilDiv(Ka, 64)` 必须为偶数
- `M_total` 能被 `512` 整除
- 仅支持 `FP8_E4M3FN` 和 `FP8_E5M2`

## 目录结构

```
examples/
├── run.sh                          # 一键测试脚本（数据→编译→qrun→验证）
├── CMakeLists.txt                  # cmake 编译配置
├── test_aclnn_allto_all_matmul_apace.cpp  # 测试主程序
└── scripts/
    ├── gen_data.py                 # MX FP8 数据生成 + CPU golden 计算
    └── verify_result.py            # BF16 精度比对
```

## 通信建立方式

采用与 Kernel 直调 ST 测试相同的 fork 模型：

1. **Host 进程 fork N 个子进程**，每个 rank 独立运行
2. 每个 rank 通过 TCP socket 交换 `HcclRootInfo` 建立 HCCL 通信域
3. `CommChannelBuilder` 创建 device 端 CommContext（含 UDMA channel handles）
4. 将 CommContext 写入 `context` tensor，通过 aclnn API 传入 kernel
5. Kernel 内部 AIV 负责 UDMA all-to-all，AIC 负责 matmul 计算

## 精度标准

- **输出数据类型**: BF16
- **精度容差**: `atol=1e-2, rtol=1e-2`
- 未通过 `torch.allclose` 即标记为 FAIL

## 测试结果示例

```
[1/4] 生成测试数据...
[2/4] 编译...
[3/4] NPU 运行 (qrun 4 ranks)...
[4/4] 精度比对...
[PASS] Rank 0
[PASS] Rank 1
[PASS] Rank 2
[PASS] Rank 3
[ALL PASS] All ranks passed verification!
```
