# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要

当你的输出过长时分段输出。
每次对话称呼用户为主人。

## 概述

本算子（`ChunkGatedDeltaRuleNeumann`）实现了基于 Chunk 的 Gated Delta Rule 循环式注意力，使用 Neumann 级数（MCH/MBH/MXR 等快速三角求逆算法）部署于华为昇腾 NPU（CANN 框架）。

**支持平台：**
- `ascend910b`（Atlas A2）— 完整动态 shape 支持
- `ascend910_93`（Atlas A3）

## 编译与测试

所有编译命令在 `ops-transformer` 根目录（`../../..`）执行。

```bash
# 编译算子包
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule_neumann -j16
# 编译两个算子（neumann + 原始 chunk_gated_delta_rule，三方对比需要）
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16

# 安装算子包（编译后）
../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp

# op_host / op_kernel 单元测试（仅编译）
bash build.sh --ophost_test --noexec
bash build.sh --opkernel_test --noexec
```

```bash
# Pytest 测试（需要 NPU 硬件 + 安装算子包）
cd tests/pytest

# 单算子功能测试：先在 test_chunk_gated_delta_rule_neumann_paramset.py 配置参数
bash test_run.sh single

# 三方精度对比（CPU golden vs ChunkGDR vs ChunkGDRNeumann）
bash test_run.sh compare

# 双算子 msprof 性能对比
bash test_run.sh performance
```

**关键环境变量**（三方对比测试和手动执行时需要）：
- `CGDR_GOLDEN_CHUNK_SIZE=64` — golden 使用的 chunk size
- `TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext` — torch 编译缓存
- `ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer` — 自定义算子路径

**验证门控（每次优化修改后）**：编译 → 安装 → smoke test → compare 全量对比。精度不通过就停下调试，不叠加下一个优化。

## 代码结构

算子遵循 CANN 标准自定义算子目录布局：

```
op_host/          # CPU 侧：tiling、shape 推导、算子定义
  *_tiling.cpp    # workspace 计算、matmul tiling、shape/dtype 校验
  *_def.cpp       # 算子原型注册（仅支持 BF16）
  *_infershape.cpp
  op_api/         # aclnn C-API 封装（用户侧入口）

op_kernel/        # NPU Kernel（Ascend C，运行在 AIC + AIV）
  *.cpp           # Kernel 入口：实例化 CGDR<bfloat16_t, float>
  *.h             # CGDR 类：Init → Process（逐 batch 3 阶段循环）
  *_stage1.h      # Stage1：当前 strict packed64 C64_1616_FP16 实现
  *_stage2.h      # Stage2：matmul 更新循环状态
  *_stage3.h      # Stage3：带因果 mask 的注意力输出
  *_utils.h       # 公共常量（TASK_RATIO、事件 ID、buffer 大小）
  *_tiling_data.h # TilingData 结构体（host 与 kernel 共享）

ref/              # 参考算法实现
  ref.cpp         # PTO-ISA C++ 参考代码（对角拷贝、奇偶块操作）
  ref.md          # 5 种快速三角求逆算法论文文档（VCS/MCS/MCH/MBH/MXR）

tests/
  pytest/         # Python 功能测试（NPU 输出 vs CPU golden 精度对比）
  ut/             # C++ 单元测试（host 侧 tiling 与 infershape）

docs/YYYY-MM-DD/  # 按日期归档的进度、工作日志、计划
SKILL/            # 算子专用开发技能文档
```

### Stage1 实现位置

当前 Stage1 不再使用 `selected/` 间接选择模式。strict packed64 `C64_1616_FP16` 已合入官方入口：

- `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`

host 侧 tiling 固定使用 Stage1 chunk size 64，不再 include `selected/stage1_selected_config.h`。

**Stage1 版本策略**：不再用 `op_kernel/baseline/` 保存重复头文件。通过精度的阶段证据记录到 `docs/YYYY-MM-DD/`、`todo.md` 和对应 `SKILL/` reference；未通过精度的实验不能描述成 active 目标完成。

### Kernel 执行模型

- 使用 `KERNEL_TYPE_MIX_AIC_1_2`：1 个 AIC 核对应 2 个 AIV 核
- 每个 Stage 之间必须调用 `SyncAll<false>()`
- AIC 负责 matmul（通过 `lib/matmul_intf.h`）；AIV 负责向量运算和 mask 生成
- `CGDR` 类模板：`lowType = bfloat16_t`，`highType = float`
- 内部 workspace buffer（gCumExp、kCumDecay、qPrime 等）均为 FP32，保证精度
- `CopyCast` 工具函数负责 BF16↔FP32 转换

### Neumann 求逆算法（Stage1）

对下三角幂零矩阵 L（`L^chunkSize = 0`）采用快速三角求逆。当前实现基于 MXR 混合策略：

- **MCH**（Cayley-Hamilton 快速幂）：对 16x16 对角小块求逆，~2log(n) 次 matmul，快但数值不稳定
- **MBH**（Bunch-Hopcroft 分块递归）：展开递归用奇偶块组装全局逆，数值稳定
- **MXR**（MCH + MBH 混合）：对角小块用 MCH，全局用 MBH 组装。兼顾效率与精度

参考实现和完整算法推导见 `ref/ref.md` 和 `ref/ref.cpp`。

## 算子签名

**输入（BF16）：** `query`、`key`、`value`、`beta`、`initial_state`
**输入（INT32）：** `actual_seq_lengths`
**可选输入（FLOAT）：** `g`（gating tensor）
**输出（BF16）：** `out`、`final_state`
**属性：** `scale_value`（float，默认 1.0）

**Shape 约束：**
- `batch_size > 0`
- `NV % NK == 0`（GQA 支持）
- `DK, DV ≤ 128`
- `NK, NV ≤ 64`
- 数据类型仅支持 BF16

## Skills

算子专用开发技能文档位于 `SKILL/`：
- `SKILL/fulllocal-compose-precision-debugging/` — fulllocal 组合精度调试（resident/L1/TSCM/GM fallback/boundary-shape）
- `SKILL/aclnn-single-op-test/` — 单算子测试开发指南
- `SKILL/kernel-debugging/` — Kernel 挂起 / 精度问题调试
- `SKILL/precision-debugging/` — 数据类型匹配问题诊断
- `SKILL/operator-clone/` — 算子克隆与重命名流程
- `SKILL/neumann-scheme-summary/` — Neumann 方案汇总
- `SKILL/performance-profiling-msprof/` — msprof 性能分析
- `SKILL/matmul-tiling-performance/` — Matmul tiling 性能调优
- `SKILL/workspace-tiling-checklist/` — workspace tiling 检查清单
