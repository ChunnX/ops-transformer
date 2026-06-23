---
name: ascendc-perf-optimize
description: Use when designing ops-transformer Ascend C operator performance strategies from tiling models, pipeline analysis, profiling data, bound diagnosis, or reusable AI Core optimization evidence.
---

# AscendC 性能优化策略设计

本 skill 用于在修改代码前制定优化策略。它输出假设、验证矩阵和风险边界，不替代精度或 profiling 证据。

## Repo Defaults

- 仓库：`ops-transformer`
- 目标 SoC：`ascend910b`
- 架构目录：`arch22`
- 主线目录：`attention/`、`ffn/`、`gmm/`、`mc2/`、`moe/`、`posembedding/`
- custom package vendor：`custom_transformer`

如果未来存在 `agent_ops/<op>/manifest.json`，先读取其状态、精度和性能结论。当前没有账本时，按 `agent/performance-optimization-method.md` 从 candidate gate 开始。

## 输入

至少收集：

- 算子名、domain、API。
- 目标 shape、dtype、format、attrs、layout。
- `op_host` tiling 逻辑和 `TilingData`。
- `op_kernel` 数据流和 buffer 规划。
- 已有精度入口和 profiling 数据；没有 profiling 时先输出 baseline 采集计划。
- package/load gate 风险。

## Strategy Flow

```text
Step 1: Tiling model
Step 2: Inter-device or MC2 communication strategy, if applicable
Step 3: Inter-core balance and synchronization strategy
Step 4: Single-core pipeline and memory strategy
Step 5: Verification matrix and rollback condition
```

### Step 1: Tiling Model

输出：

- 多核切分维度、blockDim 和每核任务量。
- 单核 UB/L1/L0 tile 预算。
- tail 处理和 dtype/shape/attrs 分支。
- tiling key 与 kernel suffix 的对应关系。
- repeat、mask、DataCopy 参数的上限约束。

相关 reference：`references/tiling/index.md`。只在需要具体模型时读取。

### Step 2: Communication Strategy

仅适用于 MC2 或通信/计算耦合算子。必须记录 rank、world size、HCCL、拓扑和通信 task；单卡 smoke 不能代表真实性能。

如果缺少多卡 profiling，输出“通信策略缺证据”，不要推断 overlap 收益。

### Step 3: Inter-Core Strategy

检查：

- 核间负载均衡。
- 尾块是否集中在少数核。
- 是否存在跨核同步或 atomic 热点。
- blockDim 是否匹配 AIV/AIC 或 MIX 物理核限制。

### Step 4: Single-Core Strategy

按 bound 类型输出候选：

- MTE2/MTE3 bound：提高单次搬运量、对齐、stride DataCopy、减少小 DMA 描述。
- Vector bound：减少重复 vector 指令、使用更合适的 API、避免不必要 cast。
- Scalar bound：减少循环控制、常量折叠阻塞、复杂分支和小 tile 次数。
- Memory bound：UB/L1/L0 驻留复用、bank conflict 规避、减少 GM 往返。
- Pipeline bound：CopyIn/Compute/CopyOut overlap、double buffer、同步粒度。

相关 reference：`references/single-core-pipeline/`。有 profiling 或仿真图时再读取具体文件。

## 输出格式

```text
operator:
target:
current evidence:
candidate gate:
package/load gate:
accuracy gate:
baseline matrix:
hypotheses:
  - id:
    bound:
    files:
    expected effect:
    risks:
    validation:
rollback conditions:
skill or lesson feedback:
```

## Decision Rules

- 策略只能指导实验，不能写成有效优化结论。
- 从 `ops-math` 或同族算子迁移来的经验必须重新验证。
- 目标大 shape 优化要保留小/边界/中等矩阵证据。
- 降低 GM 字节数、减少理论指令或清理 API 用法不等于更快，必须看 `Task Duration(us)` 和 total device time。
- 修改 `op_host`、tiling、opdef、op_api、metadata 或 workspace 后，默认重新 `--pkg` 并安装 `custom_transformer`。
