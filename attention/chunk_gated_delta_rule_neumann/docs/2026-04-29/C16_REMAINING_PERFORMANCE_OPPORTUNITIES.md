# 2026-04-29 C16 当前方案剩余性能收益点

## 背景

当前默认实现为 `stage1_C16_FP32_CVInverse`：

- `chunkSize=16`
- AIV 生成 `L_powWsGm_ = L` 与 `INVWsGm_ = I-L`
- AIC 用 FP32 matmul 计算 `L²/L⁴/L⁸`
- AIC 用 matmul atomic add 完成 `INV += INV @ L_pow`

32-shape msprof 结果显示：

- `S=16` 已从旧方案 Neumann/normal 平均 `1.119x` 慢改善到 `0.994x`，基本持平/略优。
- `S=64` 仍为 `2.130x` 慢。
- `S>=128` 主要退化到 `2.4x~4.0x`。
- `S>=64` 平均指标：AIC scalar ratio `4.701x`、AIC MTE2 ratio `3.115x`、AIC FixPipe ratio `2.697x`、AIV vec ratio `0.654x`。

结论：AIV 累加已不是第一瓶颈；当前主要瓶颈是 C16 chunk 数放大后的 tiny matmul 调度、GM 读写和跨 stage 中间结果搬运。

## 当前数据流中可疑搬运点

### 1. `tmpWsGm_` 已无实际用途

方案A改为 `mmFp32.IterateAll(INVWsGm_, 1)` 后，Neumann 累加不再写 `tmpWsGm_`。

当前仍存在：

- `SetGlobalTensors()` 中给 `tmpWsGm_` 分配 `chunkSize × chunkSize`
- host tiling 中仍按 `L_pow + tmp + INV` 三块矩阵计 workspace
- `NeumannAddCompute()` 仍保留旧逻辑但主流程不再调用

收益判断：

- 主要减少 workspace 与地址切分复杂度。
- 对端到端性能收益较小，但属于低风险清理项。

建议：

- 删除 `tmpWsGm_` 成员和 workspace 分配。
- host tiling 将 Neumann workspace 从 `3 * C * C` 改为 `2 * C * C`。
- 删除未调用的 `NeumannAddCompute()`。

### 2. `q/k` BF16 → FP32 连续化写 GM 后再被 AIC 读

Stage1 当前流程：

1. AIV 从原始 `query/key` 读 BF16。
2. Cast 成 FP32。
3. 写 `queryContinousGm_` / `keyContinousGm_`。
4. AIC 再从 GM 读连续 FP32 做 `K @ K^T` 和 `Q @ K^T`。

这是当前最大显式重复搬运之一。以 `C=16,D=128` 粗略估算，每 chunk 仅连续化就有：

- BF16 读 `2*C*D*2 = 8KB`
- FP32 写 `2*C*D*4 = 16KB`
- AIC matmul 后续还会再读这两份 FP32 GM

收益判断：

- 如果能让 AIC 直接吃 BF16 原始输入并在 Cube 内完成计算，收益最大。
- 但当前 matmul 模板是 GM float 输入；改为 BF16/FP16 输入会牵涉 host tiling、MatmulType、精度和 layout。

建议：

- 中期做一个 `stage1_C16_BF16Input_FP32Acc_CVInverse` 实验分支：`key@key.T` 和 `query@key.T` 直接使用 BF16/half 输入、FP32 输出。
- 若精度通过，可删除 `queryContinousGm_`/`keyContinousGm_` 两块 stage workspace 和 AIV cast/write 路径。

### 3. `L_powWsGm_` 和 `INVWsGm_` 都由 AIV 写 GM，AIC 再读

当前 AIV 已经在 UB 里得到完整 `L` 半块，然后分别写：

- `L_powWsGm_ = L`
- `INVWsGm_ = I-L`

AIC 随后读 `L_powWsGm_` 做 6 次 Neumann matmul，并持续读写 `INVWsGm_`。

收益判断：

- `INVWsGm_` 必须作为后续 AIC matmul输入/输出存在。
- `L_powWsGm_` 也必须作为 `L²/L⁴/L⁸` ping-pong 输入存在。
- 当前无法完全省掉，但可以考虑压成 FP16/BF16 副本，减少 MTE2 读带宽。

建议：

- 如果精度允许，把 `L_powWsGm_` 改为 half/bf16，`INVWsGm_` 保持 FP32 或 C 输出 FP32。
- 这需要解决 `float -> half` cast 写 GM 或 AIC 侧随路 cast，目前没有确认可直接随路完成。

### 4. `gBKWsGm_` / `vBetaWsGm_` 是 AIV 生成、AIC 读取的中间矩阵

当前流程：

- AIV 写 `gBKWsGm_ = -beta * k * g_cum_exp`
- AIC 做 `INV @ gBKWsGm_ -> outKCumdecayGm_`
- AIV 写 `vBetaWsGm_ = value * beta`
- AIC 做 `INV @ vBetaWsGm_ -> outVInnerGm_`

收益判断：

- 这是 stage1 内部 AIV→AIC 的硬依赖。
- 直接减少搬运的难度较高，因为 `gBK/vBeta` 是 `C×D`，AIC matmul输入要求 GM。
- 但可减少同步等待：当前 AIC 必须分别等 `0x6` 和 `0x5`，AIV 也按固定顺序串行计算。

建议：

- 先不做大改。
- 若继续优化当前 C16，可尝试重排 AIV 顺序：优先 `GBKCompute()`，让 AIC 尽早启动 `INV @ gBK`，同时 AIV 再算 `VBetaCompute()`/`QPrimeCompute()`。
- 当前已接近这个顺序，收益有限。

### 5. `qK` 写 GM 后 Stage3 再读、mask、写 tmpGM、AIC 再读

跨 stage 当前流程：

1. Stage1 AIC 写 `outQkGm_ = Q @ K^T`。
2. Stage3 AIV 读 `outQkGm_`。
3. Stage3 AIV 乘 scale/gamma/stageThreeMask。
4. Stage3 AIV 写 `tmpGM_`。
5. Stage3 AIC 读 `tmpGM_` 做 `masked_qkt @ vInner`。

这是比 stage1 内部更大的重复搬运链路，尤其 C16 chunk 多时会重复发生。

收益判断：

- 性能潜力中高，尤其影响 Stage3 的 MTE2/MTE3 与 AIV scalar/vec。
- 但改动跨 stage，风险高于 stage1 内部清理。

建议：

- 若保持三 stage 架构，可把 `scale` 尽量前移到 Stage1 的 `outQkGm_`，Stage3 少一次 `Muls`。
- `gOptional_` 存在时 gamma 仍依赖 `gCumExp`，mask 仍需 Stage3 处理。
- 更大方案是 Stage1 直接输出 masked/scaled qkt，但需要确认 Stage3 mask 与 Stage1 mask 语义，且会改变 `outQkGm_` 复用含义。

### 6. `gOptional=false` 特化可跳过 Stage2 的 state 乘 1 写回

Stage2 当前即使 `gOptional=false`，也会：

- AIV `CopyIn(curState)`
- `CalGCumExp()` 中取 `last_g_cum_exp=1.0`
- `Muls(state_out, state_in, 1.0)`
- 写回 `curState`

当前性能脚本传入 `g`，所以这个点不影响本轮 profiling。

收益判断：

- 对无 g 场景是确定收益。
- 对当前带 g 性能批测无收益。

建议：

- 做无 g 专用 fast path：`gOptional=false` 时跳过 `CalGCumExp()` 的 state 复制写回，只保留 AIC 读 state 与 state update 的同步安全。

## 优先级

| 优先级 | 优化项 | 风险 | 预期收益 |
|---|---|---|---|
| P0 | 删除 `tmpWsGm_` / `NeumannAddCompute()` / workspace 第三块 | 低 | 小，主要清理 |
| P1 | `QK` 直接 BF16/half 输入 AIC，删除 `query/key` FP32 连续化 GM | 中高 | 中到高 |
| P1 | C64 分块/32×32 Neumann，减少 chunk 数和 tiny matmul 调度 | 中高 | 高 |
| P2 | Stage3 `qkt -> masked tmpGM` 链路融合或前移 scale | 中 | 中 |
| P2 | `L_pow` half/bf16 存储，`INV` FP32 保留 | 中 | 中 |
| P3 | 无 g 场景 Stage2 fast path | 低中 | 仅无 g 场景有效 |

## 推荐下一步

当前最务实的顺序：

1. 先做 P0 清理，确保代码和 workspace 与方案A一致。
2. 跑一次全量精度与 32-shape 性能，确认 P0 不引入退化。
3. 并行推进 P1 的 C64 分块求逆或 BF16/half 输入 AIC 实验，不建议继续在 C16 上只做零碎 AIV 搬运优化。

