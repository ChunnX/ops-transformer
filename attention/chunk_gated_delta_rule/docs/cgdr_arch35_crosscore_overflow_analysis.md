# arch35 CrossCore 同步修复分析

## 1. 问题概述

### 1.1 原始问题：flag ID 溢出

在 arch35 (Ascend950/DAV_3510, `KERNEL_TYPE_MIX_AIC_1_2`) ESL 仿真平台上运行 CGDR 算子时报错：

```
set_cross_core flag ID overflows!
```

### 1.2 修复过程中引入的次生问题

修复溢出后，又依次遇到：
1. **死锁**：1:2 模式下 intra-block 同步 Set/Wait 数量不匹配
2. **精度偏差**：WaitFlag 使用的 pipe 参数导致部分管线未阻塞，数据竞争产生浮点累积差异

## 2. 溢出根因

### 2.1 CrossCoreWaitFlag 默认模板参数

`CrossCoreWaitFlag` 声明（`kernel_operator_block_sync_intf.h:85-86`）：

```cpp
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

CGDR 代码中所有 `CrossCoreWaitFlag(flagId)` 调用**未指定模板参数**，默认为 `<0, PIPE_S>`，走 `wait_flag_dev` 路径。

对应的 `CrossCoreSetFlag` 使用 `<0x2, PIPE_MTE3>` (AIV) 和 `<0x2, PIPE_FIX>` (AIC)，走 `ffts_cross_core_sync` 路径。

### 2.2 arch35 的 IsSplitCubePipe 差异

`kernel_event.h:262-272`：

```cpp
// arch35 (3510)
IsSplitVectorPipe = { PIPE_S, PIPE_V, PIPE_MTE2, PIPE_MTE3 }
IsSplitCubePipe   = { PIPE_S, PIPE_MTE1, PIPE_MTE2, PIPE_FIX, PIPE_M }

// arch22 (2002) — 对比
IsSplitCubePipe   = { PIPE_S, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_M }
```

arch35 的 `IsSplitCubePipe` **不包含 `PIPE_MTE3`**，arch22 包含。

### 2.3 三重不匹配导致 flag 累积

以 stage1.h 同步0 为例：

| 操作 | 模板参数 | 实际执行 | modeId | pipe |
|------|---------|---------|--------|------|
| AIV Set | `<0x2, PIPE_MTE3>(0x9)` | `ffts_cross_core_sync(PIPE_MTE3, msg)` | 0x2 | PIPE_MTE3 |
| AIC Wait | `(0x9)` → 默认 `<0, PIPE_S>` | `wait_flag_dev(PIPE_S, 0x9)` | 0x0 | PIPE_S |

1. **modeId 不匹配**：Set=0x2, Wait=0x0。`GetffstMsg` 编码不同 mode 到不同 flag slot。
2. **pipe 不匹配**：Set=PIPE_MTE3, Wait=PIPE_S。仿真器 `HSetWaitEventTable` 按 `(pipe, flagId)` 索引。
3. **同步路径不匹配**：1:2 模式应使用 intra-block（modeId=4），但代码使用了 cross-core（modeId=0x2）。

AIV 不断 Set，flag 被推入等待表但**永远找不到匹配的 Wait**，累积超过 `SYNC_FLAG_ID_MAX=16` 后溢出。

### 2.4 仿真器 flag 生命周期

从 `libpem_davinci.so` 符号表推断的 flag 管理：

```
SET: decode → dispatch_Hset_flag → push_set_flag_to_tab → release_set_cross_core [溢出检查] → recv_set_cross_core
WAIT: dispatch_wait_flag → exec_wait_flag_dev → wait_flag_success → retire_set_flag
```

SET 无法被 WAIT 匹配时，`HSetWaitEventTable` 条目持续累积直至溢出。

## 3. 死锁根因

### 3.1 modeId 0x2 → 0x4 的选择

SDK 内部 `SyncAllImpl`（`kernel_operator_sync_impl.h:183-212`）根据 `__MIX_CORE_AIC_RATION__` 选择同步策略：

- **1:1 模式**：AIC↔AIV 用 cross-core（`ffts_cross_core_sync`/`wait_flag_dev`, modeId=0x2）
- **1:2 模式**：AIC↔AIV 用 intra-block（`set_intra_block`/`wait_intra_block`, modeId=4）

CGDR 在 1:2 模式下应使用 modeId=4。

### 3.2 intra-block 的 Set/Wait 严格配对

`modeId=4` 的 `NotifyEventImpl`/`WaitEventImpl` 调用 `set_intra_block`/`wait_intra_block`，要求 Set 和 Wait **严格一对一**。

1:2 模式下 (1 AIC + 2 AIV sub-blocks)：

| 方向 | Set 次数 | Wait 次数 | 缺口 |
|------|---------|----------|------|
| AIV→AIC | 2 (两个 sub-block 各 Set) | 1 (AIC 只 Wait) | 缺 1 次 Wait |
| AIC→AIV | 1 (AIC 只 Set) | 2 (两个 sub-block 各 Wait) | 缺 1 次 Set |

未消耗的 flag 留到下一轮循环，Wait 匹配到上一轮遗留 Set，同步错乱死锁。

### 3.3 FLAG_ID_MAX 补齐方案

参照 GMM `weight_quant_basic_block.h:426-453` 的 1:2 处理方式，AIC 侧用 `flagId` 和 `flagId + FLAG_ID_MAX`(16) 分别对应两个 AIV sub-block：

- **AIV→AIC**：AIC 增加 `Wait(flagId + FLAG_ID_MAX)`
- **AIC→AIV**：AIC 增加 `Set(flagId + FLAG_ID_MAX)`

## 4. 精度偏差根因

### 4.1 现象

修复溢出和死锁后，RDV 测试发现 output 和 final_state 与 baseline 存在差异，且**随 chunk 数增长而放大**：

| 配置 | output max_diff | state max_diff |
|------|----------------|----------------|
| 单 chunk (S=64) | 0.00019 | 0.0028 |
| 4 chunks (S=256) | 0.00793 | 0.09721 |

### 4.2 wait_intra_block 的 per-pipe 阻塞语义

`WaitEventImpl`（modeId=4）调用 `wait_intra_block(pipe, flagId)`，该指令**只阻塞指定的 pipe**，其他 pipe 的指令可继续派发执行。

```cpp
// kernel_operator_sync_impl.h:829-842
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId) {
    if constexpr (modeId == 4) {
        if constexpr (IsSplitVectorPipe<pipe>()) {
            if ASCEND_IS_AIV { wait_intra_block(pipe, flagId); }  // 只阻塞 pipe
        }
        if constexpr (IsSplitCubePipe<pipe>()) {
            if ASCEND_IS_AIC { wait_intra_block(pipe, flagId); }  // 只阻塞 pipe
        }
    }
}
```

### 4.3 数据竞争机制

以 stage2.h `ProcessAivFp32` 为例，分析 AIC→AIV 同步（flagId 0x2）：

```
AIC 侧:  CalVPrime(matmul) → CalAttnInter(matmul) → Set<0x4, PIPE_FIX>(0x2)
                                                      ↑ FIX 管线完成后才 Set

AIV 侧:  Wait<0x4, PIPE_MTE3>(0x2) → CastGmToGm(vInner, MTE2 读 GM)
           ↑ 只阻塞 MTE3 管线      ↑ MTE2 管线未被阻塞，可提前执行
```

当 AIV 使用 `PIPE_MTE3` Wait 时：
- `wait_intra_block(PIPE_MTE3, 0x2)` 只阻塞 MTE3 (UB→GM 写) 管线
- **MTE2 (GM→UB 读) 管线未被阻塞**，可以在 AIC 的 FIX 管线写出 matmul 结果之前就从 GM 读取数据
- AIV 读到的是上一轮或未完整的 matmul 结果

同理，AIC 侧使用 `PIPE_MTE1` Wait 时：
- `wait_intra_block(PIPE_MTE1, flagId)` 只阻塞 MTE1 (GM→L1 拷贝)
- **M/FIX 管线未被阻塞**，可以继续处理上一轮的 matmul，与新一轮数据交叉

### 4.4 误差累积放大

单 chunk 内的数据竞争产生微小浮点差异。由于 CGDR 的 stage2 在 chunk 循环中**前后依赖**（`stateOut[nvId * Dv * Dk]` 在每个 chunk 被读写），第 N 个 chunk 的输入依赖第 N-1 个 chunk 的输出。每个 chunk 的微小误差被传递并累积：

```
chunk 0: 误差 ε₀ ≈ 0.003 (单次数据竞争)
chunk 1: 误差 ε₁ ≈ ε₀ + f(ε₀) ≈ 0.006
chunk 2: 误差 ε₂ ≈ ε₁ + f(ε₁) ≈ 0.012
chunk 3: 误差 ε₃ ≈ ε₂ + f(ε₂) ≈ 0.097 (指数增长)
```

### 4.5 为什么 baseline 没有此问题

Baseline 的 `CrossCoreWaitFlag(flagId)` 默认 `<0, PIPE_S>`，走 `wait_flag_dev(PIPE_S, flagId)` 路径。`PIPE_S` 是 scalar 管线，scalar 指令是所有管线派发的前置条件——阻塞 `PIPE_S` 等效于**阻塞全部管线**，实现完全串行：

```
baseline:  wait_flag_dev(PIPE_S, flagId)
           → scalar 管线停摆 → 所有管线停止派发 → 完全串行等待
```

### 4.6 PIPE_S 修复方案

将所有 `CrossCoreWaitFlag` 的 pipe 统一改为 `PIPE_S`：

```cpp
// 修改前 (精度有差异)
CrossCoreWaitFlag<0x4, PIPE_MTE1>(flagId);  // AIC: 只阻塞 MTE1
CrossCoreWaitFlag<0x4, PIPE_MTE3>(flagId);  // AIV: 只阻塞 MTE3

// 修改后 (与 baseline 一致)
CrossCoreWaitFlag<0x4, PIPE_S>(flagId);     // AIC/AIV: 阻塞 scalar = 阻塞全部
```

`PIPE_S` 同时在 `IsSplitVectorPipe` 和 `IsSplitCubePipe` 中，因此 AIC 和 AIV 都会执行 `wait_intra_block(PIPE_S, flagId)`，阻塞 scalar 管线即阻塞全部指令派发，与 baseline 的 `wait_flag_dev(PIPE_S,...)` 行为一致。

### 4.7 为什么 GMM 参考可以用 PIPE_MTE1/PIPE_MTE3

GMM `weight_quant_basic_block.h` 使用 `PIPE_MTE1`(AIC Wait) / `PIPE_MTE3`(AIV Wait)，但 GMM 的 basic block 循环**无跨迭代数据依赖**：每个 basic block 的 matmul 输入和输出是独立的 GM 区域，不存在上一轮输出作为下一轮输入的链路。即使部分管线提前执行，也不会读到错误数据。

CGDR 的 stage2 则不同：`stateOut` 在每个 chunk 被读写，形成严格的串行依赖链，任何管线提前执行都会导致数据竞争。

### 4.8 SetFlag pipe 选择

SetFlag 的 pipe 决定 flag 在哪个管线完成后被设置：

| Set 位置 | pipe | 语义 | 选择理由 |
|---------|------|------|---------|
| AIV Set | `PIPE_MTE3` | AIV 的 UB→GM 写完成后 Set | AIV 通过 MTE3 将结果写出 GM，MTE3 完成后数据可见 |
| AIC Set | `PIPE_FIX` | AIC 的 FIX 管线完成后 Set | AIC 通过 FIX 将 matmul 结果写回 GM，FIX 完成后数据可见 |

SetFlag 的 pipe 只影响 "何时设置 flag"，不影响阻塞行为，因此只需匹配数据写出的管线即可。

## 5. 完整修改清单

### 5.1 CrossCore 同步修改

| 修改项 | 修改前 (baseline) | 修改后 | 原因 |
|--------|------------------|--------|------|
| modeId | 0x2 (cross-core) | 0x4 (intra-block) | 1:2 模式应用 intra-block 同步 |
| AIC Set pipe | PIPE_FIX | PIPE_FIX (不变) | matmul 结果通过 FIX 写回 GM |
| AIV Set pipe | PIPE_MTE3 | PIPE_MTE3 (不变) | 数据通过 MTE3 写出 GM |
| AIC Wait pipe | 默认 PIPE_S | PIPE_S (显式) | 阻塞 scalar = 阻塞全部管线 |
| AIV Wait pipe | 默认 PIPE_S | PIPE_S (显式) | 阻塞 scalar = 阻塞全部管线 |
| WaitFlag 模板参数 | 缺失 | 显式 `<0x4, PIPE_S>` | 避免 modeId/pipe 默认值不匹配 |
| 1:2 Set/Wait 补齐 | 无 | AIC 侧 +FLAG_ID_MAX | 1 AIC 对 2 AIV 的严格配对 |

### 5.2 FLAG_ID_MAX 配对规则

`FLAG_ID_MAX = 16`，flagId 低 4 位为实际 flag 编号，加 16 后映射到第二个 sub-block 的 flag 空间：

```
AIV sub-block 0:  Set(flagId)         →  AIC: Wait(flagId)
AIV sub-block 1:  Set(flagId)         →  AIC: Wait(flagId + 16)
AIC:              Set(flagId + 16)    →  AIV sub-block 1: Wait(flagId + 16)
AIC:              Set(flagId)         →  AIV sub-block 0: Wait(flagId)
```

### 5.3 V→S 同步修复

`stage2.h` 中 `CalGCumExpFp32`/`CalGCumExpBf16` 在 `Exp`（V 管道）后直接 `GetValue`（S 管道），缺少 V→S 同步。添加 `SetFlag/WaitFlag<V_S>`。

此修改不影响精度（经验证加与不加结果完全一致），但符合 AscendC 编程规范。

## 6. 验证

### 6.1 测试方法

在 NPU 上用 `cann_ops_transformer.chunk_gated_delta_rule` 直接调用，对比 baseline (origin/master) 和 fix 的 output/final_state。

### 6.2 测试配置

| 配置 | B | S | chunks | Nk | Nv | Dk | Dv | g | state dtype |
|------|---|---|--------|-----|-----|-----|-----|---|-------------|
| small_no_g | 1 | 64 | 1 | 4 | 4 | 128 | 128 | 无 | bf16 |
| small_with_g | 1 | 64 | 1 | 4 | 4 | 128 | 128 | 有 | bf16 |
| multi_chunk | 1 | 256 | 4 | 4 | 4 | 128 | 128 | 有 | bf16 |
| multi_batch | 2 | 128 | 2 | 4 | 4 | 128 | 128 | 有 | bf16 |
| var_len | 2 | 128 | 2 | 4 | 4 | 128 | 128 | 有 | bf16 |
| fp32_state | 1 | 64 | 1 | 4 | 4 | 128 | 128 | 有 | fp32 |

### 6.3 结果

所有 6 个配置的 output 和 final_state 均与 baseline **bit-exact** 一致（`np.allclose(atol=0, rtol=0)`）。

### 6.4 逐步隔离验证

| 版本 | 单chunk state diff | 4chunks state diff | 问题 |
|------|-------------------|--------------------|----|
| modeId=0x2 + PIPE_FIX Wait | — | — | 编译失败/挂死 |
| modeId=0x4 + PIPE_MTE1/MTE3 Wait | 0.0028 | 0.0972 | 精度偏差 |
| modeId=0x4 + PIPE_MTE2 Wait (AIV) | 0.0028 | 0.0066 | 部分改善 |
| modeId=0x4 + PIPE_S Wait (全部) | 0 | 0 | **bit-exact** |

## 7. 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `chunk_gated_delta_rule_utils.h` | 新增 `FLAG_ID_MAX = 16` 常量 |
| `chunk_gated_delta_rule_stage1.h` | 12 处 CrossCore + 6 处 FLAG_ID_MAX 补齐 |
| `chunk_gated_delta_rule_stage2.h` | 11 处 CrossCore + 4 处 FLAG_ID_MAX 补齐 + 2 处 V_S 同步 |
| `chunk_gated_delta_rule_stage3.h` | 4 处 CrossCore + 2 处 FLAG_ID_MAX 补齐 |
