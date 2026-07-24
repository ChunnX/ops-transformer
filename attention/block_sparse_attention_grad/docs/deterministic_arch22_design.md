# block_sparse_attention_grad arch22 确定性计算设计方案

## 1. 背景

`block_sparse_attention_grad` 算子在 arch22（Ascend 910B / DAV_2201）上需要支持 `torch.use_deterministic_algorithms(True)`。当用户启用确定性模式时，框架通过 `GetDeterministic()` 查询算子是否注册了确定性支持，若已注册则设置 `deterministic=1` tiling 标志。

算子中的 dK/dV 梯度计算使用 `SetAtomicAdd`（原子加）将多个 AIC 核的中间结果累加到共享 GM 输出区。跨核原子加的执行顺序不确定，导致浮点累加舍入不同 → 非确定性。

## 2. 硬件约束

| 约束 | 说明 | 影响 |
|------|------|------|
| `SyncAll<false>` 是跨 AIC↔AIV 同步 | `isAIVOnly=false`，AIC 侧调用 `wait_flag_dev(SYNC_AIV_FLAG)` 等待 AIV 信号 | 在 cube 循环内使用会打乱 FIX 管线时序，导致 FIX 写不落地 |
| `SyncAll<true>` 是 AIV-only 同步 | `isAIVOnly=true`，仅同步 AIV 核 | 可安全用于 AIV 侧同步 |
| AIV MTE3 写→读 DCache 不一致 | DataCopy 写 gDvShared 后，同核后续 DataCopy 读 gDvShared 间歇性读到旧 DCache 数据 | 方案C（AIV 聚合）不可行 |
| AIC FIX 写绕过 DCache | FIX 管线直接写 GM，不经过 DCache | VecPost 的 MTE3 读从 GM 获取正确数据 |
| `dcci` 未公开 | Data Cache Clean/Invalidate 是 arch22 内部 intrinsic | 无法手动 invalidate DCache |
| `ReadGmByPassDCache` 仅支持 int 类型 | 不支持 float | 无法直接绕过 DCache 读 float 数据 |
| `SetAtomicAdd(false)` 导致 FIX 不写 | arch22 上 `SetAtomicNone` + FIX = 不写 | 必须使用 `SetAtomicAdd(true)` |
| UB_SIZE = 192KB | AIV UB 总大小 | 聚合 UB 缓冲区需在 184KB~192KB 范围内 |

## 3. 方案演进

### 3.1 方案B：per-core slot + per-round SyncAll + AIV 聚合（失败）

**思路**：每个 AIC 核写私有 slot（无跨核碰撞），AIV 每轮聚合各核 slot 到共享输出。

**失败原因**：`SyncAll<false>` 在 cube 循环内破坏 FIX 管线时序，导致 dV=0（FIX 写不落地）。即使加 `PipeBarrier<PIPE_FIX>` 也无法修复。

### 3.2 方案C：post-loop 聚合（失败）

**思路**：移除所有逐轮 SyncAll，cube 写 per-core slot，cube 循环结束后 AIV 统一聚合所有 slot 到共享 gDk/gDv。

**失败原因**：AIV 的 MTE3 写（DataCopy 聚合写 gDvShared）对同核后续 MTE3 读（VecPost DataCopy 读 gDvShared）不可靠可见。arch22 上 MTE3 管线的写缓冲/DCache 不保证 write→read ordering。

**DumpTensor 定位证据**：

| Dump 点 | 内容 | 跨运行一致性 |
|---|---|---|
| dV slot core0 [0:32] | AIC 写 slot 后 | ✅ SAME |
| dV slot core1 [0:32] | AIC 写 slot 后 | ✅ SAME |
| dV slot core0 [1024:1056] | AIC 写 slot 后 | ✅ SAME |
| gDvShared [0:32] 聚合后 | AIV 聚合写后 | ✅ SAME |
| gDvShared [1024:1056] 聚合后 | AIV 聚合写后 | ✅ SAME |
| gDvShared [0:32] VecPost 前 | VecPost 读前 | ✅ SAME |
| **dV 最终输出** | VecPost 写出 | ❌ **DIFF** |

gDvShared 数据完全正确且确定，但 VecPost 的 MTE3 读间歇性读到旧 DCache 数据。

**尝试的修复**：
- `PipeBarrier<PIPE_ALL>` / `PipeBarrier<PIPE_MTE3>`：无效
- `WriteGmByPassDCache` 写 gDvShared：部分改善（8/10→5/10 失败）
- `ReadGmByPassDCache` 刷新 DCache：部分改善（5/10→4/10 失败）
- ALL AIV core 各自聚合各自 VecPost 分区：无效（同核 MTE3 写→读仍有问题）

### 3.3 方案D：cube 写共享 + PipeBarrier<PIPE_FIX>（部分成功）

**思路**：cube 直接写共享 gDk/gDv（SetAtomicAdd），无 per-core slot / 无聚合。`PipeBarrier<PIPE_FIX>` 修复 SetAtomicAdd/SetAtomicNone 竞争。

**结果**：
- 12/12 精度通过 ✅
- 6/12 bitwise 确定 ✅（无跨核碰撞的 shape）
- 6/12 非确定 ❌（有跨核碰撞的 shape — 多个 AIC 核的 mmad3 写同一 kvOffset，atomic add 顺序不确定）

### 3.4 方案E：GM 锁序列化 AIC cube loop（最终方案）

在方案D基础上，用 GM 锁序列化 AIC 核的 cube loop 执行顺序。cube 仍写共享 gDk/gDv（FIX 管线绕过 DCache），但 AIC 核按 coreIdx 顺序串行执行，确保 atomic add 顺序固定 → bitwise 确定。

## 4. 方案E 详细设计

### 4.1 核心思路

确定性来源于两个机制：

1. **GM 锁序列化**：AIC 核按 `coreIdx` 顺序串行执行 cube loop，确保 `SetAtomicAdd` 的写入顺序固定 → atomic add 顺序确定 → bitwise 确定
2. **FIX 管线写共享 gDk/gDv**：FIX 写绕过 DCache → VecPost 的 MTE3 读从 GM 获取正确数据（无 DCache 一致性问题）

### 4.2 数据流

```
AIV (vector) 启动:
  VecPre → SyncAll<false>(init) → VecOp:
    清零 slot → WriteGmByPassDCache(lockGm, 0) → CrossCoreSetFlag(DETER_V2C)
    softmax 循环 → CrossCoreSetFlag(VEC2CUBE) per round
  → WaitEvent(CUBE2POST) → SyncAll<true> → VecPost

AIC (cube) 启动:
  CubePre → SyncAll<false>(init)
  → CrossCoreWaitFlag(DETER_V2C)           // 等 AIV 清零+锁初始化
  → while (ReadGmByPassDCache(lockGm) != coreIdx) {}  // GM 锁等待
  → SetFlag()
  → cube loop:
      mmad1(Q×K→S) → CrossCoreSetFlag(CUBE2VEC) → WaitEvent(VEC2CUBE)
      → mmad2(dOut×K→dS) → mmad3(dP×dOut→dV, dS×Q→dK)  // 写共享 gDk/gDv
  → PipeBarrier<PIPE_FIX>()                // 确保所有 FIX 写完成
  → WriteGmByPassDCache(lockGm, coreIdx+1) // 释放 GM 锁
  → CrossCoreSetFlag(CUBE2POST)
```

### 4.3 GM 锁机制

```cpp
// AIV: 初始化锁为 0 (在 slot 清零后, CrossCoreSetFlag 前)
__gm__ int32_t* lockGm = (__gm__ int32_t*)(params.workspace + tilingData->deterCoreInfoGmOffset);
AscendC::WriteGmByPassDCache<int32_t>(lockGm, 0);

// AIC: 获取锁 (轮询直到 lockGm == coreIdx)
while (AscendC::ReadGmByPassDCache<int32_t>(lockGm) != (int32_t)coreIdx) {}

// AIC: 执行 cube loop (mmad1/mmad2/mmad3, 写共享 gDk/gDv)

// AIC: 释放锁 (lockGm = coreIdx + 1, 下一个核获取)
AscendC::PipeBarrier<PIPE_FIX>();  // 确保 FIX 写完成
AscendC::WriteGmByPassDCache<int32_t>(lockGm, (int32_t)(coreIdx + 1));
```

- `ReadGmByPassDCache` / `WriteGmByPassDCache` 绕过 DCache，直接读写 GM → 跨核可见
- `PipeBarrier<PIPE_FIX>` 确保当前核的所有 FIX 写完成后再释放锁 → 下一个核的 atomic add 不会与当前核的写竞争

### 4.4 PipeBarrier<PIPE_FIX> 修复

`block_mmad_bsag2.hpp` 中的 `SetAtomicAdd`/`SetAtomicNone` 竞争修复：

```cpp
// 修改前（有 bug）：
if (isAtomicAdd) {
    AscendC::SetAtomicAdd<ElementC>();
    copyL0CToGm(gC, l0CTensor[pingpongFlag], layoutC, layoutInL0C);  // FIX 异步写
    AscendC::SetAtomicNone();  // ← 立即切换，FIX 可能还没写完！
}

// 修改后：
if (isAtomicAdd) {
    AscendC::SetAtomicAdd<ElementC>();
    copyL0CToGm(gC, l0CTensor[pingpongFlag], layoutC, layoutInL0C);
    AscendC::PipeBarrier<PIPE_FIX>();  // ← 确保 FIX 写完成后再 SetAtomicNone
    AscendC::SetAtomicNone();
}
```

`copyL0CToGm` 是异步操作（FIX 管线），`SetAtomicNone()` 紧随其后立即改变全局 atomic 模式。如果 FIX 写尚未完成，该写操作可能以非 atomic 模式执行（arch22 上 `SetAtomicNone` + FIX = 不写），导致数据丢失。每次 cube 循环迭代有两个 mmad3 调用（先 dV 后 dK），第一个 mmad3 的 `SetAtomicNone` 可能影响第二个 mmad3 的 FIX 写。

### 4.5 Tiling workspace 布局

```
libapi | sOut | dPOut | dQOut | dKOut | dVOut | sftmgGm(gradSize) | GAP(64MB) | dkSlots | dvSlots | runInfo(含GM锁)
```

- sftmgGm 在 slot 之前，溢出向后通过 64MB GAP 缓冲
- GM 锁使用 `deterCoreInfoGmOffset` 区域（runInfo 的一部分，1 个 int32_t）
- GAP 从 256KB 增大到 64MB（防 sftmgGm 溢出到 deter workspace 区域）

### 4.6 确定性分析

**为什么 bitwise 确定？**

1. **AIC 核串行执行**：GM 锁确保 core 0 先执行完整个 cube loop，然后 core 1，依此类推。同一 kvOffset 的 atomic add 顺序固定（core 0 → core 1 → ... → core N-1）

2. **单核内顺序确定**：同一 AIC 核内，多个迭代可能写同一 kvOffset。`PipeBarrier<PIPE_FIX>` 确保每次 mmad3 的 FIX 写完成后再继续，迭代间顺序固定

3. **FIX 写绕过 DCache**：FIX 管线直接写 GM，不经过 DCache。VecPost 的 MTE3 读从 GM 获取数据（DCache miss → fetch from GM → 正确数据）

4. **GM 锁用 ReadGmByPassDCache/WriteGmByPassDCache**：绕过 DCache，确保跨核锁可见性

### 4.7 性能影响

AIC 核串行执行，性能下降 ~N×（N=AIC 核数，通常 10-20）。后续可优化为按 kvOffset 粒度加锁（仅序列化写同一地址的核，不同地址的核可并行）。

## 5. 代码改动

### 5.1 文件清单

| 文件 | 改动 |
|------|------|
| `op_kernel/arc22/attn_infra/gemm/block/block_mmad_bsag2.hpp` | +1 行：`PipeBarrier<PIPE_FIX>()` |
| `op_kernel/arc22/block_sparse_attention_grad_kernel.h` | +16/-3 行：GM 锁 + CrossCoreWaitFlag |
| `op_host/arch22/block_sparse_attention_grad_tiling_arch22.cpp` | +1/-1 行：GAP 256KB→64MB |

### 5.2 kernel.h 关键改动

**AIC 侧 — 锁等待**：
```cpp
// 等 AIV 清零+锁初始化
if (deter) {
    AscendC::CrossCoreWaitFlag(DETER_V2C);
    // GM 锁等待
    __gm__ int32_t* lockGm = (__gm__ int32_t*)(params.workspace + tilingData->deterCoreInfoGmOffset);
    while (AscendC::ReadGmByPassDCache<int32_t>(lockGm) != (int32_t)coreIdx) {}
}
SetFlag();
```

**AIC 侧 — 锁释放**：
```cpp
if (deter) {
    AscendC::PipeBarrier<PIPE_FIX>();
    __gm__ int32_t* lockGm = (__gm__ int32_t*)(params.workspace + tilingData->deterCoreInfoGmOffset);
    AscendC::WriteGmByPassDCache<int32_t>(lockGm, (int32_t)(coreIdx + 1));
}
AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
```

**AIV 侧 — 锁初始化**：
```cpp
// 方案E: 初始化 GM 锁为 0
__gm__ int32_t* lockGm = (__gm__ int32_t*)(params.workspace + tilingData->deterCoreInfoGmOffset);
AscendC::WriteGmByPassDCache<int32_t>(lockGm, 0);
AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(DETER_V2C);
```

## 6. 测试结果

### 6.1 测试 Shape 覆盖

| # | B | N | kvH | S1 | S2 | D | sparsity | 类型 | 精度 | 确定性 |
|---|---|---|-----|-----|-----|---|----------|------|------|--------|
| 1 | 1 | 2 | 1 | 128 | 128 | 128 | 1.0 | GQA | ✅ | ✅ |
| 2 | 10 | 12 | 4 | 256 | 357 | 128 | 0.5 | GQA | ✅ | ✅ |
| 3 | 1 | 2 | 2 | 128 | 128 | 128 | 1.0 | MHA | ✅ | ✅ |
| 4 | 2 | 8 | 8 | 256 | 256 | 128 | 0.5 | MHA | ✅ | ✅ |
| 5 | 4 | 8 | 2 | 256 | 256 | 128 | 0.5 | GQA | ✅ | ✅ |
| 6 | 2 | 4 | 2 | 512 | 512 | 128 | 0.3 | GQA | ✅ | ✅ |
| 7 | 1 | 4 | 4 | 1024 | 1024 | 128 | 0.5 | GQA | ✅ | ✅ |
| 8 | 3 | 16 | 4 | 128 | 128 | 64 | 0.5 | GQA | ✅ | ✅ |
| 9 | 2 | 8 | 2 | 256 | 384 | 128 | 0.7 | GQA | ✅ | ✅ |
| 10 | 4 | 4 | 4 | 192 | 192 | 128 | 0.5 | MHA | ✅ | ✅ |
| 11 | 1 | 2 | 1 | 128 | 128 | 128 | 0.0 | GQA | ✅ | ✅ |
| 12 | 2 | 6 | 3 | 256 | 256 | 128 | 0.5 | GQA | ✅ | ✅ |

### 6.2 回归测试

- 非 deter 小 shape：9T/0F ✅ 无回归
- deter 小 shape：dV=0.248=golden ✅，10 次运行一致 ✅

### 6.3 确定性验证

- 10× 循环测试（碰撞 shape B=4,N=8,kvH=2,S=256,sp=0.5）：10/10 identical ✅
- 12 shape 全量测试：12/12 精度 + 确定性通过 ✅

## 7. 提交记录

```
6b13ecea1  方案E: GM锁序列化AIC cube loop, 支持所有shape确定性
cf3451031  方案D: cube 写共享 + PipeBarrier PIPE_FIX 修复
de0b0972a  方案B: 修复重复聚合+清零同步+flag冲突
0084f1fb1  方案B: per-core slot + RMW 聚合
8f3107b51  初版 deter 支持
```

## 8. 后续优化方向

1. **kvOffset 粒度加锁**：当前 GM 锁序列化整个 cube loop。可改为仅序列化写同一 kvOffset 的核，不同 kvOffset 的核可并行执行，大幅降低性能损失

2. **per-core slot + AIC 侧聚合**：若未来 CANN 公开 DCache invalidate API（`dcci`），可回到方案C（per-core slot + AIV 聚合），避免串行性能损失

3. **硬件级原子加顺序保证**：若未来 arch22 固件更新支持原子加顺序保证，可移除 GM 锁，恢复全并行执行
