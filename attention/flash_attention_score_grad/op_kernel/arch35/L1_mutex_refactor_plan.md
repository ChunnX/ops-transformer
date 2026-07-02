# L1 事件同步改为 Mutex（FAG arch35）分阶段计划

改动范围：仅 `attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_block_cube.h`。
不改 `attention/common/op_kernel/buffer.h`、`buffer_manager.h`、`buffers_policy.h`。

## 已确认的决策
- 只把 **L1**（生产者 MTE2 = DataCopy，消费者 MTE1 = matmul 载入）的 `Set/Wait` 换成 mutex；**L0C 的 `M_FIX/FIX_M` 事件保持不变**（全仓无 M/FIX mutex 先例，风险高）。
- **每块物理 buffer 一个 MutexID**，保住 DB/3buff 并行。
- 保持不改底层三个头文件。

## Mutex 官方说明（CANN）
Mutex 用于核内异步流水指令之间的同步处理，功能类似传统 CPU 的锁机制：通过锁定指定流水再释放流水完成流水间的同步依赖。

- 每个锁对应一个固定 `MutexID`，ID 可由用户自定义（**范围 0-27**），也可通过 `AllocMutexID` / `ReleaseMutexID` 申请释放。
- 用户自定义配置时接口性能最优，但**无法与 TQue 等接口配合**使用；通过 `AllocMutexID` / `ReleaseMutexID` 申请释放，则**可与 TQue 等框架接口混合**使用。
- `Lock`：根据 `MutexID` 获取 Mutex，并阻塞当前流水指令队列，直到对应 `MutexID` 被 `Unlock`。
- `Unlock`：直到前置当前流水指令退出后，根据 `MutexID` 释放对应 Mutex。

原型定义：
```cpp
using MutexID = uint8_t;
class Mutex {
    template <pipe_t pipe>
    static __aicore__ inline void Lock(MutexID id);
    template <pipe_t pipe>
    static __aicore__ inline void Unlock(MutexID id);
};
```

对本方案的影响：
- 本方案统一用 `AllocMutexID` / `ReleaseMutexID`（在 `PerBufferMutex::Init/Uninit` 里），因此可与 `Buffer`/`TQue` 等框架接口混用，符合“不改底层三头文件”的约束。
- MutexID 可用范围为 0-27，共 **28 个**，preload 单配置需 9 个（见下方风险），余量充足。

## 核心机制映射（L1）
一块 L1 物理 buffer 的生命周期含两段临界区，用同一个 MutexID 覆盖（正是现有 preload vL1 的写法）：

- 生产段（MTE2 写入）：`Wait<MTE1_MTE2>`（等消费者释放）→ `Lock<PIPE_MTE2>(idx)`；`Set<MTE2_MTE1>`（生产完成）→ `Unlock<PIPE_MTE2>(idx)`。
- 消费段（MTE1 载入）：`Wait<MTE2_MTE1>`（等生产完成）→ `Lock<PIPE_MTE1>(idx)`；`Set<MTE1_MTE2>`（消费完成）→ `Unlock<PIPE_MTE1>(idx)`。

同一 MutexID 保证：消费段 `Lock<MTE1>` 必等生产段 `Unlock<MTE2>`；下一轮生产段 `Lock<MTE2>` 必等上一轮消费段 `Unlock<MTE1>`，与原正反向事件语义一一对应。

`idx` 用返回引用地址识别物理块（避免复刻 policy 内部 flag 轮转）：
```cpp
auto &ref = policy.GetXxx(...);
uint32_t idx = xxxPerBufMutex.IndexOf(&ref);
localBuffer = ref;
```

## 实施策略（关键）
保留原有 `Set/Wait` 调用（在 preload 下因 buffer 为 NO_SYNC 变成空操作，在 reuse/非 reuse 下仍是真实事件），
仅新增 `if constexpr (IS_L1_PRELOAD) { mutex.Lock/Unlock(...) }`。这样：
- preload 实例：buffer=NO_SYNC → Set/Wait 空操作 → mutex 负责同步。
- reuse / 非 reuse 实例：buffer=INNER_CORE_SYNC → 事件真实生效 → mutex 代码被 `if constexpr` 编译掉。

## 分阶段
- **Phase A（本次）**：仅 preload 路径（IS_L1_PRELOAD）。把 dY / v / q / k 四个 L1 policy 在 preload 下转 NO_SYNC + per-buffer mutex，
  覆盖 IterateMmDyV / IterateMmQK 的生产消费，以及 IterateMmDsK / IterateMmDsQ / IterateMmPDy（含 OlderDeter）里的复用释放。
- **Phase B**：扩到 IS_L1_REUSE（非 preload）。
- **Phase C**：扩到非 reuse 的 commonL1Buf / fp32L1Buf。

## Phase A 具体步骤
1. 新增 `PerBufferMutex<N>`（namespace FagBaseApi，FAGBlockCube 之前），基于 `IndexOf(&ref)` 地址识别。
2. preload 分支 selector 改 NO_SYNC：`DyL1BuffSelector` / `QL1BuffSelector` / `KL1BuffSelector` 的 `IS_L1_PRELOAD` 子分支。
3. 成员：删除 `MutexID vL1BufMutexId;`，新增 `dyL1PerBufMutex / vL1PerBufMutex / qL1PerBufMutex / kL1PerBufMutex` 及常量 `DY_L1_BUF_NUM / Q_L1_BUF_NUM`。
4. `InitCubeBuffer` / 析构：在 `if constexpr (IS_L1_PRELOAD)` 分支内 Init / Uninit 上述 4 个 mutex。
5. 局部 `Buffer<BufferType::L1>` 声明（dy/dyNext/q/qNext/k）改为 `Buffer<BufferType::L1, SYNC_TYPE>` 并补 `.template`。
6. 在各函数 preload 分支按“核心机制映射”新增 Lock/Unlock，vL1 迁移到 `vL1PerBufMutex`。

## 保持不变
- 所有 L0C 的 `Set/Wait<M_FIX/FIX_M>`；L0C policy 仍默认 INNER_CORE_SYNC。
- L0A/L0B 同步（在 matmul.h 内部）。
- `ENABLE_UNITFLAG` 分支逻辑。

## 风险与验证
- **MutexID 数量上限**：MutexID 范围 0-27（共 28 个）。preload 单配置需 `dY3 + v1 + q3 + k2 = 9` 个，通过 `AllocMutexID` 动态申请，余量充足，**不再是阻塞项**；仍需注意与 `Buffer` 内部/L0C 等其它 `AllocMutexID` 使用者叠加后不超 28。
- **死锁 / 跨函数持锁**：dy/q/k 的消费段 `Lock<MTE1>` 在 mm1/mm2，释放 `Unlock<MTE1>` 在 mm3/mm4/mm5（软件流水），靠 `IndexOf(&ref)` 保证同一物理块同一 MutexID；需上板确认锁成对、无重复 Lock/Unlock。
- 每阶段单独编译 + 上板精度/性能回归。

## 自检清单
- 底层三头文件未改。
- preload 下 L1 的 Set/Wait 为空操作，同步由 mutex 承担；reuse/非 reuse 事件不受影响。
- 所有 L0C 的 `Set/Wait<M_FIX/FIX_M>` 仍在。
- 每块 L1 物理 buffer 的 MTE2 段与 MTE1 段使用同一 idx/MutexID。
- arch35 编译通过。
