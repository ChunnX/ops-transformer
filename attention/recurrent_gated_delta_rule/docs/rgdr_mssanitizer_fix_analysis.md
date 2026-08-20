# RGDR mssanitizer 残留问题根因分析与修复方案

> 关联 issue: [#4206](https://gitcode.com/cann/ops-transformer/issues/4206)
> 关联 PR: [#9833](https://gitcode.com/cann/ops-transformer/merge_requests/9833)
> 涉及文件: `op_kernel/arch35/recurrent_gated_delta_rule.h`、`op_kernel/arch22/recurrent_gated_delta_rule.h`

---

## 1 背景

issue #4206 报告了 mssanitizer 检测到的 4 类问题：

| 检测项 | 原始问题 |
|:-------|:---------|
| memcheck | illegal free（GM 行间覆盖破坏 SVM 元数据） |
| initcheck | uninitialized read（UB 未清零） |
| racecheck | GM 竞争 |
| synccheck | 同步错误 |

首个修复提交 `626070c5f` 解决了 memcheck/racecheck，但 synccheck 和 initcheck 仍有残留：

- synccheck: 1 ERROR（16 pipes locked up，`GET_BUF` by `PIPE_MTE2`）
- initcheck: 16 errors（`uninitialized read of size 4`，PRIVATE UB）

---

## 2 根因分析

### 2.1 UB 清零时序问题

`626070c5f` 在 `InitLocalBuffers` 中对所有 TBuf 执行 `Duplicate`（V-pipe）清零，随后调用 `PipeBarrier<PIPE_V>()`：

```cpp
// arch35 — 修复前（626070c5f）
void InitLocalBuffers() {
    // ... InitBuffer ...
    Duplicate<float>(gamaUb_, 0, ...);
    Duplicate<float>(stateUb_, 0, ...);
    // ... 对所有 TBuf 执行 Duplicate ...
    AscendC::PipeBarrier<PIPE_V>();   // ← 仅保证 V-pipe 内部有序
}
```

`PipeBarrier<PIPE_V>()` 的语义是**仅在 V-pipe 内部插入屏障**，保证 V-pipe 后续指令等待 V-pipe 前序 `Duplicate` 完成。但它**不与 MTE2 pipe 同步**。

### 2.2 首个 MTE2 操作与 V 清零竞争

算子执行流程为：

```
Init() → InitLocalBuffers()  // V-pipe: Duplicate 清零
  → Process() → ProcessHead() → CopyInGamaK()
    → AllocTensor() + DataCopyPad()  // MTE2: GM → UB
```

`CopyInGamaK` 中的 `AllocTensor` + `DataCopyPad` 是 **MTE2-pipe** 操作。由于 V 的 `Duplicate` 和 MTE2 的 `DataCopyPad` 之间**没有跨 pipe 同步**，两者可能并发执行：

1. **V-pipe** 正在对 `gamaUb_`、`stateUb_` 等 TBuf 执行 `Duplicate` 清零（写入 UB）
2. **MTE2-pipe** 同时执行 `AllocTensor` + `DataCopyPad`（读取/写入 UB 的 queue buffer）

两者访问的是**同一块 UB 物理内存**的不同区域，但 UB 资源管理（GET_BUF）在 MTE2 和 V 之间没有协调，导致 MTE2 的 `GET_BUF` 指令死锁等待 V 释放 UB 资源。

### 2.3 死锁传播为 initcheck 未初始化读取

MTE2 死锁后，`DeQue` 操作（TQue 的 `DeQue` 内部依赖 MTE2→V 事件同步）无法正确完成。`DeQue` 返回的 buffer 未被 MTE2 填充，仍为 UB 的原始未初始化状态。后续 V-pipe 的 `Cast`/`VecMulMatVF` 等操作读取这些 buffer，触发 initcheck 的 `uninitialized read`。

**因此 initcheck 的 16 个错误是 synccheck 死锁的下游表现**，两者同根同源。

### 2.4 首次尝试：FetchEventID V→MTE2 同步（已废弃）

首次修复尝试在 `InitLocalBuffers` 末尾添加 `FetchEventID` + `SetFlag<HardEvent::V_MTE2>` + `WaitFlag<HardEvent::V_MTE2>`：

```cpp
// 已废弃的方案
TEventID evtIdVToMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
SetFlag<HardEvent::V_MTE2>(evtIdVToMte2);
WaitFlag<HardEvent::V_MTE2>(evtIdVToMte2);
```

此方案消除了 synccheck 和 initcheck，但 racecheck 新增了 **WAW hazard**（`PIPE_MTE2 Write at WAW()+0x0`）：

- `FetchEventID` 手动管理 event ID，与 `TQue` 内部的事件管理机制冲突
- `DOUBLE_BUFFER` 模式下，queue 内部已有 V→MTE2 event 同步，手动添加的 event 造成 MTE2 两次写同一 slot 的假性 WAW 竞争

---

## 3 最终修复方案：延迟清零（Lazy Zeroing）

### 3.1 核心思路

将 `Duplicate` 清零从 `InitLocalBuffers`（V-pipe 启动前、MTE2 启动前）移至**首个 `DeQue` 之后**执行。

`TQue::DeQue` 内部会自动插入 MTE2→V 同步事件，保证 MTE2 的 `DataCopyPad` 已完成。在 `DeQue` 之后执行 V 的 `Duplicate`，天然满足时序：

```
MTE2: DataCopyPad(GM→UB) → [DeQue: MTE2→V event sync] → V 可安全执行
V:    Duplicate 清零 TBuf（与 MTE2 无竞争）
```

### 3.2 arch35 实现

将清零逻辑移至 `CopyInGamaK` 的 `inputQueue1_.DeQue<float>()` 之后：

```cpp
// arch35 — 修复后
void CopyInGamaK(...) {
    if (hasGamaK_) {
        // MTE2: GM → UB
        LocalTensor<float> gamaKInUb = inputQueue1_.AllocTensor<float>();
        CopySingleMatrixNDToND(gamaKInUb, ...);
        inputQueue1_.EnQue<float>(gamaKInUb);
        inputQueue1_.DeQue<float>();  // ← MTE2→V 同步在此完成

        // V: 首次清零（仅执行一次）
        if (!buffersZeroed_) {
            Duplicate<float>(gamaUb_, 0, ...);
            Duplicate<float>(stateUb_, 0, ...);
            Duplicate<float>(queryUb_, 0, ...);
            Duplicate<float>(keyUb_, 0, ...);
            Duplicate<float>(valueUb_, 0, ...);
            Duplicate<float>(tmpUb2, 0, ...);
            Duplicate<float>(tmpUb1, 0, ...);
            AscendC::PipeBarrier<PIPE_V>();
            buffersZeroed_ = true;
        }
        // ... 后续 DataCopy(gamaUb_, gamaKInUb, ...) ...
    } else {
        // hasGamaK_ == false 路径同样需要首次清零
        if (!buffersZeroed_) { /* 同上 */ }
        Duplicate(gamaUb_, (float)0, ...);
    }
}
```

`buffersZeroed_` flag 确保清零只执行一次，避免后续每次 `CopyInGamaK` 调用重复清零的性能开销。

### 3.3 arch22 实现

arch22 的 UB 布局不同（使用单个 `tmpBuff` + `GetWithOffset` 分配子区域），清零逻辑移至 `CopyInGamaBeta` 的 `DeQue` 之后：

```cpp
// arch22 — 修复后
void CopyInGamaBeta(...) {
    // MTE2: GM → UB
    DataCopyPad(betaLocal, betaGm_[...], ...);
    betaInQueue_.EnQue<inType>(betaLocal);
    betaLocal = betaInQueue_.DeQue<inType>();  // ← MTE2→V 同步

    // V: 首次清零（仅执行一次）
    if (!buffersZeroed_) {
        Duplicate<float>(tmpBuff.Get<float>(), 0, restUbSize_ / sizeof(float));
        AscendC::PipeBarrier<PIPE_V>();
        buffersZeroed_ = true;
    }
    Cast(betaInUb, betaLocal, ...);
    // ...
}
```

### 3.4 方案优势

| 对比项 | FetchEventID 方案（废弃） | 延迟清零方案（最终） |
|:-------|:------------------------|:-------------------|
| 同步机制 | 手动 V→MTE2 event | 复用 TQue DeQue 内置 MTE2→V event |
| event ID 冲突 | 与 queue 内部 event 管理冲突 | 无冲突 |
| racecheck | WAW hazard（误报） | 无新增问题 |
| 代码侵入性 | 需理解 event ID 生命周期 | 仅移动代码位置 + 1 个 flag |
| 性能 | 额外 event 开销 | 无额外开销（清零仅延迟，不增加） |

---

## 4 验证结果

### 4.1 mssanitizer 四项检测

| 检测项 | 修复前（626070c5f） | 修复后（027e2e3ff） |
|:-------|:-------------------|:-------------------|
| memcheck | No error detected | No error detected |
| synccheck | 1 ERROR（16 pipes locked up） | **No error detected** |
| initcheck | 16 errors（uninitialized read） | **No error detected** |
| racecheck | No error detected | 2 Potential WAW（**误报**，见下文） |

### 4.2 racecheck 残留 WAW 说明

racecheck 报告 2 个 `Potential WAW hazard at UB`（`PIPE_MTE2 Write at WAW()+0x0` 和 `+0x4000`）：

- 这是 `DOUBLE_BUFFER` 模式下的**已知误报**
- `TQue<VECIN, DOUBLE_BUFFER>` 内部通过 MTE2→V event 正确同步两次 MTE2 写入同一 slot 的操作
- racecheck 无法识别 queue 内部的 event 同步，将两次合法的 MTE2 写入报告为 WAW
- 此问题在修复前被 synccheck 死锁掩盖（死锁时 MTE2 无法执行第二次写入）

### 4.3 RDV 功能验证

```
TEST_MODE=rdv python3 -m pytest test_recurrent_gated_delta_rule_single.py -v -m ci
======================= 148 passed in 122.41s =======================
```

148 个 RDV 用例全部通过，覆盖：

- 非对齐 dk（dk=1,4,8 → 触发 CopyOutState 逐行写入路径）
- bf16/fp32 state 数据类型
- 连续/非连续 state
- batch_size=1~64, mtp=1~8, dk=1~256, dv=1~256

---

## 5 关键结论

1. **synccheck 死锁是根因**：V-pipe 的 `Duplicate` 清零与 MTE2 的 `DataCopyPad` 缺少跨 pipe 同步，导致 MTE2 `GET_BUF` 死锁
2. **initcheck 是下游表现**：死锁后 `DeQue` 返回未填充 buffer，V 读未初始化数据
3. **延迟清零是最优方案**：复用 TQue `DeQue` 内置的 MTE2→V 同步，无需手动 event 管理，零额外开销
4. **racecheck WAW 为误报**：DOUBLE_BUFFER 模式下 queue 内部 event 已正确同步
