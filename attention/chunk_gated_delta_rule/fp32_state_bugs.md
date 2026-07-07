# ChunkGatedDeltaRule FP32 State 特性 Bug 分析

> 分析目标 commit: `db6d5350a` (fix(chunk_gated_delta_rule): fix FP32 state accumulation bug and related issues)
>
> 修复状态: Bug 1/2/5/6/7 已修复，Bug 3/4 为潜在隐患（无实际影响）

---

## Bug 1（P0-严重）：首个 chunk 读取未初始化的 `stateDeltaWk_`，导致状态累加错误

**状态**: ✅ 已修复 — 对 `cId == 0` 跳过 delta 的 CopyIn/Cast/Add

**文件**: `op_kernel/arch35/chunk_gated_delta_rule_stage2.h:156-165`

**根因**: FP32 路径采用流水线设计——chunk N 的 delta 由 chunk N-1 的 AIC 计算并写入 `stateDeltaWk_`，chunk N 的 AIV 消费该 delta 进行状态累加。但每组第一个 chunk（cId == 0）不存在 chunk -1，`stateDeltaWk_` 存放的是**未初始化的 workspace 内存**或**前一组最后一个 chunk 的残留 delta**。

**流水线时序**:

```
chunk 0:
  AIC: WaitFlag(0x3) → 读 curState → CalVPrime → CalAttnInter → SetFlag(0x2, 0x2)
       → WaitFlag(0x5) ← 阻塞等待 AIV
  AIV: CopyIn<float>(stateFp32In) → CalGCumExp → WaitFlag(0x2)
       → CopyIn<bfloat16_t>(stateDeltaWk_) ← 此时 AIC 尚未写入 delta！
       → Cast → Add → CopyOut → SetFlag(0x5)

  AIC: ← 收到 0x5
       → CalStateNew → 写入 delta 到 stateDeltaWk_（太晚了！）
```

**影响**:
- 第一个 batch/group 的 chunk 0: 读到的 delta = workspace 垃圾值 → 状态累加错误
- 后续 group 的 chunk 0: 读到的 delta = 前一个 group 最后一个 chunk 的 delta → 错误复用

**修复建议**: 在每组开始时将 `stateDeltaWk_` 清零；或对 chunk 0 跳过 delta 累加（初始 delta 应为全零）。

**关键代码**:
```cpp
// stage2.h:156-162
CrossCoreWaitFlag(0x2);
if (GetSubBlockIdx() == 0) {
    CopyIn<bfloat16_t>(sTP_->finalState[nvId * Dv_ * Dk_], Dv_, Dk_);  // ← 读取未初始化数据
    auto deltaIn = inQueue_.DeQue<bfloat16_t>();
    Cast(stateDeltaUb_, deltaIn, RoundMode::CAST_NONE, Dv_ * curDk_);
    inQueue_.FreeTensor(deltaIn);
    Add(stateFp32Ub_, stateFp32Ub_, stateDeltaUb_, Dv_ * Dk_);  // ← 累加了错误值
```

---

## Bug 2（P1-中等）：`Cast` 使用的元素数与 `stateFp32Ub_` 的 padded 分配不匹配

**状态**: ✅ 已修复 — 所有向量操作统一使用 `Dv_ * curDk_`

**文件**: `op_kernel/arch35/chunk_gated_delta_rule_stage2.h:149`, `:169`, `:242`

**根因**: `stateFp32Ub_` 按 padded 维度 `Dv_ * curDkF32_` 分配（`curDkF32_ = Ceil(Dk_, 8) * 8`），但 `Cast` 按 `Dv_ * Dk_` 连续读取。当 `Dk_ % 8 != 0` 时，padding 元素被错误地当作数据读取，导致后续行偏移错位。

**触发条件**: `Dk_ % 8 != 0`（常见 Dk_ 值如 64、128 不受影响）

**示例**（`Dk_ = 20, curDkF32_ = 24, Dv_ = 2`）:

```
stateFp32Ub_ 布局:
  Row 0: [0..19 有效数据] [20..23 padding=0]
  Row 1: [0..19 有效数据] [20..23 padding=0]

Cast(bf16Init, stateFp32Ub_, ..., Dv_ * Dk_ = 40) 连续读取:
  [0..19] → Row 0 有效数据 OK
  [20..23] → Row 0 padding ← 误读为数据！
  [24..39] → Row 1 前 16 列 ← 缺少 Row 1 末尾 4 列！
```

**修复建议**: `Cast` 应使用 `Dv_ * curDkF32_`，或改用逐行 Cast 避免跨 padding 读取。

**关键代码**:
```cpp
// stage2.h:89
stateFp32Ub_ = tmpBuff_.GetWithOffset<float>(
    static_cast<uint32_t>(Dv_ * curDkF32_), buffOffset);  // ← padded 分配

// stage2.h:149
Cast(bf16Init, stateFp32Ub_, RoundMode::CAST_RINT, Dv_ * Dk_);  // ← 按 unpadded 读取
```

---

## Bug 3（P3-低）：`CopyOut` 写入顺序 — FP32 输出早于 BF16 中间结果

**文件**: `op_kernel/arch35/chunk_gated_delta_rule_stage2.h:167-171`

**根因**: 在 AIV 写入输出时，FP32 状态先写入用户 tensor `stateFp32Out`，后写入 workspace `curState`（`stateBf16Wk_`）。这两块是不同的内存地址，且下一 chunk 的 AIV 通过 `stateFp32Out` 读取 FP32 状态（不依赖 `curState`），因此**当前场景下无实际影响**。但写入顺序与直觉相反，若未来有代码同时依赖两处输出且有访存顺序要求，可能引入隐蔽 bug。

**关键代码**:
```cpp
// stage2.h:167-171
CopyOut<float>(sTP_->stateFp32Out[nvId * Dv_ * Dk_], Dv_, Dk_, false);    // ① 先写用户 tensor
auto bf16Accum = outQueue_.AllocTensor<bfloat16_t>();
Cast(bf16Accum, stateFp32Ub_, RoundMode::CAST_RINT, Dv_ * Dk_);
outQueue_.EnQue(bf16Accum);
CopyOut<bfloat16_t>(sTP_->curState[nvId * Dv_ * Dk_], Dv_, Dk_, false);    // ② 后写中间 workspace
```

---

## Bug 4（P3-低）：`stateBf16Wk_` / `stateDeltaWk_` 跨 group 复用无显式初始化

**文件**: `op_kernel/arch35/chunk_gated_delta_rule.h:148-151`

**根因**: `stateBf16Wk_` 和 `stateDeltaWk_` 与 `highStateAddr_` 指向同一块 workspace，不同 batch/group 之间复用但未被显式清零。虽然流水线设计保证了 chunk 1+ 的数据有效性，但首次写入之前或 group 边界处这些 buffer 处于未定义状态。

**关键代码**:
```cpp
// chunk_gated_delta_rule.h:148-151
stateBf16Wk_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(highStateAddr_), stateWkElemCnt);
stateDeltaWk_.SetGlobalBuffer(
    reinterpret_cast<__gm__ lowType *>(highStateAddr_) + stateWkElemCnt, stateWkElemCnt);
// ↑ 只分配不初始化
```

---

## 总结

| 优先级 | Bug | 是否本次 patch 引入 | 影响范围 | 修复状态 |
|:------:|-----|:---:|------|:---:|
| P0 | chunk 0 读未初始化 delta → 状态累加错误 | 否（预存） | 所有 FP32 state 路径 | ✅ 已修复 |
| P1 | Cast 未处理 padding → Dk_%8≠0 时数据错位 | 否（预存） | Dk_ 非 8 倍数场景 | ✅ 已修复 |
| P1 | Muls 未处理 padding（Bug 2 扩展） | 否（预存） | Dk_ 非 8 倍数场景 | ✅ 已修复 |
| P1 | Add 操作数 padding 不一致 → 行错位 | 否（预存） | Dk_ 非 8 倍数场景 | ✅ 已修复 |
| P2 | CopyOut\<float\> srcStride=0 与 padded 源不匹配 | 否（预存） | Dk_ 非 8 倍数场景 | ✅ 已修复 |
| P3 | CopyOut 写入顺序反直觉 | 否（预存） | 潜在隐患 | 未修复（无实际影响） |
| P3 | workspace 跨 group 无显式清零 | 否（预存） | 潜在隐患 | 未修复（Bug 1 已规避） |

**修复内容**:

1. **Bug 1**: 对 `cId == 0` 跳过 delta 的 CopyIn/Cast/Add，因 chunk 0 无前驱 delta 应为零
2. **Bug 2/5/6**: `curDkF32_` 改为 `= curDk_`（16 对齐，兼容 float 8 对齐），使 `stateFp32Ub_` 与 `stateDeltaUb_` 行步长一致；所有向量操作（Cast/Muls/Add）统一使用 `Dv_ * curDk_`
3. **Bug 7**: `CopyOut` 新增 `srcCol` 参数计算 `srcStride`；新增 `CopyInFp32State` 将 FP32 状态 pad 到 `curDk_`（与 BF16 路径一致）
4. **Bug 3/4**: 未修复，因当前无实际影响（Bug 1 修复后 delta 不再被 chunk 0 读取）
