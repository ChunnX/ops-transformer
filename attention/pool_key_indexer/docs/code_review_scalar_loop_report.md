# PoolKeyIndexer Kernel 标量循环检视报告

> 检视日期：2026-09-02
> 检视主题：op_kernel 实现中低性能标量循环的排查与分级
> 检视范围：`op_kernel/arch22/*`、`op_kernel/arch35/*`（含 `arch35/vf/*`）、`op_kernel/common`
> 检视方法：以「热路径（per-token / per-s1-row / per-s2-block）是否存在逐元素标量 GetValue/SetValue/乘加」为判定基准，逐文件检索标量循环，并与 `docs/build_fix_record.md` 已登记的史实交叉核对；按「必须优化 / 建议优化 / 不需要优化」三级归类。

---

## 总览

| 类别 | 数量 | 位置 |
|------|------|------|
| 必须优化 | 1 | `arch35/pool_key_indexer_service_vector.h:638-649`（quant 预乘标量循环） |
| 建议优化 | 2 | `ExpandAndAppendIndices` 逐池 GetValue + 非 8 倍数 ps 全标量展开（arch22/arch35） |
| 不需要优化 | 4 | 尾块标量写、PA block_table 逐块 GetValue、元数据 GetValue、一次性常量 SetValue |

> 总体结论：kernel 中**不存在**因疏忽导致的、会随 topk/token 数线性放大且无解的标量 hot loop；所有标量循环要么是**有意为之的正确性退化路径**（32B 对齐约束），要么是**固有成本**（Duplicate 广播源），且大部分已在 `build_fix_record.md` 登记并预留向量化方案。唯一"新引入且位于 score 计算热路径、有清晰向量化路径"的是本次 quantmode 提交加入的 weight×scale 预乘标量循环，列为必须优化。

---

## 一、必须优化

### [M-1] 量化 weight×qScale 逐元素标量预乘循环（arch35 quant mode=0 热路径）

- **位置**：`arch35/pool_key_indexer_service_vector.h:638-649`（`PoolKeyIndexerServiceVector::ProcessVec1`，`if constexpr (LIT::isFp8PerToken)` 分支）
- **触发条件**：quant_mode=0（FP8 per-token-head），仅 Ascend950/arch35。
- **代码**：

```cpp
for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
    uint64_t srcOff = pingpong * ... + s1IdxTmp * gSizeAlign16;
    uint64_t dstOff = pingpong * ... + s1IdxTmp * 128;
    for (int64_t g = 0; g < gSize_; g++) {
        float wv = static_cast<float>(weightUB_.GetValue(srcOff + g)); // fp16 weight -> fp32
        float qs = qScaleUB_.GetValue(dstOff + g);                     // fp32 qScale
        weightScaleUB_.SetValue(dstOff + g, wv * qs);                  // 逐元素标量乘
    }
}
```

- **问题**：`ProcessVec1` 处于分数计算热路径（每 gS1 块 × 每 s2 块迭代一次），该循环对每个 s1 行做 `gSize_`（=n1，规格上限 64）次标量 `GetValue + GetValue + 浮点乘 + SetValue`。n1=64 时每个 s2 块、每个 s1 行就是 64 次标量乘，叠加 3 次标量访存，随 s2 块数线性放大。
- **为什么写成标量**：代码注释自述「Gather 对寄存器 Mul 产物 lane 布局不兼容，预乘后 WithScale 走纯 fp32 LoadAlign+Gather」。即 constrait 是在**后续** `MulWeightAndReduceSumWithScale` 的 LoadAlign+Gather 消费侧，而本循环的产物 `weightScaleUB_` 是「每 s1 行 gSize 个连续 fp32、行距 128」的 UB 缓冲。
- **为什么可以向量化**：本循环只是把「fp16 weight → fp32」再「× fp32 scale」写入一段**连续 fp32 UB 缓冲**（行距 128 不变），不涉及寄存器级 Gather lane 布局。产物布局由标量写逐元素确定，等价于「向量 Cast + 向量 Mul」写出同样布局，消费侧 `LoadAlign+Gather` 无感知。
- **修改建议**：

```cpp
// 用向量 Cast（half→float）+ 向量 Mul（fp32）替换三重标量循环：
//   1) Cast: weightUB_[srcOff..srcOff+gSize) (fp16) -> weightFp32UB_[dstOff..) (fp32)
//   2) Mul : weightScaleUB_[dstOff] = weightFp32UB_[dstOff] * qScaleUB_[dstOff]
Cast(weightFp32UB_[dstOff], weightUB_[srcOff], CastMode::ROUND_NONE, gSize_);
Mul<float>(weightScaleUB_[dstOff], weightFp32UB_[dstOff], qScaleUB_[dstOff], gSize_);
```

  实现要点：
  1. `weightFp32UB_` 复用一段 fp32 UB 缓冲（可复用 `weightScaleUB_` 的另半区做临时，或单独分配 ≤ `16*128*4B` 的小缓冲）；
  2. 向量 `Mul<float>` 要求 32B（8 个 fp32）粒度，`gSize_` 非 8 倍数时（如 n1=4/5/12）按「8 对齐段向量 + 尾段标量」处理（`gSize_≤64`，尾段 ≤7 次标量，可接受）；
  3. 保留现有 `MTE2ToSSync/VToSSync` 前的同步时序点不变，仅把「标量乘」换成「向量 Cast+Mul」，同步语义（`SToVSync` 后再进入 WithScale）不受影响；
  4. 落地后需回归 `test_quant.py`/`test_quant_general*.py` 的 mode=0 全量用例 + `compare_values`（values 参考通道关注 max_abs_err 不劣化）。

- **预期收益**：将每 s1 行 gSize 次标量乘降为 2~4 条向量指令 + 常数次边界标量；n1 越大、s2 越长的 FP8 场景收益越明显。

---

## 二、建议优化

### [S-1] `ExpandAndAppendIndices` 逐池标量 GetValue（每行 sparseCount 次）

- **位置**：
  - `arch22/pool_key_indexer_service_vector.h:398-405`（vec 分支 `base = poolIndices.GetValue(k) * poolSize` 后 `Duplicate + Add`）
  - `arch35/pool_key_indexer_service_vector.h:521-527`（同构 vec 分支）
- **性质**：主路径（ps%8==0）每行仍需 `sparseCount`（=topk/ps）次标量 `GetValue` 读取选中 pool 序号，作为 `Duplicate` 广播源。这是「逐池必须取一次 pool 序号」的固有成本，非疏忽。
- **状态**：已在 `build_fix_record.md:1462-1469`（"残留标量与后续优化方向"）登记，并给出 Gather 向量化方案：
  `out[i] = i + ps * (poolIdx[i/ps] - i/ps)`，逐行 `CreateVecIndex(iota) + Gather(sel, poolIdx, kTpl) + Sub/Muls/Add` 约 5 条向量指令，代价 +1 个 outputLen 长度 UB 模板。
- **修改建议**：以 msprof 确认「逐池 GetValue」成为真实瓶颈后再实施 Gather 方案（`Gather` Level-2 已确认 dav_3510/arch35 支持 int32；arch22 需单独确认 pipe 语义）。无数据不提前做。

### [S-2] 非 8 倍数 poolSize 的逐 token 全标量展开写（O(topk) 标量写/行）

- **位置**：
  - `arch22/pool_key_indexer_service_vector.h:406-410`（`else { for(p<ps) SetValue }`）
  - `arch35/pool_key_indexer_service_vector.h:486-491`（ps%8!=0 专用标量退化路径）
- **性质**：ps 非 8 倍数时（12/24/96/100/127…），向量写偏移不满足 32B 操作数对齐，无法用 `Duplicate/Add` 展开，退化为 `expandRounds × poolSize = topk` 次标量 `SetValue`（topk=8192 时每行 8192 次）。这是缺陷 C（aicore exception 507015）的正确性退化修复，**不是误写**。
- **状态**：与 S-1 同源于 `ExpandAndAppendIndices`，`build_fix_record.md` 已登记的 Gather 方案同样可统一解决（「与 ps%8!=0 路径的完全向量化可由 Gather 方案统一」）。
- **修改建议**：与 S-1 合并处理——实施 Gather 展开后，ps 非 8 倍数路径的标量退化可整体删除，一并消除「标量精确写 + VToSSync/SToMTE3Sync」的额外同步开销。触发条件不变：非 8 倍数 ps 成为真实业务配置或逐池 GetValue 成为瓶颈时再实施。

---

## 三、不需要优化

### [N-1] 尾块标量写（≤ poolSize-1 次/行）

- **位置**：`arch22/pool_key_indexer_service_vector.h:436-438`；`arch35/pool_key_indexer_service_vector.h:494-496`（scalar 分支）与 `:549-551`（vec 分支）。
- **理由**：尾块容量固定为 `poolSize-1`，每条 s1 行最多 127 次标量 `SetValue`，且必须「先 Align(ps-1,8) 对齐清空 + 标量精确写 visibleTailK 个」以避免越出尾区容量（与 arch22 修复一致）。量级小、正确性必需，保留。

### [N-2] PA block_table 逐块标量 GetValue（block 粒度）

- **位置**：`arch35/pool_key_indexer_service_cube.h:348/430`；`arch22/pool_key_indexer_service_cube.h:255`。
- **理由**：每个 s2 块（`s2BaseSize` 粒度）做 1 次 `blkTableGm_.GetValue` 读物理块号，是 PA 分页块表寻址的必要间接（块粒度，非 token 粒度）。块数极大时 GM 标量读延迟才需考虑，当前非瓶颈。

### [N-3] 运行期元数据 GetValue（batch / 控制流粒度）

- **位置**：`arch22/pool_key_indexer_kernel.h:239-253`、`arch35/pool_key_indexer_kernel.h:269-284`（`actualSeq / poolTailK.GetValue`，每 batch 1 次）；`arch22/pool_key_indexer_service_vector.h:891-899`（wsInfo 每行 13 个字段，LD 归并控制流）。
- **理由**：均为 batch 级或行级元数据读取 / 归并控制，读取次数与 batch / 核数同阶，不随 token 数线性放大，非性能热点。

### [N-4] 一次性常量 SetValue（非循环）

- **位置**：`arch22/pool_key_indexer_service_vector.h:746-761`（tempB 13 个常量字段写）；`arch35/pool_key_indexer_service_vector.h:630-633/750-751` 的 `Duplicate` 铺常量。
- **理由**：固定次数的初始化写（非循环体），或已是 `Duplicate` 向量铺常量，无需处理。

---

## 附：逐文件标量循环分布（供复核）

| 文件 | GetValue/SetValue 处数 | 热路径标量循环 | 分级 |
|------|:---:|------|------|
| arch22/pool_key_indexer_service_vector.h | 32 | 逐池 GetValue + 非 8 倍数 ps 展开 + 尾块写 | S-1/S-2/N-1 |
| arch35/pool_key_indexer_service_vector.h | 20 | 同上 + **quant 预乘循环** | M-1/S-1/S-2/N-1 |
| arch22/pool_key_indexer_service_cube.h | 1 | block_table 逐块 GetValue | N-2 |
| arch35/pool_key_indexer_service_cube.h | 2 | block_table 逐块 GetValue | N-2 |
| arch22/pool_key_indexer_kernel.h | 8 | actualSeq/poolTailK 元数据 | N-3 |
| arch35/pool_key_indexer_kernel.h | 8 | 同上 | N-3 |
| arch35/vf/*（topk/gather/vector1） | 0 | 无（TopK 走 VF 直通向量路径） | — |
