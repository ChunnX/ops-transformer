# mhc_pre_backward 算子性能优化分析

## 一、当前性能瓶颈分析

### 1.1 流水线阻塞

当前 AIV↔AIC 协作采用**严格串行同步**：

```
AIV: GradPre(8轮) → GradHHat2 → SetFlag(8) → [等待Flag 9] → GradX1(8轮)
AIC:                        WaitFlag(8) → mm1 → mm2 → SetFlag(9)
```

- AIV 在 `CrossCoreWaitFlag<0x2>(9)` 处**完全阻塞**，等待 AIC 完成两个 matmul
- AIC 在 `CrossCoreWaitFlag<0x2>(8)` 处**完全阻塞**，等待 AIV 完成所有 Vector 计算
- **无乒乓流水**：外层 tile 循环之间没有重叠，每个 tile 必须完全结束后才开始下一个

### 1.2 ComputeGradPre 中的手动树形归约

```cpp
// L958-968: 手动 4 段树形归约 (c0=256 时)
Add(xCastLocal_[64], xCastLocal_[64], xCastLocal_[128+64], ...);
Add(xCastLocal_,    xCastLocal_,    xCastLocal_[128],    ...);
Add(xCastLocal_,    xCastLocal_,    xCastLocal_[64],     ...);
```

- 硬编码 3 次 Add + 2 次 PipeBarrier，仅支持 c0=256
- 每次 Add 后都需要 PipeBarrier（A2 架构 PIPE_V 内 RAW 依赖不能自动串行）
- 使用 `WholeReduceSum` 替代可减少指令数和 PipeBarrier 次数

### 1.3 ComputeGradPre 内层循环的低效

```cpp
// L938-944: N=4 次循环, 每次内层按 64 元素分段
for (loopIdN = 0; loopIdN < n_; loopIdN++) {
    for (loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += 64) {
        Mul(xCastLocal_[...], xCastOutLocal[...], gradYCastLocal_[...], mask, tileTaskCount, {...});
    }
}
```

- N=4 × (256/64=4) = **16 次 Mul 调用**，每次只处理 64 个元素
- 可用 `BinaryRepeatParams` 的 stride 参数一次性处理整个 N×c0 矩阵

### 1.4 ComputeGradX1 中的类似问题

```cpp
// L790-816: N=4 次循环 × 4 次内层循环 = 16 次 Mul + 16 次 MulAddDst
for (loopIdN = 0; loopIdN < n_; loopIdN++) {
    for (loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += 64) {
        Mul(gradXCastLocal_[...], gradYCastLocal_[...], preBrcbLocal_[...], ...);
    }
}
// ... 同样结构的 MulAddDst 循环
```

- 同样存在 16 次小粒度 Mul + 16 次小粒度 MulAddDst
- 两次循环之间有 PipeBarrier，进一步增加延迟

### 1.5 h_pre/dRsqrt 广播位置

```cpp
// L1188-1217: 在 GradX1 的内层循环中, 每个 inner iteration 都重新广播
for (taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
    Brcb(preBrcbLocal_, preLocal_, ...);    // 每个 inner iteration 都执行
    Brcb(dRsqrtBrcbLocal_, gradRsqrtLocal_, ...);
    ComputeGradX1(...);
}
```

- h_pre 在整个 tile 内不变，但每个 inner iteration 都重新广播
- dRsqrt 在整个 tile 内不变，同理

### 1.6 PipeBarrier 过多

全局搜索 PipeBarrier<PIPE_V>()：

| 位置 | 次数 | 原因 |
|------|------|------|
| ComputeGradPre | ~5 | 树形归约间 |
| ComputeGradHHat2 | ~8 | Mul/Reduce/Muls 间 |
| ComputeGradX1 | ~5 | Cast/Mul/MulAddDst/Add 间 |
| ComputeScaleBias | ~6 | Add 归约间 |
| **总计** | **~24** | A2 架构要求 |

### 1.7 matmul 未分块

```cpp
// tiling L371, L385
mm1Tiling.SetDim(aivNum);   // 48核切分
// kernel L1292-1293
mm1_.SetOrgShape(mm1M, mm1N_, mm1K_);    // 整个shape
mm1_.SetSingleShape(mm1M, mm1N_, mm1K_); // 不分块
```

- mm1: M=64, N=2048, K=24（BSNN），整个 shape 作为 single shape
- mm2: M=24, N=2048, K=B×S，K 可能非常大
- 未做 K 轴分块，大 K 场景可能 L1 cache 不友好

---

## 二、优化方案与预期提升

### 优化1：AIV-AIC 乒乓流水（预期提升 30-40%）

**当前**：串行执行，AIV 等待 AIC 期间完全空闲

```
tile0: [AIV0]----[wait]----[AIV1]
                  [AIC0]----[AIC1]
tile1:                                    [AIV0]----[wait]----[AIV1]
                                                    [AIC0]----[AIC1]
```

**优化后**：双 buffer 乒乓，AIV 处理 tile(i+1) 的同时 AIC 处理 tile(i)

```
tile0: [AIV0]----[AIV1]----
       [AIC0]----[AIC1]----
tile1: [AIV0]----[AIV1]----
       [AIC0]----[AIC1]----
```

**实现要点**：
- 分配双份 workspace：`gradH2Global_[2]`, `xWorkspaceGlobal_[2]`, `gradXCubeGlobal_[2]`
- AIV 写 buffer[i%2]，AIC 读 buffer[(i-1)%2]
- Flag 编号使用 `(i%2)*2 + 8` 和 `(i%2)*2 + 9` 交替
- 首轮 AIC 等待，末轮 AIV 等待

**预期提升**：AIV 和 AIC 的执行时间重叠，理论可提升 ~40%（取两者较大者为准）

**风险**：workspace 内存翻倍，需评估 GM 容量

---

### 优化2：ComputeGradPre 用 WholeReduceSum 替代手动树形归约（预期提升 5-8%）

**当前**：3 次 Add + 2 次 PipeBarrier，硬编码 c0=256

```cpp
Add(xCastLocal_[64], xCastLocal_[64], xCastLocal_[128+64], ...);
PipeBarrier<PIPE_V>();
Add(xCastLocal_,    xCastLocal_,    xCastLocal_[128],    ...);
PipeBarrier<PIPE_V>();
Add(xCastLocal_,    xCastLocal_,    xCastLocal_[64],     ...);
```

**优化后**：1 次 WholeReduceSum + 1 次 PipeBarrier

```cpp
WholeReduceSum(xCastLocal_, xCastLocal_, c0_, tileTaskCount * n_, 1, 1, c0_/8);
PipeBarrier<PIPE_V>();
```

**预期提升**：减少 2 次 PipeBarrier + 2 次 Add 指令开销，约 5-8%

**风险**：需验证 WholeReduceSum 在 srcRepStride=c0_/8=32 时的精度

---

### 优化3：合并 ComputeGradPre 的 N×c0 循环（预期提升 5-10%）

**当前**：N=4 × 4 = 16 次小粒度 Mul

```cpp
for (loopIdN = 0; loopIdN < n_; loopIdN++) {
    for (loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0 += 64) {
        Mul(..., mask, tileTaskCount, {1, 1, 1, blkStride3, blkStride3, blkStride});
    }
}
```

**优化后**：1 次大粒度 Mul，利用 stride 参数

```cpp
// 一次性处理 N*c0 元素, repeat = tileTaskCount
Mul(xCastLocal_, xCastOutLocal, gradYCastLocal_, n_ * c0_, tileTaskCount,
    {n_ * blkStride, blkStride, blkStride, n_ * c0_, n_ * c0_, c0_});
```

**预期提升**：减少 15 次指令发射开销，约 5-10%

**风险**：需验证 stride 参数正确性，特别是 src1RepStride（广播 grad_y）

---

### 优化4：ComputeGradX1 合并 Mul+MulAddDst 循环（预期提升 5-10%）

**当前**：两轮 N×4=16 次循环，中间有 PipeBarrier

```cpp
// 第一轮: 16 次 Mul (x_grad_vec3)
for (loopIdN...) for (loopOffsetC0...) Mul(gradXCastLocal_, gradYCastLocal_, preBrcb_, ...);
PipeBarrier<PIPE_V>();
// 第二轮: 16 次 MulAddDst (x_grad_vec1)
for (loopIdN...) for (loopOffsetC0...) MulAddDst(gradXCastLocal_, xCastLocal_, dRsqrtBrcb_, ...);
```

**优化后**：合并为 1 次大粒度 Mul + 1 次大粒度 MulAddDst

```cpp
Mul(gradXCastLocal_, gradYCastLocal_, preBrcb_, n_ * c0_, tileTaskCount, {...});
PipeBarrier<PIPE_V>();
MulAddDst(gradXCastLocal_, xCastLocal_, dRsqrtBrcb_, n_ * c0_, tileTaskCount, {...});
```

**预期提升**：减少 30 次指令发射开销，约 5-10%

---

### 优化5：h_pre/dRsqrt 广播提到内层循环外（预期提升 2-3%）

**当前**：每个 inner iteration（4次）都重新 Brcb

```cpp
for (taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
    Brcb(preBrcbLocal_, preLocal_, ...);       // 重复4次
    Brcb(dRsqrtBrcbLocal_, gradRsqrtLocal_, ...);  // 重复4次
    ComputeGradX1(...);
}
```

**优化后**：广播提到循环外，只执行1次

```cpp
Brcb(preBrcbLocal_, preLocal_, ...);           // 只执行1次
Brcb(dRsqrtBrcbLocal_, gradRsqrtLocal_, ...);  // 只执行1次
for (taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
    ComputeGradX1(...);
}
```

**预期提升**：减少 6 次 Brcb 调用，约 2-3%

**注意**：此优化已在前次会话中实施并验证通过

---

### 优化6：ComputeScaleBias 删除冗余 WholeReduceSum（预期提升 1-2%）

**当前**：4 次 WholeReduceSum，但第 4 次结果未使用

```cpp
WholeReduceSum(dscaleOut,    dScaleLocal,     4, 1, 1, 3, 8);   // α_pre_grad
WholeReduceSum(dscaleOut[1], dScaleLocal,     4, 1, 1, 3, 8);   // α_post_grad (注意: src偏移有误?)
WholeReduceSum(dscaleOut[2], dScaleLocal[8],  8, 1, 1, 3, 8);   // α_res_grad
WholeReduceSum(dscaleOut[3], dScaleLocal,     8, 1, 1, 3, 8);   // ← 结果未使用
```

**优化后**：删除第 4 次 WholeReduceSum

**预期提升**：减少 1 次 ReduceSum + 1 次 PipeBarrier，约 1-2%

**注意**：此优化已在前次会话中实施并验证通过

---

### 优化7：ComputeGradHHat2 中的 Mul 循环合并（预期提升 3-5%）

**当前**：dhat2Local 的 Mul 按 block 循环

```cpp
for (int32_t blk = 0; blk < totalBlocks; blk++) {
    auto scaleSrc = (blk == 0) ? scaleLocal_ : scaleLocal_[ELEMENTS_SIZE_PER_BLOCK];
    Mul(dhat2Local[blk * 8], scaleSrc, dhatBeforeNormLocal[blk * 8], ...);
}
```

**优化后**：利用 src1RepStride=0 广播，1 次 Mul 完成

```cpp
// block 0: src1 = scaleLocal_ (pre+post)
Mul(dhat2Local, scaleLocal_, dhatBeforeNormLocal, 2 * N, tileTaskCount,
    {totalBlocks, 0, totalBlocks, fusionSize_, 0, fusionSize_});
// block 1+: src1 = scaleLocal_[8] (res)
Mul(dhat2Local_[2*N], scaleLocal_[8], dhatBeforeNormLocal_[2*N], factN, tileTaskCount,
    {totalBlocks-1, 0, totalBlocks-1, fusionSize_, 0, fusionSize_});
```

**预期提升**：BSN! 从 4 次循环减为 2 次，BSNN 从 3 次减为 2 次，约 3-5%

---

### 优化8：matmul K 轴分块（预期提升 5-15%，大 BS 场景）

**当前**：mm2 的 K = B×S，大场景下可能超出 L1 cache

```cpp
mm2_.SetSingleShape(mm2M_, mm2N_, mm2K_);  // mm2K = B*S, 不分块
```

**优化后**：K 轴分块，多次 Iterate

```cpp
int32_t kBlock = 1024;  // 根据 L1 大小调整
for (int32_t k = 0; k < mm2K_; k += kBlock) {
    int32_t curK = min(kBlock, mm2K_ - k);
    mm2_.SetSingleShape(mm2M_, mm2N_, curK);
    mm2_.SetTensorA(gradH2Global_[k * mm2M_], true);
    mm2_.SetTensorB(xWorkspaceGlobal_[k * mm2N_]);
    if (k == 0) {
        mm2_.template IterateAll<false>(output);  // 首次写入
    } else {
        mm2_.template IterateAll<true>(output);   // 后续累加
    }
}
```

**预期提升**：大 BS 场景（B×S > 4096）约 5-15%，小 BS 场景无变化

**风险**：确定性场景下 mm2 写 per-core workspace，K 分块后累加逻辑需调整

---

### 优化9：inputGradQueue 单缓冲改双缓冲（预期提升 2-3%）

**当前**：inputGradQueue 单缓冲

```cpp
pipe_->InitBuffer(inputGradQueue, 1, ...);  // 单缓冲
```

**优化后**：改为双缓冲

```cpp
pipe_->InitBuffer(inputGradQueue, 2, ...);  // 双缓冲
```

**预期提升**：MTE2 搬运与 V 计算流水，约 2-3%

**风险**：UB 空间不足，需评估

---

## 三、优化优先级排序

| 优先级 | 优化项 | 预期提升 | 难度 | 风险 | 状态 |
|--------|--------|----------|------|------|------|
| P0 | 优化1: AIV-AIC 乒乓流水 | 30-40% | 高 | GM 内存翻倍 | 待实施 |
| P1 | 优化3: 合并 GradPre N×c0 循环 | 5-10% | 中 | stride 验证 | 待实施 |
| P1 | 优化4: 合并 GradX1 Mul+MulAddDst | 5-10% | 中 | stride 验证 | 待实施 |
| P2 | 优化2: WholeReduceSum 替代树形归约 | 5-8% | 低 | 精度验证 | 待实施 |
| P2 | 优化8: matmul K 轴分块 | 5-15% | 中 | 确定性适配 | 待实施 |
| P3 | 优化7: dhat2Local Mul 循环合并 | 3-5% | 低 | stride 验证 | 待实施 |
| P3 | 优化5: h_pre/dRsqrt 广播外提 | 2-3% | 低 | 已验证安全 | **已实施** |
| P3 | 优化6: 删除冗余 ReduceSum | 1-2% | 低 | 已验证安全 | **已实施** |
| P4 | 优化9: inputGradQueue 双缓冲 | 2-3% | 低 | UB 不足 | 待评估 |

---

## 四、综合预期

| 场景 | 当前 | 优化后(预期) | 提升幅度 |
|------|------|-------------|----------|
| 小 BS (B=1,S=1) | 基准 | -10~15% | 10-15% |
| 中 BS (B=4,S=4) | 基准 | -35~50% | 35-50% |
| 大 BS (B=8,S=512) | 基准 | -40~55% | 40-55% |

**关键说明**：
- 优化1（乒乓流水）是最大收益项，但实现难度最高
- 优化2-4 可作为第一批快速实施，风险可控
- 优化8 仅对大 BS 场景有效
- 所有优化需在 **精度回归通过** 后才能合入

---

## 五、PipeBarrier 优化说明

A2 架构同一 PIPE_V 内的 Vector 指令 RAW 依赖**不能**被硬件自动串行化，必须保留显式 PipeBarrier。

**不能删除的 PipeBarrier**：
- 同一 buffer 上的读后写（RAW）依赖
- Add/Mul 链式计算之间

**可以优化的 PipeBarrier**：
- 通过合并循环减少指令数，间接减少 PipeBarrier 数量
- 通过 WholeReduceSum 替代手动归约，减少 PipeBarrier 次数
