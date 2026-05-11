# GMMSQ 与 GMMFR 差异分析

> 本文档基于 GMMFR（GroupedMatmulFinalizeRouting）MxA8W4 数据流的完整实现，逐层分析 GMMSQ（GroupedMatmulSwigluQuantV2）MxA8W4 数据流与其的差异点，为 GMMSQ 的改写实现提供依据。

---

## 1. 功能差异

| 维度 | GMMFR | GMMSQ |
|------|-------|-------|
| **核心计算** | GMM（分组矩阵乘） | GMM（分组矩阵乘） |
| **后处理** | Routing（Scatter-Add 到全局输出） | SwiGLU 激活 + MX 量化输出 |
| **输出语义** | 多 Expert 输出按 rowIndex 合并到同一张输出矩阵 | 每个 Expert 独立输出 SwiGLU 后的量化结果 |
| **输出维度** | `(M, N)` — 完整 N 输出 | `(M, N/2)` — N 分两半后输出一半 |

**GMMFR 后处理流程**：
```
GMM 结果 (FP32) → MulLogits (y *= logits * 64.0) → RoutingYToGm (AtomicAdd Scatter)
```

**GMMSQ 后处理流程**：
```
GMM 结果 (FP32) → SwiGLU (Swish(前N/2) * Gate(后N/2)) → MX Quant (shared_exp + quantize_to_fp8)
```

---

## 2. 数据类型差异

### 2.1 输入数据类型

| 参数 | GMMFR | GMMSQ | 差异 |
|------|-------|-------|------|
| `x`（激活） | `fp8_e4m3fn_t` | `fp8_e4m3fn_t` | **相同** |
| `weight`（权重） | `fp4x2_e2m1_t` | `fp4_e2m1` | **相同**（4bit 打包存储） |
| `weightScale` | `fp8_e8m0_t` | `fp8_e8m0_t` | **相同** |
| `xScale`（per-token scale） | `fp8_e8m0_t` | `fp8_e8m0_t` | **相同** |
| `bias` | `bfloat16_t` | 不使用（MxA8W4 场景为 null） | GMMSQ 无 Bias |
| `groupList` | `int64_t` | `int64_t` | **相同** |

### 2.2 输出数据类型

| 参数 | GMMFR | GMMSQ |
|------|-------|-------|
| `y`（主输出） | `float`（FP32） | `fp8_e4m3fn_t`（FP8） |
| `yScale`（量化 scale） | 无 | `fp8_e8m0_t`（E8M0） |
| 输出 shape | `(M, N)` | `(M, N/2)` |

### 2.3 GMMFR 独有参数（GMMSQ 不需要）

| 参数 | 类型 | 用途 |
|------|------|------|
| `logit` | `float` | MoE 专家对每个 token 的 logit 权重 |
| `rowIndex` | `int64_t` | MoE Routing 的 scatter 目标索引 |
| `sharedInput` | `bfloat16_t` | MoE Shared Expert 的输出（需 combine） |

### 2.4 GMMSQ 独有参数（GMMFR 不需要）

| 参数 | 类型 | 取值 | 用途 |
|------|------|------|------|
| `dequantMode` | `int64` | `{2}` | 反量化模式（MX 量化） |
| `dequantDtype` | `int64` | `{0}` | 中间计算类型（DT_FLOAT） |
| `quantMode` | `int64` | `{2}` | 量化模式（MX 量化） |

---

## 3. N 轴处理差异（核心差异）

### 3.1 GMMFR：完整 N 输出

```
GMM: X[M, K] @ W[N, K].T = Y[M, N]
输出: Y 的完整 N 列全部保留，直接做 Routing
```

### 3.2 GMMSQ：N 分两半，输出 N/2

```
GMM: X[M, K] @ W[N, K].T = Y[M, N]
SwiGLU 划分:
  - Y[:, 0:N/2]   → Swish 分支: Swish(Y_act)
  - Y[:, N/2:N]   → Gate 分支: gate
  - S = Swish(Y_act) * gate
输出: S[M, N/2] → MX Quant → output[M, N/2] + outputScale[M, ceil(N/2/64), 2]
```

**关键约束**：
- N 必须 ≥ 128 且 128 对齐（因为 N/2 需要 64 对齐）
- 输出维度从 N 缩减为 N/2

### 3.3 Tiling 层面的 N 轴切分策略

GMMFR 的 N 轴切分：每个 BasicBlock 处理 `nL1Size=256` 的完整 N 分块。

GMMSQ 的 N 轴切分策略（参考 `gmmsq_kernel_impl.md`）：
- 每个 BasicBlock 取 `baseN=128`，构成 256 的基本块（128 Swish + 128 Gate）
- 这样保证 GMM 性能的同时，SwiGLU 计算时 Swish 和 Gate 结果都在同一个 BasicBlock 内
- 尾块按 N/2 计算尾块

```
N 轴切分示意（N=512）:
┌──────────────┬────────────────┬───────────────┬──────────────┐
│ Swish[0:128] │ Swish[128:256] │ Gate[256:384] │ Gate[384:512]│
│   BB0 处理    │   BB1 处理     │   BB0 处理     │   BB1 处理   │
│  输出[0:128]  │ 输出[128:256]  │               │              │
└──────────────┴────────────────┴───────────────┴──────────────┘
```

---

## 4. 后处理差异

### 4.1 GMMFR 后处理（PhaseB）

```
PhaseB（AIV 侧，处理上一个 BasicBlock 的 MMA 结果）:
  1. CopyRowIndexLogitsGmToUb()  — MTE2 搬运 rowIndex + logits 到 UB
  2. SetAivToAic(FLAG4)          — 通知 AIC 可以 Fixpipe
  3. WaitAicToAiv(FLAG3)         — 等待 AIC Fixpipe 完成
  4. MulLogits()                 — VF: y *= logits * 64.0
  5. RoutingYToGm()              — MTE3: AtomicAdd Scatter 写回 GM
```

### 4.2 GMMSQ 后处理（PhaseB）

```
PhaseB（AIV 侧，处理上一个 BasicBlock 的 MMA 结果）:
  1. SetAivToAic(FLAG4)          — 通知 AIC 可以 Fixpipe
  2. WaitAicToAiv(FLAG3)         — 等待 AIC Fixpipe 完成（结果在 yF32Buffer_）
  3. SwiGLU 计算                 — VF: Swish(yF32[:, 0:N/2]) * yF32[:, N/2:N]
  4. ComputeMaxExp()             — VF: 计算每 32 个元素的 max exponent
  5. ComputeScale()              — VF: 计算 shared_exp → E8M0 scale
  6. ComputeDataForQuantTargetFp8() — VF: 量化到 FP8_E4M3FN
  7. MTE3: 写出 yFp8 到 GM       — CopyOutputFromUb2Gm
  8. MTE3: 写出 scaleE8M0 到 GM  — CopyScaleFromUb2Gm
```

### 4.3 后处理差异对比

| 步骤 | GMMFR | GMMSQ |
|------|-------|-------|
| 前置搬运 | CopyRowIndexLogitsGmToUb（搬运 rowIndex + logits） | 无（不需要额外搬运） |
| VF 计算 1 | MulLogits（y *= logits * 64.0） | SwiGLU（Swish(前N/2) * Gate(后N/2)） |
| VF 计算 2 | 无 | ComputeMaxExp（max exponent） |
| VF 计算 3 | 无 | ComputeScale（shared_exp → E8M0） |
| VF 计算 4 | 无 | ComputeDataForQuantTargetFp8（量化到 FP8） |
| MTE3 写出 1 | RoutingYToGm（AtomicAdd Scatter） | CopyOutputFromUb2Gm（连续写出 yFp8） |
| MTE3 写出 2 | 无 | CopyScaleFromUb2Gm（写出 scaleE8M0） |

---

## 5. Buffer 分配差异

### 5.1 UB Buffer 对比

| Buffer | GMMFR 大小 | GMMSQ 大小 | 说明 |
|--------|-----------|-----------|------|
| `weightLowBit_` | 80KB | 64KB | GMMSQ 减少（ubMte2BufferNum 调整） |
| `weightHighBit_` | 160KB（4-buffer） | 128KB（4-buffer） | GMMSQ 减少（单 buffer 从 40KB→32KB） |
| `bias_` | 1KB | 1KB | 相同 |
| `biasRescale_` | 1KB | 1KB | 相同 |
| `rowIndex_` | 2KB（2-buffer） | **无** | GMMSQ 不需要 |
| `logits_` | 1KB（2-buffer） | **无** | GMMSQ 不需要 |
| `yF32Buffer_` | 复用 weightHighBit 空间 | 128KB | **新增**：MMA Fixpipe 输出（复用 weightHighBit 空间） |
| `maxB16Buffer_` | **无** | 2KB | **新增**：ComputeMaxExp 的中间结果 |
| `scaleE8M0Buffer_` | **无** | 16KB | **新增**：MX 量化 scale 输出 |
| `yFp8Buffer_` | **无** | 32KB | **新增**：量化后 FP8 输出 |

**GMMFR UB 总使用量**：80 + 160 + 1 + 1 + 2 + 1 = **245KB**（UB_AVAILABLE = 248KB）

**GMMSQ UB 总使用量**：64 + 128 + 1 + 1 + 128 + 2 + 16 + 32 = **372KB** → **超出 UB 限制！**

> ⚠️ **关键风险**：GMMSQ 的 UB 空间需求远超 248KB。需要复用策略：
> - `yF32Buffer_` 复用 `weightHighBit_` 的空间（两者不同时活跃）
> - `yFp8Buffer_` 复用 `weightLowBit_` 的空间（PhaseB 阶段 weightLowBit 已不再使用）
> - 实际 UB 使用量：64 + 128 + 1 + 1 + 2 + 16 = **212KB**（复用后）

### 5.2 L1 Buffer 对比

GMMFR 和 GMMSQ 的 L1 Buffer 分配**完全相同**（Weight L1、Bias L1、ScaleA L1、ScaleB L1、A L1），因为 MMA 计算阶段的数据流一致。

### 5.3 L0 Buffer 对比

**完全相同**（L0A 64KB、L0B 64KB、L0C 256KB、BiasTable 4KB）。

---

## 6. Tiling 差异

### 6.1 TilingData 结构体对比

| 字段 | GMMFR | GMMSQ | 说明 |
|------|-------|-------|------|
| `groupListType` | ✅ | ✅ | 相同 |
| `hasBias` | ✅ | ✅ | 相同 |
| `groupNum` | ✅ | ✅ | 相同 |
| `coreNum` | ✅ | ✅ | 相同 |
| `outputBs` | ✅ | ✅ | 相同 |
| `sharedInputOffset` | ✅ | ❌ | GMMSQ 无 sharedInput |
| `sharedInputLen` | ✅ | ❌ | GMMSQ 无 sharedInput |
| `sharedInputWeight` | ✅ | ❌ | GMMSQ 无 sharedInput |
| `groupSize` | ✅ | ✅ | 相同（MX_GROUPSIZE=32） |
| `kSize` | ✅ | ✅ | 相同 |
| `nSize` | ✅ | ✅ | 相同（但语义上 GMMSQ 输出 N/2） |
| `initSize` | ✅ | ❌ | GMMSQ 不需要 GM 清零（无 AtomicAdd） |

### 6.2 Tiling Key 对比

| 参数 | GMMFR | GMMSQ |
|------|-------|-------|
| `ATRANS` | 0（固定） | 0（固定） |
| `BTRANS` | 0, 1 | 0, 1 |
| `HASBIAS` | 0, 1 | 0（MxA8W4 场景无 Bias） |

GMMSQ MxA8W4 场景的 Tiling Key 组合更少：`ATRANS=0 × BTRANS={0,1} × HASBIAS=0` = 2 种场景。

### 6.3 BasicBlockOffsetParam 对比

| 字段 | GMMFR | GMMSQ | 说明 |
|------|-------|-------|------|
| `mL1Size` | ✅ | ✅ | 相同 |
| `kaL1Size` | ✅ | ✅ | 相同 |
| `kbL1Size` | ✅ | ✅ | 相同 |
| `nL1Size` | ✅ | ✅ | 相同（但语义变为 N/2 分块） |
| `mOffset` | ✅ | ✅ | 相同 |
| `nOffset` | ✅ | ✅ | 相同 |
| `mSize` | ✅ | ✅ | 相同 |
| `kSize` | ✅ | ✅ | 相同 |
| `nSize` | ✅ | ✅ | 相同 |
| `kAlign` | ✅ | ✅ | 相同 |
| `nAlign` | ✅ | ✅ | 相同 |
| `yGmAddr` | ✅ | ✅ | 相同（但指向 FP8 输出） |
| `yScaleGmAddr` | ❌ | ✅ | **新增**：输出 scale 的 GM 地址 |

---

## 7. API 参数差异

### 7.1 Init() 参数对比

**GMMFR Init()**：
```cpp
controller.Init(x, w, scale, bias, group_list, pertoken_scale, logit, row_index, y, share_input, tiling);
```

**GMMSQ Init()**（预期）：
```cpp
controller.Init(x, w, scale, bias, group_list, pertoken_scale, y, yScale, tiling);
// 无 logit, row_index, share_input
// 新增 yScale（输出量化 scale 的 GM 地址）
```

### 7.2 模板参数对比

| 模板参数 | GMMFR | GMMSQ |
|----------|-------|-------|
| `xType` | `DTYPE_X` | `DTYPE_X` |
| `wType` | `DTYPE_W` | `DTYPE_W` |
| `antiQuantScaleType` | `DTYPE_SCALE` | `DTYPE_SCALE` |
| `scaleType` | `DTYPE_SCALE` | 不需要 |
| `perTokenScaleType` | `DTYPE_PERTOKEN_SCALE` | `DTYPE_PERTOKEN_SCALE` |
| `biasType` | `DTYPE_BIAS` | `DTYPE_BIAS` |
| `yType` | `DTYPE_Y`（float） | `DTYPE_Y`（fp8_e4m3fn_t） |
| `sharedInputDType` | `bfloat16_t` | **不需要** |
| `logitsType` | `float` | **不需要** |
| `rowIndexType` | `int64_t` | **不需要** |
| `yScaleType` | 无 | **新增**：`fp8_e8m0_t` |
| `hasBias` | `HASBIAS` | `HASBIAS`（MxA8W4 固定 false） |
| `wqmmConfig` | `MXA8W4_NZNK` | `MXA8W4_NZNK`（相同） |
| `vecConfig` | `VEC_ANTIQUANT_CONFIG_DYNAMIC` | 可能需要调整 |

---

## 8. Controller 层改写点

### 8.1 Process() 主循环

| 步骤 | GMMFR | GMMSQ 变化 |
|------|-------|-----------|
| `GetBlockIdx()` | AIC/AIV 配对映射 | **不变** |
| `InitOffsetParam()` | 初始化 offsetParam | **不变** |
| 缓存对齐检查 | `isCacheLineUnaligned` | **不变** |
| `InitAtomicGm()` | GM 清零 + sharedInput 拷贝 | **去除**（GMMSQ 不需要 GM 清零和 sharedInput） |
| `SyncAll()` | 所有核同步 | **不变** |
| Group 遍历 | `GetSplitValueFromGroupList()` | **不变** |
| M 轴分块 | `CeilDivide(mSize, M_L1)` | **不变** |
| `UpdateGlobalAddr()` | 更新 GM 地址 | **变化**（见 8.2） |
| `SplitNByMultiCore()` | N 轴多核分块 | **变化**：N 轴切分逻辑需适配 N/2 输出 |
| `basicBlock_.End()` | 收尾处理 | **变化**（去除 Routing 收尾） |

### 8.2 UpdateGmAddr() 变化

**GMMFR**：
```cpp
xGm_ += mSize * kSize;
weightGm_ += (nSize * kSize) >> 1;  // 4bit 权重
antiquantScaleGm_ += nSize * CeilDivide(kAlignSize, groupSize);
perTokenScaleGm_ += mSize * CeilDivide(kAlignSize, groupSize);
biasGm_ += nSize;
yGm_ += mSize * nSize;              // 完整 N 输出
logitsAddr_ += mSize;               // GMMSQ 去除
rowIndexAddr_ += mSize;             // GMMSQ 去除
```

**GMMSQ**：
```cpp
xGm_ += mSize * kSize;
weightGm_ += (nSize * kSize) >> 1;
antiquantScaleGm_ += nSize * CeilDivide(kAlignSize, groupSize);
perTokenScaleGm_ += mSize * CeilDivide(kAlignSize, groupSize);
biasGm_ += nSize;
yGm_ += mSize * (nSize / 2);        // 输出 N/2
yScaleGm_ += mSize * CeilDivide(CeilAlign(nSize/2, 64), groupSize);  // 新增
// 去除: logitsAddr_, rowIndexAddr_
```

### 8.3 SplitNByMultiCore() 变化

N 轴切分逻辑需要适配：
- GMMFR：`nL1Size = 256`，每个 BasicBlock 处理 256 个 N 元素
- GMMSQ：`nL1Size = 256`，但其中前 128 为 Swish 分支，后 128 为 Gate 分支，输出 128 个元素

---

## 9. BasicBlock 层改写点

### 9.1 Init() 变化

| 步骤 | GMMFR | GMMSQ 变化 |
|------|-------|-----------|
| L1 Buffer 分配 | Weight L1 + Bias L1 + CubeCompute L1 | **不变** |
| UB Buffer 分配 | weightLowBit + weightHighBit + bias + logits + rowIndex | **变化**：去除 logits/rowIndex，新增 yF32/maxB16/scaleE8M0/yFp8 |
| AIC 初始化 | `cubeCompute_.MxA8W4Init()` | **不变** |
| AIV 初始化 | `vecCompute_.Init(sharedInputWeight, y)` | **变化**：去除 sharedInputWeight，新增 yScale 地址 |
| 同步 | `SetAicToAiv(SYNC_AIC_AIV_FLAG)` | **不变** |

### 9.2 IterateNzNkWithAiv() 变化（AIV 路径）

**PhaseB 从 Routing 改为 SwiGLU+Quant**：

```
GMMFR PhaseB:
  CopyRowIndexLogitsGmToUb()  →  MulLogits()  →  RoutingYToGm()

GMMSQ PhaseB:
  (无前置搬运)  →  SwiGLU()  →  ComputeMaxExp()  →  ComputeScale()  →  ComputeDataForQuantTargetFp8()
                   →  CopyOutputFromUb2Gm()  →  CopyScaleFromUb2Gm()
```

**Pass1/Pass2（Weight 反量化）**：**不变**（反量化流程完全相同）。

### 9.3 IterateNzNkWithKAic() 变化（AIC 路径）

| 步骤 | GMMFR | GMMSQ 变化 |
|------|-------|-----------|
| PhaseB fixpipe | `GetTensorC(ubOutputF32Buffer_, lastParam)` | **变化**：输出到 `yF32Buffer_`（地址不同） |
| K 循环 MMA | `LaunchMatmul()` | **不变** |
| 同步信号 | FLAG8/FLAG9/FLAG4/FLAG3 | **不变** |

### 9.4 End() 变化

**GMMFR End()**：
- AIC：最后一次 fixpipe + EndSync
- AIV：最后一次 MulLogits + RoutingYToGm + 等待 AIC 完成

**GMMSQ End()**：
- AIC：最后一次 fixpipe + EndSync（**不变**）
- AIV：最后一次 SwiGLU + Quant + 写出 + 等待 AIC 完成（**后处理逻辑变化**）

### 9.5 同步信号

| 信号 | GMMFR | GMMSQ | 说明 |
|------|-------|-------|------|
| `SYNC_AIC_AIV_FLAG (9)` | AIC→AIV：A 矩阵就绪 | **不变** | |
| `SYNC_AIV_AIC_FLAG (8)` | AIV→AIC：Weight 反量化完成 | **不变** | |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG (4)` | AIV→AIC：可以 Fixpipe | **不变** | |
| `SYNC_AIC_FIX_AIV_VF_FLAG (3)` | AIC→AIV：Fixpipe 完成 | **不变** | |

同步信号数量和语义**完全不变**，因为 AIC↔AIV 的流水线模式保持一致。

---

## 10. VecCompute 层改写点

### 10.1 去除的功能

| 函数 | 说明 |
|------|------|
| `MulLogits()` | 乘 logits（`y *= logits * 64.0`） |
| `RoutingYToGm()` | AtomicAdd Scatter 写回 GM |
| `CopyRowIndexLogitsGmToUb()` | 搬运 rowIndex + logits 到 UB |
| `CopyShareInputGmToUb()` | 搬运 sharedInput 到 UB |
| `CopyShareInputUbToGm()` | 写出 sharedInput 到 GM |
| `CastAndMulWithSharedWeightVf()` | sharedInput 的 VF 处理 |
| `InitGmToZero()` | GM 清零 |

### 10.2 新增的功能

| 函数 | 说明 |
|------|------|
| `SwiGLU()` | Swish(前 N/2) * Gate(后 N/2) 的 VF 实现 |
| `ComputeMaxExp()` | 计算每 32 个元素的 max exponent（用于 MX 量化） |
| `ComputeScale()` | 根据 maxExp 计算 shared_exp → E8M0 scale |
| `ComputeDataForQuantTargetFp8()` | 将 FP32 数据量化到 FP8_E4M3FN |
| `CopyOutputFromUb2Gm()` | MTE3 写出 yFp8 到 GM |
| `CopyScaleFromUb2Gm()` | MTE3 写出 scaleE8M0 到 GM |

### 10.3 保留的功能

| 函数 | 说明 |
|------|------|
| `CopyGmToUb()` | MTE2 搬运 weight + bias 到 UB |
| `WeightAntiQuantComputeNzNk()` | Weight 反量化（4bit→8bit） |
| `AntiQuantProcessNzMxA8W4()` | 反量化参数准备 |
| `FrAntiQuantMxA8W4NzNkVf()` | VF 反量化指令 |
| `CopyWeightHighBitForAligned()` | UB→L1 搬运反量化后的 weight |
| `FrMxA8W4BiasCompute()` | Bias 反量化（如有） |

### 10.4 UB Buffer 重新分配

```
GMMFR UB 布局:
[weightLowBit 80KB][weightHighBit 160KB][bias 1KB][biasRescale 1KB][rowIndex 2KB][logits 1KB]

GMMSQ UB 布局（复用策略）:
[weightLowBit 64KB][weightHighBit 128KB][bias 1KB][biasRescale 1KB]
                                                      ↓ yF32Buffer_ 复用此空间
[maxB16 2KB][scaleE8M0 16KB][yFp8 32KB]              ← PhaseB 阶段使用
                                                      ↓ yFp8 复用 weightLowBit 空间
```

---

## 11. CubeCompute 层改写点

### 11.1 变化点

| 函数 | GMMFR | GMMSQ 变化 |
|------|-------|-----------|
| `GetTensorC()` | Fixpipe 输出到 `ubOutputF32Buffer_` | Fixpipe 输出到 `yF32Buffer_`（地址不同，语义相同） |
| `MxA8W4Init()` | L1 Buffer 分配 | **不变** |
| `UpdateGlobalAddr()` | 更新 GM 地址 | **不变**（Scale A/B 地址更新逻辑相同） |
| `CopyMxScaleGmToL1()` | Scale 搬运 | **不变** |
| `CopyAAndBiasGmToL1()` | A 矩阵搬运 | **不变** |
| `LaunchMatmul()` | MMA 计算 | **不变** |
| `LoadAAndScaleL1ToL0()` | L1→L0A | **不变** |
| `LoadBAndScaleL1ToL0()` | L1→L0B | **不变** |
| `DoMmadCompute()` | MMAD 指令 | **不变** |
| `FixL0CToDst()` | Fixpipe | **不变** |

**结论**：CubeCompute 层几乎不需要改动，只需调整 `GetTensorC()` 的目标 UB 地址。

---

## 12. 差异汇总表

| 层级 | 组件 | 差异程度 | 说明 |
|------|------|---------|------|
| **Tiling** | TilingData | 🟡 中等 | 去除 sharedInput 相关字段，新增 yScale 相关 |
| **Tiling** | Tiling Key | 🟢 小 | 减少 HASBIAS 组合（MxA8W4 固定无 Bias） |
| **Tiling** | 场景识别 | 🟡 中等 | 新增 MxA8W4 场景拦截逻辑 |
| **Kernel 入口** | apt.cpp | 🟡 中等 | 新增 V310_GMM_SQ_ANTI_QUANT 宏分支 |
| **Controller** | Process() | 🟡 中等 | 去除 InitAtomicGm，调整 N 轴切分 |
| **Controller** | UpdateGmAddr() | 🟡 中等 | 去除 logits/rowIndex，新增 yScale，输出 N/2 |
| **Controller** | Init() | 🟡 中等 | 去除 sharedInput/logits/rowIndex 参数 |
| **BasicBlock** | Init() | 🟡 中等 | UB Buffer 重新分配 |
| **BasicBlock** | IterateNzNkWithAiv() | 🔴 大 | PhaseB 从 Routing 改为 SwiGLU+Quant |
| **BasicBlock** | IterateNzNkWithKAic() | 🟢 小 | GetTensorC 目标地址变化 |
| **BasicBlock** | End() | 🟡 中等 | 后处理逻辑变化 |
| **VecCompute** | 后处理函数 | 🔴 大 | 去除 MulLogits/RoutingYToGm，新增 SwiGLU/Quant |
| **VecCompute** | 反量化函数 | 🟢 小 | 保持不变 |
| **VecCompute** | UB Buffer | 🔴 大 | 重新分配，新增 4 个 Buffer |
| **CubeCompute** | 全部 | 🟢 小 | 仅 GetTensorC 目标地址变化 |
| **Common** | Config | 🟢 小 | 可能需要新增 GMMSQ 专用配置常量 |

---

## 13. 关键风险点

1. **UB 空间不足**：GMMSQ 新增 yF32、maxB16、scaleE8M0、yFp8 四个 Buffer，需要精细的复用策略才能控制在 248KB 以内
2. **N/2 对齐要求**：N 必须 128 对齐（因为 N/2 需要 64 对齐），且 N ≥ 128
3. **SwiGLU 和 Quant 的 VF 实现**：这两个函数由其他团队实现，GMMSQ 侧需要预留接口
4. **性能瓶颈**：MX Quant 的 scatter 写出粒度较小（N/2 方向每 128 个元素一个 core），搬运 pattern 可能性能较差
5. **尾块处理**：N 轴尾块需要按 N/2 计算尾块，逻辑比 GMMFR 更复杂