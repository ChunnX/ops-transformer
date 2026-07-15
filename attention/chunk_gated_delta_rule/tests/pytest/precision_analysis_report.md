# chunk_gated_delta_rule 精度分析报告

## 1. 问题概述

`bash test_run.sh single` 执行精度测试，2 个用例均失败。错误为 `compare o failed` + `compare state failed`。

测试参数：
- Testcase0: B=1, seqlen=64, nk=4, nv=4, dk=128, dv=128, chunk_size=64, bfloat16, has_g=True
- Testcase1: B=1, seqlen=16384, nk=4, nv=4, dk=128, dv=128, chunk_size=64, bfloat16, has_g=True

## 2. 根因分析

经逐阶段对比分析，精度失败由以下 **三个独立原因** 叠加导致：

### 2.1 原因一：`check_result` 阈值匹配 Bug（测试代码问题）

**文件**: `chunk_gated_delta_rule_main.py:473-475`

```python
data_type_str = str(data_type)  # str(torch.bfloat16) = 'torch.bfloat16'
_, o_result = check_result(o_golden.to(torch.float32), o_npu.to(torch.float32), data_type_str)
```

`check_result` 函数内部判断（line 180）：
```python
if data_type == 'bfloat16':   # 'torch.bfloat16' != 'bfloat16' → False!
    rtol = 0.0078125  # bfloat16 阈值
    atol = 0.0001
else:
    rtol = 0.005      # 误用更严格的 float32 阈值
    atol = 0.000025
```

**影响**：本应使用 bfloat16 宽松阈值（rtol=0.0078125），实际误用 float32 严格阈值（rtol=0.005），导致更多元素判定为不通过。

**实测数据**（Testcase0 output, f32 vs NPU）：
| 阈值 | 不通过率 | 通过率 | 是否达标(≥99.5%) |
|------|---------|--------|-----------------|
| bfloat16 (rtol=0.0078125) | 11.30% | 88.70% | 否 |
| 误用 float32 (rtol=0.005) | 17.86% | 82.14% | 否 |

> 注：即使修复此 Bug 使用正确 bfloat16 阈值，通过率仍不达标。此 Bug 加剧了失败程度但非唯一原因。

### 2.2 原因二：bfloat16 matmul 固有精度损失（预期误差）

NPU kernel 三个 Stage 的 matmul 类型定义全部为 bfloat16 输入/输出：
- Stage1: `chunk_gated_delta_rule_stage1.h:26-28` — `aT1/bT1/cT1 = MatmulType<..., bfloat16_t>`
- Stage2: `chunk_gated_delta_rule_stage2.h:26-28` — `aT2/bT2/cT2 = MatmulType<..., bfloat16_t>`
- Stage3: `chunk_gated_delta_rule_stage3.h:26-28` — `aT3/bT3/cT3 = MatmulType<..., bfloat16_t>`

而 torch 参考实现全程使用 float32。

**逐阶段对比**（f32 vs bfloat16 模拟，即同一算法仅 matmul 精度不同）：

| 中间变量 | 计算方式 | K维 | max_abs_diff | max_rel_diff |
|---------|---------|-----|-------------|-------------|
| kkt | k_beta @ k.T | 128 | 0.00354 | 0.00471 |
| attn_inv | 逐行求逆 | - | 0.00227 | 158.9* |
| v_inner | attn @ v_beta | 64 | 0.00463 | 223.6* |
| k_cumdecay | attn @ (k_beta * g.exp()) | 64 | 0.00045 | 13.8* |
| v_prime | k_cumdecay @ state | 128 | 0.01030 | 118.3* |
| v_new | v_inner - v_prime | - | 0.01199 | 223.6* |
| **final_out** | 全部组合 | - | **0.00231** | **0.0723** |
| **final_state** | state 更新 | - | **0.00090** | **4.830** |

> *max_rel_diff 较大是因为部分元素接近 0，相对误差被放大。

**结论**：bfloat16 matmul 精度损失导致的最大绝对误差约 0.0023，最大相对误差约 7.2%。这是 bfloat16 格式的固有限制（K=128 matmul 的理论误差约 sqrt(128) * 2^-8 ≈ 5.5%）。

### 2.3 原因三：NPU 算法实现与 torch 参考的数值差异（主要误差源，逐行定位）

**这是最大的误差来源。** 通过 bfloat16 模拟（与 torch 参考算法完全一致，仅 matmul 量化为 bfloat16）与 NPU 实际输出对比：

| 对比项 | max_abs | mean_abs | 说明 |
|--------|---------|----------|------|
| f32 vs bf16_sim | 0.00231 | 0.000088 | 纯 bfloat16 精度损失 |
| f32 vs NPU | 0.06334 | 0.000667 | bfloat16 损失 + 算法差异 |
| **bf16_sim vs NPU** | **0.06348** | **0.000659** | **纯算法差异** |

NPU 与 bfloat16 模拟的差异（0.063）远大于 bfloat16 固有精度损失（0.002），说明 NPU 实现与 torch 参考算法存在数值不等价。以下逐行定位每个差异点：

---

#### 差异点 1：attn 求逆矩阵以 bfloat16 存储（影响最大）

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:529-532`（`InverseCompute` 函数）

```cpp
// line 528: 对角块在 AIV 上用 float32 逐行前代法求逆（InverseAIV, line 535-579）
InverseAIV(subOffset_, INVERSE_SHAPE);

// line 529-530: float32 求逆结果 Cast 为 bfloat16
auto tmp = outQueue_.AllocTensor<bfloat16_t>();
Cast(tmp, inverseRes_, RoundMode::CAST_RINT, curVecLen);  // ← float32 → bfloat16

// line 531-532: 以 bfloat16 写回 GM (attnWsGm_)
outQueue_.EnQue(tmp);
DataCopyOutBf16(halfChunkSize_, chunkSize_, chunkSize_, dst[subOffset_ * chunkSize_]);
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:146-151`，全程 float32，求逆结果不量化

```python
attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
attn = attn + torch.eye(chunk_size, ...)  # 全程 float32
```

**影响**：attn 矩阵是 Stage1 的核心中间结果，后续被 3 次 matmul 使用（v_inner、k_cumdecay、分块组合）。float32→bfloat16 量化引入约 0.002 的绝对误差，经 3 次 matmul 放大后影响显著。

---

#### 差异点 2：分块求逆的非对角块用 bfloat16 matmul 组合

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:792-806`（`AttnInverseMMCompute` 函数）

```cpp
// 64×64 矩阵分为 4 个 32×32 块
// line 797: 左下角 @ 左上角 → 左下角（bfloat16 matmul）
InverseAICProcess(attnWsGm_[leftDown], attnWsGm_[offset], attnWsGm_[leftDown]);

// line 802: 右下角 @ 左下角 → 左下角（bfloat16 matmul）
InverseAICProcess(attnWsGm_[rightDown], attnWsGm_[leftDown], attnWsGm_[leftDown]);
```

`InverseAICProcess`（line 819-828）使用 `StageOneMT`（bfloat16 matmul，定义于 line 26-29）：

```cpp
__aicore__ inline void InverseAICProcess(GlobalTensor<bfloat16_t> x,
                                         GlobalTensor<bfloat16_t> y,
                                         GlobalTensor<bfloat16_t> z)
{
    mm.SetOrgShape(chunkSize_, chunkSize_, chunkSize_);
    mm.SetSingleShape(INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE);  // 32×32×32
    mm.SetTensorA(x);  // bfloat16
    mm.SetTensorB(y);  // bfloat16
    mm.IterateAll(z);  // bfloat16 输出
    mm.End();
}
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:147-150`，对完整 64×64 矩阵在 float32 下逐行求逆，不分块。

**影响**：torch 是 64×64 整体求逆（O(n²) 逐行前代法），NPU 是 4 个 32×32 分块求逆 + 2 次 bfloat16 matmul 组合。分块策略本身数学等价，但 bfloat16 matmul 组合引入额外舍入误差。

---

#### 差异点 3：kkt（k_beta @ k.T）以 bfloat16 存储

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:300-302`（`ParaChunkAIC` 函数）

```cpp
// line 300-302: key @ key.T → bfloat16 kkWsGm_
AICProcess(keyConGm_[i * ckOffset_], keyConGm_[i * ckOffset_], kkWsGm_[i * ccOffset_],
           {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_}, true);
```

`AICProcess`（line 808-817）使用 bfloat16 matmul，K=dk=128。

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:146`，`k_beta @ key.transpose(-1, -2)` 在 float32 下计算。

**影响**：K=128 的 bfloat16 matmul 理论相对误差约 sqrt(128)×2⁻⁸ ≈ 5.5%，kkt 的误差直接传递给求逆步骤。

---

#### 差异点 4：v_inner 和 k_cumdecay 以 bfloat16 存储

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:322-331`（`ParaChunkAIC` 函数）

```cpp
// line 322-324: attn @ gBKWs → bfloat16 outKCumdecayGm_（K=chunkSize=64）
AICProcess(attnWsGm_[i * ccOffset_], gBKWsGm_[i * ckOffset_],
           outKCumdecayGm_[chunkRowBase_[i] * dk_],
           {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_});

// line 329-331: attn @ vBetaWs → bfloat16 outVInnerGm_（K=chunkSize=64）
AICProcess(attnWsGm_[i * ccOffset_], vBetaWsGm_[i * cvOffset_],
           outVInnerGm_[chunkRowBase_[i] * dv_],
           {chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_});
```

两个 matmul 的输入 attnWsGm_ 已经是 bfloat16（差异点 1），输出也存为 bfloat16。

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:152-153`，float32 计算，结果保持 float32。

```python
value = attn @ v_beta                    # float32
k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))  # float32
```

**影响**：v_inner 和 k_cumdecay 是 Stage1→Stage2 的关键传递数据，bfloat16 存储导致精度损失在 Stage2 中进一步放大。

---

#### 差异点 5：v_beta 和 gBK（k_cumdecay 中间量）以 bfloat16 存储

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:581-635`（`GBKCompute` 函数）

```cpp
// line 608-612: k_cumdecay 中间量 Cast 为 bfloat16 写入 gBKWsGm_
gBKLocal_ = outQueue_.AllocTensor<bfloat16_t>();
Cast(gBKLocal_, gBKUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);  // f32→bf16
outQueue_.EnQue<bfloat16_t>(gBKLocal_);
DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, gBKWsGm[gBKBeginOffset]);
```

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:658-661`（`VBetaCompute` 函数）

```cpp
// line 658-661: v_beta Cast 为 bfloat16 写入 vBetaWsGm_
vBetaLocal_ = outQueue_.AllocTensor<bfloat16_t>();
Cast(vBetaLocal_, vBetaUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dvAligned_);  // f32→bf16
outQueue_.EnQue(vBetaLocal_);
DataCopyOutBf16(halfChunkSize_, dv_, dvAligned_, vBetaWsGm[subOffset_ * dv_]);
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:134-135`，float32 计算后直接用于后续 matmul，不量化。

**影响**：这两个量作为 Stage1 AIC matmul（差异点 4）的输入，bfloat16 量化后再做 bfloat16 matmul，误差双重叠加。

---

#### 差异点 6：q_prime 和 kg 以 bfloat16 存储

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:689-692`（`QPrimeCompute` 函数）

```cpp
// line 689-692: q_prime Cast 为 bfloat16 写入 outQPrimeGm_
qPrimeLocal_ = outQueue_.AllocTensor<bfloat16_t>();
Cast(qPrimeLocal_, qPrimeUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);  // f32→bf16
outQueue_.EnQue(qPrimeLocal_);
DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outQPrimeGm[qgBeginOffset]);
```

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:630-633`（`GBKCompute` 函数）

```cpp
// line 630-633: kg Cast 为 bfloat16 写入 outKgGm_
kgLocal_ = outQueue_.AllocTensor<bfloat16_t>();
Cast(kgLocal_, kgUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);  // f32→bf16
outQueue_.EnQue<bfloat16_t>(kgLocal_);
DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outKgGm[kgBeginOffset]);
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:168, 172`，float32 计算，直接用于 Stage2/3 matmul。

**影响**：q_prime 在 Stage2 的 `CalAttnInter` 中与 state 做 K=128 的 bfloat16 matmul；kg 在 Stage2 的 `CalStateNew` 中做 K=64 的 bfloat16 matmul。bfloat16 存储进一步放大误差。

---

#### 差异点 7：qkt 以 bfloat16 存储

**NPU 代码位置**：`chunk_gated_delta_rule_stage1.h:307-310`（`ParaChunkAIC` 函数）

```cpp
// line 307-310: query @ key.T → bfloat16 outQkGm_（K=dk=128）
AICProcess(queryConGm_[i * ckOffset_], keyConGm_[i * ckOffset_],
           outQkGm_[chunkRowBase_[i] * chunkSize_],
           {validLenBatch_[i], validLenBatch_[i], dk_, ...}, true);
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:165`，`q_i @ k_i.transpose(-1, -2)` 在 float32 下计算。

**影响**：qkt 在 Stage3 中被读取并做 `scale * decay_mask * mask` 运算后再与 v_inner 做 bfloat16 matmul。bfloat16 存储的 qkt 经 Stage3 的 `CalMaskedQKT`（stage3.h:159-169）读取后再次 Cast 和运算，误差累积。

---

#### 差异点 8：Stage2 中 state 被 Cast 为 bfloat16 后做 matmul

**NPU 代码位置**：`chunk_gated_delta_rule_stage2.h:140-169`（`CalGCumExp` 函数）

```cpp
// line 160: 读取 bfloat16 state，Cast 为 float32
Cast(tmpBuffer1_, stateIn, RoundMode::CAST_NONE, Dv_ * curDk_);

// line 163: 乘以 exp(g_cum[-1])（float32）
Muls(tmpBuffer1_, tmpBuffer1_, tmpFloat, Dv_ * curDk_);

// line 164: Cast 回 bfloat16 ← 精度损失！
Cast(stateOut, tmpBuffer1_, RoundMode::CAST_RINT, Dv_ * curDk_);

// line 167: 以 bfloat16 写回 GM，供 AIC matmul 使用
outQueue_.EnQue(stateOut);
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:166-168`，state 全程 float32，直接用于 matmul。

**影响**：state 是跨 chunk 传递的递归状态，每个 chunk 都经历 bfloat16 量化。对于 seqlen=16384 的 Testcase1，有 256 个 chunk，state 量化误差累积 256 次。

---

#### 差异点 9：Stage2 的三个 bfloat16 matmul

**NPU 代码位置**：

```cpp
// stage2.h:184-195 (CalVPrime): k_cumdecay @ state.T → vInner (atomic add)
// K=Dk=128, bfloat16 × bfloat16 → bfloat16
sTP_->mm1->SetTensorA(kCumdecay, false);  // bfloat16
sTP_->mm1->SetTensorB(state, true);       // bfloat16 (已 Cast)
sTP_->mm1->IterateAll(vPrime, 1);

// stage2.h:171-182 (CalAttnInter): q_prime @ state.T → out
// K=Dk=128, bfloat16 × bfloat16 → bfloat16
sTP_->mm1->SetTensorA(qPrime, false);     // bfloat16
sTP_->mm1->SetTensorB(state, true);       // bfloat16 (已 Cast)
sTP_->mm1->IterateAll(out);

// stage2.h:197-208 (CalStateNew): vInner.T @ kg → state (atomic add)
// K=curChunkSize=64, bfloat16 × bfloat16 → bfloat16
sTP_->mm1->SetTensorA(vInner, true);      // bfloat16
sTP_->mm1->SetTensorB(kg, false);         // bfloat16
sTP_->mm1->IterateAll(state, 1);
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:166-173`，全部 float32 matmul。

**影响**：这三个 matmul 的输入均已被多次 bfloat16 量化（差异点 4/6/8），K=128 的 matmul 进一步放大误差。`CalVPrime` 使用 atomic add（`IterateAll(vPrime, 1)`），atomic add 的累加精度也可能低于直接 matmul。

---

#### 差异点 10：Stage3 的 masked_qkt 以 bfloat16 存储

**NPU 代码位置**：`chunk_gated_delta_rule_stage3.h:159-169`（`CalMaskedQKT` 函数）

```cpp
// line 159-160: 读取 bfloat16 qkt
AlignedCopyIn(sTP_->qkt[...], curChunkSize_, curChunkSize_);
auto qkt = inQueue_.DeQue<bfloat16_t>();

// line 162-166: float32 运算 (scale * decay_mask * mask)
Cast(tmpBuffer2_, qkt, RoundMode::CAST_NONE, curChunkSize_ * chunkSize_);
Muls(tmpBuffer2_, tmpBuffer2_, sTP_->scale, curChunkSize_ * chunkSize_);
Mul(tmpBuffer2_, tmpBuffer2_, tmpBuffer1_, curChunkSize_ * chunkSize_);
Mul(tmpBuffer2_, tmpBuffer2_, maskBuffer_, curChunkSize_ * chunkSize_);

// line 167: Cast 回 bfloat16 ← 精度损失！
Cast(scale_qkt, tmpBuffer2_, RoundMode::CAST_RINT, curChunkSize_ * chunkSize_);
outQueue_.EnQue(scale_qkt);
AlignedCopyOut(outGM, curChunkSize_, curChunkSize_);  // bfloat16 写回 GM
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:165`，float32 计算后直接用于 matmul。

**影响**：masked_qkt 是 Stage3 最终 matmul 的 A 矩阵，bfloat16 存储后与 bfloat16 v_inner 做 K=64 的 matmul，误差进一步累积。

---

#### 差异点 11：Stage3 最终 matmul

**NPU 代码位置**：`chunk_gated_delta_rule_stage3.h:173-184`（`AICProcess` 函数）

```cpp
// masked_qkt @ v_inner → attnOut (atomic add)
// K=curChunkSize=64, bfloat16 × bfloat16 → bfloat16
sTP_->mm3->SetTensorA(tmpGM);       // bfloat16 (masked_qkt)
sTP_->mm3->SetTensorB(vInner);      // bfloat16 (来自 Stage1)
sTP_->mm3->IterateAll(attnInter, 1); // atomic add 到输出
```

**torch 参考实现**：`chunk_gated_delta_rule_torch.py:169`，`attn @ v_new` 在 float32 下计算。

**影响**：两个输入都经过了多次 bfloat16 量化（差异点 4/7/10），atomic add 累加到最终输出，是精度损失的最终汇聚点。

---

### 2.4 误差分布特征

| 序列位置 | 不通过元素数(>0.01) | max_diff |
|---------|-------------------|----------|
| seq 0 | 298 | 0.0635 |
| seq 1 | 216 | 0.0430 |
| seq 2 | 108 | 0.0342 |
| seq 3 | 17 | 0.0146 |
| seq 4+ | 5 | 0.0144 |

误差集中在 chunk 前几个 token，后续迅速衰减。这是因为 `g` 为负值（范围 [-1, 0]），`g_cumsum` 在 chunk 末尾达到约 -34，`exp(g_cumsum)` 下溢为 0，导致状态快速衰减，前面 token 的误差对输出影响最大。

## 3. 精度差异汇总表

| # | 差异点 | NPU 代码位置 | torch 参考位置 | 数据类型 | 影响程度 |
|---|--------|-------------|---------------|---------|---------|
| 1 | attn 求逆结果 Cast 为 bf16 | stage1.h:530 | torch.py:146-151 | f32→bf16 | **高** |
| 2 | 分块求逆非对角块 bf16 matmul | stage1.h:797,802 | torch.py:147-150 (整体求逆) | bf16 matmul | **高** |
| 3 | kkt 存储为 bf16 (K=128) | stage1.h:300-302 | torch.py:146 | bf16 matmul | 中 |
| 4 | v_inner/k_cumdecay 存储为 bf16 | stage1.h:322-331 | torch.py:152-153 | bf16 matmul | **高** |
| 5 | v_beta/gBK 中间量 Cast 为 bf16 | stage1.h:608-612, 658-661 | torch.py:134-135 | f32→bf16 | 中 |
| 6 | q_prime/kg 存储为 bf16 | stage1.h:689-692, 630-633 | torch.py:168,172 | f32→bf16 | 中 |
| 7 | qkt 存储为 bf16 (K=128) | stage1.h:307-310 | torch.py:165 | bf16 matmul | 中 |
| 8 | Stage2 state Cast 为 bf16 后 matmul | stage2.h:164 | torch.py:166-168 | f32→bf16 | **高** (seqlen=16384 时累积) |
| 9 | Stage2 三个 bf16 matmul (K=128/128/64) | stage2.h:171-208 | torch.py:166-173 | bf16 matmul | **高** |
| 10 | Stage3 masked_qkt Cast 为 bf16 | stage3.h:167 | torch.py:165 | f32→bf16 | 中 |
| 11 | Stage3 最终 bf16 matmul (K=64) | stage3.h:177-182 | torch.py:169 | bf16 matmul | 中 |

> 影响程度评估依据：K 维越大误差越大（K=128 > K=64）；被多次复用的中间量影响更大（attn 被 3 次 matmul 使用）；跨 chunk 递归传递的 state 在长序列时误差累积。

## 4. 修复建议

### 4.1 修复 check_result 阈值 Bug（必做，低风险）

```python
# chunk_gated_delta_rule_main.py line 473
data_type_str = str(data_type).replace('torch.', '')  # 'torch.bfloat16' -> 'bfloat16'
```

### 4.2 提升 NPU 中间结果精度（按优先级排序）

| 优先级 | 修改项 | NPU 代码位置 | 改法 | 预期效果 |
|--------|--------|-------------|------|---------|
| P0 | attn 求逆结果改 float32 存储 | stage1.h:529-532 | `attnWsGm_` 改为 `GlobalTensor<float>`，去掉 Cast，`AICProcess`/`InverseAICProcess` 改 float32 matmul | 消除差异点 1+2，最大改善 |
| P1 | v_inner/k_cumdecay 改 float32 存储 | stage1.h:322-331 | `outVInnerGm_`/`outKCumdecayGm_` 改 `GlobalTensor<float>`，Stage2/3 matmul 输入改 float32 | 消除差异点 4 |
| P2 | Stage2 state 保持 float32 | stage2.h:160-167 | 去掉 line 164 的 Cast，state 直接以 float32 参与 matmul | 消除差异点 8 |
| P3 | q_prime/kg/qkt 改 float32 存储 | stage1.h:689-692, 630-633, 307-310 | 对应 GM 改 `GlobalTensor<float>` | 消除差异点 6+7 |
| P4 | v_beta/gBK 改 float32 存储 | stage1.h:608-612, 658-661 | 对应 workspace 改 `GlobalTensor<float>` | 消除差异点 5 |

> 注：P0-P4 修改会增大 GM 内存占用（bfloat16→float32 翻倍），需同步修改 tiling 中的 workspace size 计算。

### 4.3 放宽测试阈值（短期方案）

如果 NPU kernel 修改暂不可行，考虑：
- 降低 `pct_thd` 从 0.005 到 0.02（允许 98% 通过率）
- 或将 bfloat16 的 `rtol` 从 0.0078125 放宽到 0.015625（2^-6）

## 5. 验证方法

使用 `debug_stage.py` 脚本进行逐阶段对比：
```bash
export LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH
python3 debug_stage.py
```

使用 `debug_topn.py` 分析最大误差元素分布：
```bash
python3 debug_topn.py
```

## 6. 结论

精度失败由三个因素叠加：

| 因素 | 影响程度 | max_abs | 修复难度 |
|------|---------|---------|---------|
| check_result 阈值 Bug | 加剧失败（通过率从 88.7% 降到 82.1%） | - | 低（一行代码） |
| bfloat16 matmul 固有精度损失 | 中 | 0.0023 | 高（需 float32 matmul） |
| **NPU 算法实现数值差异** | **大** | **0.063** | **中（改中间存储精度）** |

**核心结论**：NPU kernel 有 11 处中间结果以 bfloat16 存储/传递（见汇总表），而 torch 参考全程 float32。其中影响最大的 3 处是：
1. **attn 求逆矩阵**（stage1.h:530）— 被 3 次 matmul 复用，误差放大 3 倍
2. **v_inner/k_cumdecay**（stage1.h:322-331）— Stage1→Stage2 关键传递数据
3. **Stage2 state**（stage2.h:164）— 跨 chunk 递归，长序列累积 256 次

将这 3 处中间变量从 bfloat16 提升为 float32 存储可显著改善精度。

---

## 7. 修复实施与验证结果

### 7.1 修改文件清单

共修改 6 个文件，将 NPU kernel 全部中间结果从 bfloat16 提升为 float32：

| 文件 | 改动内容 |
|------|---------|
| `op_host/chunk_gated_delta_rule_tiling.cpp:163-168` | matmul tiling 的 A/B/C 类型从 `DT_BFLOAT16` 改为 `DT_FLOAT` |
| `op_kernel/arch35/chunk_gated_delta_rule.h:173-189` | 中间 GM tensor (`kCumDecay_`/`vInner_`/`qPrime_`/`kg_`/`qkt_`) 从 `GlobalTensor<lowType>` 改为 `GlobalTensor<float>`；workspace offset 从 `sizeof(lowType)` 改为 `sizeof(float)` |
| `op_kernel/arch35/chunk_gated_delta_rule.h:206-240` | `Process()` 重构：用 `highState_`(float) 作 state、`attnInter_`(float) 作 output；前后加 `CopyCast` 进行 bfloat16↔float 转换（用户输入/输出仍为 bfloat16） |
| `op_kernel/arch35/chunk_gated_delta_rule.h:253-278` | `RunStage2`/`RunStage3` 参数改为 `GlobalTensor<float>`，输出写入 `attnInter_` 而非 `out_` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage1.h:26-29` | `MatmulType` 从 `bfloat16_t` 改为 `float` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage1.h:48-67` | `GDRStageOneInitParams` 中 `kCumdecay`/`vInner`/`qPrime`/`kG`/`qK` 改为 `GlobalTensor<float>` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage1.h:90-113` | workspace GM 全部改为 `GlobalTensor<float>`，`sizeof(bfloat16_t)` → `sizeof(float)` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage1.h:529-532` | `InverseCompute`：去除 f32→bf16 Cast，直接输出 float |
| `op_kernel/arch35/chunk_gated_delta_rule_stage1.h:608-633` | `GBKCompute`：去除 Cast，`DataCopyOutBf16` → `DataCopyOutFp32` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage1.h:658-694` | `VBetaCompute`/`QPrimeCompute`：去除 Cast，输出 float |
| `op_kernel/arch35/chunk_gated_delta_rule_stage1.h:780-790` | `DataCopyOutBf16` 改为 `DataCopyOutFp32`（float 类型） |
| `op_kernel/arch35/chunk_gated_delta_rule_stage1.h:808-828` | `AICProcess`/`InverseAICProcess` 参数改为 `GlobalTensor<float>` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage2.h:26-29` | `MatmulType` 从 `bfloat16_t` 改为 `float` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage2.h:31-49` | `StageTwoParams` 全部改为 `GlobalTensor<float>` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage2.h:140-169` | `CalGCumExp`：去除 f32→bf16 Cast（line 160/164），state 直接以 float 读写 |
| `op_kernel/arch35/chunk_gated_delta_rule_stage2.h:171-208` | `CalAttnInter`/`CalVPrime`/`CalStateNew` 参数改为 `GlobalTensor<float>` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage3.h:26-29` | `MatmulType` 从 `bfloat16_t` 改为 `float` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage3.h:31-47` | `StageThreeParams` 中 `qkt`/`vInner`/`attnOut` 改为 `GlobalTensor<float>` |
| `op_kernel/arch35/chunk_gated_delta_rule_stage3.h:133-171` | `CalMaskedQKT`：去除 Cast（line 162/167），读取 float qkt，输出 float |
| `op_kernel/arch35/chunk_gated_delta_rule_stage3.h:173-184` | `AICProcess` 参数改为 `GlobalTensor<float>` |

### 7.2 关键设计决策

1. **用户接口不变**：用户输入（query/key/value/beta）和输出（out/finalState）仍为 bfloat16，仅在 kernel 内部提升为 float32 计算
2. **CopyCast 桥接**：在 `Process()` 中用 `CopyCast` 将 bfloat16 输入转为 float（`highState_`），计算完成后再 Cast 回 bfloat16 写入用户输出
3. **attnInter_ workspace 复用**：`attnInter_` 作为 float 中间输出缓冲，按 group 从 offset 0 使用（非 `cg.startPos`），避免长序列越界
4. **workspace 无需改 tiling**：tiling 中 `interWorkspaceSz` 已用 `sizeHigh`(float=4字节) 计算，原 kernel 实际用 bfloat16(2字节) 只用了一半空间，改为 float 后正好用满

### 7.3 验证结果

```
Testcase0: B=1, seqlen=64, nk=4, nv=4, dk=128, dv=128, chunk_size=64, bfloat16, has_g=True
  output:  PctRlt=100.000000%  Pass
  state:   PctRlt=100.000000%  Pass

Testcase1: B=1, seqlen=16384, nk=4, nv=4, dk=128, dv=128, chunk_size=64, bfloat16, has_g=True
  output:  PctRlt=100.000000%  Pass
  state:   PctRlt=100.000000%  Pass

============================== 2 passed in 7.53s ==============================
```

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Testcase0 output 通过率 | 91.89% (Failed) | **100.00% (Pass)** |
| Testcase0 state 通过率 | 97.76% (Failed) | **100.00% (Pass)** |
| Testcase1 output 通过率 | 94.44% (Failed) | **100.00% (Pass)** |
| Testcase1 state 通过率 | 93.34% (Failed) | **100.00% (Pass)** |
| Max-RelativeError | 0.054 | **0.002** |
| 测试结果 | 2 failed | **2 passed** |

### 7.4 修复总结

通过将 NPU kernel 中 11 处 bfloat16 中间结果存储全部提升为 float32（3 个 Stage 的 matmul 类型、中间 GM tensor、去除所有 f32→bf16 Cast），彻底消除了 bfloat16 量化误差在阶段间传递和累积的问题。两个精度测试用例均以 100% 通过率通过，Max-RE 从 0.054 降至 0.002。

### 7.5 RDV 参数集测试结果

执行 `bash test_run.sh rdv`，共 72 个用例：

| 结果 | 数量 |
|------|------|
| Passed | 69 |
| Failed | 3 |

**失败用例参数：**

| 用例 | B | seqlen | nk | nv | dk | dv | has_g | 通过率(output) | 通过率(state) |
|------|---|--------|----|----|----|----|-------|---------------|--------------|
| param_combinations60 | 3 | 10000 | 16 | 64 | 50 | 50 | True | 88.14% | 0.30% |
| param_combinations61 | 3 | 10000 | 16 | 32 | 5 | 8 | True | 89.41% | 0.00% |
| param_combinations63 | 4 | 512 | 8 | 8 | 1 | 1 | False | 42.30% | 0.00% |

**失败特征**：dk/dv 非 32 对齐（50, 5, 8, 1），与 BLOCK_SIZE=32 对齐要求冲突，可能导致 matmul 尾块处理或 DataCopy 对齐异常。后续需针对非对齐 dk/dv 场景单独排查。
