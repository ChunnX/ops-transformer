# MhcPreSinkhornBackward 计算流程分析报告 v3

> 原则: 不局限于 golden 10 步, 按数据依赖 + UB 容量 + C 大小分场景

## ★★★ 核心约束（所有实现必须遵守）★★★

### 1. VReg 对齐 — VF 计算天然对齐

n/k 维度对齐到 `GetVRegSize()`（通常 256B = 64 个 fp32）。

```
nAlign  = CeilAlign(n, vRegElem)           // n 对齐到 VReg (n≤8 时 nAlign=64)
nnAlign = n × nAlign                        // sinkhorn: n 行 × nAlign
kAlign  = (2 + n) × nAlign                  // k 总行数 × nAlign
```

### 2. K 维 Reshape 搬运 — (2+n) 行 × n 列

**这是最关键的设计决策。** GM 中 hcBeforeNorm 按 `[B*S, kDim_]` 连续排布（kDim_ = 2n + n²），搬入 UB 时 reshape 为 `(2+n) 行 × n 列` 矩阵，每行 pad 到 nAlign。

```
GM:  [bs0: kDim_ 连续 | bs1: kDim_ 连续 | ...]
      ← 2n + n² 个元素, 连续 1D →

UB:  [bs0: (2+n) 行 × nAlign | bs1: (2+n) 行 × nAlign | ...]
      行 0:     hPre     (n 有效, pad 到 nAlign)
      行 1:     hPost    (n 有效, pad 到 nAlign)
      行 2..1+n: sinkhorn (n 行, 每行 n 有效, pad 到 nAlign)
```

搬入方式:
```cpp
CopyInQue(que, gm[offset], bsLen * (2 + n), n, 0, nAlign);
//                              ^行数        ^每行元素 ^UB对齐
```

**效果**: VF 内 DataCopy 全部对齐，无需 mask 处理尾部。计算直接用 ALL mask。

### 3. UB 复用 — 两阶段 Buffer 生命周期

PhaseA (C 维) 和 PhaseB (K 维) 使用不同 Buffer 集合，通过 pipe Reset 切换：
- PhaseA: xQue, gradHinQue, xFp32Que, gradXFromHinQue, hPreAndGradHPostQue, gradHPreBuf
- PhaseB: hcBeforeNormQue, invRmsQue, gradHResQue, zBuf, normOutForwardBuf, gradZBuf, biasBuf, gradCalcBuf, tmpBufs, sumOutBufs, normOutBuf

### 4. Broadcast 消除 — 用 Muls 替代

- `hPre[B,S,N,1] × gradHin[B,S,N,C]` → 逐 N 行 Muls(gradHin, hPre[nK])
- `invRms[B,S,1] × hcBeforeNorm[B,S,K]` → 逐 BS 行 Duplicate + Mul
- `alpha[scalar] × gradZ[B,S,K]` → Muls(gradZ, alpha)

### 5. PR 6896 功能修复（同事已验证）

- **ComputeGNormOutAndGHcBeforeNorm**: 逐 BS 算 gradNormOut + gradHcBeforeNorm，解决 bsLen>1 搬运问题
- **ComputeGradXFromRms 上提**: 移到 MatMul 前的 BS 循环内，减少一轮 BS 循环
- **ComputeMulScalar**: 用 VF 循环+UpdateMask 替代全 mask UnalignReg

### 6. 计算流程（2 轮 BS 循环）

```
BS 循环 #1: PhaseA(C) + PhaseB(K) + PhaseRms(C)
  PhaseA: x/gradHin → gradHPre + gradXFromHin→WS + xFp32→WS
  PhaseB: hcBeforeNorm → z + gradZ + gradNormOut→WS + gradHcBeforeNorm→WS + gradInvRms→UB
          gradBias/gradAlpha 累加 → WS
  PhaseRms: xFp32 + gradInvRms → gradXFromRms→WS (逐 BS)

SyncAll → CrossCore → AIC MatMul → CrossCore → SyncAll

BS 循环 #2: PhaseE(C)
  gradXFromHin + gradXFromRms + gradXFromMatmul → gradX → GM

blockIdx==0: ReduceBiasAlpha → GM
```

## 一、Shape 约束

| 维度 | 范围 | 特点 |
|------|------|------|
| n | 1~8 (主路径 4, 8) | 固定小值, K=n²+2n ≤ 80 |
| c | 64 ~ 4096+ | **可变**, 决定是否需要切 C |
| B*S | 1 ~ 数万 | 决定分核数 |

## 二、数据依赖链（重排后）

```
PhaseA (C 维):
  x[bs,n,c] + gradHin[bs,c] + hPre[bs,n]
    → gradHPre[bs,n]          (reduce_sum over C)
    → gradXFromHin[bs,n,c]    → WS
    → xFp32[bs,n,c]           → WS

PhaseB (K 维, 不涉及 C):
  hcBeforeNorm[bs,k] + invRms[bs] + bias[k] + gradHPost[bs,n] + gradHRes[bs,nn]
  + sumOut[iter,bs,n] + normOut[iter,bs,nn]
    → z_pre/post/res[bs,k]
    → gradZ_pre/post/res[bs,k]   (依赖 gradHPre from PhaseA)
    → gradBiasAcc += gradZ       (跨 BS 累加)
    → gradAlphaAcc += normOut * gradZ
    → gradHcBeforeNorm[bs,k]     → WS (供 AIC MatMul)
    → gradInvRms[bs]             → 小 buffer (4B/BS)

  SyncAll → AIC MatMul → SyncAll

PhaseE (C 维):
  xFp32[bs,n,c] + gradXFromHin[bs,n,c] + gradXFromMatmul[bs,n,c]  (from WS)
  + gradInvRms[bs]  (from PhaseB 小 buffer)
    → gradXFromRms = -(invRms³ * xFp32 * gradInvRms / NC)   (当场算, 不落 WS)
    → gradX = gradXFromHin + gradXFromRms + gradXFromMatmul
    → Cast → GM
```

**关键**: gradXFromRms 不再单独落 WS, 在 PhaseE 当场算当场加。省 1 轮 C 循环。

## 三、C 切分场景分析

### 3.1 为什么不能"完全切 C"？

C 小时 (如 c=64):
- x[bs, n=8, c=64] fp32 = bs × 2KB
- 一趟可处理 bs=64 个 BS, UB 完全装得下
- 如果强行切 C → 每次只处理 c=64 的一小块, BS 循环次数暴增, **浪费 UB 容量和带宽**

C 大时 (如 c=4096):
- x[bs, n=8, c=4096] fp32 = bs × 128KB
- 一趟只能处理 bs=1~2 个 BS
- 必须切 C, 否则 UB 装不下

### 3.2 场景划分

| 场景 | C 范围 | 策略 | C 循环 |
|------|--------|------|--------|
| **C-Short** | c × n × 4B × DB ≤ UB预算 | 不切 C, 全量处理 | 1 次 (cIdx=0, cLen=c) |
| **C-Long** | c × n × 4B × DB > UB预算 | 切 C, 内部循环 | ⌈c / ubFactorC⌉ 次 |

### 3.3 判断逻辑 (Tiling 层)

```
// PhaseA/E 的 C 相关 UB 占用 (每 BS):
perBsCostC = DB * n * cAlignedFp16 * sizeof(X_T)    // xQue
           + DB * cAlignedFp16 * sizeof(GRADHIN_T)   // gradHinQue
           + DB * n * cAlignedFp32 * sizeof(U)       // xFp32Que
           + DB * n * cAlignedFp32 * sizeof(U)       // gradXFromHinQue

// PhaseA/E 的 C 无关 UB 占用 (每 BS):
perBsCostK = DB * kAligned * sizeof(U)               // hcBeforeNormQue
           + DB * nAligned * sizeof(U)                // hPreAndGradHPostQue
           + DB * nnAligned * sizeof(U)               // gradHResQue
           + ... (zBuf, normOutBuf, tmpBufs 等)

// 固定开销:
fixedCost = kAligned * 3 * sizeof(U)                  // bias + gradCalc × 2
          + nnAligned * sizeof(U)                      // gradAlphaAndBiasQue

availUb = ubSize - RESERVE

if (perBsCostC + perBsCostK) * 1 ≤ availUb - fixedCost:
    // C-Short: 不切 C
    ubFactorC = c
    ubFactorBS = (availUb - fixedCost) / (perBsCostC + perBsCostK)
else:
    // C-Long: 切 C
    ubFactorC = 合理对齐值 (如 256 或 512)
    ubFactorBS = (availUb - fixedCost) / (perBsCostC_reduced + perBsCostK)
```

### 3.4 TilingData 调整

```cpp
struct MhcPreSinkhornBackwardArch35DeterminiticTilingData {
    // shape
    int64_t b, s, c, n;
    // core
    int64_t usedCoreNum;
    int64_t aicCoreNum;
    // BS 切分 (始终需要)
    int64_t ubFactorBS;      // UB 单次处理 BS 数
    int64_t blockFactorBS;   // 每核 BS 数
    int64_t blockTailBS;     // 尾核 BS 数
    // C 切分 (仅 C-Long 场景有效, C-Short 时 = c)
    int64_t ubFactorC;       // UB 单次处理 C 数 (= c 时不切)
    // 业务
    int64_t skIterCount;
    float eps;
    // MatMul
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};
```

## 四、UB Buffer 使用分析

### 4.1 PhaseA (C 维) — 需要 C 相关 Buffer

| Buffer | 大小 | C-Short (c=64) | C-Long (c=4096) |
|--------|------|----------------|-----------------|
| xQue_ (DB) | 2 × bs × n × cAligned × 2B | 2×64×8×64×2 = 128KB | 2×1×8×4096×2 = 128KB |
| gradHinQue_ (DB) | 2 × bs × cAligned × 2B | 2×64×64×2 = 16KB | 2×1×4096×2 = 16KB |
| xFp32Que_ (DB) | 2 × bs × n × cAligned × 4B | 2×64×8×64×4 = 256KB | 2×1×8×4096×4 = 256KB |
| gradXFromHinQue_ (DB) | 2 × bs × n × cAligned × 4B | 同上 | 同上 |
| hPreAndGradHPostQue_ (DB) | 2 × bs × nAligned × 4B | 2×64×8×4 = 4KB | 2×1×8×4 = 64B |
| gradHPreBuf_ | bs × nAligned × 4B | 64×8×4 = 2KB | 1×8×4 = 32B |

### 4.2 PhaseB (K 维) — 不涉及 C

| Buffer | 大小 | N=8, bs=64 |
|--------|------|-----------|
| hcBeforeNormQue_ (DB) | 2 × bs × kAlign × 4B | 2×64×80×4 = 40KB |
| invRmsQue_ (DB) | 2 × bs × 4B | 512B |
| gradHResQue_ (DB) | 2 × bs × nnAlign × 4B | 2×64×64×4 = 32KB |
| zBuf_ | bs × bsnnAlign × 4B | 64×64×4 = 16KB |
| normOutForwardBuf_ | bs × bsnnAlign × 4B | 16KB |
| biasBuf_ | kAlign × 4B | 320B |
| gradCalcBuf1/2_ | kAlign × 4B × 2 | 640B |
| tmpDiv_×4 + tmpMul + zResMax + zResMaxIdx + tmp + isMax | bs × nnAlign × 4B × 8 | 128KB |
| sumOutRow/ColBuf_ | skIter × bs × nAlign × 4B × 2 | 20×64×8×4×2 = 80KB |
| normOutBuf_ | skIter × bs × nnAlign × 4B | 20×64×64×4 = 320KB |
| gradAlphaAndBiasQue_ | nnAlign × 4B + 32B | 288B |

### 4.3 PhaseE (C 维) — 复用 PhaseA 的 C 相关 Buffer

PhaseE 需要的 Buffer 与 PhaseA 几乎相同 (xQue_, xFp32Que_, gradXFromHinQue_), 加上 gradXFromMatmul。可通过 pipe Reset 复用。

### 4.4 UB 峰值估算 (N=8)

| 场景 | PhaseA 峰值 | PhaseB 峰值 | 瓶颈 |
|------|------------|------------|------|
| C=64, bs=64 | ~420KB | ~620KB | PhaseB (normOutBuf × skIter) |
| C=256, bs=32 | ~420KB | ~310KB | PhaseA |
| C=1024, bs=8 | ~420KB | ~80KB | PhaseA |
| C=4096, bs=1 | ~420KB | ~10KB | PhaseA |

**结论**: C 小时 PhaseB 是 UB 瓶颈 (normOutBuf 随 skIter 增长); C 大时 PhaseA 是瓶颈。

## 五、Workspace 分析

| WS Buffer | 大小 | 写入阶段 | 读取阶段 |
|-----------|------|---------|---------|
| xFp32[B*S, N, C] | B*S × N × C × 4B | PhaseA | PhaseE |
| gradXFromHin[B*S, N, C] | B*S × N × C × 4B | PhaseA | PhaseE |
| gradHcBeforeNorm[B*S, K] | B*S × K × 4B | PhaseB | AIC MatMul |
| gradXFromMatmul[B*S, N*C] | B*S × N × C × 4B | AIC | PhaseE |
| gradAlpha[3 × usedCoreNum] | 小 | PhaseB | 跨核归约 |
| gradBias[...] | 小 | PhaseB | 跨核归约 |
| gradInvRms[B*S] | B*S × 4B | PhaseB | PhaseE (小 buffer, 可放 UB) |

**v2 优化**: gradXFromRms 不再落 WS, 省掉 B*S × N × C × 4B。

## 六、最终流程 (分场景)

### 6.1 C-Short 场景 (c × n × 4B × DB ≤ UB 预算)

```
Process():
  GetAlphaAndBias()

  // BS 循环 #1: A + B 合并 (C 不切, 1 趟过)
  for bsIdx:
    PhaseA: x/gradHin 全量搬入 → gradHPre + gradXFromHin→WS + xFp32→WS
    PhaseB: K 维全量 → gradZ + gradHcBeforeNorm→WS + gradInvRms→UB

  SyncAll → CrossCore → AIC MatMul → CrossCore → SyncAll

  // BS 循环 #2: E (C 不切, 1 趟过)
  for bsIdx:
    PhaseE: 读 WS + gradXFromRms 当场算 → sum → Cast → GM

  ReduceBiasAlpha()
```

**循环次数**: BS 2 轮, C 每轮 1 次 = **2 次 C 遍历**

### 6.2 C-Long 场景 (c × n × 4B × DB > UB 预算)

```
Process():
  GetAlphaAndBias()

  // BS 循环 #1: A(切C) + B(不切C)
  for bsIdx:
    PhaseA: for cIdx: x/gradHin 切C搬入 → gradHPre累加 + gradXFromHin→WS + xFp32→WS
    PhaseB: K 维全量 → gradZ + gradHcBeforeNorm→WS + gradInvRms→UB

  SyncAll → CrossCore → AIC MatMul → CrossCore → SyncAll

  // BS 循环 #2: E (切C)
  for bsIdx:
    PhaseE: for cIdx: 读 WS + gradXFromRms 当场算 → sum → Cast → GM

  ReduceBiasAlpha()
```

**循环次数**: BS 2 轮, C 每轮 ⌈c/ubFactorC⌉ 次

### 6.3 核心差异

| 项 | C-Short | C-Long |
|----|---------|--------|
| ubFactorC | = c | < c (对齐值) |
| PhaseA C 循环 | 1 次 | ⌈c/ubFactorC⌉ 次 |
| PhaseE C 循环 | 1 次 | ⌈c/ubFactorC⌉ 次 |
| PhaseB | 相同 | 相同 (K 维不切) |
| UB 峰值 | PhaseB 主导 | PhaseA 主导 |
| BS 一趟处理量 | 大 (多 BS) | 小 (少 BS) |

## 七、入口函数定义

```
Init(...)
  → GM 绑定 + UB Buffer + 高频变量

Process()
  ├── GetAlphaAndBias()
  ├── for bsIdx: PhaseA_CLoop + PhaseB_KDim
  ├── SyncAll + CrossCore
  ├── AIC: PhaseD_MatMul
  ├── CrossCore + SyncAll
  ├── for bsIdx: PhaseE_CLoop
  └── ReduceBiasAlpha()

PhaseA_CLoop(bsIdx, bsLen, &gradHPreOut)
  → if C-Short: 1 趟
  → if C-Long: for cIdx in ⌈c/ubFactorC⌉

PhaseB_KDim(bsIdx, bsLen, gradHPreIn)
  → K 维全量, 不切

PhaseE_CLoop(bsIdx, bsLen)
  → if C-Short: 1 趟
  → if C-Long: for cIdx in ⌈c/ubFactorC⌉
```
