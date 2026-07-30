# 确定性方案 v3：Per-core-per-Q-head 槽位 + 一次性归约

## 方案概述

为每个 `(cube_core, qHeadInGroup)` 分配独立的 workspace 槽位，所有核跑完全部 task 后，
AIV 一次性按固定顺序归约所有槽位到 shared workspace。

## 数据流

```
VecPre (AIV):
  ├── 清零 shared dQ/dK/dV workspace
  └── ZeroDetWorkspace: 清零 detDkWorkspace + detDvWorkspace

AIC (各核独立跑完全部 task):
  ├── blockMmad1: S=Q×Kᵀ, dP=dOut×Vᵀ → S/dP workspace
  ├── CUBE2VEC → AIV 算 softmax/dS → VEC2CUBE
  ├── blockMmad2: dQ=dS×K → atomicAdd 到 shared dQ (dQ 按 Q-head 隔离, 无跨核冲突)
  ├── blockMmad3: dV=Pᵀ×dOut → atomicAdd 到 per-core-per-Q-head detDvWorkspace 槽位
  │     槽位地址: detDvBase + coreIdx×groupSize×dkvSize + qHeadInGroup×dkvSize
  │     atomicAdd 仅用于同一 Q-head 的跨 Q-block 累加 (同核内顺序固定, 确定性)
  └── blockMmad3: dK=dSᵀ×Q → atomicAdd 到 per-core-per-Q-head detDkWorkspace 槽位 (同上)

CUBE2POST + SyncAll (所有核完成)

VecDeterReduce (AIV):
  ├── 按固定顺序 (core0_qh0 → core0_qh1 → ... → coreN_qhG) 读取 detDkWorkspace
  ├── FP32 累加所有 aicNum×groupSize 个槽位 → 写入 shared dK workspace
  └── 同理归约 detDvWorkspace → shared dV workspace

VecPost (AIV):
  ├── dQ: shared workspace → scale → cast(FP16) → 输出
  ├── dK: shared workspace (已被 VecDeterReduce 填充) → scale → cast → 输出
  └── dV: shared workspace (同上) → cast → 输出
```

## Workspace 布局

```
workspace:
├── [0, sOutSize)                    S  (float32)
├── [+, dPOutSize)                  dP (float32)
├── [+, dQOutSize)                  dQ (float32, AIC atomicAdd 写入)
├── [+, dKOutSize)                  dK (float32, VecDeterReduce 写入)
├── [+, dVOutSize)                  dV (float32, VecDeterReduce 写入)
├── [+, gradSize)                   sftmg 临时区
│
│   ===== 仅 det=True 时分配 =====
│
├── [+, detDkWorkspaceSize)         detDkWorkspace
│     共 aicNum × groupSize 个槽位, 每个槽位 dkvSize 个 float
│     布局: [core0_qh0 | core0_qh1 | ... | core1_qh0 | ... | coreN_qhG]
│
└── [+, detDvWorkspaceSize)         detDvWorkspace (结构同上)

detDkWorkspaceSize = detDvWorkspaceSize = aicNum × groupSize × dkvSize × sizeof(float)
```

## 关键设计决策

### 为什么用 per-core-per-Q-head 槽位而非 per-core 槽位

- per-core 槽位: 同一核的不同 Q-head 写入同一槽位的相同 KV 地址 → 需 atomicAdd → 非确定性
- per-core-per-Q-head 槽位: 不同 Q-head 写入不同槽位 → 跨 Q-head 累加由 VecDeterReduce 按序完成 → 确定性

### 为什么保留 atomicAdd (isAtomicAdd=true)

- 同一 Q-head 的不同 Q-block (S1 > blockShapeX 时) 写入同一槽位的相同 KV 地址
- 这些写入在同一核内按固定顺序执行 (FIX 管道 in-order)
- atomicAdd 仅用于此跨 Q-block 累加, 顺序固定, 确定性有保证

### VecDeterReduce 实现

- 每个 AIV 核处理 dkvSize/headDim 行的数据
- 按 block 读取 per-core 槽位到 UB, FP32 Add 累加
- 使用 MTE2/V/MTE3 管道同步, V_MTE2 防止 tempUb 被覆盖
- 写入 shared workspace 供 VecPost 读取

## 切换方式

```bash
# 确定性模式
export BSAG_DETERMINISTIC=1

# 非确定性模式 (默认, 原始代码, 性能优先)
export BSAG_DETERMINISTIC=0
```

## Workspace 大小

```
总 det workspace = 2 × aicNum × groupSize × dkvSize × sizeof(float)
```

| 场景 | groupSize | dkvSize | det workspace |
|------|-----------|---------|--------------|
| B=2,N=8,KV=2,S=128 | 4 | 65,536 | 40 MB |
| B=4,N=16,KV=4,S=256 | 4 | 524,288 | 320 MB |
| B=8,N=16,KV=4,S=512 | 4 | 2,097,152 | 1.25 GB |
| B=8,N=16,KV=2,S=1024 | 8 | 2,097,152 | 2.56 GB |
| B=32,N=16,KV=2,S=2048 | 8 | 16,777,216 | 20.5 GB |

**限制**: 大 shape (B×S 大) 时 workspace 可能超过 HBM 容量 (32GB)。

## 代码修改清单

### op_host/block_sparse_attention_grad_tiling.h
- 新增字段: detDkWorkspaceSize, detDvWorkspaceSize, aicNumForDet, groupSizeForDet, deterministic
- 新增成员: bool deterministic_

### op_host/arch22/block_sparse_attention_grad_tiling_arch22.cpp
- 读取环境变量 BSAG_DETERMINISTIC 设置 deterministic_
- CalculateWorkSpace: 条件分配 detDkWorkspace/detDvWorkspace
- FillTilingData: 设置 deterministic 字段

### op_kernel/arc22/block_sparse_attention_grad_kernel.h
- AIC: if(deterministic) gDk/gDv 指向 per-core-per-Q-head 槽位, 每次 blockMmad3 前 SetGlobalBuffer
- AIC: if(!deterministic) gDk/gDv 指向 shared workspace (原始行为)
- VecPre: if(deterministic) ZeroDetWorkspace
- AIV: if(deterministic) VecDeterReduce before VecPost

## 测试结果

80 组 shape 充分验证:
- det=True 精度: 80/80 PASS
- det=False 精度: 80/80 PASS
- det=True bit-exact 确定性: 80/80 PASS

## 已知限制

1. workspace 随 groupSize × dkvSize 线性增长, 大 shape 可能 OOM
2. VecDeterReduce 读取 aicNum×groupSize 个槽位, 大 groupSize 时性能开销大
3. 保留 atomicAdd (跨 Q-block), 依赖 FIX 管道 in-order 保证确定性
