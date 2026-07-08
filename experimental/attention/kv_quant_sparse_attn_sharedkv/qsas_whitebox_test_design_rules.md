# 白盒用例设计规则 — KvQuantSparseAttnSharedkv (arch35)

## 算子概述

`KvQuantSparseAttnSharedkv` 是一个基于量化KV Cache的稀疏注意力算子，运行于Ascend 950PR/DT平台（arch35）。KV以8bit量化格式存储（FP8_E4M3FN 或 HIFLOAT8），在kernel中在线反量化为BF16后参与注意力计算。K和V始终相同（SharedKV），支持三种计算模式：

- **SWA (Sliding Window Attention)**：仅原始KV，滑动窗口注意力
- **CFA (Compressed Flash Attention)**：原始KV + 压缩KV（密集）
- **SCFA (Sparse Compressed Flash Attention)**：原始KV + 压缩KV（稀疏TopK）

Kernel采用4-stage软件流水线：Vec0（KV加载+反量化） → Cube/BMM1（Q×K^T） → Vec1（Softmax） → Cube/BMM2+Vec2（Flash Update+输出）。

### KV 数据packed布局（每token 640B）

```
[Rope: 64×BF16 = 128B] [Nope: 448×FP8/HIF8 = 448B] [Scale: 7×FP8_E8M0 = 7B] [Pad: 57B]
```

---

## 一、参数维度边界值覆盖

### 1.1 核心维度参数

| 参数 | 最小值 | 典型值 | 最大值 | 特殊边界 |
|------|--------|--------|--------|----------|
| `B` (batch) | 1 | 48 | 大B | **0** — 空batch |
| `S1` (Q seq) | 1 | 2, 8192 | 大S1 | **0** — 空query（触发needInit） |
| `N1` (Q heads) | 64 | 64, 128 | 128 | **仅64或128**；128触发G-split |
| `N2` (KV heads) | 1 | 1 | 1 | **固定为1** |
| `D` (head dim) | 512 | 512 | 512 | **固定为512** |
| `actualS2OriSize` | 0 | 8192, 131072 | 大值 | **0** — 无原始KV |
| `actualS2CmpSize` | 0 | 2048 (=8192/4) | 大值 | **0** — 无压缩KV |

### 1.2 量化相关参数

| 参数 | 值域 | 说明 |
|------|------|------|
| `kv_quant_mode` | **1** (per-tile FP8_E4M3FN) | 量化模式 |
| `tile_size` | **64** | 量化粒度（每个scale覆盖64个nope元素） |
| `rope_head_dim` | **64** | rope维度，未量化，存储为BF16 |
| `nope_head_dim` | **448** (= 512 - 64) | nope维度，存储为FP8/HIF8 |
| `quant_scale_head_dim` | **7** (= ceil(448/64)) | 每行7个scale值（FP8_E8M0格式） |
| `dSizeVInput` | **640** | KV输入维度（含rope+nope+scale+pad） |

### 1.3 Sliding Window 参数

| 参数 | 值域 | 说明 |
|------|------|------|
| `oriMaskMode` | **4 (BAND)** | 仅支持Band模式 |
| `cmpMaskMode` | **3 (RIGHT_DOWN_CAUSAL)** | 仅支持RightDownCausal |
| `oriWinLeft` | **127** | 滑动窗口左边界 |
| `oriWinRight` | **0** | 滑动窗口右边界 |

### 1.4 压缩相关参数

| 参数 | 值域 | 说明 |
|------|------|------|
| `cmpRatio` | 4, 128 | SWA模式可为1/4/128；CFA/SCFA仅4/128 |
| `cmpTopK` (K) | 512, 1024 | CSA模式下稀疏TopK token数 |

### 1.5 反量化处理中极值

| 场景 | 说明 |
|------|------|
| scale=最小值 | FP8_E8M0 最小表示值 → 反量化输出趋于0 |
| scale=最大值 | FP8_E8M0 最大表示值 → 反量化输出放大 |
| nope=0x00 | FP8_E4M3FN零值 → 各Chanel反量化输出0 |
| nope=0x3C | FP8_E4M3FN最大值448 → 各Chanel反量化输出=scale×448 |
| nope=0x01 | FP8_E4M3FN最小非零值 → 极小值反量化 |
| rope全零/满值 | rope通道（BF16）不做反量化，直接透传 |

---

## 二、块切分与尾块覆盖

### 2.1 各轴块大小与尾块条件

| 轴 | 块大小 | 尾块条件 | 尾块代码路径 |
|----|--------|----------|--------------|
| **S1 (Q)** | `qSNumInOneBlock` = gSize (SWA/CFA) 或 1 (SCFA) | `s1RealSize < qSNumInOneBlock` | `ComputeSouterParam` |
| **S2 (KV)** | `s2BaseSize = 128` | `s2RealSize < 128` | `ComputeBmm1Tail` + softmax VF分支 |
| **M (M=Q×G)** | `mBaseSize` (SOC/模式相关) | 非整除时的余数 | metadata `s1GTailSize` → `TAIL_BLOCK`代价 |
| **K (matmul)** | `dBaseMatmulSize = 128` | dSize=512 整除；BMM1使用D/2=256 整除 | 两次matmul均无尾块 |
| **Vec0处理批** | `v0ProcessBase = 16` | 最后一批 < 16行 | `ProcessNotSparseKv()`内部loop |
| **Nope反量化** | `kvNumPerLoop = 128` | 第4次循环处理64项 (=448-3×128) | `AntiquantVFImplFp8D448`末尾 |

### 2.2 S2尾块与Softmax VF分支

| s2RealSize | 对应Softmax VF变体 | 说明 |
|------------|---------------------|------|
| 128 | `EQ_128_SCFA` | 满块对齐路径 |
| 127 - 65 | `GT_64_AND_LTE_128_SCFA` | 接近满块尾块，使用Select蒙版 |
| 64 | `GT_0_AND_LTE_64_SCFA` | 刚好分界线 |
| 63 - 1 | `GT_0_AND_LTE_64_SCFA` | 小尾块，使用半mask |
| 0 | 跳过Softmax | 空KV块 |

### 2.3 isUpdate × oriNRange 生成的6种VF Softmax组合

| VF文件 | isUpdate | oriNRange | 应用场景 |
|--------|----------|-----------|----------|
| `aligned128_update` | true | EQ_128 | 非首块，满128 |
| `aligned128_no_update` | false | EQ_128 | 首块，满128 |
| `unaligned128_update` | true | GT_64_AND_LTE_128 | 非首块，>64且<128 |
| `unaligned128_no_update` | false | GT_64_AND_LTE_128 | 首块，>64且<128 |
| `unaligned64_update` | true | GT_0_AND_LTE_64 | 非首块，≤64 |
| `unaligned64_no_update` | false | GT_0_AND_LTE_64 | 首块，≤64 |

### 2.4 subBlock非对称

- subBlock 0: `halfMRealSize = ceil(mRealSize / 2)`
- subBlock 1: `halfMRealSize = floor(mRealSize / 2)`
- **mRealSize为奇数时两侧不均**，需分别验证

### 2.5 IS_SPLIT_G下的M轴切分

- N1=128时G大小=128 → 跨2个AIC核心（每核处理64组）
- Vec0的`CalProcSize`进行4-way分配（2 AIC × 2 subBlock）
- sinks偏移量按4路计算

---

## 三、分核场景覆盖（Load Balancing）

### 3.1 分核算法各阶段

| # | 场景 | 触发条件 | 代码路径 |
|---|------|----------|----------|
| 1 | **all-zero KV短路** | 所有batch的`s1GBaseNum==0`或`s2BaseNum==0` | `BalanceSchedule`→单核心 |
| 2 | **单核心分配** | 总代价 ≤ 平均代价 | `usedCoreNum == 1` |
| 3 | **batch级分配** | 整batch代价在容忍度内 | `AssignByBatch` |
| 4 | **row级分配** | 整GS1行代价在容忍度内 | `AssignByRow` |
| 5 | **block级分配** | supportFd=true，单S2块分配 | `AssignByBlock`（仅CFA模式） |
| 6 | **强制分配** | supportFd=true且某核心0块 | `ForceAssign` |
| 7 | **跨核心KV行分裂** | `curKvSplitPart > 1` | 记录FD信息 |

### 3.2 容忍度边界

```
IsWithinTolerance(costLimit + lastBlockCost / FA_TOLERANCE_RATIO >= coreCache.cost + assignedCost)
```

- 刚好在边界上的batch/row → 跨核心分配
- 刚好在边界下的batch/row → 当前核心分配

### 3.3 N128 G-split特殊处理

- N1=128 → `isN128 = true`
- `aicCoreNum_` 减半 → 每个物理核心分配2个逻辑核心
- metadata输出2条目/物理核心（双双相同）

### 3.4 mBaseSize不同模式下的取值

| SOC | N1 | 模式 | mBaseSize | s2BaseSize |
|-----|----|------|-----------|------------|
| ASCEND950 | 128 | 任意 | `groupSize_` | 128 |
| ASCEND950 | 64 | SCFA | `1 * groupSize_` | 128 |
| ASCEND950 | 64 | CFA/SWA | `64/groupSize_ * groupSize_` | 128 |

### 3.5 代价计算的尾块（4×4矩阵）

| M轴\S2轴 | WIN_NORMAL | WIN_TAIL | CMP_NORMAL | CMP_TAIL |
|-----------|------------|----------|------------|----------|
| WIN_NORMAL | [0,0] | [0,1] | — | — |
| WIN_TAIL | [1,0] | [1,1] | — | — |
| CMP_NORMAL | — | — | [2,2] | [2,3] |
| CMP_TAIL | — | — | [3,2] | [3,3] |

代价公式：`cost = COST_WEIGHT_M(6) × align(m/16) + COST_WEIGHT_S2(10) × align(s2/64)`

---

## 四、Kernel内部分支场景

### 4.1 4-stage Pipeline状态机

| taskId%4 | Stage0: LoadQK+BMM1 | Stage1: Vec0 Dequant | Stage2: Vec1 Softmax+BMM2 | Stage3: Vec2 FlashUpdate |
|----------|---------------------|----------------------|----------------------------|--------------------------|
| 0 | ✓ | ✓ | — | — |
| 1 | ✓ | ✓ | Vec1+BMM2 | — |
| 2 | ✓ | ✓ | Vec1+BMM2 | Vec2 |
| 3+ | ✓ | ✓ | Vec1+BMM2 | Vec2 |

**Drain阶段（PRELOAD_NUM=3）：**

- `notLast` / `notLastTwoLoop` / `notLastThreeLoop` 控制3步preload排空
- `s2LoopLimit=0` → 仅dummy迭代

### 4.2 S2循环双阶段

```
s2LoopCount < oriKvLoopEndIdx → Win（原始KV）阶段 → ori_kv、oriBlockTable
s2LoopCount >= oriKvLoopEndIdx → Cmp（压缩KV）阶段 → cmp_kv、cmpBlockTable
```

| 组合 | 场景 |
|------|------|
| 仅有Win | SWA模式（cmpKvLoopEndIdx=0） |
| 仅有Cmp | 特殊无oriKV场景 |
| Win+Cmp | CFA/SCFA（需覆盖过渡时的isCmp标志切换） |

### 4.3 Vec0 KV加载与反量化路径

Vec0是8bit KV反量化的执行者，在AIV侧运行：

#### 4.3.1 两种处理模式

| 模式 | 触发条件 | 函数 | 说明 |
|------|----------|------|------|
| 普通KV | 非SCFA模式 或 Win阶段 | `ProcessNotSparseKv()` | 直接page-copy KV + 反量化 |
| 稀疏KV | SCFA模式 + Cmp阶段 | `ProcessSparseKv()` | 先按稀疏indices gather KV + 反量化 |

#### 4.3.2 ProcessNotSparseKv内部分支

```
外循环: s2 < s2End
  内循环: 每次处理 v0ProcessBase(16) 行
    1. CopyInKvNotSparse: GM→UB (PageAttention) 
    2. DequantKv: 8bit→BF16反量化
    3. CopyOutKvUb2Gm: UB→v0ResGm (NZ→ND)
```

尾批处理：最后一批 `dealNum < 16`

#### 4.3.3 ProcessSparseKv内部分支

```
每2个token一对处理：
  GetRealCmpS2Idx(token0) / GetRealCmpS2Idx(token1)
  ├── token0==-1 && token1==-1 → meetEnd, break
  ├── token1==-1 → 仅copy token0, meetEnd, break
  ├── 双token合并copy (640B合为一次DataCopyPad)
  └── 双token分离copy (keySrcStride异常时降级)
```

#### 4.3.4 IS_VEC_S2PHYADDR

- **true**：初始化阶段调用`GetKVPhyAddr`预计算稀疏KV物理地址数组，运行时使用`GetRealS2Addr()`直接查表
- **false**：运行时通过`GetRealCmpS2Idx()`从`cmpSparseIndices`动态查索引

#### 4.3.5 Vec0的CalProcSize多路分配

| 条件 | 分配方式 | 路数 |
|------|----------|------|
| `IS_SPLIT_G = true` | 4-way (2 AIC × 2 subBlock) | 4 |
| `IS_SPLIT_G = false` | 2-way (1 AIC × 2 subBlock) | 2 |

`procSize == 0` 时跳过处理。

### 4.4 反量化详细流（AntiquantVFImplFp8D448）

对每批16行KV，Nope(448)分4次处理：

| 循环 | Nope范围 | 元素数 | 说明 |
|------|----------|--------|------|
| 1 | 0-127 | 128 | 满块 |
| 2 | 128-255 | 128 | 满块 |
| 3 | 256-383 | 128 | 满块 |
| 4 | 384-447 | **64** | 尾块 |

每次循环的微操作：

```
1. Load 128×KV_T (FP8/HIF8) via LoadAlign<DIST_UNPACK4_B8>
2. Load 2×scale (FP8_E8M0) via LoadAlign<DIST_BRC_B16>，广播至128份
3. Cast KV_T→FP32 (ZEROING, NO_SAT, UNKNOWN rounding)
4. Cast FP32→BF16 (ZEROING, NO_SAT, CAST_RINT rounding)
5. Cast Scale: FP8_E8M0→BF16→FP32
6. Mul(bf16_nope, scale_bf16)
7. DeInterleave → NZ格式
8. Store → UB, blockStride=17, repeatStride=1
```

注意：Rope(64)元素由BF16直接透传，无需反量化。

### 4.5 BMM1双段Matmul

`IterateBmm1SCFA`与`sparse_flash_mla`的主要区别：

- D=512 被切分为两次MatmulK1，每次K=256
- 第一次：`outKFisrt=true` ⇒ L0C初始化（清零）
- 第二次：`outKFisrt=false` ⇒ L0C累加到已有结果
- Q也分两半加载，使用3个L1Q buffer

### 4.6 Flash Attention状态机

| s2LoopCount | sinks | 操作 | 函数 |
|-------------|-------|------|------|
| 0 | 有 | 从sinks初始化softmax状态 | `DuplicateSumWithR0` + `DataCopy` |
| 0 | 无 | 从R0初始化softmax状态 | 标准初始化 |
| >0, 非末块 | — | 累加不归一化 | `FlashUpdateNew` |
| 末块 | — | 累加后除以sum | `FlashUpdateLastNew` |
| 末块 (实为唯块) | — | 仅除以sum | `LastDivNew` |

### 4.7 Output初始化 (needInit)

```
if (S1 > S2 或 TND长度不匹配) → needInit = 1 → CleanOutput
```

- 按AIV核心数均分输出空间
- 各核心清零自己的部分

### 4.8 Layout组合

| Q Layout | KV Layout | 场景 |
|----------|-----------|------|
| BSND | PA_ND | 标准4D Q + PageAttention KV |
| TND | PA_ND | TND打包Q + PageAttention KV |

KV仅支持PA_ND（页表寻址），不支持BSND/TND。

### 4.9 可选Tensor的Null/非Null

| 参数 | 为Null时 | 非Null时 |
|------|----------|----------|
| `cmp_kv` | SWA模式 | CFA/SCFA模式 |
| `cmp_sparse_indices` | CFA模式 | SCFA模式 |
| `cu_seqlens_q` | 使用maxSeqlen（BSND） | 使用累计序列长度（TND） |
| `seqused_q` | 使用cu_seqlens或maxSeqlen | 使用实际序列长度 |
| `seqused_kv` | 使用maxSeqlen | 使用实际KV长度 |
| `ori_block_table` | 不可为null（PA_ND强制要求） | PageAttention页表 |
| `cmp_block_table` | SWA模式 | CFA/SCFA模式（PA_ND强制要求） |
| `sinks` | 不可为null（强制要求） | 初始化softmax状态 |

---

## 五、推荐测试场景矩阵

### 5.1 基本维度假定

| 编号 | 名称 | 关键参数 | 覆盖目标 |
|------|------|----------|----------|
| B-01 | SWA decode FP8 | B=1,S1=1,S2=8193,N1=64,quant=1 | SWA模式 + FP8反量化 |
| B-02 | SWA prefill FP8 | B=1,S1=8192,S2=8192,N1=64,quant=1 | SWA大prefill |
| B-03 | SWA HIF8 | B=1,S1=1,S2=8193,N1=64,quant=10 | HIFLOAT8反量化路径 |
| B-04 | SCFA decode N64 | B=48,S1=2,S2=8192,N1=64,K=1024,cmpRatio=4 | 标准量产SCFA |
| B-05 | SCFA prefill N128 | B=1,S1=8192,S2=131072,N1=128,K=1024,cmpRatio=4 | G-split + prefill |
| B-06 | CFA mode | B=1,S1=2,S2=8192,N1=64,cmpRatio=4 | CFA模板 |
| B-07 | SCFA decode K=512 | B=1,S1=2,S2=8192,N1=64,K=512,cmpRatio=4 | K=512 TopK |

### 5.2 尾块场景

| 编号 | 尾块类型 | 关键参数 | 覆盖目标 |
|------|----------|----------|----------|
| T-01 | S2尾块=127 | actualS2=(128×N+127) | GT_64_AND_LTE_128 |
| T-02 | S2尾块=65 | actualS2=(128×N+65) | GT_64_AND_LTE_128紧边界 |
| T-03 | S2尾块=64 | actualS2=(128×N+64) | GT_0_AND_LTE_64紧边界 |
| T-04 | S2尾块=63 | actualS2=(128×N+63) | GT_0_AND_LTE_64 |
| T-05 | S2尾块=1 | actualS2=(128×N+1) | 极小尾块 |
| T-06 | Vec0尾批 | s2RealSize=(16×N+8) | v0ProcessBase尾批 |
| T-07 | Nope反量化尾64 | — | AntiquantVFImpl第4次loop |
| T-08 | S1尾块 (SWA/CFA) | N1=64,gSize=64,S1=65 | s1RealSize=1 |
| T-09 | M尾块 (subBlock奇数) | mRealSize=奇数 | 不对称subBlock |
| T-10 | 双尾块 (S1G尾+S2尾) | mBaseSize非整除+s2BaseSize非整除 | 4×4代价矩阵 TAIL_TAIL |

### 5.3 分核场景

| 编号 | 分核类型 | 关键条件 | 覆盖目标 |
|------|----------|----------|----------|
| C-01 | all-zero KV | 所有batch的KV=0 | 单核心短路 |
| C-02 | 单核心 | 小负载 | usedCoreNum=1 |
| C-03 | batch级分配 | 多batch (B=48) | AssignByBatch |
| C-04 | row级分配 | 单batch内跨核心 | AssignByRow |
| C-05 | block级分配 | CFA模式 (supportFd) | AssignByBlock |
| C-06 | 强制分配 | core取0块 | ForceAssign |
| C-07 | N128 G-split | N1=128 | 双metadata槽 |
| C-08 | N64单slot | N1=64 | 单metadata槽 |

### 5.4 Kernel分支场景

| 编号 | 分支 | 条件 | 覆盖 |
|------|------|------|------|
| K-01 | ProcessNotSparseKv | SWA/CFA或Win阶段 | 普通KV+反量化 |
| K-02 | ProcessSparseKv | SCFA+Cmp阶段 | 稀疏KV+反量化 |
| K-03 | IS_VEC_S2PHYADDR=true | 向量化物理地址 | 预计算地址路径 |
| K-04 | s2LoopLimit=0 | 空KV | 无S2循环 |
| K-05 | s2LoopLimit=1 | 仅1个S2 | Stage0执行，无pipeline |
| K-06 | s2LoopLimit=2 | 2个S2 | Stage0+Stage1 |
| K-07 | s2LoopLimit=3 | 3个S2 | 首次Stage2(Vec2) |
| K-08 | s2LoopLimit≥4 | 4+个S2 | 完整4-S + drain |
| K-09 | ori→cmp过渡 | Win+Cmp | isCmp切换 |
| K-10 | 有sinks初始化 | sinks非null | 从sinks初始化softmax |
| K-11 | needInit | S1>S2或长度不匹配 | CleanOutput |
| K-12 | CSA-双token有效 | 两个tokenIdx都≥0 | 合并copy |
| K-13 | CSA-单token有效 | 仅1个tokenIdx≥0 | 分离copy |
| K-14 | CSA-双token无效 | 都=-1 | meetEnd break |
| K-15 | CSA-stride降级 | keySrcStride异常 | 降级分离copy |
| K-16 | FP8_E4M3FN | KV_DTYPE=FP8 | FP8 cast路径 |
| K-17 | HIFLOAT8 | KV_DTYPE=HIF8 | HIF8 cast路径 |
| K-18 | TND layout | Q_TND + KV_PA_ND | cuSeqlens offset |
| K-19 | BSND layout | Q_BSND + KV_PA_ND | 固定stride offset |

### 5.5 反量化专项场景

| 编号 | 描述 | 覆盖 |
|------|------|------|
| DQ-01 | per-tile FP8_E4M3FN (quant=1) | 标准反量化路径 |
| DQ-02 | HIFLOAT8 (quant=10) | HIF8反量化路径 |
| DQ-03 | Nope全零值 + scale正常 | 零nope反量化输出 |
| DQ-04 | Nope满值(0x3C) + scale最大 | nope最大值反量化 |
| DQ-05 | Nope满值 + scale=0 | scale零值处理 |
| DQ-06 | Nope随机 + scale随机 | 一般精密度验证 |
| DQ-07 | Rope BF16极值 → 透传 | rope通道无量化 |
| DQ-08 | 每行7个scale不同值 | 多tile scale差异 |

### 5.6 综合场景

| 编号 | 描述 |
|------|------|
| Z-01 | B=48 + SCFA decode + N1=64 + FP8 + S2尾块 — 量产综合 |
| Z-02 | B=1 + SCFA prefill + N1=128 + G-split — 全面prefill |
| Z-03 | B=1 + SWA + FP8 + S2尾块 — 纯窗口+量化 |
| Z-04 | B=1 + SWA + HIFLOAT8 — HIF8精度验证 |
| Z-05 | 多batch变长 + SCFA — padding场景 |
| Z-06 | S2尾块 + G-split + IS_VEC_S2PHYADDR — 多特征组合 |
| Z-07 | B=48 + SCFA + N1=64 + cmRatio=128 — 大压缩比 |
| Z-08 | B=1 + CFA + N1=64 — CFA密集压缩 |

---

## 六、用例设计Checklist

对每个用例确认覆盖以下维度：

- [ ] `kv_quant_mode` 的 **1 (FP8_E4M3FN)**
- [ ] `tile_size` 验证（固定64）
- [ ] `rope_head_dim` 验证（固定64）
- [ ] `dSizeVInput` = **640**（非512）
- [ ] `N1` 取 **64 或 128**（不取中间值）
- [ ] `N2` = **1**
- [ ] `actualS1Size` 的 **0 / 1 / <gSize / =gSize / >gSize**
- [ ] `actualS2OriSize` 的 **0 / <128 / =128 / >128**
- [ ] `actualS2CmpSize` 的 **0 / <128 / =128 / >128**
- [ ] `mRealSize` 的 **奇数/偶数**（subBlock对称性）
- [ ] `s2LoopLimit` 的 **0 / 1 / 2 / 3 / ≥4**（4-stage pipeline）
- [ ] `s2RealSize` 的 **128 / 65-127 / 1-64**（3种Softmax VF）
- [ ] `IS_SPLIT_G` = **true (N1=128) / false (N1=64)**
- [ ] `IS_VEC_S2PHYADDR` 的 **true / false**
- [ ] Template模式：**SWA / CFA / SCFA**
- [ ] Vec0处理模式：**ProcessNotSparseKv / ProcessSparseKv**
- [ ] Vec0分配：**2-way / 4-way**
- [ ] **sinks 必须提供**（FLOAT32）
- [ ] **oriMaskMode = 4 (BAND), cmpMaskMode = 3**
- [ ] **oriWinLeft = 127, oriWinRight = 0**
- [ ] **KV Layout = PA_ND**
- [ ] **没有 cmp_residual_kv**（residual恒为0）
- [ ] coreNum分配：**1核 / 部分核 / 全部核**
- [ ] cross-core row split：**有/无 FD记录**
- [ ] Nope反量化：**4次循环（3×128 + 1×64）**
- [ ] Vec0处理：**16行一批的尾批**
