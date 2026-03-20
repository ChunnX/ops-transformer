# FlashAttn Tiling 固定值说明

## 概述

`flash_attn_tiling_regbase.cpp` 中的 `GetShapeAttrsInfo()` 和 `DoOpTiling()` 两个函数
当前以**写死方式**填充 `FlashAttentionScoreSimplifiedTilingData` 的所有字段，
用于直接验证 kernel 侧计算逻辑，无需经过完整的 shape 解析流程。

固定值来源于 `examples/test_aclnn_flash_attn.cpp` 的推理场景。

---

## Example 输入参数（test_aclnn_flash_attn.cpp）

| 变量           | 值       | 说明                                  |
|--------------|----------|-------------------------------------|
| `B`          | 1        | batch size                          |
| `N_q`        | 8        | Q 的 head 数（BNSD: qShape[1]）        |
| `N_kv`       | 2        | KV 的 head 数（BSND: kShape[2]），GQA  |
| `S_q`        | 128      | Q 的序列长度（BNSD: qShape[2]）          |
| `S_kv`       | 128      | KV 的序列长度（BSND: kShape[1]）         |
| `D`          | 64       | Q/K 的 head 维度（qShape[3]）           |
| `layoutQ`    | "BNSD"   | Q 布局                               |
| `layoutKv`   | "BSND"   | KV 布局（仅支持 BSND/TND/PA_ND/PA_Nz） |
| `layoutOut`  | "BNSD"   | 输出布局                              |
| `softmaxMode`| 0.0f     | 0 表示默认缩放 = 1/√D = 1/√64 = 0.125f |
| `maskMode`   | 0        | 无注意力掩码                            |
| `winLeft`    | 0        | 无滑动窗口                             |
| `winRight`   | 0        | 无滑动窗口                             |
| `returnSoftmaxLse` | 0  | 推理场景，不输出 softmax_lse            |
| `deterministic` | 0     | 非确定性模式                            |

---

## GetShapeAttrsInfo() 写死的 InputParamsRegbase 字段

### shape 维度类

| 字段            | 固定值 | 来源/含义                                              |
|---------------|--------|------------------------------------------------------|
| `bSize`       | 1      | batch size，对应 example `B=1`                        |
| `n2Size`      | 2      | KV head 数，对应 example `N_kv=2`（BSND: kShape[2]）  |
| `gSize`       | 4      | GQA 比例 = N_q/N_kv = 8/2 = 4                        |
| `t1Size`      | 128    | TND 时为 T1 维度；此处 = S_q = 128                    |
| `t2Size`      | 128    | TND 时为 T2 维度；此处 = S_kv = 128                   |
| `s1Size`      | 128    | Q 序列长度，对应 example `S_q=128`                    |
| `s2Size`      | 128    | KV 序列长度，对应 example `S_kv=128`                  |
| `alignedS2`   | 128    | ceil(128/16)×16 = 128，已 16 对齐                     |
| `dSize`       | 64     | Q/K head 维度，对应 example `D=64`                    |
| `dSizeV`      | 64     | V head 维度，= D = 64                                 |
| `dSizeRope`   | 0      | 无 RoPE 编码                                          |

### 缩放 / 精度类

| 字段          | 固定值  | 来源/含义                                              |
|-------------|---------|------------------------------------------------------|
| `scaleValue`| 0.125f  | softmaxMode=0 → 1/√D = 1/√64 = 0.125f               |
| `implMode`  | 0       | HIGH_PRECISION（高精度模式）                           |

### Layout 类

| 字段          | 固定值 | 来源/含义                          |
|-------------|--------|----------------------------------|
| `layoutType`| 3      | BNSD = 3（对应注释 "3: BNSD"）    |

> `layoutType` 编码：1 = BSH/BSND，2 = SBH，3 = BNSD，4 = TND

### 注意力窗口类

| 字段           | 固定值 | 来源/含义                              |
|--------------|--------|--------------------------------------|
| `preTokens`  | 65536  | 无掩码时设为大值，表示全量注意力         |
| `nextTokens` | 0      | 无因果掩码                             |

### GQA 类

| 字段             | 固定值 | 来源/含义                       |
|----------------|--------|-------------------------------|
| `isGqa`        | 1      | N_q(8) ≠ N_kv(2) → GQA 有效   |
| `headNumRatio` | 4      | N_q / N_kv = 8 / 2 = 4        |

### 输出控制类

| 字段                    | 固定值 | 来源/含义                                  |
|-----------------------|--------|------------------------------------------|
| `isSoftMaxLseEnable`  | 0      | returnSoftmaxLse=0，不输出 softmax_lse    |

### Dropout / Mask / PSE / PA（均无）

| 字段                          | 固定值 | 含义                    |
|-----------------------------|--------|------------------------|
| `needDropMaskOp`             | 0      | 无 dropout              |
| `isKvContinuous`             | 1      | 非 PA，KV 内存连续       |
| `attenMaskCompressMode`      | 1      | NONE（无掩码）           |
| `attenMaskShapeType`         | 0      | 默认                    |
| `attenMaskS1Size`            | 0      | 无掩码                  |
| `attenMaskS2Size`            | 0      | 无掩码                  |
| `pseType`                    | 0      | 无 PSE 位置编码          |
| `isActualSeqLengthsNull`     | 1      | cuSeqlensQ = nullptr    |
| `isActualSeqLengthsKVNull`   | 1      | cuSeqlensKv = nullptr   |
| `isActualSharedPrefixLenNull`| 1      | 无 KV 共享前缀           |
| `blockSize`                  | 0      | 非 PA                   |
| `blockTableDim2`             | 0      | 非 PA                   |
| `paBlockNumSum`              | 0      | 非 PA                   |
| `paLayoutType`               | 0      | 非 PA                   |
| `deqScaleFlag`               | 0      | 无量化                  |
| `deqScale2Flag`              | 0      | 无量化                  |
| `isQHasLeftPadding`          | 0      | 无 padding              |
| `isKVHasLeftPadding`         | 0      | 无 padding              |
| `ropeHeadSize`               | 0      | 无 rope                 |
| `prefixSeqInnerSize`         | 0      | 无 prefix               |

---

## DoOpTiling() 写死的基本块与多核参数

### 基本块（BasicBlock）

| 成员变量          | 固定值              | 含义                                    |
|---------------|---------------------|-----------------------------------------|
| `dBasicBlock`   | 64                  | D=64，每次 cube 处理 64 列（head 维度方向） |
| `dVBasicBlock`  | 64                  | Dv=64，V 的 head 维度方向基本块             |
| `dTemplateType` | `ALIGNED_64` (=64)  | D 对齐到 64，对应 kernel 模板参数           |
| `dVTemplateType`| `ALIGNED_64` (=64)  | Dv 对齐到 64                              |
| `s1BasicBlock`  | 128                 | S_q=128，每次处理 128 行 Q（S1 方向）       |
| `s2BasicBlock`  | 128                 | S_kv=128，每次处理 128 列 KV（S2 方向）     |

### MultiCoreParamsRegbase（多核调度参数）

| 字段                       | 计算方式                                        | 固定值（本例）       |
|--------------------------|-----------------------------------------------|---------------------|
| `s1OuterSize`             | ceil(S_q / s1BasicBlock) = ceil(128/128)       | **1**               |
| `totalSize`               | B × N_kv × G × s1OuterSize = 1×2×4×1          | **8**               |
| `coreNum`                 | min(totalSize, aivNum) = min(8, 32)            | **8**               |
| `splitFactorSize`         | ceil(totalSize / coreNum) = ceil(8/8)          | **1**（每核 1 任务） |
| `splitFactorTailSize`     | totalSize − (coreNum−1) × splitFactor = 8−7×1  | **1**               |
| `bnStartIdx[i]`           | i × splitFactor（i = 0..7）                    | {0,1,2,3,4,5,6,7,…} |
| `firstFullLoadS1OuterIdx` | 0                                              | **0**               |
| `splitCoreMode`           | 0（均匀分配模式）                               | **0**               |
| `sparseStartIdx[48]`      | 全零（无稀疏注意力）                            | {0, 0, …, 0}        |

> **aivNum 兜底**：若平台信息未能正常获取（`aivNum == 0`），硬编码路径默认使用 32 核计算。
> 实际 910B / 950 硬件的 AIV 核数均为 32，因此计算结果与真实硬件一致。

---

## DoOpTiling() 写死的 DropmaskParamsRegbase

无 dropout 场景（`needDropMaskOp=0`），kernel 不访问此结构体，**全部置零**。

| 字段                   | 固定值 | 含义                                                  |
|----------------------|--------|-----------------------------------------------------|
| `multiCoreFactorSize` | 0      | dropout 多核外循环因子；无 dropout → 0                |
| `baseUbCalSize`       | 0      | dropout 单次 UB 计算元素数；无 dropout → 0            |
| `multiCoreTotalSize`  | 0      | dropout 多核总循环次数；无 dropout → 0                |
| `shapeTotalSize`      | 0      | dropout mask 的总元素数（B×N×G×S1×S2）；无 dropout → 0|
| `dropMaskAddrOffset`  | 0      | dropout mask 在 workspace 中的地址偏移；无 dropout → 0|

> 参考来源：`flash_attention_score/op_host/arch35/flash_attention_score_tiling_dropmask.cpp`
> 中 `DoOpTiling` 的 `if (!needDropMaskOp) return GRAPH_PARAM_INVALID;` 分支——
> 无 dropout 时该分支直接返回，结构体全零即可。

---

## DoOpTiling() 写死的 InitOutputParams

| 字段                        | 计算方式 / 来源                                  | 固定值（本例） |
|---------------------------|-----------------------------------------------|--------------|
| `totalOutputSize`          | B × N_q × S_q × D = 1×8×128×64（元素数）       | **65536**    |
| `totalSoftMaxLseOutputSize`| returnSoftmaxLse=0，不输出 softmax_lse         | **0**        |
| `singleCoreSize`           | totalOutputSize / coreNum = 65536 / 8（元素数） | **8192**     |
| `needInit`                 | 0：推理场景 kernel 直接写满输出，无需提前清零    | **0**        |
| `isOneN`                   | 0：N_q=8 ≠ 1，不走单 head 特殊路径              | **0**        |

> **字段含义**：  
> - `needInit=1` 用于 kernel 在计算前将输出张量初始化为 0（PA 或分块写出场景）；  
>   推理场景每个输出位置均被写入，无需初始化，故置 0。  
> - `totalOutputSize` / `singleCoreSize` 单位为**元素数**（非字节数）。  
> - `totalSoftMaxLseOutputSize` 为 softmax_lse 输出张量的总元素数；
>   推理场景该输出为 nullptr，置 0。

---

## TilingKey 构成（来自 flash_attn_tiling_basic.cpp::GetTilingKey）

| 字段              | 取值                  | 说明                   |
|-----------------|-----------------------|----------------------|
| `implMode`      | 0 (HIGH_PRECISION)    | 高精度模式              |
| `layout`        | 3 (BNSD)              | Q/Out 布局            |
| `s1Type`        | 128                   | s1BasicBlock          |
| `s2Type`        | 128                   | s2BasicBlock          |
| `dType`         | 64 (ALIGNED_64)       | dTemplateType         |
| `dvType`        | 0 (NONALIGNED)        | dVTemplateType（dType == dvType 时取 0）|
| `hasAtten`      | 0                     | 无掩码                 |
| `isPA`          | 0                     | 非 PA                 |
| `isSoftmaxLse`  | 0                     | 不输出 lse            |
| `regbase`       | 1                     | regbase 模板           |

---

## 注意事项

1. 以上所有写死值均位于 `flash_attn_tiling_regbase.cpp` 的 `GetShapeAttrsInfo()` 和
   `DoOpTiling()` 函数中，通过注释 `// 写死参数` 标识。
2. 恢复动态 tiling 时，需将 `GetShapeAttrsInfo()` 中写死的 shape/attr 部分替换为
   从 context 的 shape/attr 中动态解析（参照 `flash_attention_score` 算子实现）。
3. 该文档仅供 kernel 侧验证阶段使用，**不依赖** `flash_attention_score` 算子文件。
