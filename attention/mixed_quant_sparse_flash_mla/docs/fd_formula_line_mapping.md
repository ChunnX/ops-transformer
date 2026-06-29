# Flash Decode Formula And Line Mapping

本文整理 `mixed_quant_sparse_flash_mla` 中 Flash Decode(FD) 相关阶段的数学公式、实现入口和关键行号。

> 说明：`ComputeScaleValue_VF`、`ReduceFinalRes_const_VF` 等公共 VF API 的实现不在本算子目录内。本文按当前 kernel 调用点、FD 数据契约和 golden 公式整理其数学含义。

## 1. 总体 Attention 公式

完整 attention 语义：

```text
score = Q @ K^T * softmax_scale
O = softmax(score) @ V
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [README.md](../README.md#L18) | 18-20 | 算子整体公式 |
| [tests/pytest/mixed_quant_sparse_flash_mla_golden.py](../tests/pytest/mixed_quant_sparse_flash_mla_golden.py#L198) | 198-204 | golden 中直接 softmax 路径 |

## 2. Vec1: 局部 Softmax 状态

对每个 S2 split/tile，Vec1 负责计算局部 softmax 状态：

```text
score_i = Q @ K_i^T * softmax_scale
m_i = max(score_i)
s_i = sum(exp(score_i - m_i))
```

多 S2 loop 时使用 online-softmax 更新：

```text
m_new = max(m_old, m_tile)
scale_old = exp(m_old - m_new)
s_new = scale_old * s_old + sum(exp(score_tile - m_new))
```

sinks 场景初始化：

```text
m_init = max(score, sinks)
s_init = sum(exp(score - m_init)) + exp(sinks - m_init)
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L833) | 833-845 | 首个 S2 loop，无 sinks，调用 `ProcessVec1Vf` |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L848) | 848-872 | sinks 初始化 max/sum |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L875) | 875-885 | 有历史状态时调用 `ProcessVec1Vf` 更新 |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L909) | 909 | `SFAUpdateExpSumAndExpMax` 更新 exp/sum/max |
| [tests/pytest/mixed_quant_sparse_flash_mla_golden.py](../tests/pytest/mixed_quant_sparse_flash_mla_golden.py#L225) | 225-240 | golden 中 online-softmax 公式 |
| [tests/pytest/mixed_quant_sparse_flash_mla_golden.py](../tests/pytest/mixed_quant_sparse_flash_mla_golden.py#L285) | 285-292 | golden 中 sinks softmax 公式 |

## 3. Vec2: 局部输出 `p_i`

Vec2 负责把局部 softmax 结果乘 V，并在最后一个 S2 loop 得到归一化的局部输出。

online 输出更新：

```text
o_new = scale_old * o_old + exp(score_tile - m_new) @ V_tile
```

最后归一化：

```text
p_i = o_i / s_i
```

其中 `p_i` 是后续 FD 归约读取的 staged partial O。

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L975) | 975-976 | 首个 S2 loop 直接拷贝 BMM2 结果 |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L980) | 980-981 | 非最后 S2 loop 调 `FlashUpdateNew` |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L984) | 984-985 | 最后 S2 loop 调 `FlashUpdateLastNew` |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L993) | 993-994 | 单 S2 loop 时 `LastDivNew` 做最终除 `sum` |
| [tests/pytest/mixed_quant_sparse_flash_mla_golden.py](../tests/pytest/mixed_quant_sparse_flash_mla_golden.py#L236) | 236-242 | golden 中 `cur_o`、`cur_attn_out`、最终除 `row_sum` |

## 4. FA Producer: 写 FD Staging

当当前 `(B, N2, M)` 行被 S2 跨核切分时，FA 阶段在最后一个 S2 loop 写入 FD staging。

写入内容：

```text
slot_i stores:
  max_i = m_i
  sum_i = s_i
  partialO_i = p_i
```

slot 索引：

```text
workspaceIdx = firstFdDataWorkspaceIdx + s2SplitIdx
```

max/sum 布局：

```text
每个 M 行广播为 8 个 float
maxOffset/sumOffset = workspaceIdx * (MAX_SUM_BYTES / sizeof(float))
                   + stagingMOffset * 8
```

partial O 布局：

```text
outOffset = stagingMOffset * dSizeV
每行写 dSizeV 个 float
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L172) | 172-175 | FA 写 staging 的 `workspaceIdx` |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L178) | 178-187 | FA 写 staging 的 M offset，含 split-G 偏移 |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L912) | 912-914 | 最后 S2 loop 写 max/sum staging |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L921) | 921-926 | staging 三段区域：partialO、max、sum |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L938) | 938-942 | max/sum 广播成每行 8 float |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L946) | 946-952 | 写 max/sum 到 GM |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L996) | 996-1012 | 写 normalized partial O 到 GM |

## 5. FD Metadata 生成

AICPU metadata 算子记录需要 FD 归约的任务。

记录 FD 任务：

```text
fdBN2Idx       = splitRes.bN2End[prevCore]
fdMIdx         = splitRes.gS1End[prevCore]
fdWorkspaceIdx = preFdDataNum
fdS2SplitNum   = curKvSplitPart
fdMSize        = current logical M rows
```

`fdMSize` 计算：

```text
fdMSize =
  if splitS1GIdx is last S1G block:
      s1Size * groupSize - splitS1GIdx * mBaseSize
  else:
      mBaseSize
```

FD 按 AIV 分 M：

```text
totalFDLoad = sum(fdS2SplitNum[i] * fdMSize[i])
averageLoad = ceil(totalFDLoad / aivCoreNum)
curFDVectorNum = max(1, floor(fdS2SplitNum * fdMSize / averageLoad))
curAveMSize = ceil(fdMSize / curFDVectorNum)
curFDVectorNum = ceil(fdMSize / curAveMSize)

fdMStart[core] = vid * curAveMSize
fdMNum[core] =
  if vid is not last:
      curAveMSize
  else:
      fdMSize - vid * curAveMSize
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp](../../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp#L1020) | 1020-1037 | 判断是否需要记录 FD 信息 |
| [../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp](../../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp#L1040) | 1040-1061 | 记录 FD 任务、计算 `fdMSize` |
| [../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp](../../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp#L1068) | 1068-1102 | 维护 `firstFdDataWorkspaceIdx`、`preFdDataNum`、`curKvSplitPart` |
| [../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp](../../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp#L1140) | 1140-1165 | FD 任务按 AIV 切 M |
| [../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp](../../mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.cpp#L1257) | 1257-1270 | 写 FD metadata 到输出 tensor |

## 6. FD Consumer: 跨 Split 权重

FD 阶段读取每个 split 的 `m_i/s_i`，计算跨 split 合并权重。

```text
M = max_i(m_i)
G = sum_i(exp(m_i - M) * s_i)
w_i = exp(m_i - M) * s_i / G
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1022) | 1022-1025 | FD 入口 `ProcessFlashDecode` |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1036) | 1036-1040 | FD UB tensor：accumulatedO、lseExp、blockMax、blockSum |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1060) | 1060-1067 | 按 `FD_REDUCE_CHUNK_ROWS=16` 分 chunk |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1069) | 1069-1080 | 读取每个 split 的 max/sum |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1082) | 1082-1083 | `ComputeScaleValue_VF` 计算跨 split scale/权重 |

## 7. FD Consumer: 归约输出

FD 阶段读取每个 split 的 staged partial O，并按上一阶段权重累加。

```text
O = sum_i(w_i * p_i)
```

其中：

```text
p_i = partialO_i = local normalized output of split i
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1086) | 1086 | 当前 chunk 的累加输出 `chunkAccumO` |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1087) | 1087-1104 | 遍历 split，读 partial O 并归约 |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1096) | 1096-1098 | 读取 staged partial O |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1101) | 1101-1102 | `ReduceFinalRes_const_VF` 累加 `w_i * p_i` |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1111) | 1111 | FD 最终写 attention_out |

## 8. FD 输出地址

FD 输出地址由 batch/token base、M block base 和 FD 子任务内部 M 偏移组成。

```text
seqOffset =
  if layout is TND:
      cuSeqlensQ[bn2Idx]
  else:
      bn2Idx * s1Size

attentionOutOffset =
    seqOffset * n2GDv
  + mIdx * n2GDv
  + mStartIdx * dSizeV
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1027) | 1027-1035 | FD 输出 offset 公式 |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1108) | 1108-1111 | 用 FD offset 构造 `RunInfo` 并写出 |

## 9. 同步边界

FD 必须在所有 FA staging 写完后再读取。

```text
FA phase writes staging
SyncAll()
FD phase reads staging and reduces
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h#L404) | 404-417 | 无 FA work 的 core 仍进入 `SyncAll` 后执行 FD |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h#L555) | 555-558 | 正常 FA phase 结束后的 `SyncAll` 和 FD 调用 |

## 10. Workspace 和 UB 大小

staging 每个 split slot 的布局：

```text
partialO bytes/slot = stagingM * dTemplateAlign64 * sizeof(float)
max bytes/slot      = stagingM * 8 * sizeof(float)
sum bytes/slot      = stagingM * 8 * sizeof(float)
```

host 侧 workspace 估算：

```text
fdStagingMSize = isSplitG ? 128 : gSize
fdStagingSlotNum = isSplitG ? aicNum / 2 : aicNum

s2SplitStagingPerSlot =
    fdStagingMSize * 512 * sizeof(float) * MAX_S2_SPLIT_NUM
  + fdStagingMSize * 8 * sizeof(float) * MAX_S2_SPLIT_NUM * 2
```

FD UB 大小：

```text
maxSumTotal = workspaceNum * FD_REDUCE_CHUNK_ROWS * 8 * sizeof(float)
lseExpSize = FD_REDUCE_CHUNK_ROWS * 8 * sizeof(float)
accumOutSize = mNum * dTemplateAlign64 * sizeof(T)
partialOSize = FD_REDUCE_CHUNK_ROWS * dTemplateAlign64 * sizeof(T)
```

对应位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/util_regbase.h](../op_kernel/arch35/util_regbase.h#L26) | 26-30 | `MAX_S2_SPLIT_NUM`、广播元素数、FD chunk 行数 |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L153) | 153-160 | kernel staging partialO/max/sum bytes |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L163) | 163-169 | staging slot 数 |
| [op_host/mixed_quant_sparse_flash_mla_tiling.cpp](../op_host/mixed_quant_sparse_flash_mla_tiling.cpp#L616) | 616-621 | host workspace 大小公式 |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h#L1279) | 1279-1292 | FD UB buffer 大小公式 |

## 11. Metadata 字段索引

FD kernel 读取的 metadata 字段：

| 字段 | 含义 | 定义位置 |
| --- | --- | --- |
| `FD_CORE_ENABLE_INDEX` | 当前 AIV 是否执行 FD | [op_kernel/mixed_quant_sparse_flash_mla_metadata.h:44](../op_kernel/mixed_quant_sparse_flash_mla_metadata.h#L44) |
| `FD_BN2_IDX_INDEX` | FD 任务 BN2 索引 | [op_kernel/mixed_quant_sparse_flash_mla_metadata.h:45](../op_kernel/mixed_quant_sparse_flash_mla_metadata.h#L45) |
| `FD_M_IDX_INDEX` | FD 任务 M block 索引 | [op_kernel/mixed_quant_sparse_flash_mla_metadata.h:46](../op_kernel/mixed_quant_sparse_flash_mla_metadata.h#L46) |
| `FD_WORKSPACE_IDX_INDEX` | FD 读取的起始 workspace slot | [op_kernel/mixed_quant_sparse_flash_mla_metadata.h:47](../op_kernel/mixed_quant_sparse_flash_mla_metadata.h#L47) |
| `FD_WORKSPACE_NUM_INDEX` | 需要归约的 split 数 | [op_kernel/mixed_quant_sparse_flash_mla_metadata.h:48](../op_kernel/mixed_quant_sparse_flash_mla_metadata.h#L48) |
| `FD_M_START_INDEX` | 当前 AIV 处理的 M 起点 | [op_kernel/mixed_quant_sparse_flash_mla_metadata.h:49](../op_kernel/mixed_quant_sparse_flash_mla_metadata.h#L49) |
| `FD_M_NUM_INDEX` | 当前 AIV 处理的 M 行数 | [op_kernel/mixed_quant_sparse_flash_mla_metadata.h:50](../op_kernel/mixed_quant_sparse_flash_mla_metadata.h#L50) |

kernel 解析位置：

| 文件 | 行号 | 说明 |
| --- | --- | --- |
| [op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h](../op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h#L617) | 617-632 | `ParseFdRunInfo` 读取 FD metadata |

