# S2 Split 暂存区 & ProcessFlashDecode 实现文档

## 总体目标

为 QSMLA 算子实现 S2 轴跨核切分（S2 Split）时的中间结果暂存与 Flash Decode 归约功能。当一个 M 行对应的 S2 范围被 metadata 分配到多个核计算时，各核将 partial 结果写入 GM 暂存区，最后由 FD 规约核读取并合并为最终输出。

## 背景原理

### S2 跨核切分

metadata 算子按负载均衡将 S2 轴切分给多个 AIC/AIV 核组。一个 `(batch, head, s1_block)` 的 S2 范围可能被切成多段，每段由不同核处理。切分点处的 M 行需要跨核归约（Flash Decode）。

### Flash Attention 归约数学

对于 N 个 partial 结果 `(O_i, max_i, sum_i)`：

1. `global_max = max(max_0, ..., max_{N-1})`
2. `correction_i = exp(max_i - global_max)`
3. `global_sum = Σ correction_i * sum_i`
4. `global_O = Σ correction_i * O_i`
5. `final_O = global_O / global_sum`

### Staging 布局

每个核的暂存区布局为连续区域（非交错）：

```
[split0_output | split1_output | all_max | all_sum]
```

- `OUTPUT_BYTES = 64 * 512 * 2`（每 split 的 partial O，bf16/fp16）
- `FLOAT_BYTES = 64 * 4`（每 split 的 max 或 sum，fp32）
- 每核总占用 = `OUTPUT_BYTES * MAX_S2_SPLIT_NUM + FLOAT_BYTES * MAX_S2_SPLIT_NUM * 2`
- 每核 `MAX_S2_SPLIT_NUM = 2`，总占用 132,096 字节

## 提交记录

| Commit | 描述 |
|--------|------|
| `23b6ceb57` | 修复 `ApplyS2MetadataRange` 单位不匹配，新增 `isS2Split` 标记 |
| `dc17dc839` | S2 split 暂存区：workspace 分配、staging 写入、isS2Split 分支处理 |
| `239d0c2f4` | 实现 ProcessFlashDecode 归约逻辑，重构 staging 布局为连续区域，修复 FD buffer 溢出 |

## 修改文件清单

### `op_kernel/arch35/util_regbase.h`

- `regbaseutil` 命名空间新增常量：
  - `MAX_S2_SPLIT_NUM = 2`
  - `S2_SPLIT_STAGING_OUTPUT_BYTES = 64 * 512 * 2`
  - `S2_SPLIT_STAGING_FLOAT_BYTES = 64 * 4`
- `RunParamStr` 和 `COMMON_RUN_INFO` 各增加：
  - `bool isS2Split = false` — 当前 M 行的 S2 是否被跨核切分
  - `int64_t s2SplitIdx = 0` — 当前核在该 M 行 S2 切分中的序号

### `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h`

**成员变量**：
- `__gm__ uint8_t *s2SplitStagingBase = nullptr` — 当前核的暂存区 GM 基址

**`InitMMResBuf`**：
- 计算 staging 偏移 = V0 区域 + `STAGING_PER_CORE * aicIdx`
- 设置 `s2SplitStagingBase` 指向 workspace 中对应位置

**`Init`**：
- `InitMMResBuf` 后调用 `vecBlock.InitS2SplitStaging(s2SplitStagingBase)` 传给 vecBlock

**`ApplyS2MetadataRange`**：
- 调用处将 metadata 的 block 级 `s2StartIdx`/`s2EndIdx` 乘以 `constInfo.s2BaseSize` 转为 token 级
- 函数内部计算 `isS2Split`：
  - `hasPrevCore`：首任务看 `s2StartPoint > 0`，非首任务恒为 true
  - `hasNextCore`：末任务看 `s2EndPoint < totalLen`，非末任务恒为 true
  - `isS2Split = hasPrevCore || hasNextCore`
- 早返回路径显式设置 `isS2Split = false`

**主循环**：
- `s2SplitIdxCounter` 跟踪 split 序号（带 `< 2` 越界保护）
- `s2NoNeedCalc` 为 true 时显式重置 `isS2Split = false`
- `SetRunInfo` 传递 `isS2Split` 和 `s2SplitIdx` 到 RunInfo

**`ProcessFlashDecode()`**：
- 新增方法，从 `fdRunInfo` 构建 `RunInfo`（`halfMRealSize`、`attentionOutOffset` 等）
- `attentionOutOffset` 根据 `fdRunInfo.bn2Idx`/`mIdx`/`mStartIdx` 计算
- 调用 `vecBlock.ProcessFlashDecode(fdRunInfo, runInfo, constInfo, bmm2Buffers.Get())`

**`ProcessMainLoop`**：
- 主循环结束后 `SyncAll()` 确保所有核写完暂存区
- `fdRunInfo.coreEnable > 0` 时调用 `ProcessFlashDecode()`

### `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h`

**新增 include**：
- `vf/vf_flash_decode.h`（common 已有的 FD VF 函数）

**新增成员**：
- `__gm__ uint8_t *s2SplitStagingBase = nullptr`
- `TBuf<> fdPartialBuf` — 16KB，partial O 的 OUTPUT_T→fp32 转换暂存
- `TBuf<> fdLseExpBuf` — 512B，FD LSE exp 值暂存

**新增方法**：
- `InitS2SplitStaging(__gm__ uint8_t *base)` — 设置 staging 基址

**`ProcessVec2` 修改**（`s2LoopCount == s2LoopLimit` 最后一轮）：
- `isS2Split == true` 分支：
  - `s2LoopCount == 0` 时循环初始化 `sumUb[i] = 1.0f`（所有行）
  - 跳过 `LastDivNew`，保留未归一化的 partial O
  - Cast → DataCopyPad 写 output 到 staging 的 output 区域
  - DataCopy 写 sum/max 到 staging 的 sum/max 区域
- `isS2Split == false` 分支：原逻辑不变

**`ProcessFlashDecode` 实现**：
- 使用 common 已有的 `FaVectorApi::ComputeScaleValue_8_FD` + `FaVectorApi::ReduceFinalRes_VF`
- 流程：
  1. 从 GM 读取所有 split 的 max/sum 到 `commonTBuf`（allMaxUb + allSumUb）
  2. 初始化 `stage2OutBuf`（accumulated O）为 0
  3. 每 8 行一个 chunk：
     - `ComputeScaleValue_8_FD` → 输出 correction 到 `fdLseExpBuf`，全局 sum 到 `softmaxSumBuf[0]`
     - 每个 split：读 partial O → Cast 到 fp32（`fdPartialBuf`）→ `ReduceFinalRes_VF` 累加
  4. `LastDivNew` 归一化 → `CopyOutAttentionOut` 写出

**Buffer 分配调整**：
- `commonTBuf`：512B → 1024B（FD 需要存 allMax + allSum）
- 新增 `fdPartialBuf`：`8 * dTemplateAlign64 * sizeof(T)` = 16KB
- 新增 `fdLseExpBuf`：512B

**`SCFABlockVecDummy`**：
- 同步增加空 `InitS2SplitStaging` 和 `ProcessFlashDecode`

### `op_host/mixed_quant_sparse_flash_mla_tiling.cpp`

- workspace 增加 S2 split staging 空间：
  ```
  s2SplitStagingPerCore = M_BASE_SIZE * D_SIZE * VEC_RES_ELEM_SIZE * MAX_S2_SPLIT_NUM
      + M_BASE_SIZE * FLOAT_ELEM_SIZE * MAX_S2_SPLIT_NUM * 2
  workspaceSize += s2SplitStagingPerCore * aicNum
  ```
- 每核 ≈ 132KB，36 核总计 ≈ 4.5MB

## 关键设计决策

### 1. metadata 单位转换

metadata 算子写入的 `FA_S2_START_INDEX`/`FA_S2_END_INDEX` 是 **block 级索引**（每次 +1 对应一个 `s2BaseSize=128` tokens）。kernel 侧 `ComputeS2LoopInfo` 产出的 `s2LineStartIdx`/`s2LineOriEndIdx` 是 **token 级索引**。调用 `ApplyS2MetadataRange` 前需乘以 `s2BaseSize` 统一。

### 2. Staging 布局选择连续区域

交错式（每 split: output+max+sum）写入简单但 FD 读取时需要跳跃访问。连续区域（all outputs | all max | all sum）允许 FD 一次性 DataCopy 所有 split 的 max/sum 到 UB，减少 GM 访问次数。

### 3. 使用 common 已有 VF 函数

不自写 VF 逻辑，复用 `FaVectorApi::ComputeScaleValue_8_FD`（计算全局 max、correction、全局 sum）和 `FaVectorApi::ReduceFinalRes_VF`（累加 correction × partial_O）。每次处理 8 行。

### 4. Buffer 复用策略

| Buffer | 来源 | FD 用途 |
|--------|------|---------|
| `stage2OutBuf` | Vec2 输出 | accumulated O |
| `commonTBuf` | 通用临时 | allMax + allSum（扩至 1024B） |
| `softmaxMaxBuf[0]` | softmax | globalMaxReduce |
| `softmaxSumBuf[0]` | softmax | globalSum |
| `fdLseExpBuf` | FD 专用 | LSE exp correction 值（新增 512B） |
| `fdPartialBuf` | FD 专用 | partial O fp32 转换暂存（新增 16KB） |
| `stage1OutQue[0]` | Vec1 输出 | partial O GM 读取中转 |
| `sinksBuf` | sinks 输入 | dummy sink（learnableSink=false） |

## 后续工作

1. **FD 完成标志与 GM polling**：当前用 `SyncAll()` 做全局 barrier，后续可能需要改为 GM polling 机制（各计算核写完成标志，规约核 polling）
2. **LSE 输出**：`lse = globalMax + log(globalSum)` 尚未实现
3. **多 chunk 支持验证**：当前 `mSize` 最大 64，按 8 行切 chunk 最多 8 轮，需要验证 `commonTBuf` 和 `fdLseExpBuf` 在更大 `mSize` 下是否足够
4. **sinks 初始化**：`ProcessVec2` 中 `isS2Split` 分支的 `s2LoopCount == 0` 场景，sinks 初始化逻辑尚未对齐非 split 路径
5. **编译验证**：所有改动尚未编译测试
