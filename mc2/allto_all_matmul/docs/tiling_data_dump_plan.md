# AlltoAllMatmul TilingData Dump 方案（方案2：轻量异常 dump）

## 1. 背景与目标

通过 `aclrtGetArgsFromExceptionInfo` API，在 AlltoAllMatmul 算子执行异常时，从 device 侧 dump 出完整的 TilingData 二进制数据到 host 文件，便于离线解析定位问题。

参考文档：
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/runtimeapi/aclcppdevg_03_2140.html (aclrtGetArgsFromExceptionInfo)
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/runtimeapi/aclcppdevg_03_2139.html (aclrtBinarySetExceptionCallback)

## 2. 现状

pr_10103 已在 `allto_all_matmul_tiling.cpp` 中注册了异常回调 `AlltoAllMatmulExceptionImplWrapper`，调用 `Mc2ExceptionImplTmp` dump HCCL 通信寄存器信息（xnAddr/ckeAddr），但**未 dump TilingData**。

pr_10103 改动范围：
- `mc2/allto_all_matmul/op_host/op_tiling/allto_all_matmul_tiling.cpp`：注册 ExceptionDumpParseFunc
- `mc2/common/utils/mc2_exception_dump.h`：新增 Mc2ExceptionImplTmp 函数（dump xn/cke 寄存器）

## 3. Kernel Args 布局分析

### Kernel 签名（arch22/arch35 一致，12 个参数）

```cpp
__global__ __aicore__ void allto_all_matmul(
    GM_ADDR x1,           // param 1
    GM_ADDR x2,           // param 2
    GM_ADDR bias,         // param 3
    GM_ADDR x1_scale,     // param 4
    GM_ADDR x2_scale,     // param 5
    GM_ADDR comm_scale,   // param 6
    GM_ADDR x1_offset,    // param 7
    GM_ADDR x2_offset,    // param 8
    GM_ADDR y,            // param 9
    GM_ADDR all2all_out,  // param 10
    GM_ADDR workspaceGM,  // param 11
    GM_ADDR tilingGM      // param 12
)
```

Kernel 类型为 `KERNEL_TYPE_MIX_AIC_1_2`。

### Args 布局（A5/Ascend950）

pr_10103 的 `Mc2ExceptionImplTmp` 读 `args[0]`（offset=0）作为 hccl context 指针（`HcclCombinOpParam*`），在 A5 验证通过（无 FFTS 前缀偏移）。

推断 args 布局：

| args index | offset (bytes) | 内容         | 来源           |
|-----------|----------------|--------------|----------------|
| [0]       | 0              | hccl context | runtime 注入   |
| [1]       | 8              | x1           | kernel param 1 |
| [2]       | 16             | x2           | kernel param 2 |
| ...       | ...            | ...          | ...            |
| [11]      | 88             | workspaceGM  | kernel param 11|
| **[12]**  | **96**         | **tilingGM** | kernel param 12|

`tilingGM` 推测在 `devArgsPtr + 96`。

### TilingData 结构体大小

- `AlltoAllMatmulTilingData`（非量化）：见 `op_kernel/arch35/allto_all_matmul_tiling_data.h`
- `AlltoAllQuantMatmulTilingData`（量化）：同上文件
- 两者均 < 2048 字节，dump 2048 字节可覆盖所有场景

## 4. 改动方案

### 仅改 1 个文件

`mc2/allto_all_matmul/op_host/op_tiling/allto_all_matmul_tiling.cpp`

### 改动内容

在 `AlltoAllMatmulExceptionImplWrapper` 中，保留原有 `Mc2ExceptionImplTmp` 调用（dump xn/cke），追加 TilingData dump 逻辑：

1. 调用 `Mc2Exception::GetAclrtGetArgsFromExceptionInfoFunc()` 获取 `devArgsPtr`, `devArgsLen`
2. `OP_LOGE` 打印 `devArgsPtr` 和 `devArgsLen`（诊断 args 布局）
3. Dump 全部 args 指针（固定 256 字节）到 `args_info_AlltoAllMatmul.*.bin`——用于验证 args 布局
4. 从 `devArgsPtr + 96` 读 `tilingGM` 指针值（`aclrtMemcpy` 8 字节 D2H）
5. `aclrtMemcpy` 2048 字节从 `tilingGM` 到 host buffer
6. `DumpToFileWithSize` 写 `tiling_data_AlltoAllMatmul.*.bin`

### 复用的已有 helper（mc2_exception_dump.h）

| 函数 | 用途 |
|------|------|
| `GetAclrtGetArgsFromExceptionInfoFunc()` | 动态加载 aclrtGetArgsFromExceptionInfo |
| `GetAcldumpGetPathFunc()` | 动态加载 acldumpGetPath |
| `GenDumpFileNameWithPrefix(args, op, prefix)` | 生成 dump 文件名 |
| `DumpToFileWithSize(dir, name, id, buf, size)` | 写 dump 文件 |

无需新增公共函数，无需改动 kernel 侧。

## 5. 产出文件

异常触发后在 `acldumpGetPath(AIC_ERR_BRIEF_DUMP)` 路径下生成：

| 文件 | 内容 | 大小 |
|------|------|------|
| `args_info_AlltoAllMatmul.<streamId>.<taskId>.<ts>.bin` | 全部 args 指针（诊断） | 256B |
| `tiling_data_AlltoAllMatmul.<streamId>.<taskId>.<ts>.bin` | TilingData 原始二进制 | 2048B |
| `xn_addr_info_*` | HCCL Xn 寄存器（pr_10103 已有） | 32KB |
| `cke_addr_info_*` | HCCL CKE 寄存器（pr_10103 已有） | 1KB |

## 6. 解析 Dump 数据方案

### 6.1 解析脚本

**文件**: `mc2/tools/dump_analysis/parse_allto_all_matmul_tiling.py`

与已有的 `dump_analysis_megamoe.py` 同目录，复用代码风格。

#### 用法

```bash
python parse_allto_all_matmul_tiling.py <bin_file> [--quant] [--args] [--json]
```

| 模式 | 输入文件 | 功能 |
|------|---------|------|
| 默认 | `tiling_data_*.bin` | 按非量化 `AlltoAllMatmulTilingData`（1136B）解析 |
| `--quant` | `tiling_data_*.bin` | 按量化 `AlltoAllQuantMatmulTilingData`（1384B）解析 |
| `--args` | `args_info_*.bin` | 解析 kernel GM 地址指针（16×8B，验证 tilingGM offset） |
| `--json` | 任意 | 输出 JSON 格式（默认人类可读表格） |

#### 示例

```bash
# 解析非量化 tiling data
python parse_allto_all_matmul_tiling.py tiling_data_AlltoAllMatmul.0.1.20260819.bin

# 解析量化 tiling data
python parse_allto_all_matmul_tiling.py tiling_data_AlltoAllMatmul.0.1.20260819.bin --quant

# 解析 args_info（验证 tilingGM offset）
python parse_allto_all_matmul_tiling.py args_info_AlltoAllMatmul.0.1.20260819.bin --args

# JSON 格式输出
python parse_allto_all_matmul_tiling.py tiling_data_AlltoAllMatmul.0.1.20260819.bin --json
```

### 6.2 二进制布局（arch35，小端）

dump 出的 `tiling_data_*.bin` 是 `AlltoAllMatmulTilingData` 或 `AlltoAllQuantMatmulTilingData` 的原始内存映像。

#### 非量化 `AlltoAllMatmulTilingData`（sizeof=1136B）

| 字段 | 绝对偏移 | 大小 | 类型 | 说明 |
|------|---------|------|------|------|
| mc2InitTiling | 0 | 64 | opaque | HCCL init（可解析 Mc2InitTilingInner） |
| mc2CcTiling | 64 | 512 | opaque | HCCL cc（可解析 groupName/algConfig/opType） |
| **alltoAllMatmulTilingInfo** | **576** | **88** | - | **核心 tiling info** |
| ├ rankDim | 576 | 4 | u32 | 卡数 |
| ├ tileM | 580 | 4 | u32 | 头块M |
| ├ tileCnt | 584 | 4 | u32 | 头块数量 |
| ├ tailM | 588 | 4 | u32 | 尾块M |
| ├ tailCnt | 592 | 4 | u32 | 尾块数量 |
| ├ biasLen | 596 | 4 | u32 | bias大小 |
| ├ rankM | 600 | 4 | u32 | M轴 |
| ├ rankN | 604 | 4 | u32 | N轴 |
| ├ rankK | 608 | 4 | u32 | K轴 |
| ├ aicCoreNum | 612 | 4 | u32 | 核数 |
| ├ commLen | 616 | 8 | u64 | 通信空间 |
| ├ permuteLen | 624 | 8 | u64 | 重排空间 |
| ├ hcclDataType | 632 | 8 | u64 | hccl数据类型枚举 |
| ├ x1ScaleOptionalLen | 640 | 8 | u64 | scale空间 |
| ├ dynamicExtraSpace | 648 | 8 | u64 | 额外空间 |
| └ x1QuantDtype | 656 | 8 | u64 | 量化dtype |
| mc2MmV3TileTilingData | 664 | 236 | - | 头块matmul tiling |
| ├ TCubeTiling | 664 | 200 | 50×i32 | cube tiling |
| └ 扩展字段 | 864 | 36 | 9×u32 | mTailCnt等 |
| mc2MmV3TailTilingData | 900 | 236 | - | 尾块matmul tiling |

#### 量化 `AlltoAllQuantMatmulTilingData`（sizeof=1384B）

偏移 0-663 与非量化相同，之后：

| 字段 | 偏移 | 大小 | 说明 |
|------|------|------|------|
| mc2QuantMmTileTilingData | 664 | 360 | 头块量化matmul tiling |
| ├ Mc2QuantBatchMatmulV3DataParams | 664 | 128 | 32×u32 量化参数 |
| ├ TCubeTiling | 792 | 200 | 50×i32 cube tiling |
| ├ Mc2L2cacheTileParams | 992 | 24 | 6×u32 L2 cache |
| └ Mc2SlidingWindowParams | 1016 | 8 | 2×u32 滑动窗口 |
| mc2QuantMmTailTilingData | 1024 | 360 | 尾块量化matmul tiling |

### 6.3 解析内容（3 层）

#### 第 1 层 - HCCL 通信 tiling

**Mc2InitTilingInner**（offset 0, 64B），定义在 CANN `hccl_tiling_msg.h`:

| 字段 | 偏移 | 类型 | 说明 |
|------|------|------|------|
| version | 0 | u32 | tiling 版本 |
| mc2HcommCnt | 4 | u32 | 通信任务数 |
| offset[8] | 8 | 8×u32 | 各 cc tiling 偏移 |
| debugMode | 40 | u8 | 调试模式 |
| preparePosition | 41 | u8 | 准备位置 |
| queueNum | 42 | u16 | 队列数 |
| commBlockNum | 44 | u16 | 通信块数 |
| devType | 46 | u8 | 设备类型 |

**Mc2CcTilingInner**（offset 64, 512B），定义在 CANN `hccl_tiling_msg.h`:

| 字段 | 偏移(相对) | 类型 | 说明 |
|------|-----------|------|------|
| skipLocalRankCopy | 0 | u8 | 跳过本地rank拷贝 |
| skipBufferWindowCopy | 1 | u8 | 跳过窗口拷贝 |
| stepSize | 2 | u8 | 步长 |
| version | 3 | u8 | 版本 |
| commEngine | 13 | u8 | 通信引擎 |
| srcDataType | 14 | u8 | 源数据类型 |
| dstDataType | 15 | u8 | 目标数据类型 |
| groupName | 16 | char[128] | 通信组名 |
| algConfig | 144 | char[128] | 算法配置 |
| opType | 272 | u32 | 操作类型 |
| reduceType | 276 | u32 | 规约类型 |

#### 第 2 层 - 核心 tiling info

**AlltoAllMatmulTilingInfo**（offset 576, 88B），定义在 `allto_all_matmul_tiling_data.h`:

| 字段 | 偏移 | 类型 | 说明 |
|------|------|------|------|
| rankDim | 576 | u32 | 卡数 |
| tileM | 580 | u32 | 头块M |
| tileCnt | 584 | u32 | 头块数量 |
| tailM | 588 | u32 | 尾块M |
| tailCnt | 592 | u32 | 尾块数量 |
| biasLen | 596 | u32 | bias地址大小 |
| rankM | 600 | u32 | M轴大小（X1的M） |
| rankN | 604 | u32 | N轴大小（X2的N） |
| rankK | 608 | u32 | K轴大小（X1的K） |
| aicCoreNum | 612 | u32 | 核数 |
| commLen | 616 | u64 | 通信地址大小 |
| permuteLen | 624 | u64 | 重排空间大小 |
| hcclDataType | 632 | u64 | hccl通信枚举值 |
| x1ScaleOptionalLen | 640 | u64 | x1Scale空间 |
| dynamicExtraSpace | 648 | u64 | 动态量化额外空间 |
| x1QuantDtype | 656 | u64 | x1量化dtype |

#### 第 3 层 - matmul tiling

**非量化**: 2 × `Mc2MatMulV3TilingData`（offset 664/900, 各 236B）
- `TCubeTiling`（200B = 50 × int32）: usedCoreNum, M, N, Ka, Kb, singleCoreM/N/K, baseM/N/K, depthA1/B1, stepM/N, isBias, transLength, iterateOrder, shareMode, shareL1Size, shareL0CSize, shareUbSize, batchM/N, singleBatchM/N, stepKa/b, depthAL1CacheUB, depthBL1CacheUB, dbL0A/B/C, ALayoutInfoB/S/N/G/D, BLayoutInfoB/S/N/G/D, CLayoutInfoB/S1/N/G/S2, BatchNum, mxTypePara
- 扩展字段（36B = 9 × uint32）: mTailCnt, nTailCnt, kTailCnt, mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain, isHf32, aswWindowLen

**量化**: 2 × `Mc2QuantBatchMatmulV3TilingDataParams`（offset 664/1024, 各 360B）
- `Mc2QuantBatchMatmulV3DataParams`（128B = 32 × uint32）: batchA/B/C, batchA1-4, batchB1-4, batchC1-4, singleCoreBatch, isPerTensor, isPertoken, isDoubleScale, biasThreeDim, ubCalcM/N, needUbBuffer, realSingleCoreM/N, biasDtype, ubSize, isMClash, isNClash, groupSizeM/N/K
- `TCubeTiling`（200B）: 同上
- `Mc2L2cacheTileParams`（24B = 6 × uint32）: mTileCntL2, nTileCntL2, mTileBlock, nTileBlock, calOrder, isBasicTiling
- `Mc2SlidingWindowParams`（8B = 2 × uint32）: mTailTile, nTailTile

### 6.4 args_info 解析

`args_info_*.bin`（256B）解析前 16 个 uint64 指针：

| index | offset | 名称 | 说明 |
|-------|--------|------|------|
| [0] | 0 | hccl_context | runtime 注入 |
| [1] | 8 | x1 | kernel param 1 |
| [2] | 16 | x2 | kernel param 2 |
| [3] | 24 | bias | kernel param 3 |
| [4] | 32 | x1_scale | kernel param 4 |
| [5] | 40 | x2_scale | kernel param 5 |
| [6] | 48 | comm_scale | kernel param 6 |
| [7] | 56 | x1_offset | kernel param 7 |
| [8] | 64 | x2_offset | kernel param 8 |
| [9] | 72 | y | kernel param 9 |
| [10] | 80 | all2all_out | kernel param 10 |
| [11] | 88 | workspaceGM | kernel param 11 |
| **[12]** | **96** | **tilingGM** | kernel param 12 |
| [13-15] | 104-120 | (保留) | 额外指针 |

### 6.5 HCCL 数据类型枚举

| 值 | 名称 |
|----|------|
| 0 | HCCL_DATA_TYPE_INT8 |
| 4 | HCCL_DATA_TYPE_FLOAT16 |
| 5 | HCCL_DATA_TYPE_FLOAT32 |
| 8 | HCCL_DATA_TYPE_BF16 |
| 14 | HCCL_DATA_TYPE_FLOAT8_E5M2 |
| 15 | HCCL_DATA_TYPE_FLOAT8_E4M3FN |
| 19 | HCCL_DATA_TYPE_FLOAT4_E2M1 |

### 6.6 结构体大小汇总

| 结构体 | sizeof (bytes) | 定义位置 |
|--------|---------------|---------|
| Mc2InitTiling | 64 | CANN kernel_tiling.h |
| Mc2CcTiling | 512 | CANN kernel_tiling.h |
| TCubeTiling | 200 | CANN kernel_tiling.h |
| Mc2MatMulV3TilingData | 236 | repo mat_mul_tiling_data.h |
| Mc2QuantBatchMatmulV3DataParams | 128 | repo quant_batch_matmul_v3_tiling_data.h |
| Mc2L2cacheTileParams | 24 | 同上 |
| Mc2SlidingWindowParams | 8 | 同上 |
| Mc2QuantBatchMatmulV3TilingDataParams | 360 | 同上 |
| AlltoAllMatmulTilingInfo | 88 | repo allto_all_matmul_tiling_data.h |
| AlltoAllMatmulTilingData (非量化) | 1136 | 同上 |
| AlltoAllQuantMatmulTilingData (量化) | 1384 | 同上 |

## 7. 风险与验证

### 风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| tilingGmOffset=96 不正确 | 如有额外 FFTS 前缀，实际 offset 可能为 104 | args_info dump 用于诊断；日志打印 devArgsLen 可推算元素数 |
| tilingGM 指针为 0 或无效 | 异常可能发生在 tilingGM 未初始化前 | 代码中检查 tilingGM == 0 并跳过 |
| 仅支持 arch35 | pr_10103 已限制 Ascend950 | 与 pr_10103 一致，arch22 后续扩展 |

### 验证步骤

1. `bash build.sh` 构建算子包
2. 安装算子包到环境
3. `export ASCEND_GLOBAL_LOG_LEVEL=1`（开启 DEBUG 日志）
4. 触发 AlltoAllMatmul 异常
5. 检查日志中 `devArgsLen` 值（预期 104 = 13 × 8 字节，或 13）
6. 检查 `args_info_*.bin`，确认 args[12] 是否为有效 GM 地址
7. 检查 `tiling_data_*.bin`，按 AlltoAllMatmulTilingData 结构体解析验证字段值
8. 使用 `parse_allto_all_matmul_tiling.py` 脚本自动解析

## 8. 使用方式

### 触发条件

算子执行异常时自动触发（由 CANN runtime 调用注册的 ExceptionDumpParseFunc 回调）。

### 环境变量

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1        # 开启 DEBUG 日志查看 dump 路径
export ASCEND_SLOG_PRINT_TO_STDOUT=1    # 日志输出到终端（可选）
```

### dump 文件路径

由 `acldumpGetPath(AIC_ERR_BRIEF_DUMP)` 返回，通常为：
- `/var/log/npu/host_{pid}/dump/` 或
- CANN 安装目录下的 dump 路径

日志中会打印 `"Start to dump file. The dump path is ..."` 显示实际路径。

### 解析流程

1. 找到 dump 产生的 bin 文件（路径见日志）
2. 先用 `--args` 模式解析 `args_info_*.bin`，确认 tilingGM 在 args[12] 且为有效地址
3. 根据算子是量化还是非量化，选择默认或 `--quant` 模式解析 `tiling_data_*.bin`
4. 检查解析结果中 `alltoAllMatmulTilingInfo` 的字段值是否合理（rankDim/rankM/rankN/rankK 等）
5. 如需程序化处理，加 `--json` 输出 JSON 格式
