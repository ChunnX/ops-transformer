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

### 涉及 2 个文件

| 文件 | 改动 |
|------|------|
| `mc2/common/utils/mc2_exception_dump.h` | 新增通用 `Mc2DumpTilingData` 函数 |
| `mc2/allto_all_matmul/op_host/op_tiling/allto_all_matmul_tiling.cpp` | 调用通用函数，删除算子专属 dump 代码 |

### 4.1 通用 dump 函数

在 `mc2/common/utils/mc2_exception_dump.h` 中新增 `Mc2DumpTilingData`：

```cpp
inline void Mc2DumpTilingData(aclrtExceptionInfo *args, const char *op,
                              uint32_t tilingGmArgIdx, uint32_t argsOffset = 0U)
```

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| args | `aclrtExceptionInfo *` | 异常信息指针（回调入参） |
| op | `const char *` | 算子名（用于日志和文件名） |
| tilingGmArgIdx | `uint32_t` | tilingGM 在 kernel args 中的索引（从 0 开始） |
| argsOffset | `uint32_t` | FFTS 前缀偏移字节数（默认 0，MegaMoe 等 mix 算子需传 8） |

#### 内部逻辑

1. 调用 `GetAclrtGetArgsFromExceptionInfoFunc()` 获取 `devArgsPtr`, `devArgsLen`
2. 计算 `tilingGmOffset = argsOffset + tilingGmArgIdx * 8`，并校验 `tilingGmOffset + 8 <= devArgsLen`
3. Dump `min(devArgsLen, 256)` 字节 args 指针到 `args_info_<op>.*.bin`
4. 从 `devArgsPtr + tilingGmOffset` 读 `tilingGM` 指针值（D2H 8 字节）
5. 校验 `tilingGM != 0`
6. `aclrtMemcpy` 2048 字节从 `tilingGM` 到 host buffer
7. `DumpToFileWithSize` 写 `tiling_data_<op>.*.bin`

#### 各 mc2 算子参数对照

不同算子的 kernel 参数数量不同，tilingGM 的 args index 也不同：

| 算子 | kernel 参数数 | tilingGmArgIdx | argsOffset | 调用示例 |
|------|-------------|----------------|------------|---------|
| AlltoAllMatmul | 12 | 12 | 0 | `Mc2DumpTilingData(args, "AlltoAllMatmul", 12U, 0U)` |
| AllGatherMatmulV2 | 11 | 11 | 0 | `Mc2DumpTilingData(args, "AllGatherMatmulV2", 11U, 0U)` |
| MatmulAlltoAll | 11 | 11 | 0 | `Mc2DumpTilingData(args, "MatmulAlltoAll", 11U, 0U)` |
| MatmulReduceScatterV2 | 10 | 10 | 0 | `Mc2DumpTilingData(args, "MatmulReduceScatterV2", 10U, 0U)` |
| MegaMoe | - | - | 8 | `Mc2DumpTilingData(args, "MegaMoe", <idx>U, 8U)` |

### 4.2 AlltoAllMatmul 接入

在 `allto_all_matmul_tiling.cpp` 的 `AlltoAllMatmulExceptionImplWrapper` 中，保留原有 `Mc2ExceptionImplTmp` 调用（dump xn/cke），替换为通用函数调用：

```cpp
inline void AlltoAllMatmulExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "AlltoAllMatmul");
    constexpr uint32_t TILING_GM_ARG_IDX = 12U;
    constexpr uint32_t ARGS_OFFSET = 0U;
    Mc2Exception::Mc2DumpTilingData(args, "AlltoAllMatmul", TILING_GM_ARG_IDX, ARGS_OFFSET);
}
```

### 4.3 其他 mc2 算子接入方式

其他 mc2 算子只需在各自的 `ExceptionImplWrapper` 回调中调用 `Mc2DumpTilingData`，传入正确的 `tilingGmArgIdx` 即可，无需重复实现 dump 逻辑：

```cpp
// all_gather_matmul_v2 (11 参数，tilingGM 是第 11 个参数)
Mc2Exception::Mc2DumpTilingData(args, "AllGatherMatmulV2", 11U, 0U);

// matmul_allto_all (11 参数)
Mc2Exception::Mc2DumpTilingData(args, "MatmulAlltoAll", 11U, 0U);

// matmul_reduce_scatter_v2 (10 参数)
Mc2Exception::Mc2DumpTilingData(args, "MatmulReduceScatterV2", 10U, 0U);
```

### 4.4 复用的已有 helper（mc2_exception_dump.h）

| 函数 | 用途 |
|------|------|
| `GetAclrtGetArgsFromExceptionInfoFunc()` | 动态加载 aclrtGetArgsFromExceptionInfo |
| `GetAcldumpGetPathFunc()` | 动态加载 acldumpGetPath |
| `GenDumpFileNameWithPrefix(args, op, prefix)` | 生成 dump 文件名 |
| `DumpToFileWithSize(dir, name, id, buf, size)` | 写 dump 文件 |

无需改动 kernel 侧。

## 5. 产出文件

异常触发后在 `acldumpGetPath(AIC_ERR_BRIEF_DUMP)` 路径下生成（`<op>` 为算子名，如 `AlltoAllMatmul`）：

| 文件 | 内容 | 大小 |
|------|------|------|
| `args_info_<op>.<streamId>.<taskId>.<ts>.bin` | 全部 args 指针（诊断） | min(devArgsLen, 256)B |
| `tiling_data_<op>.<streamId>.<taskId>.<ts>.bin` | TilingData 原始二进制 | 2048B |
| `xn_addr_info_<op>.*` | HCCL Xn 寄存器（pr_10103 已有） | 32KB |
| `cke_addr_info_<op>.*` | HCCL CKE 寄存器（pr_10103 已有） | 1KB |

dump 文件固定 2048B（`MAX_DUMP_TILING_SIZE`），而实际结构体大小通常小于此值（如非量化 1136B、量化 1384B），尾部为 padding（device 上结构体之后的内存），解析时忽略即可。

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
| tilingGmArgIdx 不正确 | 各算子参数数量不同，需逐一确认 | 参见 4.1 节参数对照表；args_info dump 可验证 |
| FFTS 前缀偏移 | 部分 mix 算子 args[0] 不是 hccl context 而是 FFTS 指针 | `argsOffset` 参数支持传入偏移（如 MegaMoe 传 8） |
| tilingGM 指针为 0 或无效 | 异常可能发生在 tilingGM 未初始化前 | 代码中检查 tilingGM == 0 并跳过 |
| args 长度不足 | `tilingGmOffset + 8 > devArgsLen` | 代码中校验边界并跳过 |
| 仅支持 arch35 | pr_10103 已限制 Ascend950 | 与 pr_10103 一致，arch22 后续扩展 |

### 验证步骤

1. `bash build.sh` 构建算子包
2. 安装算子包到环境
3. `export ASCEND_GLOBAL_LOG_LEVEL=1`（开启 DEBUG 日志）
4. 触发 AlltoAllMatmul 异常
5. 检查日志中 `devArgsLen` 和 `tilingGmArgIdx` 值（AlltoAllMatmul 预期 idx=12, offset=96）
6. 检查 `args_info_*.bin`，确认 args[12] 是否为有效 GM 地址
7. 检查 `tiling_data_*.bin`，按 AlltoAllMatmulTilingData 结构体解析验证字段值
8. 使用 `parse_allto_all_matmul_tiling.py` 脚本自动解析
9. 使用 `--verify` 模式与 host 参考文件逐字节对比（详见第 9 节）

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
3. 脚本自动检测量化/非量化模式，直接解析 `tiling_data_*.bin`
4. 检查解析结果中 `alltoAllMatmulTilingInfo` 的字段值是否合理（rankDim/rankM/rankN/rankK 等）
5. 如需程序化处理，加 `--json` 输出 JSON 格式

## 9. 数据正确性验证方案

### 9.1 原理

异常时 dump 的 `tiling_data_*.bin` 与 host 侧 `PostTiling()` 写入 context 的 tiling data 是同一块内存（host 写入 → device 读取 → 异常时 dump 回来）。因此可以通过对比 host 侧参考文件与 dump 文件来验证 dump 链路的正确性。

### 9.2 host 侧参考文件生成

在 `PostTiling()` 中通过环境变量 `MC2_TILING_DUMP_REF_PATH` 控制，将 tiling data 原始二进制写入指定文件：

**涉及文件**（3 个 PostTiling 函数，各覆盖一种场景）：

| 文件 | 场景 | 结构体 | sizeof |
|------|------|--------|--------|
| `allto_all_fp_matmul_tiling_base.cpp` | 非量化 | `AlltoAllMatmulTilingData` | 1136B |
| `allto_all_kc_quant_matmul_tiling_base.cpp` | KC 量化 | `AlltoAllQuantMatmulTilingData` | 1384B |
| `allto_all_mx_quant_matmul_tiling_base.cpp` | MX 量化 | `AlltoAllQuantMatmulTilingData` | 1384B |

每个 PostTiling 函数末尾追加：

```cpp
const char *dumpRefPath = std::getenv("MC2_TILING_DUMP_REF_PATH");
if (dumpRefPath != nullptr) {
    FILE *fp = std::fopen(dumpRefPath, "wb");
    if (fp != nullptr) {
        std::fwrite(outTilingData, sizeof(<TilingDataStruct>), 1, fp);
        std::fclose(fp);
        OP_LOGI(opName_, "Tiling reference dumped to %s (%zu bytes).", dumpRefPath, sizeof(<TilingDataStruct>));
    }
}
```

### 9.3 性能影响

**不影响生产环境性能**：

- 正常运行（不设环境变量）：仅多一次 `getenv` 调用（纳秒级），在 host 侧 tiling 阶段可忽略
- 调试时（设环境变量）：执行一次 `fwrite`（1-2KB），约几十微秒，且 tiling 只在下发算子时执行一次，不在 kernel 执行热路径上
- tiling 是 host 侧逻辑，执行频率远低于 kernel 执行，即使写文件也不影响 NPU 计算性能

可安全保留在生产代码中。

### 9.4 验证流程

**步骤 1：正常执行时生成参考文件**

```bash
# 设置环境变量，指定参考文件路径
export MC2_TILING_DUMP_REF_PATH=/tmp/tiling_ref.bin

# 跑 AlltoAllMatmul（正常执行，不需要触发异常）
# PostTiling 会将 tiling data 写入 /tmp/tiling_ref.bin
# 日志中会打印 "Tiling reference dumped to /tmp/tiling_ref.bin (1136 bytes)"
```

**步骤 2：触发异常，拿到 dump 文件**

用相同输入参数跑一次（触发异常），得到 `tiling_data_*.bin`（2048B，含 padding）。

**步骤 3：逐字节对比**

```bash
python parse_allto_all_matmul_tiling.py tiling_data_*.bin --verify /tmp/tiling_ref.bin
```

输出示例（一致时）：
```
[INFO]   Verify: dump vs host reference
[INFO]   dump 文件: 2048 bytes (含 padding)
[INFO]   参考文件: 1136 bytes (结构体实际大小)
[INFO]   结果: PASS - 全部 1136 字节完全一致
[INFO]   dump 文件尾部 912 字节为 padding（不在对比范围内）
```

输出示例（不一致时）：
```
[ERROR]   结果: FAIL - 3/1136 字节不一致
[ERROR]   首个不一致偏移: 580 (0x244)
[ERROR]   所属字段: AlltoAllMatmulTilingInfo (offset 576-664)
[ERROR]     offset  580 (0x244) [AlltoAllMatmulTilingInfo]: dump=0x00, ref=0x01
```

### 9.5 对比范围

| 项目 | dump 文件 | 参考文件 | 对比方式 |
|------|----------|---------|---------|
| 大小 | 2048B（固定） | 1136B 或 1384B（结构体实际大小） | 只对比参考文件长度范围内的字节 |
| 内容 | 结构体 + padding | 结构体 | 结构体部分应完全一致 |
| padding | 尾部 912B 或 664B | 无 | 不在对比范围内 |

### 9.6 验证原理

```
host PostTiling()                    device kernel                      host 异常回调
─────────────                       ──────────────                     ──────────────
localTilingData_                    tilingGM 指向                      aclrtGetArgsFromExceptionInfo
  ↓ memcpy_s                        device 上的                        → devArgsPtr
outTilingData                       tiling data buffer                 → 读 args[12] 得 tilingGM
  ↓ fwrite (环境变量控制)           ↑                                  → aclrtMemcpy 2048B
tiling_ref.bin                      同一块内存                         tiling_data_*.bin
                                    (host 写入 → device 读取)          (异常时 dump 回 host)

对比: tiling_ref.bin  vs  tiling_data_*.bin (前 N 字节)
      ↑ host 侧写入的原始数据        ↑ device 侧 dump 回来的数据
      如果一致 → dump 链路正确（args offset 正确、tilingGM 指针正确、aclrtMemcpy 正确）
```

# MC2 算子 Workspace 空间规划 + Runtime Info + 异常 Dump 设计方案

## 1. 背景与目标

为更好定位 MC2 通信算子（AllGatherMatmulV2、AlltoAllMatmul、MatmulReduceScatterV2）执行异常问题，需要：

1. **在 TilingData 中加入 workspace 空间规划**：记录 workspace 分段地址、大小及排序
2. **在 workspace 中申请额外空间记录运行时信息**：每个 core 记录执行阶段信息（64KB = 128×512B）
3. **异常时 dump workspace 跟随 tiling data 一起落盘**：复用现有 ExceptionDumpParseFunc 机制

### 设计原则

- **仅 A5 平台**（Ascend950 / DAV_3510）
- **零性能影响**：正常执行路径开销可忽略
- **复用现有机制**：基于已有 Mc2DumpTilingData / Mc2ExceptionImplTmp 扩展

---

## 2. 现状分析

### 2.1 三个算子 Workspace 分配模式

| 算子 | WorkspaceSize | 分段顺序 |
|------|--------------|---------|
| **AllGatherMatmulV2** | `libApiWS + nd2NzWorkLen + gatherLen + biasLen` | libApi → nd2Nz(A5无) → gather → bias |
| **AlltoAllMatmul** | `libApiWS + commLen + permuteLen + biasLen` | libApi → commOut → permuteOut → bias |
| **MatmulReduceScatterV2** | `libApiWS + mmResultLen + (a2a ? mmResultLen : 0)` | libApi → mmResult → recvBuf(a2a) |

### 2.2 现有异常 Dump 状态

| 算子 | ExceptionDumpParseFunc | Mc2ExceptionImplTmp (xn/cke) | Mc2DumpTilingData | Mc2DumpWorkspace |
|------|:---:|:---:|:---:|:---:|
| **AllGatherMatmulV2** | 已有 | 已有 | **缺失** | **缺失** |
| **AlltoAllMatmul** | 已有 | 已有 | 已有 (tilingGmArgIdx=12) | **缺失** |
| **MatmulReduceScatterV2** | **缺失** | **缺失** | **缺失** | **缺失** |

### 2.3 Kernel 参数索引（含框架隐式 context arg[0]）

| 算子 | workspaceGM 索引 | tilingGM 索引 |
|------|:---:|:---:|
| **AllGatherMatmulV2** | 10 | 11 |
| **AlltoAllMatmul** | 11 | 12 |
| **MatmulReduceScatterV2** | 9 | 10 |

> 框架在 kernel args 数组首位注入 HcclCombinOpParam 指针作为 arg[0]，因此所有显式参数索引 +1。

---

## 3. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Workspace 内存布局                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  [段0 LIB_API] [段1 业务段...] [段2 业务段...] ... [段N RUNTIME_INFO(64KB)] │
│                                                  128×512B                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Host 侧                           Kernel 侧                                │
│  ┌──────────────┐                  ┌───────────────────────────────────┐   │
│  │GetWorkspace  │                  │ Init:                             │   │
│  │Size:         │                  │  读 WorkspaceLayoutInfo           │   │
│  │ 1.计算各段   │──TilingData────▶│  找到 RUNTIME_INFO 段 offset      │   │
│  │ 2.追加64KB   │                  │  计算自身 slot 地址               │   │
│  │ 3.填Layout   │                  │  写入 magicNum + coreId          │   │
│  └──────────────┘                  │                                   │   │
│                                    │ 执行中:                           │   │
│  异常时:                           │  阶段切换时更新                    │   │
│  ┌──────────────┐                  │  execTurn/Position/commPhase      │   │
│  │Mc2DumpTiling │                  │  (单次写 8B + DCI 刷缓存)         │   │
│  │Data (含Layout│                  └───────────────────────────────────┘   │
│  │+RuntimeInfo) │                                                         │
│  │Mc2DumpWork   │                  异常触发:                               │
│  │spaceContent  │◀────────────────│ D2H 拷贝所有段含 RuntimeInfo         │
│  └──────────────┘                  落盘文件                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 新增结构体定义

### 4.1 Workspace 段类型与布局信息

**文件**: `mc2/common/op_kernel/mc2_tiling_struct.h`（kernel 侧）
**文件**: `mc2/common/op_host/op_tiling/mc2_tiling_struct.h`（host 侧）

#### Kernel 侧结构体（纯 C++ 结构体）

```cpp
namespace Mc2Tiling {

constexpr uint32_t MAX_WORKSPACE_SEGMENTS = 8U;

// Workspace 段类型
enum WorkspaceSegType : uint8_t {
    WS_SEG_LIB_API       = 0,   // 系统库预留空间
    WS_SEG_ND2NZ         = 1,   // ND2NZ 格式转换空间
    WS_SEG_GATHER        = 2,   // AllGather 通信结果空间
    WS_SEG_GATHER_SCALE1 = 3,   // AllGather scale1 空间 (FP8 perblock)
    WS_SEG_GATHER_SCALE  = 4,   // AllGather scale 空间
    WS_SEG_COMM_OUT      = 5,   // AlltoAll 通信结果空间
    WS_SEG_PERMUTE_OUT   = 6,   // AlltoAll 重排空间
    WS_SEG_BIAS          = 7,   // bias 空间
    WS_SEG_MM_RESULT     = 8,   // matmul 结果暂存空间
    WS_SEG_RECV_BUF      = 9,   // ReduceScatter 接收缓冲 (a2a 路径)
    WS_SEG_COMM_INT8     = 10,  // int8 通信 workspace
    WS_SEG_DYNAMIC_QUANT = 11,  // 动态量化临时空间
    WS_SEG_RUNTIME_INFO  = 12,  // 运行时信息段 (64KB)
    WS_SEG_RESERVED      = 255,
};

// 单个段描述 (24 bytes)
struct WorkspaceSegInfo {
    uint64_t offset;      // 相对 workspace 基地址的偏移 (字节)
    uint64_t size;        // 该段大小 (字节)
    uint8_t  type;        // WorkspaceSegType
    uint8_t  reserved[7]; // 对齐
};

// Workspace 布局信息 (208 bytes)
struct WorkspaceLayoutInfo {
    uint64_t totalSize;                                // workspace 总大小
    uint32_t segCount;                                 // 实际段数
    uint32_t reserved;                                 // 对齐
    WorkspaceSegInfo segments[MAX_WORKSPACE_SEGMENTS]; // 段数组 (按 offset 升序)
};

} // namespace Mc2Tiling
```

#### Host 侧注册（TILING_DATA_DEF 宏）

```cpp
BEGIN_TILING_DATA_DEF(WorkspaceSegInfo)
TILING_DATA_FIELD_DEF(uint64_t, offset);
TILING_DATA_FIELD_DEF(uint64_t, size);
TILING_DATA_FIELD_DEF(uint8_t, type);
TILING_DATA_FIELD_DEF_ARR(uint8_t, 7, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WorkspaceSegInfoOp, WorkspaceSegInfo);

BEGIN_TILING_DATA_DEF(WorkspaceLayoutInfo)
TILING_DATA_FIELD_DEF(uint64_t, totalSize);
TILING_DATA_FIELD_DEF(uint32_t, segCount);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
TILING_DATA_FIELD_DEF_ARR(WorkspaceSegInfo, MAX_WORKSPACE_SEGMENTS, segments);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WorkspaceLayoutInfoOp, WorkspaceLayoutInfo);
```

### 4.2 Runtime Info 结构体

**文件**: `mc2/common/op_kernel/mc2_tiling_struct.h`（kernel 侧，同一文件追加）

```cpp
namespace Mc2Tiling {

constexpr uint32_t RUNTIME_INFO_CORE_MAX = 128U;       // 最大核数
constexpr uint32_t RUNTIME_INFO_PER_CORE_SIZE = 512U;  // 每核 512 字节
constexpr uint32_t RUNTIME_INFO_TOTAL_SIZE =
    RUNTIME_INFO_CORE_MAX * RUNTIME_INFO_PER_CORE_SIZE; // 64KB

// 通信/计算阶段枚举 (各算子共用)
enum RuntimePhase : uint8_t {
    RT_PHASE_INIT           = 0,
    RT_PHASE_COMM_PREPARE   = 1,   // 通信准备 (AllGather/AlltoAll prepare)
    RT_PHASE_COMM_WAIT      = 2,   // 通信等待 (hccl.Wait)
    RT_PHASE_COMM_PROCESS   = 3,   // 通信执行中
    RT_PHASE_COMPUTE_LOCAL  = 4,   // 本卡计算
    RT_PHASE_COMPUTE_GATHER = 5,   // 远端数据计算
    RT_PHASE_TRANSPOSE      = 6,   // 转置
    RT_PHASE_REDUCE         = 7,   // 归约
    RT_PHASE_FINALIZE       = 8,   // 结束清理
    RT_PHASE_RESERVED       = 255,
};

// 每个 core 的运行时信息 (512 字节)
struct alignas(512) RuntimeInfoPerCore {
    uint32_t magicNum;        // 4B  魔数 0x4D433252 ("MC2R")
    uint16_t coreId;          // 2B  core ID (C核: 0~aicCoreNum-1, V核: aicCoreNum~total-1)
    uint8_t  execTurn;        // 1B  执行轮次 (从 0 开始递增)
    uint8_t  execPosition;    // 1B  执行位置 (当前 tile/turn 索引)
    uint8_t  commPhase;       // 1B  当前阶段 (RuntimePhase)
    uint8_t  reserved[503];   // 503B 预留
}; // 512 bytes

} // namespace Mc2Tiling
```

#### 字段说明

| 字段 | 大小 | 说明 |
|------|:---:|------|
| `magicNum` | 4B | `0x4D433252`，校验该 slot 是否被有效写入 |
| `coreId` | 2B | 核编号：C核从 0 开始，V核从 `aicCoreNum` 开始 |
| `execTurn` | 1B | 执行轮次：每次进入新的大阶段递增 |
| `execPosition` | 1B | 执行位置：当前 tile/turn 的索引 |
| `commPhase` | 1B | 当前阶段：`RuntimePhase` 枚举值 |
| `reserved` | 503B | 预留，后续可扩展 |

#### Core 编排方式

```
Slot 0     Slot 1    ...  Slot (C-1)   Slot C    ...  Slot 127
┌──────┐  ┌──────┐       ┌──────┐    ┌──────┐       ┌──────┐
│C核 0 │  │C核 1 │  ...  │C核C-1│    │V核 0 │  ...  │V核V-1│
└──────┘  └──────┘       └──────┘    └──────┘       └──────┘
←── aicCoreNum 个 C核 ──→←── aivCoreNum 个 V核 ──→
```

- **C 核 (AIC)**: `slotIdx = GetBlockIdx()`
- **V 核 (AIV)**: `slotIdx = aicCoreNum + GetBlockIdx()`

---

## 5. Kernel 侧 Mc2RuntimeInfo 辅助类

**文件**: `mc2/common/op_kernel/mc2_runtime_info.h`（新增）

```cpp
#ifndef MC2_RUNTIME_INFO_H
#define MC2_RUNTIME_INFO_H

#include "mc2_tiling_struct.h"

#if defined(__CCE_AICORE__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Mc2Kernel {

class Mc2RuntimeInfo {
public:
    __aicore__ inline Mc2RuntimeInfo() = default;

    __aicore__ inline void Init(GM_ADDR workspaceGM,
                                Mc2Tiling::WorkspaceLayoutInfo *wsLayout,
                                uint32_t aicCoreNum)
    {
        runtimeInfoBase_ = nullptr;
        slotIdx_ = 0;

        if (wsLayout == nullptr) {
            return;
        }

        for (uint32_t i = 0; i < wsLayout->segCount && i < Mc2Tiling::MAX_WORKSPACE_SEGMENTS; i++) {
            if (wsLayout->segments[i].type == Mc2Tiling::WS_SEG_RUNTIME_INFO) {
                runtimeInfoBase_ = reinterpret_cast<__gm__ Mc2Tiling::RuntimeInfoPerCore *>(
                    workspaceGM + wsLayout->segments[i].offset);
                break;
            }
        }

        if (runtimeInfoBase_ == nullptr) {
            return;
        }

        if ASCEND_IS_AIC {
            slotIdx_ = GetBlockIdx();
        } else {
            slotIdx_ = aicCoreNum + GetBlockIdx();
        }

        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->magicNum = 0x4D433252U;
        mySlot->coreId = static_cast<uint16_t>(slotIdx_);
        mySlot->execTurn = 0;
        mySlot->execPosition = 0;
        mySlot->commPhase = static_cast<uint8_t>(Mc2Tiling::RT_PHASE_INIT);

        FlushCache();
    }

    __aicore__ inline void UpdatePhase(uint8_t turn, uint8_t position, uint8_t phase)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execTurn = turn;
        mySlot->execPosition = position;
        mySlot->commPhase = phase;
        FlushCache();
    }

    __aicore__ inline void UpdatePhase(uint8_t phase)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->commPhase = phase;
        FlushCache();
    }

private:
    __aicore__ inline void FlushCache()
    {
        GlobalTensor<uint8_t> cacheLine;
        cacheLine.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(&runtimeInfoBase_[slotIdx_]),
                                  Mc2Tiling::RUNTIME_INFO_PER_CORE_SIZE);
        DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(cacheLine);
    }

    __gm__ Mc2Tiling::RuntimeInfoPerCore *runtimeInfoBase_ = nullptr;
    uint32_t slotIdx_ = 0;
};

} // namespace Mc2Kernel
#endif // __CCE_AICORE__
#endif // MC2_RUNTIME_INFO_H
```

### 性能影响分析

| 维度 | 开销 | 说明 |
|------|------|------|
| Init 调用 | 1 次 / core | 写 8 字节 + 1 次 DCI，约 ~100 cycle |
| UpdatePhase 调用 | ~5-10 次 / core | 每次写 3-8 字节 + 1 次 DCI，约 ~100 cycle |
| 总额外开销 | < 0.01% | 相对于 matmul/通信的百万级 cycle 可忽略 |
| 内存开销 | +64KB | workspace 固定增加 64KB |
| 同步开销 | 无 | 每个 core 只写自己的 slot，无核间同步 |

---

## 6. TilingData 结构体修改

### 6.1 AllGatherMatmulV2

**文件**: `all_gather_matmul_v2/op_kernel/all_gather_matmul_tiling_data.h`

```cpp
struct AllGatherMatmulTilingDataV2 {
    // ... 现有字段不变 ...
    Mc2MatMulV3TilingData mc2MmV3TailTilingData;
    WorkspaceLayoutInfo workspaceLayout;  // 新增，放在末尾
};

struct AllGatherMatmulTilingDataFp8 {
    // ... 现有字段不变 ...
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams quantBmmv3TailTiling;
    WorkspaceLayoutInfo workspaceLayout;  // 新增，放在末尾
};
```

### 6.2 AlltoAllMatmul

**文件**: `allto_all_matmul/op_kernel/arch35/allto_all_matmul_tiling_data.h`

```cpp
struct AlltoAllMatmulTilingData {
    // ... 现有字段不变 ...
    Mc2MatMulV3TilingData mc2MmV3TailTilingData;
    WorkspaceLayoutInfo workspaceLayout;  // 新增
};

struct AlltoAllQuantMatmulTilingData {
    // ... 现有字段不变 ...
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams mc2QuantMmTailTilingData;
    WorkspaceLayoutInfo workspaceLayout;  // 新增
};
```

### 6.3 MatmulReduceScatterV2

**文件**: `matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_v2_c_tiling.h`

```cpp
struct MatmulReduceScatterV2TilingData {
    // ... 现有字段不变 ...
    Mc2MatMulV3TilingData mC2Mmv3TailTilingData;
    WorkspaceLayoutInfo workspaceLayout;  // 新增
};

struct QuantBatchMatmulV3ReduceScatterTilingData {
    // ... 现有字段不变 ...
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams quantBmmV3TailTiling;
    WorkspaceLayoutInfo workspaceLayout;  // 新增
};
```

---

## 7. Host 侧 BuildWorkspaceLayout

三个算子的 TilingBase 中新增 `BuildWorkspaceLayout()` 方法，在 `GetWorkspaceSize()` 中调用。

### 7.1 AllGatherMatmulV2

**文件**: `all_gather_matmul_v2/op_host/op_tiling/all_gather_matmul_tiling_base_v2.h` + `.cpp`

```cpp
// .h 新增
WorkspaceLayoutInfo workspaceLayout_{};
void BuildWorkspaceLayout();

// .cpp
void AllGatherMatmulTilingBase::BuildWorkspaceLayout()
{
    uint64_t offset = 0;
    uint32_t idx = 0;

    workspaceLayout_.segments[idx++] = {offset, libApiWorkSpaceSize_, WS_SEG_LIB_API, {}};
    offset += libApiWorkSpaceSize_;

    if (npuArch_ != NpuArch::DAV_3510 && rcsCfg_.nd2NzWorkLen > 0) {
        workspaceLayout_.segments[idx++] = {offset, rcsCfg_.nd2NzWorkLen, WS_SEG_ND2NZ, {}};
        offset += rcsCfg_.nd2NzWorkLen;
    }
    if (rcsCfg_.gatherLen > 0) {
        workspaceLayout_.segments[idx++] = {offset, rcsCfg_.gatherLen, WS_SEG_GATHER, {}};
        offset += rcsCfg_.gatherLen;
    }
    if (biasLen_ > 0) {
        workspaceLayout_.segments[idx++] = {offset, biasLen_, WS_SEG_BIAS, {}};
        offset += biasLen_;
    }

    // Runtime Info 段 (64KB)
    workspaceLayout_.segments[idx++] = {offset, RUNTIME_INFO_TOTAL_SIZE, WS_SEG_RUNTIME_INFO, {}};
    offset += RUNTIME_INFO_TOTAL_SIZE;

    workspaceLayout_.totalSize = offset;
    workspaceLayout_.segCount = idx;
}

ge::graphStatus AllGatherMatmulTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, ...);
    workspaceSize_ = libApiWorkSpaceSize_ + storageA_ + biasLen_;
    BuildWorkspaceLayout();
    workspaces[0] = workspaceLayout_.totalSize;
    return ge::GRAPH_SUCCESS;
}
```

PostTiling 中写入：

```cpp
tilingData.workspaceLayout = workspaceLayout_;
```

### 7.2 AlltoAllMatmul

```cpp
void AllToAllMatmulTilingBase::BuildWorkspaceLayout()
{
    uint64_t offset = 0;
    uint32_t idx = 0;

    workspaceLayout_.segments[idx++] = {offset, libApiWorkSpaceSize_, WS_SEG_LIB_API, {}};
    offset += libApiWorkSpaceSize_;

    workspaceLayout_.segments[idx++] = {offset, inferredInfo_.commLen, WS_SEG_COMM_OUT, {}};
    offset += inferredInfo_.commLen;

    if (inferredInfo_.permuteLen > 0) {
        workspaceLayout_.segments[idx++] = {offset, inferredInfo_.permuteLen, WS_SEG_PERMUTE_OUT, {}};
        offset += inferredInfo_.permuteLen;
    }
    if (inferredInfo_.biasLen > 0) {
        workspaceLayout_.segments[idx++] = {offset, inferredInfo_.biasLen, WS_SEG_BIAS, {}};
        offset += inferredInfo_.biasLen;
    }

    workspaceLayout_.segments[idx++] = {offset, RUNTIME_INFO_TOTAL_SIZE, WS_SEG_RUNTIME_INFO, {}};
    offset += RUNTIME_INFO_TOTAL_SIZE;

    workspaceLayout_.totalSize = offset;
    workspaceLayout_.segCount = idx;
}
```

### 7.3 MatmulReduceScatterV2

```cpp
void MatmulReduceScatterTilingBase::BuildWorkspaceLayout()
{
    uint64_t offset = 0;
    uint32_t idx = 0;

    workspaceLayout_.segments[idx++] = {offset, libApiWorkSpaceSize_, WS_SEG_LIB_API, {}};
    offset += libApiWorkSpaceSize_;

    workspaceLayout_.segments[idx++] = {offset, mmResultLen_, WS_SEG_MM_RESULT, {}};
    offset += mmResultLen_;

    if (isA2APath_) {
        workspaceLayout_.segments[idx++] = {offset, mmResultLen_, WS_SEG_RECV_BUF, {}};
        offset += mmResultLen_;
    }

    workspaceLayout_.segments[idx++] = {offset, RUNTIME_INFO_TOTAL_SIZE, WS_SEG_RUNTIME_INFO, {}};
    offset += RUNTIME_INFO_TOTAL_SIZE;

    workspaceLayout_.totalSize = offset;
    workspaceLayout_.segCount = idx;
}
```

---

## 8. Kernel 侧 Runtime Info 集成

### 8.1 AllGatherMatmulV2

**文件**: `all_gather_matmul_v2/op_kernel/arch35/all_gather_matmul_fp16_bf16.h`

```cpp
#include "../../../common/op_kernel/mc2_runtime_info.h"

class AllGatherMatmulFP16BF16 {
    // ...
    Mc2Kernel::Mc2RuntimeInfo runtimeInfo_;  // 新增成员
};

// Init
runtimeInfo_.Init(workspaceGM, &tilingData->workspaceLayout, cfg.aicCoreNum);

// Process
__aicore__ inline void Process() {
    if ASCEND_IS_AIC {
        runtimeInfo_.UpdatePhase(0, 0, Mc2Tiling::RT_PHASE_COMM_PREPARE);
        StartNotify();

        runtimeInfo_.UpdatePhase(0, 0, Mc2Tiling::RT_PHASE_COMPUTE_LOCAL);
        InnerProcess();

        runtimeInfo_.UpdatePhase(0, 0, Mc2Tiling::RT_PHASE_FINALIZE);
        EndNotify();
    }
}

// MatmulKernelComputeGather
for (uint32_t i = 0; i < count; i++) {
    runtimeInfo_.UpdatePhase(isTail ? 1 : 0, i, Mc2Tiling::RT_PHASE_COMM_WAIT);
    if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
        hccl_.Wait(hHandles_[i + shift]);
    }
    runtimeInfo_.UpdatePhase(isTail ? 1 : 0, i, Mc2Tiling::RT_PHASE_COMPUTE_GATHER);
    mmv3.UpdateSlice(i, isTail);
    mmv3.Process(isLast && (i == (count - 1)));
}
```

同理修改 `all_gather_matmul_base_v2.h`（FP8 路径）。

### 8.2 AlltoAllMatmul

**文件**: `allto_all_matmul/op_kernel/allto_all_matmul_kernel_base.h`

```cpp
Mc2Kernel::Mc2RuntimeInfo runtimeInfo_;

// Init
runtimeInfo_.Init(workspaceGM, &tilingData->workspaceLayout,
                  tilingData->alltoAllMatmulTilingInfo.aicCoreNum);
```

**文件**: `allto_all_matmul/op_kernel/allto_all_matmul_pipeline.h`

```cpp
// Process
__aicore__ inline void Process(uint32_t taskCnt) {
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            runtimeInfo_->UpdatePhase(0, index, Mc2Tiling::RT_PHASE_COMM_PROCESS);
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            runtimeInfo_->UpdatePhase(0, index, Mc2Tiling::RT_PHASE_TRANSPOSE);
            transStage_->Process(index);
            // ...
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(9);
            runtimeInfo_->UpdatePhase(0, index, Mc2Tiling::RT_PHASE_COMPUTE_GATHER);
            computeStage_->Process(index);
        }
    }
    runtimeInfo_->UpdatePhase(0, 0, Mc2Tiling::RT_PHASE_FINALIZE);
}
```

### 8.3 MatmulReduceScatterV2

**文件**: `matmul_reduce_scatter_v2/op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h`

```cpp
Mc2Kernel::Mc2RuntimeInfo runtimeInfo_;

// Init
runtimeInfo_.Init(workspaceGM, &tilingData->workspaceLayout, cfg.aicCoreNum);

// MatMulV3Compute
for (uint32_t i = 0; i < count; i++) {
    runtimeInfo_.UpdatePhase(isTail ? 1 : 0, i, Mc2Tiling::RT_PHASE_COMM_WAIT);
    // ... hccl.Wait ...
    runtimeInfo_.UpdatePhase(isTail ? 1 : 0, i, Mc2Tiling::RT_PHASE_COMPUTE_GATHER);
    mmv3.UpdateSlice(i, isTail);
    mmv3.Process(isLast && (i == (count - 1)));
}
```

**文件**: `matmul_reduce_scatter_v2/op_kernel/arch35/matmul_a2a_vec_reduce_fp16_bf16.h`（A2A 路径）

```cpp
// ExecuteAicMatMulPipeline
for (uint32_t i = 0; i < count; i++) {
    runtimeInfo_.UpdatePhase(isTail ? 1 : 0, i, Mc2Tiling::RT_PHASE_COMPUTE_LOCAL);
    mmv3.UpdateSlice(i, isTail);
    mmv3.Process(isLast && (i == (count - 1)));
    CubeNotifyVector();
}

// ExecuteAivCommReducePipeline
for (uint32_t i = 0; i < count - 1; i++) {
    runtimeInfo_.UpdatePhase(isTail ? 1 : 0, i, Mc2Tiling::RT_PHASE_COMM_WAIT);
    // ... hccl.Wait ...
    runtimeInfo_.UpdatePhase(isTail ? 1 : 0, i, Mc2Tiling::RT_PHASE_REDUCE);
    // ... reduceSum ...
}

// PostProcess
runtimeInfo_.UpdatePhase(0, 0, Mc2Tiling::RT_PHASE_FINALIZE);
```

---

## 9. 异常 Dump 机制

### 9.1 合并 Mc2DumpTilingData + Mc2DumpWorkspaceContent

**文件**: `mc2/common/utils/mc2_exception_dump.h`

将 tiling data dump 和 workspace dump 合并到 `Mc2DumpTilingData` 中，只需一次 D2H 拷贝 tiling data，内部自动完成 workspace dump。

#### 设计原理

各算子的量化/非量化 TilingData 结构体共享相同前缀（`mc2InitTiling` + `mc2CcTiling` + TilingInfo/RCSTiling），`x1QuantMode` 字段位于共享前缀中，因此在量化/非量化两个结构体中的偏移相同。`Mc2DumpTilingData` 内部已 D2H 拷贝了完整 2048B tiling data，可直接从中读取 `x1QuantMode` 判断量化模式，选择正确的 `wsLayoutOffset`，然后内部完成 workspace 逐段 dump。

#### 签名

```cpp
inline void Mc2DumpTilingData(aclrtExceptionInfo *args, const char *op,
                              uint32_t tilingGmArgIdx,
                              uint32_t argsOffset = 0U,
                              uint32_t workspaceGmArgIdx = 0U,
                              uint32_t quantModeOffset = 0U,
                              uint32_t wsLayoutOffsetNonQuant = 0U,
                              uint32_t wsLayoutOffsetQuant = 0U)
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `workspaceGmArgIdx` | workspaceGM 在 args 中的索引，0 表示不 dump workspace | 0 |
| `quantModeOffset` | `x1QuantMode` 在 tiling data 中的偏移（`offsetof` 编译期常量） | 0 |
| `wsLayoutOffsetNonQuant` | 非量化结构体中 `workspaceLayout` 的 `offsetof` | 0 |
| `wsLayoutOffsetQuant` | 量化结构体中 `workspaceLayout` 的 `offsetof` | 0 |

> 向后兼容：`workspaceGmArgIdx=0` 时行为与原 `Mc2DumpTilingData` 完全一致，不影响现有调用方。

#### 内部流程

1. D2H 拷贝 args + tiling data（2048B）并落盘（已有逻辑）
2. 若 `workspaceGmArgIdx == 0`，结束
3. 若 `wsLayoutOffsetNonQuant == wsLayoutOffsetQuant`，直接用该 offset（无量化变体的算子）
4. 否则从已拷贝的 `tilingBuf[quantModeOffset]` 读 `uint32_t`，非零用 quant offset，否则用 non-quant offset
5. 从选中 offset 解析 `WorkspaceLayoutInfo`
6. 读取 workspaceGM 指针（`tilingGmArgIdx - 1`，即 workspace 和 tiling 是相邻的最后两个参数）
7. 逐段 D2H 拷贝 workspace 并落盘

#### 量化模式检测原理

```
AlltoAllMatmulTilingData (非量化):                   AlltoAllQuantMatmulTilingData (量化):
  mc2InitTiling(64)                                    mc2InitTiling(64)
  mc2CcTiling(512)                                     mc2CcTiling(512)
  AlltoAllMatmulTilingInfo(96)                         AlltoAllMatmulTilingInfo(96)
    └ x1QuantMode (offset 616)                           └ x1QuantMode (offset 616)  ← 相同!
  ─── 分界线 ───                                       ─── 分界线 ───
  Mc2MatMulV3TilingData(236) × 2                      Mc2QuantBatchMatmulV3TilingDataParams(360) × 2
  WorkspaceLayoutInfo                                  WorkspaceLayoutInfo           ← offset 不同
```

`x1QuantMode` 在共享前缀中，两个结构体偏移相同。`offsetof(AlltoAllMatmulTilingData, alltoAllMatmulTilingInfo.x1QuantMode)` 是编译期常量，结构体布局变更时自动跟随。

> **约定**：各算子需将 `x1QuantMode`/`x2QuantMode` 放在量化/非量化结构体共享前缀中（分化点之前），确保两个变体中偏移一致。

### 9.2 各算子 Exception Wrapper 调用

#### AllGatherMatmulV2

**文件**: `all_gather_matmul_v2/op_host/op_tiling/all_gather_matmul_v2_tiling_common.cpp`

```cpp
inline void AllGatherMatmulV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) { return; }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "AllGatherMatmulV2");

    // TODO: 待 AllGatherMatmulV2 TilingData 加入 x1QuantMode 后，传入 quantModeOffset 和两个不同 offset
    // 当前无 x1QuantMode，两个 offset 相同，跳过量化检测
    Mc2Exception::Mc2DumpTilingData(args, "AllGatherMatmulV2",
        11U,   // tilingGmArgIdx
        0U,    // argsOffset
        10U,   // workspaceGmArgIdx
        0U,    // quantModeOffset (暂无)
        offsetof(Mc2Tiling::AllGatherMatmulTilingDataV2, workspaceLayout),       // nonQuant
        offsetof(Mc2Tiling::AllGatherMatmulTilingDataV2, workspaceLayout));      // quant (暂同)
}
```

#### AlltoAllMatmul

**文件**: `allto_all_matmul/op_host/op_tiling/allto_all_matmul_tiling.cpp`

```cpp
inline void AlltoAllMatmulExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) { return; }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "AlltoAllMatmul");

    Mc2Exception::Mc2DumpTilingData(args, "AlltoAllMatmul",
        12U,   // tilingGmArgIdx
        0U,    // argsOffset
        11U,   // workspaceGmArgIdx
        offsetof(AlltoAllMatmulTilingData, alltoAllMatmulTilingInfo.x1QuantMode),  // quantModeOffset
        offsetof(AlltoAllMatmulTilingData, workspaceLayout),                       // nonQuant
        offsetof(AlltoAllQuantMatmulTilingData, workspaceLayout));                 // quant
}
```

#### MatmulReduceScatterV2

**文件**: `matmul_reduce_scatter_v2/op_host/op_tiling/matmul_reduce_scatter_v2_tiling_common.cpp`

```cpp
inline void MatmulReduceScatterV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) { return; }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "MatmulReduceScatterV2");

    // TODO: 待 MatmulReduceScatterV2 TilingData 加入 x1QuantMode 后，传入 quantModeOffset 和两个不同 offset
    Mc2Exception::Mc2DumpTilingData(args, "MatmulReduceScatterV2",
        10U,   // tilingGmArgIdx
        0U,    // argsOffset
        9U,    // workspaceGmArgIdx
        0U,    // quantModeOffset (暂无)
        offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, workspaceLayout),
        offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, workspaceLayout));
}

__attribute__((constructor)) void RegisterMatmulReduceScatterV2ExceptionFunc()
{
    IMPL_OP(MatmulReduceScatterV2)
        .ExceptionDumpParseFunc(MatmulReduceScatterV2ExceptionImplWrapper);
}
```

---

## 10. 异常 Dump 产物

异常发生后，dump 目录下生成以下文件：

```
{dumpPath}/{deviceId}/
├── xn_addr_info_<op>.<streamId>.<taskId>.<ts>      # xn 地址信息 (已有)
├── cke_addr_info_<op>.<streamId>.<taskId>.<ts>     # cke 地址信息 (已有)
├── args_info_<op>.<streamId>.<taskId>.<ts>.bin     # kernel args (已有)
├── tiling_data_<op>.<streamId>.<taskId>.<ts>.bin   # tiling data (含 WorkspaceLayoutInfo)
├── workspace_seg0_type0_<op>.<streamId>.<taskId>.<ts>.bin   # 段0 (libApi)
├── workspace_seg1_type2_<op>.<streamId>.<taskId>.<ts>.bin   # 段1 (gather/commOut/mmResult)
├── workspace_seg2_type7_<op>.<streamId>.<taskId>.<ts>.bin   # 段2 (bias/permuteOut/recvBuf)
├── ...
└── workspace_segN_type12_<op>.<streamId>.<taskId>.<ts>.bin  # 段N (runtime info, 64KB)
```

### Runtime Info 解析

`workspace_seg{N}_type12_*` 文件为 Runtime Info 段，内容为 128 个 `RuntimeInfoPerCore` 结构体：

```
for i in 0..127:
    slot = read RuntimeInfoPerCore at offset i*512
    if slot.magicNum == 0x4D433252:
        print(f"Core {slot.coreId}: turn={slot.execTurn}, "
              f"position={slot.execPosition}, phase={phase_name[slot.commPhase]}")
```

示例输出：

```
Core 0  (C核):  turn=0, position=3, phase=COMM_WAIT        ← 等待第3个tile通信
Core 1  (C核):  turn=0, position=3, phase=COMPUTE_GATHER   ← 计算第3个tile
Core 24 (V核):  turn=0, position=3, phase=TRANSPOSE         ← V核正在转置
```

### 段类型对照表

| 文件名 type 值 | 含义 |
|:---:|------|
| 0 | 系统库预留空间 (通常不关注) |
| 1 | ND2NZ 转换空间 |
| 2 | AllGather 通信结果 |
| 5 | AlltoAll 通信结果 |
| 6 | AlltoAll 重排结果 |
| 7 | bias 空间 |
| 8 | matmul 结果暂存 |
| 9 | ReduceScatter 接收缓冲 |
| 12 | Runtime Info (64KB) |

---

## 11. 完整改动文件清单

| # | 文件 | 类型 | 说明 |
|:---:|------|:---:|------|
| 1 | `mc2/common/op_kernel/mc2_tiling_struct.h` | 修改 | 新增 WorkspaceSegType、WorkspaceSegInfo、WorkspaceLayoutInfo、RuntimePhase、RuntimeInfoPerCore |
| 2 | `mc2/common/op_host/op_tiling/mc2_tiling_struct.h` | 修改 | 新增对应 TILING_DATA_DEF 注册 |
| 3 | `mc2/common/op_kernel/mc2_runtime_info.h` | **新增** | Mc2RuntimeInfo 辅助类 |
| 4 | `mc2/common/utils/mc2_exception_dump.h` | 修改 | 新增 Mc2DumpWorkspaceContent 函数 |
| **AllGatherMatmulV2** | | | |
| 5 | `…/op_kernel/all_gather_matmul_tiling_data.h` | 修改 | TilingData 追加 workspaceLayout |
| 6 | `…/op_host/op_tiling/all_gather_matmul_tiling_base_v2.h` | 修改 | 新增 workspaceLayout_、BuildWorkspaceLayout |
| 7 | `…/op_host/op_tiling/all_gather_matmul_tiling_base.cpp` | 修改 | 实现 BuildWorkspaceLayout |
| 8 | `…/op_host/op_tiling/arch35/all_gather_matmul_tiling_v2.cpp` | 修改 | PostTiling 写入 workspaceLayout |
| 9 | `…/op_host/op_tiling/all_gather_matmul_v2_tiling_common.cpp` | 修改 | ExceptionWrapper 增加 tiling + workspace dump |
| 10 | `…/op_kernel/arch35/all_gather_matmul_fp16_bf16.h` | 修改 | 插入 runtimeInfo 调用 |
| 11 | `…/op_kernel/arch35/all_gather_matmul_base_v2.h` | 修改 | FP8 路径插入 runtimeInfo |
| **AlltoAllMatmul** | | | |
| 12 | `…/op_kernel/arch35/allto_all_matmul_tiling_data.h` | 修改 | TilingData 追加 workspaceLayout |
| 13 | `…/op_host/op_tiling/allto_all_matmul_tiling_base.h` | 修改 | 新增 BuildWorkspaceLayout |
| 14 | `…/op_host/op_tiling/allto_all_matmul_tiling_base.cpp` | 修改 | 实现 BuildWorkspaceLayout |
| 15 | `…/op_host/op_tiling/arch35/allto_all_fp_matmul_tiling_base.cpp` | 修改 | PostTiling 写入 workspaceLayout |
| 16 | `…/op_host/op_tiling/allto_all_matmul_tiling.cpp` | 修改 | ExceptionWrapper 增加 workspace dump |
| 17 | `…/op_kernel/allto_all_matmul_kernel_base.h` | 修改 | Init 中初始化 runtimeInfo |
| 18 | `…/op_kernel/allto_all_matmul_pipeline.h` | 修改 | Process 中插入 runtimeInfo 调用 |
| **MatmulReduceScatterV2** | | | |
| 19 | `…/op_kernel/matmul_reduce_scatter_v2_c_tiling.h` | 修改 | TilingData 追加 workspaceLayout |
| 20 | `…/op_host/op_tiling/matmul_reduce_scatter_tiling_base_v2.h` | 修改 | 新增 BuildWorkspaceLayout |
| 21 | `…/op_host/op_tiling/matmul_reduce_scatter_tiling_base.cpp` | 修改 | 实现 BuildWorkspaceLayout |
| 22 | `…/op_host/op_tiling/arch35/matmul_reduce_scatter_v2_tiling.cpp` | 修改 | PostTiling 写入 workspaceLayout |
| 23 | `…/op_host/op_tiling/matmul_reduce_scatter_v2_tiling_common.cpp` | 修改 | 新增 ExceptionWrapper 注册 |
| 24 | `…/op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h` | 修改 | 旧路径插入 runtimeInfo |
| 25 | `…/op_kernel/arch35/matmul_a2a_vec_reduce_fp16_bf16.h` | 修改 | A2A 路径插入 runtimeInfo |
| 26 | `…/op_kernel/arch35/quant_bmm_reduce_scatter_fp8_hif8.h` | 修改 | FP8 路径插入 runtimeInfo |
| 27 | `…/op_kernel/arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h` | 修改 | FP8 A2A 路径插入 runtimeInfo |

---

## 12. 关键设计要点总结

| 维度 | 决策 | 理由 |
|------|------|------|
| **Runtime Info 段位置** | workspace 末尾 | 不影响现有业务段 offset |
| **Runtime Info 段大小** | 固定 64KB (128×512B) | 覆盖 A5 最大核数 |
| **Core 编排** | C核 0~aicCoreNum-1，V核 aicCoreNum~127 | 先 C 核再 V 核 |
| **Kernel 写入方式** | `__gm__` 指针直接写 + `DataCacheCleanAndInvalid` | 单行缓存刷新，确保异常时数据可见 |
| **写入频率** | 仅阶段切换时写入（~5-10 次/core） | 性能影响 < 0.01% |
| **WorkspaceLayoutInfo 位置** | TilingData 末尾字段 | 向后兼容 |
| **Dump 触发** | 复用现有 `ExceptionDumpParseFunc` 机制 | 零改动异常检测路径 |
| **Dump 范围** | tiling data + workspace 各段（含 runtime info） | 完整调试信息 |
| **单段 dump 上限** | 64MB | 防止磁盘写满 |