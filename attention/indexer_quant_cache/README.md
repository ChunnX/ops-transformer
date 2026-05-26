# IndexerQuantCache

## 概述

对 Indexer 注意力机制的 Epilog 阶段进行 KV Cache 压缩。该算子使用逐块动态量化（Per-Block Dynamic Quantization），
将 float16/bfloat16 的激活值压缩为 FP8（E4M3/E5M2）、INT8（UINT8）或 MX-FP4 格式，同时支持 MX-FP8、HiFloat8 和
MX-FP4 量化模式。

## 支持的硬件

- Ascend 950 AI Processor

## 算子规格

### 输入

| 参数名                         | 数据类型                                             | 格式 | 说明                                  |
|-------------------------------|----------------------------------------------------|------|--------------------------------------|
| indexer_compress_cache        | float8_e4m3fn, float8_e5m2, uint8                  | ND   | 当前 KV Cache（输入/输出原地更新）。MX-FP4 模式下为 uint8，每字节打包 2 个 fp4 值 |
| indexer_compress_cache_scale  | float32, float8_e8m0                                | ND   | 每块量化 Scale（输入/输出原地更新）          |
| x                             | float16, bfloat16                                  | ND   | 待量化的激活输入                          |
| slot_mapping                  | int32                                              | ND   | Token 到 Cache Slot 的索引映射             |

### 输出

| 参数名                         | 说明                                  |
|-------------------------------|--------------------------------------|
| indexer_compress_cache        | 原地更新后的 KV Cache                    |
| indexer_compress_cache_scale  | 原地更新后的 Scale                       |

### 属性

| 属性名       | 类型   | 默认值 | 说明                                           |
|-------------|--------|--------|-----------------------------------------------|
| quant_mode  | int    | 1      | 量化模式：0=MX-FP8, 1=Normal, 2=HiFloat8, 3=MX-FP4 |
| round_scale | bool   | true   | MX-FP8 模式下是否对 scale 进行舍入                 |
| scale       | float  | 1.0    | HiFloat8 模式下的全局 scale 乘数                  |

## 约束

- slot_mapping 的维度应等于 x 的维度减 1
- 最后一维（d 轴）按每 128 个元素一组进行逐块量化；MX-FP4（quant_mode=3）按每 32 个元素一组（标准 MX 块）量化出一个 e8m0 scale
- 仅支持 Ascend 950 平台
- 输入输出为原地操作（in-place）

## 调用示例

```cpp
#include "aclnn_indexer_quant_cache.h"

// ... 创建 tensor，设置 shape 和 dtype ...

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

// 第一阶段：计算 workspace 大小
aclnnIndexerQuantCacheGetWorkspaceSize(
    indexerCompressCache, indexerCompressCacheScale,
    x, slotMapping, 1, true, 1.0f,
    indexerCompressCacheOut, indexerCompressCacheScaleOut,
    &workspaceSize, &executor);

// 分配 workspace
void *workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);

// 第二阶段：执行算子
aclnnIndexerQuantCache(workspace, workspaceSize, executor, stream);
```
