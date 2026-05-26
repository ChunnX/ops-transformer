# KvCompressEpilog

## 概述

在 KV Cache 的 Epilog 阶段对缓存进行压缩更新。该算子使用逐组动态量化（Group-wise Dynamic Quantization），
将 bfloat16 的激活值压缩为 FP8（UINT8 打包）格式，并按 slot_mapping 将量化结果原地散写（scatter）到
cache。支持 group(bf16-scale)、group(e8m0-scale)、HiFloat8 三种量化模式。

## 支持的硬件

- Ascend950PR/Ascend950DT

## 算子规格

### 输入

| 参数名             | 数据类型        | 格式 | 说明                                  |
|-------------------|----------------|------|--------------------------------------|
| cache | uint8          | ND   | 当前 KV Cache（输入/输出原地更新），每行包含 rope/fp8/scale/pad 各区段 |
| x                 | bfloat16       | ND   | 待量化的激活输入（末轴含 rope 段）        |
| slot_mapping      | int32, int64   | ND   | Token 到 Cache Slot 的索引映射，-1 表示跳过 |

### 输出

| 参数名             | 说明                    |
|-------------------|------------------------|
| cache | 原地更新后的 KV Cache     |

### 属性

| 属性名            | 类型  | 默认值 | 说明                                              |
|------------------|-------|--------|--------------------------------------------------|
| quant_group_size | int   | 128    | 量化分组大小                                       |
| quant_mode       | int   | 1      | 量化模式：0=group(bf16 scale), 1=group(e8m0 scale), 2=HiFloat8 |
| round_scale      | bool  | true   | group 模式下是否对每组 scale 向上取到 2 的幂          |
| x_scale          | float | 1.0    | HiFloat8 模式下的全局 scale 乘数                    |

## 约束

- slot_mapping 的维度应等于 x 的维度减 1
- x 的最后一维（d 轴）需满足 d % 64 == 0 且 d > 64
- group 模式下每 64 个元素一组量化出一个 scale
- 仅支持 Ascend 950 平台
- 输入输出为原地操作（in-place）

## 调用示例

```cpp
#include "aclnn_kv_compress_epilog.h"

// ... 创建 tensor，设置 shape 和 dtype ...

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

// 第一阶段：计算 workspace 大小
aclnnKvCompressEpilogGetWorkspaceSize(
    cache, x, slotMapping,
    128, 1, 1, 1.0f,
    &workspaceSize, &executor);

// 分配 workspace
void *workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);

// 第二阶段：执行算子
aclnnKvCompressEpilog(workspace, workspaceSize, executor, stream);
```