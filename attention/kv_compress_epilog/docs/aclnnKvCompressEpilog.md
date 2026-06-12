# aclnnKvCompressEpilog

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |

## 功能说明

- 接口功能：在 KV Cache 的 Epilog 阶段，对 KV Cache 进行原地压缩更新。将 bfloat16 的激活值按逐组动态量化（Group-wise Dynamic Quantization）压缩为 FP8（UINT8 打包）格式，并按 slotMapping 散写到 kvCompressCache。支持 group(bf16 scale)、group(e8m0 scale)、HiFloat8 三种量化模式。

- 计算逻辑：
  - 根据 slotMapping 中的索引值，确定每个 token 写入 cache 的目标位置，值为 -1 的 token 跳过。
  - 对 x 的最后一维（d 轴）按每 64 个元素一组，计算每组的 amax，并量化为目标 dtype。
  - quantMode=0：group 量化，scale 存储为 bfloat16。
  - quantMode=1：group 量化，scale 存储为 float8_e8m0，支持 roundScale 向上取 2 的幂。
  - quantMode=2：HiFloat8 量化，输出为 x * scale 属性后的 hifloat8。

- 示例：

  ```
    kv_compress_cache_shape: [2048, 384]
    x_shape:                 [1024, 256]
    slot_mapping_shape:      [1024]
    quant_group_size = 128
    quant_mode = 1
    round_scale = true
    scale = 1.0
    kv_compress_cache_out_shape: [2048, 384]   (原地更新)
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnKvCompressEpilogGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnKvCompressEpilog"接口执行计算。

```Cpp
aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(
  aclTensor       *kvCompressCacheRef,
  const aclTensor *x,
  const aclTensor *slotMapping,
  int64_t          quantGroupSize,
  int64_t          quantMode,
  bool             roundScale,
  float            scale,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnKvCompressEpilog(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnKvCompressEpilogGetWorkspaceSize

- **参数说明**

  - kvCompressCacheRef(aclTensor*，计算输入/输出)：Device侧的aclTensor，表示当前层的 KV Cache 向量缓存（原地更新）。shape 为 [num_slots, kvCacheCol]。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 UINT8，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。

  - x(aclTensor*，计算输入)：Device侧的aclTensor，待量化的激活输入。shape 为 [bs, d]。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。

  - slotMapping(aclTensor*，计算输入)：Device侧的aclTensor，token 到 cache slot 的索引映射，值为 -1 表示跳过该 token。shape 为 [bs]，元素取值范围为 [-1, num_slots - 1]。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 INT32、INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。

  - quantGroupSize(int64_t，计算输入)：Host侧的int，量化分组大小。默认值为 128。

  - quantMode(int64_t，计算输入)：Host侧的int，量化模式。取值：
    - 0：group 量化（scale 存储为 bfloat16）
    - 1：group 量化（默认，scale 存储为 float8_e8m0）
    - 2：HiFloat8 量化

  - roundScale(bool，计算输入)：Host侧的bool，group 模式下是否对每组 scale 向上取到 2 的幂。默认值为 true。

  - scale(float，计算输入)：Host侧的float，HiFloat8 模式下的全局 scale 乘数。仅在 quantMode=2 时生效。默认值为 1.0。

  - workspaceSize(uint64_t*，计算输出)：Host侧的uint64_t，返回执行算子所需的 workspace 大小。

  - executor(aclOpExecutor**，计算输出)：返回算子执行器。

## aclnnKvCompressEpilog

- **参数说明**

  - workspace(void*，计算输入)：workspace 内存地址。
  - workspaceSize(uint64_t，计算输入)：workspace 大小。
  - executor(aclOpExecutor*，计算输入)：算子执行器。
  - stream(aclrtStream，计算输入)：执行算子使用的 stream。

## 约束

- 仅支持 Ascend 950PR/Ascend 950DT 平台。
- slotMapping 的维度应等于 x 的维度减 1，即 slotMapping 为 x 除最后一维外的所有维度展平。
- x 的最后一维（d 轴）需满足 d % 64 == 0 且 d > 64，按每 64 个元素一组进行逐组量化。
- kvCompressCacheRef 既是输入也是输出（原地操作）。
- slotMapping 中值为 -1 的 token 会被跳过不处理。
