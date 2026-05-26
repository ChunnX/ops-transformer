# aclnnIndexerQuantCache

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |

## 功能说明

- 接口功能：在 Indexer 注意力机制的 Epilog 阶段，对 KV Cache 进行原地压缩更新。将 float16/bfloat16 的激活值按逐块动态量化（Per-Block Dynamic Quantization）压缩为 FP8（E4M3/E5M2）或 INT8（UINT8）格式，支持 Normal、MX-FP8、HiFloat8 三种量化模式。

- 计算逻辑：
  - 根据 slotMapping 中的索引值，确定每个 token 写入 cache 的目标位置。
  - 对 x 的最后一维（d 轴）按每 128 个元素一组，计算每组的 amax，并量化为目标 dtype。
  - Normal 模式：scale 存储为 float32，使用动态 block quantization。
  - MX-FP8 模式：scale 存储为 float8_e8m0，支持 round_scale 舍入。
  - HiFloat8 模式：scale 为全局 scale 属性乘以输入，输出为 hifloat8。

- 示例：

  ```
    cache_shape:       [2048, 128]
    cache_scale_shape: [2048, 1]
    x_shape:                            [1024, 128]
    slot_mapping_shape:                 [1024]
    quant_mode = 1
    round_scale = true
    x_scale = 1.0
    cache_out_shape:       [2048, 128]    (原地更新)
    cache_scale_out_shape: [2048, 1]      (原地更新)
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnIndexerQuantCacheGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnIndexerQuantCache"接口执行计算。

```Cpp
aclnnStatus aclnnIndexerQuantCacheGetWorkspaceSize(
  const aclTensor *cache,
  const aclTensor *cacheScale,
  const aclTensor *x,
  const aclTensor *slotMapping,
  int64_t          quantMode,
  bool             roundScale,
  float            xScale,
  const aclTensor *cacheOut,
  const aclTensor *cacheScaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnIndexerQuantCache(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnIndexerQuantCacheGetWorkspaceSize

- **参数说明**

  - cache(aclTensor*，计算输入/输出)：Device侧的aclTensor，表示当前层的 KV Cache 向量缓存。shape 为 [num_slots, d]。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。

  - cacheScale(aclTensor*，计算输入/输出)：Device侧的aclTensor，表示每块量化的 scale 因子。shape 为 [num_slots, CeilDiv(d, 128)]。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 FLOAT（Normal/HiFloat8 模式）、FLOAT8_E8M0（MX-FP8 模式），[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。

  - x(aclTensor*，计算输入)：Device侧的aclTensor，待量化的激活输入。shape 为 [bs, d]。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。

  - slotMapping(aclTensor*，计算输入)：Device侧的aclTensor，token 到 cache slot 的索引映射，值为 -1 表示跳过该 token。shape 为 [bs]，元素取值范围为 [-1, num_slots - 1]。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，不支持空Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持 INT32，[数据格式](../../../docs/zh/context/数据格式.md)支持 ND。

  - quantMode(int64_t，计算输入)：Host侧的int，量化模式。取值：
    - 0：MX-FP8 量化（scale 存储为 float8_e8m0）
    - 1：Normal 逐块动态量化（默认，scale 存储为 float32）
    - 2：HiFloat8 量化

  - roundScale(bool，计算输入)：Host侧的bool，MX-FP8 模式下是否对 scale 进行 round 操作。仅在 quantMode=0 时生效。默认值为 true。

  - xScale(float，计算输入)：Host侧的float，HiFloat8 模式下的全局 scale 乘数。仅在 quantMode=2 时生效。默认值为 1.0。

  - cacheOut(aclTensor*，计算输出)：Device侧的aclTensor，原地更新后的 KV Cache。shape 和 dtype 与 cache 相同。

  - cacheScaleOut(aclTensor*，计算输出)：Device侧的aclTensor，原地更新后的 scale 因子。shape 和 dtype 与 cacheScale 相同。

  - workspaceSize(uint64_t*，计算输出)：Host侧的uint64_t，返回执行算子所需的 workspace 大小。

  - executor(aclOpExecutor**，计算输出)：返回算子执行器。

## aclnnIndexerQuantCache

- **参数说明**

  - workspace(void*，计算输入)：workspace 内存地址。
  - workspaceSize(uint64_t，计算输入)：workspace 大小。
  - executor(aclOpExecutor*，计算输入)：算子执行器。
  - stream(aclrtStream，计算输入)：执行算子使用的 stream。

## 约束

- 仅支持 Ascend 950PR/Ascend 950DT 平台。
- slotMapping 的维度应等于 x 的维度减 1，即 slotMapping 为 x 除最后一维外的所有维度展平。
- x 的最后一维（d 轴）按每 128 个元素一组进行逐块量化。
- cache 和 cacheOut 必须为同一 tensor（原地操作）。
- cacheScale 和 cacheScaleOut 必须为同一 tensor（原地操作）。
- slotMapping 中值为 -1 的 token 会被跳过不处理。
