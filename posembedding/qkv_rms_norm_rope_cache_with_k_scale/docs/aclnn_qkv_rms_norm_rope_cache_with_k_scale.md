# aclnnQkvRmsNormRopeCacheWithKScale

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

`aclnnQkvRmsNormRopeCacheWithKScale`面向大语言模型推理中的PagedAttention KV Cache更新场景。接口从融合输入`qkv`中拆分Q、K、V分量，对Q/K执行RMSNorm、RoPE和共享`rotation`矩阵乘，随后将Q/K动态量化为FP8 E4M3FN；Q分支输出`qOut`和`qScale`，K分支按`slotMapping`写入`kCache`和`kScaleCache`。V分支按`vScale`缩放后量化为FP8 E4M3FN，并按`slotMapping`写入`vCache`。

主要计算过程如下：

1. 按`headNums=[Nq, Nk, Nv]`从`qkv`拆分Q、K、V。
2. Q/K沿最后一维做RMSNorm：

   $$
   y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
   $$

3. Q/K分支采用Half-and-Half RoPE。`cosSin[..., :D/2]`为cos，`cosSin[..., D/2:]`为sin；V分支不执行RoPE：

   $$
   y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
   $$

4. Q/K共享`rotation`矩阵：

   $$
   q_{rot} = q_{rope} @ rotation,\quad k_{rot} = k_{rope} @ rotation
   $$

5. Q/K按每个token和head做动态量化，FP8 E4M3FN最大有限值使用`FP8_E4M3FN_MAX`，该值为448：

   $$
   scale = max(abs(x)) / FP8\_E4M3FN\_MAX,\quad x_{fp8} = cast(x / scale)
   $$

6. V分支计算：

   $$
   v_{fp8} = cast(v * vScale)
   $$

7. Cache写回位置由`slotMapping`决定：

   $$
   blockId = slotMapping[t] / BlockSize,\quad blockOffset = slotMapping[t]\ \%\ BlockSize
   $$

## 接口原型

每次调用分为两段。第一段接口完成参数校验、workspace计算和执行器创建，第二段接口提交算子执行。

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qGamma,
    const aclTensor *kGamma,
    const aclTensor *cosSin,
    const aclTensor *slotMapping,
    aclTensor *kCacheRef,
    aclTensor *vCacheRef,
    aclTensor *kScaleCacheRef,
    const aclTensor *cuSeqLen,
    const aclTensor *actualSeqLens,
    const aclTensor *rotationOptional,
    const aclTensor *vScaleOptional,
    const aclIntArray *headNums,
    const char *qkvLayout,
    float epsilon,
    aclTensor *qOut,
    aclTensor *qScale,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

### aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize

| 参数名 | 输入/输出/属性 | 必选/可选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| qkv | 输入 | 必选 | Q/K/V融合输入。`qkvLayout`为`"TND"`时shape为`[T, Nq+Nk+Nv, D]`，为`"NTD"`时shape为`[Nq+Nk+Nv, T, D]`。 | BFLOAT16 | 3维 |
| qGamma | 输入 | 必选 | Q分支RMSNorm权重。 | FLOAT | `[D]` |
| kGamma | 输入 | 必选 | K分支RMSNorm权重。 | FLOAT | `[D]` |
| cosSin | 输入 | 必选 | RoPE位置编码表，前`D/2`列为cos，后`D/2`列为sin。 | FLOAT | `[MaxSeqLen, D]` |
| slotMapping | 输入 | 必选 | 每个token写入cache的slot索引。 | INT32 | `[T]` |
| kCacheRef | 输入/输出 | 必选 | K Cache，接口原地更新，输出与输入复用同一Tensor，支持连续和符合约束的非连续Tensor。 | FLOAT8_E4M3FN | `[BlockNum, Nk, BlockSize, D]` |
| vCacheRef | 输入/输出 | 必选 | V Cache，接口原地更新，输出与输入复用同一Tensor，支持连续和符合约束的非连续Tensor。 | FLOAT8_E4M3FN | `[BlockNum, Nv, BlockSize, D]` |
| kScaleCacheRef | 输入/输出 | 必选 | K动态量化scale cache，接口原地更新，输出与输入复用同一Tensor，支持连续和符合约束的非连续Tensor。 | FLOAT | `[BlockNum, Nk, BlockSize, 1]` |
| cuSeqLen | 输入 | 必选 | 当前调用内各batch token数的前缀和，`cuSeqLen[0]`应为0，`cuSeqLen[Batch]`应为`T`。 | INT32 | `[Batch+1]` |
| actualSeqLens | 输入 | 必选 | 每个batch追加本次token后的实际序列长度，用于计算RoPE位置历史偏移。 | INT32 | `[Batch]` |
| rotationOptional | 输入 | 必选 | Q/K共享矩阵乘权重，当前必须传入非空Tensor。 | BFLOAT16 | `[D, D]` |
| vScaleOptional | 输入 | 必选 | V分支量化缩放因子，当前必须传入非空Tensor。 | FLOAT | `[Nv]` |
| headNums | 属性 | 必选 | Q/K/V头数，按`[Nq, Nk, Nv]`传入。 | INT64数组 | 长度为3 |
| qkvLayout | 属性 | 必选 | `qkv`、`qOut`和`qScale`的N/T轴布局。仅支持大小写敏感的`"TND"`和`"NTD"`。 | CHAR* | - |
| epsilon | 属性 | 可选 | RMSNorm防除零参数，默认值为`1e-6`。 | FLOAT | - |
| qOut | 输出 | 必选 | Q分支动态量化后的输出。`"TND"`时shape为`[T, Nq, D]`，`"NTD"`时shape为`[Nq, T, D]`。 | FLOAT8_E4M3FN | 3维 |
| qScale | 输出 | 必选 | Q分支动态量化scale。`"TND"`时shape为`[T, Nq]`，`"NTD"`时shape为`[Nq, T]`。 | FLOAT | 2维 |
| workspaceSize | 输出 | 必选 | 返回用户需要申请的workspace大小。 | UINT64* | - |
| executor | 输出 | 必选 | 返回算子执行器，作为第二段接口输入。 | aclOpExecutor** | - |

### aclnnQkvRmsNormRopeCacheWithKScale

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| workspace | 输入 | Device侧workspace地址。当`workspaceSize`为0时可传入`nullptr`。 |
| workspaceSize | 输入 | 第一段接口返回的workspace大小。 |
| executor | 输入 | 第一段接口返回的执行器。 |
| stream | 输入 | ACL runtime stream。 |

## 输出说明

| 输出 | 描述 |
|---|---|
| qOut | Q分支FP8 E4M3FN量化输出。 |
| qScale | Q分支每个token/head对应的动态量化scale。 |
| kCacheRef | 原地更新后的K Cache，shape和dtype与输入`kCacheRef`一致。 |
| vCacheRef | 原地更新后的V Cache，shape和dtype与输入`vCacheRef`一致。 |
| kScaleCacheRef | 原地更新后的K scale cache，shape和dtype与输入`kScaleCacheRef`一致。 |

## 约束说明

- 仅支持<term>Ascend 950PR/Ascend 950DT</term>。
- 当前实现仅支持`D=128`，且`Nv`必须等于`Nk`。
- `headNums`必须包含3个正整数，顺序为`[Nq, Nk, Nv]`。
- `qkvLayout`大小写敏感，仅支持`"TND"`和`"NTD"`。
- `qkv`、`qOut`、`qScale`保持相同N/T布局：
  - `qkvLayout="TND"`：`qkv=[T, Nq+Nk+Nv, D]`，`qOut=[T, Nq, D]`，`qScale=[T, Nq]`。
  - `qkvLayout="NTD"`：`qkv=[Nq+Nk+Nv, T, D]`，`qOut=[Nq, T, D]`，`qScale=[Nq, T]`。
- `cosSin`第二维必须等于`D`。调用方需保证`cosSin`覆盖所有RoPE位置，即`actualSeqLens`对应的位置索引不超过`MaxSeqLen-1`。
- `cuSeqLen`为当前调用内token的前缀和；`actualSeqLens[b]`表示第`b`个batch追加当前token后的实际序列长度，必须满足`actualSeqLens[b] >= cuSeqLen[b+1] - cuSeqLen[b]`。若`actualSeqLens[b]`小于该batch本次调用的token数，行为未定义。
- `slotMapping`取值范围应为`[0, BlockNum * BlockSize - 1]`。同一次调用内多个token写入同一slot会对同一cache地址产生重复写，最终写入顺序和结果未定义。
- `rotationOptional`和`vScaleOptional`当前必须传入非空Tensor。
- 输入和输出均为ND格式，不支持私有格式；不支持空Tensor。
- `kCacheRef`、`vCacheRef`和`kScaleCacheRef`支持连续Tensor，也支持符合以下约束的非连续Tensor：
  - `kCacheRef`、`vCacheRef`和`kScaleCacheRef`均为4维正stride，最后一维stride为1。
  - `kCacheRef`和`vCacheRef`前三维stride必须一致。
  - `kScaleCacheRef`前三维stride作为block/head/token写回stride使用，输入和原地更新后的Tensor必须保持同一组前三维stride。
- 资源边界约束：`Nq+Nk <= 128`，`Nq+Nk+Nv <= 160`，`Nv <= 80`，`ceil_align(Nq,16)+ceil_align(Nk,16) <= 256`。

## 调用示例

完整样例请参考[examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp](../examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp)。核心调用流程如下：

```cpp
std::vector<int64_t> headNumsVec = {Nq, Nk, Nv};
aclIntArray *headNums = aclCreateIntArray(headNumsVec.data(), headNumsVec.size());
const char *qkvLayout = "TND";
float epsilon = 1e-6f;

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus status = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    qkv, qGamma, kGamma, cosSin, slotMapping,
    kCacheRef, vCacheRef, kScaleCacheRef,
    cuSeqLen, actualSeqLens,
    rotationOptional, vScaleOptional,
    headNums, qkvLayout, epsilon,
    qOut, qScale,
    &workspaceSize, &executor);

void *workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

status = aclnnQkvRmsNormRopeCacheWithKScale(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```
