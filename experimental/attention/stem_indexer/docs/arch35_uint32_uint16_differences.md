# arch35 中 uint32 与 uint16 分支差异

## 1. 范围

本文只总结 `experimental/attention/stem_indexer/op_kernel/arch35` 中 `uint32` 与 `uint16` score 通路存在差异的代码，不重复两条通路共用的 Cube/MM1、任务调度、动态 TopK 数量计算、因果窗口处理及 GM 输出逻辑。

两条通路由 `TOPK_SCORE_PRECISION` 在编译期选择：

| `TOPK_SCORE_PRECISION` | `SIType::scoreType` | 本文简称 |
| --- | --- | --- |
| `1` | `uint32_t` | uint32 通路 |
| `2` | `uint16_t` | uint16 通路 |

对应代码位于 `arch35/stem_indexer_common.h` 的 `SIType` 模板中：

```cpp
using scoreType =
    std::conditional_t<TOPK_SCORE_PRECISION == 1, uint32_t, uint16_t>;
```

## 2. 差异总览

| 环节 | uint32 通路 | uint16 通路 |
| --- | --- | --- |
| Vector1 score 编码 | FP32 直接编码为 32 位 sortable key | FP32 先按 `CAST_ROUND` 转为 BF16，再编码为 16 位 sortable key |
| score 精度 | 保留 FP32 位宽 | 降为 BF16 位宽，相近 FP32 score 可能变成相同 key |
| 单个 score 大小 | 4 B | 2 B |
| 每个 256 B Vector 寄存器处理的 score 数 | 64 | 128 |
| kth 定位方式 | 按 4 个字节执行 4 级直方图筛选 | 按高、低两个字节执行 2 级直方图筛选 |
| 字节索引工作区 | `idx0`、`idx1`、`idx2`、`idx3` | `idxHigh`、`idxLow`，复用 `idx0`、`idx1` 的物理位置 |
| TopK 临时索引类型 | `uint32_t` | `uint16_t` |
| 临时索引行 stride | 256 个 `uint32_t` | 512 个 `uint16_t` |
| 跨 S2 轮次历史索引 | `uint32_t`，直接拷贝 | 仍为 `uint32_t`，需从 uint16 临时索引转换或解包 |
| 最终稀疏索引输出 | `int32_t`/`uint32_t` 位宽 | `int32_t`/`uint32_t` 位宽，与 uint32 通路相同 |

## 3. 类型选择

文件：`arch35/stem_indexer_common.h`

`SIType` 只允许 `TOPK_SCORE_PRECISION` 为 1 或 2，并据此生成 `scoreType`。随后 `stem_indexer_service_vector.h` 使用：

```cpp
using SCORE_T = typename SIT::scoreType;
SIKernel::StemIndexerTopk<SCORE_T> topkOp_;
```

因此 Vector1 输出类型、TopK 模板实例、score buffer 的元素类型以及 padding VF 都在编译期分化，不存在运行时 precision 判断。

## 4. Vector1：score 编码差异

文件：`arch35/vf/stem_indexer_vector1.h`

两条通路都先计算：

```text
score = qk * rSquare + vBias
```

差异从 score 转换为可排序整数 key 时开始。

### 4.1 uint32 通路

uint32 重载直接处理 4 组 FP32 寄存器：

- 使用 `FloatSortConstCtx<float>`；
- 调用两次 `FloatX2ToSortableKey<float>`；
- 生成 4 组 `RegTensor<uint32_t>`；
- 每组写 64 个元素，依次写入行内偏移 `0`、`64`、`128`、`192`。

该通路不在编码前降低浮点位宽，sortable key 保留 FP32 的排序信息。

### 4.2 uint16 通路

uint16 重载增加了 FP32 到 BF16 的压缩步骤：

1. 对 FP32 寄存器两两执行 `DeInterleave`；
2. 分别使用 EVEN/ODD `CastTrait`，以 `RoundMode::CAST_ROUND` 转为 BF16；
3. 使用 `FloatSortConstCtx<bfloat16_t>`；
4. 调用 `FloatX2ToSortableKey<bfloat16_t>`，得到 2 组 `RegTensor<uint16_t>`；
5. 每组写 128 个元素，写入行内偏移 `0`、`128`。

因此 uint16 通路实际比较的是 BF16 舍入后的 score。若两个不同 FP32 score 舍入成同一 BF16，它们在 TopK 中会成为相同 key，最终由现有的“大于 kth 后追加等于 kth”过滤顺序处理并列项。

### 4.3 sortable key 规则

两条通路共用同一套模板逻辑，但实例化的浮点位宽不同：

| 项目 | uint32 通路 | uint16 通路 |
| --- | --- | --- |
| 浮点类型 | `float` | `bfloat16_t` |
| 整数 key 类型 | `uint32_t` | `uint16_t` |
| 符号位掩码 | `0x80000000` | `0x8000` |
| NaN key | 全 1 的 32 位值 | 全 1 的 16 位值 |

负数执行按位取反，非负数翻转符号位，使整数 key 与浮点大小保持单调关系。

## 5. TopK：kth 定位算法差异

相关文件：

- uint32：`arch35/vf/vf_topk_gather.h`
- uint16：`arch35/vf/vf_topk_16_gather.h`

### 5.1 uint32：4 级字节直方图

32 位 key 按 4 个字节逐级缩小 kth 所在桶：

| 级次 | 直方图函数 | 目标桶函数 | 输出桶 |
| --- | --- | --- | --- |
| 1 | `HistogramsFirstVFImpl` | `FindFirstTargetBinVFImpl` | `idx0` |
| 2 | `HistogramsSecondVFImpl` | `FindSecondTargetBinVFImpl` | `idx1` |
| 3 | `HistogramsThirdVFImpl` | `FindThirdTargetBinVFImpl` | `idx2` |
| 4 | `HistogramsLastVFImpl` | `FindKthVFImpl` | `idx3` 并还原完整 kth key |

输入过滤、score gather 和临时索引均使用 uint32，每个 256 B Vector 寄存器处理 64 个元素：

```cpp
constexpr uint32_t VF_B32_ELEMS = 64U;
```

### 5.2 uint16：2 级字节直方图

16 位 key 只需要高、低两个字节：

| 级次 | 直方图函数 | 目标桶函数 | 输出桶 |
| --- | --- | --- | --- |
| 1 | `HistogramsHighVFImpl` | `FindHighTargetBinVFImpl` | `idxHigh` |
| 2 | `HistogramsLowVFImpl` | `FindKthVFImpl` | `idxLow` 并合成 16 位 kth key |

score 与临时索引使用 uint16，每个 256 B Vector 寄存器处理 128 个元素：

```cpp
constexpr uint16_t VF_B16_ELEMS = 128U;
```

直方图计数、`next_k` 和跨轮历史索引仍使用 uint32。

## 6. TopK 编排与临时索引差异

文件：`arch35/vf/stem_indexer_topk.h`

`StemIndexerTopk<SCORE_T>::Batch4Rows` 使用 `if constexpr` 分派两个 VF 实现。

### 6.1 VF 调用参数

uint32 调用 `SITopkb32gather::SiTopKVF`，传入 `idx0` 至 `idx3`；uint16 调用 `SITopkb16gather::SiTopKVF`，只传入 `idx0`、`idx1`，它们在 uint16 VF 内分别作为 `idxHigh`、`idxLow`。

### 6.2 临时索引布局

两条通路为临时索引预留相同的物理空间，但元素类型和行 stride 不同：

| 项目 | uint32 通路 | uint16 通路 |
| --- | --- | --- |
| `tmpIndexLocal` 访问类型 | `LocalTensor<uint32_t>` | `ReinterpretCast<uint16_t>()` |
| 每行 stride | 256 × 4 B = 1024 B | 512 × 2 B = 1024 B |
| 4 行总空间 | 4096 B | 4096 B |

uint16 通路使用 512 个 uint16 元素作为行 stride，目的是让每行仍占 1024 B，并与下一行及后续复用区隔离；有效 TopK 临时索引只占行首部分。

### 6.3 首轮索引保存

- uint32：`DataCopy` 将 uint32 临时索引直接复制到 `hisIndexLocal[0]`；
- uint16：`AscendC::Cast(..., RoundMode::CAST_NONE)` 将 uint16 临时索引扩展为 uint32，再写入 `hisIndexLocal[0]`。

### 6.4 后续 S2 轮次 gather

- uint32 的 `FindRealIndexVFImpl` 直接加载 uint32 临时索引；
- uint16 的 `FindRealIndexVFImpl` 使用 `DIST_UNPACK_B16` 加载并扩展 uint16 临时索引，再从 uint32 历史索引中 gather 真实索引；
- 两条通路的 gather 结果都写入 uint32 `indicesOutLocal`，随后复制回 uint32 `globalIndexLocal`。

因此 uint16 只压缩 TopK 内部的 score 和临时相对索引，不改变跨轮真实索引及最终输出的 32 位语义。

## 7. TopK 复用工作区差异

文件：`arch35/vf/stem_indexer_topk.h`

以下为一次同时处理 4 行时，`reuseMm1ResLocal` 内 TopK 子区的布局。表中大小按当前常量 `TOPK_ALIGN_SIZE = 256` 计算。

| 子区 | uint32 通路 | uint16 通路 | 说明 |
| --- | ---: | ---: | --- |
| 字节桶索引区 | 16 KB | 8 KB | uint32 需要 4 组桶，uint16 只需要 2 组 |
| histogram | 4 KB | 4 KB | 256 个桶 × 4 行 × 4 B |
| `nkValue` | 1 KB | 1 KB | 64 个 uint32 × 4 行 |
| 临时索引 | 4 KB | 4 KB | 元素类型不同，物理大小相同 |
| `indicesOut` | 4 KB | 4 KB | 两条通路均为 uint32 |
| score 输出区 | 4 KB | 2 KB | 256 个 score × 4 行 |
| 合计 | **33 KB** | **23 KB** | 从复用区起点到 score 输出区末尾 |

对应偏移差异：

| 常量 | uint32 通路 | uint16 通路 |
| --- | ---: | ---: |
| `IDX_BUFFER_NUM` | 4 | 2 |
| `REUSE_INDICES_OUT_OFFSET` | 6400 个 uint32 = 25 KB | 4352 个 uint32 = 17 KB |
| `REUSE_SCORE_OUT_OFFSET_U32` | 7424 个 uint32 = 29 KB | 5376 个 uint32 = 21 KB |

## 8. Vector service 的 buffer 与 stride 差异

文件：`arch35/stem_indexer_service_vector.h`

### 8.1 `mrgValueBuf_` 物理大小相同

`mrgValueBuf_` 的申请大小固定按 `sizeof(uint32_t)` 计算：

```cpp
rows * (MAX_TOPK_COUNT + TRUNK_LEN_256) * sizeof(uint32_t)
```

因此 uint16 与 uint32 通路申请相同的物理空间。取得 tensor 后再按 `SCORE_T` 解释：

```cpp
mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
```

### 8.2 `mrgRowStride_` 元素数不同、字节数相同

```cpp
mrgRowStride_ =
    (MAX_TOPK_COUNT + TRUNK_LEN_256) * sizeof(uint32_t) /
    sizeof(SCORE_T);
```

| 项目 | uint32 通路 | uint16 通路 |
| --- | ---: | ---: |
| 每行 score 元素 stride | 512 | 1024 |
| 每行物理字节数 | 2048 B | 2048 B |

更大的 uint16 元素 stride 保证最终把该行重新解释为 uint32，并在行首打包稀疏索引时，仍有与 uint32 通路相同的物理空间。

### 8.3 padding VF 的编译期差异

文件：`arch35/vf/stem_indexer_topk_utils.h`

padding 代码共用 `SCORE_T` 模板，但以下编译期量不同：

| 项目 | uint32 通路 | uint16 通路 |
| --- | --- | --- |
| `VF_ELEMS = 256 / sizeof(SCORE_T)` | 64 | 128 |
| `IDX_T` | `int32_t` | `int16_t` |
| mask、load/store、比较类型 | uint32 | uint16 |

算法范围和填充值相同，仅向量宽度及寄存器元素类型不同。

## 9. 仅存在于单侧 VF 文件、但未进入当前主调用链的代码

- uint32 文件中的 `IndicesAddOffsetVF` / `IndicesAddOffset` 在 `arch35` 当前 TopK 编排中没有调用；
- uint16 文件中的 `FindLDRealIndexVFImpl` / `SiTopKLDGatherVF` 在 `arch35` 当前 TopK 编排中没有调用。

它们是两个独立 VF 文件之间的代码差异，但不影响当前 `StemIndexerTopk::Batch4Rows` 主通路。

## 10. 未列为差异的共用部分

以下代码在两条通路中没有算法分支，本文不展开：

- Cube/MM1 计算与 CV 同步；
- `qk * rSquare + vBias` 的 FP32 计算本身；
- 动态 TopK 数量、causal/window 范围计算；
- TopK 结果的初始块和窗口索引打包；
- `globalIndexLocal_`、`indicesOutLocal`、`sparseIndicesGm` 和 `sparseSeqLenGm` 的 32 位输出格式；
- kernel 入口与 arch35 调度框架。

## 11. 结论

uint32 通路以更大的 score 和字节桶工作区保留 FP32 排序精度，通过 4 级直方图得到 kth；uint16 通路在 Vector1 将 score 舍入为 BF16 sortable key，通过 2 级直方图降低 TopK 的数据宽度和桶工作区。uint16 的压缩范围止于 TopK 内部：跨轮真实索引和最终稀疏索引输出仍保持 uint32/int32。
