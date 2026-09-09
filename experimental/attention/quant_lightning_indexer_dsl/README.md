# Quant Lightning Indexer（DSL MXFP4）

本目录提供 `quant_lightning_indexer` 的 CANNBotDSL 融合实现。接口范围以
`算子接口.xlsx` 的 `qli` 页为准；本文档沿用 ASC QLI V2 的“功能说明、参数说明、
约束说明、调用示例”结构，但只描述当前 DSL 已实现的 MXFP4 路径，不包含
`cu_seqlens_k`、`output_idx_offset`、LD/S2 分核、其他 layout 或其他量化模式。

实现入口为 `quant_lightning_indexer.py`。一次 Python 调用只下发一个顶层
`QliFusedKernel`，Cube、Vector1、Sparse TopK 和可选 Candidate TopK 均在该融合
Kernel 内完成。Sample 只负责接口校验、输出及 workspace 分配和 Kernel 调用，
不生成测试数据，也不计算 CPU Golden。

## 功能说明

Quant Lightning Indexer 用于稀疏 Attention 的前处理。对每个 Query token，算子
计算 32 个 Query group 与有效 Key token 的 MXFP4 点积，乘以 E8M0 反量化系数，
经过 ReLU 和 FP32 权重加权后沿 group 轴归约，最后选出分数最高的 token 索引。

令逻辑 head dim 为 `D=128`、group 数为 `N1=32`，核心计算可写为：

$$
score(t,s)=\sum_{g=0}^{31}w(t,g)\cdot
\operatorname{ReLU}\left(\operatorname{Dequant}
\left(q(t,g)\cdot k(s)^T\right)\right)
$$

主要过程如下：

1. Cube 使用原生 FP4 E2M1 数据执行 128×128 QK 矩阵乘，累加类型为 FP32。
2. Vector1 按 ASC QLI V2 的舍入边界执行 ReLU、权重乘加和 32 group 归约，生成
   BF16 score code 及对应的 `uint16` 可排序 key。
3. Sparse TopK 使用 Arch35 `vf_topk_16.h` 对应的 uint16 radix 选择流程返回 token
   索引和 BF16 分数；超过 16K 的 S2 在同一 AIV 内按 trunk/history 逐段合并。
4. Candidate 开启时，每连续 8 个有效 token 形成一个 Candidate block，以该块的
   score 最大值参与同一套 TopK 比较。Candidate 的 8-token 分组与 PA 的
   128-token page 是两个独立概念。

## Python 接口

```python
def quant_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    topk: int,
    quant_mode: int,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    return_value: bool = False,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

该入口注册为
`torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer`；通过
`cann_ops_transformer.ops.ds41` 导入的同名对象就是该 Torch operator。

## 维度说明

| 符号 | 含义 | 当前范围 |
|---|---|---|
| `B` | Batch 数 | 动态；多 Batch 时由 `cu_seqlens_q.shape[0] - 1` 得到 |
| `T1` | TND Query 总 token 数 | 动态，且大于 0 |
| `N1` | Query group/head 数 | 固定为 32 |
| `N2` | Key head 数 | 固定为 1 |
| `D` | Q/K 逻辑 head dim | 固定为 128 |
| `P` | `k` 的物理 PA page 数 | 动态，且大于 0 |
| `block_size` | 每个 PA page 的 token 数 | 固定为 128 |
| `S2` | 每个 Batch 的有效 Key token 数 | 动态，容量上限为 128K |
| `C` | Candidate 输出容量 | `candidate_topk_blocks`；关闭时为 0 |

MXFP4 每个 `uint8` 存放两个 FP4 E2M1 数，因此逻辑 `D=128` 对应 Q/K 的物理末轴
长度 `D/2=64`。反量化 scale 每 32 个逻辑 D 元素一组，故末两轴为
`(D/64, 2)=(2,2)`。

## 参数说明

| 参数名 | 类型与 shape | 输入属性 | 说明 |
|---|---|---|---|
| `q` | `uint8 [T1,N1,D]` | 输入 | 打包 MXFP4 Query；，必须连续。 |
| `k` | `uint8 [block_num,block_size,N2,D]` | 输入 | 打包 MXFP4 Key，布局为 `PA_BBND`；仅支持 0 轴非连续。 |
| `w` | `float32 [T1,N1]` | 输入 | 32 个 Query group 的权重，必须连续。 |
| `q_descale` | `uint8 [T1,N1,Ceil(D/64),2]` | 输入 | E8M0 Query 反量化系数。该 shape 即 `(T1,N1,D/64,2)`，必须连续。 |
| `k_descale` | `[block_num,block_size,N2,D]` | 输入 | E8M0 Key 反量化系数；仅支持 0 轴非连续，0 轴步长应与 `k` 的物理 page 对应。 |
| `cu_seqlens_q` | `int32 [B+1]` | 可选输入 | TND 中各 Batch Query 长度的前缀和。省略时按单 Batch 处理；`B>1` 时必须传入。 |
| `seqused_q` | `int32 [B]` | 可选输入 | 各 Batch Query 的有效前缀长度。 |
| `seqused_k` | `int32 [B]` | 可选输入 | 各 Batch 的实际 S2，数值为直接长度而非累计长度，可落在 128-token page 的中间。 |
| `cmp_residual_k` | `int32 [B]` | 条件输入 | 仅 `mask_mode=3` 且 `cmp_ratio!=1` 时必须传入，表示压缩 Key 的残余长度。其他组合必须为 `None`。 |
| `block_table` | `int32 [B,max_num_blocks_per_seq]` | 可选输入 | 每个 Batch 从逻辑 Key page 到 `k/k_descale` 物理 page 的映射；尾部未用项使用 `-1`。省略时按物理 page 顺序读取。 |
| `metadata` | `None` | 保留输入 | 当前 `qli` 表格约束下尚未启用，必须传 `None` 或省略。 |
| `topk` | `int` | 必选属性 | Sparse TopK 的输出数量，范围 `[1,8192]`。 |
| `quant_mode` | `int`，默认 `1` | 属性 | 当前仅支持 `1`，表示 Q/K 为 MXFP4 E2M1、scale 为 E8M0。 |
| `max_seqlen_q` | `int`，默认 `-1` | 属性 | Query 单 Batch 最大长度；`-1` 表示使用 `T1`。显式值应不小于实际最大 Query 长度。 |
| `mask_mode` | `int`，默认 `0` | 属性 | `0`：无 mask；`3`：压缩 Key causal mask。 |
| `cmp_ratio` | `int`，默认 `1` | 属性 | Key 压缩比，范围 `[1,128]`。传入的 S2 是压缩后的长度。 |
| `layout_q` | `str`，默认 `"TND"` | 属性 | 当前仅支持 `TND`。 |
| `layout_kv` | `str`，默认 `"PA_BBND"` | 属性 | 当前仅支持 `PA_BBND`。 |
| `return_value` | `bool`，默认 `False` | 属性 | 为保持表格接口兼容而保留。当前 Torch 返回 tuple 始终包含 `sparse_values`；精度转测使用 `True`。 |
| `candidate_topk_blocks` | `int`，默认 `-1` | 属性 | `-1` 关闭 Candidate；正数表示每个 Query 的 Candidate 输出容量 `C`。 |
| `candidate_block_size` | `int`，默认 `-1` | 属性 | Candidate 关闭时必须为 `-1`；开启时固定为 `8`。 |

所有 Tensor 输入都必须位于与 `q` 相同的 NPU。除 `k` 和 `k_descale` 的 0 轴
非连续能力外，其余输入要求连续。

## 返回值

函数返回四个 Tensor 组成的 tuple：

| 位置 | 含义 | 数据类型与 shape | 说明 |
|---|---|---|---|
| `0` | `sparse_indices` | `int32 [T1,1,topk]` | 选中的逻辑 Key token 索引。 |
| `1` | `sparse_values` | `bfloat16 [T1,1,topk]` | 与 `sparse_indices` 对应的 QLI score。 |
| `2` | `candidate_block_indices` | `int32 [T1,1,C]` | Candidate block 索引；关闭 Candidate 时 shape 为 `[T1,1,0]`。 |
| `3` | `candidate_block_length` | `int32 [T1,1]` | 每个 Query 有效的 Candidate 输出数量。 |

TopK 遵循 ASC VF 的稳定流式选择语义：先按输入顺序输出大于第 K 阈值的项，再按
输入顺序补齐等于阈值的项，不承诺结果按分数完全降序排列。

## 约束说明

- 当前只支持 `quant_mode=1`、`layout_q="TND"`、`layout_kv="PA_BBND"`、
  `N1=32`、`N2=1`、逻辑 `D=128` 和 PA `block_size=128`。
- `q`、`k` 使用 `uint8` 承载打包 MXFP4；`q_descale`、`k_descale` 使用
  `uint8` 承载 E8M0 bit pattern，不接受 BF16/FP16 替代。
- `k` 仅允许 0 轴带 page padding；其内层 stride 必须为 `(64,64,1)`。
  `k_descale` 同样仅允许 0 轴非连续，其内层 stride 必须为 `(4,4,2,1)`。
  两者的 0 轴步长都必须覆盖完整 PA 存储行。
- 有 `block_table` 时，Key 容量为 `block_table.shape[1] * 128`；否则为
  `k.shape[0] * 128`。当前动态 Kernel 的 S2 容量上限为 128K。
- `seqused_k[b]` 必须不超过对应 Batch 的映射容量；S2 尾块无需补齐到 128，
  Kernel 会在 NPU 上完成尾部 mask。
- `cu_seqlens_q`、`seqused_q`、`seqused_k`、`cmp_residual_k` 和
  `block_table` 均为 NPU `int32` 连续 Tensor。
- `mask_mode=3` 且 `cmp_ratio!=1` 时，`cmp_residual_k[b]` 应满足
  `0 <= cmp_residual_k[b] < cmp_ratio`。
- Candidate 开关是编译期分支：`(-1,-1)` 会完整裁掉 Candidate 计算和对应资源；
  开启时 `candidate_topk_blocks>0` 且 `candidate_block_size=8`。
- 本实现不包含 LD/S2 多核切分。每个 Cube task 最多处理同一 Batch 的 4 个 Query
  行，并由两个 AIV 分担；一个 task 的完整 S2 始终由同一 AIC 遍历。

## 动态 Shape 与编译缓存

`B`、`T1`、物理 page 数、`block_table` 宽度、实际 S2、`max_seqlen_q` 和
`seqused_*` 的数值均通过符号 shape 或运行时标量传入，不进入编译缓存键。进程内
`_COMPILED_KERNEL` 使用锁保护并缓存已编译 runner，因此同一静态分支下切换
Batch、T1 或 S2 不会重复编译。

以下参数会改变 Kernel 的静态控制流或输出宽度，取值变化时会产生新的编译缓存项：

- `topk`；
- Candidate 开关及 `candidate_topk_blocks`；
- `mask_mode`、`cmp_ratio`；
- `cu_seqlens_q`、`seqused_q`、`seqused_k`、`cmp_residual_k`、`block_table`
  是否存在。

缓存只在当前 Python 进程内有效，进程重启后的首次调用仍需编译。

## 实现说明

- Q/K 和 scale 的公开输入保持 `uint8`。受当前 CANNBotDSL packed-FP4 动态宿主
  carrier 限制，融合 Kernel 内先由 AIV 将动态输入搬入固定 staging，再由 AIC 以
  `fp4x2_e2m1` 和 `float8_e8m0` 原生类型进入 Cube；该过程没有额外小算子下发。
- ASC Vector1 使用 BF16 `MulAddDst`。当前 CANN 9.2 下 DSL raw `vmadd` 不提供
  BF16 operand overload，因此仅 128-token Vector1 算术 carrier 使用 FP32，并在
  ASC 定义的边界执行逐步 BF16-grid round-to-nearest-even。完整 S2 不分配 FP32
  score buffer，GM 中保存的是 16 位 score code 和 sortable key。
- Sparse 与 Candidate 均使用融合 Kernel 内的 raw-VF radix TopK；跨 VF 阶段的
  标量通过 UB 存取，不使用 `vreduce_max + vscatter` 合并方案。

## 调用示例

以下示例假设输入 Tensor 已按上表在 NPU 上准备完成；Sample 接口本身不负责生成
或随机化数据：

```python
from cann_ops_transformer.ops.ds41 import (
    quant_lightning_indexer,
)

result = quant_lightning_indexer(
    q,
    k,
    w,
    q_descale,
    k_descale,
    512,
    1,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    cmp_residual_k=None,
    block_table=block_table,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_value=True,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
)

(
    sparse_indices,
    sparse_values,
    candidate_indices,
    candidate_length,
) = result
```

关闭 Candidate 时两个属性必须同时传 `-1`：

```python
result = quant_lightning_indexer(
    q,
    k,
    w,
    q_descale,
    k_descale,
    512,
    1,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    cmp_residual_k=None,
    block_table=block_table,
    metadata=None,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
)
```

## 精度验证

CPU Golden 为本工程独立实现，不引用 ASC 源码；比较入口复现 ASC QLI V2 的
`result_compare_method`。Sparse 索引/值与 Candidate 索引均进入统一比较，
Candidate 还校验有效长度、索引合法性、唯一性和尾部 `-1` padding。

```bash
cd /home/x00949573/code/cannbot-arena_qli/test/quant_lightning_indexer
./test_run.sh single

# 只运行指定用例
./test_run.sh single --cases decode_b8_s2_8k,cross_09_pa_axis0_noncontiguous

# Excel -> PT -> NPU/CPU 批量比较
./test_run.sh batch --excel /path/to/cases.xlsx --sheet Sheet1 \
  --pt-path ./pt_path --output ./result.xlsx
```

当前动态 Shape 转测 paramset 包含 decode、prefill、S2 极小尾块、128 边界、
16K TopK trunk/history merge、`cmp_ratio=2`、Batch=32、PA 0 轴非连续及
Candidate 关闭分支。最近一次回归结果为 `12 passed`；运行日志位于
`outputs/qli_dynamic_shape_paramset_12.log`。
