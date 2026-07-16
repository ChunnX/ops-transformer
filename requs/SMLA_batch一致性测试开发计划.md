# SMLA batch 一致性测试开发计划

## 1. 结论

在现有 SparseFlashMla（SMLA）pytest 上适配 batch 一致性框架可行，但不能把
它实现成统一的 `tensor[order]`。SMLA 同时支持 BSND、TND 和 PA_BBND：

- BSND 的 batch 是显式维度；
- TND 的 batch 由 `cu_seqlens_*` 分段表达；
- PA_BBND 的 KV 是物理 cache，逻辑 batch 通过 block table 引用物理块。

因此必须实现 layout-aware 的用例变换，并同时变换所有耦合张量。仅重排 Q
而不处理 KV、稀疏索引、长度和 block table，会得到语义不同但 shape 合法的
错误用例。

建议按以下优先级开发：

1. 模式 1：batch 调序；
2. 模式 2：batch 拆分；
3. metadata 白盒分析；
4. 模式 3：batch 内 token 拆分；
5. 模式 4：前插/删除 token。

模式 1、2 可作为第一期稳定 CI。模式 3、4 对 mask、query position 和稀疏索引
语义有额外约束，应在变换有效性检查通过后再运行。

## 2. 当前 pytest 基线

现有链路位于：

~~~text
attention/sparse_flash_mla/tests/pytest/
├── sparse_flash_mla_golden.py
├── result_compare_method.py
├── test_sparse_flash_mla_batch.py
├── test_run.sh
└── batch/
    ├── sparse_flash_mla_pt_save.py
    └── sparse_flash_mla_process.py
~~~

当前数据流：

~~~text
Excel 参数
 -> sparse_flash_mla_golden.gen_data()
 -> input_data
 -> torch.save(.pt)
 -> test_sparse_flash_mla_batch.py
 -> sparse_flash_mla_process.call_npu()
 -> result_compare_method.check_result()
~~~

`input_data` 已包含：

- `params`：原始测试参数；
- `metadata_input`：生成 metadata 所需的派生参数；
- `input`：主算子的张量和属性；
- `cpu_output`；
- `softmax_lse`。

这使“一份基准 case 派生多个一致性 case”成为可能，无需重写现有 golden。

## 3. 测试目标

### 3.1 黑盒目标

| 模式 | 变换 | 预期 |
| --- | --- | --- |
| 1 | 调换 batch 顺序 | 映射回原 batch/token 后输出一致 |
| 2 | 把一个多 batch case 拆成多个子 case | 合并子 case 输出后与原输出一致 |
| 3 | 把一个 batch 的 query token 拆成多个逻辑 batch | token 映射后输出一致 |
| 4 | 在 batch 首部增加或删除 token | 未改变语义的对应 token 输出一致 |

### 3.2 白盒目标

- 统计每个 FA/FD core 是否启用；
- 解析每个 core 的 `BN2/M/S2` 范围；
- 识别 token 的规约数据是否落在单核或跨核；
- 识别单核单 token、单核多 token、单核与跨核混合等覆盖；
- 按 level-1 的 S2 区间和 level-2 调度形态输出覆盖报告。

白盒分析只用于解释覆盖和指导用例生成，不能替代数值精度比较。

## 4. 总体设计

建议新增：

~~~text
attention/sparse_flash_mla/tests/pytest/batch_consistency/
├── __init__.py
├── model.py
├── transform.py
├── layouts.py
├── runner.py
├── compare.py
├── metadata_layout.py
└── metadata_analyzer.py

attention/sparse_flash_mla/tests/pytest/
└── test_sparse_flash_mla_batch_consistency.py
~~~

职责：

| 模块 | 职责 |
| --- | --- |
| `model.py` | case、token origin map、运行结果和覆盖结果的数据类 |
| `layouts.py` | BSND/TND/PA_BBND 的分段、选取、拼接原语 |
| `transform.py` | 四种一致性模式的完整用例变换 |
| `runner.py` | 调用 metadata 和主算子，返回统一结果 |
| `compare.py` | 按 origin map 对齐输出并复用现有容差 |
| `metadata_layout.py` | 与 C++ header 同源的 metadata 布局定义 |
| `metadata_analyzer.py` | FA/FD 记录解析、分类和覆盖聚合 |

第一期不要直接改写 `sparse_flash_mla_golden.py` 的大段生成逻辑。先把
`gen_data()` 的结果当成不可变基准，用纯变换模块派生 case，降低回归范围。

## 5. 统一数据模型

~~~python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class TokenOrigin:
    source_batch: int
    source_token: int


@dataclass
class ConsistencyCase:
    name: str
    input_data: Dict[str, Any]
    output_origins: List[TokenOrigin]


@dataclass
class RunResult:
    output: torch.Tensor
    softmax_lse: Optional[torch.Tensor]
    metadata: Optional[torch.Tensor]
~~~

基准 case 与派生 case 都应带 `output_origins`。比较时按 origin 对齐，而不是
假设派生输出可以简单 reshape 回原 shape。

所有变换应：

- clone 要修改的 tensor；
- 不原地修改 `torch.load()` 返回的基准字典；
- 共享不变 tensor 时只读；
- 更新 `input`、`metadata_input` 和 `params` 中的相关字段；
- 生成可复现的变换 seed，并写入测试日志。

## 6. 必须同步变换的字段

| 类别 | 字段 |
| --- | --- |
| Query | `q` |
| KV | `ori_kv`、`cmp_kv` |
| 稀疏索引 | `ori_sparse_indices`、`cmp_sparse_indices` |
| PA 映射 | `ori_block_table`、`cmp_block_table` |
| Q 分段/长度 | `cu_seqlens_q`、`seqused_q` |
| KV 分段/长度 | `cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`、`seqused_ori_kv`、`seqused_cmp_kv` |
| 压缩残差 | `cmp_residual_kv` |
| 可变 top-k | `ori_topk_length`、`cmp_topk_length` |
| 派生参数 | `B`、`T1/T2/T3`、`max_seqlen_*` |
| 输出 | 主输出、可选 `softmax_lse` |

`sinks` 通常是 head 级张量而不是 batch 级张量，应保持不变。若接口后续改变
其语义，必须根据 shape 和算子契约重新分类，不能仅凭字段名处理。

## 7. Layout 原语

### 7.1 BSND

BSND 的 batch 位于第 0 维，可直接选取：

~~~python
def select_bsnd(tensor, batch_ids):
    if tensor is None:
        return None
    index = torch.as_tensor(
        batch_ids, dtype=torch.long, device=tensor.device
    )
    return tensor.index_select(0, index)
~~~

该函数只适用于明确含 batch 第 0 维的张量。不要用于 `sinks`、共享物理 cache
或标量属性。

### 7.2 TND

TND 的 batch 必须通过 prefix sum 切段：

~~~python
def split_tnd(tensor, cu_seqlens):
    if tensor is None:
        return None
    offsets = cu_seqlens.to("cpu", torch.int64).tolist()
    return [
        tensor[offsets[i]:offsets[i + 1]]
        for i in range(len(offsets) - 1)
    ]


def concat_segments(segments):
    lengths = [segment.shape[0] for segment in segments]
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    return torch.cat(segments, dim=0), prefix
~~~

重排 Q 时，所有按 query token 展开的稀疏索引和输出都必须使用同一 segment
顺序。重排 KV 时同理更新其 `cu_seqlens_*`。

### 7.3 PA_BBND

PA_BBND 下：

- `ori_kv/cmp_kv` 是物理 cache；
- block table 的每一行才表示一个逻辑 batch；
- block id 指向物理 cache 的第 0 维。

第一期策略应是：

1. 保留物理 cache backing storage 不变；
2. 只选取/重排 block table 行；
3. 同步选取 `seqused_*`、`cmp_residual_kv` 等 per-batch 向量；
4. 根据重排后的逻辑长度重算仍参与 metadata 的 `cu_seqlens_*`；
5. 不重编号 block id；
6. 校验所有有效 block id 仍在 cache 范围内。

这样既保留共享 block/非连续 cache 场景，也避免为重排复制大量 KV。

## 8. 模式 1：batch 调序

输入条件：

- `B >= 2`；
- 生成一个非恒等 permutation；
- Q、KV、索引和长度具有可识别的 batch 语义。

变换规则：

### BSND + BSND

- `q = q[order]`；
- batch 独立的 `ori_kv/cmp_kv = kv[order]`；
- 稀疏索引按其 batch/token 维同步重排；
- 所有 `[B]` 长度向量按 `order` 重排。

### TND + TND

- 用 `cu_seqlens_q` 拆 Q segment，再按 `order` 拼接；
- 对 query-token 级稀疏索引使用同一 Q segment 顺序；
- ori/cmp KV 分别按各自 prefix sum 拆分和拼接；
- 重新计算三组 prefix sum；
- per-batch 向量按 `order` 重排。

### TND/BSND + PA_BBND

- Q 按其 layout 重排；
- 物理 KV cache 不动；
- block table 行按 `order` 重排；
- per-batch 长度/残差向量按 `order` 重排。

比较规则：

~~~python
inverse = [0] * len(order)
for new_batch, old_batch in enumerate(order):
    inverse[old_batch] = new_batch
~~~

BSND 输出可按 inverse 恢复；TND 输出应按 `TokenOrigin` 映射，不应依赖每个
batch token 数相同。

## 9. 模式 2：batch 拆分

把 `batch_ids` 随机分成 2 到 N 个非空组。每组构造独立
`ConsistencyCase` 并执行主算子。

关键规则：

- BSND 选取 batch 行；
- TND 选取并拼接 segment，prefix sum 从 0 重算；
- PA_BBND 选取 table 行，但保留完整物理 cache；
- 子 case 的 `B`、`T1/T2/T3`、`max_seqlen_*` 全部重算；
- 按 origin map 合并所有子输出，再与一次性执行的基准输出比较。

第一期每个原 batch 只出现一次。复制 batch、共享 block 和空 segment 可放入
后续边界测试。

## 10. 模式 3：batch 内 token 拆分

该模式把一个 batch 中的多个 query token 转换为多个逻辑 batch。它不是普通
batch split，必须保证每个派生 token 看到与基准相同的注意力上下文。

建议第一期限制：

- 只选择 `S1 > 1` 或 TND segment 长度大于 1 的 case；
- KV 内容保持不变；
- PA 场景复制 block table 行，不复制物理 cache；
- 每个新 batch 的 `seqused_*`、残差和 top-k 长度从源 batch 复制；
- 稀疏索引按源 query token 选取；
- 明确构造新 token 的 query position。

对于依赖 query position、causal mask 或 sliding window 的场景，把 token 从原
batch 位置移到新 batch 的位置 0 可能改变合法注意力范围。必须先用 CPU golden
检查变换前后对应 token 是否等价：

~~~text
CPU baseline token != CPU transformed token
    -> 变换语义不成立，标记 SKIP/INVALID_TRANSFORM
CPU baseline token == CPU transformed token
    -> 再比较两次 NPU 输出
~~~

不要把“CPU 也不一致”的 case 记成 NPU batch 一致性失败。

## 11. 模式 4：前插/删除 token

模式 4 风险最高。前插或删除 Q token 会改变：

- query position；
- causal/window mask；
- TND prefix sum；
- query-token 稀疏索引；
- `seqused_q`；
- 可能的 KV 有效长度和 block table；
- metadata 的 M/S2 分配。

因此模式 4 不能对所有 SMLA case 无条件启用。建议：

1. 先支持 mask 语义确认不受前缀位置影响的 case；
2. 插入 token 时同步构造其 Q、索引和必要的 KV；
3. 删除 token 时保留剩余 token 的 origin；
4. 只比较 origin 相同且上下文相同的 token；
5. 运行 CPU golden 等价性预检查；
6. 报告 `INVALID_TRANSFORM`，不要静默跳过。

当特性目标是“prefill + 多次 decode 等价于一次 prefill”时，应单独建立
stateful 序列测试，不应强行复用简单的前插/删除实现。

## 12. 精度比较策略

现有 `result_compare_method.check_result()` 已包含 BF16/FP16 边界容差和全局
超差比例。batch 一致性第一期应复用同一标准，避免出现“golden 比较通过、
一致性比较失败”仅因为阈值不同。

建议增加薄适配层：

~~~python
def compare_pair(expected, actual):
    result = result_compare_method.check_result(expected, actual)
    status, percent = result[:2]
    return {
        "pass": status.lower() in {"pass", "passed"},
        "fulfill_percent": float(percent),
    }
~~~

需要同时比较：

- 主输出；
- `return_softmax_lse=True` 时的 LSE；
- 每个 origin 的局部结果；
- 合并后的整体结果。

MD5 只能作为“完全一致”的附加诊断：

- MD5 相同可快速确认 bitwise equal；
- MD5 不同不能直接判精度失败；
- 不同执行切分可能产生合法的浮点累加顺序差异；
- 最终判定仍使用数值容差。

## 13. metadata 白盒分析

### 13.1 布局来源

metadata 公共定义位于：

~~~text
attention/sparse_flash_mla/op_kernel/sparse_flash_mla_metadata.h
attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_metadata.h
~~~

当前布局为：

- 总长度 `1024 * int32`；
- FA：36 个 core，每条 9 个字段，共 324 个 int32；
- FD：72 个 core，每条 8 个字段，共 576 个 int32；
- 尾部为预留空间。

命名字段包括：

- FA：enable、BN2 start/end、M start/end、S2 start/end、
  first FD workspace index；
- FD：enable、BN2 index、M index、workspace index/num、
  M start/num。

### 13.2 不要复制魔法偏移

`metadata_layout.py` 应与 C++ header 同源，建议二选一：

1. 构建/测试时解析 header 中的 constexpr，生成 Python 常量；
2. 手工镜像常量，但增加 layout version、总大小和 record size 断言，并由
   单元测试对照 header。

解析入口示例：

~~~python
def parse_metadata(metadata, layout):
    flat = metadata.detach().to("cpu", torch.int32).reshape(-1)
    if flat.numel() != layout.total_size:
        raise ValueError(
            f"metadata size {flat.numel()} != {layout.total_size}"
        )

    fa_end = layout.fa_core_count * layout.fa_record_size
    fd_end = fa_end + layout.fd_core_count * layout.fd_record_size
    fa = flat[:fa_end].reshape(
        layout.fa_core_count, layout.fa_record_size
    )
    fd = flat[fa_end:fd_end].reshape(
        layout.fd_core_count, layout.fd_record_size
    )
    return fa, fd
~~~

不要在多个测试文件散落 `36`、`72`、`9`、`8` 和字段偏移。

### 13.3 数据采集

调整 `call_npu()`，允许可选返回 metadata：

~~~python
def call_npu(input_data, capture_metadata=False):
    metadata = build_metadata(input_data)
    output, lse = call_main_op(input_data, metadata)
    if capture_metadata:
        return RunResult(output, lse, metadata.detach().cpu())
    return output, lse
~~~

默认行为必须保持原二元返回，或一次性把调用方迁移到 `RunResult`，不能让现有
batch 用例悄然解包失败。

### 13.4 覆盖分类

解析器先输出原始记录，再由 analyzer 计算：

- 启用 FA/FD core 数；
- 每个 FA core 覆盖的 token 和 S2 范围；
- 每个 token 涉及的 core 集合；
- 单核规约/跨核规约；
- 每个 core 上的 token 数；
- 单核数据与跨核数据是否混合；
- level-1 的四类 S2 区间。

metadata 字段的确切闭开区间和 workspace 语义必须与 kernel 消费代码核对。
不能仅凭字段名猜测 `start/end` 是否包含末端。

覆盖输出建议使用 JSON：

~~~json
{
  "layout_version": 1,
  "enabled_fa_cores": 24,
  "enabled_fd_cores": 12,
  "single_core_tokens": 31,
  "cross_core_tokens": 3,
  "core_patterns": {
    "single_only": 12,
    "cross_only": 2,
    "mixed": 10
  }
}
~~~

## 14. 测试矩阵

| 维度 | 第一阶段覆盖 |
| --- | --- |
| template | SWA、HCA、CSA |
| Q layout | BSND、TND |
| KV layout | BSND、TND、PA_BBND |
| batch | 2、4、非 2 次幂 |
| Q 长度 | decode=1、短 prefill、变长 TND |
| KV 长度 | 相同长度、batch 间变长、边界长度 |
| KV 来源 | ori only、ori+cmp |
| sparse | 无索引、固定 top-k、可变 top-k |
| output | 主输出、主输出+LSE |
| 一致性模式 | 1、2 为必跑；3、4 为条件跑 |

白盒 S2 边界至少覆盖：

- `S2_b / 32 == 128`；
- `S2_b / 32 < 128`；
- `S2_b / 32 > 128` 且可整除；
- `S2_b / 32 > 128` 且有尾块。

## 15. 实施阶段

### 阶段 A：基线固化

- 为 `input_data` 写 schema 校验；
- 抽取 metadata 与主算子调用函数；
- 引入 `RunResult`；
- 为现有 PT 用例做零行为变更回归。

### 阶段 B：模式 1、2

- 实现 BSND/TND/PA_BBND 原语；
- 实现 origin map；
- 实现 batch permutation 和 split；
- 复用现有 compare；
- 接入 pytest 参数化。

### 阶段 C：metadata 分析

- 建立 header 同源布局；
- 解析 FA/FD 记录；
- 输出 token/core 关系；
- 聚合 level-1/level-2 覆盖；
- 给解析器增加手工 metadata 单元测试。

### 阶段 D：模式 3

- 先支持语义明确的 layout/mask 组合；
- 增加 CPU 变换有效性 oracle；
- 支持 PA block table 行复制；
- 补充 token origin compare。

### 阶段 E：模式 4

- 与特性负责人确认 token position/mask 约束；
- 分别实现 prepend 和 delete；
- 对不满足前提的 case 输出明确 skip reason；
- 补充 stateful prefill/decode 专项方案。

### 阶段 F：入口和 CI

在 `test_run.sh` 增加独立入口，例如：

~~~bash
bash test_run.sh consistency \
  -P ./data \
  -M 1,2 \
  --seed 20260716
~~~

不要默认改变现有 `single/save/load` 行为。CI 先放小 case，白盒覆盖报告作为
artifact；大 KV 和全矩阵放夜间任务。

## 16. 单元测试

不依赖 NPU 的测试：

- TND segment 拆分/拼接往返；
- permutation + inverse 恢复；
- split 后 origin 完整且不重复；
- PA 重排后 block id 不变且不越界；
- 所有 per-batch 向量同步；
- metadata shape/version 校验；
- FA/FD synthetic record 分类；
- 非法变换给出稳定错误原因。

依赖 NPU 的测试：

- 模式 1、2 各 layout 至少一条；
- ori-only 与 ori+cmp；
- `return_softmax_lse` 两种状态；
- eager 与 aclgraph 路径分别验证；
- metadata 单核、跨核和混合覆盖样例。

## 17. 风险与控制

| 风险 | 控制 |
| --- | --- |
| TND 只重排 Q，漏掉 token 级索引 | 所有变换通过统一 origin/segment API |
| PA 物理 cache 被错误重排 | 默认 cache 不动，只变 table 行 |
| prefix sum、seqused 和实际 tensor 长度不一致 | 变换后执行 schema invariant 校验 |
| 模式 3/4 改变 mask 语义 | CPU golden 等价性预检查 |
| MD5 把合法浮点差异判失败 | MD5 仅作诊断，数值标准作最终判定 |
| metadata header 演进 | 同源常量、version/size guard、header 对照测试 |
| 大 case 内存翻倍 | PA 共享物理 cache，分组串行执行 |
| 随机变换不可复现 | seed、order、split 方案写日志和结果 |
| 修改现有 load 入口造成回归 | 一致性测试使用独立 pytest 和 shell 入口 |

## 18. 验收标准

第一阶段完成条件（2026-07-17 按实际交付范围校正）：

- 模式 1、2 覆盖 BSND、TND、PA_BBND；
- 变换后所有 schema invariant 通过；
- 主输出和可选 LSE 均能按 origin 对齐；
- 数值标准与现有 pytest 对齐；
- 原有 `single/save/load` 测试结果不变；
- 失败日志包含基准 case、模式、seed、order/split、layout 和首个失败 origin。

原计划将 metadata parser 也放在“第一阶段完成条件”中。实际实施时为了先形成可审查的
黑盒闭环，并避免在字段区间/workspace 语义未确认前写出错误白盒结论，metadata 捕获、
解析和覆盖分析统一移入第二阶段。该范围调整不代表 metadata 已完成。

全部完成条件：

- 模式 3、4 对支持组合有明确前置条件；
- 不支持组合报告 `INVALID_TRANSFORM` 或带原因的 skip；
- level-1 四类边界均有用例；
- level-2 单核、跨核、单核多 token 和混合模式均有覆盖；
- 小矩阵进入 CI，大矩阵进入夜间回归；
- 文档、代码常量与 kernel metadata header 保持一致。

## 19. 2026-07-17 实际交付状态

本节及后续内容是第一阶段实现完成后的状态记录和第二阶段联合分析；如与前面的实施前
设想有冲突，以本节记录的实际代码状态为准。

### 19.1 已完成

- 模式 1（batch reorder）和模式 2（batch split）已经实现；
- Q 支持 BSND/TND，KV 支持 BSND/TND/PA_BBND；
- `TokenOrigin(source_batch, source_token)` 已用于主输出和可选 LSE 对齐；
- schema/invariant、per-batch 字段同步、TND prefix sum、PA block table 行映射已实现；
- 比较复用原 `result_compare_method.check_result`，BF16 MD5 仅作 bitwise 诊断；
- runner 设置 level 3、读取并恢复真实旧 level，当前运行时拒绝 level 3 时转换为
  `BatchConsistencyUnsupportedError`；
- 独立 pytest 和 `test_run.sh consistency` 入口默认关闭，不改变原
  `single/save/load/load_graph` 路径；
- 85 条 CPU 单测通过；三类 B=2 实际 PT 的纯变换和 invariant 通过；普通 level-0
  SMLA+LSE NPU 基线为 `Pass，100%`。

### 19.2 尚未完成

- metadata 的实际捕获、结构解析和 level-1/level-2 白盒覆盖分析；
- 模式 3（batch 内 token 拆分）和模式 4（前插/删除 token）；
- eager/graph 一致性统一执行入口；
- level-3 官方能力查询或可靠的 SMLA-specific 预检；
- 命名模式 registry、每 case 配置和正式 CI/nightly 矩阵。

### 19.3 当前环境阻塞

在 Ascend910_9362（910_93 系列）和当前 torch_npu/CANN 9.2 环境中，
`set_deterministic_level(3)` 可以返回且 `_get_deterministic_level()` 可读回 3，但第一次
NPU 操作会报“仅支持 0/1/2”。因此当前结论是：

- 测试框架能够把该环境分类为 `UNSUPPORTED`；
- 这不是精度比较失败；
- level-0 普通算子通过不能作为 batch 一致性通过证据；
- 真正 NPU 闭环必须在支持 level 3 的 torch_npu/CANN/kernel 组合上复验。

本地源代码检索还未发现 SMLA tiling key 中的 `BATCH_CONSISTENCY` 参数或
`GetDeterministicLevel` 调用。它与运行时现象相符，但只能说明当前工作树/当前环境状态，
不能代替上游版本计划；测试侧不能通过绕开 level 3 来伪造特性通过。

## 20. 与 GLM-5.2 的第二阶段联合审议记录

### 20.1 交互方式

本次使用模型 `cannbot/glm-5.2`，在仓库根目录进行两轮只读交互：

1. 第一轮让 GLM 阅读 4 份需求材料、当前开发计划、第一阶段全部实现、原
   `call_npu`/graph 入口以及 metadata header，独立给出第二阶段方案；
2. 第二轮使用同一 session，让 GLM 对第一轮做反向审查，逐项核对第一阶段真实状态、
   metadata 未知语义、level-3 延迟拒绝、配置兼容、graph、模式 3/4 和分阶段回滚。

两轮均明确禁止修改文件；GLM 没有写入仓库。本机 CANNBot 1.0.5 的默认数据库存在
`SQLiteError: no such column: replacement_seq` 兼容问题，因此本次使用隔离的
`XDG_DATA_HOME`，只复制认证文件，结束后清理临时目录，没有修改用户原数据库。

### 20.2 共同结论

- `-M 1,2` 是第一阶段临时兼容接口，不是长期设计；
- 模式定义应收敛到 Python Enum/registry，shell 只负责透传；
- metadata 的 1024/36x9/72x8 结构可由 header 证实，但字段闭开区间、FD 第 8 槽位
  和 workspace 关系不能猜测；
- metadata 捕获必须对应主算子真正消费的那一份，不能把“第二次重算结果”默认当成
  执行证据；
- graph 只能复用 deterministic guard、变换和比较层，需要单独的执行 adapter；
- 模式 3/4 必须先由 CPU golden oracle 证明变换前后语义等价，再进入 NPU 比较；
- `UNSUPPORTED`、`INVALID_TRANSFORM` 和真实数值 `FAIL` 必须分开统计；
- 所有第二阶段能力继续默认关闭，按小提交、可回滚方式逐步启用。

### 20.3 主审修正

GLM 输出是设计参考，不直接作为源码事实。联合审议后作出以下修正：

1. **mask code 以 golden 源码为准。** 当前 `GeneralizedSFA` 中，`ori_mask_mode=0`
   使用完整有效 KV；`ori_mask_mode=3` 的有效范围随 `cur_ori_act_kv-cur_act_q+i_S1+1`
   变化；`ori_mask_mode=4` 还叠加 window。不能采用模型曾给出的“0=causal、3=full”
   口头映射。完整枚举和接口契约仍需负责人确认。
2. **level-3 通用小算子成功不能证明 SMLA 支持。** generic `torch.add` 最多得到
   `UNKNOWN`；只有官方查询 API 或真实 SMLA level-3 smoke 才能标记 `SUPPORTED`。
3. **CPU oracle 不能调用 `gen_data(params)` 重生随机输入。** oracle 必须用已经变换的
   Q/KV/索引/长度/block table 调用可复用 golden 计算入口，否则比较的是另一份数据。
4. **metadata 不建议通过重复调用 metadata op 捕获。** 两次结果是否相同尚未证实，
   且第二份不一定是主算子实际消费的数据；优先从真实调用路径返回/观察原 metadata。
5. **配置来源不能靠运行时重新读 Excel 并模糊匹配 PT 名称。** 应先统一 registry，
   再由保存链路把可选配置写入 versioned namespace，或使用精确键的 sidecar manifest。

## 21. 第二阶段目标与设计原则

第二阶段不是一次性把模式 3/4、metadata 和 graph 全部塞进同一提交，而是建立可扩展
控制面和可验证的数据面：

1. 先把数字模式升级为命名 registry，并定义统一状态模型；
2. 只做 metadata 结构解析和原始记录输出，字段语义确认后再做覆盖推导；
3. 捕获主算子实际消费的 metadata，同时保持旧 `call_npu()` 二元返回；
4. 建立 SMLA-specific level-3 能力检查及运行时兜底；
5. 先接 graph 模式 1，再评估会改变 shape 的 graph 模式 2；
6. 抽取“对实际输入求 CPU golden”的 oracle 后再实现模式 3/4；
7. 最后接入 CI/nightly 和 artifact，不以 skip 数量掩盖未执行能力。

约束保持不变：现有 `single/save/load/load_graph` 默认行为不可变化；第二阶段实现前需要
先完成本计划中的阻断项确认。本次只更新分析文档，不修改第二阶段代码。

## 22. 配置接口演进

### 22.1 为什么 `-M 1,2` 只能是临时入口

当前数字接口有四个问题：

- `1/2` 对使用者没有自解释性；
- shell 和 Python 各自硬编码模式含义，新增模式需要同步修改；
- 无法携带 `implemented`、`requires_oracle`、`supports_graph` 等能力信息；
- 后续 metadata、graph、profile 和 per-case 配置无法继续用数字平铺。

第一阶段保留它是为了最小改动贯通 shell -> env -> pytest，并不是最终架构。

### 22.2 目标模型

建议新增 `batch_consistency/registry.py`，并在 `model.py` 定义规范枚举：

~~~python
class ConsistencyMode(Enum):
    REORDER = "reorder"
    SPLIT = "split"
    TOKEN_SPLIT = "token-split"
    PREPEND_DELETE = "prepend-delete"


@dataclass(frozen=True)
class ModeSpec:
    mode: ConsistencyMode
    numeric_alias: int
    implemented: bool
    min_batch: int
    requires_cpu_oracle: bool
    eager_handler: Optional[Callable]
    graph_handler: Optional[Callable]
~~~

registry 统一负责：

- 把 `reorder,split` 和兼容形式 `1,2` 解析为同一 Enum 集合；
- 检查重复、未知名称、互斥配置和未实现模式；
- 暴露模式能力，不让 `test_run.sh` 理解数字语义；
- 生成帮助信息和日志中的规范名称；
- 给 `--profile blackbox-basic` 映射为 `reorder,split`。

### 22.3 CLI 与兼容周期

推荐命令：

~~~bash
bash test_run.sh consistency \
  -P ./data \
  --modes reorder,split \
  --seed 20260716
~~~

迁移分三步：

| 阶段 | 行为 |
| --- | --- |
| 2A | 新增 `--modes` 和 registry；`-M 1,2` 继续可用并映射到命名模式 |
| 2B | 文档和 CI 全部切到命名形式；数字形式输出 deprecated warning |
| 2C | 数字形式只保留兼容解析；是否删除要统计下游使用并单独评审，不自动移除 |

shell 默认值也应调整：未显式传 `--modes` 时不要强制导出 `1,2`，由 Python 配置解析器
选择 case 配置或默认 profile，避免 shell 默认值永远覆盖后续 per-case 配置。

### 22.4 配置来源和优先级

建议使用明确的覆盖顺序：

~~~text
显式 CLI --modes
  > 显式环境变量 SMLA_CONSISTENCY_MODES
  > PT 中可选的 consistency 配置 / 精确键 sidecar manifest
  > --profile 或默认 profile blackbox-basic
~~~

Excel 链路打通后，推荐在保存 PT 时生成可选、带版本的命名空间，而不是测试时再次读取
Excel 并按文件名模糊匹配：

~~~python
"consistency": {
    "schema_version": 1,
    "modes": ["reorder", "split"],
    "metadata": False,
    "graph": False,
}
~~~

旧 PT 没有该字段时保持现有默认；schema 校验器只校验该 namespace 存在时的内容。
若团队不允许扩展 PT，则使用以 PT 精确 basename/case id 为键的 versioned JSON manifest，
两种方式二选一，不能同时形成两个真相源。

### 22.5 统一状态模型

第二阶段汇总至少区分：

| 状态 | 含义 | 是否计为失败 |
| --- | --- | --- |
| `PASS` | 环境支持、变换有效且数值通过 | 否 |
| `FAIL` | 环境支持、变换有效，但主输出/LSE 数值不一致或执行异常 | 是 |
| `UNSUPPORTED` | 当前 runtime/device/graph 不支持必要能力 | 否，单独计数 |
| `INVALID_TRANSFORM` | 源 case 在该模式下无法保持等价语义 | 否，单独计数并报告原因 |
| `CONFIG_ERROR` | 未知模式、非法组合、配置 schema 错误 | 是，入口错误 |

`skip` 是 pytest 表达机制，不应成为对外唯一状态；shell/JSON artifact 必须能区分
`UNSUPPORTED`、`INVALID_TRANSFORM` 和普通 deselected。

## 23. Metadata 白盒实现方案

### 23.1 已由当前 header 证实的结构

两个 header 的符号名略有不同，但数值布局一致：

~~~text
总长: 1024 * int32
FA:   offset [0, 324),   36 cores * 9 fields
FD:   offset [324, 900), 72 cores * 8 fields
尾部: offset [900, 1024), 124 int32 reserved
~~~

FA 的 9 个索引均有命名；FD record size 是 8，但当前 header 只命名了索引 0..6。
此外 backing tensor 定义为 `int32_t`，struct 字段为 `uint32_t`，解析器必须明确按位解释
无符号值，避免高位值被当成负数。

### 23.2 阻断确认项

在下列问题确认前，只能输出 raw record，不得声称完成 level-2 覆盖分类：

1. `BN2/M/S2 START/END` 是 `[start, end)` 还是闭区间；
2. FD 索引 7 是预留还是有未公开语义；
3. `FIRST_FD_DATA_WORKSPACE_IDX` 的索引基和地址空间；
4. `FD_WORKSPACE_NUM` 与 S2 切片、partial result 数量的准确关系；
5. split-G/non-split-G 是否使用相同 9 字段 FA record，`FA_S2_MAX_NUM` 在何时有效；
6. metadata header 与实际部署二进制的版本如何关联。

这些问题需要 kernel/metadata 负责人结合生产和消费 metadata 的源码确认，不能仅凭名称
反推。

### 23.3 模块拆分

建议新增：

| 文件 | 职责 |
| --- | --- |
| `metadata_layout.py` | versioned layout、header 镜像常量、size/offset/dtype guard、raw FA/FD 切片 |
| `metadata_analyzer.py` | 在字段语义确认后构造 token/core/workspace 关系并聚合覆盖 |
| `metadata_report.py`（可选） | JSON schema、artifact 写入和跨 case 聚合；不把 I/O 混入 parser |

`metadata_layout.py` 的第一批单测应包含 synthetic 1024 tensor、shape/dtype/size 错误、
FA/FD 边界、reserved 区域和 header 常量对照。header 对照测试只校验结构常量，不通过
正则猜测字段语义。

### 23.4 捕获真实 metadata 且保持旧接口

当前 `call_npu()` 和 `call_npu_graph()` 都在内部创建 metadata，并始终返回
`(output, softmax_lse)`。目标是捕获“实际传给主算子的同一 tensor”，同时不让旧调用方
出现可变长度返回值。

推荐优先级：

1. 抽取私有执行实现，使其返回 `RunResult(output, lse, metadata)`；保留旧
   `call_npu(input_data)` wrapper，只调用 `as_legacy_tuple()`；新增一致性专用 sibling
   函数返回 `RunResult`；
2. 如果大函数重构风险过高，给旧函数增加默认关闭的 `metadata_sink` callback，返回值
   永远仍是二元组，在实际主算子完成后把同一 metadata 复制到 CPU；
3. 不推荐在 runner 中再次调用 metadata op 作为最终方案，因为第二次结果不一定就是
   主算子消费的数据，也会重复执行算子。

无论选择 1 还是 2，都要增加旧 batch/eager/graph 调用方的解包回归测试，并保证
`capture_metadata=False` 时行为和性能无变化。

### 23.5 Analyzer 输出

语义确认后，analyzer 应先输出每核 raw record，再推导：

- enabled FA/FD core 数；
- 每个 `(batch, n2, query-token)` 的 FA/FD core 集合；
- 单核规约、跨核规约、单核多 token 和单核/跨核混合；
- 每核 token 数、S2 范围、workspace 范围；
- level-1 的四类 S2 条件；
- level-2 和“核视角”覆盖；
- 无法解释或越界的 record，作为 analyzer error 而不是静默忽略。

建议 JSON artifact：

~~~json
{
  "schema_version": 1,
  "layout_version": 1,
  "case": "...",
  "device": "...",
  "runtime": "...",
  "enabled_fa_cores": 0,
  "enabled_fd_cores": 0,
  "level1": {},
  "level2": {},
  "raw_record_digest": "...",
  "unknown_fields": []
}
~~~

## 24. Level-3 能力检查与结果分类

### 24.1 当前问题

当前环境证明 `set/get level=3` 不是有效预检：setter 成功、getter 返回 3，第一次 NPU
操作才拒绝。仅根据 setter 成功标记 `SUPPORTED` 会产生假阳性。

### 24.2 分层检查

建议能力检查按可靠性从高到低执行：

1. 如果 torch_npu/CANN 提供官方 supported-level 查询 API，使用它并记录版本；
2. 否则用一个最小但真实有效的 SMLA PT 执行 metadata + main op level-3 smoke；
3. 通用 NPU kernel 只能判断运行时是否普遍拒绝，成功时返回 `UNKNOWN`，不能证明 SMLA；
4. `UNKNOWN` 继续执行当前真实 case，并保留 `BatchConsistencyUnsupportedError` 兜底。

probe 必须保存/恢复旧 level，结果按 `(device, torch_npu, CANN, op build)` 缓存。当前 shell
每 PT 一个 pytest 进程，如果要跨 case 复用，应在 shell 入口先执行一次专用 probe 并用
机器可读结果传给子进程，而不是每个 case 重复探测。

### 24.3 错误分类

- 明确由 level 3 不支持引起的错误 -> `UNSUPPORTED`；
- 未识别的 runtime 异常 -> `FAIL`，不能用宽泛字符串吞掉；
- 支持 level 3 后产生的数值不一致 -> `FAIL`；
- 设置或执行后始终在 `finally` 恢复旧 level，恢复失败本身也要报告 `FAIL`；
- 字符串匹配只能作为当前版本 fallback，并用 monkeypatch 单测锁定已知错误文本。

## 25. Graph 一致性方案

现有 `call_npu_graph()` 使用 `torch.compile(fullgraph=True, dynamic=False)`，并在 graph 外
生成 metadata。graph 与 eager 可以共享 schema、transform、origin compare、status 和
deterministic guard，但不能共用同一个执行函数。

建议把 runner 拆成：

~~~text
batch_consistency_level_guard
  + eager_executor(input_data) -> RunResult
  + graph_executor(input_data) -> RunResult
  + run_mode(mode, executor, oracle, compare)
~~~

落地顺序：

1. 提取并单测 level guard，确保 eager 行为零变化；
2. 新增 `test_sparse_flash_mla_batch_consistency_graph.py`，避免与 eager 收集混在一起；
3. 先支持模式 1，因为 reorder 不改变 shape；
4. 模式 2 每个子 case B/T shape 变化，需要分别编译，先进入 nightly，再根据编译成本
   决定是否进小 CI；
5. graph metadata 必须捕获 graph 实际执行所用的同一 tensor。

开始实现前需确认：level 3 在编译期还是执行期生效；`torch._dynamo.reset()` 是否影响该
状态；graph 拒绝 level 3 的错误类型；同 shape reorder 是否能安全复用编译产物。

## 26. 模式 3/4 与 CPU Golden Oracle

### 26.1 Oracle 必须使用实际输入

模式 3/4 会修改 Q、KV 引用、长度、索引和位置语义。新的 `oracle.py` 应把
`GeneralizedSFA` 的“对现有 tensor 计算”能力封装为：

~~~text
baseline input_data --actual tensors--> CPU result A
derived input_data  --actual tensors--> CPU result B
result B --TokenOrigin--> align to A
数值一致 -> VALID，才运行 NPU
数值不一致 -> INVALID_TRANSFORM
~~~

不能直接调用 `gen_data(params)`，因为它会随机生成另一份 Q/KV/索引。第一阶段派生 case
已主动清空 `cpu_output`/`softmax_lse`，第二阶段应新增 actual-input golden adapter，而不是
复用过期输出。

### 26.2 mask 语义的当前源码事实

当前 golden 中可观察到：

- `ori_mask_mode=0`：`ori_win_start=0`、`ori_win_end=cur_ori_act_kv`，使用完整有效 KV；
- `ori_mask_mode=3`：end 随 `cur_ori_act_kv-cur_act_q+i_S1+1` 变化，query 位置敏感；
- `ori_mask_mode=4`：在上述位置关系上叠加 `ori_win_left/right`；
- cmp sparse/top-k 的有效阈值也会根据 mask mode、query 位置和 `cmp_ratio` 变化。

因此模式 3/4 的第一批候选可以从 full-KV 组合开始，但不能只用 mask 数字白名单替代
oracle。完整枚举、边界闭开和 host/kernel 契约仍需接口负责人确认。

### 26.3 模式 3：batch 内 token 拆分

模式 3 需要把一个源 batch 的 query token 分配到多个新逻辑 batch，并同步：

- Q token、query-token sparse indices 和 top-k length；
- BSND/TND KV 上下文，或 PA_BBND block table 行；
- `cu_seqlens_*`、`seqused_*`、`cmp_residual_kv`；
- `B/S/T/max_seqlen_*` 和 origin map。

PA 仍只复制逻辑 table 行，不复制物理 cache。BSND/TND KV 是否整段复制、裁剪到 prefix
或需要显式 query-position 补偿，必须由 oracle 和接口契约决定。无 token 可拆属于
`INVALID_TRANSFORM` 或 no-op deselect（最终规范需统一），不能记 PASS。

### 26.4 模式 4：前插/删除 token

模式 4 比模式 3 风险更高：它直接改变原 token 的位置、TND prefix sum、mask/window
边界和 sparse/top-k 索引。应拆成两个显式操作 `prepend`/`delete-prefix`，每个操作记录：

- 新增/删除 token 的来源和值；
- 哪些旧 token 仍参与比较；
- origin 映射和不可比较 token；
- KV/context 是否同步变化；
- CPU oracle 首个不一致 origin 和原因。

stateful 的 `prefill(input) + decode(token)` 与 `prefill(input+token)` 属于另一类时序等价性，
不能偷偷塞进简单 prepend/delete；是否纳入第二阶段需特性负责人单独定范围。

### 26.5 判定边界

| 情况 | 状态 |
| --- | --- |
| CPU oracle 不等价 | `INVALID_TRANSFORM` |
| 没有可执行 token/组合不在当前模式能力集 | `INVALID_TRANSFORM` 或明确 deselect，不能 PASS |
| level 3/graph 环境不支持 | `UNSUPPORTED` |
| CPU 等价、环境支持、NPU 不一致 | `FAIL` |
| CPU 等价、环境支持、NPU 一致 | `PASS` |

## 27. CI、Artifact 与测试矩阵

### 27.1 分层矩阵

| 层级 | 内容 | 频率 |
| --- | --- | --- |
| CPU PR 门禁 | schema/layout/registry/parser/oracle/compare synthetic 单测 | 每次提交 |
| NPU smoke | 三类 KV layout、模式 1/2、主输出+LSE、小 B/短序列 | 支持 level 3 的每次提交环境 |
| Graph smoke | 模式 1、固定 shape、小 case | 支持 graph+level 3 的专用环境 |
| Nightly | SWA/HCA/CSA、BSND/TND/PA、ori-only/ori+cmp、sparse/top-k、模式 2 graph、模式 3/4 有效组合 | 每夜 |
| Metadata coverage | level-1 四边界、level-2 单核/跨核/一核多 token/混合 | 专项或 nightly |

当前 910_93 环境若仍不支持 level 3，只运行 CPU 门禁和普通基线，NPU batch consistency
结果必须显示 `UNSUPPORTED`，不能把所有 skip 的 job 标绿为“特性通过”。

### 27.2 Artifact

每次一致性运行建议产生 versioned JSON/文本：

- case、mode、profile、seed、order/groups、layout；
- device、torch_npu、CANN、op build、level-3 probe；
- PASS/FAIL/UNSUPPORTED/INVALID_TRANSFORM/CONFIG_ERROR；
- output/LSE fulfill percent、MD5 诊断、首个失败 origin；
- metadata layout version、raw digest、FA/FD coverage；
- graph compile/run 标识和耗时。

大 tensor 不默认保存；只有显式 debug 开关才保存失败输入/输出，并设置大小上限和清理
周期。

## 28. 第二阶段里程碑与文件级改造

| 里程碑 | 主要改动 | 前置/门禁 |
| --- | --- | --- |
| M2.0 语义确认 | kernel/runtime/graph/mask 负责人确认阻断项，冻结 JSON/config schema | 不改代码 |
| M2.1 Mode registry | 修改 `model.py`，新增 `registry.py`，调整 shell/pytest 解析和单测 | `-M 1,2` 兼容回归 |
| M2.2 Raw metadata | 新增 `metadata_layout.py` 和 synthetic/header 对照单测 | 只承诺结构解析 |
| M2.3 精确捕获 | 最小修改原执行函数或增加 sibling，runner 填充 `RunResult.metadata` | 旧二元返回和 eager/graph 回归 |
| M2.4 Level-3 probe | 新增能力枚举、SMLA smoke、缓存和机器可读输出 | 不得把 generic op 成功当 SUPPORTED |
| M2.5 Analyzer | 新增 `metadata_analyzer.py`/report，输出 level-1/2 覆盖 | metadata 字段语义已确认 |
| M2.6 Graph | 新增 graph consistency 测试和 executor，先模式 1 | graph/level-3 契约确认 |
| M2.7 Mode 3 | 新增 `oracle.py`，实现 actual-input golden、token split、状态汇总 | mask/position/PA 语义确认 |
| M2.8 Mode 4 | 扩展 oracle 和 transform，分开 prepend/delete | 模式 3 稳定、stateful 范围确认 |
| M2.9 CI | 小矩阵、nightly、artifact 和 dashboard | 前述阶段独立通过 |

建议每个里程碑独立提交，不把 registry、metadata、graph、模式 3/4 合并成一个不可回滚
提交。所有新能力默认 off，可分别通过 modes/profile/metadata/graph 开关回退到稳定的模式
1/2。

## 29. 第二阶段阻断项清单

| 编号 | 问题 | 负责人 | 阻断范围 |
| --- | --- | --- | --- |
| META-1 | START/END 闭开区间 | kernel/metadata | token/core 覆盖分析 |
| META-2 | FD 索引 7、workspace index/num、split-G 布局 | kernel/metadata | FD/FA 关联和 layout version |
| RT-1 | level 3 官方能力查询及通用/per-op 校验位置 | torch_npu/CANN | probe 的 SUPPORTED 判定 |
| RT-2 | 支持 910_93 level 3 的版本/镜像 | CANN/kernel | NPU 闭环环境 |
| GRAPH-1 | level 3 在 compile/run 阶段的生效点 | torchair | graph executor |
| GRAPH-2 | `torch._dynamo.reset()` 和静态 graph 缓存行为 | torchair | guard/复用策略 |
| SEM-1 | mask mode 完整枚举、query position、window 边界 | 算子接口 | 模式 3/4 |
| SEM-2 | sparse indices/top-k 的 token/batch 语义 | CSA/接口 | 模式 3/4 |
| SEM-3 | PA table 复制与 residual/physical cache 约束 | PA/kernel | 模式 3/4 PA |
| CFG-1 | Excel/PT/sidecar 的唯一配置载体和 case id | 测试框架 | per-case 配置 |
| SCOPE-1 | stateful prefill/decode 是否纳入本阶段 | 特性负责人 | 模式 4 范围 |

## 30. 第二阶段验收门禁

第二阶段不能以“代码已写”作为完成标准，至少满足：

1. `-M 1,2` 与 `--modes reorder,split` 对模式 1/2 行为完全等价，原命令不回归；
2. metadata parser 能严格切分 1024 int32，并在未知字段存在时显式报告；
3. 捕获到的 metadata 与主算子实际消费对象一致，旧 `call_npu` 始终二元返回；
4. level-3 probe 不出现已知假阳性，UNKNOWN/UNSUPPORTED/FAIL 可机器读取；
5. analyzer 的区间和 workspace 语义有源码/负责人依据及 synthetic 单测；
6. graph 模式 1 在支持环境通过，模式 2 的编译成本和执行策略有数据；
7. 模式 3/4 每个进入 NPU 的 case 先通过 actual-input CPU oracle；
8. `INVALID_TRANSFORM` 有明确原因和 origin，不混入 PASS/FAIL；
9. 原 `single/save/load/load_graph` 和第一阶段 85 条 CPU 单测保持通过；
10. CI 至少有一套真正支持 level 3 的环境；否则只能声明框架门禁通过，不能声明特性
    NPU 验收完成。

## 31. CANNBot/GLM-5.2 命令说明

最小调用方式：

~~~bash
cannbot run \
  -m cannbot/glm-5.2 \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '只读分析任务内容；不要修改文件'
~~~

继续上一轮 session 做复审：

~~~bash
cannbot run \
  -c \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '继续上一轮，对方案做反向审查；仍然不要修改文件'
~~~

也可以从第一轮输出的 session id 精确恢复：

~~~bash
cannbot run \
  -s <session-id> \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '继续任务'
~~~

如果 CANNBot 1.0.5 的默认数据库报 `no such column: replacement_seq`，不要直接删除或
改写用户数据库。可以新建临时 `XDG_DATA_HOME`，以 0700/0600 权限复制已有认证文件后
运行；任务结束立即清理，因为临时目录含认证信息。长期方案应升级/迁移 CANNBot 数据库，
而不是一直依赖临时目录。
