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

第一阶段完成条件：

- 模式 1、2 覆盖 BSND、TND、PA_BBND；
- 变换后所有 schema invariant 通过；
- 主输出和可选 LSE 均能按 origin 对齐；
- 数值标准与现有 pytest 对齐；
- metadata parser 能解析 1024 int32 并输出 FA/FD 覆盖；
- 原有 `single/save/load` 测试结果不变；
- 失败日志包含基准 case、模式、seed、order/split、layout 和首个失败 origin。

全部完成条件：

- 模式 3、4 对支持组合有明确前置条件；
- 不支持组合报告 `INVALID_TRANSFORM` 或带原因的 skip；
- level-1 四类边界均有用例；
- level-2 单核、跨核、单核多 token 和混合模式均有覆盖；
- 小矩阵进入 CI，大矩阵进入夜间回归；
- 文档、代码常量与 kernel metadata header 保持一致。
