# `requs/data` 外部 batch/seq 一致性方案评估

> 评估日期：2026-07-21
> 结论：本目录中的原始文件作为设计输入和历史证据保留，**不属于 SMLA/MQSMLA 正式执行链**。
> 正式实现仍位于两个算子各自的 `tests/pytest_v2/batch/consistency/`。

三份原始输入按收到时的字节保留，包括参考脚本的 CRLF 和原文 EOF 空行；它们不参与正式
源码 lint/CI，也不会被导入执行。参考 Python 仅用内存 `compile()` 做无副作用语法读取。

## 1. 文件清单与原方案目标

| 文件 | 原方案用途 | 本项目处理 |
| --- | --- | --- |
| `算子多batch一致验证.md` | 为 aclnnfuzz 定义输入 seed、输出分组标识、level0/1/2 和覆盖数量 | 作为通用测试框架参考资料保留 |
| `aclnnAdd.xlsx` | 8 条 aclnnAdd 示例，展示重复 `attr_*` 列中的 input/output seed info | 只用于理解 schema，不转成 SMLA/MQSMLA 用例 |
| `compare_batch_tensor_by_result.py` | 预检 Excel，读取结果 bin，并按 output seed 分组做逐 bit 相等比较 | 不导入、不调用、不复制到 pytest_v2 |

原方案解决的是通用算子的“给某个 tensor 的某个轴切片生成相同数据，然后把应相同的输出切片
归到同一组比较”。其输入项为 6 元素：

```text
[tensor_index, batch_axis, batch_indices, range, seed, pre_shape_index]
```

输出项为 5 元素：

```text
[tensor_index, batch_axis, batch_indices, output_seed, pre_shape_index]
```

这里的 `output_seed` 是**关系分组标签**，不是“算子的输出值等于输入 seed 之和”。

## 2. 与 SMLA/MQSMLA 的本质差异

SMLA/MQSMLA 不能仅靠固定一个 tensor 轴的随机 seed 证明数学输入相同。一个 query token 的
计算上下文同时依赖：

- Q、ori/cmp KV、有效长度和物理长度；
- BSND/TND/PA_BBND 的不同寻址方式；
- TND `cu_seqlens` 前缀和、PA block table；
- RightDownCausal/Band 的位置窗口；
- sparse indices、top-k、sinks、cmp ratio 和 residual；
- MQSMLA 的量化 KV、scale 和量化模式。

因此本项目继续采用“从一个真实 baseline PT 同步派生输入 + actual-input semantic oracle”的
路线。seed 只控制可复现的排列或 fuzz，不作为输入等价性的证明。

另外，外部文档的 level0/level1/level2 是它自己的**用例分类**，与 CANN deterministic
level 0/1/2/3 不是同一概念；也不能把它们与本项目模式 1-4 做一一映射。本项目四种模式都
包含 baseline/derived 多次完整调用。

## 3. 逐项采纳结论

| 外部设计 | 结论 | 在本项目中的落点 |
| --- | --- | --- |
| 显式声明“哪些输出应相同” | 采纳思想 | 使用 `TokenOrigin(source_batch, source_token)`，并输出 versioned relation manifest |
| 输出关系分组 | 采纳但增强 | relation id 绑定 baseline 内容指纹；origin 压缩为连续 spans，避免 16K token JSON 膨胀 |
| Excel/配置执行前预检 | 采纳 | canonical mode、groups、split sizes、batch/token 边界、重复 origin、scope 数量和 oracle 均在 NPU 前检查 |
| min/mid/max 位置覆盖 | 采纳为诊断 | manifest 报告 source batch/token 的 min/mid/max 命中次数，不把数量当 PASS |
| level0/1/2 分层思路 | 部分采纳 | 转成 mode/layout/mask/visible-S2/来源位置覆盖清单，不复用原 sheet 名义 |
| 相同 seed + range 重生成切片 | 不采纳 | 派生 tensor 直接来自 baseline PT，避免两套等价性来源 |
| 单 `batch_axis`/`pre_shape_index` 切片 | 不采纳 | 无法表达 TND prefix sum、PA table 和位置 mask 联动 |
| `output_seed` 数字作为唯一关系键 | 不采纳 | 可能碰撞且丢失 batch/token 坐标；改用有类型的 TokenOrigin |
| 外部 Python 脚本 | 不采纳 | 有正确性和安全问题，见第 4 节 |
| 500/200/200 固定数量 | 不采纳 | SMLA 以需求模式、layout、mask、S2 分类和 STC 状态覆盖为准 |

已有能力不重复建设：PT save/load、CPU golden、MD5/原精度双门禁、FP8 原始 bit 比较、
actual-input oracle 和 metadata 白盒分析均继续使用 pytest_v2 原实现。

## 4. 外部脚本审计

以下问题是“不直接复用”的具体依据，原文件保持不动以便追溯：

1. 文档明确 input seed info 是 6 元素，但脚本在 `create_input_info` 和
   `vailed_input_seed_info_attr` 中判断 `len(...) == 5`，随后又访问索引 5。对文档中的合法
   6 元素输入，循环会直接跳过，所谓 input 预检实际没有执行；若真传 5 元素则会越界。
2. CSV/Excel 的 attrs、shape、dtype 和动态 slice 多处直接使用 `eval`。这些输入一旦可由
   外部修改，就存在任意 Python 代码执行风险；正式工具至少应使用严格 schema、
   `ast.literal_eval` 或 JSON，并直接构造索引 tuple。
3. 只检查 `index >= shape`，没有统一检查负 index、axis 范围、`pre_shape_index` 长度、空
   batch list、重复 index、bin 文件元素数和 reshape 前 byte size。
4. 文档要求“相同 seed 必须绑定相同 dtype/range/slice shape”，脚本没有建立全局约束表去
   验证这些条件。
5. 比较分组键只有 `output_seed/tensor_index/dtype`，没有 operator、关系 id 和完整输出语义；
   不同算子或不相关 case 复用 seed 时可能被误放进同一组。
6. `np.array_equal` 只提供一种原始相等门禁，没有本项目要求的 TokenOrigin 对齐、原 CPU
   精度门禁、LSE、有效 token/padding 处理和 FP8 NaN bit 特判。
7. 日志固定追加到当前目录 `auto_run.txt`，没有显式 artifact 目录或并发隔离；预检函数的
   返回值也没有汇总为可机器判断的 PASS/CONFIG_ERROR 报告。

`get_numpy_dtype` 把 fp32/int32 映射为 `uint32`、fp16/bf16 映射为 `uint16`，若目标是比较
原始存储 bit，这个“同位宽无符号重解释”本身并非错误；但脚本没有明确声明该契约，也没有
校验 endian、文件 byte size 或覆盖 FP8 类型，所以不能把它当成数值精度比较器。这一点与
DeepSeek 审议中“float32 映射 uint32 必然错误”的判断不同，主审按代码真实用途作了修正。

`aclnnAdd.xlsx` 当前只有一个 `test` sheet 和 8 条示例，并没有实际提供文档所述的
level0/level1/level2 三套验收集，因此它只能证明字段写法，不能证明覆盖计划已经实现。

## 5. 已落地：relation manifest v1

两个算子新增同源文件：

```text
tests/pytest_v2/batch/consistency/relation_manifest.py
```

`consistency_prepare` 现在除原报告外，还生成：

```text
case_XXX/consistency_relation_manifest.json
smla_consistency_relation_manifest.json
mqsmla_consistency_relation_manifest.json
```

核心结构示意：

```json
{
  "schema_version": 1,
  "kind": "smla_mqsmla_batch_consistency_relation",
  "operator": "smla",
  "claim_scope": "CPU-only relation precheck; no NPU batch-consistency PASS claim",
  "scenario_coverage": {
    "implemented": ["same-source-cross-call"],
    "supplemental_gaps": ["independent-cross-case", "single-call-intra-case"]
  },
  "baselines": [{
    "case_id": "smla:case_name:<content-sha-prefix>",
    "input_content_fingerprint_sha256": "...",
    "valid_q_lengths": [4, 4]
  }],
  "relations": [{
    "mode": "reorder",
    "scenario": "same-source-cross-call",
    "baseline_case_id": "...",
    "comparison_identity": {
      "type": "TokenOrigin",
      "mapped_token_count": 8,
      "origin_spans": [
        {"target_start": 0, "target_end": 4,
         "source_batch": 1, "source_token_start": 0, "source_token_end": 4}
      ]
    },
    "precheck": {"status": "PASS"}
  }]
}
```

manifest v1 当前完成的是“实际 baseline PT -> pytest 自动派生调用”的机器可读关系和离线
预检，是开发计划 P3.4 的数据契约基础；它**还不是** STC 001-027 的完整输入清单。STC 中
014/026 等冲突未由 owner 修正前，不会把猜测写成 `CONFIRMED` case。

预检门禁包括：

- mode 必须是 canonical name，Excel 中的数字 `1,2` 被拒绝；
- baseline id 绑定去除临时路径后的输入内容 SHA256；来源路径仍单独保留；
- TokenOrigin 不为空、不重复，并落在 baseline 有效 batch/token 范围；
- origin spans 目标区间连续、源/目标宽度一致，且有独立 SHA256 防篡改；
- mapped token 数与 `_consistency_expected_scope` 一致；
- 模式 3/4 semantic oracle 必须通过；不完整 partition 明确 warning；
- 汇总 mode、layout、mask pair、visible S2 四类、floor/ceil 冲突计数和 min/mid/max 命中；
- `claim_scope` 明确只代表 CPU 输入/关系预检，不能冒充 NPU 或 kernel metadata 覆盖。

## 6. 多模型审议与主审修正

本轮使用隔离的 CANNBot session，分别让以下模型只读审议同一批资料：

- `cannbot/glm-5.2`：架构复用与最小增量；
- `cannbot/deepseek-v4-pro`：脚本安全、正确性和 manifest 反审；
- `cannbot/qwen3.7-max`：需求映射、覆盖维度和边界遗漏。

共同有效意见是：不引入 seed 平行生成器；保留 TokenOrigin；借鉴声明式关系、预检、覆盖清单
和 min/mid/max 诊断；manifest 必须绑定 PT/input digest，并保留 NPU claim 边界。

主审纠正了三类模型表述：

1. 本项目 mode 1-4 都是 baseline/derived 多次完整调用，不能把 mode 1 或 mode 3 称为外部
   level1 的“单次调用内比较”，也不能把 mode 1 简称“跨 case 模式”；
2. `output_seed` 是分组标签，不是 attention 输出值等于 seed 的算术关系；
3. 同位宽 `uint32/uint16` 读取可用于 raw-bit equality，问题是契约和校验不完整，而不是
   fp32 映射 uint32 这一行天然错误。

多模型只负责找遗漏和提出备选方案，没有直接修改仓库。最终代码、状态和文档由主审依据
需求、源码和本地测试裁决。

## 7. 后续计划

不等待 NPU 仍可继续：

1. 给 STC 001-027 增加 source manifest，引用本次 relation schema，而不是另造 output seed；
2. 等 owner 修正 STC 014/026 等数据后，将状态从 `BLOCKED_BY_SPEC/INVALID_DATA` 升级；
3. 增加 TND、PA_BBND、RightDown/Band 小 shape workbook，观察 relation coverage 的缺口；
4. 把多个 workbook/manifest 汇总成需求覆盖矩阵，但不把未运行 NPU 的 cell 标为 PASS；
5. level 3 环境可用后，在同一 relation id 下追加 NPU output/metadata digest 和最终状态。

NPU 主算子、runtime level 3、PR floor/需求 ceil 和真实 FA/FD coverage 仍需等待对应 owner，
本目录参考方案不能解除这些阻断。

## 8. 两类场景的精确映射（新增检视）

外部文档提出的“不同用例间比较”和“同一用例内位置比较”本身合理，与模式1-4不冲突；
但`case`和“同一用例”必须区分生成来源与算子调用边界：

| 分类 | 调用/来源 | 当前状态 |
| --- | --- | --- |
| `same-source-cross-call` | 同一baseline PT同步派生，执行多个完整算子调用 | 当前模式1-4 |
| `independent-cross-case` | 两个独立编写或生成的PT，通过显式关系声明等价输出位置 | 外部要求的补充场景，待实现 |
| `single-call-intra-case` | 一次算子调用的输出内，两个数学等价位置直接互比 | 外部要求的补充场景，待实现 |

因此不能因为baseline和derived在pytest里表现成两个case，就把模式1称为“独立跨case”；也不能
因为模式3比较不同token区间，就称为“单次调用内比较”。relation manifest和在线result现已
固定记录第一类，并显式输出后两类gap。

后两类会继续借鉴外部`output_seed`的“关系标签”思想，但正式关系必须绑定operator、两个PT
内容指纹或单次输出位置、TokenOrigin/位置范围和actual-input等价证据。随机seed碰巧生成相同值
不构成Attention数学输入等价，也不能成为PASS依据。
