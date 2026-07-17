# SMLA/MQSMLA batch 一致性测试开发计划

> 版本：2026-07-17 检视修订版
> 范围：`sparse_flash_mla` 与 `mixed_quant_sparse_flash_mla`
> 状态来源：代码、STC、会议纪要、metadata header/AICPU 源码、已安装 MQ metadata 实测
> 需求与历史台账：`requs.md`

## 1. 当前结论

测试框架不能单方面“打通”尚未支持的 batch consistency。当前工作应拆成三层，并分别
声明状态：

1. **测试控制与变换层**：命名模式、BSND/TND/PA 变换、origin 对齐、MD5/精度比较；
2. **metadata 观察层**：捕获、严格解析和覆盖分析；
3. **主算子验收层**：SMLA/MQSMLA 真正在 level 3 下执行并逐 bit 通过。

本轮已经完成第一层的模式 1/2 和受限模式 3，完成第二层的 parser、SMLA eager 同对象
捕获、MQ 独立 probe 和保守 analyzer。第三层仍被两个事实阻断：

- 当前 SMLA/MQSMLA 主算子源码未形成可验收的 batch-consistency 完整路径；
- 本机 Ascend910_9362 + torch_npu/CANN 9.2 在第一次 NPU 操作时拒绝 deterministic
  level 3，即使 setter/getter 表面成功。

因此当前可声明“测试框架 CPU 门禁通过、metadata level 0 探针通过”，不能声明
“SMLA/MQSMLA batch 一致性通过”。

## 2. 资料裁决原则

实现按以下顺序取证：

```text
STC 001-027 的验收关系
  + 会议纪要的开发者解释
  + batch一致性测试框架的模式定义
  + 两份特性设计文档
  + metadata解析文档
  + 当前生产/消费 metadata 的源码事实
```

STC 决定需要验收什么，源码决定当前接口实际上能做什么。冲突项不能靠模型推测解决：

- 模式 4的 prepend/append 冲突保持阻断；
- `S2=5120` 按数学事实归入 `S2 % 32 == 0`；
- STC 026 的 `B=8` 与 5 个 `cu_seqlens_q` 值不一致，不能直接作为生成参数；
- metadata FD M 是 s1g row，缺上下文时不直接换算成 query token。

## 3. 多模型联合审议记录

### 3.1 已使用模型

本轮通过 CANNBot 分别让以下模型只读审议同一组材料和当前实现：

- `cannbot/glm-5.2`；
- `cannbot/deepseek-v4-pro`；
- `cannbot/qwen3.7-max`。

输入包括 `requs.md`、本计划、五份需求资料、STC Excel、会议纪要、第一阶段代码、SMLA/
MQSMLA metadata header、MQ metadata AICPU 生产代码和主算子消费代码。三个模型未直接
写仓库，最终裁决由主审结合源码和实测完成。

### 3.2 共同关注点

三个模型一致指出：

- 模式 2应逐子 case 与基准 origin 直比，完整 merge 只能辅助诊断；
- 默认 STC 不应随机分组，seed 应限定在可复现排列或 fuzz profile；
- 模式 3必须保持 KV，并明确目标 token 的位置/mask 语义；
- 模式 4在 causal/window mask 下风险最高，必须先有 actual-input oracle；
- 数字模式应收敛到 Python registry，shell 不维护业务含义；
- metadata 必须区分“主算子实际消费的同一对象”和“独立重复调用的 probe”；
- `UNSUPPORTED`、`INVALID_TRANSFORM` 与数值 `FAIL` 不能混为 pytest skip；
- 当前主算子能力和 runtime 能力都不能从普通精度 PASS 推导。

### 3.3 主审对模型意见的修正

模型意见不是源码事实，以下内容已按证据修正：

1. **模式 3采用 STC token 子集语义。** 部分模型把它理解成每个 token 都复制成一个
   新逻辑 batch；STC 002/006/007/012/016 实际是从基准 case 选取 batch/token 子集。
2. **5120 可被 32 整除。** STC 描述和个别模型分类错误，代码不跟随错误文字。
3. **FD task 不等于一个 token。** `FD_M_IDX` 是 s1g index，完整展开需要 group size。
4. **generic NPU op 成功不能证明 SMLA/MQSMLA 支持 level 3。** 只能真实运行目标算子或
   使用官方 per-op 能力查询。
5. **CPU oracle 不能重新随机生成输入。** 必须消费已经变换后的实际 Q/KV/index/table。
6. **MQ metadata probe 不能证明 MQ 主算子 PASS。** 它只证明部署 metadata op 的输出。

## 4. 总体架构

```text
CLI / profile / per-case config
            |
            v
      ModeRegistry
            |
            v
       CaseAdapter -------- SMLA input / MQSMLA op_input
            |
            v
       CaseSchema --------- physical length / valid length / invariant
            |
            v
  BatchCaseTransformer ---- reorder / split / token-split / shape-change guard
            |
            +----> actual-input CPU oracle（待完成）
            |
            v
  operator-specific executor
      |                  |
  SMLA eager         MQSMLA eager（待在 MQ pytest 接入）
      |
  exact metadata capture
      |
      v
  OriginComparator + TensorDigest + existing numeric checker
      |
      +----> MetadataLayout + MetadataCoverageAnalyzer
      |
      v
  PASS / FAIL / UNSUPPORTED / INVALID_TRANSFORM / CONFIG_ERROR
```

跨算子共享的是数据模型、变换、比较和 metadata 解析；主算子调用必须由各自 pytest 的
executor 负责，不能把 MQSMLA 的整段 op 调用复制进 SMLA runner。

## 5. 模式设计

### 5.1 模式 1：`reorder`

目标：同一批 batch 改变顺序后，每个源 token 输出逐 bit 相同。

变换：

- BSND Q/KV 按 batch dim 选择；
- TND Q/KV 按原 prefix sum 切段、重排、拼接并重算 prefix sum；
- PA_BBND 保持物理 cache，只重排 block table 行；
- sparse indices/top-k length 随 Q token；
- `seqused_*`、`cmp_residual_kv` 等 per-batch vector 使用同一 order；
- 更新 B/S/T/max length 和 params/metadata 中存在的镜像字段；
- order 由 seed 固定并保证非恒等。

比较：按 `TokenOrigin(source_batch, source_token)` 还原有效 token，再检查 MD5 和原数值
精度。BSND padding 不进入 digest。

### 5.2 模式 2：`split`

目标：基准多 batch 调用与任意选中 batch 子调用的对应结果一致。

主验收：

```text
for each derived subcase:
    expected = select baseline output by derived origins
    compare expected directly with derived output
```

规则：

- 显式 groups 可只覆盖部分 batch；
- 不同 group 间不允许重复 batch；
- 默认基础 profile 使用连续确定性二分，与 seed 无关；
- `--randomize` 才打乱 batch 并使用 seed；
- groups 完整覆盖时可额外 merge，但不作为唯一或首要 oracle。

### 5.3 模式 3：`token-split`

目标：选中 query token 的结果不受同 batch 其他 query token 组合影响。

当前实现：

- 显式 `batch_ids + token_ids_by_batch`；默认选每个 batch 的第一个有效 token；
- BSND 支持不同 batch 选取不同 token 数，物理 S1 pad 到最大值，`seqused_q` 保存有效数；
- TND 直接拼接选中 token并重算 `cu_seqlens_q`；
- Q、ori/cmp sparse indices、ori/cmp top-k length 同步选择；
- KV、block table、KV lengths 和 residual 保持所选 batch 的完整上下文；
- 比较只读取有效 token，不比较 padding；
- 位置敏感 mask 检查目标 token 到有效 Q 右边界的距离，变化则 `INVALID_TRANSFORM`。

当前限制：静态 guard 只覆盖已知位置关系，不能代替 actual-input golden；随机跨 batch/token
重组和所有 mask/top-k 组合进入正式 CI 前，必须完成第 10 节 oracle。

### 5.4 模式 4：`shape-change`

当前不实现。待解决：

- STC 要求尾部追加 token；
- 框架/会议建议首部插入以维持右对齐；
- “前 300 token 相同”背景又偏向前缀比较；
- causal/window、sparse threshold 和 TND prefix 都可能变化。

完成条件是接口负责人给出规范变换，并且 actual-input oracle 对每个目标 origin 证明等价。

## 6. STC 001-027 映射

| STC | 场景 | 框架能力 | 当前状态 |
| --- | --- | --- | --- |
| 001 | 128 调度块基准 | 基准 + metadata Level-1/2 | parser 可用，主算子验收阻塞 |
| 002 | 所有 batch 第一个 token | mode 3 | 变换已支持 |
| 003 | 选部分 batch | mode 2 subset | 已支持直比 |
| 004 | 追加 token 改 S1 | mode 4 | 资料冲突，阻塞 |
| 005 | 大于 128 调度块基准 | 基准 + metadata | parser/classifier 可用 |
| 006 | 1 batch / 1 token | mode 3 + batch subset | API 已支持 |
| 007 | 每 batch 前 x token | mode 3 variable valid length | 已支持 BSND padding 过滤 |
| 008 | 选 2 batch | mode 2 subset | 已支持 |
| 009 | 追加 token | mode 4 | 阻塞 |
| 010 | `S2/32 < 128` 基准 | Level-1 less | classifier 已支持 |
| 011 | 选部分 batch | mode 2 | 已支持 |
| 012 | 每 batch 第一个 token | mode 3 | 已支持 |
| 013 | 追加 token | mode 4 | 阻塞 |
| 014 | `S2=5120` 基准 | Level-1 greater/divisible | 分类已纠正；STC 文字待修 |
| 015 | 选部分 batch | mode 2 | 已支持 |
| 016 | 每 batch 第一个 token | mode 3 | 已支持 |
| 017 | 追加 token | mode 4 | 阻塞 |
| 018 | 多次打乱 batch | mode 1 fuzz | 可由 seed/profile 驱动 |
| 019 | 多次打乱 token | mode 3 fuzz | API 支持任意顺序；正式 fuzz 待 oracle |
| 020 | batch/token 随机组合 | mode 1+2+3 | 原语具备；组合配置与 oracle 待完成 |
| 021 | ACLNN 直调 | executor variant | 未接入 |
| 022 | torch 直调 | eager executor | SMLA 框架具备，level 3阻塞 |
| 023 | ACL graph | graph executor | 未接入 |
| 024 | N=125 随机组合 | nightly matrix | 待主算子、oracle 和大矩阵 |
| 025 | N=3 随机组合 | nightly matrix | 同上 |
| 026 | TND 随机组合 | TND 原语 | 原语已支持；STC B/cu_seqlens 数据冲突待确认 |
| 027 | PA_BBND 随机组合 | PA table 原语 | 原语已支持；组合验收待完成 |

STC 是验收场景表，不是可直接运行的完整 Excel generator 输入。001-017 的 baseline/derived
引用关系和 018-027 的随机 profile 需要转成 versioned per-case 配置。

## 7. 双算子适配策略

### 7.1 已完成的共享数据面

`CaseAdapter` 识别：

| 标准概念 | SMLA | MQSMLA |
| --- | --- | --- |
| tensor 容器 | `input` | `op_input` |
| batch | 顶层/metadata `B` | metadata `batch_size` |
| Q heads | `N1` | `num_heads_q` |
| KV heads | `N2` | `num_heads_kv` |
| head dim | `D` | `head_dim` |

同一 schema/transform 已用 MQSMLA synthetic case 验证 reorder 和 invariant，不再维护两套
字段变换。

### 7.2 MQSMLA executor 的落地方式

MQ 主算子 executor 尚未接入，计划如下：

1. 在 `attention/mixed_quant_sparse_flash_mla/tests/pytest` 内新增很薄的 executor；
2. executor 调用 MQ 现有 process，并返回统一 `RunResult`；
3. MQ process 使用与 SMLA 相同的默认关闭 `metadata_sink` 或 sibling API，捕获主算子实际
   消费的 metadata；
4. 通过公共包位置或测试根目录配置复用 `CaseAdapter/transform/compare`，不能复制源码；
5. 保留 MQ 原 single/batch/graph 返回契约；
6. 先接 eager 模式 1/2，再接受限模式 3和 graph。

在原始“只允许修改 SMLA pytest”边界未解除前，本轮不修改 MQ pytest。后续实施 MQ executor
需要用户明确允许跨到 MQ tests 目录，或先批准把共享包移动到两个算子都可依赖的位置。

## 8. 配置接口

### 8.1 已实现

规范模式名：

```text
reorder, split, token-split, shape-change
```

profiles：

- `blackbox-basic = reorder,split`；
- `blackbox-supported = reorder,split,token-split`；
- `blackbox-all` 包含 shape-change，当前会明确配置失败。

推荐：

```bash
bash test_run.sh consistency -P ./data --modes reorder,split
bash test_run.sh consistency -P ./data --profile blackbox-supported
```

兼容形式 `-M 1,2` 由 Python registry 映射并发 deprecated warning。shell 不再理解
`1/2/3/4` 的含义。

### 8.2 待实现的 per-case 配置

优先在保存 PT 时写入可选 namespace：

```python
"consistency": {
    "schema_version": 1,
    "profile": "stc-fixed",
    "modes": ["split", "token-split"],
    "split_groups": [[0, 3]],
    "token_batches": [0, 2],
    "token_ids": [[0], [0, 1]],
    "baseline_case_id": "batch_consistency_001",
}
```

若不允许扩展 PT，则使用以精确 case id 为键的 versioned JSON manifest。PT namespace 与
sidecar 必须二选一，不能形成两个真相源，也不能运行时按模糊文件名重新读 Excel。

配置优先级计划：

```text
显式 CLI > 显式环境变量 > PT/manifest per-case > profile > blackbox-basic
```

## 9. Metadata 阶段二方案与进展

### 9.1 M2.2 严格布局解析：已完成

`MetadataLayout`：

- 只接受 `torch.int32` 和恰好 1024 个元素；
- 当前布局为 36x9 + 72x8 + 124 reserved；
- legacy 文档布局为 36x8 + 72x8 + 160 reserved，单独 version 0；
- 负 `int32` 按同 bit pattern 转成 `uint32`；
- FD slot 7 和 reserved 非零均显式报告。

### 9.2 M2.3 SMLA eager 精确捕获：已完成

原 `call_npu(input_data)` 增加默认 `None` 的 `metadata_sink`：

- metadata 先生成并作为同一个 tensor 传给 SMLA；
- 主算子和 synchronize 成功后才复制到 CPU sink；
- 旧调用仍得到 `(output, softmax_lse)`；
- sink 未启用时不产生额外 CPU copy；
- 一致性 `SmlaExecutor` 把捕获值写入 `RunResult.metadata`。

graph 和 MQ 主算子尚未使用该精确捕获方式。

### 9.3 MQ metadata-only probe：已完成

CLI：

```bash
python3 -m batch_consistency.metadata_probe \
  --pt /path/to/mqsmla.pt --device 0 --level 3 --s2 4096,5120
```

probe 会恢复旧 deterministic level，输出 parser/analyzer JSON，并固定写明：

```text
metadata-only; main operator batch consistency not asserted
```

本机实测 level 0 成功、level 3在第一次 NPU op 被 runtime 拒绝。安装目录为 root 0750 导致
developer 不能直接 source，自定义包权限需环境侧修正。

### 9.4 M2.5 保守 analyzer：部分完成

已输出：

- enabled FA/FD core；
- raw FA start/end 和 endpoint chain 是否连续；
- FD reduction task 的 `(bn2, m, workspace_index, workspace_num)`；
- 同一 FD task 的 AIV core 与 M ranges；
- workspace split 最大值、空 M range、非法 split 数和未知 slot；
- Level-1 四类 S2、规约块大小和块数；
- reserved 非零和分析限制。

完整 Level-2/token-core 展开仍需：

1. 把 `bn2` 用 N2 映射为 `(batch, n2)`；
2. 用 group size 和 `mBaseSize` 把 s1g/M range 映射为 query token/head group；
3. 用每 batch 的 ori/cmp 有效 S2、mask、top-k 重建每行的 S2 block 数；
4. 按 `[start,end)` 展开每个 FA core 的首/尾 partial row 与中间完整 row；
5. 处理 split-G 成对重复 FA record；
6. 用 FD workspace 关联跨核 partial，并校验无重叠/无缺失；
7. 聚合“全单核、存在跨核、一核两个跨核头、单核+跨核混合”等覆盖标签。

### 9.5 新发现的源码阻断

当前 MQ metadata AICPU：

- 已存在按 block 连续切分和 FD record 生成代码；
- `supportFd` 成员默认 `false`；
- 当前工作树未检索到把 `supportFd` 设为 true 或读取 level 3 的路径。

需 metadata 开发者确认 PR/安装包是否漏合、是否由外部变体设置，或是否还有未同步源码。
在此确认前，analyzer 可以解析，但自动场景不能假设 FD 一定产生。

## 10. Actual-input CPU golden oracle

模式 3/4 的正式门禁必须使用已经变换的实际 tensor：

```text
baseline actual tensors -> CPU result A
derived actual tensors  -> CPU result B
B by TokenOrigin -> align to A
等价 -> VALID，允许 NPU
不等价 -> INVALID_TRANSFORM，记录第一个 origin
```

不能调用 `gen_data(params)` 重生随机 Q/KV。实现步骤：

1. 从 SMLA golden 抽出“消费现有 Q/KV/sparse/table/length 的 forward adapter”；
2. 保持原随机生成与保存接口不变；
3. 先支持 BSND + full mask + 无 sparse 的小 case；
4. 增加 TND、PA、cmp、top-k、causal/window；
5. 每次扩展都用正例和故意破坏位置的负例锁定；
6. MQSMLA 复用统一 oracle 协议，但量化输入需要自己的 actual-input golden adapter；
7. oracle 通过只证明变换等价，最终仍需 level-3 NPU MD5。

当前模式 3的 `TransformSemanticGuard` 是过渡门禁，不应在文档或 CI 中标成 oracle 完成。

## 11. Graph 与 ACLNN

### 11.1 Graph

graph 可共享 transform/compare/guard，但需要独立 executor：

1. 先实现 mode 1，同 shape 可评估编译复用；
2. mode 2/3 shape 改变时每个子 case 可能单独编译，先放 nightly；
3. 捕获 graph 实际消费的 metadata；
4. 明确 level 3在 compile 还是 run 时读取；
5. 记录 `torch._dynamo.reset()`、静态 kernel build dir 和 cache 对结果的影响。

### 11.2 ACLNN

STC 021 需要 C++/ACLNN executor：

- 使用 `aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3)`；
- 保存/恢复原系统配置；
- 复用同一 PT/manifest 和 origin 期望；
- 输出与 torch eager 相同的机器可读状态和 digest；
- 不把 Python setter 成功当成 ACLNN 路径证据。

## 12. 状态模型

| 状态 | 定义 | CI 含义 |
| --- | --- | --- |
| `PASS` | 变换有效、环境支持、MD5 与数值精度都通过 | 成功 |
| `FAIL` | 执行异常、metadata/analyzer 错误或有效 case 不一致 | 失败 |
| `UNSUPPORTED` | runtime/device/operator/graph 明确不支持必要能力 | 单列，不得称特性通过 |
| `INVALID_TRANSFORM` | 源 case 无法在该模式保持数学等价 | 单列，需原因/origin |
| `NOT_APPLICABLE` | 当前 PT 对所选模式没有可执行 test item | 单列 |
| `CONFIG_ERROR` | mode/profile/per-case schema 非法 | 入口失败 |

pytest skip 只是内部表达方式；shell/JSON artifact 必须保留上述分类。一个 PT 同时运行多个
mode 时，后续 JSON 应按 `(case, mode)` 记录，避免当前 shell 的单 case 汇总丢失混合状态。

## 13. 测试矩阵

| 层级 | 内容 | 当前状态 |
| --- | --- | --- |
| CPU PR | registry、schema、layout、mode 1/2/3、origin、padding、MQ 格式、metadata parser/analyzer | 114 passed |
| 默认关闭检查 | 新一致性入口不触发 NPU | 3 skipped |
| 原功能收集 | SMLA single/batch/graph | 15 collected |
| NPU smoke | mode 1/2、BSND/TND/PA、output+LSE | level 3阻塞 |
| Metadata probe | MQ level 0 raw layout | 已实测 |
| Graph smoke | mode 1 eager-equivalent | 未实现 |
| Nightly | SWA/HCA/CSA、N=3/125、随机组合、mode 2/3 graph | 未实现 |
| STC 001-027 | baseline/derived case pair + metadata coverage | 配置链和主算子阻塞 |

每次提交最低门禁：

```bash
source /mnt/workspace/gitCode/env.sh
cd attention/sparse_flash_mla/tests/pytest

python3 -m pytest -q test_batch_consistency_unit.py
python3 -m pytest -q test_sparse_flash_mla_batch_consistency.py
python3 -m pytest --collect-only -q \
  test_sparse_flash_mla_single.py \
  test_sparse_flash_mla_batch.py \
  test_sparse_flash_mla_batch_graph.py
bash -n test_run.sh
git diff --check
```

## 14. 阶段与里程碑状态

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| P1.0 | 模式 1/2、三类 layout、origin compare、level guard | 已完成，基线 `b680bfbdd` |
| P1.R1 | 检视修复：类封装、mode 2直比、确定性分组、双算子背景 | 本轮完成 |
| M2.1 | 命名 registry、profile、legacy alias | 本轮完成 |
| M2.2 | raw metadata parser + synthetic tests | 本轮完成 |
| M2.3 | SMLA eager exact metadata capture | 本轮完成 |
| M2.4 | MQ metadata probe + level 3真实能力证据 | probe 完成，环境结论 UNSUPPORTED |
| M2.5a | 保守 FA/FD analyzer | 本轮完成 |
| M2.5b | 带上下文的完整 Level-2 token/core analyzer | 待字段/算法确认 |
| M2.6 | SMLA graph executor/capture | 待实现 |
| M2.7a | STC token subset transform | 本轮受限完成 |
| M2.7b | actual-input oracle + mode 3正式矩阵 | 待实现 |
| M2.8 | mode 4 shape change | 阻塞 |
| M2.9 | MQ main executor + exact capture | 待授权跨 MQ pytest |
| M2.10 | per-case STC config、JSON artifact、CI/nightly | 待实现 |

## 15. 后续实施顺序

### 优先级 A：让真实特性可运行

1. torch_npu/CANN 确认支持 level 3的版本和 910_93/950 范围；
2. SMLA/MQSMLA tiling 确认 `GetDeterministicLevel()==3` 路径和 tiling key；
3. MQ metadata 开发者解释 `supportFd` 的设置入口和已安装包版本；
4. 获得一个可运行的最小 SMLA 与 MQSMLA level-3 smoke case。

### 优先级 B：冻结 STC 数据契约

1. 修正 STC 014 的 5120 描述；
2. 修正/解释 STC 026 的 B 与 cu_seqlens 数量；
3. 确定 001/005/010/014 到派生 case 的精确 tensor 复用关系；
4. 选择 PT namespace 或 JSON manifest；
5. 固化 mode 4 prepend/append 规范。

### 优先级 C：完成可证明的模式 3/4

1. actual-input SMLA golden adapter；
2. mode 3 full-mask 小矩阵；
3. mode 3 position-sensitive/sparse/PA 扩展；
4. mode 4按裁决实现；
5. MQ quantized actual-input oracle。

### 优先级 D：执行面扩展

1. MQ eager executor；
2. SMLA/MQSMLA graph mode 1；
3. mode 2/3 graph nightly；
4. ACLNN STC 021；
5. JSON artifact 和真正支持 level 3的 CI。

## 16. 阻断项清单

| 编号 | 问题 | 需要确认方 | 阻断范围 |
| --- | --- | --- | --- |
| RT-1 | level 3支持的 torch_npu/CANN/SoC 组合 | runtime/CANN | 全部 NPU 验收 |
| OP-1 | SMLA/MQSMLA level-3 tiling key 和功能合入状态 | 主算子开发 | 主算子 PASS |
| META-1 | MQ `supportFd` 设置路径/安装包源码版本 | metadata 开发 | FD 生成与自动覆盖 |
| META-2 | FD slot 7最终定义 | metadata/kernel | layout version 冻结 |
| META-3 | split-G 和 group/M 展开契约 | metadata/kernel | 完整 Level-2 analyzer |
| SEM-1 | mode 4 prepend/append 正式语义 | STC/接口负责人 | mode 4 |
| SEM-2 | mask/top-k/position 完整规则 | 接口/golden | mode 3/4 oracle |
| DATA-1 | STC 014 5120 描述修订 | STC 负责人 | Level-1 报告一致性 |
| DATA-2 | STC 026 B/cu_seqlens 冲突 | STC 负责人 | TND STC |
| CFG-1 | PT namespace 或 manifest | 测试框架 | 001-027 自动化 |
| SCOPE-1 | 允许修改 MQ pytest/公共包位置 | 用户/维护者 | MQ executor |
| GRAPH-1 | level 3读取时机和 compile cache | torchair | graph |

## 17. 验收门禁

最终不能以“代码已写”作为完成标准。至少需要：

1. SMLA 与 MQSMLA 各有真实 level-3 eager smoke PASS；
2. 模式 1/2 每个派生 case 对应 origin MD5 与数值精度都通过；
3. 模式 3/4 所有 NPU case 先通过 actual-input oracle；
4. BSND/TND/PA_BBND 和 SMLA/MQSMLA 双格式均有覆盖；
5. output 与可选 LSE 都进入 bitwise 检查；
6. metadata 来自主算子实际消费对象，parser/layout 与部署版本一致；
7. Level-1 四类和 Level-2 指定覆盖标签能自动报告；
8. STC 001-027 每项都有 PASS/FAIL/UNSUPPORTED/INVALID 的机器可读结果；
9. torch eager、ACLNN、graph 的要求分别验收；
10. 原 single/save/load/load_graph 无行为回归；
11. CI 至少存在一套真正支持 level 3的环境；
12. 文档记录版本、环境、seed/profile、metadata digest 和失败 origin。

## 18. CANNBot 使用说明

启动一次只读审议：

```bash
cannbot run \
  -m cannbot/glm-5.2 \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '阅读 requs/requs.md、开发计划、STC、会议纪要和当前代码；只读分析，不修改文件'
```

替换模型即可并行获得不同视角：

```bash
cannbot run -m cannbot/deepseek-v4-pro \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '只读反向审查 batch consistency 方案，不修改文件'

cannbot run -m cannbot/qwen3.7-max \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '只读核对 STC 与实现差异，不修改文件'
```

继续最近 session：

```bash
cannbot run -c \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '继续上一轮，根据新证据复审；仍然不要修改文件'
```

或用首轮输出的 session id：

```bash
cannbot run -s <session-id> \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '继续任务'
```

本机 CANNBot 1.0.5 的默认数据库曾报
`SQLiteError: no such column: replacement_seq`。不要删除或直接改写用户数据库；优先升级/
迁移 CANNBot。若临时隔离 `XDG_DATA_HOME`，目录和认证文件必须分别为 0700/0600，任务后
立即清理，绝不能把认证信息写进仓库或日志。
