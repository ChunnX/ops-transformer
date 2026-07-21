# SMLA/MQSMLA batch 一致性测试开发计划

> 版本：2026-07-21 检视意见三新增意见推进版
> 范围：`sparse_flash_mla`（SMLA）与 `mixed_quant_sparse_flash_mla`（MQSMLA）
> 第一需求源：`docs/SMLA batch一致性.md`
> 需求、冲突、文件级开发日志与逐项检视答复：`requs.md`

## 1. 当前结论

阶段二的测试侧代码已经完成到以下边界：

1. 两个算子分别拥有自己的 `tests/pytest_v2`，并保留各自原 pytest 的输入生成、CPU
   golden、PT save/load、metadata + 主算子、精度比较和 graph 文件；
2. 模式 1-4 均有命名配置、layout-aware 变换、`TokenOrigin` 对齐、NPU MD5 与原 CPU
   `check_result` 双门禁；
3. 模式 3/4 有 actual-input 语义 oracle，验证 mapped Q、可见 KV、mask window、sparse
   indices、top-k 与 sinks；
4. MQSMLA 已有自己的 eager executor 和 `metadata_sink`，不再只做 metadata probe；
5. metadata 白盒分析可从 1024-int32 展开到 batch/N2/query-token/S2-block/FA core/FD
   workspace，并报告 Level-1/2 覆盖、结构错误和 PR floor/需求 ceil 公式冲突；
6. 双算子已有正式 CPU-only `consistency_prepare`：真实跑通 Excel、原 golden、baseline PT、
   模式1-4、派生PT round-trip和详细输入报告；
7. 公共实现已收拢到各自 `batch/consistency/` 子包，临时 metadata-only probe 已删除，镜像
   一致性由单测门禁；
8. 借鉴外部aclnnfuzz方案的“显式关系+预检+覆盖诊断”，双算子离线链新增relation manifest
   v1；保留TokenOrigin，不引入seed重生成平行框架。
9. 在线执行新增versioned suite config v1：同一文档按精确selector为每个PT解析独立有效配置，
   并在NPU前完成B/Q相关预检；
10. 每个`(operator,PT,mode)`均输出结构化JSON，汇总器只统计通过schema和level3门禁的artifact，
    不再解析pytest文本；
11. 当前模式1-4统一标注为`same-source-cross-call`，独立跨case和单次调用内位置比较作为两个
    补充gap推进；
12. metadata打桩已完成CPU-only候选规划器与默认关闭hook，真实注入仍受包版本、FD契约、
    公式和level3阻断。

但“代码完成”不等于“特性验收完成”。当前本机 torch_npu/CANN 在设置 level 3 后的第一条
NPU 操作处拒绝该等级；本地算子源码快照也没有 PR8737 的完整改动。因此当前只能声明：

```text
测试侧变换/比较/白盒分析：CPU 门禁通过
Excel -> CPU golden -> baseline/derived PT：双算子离线门禁通过
真实 SMLA/MQSMLA level-3 batch consistency：UNSUPPORTED / 待版本对齐
```

严禁自动退回 level 0 后把结果标成 batch consistency PASS。

## 2. 需求裁决与冲突处理

### 2.1 资料采用顺序

| 优先级 | 资料 | 决定内容 |
| --- | --- | --- |
| 1 | `docs/SMLA batch一致性.md` | token 粒度目标、模式 1-4、level 3 与固定规约语义 |
| 2 | 两算子 README、golden、pytest process、metadata/opapi/tiling/kernel 源码、PR8737 | 参数、mask 数学语义、metadata 布局和当前/目标实现事实 |
| 3 | STC 001-027 Excel | 验收 case 草案；错误数据需修订，不能覆盖主需求 |
| 4 | `docs/batch一致性测试框架.md`、会议纪要 | 开发方法与历史理解 |
| 5 | metadata 解析等补充文档 | 旧布局与设计背景 |
| 6 | `data/`外部aclnnfuzz方案 | 关系声明、预检和覆盖方法参考；不决定算子语义 |

冲突时必须在 `SMLA_MQSMLA需求冲突与对齐清单.md` 登记。无法通过实际输入证明数学等价的派生 case 返回
`INVALID_TRANSFORM`，不能包装成数值 `FAIL`。

### 2.2 已裁决的关键冲突

- 模式 2 是测试侧构造独立算子调用，不是 kernel 内部分组；
- 模式 3 是完整连续 token 分区，如 `16K -> 4K+12K`；
- 模式 4 采用主需求的共同前缀，如 `300+100` 对 `300+200`；
- `S2 / 32 == 128` 是精确等式，即 `S2=4096`；
- `S2=5120` 可被 32 整除，STC 014 的文字错误不进入实现；
- STC 026 的 B 与 `cu_seqlens_q` 数量冲突，修订前不自动生成；
- 当前 metadata 布局按 FA `36x9`、FD `72x8`、reserved 124 解析；旧 `36x8`
  作为独立 legacy version；
- FD slot 7 未在当前 header 中命名，只能报告 unknown，不能猜字段含义。

## 3. 多模型联合审议

### 3.1 审议方式

已通过 CANNBot 对同一组需求、STC、代码与计划分别进行只读审议：

- `cannbot/glm-5.2`；
- `cannbot/deepseek-v4-pro`；
- `cannbot/qwen3.7-max`。

模型不直接修改仓库。主审把它们作为遗漏检查与反向审查来源，再用源码、PR 和可重复测试
裁决。

### 3.2 有效共识

- 两算子必须有 operator-owned pytest tree 和 executor；
- MQ 必须调用 metadata + 主算子并捕获同一 metadata；
- 一致性入口必须保留原 PT/golden/process/compare 流程；
- 模式 2 应逐 subcase 与 baseline origin 直比；
- 模式 3/4 必须考虑位置敏感 mask，不能只切/扩 Q；
- 白盒分析必须带 B/N2/G/Q/KV/mask 上下文；
- `UNSUPPORTED`、`INVALID_TRANSFORM`、`NOT_APPLICABLE` 与 `FAIL` 必须分开。

### 3.3 已纠正的模型错误

1. 外层 `DeterministicLevelGuard` 已包住 executor；executor 的 `call_npu` 内部按顺序
   调 metadata 和主算子，因此两段都处于 level 3，不存在“只包第二段”的问题。
2. `S2 / 32 == 128` 不是 Python/C++ 整数除法后的 `[4096,4127]` 区间，而是需求中的
   数学等式 `S2=4096`。
3. FD `workspace_num` 应与该 token 的规约块总数核对，不是简单等于参与 FA 的 core 数。
4. DeepSeek/Qwen 认为 oracle 未检查 mask window，但代码的 `_windows()` 和实际KV slice检查
   已存在；有效建议是把这项证据写入报告，而不是重复实现。
5. prepend 并非一概数学错误：Q位置同步右移时 threshold 可保持；它与当前主需求冲突在于
   “共同前缀”定义，而不是 mask 必然变化。
6. GLM担心 visible S2 不符合PR；PR head实际正是 `actOriS2Size+actCmpS2Size`。真正冲突是
   C++整数floor与需求/analyzer的ceil，现已双轨报告。

### 3.4 2026-07-21 检视意见三联合审议

本轮三模型围绕`requs/data`外部方案再次独立审议：GLM负责架构取舍，DeepSeek负责脚本安全与
正确性反审，Qwen负责需求映射和覆盖维度。共同有效意见是：

- 不导入`tensor_input_seed_info`随机切片生成器，继续从同一baseline PT派生完整等价输入；
- 将外部`output_seed`“关系标签”思想升级为versioned TokenOrigin relation manifest；
- 增加origin边界/重复/scope/oracle/digest预检和min/mid/max来源位置诊断；
- 覆盖清单区分CPU输入关系与真实kernel metadata，不能因离线cell存在就标NPU PASS；
- manifest绑定去除临时来源路径后的输入内容SHA256，来源路径另行保留。

主审纠正了模型的三处误读：四模式都是baseline/derived多次完整调用，不能直接映射成外部
level0/1/2；`output_seed`不是输出数值公式；同位宽unsigned读取可用于raw-bit equality，外部
脚本的真实问题是`eval`、6元素输入被`len==5`跳过、边界/schema/分组键不足。完整审计见
`data/README.md`。

新增意见3-6的第二轮复审进一步确认：GLM建议suite按case解析，DeepSeek强调结构化结果与异常
artifact，Qwen强调三类场景边界和metadata注入契约。主审未采纳“CPU FakeExecutor会形成NPU
PASS”和“relation只校验外壳”两项误报；最终落地为suite config/result artifact/scenario gap/
CPU-only stub planner四条独立控制线。

## 4. pytest_v2 总体架构

```text
原参数/Excel
    -> 原 CPU golden + PT save
    -> 原 PT load
    -> suite config按PT精确匹配 + shape-aware预检
    -> CaseAdapter + schema + mode transform
    -> actual-input semantic oracle
    -> relation manifest（TokenOrigin spans/content digest/config precheck/coverage）
    -> DeterministicLevelGuard(level=3)
    -> operator-owned metadata op
    -> operator-owned main op（消费同一个 metadata）
    -> NPU baseline/derived origin 对齐 + bitwise MD5
    -> 原 CPU golden + check_result
    -> MetadataLayout + MetadataCoverageAnalyzer
    -> baseline/derived InputSummary（来源、shape/dtype/digest/sample）
    -> 每个(operator,PT,mode)原子写result JSON
    -> JSON-only汇总 + artifact完整性/pytest进程状态核对
    -> PASS / FAIL / UNSUPPORTED / INVALID_TRANSFORM / NOT_APPLICABLE / CONFIG_ERROR
    -> finally 恢复进入测试前的 deterministic level
```

两棵 v2 在 `batch/consistency/*.py` 上保持同源镜像，但 executor、prepare wrapper 和
process 各自留在算子目录。这样外部检视者可从本算子的 `test_*_batch_consistency.py`
顺着原 process 看完整调用，不需要跳到另一算子目录。

CPU-only分支在 mode transform 后不进入 level guard/NPU，而是按`TokenOrigin`物化比较范围的
expected output，保存并重载derived PT。模式4额外token明确记为`unmapped_tokens`，不冒充完整
CPU golden。每个baseline和算子汇总目录同时保存relation manifest；其中PASS只代表输入关系
预检，不代表NPU或metadata核覆盖。

## 5. 模式与正式配置

### 5.1 命名接口

规范名称：

```text
reorder, split, token-split, shape-change
```

| profile | modes | 用途 |
| --- | --- | --- |
| `blackbox-basic` | `reorder,split` | 最小稳定集 |
| `blackbox-supported` | 四种模式 | 当前测试侧完整黑盒集 |
| `blackbox-all` | 四种模式 | 与 supported 暂时等价，预留后续模式扩展 |

推荐命令：

```bash
bash test_run.sh consistency --modes reorder,split
bash test_run.sh consistency --profile blackbox-supported
```

`-M 1,2` 只保留为 SMLA 旧入口下游迁移兼容，会映射到命名 mode 并发 deprecated warning。
正式配置不再依赖含义不透明的数字，也不需要等待真实 NPU 打通后才迁移。

#### 5.1.1 versioned suite正式入口

推荐把多参数写入`cases/SMLA_MQSMLA_batch一致性执行配置.json`，并运行：

```bash
bash test_run.sh consistency \
  --pt-dir /path/to/baseline \
  --config-file ../../../../requs/cases/SMLA_MQSMLA_batch一致性执行配置.json \
  --report-dir ./result
```

一个suite文档可覆盖一批PT，但每个PT独立匹配和解析，不共享`3+5`、`4K+12K`等shape相关
常数。selector支持operator、case name、PT basename和输入内容SHA256；默认未匹配、重复匹配、
未知字段、越界groups和token总数不符均在NPU前报`CONFIG_ERROR`。配置文件和旧详细CLI互斥。

### 5.2 模式 1：reorder

- 同一组 batch 使用非恒等 order；
- BSND 按 batch dim 重排；
- TND 切段、重排、拼接并重算 prefix sum；
- PA_BBND 保持物理 cache，只重排 block-table 行；
- Q/KV、有效长度、sparse index、top-k、residual 和 metadata/params 镜像字段同步；
- seed 固定 permutation，输出按 `TokenOrigin(source_batch,source_token)` 还原后比较。

### 5.3 模式 2：split

主需求示例：

```text
baseline:  B=8, [0,1,2,3,4,5,6,7]
derived A: B=3, [0,1,2]
derived B: B=5, [3,4,5,6,7]
```

正式验收显式配置：

```bash
--split-groups '0,1,2;3,4,5,6,7'
```

每个 derived subcase 直接选择 baseline 对应 origin 做 MD5 和数值精度比较。groups 完整覆盖
时可额外 merge 诊断，但 merge 不是主 oracle。默认 B=8 也是 3+5；其他 B 使用连续确定性
二分。只有显式 `--randomize` 才让 seed 参与分组。

### 5.4 模式 3：token-split

主需求示例：

```text
B=1, Token=16K
  -> call A: [0,4K)
  -> call B: [4K,16K)
```

至少两个正 split size，且 `sum(split_sizes)==old_Q_length`；所有区间连续、不重叠、完整
覆盖。每段独立调用并直接和 baseline 对应 token 比较。

对位置敏感 mask，原区间 `[start,end)` 使用：

```text
new_ori_S2 = old_ori_S2 - old_Q_length + end
```

cmp 长度与 residual 按 `cmp_ratio` 同步，确保派生 token 的可见 KV 窗口等于原 token。

### 5.5 模式 4：shape-change

```text
baseline: 300 common + 100 extra
derived:  300 common + 200 extra
compare:  前 300 token
```

对位置敏感 mask 使用：

```text
new_ori_S2 = old_ori_S2 + new_Q_length - old_Q_length
```

该式保持 `ori_S2-Q_length` 不变。新增逻辑 KV 位于共同 token 的未来；cmp 关系和 PA block
容量同步检查。无法构造安全上下文时返回 `INVALID_TRANSFORM`。

### 5.6 actual-input 语义 oracle

模式 3/4 在任何 NPU 调用前验证：

- mapped Q tensor 完全相同；
- ori/cmp mask window 与其中实际可见 KV slice 相同；
- ori/cmp sparse indices 与 top-k length 相同；
- sinks 相同；
- layout、有效长度、block table 可寻址范围合法。

oracle 不重新调用随机 `gen_data(params)`。通过后，derived NPU 才允许复用 baseline PT 中
对应 token 的原 `cpu_output` 作为数值 golden；任何无法证明的 case 都拒绝执行。

## 6. 阶段二：metadata 白盒分析

### 6.1 布局与上下文

`MetadataLayout` 严格要求 `torch.int32` 和恰好 1024 个元素；current 布局为 FA 36x9、
FD 72x8、tail 124，legacy FA 36x8 单独版本化。负 int32 按相同 uint32 bit pattern 解释，
FD slot 7 和 reserved 非零均显式报告。

`CoverageContext.from_input_data(case)` 提供 B、N1、N2、G、每 batch Q/KV 长度、cmp residual/
ratio、mask/window、sparse top-k 和 layout。analyzer 随后：

1. `bn2 -> (batch,n2)`；
2. 按 PR8737 当前 `mBaseSize=G` 将 `m_index` 解释为 query token；
3. 计算每个 token 的可见 `ori+cmp` S2；
4. 同时计算PR8737 head的`align_up(floor(actual_S2/32),128)`与需求
   `align_up(ceil(actual_S2/32),128)`，结构展开暂按PR并报告冲突/zero edge；
5. 展开 FA `[start,end)` 到 token/S2 block；
6. 合并 split-G 镜像的重复 FA range；
7. 关联 FD workspace 与 `m_start/m_num`；
8. 检查 `workspace_num`、G 行覆盖、range 连续性、unknown/reserved；
9. 输出 token/core 视角、Level-1 四类、Level-2 目标与 missing coverage。

### 6.2 PR8737 与本地版本差异

| 提交 | 只读证据 |
| --- | --- |
| `e14f1559` | MQ metadata 读取 `ACL_OPT_DETERMINISTIC` 并传递 batch-consistency 状态 |
| `8b4ea3a4` | level-3 路径开启 FD 并使用固定规约 tile |
| `f2424c8c` | 增加 FD |
| `f7a7856f`、`31a3b6d3` | 后续 FD 修复；公开 MR head 为 `31a3b6d3` |

当前本地快照未包含这些改动并仍表现为 `supportFd=false`。已安装 custom 包来源可能不同，
但当前 runtime 先于 metadata 执行拒绝 level 3。此前用于接口探索的 level-0 MQ probe 已从
正式目录删除；历史观察只保留为开发记录，不可证明 level-3 固定调度、FD 或主算子一致性。

## 7. 状态模型

| 状态 | 定义 | 处理 |
| --- | --- | --- |
| `PASS` | 变换有效、环境支持、NPU MD5、CPU 精度和 metadata integrity 全通过 | 成功 |
| `FAIL` | 有效 case 执行异常或任一真实门禁失败 | 失败 |
| `UNSUPPORTED` | runtime/device/operator/graph 明确不支持必要能力 | 单列，不能称通过 |
| `INVALID_TRANSFORM` | 当前 PT 无法保持目标 token 数学输入等价 | 单列并记录原因 |
| `NOT_APPLICABLE` | PT 对所选模式无可执行 item | 单列 |
| `CONFIG_ERROR` | mode/profile/groups/sizes schema 非法 | 入口失败 |

JSON artifact现已按`(operator,PT,mode)`落盘。`PASS`还必须证明执行报告通过，且所有记录的
deterministic level均为requested=3、observed=3。聚合器核对预期artifact总数、唯一PT数、唯一
PT/mode矩阵、损坏/重复/异算子artifact和pytest子进程失败数，并输出
`PASS/PARTIAL/UNSUPPORTED/INCOMPLETE/FAIL/NO_RESULTS`；文本日志只用于诊断。

## 8. 阶段与里程碑

| 里程碑 | 内容 | 当前状态 |
| --- | --- | --- |
| P1.0 | v1 模式 1/2、layout 原语、origin compare、level guard | 完成，`b680bfbdd` |
| P1.R1 | v1 检视修复、命名接口、metadata 初步分析 | 完成，`183beb730` |
| P2.1 | SMLA `pytest_v2` + 原流程整合 | 完成 |
| P2.2 | MQSMLA `pytest_v2` + operator-owned executor | 完成 |
| P2.3 | 模式 2 规范 `B=8 -> 3+5`、显式 groups | 完成 |
| P2.4 | 模式 3 完整 token partition + mask 对齐 | 完成 |
| P2.5 | 模式 4 共同前缀 shape change + mask 对齐 | 完成 |
| P2.6 | actual-input 语义 oracle + 原 CPU golden 复用 | 完成 |
| P2.7 | current metadata parser + contextual Level-1/2 analyzer | 完成 |
| P2.8 | PR8737源码与版本边界记录 | 完成；临时probe已删除，真实level3阻塞 |
| P2.9 | 教学、两段调用、文件职责和逐STC冲突台账 | 完成（2026-07-20扩写） |
| P2.10 | 双算子Excel->golden->PT离线预检 | 完成；各1 baseline + 6 derived round-trip |
| P2.11 | 白盒原始输入摘要与floor/ceil公式审计 | 完成 |
| P2.12 | pytest_v2目录整理与双树镜像门禁 | 完成 |
| P2.13 | 外部aclnnfuzz方案评估、relation manifest v1与覆盖清单 | 完成（2026-07-21） |
| P2.14 | per-PT suite config v1与shape-aware预检 | 完成；正式JSON示例已验证双算子baseline |
| P2.15 | 逐模式result artifact与JSON-only汇总 | 完成；双算子在线各4条UNSUPPORTED且无缺失 |
| P2.16 | metadata单核->多核候选planner和默认关闭hook | 安全骨架完成；真实注入待P3.1及FD契约 |
| P3.1 | 支持 level 3 的 runtime/custom/operator 版本对齐 | 阻塞 |
| P3.2 | SMLA eager 模式 1-4 真实 NPU | 待 P3.1 |
| P3.3 | MQSMLA eager 模式 1-4 真实 NPU | 待 P3.1 |
| P3.4 | STC 001-027 versioned manifest 与修订数据 | relation schema/自动派生manifest完成；STC source entries待owner修订 |
| P3.5 | 覆盖矩阵 | CPU relation inventory完成；真实metadata Level-1/2待P3.1/P3.4 |
| P4.1 | SMLA/MQSMLA graph executor + exact capture | 待实现 |
| P4.2 | ACLNN STC 021 | 待实现 |
| P4.3 | N=3/N=125、TND、PA_BBND nightly | 待 P3/P4 |

## 9. 当前测试矩阵

| 层级 | 内容 | 当前结果 |
| --- | --- | --- |
| SMLA v2 CPU | modes 1-4、配置、artifact、planner、metadata、relation/mirror | `39 passed`（2026-07-21实测） |
| MQSMLA v2 CPU | 与 SMLA 同源门禁 | `39 passed`（2026-07-21实测） |
| SMLA离线prepare | Excel/原golden/baseline+derived PT/relation manifest | 1 baseline + 6 derived，round-trip与6条relation预检通过 |
| MQSMLA离线prepare | Excel/原量化golden/baseline+derived PT/relation manifest | 1 baseline + 6 derived，round-trip与6条relation预检通过 |
| CPU relation coverage gaps | mode/layout/mask/S2/source landmarks | 四mode齐；SMLA缺TND/PA/位置mask，MQ缺TND/BSND KV/no-mask，均单列missing |
| SMLA opt-in 入口默认关闭 | 四模式不触发 NPU | `4 skipped` |
| MQSMLA opt-in 入口默认关闭 | 四模式不触发 NPU | `4 skipped` |
| shell | 两份 `test_run.sh` | `bash -n` 通过 |
| source lint | 两份一致性新增文件 | ruff 通过 |
| Python compile | 两棵 v2 | compileall 通过 |
| 原 single/batch/graph collect | 两棵 v2 | SMLA 15 项、MQSMLA 3 项 |
| 双算子 level 3 | metadata + 主算子 | runtime 拒绝，`UNSUPPORTED` |
| 双算子在线结构化结果 | 各1 baseline x 4 modes | 各4条`UNSUPPORTED`，artifact无缺失，未误报PASS |

## 10. 剩余阶段二与第三阶段实施顺序

### A. 不依赖NPU，优先关闭数据和规格问题

1. 由需求/算子/STC owner逐项处理`SMLA_MQSMLA需求冲突与对齐清单.md`；
2. 先确认PR floor/需求ceil与`S2<32`zero edge，再冻结白盒预期；
3. 修STC 014/015、018-020、024-027，并补齐001-020公共字段；
4. 基于已冻结的relation manifest v1补STC source entries，把001-027变为机器可读关系；
5. 给MQ CPU golden补LSE后，再开启`return_softmax_lse=True`一致性验收；
6. 为TND/PA/position mask各补一个小shape离线workbook case。
7. 定义`independent-cross-case`关系source entry和`single-call-intra-case`位置关系schema，先补
   actual-input等价证明和最小CPU用例，再接NPU；
8. 用metadata stub planner筛选自然单核且S2 block>=2的候选，保存计划；在FD契约冻结前不生成
   synthetic metadata。

### B. 解除真实执行阻断（环境可用时）

1. 确认支持 deterministic level 3 的 torch_npu/CANN/SoC 版本矩阵；
2. 对齐本地 operator source、PR8737 head 与已安装 custom 包构建版本；
3. 验证第一条 SMLA、MQSMLA metadata NPU op 均可在 level 3 执行；
4. 固化一个最小 SMLA 和一个最小 MQSMLA level-3 PT smoke；
5. 记录软件版本、SoC、custom 包 hash、metadata digest。

### C. 跑通 eager 四模式

1. SMLA：BSND no-mask 模式 1-4；
2. SMLA：RightDownCausal/Band，验证模式 3/4 上下文公式；
3. SMLA：TND、PA_BBND、sparse/top-k、可选 LSE；
4. MQSMLA：同样顺序，额外验证量化 KV 与 `quant_mode/rope_head_dim`；
5. 每个 subcase 保存 baseline/derived MD5、CPU 精度、metadata digest、失败 origin。

### D. 冻结 STC 数据契约

已选择versioned JSON relation manifest，不向PT增加`output_seed`式平行字段。当前自动派生artifact
结构如下；下一步在其上增加STC source entry和状态，而不是再选择另一套schema：

```json
{
  "schema_version": 1,
  "kind": "smla_mqsmla_batch_consistency_relation",
  "operator": "smla",
  "scenario_coverage": {
    "implemented": ["same-source-cross-call"],
    "supplemental_gaps": ["independent-cross-case", "single-call-intra-case"]
  },
  "baselines": [{
    "case_id": "smla:batch_consistency_001:<content-sha-prefix>",
    "input_content_fingerprint_sha256": "..."
  }],
  "relations": [{
    "mode": "split",
    "scenario": "same-source-cross-call",
    "baseline_case_id": "smla:batch_consistency_001:<content-sha-prefix>",
    "comparison_identity": {
      "type": "TokenOrigin",
      "origin_spans": [{
        "target_start": 0, "target_end": 3,
        "source_batch": 0, "source_token_start": 0, "source_token_end": 3
      }]
    }
  }]
}
```

在线配置优先级改为二选一：versioned `--config-file`，或deprecated CLI/environment；二者不得
混用。suite内部是`defaults + 一个精确case override`，case的`modes/profile`互相替换。自动派生
manifest已经可生成和校验；仍需先修正STC 014/026，并确认001-027各自引用哪个baseline PT。
source manifest不得把`BLOCKED_BY_SPEC/INVALID_DATA`伪装成`CONFIRMED`。

### E. 扩展执行面

1. 实现 SMLA/MQSMLA graph consistency executor；
2. 确认 level 3 在 compile 还是 run 时读取，并记录 compile cache；
3. 捕获 graph 主算子实际消费的 metadata；
4. 实现 ACLNN STC 021，使用 `aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC,3)`；
5. 最后接入 N=3/N=125、随机组合、TND/PA nightly。

## 11. 阻断项

| 编号 | 问题 | 需要确认方 | 阻断范围 |
| --- | --- | --- | --- |
| RT-1 | 支持 level 3 的 torch_npu/CANN/SoC 组合 | runtime/CANN | 全部真实 NPU 验收 |
| OP-1 | SMLA/MQSMLA level-3 tiling/main-op 合入版本 | 主算子开发 | 双主算子 PASS |
| PKG-1 | 本地源码、PR8737、已安装 custom 包 hash 对齐 | 构建/环境维护 | MQ metadata/FD 证据 |
| META-1 | FD slot 7 最终定义与部署 layout version | metadata/kernel | 最终 schema 冻结 |
| FORMULA-1 | PR floor、需求ceil和S2小于32的最终公式 | tiling/需求负责人 | 白盒结构预期 |
| DATA-1 | STC逐项错误与缺字段修订 | STC负责人 | STC manifest |
| MODE4-1 | 共同前缀+KV增长是否为正式产品口径 | 需求/mask负责人 | 模式4最终冻结 |
| CFG-1 | STC 001-027到relation manifest v1的source entry映射 | 测试/STC维护者 | 001-027 自动化 |
| GRAPH-1 | level 3 读取时机与 compile cache 约束 | torchair | graph 验收 |
| SCENARIO-1 | 独立跨case和单次调用内等价位置的正式输入/关系 | 用例与需求owner | 两个补充黑盒场景 |
| STUB-1 | synthetic metadata生成、结构校验和独立coverage口径 | metadata/kernel/test | 单核到多核打桩激活 |

MQ pytest修改权限和上下文analyzer已不再是阻断项；模式4已有安全暂定实现，但产品口径仍需
owner确认。

## 12. 最终验收门禁

1. SMLA 与 MQSMLA 各有真实 level-3 eager smoke PASS；
2. 模式 1/2/3/4 每个派生 case 的对应 origin MD5 与 CPU 精度均通过；
3. 模式 3/4 每个 NPU case 先通过 actual-input 语义 oracle；
4. BSND、TND、PA_BBND 与 sparse/top-k/position-sensitive mask 有覆盖；
5. SMLA 可选 LSE 与 output 一起进入 bitwise 检查；
6. metadata 来自主算子实际消费对象，且 parser layout 与部署版本一致；
7. Level-1 四类和 Level-2 指定核视角均能自动报告；
8. STC 001-027 每项有机器可读状态和 baseline/derived 关系；
9. eager、graph、ACLNN 各自独立验收；
10. 原 single/save/load/load_graph 无行为回归；
11. CI 至少有一套真正支持 level 3 的环境；
12. artifact 记录版本、case、mode、seed/profile、metadata/output digest 和失败 origin。
13. 白盒artifact携带baseline/derived原始输入摘要及floor/ceil公式审计；
14. MQ `return_softmax_lse=True` 只有在process保留LSE且CPU golden生成LSE后才可列入通过项。
15. relation manifest通过baseline内容指纹、origin span digest、边界/重复/scope/oracle预检；
    其CPU coverage与真实metadata coverage分栏报告。
16. 每个预期`(operator,PT,mode)`都有唯一结果JSON；缺失/重复/pytest进程失败不能汇总为PASS。
17. `independent-cross-case`与`single-call-intra-case`分别有可信输入关系和独立coverage，不借
    `same-source-cross-call`模式改名冒充。
18. synthetic metadata必须单独标识，不能计入自然metadata coverage；注入前后结构与workspace
    契约均有门禁。

## 13. CANNBot 交互方式

启动独立只读审议：

```bash
cannbot run -m cannbot/glm-5.2 \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '阅读 requs/requs.md、开发计划、主需求和当前代码；只读审议，不修改文件'

cannbot run -m cannbot/deepseek-v4-pro \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '反向检查模式3/4数学语义、metadata和验收漏洞；只读，不修改文件'

cannbot run -m cannbot/qwen3.7-max \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '核对STC、主需求、两算子pytest_v2的差异；只读，不修改文件'
```

继续最近 session：

```bash
cannbot run -c \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '根据新测试结果继续复审；仍然不要修改文件'
```

或使用首轮返回的 session id：

```bash
cannbot run -s <session-id> \
  --dir /mnt/workspace/gitCode/cann/ops-transformer_0716 \
  '继续任务'
```

若 CANNBot 数据库出现 migration 错误，不删除或直接改写用户数据库；优先升级/迁移工具。
若必须临时隔离 `XDG_DATA_HOME`，目录与认证文件权限分别保持 0700/0600，完成后清理，
认证信息不得进入仓库、日志或提交。

2026-07-21实测默认数据库因缺少`replacement_seq`列启动失败。本轮没有迁移/删除原数据库，
而是为三个模型分别使用0700临时`XDG_DATA_HOME`，只复制0600认证文件；三个session均只读完成。
