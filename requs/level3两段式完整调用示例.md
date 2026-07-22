# deterministic level 3：用例如何调用 batch 一致性

> 版本：2026-07-23 检视意见四实机复验版
> 主线：只讲 `tests/pytest_v2`；旧 pytest/v1 不再承载新功能。

## 1. 先直接回答：算子参数有没有新增

**没有为 batch 一致性给 metadata API 或主算子新增 `mode=1/2/3/4` 参数。**

用例仍然提供原有的 Q/KV、length、mask、block table、sparse index 等字段。四种模式改变的是这些
已有字段的 shape/value，以及一次 pytest 要发起几次调用。唯一的新开关是进程级：

```python
torch_npu.npu.set_deterministic_level(3)
```

level 3 不是 Excel 列，也不是某个 operator argument。runtime/tiling 在进程状态中读取它，选择固定
规约块、FA/FD 分核和固定规约路径。

| 层 | 接收什么 | 做什么 | 不做什么 |
| --- | --- | --- | --- |
| 用例/Excel/PT | 原算子字段；可另给 pytest 模式配置 | 生成 baseline 的真实输入和 CPU golden | 不指定某 token 分到 NPU 第几核 |
| pytest_v2 | `--modes`、split groups、token ranges、common prefix | 构造 baseline/derived 多次调用，维护 `TokenOrigin`，比较结果 | 不实现 kernel 内 FA/FD 规约 |
| runtime + 算子 | level=3 和每次调用的原参数 | metadata tiling、规约块、分核、workspace、kernel combine | 不知道“这是模式2的第几组” |

## 2. 参数究竟怎样到 NPU

```text
Excel row
  -> 原 utils 解析/default fill
  -> 原 golden 生成真实 tensor + cpu_output
  -> 原 PT save
  -> pytest_v2 load PT
  -> mode transform 改已有 tensor/length/table/index
  -> capability probe: set 3 -> readback 3 -> 普通NPU dispatch -> restore
  -> set_deterministic_level(3)                  # 进程状态
  -> metadata_op(本次调用的 shape/length/mask)   # 返回本次 metadata
  -> main_op(同一批 tensor + 同一个 metadata)    # NPU执行FA/FD/规约
  -> origin 对齐 + bitwise MD5 + CPU precision + 白盒报告
  -> restore old deterministic level
```

所以“模式 2 把 B=8 拆成 B=3 与 B=5”表示 pytest 发起三个完整调用：一个 B=8 baseline，
一个 B=3 derived，一个 B=5 derived。每次调用内部，NPU 又根据各自 metadata 把 token/S2 block
分到核上。这两个“拆分”发生在不同层，不能混为一谈。

模式名本身不会传入 NPU。NPU 能看到的是：

- 变化后的 `q/ori_kv/cmp_kv`；
- `cu_seqlens_*`、`seqused_*`、`cmp_residual_kv`；
- block table、sparse indices、top-k length、sinks；
- B/S1/S2/T1/N1/N2/D、mask/window、layout；
- 进程级 deterministic level 3。

## 3. 用例侧到底准备一条还是多条 Excel case

正式推荐是：**Excel/原 PT 只准备 baseline，派生 case 由 pytest_v2 生成。** 这样 Q/KV 真正来自
同一份 baseline，不会因为 Excel generator 再次随机采样而破坏等价关系。

| 模式 | baseline 最低条件 | pytest 生成的 derived | 同步变化字段 | 通常不变字段 |
| --- | --- | --- | --- | --- |
| `reorder` | B>=2；每个 batch 输入完整 | 同 B 的非恒等 permutation | Q/KV 或 table 行、length、index/top-k；TND prefix sum重算 | dtype、N1/N2/D、mask参数、tensor内容本身 |
| `split` | B>=2；需求示例 B=8 | B=3 与 B=5 等独立调用 | 选择对应 batch 的所有 per-batch 字段；TND重新拼接 | 每个被选 batch 内的 Q/KV/mask语义 |
| `token-split` | 指定 batch 有>=2个有效 Q token | 例如16K拆为4K、12K两次完整连续分区 | Q及Q对齐index、Q length；位置mask下同步effective KV/cmp | 每个 mapped token 的实际 Q/可见KV/window/sinks |
| `shape-change` | 有共同前缀；如400=300+100 | 如500=300+200，只比较前300 | Q shape/effective length；位置mask下KV/cmp length；PA需容量 | 共同前缀Q、其可见KV/window/index/sinks |

### 3.1 模式 1：reorder

用例不需要手写第二条 reorder Excel。baseline 至少 B=2，然后：

```bash
bash test_run.sh consistency -P ./data --modes reorder --seed 20260716
```

seed 只固定 permutation。输出先按 `TokenOrigin(source_batch, source_token)` 还原，再做 MD5；
不能直接拿重排后物理顺序的整 tensor 求 MD5。

### 3.2 模式 2：split

需求 B=8 -> B=3+B=5：

```bash
bash test_run.sh consistency -P ./data --modes split \
  --split-groups '0,1,2;3,4,5,6,7'
```

两组分别调用 metadata + 主算子，并各自直接与 B=8 baseline 的对应 origin 比较。merge 只可作为
完整分区诊断，不是主 oracle。随机分组只有显式 `--randomize` 才启用。

### 3.3 模式 3：token-split

baseline 必须真的有 16K 有效 Q，而不只是物理 padding 到16K：

```bash
bash test_run.sh consistency -P ./data --modes token-split \
  --mode3-batch 0 --token-split-sizes '4096,12288'
```

要求每段>0、连续、不重叠且总和恰好等于 old Q length。对源区间 `[start,end)`，位置 mask 使用：

```text
new_ori_S2 = old_ori_S2 - old_Q_length + end
```

这不是为了操纵 NPU 分核，而是让被比较 token 在这次较短调用中的 RightDown/Band window 与
baseline 相同。随后 NPU 自己依据新 shape 重新分核。

### 3.4 模式 4：shape-change

baseline 需要包含“共同 token + baseline extra”。需求例子是400=300+100：

```bash
bash test_run.sh consistency -P ./data --modes shape-change \
  --mode3-batch 0 --common-tokens 300 --derived-extra 200
```

derived 变为500=300+200，只比较共同前300。位置 mask 使用：

```text
new_ori_S2 = old_ori_S2 + new_Q_length - old_Q_length
```

这保持 `ori_S2-Q_length` 不变。若 PA block table 没有多出的物理容量，框架返回
`INVALID_TRANSFORM`；它不会伪造 block id。模式4的 prepend/append 冲突和完整 mask 数学见
`SMLA_MQSMLA需求冲突与对齐清单.md`。

## 4. metadata 与主算子的“两段式”规则

每一个 baseline/derived 调用都必须遵守：

```python
old_level = torch_npu.npu._get_deterministic_level()
try:
    torch_npu.npu.set_deterministic_level(3)
    metadata = metadata_op(**metadata_args_for_this_call)
    output, softmax_lse = main_op(
        **tensor_args_for_this_call,
        metadata=metadata,  # 必须是本次第一段返回的同一个 tensor
    )
    torch.npu.synchronize()
finally:
    torch_npu.npu.set_deterministic_level(old_level)
```

关键点：

1. level 3 必须在 metadata 前已生效，并覆盖主算子和同步点；
2. derived shape 改变后必须重新调用 metadata op，不能复用 baseline metadata；
3. metadata 参数和主算子参数必须来自同一个 transformed case；
4. getter/setter/首次 NPU 操作任何一步拒绝 level3，都归 `UNSUPPORTED`；
5. 禁止自动退回 level0 后标 batch consistency PASS。

正式批跑会先在独立进程执行一次 capability probe。setter返回且getter读回3只证明Python状态已
暂存，只有一条普通NPU dispatch也成功，才允许加载算子executor并进入上述metadata+主算子链。
探针失败时每个模式仍先做纯CPU transform/semantic oracle，避免掩盖`INVALID_TRANSFORM`；随后artifact
绑定同一份`runtime_capability.json`，不加载executor、不重复下发已知必失败的算子。
可用`--probe-attempts 10`做异常恢复短压测；这仍不等于算子smoke或精度压测。

实际代码入口：

| 算子 | executor | 原两段式 process |
| --- | --- | --- |
| SMLA | `attention/sparse_flash_mla/tests/pytest_v2/batch/sparse_flash_mla_consistency.py` | `batch/sparse_flash_mla_process.py` |
| MQSMLA | `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/batch/mixed_quant_sparse_flash_mla_consistency.py` | `batch/mixed_quant_sparse_flash_mla_process.py` |

公共 guard/模式/报告位于各算子自己的 `batch/consistency/` 镜像目录。SMLA 与 MQSMLA 没有跨目录
import，单测会检查两份核心镜像是否漂移。

## 5. 为什么原 golden/PT schema 没增加 mode 字段

CPU golden回答的是“一次具体算子输入的数学预期”；batch consistency回答的是“同一数学 token
在另一种调用 shape/调度下是否 bitwise 一致”。因此：

- baseline 仍由原 golden生成并随原 schema 存 PT；
- pytest 用 `TokenOrigin` 从 baseline CPU output物化 mapped token 的 derived expected output；
- 模式4额外 token 不属于比较集合，不能拿零填充值冒充它们的完整 golden；
- 离线预检报告会明确写 `origin_mapped_cpu_golden`、physical/compared length和unmapped token数；
- 如果将来需求要求验证派生输入的所有额外 token，必须给 golden增加“对指定现成 tensor 重新计算”
  的正式接口，那是另一项功能，不能在本轮静默假装已经具备。

离线验收表额外放了 `consistency_modes/split_groups/...` 列，只供
`consistency_prepare` 读取；原 golden parser忽略这些辅助列，正式 operator API/PT schema不变。

2026-07-21新增的relation manifest也是PT旁路JSON artifact，不向operator API或原PT schema增加
mode/output-seed字段。它把自动派生后的真实关系记录为`baseline_case_id + TokenOrigin spans`，
并绑定baseline/derived输入内容digest，供预检、复现和未来追加NPU结果。

## 6. 不用 NPU先验证用例链

SMLA：

```bash
cd attention/sparse_flash_mla/tests/pytest_v2
bash test_run.sh consistency_prepare \
  --excel ../../../../requs/cases/SMLA_MQSMLA_batch一致性离线预检.xlsx \
  --sheet SMLA --pt-dir /tmp/smla_pt --report-dir /tmp/smla_report
```

MQSMLA：

```bash
cd attention/mixed_quant_sparse_flash_mla/tests/pytest_v2
bash test_run.sh consistency_prepare \
  --excel ../../../../requs/cases/SMLA_MQSMLA_batch一致性离线预检.xlsx \
  --sheet MQSMLA --pt-dir /tmp/mqsmla_pt --report-dir /tmp/mqsmla_report
```

它验证 Excel解析、原 CPU golden、baseline PT save/reload、模式1-4 transform、actual-input oracle、
origin-mapped expected output、derived PT save/reload、schema与digest；同时生成version 1 relation
manifest，检查origin边界/重复、comparison scope、span digest和输入关系覆盖。它不调用 NPU，
因此报告明确写“CPU-only；不能声称 batch consistency PASS”。

## 7. 当前环境预期

2026-07-23按用户更新后的环境和custom包复验：torch
`2.10.0+cpu`、torch_npu `2.10.0.post3.dev20260709`、Ascend910_9362、driver 25.5.5。
setter返回成功且getter读回3，但第一条NPU操作拒绝level3：

```text
'torch_npu.npu.set_deterministic_level' currently only supports configuring 0/1/2 !
```

所以当前可以完成第6节的离线预检，真实两段式只能记录 `UNSUPPORTED`。MQSMLA 当前普通 process
还丢弃主算子返回的 LSE，且 MQ CPU golden不生成 LSE；在这两点补齐前，MQ用例应保持
`return_softmax_lse=False`，不能把 LSE列入已验收范围。

连续10次`set 3 -> readback -> NPU dispatch -> restore`结果完全一致：10次均在`npu_smoke`阶段
返回`LEVEL3_ACCEPTED_BUT_NPU_REJECTED`，10次都从0读回3并成功恢复0。因此当前值得保留的是恢复
短测和CPU门禁，不应启动四模式长精度/随机压力测试。

`pr_8`的MQ tiling源码当前以`GetDeterministicLevel()==1`或
`ACL_OPT_DETERMINISTIC==1`开启batch consistency，其pytest也显式设置1；这不能证明需求level3会
由runtime映射成内部1。当前runtime恰好在dispatch阶段拒绝3，已直接否定“本环境已完成映射”的
假设。只有支持3的runtime下，tiling日志/打桩观测到实际context值并命中batch-consistency tiling
key，才能冻结映射契约；禁止用level1结果替代level3验收。

已安装custom OPP全树内容指纹为
`81730ebc09a01b62ea8639ce2f4977d489218817329fcd954356bd88055c7a5e`（776文件，
21194488字节），三个关键二进制与`pr_8`工作树中2026-07-23 01:53构建产物逐字节一致；但包内
`version.info`没有source commit，所以这是强本地关联证据，不是可跨机器验证的commit provenance。

## 8. versioned在线配置和结果（2026-07-21新增）

正式批跑不再建议在命令行堆叠groups/sizes/common等参数，而是使用：

```bash
bash test_run.sh consistency \
  --pt-dir /path/to/baseline_pt \
  --config-file ../../../../requs/cases/SMLA_MQSMLA_batch一致性执行配置.json \
  --report-dir ./result/consistency \
  --probe-attempts 1
```

SMLA入口也接受`-P`作为`--pt-dir`兼容别名。suite文档按每个PT精确匹配和shape校验；
`--config-file`不能与`--modes/--profile/--seed/--split-groups/--token-split-sizes`等混用。

每次执行会生成独立run目录。`runtime_capability.json`记录失败阶段、每次level恢复、软件/SoC版本、
custom OPP全树和关键文件SHA256；逐模式JSON位于`results/`并嵌入该证据，权威汇总是
`consistency_result_summary.json`；文本`pytest.log`只用于诊断。当前环境的预期真实结果是
`UNSUPPORTED`且命令返回2，不是“pytest skip所以PASS”。只有两段均真实运行、比较报告通过且
requested/observed deterministic level都为3，artifact才允许写`PASS`。
