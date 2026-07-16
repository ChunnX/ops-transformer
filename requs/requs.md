# 需求一

requs/docs下提供了`batch一致性测试框架.md`、`SMLA支持batch一致性.md`、`SMLA batch一致性.md`、`metadata输出结果解析.md`文档，batch一致性这是SMLA算子的一个特性，`batch一致性测试框架.md`是测试框架适配的开发需求规划，现在我需要在SMLA pytest上去适配这个batch一致性，请chatgpt和glm查阅文档和ops-transformer/attention/sparse_flash_mla/tests/pytest的实现，`batch一致性测试框架.md`的`# 大纲`下图片展示了当前开发流程与重要点，chatgpt按照`batch一致性测试框架.md`一步步去`SMLA_batch一致性测试开发计划.md`规划设计指导glm-5.2完成开发任务（通过`SMLA_batch一致性测试开发计划.md`文档沟通，如何判断对方是否回答完：可以通过在文档中设置时间节点），
入门先优先适配`# 支持的黑盒场景`，一步步走，比如说
ops-transformer/attention/sparse_flash_mla下面其他目录有算子具体实现pytest要跑`batch一致性`的用例，肯定要在当前SMLA pytest基础上加一个开关，去控制`batch一致性`的分支逻辑
有了开关后肯定开始黑盒场景实现，`batch一致性测试框架.md`中也提供了一些思路，白盒就重点对场景规约情况做分析了，在`SMLA batch一致性.md`、`SMLA支持batch一致性.md`、`metadata输出结果解析.md`就对`batch一致性`做了重点介绍，一定要多查阅，`metadata输出结果解析.md`也有FD相关分核结果理解展示

关于实现过程glm-5.2有问题的可以和chatgpt探讨一下，但有一个前提是`batch一致性测试`实现时不要影响到SMLA pytest已有的一些功能

环境支持，当前环境是ascend910_93的，根据ops-transformer_0716/attention/sparse_flash_mla/op_host/sparse_flash_mla_def.cpp可知SMLA是支持ascend910_93，可以尝试构造`batch一致性`的excel用例验证跑下(当时如果`batch一致性`这个特性只在ascend950实现的话，本环境肯定不支持了，告知我一声就行),还有不要在沙箱内用N卡，沙箱外执行，python环境支持请在跑用例前`source /mnt/workspace/gitCode/env.sh`

/mnt/workspace/gitCode/cann/ops-transformer_0716/attention/sparse_flash_mla是SMLA的最新代码，可以查阅一下，我只允许你们动/mnt/workspace/gitCode/cann/ops-transformer_0716/attention/sparse_flash_mla/tests/pytest目录，注意代码规范

# 第一阶段开发记录（2026-07-17）

## 1. 本阶段结论与变更边界

第一阶段按照“先打通黑盒模式、保持原 pytest git  1
（batch 调序）和模式 2（batch 拆分）。实现支持 Q 的 BSND/TND layout，以及 KV
的 BSND/TND/PA_BBND layout；主输出和可选 LSE 均按源 token 对齐后复用原 pytest
的数值精度标准比较。

本阶段代码变更只位于：

~~~text
attention/sparse_flash_mla/tests/pytest/
~~~

没有修改 SMLA host/kernel、`sparse_flash_mla_golden.py`、原 batch 测试入口或
`batch/sparse_flash_mla_process.py`。一致性测试默认关闭，原有
`single/save/load/load_graph` 行为不变。另更新本需求记录和开发计划文档，用于
记录实际交付边界和第二阶段设计。

当前实现状态：

| 范围 | 状态 | 说明 |
| --- | --- | --- |
| 模式 1：batch 调序 | 已实现 | 随机种子可复现，强制生成非恒等排列 |
| 模式 2：batch 拆分 | 已实现 | 分组非空、无重复、完整覆盖原 batch |
| BSND/TND/PA_BBND | 已实现 | 使用 layout-aware 变换，不做统一 `tensor[order]` |
| 主输出/LSE 对齐 | 已实现 | 使用 `TokenOrigin(source_batch, source_token)` |
| metadata 白盒解析 | 未实现 | 进入第二阶段设计，当前 `RunResult.metadata` 为 `None` |
| 模式 3/4 | 未实现 | 明确报 unsupported，不做语义不可靠的伪实现 |
| graph 一致性 | 未实现 | 原 graph 测试未受影响，第二阶段单独适配 |
| level 3 NPU 验证 | 环境阻塞 | 当前 910_93 环境的 torch_npu/CANN 9.2 首次 NPU 操作拒绝 level 3 |

## 2. 本次每个文件的职责

| 文件 | 类型 | 本次职责 |
| --- | --- | --- |
| `pytest.ini` | 修改 | 注册 `consistency` marker，避免 pytest 未注册告警，并表明该测试默认关闭 |
| `test_run.sh` | 修改 | 增加独立 `consistency` 子命令；解析 PT 目录、模式和 seed；筛选 `B>=2` 的 PT；每个 case 使用独立 pytest 进程；汇总 PASS/UNSUPPORTED/FAIL |
| `batch_consistency/__init__.py` | 新增 | 声明 batch consistency Python 包；不承载业务逻辑 |
| `batch_consistency/model.py` | 新增 | 定义 `TokenOrigin`、`ConsistencyCase` 和 `RunResult`；保存源 token 映射、变换元信息和统一运行结果 |
| `batch_consistency/layouts.py` | 新增 | 提供 BSND batch 选择、TND 分段/拼接/重算 prefix sum、PA block table 行选择、per-batch 向量选择等纯变换原语 |
| `batch_consistency/schema.py` | 新增 | 校验 PT 必需字段、layout 和 tensor rank；提取物理 Q 长度；检查变换后的 B、prefix sum、seqused、max length 和 PA block id invariant；构造基准 origin |
| `batch_consistency/transform.py` | 新增 | 实现模式 1/2 共用的 `_select_batches`，同步所有 batch/token 耦合字段并更新派生标量；模式 3/4 仅提供带原因的显式 unsupported |
| `batch_consistency/compare.py` | 新增 | 按 `TokenOrigin` 对齐主输出/LSE；合并模式 2 的子 case 输出；复用 `result_compare_method.check_result`；BF16-safe MD5 仅用于 bitwise 诊断 |
| `batch_consistency/runner.py` | 新增 | 在不改旧 `call_npu()` 二元返回契约的前提下设置 deterministic level 3、运行基准/派生 case，并在 `finally` 中恢复真实旧 level；识别当前运行时的 level-3 unsupported |
| `test_batch_consistency_unit.py` | 新增 | 85 条 CPU 单测，覆盖三类 layout、模式 1/2、字段同步、schema/invariant、origin/LSE、BF16 MD5、分组边界、模式 3/4 unsupported 和 deterministic level 恢复 |
| `test_sparse_flash_mla_batch_consistency.py` | 新增 | NPU pytest 入口；默认 skip，仅在 `SMLA_CONSISTENCY_TEST=1` 时收集执行；从 PT 加载 `B>=2` case，分别执行模式 1/2 并输出可诊断信息 |
| `requs/requs.md` | 修改 | 保留原始需求，并追加第一阶段逐文件开发记录、模式分析、验证状态和配置演进结论 |
| `requs/SMLA_batch一致性测试开发计划.md` | 修改 | 校正实际阶段边界，记录与 GLM-5.2 的第二阶段联合审议、阻断项、里程碑和验收门禁 |

## 3. 第一阶段执行链路

~~~text
bash test_run.sh consistency -P <pt_dir> -M 1,2 --seed <seed>
  -> 设置独立的 SMLA_CONSISTENCY_* 环境变量
  -> 仅选择 B>=2 的已有 PT
  -> 对每个 PT 启动独立 pytest 进程
  -> schema 校验并在 level 3 下运行原始基准 case
  -> 生成模式 1 或模式 2 派生 case
  -> 再次执行 SMLA
  -> 按 TokenOrigin 对齐/合并主输出和可选 LSE
  -> 复用原数值容差判断
  -> 汇总 PASS / UNSUPPORTED / FAIL
~~~

每个 PT 独立进程可以隔离 NPU/编译状态及单 case 异常；一致性入口不复用原
`load`/`load_graph` 分支，因此不会改变已有测试的默认收集和执行路径。

## 4. 模式 1：batch 调序实现分析

模式 1 的输入是原始 `B>=2` case 和一个 `[0, B)` 的排列。未显式传入排列时，
使用 `seed` 生成非恒等排列；如果传入自定义排列，则校验它不遗漏、不重复且不越界。

变换过程不是只重排 Q，而是通过共用 `_select_batches` 同步处理完整语义：

1. BSND Q/KV 按显式 batch 第 0 维 `index_select`；
2. TND Q/KV 先用原 `cu_seqlens_*` 切成 per-batch segment，再按排列拼接，最后
   重算新的 prefix sum；
3. PA_BBND 的 KV 是共享物理 cache，不能随逻辑 batch 搬动；实现保持 cache tensor
   原样，仅重排 `ori_block_table`/`cmp_block_table` 的行；
4. 稀疏索引和可变 top-k 按 Q token 的原始分段同步重排；
5. `seqused_*`、`cmp_residual_kv` 等 per-batch 向量使用同一个排列；
6. 更新顶层、`params` 和 `metadata_input` 中的 B、S/T、prefix sum 和 max length；
7. 为派生输出生成 origin 序列，例如新 batch 0 来自旧 batch 2，则该 batch 的
   token 记录为 `(2, 0)`, `(2, 1)` 等；
8. 派生执行结束后，按 origin 把主输出和 LSE 恢复到基准顺序再比较。

因此模式 1 验证的是“改变 batch 调度顺序不应改变每个源 token 的结果”，而不是
比较两个 shape 恰好相同但语义已经错位的 tensor。

## 5. 模式 2：batch 拆分实现分析

模式 2 将 `[0, B)` 划分为 2 到 B 个非空子组。默认分组由 `seed` 决定；自定义分组
必须满足所有原 batch 恰好出现一次。每个子组调用与模式 1 相同的
`_select_batches`，所以三类 layout 和全部耦合字段只有一套变换规则。

执行和比较过程如下：

1. 原多 batch case 在 level 3 下执行一次，得到基准输出和基准 origin；
2. 每个子组派生为独立 case，B、S/T、prefix sum、seqused、metadata 派生字段都
   调整到子 case 的实际值；
3. 子 case 串行执行，避免同时保留多份大 KV；PA_BBND 仍共享原物理 cache；
4. BSND 主输出按 `[batch, token]`、TND 主输出按扁平 token 收集到
   `TokenOrigin -> tensor` 映射；
5. 可选 LSE 使用相同 origin 规则合并；
6. 检查 origin 无重复、无缺失，再按原 batch/token 顺序重建完整输出并与基准比较。

该模式验证的是“一个多 batch 调用与若干子 batch 调用合并后的逐 token 结果一致”。
拆分组本身可以乱序，但比较不依赖子组执行顺序。

## 6. Layout 与字段同步规则

| 类别 | 字段/规则 | 处理方式 |
| --- | --- | --- |
| Q | `q` | BSND 选 batch 行；TND 按原 Q prefix sum 选 segment |
| KV | `ori_kv`、`cmp_kv` | BSND 选 batch 行；TND 选 segment；PA 保持物理 cache 不动 |
| 稀疏索引 | `ori_sparse_indices`、`cmp_sparse_indices` | 与 Q token 使用同一分段和顺序 |
| PA 映射 | `ori_block_table`、`cmp_block_table` | 只选/重排行，不复制或重排 cache block |
| Q 长度 | `cu_seqlens_q`、`seqused_q` | 重算 prefix sum，选择 per-batch valid length |
| KV 长度 | `cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`、`seqused_ori_kv`、`seqused_cmp_kv` | 按 layout 重算或选择 |
| 压缩残差 | `cmp_residual_kv` | 按 batch 选择 |
| 可变 top-k | `ori_topk_length`、`cmp_topk_length` | BSND 按 batch；TND 使用变换前 Q prefix sum 分段 |
| 派生字段 | `B/S1/S2/T1/T2/T3`、`max_seqlen_*` | 同步更新顶层、`params`、`metadata_input` 中实际存在的字段 |
| 不变字段 | `sinks`、scale、mask/window、layout、PA 物理 cache | 语义不是 batch 行，不做机械重排 |
| 旧 golden | `cpu_output`、`softmax_lse` | 派生后已过期，明确清空，绝不伪装成派生 case golden |

变换完成后立即执行 invariant 校验；prefix sum 长度、tensor token 总数、B 维、
seqused 上界、metadata max length 或 PA block id 任一不一致都会在调用 NPU 前失败。

## 7. 精度判定与兼容性设计

- 主输出和 LSE 最终调用已有 `result_compare_method.check_result`，保持与 SMLA 原
  pytest 相同的容差和通过率标准；
- MD5 只报告是否 bitwise equal，不作为最终通过条件，因为不同调度可能产生合法的
  浮点累加顺序差异；BF16 通过 `view(torch.uint8)` 取原始字节，避免 NumPy dtype
  不支持；
- 旧 `call_npu(input_data) -> (output, softmax_lse)` 未修改；一致性 runner 只在外层
  包装为 `RunResult`；
- 每次 NPU 调用前读取真实旧 deterministic level，设置并校验 level 3，无论成功、
  精度失败还是异常都在 `finally` 中恢复旧 level；
- 模式 3/4 在 shell、pytest 和 transform 层都有明确原因，不会因“全部 skip”被误报
  为模式已通过。

## 8. 验证记录与当前环境状态

已完成的验证：

| 验证项 | 结果 |
| --- | --- |
| CPU 单元测试 | `85 passed` |
| 新 NPU 入口默认关闭 | `2 skipped`，符合 opt-in 设计 |
| 原 single/batch/graph 测试收集 | 15 个原测试可正常收集 |
| shell/static/diff 门禁 | `bash -n`、Python 静态检查、`git diff --check` 通过 |
| 三类真实 PT 纯变换 | 临时生成 BSND、TND、PA_BBND 的 B=2 case，模式 1/2 和 schema invariant 均通过 |
| 普通 SMLA+LSE NPU 基线 | Ascend910_9362（910_93 系列）deterministic level 0 下 `Pass，100%` |

当前 NPU 阻塞不是精度失败。环境中的 torch_npu/CANN 9.2 允许先设置并读回 level 3，
但在第一次 NPU 操作时返回：

~~~text
torch_npu.npu.set_deterministic_level currently only supports configuring 0/1/2
~~~

runner 将其转换为 `BatchConsistencyUnsupportedError`，pytest 记为 skip，shell 汇总为
`UNSUPPORTED`；它不会被统计为 PASS 或 FAIL。也就是说，第一阶段的软件框架和纯变换
已经完成，但真正的 level-3 batch 一致性 NPU 精度闭环仍需支持该能力的 torch_npu/CANN
及 kernel 环境复验。普通 level-0 基线通过不能替代 level-3 验收。

验证过程中产生的临时 Excel/PT、日志、cache 和隔离 CANNBot 数据均已清理，未纳入
版本管理。

## 9. 关于 `-M 1,2` 的结论

`-M 1,2` 确实比较简陋，它是第一阶段为了低耦合打通模式 1/2 而保留的临时兼容入口，
不是长期配置设计。第一阶段只有两个可执行模式，数字参数可以让 shell、环境变量和
pytest 快速贯通，同时避免改动 Excel/PT 生成链路；但它存在明显问题：数字不可读、
shell 中有模式语义、增加模式时需要修改多处，也无法表达 graph/metadata/profile 等能力。

第二阶段计划改成 Python 侧统一的命名 Enum/registry，推荐入口为：

~~~bash
bash test_run.sh consistency \
  -P ./data \
  --modes reorder,split \
  --seed 20260716
~~~

计划中的迁移规则：

1. `reorder`、`split`、`token-split`、`prepend-delete` 成为唯一规范名称；
2. shell 只透传模式字符串，不再理解 `1/2/3/4` 的业务含义；
3. 第二阶段继续接受 `-M 1,2`，统一在 registry 中映射并给出 deprecated 提示；
4. CI 和文档迁移到 `--modes reorder,split` 后，数字形式只作为兼容别名；是否最终删除
   要在下游脚本完成迁移后单独评审，不在本阶段强制移除；
5. 增加 `--profile blackbox-basic` 等组合配置，并在 Excel/每 case 配置链路打通后使用
   versioned、namespaced 配置，避免继续增加魔法数字。

本次按要求只记录第二阶段设计，尚未修改上述配置代码。详细方案见
`requs/SMLA_batch一致性测试开发计划.md` 的第二阶段联合分析。

# 第一阶段开发检视一

cannbot模型不止glm-5.2还有deepseek v4和qwen3.7-max，我希望chatgpt能和多方合作沟通商量规划，毕竟别的模型有可能关注到chatgpt没关注到的点，实现的时候chatgpt可以指导其他模型帮你修改一下

```
# batch一致性测试框架开发大纲

## 1. 图例说明 (Legend)
- 🔴 **实现速度快，优先实现** (粉色背景)
- ⬜ **现有功能** (白色背景)
- 🔵 **实现速度慢，后期实现** (蓝色背景)

## 2. 核心流程 (Main Flow)
- **S2_B分析模块** 🔵
  - *描述：保障至少有token能被切到多核*
  - ⬇️
- **给定的参数** ⬜
  - ⬇️
- **当前生成golden的代码** ⬜
  - ⬇️
- **黑盒场景覆盖模块** 🔴
  - ⬇️
- **metadata算子调用** ⬜
  - ⬇️
- **自动场景覆盖模块** 🔵
  - ⬇️
- **白盒覆盖情况自动分析** 🔵
  - ⬇️
- **一致性对比模块** 🔴
  - ⬇️
- **精度对比** ⬜

## 3. 并行/辅助逻辑 (Parallel Logic)
- **两次执行间random方式** 🔴
  - *说明：此步骤与“黑盒场景覆盖模块”并行，并汇入“一致性对比模块”*
```

1. 首先一个问题就是代码规范：我检视你第一阶段修改的代码，有些函数采用了`_`开头不对外开放，但这种风格不符合开源规范，可以看看pytest之前的规范风格，对于不想开放的，封装在类里面

2. 项目背景我更新一下，一是要适配的算子应该是sparse_flash_mla和mixed_quant_sparse_flash_mla两个算子，两个算子需求基本都一样。
二是准确的说`batch一致性`当前SMLA和MQSMLA应该并没有支持，也就是cann环境里面没有支持相关功能，这是我疏忽的点。和开发对了之后，应该是SMLA/MQSMLA支持`batch一致性`和pytest适配`batch一致性测试`应该是同步进行的，后面SMLA/MQSMLA支持`batch一致性`的功能开发完成时就需要用到pytest进行验收测试，但当前我并不清楚有什么方法可以完成测试

`ops-transformer_0716/requs/docs/Milan&TorinoC12_RP3_B050_G项目_STC用例.xlsx`就是当前SMLA/MQSMLA支持`batch一致性`特性的STC验收用例，查阅一下对你理解推进当前pytest的需求有帮助

`/mnt/workspace/gitCode/cann/ops-transformer_0716/requs/docs/会议纪要.md`是请SMLA/MQSMLA支持`batch一致性`的开发者讲解思路，看看有没有信息能用，对于黑盒四模式的分析我觉得还重要的

虽然SMLA/MQSMLA目前还不支持`batch一致性`，但mixed_quant_sparse_flash_mla_metadata算子的`batch一致性`编码已完成： https://gitcode.com/cann/ops-transformer/pull/8737，该修改我已下载到ops-transformer下，并编译了自定义算子包安装（不理解自定义算子包的可以在ops-transformer/docs/zh中找相关文档）
```
# ops-transformer仓
bash build.sh --pkg --ops=mixed_quant_sparse_flash_mla_metadata,mixed_quant_sparse_flash_mla --soc=ascend910_93
cd build_out/
sudo bash cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/Ascend/
```
要使用mixed_quant_sparse_flash_mla_metadata相关修改
```
source /mnt/workspace/gitCode/env.sh
source /home/Ascend/vendors/custom_transformer/bin/set_env.bash
```
可以用这个修改跑MQSMLA metadata结果，来辅助验证和代码开发（有能力甚至可以自己尝试去修改sparse_flash_mla_grad_metadata代码编译装包来辅助验证）

3. 查阅`SMLA_batch一致性测试开发计划.md`文档，我对模式2、模式3的的理解有点不同，模式2把一个多 batch case 拆成多个子 case，但预期应该不用合并子 case 吧，各case对应输出一致就行，当然如果你觉得合并起来不影响各自case输出并且好比较，那就合并吧；对于模式3，我就不太理解原规划是什么目的，可以麻烦结合所有文档讲解一下吗

默认分组由 `seed` 决定————这个决策评估一下合理吗，符合文档需求？看看分组这事在算子侧是怎么实现的？


