# SMLA/MQSMLA batch 一致性需求与开发台账

> 最近整理：2026-07-21
> 第一阶段基线提交：`b680bfbdd test(smla): add phase-one batch consistency framework`
> 第一阶段二次推进提交：`183beb730 test(smla): advance batch consistency framework phase two`
> pytest_v2阶段二基线提交：`f9ca0da86 test(smla): integrate dual-op batch consistency pytest v2`
> 本文用途：归档需求背景、权威资料、需求冲突裁决、实现记录、检视答复和当前状态。具体后续里程碑见
> `SMLA_batch一致性测试开发计划.md`。
> 阅读提示：第 6-12 节和前两轮“检视意见处理记录”保留历史快照；当前权威状态见原始
> `# 检视意见三`后追加的“检视意见三新增意见处理记录（2026-07-21）”。逐STC/公式/mask冲突见
> `SMLA_MQSMLA需求冲突与对齐清单.md`；外部框架逐项评估见`data/README.md`。若历史状态与
> 新记录冲突，以2026-07-21记录为准。

## 1. 项目背景与最终范围

batch 一致性的根因不是普通数值精度不足，而是 FlashDecode 在 S2 方向切到不同核后，
FA partial result 的归约顺序可能随 batch、token 或 shape 改变。浮点加法不满足结合律（A + B + C != A + (B + C)），
所以结果虽然可能仍通过普通容差，却不再逐 bit 一致。

本需求的最终范围同时包含两个主算子：

- `sparse_flash_mla`，简称 SMLA；
- `mixed_quant_sparse_flash_mla`，简称 MQSMLA。

二者要同步完成两条开发线：

1. 主算子读取 deterministic level 3，并实现与其他 token 调度无关的固定规约语义；
2. pytest 构造等价输入、执行 level 3、按源 token 对齐输出，并同时检查 MD5 与原精度标准。

当前 SMLA/MQSMLA 主算子和本机 torch_npu/CANN 组合都不能视为已支持 batch 一致性。
pytest 框架通过只代表测试侧逻辑可用，普通 level 0 精度通过也不能代替 level 3 特性验收。

## 2. 需求资料与采用顺序

| 优先级 | 资料 | 用途 |
| --- | --- | --- |
| 1 | `docs/SMLA batch一致性.md` | 需求目标、模式 1-4、level 3 和固定规约语义的主基线 |
| 2 | 两算子 README、golden、metadata/opapi/tiling/kernel 源码及 PR8737 | 解释接口、mask 数学语义和当前/目标实现事实 |
| 3 | `docs/Milan&TorinoC12_RP3_B050_G项目_STC用例.xlsx` | 验收 case 草案；已知错误必须进入冲突台账，不能反向覆盖主需求 |
| 4 | `docs/batch一致性测试框架.md`、`docs/会议纪要.md` | 开发侧方法草案和辅助解释，不作为冲突时的最终裁决 |
| 5 | `docs/SMLA支持batch一致性.md`、`docs/metadata输出结果解析.md` | 设计补充和历史布局参考 |
| 6 | `data/算子多batch一致验证.md`、示例Excel/脚本 | 外部aclnnfuzz工程实践参考；不决定SMLA语义，不进入正式执行链 |

出现冲突时不静默选择一个说法：`SMLA batch一致性.md` 决定需求目标，算子 README/golden/
kernel 和 PR 决定可执行语义，STC 与临时框架作为待修订材料。仍不能证明数学输入等价的
变换必须报告 `INVALID_TRANSFORM`，不能把需求冲突包装成算子精度失败。

## 3. 已确认的需求事实与资料冲突

### 3.1 deterministic level

- level 3 是 batch 一致性功能开关；
- setter/getter 成功不代表算子可执行，当前环境会在第一次 NPU 操作时延迟拒绝；
- 测试必须保存并恢复真实旧 level；
- 不允许退回 level 0 后把结果标成 batch consistency PASS。

### 3.2 metadata 当前布局

SMLA 和 MQSMLA 当前 header 均定义：

```text
总长: 1024 * int32
FA:   [0, 324)   = 36 cores * 9 fields
FD:   [324, 900) = 72 cores * 8 fields
尾部: [900, 1024)= 124 int32 reserved
```

FA 索引 0-8 均已命名；FD record size 为 8，但 header 只命名 0-6，索引 7 必须按未知/
预留槽报告。`int32_t` backing tensor 对应 `uint32_t` struct，解析时按原始 bit pattern 转成
无符号值。

MQSMLA metadata AICPU 源码已经明确：

- `s2End` 是取不到的右开边界；
- `bN2End/gS1End/s2End` 是连续任务端点，下一核 start 来自上一核 end；
- FD 的 `workspace_num` 是该归约任务的 S2 split 数，`m_start/m_num` 是同一任务进一步
  分配到 AIV 核的 M 范围；
- FD `M` 是 `s1g` 行，缺少 group-size 等上下文时不能直接称为一个 query token。

### 3.3 S2 分类冲突

框架定义 Level-1 四类：

1. `S2 / 32 == 128`；
2. `S2 / 32 < 128`；
3. `S2 / 32 > 128 && S2 % 32 == 0`；
4. `S2 / 32 > 128 && S2 % 32 != 0`。

STC 014 把 `S2=5120` 描述为“不是整数倍”，但 `5120 % 32 == 0`，并且两份设计文档都
给出 `R=align_up(5120/32, 128)=256`。因此代码按可计算事实把 5120 归入第 3 类；STC
文字需由用例负责人修订，不能在代码里制造错误分类来迁就描述。

### 3.4 2026-07-20 新发现的 floor/ceil 公式冲突

PR8737 head `31a3b6d3` 的 C++ 使用整数 `actTotalS2Size / 32`，即先 floor；需求文档与此前
analyzer按ceil理解。两者在4097处会分别得到tile128/256，在1..31还存在PR表达式zero edge。
当前白盒报告同时输出两套quotient/tile/conflict，结构展开暂按PR head并显式记录zero guard。
这不是已裁决事项，必须由tiling/需求owner确认。完整证据和数值表见
`SMLA_MQSMLA需求冲突与对齐清单.md`。

## 4. 四种黑盒模式的裁决

| 模式 | 规范名称 | 当前裁决 | STC 对应 |
| --- | --- | --- | --- |
| 1 | `reorder` | 同一组 batch 改变执行顺序，Q/KV/长度/索引同步重排，按 origin 还原后逐 token 比较 | 018，以及 020/024-027 的组合场景 |
| 2 | `split` | 从基准多 batch case 选取一个或多个 batch 子集；每个子 case 直接和基准对应 batch 比较 | 003/008/011/015，随机组合见 020/024-027 |
| 3 | `token-split` | 对一个 batch 的 Q token 做完整、不重叠的连续分区；例如 `16K -> 4K+12K` 两次调用，每段都与基准 origin 直比 | 002/006/007/012/016，随机 token 见 019/020/024-027 |
| 4 | `shape-change` | 按主需求比较共同前缀；例如 `300+100` 与 `300+200` 的前 300 个 token，并为位置敏感 mask 同步调整逻辑 KV 右边界 | 004/009/013/017 |

### 4.1 模式 2 为什么不以合并结果为主

用户检视意见正确。模式 2 的主验收应是：

```text
基准 case 的对应 origin 输出 <-> 每个派生子 case 的输出
```

每个子 case 单独报告 MD5、数值精度和 batch id。只有显式 groups 恰好完整覆盖原 batch
时，框架才额外合并一次做诊断；合并失败不能掩盖某个子 case 的直接结果，不完整子集也
不再被强制补全。

### 4.2 模式 3 到底在测什么

模式 3 以 `docs/SMLA batch一致性.md` 的完整拆分示例为准：一个 `B=1, Token=16K` 的基准
调用拆成 `[0,4K)` 和 `[4K,16K)` 两个独立调用，两个派生区间必须完整、不重叠地覆盖原
query token。每段按 `TokenOrigin` 直接与基准对应 token 比较，而不是只抽一个 token 做 smoke。

对无位置 mask，KV 上下文保持不变。对 RightDownCausal/Band 等位置敏感 mask，不能只切 Q；
对原区间 `[start,end)` 使用：

```text
new_ori_S2 = old_ori_S2 - old_Q_length + end
```

这样派生位置 `old_token-start` 与原位置 `old_token` 的可见 KV 窗口相同。cmp 长度和 residual
也按 `cmp_ratio` 同步。实际输入语义 oracle 随后逐 token 校验 Q、ori/cmp 可见 KV、mask
窗口、sparse indices、top-k 和 sinks；无法证明等价即报告 `INVALID_TRANSFORM`。

### 4.3 模式 4 如何按主需求实现

模式 4 不再以 STC 的“追加”文字或临时框架的 prepend 建议作最终裁决，而是直接实现主需求
“共同前缀、不同 shape”：基准 `300+100`，派生 `300+200`，只比较共同前 300 个 token。
对位置敏感 mask 使用：

```text
new_ori_S2 = old_ori_S2 + new_Q_length - old_Q_length
```

该式保持 `ori_S2-Q_length` 不变，新增长度位于共同 token 的未来，不改变其可见窗口；cmp
上下文同步调整。对 BSND/TND/PA_BBND 容量、mask 或索引无法证明安全的 case，语义 oracle
会拒绝执行，不把输入构造错误归因成算子一致性失败。

## 5. 随机策略与 seed 的检视结论

默认分组完全由 seed 决定不适合作为基础 STC：失败 case 不直观，固定用例与调度覆盖也
难对应。当前规则调整为：

- 模式 1仍按需求打乱 batch，但固定 seed 使排列可复现且强制非恒等；
- 模式 2主需求的规范默认是 `B=8 -> [0,1,2] + [3,4,5,6,7]`，其他 B 才使用确定性连续二分；
- 正式验收优先通过 `--split-groups '0,1,2;3,4,5,6,7'` 显式给组，子集场景也可不覆盖全集；
- 只有 `--randomize`/fuzz profile 才用 seed 打乱模式 2分组；
- 随机场景主要对应 STC 018-020、024-027，不能替代 001-017 的显式基准/派生关系。

算子侧不是按测试 seed 分组。metadata 根据 B/N2/S1/S2、有效长度、mask、top-k 和核数
连续分配 FA 任务；seed 只是测试输入组合控制，不是 kernel 调度参数。

## 6. 第一阶段历史快照（截至提交 183beb730）

> 本节至第 12 节用于追溯第一阶段为何作出当时的判断，不代表 2026-07-19 的最终能力。
> 当前实现、逐项答复和剩余阻断统一见原始 `# 检视意见` 后的处理记录。

### 6.1 第一阶段开发检视一：逐项答复与修复

| 检视项 | 处理结果 | 当前状态 |
| --- | --- | --- |
| 模块级 `_private_function` 不符合本项目风格 | schema、transform、compare、runner、配置、registry、metadata 逻辑封装进职责类；模块只保留兼容的公开函数 | 已修复，生产 `batch_consistency/*.py` 无模块级 `def _...` |
| 范围遗漏 MQSMLA | 新增 `CaseAdapter` 统一 `input/op_input` 和 metadata 命名；变换/schema 有 MQSMLA synthetic 回归；新增已安装 MQ metadata 探针 | 数据面已适配；MQ 主算子 executor 尚未接入 |
| 误以为只是环境不支持 | 文档和状态改为“主算子功能与 pytest 同步开发，当前均不能验收” | 已修正 |
| 模式 2不应依赖完整合并 | 每个子 case 用 `compare_subset` 直接比较，merge 仅完整分区诊断 | 已修复 |
| 模式 3目的不清 | 按 STC token 子集语义实现，并记录资料差异和 mask 前置条件 | 已实现受限版本 |
| 默认 seed 分组合理性 | 基础模式改为确定性二分，随机仅显式启用 | 已修复 |
| `-M 1,2` 太简陋 | 新增命名 Enum/registry、`--modes`、`--profile`；数字形式仅 deprecated 兼容别名 | 已修复控制面 |
| metadata 应利用已安装 MQ 包 | 新增严格 parser、保守 analyzer、CLI probe，并完成真实 level 0/3 探测 | 已推进 |

## 7. 当前执行与比较链路

```text
命名 modes/profile
  -> schema + CaseAdapter 识别 SMLA/MQSMLA case 格式
  -> 在任何 NPU 调用前生成并校验派生 case
  -> 保存旧 deterministic level，设置并校验 level 3
  -> 执行基准 case，SMLA eager 捕获主算子实际消费的 metadata
  -> 执行派生 case
  -> TokenOrigin(source_batch, source_token) 对齐有效 token
  -> 去除 BSND padding，打包成规范 token-major tensor
  -> MD5 bitwise 必须相等，同时复用原 check_result 数值精度检查
  -> finally 恢复真实旧 level
  -> PASS / FAIL / UNSUPPORTED / INVALID_TRANSFORM / NOT_APPLICABLE
```

MD5 不再只是日志诊断。会议和 STC 明确要求对应 token 结果逐 bit 一致，因此 batch
consistency PASS 同时要求 bitwise 与数值精度通过。无效 padding 不属于对应 token，不能
参与 MD5；框架保留物理 Q 长度用于定位，同时用 `seqused_q` 记录有效长度。

## 8. 文件级开发记录

以下均位于 `attention/sparse_flash_mla/tests/pytest/`，除文档外没有修改主算子 host/kernel。

| 文件 | 本轮职责 |
| --- | --- |
| `batch/sparse_flash_mla_process.py` | 给 eager `call_npu` 增加默认关闭的 `metadata_sink`；旧二元返回不变，一致性 runner 可捕获主算子实际使用的同一 metadata tensor |
| `batch_consistency/model.py` | 保留 `TokenOrigin`、`ConsistencyCase`、`RunResult` 统一数据模型和旧 tuple 适配 |
| `batch_consistency/layouts.py` | 保留 BSND/TND/PA_BBND 选择、TND prefix sum、PA table 和 per-batch 向量原语 |
| `batch_consistency/adapter.py` | 新增 SMLA/MQSMLA case 格式描述；统一 tensor 容器、B/N1/N2/D 命名和 layout 访问 |
| `batch_consistency/registry.py` | 新增 `reorder/split/token-split/shape-change` Enum、数字兼容别名、profile、能力状态和统一解析 |
| `batch_consistency/schema.py` | 将内部逻辑收敛到 `CaseSchema`；支持两种 case 格式，区分物理/有效 Q 长度并校验变换 invariant |
| `batch_consistency/transform.py` | 将内部逻辑收敛到 `BatchCaseTransformer`/`TransformSemanticGuard`；修正模式 2，新增模式 3 BSND/TND token 选择，模式 4保留冲突说明 |
| `batch_consistency/compare.py` | 将 origin 对齐、digest、merge 封装成类；新增逐 subset 比较；只打包有效 token 做 MD5/LSE 比较，完整 merge 降为诊断 |
| `batch_consistency/runner.py` | 类化 level guard/executor/orchestration；先做 CPU 变换校验再碰 NPU；模式 2逐 case 比较；SMLA metadata 填入 `RunResult` |
| `batch_consistency/metadata_layout.py` | 新增严格 1024-int32 parser；支持当前 9-field 和文档 legacy 8-field FA 布局，显式解析未知/预留区域 |
| `batch_consistency/metadata_analyzer.py` | 新增 S2 Level-1 分类、FA endpoint chain、FD task/workspace/M-range 和异常报告；不猜缺上下文的 token 映射 |
| `batch_consistency/metadata_probe.py` | 新增 MQSMLA metadata-only 探针和 CLI；支持指定 level 并恢复旧值；明确不代表主算子 PASS |
| `batch_consistency/__init__.py` | 只导出无 PyTorch 依赖的 registry；其他 API 从具体子模块导入，避免 CLI 配置预检初始化 NPU backend |
| `test_batch_consistency_unit.py` | 从第一阶段 85 条扩充到 114 条 CPU 测试；新增 registry/profile、模式 2 subset、模式 3变长/TND、padding、MQ 格式、metadata parser/analyzer 和捕获回归 |
| `test_sparse_flash_mla_batch_consistency.py` | 使用类化配置和命名 registry；新增 mode 3入口；只收集 SMLA PT，避免误用 SMLA executor 执行 MQ PT |
| `test_run.sh` | shell 仅透传命名配置；新增 `--modes/--profile/--randomize`；保留 `-M`；区分 UNSUPPORTED/INVALID/NOT_APPLICABLE |
| `pytest.ini` | 第一阶段已注册 `consistency` marker，本轮未改变原 marker 行为 |
| `requs.md` | 本文；归档背景、资料、裁决、检视答复、文件职责、实测和状态 |
| `SMLA_batch一致性测试开发计划.md` | 记录多模型审议、阶段二实际进展、剩余里程碑和验收门禁 |

## 9. 配置接口与运行方法

推荐使用可读名称：

```bash
cd attention/sparse_flash_mla/tests/pytest
source /mnt/workspace/gitCode/env.sh

bash test_run.sh consistency \
  -P ./data \
  --modes reorder,split \
  --seed 20260716

bash test_run.sh consistency \
  -P ./data \
  --profile blackbox-supported
```

当前 profile：

| profile | modes | 用途 |
| --- | --- | --- |
| `blackbox-basic` | `reorder,split` | 默认稳定入口 |
| `blackbox-supported` | `reorder,split,token-split` | 启用当前所有已实现变换，部分 case 可报 INVALID_TRANSFORM |
| `blackbox-all` | 再加 `shape-change` | 用于显式暴露未实现能力；当前会配置失败，不会伪 PASS |

`-M 1,2` 仍映射到 `reorder,split` 并发出 deprecated warning，只用于下游迁移期。命名形式
已经是正式入口，不需要等整个 batch consistency 特性打通后再调整。

MQ metadata CLI：

```bash
source /mnt/workspace/gitCode/env.sh
source /home/Ascend/vendors/custom_transformer/bin/set_env.bash
python3 -m batch_consistency.metadata_probe \
  --pt /path/to/mqsmla_case.pt \
  --level 3 \
  --s2 4096,2048,5120
```

本机 `/home/Ascend/vendors/custom_transformer` 当前为 `root:root 0750`，普通 developer 用户
无法读取 `set_env.bash`。这属于安装权限问题，不是探针代码问题；应由环境维护者修正为
团队约定的可读权限，或在受控 root 会话中运行，不能通过把目录全局改成 0777 来规避。

## 10. 2026-07-17 验证与真实环境结论

### 10.1 CPU/兼容门禁

| 验证项 | 当前结果 |
| --- | --- |
| batch consistency CPU 单测 | `114 passed` |
| 新 NPU 入口默认关闭 | `3 skipped`，符合 opt-in 设计 |
| 原 single/batch/graph 收集 | 15 个原测试可收集 |
| Python/shell/diff 静态门禁 | `compileall`、`bash -n`、`git diff --check` 通过 |

第一阶段曾在普通 level 0 下跑通 SMLA+LSE，结果为 `Pass，100%`；该结果只证明普通基线
可用，不计入 batch consistency 验收。

### 10.2 已安装 MQ metadata 实测

运行前按用户提供的两个 env 脚本配置，并因当前安装目录权限使用受控 root 会话。实测：

1. level 0 调用成功，返回 `(1024,) torch.int32 npu:0`；
2. 示例 `B=1, S1=1, S2=8192, TND+PA_BBND` 解析得到 1 个 enabled FA core、0 个 FD
   core、reserved 全 0；这是普通调度结果，不是 batch 一致性证据；
3. 设置 level 3 后 getter 返回 3，但第一次 MQ metadata NPU 调用报：

```text
'torch_npu.npu.set_deterministic_level' currently only supports configuring 0/1/2 !
```

4. 探测结束后 level 已恢复为 0；
5. 当前工作树 MQ metadata AICPU 中 `supportFd` 默认 `false`，未检索到赋值路径。静态源码
   和 level 0 的无 FD 结果一致；但 level 3 在 runtime 层已被拒绝，所以不能据此断言已安装
   二进制的 level-3 路径最终行为。

## 11. 当前能力状态

| 能力 | 状态 | 可声明边界 |
| --- | --- | --- |
| 模式 1 reorder | 测试框架完成 | 主算子/level 3未支持，不能声明 NPU PASS |
| 模式 2 subset split | 测试框架完成 | 每个子 case 直比；merge 仅诊断 |
| 模式 3 token split | 受限实现 | BSND/TND、KV 保持、静态 mask guard；actual-input oracle 尚缺 |
| 模式 4 shape change | 未实现 | STC 与会议冲突待裁决 |
| SMLA/MQSMLA 双格式变换 | 完成 | MQ synthetic CPU 回归通过 |
| SMLA eager metadata 同对象捕获 | 完成 | 旧 `call_npu` 返回契约不变 |
| MQ metadata probe | 完成 | metadata-only，不执行/验收 MQ 主算子 |
| raw metadata parser | 完成 | 当前 9-field + legacy 8-field，严格 size/dtype |
| conservative analyzer | 部分完成 | Level-1、FA chain、FD task；完整 token/core 展开尚缺上下文 |
| MQ 主算子 executor | 未接入 | 应在 MQ pytest 侧复用共享框架，不能复制整套 op 调用 |
| graph consistency | 未实现 | 原 `load_graph` 不受影响 |
| 支持 level 3 的真实 CI | 阻塞 | 当前 910_93 + torch_npu/CANN 9.2 延迟拒绝 |

## 12. 原始需求归档

最初需求要求：

- 只修改 `attention/sparse_flash_mla/tests/pytest` 内测试代码，不影响 SMLA host/kernel；
- 先完成黑盒场景，再推进 metadata 白盒分析；
- 新增显式开关，默认不改变 `single/save/load/load_graph`；
- 复用当前 golden 和精度比较逻辑；
- 在 NPU 运行前 `source /mnt/workspace/gitCode/env.sh`；
- 通过开发计划与 GLM-5.2 交互，并做好文档和 Git 版本管理。

第一次阶段检视进一步修正为：

- 目标是 SMLA 和 MQSMLA 两个算子；
- 主算子功能本身尚未支持，pytest 与功能同步开发；
- 查阅 STC Excel 和会议纪要后重新裁决模式 2/3；
- 使用已安装 MQ metadata 自定义包辅助验证；
- 邀请 GLM-5.2、DeepSeek V4、Qwen3.7-Max 多方独立审议；
- 清理模块级私有函数风格，说明 `-M 1,2` 的长期替代方案。

上述要求已转成本文的可追踪条目；未完成部分进入开发计划的阻断项和后续里程碑，不再
把原始口述需求与当前实现状态混写在同一段落。


# 检视意见

cannbot模型不止glm-5.2还有deepseek v4和qwen3.7-max，我希望chatgpt能和多方合作沟通商量规划，毕竟别的模型有可能关注到chatgpt没关注到的点，实现的时候chatgpt可以指导其他模型帮你修改一下

做好开发日志记录和文档记录

1. 我发现你将SMLA、MQSMLA的pytest修改全部放在ops-transformer_0716/attention/sparse_flash_mla/tests/pytest下，变通一下行吗，我不是说了之前的背景有点错误吗？不止要修改SMLA，还要修改MQSMLA，既然要修改MQSMLA，那其pytest的修改应该放在对应算子下面（ops-transformer_0716/attention/mixed_quant_sparse_flash_mla/tests/pytest），懂吗

2. 关于项目背景在`SMLA batch一致性.md`里面应该有详细介绍说明了，我读取了相关文档知道了算子侧是通过`level 3`去控制batch 一致性，那可以举个例子说明在pytest用例中是如何开启的呢，因为这两个算子都是两段式调用（metadata + 主算子），所以把参数给全。

3. MQSMLA metadata算子的batch一致性编码https://gitcode.com/cann/ops-transformer/pull/8737 ，已在ops-transformer中，你觉得对你有没有帮助，能启用一些临时验证吗？因为研发进度很赶，要快速推进

4. STC用例可能有些地方有失误，后续开发STC可能还会修正，还有`batch一致性测试框架.md`文档只是列出来的一个临时开发计划，我建议有冲突的地方以`SMLA batch一致性.md`为准，例如`SMLA batch一致性.md`文档的目标中第一点就涵盖了1/2模式，模式3/4也解释得很清楚，建议还是更新一下模式需求，`batch一致性测试框架.md`和`会议纪要.md`是开发侧理解的测试，但实际应该以`SMLA batch一致性.md`需求为准，就像`batch一致性测试框架.md`提到的一些模式解决方法不一定准备哈，建议实现过程中查看源码评估一下，就是你说“
- 框架和会议纪要建议在首部 prepend，以维持右下角对齐 mask；
- STC 004/009/013/017 写的是在各 batch 后面追加 token；
- 另一份背景材料又用“前 300 token 相同”描述前缀比较。
”
这时候建议查看{op}算子目录下的实现评估一下该如何实现需求。（文档需求有冲突的集中统计一下）

5. 关于 `默认分组` 的事我并不清楚，究竟是按seed来做模式2，还是按其他方式，我建议看`SMLA batch一致性.md`的需求或其他文档，因为pytest只是测试框架层：做参数处理、算子torch_npu接口调用、cpu golden实现和调用、compare。算子内部实现不在我们范围，测试框架应该和算子内部对齐，算子内部时如何做分组的呢，建议查一查，难不成模式2的分组放在我们测试框架参数处理层吗？

6. 说真的，我对算子开发的信息根本不熟，就像各个维度参数（B/S1/S2和token等等）的意义我根本不太了解，太迷糊了，请问你能生成图像绘图吗，可以的话帮我在ops-transformer_0716/requs生成一个教学文档，对这些详细解释一下，简单明了的

7. 现阶段实现你都是将模块拆分放在batch_consistency（我说的不影响pytest原有功能，但允许你动原有文件啊），但实际最后验收时应该按照pytest的原有框架来（获取输入，做参数处理、算子torch_npu接口调用、cpu golden实现和调用、compare外加上pt保存执行策略等等）的，当前完全抽离，只是入口一致，虽然方便开发，但对外部使用者查阅代码时很困难。我建议在tests创建一个pytest_v2副本，去将单独抽离的模块迁回原有框架体系，方便我检视。

8. `/home/Ascend/vendors/custom_transformer`权限不够，就加sudo
```bash
sudo source /mnt/workspace/gitCode/env.sh
sudo source /home/Ascend/vendors/custom_transformer/bin/set_env.bash
```

9. 算子调用不懂的看ops-transformer_0716/attention/{op}/README.md，op代指算子名

10. 黑盒做完后搞一下`batch一致性测试框架.md`文档中的`白盒覆盖情况自动分析`

## 检视意见处理记录（2026-07-19）

> 历史快照：本节记录提交`f9ca0da86`前后的设计。其平铺`batch/consistency_*.py`路径和
> metadata-only probe已在2026-07-20整理/删除；当前文件结构与状态以“检视意见二处理记录”为准。

### 处理结论

本轮已把测试侧实现迁入两个算子各自的 `tests/pytest_v2`，并在原 pytest 的输入生成、CPU
golden、PT 保存/加载、两段式 NPU 调用和精度比较流程上接入模式 1-4、MD5 与 metadata
白盒分析。阶段二的 pytest 代码已经推进到“可在支持 level 3 的环境执行真实用例”的状态。

当前不能声明 SMLA/MQSMLA batch 一致性已经在 NPU 通过：本机 torch_npu/CANN 会在设置
level 3 后的第一次 NPU 操作处拒绝该等级；本地算子源码快照与 PR8737 也不处于同一版本。
因此本轮的准确边界是“测试框架与 CPU 门禁完成，真实 level-3 双算子验收被环境/版本阻断”。

### 10 条检视意见逐项答复

| 编号 | 答复与修复 | 证据/状态 |
| --- | --- | --- |
| 1 | 同意两个算子必须各自拥有 pytest。新建 SMLA 与 MQSMLA 两棵独立 `pytest_v2`；MQ executor、process、入口和 README 均位于 MQ 算子目录，不再让 SMLA executor 代跑 MQ PT | `attention/sparse_flash_mla/tests/pytest_v2/`、`attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/`，已完成 |
| 2 | level 3 在任何 tensor 上 NPU 前设置，作用域同时包住 baseline/derived 的 metadata 与主算子，并在 `finally` 恢复真实旧值；两算子全参数示例单独成文 | `level3两段式完整调用示例.md`、`batch/consistency_flow.py`，已完成；真实运行受 runtime 阻断 |
| 3 | PR8737 对字段语义、FD 开启条件和规约块公式有直接帮助。已实现 MQ metadata-only probe、1024-int32 parser 与 FA/FD analyzer；但 level 0 只能验证接口/布局/普通 FA 链，不能证明 level 3、FD 或主算子一致性 | `mixed_quant_sparse_flash_mla_metadata_probe.py`、`metadata_layout.py`、`metadata_coverage.py`，临时验证入口完成 |
| 4 | 已把 `docs/SMLA batch一致性.md` 设为第一需求源，STC/会议/临时框架降为辅助资料；模式 3/4 按主需求和算子 mask 语义实现，所有冲突集中登记，不再按错误资料静默实现 | 见下方“需求冲突集中台账”，已完成 |
| 5 | 模式 2 的 `3+5` 确实由 pytest 参数处理层构造为两次算子调用；它不是算子内部的核分组。正式验收使用显式 groups，规范示例 `B=8 -> 3+5`；seed 只影响显式 fuzz/randomize | `--split-groups '0,1,2;3,4,5,6,7'`，已完成 |
| 6 | 已新增面向初学者的教学文档和确定性 SVG，解释 B/S1/S2/N1/N2/D/G/T、token、Q/KV、三种 layout、metadata/主算子、FA/FD 和四种模式 | `SMLA_MQSMLA入门教学.md`、`images/smla_mqsmla_batch_consistency_overview.svg`，已完成 |
| 7 | 已按建议创建 `pytest_v2` 副本。一致性入口读取原 PT，调用各自原 process 和 CPU golden/`check_result`，保留 single/save/load/graph 文件；辅助代码放回原 `batch/` 体系，没有依赖 v1 的独立 `batch_consistency/` 包 | 两棵 `pytest_v2` README 与下方文件表，已完成 |
| 8 | `source` 是 shell builtin，`sudo source ...` 无法生效。正确方式是让 sudo 启动一个 shell，再在同一 shell 里 source 两个脚本并执行命令 | `sudo -n bash -lc 'source ... && source ... && <command>'`，已在教学文档给出 |
| 9 | 已以两个算子 README 和实际 process/op 调用为接口依据；MQ 保留量化参数，SMLA 保留可选 LSE，不用猜测公共调用形式 | 两算子 `README.md`、两棵 `pytest_v2/batch/*_process.py` |
| 10 | 黑盒模式完成后已实现白盒自动分析：解析 FA/FD，结合实际 case 上下文展开 batch/N2/query-token/S2-block/workspace，输出 Level-1/2 覆盖、核视角、缺失覆盖和结构错误 | `batch/metadata_coverage.py`，CPU synthetic 门禁已覆盖；真实 level-3 metadata 待环境 |

### level 3 两段式调用的关键点

pytest 中不是给 metadata 或主算子单独增加一个 `level=3` 参数，而是设置进程级 deterministic
level，再调用原两段流程：

```python
old_level = torch_npu.npu._get_deterministic_level()
try:
    torch_npu.npu.set_deterministic_level(3)
    assert torch_npu.npu._get_deterministic_level() == 3
    metadata = metadata_op(all_metadata_arguments)
    output = main_op(all_operator_arguments, metadata=metadata)
    torch.npu.synchronize()
finally:
    torch_npu.npu.set_deterministic_level(old_level)
```

实际代码中 `ConsistencyFlow` 的 guard 包住 executor，而 executor 调用的 `call_npu` 内部依次
执行 metadata 和主算子，所以两个阶段处在同一个 level-3 作用域，且主算子消费的正是第一段
返回的同一个 metadata tensor。完整 SMLA/MQSMLA 参数列表见 `level3两段式完整调用示例.md`。

### 需求冲突集中台账

| 编号 | 冲突/疑点 | 采用结论 | 处理方式 |
| --- | --- | --- | --- |
| C-01 | STC、会议纪要、临时框架与主需求的优先级不一致 | `docs/SMLA batch一致性.md` 是需求第一基线；算子 README、golden、源码与 PR 决定可执行语义 | 次级资料错误只登记，不反向覆盖主需求 |
| C-02 | 模式 2 被误解为 kernel 内部分组或依赖 seed | 模式 2 是测试侧把一次 `B=8` 构造成独立 `B=3`、`B=5` 调用 | 显式 groups 为验收接口，seed 仅 fuzz |
| C-03 | 模式 3 曾被实现成抽取少量 token | 主需求要求完整 token 分区，例如 `16K -> 4K+12K` | 强制 `sum(split_sizes)==old_Q_length`，每段直比 |
| C-04 | 模式 3 只切 Q 会改变 RightDownCausal/Band 位置 | 对 `[start,end)` 保持原 token 可见窗口 | `new_ori_S2=old_ori_S2-old_Q_length+end`，cmp 同步后运行语义 oracle |
| C-05 | 模式 4 有 prepend、尾部追加、共同前缀三种说法 | 主需求采用 `300+100` 对 `300+200` 的共同前 300 token | `new_ori_S2=old_ori_S2+new_Q_length-old_Q_length`，无法证明则 `INVALID_TRANSFORM` |
| C-06 | STC 014 把 `S2=5120` 写成不能被 32 整除 | 数学事实为 `5120 % 32 == 0`，属于大于 128 且整除类 | analyzer 按事实分类；STC 文案待修 |
| C-07 | `S2 / 32 == 128` 被误读成整数除法区间 | 需求表达的是精确等式，即 `S2=4096` | `classify_s2` 使用精确边界 |
| C-08 | STC 026 的 B 与 `cu_seqlens_q` 数量不一致 | 不能直接作为生成参数 | 待 STC 负责人修正后进入 manifest |
| C-09 | metadata 文档与 header 的 FA/FD 布局有版本差异 | 当前布局为 FA `36x9`、FD `72x8`、尾部 124；legacy `36x8` 单独解析 | parser 显式版本化；FD slot 7 按 unknown 报告 |
| C-10 | 本地源码、PR8737 与已安装 custom 包是否一致不明 | 三者不能互相替代为验收证据 | 报告各自版本边界，真实 CI 前必须对齐 |

### PR8737 证据与临时验证结论

对公开 MR ref `refs/merge-requests/8737/head` 的隔离只读检查得到 head
`31a3b6d3 fix fd bug`，相关提交链为：

| 提交 | 可用证据 |
| --- | --- |
| `e14f1559 mqsmla metadata support batch consistency` | 读取 `ACL_OPT_DETERMINISTIC`，向 metadata tiling 传递 `is_batch_consistency` |
| `8b4ea3a4 mqsmla metadata adapt batch consistency` | level-3 路径开启 `supportFd_`，实现固定 reduction tile 计算 |
| `f2424c8c add FD` | 增加 FD 生成路径 |
| `f7a7856f fix bug`、`31a3b6d3 fix fd bug` | 后续修复，说明验证必须绑定最终 head，不能只看早期提交 |

当前本地工作树快照没有这组 PR 改动，静态路径仍表现为 `supportFd=false`；已安装 custom 包可能
来自 PR，但当前 torch_npu/CANN 在第一条 NPU 操作前拒绝 level 3。因此本轮只把 level-0 probe
作为接口、布局和普通 FA 链的快速检查，不用它证明 level-3 调度、FD 或 MQSMLA 主算子 PASS，
也不在本次 pytest 范围内擅自合入/安装算子源码。

### 多模型联合审议与主审修正

本轮通过 CANNBot 让 `cannbot/glm-5.2`、`cannbot/deepseek-v4-pro`、
`cannbot/qwen3.7-max` 分别只读审议同一组需求、STC、源码和 pytest。三个模型共同建议：

- SMLA/MQSMLA 使用各自 pytest 树和 operator-owned executor；
- MQ executor 要接入主算子并捕获同一 metadata，而不只做 metadata probe；
- 必须保留原 PT/golden/process/compare 流程；
- FA/FD 白盒分析必须带 B/N2/G/Q/KV/mask 上下文，不能只打印 raw record。

模型意见经源码复核后有两处明确纠正：

1. 有模型认为 level guard 只包住主算子。实际 executor 调用的 `call_npu` 内部先 metadata、再
   主算子，外层 guard 已完整覆盖两段，不需要再嵌套一次 guard。
2. 有模型把 `S2 / 32 == 128` 按整数除法解释为 `[4096,4127]`。需求是数学精确等式，代码按
   `S2=4096` 处理。

多模型没有直接写仓库；它们用于发现遗漏和反向审查，最终实现与结论由主审按源码、PR 和
可重复测试裁决。

### pytest_v2 与原框架的完整执行链

```text
参数/Excel
  -> 原 golden 生成 input/op_input + cpu_output
  -> 原 PT save
  -> 原 batch PT load
  -> 一致性参数变换（Q/KV/长度/table/index 同步）
  -> actual-input 语义 oracle
  -> 保存旧 level，设置并校验 level 3
  -> 原 metadata op
  -> 原主算子（消费同一个 metadata）
  -> NPU baseline/derived origin 对齐 + MD5
  -> derived NPU 与原 cpu_output 的 check_result
  -> metadata Level-1/2 自动分析
  -> 恢复旧 level
```

一致性 PASS 同时要求 NPU baseline/derived bitwise MD5 相同、原 CPU 数值精度通过、metadata
结构无错误。BSND padding 不属于有效 token，不进入 digest；模式 2/3/4 都逐派生 case 直接
比较，不靠合并结果掩盖单个失败。

### 两棵 pytest_v2 的文件职责

以下 10 个辅助模块在两棵 `pytest_v2/batch/` 中保持逐字一致，但各自随算子 pytest 独立提交，
避免 MQ pytest 反向依赖 SMLA 目录。未来如项目接受公共 tests 包，可再抽取；本轮优先满足
operator-owned、便于检视和互不串用 executor 的要求。

| 通用文件 | 职责 |
| --- | --- |
| `batch/consistency_adapter.py` | 识别 SMLA `input` 与 MQSMLA `op_input`，统一 B/N1/N2/D 和镜像字段访问 |
| `batch/consistency_model.py` | 定义 `TokenOrigin`、`ConsistencyCase`、`RunResult` |
| `batch/consistency_layouts.py` | BSND/TND/PA_BBND 的 batch/token 选择、TND prefix sum、PA table 原语 |
| `batch/consistency_schema.py` | 校验 PT schema，区分物理/有效 Q 长度并检查派生 invariant |
| `batch/consistency_transform.py` | 实现模式 1-4、显式分组/分段、mask 上下文公式和 actual-input 语义 oracle |
| `batch/consistency_compare.py` | origin 对齐、有效 token 打包、MD5、原 `check_result` 与可选 LSE 比较 |
| `batch/consistency_config.py` | 命名 mode/profile、deprecated 数字别名和显式 groups/sizes 解析 |
| `batch/consistency_flow.py` | level guard、baseline/derived 执行、双门禁比较、metadata 报告和状态编排 |
| `batch/metadata_layout.py` | 严格解析 1024-int32、FA 36x9、FD 72x8、unknown/reserved |
| `batch/metadata_coverage.py` | 结合实际输入展开 FA/FD token/core/S2/workspace，统计 Level-1/2 与结构错误 |

SMLA `attention/sparse_flash_mla/tests/pytest_v2/`：

| 文件 | 职责/本轮变化 |
| --- | --- |
| `README.md` | 解释 v2 目的、完整数据流、模式命令和 level-3 阻断边界 |
| `pytest.ini` | 保留原 marker，并注册默认关闭的 `consistency` marker |
| `test_run.sh` | 保留 single/save/load/load_graph；新增 consistency、命名 modes/profile 和显式模式参数 |
| `sparse_flash_mla_paramset.py` | 原单用例参数入口，保持不变 |
| `sparse_flash_mla_golden.py` | 原 CPU golden 与输入生成，派生 NPU 精度仍使用其保存的 `cpu_output` |
| `result_compare_method.py` | 原数值精度判定，和 MD5 一起作为 PASS 门禁 |
| `utils.py` | 原日志/结果等工具，保持原框架能力 |
| `test_sparse_flash_mla_single.py` | 原 single 流程副本 |
| `test_sparse_flash_mla_batch.py` | 原 PT batch/load 流程副本 |
| `test_sparse_flash_mla_batch_graph.py` | 原 graph/load 流程副本；一致性 graph 尚未接入 |
| `batch/sparse_flash_mla_pt_save.py` | 原 PT 保存流程副本 |
| `batch/sparse_flash_mla_process.py` | 原 metadata+主算子 eager/graph；eager 新增默认关闭的 `metadata_sink`，旧返回契约不变 |
| `batch/sparse_flash_mla_consistency.py` | SMLA 专用 executor，将原 process 输出适配为 `RunResult` 并捕获实际 metadata |
| `test_sparse_flash_mla_batch_consistency.py` | 从 SMLA PT 收集模式 1-4，处理 opt-in 与状态分类 |
| `test_batch_consistency_unit.py` | 四模式、双格式、oracle、MD5 和 FA/FD analyzer 的 CPU synthetic 回归 |
| `batch/__init__.py` | 保留 batch package 边界，不在导入时触发 NPU 调用 |

MQSMLA `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/`：

| 文件 | 职责/本轮变化 |
| --- | --- |
| `README.md` | 保留原 MQ 使用说明并补充 operator-owned consistency 设计与命令 |
| `pytest.ini` | 保留 ci/graph marker，新增默认关闭的 `consistency` marker |
| `test_run.sh` | 保留 single/batch_save/batch_exec/batch_exec_graph/batch；新增 consistency 与模式参数 |
| `mixed_quant_sparse_flash_mla_paramset.py` | 原 MQ 单用例参数入口 |
| `mixed_quant_sparse_flash_mla_golden.py` | 原量化 CPU golden 和输入生成 |
| `check_valid_param.py`、`conftest.py` | 原参数合法性与 pytest 配置 |
| `result_compare_method.py`、`utils.py` | 原数值精度、结果记录与通用工具 |
| `perf_run.sh`、`process_perf_data.py` | 原性能执行与数据处理工具，保持不变 |
| `test_mixed_quant_sparse_flash_mla_single.py` | 原 MQ single 流程副本 |
| `test_mixed_quant_sparse_flash_mla_batch.py` | 原 MQ PT batch/load 流程副本 |
| `test_mixed_quant_sparse_flash_mla_batch_graph.py` | 原 MQ graph/load 流程副本；一致性 graph 尚未接入 |
| `batch/test_mixed_quant_sparse_flash_mla_pt_save.py` | 原 Excel -> golden -> PT 保存流程 |
| `batch/mixed_quant_sparse_flash_mla_process.py` | 原 MQ metadata+主算子；新增可选 `metadata_sink`，保持 `(npu_result,cpu_output)` 返回契约 |
| `batch/mixed_quant_sparse_flash_mla_consistency.py` | MQ 专用 executor，真实调用 metadata+MQ 主算子，不借用 SMLA executor |
| `batch/mixed_quant_sparse_flash_mla_metadata_probe.py` | PR8737 metadata-only 临时验证，恢复旧 level 并输出白盒 JSON |
| `test_mixed_quant_sparse_flash_mla_batch_consistency.py` | 从 MQ PT 收集模式 1-4，处理 opt-in 与状态分类 |
| `test_batch_consistency_unit.py` | 与 SMLA 同源的 CPU 门禁，保证两份辅助实现没有漂移 |
| `batch/__init__.py` | MQ batch package 边界 |

### 四种模式的最终实现分析

| 模式 | 正式配置 | 派生调用 | 等价性措施 | 验收 |
| --- | --- | --- | --- | --- |
| `reorder` | `--modes reorder --seed N` | 同 B，非恒等 batch order | Q/KV/table/长度/index/top-k 同步重排；TND 重算 prefix sum | 按 origin 还原后 MD5 + CPU 精度 + metadata |
| `split` | `--modes split --split-groups '0,1,2;3,4,5,6,7'` | `B=8` 基准对 `B=3`、`B=5` | 每个 group 是测试侧独立 case；显式 group 优先，seed 只用于 `--randomize` | 每个 subcase 直接对基准对应 batch；完整 merge 仅诊断 |
| `token-split` | `--modes token-split --token-split-sizes '4096,12288'` | 一个 16K batch 对 4K、12K 两个完整分段 | 强制完整分区；位置 mask 用 `old_S2-old_Q+end`；逐 token actual-input oracle | 每段直接按 origin 对基准，任一段失败则模式失败 |
| `shape-change` | `--modes shape-change --common-tokens 300 --derived-extra 200` | `300+100` 基准对 `300+200` 派生 | 共同前缀 Q 不变；位置 mask 用 `old_S2+new_Q-old_Q`；cmp/PA 容量同步校验 | 只比较共同 300 token，额外 token 不进入 digest |

`--modes` 和 `--profile` 是正式接口；`-M 1,2` 已不是长期设计，只在 SMLA 旧入口下游迁移期
映射为命名模式并发出 deprecated warning。无需等待 batch 一致性全部打通再改接口，后续新增
STC manifest 或 nightly profile 也只扩展名字，不继续增加含义不明的数字。

### 白盒自动分析的实现边界

当前 analyzer 会：

1. 解析 FA `36x9`、FD `72x8` 和 reserved tail，按 uint32 bit pattern 解释负 int32；
2. 用 `bn2 -> (batch,n2)`，并依据 PR8737 当前 `mBaseSize=G` 将 `m_index` 展开成 query token；
3. 根据每个 token 的 ori/cmp 可见 S2 计算
   `R=align_up(ceil(actual_S2/32),128)` 和规约块总数；
4. 展开 FA `[start,end)`，合并 split-G 的重复范围，避免把镜像 record 误判成跨核；
5. 关联 FD workspace，检查 `workspace_num` 是否等于该 token 的规约块总数，并检查 G 行覆盖；
6. 输出 token 视角、core 视角、Level-1 四类、Level-2 目标、缺失项和结构错误。

它不能凭 level-0 metadata 推断 level-3 FD 一定产生，也不能替代主算子 MD5。真实 coverage
验收必须使用与主算子同一次调用实际消费的 level-3 metadata。

### 当前验证状态与下一阶段

本轮 CPU/静态门禁：两算子 `pytest_v2` 的一致性单测各 `20 passed`；两个 opt-in NPU 入口在
默认关闭时各 `4 skipped`；新增一致性源码 ruff、两份 shell `bash -n` 和两棵目录 compileall
均通过；原 single/batch/graph 在 SMLA 收集 15 项、MQSMLA 收集 3 项。

下一阶段按以下顺序推进：

1. 对齐支持 deterministic level 3 的 torch_npu/CANN、SoC 和 PR8737/custom 包版本；
2. 在该环境分别运行 SMLA/MQSMLA eager 模式 1-4，保存 metadata digest、MD5 和失败 origin；
3. 把 STC 001-027 修订成 versioned PT namespace 或 JSON manifest，修正 014/026 数据问题；
4. 接入 SMLA/MQSMLA graph executor/capture，再实现 ACLNN STC 021；
5. 在真实 level-3 metadata 上补齐 Level-1/2 覆盖矩阵和 N=3/N=125、TND、PA_BBND nightly。

在第 1 项完成前，任何 level 0 的普通精度 PASS 都只记录为环境 smoke，不得写成 batch
consistency PASS。


# 检视意见二

cannbot模型不止glm-5.2还有deepseek v4和qwen3.7-max，我希望chatgpt能和多方合作沟通商量规划，毕竟别的模型有可能关注到chatgpt没关注到的点，实现的时候chatgpt可以指导其他模型帮你修改一下

做好开发日志记录和文档记录

1. 关于上一个检视的分核和规约等事，我挺好奇一点，模式1~4究竟是pytest框架来做输入前的处理？还是说让NPU做分核规约的事？我搞不懂这究竟是哪一侧的任务（算子开发/pytest框架适配），如果全交给pytest做参数参数处理，那还有算子开发什么事？请结合当前ops-transformer_0716/requs/docs分析一下，算子开发/pytest框架适配分别做什么。就像你上一个检视实现的`### 四种模式的最终实现分析`关于1~4分核规约的问题全部交给pytest参数控制，那问题来了，需求文档中算子开发干了啥？我们设置的参数能传入NPU侧？

2. `### 需求冲突集中台账`写详细一点，或者单独列出一个文档，比如STC有冲突，就列出哪一条用例冲突了什么，后面我好找算子开发那边的人对齐，模式4冲突，就详细说明具体问题（哪个资料指导了什么，和其他文档冲突，mask详细冲突问题我建议仔细说说，这和后期可能需要核对，mask怎么实现的，右下角计算会和当前有什么冲突）

3. 当前由于MQSMLA的pytest侧没适配batch一致性，只有pytest_v2做了适配，当前代码实现应该切到pytest_v2为主了，同时`level3两段式完整调用示例.md`我并不满意，关于上一个检视的意见2我提的是用例侧如何调用batch一致性（由于改了metadata，我想知道用例参数会不会有什么变动，没有就直说，有就模式1~4分别需要什么样的用例）

4. 关于pytest适配batch一致性绘一张流程图，我实在不理解，为什么batch一致性会不需要改动golden和输入pt保存，讲解一下，我理解一直是batch一致性是CPU golden的一个特性，是用例集参数的变动，但我看你的修改好像需求是框架侧的问题

5. SMLA和MQSMLA算子虽然NPU侧的算子开发还没有实现，但输入处理、CPU golden生成、pt文件保存一直都是框架独立支持的功能（除了compare需要NPU结果），我觉得关于batch一致性可以生成excel用例表单独验下输入处理、CPU golden生成、pt文件保存的功能。对了有些临时验证的文件别提交上去了，记得及时删除（excel文件可以保留），| `batch/mixed_quant_sparse_flash_mla_metadata_probe.py` | PR8737 metadata-only 临时验证，恢复旧 level 并输出白盒 JSON |这种不是真正的适配代码都放进batch中提交上去不合理，有些可以创一个临时目录去放临时测试文件

6. 关于`### 当前验证状态与下一阶段`的NPU算子开发侧实现也不急，这个只能等，先看看`batch一致性测试框架.md`文档的大纲里面还有什么能实现的

7. 上一个检视的意见6生成的SMLA_MQSMLA入门教学.md我觉得还是不够详细，对于列出的很多字段关系我还是有点不太理解，能生成更多的关系图逻辑图以及思维导图辅助理解吗？（因为我原来本就不是底层transform的开发人员，属于一个小白接触这些）

8. 白盒分析带上原有输入，列出详细信息。当前batch一致性的实现文件还是有点杂乱，在整理验证一下


## 检视意见二处理记录（2026-07-20）

### 一、处理结论

本轮没有等待NPU算子/环境，先把所有可离线闭环的工作推进完：

1. 明确pytest与算子侧的责任边界，并重写用例调用说明；
2. 建立逐STC、模式4、mask和PR公式的独立冲突文档；
3. 正式只在两棵`pytest_v2`推进新功能，整理`batch/consistency/`子包；
4. 删除MQSMLA metadata-only临时probe；
5. 新增双算子Excel→原CPU golden→baseline PT→模式1-4→derived PT的无NPU验收；
6. 白盒报告增加原始输入来源、参数、tensor shape/dtype/device/byte/digest/sample；
7. analyzer同时报告PR floor与需求ceil公式，避免静默掩盖冲突；
8. 扩充小白教学并增加5张关系/流程/模式/mask SVG；
9. 发现并修复MQSMLA FP8相同NaN payload被`torch.equal`误判不一致的问题。

当前准确状态：**两算子输入/golden/PT/模式变换的CPU-only链已真实通过；真实NPU level3仍是
`UNSUPPORTED`，没有被包装成PASS。**

### 二、8条检视意见逐项答复

| 编号 | 回答、修复与实现 | 证据/状态 |
| --- | --- | --- |
| 1 | 四种模式是pytest构造baseline/derived多次调用；metadata tiling、固定规约块、FA/FD分核、workspace和kernel combine仍由算子完成。模式不会作为op参数传入，变化后的原Q/KV/length/table/index参数会正常进入metadata/main op；level3通过进程状态被runtime/tiling读取 | `level3两段式完整调用示例.md`、`images/responsibility_boundary.svg`；职责已澄清 |
| 2 | 新建独立冲突清单，逐条列出STC 001-027原文/字段/问题/修订项/待确认角色；模式4单列主需求、会议prepend、STC尾部追加三种说法，并推导RightDown/Band threshold/start/end；新增PR floor/需求ceil冲突 | `SMLA_MQSMLA需求冲突与对齐清单.md`；待owner确认项未伪装成已裁决 |
| 3 | 新开发主线正式指向`tests/pytest_v2`。重写两段调用文档，开头直接回答“operator API无新增mode参数”；逐模式列baseline最低条件、derived调用、同步字段和命令。原Excel/PT只需baseline，derived由pytest生成，避免generator二次随机 | `level3两段式完整调用示例.md`、两棵README；已完成 |
| 4 | 新增完整流程图，并解释golden/PT为何不加mode字段：golden定义一次具体输入；一致性新增的是多次调用间的TokenOrigin关系。baseline复用原golden；derived只从baseline CPU output物化有origin的比较token，模式4额外token明确不冒充完整golden | `images/batch_consistency_flow.svg`、调用文档第5节；已完成 |
| 5 | 新增正式Excel和`consistency_prepare`命令，双算子真实跑通原parser、原golden、baseline PT save/reload、模式1-4、oracle、derived PT save/reload、schema/digest。临时MQ probe已删除 | `cases/SMLA_MQSMLA_batch一致性离线预检.xlsx`、两棵`batch/*_consistency_prepare.py`；已通过 |
| 6 | 对照框架大纲逐项标记无NPU可做/已做/仍依赖NPU；完成输入表、golden、PT、黑盒、random、输入报告和白盒算法部分。NPU优先级下调，先关闭公式/STC/manifest/MQ LSE等离线问题 | `docs/batch一致性测试框架.md`、开发计划第10节 |
| 7 | 教学增加字段关系思维导图、职责边界、完整流程、四模式TokenOrigin映射和RightDown/Band三方案对比；新增物理/有效长度、手算例、报告读法、文件导航和小白自检 | `SMLA_MQSMLA入门教学.md`及`images/*.svg`；已完成 |
| 8 | 白盒`input_summary`带Excel/sheet/PT来源、baseline/derived、B/S/N/D/G/layout/mask、所有输入tensor的完整digest和有界sample；控制tensor给完整values。公共文件移入`batch/consistency/`，operator process/executor/prepare留顶层；双树核心增加字节级镜像门禁 | `batch/consistency/reporting.py`、整理后的文件树和unit test；已完成 |

### 三、算子开发与pytest适配的最终责任边界

```text
pytest_v2：
  Excel/PT -> baseline输入 -> modes 1-4派生输入 -> 多次metadata+main调用
  -> TokenOrigin对齐 -> NPU MD5 -> CPU精度 -> metadata/input报告

runtime：
  保存/设置/读取/恢复 deterministic level 3

算子 metadata/tiling：
  根据“本次调用”的B/S1/S2/mask/length计算规约tile、负载与FA/FD任务

算子 kernel：
  执行FA局部结果、同核固定combine、跨核workspace与FD固定combine
```

所以算子开发不是“被pytest替代”。pytest只制造会改变调度但应保持数学等价的调用组合，并验证
结果；如果算子没有level3固定规约实现，pytest会得到真实不一致或`UNSUPPORTED`。pytest配置的
`split_groups`不会传入NPU，NPU收到的是每个derived调用正常的B、tensor和length等已有参数。

### 四、golden与PT的边界

本轮保留原golden/PT schema，不是因为batch一致性属于“纯框架、不需要CPU”，而是因为职责不同：

1. 原golden已经为baseline的具体Q/KV计算`cpu_output`；
2. batch一致性要证明mapped token的实际数学输入未变，因此它的CPU预期就是baseline中对应
   `TokenOrigin`的output，而不是再次随机生成另一份输入；
3. reorder/split/token-split可为所有derived有效token物化expected output；
4. shape-change只为共同前缀物化expected，额外token写入`unmapped_tokens`且不进入比较；
5. 若未来要求验证shape-change额外token，必须让原golden支持“对给定现成tensor重新计算”，
   本轮没有用零模板假装这项能力。

离线prepare因此验证的是“原baseline golden + 一致性comparison golden scope”，不是NPU结果。

### 五、多模型审议记录与主审裁决

本轮沿用隔离CANNBot会话对`cannbot/glm-5.2`、`cannbot/deepseek-v4-pro`、
`cannbot/qwen3.7-max`的独立只读审议。认证和临时数据未进入仓库；模型不直接修改代码，主审按
源码、需求和可重复测试裁决。

三方有效共识：

- 明确pytest/算子边界；
- 用Excel→golden→PT做无NPU验收；
- 删除正式`batch/`中的metadata-only临时probe；
- 白盒报告带tensor shape/dtype/digest/PT来源；
- 整理双树重复文件并加镜像一致性门禁。

模型意见中已纠正的错误：

1. DeepSeek/Qwen说oracle没检查mask window；实际`ActualInputSemanticOracle._windows()`和可见KV
   slice检查已存在，本轮是把证据写进报告和教学；
2. DeepSeek/Qwen把prepend一律判错；实际Q位置同步右移时threshold可保持，冲突在于不符合
   主需求“共同前缀”；
3. GLM怀疑visible ori+cmp与PR不符；PR head实际正用`actOriS2Size+actCmpS2Size`，真正冲突是
   C++ floor与需求ceil；
4. 多模型都未提前指出FP8 NaN bit比较问题；真实MQ离线链暴露后由主审定位并修复。

### 六、本轮代码实现与文件职责

#### 6.1 双算子镜像子包

以下文件在SMLA/MQSMLA两棵`tests/pytest_v2/batch/consistency/`逐字镜像，各算子独立拥有，
没有跨算子import：

| 文件 | 职责 |
| --- | --- |
| `adapter.py` | 适配SMLA `input`与MQSMLA `op_input`及metadata字段名 |
| `model.py` | `TokenOrigin`、`ConsistencyCase`、`RunResult`数据模型 |
| `layouts.py` | BSND/TND/PA的batch/token选择、prefix sum和block table原语 |
| `schema.py` | PT schema、物理/有效长度和派生invariant校验 |
| `transform.py` | 模式1-4、mask对齐、actual-input oracle；FP8按存储bit比较 |
| `compare.py` | origin对齐、有效token打包、MD5和原`check_result`复用 |
| `config.py` | 命名mode/profile和CLI groups/sizes解析；数字别名仅deprecated兼容 |
| `flow.py` | level3 guard、baseline/derived、MD5/CPU/metadata三重门禁和input summary |
| `metadata_layout.py` | 严格解析1024-int32、FA36x9、FD72x8、unknown/reserved |
| `metadata_coverage.py` | token/core/S2/workspace展开、Level-1/2和floor/ceil公式审计 |
| `reporting.py` | provenance、参数和所有tensor的shape/dtype/device/byte/MD5/SHA256/sample |
| `offline.py` | Excel辅助配置解析、mapped CPU expected、derived PT save/reload报告 |
| `__init__.py` | 明确子包边界，不在import时触发NPU |

#### 6.2 SMLA operator-owned文件

| 文件 | 本轮作用/变化 |
| --- | --- |
| `batch/sparse_flash_mla_process.py` | 保留原metadata+main op，仍由SMLA executor调用 |
| `batch/sparse_flash_mla_consistency.py` | SMLA专用eager executor，捕获同次metadata |
| `batch/sparse_flash_mla_pt_save.py` | 原baseline PT保存，不改schema |
| `batch/sparse_flash_mla_consistency_prepare.py` | 新增SMLA Excel→原golden→PT CPU-only CLI |
| `sparse_flash_mla_golden.py` | `SMLA_CPU_ONLY=1`时跳过Ascend950设备属性/stride分支，使无NPU生成可重复 |
| `test_sparse_flash_mla_batch_consistency.py` | PT加载时记录source path，模式1-4正式opt-in入口 |
| `test_batch_consistency_unit.py` | 新增input report、公式、FP8 bit、离线配置和镜像门禁测试 |
| `test_run.sh` | 新增`consistency_prepare`命令，原single/save/load/load_graph不变 |
| `README.md` | 更新子包结构、离线/NPU命令和状态边界 |

#### 6.3 MQSMLA operator-owned文件

| 文件 | 本轮作用/变化 |
| --- | --- |
| `batch/mixed_quant_sparse_flash_mla_process.py` | 保留原metadata+main op及旧返回契约 |
| `batch/mixed_quant_sparse_flash_mla_consistency.py` | MQ专用executor，不借用SMLA调用 |
| `batch/test_mixed_quant_sparse_flash_mla_pt_save.py` | 原Excel→量化golden→baseline PT流程 |
| `batch/mixed_quant_sparse_flash_mla_consistency_prepare.py` | 新增MQ Excel→原量化golden→PT CPU-only CLI |
| `batch/mixed_quant_sparse_flash_mla_metadata_probe.py` | **已删除**；它是临时探索，不属于正式适配 |
| `test_mixed_quant_sparse_flash_mla_batch_consistency.py` | PT加载时记录source path，模式1-4正式opt-in入口 |
| `test_batch_consistency_unit.py` | 与SMLA同源的CPU与镜像门禁 |
| `test_run.sh` | 新增`consistency_prepare`命令，原MQ命令不变 |
| `README.md` | 删除probe说明，补正式子包、离线链和状态边界 |

#### 6.4 需求、教学和验收资料

| 文件 | 作用 |
| --- | --- |
| `cases/SMLA_MQSMLA_batch一致性离线预检.xlsx` | SMLA/MQSMLA各1个小shape baseline及说明sheet；不替代调度STC |
| `SMLA_MQSMLA需求冲突与对齐清单.md` | 逐STC 001-027、模式4、mask、PR公式和待确认人 |
| `level3两段式完整调用示例.md` | 重写为用例参数、四模式调用和参数到NPU的说明 |
| `SMLA_MQSMLA入门教学.md` | 扩写字段、例子、报告和文件导航 |
| `docs/batch一致性测试框架.md` | 标记原大纲各模块的离线可做性和完成状态 |
| `SMLA_batch一致性测试开发计划.md` | 调低NPU等待优先级，增加离线/冲突/报告里程碑 |
| `images/responsibility_boundary.svg` | pytest/runtime/算子责任边界 |
| `images/batch_consistency_flow.svg` | Excel/golden/PT/baseline/derived/NPU/compare完整流程 |
| `images/dimension_layout_mindmap.svg` | B/S/N/D/T、length、layout、metadata关系思维导图 |
| `images/four_modes_mapping.svg` | 四模式调用与TokenOrigin映射 |
| `images/rightdown_band_mask_alignment.svg` | 尾部只加Q、prepend、Q+KV同步增长的mask对比 |

### 七、离线验收实测

使用正式workbook和两棵`test_run.sh consistency_prepare`真实运行：

| 算子 | baseline | derived调用 | 覆盖 | 结果 |
| --- | ---: | ---: | --- | --- |
| SMLA | 1 | 6 | reorder 1、split 2、token-split 2、shape-change 1 | 全部schema有效，CPU output PT round-trip MD5一致 |
| MQSMLA | 1 | 6 | reorder 1、split 2、token-split 2、shape-change 1 | 全部schema有效，CPU output PT round-trip MD5一致 |

MQ首次运行暴露FP8打包KV中相同NaN bit被`torch.equal`判不等。修复为shape/dtype一致后比较完整
uint8存储bit，再次运行通过。这也证明离线链不是只做mock单测，而是真正调用了原量化golden。

报告同时包含baseline/derived PT绝对来源、Excel/sheet、完整tensor digest和模式4golden scope。
生成的PT/JSON放在`/tmp`验证目录，未提交；只提交可复现的Excel。

### 八、PR公式的新阻断

PR8737 head当前表达式：

```text
PR:          align_up(floor(visible_S2/32), 128)
requirement: align_up(ceil (visible_S2/32), 128)
```

4097会得到128/256两种不同tile；1..31的PR表达式为0。analyzer现在输出两套值和冲突字段，
结构检查暂按PR head且对0使用base-size保护。此处理只保证工具不会崩溃，不代表需求已裁决。
需要tiling owner明确：是否源码少了`+31`、是否有前置clamp，以及白盒STC应以哪一式为准。

### 九、当前状态与后续顺序

已完成且不依赖NPU：

- 双算子pytest_v2模式1-4与三种layout变换基础；
- actual-input oracle、origin比较、CPU精度/MD5编排；
- Excel/golden/baseline+derived PT离线链；
- 原始输入白盒摘要、metadata parser/analyzer和公式冲突报告；
- 目录整理、镜像门禁、冲突/教学/流程文档。

下一步先做，不等NPU：

1. 找owner关闭floor/ceil/zero edge；
2. 修STC 014/015、018-020、024-027及公共缺字段；
3. 选择versioned JSON manifest或PT namespace，把001-027机器化；
4. 为TND、PA、position mask增加更多小shape离线表；
5. MQ process保留LSE且MQ CPU golden生成LSE后，再启用`return_softmax_lse=True`；
6. 完成graph/ACLNN调用边界设计，但不在eager level3前虚报验收。

环境/算子可用后再做：

1. SMLA/MQSMLA各一个真实level3 smoke；
2. eager模式1-4的NPU baseline/derived MD5+CPU精度+同次metadata；
3. 真实Level-1/2 coverage matrix；
4. graph、ACLNN、N=3/N=125、TND/PA nightly。

当前`-M 1,2`仍只作为SMLA旧入口迁移兼容并发deprecated warning；正式接口已经是
`--modes reorder,split,...`或`--profile`，不会等NPU打通后才迁移，也不会继续给数字增加新含义。

# 检视意见三

cannbot模型不止glm-5.2还有deepseek v4和qwen3.7-max，我希望chatgpt能和多方合作沟通商量规划，毕竟别的模型有可能关注到chatgpt没关注到的点，实现的时候chatgpt可以指导其他模型帮你修改一下

做好开发日志记录和文档记录

1. NPU侧算子开发还要继续等

2. 责任田划分我大致理解了，现在ops-transformer_0716/requs/data目录下有另一个开发者基于另一个测试框架做的batch 一致性的一些测试方案和实现，我觉得有一些可取可借鉴的东西，评估一下呢？如果觉得这设计不错就可以采取并实现

3. 我之前看过你做的`### 四种模式的最终实现分析`，在正式配置一项中，我知道实际上模式指定时还需要手动设置一些配置参数，但我觉得这种方式不太好，对于多个配置参数是不是写进文件中，需要的时候修改文件更好？而且batch批跑是是不是加载同一套配置，同一套配置是否适用于所有用例？这都是值得考虑，有时候我给你举的例子但不一定是普遍情况，ops-transformer_0716/requs/data目录下对这种是怎么处理的呢

4. 当前pytest适配batch 一致性测试需求，输出的结果是怎么的呢，结果文件是怎么统计的呢

5. ops-transformer_0716/requs/data认为
```
batch一致性功能测试分成两种场景：
（1）不同用例间的batch/seq比较（必须包含），验证跨case输出一致性
（2）同一用例内的batch/seq比较，验证同一case内不同batch/seq位置输出标识正确性
```
这个和当前有冲突吗，没冲突适配的怎样

6. 测试场景：打桩metadata  
例如某一个Shape  
按照负载均衡计算，某一个规约任务，在单核完成  

打桩metadata，让规约任务，在多核执行

## 检视意见三处理记录（2026-07-21）

### 一、直接答复

1. **NPU侧继续等待。** 本轮没有修改SMLA/MQSMLA host、tiling、kernel或已安装custom包，也没有
   用level0结果替代level3。`RT-1/OP-1/PKG-1/FORMULA-1`状态不变：支持level3的runtime、主算子
   合入版本、PR floor/需求ceil仍需对应owner处理。
2. **外部方案有可取之处，但不能原样接入。** 采纳“显式输出关系、执行前预检、分层覆盖、
   min/mid/max位置诊断”四个思想；不采纳seed重生成、单batch_axis切片、output_seed数字关系键、
   固定500/200/200数量和原比较脚本。
3. **已把采纳项实现到pytest_v2主线。** 双算子新增versioned relation manifest v1，继续以
   `TokenOrigin`为唯一比较身份；`consistency_prepare`真实生成、校验并汇总关系和CPU输入覆盖。
4. **开发计划已推进但没有虚报完成。** relation schema和自动派生manifest已完成，P3.4从“未
   实现”推进到“schema/自动artifact完成，STC 001-027 source entries待owner修订”；真实metadata
   coverage仍等level3。

外部资料和逐项代码审计见`data/README.md`。`data/算子多batch一致验证.md`、
`data/compare_batch_tensor_by_result.py`和`data/aclnnAdd.xlsx`作为用户提供的原始输入保持不改，
避免评估过程中反向篡改证据。

### 二、外部方案到底解决什么

外部aclnnfuzz方案通过`tensor_input_seed_info`让某些tensor轴切片由相同seed/range生成，再通过
`tensor_output_seed_info.output_seed`把“应相同”的输出切片放进同一比较组。这个思路适合
逐元素、reduce、broadcast等能用普通轴切片描述依赖关系的通用算子。

SMLA/MQSMLA的一致query token却同时依赖Q、KV、有效长度、TND prefix sum、PA table、mask、
sparse/top-k、cmp residual和MQ量化参数。只给Q或某个KV切片同seed，不能证明attention数学输入
相同。因此当前“baseline PT -> 同步变换全部相关字段 -> actual-input oracle”的可靠性高于重新
生成随机切片，不能为了复用外部字段而退回较弱模型。

外部文档的level0/level1/level2还是**用例分类**，不是CANN deterministic level，也不与模式
1-4一一对应。本项目四模式都是baseline与一个或多个derived的多次完整调用：

| 外部思想 | 本项目裁决 |
| --- | --- |
| input seed固定切片 | 不接入；直接复用baseline真实tensor并同步全部语义字段 |
| output seed标记相等组 | 升级为`baseline_case_id + TokenOrigin spans` |
| 跨case/单case/边界分层 | 改为mode/layout/mask/visible-S2/source-position coverage inventory |
| min/mid/max位置 | 作为覆盖诊断，不作为数值PASS |
| Excel预检 | 正式parser拒绝数字mode，manifest校验origin/scope/oracle/digest |
| raw binary相等 | 已有tensor原始byte MD5 + 原CPU精度双门禁，不复制numpy脚本 |

### 三、外部实现审计结论

`compare_batch_tensor_by_result.py`不能直接进入生产链，关键证据如下：

1. 文档的input seed info是6元素，脚本两处判断`len==5`后又访问索引5。合法6元素会被静默
   跳过，5元素才会进入并越界，input格式预检实际上失效；
2. CSV/Excel attrs、shape、dtype和动态slice多处直接`eval`，存在代码执行风险；
3. 没有完整检查负index、axis、pre-index数量、重复项、bin byte size以及“同seed必须同
   range/dtype/slice shape”的文档约束；
4. output分组键没有operator/relation id，seed复用可能把无关case混组；
5. 没有TokenOrigin、有效token/padding、LSE、原精度、metadata和两段式调用语义；
6. 示例Excel只有`test` sheet和8条aclnnAdd数据，不是文档声称的level0/1/2完整用例集。

脚本把fp32/bf16映射成同位宽unsigned读取，在“按原始存储bit比较”前提下并非天然错误；真正
缺口是没有声明raw-bit契约、endianness/byte-size/FP8校验，也没有数值精度门禁。该判断修正了
DeepSeek把`float32 -> uint32`本身视为错误的意见。

### 四、本轮实现的数据流

```text
Excel canonical mode配置
  -> 原golden生成baseline PT
  -> mode 1-4派生完整输入 + semantic oracle
  -> origin-mapped CPU expected scope
  -> derived PT save/reload + provenance fingerprint + content fingerprint
  -> TokenOrigin压缩为连续source/target spans
  -> manifest边界/重复/scope/oracle/span digest预检
  -> mode/layout/mask/visible-S2/min-mid-max coverage inventory
  -> CPU-only PASS（不触碰NPU，不声称batch consistency PASS）
```

每个baseline新增`consistency_relation_manifest.json`；每个算子汇总新增
`smla_consistency_relation_manifest.json`或`mqsmla_consistency_relation_manifest.json`。
manifest使用`schema_version=1`和固定`kind`，baseline id绑定去掉临时路径后的输入内容SHA256；
来源Excel/PT绝对路径仍作为provenance单独保留。这样同一输入换临时目录不会改变身份，而PT内容
变化会改变baseline id。

TokenOrigin逐token列表会让16K用例artifact膨胀，因此manifest只保存连续span：

```json
{
  "target_start": 0,
  "target_end": 4096,
  "source_batch": 0,
  "source_token_start": 0,
  "source_token_end": 4096
}
```

校验器保证target spans从0连续、源/目标宽度相同、源token不重复且不越baseline有效长度，
`mapped_token_count`与`_consistency_expected_scope.compared_q_lengths`一致。模式4新增但没有origin
的token仍只记为`unmapped_tokens`，不会被伪装成golden。

### 五、配置预检的变化

Excel的`consistency_modes`现在必须使用`reorder/split/token-split/shape-change`正式名称；数字
`1,2`直接报配置错误。`split_groups`和`token_split_sizes`复用正式`CaseConfigParser`，不再由
offline模块维护一份较弱的字符串拆分逻辑；负`mode_batch`、非正`common_tokens`和负
`derived_extra`也会在生成前拒绝。

这不改变CLI迁移策略：旧SMLA入口的`-M 1,2`仍可deprecated兼容，但新Excel和relation
manifest不再接受数字别名。

### 六、多模型联合审议与主审裁决

本轮为同一份检视意见、外部三文件、主需求、开发计划和pytest_v2建立三个隔离CANNBot会话：

| 模型 | 关注点 | 有效贡献 |
| --- | --- | --- |
| `cannbot/glm-5.2` | 架构复用和最小增量 | 建议只借鉴预检/覆盖，不另起seed框架 |
| `cannbot/deepseek-v4-pro` | 安全与manifest反审 | 找到`eval`、6/5元素矛盾，建议versioned关系和coverage artifact |
| `cannbot/qwen3.7-max` | 模式、layout、mask和边界 | 提醒PA容量、cmp residual、TND prefix、S2<32和PT digest绑定 |

CANNBot 1.0.5默认数据库因缺少`replacement_seq`列启动失败。本轮没有删除、迁移或直接修改用户
数据库；按既有安全约束使用0700隔离`XDG_DATA_HOME`并只复制0600认证文件，三模型均只读。
模型没有并发写仓库，避免不可追踪的共享工作树覆盖；主审根据它们的建议实现并承担最终裁决。

除前述dtype误读外，主审还纠正：外部`output_seed`只是关系标签，不表示attention输出等于seed
之和；外部level分类与四模式无一一对应；mode 1也不是“两个预生成case”的专用模式，而是从
同一baseline派生重排调用。

### 七、文件级开发记录

#### 7.1 双算子镜像核心

| 文件 | 本轮作用/变化 |
| --- | --- |
| `batch/consistency/relation_manifest.py` | 新增manifest v1、stable baseline id、origin span压缩/校验、覆盖汇总与多baseline合并 |
| `batch/consistency/offline.py` | 正式parser预检；baseline/derived provenance和content双指纹；保存per-case relation manifest |
| `batch/consistency/reporting.py` | 新增不受临时来源路径影响的`input_content_digest` |
| `test_batch_consistency_unit.py` | 新增canonical Excel、6条relation、coverage、越界origin、digest篡改和路径无关内容指纹回归；镜像门禁纳入新模块 |

#### 7.2 operator-owned文件

| 文件 | 本轮作用/变化 |
| --- | --- |
| SMLA `batch/sparse_flash_mla_consistency_prepare.py` | 合并per-case manifests，输出算子级relation/coverage artifact |
| MQSMLA `batch/mixed_quant_sparse_flash_mla_consistency_prepare.py` | MQ独立完成相同汇总，不借用SMLA wrapper |
| 两棵`README.md` | 增加manifest职责、输出文件和CPU-only claim边界 |

#### 7.3 需求与计划文档

| 文件 | 本轮作用/变化 |
| --- | --- |
| `data/README.md` | 外部三文件逐项评估、缺陷、采纳/拒绝表、多模型结论和后续接口 |
| `docs/batch一致性测试框架.md` | 在历史大纲后追加relation manifest与真实metadata coverage边界 |
| `SMLA_batch一致性测试开发计划.md` | 更新P2.13、P3.4/P3.5状态、测试矩阵和STC source manifest路线 |
| `requs.md` | 保留原检视意见三并追加本处理记录 |

用户新增的`data/`三份原始资料一并纳入版本管理，但原内容不改；新增README明确它们是
reference-only，防止后续维护者误把有`eval`和预检bug的脚本放进CI。

### 八、当前验证状态

本轮已在加载`/mnt/workspace/gitCode/env.sh`后完成：

- SMLA CPU unit：`26 passed`；
- MQSMLA CPU unit：`26 passed`；
- 双算子离线prepare：各1个baseline、6个derived、6条relation，round-trip与manifest
  precheck均通过；
- 算子级manifest：四个mode均有relation，`missing_modes=[]`；
- 输入覆盖缺口被如实保留：SMLA缺TND Q、TND/PA KV和position-sensitive mask；MQSMLA缺
  TND Q、BSND/TND KV和no-mask。这些是后续workbook待补项，不影响当前小shape关系预检，
  也没有被标成NPU FAIL；
- 新增源码ruff、双树字节镜像、compileall、原用例收集、默认opt-in skip和shell语法通过；
  正式实现及新增说明文档通过`git diff --check`。`data/`三份原始输入按收到时字节保留，
  其中参考脚本只做内存`compile()`，不进入lint或CI执行；
- 临时PT/JSON、CANNBot认证和日志均未进入仓库。

真实NPU状态仍为：

```text
SMLA level-3 metadata + main op: UNSUPPORTED
MQSMLA level-3 metadata + main op: UNSUPPORTED
原因：当前runtime只接受deterministic level 0/1/2
```

### 九、开发计划继续推进顺序

不等NPU，按顺序继续：

1. 用relation manifest v1定义STC source entry状态，但只录入无冲突项；
2. 由owner修正014/015、018-020、024-027及001-020公共字段，尤其026的B/cu_seqlens冲突；
3. 为SMLA/MQSMLA各补TND、PA_BBND、RightDown/Band小shape workbook，关闭CPU relation
   coverage缺口；
4. MQ process保留LSE且MQ CPU golden生成LSE后，再开启MQ LSE关系和精度门禁；
5. 等tiling owner裁决PR floor/需求ceil与`S2<32`zero edge。

环境就绪后再做：双算子真实level3 smoke -> eager四模式 -> 真实metadata Level-1/2 matrix ->
graph -> ACLNN -> N=3/N=125和nightly。relation manifest届时复用同一relation id追加NPU output、
metadata digest和最终状态，不再重新设计case身份。

## 检视意见三新增意见处理记录（2026-07-21）

### 一、对新增意见3-6的直接答复

| 检视意见 | 答复与处理 | 当前状态 |
| --- | --- | --- |
| 多个模式参数是否应写入文件 | 是。新增versioned JSON suite，正式执行使用`--config-file`；模式名、order、groups、token sizes、源batch和shape-change参数均可按PT配置 | 已实现并有双算子示例 |
| 批跑是否给所有PT加载同一套参数 | 只加载同一个suite文档，但**不是把同一组数值套给所有PT**。每个PT按精确selector匹配一项，再结合自身B/Q长度解析和预检 | 已实现；默认未匹配即`CONFIG_ERROR` |
| pytest结果和统计是什么 | 原入口依赖退出码和日志`grep`，会把skip误解成PASS。现在每个`(operator,PT,mode)`写一个JSON，汇总器只读取校验后的JSON | 已替换双算子在线入口 |
| 外部“两种场景”是否冲突 | 不冲突，但也不能说已经全部适配。当前模式1-4是`same-source-cross-call`；外部的独立跨case与单次调用内位置比较是两个补充场景 | 已机器标注现状和缺口；未伪装完成 |
| 单核规约任务打桩成多核 | 已完成CPU-only候选规划器和默认关闭的executor hook；因FD契约、公式、包版本和level3未确认，本轮不生成或上NPU执行伪metadata | 安全骨架完成，真实注入阻塞 |

`-M 1,2`不是“等batch一致性打通后再想办法”的正式方案。现在已经完成控制面迁移：

- versioned JSON和Excel正式字段只接受`reorder/split/token-split/shape-change`；
- `--config-file`是推荐入口；
- `--modes/--profile`仍可用于单一shape的临时运行；
- 数字`-M 1,2`只在旧CLI兼容层转换并发deprecated warning，何时删除取决于下游调用者迁移，
  与NPU功能何时打通无关。

### 二、配置文件如何解决“例子不普遍”

正式示例为`cases/SMLA_MQSMLA_batch一致性执行配置.json`，固定：

```json
{
  "schema_version": 1,
  "kind": "smla_mqsmla_batch_consistency_suite",
  "unmatched_case_policy": "error",
  "defaults": {"seed": 20260716, "randomize": false},
  "cases": [{
    "id": "offline-smla-b2-s4",
    "match": {"operator": "smla", "case_name": "offline_smla_b2_s4"},
    "config": {
      "modes": ["reorder", "split", "token-split", "shape-change"],
      "split_groups": [[0], [1]],
      "token_split_sizes": [2, 2],
      "mode_batch": 0,
      "common_tokens": 3,
      "derived_extra": 2
    }
  }]
}
```

selector只做精确相等，支持`operator/case_name/pt_basename/input_content_fingerprint_sha256`；
同一PT匹配多项会报歧义，默认未匹配也报错。`defaults`只放真正通用的seed、是否randomize等；
case项的`modes`会替换默认`profile`，反之亦然。

每个PT加载后才做shape-aware解析：

- `reorder_order`必须是当前`[0,B)`的非恒等排列；
- `split_groups`必须索引合法、无重复，默认要求完整覆盖当前B；
- 未显式写groups时，只有`B=8`采用需求例子`3+5`，其他B按自身shape确定性二分；
- `token_split_sizes`必须全为正数且总和等于所选batch的有效Q长度；未显式写时按该PT的Q长度
  分半，16K才使用4K+12K；
- `mode_batch/common_tokens/derived_extra`均相对当前PT校验，shape-change目标长度不能与baseline
  相同；
- B=1导致reorder/split天然不适用、Q不足2导致token-split不适用，记
  `NOT_APPLICABLE`；用户显式写越界group或错误token总和才记`CONFIG_ERROR`。

外部aclnnfuzz方案也是把seed/range等字段写在Excel的每一条case上，并非一套`3+5`配置覆盖
所有case；这一点值得借鉴。本项目采用更严格的per-case selector和shape预检，但不复制其
单tensor轴seed模型、`eval`脚本或有缺陷的5/6元素检查。

配置文件与旧详细CLI严格互斥，以下命令会在执行前失败，避免两套优先级悄悄覆盖：

```bash
bash test_run.sh consistency --config-file suite.json --seed 1
```

### 三、结果文件、统计口径和退出语义

每次`test_run.sh consistency`在`--report-dir`下创建独立`YYYYMMDD_HHMMSS_PID`目录：

```text
<run-dir>/
  results/
    <operator>__<pt>__<case>__<fingerprint>__<mode>.json
  consistency_result_summary.json
  pytest.log
```

逐模式JSON包含schema/kind、场景分类、算子、mode、case名、PT绝对路径、输入内容SHA256、最终
生效配置、reason code、执行报告。状态只有：

| 状态 | 含义 |
| --- | --- |
| `PASS` | 报告本身通过，且所有baseline/derived记录的requested和observed level都严格为3 |
| `FAIL` | NPU执行异常、bitwise/CPU精度/metadata integrity失败，或PASS门禁拒绝 |
| `UNSUPPORTED` | runtime/operator明确拒绝level3等必要能力 |
| `INVALID_TRANSFORM` | 该PT不能构造数学等价派生输入 |
| `NOT_APPLICABLE` | mode未选择或shape天然不适用 |
| `CONFIG_ERROR` | suite歧义、未命中、字段、范围或shape配置错误 |

聚合器按status/mode/operator计数，并同时核对`PT数 x 4个pytest mode`总数、唯一PT数和唯一
`(operator,PT,mode)`矩阵。缺文件、多余/异算子旧文件、重复记录、损坏JSON或pytest子进程非零
都进入JSON汇总，不能用“总数刚好相等”或日志文字绕过。算子级结果为：

- `PASS`：至少有真实level3 PASS，无FAIL/不完整/unsupported/invalid；未选择模式的
  `NOT_APPLICABLE`不阻塞；
- `PARTIAL`：有PASS，同时还有`UNSUPPORTED/INVALID_TRANSFORM`；
- `UNSUPPORTED`：没有PASS且环境能力明确不支持；
- `INCOMPLETE`：缺artifact、只有不适用/无效变换等，无法形成验收结论；
- `FAIL`：有`FAIL/CONFIG_ERROR`或pytest进程异常；
- `NO_RESULTS`：没有artifact且没有给出预期case数。

CLI仅在完整`PASS`时返回0，`FAIL`返回1，其余未完成状态返回2。`pytest.log`只用于诊断，不再是
权威统计输入。本机用正式配置实测：SMLA和MQSMLA各1个PT、4个mode均写出
`UNSUPPORTED`，`missing_artifact_count=0`；pytest自身为`4 skipped`也没有被统计成PASS。

### 四、外部两种功能场景与当前四模式的关系

| 场景ID | 定义 | 当前适配 |
| --- | --- | --- |
| `same-source-cross-call` | 一个baseline PT同步派生输入，进行两次或多次完整算子调用 | 模式1-4已实现 |
| `independent-cross-case` | 两个独立编写/生成的PT之间声明等价输出位置 | 未实现；relation schema待扩展 |
| `single-call-intra-case` | 同一次算子调用中两个数学等价位置互相比对 | 未实现；需要显式位置关系和专用case |

所以外部说的两种场景与当前没有语义冲突，缺口在于来源和调用边界不同。不能把模式1叫作
“独立跨case”：它虽然执行baseline和derived两个调用，但derived仍由同一个baseline同步派生；
也不能把模式3叫作“单次调用内比较”：它会执行baseline和多个token分段调用。

relation manifest与在线summary现在都明确写入`same-source-cross-call`，并列出另外两个
`supplemental_gaps`，防止报告读者误判全覆盖。下一步实现顺序是：

1. 为`independent-cross-case`增加versioned relation source entry，要求两个PT各自内容指纹、
   完整TokenOrigin/位置映射及actual-input等价证据；外部`output_seed`只能作为关系标签参考；
2. 新增最小`single-call-intra-case`输入，使同一调用内至少两个位置拥有完全相同Q、可见KV、
   mask/sparse/top-k/sinks语义，再用位置对直接做bitwise+CPU精度比较；
3. 两者各自拥有结果artifact和coverage cell，不并入模式1-4数字含义；
4. 在没有可信用例与映射前保持gap，不用随机碰巧相等的输出冒充覆盖。

### 五、metadata单核到多核打桩的实现边界

新增`metadata_stub.py`的CPU-only planner：

```text
自然metadata + 对应PT上下文
  -> 严格解析FA/FD
  -> 展开到bn2/query-token/S2-block/core
  -> 只选“自然状态由一个逻辑FA核完整覆盖”的token
  -> 要求该token至少有两个S2 block
  -> 给出拟拆成两个非空block区间的候选计划
  -> 输出自然metadata SHA256、公式审计和activation blockers
  -> 不生成、不修改、不执行synthetic metadata
```

双算子executor/process已增加默认`None`的`metadata_transform`注入点；不传时原路径和返回契约
不变，当前CLI也没有开启该hook的参数。正式激活前必须同时关闭这些阻断：

1. `supportFd=false`与目标实现的关系确认；
2. 本地源码、PR head和已安装custom包build id对齐；
3. FD record slot 7、workspace index/num和M范围的部署契约冻结；
4. PR floor/需求ceil及`S2<32`zero edge裁决；
5. runtime和双主算子真实支持deterministic level 3；
6. 注入前后各自通过metadata结构检查，且synthetic结果单独标识，永远不能计作自然metadata覆盖。

这不是拒绝打桩场景，而是先完成可审计的选点和注入接口，避免用错误FD记录越界写workspace，
或让kernel消费一个“看起来多核、实际契约错误”的tensor。

### 六、本轮文件级开发记录

#### 6.1 双算子逐字镜像核心

| 文件 | 本轮作用 |
| --- | --- |
| `batch/consistency/suite_config.py` | suite v1解析、未知字段拒绝、精确selector、defaults/case合并、legacy转换和per-PT shape预检 |
| `batch/consistency/result_artifact.py` | 原子写逐模式JSON、level3 PASS硬门禁、PT/mode矩阵与损坏/重复/异算子artifact核对、状态聚合及CLI |
| `batch/consistency/metadata_stub.py` | 自然单核token候选规划、拟拆分block ranges、安全阻断项及受约束hook helper |
| `batch/consistency/relation_manifest.py` | relation增加`same-source-cross-call`，并机器记录两个补充场景缺口 |
| `test_batch_consistency_unit.py` | 配置匹配/歧义/越界/shape默认、artifact PASS/矩阵/损坏门禁、场景分类、planner/hook回归；镜像门禁纳入三个新模块 |

#### 6.2 operator-owned在线链

| 文件 | 本轮作用 |
| --- | --- |
| 两个`test_*_batch_consistency.py` | 每个PT分别解析配置；所有skip/error/pass在pytest结束前落最终JSON；不再只打印report |
| 两个`test_run.sh` | 新增`--config-file/--report-dir`、每次独立目录、逐PT进程、JSON聚合；删除一致性路径的grep统计 |
| `batch/sparse_flash_mla_consistency.py`与MQ对应文件 | executor接收默认关闭的metadata transform并继续捕获实际消费对象 |
| `batch/sparse_flash_mla_process.py`与MQ对应文件 | metadata op与主算子之间预留受约束hook；默认`None`保持原调用 |
| `cases/SMLA_MQSMLA_batch一致性执行配置.json` | 与现有离线表两个小shape一一对应的正式配置示例 |

本轮仍未修改两算子host、tiling、kernel、torch extension schema或已安装custom包；`data/`原始
三文件也保持字节不变。

### 七、验证结果与阶段推进

- 双算子CPU unit当前各`39 passed`；
- 双算子离线prepare仍各1 baseline、6 derived、6 relation，新增scenario字段通过manifest
  round-trip校验；
- 正式suite对两个真实baseline分别命中`offline-smla-b2-s4`和
  `offline-mqsmla-b2-s4`，生效groups/token sizes/shape-change值与各自shape一致；
- 双算子在线各4条结果JSON、1条summary，无缺失；当前结果均为`UNSUPPORTED`，原因仍为
  runtime只支持level 0/1/2；
- 两棵共享目录ruff、compileall、字节镜像和两份shell语法通过；原始入口与最终完整门禁见本次
  提交前验证记录。

阶段二因此新增完成三项：per-case suite配置、结构化在线结果、metadata stub候选规划；阶段三
仍按“版本对齐 -> 双算子level3 smoke -> eager四模式 -> 自然metadata矩阵”推进。两个补充黑盒
场景并行进入关系设计/用例准备，不抢在数学等价证据之前接NPU。metadata真实注入则在
`PKG-1/META-1/FORMULA-1/RT-1/OP-1`全部关闭后才启用。

### 八、新增意见的多模型复审记录

本轮再次用三个隔离、只读CANNBot会话审阅新增意见3-6和未提交设计：

| 模型 | 本轮关注 | 纳入裁决的意见 |
| --- | --- | --- |
| `cannbot/glm-5.2` | suite配置与现有pytest接法 | 一个suite文档、per-case selector、shape解析，legacy CLI与文件互斥 |
| `cannbot/deepseek-v4-pro` | 结果可信度和异常路径 | pytest退出码/skip文本不能作为PASS；每个mode先落状态，聚合器检查缺失 |
| `cannbot/qwen3.7-max` | 外部两类场景和metadata打桩 | 四模式应标same-source cross-call；独立跨case/单次调用内是gap；注入前冻结FD契约 |

主审还纠正了复审中的两项过度判断：CPU `FakeExecutor`只用于显式单测，不会产生在线NPU
PASS；现有relation validator已经逐relation检查span digest、边界和scope，不是只验证manifest
外壳。有效风险是PASS必须绑定observed level3、异常也必须有artifact，本轮已经落实。

模型没有写共享工作树。临时CANNBot目录只包含隔离数据库/认证副本和审议日志，认证权限保持
0600；验证完成后整目录删除，不进入Git。最终实现和安全边界仍由主审依据源码、实测与需求负责。


# 检视意见四

cannbot模型不止glm-5.2还有deepseek v4和qwen3.7-max，我希望chatgpt能和多方合作沟通商量规划，毕竟别的模型有可能关注到chatgpt没关注到的点，实现的时候chatgpt可以指导其他模型帮你修改一下

做好开发日志记录和文档记录

1. NPU侧算子开发已基本实现，/mnt/workspace/gitCode/cann/ops-transformer下分支pr_8的最新提交就是，虽然部分场景可能有精度问题，但核心逻辑应该是正确的
我已编好自定义算子包（关于这个不懂的可以查阅本文档之前的开发日志）并安装在/home/Ascend，使用前配置环境
```
source /mnt/workspace/gitCode/env.sh
source /home/Ascend/vendors/custom_transformer/bin/set_env.bash
```
你现在看看能不能设置level3了，能设置了的话，尝试继续推进需求任务，做好相关验证，压测等等问题
如果相关背景忘记了查阅本文档之前的开发记录和/mnt/workspace/gitCode/cann/ops-transformer_0716/requs/SMLA_batch一致性测试开发计划.md文档

## 检视意见四处理记录（2026-07-23）

### 一、直接答复

1. **现在只能“设置并读回level 3”，还不能在NPU上执行level 3。** 更新环境中
   `torch_npu.npu.set_deterministic_level(3)`返回成功，getter读回3；但设置后的第一条普通
   `torch.empty((1,), float32).npu()`在runtime dispatch阶段报
   `currently only supports configuring 0/1/2`。因此level3 Python状态接口已更新，runtime
   capability仍未打通，真实SMLA/MQSMLA不能写PASS。
2. **已经检查`pr_8`最新实现和已安装custom OPP。** 源码HEAD是
   `48f862782051cdea2516b2227634ab9507825c50`（`almost pass`）。MQ kernel已增加
   `BATCH_CONSISTENCY` tiling key、跨核/核内workspace和固定规约路径；安装包中的关键kernel
   source与HEAD逐字节相同，三个关键二进制也与该工作树2026-07-23本地build逐字节相同。
3. **没有用level 1或level 0替代需求level 3。** `pr_8` tiling当前检查内部值1，不足以证明
   公共level3会被runtime映射为1；本机runtime反而在dispatch时拒绝3。映射契约仍需runtime和
   tiling实测证据，不能靠推测关闭。
4. **已推进验证和有意义的短压测。** 连续10次执行
   `set3 -> readback3 -> NPU dispatch -> restore0`，10次均稳定在dispatch阶段拒绝，10次均成功
   恢复0。双算子正式入口各生成4条`UNSUPPORTED` artifact，无缺失、无进程失败、无PASS误报。
5. **没有启动无意义的长精度压测。** 普通NPU dispatch尚未过门时，四模式、随机shape和百次
   主算子压力只会重复同一个runtime错误，不能提供算子精度证据。capability通过后才按本节末尾
   的三级顺序解锁。
6. **多模型已合作复审，最终裁决仍由主审负责。** GLM-5.2、DeepSeek V4 Pro、Qwen3.7 Max
   分别做架构、反向审查和测试矩阵检查；模型不直接并发修改共享工作树，避免交叉覆盖。有效建议
   已落到代码，模型的错误建议已由源码和实测排除。

### 二、更新环境与level3三层事实

| 层次 | 2026-07-23实测 | 裁决 |
| --- | --- | --- |
| 状态写入 | setter(3)返回成功 | 只证明Python/C++状态接受暂存值 |
| 状态读回 | getter返回3 | 只证明requested/observed一致 |
| NPU dispatch | 第一条普通NPU tensor操作报只支持0/1/2 | runtime capability为`UNSUPPORTED` |
| 算子metadata/main-op | 未进入 | 不能推断FD、tiling key或主算子精度 |
| 异常恢复 | 10次均从3恢复0 | 恢复短测通过，不等于level3功能通过 |

环境证据：

- SoC：`Ascend910_9362`，20 cube core、40 vector core；
- torch：`2.10.0+cpu`；
- torch_npu：`2.10.0.post3.dev20260709`；
- driver：`25.5.5`，`ascendhal_version=7.35.23`；
- CANN安装根：`/home/Ascend/cann-9.2.0`，安装组件`version.info`报告9.1.0、时间戳
  `20260715_000325483`；
- custom OPP：`/home/Ascend/vendors/custom_transformer`。

torch_npu Python实现本身只把level传给`torch_npu._C._npu_set_deterministic_level(level)`，并用
getter读回；它没有在Python setter中完成真实NPU capability检查。这解释了为什么setter/getter
成功而首个NPU操作才暴露runtime错误。框架现在显式记录这种
`LEVEL3_ACCEPTED_BUT_NPU_REJECTED`状态，不再把它笼统写成“setter失败”。

### 三、`pr_8`与安装包审计

`/mnt/workspace/gitCode/cann/ops-transformer`保持在`pr_8`，HEAD为`48f862782`。本轮只读审计，未
修改或提交该仓库；用户已有未跟踪目录`.arts/`保持原样。

#### 3.1 NPU侧模式实现

`pr_8`的MQSMLA实现已经具备以下核心结构：

1. host tiling调用`IsBatchConsistencyEnabled()`，把开关写入tiling key；
2. kernel template新增`BATCH_CONSISTENCY`布尔维度；
3. batch-consistency分支为同token的固定分块结果规划核内、跨核及FD workspace；
4. vector路径分阶段保存partial max/sum/output并按固定顺序combine；
5. 最终FD/拷出路径消费规约后的结果。

但仍有三个不能跳过的验收点：

- host代码以`GetDeterministicLevel()==1`或`ACL_OPT_DETERMINISTIC==1`启用，测试也显式设置1；
- kernel中仍有大量`[MQSMLA_DBG]`/`[MQSMLA_DEBUG]`打印，提交名也是`almost pass`；
- 用户已提示部分场景可能有精度问题，当前runtime又阻止level3真实调用，所以本轮只能认可“核心
  结构已实现”，不能替算子侧宣布全场景精度验收完成。

#### 3.2 level3与内部值1的映射裁决

当前证据不支持“公共level3必然自动映射为内部1”：

1. 主需求明确公共接口使用3，结果artifact也要求requested/observed均为3；
2. `pr_8`源码只证明它在值1时开key，没有转换代码证据；
3. 当前runtime在第一条NPU dispatch处直接拒绝3，说明本环境至少没有一条可工作的3到1链路；
4. 用level1跑通只能证明level1代码路径，不能证明需求中的batch consistency语义。

关闭`MAP-1`需要：支持level3的runtime接受dispatch；tiling日志或受控打桩记录
`GetDeterministicLevel()`实际值；tiling key确认命中`BATCH_CONSISTENCY=1`；metadata和主算子均在
该上下文执行。四项齐全后再决定host条件应判断3、1或runtime定义的映射值。

#### 3.3 安装包身份

新probe对安装树做了稳定内容manifest：相对路径、文件大小和逐文件SHA256共同生成全树指纹。
本机结果为：

```text
file_count = 776
total_bytes = 21194488
content_manifest_sha256 = 81730ebc09a01b62ea8639ce2f4977d489218817329fcd954356bd88055c7a5e
```

关键二进制：

| 文件 | SHA256 |
| --- | --- |
| `libcust_opmaster_rt2.0.so` | `3ad69362ee08b2eecb7248d068664a1e7c08426dfc82ed8d0b7fb1e667f75507` |
| `libtransformer_aicpu_kernels.so` | `dad01989062d98774200b1437c4fef8b38eb493522ede8fde103c7e2c3d04ba8` |
| `libcust_opapi.so` | `ba115cd4a9faaa2c36fe36e80e2ac65122c5c757056aef283d79793e4aa6209a` |

这三个hash与`pr_8`工作树`build/`及CPack staging payload完全相同；staging与安装目录的payload
diff只多出安装器生成的`bin/`和`scripts/`。build时间戳为`20260723_015357643`，晚于HEAD提交，
且tracked源码无修改。这是“本机安装包来自当前本地build”的强关联证据。

仍不能把`PKG-1`完全标记为跨机器关闭：安装包`version.info`只有
`custom_opp_compiler_version=9.1.0`，没有source commit/build id。以后构建应把commit、dirty状态、
build id和payload manifest写入包内；内容hash可以证明两个payload相同，但不能单独反推出源码commit。

### 四、本轮实现模式与执行状态机

本轮没有修改四种黑盒模式的数学定义；新增的是所有模式共同依赖的runtime门禁：

```text
一次性 capability probe（shell先生成证据）
  -> set level 3
  -> getter必须读回3
  -> 下发1元素float32普通NPU tensor并synchronize
  -> finally恢复进入前level并再次读回
每个PT/mode仍先做config + CPU transform + semantic oracle
  -> INVALID_TRANSFORM/CONFIG_ERROR: 保留输入侧真实状态
  -> SUPPORTED: 才懒加载operator executor
       -> 每个模式执行真实metadata + 主算子 + compare
  -> UNSUPPORTED: 每个已选择模式绑定同一probe证据，直接写UNSUPPORTED
  -> ERROR/恢复失败: 写FAIL，聚合器不得降级
```

普通NPU probe只证明runtime dispatch能力，不是SMLA/MQSMLA smoke。最终PASS仍必须经过：

1. probe为`SUPPORTED`；
2. 算子自己的metadata op成功；
3. 主算子消费同一metadata并成功同步；
4. origin bitwise MD5、CPU precision和metadata integrity全部通过；
5. 每个执行报告requested/observed level均为3。

这样既解决了setter/getter假阳性，也没有拿普通op替代主算子验收。阻断环境下仍保留所有NPU前
数学预检，并使用lazy import；MQSMLA不会再为了四个必定跳过的模式提前加载/编译operator executor。

### 五、本轮文件级开发记录

双算子的`batch/consistency/`继续保持逐字镜像，单测包含新文件防漂移。

| 文件 | 本轮职责 |
| --- | --- |
| 两棵`batch/consistency/runtime_capability.py` | 新增set/readback/NPU dispatch/restore探针；支持1-100次短测；收集torch、torch_npu、CANN、driver、SoC、custom OPP全树/关键文件hash；原子写JSON |
| 两棵`batch/consistency/flow.py` | 复用统一runtime拒绝识别；把直接set拒绝与“接受状态后在operator拒绝”分别编码；恢复失败仍为硬错误 |
| 两棵`batch/consistency/result_artifact.py` | 校验并嵌入capability证据及digest；PASS强制绑定SUPPORTED probe；混用/缺失probe的结果矩阵直接FAIL；汇总输出唯一probe状态/hash |
| 两个`test_*_batch_consistency.py` | 每模式先做CPU transform/oracle再消费一次性probe；UNSUPPORTED快速落artifact；probe ERROR落FAIL；executor改为通过门禁后lazy import |
| 两棵`test_run.sh` | 正式在线入口先生成`runtime_capability.json`；新增`--probe-attempts 1..100`；probe退出2继续生成完整逐模式矩阵 |
| 两棵`test_batch_consistency_unit.py` | 新增延迟拒绝、成功smoke、异常恢复和PASS/probe冲突门禁；镜像清单纳入新模块 |
| 两棵`README.md` | 记录三级门禁、命令、artifact和短压测语义 |
| `level3两段式完整调用示例.md` | 补充分层事实、内部值映射、包指纹和更新后的正式调用 |
| `SMLA_batch一致性测试开发计划.md` | 增加P2.17、MAP-1、实测矩阵和阶段三解锁顺序 |
| `requs.md` | 保留原检视意见并追加本处理记录、证据、答复和后续计划 |

`-M 1,2`仍只在旧CLI转换层做deprecated兼容；本轮正式入口和文档继续使用命名mode/profile及
versioned suite config，没有因为runtime阻断退回数字模式。

### 六、验证与压力结果

#### 6.1 CPU和静态门禁

| 项目 | 结果 |
| --- | --- |
| SMLA一致性单测 | `44 passed` |
| MQSMLA一致性单测 | `44 passed` |
| ruff（本轮Python文件） | 通过 |
| compileall（本轮Python文件） | 通过 |
| 两棵`test_run.sh` bash语法 | 通过 |
| 双树核心镜像 | 通过 |

双算子`consistency_prepare`再次从同一workbook真实执行：各1个baseline、模式1/4各1条relation、
模式2/3各2条relation，共各6条派生关系；Excel解析、原CPU golden、baseline/derived PT round-trip、
relation manifest均通过。该结果仍明确是CPU relation PASS，不是NPU PASS。

#### 6.2 level3恢复短压测

10次结果完全一致：

```text
status=UNSUPPORTED
reason_code=LEVEL3_ACCEPTED_BUT_NPU_REJECTED
failure_stage=npu_smoke
set_call_returned=10/10
readback_matches(level=3)=10/10
npu_smoke_passed=0/10
restore_to_level0=10/10
```

短压测报告文件SHA256为
`44712eed8ec594c36438e5632adac057a85d8ddfbed263ce91cb18173a6c9629`。报告只作本轮临时验收，
关键结论和环境指纹已归档在本节；临时目录在提交前清理，不进入Git。

#### 6.3 双算子正式在线入口

使用versioned suite config和各自离线baseline PT运行：

| 算子 | 逐模式矩阵 | 汇总 | artifact完整性 | 主算子是否执行 |
| --- | --- | --- | --- | --- |
| SMLA | 4条`UNSUPPORTED` | `UNSUPPORTED`，退出2 | 4/4，无校验错误 | 否，capability先阻断 |
| MQSMLA | 4条`UNSUPPORTED` | `UNSUPPORTED`，退出2 | 4/4，无校验错误 | 否，capability先阻断 |

每条artifact的reason均为`LEVEL3_ACCEPTED_BUT_NPU_REJECTED`，并携带probe SHA256、环境版本和
custom OPP manifest；pytest测试本身虽然是skip，JSON-only聚合器仍返回2，没有把skip误算PASS。

### 七、多模型复审与主审裁决

三个模型通过隔离的CANNBot数据目录只读审议。认证副本权限0600，临时目录权限0700；模型访问
兄弟仓库会触发CANNBot交互授权，因此主审没有放宽权限，而是先用本机Git只读核实`pr_8`事实，
再把固定事实交给模型复审。

- GLM-5.2有效指出setter/getter假阳性、内部值1映射无证据、PASS边界和包commit缺口；
- DeepSeek有效指出失败阶段、runtime/driver/SoC/package字段、restore短测和payload hash需求；
- Qwen有效确认阻断时应为`UNSUPPORTED`并停止长压测。

主审没有采纳三项不严谨建议：Qwen给出的简单MatMul shape不是SMLA/MQSMLA smoke；GLM建议的
level0 metadata probe不能放入验收主线；DeepSeek把“设备能力白名单”当唯一UNSUPPORTED判据过于
理想化，本机已有runtime明确拒绝消息，足以按当前状态模型判`UNSUPPORTED`。模型没有修改共享
工作树，最终代码、测试和文档由主审完成并验证。

### 八、阶段状态与下一步

| 里程碑 | 当前状态 |
| --- | --- |
| P2.17 runtime capability/证据/恢复短测 | 完成 |
| P3.1 runtime/custom/operator版本对齐 | 部分推进；custom本地关联已证明，runtime仍阻断，包commit字段待补 |
| P3.2 SMLA eager四模式真实NPU | 等待capability `SUPPORTED` |
| P3.3 MQSMLA eager四模式真实NPU | 等待capability `SUPPORTED`和MAP-1 |
| P3.4 STC 001-027 manifest | 仍等待owner修订source data |
| P3.5 真实metadata覆盖 | 等待P3.1及公式/FD契约 |

环境更新后按以下顺序继续，不能越级：

1. 先运行1次capability probe；若支持，再运行10次恢复/稳定性短测；
2. 分别固定一个SMLA、MQSMLA最小真实PT，只跑baseline metadata+主算子smoke；
3. 观测tiling context和key，关闭level3到内部值1的MAP-1；
4. SMLA按BSND no-mask、位置mask、TND、PA_BBND顺序跑四模式；
5. MQSMLA按相同顺序增加quant mode/rope head dim/FP8-HIF8边界；
6. 最小矩阵连续通过后再做N=3/N=125、S2边界、100次同输入、跨进程和nightly随机压力；
7. 最后推进graph、ACLNN和synthetic metadata，不把任何stub结果计入自然metadata coverage。

当前结论不是“开发失败”，而是测试侧已经把阻断位置精确收敛到runtime dispatch，并建立了不会
误报PASS的可复现门禁。解除RT-1后，现有入口会自动从快速`UNSUPPORTED`切换到两算子真实
metadata/main-op执行，无需再改四模式配置接口。


# 检视意见五

cannbot模型不止glm-5.2还有deepseek v4和qwen3.7-max，我希望chatgpt能和多方合作沟通商量规划，毕竟别的模型有可能关注到chatgpt没关注到的点，实现的时候chatgpt可以指导其他模型帮你修改一下

你有时会忘记之前的任务要求，最好执行前将工作量大纲规划提前总结记录在本文档，依据文档一步步去推行实现，实现一步记录在本并写明下一步该做什么，后续你就知道前面执行的进度而安排后面的事。最好不要像你之前把所有工作做完了才去写日志记录，这样你有可能会丢失很多前面的记忆，懂吗？最后有需要才尝试整理一下文档

前置要求：一是验证前需要加载环境"source /mnt/workspace/gitCode/env.sh "，这里面有python + cann环境；二是如何对一些执行命令有疑问，建议查阅本文档之前的内容

1. ops-transformer关于batch一致性的修改更新了，切到了pr_9103分支，尝试基于新分支推进验证测试。

2. 把SMLA/MQSMLA的pytest batch一致性测试在ops-transformer_0716/requs输出一个文档详细介绍一下，从环境到适配到执行再到结果查看，同时在pytest/excel下准备用例表验证介绍

## 检视意见五执行大纲（已完成）

> 开始时间：2026-07-26。本文档从本轮第一项动作起持续记录；每完成一个阶段，立即补充实际命令、
> 证据、结论、阻断项和下一步，不在全部工作结束后集中补写。所有真实验证命令必须先执行
> `source /mnt/workspace/gitCode/env.sh`；需要调用自定义算子时，还必须执行
> `source /home/Ascend/vendors/custom_transformer/bin/set_env.bash`。

### 一、工作量与阶段验收条件

| 阶段 | 工作内容 | 本阶段验收条件 | 初始状态 |
| --- | --- | --- | --- |
| A | 审计`ops-transformer/pr_9103`的SMLA/MQSMLA batch-consistency实现、构建物、已安装custom OPP和runtime | 明确两算子的开关值、tiling key、workspace/FD、规约路径及源码/构建/安装身份；不得沿用`pr_8`结论 | 已完成 |
| B | 通过CANNBot邀请GLM-5.2、DeepSeek V4 Pro、Qwen3.7 Max做只读复审 | 三方分别给出风险与测试建议；主审用源码/实测逐项裁决；模型不直接并发修改共享工作树 | 已完成 |
| C | 基于新分支重跑level3 capability、恢复短测及双算子最小真实smoke | probe证据包含set/readback、首个NPU dispatch、restore、runtime/SoC/package身份；仅`SUPPORTED`才进入主算子 | 已完成；runtime阻断smoke |
| D | 准备并实际解析两算子`pytest_v2/excel`用例表 | 表格严格匹配现有parser schema；至少覆盖baseline、四种命名模式、位置敏感mask及边界候选；CPU golden/PT round-trip通过 | 已完成 |
| E | 依门禁推进四模式、短压力和结果artifact验收 | 每模式真实metadata+主算子+同步+精度/bitwise检查通过后才可PASS；环境不支持时准确标记`UNSUPPORTED`且不跑长压测 | 已完成；正确阻断主算子 |
| F | 输出详细使用指南、回归检查、整理开发记录并独立提交 | 文档覆盖环境、适配、Excel/PT、命令、artifact、状态与排障；双树回归、静态检查、工作树审计通过 | 已完成 |

正式接口继续使用命名模式/profile和versioned suite config；`-M 1,2`仅保留为旧CLI的deprecated兼容，
本轮不会退回数字模式。真实NPU长压测的解锁顺序固定为：capability `SUPPORTED` -> 双算子最小
metadata/main-op smoke -> 四模式基础矩阵 -> 重复/跨进程/边界压力。

### 二、开始时状态与边界

- 测试仓：`master@b64b8149f`，相对`origin/master` ahead 3；开始时唯一修改是本节上方用户新增的
  `# 检视意见五`，必须完整保留。
- 算子开发仓：`pr_9103@1acab9aec`；存在用户未跟踪目录`.arts/`，本轮不读取后写入、不删除、
  不纳入提交。
- 上一轮`pr_8@48f862782`及旧安装包的结论仅作历史参照；新分支切换后，源码、构建物、安装包和
  runtime capability必须重新取证。
- 不修改`/home/Ascend`安装树；只做身份核验、环境加载和真实调用。CANNBot模型只读审议，不直接
  修改测试仓或算子仓，最终实现与裁决由主审负责。

### 三、当前步骤与紧接着的下一步

当前执行阶段A：先读取`pr_9103`最近batch-consistency提交及两算子host/kernel/pytest实现，重点
核对公共level3在新分支如何触发、SMLA与MQSMLA是否同时落地、是否仍有调试代码或未闭合分支。

紧接着的下一步：比较`pr_9103`的build/CPack产物与
`/home/Ascend/vendors/custom_transformer`已安装payload，记录关键文件hash和全树manifest；若身份
不一致，只报告并停止以旧安装包做主算子结论，不擅自覆盖安装目录。

### 四、阶段A完成记录：`pr_9103`源码、构建与安装身份

完成时间：2026-07-26。所有版本/构建核验均先加载`/mnt/workspace/gitCode/env.sh`；读取旧安装树时
还加载了其`bin/set_env.bash`。算子仓保持`pr_9103@1acab9aec`，tracked文件无修改，用户的`.arts/`
仍是唯一未跟踪目录。

#### 4.1 源码结论

1. MQSMLA host已经从`pr_8`的“context level 1或ACL开关1”改为只在
   `TilingContext::GetDeterministicLevel()==3`时开启；`batchConsistency`进入tiling info和新增的
   `BATCH_CONSISTENCY` tiling-key模板维度。因此上一轮`MAP-1`的核心歧义在新源码中已经关闭：
   公共level3直接对应host判断3，不再假设内部值1。
2. MQSMLA batch路径按`BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM=33`分配核内/跨核combine workspace，
   按固定32份规划reduce block，并在vector路径保存max/sum/partial-O、做增量merge和跨核最终规约；
   非batch路径继续使用普通FD两slot staging。tiling key、workspace和kernel模板三处开关一致。
3. `final`提交已清除上一版`[MQSMLA_DBG]`/`[MQSMLA_DEBUG]`类设备打印；相关host/kernel范围未发现
   `printf`、`TODO`或`FIXME`残留。
4. 算子仓自带旧pytest仍有一处真实不一致：
   `tests/pytest/batch/mixed_quant_sparse_flash_mla_process.py`设置level 1，而新host只识别3；同目录
   single入口已经设置3。这条batch入口不能用于验收新实现。本测试仓`pytest_v2`正式入口固定请求3，
   后续文档会明确禁止照抄该旧入口。
5. SMLA当前host/kernel没有`GetDeterministicLevel`判断，也没有`BATCH_CONSISTENCY` tiling-key维度。
   `7ca0cd3c6`对SMLA目录的改动是扩展被MQSMLA复用的`flash_decode.h`并调整SMLA staging offset，
   不能据此声称SMLA已经存在显式level3专用kernel分支。SMLA仍可做level3黑盒一致性测试，但“是否
   按产品设计需要显式开关/固定规约路径”新增为算子owner确认项，不能由测试侧猜测。

#### 4.2 旧安装树不属于新分支

开始时本地`build/CPack`早于`pr_9103`提交，且build与安装树也不相同。三个关键文件对比如下：

| 文件 | 旧安装树SHA256 | 开始时旧build SHA256 | 裁决 |
| --- | --- | --- | --- |
| `libcust_opmaster_rt2.0.so` | `3ad69362ee08b2eecb7248d068664a1e7c08426dfc82ed8d0b7fb1e667f75507` | `bccbc5b04a261b12257eb495fdc112e180a287ea2a52c2b2beb4ba0dce59d0bf` | 不同，均不能证明`pr_9103` |
| `libtransformer_aicpu_kernels.so` | `dad01989062d98774200b1437c4fef8b38eb493522ede8fde103c7e2c3d04ba8` | `47e2c6fd201066059749c3a54b13cd26f032aeb9291bfbc2d5a324c833f68284` | 不同，均不能证明`pr_9103` |
| `libcust_opapi.so` | `ba115cd4a9faaa2c36fe36e80e2ac65122c5c757056aef283d79793e4aa6209a` | `88d00cbc4d99639616b9ece1cdadf0fb7b78269e904de6c334dc92e3013f89e7` | 不同，均不能证明`pr_9103` |

旧安装树manifest仍为`81730ebc09a01b62ea8639ce2f4977d489218817329fcd954356bd88055c7a5e`，
776个文件、21194488字节。特别是旧安装树中的MQ kernel入口和tiling-key头与当前源码hash不同，
所以不得用该安装树的调用结果替`pr_9103`下结论。

#### 4.3 已只在本地重建`pr_9103`，未覆盖系统安装

实际执行：

```bash
source /mnt/workspace/gitCode/env.sh
bash build.sh --pkg --soc=ascend910_93 --vendor_name=custom \
  --ops=sparse_flash_mla,sparse_flash_mla_metadata,mixed_quant_sparse_flash_mla,mixed_quant_sparse_flash_mla_metadata \
  -j16
```

构建成功，生成`cann-ops-transformer-custom_linux-aarch64.run`；包SHA256为
`fa30e3c778cd07d595cedd9237b4144459ca99a98db02024b56c4b9d521b9840`。新CPack staging为1414个
文件、25293615字节，内容manifest为
`ea11138e941250d4b0241d360b6b54ea6bb960f291e8dc4c5af90d1960a379a4`。关键二进制为：

| 文件 | 新`pr_9103`本地构建SHA256 |
| --- | --- |
| `libcust_opmaster_rt2.0.so` | `6ab959c0088a6dc95c5e8aead320f5048111db77365af8098286be63243bc245` |
| `libtransformer_aicpu_kernels.so` | `698fb75a1bc79bacb66e8c6bda254f40029e75b0b9e0365802ca927fa3815e3f` |
| `libcust_opapi.so` | `1755dc9287265781ab95a5f0cab1d9ad0d6309e6a64cf383d8e8369b4f694273` |

CPack内MQ kernel入口、MQ tiling-key头和SMLA kernel入口分别与当前工作树逐字节同hash，证明本地
staging消费了当前源码；但`version.info`仍没有source commit字段，所以报告只能表述为“本轮命令、
时间戳和逐文件hash强关联”，不能伪造包自证commit。

系统`/home/Ascend/vendors/custom_transformer`未被修改。后续新分支验证将让独立Python进程把本地
CPack staging置于`ASCEND_CUSTOM_OPP_PATH`和`LD_LIBRARY_PATH`首位；artifact必须同时记录新manifest，
否则视为加载身份错误。

阶段B当前步骤：把上述固定事实、检视意见五和现有pytest_v2状态分别交给GLM-5.2、DeepSeek V4
Pro、Qwen3.7 Max只读复审。紧接着的下一步是汇总三方对“本地staging加载可信度、SMLA无显式开关、
MQ旧level1入口、Excel最小矩阵和压力解锁”的独立意见，再由主审逐项用源码和实测裁决。

### 五、阶段B完成记录：三模型复审与主审裁决

完成时间：2026-07-26。三次CANNBot会话分别使用0700临时`XDG_DATA_HOME`和0600认证副本；结束后
临时目录已精确清理。三个模型均只读，没有修改共享工作树。

| 模型 | 分工 | 找到的有效问题 |
| --- | --- | --- |
| GLM-5.2 | 架构、状态机和证据链 | 正式pytest若缺capability report仍可能先执行NPU；probe device和测试device未绑定；新staging虽有manifest但未记录`LD_LIBRARY_PATH`/期望manifest，加载优先级证据不足；SMLA的黑盒结果与产品路径证明应分开 |
| DeepSeek V4 Pro | 误报PASS、缓存和失败路径反审 | 两棵`test_run.sh`未提供显式staging参数；旧artifact/run目录复用有污染风险；SMLA shell严格模式与MQ不一致；离线`pass`容易被脱离scope误读；旧MQ level1入口必须显式告警 |
| Qwen3.7 Max | Excel/schema、命令和用户文档 | 当前workbook每算子只有1条小shape CPU-only行；SMLA/MQ用例名列大小写不同；MQ一致性扩展列缺少类型归一化；文档必须解释退出码、四层覆盖、模式4 unmapped token和配置互斥 |

#### 5.1 采纳并进入本轮实现

1. capability report从“正式shell通常会提供”提升为测试函数的强制前置条件；缺失、无效、设备不匹配
   均在任何executor/NPU调用前落`CONFIG_ERROR`/`FAIL`，不得依赖最终PASS校验兜底。
2. probe新增运行身份契约：记录`LD_LIBRARY_PATH`，支持声明期望custom OPP根和manifest；根不在
   `ASCEND_CUSTOM_OPP_PATH`首位、op-api lib不在`LD_LIBRARY_PATH`首位或manifest不符时为`ERROR`。
   两棵正式shell增加通用参数，不把本机绝对staging路径硬编码进测试代码。
3. probe device与pytest执行device强绑定；结果artifact继续要求同一probe，正式run使用独立report目录
   并拒绝消费本次运行前的旧artifact。
4. 双算子Excel各自落在本算子`pytest_v2/excel`，严格使用各自原parser列名；一致性扩展列在prepare
   层做显式类型/非空校验，禁止Excel数值自动转换导致静默跳过模式。
5. 新使用指南采用CPU离线、runtime capability、metadata/main-op、压力四层口径；清楚说明
   `consistency_prepare`的PASS不是NPU PASS、`UNSUPPORTED`不是算子FAIL、命名模式是正式接口。
6. SMLA artifact/文档增加assurance scope：可报告level3黑盒关系是否PASS，但在owner确认默认路径固定
   规约语义前，不把它包装成“已证明命中专用tiling key”；MQSMLA则同时要求其level3专用路径证据。

#### 5.2 不直接采纳或调整后的建议

- 不采纳“恢复失败时仍优先标原始`UNSUPPORTED`”。恢复失败意味着全局/线程状态可能污染，按既有
  门禁必须是硬`FAIL`；实现只需同时保留原始异常与restore异常，便于排障，不能降级。
- 不采纳“最小SMLA小shape连续相同就说明误报，正确行为应至少一次不一致”。未观察到不一致不能
  反证确定性；正确做法是给黑盒结果限定claim scope，并由owner/tiling证据关闭产品语义缺口。
- 不采纳“probe进程恢复0后要求另一个pytest进程进入guard前必须读回0”。level状态可能有合法的
  进入前值，guard的契约是记录、设3、读回、finally恢复进入值；跨进程残留目前没有源码或实测证据。
- 不把`ldd`输出当“实际已加载路径”。`ldd`只解析依赖，不证明后续进程最终加载对象；本轮先用
  环境首位、全树manifest和关键库hash形成可执行硬门禁，主算子进程的`/proc/self/maps`证据作为
  后续增强项。
- 不直接照搬“每算子7条且立刻跑S2=4096/5120”的数量建议。先由真实parser和CPU golden验证最小
  smoke、四模式、位置mask、layout和边界候选；大shape只在资源估算通过且capability解锁后进入NPU。

阶段C当前步骤：先实现并单测capability加载身份/设备/缺失报告门禁，再用新本地staging运行1次
level3 probe。紧接着的下一步：若probe为`SUPPORTED`，分别只跑SMLA/MQSMLA各一个最小
metadata+main-op smoke；若仍为`UNSUPPORTED`，停止主算子和长压测，但继续阶段D的Excel真实解析、
CPU golden/PT round-trip及文档建设。

#### 5.3 阶段C进度1：capability/provenance强门禁已实现

完成时间：2026-07-26。双算子的镜像`runtime_capability.py`、`result_artifact.py`和正式pytest入口
已完成以下增量：

- probe可接收期望custom OPP根和全树manifest；只有该根同时位于`ASCEND_CUSTOM_OPP_PATH`首位、
  其`op_api/lib`位于`LD_LIBRARY_PATH`首位、实际哈希manifest吻合时身份契约才为`PASS`；否则在
  设置level和NPU smoke之前返回`ERROR/CUSTOM_OPP_IDENTITY_MISMATCH`。
- 环境证据新增`LD_LIBRARY_PATH`；正式shell新增`--custom-opp-root`、
  `--expected-custom-opp-manifest`和显式device参数，路径参数是通用接口，没有把本机staging写死。
- pytest执行任何NPU前强制要求capability report，并校验probe device等于本次test device；缺失或
  不匹配写`CONFIG_ERROR`，不再等PASS artifact末端才兜底。
- artifact不但校验嵌入probe digest，还会重新读取源`runtime_capability.json`并校验文件SHA256和
  JSON内容一致，防止旧/篡改副本混入；SMLA PASS的claim scope明确限定为level3黑盒关系结果。

本步验证全部先加载`/mnt/workspace/gitCode/env.sh`：SMLA与MQSMLA镜像单测各`47 passed`；相关Python
ruff通过；两份`test_run.sh`的`bash -n`通过；`git diff --check`通过。

当前下一步：以新CPack staging根和
`ea11138e941250d4b0241d360b6b54ea6bb960f291e8dc4c5af90d1960a379a4`作为硬期望运行1次真实
level3 probe。先故意保留旧安装树环境，再由新CLI把staging插到首位，以证明身份门禁确实选择
新分支payload而不是旧`/home/Ascend`包。

#### 5.4 阶段C完成记录：`pr_9103`真实level3 capability仍为`UNSUPPORTED`

完成时间：2026-07-27。实测先依次加载`env.sh`和旧安装树`set_env.bash`，再由本轮新接口把
`pr_9103`本地CPack staging插到两个搜索路径首位，并把新manifest作为硬期望。身份契约结果为：

```text
custom_opp_identity_contract.status = PASS
ASCEND_CUSTOM_OPP_PATH[0] = pr_9103 CPack staging
LD_LIBRARY_PATH[0] = pr_9103 CPack staging/op_api/lib
observed_manifest = expected_manifest
                  = ea11138e941250d4b0241d360b6b54ea6bb960f291e8dc4c5af90d1960a379a4
```

这排除了“实际仍在验证旧`/home/Ascend`payload”的歧义。一次探针结果：退出2、`UNSUPPORTED`、
`failure_stage=npu_smoke`、`reason_code=LEVEL3_ACCEPTED_BUT_NPU_REJECTED`；报告SHA256为
`30359741a52cd299b5fe0473ad795e815448206dbc440911c5dd5d04cd5a1de2`。

随后10次恢复短测结果完全一致：

```text
set_call_returned = 10/10
readback_matches(level=3) = 10/10
npu_smoke_passed = 0/10
restore_to_level0 = 10/10
overall = UNSUPPORTED / LEVEL3_ACCEPTED_BUT_NPU_REJECTED
identity_contract = PASS
```

10次报告SHA256为`c641dcf8a9f9ab967db10a3b76acf4b63bafcd0f3cada74e92be3e2e1f128200`。
两个临时报告的关键结论已归档于此，0700临时目录已清理。

按门禁，双算子的metadata和主算子本轮均**没有执行**，这不是主算子FAIL，也不能验证或否定
`pr_9103`的kernel精度；最小真实smoke保持“等待runtime capability SUPPORTED”。阶段E的四模式和
长压测同样不得解锁。

阶段D当前步骤：读取两算子原Excel parser的真实schema和golden约束，在各自`pytest_v2/excel`
准备独立workbook，不复用一个大小写列名混杂的sheet。紧接着的下一步是先用小shape baseline、
位置敏感mask和layout候选跑实际`consistency_prepare`；通过后再把大S2/N边界列为“压力候选”，不在
当前runtime阻断环境冒充NPU覆盖。

#### 5.5 阶段D进度1：Excel扩展列类型门禁回归完成

完成时间：2026-07-27。两棵`offline.py`现在统一要求`consistency_modes`、`split_groups`和
`token_split_sizes`只能是文本或空值；Excel把这些列自动解析成整数、浮点数等非字符串类型时，
prepare会明确报出`Excel <column> must be text`，不再静默忽略并退回默认模式。双树单测新增
`consistency_modes=1234`的反例，验证数值型模式配置必然失败。

本步所有验证先加载`/mnt/workspace/gitCode/env.sh`：SMLA与MQSMLA单测各`47 passed`；
`ruff 0.15.14`、相关Python `compileall`、双shell `bash -n`和`git diff --check`均通过。最初尝试的
`python -m ruff`因当前Python环境未安装ruff模块而不能作为成功证据，随后改用环境中实际提供的
`/opt/buildtools/Python-3.11.4/bin/ruff`重新运行并通过。

当前下一步：依据两算子各自原始parser的大小写敏感schema生成可重复构建的workbook；先逐行运行
`consistency_prepare`验证Excel解析、CPU golden、baseline/derived PT和四个命名模式的round-trip，
再把大`S2/N`组合放进单独压力候选sheet，避免把未执行的候选写成NPU PASS。

#### 5.6 阶段D完成记录：双算子Excel与CPU离线关系矩阵通过

完成时间：2026-07-27。新增可重复生成脚本
`requs/cases/generate_pytest_v2_consistency_workbooks.py`，实际产出：

- `attention/sparse_flash_mla/tests/pytest_v2/excel/smla_batch_consistency_cases.xlsx`，主sheet为
  `SMLA`且用例名列严格使用`testcase_name`；SHA256为
  `42b294d0a3369a0890deb5b9d494e8d5d6e2d0736d27c1a99e42e3dc733d7d59`。
- `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/excel/mqsmla_batch_consistency_cases.xlsx`，主sheet为
  `MQSMLA`且用例名列严格使用`Testcase_Name`；SHA256为
  `d531b2d9c389ca8662dc68243acb084674fc511f9517b0a5a665d1d103c1cbac`。

生成器还会固定workbook属性和ZIP成员时间/顺序；连续生成两次的两份SHA256均逐字节不变，避免
内容相同却因XLSX容器时间戳导致provenance漂移。

每份主sheet均含4条小shape可执行用例，覆盖BSND、TND、PA_BBND、no-mask、位置敏感mask和四个
正式命名模式；`PressureCandidates`另列`S2=4096`、`S2=5120`、`N1=3`、`N1=125+PA`四条候选，
状态全部是`NOT_EXECUTED_CANDIDATE`，且不属于主sheet，parser不会把它们当成已验证用例。

实际`consistency_prepare`结果如下；命令均先加载`/mnt/workspace/gitCode/env.sh`，未加载custom OPP、
未调用NPU：

| 算子 | baseline | derived关系 | 模式计数（reorder/split/token-split/shape-change） | 结论 |
| --- | ---: | ---: | --- | --- |
| SMLA | 4 | 21 | `3/6/8/4` | CPU prepare PASS |
| MQSMLA | 4 | 21 | `3/6/8/4` | CPU prepare PASS |

SMLA summary/合并关系manifest SHA256分别为
`ea2dd9e5044601030a771de1840be5a167b5babb229124bc82a1970a01c2da1b`和
`49b72298620414e92c55b50e85c9dce15b490f6ac68a6995925a338361376bf8`；MQSMLA分别为
`96dfc141689c863ea83cbccb68c047aa84f8efaf3a7378ba6efdfa6ae66f9edb`和
`9fdb20e652a83caad6ab47908bce708ada31b07fe92b016c0e7ec4e40244c0af`。两个合并manifest均报告
四模式、两种Q layout、三种KV layout和两类mask无缺口；大S2三个类别仍明确缺失，与压力sheet的
未解锁状态一致。

MQSMLA第一次执行在BSND/no-mask baseline暴露原CPU golden缺陷：`ori_mask_mode=0`没有初始化
`ori_win_start/end`，抛出`UnboundLocalError`。本轮没有删除该用例来掩盖失败，而是在
`mixed_quant_sparse_flash_mla_golden.py`补齐0（全窗口）、3（RightDown）和4（Band）统一窗口计算及
越界裁剪，并新增单测覆盖三个合法模式和非法模式；MQSMLA单测随后为`48 passed`，完整4行prepare
重跑通过。

上述PASS严格限定为Excel解析、CPU golden、baseline/derived PT schema、语义oracle、关系manifest和
保存/重载一致；不是NPU、metadata或batch-consistency PASS。阶段D完成，阶段E当前步骤是给8个新
baseline补齐versioned suite selector，然后从正式`test_run.sh consistency`重跑身份probe和状态矩阵。
按阶段C已知门禁，预期为完整`UNSUPPORTED`且executor调用数为0；任何主算子执行都属于门禁回归。

#### 5.7 阶段E完成记录：正式入口完整落为`UNSUPPORTED`，未误跑主算子

完成时间：2026-07-27。`requs/cases/SMLA_MQSMLA_batch一致性执行配置.json`保留原离线预检
selector并新增8个pytest_v2 baseline的精确selector；加载真实PT逐个解析后，8/8恰好匹配一条配置，
没有未匹配、多重匹配或shape参数越界。正式执行均先加载`env.sh`和旧安装树`set_env.bash`，再通过
`--custom-opp-root`把`pr_9103` CPack staging置于`ASCEND_CUSTOM_OPP_PATH`与`LD_LIBRARY_PATH`首位，
并以`ea11138e...379a4`作为manifest硬门禁。

SMLA与MQSMLA分别通过本算子的`test_run.sh consistency`执行4个baseline、每个baseline固定汇总四个
模式槽。两边结果完全同构：

| 证据 | SMLA | MQSMLA |
| --- | --- | --- |
| shell退出码/overall | `2 / UNSUPPORTED` | `2 / UNSUPPORTED` |
| artifact | `16/16` | `16/16` |
| 状态 | `UNSUPPORTED=14, NOT_APPLICABLE=2` | `UNSUPPORTED=14, NOT_APPLICABLE=2` |
| PASS/FAIL/CONFIG_ERROR/INVALID_TRANSFORM | 全部0 | 全部0 |
| artifact校验错误/pytest进程失败 | `0/0` | `0/0` |
| identity/restore | `PASS/true` | `PASS/true` |
| capability report SHA256 | `8e8be7654d0596ee126c4cb409200b245df50998c42c492d4db307b231bcad0b` | `1cd8a1a36d44375116890589638cc86a7da48e87873fcd7d9640c2077f25ee45` |
| result summary SHA256 | `a3ed65585518b39953ffec1b29b4e58aa8f897e8a4e69c6872e5de4c551d9f55` | `a54e323c8ddb210edefe7366bd6b15a904926668ca5fbe8a91b8110403be6385` |

每个可执行槽的reason均为`LEVEL3_ACCEPTED_BUT_NPU_REJECTED`；两个专项case只选择模式3/4，故模式1/2
各产生一个`MODE_NOT_SELECTED/NOT_APPLICABLE`槽，这属于预期完整矩阵，不是漏测。所有14个
`UNSUPPORTED` artifact的执行`report`均为`null`；结合测试入口在capability判断之后才调用
`set_device/executor`的控制流，证明本轮没有执行SMLA/MQSMLA metadata或主算子。探针仍是唯一NPU
调用，且它在level3 smoke失败后恢复进入前的level 0。

阶段E因此按当前硬件/runtime条件完成，但完成含义是“门禁和状态链正确”，不是kernel验收完成。
`pr_9103`精度、bitwise、metadata和压力仍处于等待runtime支持状态；SMLA即使未来黑盒PASS，也仍需
owner关闭“没有显式level3 tiling-key/固定规约路径”的产品语义问题。

阶段F执行内容：输出端到端使用指南，更新双README、开发计划和冲突清单，把本轮各文件职责、命令、
状态/退出码、artifact检查和压力解锁顺序固化；随后执行最终镜像/回归/工作树审计并创建独立提交。
上述工作已完成，最终证据和后续计划见下一节。

### 六、阶段F完成记录：逐项答复、文件职责、最终回归与后续计划

完成时间：2026-07-27。阶段F所有验证继续先加载`/mnt/workspace/gitCode/env.sh`；本节落盘后以一个
新的独立Git提交保存本轮修改，不amend已有提交。算子开发仓没有代码改动，系统custom OPP也没有被
覆盖。

#### 6.1 对检视意见五的最终逐项答复

1. **多模型合作已完成且结论已经落地。** GLM-5.2、DeepSeek V4 Pro、Qwen3.7 Max分别从状态机/
   provenance、误报PASS/失败路径、Excel/schema/用户入口三个角度独立只读审议。主审没有把模型意见
   直接当事实，而是用`pr_9103`源码、本地构建、真实probe和pytest结果逐项裁决。采纳项已落实为
   capability强门禁、probe/test device绑定、custom OPP双搜索路径与manifest身份契约、artifact源
   文件复核、Excel类型门禁、算子专属workbook和SMLA claim scope；三模型没有并发修改共享工作树。
2. **按阶段即时记录的工作方式已执行。** 本轮开始即写入A-F工作量、验收条件、当前步骤和下一步；
   A到E每阶段结束后都立即记录命令、证据、结论与阻断，再进入下一阶段。本节只做最终收口，没有
   反向伪造此前过程。以后继续以本节“后续解锁顺序”和开发计划状态表作为恢复点。
3. **已经切换并基于`ops-transformer/pr_9103@1acab9aec`验证。** 本地重新构建的run包SHA256为
   `fa30e3c778cd07d595cedd9237b4144459ca99a98db02024b56c4b9d521b9840`，staging全树manifest为
   `ea11138e941250d4b0241d360b6b54ea6bb960f291e8dc4c5af90d1960a379a4`。正式probe证明该staging位于
   `ASCEND_CUSTOM_OPP_PATH`和`LD_LIBRARY_PATH`首位且manifest吻合；旧`/home/Ascend`包没有冒充
   新分支，也没有被覆盖。当前runtime仍在level3首个NPU dispatch处拒绝，因此结论是环境
   `UNSUPPORTED`，不是`pr_9103`算子精度失败。
4. **完整pytest batch一致性指南和两算子用例表已交付并实际执行。** 新指南覆盖环境、源码/包身份、
   Excel列、CPU prepare、四模式、versioned suite、正式命令、artifact、退出码、排障和压力解锁。
   两份workbook各有4条可运行小shape及4条隔离的未执行压力候选；实际CPU prepare各得到4个baseline、
   21条关系，正式在线入口各得到16/16个结构化槽位。正式接口继续使用命名模式/profile/suite JSON；
   `-M 1,2`只保留deprecated迁移兼容，不会等batch一致性打通后再设计正式接口。

#### 6.2 本轮修复与实现结果

- capability report从可选证据提升为pytest主算子执行的硬前置；缺失、损坏或device不一致在executor
  前写`CONFIG_ERROR`。probe只有在期望root、搜索路径首位和全树manifest全部吻合后才设置level并做
  NPU smoke；身份契约错误准确落`CONFIG_ERROR`，其他真实probe异常仍落`FAIL`；level3异常后严格
  恢复进入前level。
- 每个结果artifact重新读取capability源JSON，校验源文件SHA256、嵌入probe SHA256和JSON内容完全
  相同；SMLA即使未来PASS也只声明level3黑盒关系，不能替代owner对固定规约产品路径的证明。
- Excel一致性扩展列只接受文本或空值，数值型`consistency_modes/split_groups/token_split_sizes`
  明确失败；workbook生成器固定属性、ZIP成员顺序和时间，当前环境连续生成两次字节完全一致。
- MQSMLA原golden的no-mask未初始化窗口缺陷已修复；ori mask 0/3/4统一计算可见KV窗口并裁剪，合法
  BSND no-mask用例不再被删除或绕过，非法模式有反例单测。
- versioned suite新增8个精确selector；当前重生成的8个baseline逐个验证为恰好命中一条，模式与shape
  校验全部通过。正式NPU入口在runtime阻断时完整输出`UNSUPPORTED/NOT_APPLICABLE`矩阵，未进入
  metadata/main-op executor。

#### 6.3 本轮23个文件的职责

| 文件 | 本轮职责 |
| --- | --- |
| `attention/sparse_flash_mla/tests/pytest_v2/README.md` | 补充SMLA用例表、正式staging参数、capability/artifact状态和详细指南入口 |
| `attention/sparse_flash_mla/tests/pytest_v2/batch/consistency/offline.py` | 严格校验Excel三个一致性扩展列的文本类型 |
| `attention/sparse_flash_mla/tests/pytest_v2/batch/consistency/result_artifact.py` | 复核capability源文件，收紧SMLA PASS声明范围 |
| `attention/sparse_flash_mla/tests/pytest_v2/batch/consistency/runtime_capability.py` | 绑定custom OPP根/manifest/搜索路径、device和runtime证据 |
| `attention/sparse_flash_mla/tests/pytest_v2/test_batch_consistency_unit.py` | 覆盖Excel类型、身份契约、device绑定和capability篡改反例 |
| `attention/sparse_flash_mla/tests/pytest_v2/test_sparse_flash_mla_batch_consistency.py` | 在lazy executor之前强制消费同device capability并落准确状态 |
| `attention/sparse_flash_mla/tests/pytest_v2/test_run.sh` | 新增device、custom OPP根和期望manifest正式CLI及环境首位设置 |
| `attention/sparse_flash_mla/tests/pytest_v2/excel/smla_batch_consistency_cases.xlsx` | SMLA 4条CPU可运行关系用例、4条隔离压力候选及表内指南 |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/README.md` | 补充MQSMLA用例表、正式staging参数、结果语义和详细指南入口 |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/batch/consistency/offline.py` | 与SMLA镜像，严格校验Excel扩展列类型 |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/batch/consistency/result_artifact.py` | 与SMLA镜像，校验capability源与结果声明 |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/batch/consistency/runtime_capability.py` | 与SMLA镜像，执行身份、device和level3 capability门禁 |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/mixed_quant_sparse_flash_mla_golden.py` | 修复ori mask 0/3/4的可见KV窗口计算和边界裁剪 |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/test_batch_consistency_unit.py` | 覆盖共享门禁并新增MQ golden窗口合法/非法模式回归 |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/test_mixed_quant_sparse_flash_mla_batch_consistency.py` | 在MQ executor之前强制消费同device capability并落准确状态 |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/test_run.sh` | 为MQ正式入口增加custom OPP身份参数并透传既有device |
| `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/excel/mqsmla_batch_consistency_cases.xlsx` | MQSMLA 4条CPU可运行关系用例、4条隔离压力候选及表内指南 |
| `requs/SMLA_MQSMLA_pytest_batch一致性测试使用指南.md` | 新增从环境、适配、Excel/PT到在线执行、结果查看和排障的端到端指南 |
| `requs/SMLA_MQSMLA需求冲突与对齐清单.md` | 登记SMLA level3路径、MQ旧level1入口和新旧包身份三项新冲突 |
| `requs/SMLA_batch一致性测试开发计划.md` | 把权威分支、完成项、阻断项及后续P3/P4顺序更新到`pr_9103`现状 |
| `requs/cases/SMLA_MQSMLA_batch一致性执行配置.json` | 为8个新baseline增加命名模式和shape-aware精确selector |
| `requs/cases/generate_pytest_v2_consistency_workbooks.py` | 可重复生成两算子专属workbook、压力候选和表内Guide |
| `requs/requs.md` | 保留用户原始检视意见五并持续记录A-F全过程、证据、答复和下一步 |

#### 6.4 最终回归证据

| 检查 | 最终结果 |
| --- | --- |
| SMLA CPU单测 | `48 passed` |
| MQSMLA CPU单测 | `49 passed` |
| Ruff | 11个本轮核心Python文件全部通过；MQ golden以`E9,F63,F7,F82`通过 |
| Python编译 | 本轮Python文件`compileall`通过 |
| Shell/JSON/diff | 两份`test_run.sh`均`bash -n`通过；suite JSON可解析；`git diff --check`通过 |
| 双树镜像 | `offline.py`、`result_artifact.py`、`runtime_capability.py`三组逐字节相同 |
| workbook可重复性 | 当前生成器连续执行两次SHA256一致；SMLA=`42b294...d7d59`，MQSMLA=`d531b2...1cbac` |
| workbook结构 | 各为主sheet+`PressureCandidates`+`Guide`；4条可运行行、4条明确未执行候选 |
| 最终CPU prepare | 两算子均`pass=true`、4 baseline、21 relation；模式计数`3/6/8/4` |
| suite selector | 当前8个baseline全部唯一匹配，无未匹配、多重匹配或shape错误 |
| staging复核 | manifest仍为`ea11138...379a4`，1414文件、25293615字节、0哈希错误；run包hash未变 |
| 身份负向门禁 | 双算子故意传错manifest后均为14 `CONFIG_ERROR`+2 N/A；probe attempt=0、device=`NOT_RUN`、执行report全空 |
| 正式在线状态 | 两算子各16/16 artifact：14 `UNSUPPORTED`+2 `NOT_APPLICABLE`；0 PASS/FAIL/配置错误/缺失 |

阶段F重放生成器时发现阶段D记录的workbook字节hash已不是当前生成器的输出；本节已经把阶段D和最终
表格hash统一修正为当前产物，并再次连续生成两次确认稳定。表格逻辑内容和CPU prepare覆盖计数未
变化。临时PT和报告均位于`/tmp`，不进入Git。

#### 6.5 当前未完成项与后续解锁顺序

当前测试框架、配置、Excel和误报PASS门禁已经完成，但真实kernel验收仍被以下外部条件阻断：

1. runtime/CANN owner提供并确认支持deterministic level 3真实NPU dispatch的版本组合；先要求同一
   staging身份下`runtime_capability.json.status=SUPPORTED`。
2. capability解锁后只跑每算子1条最小metadata+main-op smoke；MQSMLA同时确认命中
   `BATCH_CONSISTENCY` tiling key，SMLA由owner确认“无显式level3分支”是设计还是实现缺口。
3. 最小smoke通过后执行当前4 baseline/16模式槽基础矩阵，再做重复、跨进程和恢复短压测；任何FAIL
   先停在基础矩阵定位，不能直接上长压。
4. 基础矩阵稳定后，按资源估算依次解锁`S2=4096`、`S2=5120`、`N1=3`、`N1=125+PA`候选；候选在
   执行前一直保持`NOT_EXECUTED_CANDIDATE`。
5. 并行由需求/STC owner修正公式、N语义、数组和block映射；由构建owner给custom包补source
   commit/build id；由算子pytest owner修复旧MQ level1 batch入口。之后再推进graph、ACLNN和nightly。

因此本轮“检视意见五”开发任务已完成；尚未完成的是依赖新runtime和owner裁决的真实level3算子
验收，不能在当前环境下写成PASS。
