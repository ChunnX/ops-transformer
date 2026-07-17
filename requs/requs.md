# SMLA/MQSMLA batch 一致性需求与开发台账

> 最近整理：2026-07-17
> 第一阶段基线提交：`b680bfbdd test(smla): add phase-one batch consistency framework`
> 本文用途：归档需求背景、权威资料、需求冲突裁决、实现记录、检视答复和当前状态。具体后续里程碑见
> `SMLA_batch一致性测试开发计划.md`。

## 1. 项目背景与最终范围

batch 一致性的根因不是普通数值精度不足，而是 FlashDecode 在 S2 方向切到不同核后，
FA partial result 的归约顺序可能随 batch、token 或 shape 改变。浮点加法不满足结合律，
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
| 1 | `docs/Milan&TorinoC12_RP3_B050_G项目_STC用例.xlsx` | 001-027 正式验收场景、基准/派生 case 关系和 shape |
| 2 | `docs/会议纪要.md` | 开发者对黑盒模式、S2 切分、FA/FD 和 metadata 控制方式的解释 |
| 3 | `docs/batch一致性测试框架.md` | 四种黑盒模式、白盒层级和自动化流程 |
| 4 | `docs/SMLA batch一致性.md`、`docs/SMLA支持batch一致性.md` | 特性背景、level 3 开关、Level-1/2 规约设计 |
| 5 | `docs/metadata输出结果解析.md` | FA/FD record 的旧版与 split-G 布局说明 |
| 6 | 当前 header、metadata AICPU 和 kernel 消费源码 | 字段偏移、闭开区间和部署代码现状的最终静态证据 |

出现冲突时不静默选择一个说法：STC 决定验收意图，源码决定当前接口事实；仍不能证明
等价的变换必须报告 `INVALID_TRANSFORM` 或保持未实现。

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

## 4. 四种黑盒模式的裁决

| 模式 | 规范名称 | 当前裁决 | STC 对应 |
| --- | --- | --- | --- |
| 1 | `reorder` | 同一组 batch 改变执行顺序，Q/KV/长度/索引同步重排，按 origin 还原后逐 token 比较 | 018，以及 020/024-027 的组合场景 |
| 2 | `split` | 从基准多 batch case 选取一个或多个 batch 子集；每个子 case 直接和基准对应 batch 比较 | 003/008/011/015，随机组合见 020/024-027 |
| 3 | `token-split` | 保持所选 batch 的 KV，上下文不变；选取 batch 内指定 Q token，比较这些 token 的对应输出 | 002/006/007/012/016，随机 token 见 019/020/024-027 |
| 4 | `shape-change` | 保留目标 token，增减其他 Q token 以改变 batch shape；当前资料冲突，暂不执行 | 004/009/013/017 |

### 4.1 模式 2 为什么不以合并结果为主

用户检视意见正确。模式 2 的主验收应是：

```text
基准 case 的对应 origin 输出 <-> 每个派生子 case 的输出
```

每个子 case 单独报告 MD5、数值精度和 batch id。只有显式 groups 恰好完整覆盖原 batch
时，框架才额外合并一次做诊断；合并失败不能掩盖某个子 case 的直接结果，不完整子集也
不再被强制补全。

### 4.2 模式 3 到底在测什么

模式 3 要验证一个 query token 的结果不受同 batch 其他 query token 的组合与调度影响。
Attention 对不同 Q token 数学上独立，但 metadata 分核会看到整组 shape；抽取 token 后
调度可能变化，因此目标 token 必须仍逐 bit 一致。

资料中有三种表述：

- STC：从基准选取每个 batch 的第一个 token、前 x 个 token或单个 token，派生 S1 可变；
- 框架方法 1：把一个 batch 的 token 分到多个逻辑 batch，保持 KV；
- 会议/框架方法 2：保持 shape 和目标 token 位置，只替换其他 Q token，保持 KV。

当前实现优先满足有明确 case 对的 STC 语义，即“token 子集选择”。同 shape 替换其他 Q
并不会改变 shape 驱动的 metadata 调度，单独作为主验收价值较弱，后续可作为 query
独立性补充测试。

位置敏感 mask 不能机械抽取 token。当前 guard 检查目标 token 到有效 Q 右边界的距离：
距离改变则报告 `INVALID_TRANSFORM`；选择后缀并保持右对齐时可以继续。后续 actual-input
CPU golden oracle 建立后，再用真实计算代替有限的静态 guard。

### 4.3 模式 4 为什么仍未实现

- 框架和会议纪要建议在首部 prepend，以维持右下角对齐 mask；
- STC 004/009/013/017 写的是在各 batch 后面追加 token；
- 另一份背景材料又用“前 300 token 相同”描述前缀比较。

三者在 causal/window mask 下并不等价。没有 actual-input golden oracle 和接口负责人裁决
前，模式 4明确报 unsupported，不能用某一种随机实现冒充验收。

## 5. 随机策略与 seed 的检视结论

默认分组完全由 seed 决定不适合作为基础 STC：失败 case 不直观，固定用例与调度覆盖也
难对应。当前规则调整为：

- 模式 1仍按需求打乱 batch，但固定 seed 使排列可复现且强制非恒等；
- 模式 2基础 profile 默认确定性二分，例如 B=5 得到 `[0,1,2] + [3,4]`，seed 不改变它；
- STC 可通过 Python API 显式给出任意 batch 子集，子集不必覆盖全集；
- 只有 `--randomize`/fuzz profile 才用 seed 打乱模式 2分组；
- 随机场景主要对应 STC 018-020、024-027，不能替代 001-017 的显式基准/派生关系。

算子侧不是按测试 seed 分组。metadata 根据 B/N2/S1/S2、有效长度、mask、top-k 和核数
连续分配 FA 任务；seed 只是测试输入组合控制，不是 kernel 调度参数。

## 6. 第一阶段开发检视一：逐项答复与修复

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
