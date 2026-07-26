# SMLA/MQSMLA pytest_v2 batch一致性测试使用指南

> 版本：2026-07-27，适配`ops-transformer/pr_9103@1acab9aec`。
> 本文覆盖环境、用例表、CPU离线预检、versioned配置、runtime探针、正式执行、结果判读和压力解锁。
> 当前机器的level 3真实NPU dispatch为`UNSUPPORTED`；本文不会把CPU prepare或pytest skip写成NPU PASS。

## 1. 测试要回答什么

batch一致性不是“同一输入重复跑两次”，而是证明同一批源token在不同调用形态下仍得到一致输出。
当前实现的主场景是`same-source-cross-call`：从一个baseline PT派生多次调用，通过`TokenOrigin`把
每个派生输出映射回baseline的源batch和源token，再同时检查bitwise MD5与原CPU精度门禁。

测试分四层，结论不能跨层提升：

| 层级 | 实际动作 | 成功能说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| L0 Excel/CPU | parser、原golden、四模式变换、语义oracle、PT round-trip | 输入关系与离线数据链成立 | 未调用NPU，不能称batch一致性PASS |
| L1 runtime capability | set level 3、readback、普通NPU dispatch、finally restore | 当前runtime/device能进入level3 NPU执行 | 未证明SMLA/MQSMLA算子正确 |
| L2 metadata/main-op | 每个派生case真实metadata和主算子、同步、输出比较 | 对应算子/case/mode的eager黑盒结果 | 未覆盖压力、graph或ACLNN |
| L3 压力/白盒 | 重复、跨进程、大S2/N、metadata FA/FD和workspace | 指定版本范围内的扩展验收 | 不能外推到其他包、SoC或runtime |

只有L1为`SUPPORTED`时，正式入口才允许进入L2。当前机器停在L1，正确结果是`UNSUPPORTED`，
不是`FAIL`，也不是`PASS`。

## 2. 路径与文件职责

仓库根目录：

```text
/mnt/workspace/gitCode/cann/ops-transformer_0716
```

两算子的入口与用例表：

| 算子 | pytest_v2目录 | Excel主sheet | 用例名列 |
| --- | --- | --- | --- |
| SMLA | `attention/sparse_flash_mla/tests/pytest_v2` | `SMLA` | `testcase_name` |
| MQSMLA | `attention/mixed_quant_sparse_flash_mla/tests/pytest_v2` | `MQSMLA` | `Testcase_Name` |

关键文件：

| 文件/目录 | 职责 |
| --- | --- |
| 两棵`test_run.sh` | 参数校验、独立run目录、staging搜索路径、capability先行、逐PT隔离pytest、最终JSON汇总 |
| 两个`test_*_batch_consistency.py` | 四模式pytest入口；强制capability文件和device一致；在任何executor/NPU调用前处理配置错误或`UNSUPPORTED` |
| 两个`batch/*_consistency_prepare.py` | Excel到原CPU golden、baseline PT、derived PT、关系manifest的L0入口 |
| 两个`batch/*_consistency.py` | L2 operator executor；同时调用metadata与主算子，并捕获主算子实际消费的metadata |
| 两棵`batch/consistency/config.py` | 正式命名模式、profile以及deprecated数字别名 |
| 两棵`batch/consistency/suite_config.py` | versioned JSON、精确selector、每个PT的shape-aware配置校验 |
| 两棵`batch/consistency/transform.py` | BSND/TND/PA_BBND四模式变换、位置mask上下文联动、actual-input语义oracle |
| 两棵`batch/consistency/offline.py` | Excel扩展列类型检查、origin-mapped expected output、derived PT保存/重载 |
| 两棵`batch/consistency/relation_manifest.py` | baseline/derived关系、内容指纹、边界/重复/scope预检和CPU coverage inventory |
| 两棵`batch/consistency/runtime_capability.py` | level3 set/readback/NPU smoke/restore、device、runtime、SoC、custom OPP manifest和搜索路径证据 |
| 两棵`batch/consistency/result_artifact.py` | 逐模式权威JSON、probe源文件hash/内容复核、完整矩阵聚合和退出码 |
| 两棵`batch/consistency/metadata_*` | 1024-int32布局、FA/FD/workspace解析、floor/ceil公式双轨审计和默认关闭stub |
| 两棵`test_batch_consistency_unit.py` | 纯CPU回归；四模式、配置、provenance、capability、artifact、metadata和故障路径 |
| `mixed_quant_sparse_flash_mla_golden.py` | MQSMLA CPU golden；本轮补齐ori mask 0/3/4窗口计算 |
| `requs/cases/generate_pytest_v2_consistency_workbooks.py` | 可重复生成两份算子专属workbook |
| `requs/cases/SMLA_MQSMLA_batch一致性执行配置.json` | 当前Excel baseline与正式入口共用的schema v1 suite selector |

两棵`batch/consistency`目前是有意镜像。修改公共逻辑时必须同步两边，并在最终检查中逐文件比较，
防止一个算子修复而另一个继续旧行为。

## 3. 环境与`pr_9103`身份

### 3.1 每次验证的公共前置

所有验证命令都先执行：

```bash
source /mnt/workspace/gitCode/env.sh
```

它提供本仓所需Python、PyTorch、torch_npu与CANN环境。不要直接使用系统Python猜测依赖。

### 3.2 调用自定义算子时再加载安装树

CPU-only的`consistency_prepare`不需要custom OPP。运行capability或正式NPU入口时再执行：

```bash
source /home/Ascend/vendors/custom_transformer/bin/set_env.bash
```

旧安装树只是提供通用环境，不能代表`pr_9103`。正式命令必须再通过参数选择本地staging：

```bash
STAGING=/mnt/workspace/gitCode/cann/ops-transformer/build/_CPack_Packages/Linux/External/\
cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer
MANIFEST=ea11138e941250d4b0241d360b6b54ea6bb960f291e8dc4c5af90d1960a379a4
```

shell会把：

```text
$STAGING
$STAGING/op_api/lib
```

分别插到`ASCEND_CUSTOM_OPP_PATH`和`LD_LIBRARY_PATH`首位。probe还会重新计算全树manifest；根目录、
首位顺序或manifest任一不符都会在set level和NPU smoke之前返回
`ERROR/CUSTOM_OPP_IDENTITY_MISMATCH`。

当前staging的`version.info`没有source commit字段，因此可声称“内容manifest与本轮`pr_9103`构建
逐文件强关联”，不能声称“包自身证明了commit”。关键构建hash和构建命令见`requs.md`检视意见五。

## 4. Excel用例表

### 4.1 生成与检查

从仓库根目录运行：

```bash
source /mnt/workspace/gitCode/env.sh
python requs/cases/generate_pytest_v2_consistency_workbooks.py
```

生成：

```text
attention/sparse_flash_mla/tests/pytest_v2/excel/smla_batch_consistency_cases.xlsx
attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/excel/mqsmla_batch_consistency_cases.xlsx
```

每份workbook有三个sheet：

| sheet | 是否由prepare执行 | 内容 |
| --- | --- | --- |
| `SMLA`或`MQSMLA` | 是 | 4条小shape可运行baseline |
| `PressureCandidates` | 否 | S2=4096/5120、N1=3/125等未解锁候选 |
| `Guide` | 否 | scope、模式语法和解锁顺序 |

`PressureCandidates`不使用主parser sheet名，且每行明确标为`NOT_EXECUTED_CANDIDATE`；它只是一张
待办表，不能出现在NPU覆盖或PASS计数中。

### 4.2 当前可运行矩阵

两算子都包含同一关系覆盖结构：

| case类型 | Q/KV layout | mask | 模式 | 目的 |
| --- | --- | --- | --- | --- |
| BSND no-mask | BSND/BSND | ori=0 | 1-4 | 基础四模式与MQ no-mask golden |
| BSND RightDown | BSND/BSND | ori=3 | 3/4 | 位置敏感token分段和shape联动 |
| TND no-mask | TND/TND | ori=0 | 1-4 | prefix sum重排、分组、分段、增长 |
| PA Band | BSND/PA_BBND | ori=4 | 1-4 | block table重排/拆分；mode4在映射容量内缩短 |

当前小矩阵是L0快速门禁，不替代STC大shape覆盖。它实际产生每算子4个baseline、21条derived关系：

```text
reorder=3, split=6, token-split=8, shape-change=4
```

### 4.3 一致性扩展列

主sheet在原算子schema末尾增加：

| 列 | 示例 | 规则 |
| --- | --- | --- |
| `consistency_modes` | `reorder,split,token-split,shape-change` | 只能使用正式名字，数字和数值单元格拒绝 |
| `split_groups` | `0;1`或`0,1,2;3,4` | 组内逗号、组间分号；正式验收通常要求完整partition |
| `token_split_sizes` | `2,2`或`4096,12288` | 至少两个正整数，和必须等于源batch有效Q长度 |
| `mode_batch` | `0` | 模式3/4使用的源batch |
| `common_tokens` | `3`或`300` | 模式4与baseline共同的前缀长度 |
| `derived_extra` | `2`或`200` | 模式4派生case共同前缀后的token数 |

前三列只能是文本或空值。Excel把`consistency_modes`存成数值`1234`时，prepare会报
`Excel consistency_modes must be text`，不会静默退回默认模式。

## 5. L0：CPU离线预检

### 5.1 SMLA

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer_0716/attention/sparse_flash_mla/tests/pytest_v2
source /mnt/workspace/gitCode/env.sh

bash test_run.sh consistency_prepare \
  --excel ./excel/smla_batch_consistency_cases.xlsx \
  --sheet SMLA \
  --pt-dir /tmp/smla_consistency_pt \
  --report-dir /tmp/smla_consistency_prepare \
  --seed 20260716
```

### 5.2 MQSMLA

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer_0716/attention/mixed_quant_sparse_flash_mla/tests/pytest_v2
source /mnt/workspace/gitCode/env.sh

bash test_run.sh consistency_prepare \
  --excel ./excel/mqsmla_batch_consistency_cases.xlsx \
  --sheet MQSMLA \
  --pt-dir /tmp/mqsmla_consistency_pt \
  --report-dir /tmp/mqsmla_consistency_prepare \
  --seed 20260716
```

输出结构：

```text
/tmp/<op>_consistency_pt/
  baseline/*.pt

/tmp/<op>_consistency_prepare/
  case_000/
    consistency_prepare_report.json
    consistency_relation_manifest.json
    derived/*.pt
  case_001/...
  <op>_consistency_prepare_summary.json
  <op>_consistency_relation_manifest.json
```

重点检查：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/smla_consistency_prepare/smla_consistency_prepare_summary.json")
doc = json.loads(path.read_text(encoding="utf-8"))
print("scope:", doc["scope"])
print("pass:", doc["pass"])
print("case_count:", doc["case_count"])
print("relation_coverage:", doc["relation_coverage"])
PY
```

这里的`pass=true`只属于`CPU-only offline acceptance`。若把它写成NPU PASS，就是结论升级错误。

## 6. 正式模式与suite配置

### 6.1 为什么不用`-M 1,2`

正式接口使用：

```text
reorder, split, token-split, shape-change
```

`-M 1,2`只为旧命令兼容，运行时会发出deprecated警告。数字无法表达“2是batch拆分还是token拆分”
一类语义，也不携带shape相关参数，不适合作为稳定配置。正式批量验收使用versioned JSON：

```text
requs/cases/SMLA_MQSMLA_batch一致性执行配置.json
```

配置以`operator + case_name`精确匹配PT；也支持`pt_basename`和输入内容SHA256。未匹配、多重匹配、
未知字段、重复batch、越界id、token分段总和错误均在NPU前落`CONFIG_ERROR`。

`--config-file`和`--modes/--profile/--split-groups/...`互斥，不能混用两套来源。

### 6.2 四种模式实现

#### 模式1：`reorder`

- BSND按batch维重排Q/KV/长度/索引；
- TND切分各batch token段后重排并重算prefix sum；
- PA保持物理cache不动，只重排block table行；
- permutation必须非恒等；输出按`TokenOrigin`还原比较。

#### 模式2：`split`

- 把一个baseline batch集合拆成至少两个独立调用；
- 示例B=8明确写成`[[0,1,2],[3,4,5,6,7]]`，即3+5；
- 每个subcase直接与自身origin比较；完整partition可额外做merge诊断，但merge不是主oracle。

#### 模式3：`token-split`

- 对一个源batch按连续区间拆成至少两次调用；
- `sum(token_split_sizes)`必须等于有效Q长度，区间完整、无重叠、无遗漏；
- 对RightDown/Band，派生片段同步设置`new_ori_S2 = old_ori_S2 - old_Q + end`；
- cmp长度与residual按`divmod(new_ori_S2, cmp_ratio)`同步；
- NPU前逐token证明Q、可见KV、mask窗口、sparse index、top-k和sinks相同。

#### 模式4：`shape-change`

- baseline和derived共享前`common_tokens`，但总Q shape必须不同；
- 对位置敏感mask同步设置`new_ori_S2 = old_ori_S2 + new_Q - old_Q`；
- BSND/TND可按需增长物理tensor；PA必须已有足够block映射，否则为`INVALID_TRANSFORM`；
- 新增token没有baseline origin，不参与bitwise比较；报告必须记录`unmapped_tokens`。

模式4“共同前缀+KV同步变化”目前是可证明输入等价的测试侧暂定口径，产品定义仍需需求/mask owner
确认，详见冲突清单。

## 7. L1/L2：正式在线执行

先准备好L0生成的`baseline/*.pt`，再运行正式入口。以下命令使用当前`pr_9103` staging；换机器或
重新构建后必须重新计算manifest，不能照抄旧值。

### 7.1 SMLA

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer_0716/attention/sparse_flash_mla/tests/pytest_v2
source /mnt/workspace/gitCode/env.sh
source /home/Ascend/vendors/custom_transformer/bin/set_env.bash

STAGING=/mnt/workspace/gitCode/cann/ops-transformer/build/_CPack_Packages/Linux/External/\
cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer
MANIFEST=ea11138e941250d4b0241d360b6b54ea6bb960f291e8dc4c5af90d1960a379a4

bash test_run.sh consistency \
  --pt-dir /tmp/smla_consistency_pt/baseline \
  --config-file /mnt/workspace/gitCode/cann/ops-transformer_0716/requs/cases/SMLA_MQSMLA_batch一致性执行配置.json \
  --report-dir /tmp/smla_consistency_result \
  --probe-attempts 1 \
  --device-id 0 \
  --custom-opp-root "$STAGING" \
  --expected-custom-opp-manifest "$MANIFEST"
```

### 7.2 MQSMLA

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer_0716/attention/mixed_quant_sparse_flash_mla/tests/pytest_v2
source /mnt/workspace/gitCode/env.sh
source /home/Ascend/vendors/custom_transformer/bin/set_env.bash

STAGING=/mnt/workspace/gitCode/cann/ops-transformer/build/_CPack_Packages/Linux/External/\
cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer
MANIFEST=ea11138e941250d4b0241d360b6b54ea6bb960f291e8dc4c5af90d1960a379a4

bash test_run.sh consistency \
  --pt-dir /tmp/mqsmla_consistency_pt/baseline \
  --config-file /mnt/workspace/gitCode/cann/ops-transformer_0716/requs/cases/SMLA_MQSMLA_batch一致性执行配置.json \
  --report-dir /tmp/mqsmla_consistency_result \
  --probe-attempts 1 \
  --device-id 0 \
  --custom-opp-root "$STAGING" \
  --expected-custom-opp-manifest "$MANIFEST"
```

每次执行在`--report-dir`下新建`YYYYmmdd_HHMMSS_PID`目录，不复用旧artifact。执行顺序固定为：

```text
参数/路径/manifest预检
  -> 一次runtime capability report
  -> 每个baseline独立pytest进程
  -> 每个PT固定四个模式artifact
  -> JSON-only聚合与完整性检查
```

capability为`UNSUPPORTED`时，pytest仍为每个预期槽写结构化结果，但在`set_device`和executor之前
skip；不会调用metadata或主算子，也不会跑压力。

## 8. 结果查看

一次正式run目录包含：

```text
<run_dir>/
  runtime_capability.json
  pytest.log
  results/*.json
  consistency_result_summary.json
```

权威顺序是：

1. `consistency_result_summary.json`；
2. `results/*.json`；
3. `runtime_capability.json`；
4. `pytest.log`只作诊断，不能用“pytest退出0/skip”替代JSON结论。

快速检查：

```bash
python - <<'PY'
import json
from pathlib import Path

run_dir = Path("/tmp/smla_consistency_result/<run-id>")
summary = json.loads((run_dir / "consistency_result_summary.json").read_text())
probe = json.loads((run_dir / "runtime_capability.json").read_text())
print("overall:", summary["overall_status"])
print("counts:", summary["status_counts"])
print("artifact errors:", summary["artifact_validation_errors"])
print("probe:", probe["status"], probe["reason_code"])
print("identity:", probe["custom_opp_identity_contract"]["status"])
print("restored:", probe["restoration_pass"])
PY
```

### 8.1 状态定义

| 状态 | 含义 | 是否算PASS |
| --- | --- | --- |
| `PASS` | level3 probe受支持，真实metadata+主算子+同步+bitwise/精度门禁全过 | 是，但仅限artifact声明scope |
| `FAIL` | executor、比较、恢复或其他真实执行失败 | 否，硬失败 |
| `CONFIG_ERROR` | 缺probe、device不一致、suite不匹配、身份不符或配置非法 | 否，先修配置/证据 |
| `UNSUPPORTED` | 当前runtime/device/package明确不支持必要level3 dispatch | 否；不是算子精度FAIL |
| `INVALID_TRANSFORM` | 无法证明派生输入语义等价或PA容量不足 | 否；修用例关系 |
| `NOT_APPLICABLE` | 本PT未选择该模式或算子不匹配 | 中性，但矩阵槽仍必须存在 |

聚合overall还可能是`PARTIAL`、`INCOMPLETE`或`NO_RESULTS`。缺失、重复、外来artifact或pytest子进程
失败不能汇总为PASS。

### 8.2 退出码

| 退出码 | 聚合含义 |
| ---: | --- |
| 0 | `overall_status=PASS` |
| 1 | `overall_status=FAIL` |
| 2 | 非PASS且非硬FAIL，例如`UNSUPPORTED/PARTIAL/INCOMPLETE/NO_RESULTS`；必须读取JSON区分 |

因此CI不能简单把退出2改成成功，也不能把`pytest skipped`当通过。

### 8.3 当前机器的已验证结果

2026-07-27基于`pr_9103` staging实测，两算子均为：

```text
case_count=4
artifact_count=16/16
UNSUPPORTED=14
NOT_APPLICABLE=2
PASS/FAIL/CONFIG_ERROR/INVALID_TRANSFORM=0
overall_status=UNSUPPORTED
shell exit=2
identity=PASS
restore=true
```

reason为`LEVEL3_ACCEPTED_BUT_NPU_REJECTED`。该结论只证明门禁和状态链正确；metadata/main-op没有执行，
不能据此判断`pr_9103` kernel精度好坏。

## 9. 常见问题与排障

### 9.1 `CUSTOM_OPP_IDENTITY_MISMATCH`

检查：

- `--custom-opp-root`是否指向`packages/vendors/custom_transformer`本身；
- `$root/op_api/lib`是否存在；
- 两个搜索路径首项是否分别是root和root/op_api/lib；
- manifest是否来自当前staging，而非旧安装树；
- 是否在构建后继续沿用旧manifest。

身份错误不要通过删除`--expected-custom-opp-manifest`绕过，应重新构建或重新取证。
正式入口会把可执行模式槽写成`CONFIG_ERROR/CUSTOM_OPP_IDENTITY_MISMATCH`并以整体`FAIL`退出；probe
应显示`attempt_count_executed=0`、`device_init.status=NOT_RUN`，所有artifact执行`report`均为空。
如果身份错误时仍出现level3 attempt或主算子report，属于门禁回归。

### 9.2 `LEVEL3_ACCEPTED_BUT_NPU_REJECTED`

这表示setter返回且readback为3，但首个普通NPU dispatch被runtime拒绝。当前环境的错误明确说只支持
0/1/2。处理方向是对齐torch_npu/CANN/runtime/SoC，不是把测试降到level0，也不是把状态改成PASS。

### 9.3 `CONFIG_ERROR: ... has no suite config match`

PT中的case name与suite selector不同。先打印PT身份，再更新versioned配置；不要改成
`unmatched_case_policy=defaults`掩盖正式验收遗漏。

### 9.4 `token_split_sizes ... expected ...`

分段总和必须等于`mode_batch`的有效Q长度，不是物理padding长度。TND依据`cu_seqlens_q`，BSND有
`seqused_q`时依据有效长度。

### 9.5 `PA_BBND ... only N blocks mapped`

位置敏感模式3/4会联动逻辑KV长度。增长后的长度需要更多block时，Excel baseline必须预留真实可寻址
block映射；测试不会虚构block id。无法满足时正确状态是`INVALID_TRANSFORM`。

### 9.6 CPU prepare通过但正式入口`UNSUPPORTED`

这是正常分层结果。prepare验证L0，正式入口在L1阻断，两者不矛盾。

### 9.7 SMLA未来黑盒PASS怎么表述

当前`pr_9103` SMLA源码没有显式`GetDeterministicLevel()==3`判断或batch-consistency tiling-key。
即使未来level3黑盒关系PASS，也只能表述为“level3环境下黑盒跨调用结果通过”；在算子owner确认固定
规约产品语义前，不能表述为“已证明命中专用batch-consistency kernel路径”。

## 10. 压力与后续解锁顺序

严格按以下顺序推进，前一步失败时停止后续NPU动作：

1. capability连续短测全部`SUPPORTED`且每次restore通过；
2. SMLA、MQSMLA各1条最小metadata+main-op smoke；
3. 当前小矩阵四模式；
4. 重复执行和跨进程恢复；
5. `S2=4096`精确128边界；
6. `S2=5120`大于128且可整除候选；
7. 真正的`S2/32`有余数、`S2<32`和floor/ceil冲突点；
8. owner确认后的`N1=3/N1=125`、TND、PA_BBND nightly；
9. graph和ACLNN独立执行面。

压力候选必须先补全原STC缺失的dtype、N1/N2/G、mask、有效长度、block mapping、baseline关系和
资源预算。不能仅把大数字填入当前小表就称为STC覆盖。

## 11. 提交前回归

从仓库根目录执行：

```bash
source /mnt/workspace/gitCode/env.sh

python -m pytest -q \
  attention/sparse_flash_mla/tests/pytest_v2/test_batch_consistency_unit.py
python -m pytest -q \
  attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/test_batch_consistency_unit.py

/opt/buildtools/Python-3.11.4/bin/ruff check \
  attention/sparse_flash_mla/tests/pytest_v2/batch/consistency \
  attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/batch/consistency \
  requs/cases/generate_pytest_v2_consistency_workbooks.py

bash -n \
  attention/sparse_flash_mla/tests/pytest_v2/test_run.sh \
  attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/test_run.sh

git diff --check
```

还要确认：

- 两棵镜像公共模块没有无意差异；
- workbook可由generator重建且主sheet行数/列名正确；
- 临时PT、报告、CANNBot认证和模型会话没有进入Git；
- `ops-transformer`算子仓tracked文件仍未被测试工作修改；
- `/home/Ascend`安装树未被覆盖。
