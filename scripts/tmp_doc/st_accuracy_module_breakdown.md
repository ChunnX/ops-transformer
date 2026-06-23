# st_accuracy 模块端到端说明

## 1. 模块定位

`scripts/st_accuracy` 是当前仓库本地 ST 精度测试框架。它的目标不是替代全量 CI，而是在本仓库内完成一条可复核的本地精度闭环：

```text
发现 AscendC 算子 -> 装载 ST JSON case -> 过滤和支持性判定 -> 真实执行时强制构建/安装 custom package
-> 生成 native batch 输入 -> C++ ACLNN runner 设备执行 -> host golden compare -> 写出结构化报告
```

当前默认发现范围是 `attention/`，当前重点验证范围是：

- `attention/lightning_indexer`
- `attention/recurrent_gated_delta_rule`

当前两个重点算子的 ST case 数量：

- `lightning_indexer`：10 个 case，API 为 `LightningIndexer`
- `recurrent_gated_delta_rule`：8 个 case，API 为 `RecurrentGatedDeltaRule`
- 合计：2 个算子、18 个 case

该框架已经完成 P0-P6 基础优化：

| 阶段 | 模块 | 当前状态 |
|---|---|---|
| P0 | Discovery gate | 已完成；显式 `--ops` 下缺失算子、0 case、坏 ST 资产会失败 |
| P1 | 分阶段耗时统计 | 已完成基础版；summary 写出 Python/native/report 阶段耗时 |
| P2 | Native C++ 内部耗时 | 已完成基础版；两个 attention API 写出 parse/input/golden/workspace/launch/D2H/compare |
| P3 | Python 重复计算 | 已完成基础版；`StCase` 派生视图缓存，coverage/taxonomy 复用 support 判定 |
| P4 | Native 桥接与构建缓存 | 已完成基础版；runner binary 进入跨 `--out` 的 cache，并写出 load 诊断 |
| P5 | Adapter 结构拆分 | 已完成基础版；共享 helper 与两个 attention adapter 已拆分 |
| P6 | C++ runner 结构拆分 | 已完成基础版；两个 attention API 的 C++ golden/compare/runner 已拆入 `attention_runner.inc` |

诚信边界保持不变：

- 不修改 golden、case JSON、比较阈值来制造精度通过。
- 不跳过 host golden compare。
- 不把 `coverage`、`taxonomy` 或 `--no-execute` 写成设备侧精度通过。
- 不把旧 package、旧 vendor 产物或单次静态扫描当作 package/load gate 通过证据。

## 2. 对外命令

所有命令入口都在 `scripts/st_accuracy/run_st_accuracy.py`。

| 命令 | 功能 | 主要输入 | 主要输出 | 是否执行设备 |
|---|---|---|---|---|
| `list` | 列出可发现算子、API 和 ST case 数量 | `--ops`、`--json-pattern`、仓库源码树 | stdout text/json；含 discovery report 和 timings | 否 |
| `coverage` | 统计当前 native runner 可执行覆盖率 | operator 列表、ST case、adapter support gate | `coverage_summary.json`、`coverage_by_op.jsonl`、`coverage_by_api.jsonl` | 否 |
| `taxonomy` | 分类 unsupported 原因并给出 adapter 优先级 | operator 列表、ST case、adapter support gate、example/header/CMake 信号 | `taxonomy_summary.json`、`taxonomy_by_api.jsonl`、`taxonomy_by_op.jsonl`、`taxonomy_priority.jsonl` | 否 |
| `run --no-execute` | 只做发现、筛选、support gate 和结果落盘 | ST case、selector 参数、adapter support gate | `summary.json`、`cases.jsonl`，状态为 `skipped` 或 `unsupported` | 否 |
| `run` | 先强制构建/安装 selected op 的 custom package，再真实执行 selected case 并比较 golden | ST case、selector 参数、adapter、native runner、custom package | `summary.json`、`cases.jsonl`、native cache、batch TSV | 是 |
| `run --prepare` | 兼容旧脚本的保留参数；真实设备 `run` 无论该参数如何都会 package/install，`--no-execute` 不安装 | `--soc`、`--vendor`、`--jobs`、`--opt` | 真实执行时写入 `prepare_pkg_ms` | 取决于是否 `--no-execute` |

常用命令示例：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py list \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --format json

python3 scripts/st_accuracy/run_st_accuracy.py coverage \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --out build/st_accuracy/coverage

python3 scripts/st_accuracy/run_st_accuracy.py taxonomy \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --out build/st_accuracy/taxonomy

python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --no-execute \
  --out build/st_accuracy/no_execute

LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} \
python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --vendor custom \
  --soc ascend910b \
  --device 0 \
  --out build/st_accuracy/device
```

`--device` 是传给 `aclrtSetDevice()` 的 ACL runtime logical device id，不等同于一定是 `npu-smi` 物理编号。

真实设备侧 `run` 的 package/load gate 是强制行为：当没有指定 `--no-execute` 且筛选后存在 case 时，框架会对 selected case 的 op 集合执行 `build.sh --pkg` 并安装最新 custom package。静态命令和 `run --no-execute` 不安装 package，因为它们不产生设备侧精度结论。

## 3. 端到端数据流

### 3.1 静态链路

`list`、`coverage`、`taxonomy` 和 `run --no-execute` 都走静态链路：

```text
run_st_accuracy.py
  -> inventory.discover_operators()
  -> st_cases.load_cases_with_diagnostics()
  -> discovery gate
  -> CaseSelector
  -> adapters.get_adapter().supports()
  -> coverage/taxonomy/report
```

静态链路的作用是确认“有哪些 case、哪些可以被当前 native runner 合法执行、哪些不能执行以及原因是什么”。它不调用 ACLNN，不产生设备侧精度结论。

### 3.2 设备执行链路

真实 `run` 在 support gate 之后进入 native 链路：

```text
run_st_accuracy.py
  -> CaseSelector
  -> prepare_pkg(build.sh --pkg + 安装 .run)
  -> runner.run_cases()
  -> native.NativeRunner.run_many()
  -> native cache build/load
  -> native/batch_cases.tsv
  -> native/st_accuracy_aclnn_runner.cpp
  -> dlopen libcust_opapi.so
  -> aclrtSetDevice + aclrtStream
  -> aclnn*GetWorkspaceSize + aclnn*
  -> D2H
  -> host golden compare
  -> stdout JSONL
  -> Python parse JSONL
  -> CaseResult
  -> report.ResultReporter
```

设备执行链路的精度通过条件是：C++ runner 成功调用当前 custom package 的 ACLNN API，并且设备输出与 host golden compare 通过。

## 4. 输入与输出模型

### 4.1 外部输入

| 输入 | 来源 | 用途 |
|---|---|---|
| 仓库算子目录 | `attention/<op>/` | `inventory.py` 判断是否是可优化 AscendC 算子 |
| ST JSON | `<op>/tests/st/<api_dir>/all_*.json` | `st_cases.py` 装载 case |
| CLI selector | `--ops`、`--apis`、`--case-ids`、`--dtype`、`--limit` | `runner.CaseSelector` 筛选 case |
| package 参数 | `--vendor`、`--soc`、`--jobs`、`--opt` | 真实设备 run 的强制 package build/install 与 native compile context |
| runtime 参数 | `--device`、`ASCEND_HOME_PATH`、`LD_LIBRARY_PATH` | native runner 编译、dlopen 和设备执行 |
| native runner source | `native/st_accuracy_aclnn_runner.cpp`、`native/attention_runner.inc` | 构建 C++ runner，并参与 cache key |

ST JSON case 的核心字段来自原 ST 资产，不在此框架内改写。框架主要读取：

- `id`
- `aclnn_name`
- `inputs`
- tensor/scalar/attr 的 `name`、`type`、`dtype`、`shape`、`range_values`、`value`

### 4.2 内部数据模型

| 数据模型 | 文件 | 关键字段 | 作用 |
|---|---|---|---|
| `Operator` | `inventory.py` | `name`、`category`、`path` | 表示一个被发现的 AscendC 算子 |
| `StCase` | `st_cases.py` | `op_name`、`category`、`api_dir`、`aclnn_name`、`json_path`、`case_id`、`raw` | 表示一个 ST JSON case |
| `CaseLoadResult` | `st_cases.py` | `cases`、`skipped_assets` | 同时返回合法 case 和 ST 资产诊断 |
| `StAssetDiagnostic` | `st_cases.py` | `reason`、`message`、`json_path`、`case_index` | 描述缺失/非法 ST 资产 |
| `CaseResult` | `st_cases.py` | `op_name`、`api`、`case_id`、`status`、`message`、`duration_ms`、`extra` | 所有 `run` 输出行的统一格式 |
| `TimingCollector` | `timing.py` | `*_ms`、`total_ms` | 记录阶段耗时 |

`StCase` 已缓存常用派生视图：

- `inputs`
- `flat_inputs`
- `tensor_items`
- `tensor_list_items`
- `scalar_items`
- `tensor_dtypes`
- `all_dtypes`

这部分是 P3 优化结果，目的是避免 adapter、coverage、taxonomy 反复 flatten 同一个 case。

### 4.3 对外输出

`run` 输出目录：

| 文件 | 内容 |
|---|---|
| `summary.json` | `total`、`passed`、`failed`、`error`、`unsupported`、`skipped`、`timings`、`native` |
| `cases.jsonl` | 每个 case 一行 `CaseResult` JSON |
| `native/batch_cases.tsv` | Python 传给 C++ runner 的 batch 输入 |

`summary.json.timings` 目前包含：

- `inventory_discovery_ms`
- `case_load_ms`
- `discovery_report_ms`
- `case_select_ms`
- `support_gate_ms`
- `native_build_ms`
- `batch_write_ms`
- `native_subprocess_ms`
- `native_output_parse_ms`
- `report_write_ms`
- `total_ms`

两个 attention API 的 `cases.jsonl.extra.timings` 目前包含：

- `parse_ms`
- `input_generate_h2d_ms`
- `golden_ms`
- `workspace_ms`
- `launch_sync_ms`
- `d2h_ms`
- `compare_ms`

`coverage` 输出目录：

| 文件 | 内容 |
|---|---|
| `coverage_summary.json` | 算子数、case 数、executable/unsupported 计数、unsupported reason 汇总、timings |
| `coverage_by_op.jsonl` | 每个 op 的 case/API/可执行/unsupported/zero-sized 汇总 |
| `coverage_by_api.jsonl` | 每个 API 的 case/op/可执行/unsupported 汇总 |

`taxonomy` 输出目录：

| 文件 | 内容 |
|---|---|
| `taxonomy_summary.json` | API 数、case 数、unsupported category、recommended action 汇总、timings |
| `taxonomy_by_api.jsonl` | 每个 API 的支持状态、unsupported 分类、example/header/CMake 信号 |
| `taxonomy_by_op.jsonl` | 每个 op 的 API 和 unsupported 分类 |
| `taxonomy_priority.jsonl` | unsupported API 的后续 adapter 优先级 |

native cache 输出：

| 文件/目录 | 内容 |
|---|---|
| `build/st_accuracy/native_cache/<cache_key>/st_accuracy_aclnn_runner` | 按 compile context 复用的 C++ runner binary |
| `build/st_accuracy/native_cache/<cache_key>/manifest.json` | `compile_context` 和 `compile_command` |

## 5. 子模块说明

### 5.1 `run_st_accuracy.py`：CLI 与总编排

模块职责：

- 定义 `list`、`coverage`、`taxonomy`、`run` 四类命令。
- 统一执行 discovery/load/report gate。
- 在真实设备 `run` 中串联 selector、强制 package/install、support gate、native runner 和 report writer。
- 在 `run --no-execute` 中跳过 package/install，只写静态筛选结果。

输入：

- CLI 参数：`--ops`、`--apis`、`--case-ids`、`--dtype`、`--limit`、`--out`、`--vendor`、`--soc`、`--device`、`--json-pattern`、兼容参数 `--prepare`。
- 仓库根目录，由 `_repo_root()` 从脚本位置推导。
- `inventory.py` 与 `st_cases.py` 返回的数据。

输出：

- stdout：命令摘要或 discovery error JSON。
- 返回码：discovery gate 失败或 `run` 存在 `failed/error` 时返回非 0。
- 输出目录：交给 `coverage.py`、`taxonomy.py` 或 `report.py` 写文件。

边界：

- 它只负责编排，不直接理解单个 API 的 shape/dtype 约束。
- 显式 `--ops` 下，`missing_ops`、`zero_case_ops`、`skipped_assets` 会阻止 `run` 继续执行。

### 5.2 `inventory.py`：算子发现

模块职责：

- 在默认 category 中扫描可优化算子。
- 判断一个目录是否具备 AscendC AI Core 算子特征。

输入：

- 仓库根目录。
- category 列表；当前默认是 `("attention",)`。

判定条件：

- 目录存在 `CMakeLists.txt`。
- `op_host` 下存在 `*_def*.cpp`。
- `op_kernel` 下源码包含 AscendC 特征，如 `kernel_operator`、`__aicore__`、`GlobalTensor`、`TPipe`、`GET_TILING_DATA` 等。

输出：

- `list[Operator]`，其中 `Operator` 包含 `name`、`category`、`path`。

边界：

- 当前默认不是全仓库扫描；如果后续要覆盖 `ffn/gmm/mc2/moe/posembedding`，应先扩展 category 策略，并重新验证 discovery gate。

### 5.3 `st_cases.py`：ST case 装载与诊断

模块职责：

- 从每个 `Operator` 的 `tests/st/*/all_*.json` 装载 case。
- 将原始 JSON case 包装为 `StCase`。
- 对缺失目录、非法 JSON、空 payload、非法 case id 等资产问题生成诊断。

输入：

- `Operator` 列表。
- `json_pattern`，默认 `all_*.json`。
- 文件路径：`<op>/tests/st/<api_dir>/<json_pattern>`。

输出：

- `CaseLoadResult.cases`：合法 `StCase` 列表。
- `CaseLoadResult.skipped_assets`：`StAssetDiagnostic` 列表。

边界：

- 它只做装载和基本结构校验，不判断该 case 是否能由 native runner 执行。
- 它不修改原始 JSON。
- `StCase` 的缓存派生视图只在单个对象生命周期内生效，不跨命令落盘。

### 5.4 `adapter_utils.py`：Adapter 基类与共享约束工具

模块职责：

- 定义 `Adapter` 基类和通用 `supports()` 默认逻辑。
- 提供 dtype 集合、shape/numel/broadcast、scalar/attr 解析、range 判定等共享工具。

输入：

- `StCase`。
- `case.flat_inputs` 中的 tensor/scalar/attr 元数据。

输出：

- `Adapter.supports(case) -> (bool, reason)`。
- 各类 helper 的解析结果，例如 normalized shape、attr value、broadcast shape。

边界：

- support gate 的含义是“当前 native runner 能否合法执行并比较这个 case”，不是官方 API 的完整能力声明。
- 不能通过放宽 support gate 来隐藏 runner/golden 不支持的输入域。

### 5.5 `attention_adapters.py`：两个重点 attention API 的支持域

模块职责：

- 定义 `LightningIndexerAdapter`。
- 定义 `RecurrentGatedDeltaRuleAdapter`。
- 对当前两个重点 attention API 做严格 shape/dtype/layout/attr/optional input gate。

输入：

- `StCase`。
- named tensor：如 `query`、`key`、`weights`、`value`、`beta`、`stateRef`、`actualSeqLengths` 等。
- attrs：如 `layoutQuery`、`layoutKey`、`sparseCount`、`sparseMode`、`returnValues`、`scaleValue` 等。

输出：

- 支持时返回 `(True, "")`。
- 不支持时返回 `(False, reason)`，reason 会进入 `coverage`、`taxonomy` 或 `cases.jsonl`。

当前关键约束：

- `LightningIndexer` 支持 `BSND`、`TND`、`PA_BSND` 相关输入域，并要求 head dim 为 128、key N2 为 1、`sparseMode` 为 0 或 3。
- `RecurrentGatedDeltaRule` 要求核心输入为 bf16，辅助索引为 int32，`stateRef` shape 与 query/value 维度匹配，且当前 native golden 要求 `Nv` 可被 `Nk` 整除。

边界：

- 这里保护的是当前 C++ host golden 和 ACLNN runner 的可验证输入域。
- 新增 attention case 时，应先确认 adapter support gate 是否明确表达了输入域，不应直接让 native runner 执行未知域。

### 5.6 `adapters.py`：通用 adapter 与 registry facade

模块职责：

- 保存通用 elementwise/reduction/tensor-list/slice/cast 等 adapter。
- 从 `attention_adapters.py` 引入两个 attention adapter。
- 维护 `ADAPTERS` registry。
- 提供稳定外部入口 `get_adapter(aclnn_name)`。

输入：

- ACLNN API 名，例如 `aclnnLightningIndexer`、`LightningIndexer`、`Add`。
- `StCase`。

输出：

- `Adapter | None`。
- adapter support 判定结果。

边界：

- `get_adapter()` 会同时尝试原名和移除 `aclnn` 前缀后的名称。
- 该文件仍包含大量通用 adapter；P5 只完成了共享 helper 与 attention adapter 拆分，不代表所有 adapter 已模块化。

### 5.7 `runner.py`：case 选择、support gate 与执行分流

模块职责：

- `CaseSelector` 根据 CLI 条件筛选 case。
- `run_cases()` 执行 adapter lookup、support gate、`--no-execute` 分流和 native batch 调用。

输入：

- `Iterable[StCase]`。
- selector 条件：ops/apis/dtypes/case_ids/limit。
- runtime 参数：repo root、out dir、vendor、soc、device、execute flag。

输出：

- `list[CaseResult]`。

状态流转：

- 无 adapter：`unsupported`
- adapter 不支持：`unsupported`
- `--no-execute`：`skipped`
- native runner 返回：`passed`、`failed` 或 `error`

边界：

- `runner.py` 不生成 golden，也不调用 ACLNN。
- 真实设备执行只发生在 `NativeRunner.run_many()`。

### 5.8 `native.py`：Python 到 C++ native runner 的桥接层

模块职责：

- 维护 native runner 编译上下文和 cache key。
- 编译 `native/st_accuracy_aclnn_runner.cpp`。
- 将 selected executable case 写成 `native/batch_cases.tsv`。
- 调用 C++ runner 子进程。
- 解析 C++ stdout JSONL，并转换为 `CaseResult`。
- 收集 runner binary 和 op_api lib 的 load diagnostics。

输入：

- `list[tuple[StCase, Adapter]]`。
- `vendor`、`soc`、`device`。
- `ASCEND_HOME_PATH`。
- C++ source files：
  - `scripts/st_accuracy/native/st_accuracy_aclnn_runner.cpp`
  - `scripts/st_accuracy/native/attention_runner.inc`

输出：

- `native/batch_cases.tsv`。
- C++ runner binary：`build/st_accuracy/native_cache/<cache_key>/st_accuracy_aclnn_runner`。
- manifest：`build/st_accuracy/native_cache/<cache_key>/manifest.json`。
- `CaseResult` 列表。
- `summary.json.native` 诊断信息。

cache key 组成：

- source files sha256 聚合值。
- vendor。
- soc。
- compiler/std/opt 参数。
- Ascend include/lib 路径。
- vendor include/lib 路径。

op_api lib 选择：

1. 优先使用仓库内 `build/libcust_opapi.so`。
2. 否则使用 `${ASCEND_HOME_PATH}/opp/vendors/${vendor}_transformer/op_api/lib/libcust_opapi.so`。

边界：

- `summary.json.native.opapi_lib.source` 只能说明 `dlopen` 的 op_api lib 来自 build 还是 vendor，不能单独证明完整 package/load gate。
- 真实设备侧闭环仍需要结合 package 安装、vendor OPP 目录和 18 case golden compare 结果。

### 5.9 `native/st_accuracy_aclnn_runner.cpp`：C++ ACLNN batch runner

模块职责：

- 解析命令行和 batch TSV。
- 初始化 ACL runtime 和 stream。
- `dlopen` op_api 动态库并解析 ACLNN 符号。
- 生成 host 输入数据并 H2D。
- 调用 ACLNN 两段式接口。
- D2H 后执行 host golden compare。
- 将每个 case 的结果以 JSONL 输出到 stdout。

输入：

- CLI：
  - `--batch-file`
  - `--device`
  - `--opapi-lib`
- batch TSV，每行包含 `case_index`、`api`、dtype/shape/range/attr 等字段。
- custom package 中的 op_api 动态库。

输出：

- stdout JSONL，每行至少包含：
  - `case_index`
  - `status`
  - `message`
  - `duration_ms`
  - 可选 `timings`
- 进程返回码。

主要内部职责：

- `Args`：统一承载 parsed batch line 参数。
- `ParseBatchLine()`：把 Python TSV 协议转成 `Args`。
- `DispatchWithStream()`/`RunBatch()`：按 API 分发执行。
- `PrintResultJson()`/`PrintStageTimingsJson()`：输出 Python 可解析 JSON。
- 通用 golden：elementwise、reduction、tensor-list、slice、cast 等。
- fresh ACL child path：部分 API 如 `Cat/Stack/ChunkCat/SplitWithSize` 需要独立 ACL child 执行路径。

边界：

- P6 后仍保持单 translation unit 编译方式。
- `Args`、`ParseBatchLine()`、dispatch 和 JSON schema 仍是 Python/C++ 共享协议，后续若要重构必须单独设计协议边界。

### 5.10 `native/attention_runner.inc`：attention C++ golden/compare/runner

模块职责：

- 承载两个重点 attention API 的 C++ 实现：
  - `LightningIndexer`
  - `RecurrentGatedDeltaRule`
- 生成 host golden。
- 调用 attention ACLNN API。
- 比较 attention 输出。
- 输出 attention 内部分段耗时。

输入：

- `Args`，来自主 cpp 的 `ParseBatchLine()`。
- ACL stream。
- 已加载的 ACLNN 函数指针。

输出：

- `NativeResult`。
- `NativeStageTimings` 中的 parse/input/golden/workspace/launch/D2H/compare。

边界：

- 该文件通过 `#include "attention_runner.inc"` 进入主 cpp，不单独编译。
- 它依赖主 cpp 中定义的通用类型、tensor holder、ACL helper、JSON 输出和 dispatch 结构。
- P6 只完成 attention 区段拆分，未改变 TSV、JSON、dispatch 或 native batch 协议。

### 5.11 `coverage.py`：可执行覆盖率报告

模块职责：

- 按 op/API 统计当前 ST case 中哪些可由 native runner 执行。
- 汇总 unsupported reason 和 zero-sized case。

输入：

- `Operator` 列表。
- `StCase` 列表。
- `get_adapter(case.aclnn_name).supports(case)`。

输出：

- `coverage_summary.json`
- `coverage_by_op.jsonl`
- `coverage_by_api.jsonl`

边界：

- 它不执行设备。
- 它的 `executable` 只表示 adapter support gate 通过，不表示精度 passed。

### 5.12 `taxonomy.py`：unsupported 分类与 adapter 优先级

模块职责：

- 将 unsupported reason 归类为 `missing_adapter`、`dtype_gate`、`shape_gate`、`attr_gate`、`range_gate`、`package_soc_unsupported` 等。
- 检查 API 是否存在 example、op_api header、CMake 支持信号。
- 给 unsupported API 生成 recommended action。

输入：

- `Operator` 列表。
- `StCase` 列表。
- 仓库源码中的 examples、op_api headers、CMakeLists。
- adapter support 判定结果。

输出：

- `taxonomy_summary.json`
- `taxonomy_by_api.jsonl`
- `taxonomy_by_op.jsonl`
- `taxonomy_priority.jsonl`

recommended action 示例：

- `covered`
- `implement_adapter_from_examples`
- `extend_existing_adapter`
- `inspect_api_assets`
- `review_package_or_soc_support`
- `documented_exclusion`

边界：

- 它用于规划后续 adapter/native runner 扩展，不是精度验证命令。

### 5.13 `report.py`：run 结果落盘

模块职责：

- 将 `CaseResult` 写入 `cases.jsonl`。
- 汇总状态计数写入 `summary.json`。
- 合并 timings 和 native load diagnostics。

输入：

- `Iterable[CaseResult]`。
- `TimingCollector`。
- `run_metadata`，当前主要包含 `native`。

输出：

- `cases.jsonl`
- `summary.json`

状态字段：

- `passed`
- `failed`
- `error`
- `unsupported`
- `skipped`

边界：

- 它只负责结果序列化，不重新解释结果含义。

### 5.14 `timing.py`：阶段耗时采集

模块职责：

- 提供 `TimingCollector.measure(key)` 上下文管理器。
- 对同名 key 累加耗时。
- 输出三位小数的毫秒级 timing dict，并补充 `total_ms`。

输入：

- 被测代码块。
- timing key。

输出：

- `dict[str, float]`。

边界：

- 它记录的是框架阶段耗时，不是硬件 profiling。
- 设备算子内部性能分析仍应使用 profiling 工具，不能只依赖这里的 Python timing。

### 5.15 `prepare.py`：package 构建与安装辅助

模块职责：

- 在真实设备侧 `run` 时执行强制 package build。
- 找到 `build_out/cann-ops-transformer-<vendor>_linux-*.run` 并安装。

输入：

- `ops`
- `soc`
- `vendor`
- `jobs`
- `opt_level`

执行命令形态：

```text
bash build.sh --pkg --soc=<soc> --ops=<ops> --vendor_name=<vendor> -j<jobs> -O<opt_level>
build_out/cann-ops-transformer-<vendor>_linux-*.run
```

输出：

- 构建产物。
- 已安装 custom package。

边界：

- package op 列表来自 selected case 的 op 集合；如果筛选后没有 case，则不会触发 package/install。
- `--no-execute` 不触发 package/install。
- `--prepare` 当前只作为兼容旧脚本的参数保留，不能绕过真实设备 `run` 的强制安装。
- 安装是否完整仍需要通过后续设备侧 `run` 和 golden compare 验证。

## 6. 当前协议边界

### 6.1 Python -> C++ batch TSV

`native.py` 将每个 executable case 写成一行 key-value TSV。基础字段包括：

- `case_index`
- `api`
- `dtype`
- shape/range 字段
- scalar/attr 字段

两个 attention API 使用专用字段，例如：

- `LightningIndexer`：`query_shape`、`key_shape`、`weights_shape`、`layout_query`、`layout_key`、`sparse_count`、`sparse_mode`、`return_values`、`actual_seq_lengths_query`、`actual_seq_lengths_key`、`block_table` 等。
- `RecurrentGatedDeltaRule`：`query_shape`、`key_shape`、`value_shape`、`beta_shape`、`state_shape`、`actual_seq_lengths`、`ssm_state_indices`、`num_accepted_tokens`、`scale_value`、`g_shape`、`gk_shape` 等。

该协议当前是隐式共享协议：Python 负责写，C++ `ParseBatchLine()` 负责读。后续如果继续 P6 深水区，应先设计协议边界，不应顺手改字段名。

### 6.2 C++ -> Python JSONL

C++ runner 每行输出一个 JSON object，Python 按 `case_index` 归还给原 case。核心字段包括：

- `case_index`
- `status`
- `message`
- `duration_ms`
- `timings`

Python 会把除 `status/case_index/duration_ms/message` 外的字段放入 `CaseResult.extra`。

### 6.3 status 语义

| status | 来源 | 含义 |
|---|---|---|
| `passed` | C++ runner | 设备输出与 host golden compare 通过 |
| `failed` | C++ runner 或 native 输出异常 | 设备执行完成但结果不符合预期，或 native 正常退出但没有有效输出 |
| `error` | Python/native 异常 | 编译、子进程、JSON 解析、ACL 初始化等异常 |
| `unsupported` | adapter support gate | 当前 native runner 不支持该 case 输入域 |
| `skipped` | `--no-execute` | 明确禁止执行设备侧 |

## 7. P0-P6 优化后的当前能力

### 7.1 已完成能力

- discovery gate 能阻止显式 `--ops` 下的假成功。
- timing 能覆盖 Python 静态链路、native bridge、report 写入和两个 attention API 的 C++ 内部阶段。
- `StCase` 派生视图与 coverage/taxonomy support 判定已避免明显重复计算。
- native runner binary 已按 compile context 进入跨输出目录 cache。
- 真实设备侧 `run` 已强制重新 package/install selected ops，避免使用旧 custom package 进行精度测试。
- run summary 已记录 runner binary 与 op_api lib 诊断。
- adapter 基础结构已拆分出共享 helper 和 attention adapter。
- attention C++ runner 区段已从主 cpp 拆入 include 文件。

### 7.2 当前验证状态

当前最近一轮 P6 闭环验证：

- `python3 -m unittest tests/st_accuracy/test_transformer_st_accuracy.py`：30 个测试通过。
- `python3 -m py_compile scripts/st_accuracy/native.py scripts/st_accuracy/run_st_accuracy.py scripts/st_accuracy/runner.py scripts/st_accuracy/adapters.py scripts/st_accuracy/adapter_utils.py scripts/st_accuracy/attention_adapters.py`：通过。
- `coverage --ops lightning_indexer,recurrent_gated_delta_rule`：18 executable、0 unsupported。
- `taxonomy --ops lightning_indexer,recurrent_gated_delta_rule`：18 cases、0 unsupported。
- `run --no-execute --ops lightning_indexer,recurrent_gated_delta_rule`：写出 18 result rows。
- 强制 package/install 设备侧验证 `build/st_accuracy/force_pkg_device`：先执行 `build.sh --pkg` 并安装 `custom_transformer`，`prepare_pkg_ms=104151.404`；随后 18 passed、0 failed、0 error、0 unsupported。
- 设备侧冷编译：18 passed、0 failed、0 error、0 unsupported，`cache_key=b251c21a1d70a8f7ba89dd56`，`native_build_ms=94539.489`。
- 设备侧 cache 复用：18 passed、0 failed、0 error、0 unsupported，同一 `cache_key=b251c21a1d70a8f7ba89dd56`，`native_build_ms=0.048`。
- `summary.json.native.runner_binary.source_files` 已覆盖主 cpp 与 `attention_runner.inc`。

### 7.3 仍需注意的边界

- 当前默认 discovery scope 仍是 `attention/`，不是全仓库所有 domain。
- 当前真实设备侧重点覆盖两个 attention 算子，不代表所有 adapter registry 中的 API 都已在本仓库设备侧闭环。
- C++ 主 runner 仍较大，`Args`、`ParseBatchLine()`、dispatch、JSON 输出仍集中在主 cpp 中。
- `coverage/taxonomy` 是规划和静态分析产物，不能替代设备侧 `run`。
- `summary.json.native.opapi_lib` 是 load 诊断，不是完整 package/load gate 的唯一证据。

## 8. 如果要判断是否符合需求

如果你的需求是“稳定验证当前两个重点 attention 算子的本地 ST 精度”，当前框架已经具备闭环能力：

- 能发现目标算子和 18 个 case。
- 能在执行前给出明确 support gate。
- 能设备侧调用 custom package。
- 能做 host golden compare。
- 能输出 per-case 结果、summary、timing 和 native load 诊断。
- 能复用 C++ runner cache，避免每个新 `--out` 重编译。

如果你的需求是“扩展为全仓库 ST 精度框架”，还需要额外工作：

- 扩展 `inventory.DEFAULT_CATEGORIES` 或引入可配置 category。
- 为更多 API 建立 adapter support gate。
- 为更多 API 在 C++ runner 中实现 batch 协议、ACLNN 调用和 host golden compare。
- 为新增 API 建立 coverage/taxonomy/no-execute/device 四层验证。
- 增强 package/load gate，明确区分 build lib、vendor lib、kernel object、op info 和 runtime config 是否均来自当前产物。

如果你的需求是“把 P6 继续做深”，建议顺序是：

1. 先设计 `Args` 与 `ParseBatchLine()` 的协议边界。
2. 再评估是否需要 C++ 层可独立编译的协议/golden 单测。
3. 最后再拆通用 math runner 或改多 translation unit 编译方式。

当前不建议直接做的事情：

- 不建议在没有新需求的情况下扩大 default discovery scope。
- 不建议为了减少 unsupported 放宽 adapter gate。
- 不建议为了结构拆分同步修改 JSON/timing 字段名。
- 不建议把 native runner 拆成多个 translation unit，除非先补齐编译和链接验证设计。
