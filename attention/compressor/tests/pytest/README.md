# Compressor算子测试框架

## 功能说明

基于pytest测试框架，实现Compressor算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能
- **双模式执行隔离**：支持直接pytest多进程执行和shell层进程隔离两种批量模式
- **性能采集**：支持挂载msprof采集算子性能数据并汇总输出

## 当前实现范围

### 参数限制

- 支持D为128/512。
- 支持H为1K~10K，512对齐。
- 支持cmp_ratio为2、4、8、16、32、64、128。
- 当cmp_ratio为4/128时，支持如下三种情况：
    - C4A: D=512, coff=2, cmp_ratio=4;
    - C4Li: D=128, coff=2, cmp_ratio=4;
    - C128A: D=512, coff=1, cmp_ratio=128。

### 环境配置

#### 前置要求

1、 确认TorchNPU为最新版本  

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                                  # 执行脚本，支持single/batch两种命令
- batch_isolated_run.sh                        # 批量隔离执行脚本（shell层进程隔离+msprof性能采集）
- compressor_golden.py                         # cpu侧算子golden实现以及精度对比
- collect_perf_data.py                         # msprof性能数据收集与汇总
- pytest.ini                                   # 创建测试标记

单用例测试：

- test_compressor_single.py                    # 测试单用例运行主程序
- compressor_operator_single.py                # CPU侧算子逻辑实现获取golden与npu算子直调
- test_compressor_paramset.py                  # 单用例入参配置

批量测试：

- test_compressor_batch.py                     # 用例批量测试主程序并生成excel文件保存结果（支持隔离模式）
- ./batch/compressor_pt_loadprocess.py         # 读取pt文件并调用算子获取npu输出
- ./batch/compressor_pt_save.py                # 读取excel表格批量生成用例pt文件
- ./batch/replace_path.py                      # test_compressor_batch.py占位符替换

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_compressor_paramset.py的ENABLED_PARAMS参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### 用例的批量生成与测试

##### 方式A：test_run.sh 批量执行

1、excel路径下存放用例excel表格

2、test_run.sh中设置读取的用例excel表格路径（PATH1）和pt文件存放路径（PATH2）

3、执行指令：

``` bash
bash test_run.sh batch
```

##### 方式B：手工分步执行

1、生成pt文件：

``` bash
python3 batch/compressor_pt_save.py "excel/*.xlsx" pt_path
```

2、替换测试脚本路径：

``` bash
python3 batch/replace_path.py test_compressor_batch.py pt_path
```

3、执行测试：

``` bash
python3 -m pytest -rA -s test_compressor_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

4、恢复测试脚本：

``` bash
cp test_compressor_batch.py.bak test_compressor_batch.py
```

##### 方式C：批量隔离执行（推荐用于性能采集）

对每条用例单独拉起一个pytest进程，实现进程间完全隔离，避免单条用例崩溃影响其他用例。
用例获取逻辑：优先从 `./excel` 目录读取Excel文件的 `Testcase_Name` 列映射为 `.pt` 文件路径；若无Excel文件或缺少 `Testcase_Name` 列，则自动回退为扫描用例目录下所有 `.pt` 文件。

``` bash
bash batch_isolated_run.sh ./pt_path 0         # 不采集性能
bash batch_isolated_run.sh ./pt_path 1         # 采集性能（挂载msprof）
```

## Excel 用例表格式

`excel/test_cases.xlsx` 的 Sheet1 需包含以下列：

| 列名 | 类型 | 示例 | 说明 |
|---|---|---|---|
| Testcase_Name | str | `Prefill0` | 用例名，也作为生成的.pt文件名 |
| batch_size | int | `1` | |
| hidden_size | int | `4096` | |
| Seq_len | int | `8192` | |
| head_dim | int | `128` / `512` | |
| block_size | int | `128` | |
| cmp_ratio | int | `4` / `128` | |
| coff | int | `1` / `2` | |
| start_p | int | `0` / `8192` | |
| cache_mode | int | `1` / `2` | 1:连续buffer, 2:循环buffer |
| layout_x | str | `TH` / `BSH` | |
| data_type | str | `BF16` / `FP16` | |
| cu_seqlens | None/str | `None` 或 `"[0,8192]"` | TH布局必填 |
| seqused | None/str | `None` 或 `"[8192]"` | |
| start_pos | None/str | `None` 或 `"[0]"` | |
| x_datarange | str | `"[-10,10]"` | |
| wkv_datarange | str | `"[-10,10]"` | |
| wgate_datarange | str | `"[-10,10]"` | |
| ape_datarange | str | `"[-10,10]"` | |
| kv_state_datarange | str | `"[-10,10]"` | |
| score_state_datarange | str | `"[-10,10]"` | |

**注意事项**：

- `layout_x=TH` 时，`cu_seqlens` 为必填项（长度=batch_size+1的列表）

## 输出文件

| 文件 | 说明 |
|---|---|
| `result.xlsx` | 测试结果（精度、参数等） |
| `result_perf.xlsx` | 测试结果 + 性能数据（仅msprof模式） |
| `batch_summary.log` | 批量隔离执行详细日志 |
| `batch_fail_list.log` | 失败用例清单 |
| `PROF_*/` | msprof性能原始数据目录 |

## Pro0921 latent grouped 用例（新增）

正式 Compressor 增加了 `x_cache` / `x_cache_slot_mapping` / `x_cache_update` grouped
输入。原来的普通路径与新的 grouped 路径**显式二选一**，不使用异常回退或旧接口动态兼容。

### 用例契约

- `path=normal`：关闭 grouped，不传 `x_cache`，语义与既有普通路径一致。
- `path=grouped`：必须携带 `x_cache_update`（布尔）以及夹具生成的 `x_cache` /
  `x_cache_slot_mapping`。缺失 grouped 字段直接报错，绝不静默当作 normal。

用例集是纯 JSON 文件（`contract=pro0921-grouped-compressor-v1`），新增用例无需修改 Excel。

### 直接可执行入口

```bash
# 1) 物化冻结的 B1 用例集（夹具 + golden + cases.json），纯 numpy，不碰设备
python3 pro0921_grouped_cases.py --emit /tmp/pro0921_compressor_cases

# 2) 板上精度：退出码非 0 表示不通过；无 NPU 时明确失败，不产生伪结果
python3 pro0921_grouped_cases.py --run /tmp/pro0921_compressor_cases/cases.json

# 3) 只看计划（不碰设备）
python3 pro0921_grouped_cases.py --plan

# 4) 生成 test_compressor_batch.py 可消费的 .pt（自动带 grouped 字段）
python3 pro0921_grouped_cases.py --emit-pt /tmp/pro0921_compressor_pt
```

`compressor_golden.py` 也提供显式 grouped 入口，普通 CLI 语义不变：

```bash
python3 compressor_golden.py --grouped_cases cases.json --emit /tmp/pro0921_compressor_cases
```

批量入口 `batch/compressor_pt_save.py`：

```bash
# JSON 用例集入口（不要求手改 Excel）
python3 batch/compressor_pt_save.py --cases-json cases.json /tmp/pro0921_compressor_pt

# Excel 入口：新增可选的 path / case_variant 列；缺列时全部按既有 normal 处理
python3 batch/compressor_pt_save.py excel/test_cases.xlsx pt_path
```

`path=grouped` 的 Excel 行必须给出 `case_variant`（`boundary`/`update_only`/
`coff2_boundary`/`li_read_shared`），且 `cmp_ratio`/`coff`/`head_dim`/`hidden_size`/
`cache_mode` 要与该 variant 的冻结几何一致，否则报错。`batch/compressor_pt_loadprocess.py`
按 `.pt` 内的 `path` 字段显式选择普通或 grouped 调用。

**`.pt` 路径契约（无兼容回退）**：当前 writer 生成的每个 `.pt` 都带显式 `path`
（`normal`/`grouped`）。Excel 无 `path` 列时是该列在**生成阶段**默认写 `normal`，
不是读取旧 `.pt` 时的运行期回退。`pt_grouped_args` 要求 `path` 显式存在，缺失即报错；
历史无 `path` 的旧 `.pt` 不支持直接运行，请用当前 writer 重新生成。grouped `.pt`
额外带完整 `expected_x_cache`，`batch/compressor_pt_loadprocess.py` 调用后用同步读回
与该 golden 精确比较（bf16→fp32），并对全量 `state_cache` 与独立 golden 比较，失败直接
`raise AssertionError`。

### 冻结 B1 覆盖

| 用例 | path | R / coff | 覆盖 |
| --- | --- | --- | --- |
| `hca_normal_r128` | normal | 128 / 1 | 开关关闭、普通路径 |
| `hca_grouped_boundary` | grouped | 128 / 1 | 边界投影 |
| `hca_grouped_update_only` | grouped | 128 / 1 | 非边界只更新，state 不变 |
| `csa_grouped_coff2_boundary` | grouped | 4 / 2 | coff2 上一组状态取自原 state |
| `csa_grouped_li_read_shared` | grouped | 4 / 2 | LI 只读共享缓存（`x_cache_update=false`） |

精度对比使用 `../pro0921_latent_smoke/latent_ref.py` 独立 numpy golden，并用
`independent_contract_check` 以另一条纯 numpy 路径二次重算，避免两条实现互比；比较时初始
state 与逻辑历史一致；边界与非边界数学语义不同，不要求二者相等。

### 板上精度/性能配对 runner

```bash
python3 pro0921_accuracy_perf.py --mode plan
python3 pro0921_accuracy_perf.py --mode accuracy --accuracy-out /tmp/pro0921_accuracy.json
python3 pro0921_accuracy_perf.py --mode perf --perf-out /tmp/pro0921_perf.json \
    --warmup 5 --repeats 20
```

- 功能与性能写入**不同文件**（`--accuracy-out` / `--perf-out`），不混同。
- 固定 seed/dtype/shape，显式 `--warmup` / `--repeats`；每次 trial 从 pristine 设备副本
  重置 ring/state，计时区间不含 reset 与 H2D；用 `torch.npu.Event` + `synchronize()` 计时，
  记录 median/min 及 raw samples。
- 12 场景矩阵：`geom ∈ {csa(R=4,coff=2), hca(R=128,coff=1)}` × `path ∈ {normal,grouped}`
  × `kind ∈ {boundary,nonboundary,cycle}`。
  - `boundary`：normal/grouped **两条路径都严格 1 次 S1 调用**（normal 用计时外预热的
    state 读回、grouped 用 ring 读回），不再把整周期与单 token 混比；
  - `nonboundary`：两条路径都 1 次 S1 调用，且只预热本 token 之前的 prefix，不填未来
    token 伪造历史；
  - `cycle`：两条路径都 `ratio` 次逐 token 调用。
- 配对加速比在相除前断言 scope 相等（kind、tokens/calls、shape、seed），不只比 geom/kind 标签；
  golden 最终结果与 fullgroup 独立 reference 比较，normal/state 各自比。
- 每个场景计时前的功能验证对副作用做逐元素检查：`x_cache` 在缓存 dtype 上全量精确相等，
  `state_cache` 用 pristine 区分未改区域（全量精确）与更新区域（逐元素 `isclose`），
  不使用输出投影的整块百分比容差，单个元素损坏或整 token 漏写一定判失败。
- 无 NPU 时明确失败（退出码 2）或只 `--mode plan`，绝不生成伪 perf 数值。

### 板上必要前置

1. 安装 torch → torch_npu → 本任务正式 provider（`cann_ops_transformer`）并保持导入顺序。
2. 安装当前双仓算子与 wheel；仅安装成功不证明 `libc10.so` 运行时可加载。
3. 性能 shape 由固定的 12 场景矩阵决定（无 `--hidden`/`--head` 覆盖），按原线程 profile
   在板上运行；本地只做聚焦 CPU 测试。

### 聚焦 CPU 测试

```bash
python3 -m pytest -q tests/test_pro0921_grouped_accuracy_cases.py
```

覆盖契约校验、独立 golden、cycle state/reset、边界/非边界等价、计时聚合与 `.pt` 契约。
