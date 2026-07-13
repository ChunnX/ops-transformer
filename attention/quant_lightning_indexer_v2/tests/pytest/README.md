# quant_lightning_indexer_v2算子测试框架
## 功能说明
基于pytest测试框架，实现quant_lightning_indexer_v2算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能
- **双模式执行隔离**：支持直接pytest多进程执行和shell层进程隔离两种批量模式
- **性能采集**：支持挂载msprof采集算子性能数据并汇总输出
- **运行模式切换**：支持eager直接调用和graph（torch.compile + torchair）两种算子调用模式

## 当前实现范围
### 参数限制

- **数据格式**:
  - **query_layout**：BSND、TND
  - **key_layout**: PA_BBND、BSND、TND

- **数据类型**:
  - **qk_dtype**: FLOAT8_E4M3FN、INT8、HIFLOAT8
  - **dequant_dtype**: FP32（Ascend950）、FP16（Ascend910_93）
  - **actual_seq_dtype**: INT32

- **运行模式**:
  - **eager**：直接调用 `torch.ops.cann_ops_transformer.quant_lightning_indexer`
  - **graph**：通过 `torch.compile` + `torchair` 后端编译执行（需torchair支持）

### 环境配置

#### 前置要求
1、 确认torch_npu为最新版本
2、 激活CANN包和自定义算子包
3、 graph模式需要安装torchair编译器后端

#### custom包调用
支持custom包调用

## 文件结构
#### pytest文件结构说明
- test_run.sh                                  # 执行脚本，支持single/batch两种命令
- batch_isolated_run.sh                        # 批量隔离执行脚本（shell层进程隔离+msprof性能采集）
- quant_lightning_indexer_v2_golden.py         # cpu侧算子golden实现
- quant_lightning_indexer_v2_acl_graph.py      # graph模式torchair后端实现
- result_compare_method.py                     # cpu golden与npu输出精度对比
- collect_perf_data.py                         # msprof性能数据收集与汇总
- pytest.ini                                   # 创建测试标记

单用例测试：
- test_quant_lightning_indexer_v2_single.py    # pytest测试单用例运行主程序
- test_quant_lightning_indexer_v2_paramset.py  # 单用例入参配置，按芯片型号自动选择用例

批量测试：
- test_quant_lightning_indexer_v2_batch.py     # 用例批量测试主程序并生成excel文件保存结果
- ./batch/quant_lightning_indexer_v2_pt_loadprocess.py   # 读取pt文件并调用算子获取npu输出
- ./batch/quant_lightning_indexer_v2_pt_save.py          # 读取excel表格批量生成用例pt文件
- ./batch/replace_path.py                                # test_quant_lightning_indexer_v2_batch.py占位符替换


## 架构说明

- **single 模式**：`qliv2_output_acl_graph` 调用 `qliv2_output_single(is_batch=True)` 即时生成数据 → `torch.compile` + `torchair` 执行
- **batch 模式**：`qliv2_output_acl_graph_from_pt` 直接从 .pt 文件读取 pre-computed 数据 → 共用同一 `torch.compile` 路径，跳过重复生成
- 两路共用 `_qliv2_prepare_tensors_and_metadata` 和 `_qliv2_run_compiled_graph`，统一使用 `fullgraph=False`

## 使用方法
在pytest文件夹路径下执行：

### 运行测试用例
#### 单用例调测
1、手动配置test_quant_lightning_indexer_v2_paramset.py的ENABLED_PARAMS参数

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
bash test_run.sh batch          # eager模式（默认）
bash test_run.sh batch eager    # 显式指定eager模式
bash test_run.sh batch graph    # graph模式
```

##### 方式B：手工分步执行
1、生成pt文件：
``` bash
python3 batch/quant_lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path
```

2、替换测试脚本路径：
``` bash
python3 batch/replace_path.py test_quant_lightning_indexer_v2_batch.py pt_path
```

3、执行测试：
``` bash
python3 -m pytest -rA -s test_quant_lightning_indexer_v2_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

4、恢复测试脚本：
``` bash
cp test_quant_lightning_indexer_v2_batch.py.bak test_quant_lightning_indexer_v2_batch.py
```

##### 方式C：批量隔离执行（推荐用于性能采集）
对每条用例单独拉起一个pytest进程，实现进程间完全隔离，避免单条用例崩溃影响其他用例。
``` bash
bash batch_isolated_run.sh ./pt_path 0         # 不采集性能
bash batch_isolated_run.sh ./pt_path 1         # 采集性能（挂载msprof）
bash batch_isolated_run.sh ./pt_path 0 graph   # graph模式 + 不采集性能
bash batch_isolated_run.sh ./pt_path 1 graph   # graph模式 + 性能采集
```

## Excel 用例表格式

`excel/test_cases.xlsx` 需包含以下列（Sheet1）：

| 列名 | 类型 | 示例 |
|---|---|---|
| Testcase_Name | str | `test_case_01` |
| batch_size | int | `8` |
| q_seq | int | `15` |
| k_seq | int | `111` |
| q_t_size | int | `8` |
| k_t_size | int | `15` |
| q_head_num | int | `64` |
| k_head_num | int | `1` |
| head_dim | int | `128` |
| block_size | int | `512` |
| block_num | int | `8` |
| qk_dtype | str | `FLOAT8_E4M3FN` / `INT8` / `HIFLOAT8` |
| dequant_dtype | str | `FP32` / `FP16` |
| actual_seq_dtype | str | `INT32` |
| cu_seqlens_q | None/str | `None` 或 `"[0, 1]"` |
| cu_seqlens_k | None/str | `None` 或 `"[0, 1]"` |
| seqused_q | None/str | `None` 或 `"[3,3,3,3,3,3,3,3]"` |
| seqused_k | str | `"[28,24,80,96,47,76,0,111]"` |
| cmp_residual_k | None/str | `None` 或 `"[0,0,0,0,0,0,0,0]"`（cmp_ratio>1时必填）|
| max_seqlen_q | int | `-1` |
| quant_mode | int | `1` / `2` / `4` |
| layout_query | str | `BSND` / `TND` |
| layout_key | str | `PA_BBND` |
| sparse_count | int | `512` |
| sparse_mode | int | `0` / `3` |
| query_datarange | str | `"[-448,448]"` |
| key_datarange | str | `"[-20,20]"` |
| weights_datarange | str | `"[-123,123]"` |
| q_scale_datarange | str | `"[0,255]"` |
| k_scale_datarange | str | `"[0,65504]"` |
| cmp_ratio | int | `1` / `4` |
| return_value | int | `0` / `1` |
| output_idx_offset | None/str | `None` 或列表字符串 |

**注意事项**：
- `dequant_dtype`：Ascend950仅支持`FP32`，Ascend910_93支持`FP16`
- `cmp_ratio > 1`且`sparse_mode != 0`时，`cmp_residual_k`必填（长度=batch_size的列表）
- `return_value=1`时，`output_idx_offset`需提供有效值
- Ascend910_93要求`quant_mode=2`

## 输出文件

| 文件 | 说明 |
|---|---|
| `result.xlsx` | 测试结果（精度、参数等） |
| `result_perf.xlsx` | 测试结果 + 性能数据（仅msprof模式） |
| `batch_summary.log` | 批量执行详细日志 |
| `batch_fail_list.log` | 失败用例清单 |
| `PROF_*/` | msprof性能原始数据目录 |
