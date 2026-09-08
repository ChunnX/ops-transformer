# mixed_quant_sparse_flash_mla算子测试框架

## 功能说明

基于pytest测试框架，实现mixed_quant_sparse_flash_mla算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调、图模式调用获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- **数据格式**:
    - **layout_q**：BSND、TND
    - **layout_kv**：PA_BBND

### 环境配置

#### 前置要求

1、 确认TorchNPU为最新版本

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                                               # 用例执行脚本
- mixed_quant_sparse_flash_mla_golden.py                   # 算子入参处理及cpu侧golden实现
- result_compare_method.py                                  # cpu golden与npu结果精度对比脚本
- pytest.ini                                                # 创建测试标记

单用例测试：

- test_mixed_quant_sparse_flash_mla_single.py              # pytest测试单用例运行主程序
- mixed_quant_sparse_flash_mla_paramset.py                 # 单用例入参数配置

批量测试：

- test_mixed_quant_sparse_flash_mla_batch.py               # 用例批量测试主程序，读取pt用例并获取npu输出，结果保存至excel文件
- ./batch/test_mixed_quant_sparse_flash_mla_pt_save.py     # 读取excel表格批量生成用例pt文件
- ./batch/mixed_quant_sparse_flash_mla_process.py          # 调用算子接口获取npu输出

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例测试

1、手动配置mixed_quant_sparse_flash_mla_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single --pt-dir mqsmla_testcase
```

`single` 仅生成并保存算子输入，不计算 CPU golden，也不执行 NPU。保存的 `.pt` 文件包含
`params`、`metadata_input` 和 `op_input`，不包含 `golden_state`、`cpu_output` 或 `cpu_lse`。
直接运行 `test_mixed_quant_sparse_flash_mla_single.py` 时默认行为相同。

执行保存的输入：

``` bash
bash test_run.sh batch_exec --mode 0 --pt-dir mqsmla_testcase
# Graph模式：
bash test_run.sh batch_exec_graph --mode 0 --pt-dir mqsmla_testcase
```

执行时默认跳过输出/LSE精度对比和 batch consistency 对比，只检查 NPU 是否正常完成。
结果表记录 `Executed (accuracy skipped)`，`fulfill_percent` 留空；NPU执行错误仍使pytest失败。
对于包含 CPU golden 的旧用例，可设置 `MQSMLA_SKIP_ACCURACY=0` 恢复原精度/一致性校验流程。

#### 单用例性能测试

配置 `mixed_quant_sparse_flash_mla_paramset.py` 中的 `ENABLED_PARAMS`。
如需性能基线对比，可在本目录准备与当前用例顺序一致的 `perf_golden.xlsx`
（包含 `metadata_duration` 和 `qsmla_duration` 列，单位为微秒）；没有该文件时仅生成性能报告。
在已配置 CANN、`msprof` 和 Python 测试依赖的环境中执行：

``` bash
bash perf_run.sh
```

脚本通过 `msprof` 运行 `test_mixed_quant_sparse_flash_mla_single.py`，设置
`MQSMLA_INPUTS_ONLY=0`，生成并保存输入后执行 NPU，不计算 CPU golden、不进行精度对比。
无需预先生成 `.pt` 文件。性能测试固定关闭 batch consistency，保证每个用例仅执行一组
metadata / attention 算子。默认输出 `res.log`、`result.xlsx` 和 `result_perf.xlsx`；
可通过 `MQSMLA_RESULT_SAVE_PATH` 修改结果路径，相对路径以本脚本所在目录为准。
结果表每次运行前清空，性能报告成功保存后删除本次处理的 `PROF*` 目录。
处理程序读取首个匹配的 `PROF*` 目录，运行前请将历史 profiling 目录移出本目录。

如果已采集性能，仅因缺少基线文件导致报告生成失败，可保留现有 `PROF*` 目录并直接执行，
无需重新运行 NPU：

``` bash
python3 process_perf_data.py --test_result_path=result.xlsx
```

#### 批量测试

支持三种批量测试模式，均可通过选项指定excel文件、sheet名、pt保存目录：

| 命令 | 功能 |
|------|------|
| `batch_save` | 从excel读取用例 → golden计算 → 保存pt文件 |
| `batch_exec` | 读取single/batch_save生成的pt文件 → NPU执行，默认跳过精度对比 |
| `batch` | 全流程：`batch_save` + `batch_exec`，默认完成后清理pt文件 |

##### 选项参数

| 选项 | 说明 | 默认值 | 适用命令 |
|------|------|--------|---------|
| `--excel <路径>` | 指定excel文件路径 | `./excel/example.xlsx` | batch_save / batch_exec / batch |
| `--sheet <名称>` | 指定excel sheet名 | `decode` | batch_save / batch |
| `--pt-dir <目录>` | 指定pt文件保存/读取目录 | `mqsmla_testcase` | single / batch_save / batch_exec / batch |
| `--keep-pt` | 执行完成后保留pt文件 | 默认清理 | 仅batch |

##### 使用示例

``` bash
# batch_save：只生成pt文件
bash test_run.sh batch_save
bash test_run.sh batch_save --excel my.xlsx --sheet prefill --pt-dir my_pt

# batch_exec：只执行NPU测试（需先batch_save生成pt文件）
bash test_run.sh batch_exec
bash test_run.sh batch_exec --pt-dir my_pt

# batch：全流程（默认清理pt文件）
bash test_run.sh batch

# batch：全流程，保留pt文件
bash test_run.sh batch --keep-pt

# batch：全流程，自定义所有参数
bash test_run.sh batch --excel my.xlsx --sheet prefill --pt-dir my_pt --keep-pt
```
