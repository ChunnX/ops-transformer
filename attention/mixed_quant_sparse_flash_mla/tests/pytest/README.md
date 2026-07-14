# mixed_quant_sparse_flash_mla算子测试框架
## 功能说明
基于pytest测试框架，实现mixed_quant_sparse_flash_mla算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调、图模式调用获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围
### 参数限制

- **数据格式**:
    - **layout_q**：BSND、TND
    - **layout_kv**：PA_BBND

### 环境配置

#### 前置要求
1、 确认torch_npu为最新版本

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

性能测试：
- perf_run.sh                                               # 性能数据分析脚本（单进程msprof采集）
- process_perf_data.py                                      # 单PROF文件夹性能数据解析脚本
- batch_isolated_run.sh                                     # 批量隔离执行脚本，逐用例独立进程+msprof采集
- collect_perf_data.py                                      # 多PROF文件夹性能数据批量收集脚本

## 使用方法
在pytest文件夹路径下执行：

### 运行测试用例
#### 单用例测试
1、手动配置mixed_quant_sparse_flash_mla_paramset.py的参数

2、执行指令：
``` bash
bash test_run.sh single
```
#### 批量测试

支持三种批量测试模式，均可通过选项指定excel文件、sheet名、pt保存目录：

| 命令 | 功能 |
|------|------|
| `batch_save` | 从excel读取用例 → golden计算 → 保存pt文件 |
| `batch_exec` | 读取pt文件 → NPU测试（需先`batch_save`） |
| `batch` | 全流程：`batch_save` + `batch_exec`，默认完成后清理pt文件 |

##### 选项参数

| 选项 | 说明 | 默认值 | 适用命令 |
|------|------|--------|---------|
| `--excel <路径>` | 指定excel文件路径 | `./excel/example.xlsx` | batch_save / batch_exec / batch |
| `--sheet <名称>` | 指定excel sheet名 | `decode` | batch_save / batch |
| `--pt-dir <目录>` | 指定pt文件保存/读取目录 | `qsmla_testcase` | batch_save / batch_exec / batch |
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

### 性能测试

#### 方式一：单进程msprof采集（perf_run.sh）

所有用例在同一个msprof进程中执行，性能数据汇总到单个PROF文件夹：

``` bash
bash perf_run.sh
```

执行完成后生成 `result_perf.xlsx`，包含 `metadata_duration`、`qsmla_duration` 及与基线对比结果。

#### 方式二：批量隔离执行（batch_isolated_run.sh）

每条用例单独拉起一个独立pytest进程，进程间完全隔离：
- 某条用例device越界/崩溃不会影响后续用例
- msprof挂载在单用例进程上，性能数据采集互不干扰，每条用例生成独立PROF文件夹
- 启用msprof时每条用例执行多次（默认5次）取平均，提升性能数据准确性：
  - 第1次：精度校验 + 性能采集
  - 第2~N次：仅性能采集（跳过精度校验与结果保存）

``` bash
# 不采集性能，仅隔离执行
bash batch_isolated_run.sh qsmla_testcase 0

# 启用msprof采集性能数据，默认采集5次取平均
bash batch_isolated_run.sh qsmla_testcase 1

# 启用msprof，指定采集3次
bash batch_isolated_run.sh qsmla_testcase 1 3
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 参数1 | 用例目录（.pt文件所在路径） | `qsmla_testcase` |
| 参数2 | 是否启用msprof采集（0:否 1:是） | `0` |
| 参数3 | 每条用例性能采集次数（取平均） | `5` |

启用msprof时，执行完成后自动调用 `collect_perf_data.py`，按采集次数分组PROF文件夹，提取 `MixedQuantSparseFlashMlaMetadata` 和 `MixedQuantSparseFlashMla` 两个算子的 `Task Duration(us)` 并取平均值，生成 `result_perf.xlsx`。

执行产物：
| 文件 | 说明 |
|------|------|
| `result.xlsx` | 测试结果表格（精度数据） |
| `result_perf.xlsx` | 性能数据表格（含duration均值、各次详情及基线对比） |
| `batch_summary.log` | 批量执行详细日志 |
| `batch_fail_list.log` | 失败用例清单 |