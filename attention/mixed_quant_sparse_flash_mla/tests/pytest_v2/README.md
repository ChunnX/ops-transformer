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

## pytest_v2 batch 一致性扩展

本目录是 MQSMLA 自己的实现，不再从 SMLA pytest 借用主算子 executor。原输入生成、量化 CPU
golden、PT save/load、process 和 `check_result` 均保留；一致性入口在这些原流程上增加模式
变换、level 3、MD5 和 metadata 白盒分析。

| 文件 | 职责 |
| --- | --- |
| `test_mixed_quant_sparse_flash_mla_batch_consistency.py` | 模式 1-4 的 MQ PT 入口 |
| `batch/mixed_quant_sparse_flash_mla_consistency.py` | MQ 专用 executor |
| `batch/mixed_quant_sparse_flash_mla_process.py` | 原两段调用；新增可选 `metadata_sink`，旧返回值不变 |
| `batch/mixed_quant_sparse_flash_mla_metadata_probe.py` | PR8737 metadata-only 临时验证和白盒报告 |
| `batch/consistency_transform.py` | 双 layout/PA 变换与实际量化输入语义检查 |
| `batch/metadata_coverage.py` | PR8737 规约块公式和 FA/FD token/core/workspace 展开 |

```bash
# B=8 -> B=3 + B=5
bash test_run.sh consistency --pt-dir ./mqsmla_testcase \
  --modes reorder,split \
  --split-groups '0,1,2;3,4,5,6,7'

# metadata-only；不代表 MQSMLA 主算子 PASS
python3 -m batch.mixed_quant_sparse_flash_mla_metadata_probe \
  --pt /path/to/mqsmla_case.pt --level 3
```

当前本机 runtime 会在第一条 NPU 操作时拒绝 level 3。level 0 探针只可验证接口、布局和普通
FA 调度；必须等支持 level 3 的 runtime/custom 包组合后，才能验收 FD 与主算子一致性。
