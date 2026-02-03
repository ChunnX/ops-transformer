# QSAS算子测试框架
## 功能说明
基于pytest测试框架，实现QSAS算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调、图模式调用获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围
### 参数限制

- **数据格式**:
- **query_layout**：BSND TND
- **key_layout**: PA_ND
- **value_layout**: PA_ND

### 环境配置

#### 前置要求
1. 确认torch_npu为最新版本
2. source CANN包

#### Custom包调用
支持custom包调用

## 文件结构
#### pytest文件结构说明
- test_qsas_pytestcase.py                           # 读取kv_quant_sparse_attn_sharedkv_paramset.py中自定义参数执行用例，支持单跑及批量测试
- kv_quant_sparse_attn_sharedkv_paramset.py         # test_qsas_pytestcase.py 测试所需的自定义参数配置
- kv_quant_sparse_attn_sharedkv_process_ci_graph.py # torch直调、图模式调用算子入口
- kv_quant_sparse_attn_sharedkv_golden.py           # 根据入参生成torch接口所需tensor及actlen等参数，执行cpu计算生成golden，支持pt保存功能
- check_result.py                                   # cpu golden与npu结果精度对比脚本
- check_valid_param.py                              # 参数合法性检查
- pytest.ini                                        # 创建CI单算子、graph图模式的测试标记
- test_run.sh                                       # 用例执行脚本
- ./batch
    - test_qsas_pt_batch.py                 # 直调模式，读取pt文件批量测试，获取npu输出、npu输出与cpu golden精度对比结果，并保存为excel表格
    - test_qsas_pt_batch_graph.py           # 图模式，读取pt文件批量测试，获取npu输出、npu输出与cpu golden精度对比结果，并保存为excel表格
    - test_qsas_pt_save_from_excelcase.py           # 读取表格批量生成用例pt文件
- ./excel
    - example.xlsx                                  # 批量用例pt生成示例表格

## 使用方法
在pytest文件夹路径下执行：

### 运行测试用例
#### 单用例测试
1、手动配置kv_quant_sparse_attn_sharedkv_paramset.py的参数

2、执行指令：
``` bash
bash test_run.sh single
```
#### 用例的批量生成与测试
1、excel路径下存放用例excel表格

2、执行指令：
``` bash
bash test_run.sh batch
```