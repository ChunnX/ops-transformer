## SAS算子测试框架
## 文件结构

pytest/
- test_run.sh                     # pytest测试用例运行主程序
- test_sas.py                     # 执行单跑功能
- check_valid_param.py            # 检验入参的合法性
- testcases_sas.py                # 测试用例入参配置
- check_result.py                 # cpu结果和npu结果精度对比
- sparse_attn_sharedkv_process.py # CPU侧算子逻辑实现获取golden，npu算子直调获取算子输出
- pytest.ini                      # 创建ci单算子和graph图模式的测试标记
- gen_and_save_data.py            # 读取excel文件中的用例信息并批量生成用例保存为pt文件
- test_sas_load.py                # 读取保存为pt文件的用例并执行

## 功能说明

基于pytest测试框架，实现SAS算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

### 当前实现范围
### 参数限制

## 环境配置

### 前置要求

1. 确认torch_npu为最新版本
2. source CANN包

### Custom包调用

支持custom包调用

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例
单算子直调
```bash
bash run_sas.sh single
```
#### 用例的批量生成与测试
1、excel路径下存放用例excel表格

2、执行指令：
``` bash
bash test_run.sh save # 默认路径执行用例生成
```
支持从指定路径下读取excel表指定sheet页，pt文件保存到指定文件夹：
``` bash
bash test_run.sh save -E "./cases/sas_redline_L0.xlsx" -S "Sheet1" -P "./data"
```
其中，-E设置excel文件存储路径，-S设置Sheet名，-P设置pt存储文件夹，各参数可单独配置
#### 用例的批量生成与测试
1、pt路径下存放用例pt文件

2、执行指令：
``` bash
bash test_run.sh load
```
支持从指定路径下读取pt文件，结果保存到指定文件夹：
``` bash
test_run.sh load -P "./data" -R "./result/sas_result.xlsx"
```
其中，-P设置pt读取文件夹，-R设置结果存储路径，各参数可单独配置
