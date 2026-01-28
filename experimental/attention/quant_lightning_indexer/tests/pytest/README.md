## QLI算子测试框架
## 文件结构

pytest文件说明
- test_qli.py                     # pytest测试用例运行主程序
- testcases.py                    # 测试用例入参配置
- check_result.py                 # cpu结果和npu结果精度对比, 原fuzz中precision method=1
- check_result_a5.py              # 开发自写精度对比方法：排序逐次比较硬阈值判断
- check_result_method13.py        # cpu结果和npu结果精度对比方法：原fuzz中precision method=13,适用于QLI           
- qli_single.py                   # CPU侧算子逻辑实现获取golden，npu算子直调获取算子输出，golden采用numpy类型实现，适用于A3
- qli_single_tensor.py            # 作用同qli_single.py，golden实现采用tensor类型，适用于A5中float8_e4m3fn类型
- pytest.ini                      # 创建ci单算子和graph图模式的测试标记
- qli_pt_process.py               # 批跑中读取pt文件并调用算子获得npu结果
- qli_store.py                    # 读取excel批量生成用例pt文件, cpu golden实现调用qli_single_tensor.py
- qli_store_a5.py                 # 读取excel批量生成用例pt文件, cpu golden实现为开发自写脚本嵌入
- test_qli_rdv.py                 # 批量用例测试主程序
- qli_testexcel_2_myexcel.py      # 测试提供的excel用例表格转换为pytest适用表格

## 功能说明

基于pytest测试框架，实现QLI算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

### 当前实现范围
### 参数限制

- **数据格式**:
- **query_layout**：BSND TND PA_BSND
- **key_layout**: BSND TND PA_BSND

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
``` bash
python3 -m pytest -rA -s test_qli.py -v -m ci
```
用例的批量生成与测试
``` bash
#下面区别在于golden实现的不同
#需要配置读取的excel表格路径和pt文件存放路径
python qli_store.py 
python qli_store_a5.py

#批量测试
#需要配置读取的pt文件路径以及excel结果路径
python3 -m pytest -rA -s test_qli_rdv.py -v -m ci
```
