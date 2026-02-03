# Pytest测试框架 
## operator为算子名称

## 文件结构

pytest/
- test_operator.py                # 测试用例运行主程序
- testcases_operator.py           # 泛化测试用例入参配置
- check_valid_param.py            # 入参合法性校验
- operator_single.py              # CPU侧算子逻辑实现获取golden与npu算子直调
- pytest.ini                      # 创建ci单算子和graph图模式的测试标记
- check_result.py                 # npu的结果与golden值的精度对比

## 功能说明

基于pytest测试框架，实现GQA算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 环境配置

### 前置要求

1. 确认torch_npu为最新版本
2. source CANN包

### Custom包调用

支持custom包调用

## 使用方法

在pytest文件夹路径下执行：

单算子直调
```bash
python3 -m pytest -rA -s test_operator.py -v -m ci
```