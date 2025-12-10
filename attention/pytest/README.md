# GQA算子测试框架

## 文件结构

pytest/
- test.py                      # pytest泛化测试用例运行主程序
- test_excel.py                   # excel读取测试用例运行主程序, 测试用例的excel文件需放入pytest/test_excel目录下
- testcases.py                    # 泛化测试用例入参配置
- check_valid_param.py            # 入参检查及精度对比
- gqa_no_quant_bnsd_bsnd.py       # CPU侧算子逻辑实现获取golden，npu算子直调
- gqa_no_quant_bnsd_bsnd_ge.py    # 图模式编译调用，与单算子调用结果精度对比
- pytest.ini                      # 创建ci单算子和graph图模式的测试标记

## 功能说明

基于pytest测试框架，实现GQA算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据， 通过torchair入图
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

### 当前实现范围

✅**已实现**：基础GQA非量化PA场景
❌**未实现**：mask功能

### 参数限制

- **数据格式**：FP16 / BF16
- **in_layout**：BSND /BNSD
- **kvcache_layout**: BNBD / BBH
  - BSND格式仅支持BBH

## 环境配置

### 前置要求

1. 确认torch_npu为最新版本
2. source CANN包

### Custom包调用

支持custom包调用

## 使用方法

在pytest文件夹路径下执行：

### 方式一：运行泛化测试用例
单算子直调+图模式
```bash
python3 -m pytest -rA -s test.py
```

单算子直调
```bash
python3 -m pytest -rA -s test.py -v -m ci
```

图模式
```bash
python3 -m pytest -rA -s test.py -v -m graph
```

### 方式二：运行Excel测试用例
```bash
EXCEL_FILE="./test_excel/IFA_FIA_Case_PA.xlsx" SHEET_NAME="IFA_FIA_Case" python3 -m pytest -rA -s test_excel.py
EXCEL_FILE="./test_excel/IFA_FIA_Case_PA.xlsx" SHEET_NAME="IFA_FIA_Case" python3 -m pytest -rA -s test_excel.py -v -m ci
EXCEL_FILE="./test_excel/IFA_FIA_Case_PA.xlsx" SHEET_NAME="IFA_FIA_Case" python3 -m pytest -rA -s test_excel.py -v -m graph
```

## 测试用例生成

网站生成的excel测试用例文件存放在 `test_excel/` 目录下。