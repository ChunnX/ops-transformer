# SAS算子测试框架

## 文件结构
pytest/
- test_sas_pytestcase.py          # pytest泛化测试用例运行主程序,支持多组param，支持将input及golden存储为pt
  - savePt：是否存储input及golden为pt
  - save_path：pt存储路径
  - device_id：指定device
- test_sas_store_pt_from_excel.py # 读取表格批量生成input及golden存储为pt文件
  - pt_dir：pt文件路径
  - result_path：结果表格存储路径
  - device_id：指定device
- test_sas_load_from_pt.py        # 加载pt文件测试，支持遍历目录批跑或指定文件路径列表测试
  - template_run_mode：指定算子，"SWA"、"CFA"、"SCFA"
  - actlen_mode：full、random，为full时actlen等于s1及s2，为random时每个batch在[1, S1]、[S2/2, S2]随机泛化
  - S1EQS2：True、False，为True时s1_actlen=s2actlen
  - save_path：pt存储位置
- testcases_sas.py                # 泛化测试用例入参配置
    - 该文件中为None的参数可自动生成，或传入具体数值则使用传入值。
    - TEST_PARAMS中可定义多组参数，在ENABLED_PARAMS中指定，可批跑
- check_valid_param.py            # 入参检查及精度对比
- check_result.py                 # cpu/npu结果对比校验
- read_excel.py                   # 解析excel用例表格
- pytest.ini                      # 创建ci单算子和graph图模式的测试标记

## 功能说明
基于pytest测试框架，实现SAS算子的精度验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能，并将结果存储为excel表格

## 使用方法
在pytest文件夹路径下执行：

### 运行方式：运行泛化测试用例
单算子直调
```bash
python3 -m pytest -rA -s test_sas_xx.py -v -m ci
```