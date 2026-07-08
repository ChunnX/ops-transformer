# FA 精度测试框架

面向昇腾NPU算子的通用精度测试框架，从4个既有FA测试框架提取最优方案移植。

## 快速开始

```bash
# CLI模式
python test.py --stages gen,cpu,npu,compare

# pytest模式
pytest test.py --stages gen,cpu,npu,compare

# golden独立CLI
python test.py --stages gen,cpu
```

## 目录结构

- `test.py` — 纯入口(CLI main + test函数 + fixtures + hookwrapper, ~150行)
- `conftest.py` — 极薄pytest hooks(addoption + generate_tests, ~50行)
- `orchestrator.py` — 引擎+报告(run_case + Reporter + helpers, ~250行)
- `compare.py` — 精度判定(compare_outputs, 纯函数不打印)
- `precision_analyze.py` — 失败诊断(analyze_failure, 独立)
- `csv_parser.py` — CSV→paramset转换工具(独立)
- `pytest.ini` — marker静态注册(STC/RDV/PROF/WHITEBOX/REDLINE)
- `core/` — ABC基类(__init__.py) + Backend实现(gen_data/cpu/npu/gpu) + case_loader + quant + utils/
- `test_case/` — 5个paramset文件(stc/rdv/prof/whitebox/redline)
- `prof/` — 性能采集虚函数接口(npuprof/npuprofop/gpuprof/l2_flush/perf_parse)
- `data/` — pt中间产物(运行时生成)
- `result/` — 测试结果(compare时自动创建)

## 新算子适配

1. 创建 `test_case/{prefix}_paramset.py` 定义参数+用例集
2. 实现 `core/gen_data.py` 中 GenDataBackend
3. 实现 `core/cpu.py` 中 CpuBackend
4. 实现 `core/npu.py` 中 NpuBackend
5. (可选) 实现 `core/gpu.py` / `core/quant.py` / `core/__init__.py`中forward门面

框架核心无需修改即可接入新算子。

## 设计文档

见 `designs/` 目录。
