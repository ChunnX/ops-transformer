# FA 精度测试框架

面向昇腾NPU算子的精度测试框架。

## 快速开始

```bash
# CLI — 全链路
python test.py --stages gen,cpu,npu,compare --case-set ci

# pytest — 全链路
pytest test.py --stages gen,cpu,npu,compare --case-set ci

# 仅生成golden（不依赖NPU）
python test.py --stages gen,cpu --case-set debug

# 断点续跑（跳过gen/cpu，从pt加载，只跑npu+compare）
python test.py --stages npu,compare --case-set ci

# 静默日志（批量跑只看结果）
python test.py --stages gen,cpu,npu,compare --case-set ci --log-level WARNING
```

## 核心功能

| 功能 | 说明 | CLI参数 |
|------|------|---------|
| 流程任意组合 | gen→cpu→npu→gpu→compare按固定顺序执行，`--stages`选子集 | `--stages gen,cpu,compare` |
| 断点续跑 | 跳过的阶段自动从pt加载 | `--stages cpu,compare` |
| 外部pt导入 | 从整网pt加载inputs替代gen | `--load-pt path/to/input.pt` |
| pt中间产物 | 每阶段产出存pt，默认保留 | `--keep-pt` |
| 崩溃隔离 | npu_compute用ThreadPoolExecutor隔离，单case崩溃不中断批跑 | `--max-retries 3` |
| 大case并行 | compare阶段支持torch.multiprocessing分块并行 | `--chunk-size 1000000 --num-workers 4` |
| 精度失败分析 | 失败时输出点位详情+Max-RE+预览 | `--fail-analysis --verbose-diff` |
| 日志系统 | 每阶段进度+耗时+shape，可调级别 | `--log-level INFO` |
| 图模式 | npu_compute支持aclgraph | `--graph-path 7` |
| 用例集分层 | ci/debug/func_rdv等用例集 | `--case-set ci` |
| 用例过滤 | 精确/前缀/后缀三模式匹配 | `--case-id FP16_BNSD_01` |

## 目录结构

```
test.py              纯入口(CLI + test函数 + fixtures)
conftest.py          pytest hooks
orchestrator.py      引擎(run_case + Reporter)
compare.py           精度判定(纯函数)
precision_analyze.py 失败诊断
csv_parser.py        CSV→paramset转换工具
pytest.ini           marker注册 + 日志配置
core/
  __init__.py        ABC基类 + forward门面
  case_loader.py     用例加载+笛卡尔积展开
  gen_data.py        GenDataBackend
  cpu.py             CpuBackend
  npu.py             NpuBackend
  gpu.py             GpuBackend(可选)
  quant.py           量化与反量化(可选)
  utils/             算子特有工具(input_spec/cpu_golden/data_utils)
test_case/           paramset文件(stc/rdv/prof/whitebox/redline)
prof/                性能采集虚函数(npuprof/npuprofop/gpuprof/l2_flush/perf_parse)
data/                pt中间产物(运行时生成)
result/              CSV + 失败分析(compare时自动创建)
```

## 新算子适配

1. `test_case/{prefix}_paramset.py` — 定义参数+用例集
2. `core/gen_data.py` — GenDataBackend(gen_inputs + normalize_params)
3. `core/cpu.py` — CpuBackend(cpu_compute)
4. `core/npu.py` — NpuBackend(npu_compute)
5. (可选) `core/gpu.py` / `core/quant.py` / `core/__init__.py`中forward

框架核心无需修改。

## 设计文档

见 `designs/` 目录。
