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
| 外部pt导入 | 从整网pt加载inputs替代gen | `--load-network-pt path/to/input.pt` |
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
tests2/
├── test.py                 # 统一入口（CLI main + test函数 + fixtures）
├── conftest.py             # pytest hooks（addoption + generate_tests）
├── orchestrator.py         # 编排引擎（run_case + Reporter + helpers）
├── compare.py              # 精度判定（compare_outputs，纯函数）
├── precision_analyze.py    # 失败诊断（analyze_failure）
├── csv_parser.py           # CSV→paramset 转换工具
├── pytest.ini              # marker 静态注册 + 日志配置
├── core/
│   ├── __init__.py         # ABC 基类 + forward() 门面函数
│   ├── case_loader.py      # 用例加载 + 笛卡尔积展开
│   ├── gen_data.py         # GenDataBackend（gen_inputs + normalize_params）
│   ├── cpu.py              # CpuBackend（cpu_compute）
│   ├── npu.py              # NpuBackend（npu_compute）
│   ├── gpu.py              # GpuBackend（gpu_compute，可选）
│   ├── quant.py            # 量化与反量化（可选）
│   └── utils/              # 算子特有工具（input_spec / cpu_golden / data_utils）
├── test_case/
│   ├── stc_paramset.py     # STC 用例集（ci + debug）
│   ├── rdv_paramset.py     # RDV 用例集（func_rdv）
│   ├── prof_paramset.py    # PROF 用例集
│   ├── whitebox_paramset.py # WHITEBOX 用例集
│   └── redline_paramset.py # REDLINE 用例集
├── prof/                   # 性能采集虚函数接口
│   ├── npuprof/            # NPU 性能采集
│   ├── npuprofop/          # NPU 单算子性能
│   ├── gpuprof/            # GPU 性能
│   ├── l2_flush/           # L2 缓存刷新
│   └── perf_parse/         # 性能产物解析
├── data/                   # pt 中间产物（运行时生成）
├── result/                 # 测试结果（CSV + 失败分析，compare 时自动创建）
└── designs/                # 设计文档
```

## 设计思路

**纯入口 + 引擎分离**：`test.py` 只做 CLI/pytest 委托，`orchestrator.py` 是可独立 UT 的引擎。入口不包含被测逻辑。

**固定顺序 + 子集选择**：阶段间有数据依赖（gen→cpu→npu→gpu→compare），用户通过 `--stages` 选子集，框架按固定顺序执行。跳过的阶段自动从 pt 加载。

**判定与诊断分离**：`compare.py` 只做精度判定（返回 dict，不打印），`precision_analyze.py` 只做失败诊断（打印）。变化频率不同，互不干扰。

**声明式用例管理**：`test_case/*.py` 声明参数值（PARAM_NAMES + DEFAULTS + CASE_SETS），`case_loader` 自动笛卡尔积展开。新增参数只需加到 DEFAULTS，所有用例自动继承。

**ABC 基类集中**：接口契约定义在 `core/__init__.py`（GenDataBase/CpuBase/NpuBase/GpuBase），实现分散在各模块。接口给用户看，实现给维护者看。

**标准 logging**：用 Python 官方 `logging.getLogger(__name__)`，不在基类加自定义属性。orchestrator 统一输出阶段进度日志，Backend 内部日志是 opt-in。

## 新算子适配

1. `test_case/{prefix}_paramset.py` — 定义参数+用例集
2. `core/gen_data.py` — GenDataBackend(gen_inputs + normalize_params)
3. `core/cpu.py` — CpuBackend(cpu_compute)
4. `core/npu.py` — NpuBackend(npu_compute)
5. (可选) `core/gpu.py` / `core/quant.py` / `core/__init__.py`中forward

框架核心无需修改。

## 设计文档

见 `designs/` 目录。
