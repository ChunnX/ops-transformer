# NPUKernelBench

华为昇腾（Ascend）NPU 算子基准测试框架。支持从 LLM 代码生成、编译、精度验证到性能测试的全流程自动化。

## 环境准备

```bash
source /path/to/cann/set_env.sh     # 设置 CANN 环境
pip install pyyaml pandas psutil tqdm tabulate openai
```

| 依赖 | 版本 |
|------|------|
| Python | >= 3.11 |
| PyTorch | 2.5.1 |
| torch-npu | 对应 CANN 8.5.0 |
| CANN SDK | 8.5.0 |

## 流水线 Stage 说明

框架以流水线方式执行四个阶段，每个阶段可独立开关。每个阶段接收上游产物，处理后传递给下游。

### code_gen — 代码生成(可选)

通过 LLM 生成 AscendC 算子代码。构建包含算子 API 描述、测试用例 shape 信息、硬件规格和参考案例的提示词，调用 LLM API 获取生成结果，解析提取 kernel 实现代码。

| 项目 | 说明 |
|------|------|
| 输入 | 算子定义目录（`question/` 模板 + `validation/test_cases.csv`） |
| 输出 | `gen_content`、`kernel_content`（kernel_only 模式）或 `tiling_def` + `tiling_content` + `kernel_content`（full 模式） |
| 产物文件 | `log/*_rag_simple_code_gen_prompt.json`（含 prompt 和生成内容） |
| 触发 | `-chat` 启用 LLM 生成；不启用时跳过，使用标杆参考实现 |

### compile — 编译

将模板代码 + 生成的 kernel 代码组装为完整的 msopgen 工程，调用编译工具链生成 `.so` 动态库。

| 项目 | 说明 |
|------|------|
| 输入 | `code_gen` 阶段的生成内容，或 `answer/` 目录下的标杆参考实现 |
| 输出 | `compile_result`（编译成功/失败） |
| 产物文件 | `src/`（源码）、`build/`（编译产物）、`log/*_compile.log` |
| 模式 | `msopgen`（kernel_only 默认）或 `cann-ops`（full 模式） |

编译分为两个子阶段：**算子编译**（msopgen cmake build，生成 `libcust_opapi.so`、kernel `.o` 等）和 **pybind 编译**（生成 `kernel_gen_ops.*.so` Python 扩展）。默认每次全量编译。使用 `-skip_op_build` 可在算子产物已存在时跳过算子编译，仅重建 pybind 扩展（秒级），适用于修复 pybind 绑定后快速重编译（算子编译可达 10+ 分钟）。

### precision — 精度验证

在 NPU 上运行自定义算子，与 PyTorch CPU 参考实现逐元素对比，统计绝对误差和相对误差。

| 项目 | 说明 |
|------|------|
| 输入 | 编译产物 + `validation/` 下的 `module.py`（参考实现）和 `test_cases.csv` |
| 输出 | `precision_result`（逐 case 的通过/失败、误差统计） |
| 产物文件 | `case_{id}/log/*_precision.log` |
| 阈值 | `max_rel_error` 和 `max_abs_error`（默认 0.001） |

### perf — 性能测试

测量自定义算子在 NPU 上的执行耗时，计算吞吐量和 FLOPS。

| 项目 | 说明 |
|------|------|
| 输入 | 编译产物 + 精度验证通过的 case |
| 输出 | `perf_result`（延迟、吞吐量） |
| 产物文件 | `case_{id}/log/*_perf.log` |
| 配置 | `num_perf_trials`（测试轮次，默认 100） |

## 命令参考

### 基础命令

```bash
python scripts/run_test.py [选项]
```

### 选项列表

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-task_name` | 多值 | 全部 | 算子名称（如 `Sqrt Add LayerNorm`） |
| `-stages` | 多值 | `code_gen compile precision` | 执行阶段：`code_gen` `compile` `precision` `perf` |
| `-chat` | 开关 | 关闭 | 启用 LLM 代码生成 |
| `-n_sample` | 整数 | 100 | LLM 生成样本数 |
| `-n_case` | 整数 | 1 | 测试用例数（0 = CSV 中全部） |
| `-test_mode` | 选择 | `static` | `static`（静态 shape）或 `dynamic`（动态 shape） |
| `-template_mode` | 选择 | `kernel_only` | `kernel_only`（仅 kernel）或 `full`（完整模板） |
| `-run_dir` | 字符串 | `runs` | 输出根目录 |
| `-config` | 字符串 | `base_config.yaml` | 配置文件路径 |
| `-tiling_dump` | 开关 | 关闭 | 自动注入 tiling 参数打印到编译代码中 |
| `-skip_op_build` | 开关 | 关闭 | 跳过算子编译，仅重建 pybind 扩展（需算子产物已存在） |

### 常用命令示例

| 场景 | 命令 |
|------|------|
| 编译 + 精度验证 | `python scripts/run_test.py -task_name Sqrt -stages compile precision` |
| 完整流程（含性能） | `python scripts/run_test.py -task_name LayerNorm -stages compile precision perf` |
| LLM 生成 + 编译 + 验证 | `python scripts/run_test.py -chat -task_name Sqrt -stages code_gen compile precision` |
| 仅编译 | `python scripts/run_test.py -task_name Sqrt -stages compile` |
| 多算子批量 | `python scripts/run_test.py -task_name Sqrt Gelu Add -stages compile precision` |
| 指定用例数 | `python scripts/run_test.py -task_name Matmul -stages compile precision -n_case 5` |
| 动态 shape 模式 | `python scripts/run_test.py -task_name Sqrt -stages compile precision -test_mode dynamic` |
| 完整模板模式 | `python scripts/run_test.py -task_name Sqrt -stages compile -template_mode full` |
| 注入 tiling 打印 | `python scripts/run_test.py -task_name Sqrt -stages compile precision -tiling_dump` |
| LLM 多样本 | `python scripts/run_test.py -chat -task_name Sqrt -stages code_gen -n_sample 5` |
| 仅重建 pybind（跳过算子编译） | `python scripts/run_test.py -task_name Sqrt -stages compile -skip_op_build` |

## 项目结构

```
npu-kernel-bench/
├── npukernelbench/              # 主包
│   ├── config/                  #   配置（FullConfig frozen dataclass + YAML + CLI 合并）
│   ├── models/                  #   数据模型（BuildHandle, TestHandle, Result）
│   ├── discovery/               #   任务发现（扫描 tasks/ 目录 + 创建 Handle）
│   ├── pipeline/                #   流水线（Stage Protocol + Runner）
│   ├── stages/                  #   各阶段实现
│   │   ├── code_gen/            #     代码生成（prompt 构建 + LLM API + 解析）
│   │   ├── compile/             #     编译（源码准备 + msopgen 调用 + 产物验证）
│   │   ├── precision/           #     精度验证（NPU 运行 + 误差统计）
│   │   └── perf/                #     性能测试（计时 + 吞吐量）
│   ├── code_gen/                #   代码生成核心
│   │   ├── prompts/             #     提示词构建 + SFT 参考案例
│   │   └── llm_api.py           #     LLM API 调用（OpenAI 兼容）
│   ├── statistics/              #   统计报告（结果收集 + 评分展示）
│   └── utils/                   #   工具（IO、日志、模块加载）
│
├── scripts/                     # 入口脚本
│   └── run_test.py              #   运行入口（单算子/多算子/批量）
│
├── tasks/                       # 算子任务定义
│   ├── level1/                  #   基础算子（element-wise）
│   ├── level2/                  #   标准算子（LayerNorm, Conv2D）
│   └── level3/                  #   复杂算子（FlashAttention, Matmul）
│
├── libs/                        # 编译模板
│   └── common_include/          #   msopgen / cann-ops 公共文件
│
└── base_config.yaml             # 默认配置
```

## 算子任务定义

每个算子位于 `tasks/level{N}/{category}/{OperatorName}/` 下：

```
tasks/level1/Math/Sqrt/
├── question/                    # 代码模板（LLM 生成时的输入）
│   ├── api_desc.md              #   算子 API 描述与公式
│   ├── op_kernel/               #   Device 侧 kernel 模板 (.cpp)
│   └── op_host/                 #   Host 侧实现模板 (.cpp)
├── answer/                      # 标杆参考实现（可含多个版本 0/, 1/...）
└── validation/                  # 验证相关
    ├── test_cases.csv           #   测试用例（shape、dtype）
    ├── prepare_inputs.py        #   输入数据生成
    └── module.py                #   PyTorch 参考实现
```

## 配置

通过 `base_config.yaml` + CLI 参数共同控制，CLI 参数优先级更高。

```yaml
run_dir: "runs"
n_sample: 100
n_case: 1
static_shape_mode: True
kernel_only_mode: True

compile:
  mode: "msopgen"
  timeout: 400
  num_processes: 128

eval:
  gpu_devices: [0,1,2,3,4,5,6,7]
  max_rel_error: 0.001
  max_abs_error: 0.001
  timeout_precision: 300

chat:
  api_model: "vllm"
  base_url: "http://0.0.0.0:5600/v1/"
  model_path: ""
  temperature: 0.7
  max_tokens: 16384
```

LLM API Key 通过环境变量 `LLM_API_KEY` 设置。