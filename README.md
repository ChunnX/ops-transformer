# NPUKernelBench

华为昇腾（Ascend）NPU 算子基准测试框架。支持从算子代码编译、精度验证的全流程自动化，用于验证迁移算子的正确性。

## 环境准备

```bash
source /path/to/cann/set_env.sh     # 设置 CANN 环境(最好是9.0.0)
pip install -r requirements.txt
```

## 流水线 Stage 说明

框架以流水线方式执行，每个阶段可独立开关。每个阶段接收上游产物，处理后传递给下游。

### compile — 编译

将模板代码 + 算子 kernel 代码组装为完整的 msopgen 工程，调用编译工具链生成 `.so` 动态库。

| 项目 | 说明 |
|------|------|
| 输入 | `answer/` 目录下的算子实现 |
| 输出 | `compile_result`（编译成功/失败） |
| 产物文件 | `src/`（源码）、`build/`（编译产物）、`log/*_compile.log` |
| 模式 | `msopgen`（kernel_only 默认）或 `cann-ops`（full 模式） |

编译分为两个子阶段：**算子编译**（msopgen cmake build，生成 `libcust_opapi.so`、kernel `.o` 等）和 **pybind 编译**（生成 `kernel_gen_ops.*.so` Python 扩展）。默认每次全量编译。使用 `-skip_op_build` 可在算子产物已存在时跳过算子编译，仅重建 pybind 扩展（秒级），适用于修复 pybind 绑定后快速重编译（某些算子编译可达 10+ 分钟）。

### precision — 精度验证

在 NPU 上运行自定义算子，与 PyTorch CPU 参考实现逐元素对比，统计绝对误差和相对误差。

| 项目 | 说明 |
|------|------|
| 输入 | 编译产物 + `validation/` 下的 `module.py`（参考实现）和 `test_cases.csv` |
| 输出 | `precision_result`（逐 case 的通过/失败、误差统计） |
| 产物文件 | `case_{id}/log/*_precision.log` |
| 阈值 | `max_rel_error` 和 `max_abs_error`（默认 0.001） |

## 命令参考

### 基础命令

```bash
python scripts/run_test.py [选项]
```

### 选项列表

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-task_name` | 多值 | 全部 | 算子名称（如 `scatterElementsV2 Add LayerNorm`），支持同时指定多个算子 |
| `-stages` | 多值 | `compile precision` | 执行阶段：`compile` `precision` |
| `-n_case` | 整数 | 1 | 测试用例数（0 = CSV 中全部） |
| `-test_mode` | 选择 | `static` | `static`（静态 shape）或 `dynamic`（动态 shape） |
| `-template_mode` | 选择 | `kernel_only` | `kernel_only`（仅 kernel）或 `full`（完整模板） |
| `-run_dir` | 字符串 | `runs` | 输出根目录 |
| `-config` | 字符串 | `base_config.yaml` | 配置文件路径 |
| `-tiling_dump` | 开关 | 关闭 | 自动注入 tiling 参数打印到编译代码中，用于调试 tiling 计算 |
| `-skip_op_build` | 开关 | 关闭 | 跳过算子编译，仅重建 pybind 扩展（需算子产物已存在） |

### 常用命令示例

| 场景 | 命令 |
|------|------|
| 编译 + 精度验证 | `python scripts/run_test.py -task_name scatterElementsV2 -stages compile precision` |
| 多算子批量 | `python scripts/run_test.py -task_name scatterElementsV2 Gelu Add -stages compile precision` |
| 指定用例数 | `python scripts/run_test.py -task_name scatterElementsV2 -stages compile precision -n_case 5` |
| 运行answer下所有的目录用例(以0、1、2为例) | `python scripts/run_test.py -task_name scatterElementsV2 -stages compile precision -n_sample 3` |
| 注入 tiling 打印 | `python scripts/run_test.py -task_name scatterElementsV2 -stages compile precision -tiling_dump` |

## 项目结构

```
npu-kernel-bench/
├── npukernelbench/              # 主包
│   ├── config/                  #   配置（FullConfig frozen dataclass + YAML + CLI 合并）
│   ├── models/                  #   数据模型（BuildHandle, TestHandle, Result）
│   ├── discovery/               #   任务发现（扫描 tasks/ 目录 + 创建 Handle）
│   ├── pipeline/                #   流水线（Stage Protocol + Runner）
│   ├── stages/                  #   各阶段实现
│   │   ├── compile/             #     编译（源码准备 + msopgen 调用 + 产物验证）
│   │   └── precision/           #     精度验证（NPU 运行 + 误差统计）
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
tasks/level1/Math/scatterElementsV2/
├── question/                    # 代码模板
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
```
