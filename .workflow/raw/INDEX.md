# raw/ — 原始文档索引

> 本目录保存所有原始文档，**不可修改**。其他文件是衍生物，这里是 truth source。

## 目录结构

```
raw/
├── INDEX.md        ← 本文件（检索入口）
├── spec/           ← 算子原型规格说明书（接口定义、shape/dtype/约束）
├── design/         ← 设计文档（初稿设计、实现方案）
├── docs/           ← 参考文档
└── reference/      ← 参考文档（代码走读、分析等）
```

## spec/ — 算子原型规格说明书

> 算子原型规格说明书定义接口契约：输入输出 shape/dtype/format、属性参数、约束条件。
> 是 ACLNN 接口开发的 truth source。

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [spec/gmmsq_op.md](spec/gmmsq_op.md) | GMMSQ 算子原型规格 | 参考 GMMSQ 接口定义 |
| [spec/gmmfr_op.md](spec/gmmfr_op.md) | GMMFR 算子原型规格 | 参考 GMMFR 接口定义 |
| [spec/qkv_rms_norm_rope_cache_with_kscale_op.md](spec/qkv_rms_norm_rope_cache_with_kscale_op.md) | **本算子原型规格** | **ACLNN 接口开发的 truth source** |

## design/ — 设计文档

> 设计文档记录实现方案，包括架构设计、计算流程、tiling 策略、kernel 设计等。
> 设计文档可能随开发迭代更新，但 spec/ 中的接口契约保持稳定。

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [design/qkv_rms_norm_rope_cache_with_kscale/](design/qkv_rms_norm_rope_cache_with_kscale/) | QkvRmsNormRopeCacheWithKScale 设计文档目录 | 开发本算子时参考 |

### QkvRmsNormRopeCacheWithKScale 设计文档

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [qkv_rms_norm_rope_cache_with_kscale_design.md](design/qkv_rms_norm_rope_cache_with_kscale/qkv_rms_norm_rope_cache_with_kscale_design.md) | 完整设计文档（主文档/初稿） | 了解算子完整设计 |
| [qkv_rms_norm_rope_cache_with_kscale_tasks/](design/qkv_rms_norm_rope_cache_with_kscale/qkv_rms_norm_rope_cache_with_kscale_tasks/) | 拆分设计文档目录 | 按模块开发时参考 |

#### 拆分文档目录

| 文件 | 内容 | 何时读 |
|------|------|--------|
| `01_operator_proto_and_formula.md` | 算子原型与计算公式 | 实现接口/checker 时 |
| `02_deliverables_overview.md` | 交付件纵览 | 了解需要交付哪些文件 |
| `03_golden_design.md` | Golden 设计 | 后续阶段 |
| `04_tiling_and_ut_design.md` | Tiling & UT 设计 | 后续阶段 |
| `05_kernel_design.md` | Kernel 设计 | 后续阶段 |
| `06_inter_core_sync_design.md` | 核间同步设计 | 后续阶段 |
| `07_intra_core_sync_design.md` | 核内同步设计 | 后续阶段 |
| `08_implementation_task_breakdown.md` | 实现子任务拆解 | 了解任务拆分 |

## reference/ — 参考文档

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [reference/01_architecture.md](reference/01_architecture.md) | GMMFR 架构总览与数据流 | 了解参考算子整体架构 |
| [reference/02_controller.md](reference/02_controller.md) | Controller 层代码走读 | 实现 Controller 时参考 |
| [reference/03_basic_block.md](reference/03_basic_block.md) | BasicBlock 层代码走读 | 实现 BasicBlock 时参考 |
| [reference/04_vec_compute.md](reference/04_vec_compute.md) | VecCompute 层代码走读 | 实现向量计算时参考 |
| [reference/05_cube_compute.md](reference/05_cube_compute.md) | CubeCompute 层代码走读 | 实现矩阵计算时参考 |
| [reference/06_sync_mechanism.md](reference/06_sync_mechanism.md) | 跨核同步机制分析 | 处理 AIC/AIV 同步时参考 |

## 检索场景速查

| 我需要... | 读哪个文件 |
|-----------|-----------|
| 了解算子接口契约（shape/dtype/约束） | `spec/` 下对应算子规格文档 |
| 了解实现方案和计算流程 | `design/` 下对应设计文档 |
| 了解参考算子的代码走读 | `reference/` 下走读文档 |
| 查看用户的详细任务要求 | `.workflow/reference/next_steps/` 下对应文件 |
