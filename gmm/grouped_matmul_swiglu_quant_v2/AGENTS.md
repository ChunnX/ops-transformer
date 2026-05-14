# AGENTS.md — GMMSQ (GroupedMatmulSwigluQuantV2)

> 本项目使用 **Agent 驱动的知识库 + 关键文件架构** 进行开发。每次新会话启动时，请按本文档指引读取相关文件，确保跨会话信息不丢失、不偏离。

---

## 工作流：启动新会话必读

每次进入新会话，按顺序读取以下文件（渐进式披露，从宏观到微观）：

1. **本文档** → 了解项目概况和目录结构
2. **[.agent/GOAL.md](.agent/GOAL.md)** → 了解当前总体目标和方向
3. **[.agent/PLAN.md](.agent/PLAN.md)** → 了解当前阶段任务拆解
4. **[.agent/NEXT_STEPS.md](.agent/NEXT_STEPS.md)** → 了解用户下一步计划
5. 如有歧义 → 读取 **[.agent/raw/](.agent/raw/)** 中的原始意图文档

### 工作期间的行为准则

| 场景 | 应读取的文件 |
|------|-------------|
| 需要技术细节 | [.agent/DESIGN.md](.agent/DESIGN.md) → 索引入口，指向 `raw/design/` 和 `raw/docs/` |
| 做重大决策前 | [.agent/CHECKLIST.md](.agent/CHECKLIST.md) → 按清单逐项检查 |
| 发现新信息 | 立即写入对应文件（见下方文件职责表） |
| 不确定原始意图 | [.agent/raw/](.agent/raw/) → 原始意图，不可修改 |

### 完成工作后的行为准则

1. 按 [.agent/CHECKLIST.md](.agent/CHECKLIST.md) 逐项验证
2. 在 [.agent/CHANGELIST.md](.agent/CHANGELIST.md) 追加变更记录（**严格按格式**，见下方）
3. 更新 [.agent/NEXT_STEPS.md](.agent/NEXT_STEPS.md) 和 [.agent/PLAN.md](.agent/PLAN.md) 的状态
4. 如目标进度变更，更新 [.agent/GOAL.md](.agent/GOAL.md)

---

## 核心原则

### 原则一：关键信息必须落盘
任何有价值的信息都应该通过文档持久化保存，**不要依赖上下文记忆**。发现重要信息 → 立即写入文件。

### 原则二：原始信息必须完整保留
用户的原始输入、想法、设计意图必须以**未压缩、未总结、未概括**的形式保存在 `.agent/raw/` 中。保持原始语言（中文存中文，英文存英文），不要翻译或改写。

---

## 项目概况

### 算子功能
融合 GroupedMatmul + dequant + SwiGLU + quant 的 Ascend C 算子，支持多种量化场景：
- **A8W8**: INT8 激活 + INT8 权重（Atlas A2/A3）
- **A8W4**: INT8 激活 + INT4 权重（Atlas A2/A3）
- **MX量化**: FP8/FP4 微缩放量化（Ascend 950PR/950DT）
- **MXA8W4量化**: FP8 + FP4，MX量化的伪量化数据流（Ascend 950PR/950DT，**当前开发重点**)

### 当前目标
在已有全量化基础上支持 **MxA8W4 伪量化数据流**，确保 tiling 拦截校验正确、kernel 精度正确，且不影响原有全量化数据流功能。

### 当前阶段
**阶段二-Part2/Part3** — Tiling 测试验证 + Kernel 代码开发（进行中）

---

## 目录结构

```
gmm/grouped_matmul_swiglu_quant_v2/
├── .agent/                    ← 跨会话记忆系统
│   ├── GOAL.md                ← 总体目标（方向）
│   ├── PLAN.md                ← 阶段计划（路径）
│   ├── CHECKLIST.md           ← 质量检查清单
│   ├── CHANGELIST.md          ← 变更记录（追加式）
│   ├── DESIGN.md              ← 设计文档索引
│   ├── NEXT_STEPS.md          ← 用户下一步计划
│   ├── reference/             ← 方法论文档
│   │   └── vibe-coding-philosophy.md
│   └── raw/                   ← 原始意图（不可修改）
│       ├── design/            ← 原始设计文档
│       ├── docs/              ← GMMFR算子走读文档
│       ├── development/       ← SRS/STC/编码方案
│       └── next_steps/        ← 用户原始任务描述
│
├── op_api/                    ← ACLNN 接口层（aclnn_grouped_matmul_swiglu_quant_v2.cpp）
├── op_host/                   ← Host 侧（InferShape、Tiling）
│   └── op_tiling/             ← Tiling 模板
│       └── arch35/            ← Ascend 950 架构 Tiling
├── op_kernel/                 ← Device 侧 Kernel（Ascend C）
│   ├── arch35/                ← Ascend 950 架构 Kernel
│   │   ├── common/            ← 公共头文件（从 GMMFR 复制）
│   │   └── weight_quant_basic_block/  ← 权重量化核心模块
│   └── grouped_matmul_swiglu_quant_v2_apt.cpp  ← Kernel 入口
├── op_graph/                  ← Graph 侧（fallback）
├── tests/                     ← 单元测试
│   └── ut/
│       ├── op_host/           ← Tiling UT（CSV + cpp）
│       └── op_api/            ← API UT
└── examples/                  ← 调用示例
```

### 关键文件速查

| 文件 | 作用 |
|------|------|
| `op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp` | Kernel 入口，含 `V310_GMM_SQ_ANTI_QUANT` 宏分支 |
| `op_host/op_tiling/grouped_matmul_swiglu_quant_v2_tiling.cpp` | Tiling 分发逻辑（TilingFunc） |
| `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp` | MxA8W4 Tiling 模板实现 |
| `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h` | Controller 层 |
| `op_kernel/arch35/common/basic_block_config.h` | BasicBlock 配置常量 |

---

## 构建与验证命令

### 构建 ophost（Tiling 侧）
```bash
bash build.sh --ophost --ops=grouped_matmul_swiglu_quant_v2 -j16
```

### 构建 kernel（Device 侧）
```bash
bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul_swiglu_quant_v2
```

### 运行 Tiling UT
```bash
bash build.sh --ophost_test --ops=grouped_matmul_swiglu_quant_v2
# 仅编译不执行：
bash build.sh --ophost_test --ops=grouped_matmul_swiglu_quant_v2 --noexec
```

### 运行 API UT
```bash
bash build.sh --opapi_test --ops=grouped_matmul_swiglu_quant_v2
```

### 编译验证顺序
**Tiling 开发**: `ophost 编译 → ophost_test 验证`
**Kernel 开发**: `opkernel 编译 → 检查编译错误`

---

## 开发约定

### Tiling 开发
- Tiling 模板继承 `TilingBaseClass`，实现 `InferScenario` + `DoOpTiling` + `GetTilingKey`
- 通过 `REGISTER_OPS_TILING_TEMPLATE(OpName, TilingClass, TemplateID)` 注册
- TilingFunc 中通过 dtype 检测分发到对应模板
- TilingData 使用 `#pragma pack(push, 8)` 保证 8 字节对齐

### Kernel 开发
- 架构分层：Controller → BasicBlock → VecCompute / CubeCompute
- MxA8W4 场景使用 `V310_GMM_SQ_ANTI_QUANT` 宏标识
- 从 GMMFR 算子迁移时，注意 namespace 为 `WeightQuantBatchMatmulV2::Arch35`
- 使用 `using` 声明引入常用工具：`CeilDivide`, `DataCopyPad2D` 等

### 代码规范
- Header guard 宏必须与完整文件名匹配
- 使用 `GetInputDesc()` 获取 descriptor，`GetInputShape()->GetOriginShape()/GetStorageShape()` 获取 shape
- compileInfoPtr 成员名带下划线后缀：`aicNum_`, `ubSize_`
- 算子不支持 bias 输入，TilingData 中不包含 hasBias 字段

---

## 参考算子

**GMMFR (GroupedMatmulFinalizeRouting)** 是 MxA8W4 数据流的参考实现：
- Kernel 入口: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_apt.cpp`
- Controller: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h`
- 完整走读文档: `.agent/raw/docs/` 目录

---

## CHANGELIST.md 记录格式

每次变更后，在 [.agent/CHANGELIST.md](.agent/CHANGELIST.md) 追加记录，**严格按以下格式**：

```markdown
### YYYY-MM-DD | 变更标题

**类型**: 🆕 新功能 | 🔧 修改 | 🐛 修复 | 🗑️ 删除 | 📝 文档
**范围**: 影响范围

#### 变更内容
- 具体变更了什么

#### 关键决策
- 做了什么决策，为什么

#### 影响范围
- 对现有功能的影响
```

**注意**: 不要删除已有条目，只追加。
