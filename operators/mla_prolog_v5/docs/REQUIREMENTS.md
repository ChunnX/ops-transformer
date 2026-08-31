# mla_prolog_v5 需求分析（快速迁移模式）

> 本任务为"快速迁移"模式：精简设计流程，只迁移原始代码仓代码，不做功能增强。

## 1. 需求来源

GitCode issue-65：将 ops-transformer 仓 `attention/mla_prolog_v3` 算子快速迁移为独立自定义算子工程
`mla_prolog_v5`，并打通 pybind 接口调用流程。

- 源算子：`attention/mla_prolog_v3`（依赖 `attention/mla_prolog` 的 arch22 kernel 头与 tiling/infershape 实现）
- 目标：一个独立可编译的算子工程 + Python 接口测试
- 要求：快速迁移、减少人工交互、完成基本功能验证

## 2. 功能描述

MlaProlog 为推理场景 Multi-Head Latent Attention（MLA）前处理算子，四路计算：

- $c^Q = \alpha_q \cdot \mathrm{RmsNorm}(x \cdot W^{DQ})$
- $q^N = c^Q \cdot W^{UK} \cdot W^{UQ}$，$q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})$
- $c^{KV} = \alpha_{kv} \cdot \mathrm{RmsNorm}(x \cdot W^{DKV}) \rightarrow$ 写入 kvCache（原地）
- $k^R = \mathrm{ROPE}(x \cdot W^{KR}) \rightarrow$ 写入 krCache（原地）

详细公式、参数、约束与 v3 原版一致（见源仓 `attention/mla_prolog_v3/README.md`，
迁移至本工程 `README.md`）。

## 3. 支持硬件与数据类型（迁移范围）

- **硬件：仅 A2（ascend910b，910B 系列，本机 910B3）**。ascend950/arch35 相关代码不迁移。
- **数据类型/量化模式**：保持 v3 原版（非量化 bf16、部分量化 int8、全量化 int8 等场景），
  不做功能增强。A2 上可执行的量化组合以 aclnn 外层校验为准（DAV_2201 分支）。

## 4. 使能方式

- **Aclnn 直调**：独立工程编译产出 `custom_opp_*.run` 算子包（cust_optiling / cust_opapi /
  kernel binary），安装后通过两段式 `aclnnMlaPrologV5WeightNz(GetWorkspaceSize)` 调用。
- **pybind**：torch_extension（pybind11 封装 aclnn 两段式接口），Python 侧
  `import` 后直接以 torch.Tensor 调用。

## 5. 接口命名映射

| 源（ops-transformer） | 目标（mla_prolog_v5 独立工程） |
|---|---|
| 算子类型 `MlaPrologV3`（def 注册，含 do_rope 属性） | `MlaPrologV5` |
| kernel 入口 `mla_prolog_v3` | `mla_prolog_v5` |
| `aclnnMlaPrologV3/V4WeightNz`（外层 wrapper + 仓框架生成的 Inner） | `aclnnMlaPrologV5WeightNz`（外层迁移 + **手写 L0/Inner 层**，独立工程无仓级生成框架） |
| `torch_extension/csrc/mla_prolog.cpp`（调 V4 API） | `torch_extension/csrc/mla_prolog_v5.cpp`（调 V5 API） |

> 说明：aclnn 外层语义取 V4 版（带 doRope 开关参数），因其为 v3 目录内现成最全接口，
> 且 torch_extension 现成代码即调用该版本；算子本体（kernel/tiling/infershape）为 v3 原版，无功能增强。

## 6. 验收标准（基本功能验证）

1. 独立工程 `bash build.sh --soc=ascend910b` 编译通过：生成 kernel binary、cust_optiling.so、
   cust_opapi.so、custom_opp_*.run 包。
2. 算子包可安装（.run 安装到 opp/vendors）。
3. pybind 扩展编译通过，Python 接口在 910B3 NPU 上实际运行成功。
4. 精度：非量化 bf16 场景 NPU 结果与 CPU golden 比对通过（相对误差阈值参考社区标准，
   bf16 放宽标准）；可加测部分量化场景（时间允许时）。

## 7. 约束限制

- 只迁移，不增强：kernel/host 逻辑保持源仓实现（除 A2 裁剪与重命名必需改动）。
- 源仓 `attention/`、`common/` 等目录不做任何修改（新工程完全独立、自包含 vendored 依赖）。
- 不引入对仓级构建框架（add_modules_sources 等）的依赖。
