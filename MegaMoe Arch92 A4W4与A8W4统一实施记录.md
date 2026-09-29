# MegaMoe Arch92 A4W4 与 A8W4 统一实施记录

## 范围与状态

本次在 `ops-transformer_5` 原有 Arch35/92 公共层上扩展 Arch92 MTE Wave。当前暂以两段 FP4 A/FP4 B 的 Tensor QGMM BlockMmad 接通 A4W4；这是 A8W4 直接接口尚未放开时的过渡方案，不是 A4W4 最终的 SwiGLU 输出类型。A8W4 保留同一 Wave 的直连目标，但当前 Host 拒绝、Kernel 不实例化混合 FP8/FP4 接口。没有引入 Arch35 的 W4->W8 prologue，也没有修改 Arch35 的默认执行策略。URMA/Layered 不在本次 Arch92 范围内。

这里的“接通”指源码路径和数据类型链已连通，**不是**编译、精度或性能验收。本机没有 `bisheng`/CMake 和 A6 运行环境，仍需在目标环境执行编译与三方精度比对。

## 代码变化

1. **模式选择与入口。** Host 的 [`CheckArchitectureCapability`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_host/op_tiling/arch35_92/mega_moe_tiling.cpp:2603) 在 Arch92 MTE 上放行 A8W8_ND/NZ、A4W4_ND/NZ，继续拒绝 A8W4 与 URMA。Kernel [`mega_moe`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_apt.cpp:63) 接受 FP4 权重的合法 A4W4 模板组合，并通过 `if constexpr` 使 A8W4 分支不实例化；Wave 本身也以 `static_assert` 防止误放开。Arch35 的入口和 Host 支持矩阵未改变。

2. **当前过渡阶段的 A4W4 两段 GMM 都使用 FP4。** Arch35 原 [`QuantConfig`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_types.h:194) 为 A4W4 选择 FP8 SwiGLU 量化输出，从而让 GMM2 走 A8W4。`FP4_SWIGLU_OUTPUT<IsA4W4>` 按 MoE/shared 各自的量化模式计算：Arch92 仅 A4W4 为 `true`，A8W8/A8W4 为 `false`；Arch35 始终为 `false`。定义见 [`ExecutionPolicy`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_execution_policy.h:18)，实例化见 [`MegaMoe`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:59)。因此 Arch92 A4W4 的输入量化、GMM1 权重、SwiGLU 输出、GMM2 输入与权重均为 FP4，SwiGLU 的 BF16 运算和 FP4 打包量化使用已有 epilogue 实现；不新增激活数学。

3. **复用同一套 Wave 和通用 BlockMmad 调用。** [`ProcessGmm1Wave`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:294) 与 [`ProcessGmm2WaveProblem`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:375) 继续调用 `RunGmm1Generic` / `RunGmm2Generic`。`GmmKernel::Config<false>` 选择 Tensor QGMM，FP4/FP8 的 C0 和 L1 字节数由输入类型决定。文件和类分别统一命名为 `mega_moe_wave.h`、`MegaMoeArch92Wave`，当前承载 A8W8/A4W4。未来 A8W4 的目标同样是此 Wave，但必须先确认直接混合类型 BlockMmad 的实际接口。

4. **打包数据的 GM 地址与容量。** routed GMM1 输出和 GMM2 输入的地址绑定都改用 `ActivationQuantOutType`，而不再误用输入 `ActivationType`（FP4 输入的物理存储类型是 `uint8_t`），见 [`ProcessGmm1Wave` / `ProcessGmm2WaveProblem`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:342)。[`WorkspaceLayout`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_workspace.h:125) 在 Arch92 A4W4 的 GM handoff 布局中按每字节两个 FP4 元素分配 routed/shared 激活输出；Arch35 和 Arch92 A8W8 仍按每字节一个元素分配。GMM2 routed/shared 的底层输入地址函数原本已按 `PackedElementTraits` 计算，没有修改其函数逻辑，见 [`UpdateMoeExpertGmm2GlobalBuffer`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm2_combine.h:971) 与 [`UpdateSharedExpertGmm2GlobalBuffer`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm2_combine.h:1020)。FP4 写回原本就在 [`StoreQuantOutput`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/blaze/epilogue/block_epilogue_activation_mx_quant.h:374) 中实现。

5. **BlockMmad context 的类型边界。** A4W4_ND 的 weight1=ND、weight2=NZ_C0_32，两段可能实例化不同 BlockMmad 类型。Wave 不再要求 GMM1/GMM2 类型相同；类型相同时复用原 persistent context，类型不同时 GMM2 传空指针，沿用通用 GMM2 函数内的局部 context。共享专家两段类型不同时，两段都不传 persistent context。判断见 [`GetRoutedGmm2MmadContext` / `GetSharedMmadContext`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:98)，局部 context 的原有分支见 [`Gmm2ExecGeneric`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/stage/mega_moe_gmm2_combine.h:775)。

6. **共享专家 GMM2。** [`RunSharedExpertGmm2`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:613) 只对 Arch35 A4W4 保留旧 A8W4 路径；Arch92 A4W4 使用 Generic FP4/FP4 路径。共享 GMM1/激活仍复用公共实现，其输出类型由 Arch92 策略决定。

## A8W4 后续放开条件

当前 A8W4 的 GMM1 输入/权重为 FP8/FP4，GMM2 输入/权重也为 FP8/FP4。现有 Tensor QGMM 的 FP4 逻辑主要按 A/B 同精度处理，不能把 `Config<false>` 的 A4W4 实例机械套到混合类型上。待 A6 直接 A8W4 BlockMmad 接口到位后，应在通用 Config/选择器中确认混合类型的 C0、L1 字节容量、layout、scaleA/B 与 `actualShape` 参数，再解除 Host、Kernel 和 Wave 三处门禁；不需要复制第二套 Wave，也不恢复 Arch35 的 W4->W8 prologue。

**后续必须同步调整 A4W4：**一旦 Arch92 能直接计算 A8W4，A4W4 的 SwiGLU 量化输出也应恢复为与 Arch35 一样的 FP8 E4M3；届时 GMM1 仍是 FP4/FP4，但 GMM2 变为 FP8 A/FP4 B，而不是继续保持当前的 FP4/FP4。需要把 Arch92 A4W4 的 [`FP4_SWIGLU_OUTPUT<IsA4W4>`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_execution_policy.h:28) 选择改回 `false`，并让 routed [`ProcessGmm2WaveProblem`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch92/mega_moe_wave.h:375) 与共享专家 [`RunSharedExpertGmm2`](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/mega_moe.h:617) 都调用 A6 直接 A8W4 接口。FP8 每元素占一字节，还必须同步恢复 [`WorkspaceLayout` 的 routed/shared 激活区容量](/D:/gitcode/ops-transformer_5/mc2/mega_moe/op_kernel/arch35_92/common/mega_moe_workspace.h:125)，并复核 GMM1 写回和 GMM2 读取的偏移。不能只解除 A8W4 门禁就认为 A4W4 已切换完成；应重新跑无共享/共享专家的编译和精度验证。

## 验证清单

- 已做：`git diff --check` 通过；核对 Host/Kernel workspace 均按同一 `WorkspaceLayoutPolicy::GM_HANDOFF_EXPERT_WAVE` 构造，A4W4 routed/shared 数据与 scale 地址链分别按打包密度和原有 scale 宽度计算。
- 待做：在 A6 CANN/ops-tensor 目标版本上编译 A4W4_ND、A4W4_NZ，以及无共享/共享、量化/非量化 Combine 的模板实例。
- 待做：运行 A8W8 回归和 A4W4 两段 GMM 精度验证，特别检验 NZ_C0_32 物理布局、K tail、不同类型 context 切换与 shared 专家结果。
- 待做：在目标编译环境验证 A8W4 tiling 明确拒绝，且其未支持的 Tensor 实现不会被实例化。
