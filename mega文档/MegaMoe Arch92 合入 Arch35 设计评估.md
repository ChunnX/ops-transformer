# MegaMoe arch92 合入 arch35 设计评估

## 1. 目标与结论

本文评估将当前 MegaMoe arch92 实现与历史 arch35 实现收敛到同一套代码目录和构建入口时，哪些代码应直接共用、哪些差异适合用编译宏隔离、哪些差异应保留为架构专用文件。

本文采用以下设计标准：

1. 仅有少量 include、类型名、常量或单点 API 差异时，在公共文件中使用编译宏或类型别名隔离。
2. 算法主体相同、差异可由模板参数或 policy 表达时，保留公共实现，通过 `if constexpr` 或 policy 选择，不使用大段预处理分支。
3. 控制流、核角色、同步协议或资源生命周期存在较大差异时，保留架构专用文件，通过公共接口调用共享基础能力。
4. 不为了减少文件数量，把完整函数分别放进一个文件中的 `#if/#else` 两侧；这种写法只是把两个文件藏进了一个文件。

结论是：**不建议把 arch92 全部机械塞回 arch35 文件，也不建议继续维护两套完整目录。推荐形成“公共基础实现 + arch35 专用执行文件 + arch92 专用执行文件”的三层结构。**

按当前差异，arch92 建议保留 6 个左右的专用文件：

- arch92 平台 policy；
- arch92 量化接口兼容头；
- arch92 A8W8 Wave；
- arch92 Dispatch 调度封装；
- arch92 GMM1/Activation 调度封装；
- arch92 GMM2/Combine 调度封装。

其余数据结构、Workspace、通用工具、基础搬运、量化、SendMask、共享专家输入和 Unpermute 应合并为公共实现。

## 2. 比较基线

历史提交关系如下：

| 提交 | 用途 |
| --- | --- |
| `e2817c8f4` | 首次增加 arch92 Host、Kernel 和迁移文档 |
| `add347aff` | 删除 arch35 前的最后一版完整 arch35/arch92 共存代码，作为本报告的 arch35 比较基线 |
| `0ec19e054` | 删除 arch35/arch22，形成当前只保留 arch92 的目录 |

历史 arch35 文件可通过下面的命令检视：

```bash
git show add347aff:mc2/mega_moe/op_kernel/arch35/<file>
git show add347aff:mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp
```

当前 arch92 入口为 [`mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:33)，主 Wave 为 [`MegaMoeArch92A8W8Wave`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:47)，Host 入口为 [`MegaMoeTilingFuncImplPublic`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2018)。

## 3. 差异规模

`add347aff` 中 arch35 有 29 个 Kernel 文件，当前 arch92 有 23 个 Kernel 文件，其中 19 个文件能够按相同相对路径直接对应。

以下增删行只用于判断差异规模，不表示每一行都是架构差异；格式调整、函数签名重构和后续公共修复也包含在内。

| 对应文件 | arch35 行数 | arch92 行数 | 增/删规模 | 初步分类 |
| --- | ---: | ---: | ---: | --- |
| `block_scheduler_swizzle.h` | 91 | 91 | 0 / 0 | 直接共用 |
| `mega_moe_constants.h` | 74 | 74 | 0 / 0 | 直接共用 |
| `mega_moe_tiling.h` | 192 | 192 | 0 / 0 | 直接共用 |
| `mega_moe_send_mask.h` | 142 | 142 | 0 / 0 | 直接共用 |
| `mega_moe_shared_expert_input.h` | 80 | 80 | 0 / 0 | 直接共用 |
| `mega_moe_token_quant.h` | 136 | 136 | 0 / 0 | 直接共用 |
| `mega_moe_unpermute.h` | 261 | 261 | 0 / 0 | 直接共用 |
| `gated_activation.h` | 492 | 492 | 9 / 9 | 公共超集 |
| `mega_moe_mxfp8_utils.h` | 120 | 116 | 1 / 5 | 少量兼容隔离 |
| `mega_moe_types.h` | 139 | 124 | 9 / 23 | 公共结构 + 类型兼容 |
| `mega_moe_utils.h` | 302 | 302 | 39 / 39 | 公共实现 + policy |
| `mega_moe_tiling_key.h` | 53 | 55 | 37 / 34 | 使用 arch35 能力超集 |
| `block_epilogue_activation_mx_quant.h` | 703 | 687 | 62 / 76 | 公共模板超集 |
| `mega_moe_gmm_common.h` | 404 | 368 | 14 / 53 | 公共 Generic + arch35 A8W4 policy |
| `mega_moe_workspace.h` | 315 | 333 | 68 / 48 | 公共布局 + Workspace policy |
| `mega_moe_workspace_reset.h` | 82 | 92 | 29 / 19 | 公共清零 + Workspace policy |
| `mega_moe_token_dispatch.h` | 591 | 621 | 192 / 145 | 拆公共数据面与架构调度面 |
| `mega_moe_gmm1_activation.h` | 1121 | 1138 | 506 / 450 | 保留架构调度文件 |
| `mega_moe_gmm2_combine.h` | 1113 | 1322 | 569 / 315 | 保留架构调度文件 |

从规模上看，真正不适合靠零散宏合并的是 Dispatch、GMM1、GMM2 和 Wave。其他大多数文件可以共用或参数化。

## 4. 推荐目录结构

```text
mc2/mega_moe/op_kernel/
├── mega_moe_apt.cpp
├── common/
│   ├── mega_moe_base.h
│   ├── mega_moe_constants.h
│   ├── mega_moe_types.h
│   ├── mega_moe_utils.h
│   ├── mega_moe_workspace.h
│   ├── mega_moe_gmm_common.h
│   └── mega_moe_mxfp8_utils.h
├── stage/
│   ├── mega_moe_token_quant.h
│   ├── mega_moe_send_mask.h
│   ├── mega_moe_shared_expert_input.h
│   ├── mega_moe_unpermute.h
│   ├── mega_moe_token_dispatch_common.h
│   ├── mega_moe_gmm1_common.h
│   └── mega_moe_gmm2_common.h
├── arch35/
│   ├── mega_moe_arch35_policy.h
│   ├── mega_moe_arch35.h
│   ├── mega_moe_wave.h
│   ├── mega_moe_wave_a8w4.h
│   ├── mega_moe_layered.h
│   ├── stage/
│   │   ├── mega_moe_token_dispatch_arch35.h
│   │   ├── mega_moe_gmm1_activation_arch35.h
│   │   └── mega_moe_gmm2_combine_arch35.h
│   └── blaze/
│       └── A8W4 BlockMmad 与 Prologue 文件
└── arch92/
    ├── mega_moe_arch92_policy.h
    ├── mega_moe_a8w8_wave.h
    ├── mega_moe_quant_compat.h
    └── stage/
        ├── mega_moe_token_dispatch_arch92.h
        ├── mega_moe_gmm1_activation_arch92.h
        └── mega_moe_gmm2_combine_arch92.h
```

这里的 `common` 表示两个平台实际共用的算法，不表示把 arch35/arch92 两份文件简单移动到同一目录。

## 5. 逐文件处理建议

### 5.1 直接共用，不增加架构宏

下面 7 个文件在比较基线中内容完全相同，应只保留一份：

- `blaze/gemm/block/block_scheduler_swizzle.h`
- `common/mega_moe_constants.h`
- `mega_moe_tiling.h`
- `stage/mega_moe_send_mask.h`
- `stage/mega_moe_shared_expert_input.h`
- `stage/mega_moe_token_quant.h`
- `stage/mega_moe_unpermute.h`

当前对应实现可从 [`SendMask`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_send_mask.h:34)、[`SharedExpertInput`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_shared_expert_input.h:34)、[`TokenQuant`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_quant.h:76) 和 [`Unpermute`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_unpermute.h:240) 检视。

这些文件加入架构宏没有收益，反而会制造不存在的平台差异。

### 5.2 少量修改，用宏或类型别名隔离

#### 5.2.1 `mega_moe_quant_compat.h`

arch92 的差异只有量化命名空间和 include 搜索路径。当前代码在 `__NPU_ARCH__ == 9201` 时将 `Quant` 映射到 `quant`，见 [`mega_moe_quant_compat.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_quant_compat.h:14)。

建议保留独立兼容头，由公共 `mega_moe_mxfp8_utils.h` 引用。不要在每个量化调用点写架构宏。

#### 5.2.2 `mega_moe_types.h`

公共状态结构应采用当前较清晰的 `ExpertLoopState { problemShape, rowOffset }`。真正的平台差异主要是 Hcomm 类型接口，可以集中成一个别名：

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
using MegaMoeHcomm = Hcomm<CommEngine::AIV, CommProtocol::ROCE>;
#else
using MegaMoeHcomm = Hcomm<COMM_PROTOCOL_UBC_CTP>;
#endif
```

之后 `CombineCommParams` 只引用 `MegaMoeHcomm`。宏只出现一次，结构体本身共用。

#### 5.2.3 `mega_moe_utils.h`

格式调整和专家状态更新不属于平台差异，应统一采用当前实现。arch35 特有的 `GetBlockNum() * 2`、`GetBlockIdx() / GetTaskRation()` 不应继续写死在通用工具里，应改为接受 `JobContext` 或 policy：

```cpp
WorkRange TilingByJobContext(totalLen, job.jobIndex, job.totalJobs, align);
```

只有确实无法通过参数传入的底层 API 差异才使用宏。

#### 5.2.4 `mega_moe_tiling_key.h`

建议保留 arch35 的能力超集：E2M1、MTE、URMA 等 tiling key 都在同一 schema 中声明。arch92 Host 只生成 A8W8 + MTE 子集，不需要再维护一份裁剪后的 tiling key。

这不会让 arch92 自动支持 A8W4/URMA；是否实例化仍由 Host 能力门禁和 Kernel 根入口共同决定。

### 5.3 算法相同，用模板或 policy，不使用大段宏

#### 5.3.1 `block_epilogue_activation_mx_quant.h`

arch35 版本支持 interleaved ping-pong，arch92 当前只使用两块独立 Gate/Up UB。两者的激活和 MX 量化主体相同，适合保留公共模板超集：

```cpp
template <..., bool IsInterleaved = false>
class BlockEpilogueActivationMxQuant;
```

arch35 相应路径实例化 `IsInterleaved=true`，arch92 固定实例化 `false`。这比复制 700 行 epilogue 或在函数内部判断 `__NPU_ARCH__` 更清晰。

#### 5.3.2 `mega_moe_gmm_common.h`

Generic A8W8 QBMM MX 在两边可以共用。arch35 独有的 A8W4 BlockMmad、FP4 Prologue 和 layout selector 应放在 arch35 专用 policy 头中，再由公共 `Config` 接收 selector。

当前 arch92 在 [`GmmKernel::Config`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h:47) 中直接禁止 `IsA8W4`。合并后不应删除这个约束，而应让 arch92 根入口无法选中 A8W4 policy；arch35 实例才包含并实例化 A8W4 类型。

建议只有 A8W4 专用 include 使用架构宏：

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/blaze/gemm/block/block_mmad_mx_fp8fp4.h"
#include "arch35/blaze/prologue/block_prologue_mx_fp8fp4.h"
#endif
```

Generic 配置本身不需要宏。

#### 5.3.3 `mega_moe_workspace.h`

Workspace 中各区域的语义在两边基本一致，差异主要是哪些条件区需要分配以及 Combine slot stride。当前构造函数已经接收 `WorkspaceLayoutPolicy`，见 [`WorkspaceInfo`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_workspace.h:146)。

建议保留一个公共 `WorkspaceInfo`，由 Host 和 Kernel 使用同一个 policy：

- arch35 普通 A8W8：默认布局；
- arch35 A8W4 Wave：仍使用 `DEFAULT` policy，但由 `groupedMatmulMode == A8W4` 分配 cumsum、GMM1 和 GMM2 的 GM 区域；
- arch92 A8W8 Wave：GM handoff Expert Wave 布局。

arch35 A8W4 与 arch92 A8W8 都经过 GM 交接，不代表两者可以使用同一个 Workspace policy。前者的条件区由 A8W4 mode 隐式决定；后者是 A8W8 mode，却需要额外的 Expert Wave flag 和 handoff 区域，因此才新增 `GM_HANDOFF_EXPERT_WAVE`。不要为 arch92 再复制一份 Workspace 公式，也不要在构造函数中直接判断 `__NPU_ARCH__`。Workspace 必须由“实际 Kernel variant”决定，而不只是由芯片代际决定。

#### 5.3.4 `mega_moe_workspace_reset.h`

清零算法完全可以共用，差异只是连续 flag 区总长度。`CalcResetFlagElementCount` 应接收与 Workspace 构造相同的 layout policy，并保证 Host 预算、Kernel 清零和 Workspace 地址推进使用同一条件。

### 5.4 大量控制流差异，保留架构专用文件

#### 5.4.1 Wave

arch35 A8W8、arch35 A8W4 与 arch92 A8W8 的职责分配不同：

- arch35 是 1C2V；
- arch35 A8W4 使用 AIV0 Prologue 和 AIV1 Epilogue/Dispatch/Combine；
- arch92 是 1C1V，只保留原 AIV1 执行链，并使用 token-grained Wave。

当前 arch92 Wave 的专家位置推进、下一 Wave Dispatch 和当前 GMM1 交错位于 [`ProcessMoeExpertStages`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h:236)。这一整段不适合包在 `#if 9201` 中与 arch35 Wave 混写。

结论：保留以下独立文件：

- `arch35/mega_moe_wave.h`
- `arch35/mega_moe_wave_a8w4.h`
- `arch92/mega_moe_a8w8_wave.h`

#### 5.4.2 Token Dispatch

两版约有 192 行新增、145 行删除。差异涉及：

- rank sharding 的调用方式；
- token count 的发布和等待；
- cumsum 是否跨 Wave 写入 GM；
- Dispatch 与 GMM1 tile ready 的协议；
- 物理 AIV 角色守卫。

建议拆成：

- `mega_moe_token_dispatch_common.h`：mask 读取、rank range、token/meta 搬运、GM 地址计算；
- `mega_moe_token_dispatch_arch35.h`：1C2V 角色选择和 arch35 调用签名；
- `mega_moe_token_dispatch_arch92.h`：1C1V rank sharding、持久 cumsum 和 Wave 接口。

当前 arch92 的单专家入口见 [`RunMoeExpertDispatchJob`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h:624)。它可以作为 arch92 专用封装，内部继续调用公共数据搬运函数。

#### 5.4.3 GMM1/Activation

两版约有 506 行新增、450 行删除。差异不仅是 API 名称，而是：

- arch35 AIV0/AIV1 职责；
- A8W4 Prologue；
- L0C 到 UB/GM 交接方式；
- interleaved ping-pong；
- arch92 A8W8 GM handoff；
- Wave 粒度通知和 BlockMmad context 传递。

建议公共文件只保留：

- GMM 地址和 shape 构造；
- Generic AIC MMAD 调用；
- 公共 tile scheduler；
- 公共 activation/quant 算法。

arch35 与 arch92 分别保留上层执行封装。把这 900 行差异放进一个文件的 `#if/#else` 不符合本次设计标准。

#### 5.4.4 GMM2/Combine

两版约有 569 行新增、315 行删除。差异涉及：

- A8W4 Prologue；
- AIC/AIV producer-consumer 数量；
- GM handoff；
- Combine counter stride；
- Wave ready 与 tile completion 通知；
- 1C1V 下完整 consumer 集合。

基础的 Combine 数据搬运、量化、meta 读取可以共用；调度与同步封装应保持架构独立。

当前 arch92 的 GMM2 入口和 Wave 通知集中在 [`mega_moe_gmm2_combine.h`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h:853)。建议抽取公共计算函数，而不是把整个文件直接覆盖到 arch35。

## 6. 平台 policy 设计

当前 [`MegaMoeArch92ArchitecturePolicy`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18) 同时表达了物理架构和执行路径。合并时建议拆成两个维度。

### 6.1 Hardware policy

只描述硬件和 task type 导致的事实：

```cpp
struct Arch35HardwarePolicy {
    static constexpr uint32_t VECTOR_PER_CUBE = 2U;

    __aicore__ static uint32_t GetLogicalBlockIdx()
    {
        return GetBlockIdx() / GetTaskRation();
    }
};

struct Arch92HardwarePolicy {
    static constexpr uint32_t VECTOR_PER_CUBE = 1U;

    __aicore__ static uint32_t GetLogicalBlockIdx()
    {
        return GetBlockIdx();
    }
};
```

### 6.2 Execution policy

描述某条 Kernel variant 的交接方式：

```cpp
struct DefaultA8W8FlowPolicy {
    static constexpr bool PERSIST_DISPATCH_CUMSUM = false;
    static constexpr bool USE_GM_HANDOFF = false;
    static constexpr WorkspaceLayoutPolicy WORKSPACE_LAYOUT = WorkspaceLayoutPolicy::DEFAULT;
};

struct Arch35A8W4WaveFlowPolicy {
    static constexpr bool PERSIST_DISPATCH_CUMSUM = true;
    static constexpr bool USE_GM_HANDOFF = true;
    // 历史 arch35 Workspace 通过 groupedMatmulMode == A8W4 选择条件区。
    static constexpr WorkspaceLayoutPolicy WORKSPACE_LAYOUT = WorkspaceLayoutPolicy::DEFAULT;
};

struct Arch92ExpertWaveFlowPolicy {
    static constexpr bool PERSIST_DISPATCH_CUMSUM = true;
    static constexpr bool USE_GM_HANDOFF = true;
    static constexpr WorkspaceLayoutPolicy WORKSPACE_LAYOUT =
        WorkspaceLayoutPolicy::GM_HANDOFF_EXPERT_WAVE;
};
```

这样可以表达：

| 平台/路径 | Hardware policy | Execution policy |
| --- | --- | --- |
| arch35 普通 A8W8 | `Arch35HardwarePolicy` | `DefaultA8W8FlowPolicy` |
| arch35 A8W4 Wave | `Arch35HardwarePolicy` | `Arch35A8W4WaveFlowPolicy` |
| arch92 A8W8 Wave | `Arch92HardwarePolicy` | `Arch92ExpertWaveFlowPolicy` |

这三个显式参数不是为了给现有代码换名字，而是为了解除 arch35 中成立、合到 arch92 后不再成立的隐式绑定：

1. arch35 用 `EnableA8W4` 同时决定 Dispatch cumsum 是否写 GM；arch92 虽然是 A8W8，也需要跨 Wave 持久化 cumsum。
2. arch35 通过选择 `Gmm1AicMmadTileToGmGeneric` 或 `Gmm1AicMmadTileToUbGeneric` 隐式决定 handoff；arch92 A8W8 也要走 GM handoff，不能再从量化模式反推。
3. arch35 Workspace 根据 `groupedMatmulMode == A8W4` 分配 handoff 区；arch92 的 grouped mode 是 A8W8，只能显式传入 `GM_HANDOFF_EXPERT_WAVE` 才能让 Host 预算、Kernel 地址推进和 reset 范围一致。

因此，三个 policy 值必须由具体 Kernel variant 选择，不能只按 arch35/arch92 或 A8W8/A8W4 二选一。这能避免把“arch92”错误等同于“必须 GM handoff”，也避免把“A8W8”错误等同于“不需要持久 cumsum”。当前 arch92 三个值的集中定义可在 [`MegaMoeArch92ArchitecturePolicy`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/arch92/common/mega_moe_core_context.h:18) 检视。

## 7. 编译宏边界

可以参考 [`moe_distribute_dispatch_v2.h`](/D:/gitcode/ops-transformer_910b2/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h:23) 使用 `__NPU_ARCH__`，但 MegaMoe 只应在以下位置使用。

### 7.1 允许使用宏的位置

1. 根入口选择 task type：

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
```

当前 arch92 固定使用 1C1V，见 [`mega_moe_apt.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_kernel/mega_moe_apt.cpp:44)。

2. 根入口 include 和 Kernel variant 实例化。
3. arch35 独有 A8W4/Layered 类型的 include。
4. Hcomm、量化命名空间等无法用同一 C++ 类型表达的兼容点。

### 7.2 不应使用宏的位置

以下内容应由 policy 或普通参数表达：

- `blockIdx`、`blockNum` 和 consumer 数量；
- Workspace layout；
- cumsum 是否持久化；
- GM/UB handoff；
- rank sharding 的 job 数；
- Combine slot stride；
- UB buffer 数量和 batch 大小。

这些值会同时影响 Host 和 Kernel。直接使用设备宏会让 Host 无法共享同一套决策。

## 8. Host Tiling 合并方案

Host 代码不能依赖 `__NPU_ARCH__`，应从平台信息取得架构：

```cpp
auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
NpuArch npuArch = platform.GetCurNpuArch();
```

建议只保留一个 `mega_moe_tiling.cpp` 和一个 `IMPL_OP_OPTILING(MegaMoe)` 注册，内部构造 Host policy：

```cpp
struct MegaMoeHostPolicy {
    bool isArch92;
    uint32_t vectorPerCube;
    WorkspaceLayoutPolicy workspaceLayout;
    bool supportsUrma;
    bool supportsA8W4;
};
```

Host 的处理边界如下：

| 内容 | 处理方式 |
| --- | --- |
| 输入 shape、dtype、TensorList 一致性校验 | 公共 |
| UB 容量读取和自适应 batch 公式 | 公共，直接使用平台真实 UB |
| arch35 A8W4/A4W4/URMA 能力 | arch35 Host policy |
| arch92 当前 A8W8 + MTE 门禁 | arch92 Host policy |
| `aivNum == 2 * aicNum` 或 `aivNum == aicNum` | policy 校验 |
| Combine slot 数量 | 根据逻辑 consumer 数计算，不在公式里硬编码 `/2` |
| Workspace 构造 | 将同一个 `workspaceLayout` 传给公共 `WorkspaceInfo` |

当前 arch92 的能力门禁和 1C1V 核数检查位于 [`CheckArchitectureCapability`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:1978) 和 [`MegaMoeTilingFuncImplPublic`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/op_tiling/arch92/mega_moe_tiling.cpp:2018)。这些适合提取成小型 `arch92/mega_moe_tiling_policy.h`，不需要保留整份 arch92 tiling cpp。

## 9. 构建与注册

### 9.1 OpDef

当前 OpDef 只注册 `ascend910_96`，见 [`mega_moe_def.cpp`](/D:/gitcode/ops-transformer_910b2/mc2/mega_moe/op_host/mega_moe_def.cpp:194)。合并后需恢复 arch35 的 `ascend950` 配置，同时保留 A6 配置。

两种 SoC 可以继续指向同一个 `mega_moe_apt`，由设备编译宏选择实际入口。

### 9.2 CMake

历史 arch35 使用 `--cce-auto-sync=off`，当前 arch92 使用 `--cce-auto-sync=on`。这是编译选项差异，不能仅靠头文件宏处理，CMake 必须按 `ASCEND_COMPUTE_UNIT` 保留分支。

### 9.3 Tiling 注册

必须只有一个：

```cpp
IMPL_OP_OPTILING(MegaMoe)
```

不能同时编译历史 arch35 tiling cpp 和当前 arch92 tiling cpp，否则会发生重复注册或运行时选择不明确。

## 10. 实施顺序

建议按以下顺序进行，每一步都保持至少一个平台可编译：

1. 从 `add347aff` 恢复 arch35 专用 Wave、A8W4 Prologue、Layered 和必要的构建配置，不恢复历史调试代码。
2. 将 7 个完全相同文件移到公共目录，修改双方 include，行为不变。
3. 统一 `mega_moe_tiling.h` 和 tiling key 能力超集，先固定 Host/Kernel ABI。
4. 增加 Hardware policy 和 Execution policy，替换公共代码中的固定 `*2`、`/2`、`GetTaskRation()` 和 handoff 条件。
5. 合并 Workspace 和 reset 计算，增加 Host/Kernel policy 一致性静态检查或日志。
6. 抽取 Dispatch 公共数据面，保留两个架构调度封装。
7. 抽取 GMM1/GMM2 公共 MMAD、地址和量化函数，保留架构调度封装。
8. 合并根入口，通过 `__NPU_ARCH__` 选择 task type、include 和 Kernel variant。
9. 合并 Host tiling，按 `GetCurNpuArch()` 选择 Host policy，只保留一次 tiling 注册。
10. 恢复 `ascend950` OpDef 与 CMake 编译选项，完成两平台编译和回归。

不要先把所有文件移动到同一目录再一次性修编译错误。那样很难区分 include 冲突、模板实例化错误和同步协议错误。

## 11. 回归矩阵

### 11.1 最低编译矩阵

| 平台 | 路径 |
| --- | --- |
| arch35 / DAV_3510 | A8W8 ND |
| arch35 / DAV_3510 | A8W8 NZ |
| arch35 / DAV_3510 | A8W4 |
| arch35 / DAV_3510 | A4W4 预留实例 |
| arch35 / DAV_3510 | URMA/Layered（若继续作为正式能力） |
| arch92 / DAV_9201 | A8W8 ND |
| arch92 / DAV_9201 | A8W8 NZ |

### 11.2 功能矩阵

- 单卡和多卡；
- `topkWeightsPrefetch` 开/关；
- 无共享专家和 1～4 个共享专家；
- Combine no-quant、E4M3、E5M2；
- 零 token expert、小 expert、热点 expert 和跨 Wave expert；
- `maxRecvTokenNum=0` 与显式上界；
- 最大 H、hiddenDim、topK 和 BS；
- expert 数不能被核数整除的分配；
- expert token count 输出；
- Workspace flag 初值、producer 次数和 consumer 目标值。

### 11.3 协议重点

1. arch35 的逻辑 AIC job 数不能误用物理 AIV 数。
2. arch92 的 `blockAivNum` 已经是完整 1C1V consumer 数，不能再除以 2。
3. Host `combineSyncSlotCountPerExpert`、Kernel slot 索引和 reset 元素数必须使用相同 policy。
4. `WorkspaceInfo` 的 Host 空基址预算和 Kernel 实际地址推进必须使用相同 layout。
5. arch35 AIV0/AIV1 角色不能因为公共化而同时执行同一份 epilogue。
6. arch92 不得意外实例化 A8W4 Prologue 或 URMA/Layered。

## 12. 工作量评估

| 工作项 | 预计难度 | 说明 |
| --- | --- | --- |
| 恢复历史 arch35 专用文件 | 低 | Git 历史完整，但需排除历史调试代码 |
| 完全相同文件公共化 | 低 | 主要是 include 修改 |
| policy 和根入口 | 中 | 需同时覆盖 1C2V/1C1V |
| Workspace/Reset 合并 | 中 | 公式不复杂，但错误会导致地址错位或死锁 |
| Host tiling 合并 | 中 | 约 279/241 行差异，需单注册和运行时平台选择 |
| Dispatch 拆分 | 中高 | 涉及 cumsum、rank sharding 和 tile ready |
| GMM1/GMM2 拆分 | 高 | 涉及 Prologue、handoff、Wave 和同步协议 |
| 双平台回归 | 高 | FastModel 单次耗时较长，组合数量多 |

初步估计：

- 只完成代码结构合并和两个平台基础编译：3～5 个工作日；
- 保留 arch35 全部现有能力并完成关键功能回归：7～10 个工作日；
- 若 URMA/Layered 也要求完整回归，日历时间可能进一步增加。

## 13. 最终建议

按照“大改动拉新文件、小改动用宏隔离”的标准，推荐边界如下：

### 保留 arch92 专用文件

1. `mega_moe_arch92_policy.h`
2. `mega_moe_quant_compat.h`
3. `mega_moe_a8w8_wave.h`
4. `mega_moe_token_dispatch_arch92.h`
5. `mega_moe_gmm1_activation_arch92.h`
6. `mega_moe_gmm2_combine_arch92.h`

### 使用少量宏隔离

1. 根入口的 1C1V/1C2V task type；
2. arch35 A8W4/Layered 专用 include；
3. Hcomm 类型别名；
4. arch92 量化命名空间兼容；
5. 根入口 Kernel variant 实例化。

### 合并为公共代码

1. Tiling schema 和 tiling key 超集；
2. 常量、类型主体和工具函数；
3. Workspace 和 reset 算法；
4. Quant、SendMask、共享专家输入、Unpermute；
5. Generic A8W8 BlockMmad 配置；
6. Dispatch/GMM1/GMM2 的底层数据操作和计算函数。

这个方案不会追求最少文件数，而是让每个架构专用文件都对应一个真实的平台控制流差异。代码检视时可以先看公共协议，再分别检查 arch35 和 arch92 的薄调度层，不需要在上千行文件中追踪交错的预处理分支。
