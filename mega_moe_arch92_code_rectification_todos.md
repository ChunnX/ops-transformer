# MegaMoe arch92 未完成代码整改 TODO

## 1. 文档用途

本文档只汇总 A6（arch92）A8W8 迁移中尚未完成、需要继续修改代码的事项，作为后续整改入口。

不纳入本清单：

- 已经完成并验证的迁移修改。
- 单纯的测试用例补充、性能测试和文档绘图。
- `ascend910_96 -> arch92` 等 MegaMoe 目录之外的最终编译仓路由配置。
- A6 明确不支持的 URMA、Layered、A8W4 和 A4W4 功能开发。
- 只需升级外部 CANN、无需修改本仓代码的问题；如果本仓存在临时绕过，则只记录绕过的恢复和清理工作。

优先级定义：

- **P0**：影响当前 A8W8 正确性或 A5 已有特性完整性，必须先完成。
- **P1**：消除架构迁移遗留和无效分支，降低后续修改风险。
- **P2**：利用 A6 384KB UB 或调整流水的性能整改。
- **P3**：依赖 profiling 或较大范围协议重构的后续优化。

## 2. 总览

| 编号 | 优先级 | 状态 | 整改项 | 主要依赖 |
|---|---|---|---|---|
| R01 | P0 | 定位中 | 修正 arch92 自研 A8W8 BlockMmad 的 GMM2 结果 | A6 实机日志 |
| R02 | P0 | 外部阻塞/本仓有绕过 | 恢复共享专家 GMM1 路径 | CANN CopyL0C2UB 接口一致性或本地兼容实现 |
| R03 | P1 | 未完成 | 删除 arch92 A8W4/A4W4 类型和控制流残留 | R01 稳定后执行 |
| R04 | P1 | 未完成 | 系统清理 1C2V、AIV0/AIV1、subBlock 遗留语义 | 无 |
| R05 | P1 | 未完成 | 收敛 arch92 Dispatch cumsum 策略接口 | R08 决定最终存储方式 |
| R06 | P1 | 未完成 | 裁剪 arch92 Host Tiling 中不可达模式 | R03 |
| R07 | P1 | 未完成 | 收敛 Workspace 条件分配和 Host/Kernel 公式 | R02、R08 |
| R08 | P2 | 未实现 | 将 Dispatch cumsum 常驻高地址 UB | R09 |
| R09 | P1/P2 | 未实现 | 建立 arch92 384KB UB layout policy | 无 |
| R10 | P2 | 未实现 | GMM1 GM 输出到 Activation 的 128 行双缓冲 | R01、R09 |
| R11 | P2 | 未实现 | 非量化 GMM2 输出到 Combine 的 tile 双缓冲 | R01、R09 |
| R12 | P2 | 未实现 | 前后处理阶段独占并复用 UB | R09 |
| R13 | P3 | 待 profiling | 扩大量化 Combine ring | R09 |
| R14 | P3 | 未实现 | Dispatch/Activation/Combine 跨阶段 MTE 重叠 | R09、R10/R11 |
| R15 | P3 | 明确延后 | 改进 Wave 专家装箱，使每轮在超过目标负载后尽量贴近目标 | 当前首版正确性稳定 |
| R16 | P1 | 待正确性闭环后清理 | 删除 BlockMmad 临时诊断代码和解码脚本 | R01 |
| R17 | P1 | 未完成 | 清理 arch92 Params/公共类型中的 URMA 不可达代码 | R01、R02 稳定后执行 |

## 3. P0 正确性与功能缺口

### R01 修正自研 A8W8 BlockMmad 的 GMM2 结果

**当前事实**

- Dispatch、GMM1 gate/up、SwiGLU、Activation 量化数据和量化 scale 已与 golden 对齐。
- GMM2 的完整 Activation、ScaleA、Weight2、ScaleB hash 已与 golden 对齐。
- 第一个错误边界是 GMM2 输出。
- 当前典型问题形状为 `M=1, N=4096, K=512, scaleK=16`，首个输出 tile 为 `1x256`。
- arch92 使用仓内专用 `mega_moe_block_mmad_mx_fp8.h`，不能依赖 A6 CANN 包中不存在的 A5 `BlockMmad<MatmulWithScaleMx>` 实现。

**待修改**

1. 根据一次性打点结果确定首次错误发生在：
   - 第一个或后续 K=128 累加片段；
   - MX ScaleA/ScaleB 的 K-group 地址或装载布局；
   - L1/L0A/L0B 搬运和 ND/NZ 路由；
   - MMAD 累加初始化/复用；
   - L0C 到 GM 的 Fixpipe 输出。
2. 对照 `ops-tensor/include/blaze/gemm/block` 中 A6 原生 A8W8 实现核对以下行为，而不是直接引用 ops-tensor 头文件：
   - `baseM/baseN/baseK` 和小 M 路由；
   - ScaleA/ScaleB 的物理排布、stride 和 K-group 推进；
   - K-tail 与 `scaleK` 对齐；
   - L0A/L0B ping-pong 和 L0C 累加生命周期；
   - Fixpipe 的 dtype、quant mode、stride 和同步。
3. 保持 GMM1 与 GMM2 共用同一 BlockMmad 主体，但不能以“GMM1 正确”推断 GMM2 的所有参数组合正确；两者的 M、N、K、尾块和权重布局不同。
4. 修复必须作用于业务输出路径，不能只修改诊断用的 L0C2UB 旁路。

**主要文件**

- `mc2/mega_moe/op_kernel/arch92/blaze/gemm/block/mega_moe_block_mmad_mx_fp8.h`
- `mc2/mega_moe/op_kernel/arch92/common/mega_moe_gmm_common.h`
- `mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h`
- `mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h`

**完成标准**

- GMM1 和 GMM2 的关键中间值及最终输出均与 golden 对齐。
- 覆盖至少小 M、正常 M、K-tail、ND/NZ 权重和 E4M3/E5M2 合法组合。
- 修复后没有保留会覆盖业务结果的诊断旁路。

### R02 恢复共享专家 GMM1

**当前事实**

- 共享专家 GMM1 当前因 CANN Tensor API `CopyL0C2UB` 参数签名不一致被临时注释。
- A6 除 URMA 外应支持 A5 已有特性，因此跳过共享专家不是最终可接受实现。
- 该问题触发于共享专家 `RunGmm1Generic<..., QF322BF16_PRE, ...>` 的 L0C 到 UB 路径，普通 MoE 专家当前未由此报错。

**待修改**

1. 优先在兼容版本 CANN 上恢复 `ProcessSharedExpertGmm1` 并验证。
2. 如果目标 CANN 仍不一致，为 arch92 提供局部兼容实现，避免修改 A5 文件或全局篡改 CANN 头：
   - 使用目标 SoC 实际生效的 `asc_copy_l0c2ub` 重载；或
   - 将共享专家 epilogue 改为 arch92 可编译且语义一致的输出路径。
3. 删除 `skip for now`/注释调用等临时绕过。
4. 重新检查共享专家 GMM1、Activation、GMM2 的 Workspace、同步和最终融合路径都实际可达。

**主要文件**

- `mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h`
- `mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h`
- `mc2/mega_moe/op_kernel/arch92/blaze/epilogue/block_epilogue_activation_mx_quant.h`

**完成标准**

- `sharedExpertNum > 0` 不再跳过 GMM1。
- 共享专家开启/关闭均能编译，开启时结果正确且不会破坏 MoE 专家流水。

## 4. P1 架构代码收敛

### R03 删除 arch92 A8W4/A4W4 残留

arch92 当前产品范围只有 A8W8，但本地主类和部分 stage 仍保留从 arch35 复制来的模式定义与兼容参数，例如：

- `ENABLE_A8W4`、`ENABLE_A4W4`。
- 名字或模板参数中的 `EnableA8W4`。
- `if constexpr (ENABLE_A8W4 || ...)` 等永远不会选择第二条有效 A6 路径的判断。
- 为旧调用签名保留的 Dispatch wrapper。
- Workspace 中以 A8W4/A4W4 名义存在、但 arch92 实际由 GM handoff 或 A8W8 policy 使用的条件。

**待修改**

1. 从 arch92 主类、stage、common 和 Host Tiling 中删除不可达的数据类型与模板实例。
2. 对实际仍需要的行为重新按语义命名，例如 `UseGmHandoff`、`UseRankSharding`，不要借用 `EnableA8W4` 表达 A6 行为。
3. 删除只为兼容 arch35 调用签名存在的 wrapper，让 arch92 调用点直接使用本地接口。
4. 保留 A5 arch35 文件零侵入。

**完成标准**

- `rg "A8W4|A4W4|EnableA8W4|ENABLE_A8W4|ENABLE_A4W4" mc2/mega_moe/op_kernel/arch92 mc2/mega_moe/op_host/op_tiling/arch92` 只剩必要注释或为 Host 明确拒绝非法输入的常量，不再有可实例化 Kernel 路径。

### R04 清理 1C2V/AIV0/AIV1/subBlock 遗留

A6 是默认 1C1V，不应通过 `oneAiv` 等命名体现“特殊模式”，也不应残留 A5 的角色映射。

**重点审计**

1. 删除仅服务 AIV0 Prologue 的等待、flag 和状态变量，包括已经没有生产者/消费者的 `WaitUntilGmFlagIsNonZero` 类逻辑。
2. 删除 `GetSubBlockIdx()==0/1` 对 AIV0/AIV1 的角色选择；1C1V 下 AIV job 应由物理 block/job context 直接表达。
3. 检查 `GetTaskRation()`、`GetBlockNum()*2`、`aivNum=2*aicNum` 和偶/奇 AIV 映射。
4. 检查 Dispatch、SwiGLU、Combine 的参与核数、job index 和同步目标是否仍隐含 1C2V。
5. 删除已无消费者的 Workspace flag、CV flag 和 reset 范围，不能只删等待代码而保留无效协议。
6. 保留 `TilingByCore` 等仍参与 1C1V 分核的通用函数；整改目标是删除旧角色语义，不是删除分核本身。

**完成标准**

- arch92 活跃路径不再通过 subBlock 区分两个 Vector 角色。
- 每个 flag 都能明确列出唯一生产者、消费者和 reset 方。
- AIC/AIV 在同一物理 block 上使用一致的 block index 语义。

### R05 收敛 Dispatch cumsum 策略接口

`MegaMoeArch92ArchitecturePolicy::PERSIST_DISPATCH_CUMSUM` 当前恒为 `true`，`PersistDispatchCumsum` 的 `if constexpr` 不产生运行时成本，但保留了不存在的第二种 arch92 实例和错误配置空间。

**待修改**

- R08 实施前：arch92 本地 Dispatch 直接无条件执行 GM 恢复/写回，删除模板参数和 policy 常量。
- R08 实施后：删除 GM 恢复/写回、`cumsumInfoGlobalTensor` 和对应 Workspace，直接使用常驻 UB。
- 不应只把常量改成 `false` 后继续保留双路径代码。

### R06 裁剪 arch92 Host Tiling 不可达模式

arch92 Host Tiling 由 arch35 副本演化而来，需要在 Kernel 正确性稳定后进一步收窄：

1. 只生成 arch92 A8W8 支持的 tiling key 和 dtype/layout 组合。
2. 删除只为 A8W4、A4W4、Layered、URMA 服务的 buffer、workspace 和调度计算。
3. 保留并明确校验 1C1V：`blockAivNum == aivNum == aicNum`。
4. `availableUbBytes` 必须来自平台查询，不写死 `384 * 1024`。
5. Host 拒绝的模式必须与 Kernel 不再实例化的模式一致。

### R07 收敛 Workspace 条件分配与公式

**待修改**

1. 清理 R03/R04 后不再使用的 A8W4/A4W4、AIV0/AIV1 flag 和 buffer。
2. R08 完成后删除 `cumsumInfoPtr` 及每核 cumsum GM 备份空间。
3. R02 恢复后重新核对共享专家 Workspace，避免“已分配但业务被跳过”或恢复后空间不足。
4. Host Tiling 和 Kernel 必须使用同一个布局策略/同一组公式描述以下区域：
   - GMM1/GMM2 GM handoff；
   - dispatch/gmm/activation/combine flags；
   - Wave sequence/group counter；
   - Combine quant/non-quant 条件区域；
   - shared expert 区域。
5. 每次删除区域时同步修改 reset 范围和总 Workspace 大小。

### R17 清理 URMA 不可达类型和参数

A6 首版明确不支持 URMA。当前为了低风险迁移可能仍在公共 `Params` 或 Hcomm 兼容签名中保留 URMA 字段。

**待修改**

1. 在 R01/R02 正确性稳定后，沿 arch92 本地类型闭包删除不会被 MTE 路径读取的 URMA 参数、分支和辅助函数。
2. 若公共 ABI/模板签名要求保留字段，明确限制为“结构兼容但不可达”，不得在 arch92 生成 URMA 实例。
3. 不为清理 arch92 而修改 arch35 公共热路径；必要时继续使用 arch92 本地副本。

## 5. 384KB UB 整改

### R09 建立 arch92 UB layout policy

这是 R08、R10、R11 和 R12 的共同前置工作。

**待修改**

1. 将 arch92 固定 UB 地址和 256KB 容量判断集中到本地 layout policy，至少覆盖：
   - epilogue 中的 `256U * 1024U` 判断；
   - `META_INFO_TENSOR_ADDR = 200KB`；
   - `WAVE_COMBINE_UB_BASE = 64KB`；
   - `WAVE_GMM2_READY_SCAN_UB_ADDR = 184KB`；
   - Dispatch、Activation、Combine 与 cumsum 持久区边界。
2. Host 从平台读取实际可用 UB，并把容量/布局所需参数传给 Kernel。
3. 每个 stage 声明自己的最大地址和生命周期；复用只能发生在已完成所有异步操作的阶段之间。
4. A5 继续使用原有 256KB 布局。

**完成标准**

- arch92 没有散落的、未经 policy 解释的 256KB 假设或高地址常量。
- 所有合法特性组合下，最高 UB 地址均不超过平台提供容量。

### R08 Dispatch cumsum 常驻 UB

`cumsumInfoTensor` 大小为：

```text
Align32(moeExpertPerRank * epWorldSize * sizeof(int32_t))
= Align32(moeExpertNum * 4B)
```

当前 `moeExpertNum <= 2048`，因此最多约 8KB，适合从 A6 高地址 UB 划出常驻区。

**待修改**

1. 从实际可用 UB 顶端预留动态对齐的 cumsum 区，不直接写死 384KB 地址。
2. Dispatch、Activation、Combine 和后处理均不得覆盖该区。
3. 每个专家直接在 UB 中延续前缀和，删除每轮 GM 恢复和回写。
4. `ExportExpertTokenCounts` 直接从常驻 UB 导出。
5. 删除 `cumsumInfoPtr`、`cumsumInfoGlobalTensor`、相关 Workspace 和 `PersistDispatchCumsum` 模板维度。

**收益判断**

当前实现对专家 `e` 恢复前 `e` 个专家并写回前 `e+1` 个专家的 prefix，累计搬运量近似随本卡专家数二次增长。收益不只是省去一次 8KB copy，而是消除逐专家 MTE2/MTE3 和等待。

### R10 GMM1 到 Activation 的 128 行双缓冲

保持 AIC GMM1 scheduler tile 为 256 行，AIV 将 GM 中的 GMM1 输出按 128 行消费：

```text
bank0: gate 64KB + up 64KB = 128KB
bank1: gate 64KB + up 64KB = 128KB
其余                             128KB
```

**待修改**

1. 重叠 half0 Vector/量化与 half1 GM2UB，并尝试预取下一 tile。
2. 不改变外部 256 行 GMM ready 协议，除非同步 flag、Host 预留和 reset 一并重构。
3. topk weights 和 metadata 不得与两个活跃 bank 或 R08 常驻区重叠。
4. 覆盖所有 A5 已支持激活函数。

### R11 非量化 GMM2 到 Combine 的 tile 双缓冲

一个 `256x256` BF16 GMM2 输出 tile 约 128KB，可使用两个 bank，并把 metadata/ready scan 放入剩余区域。

**待修改**

1. 重叠当前 tile 的 MTE3/发送与下一 tile 的 MTE2。
2. bank 复用前等待对应异步操作完成。
3. AIV 仍按 AIC 发布的 sequence 消费，不能越过未完成 tile。
4. 非量化 Combine 保持每个 GMM2 tile 启动一次；量化 Combine 不套用该设计。

### R12 前后处理阶段独占并复用 UB

当前 Reset、Quant、SendMask 等阶段严格串行，却可能长期保留互不重叠视图。后续改为阶段级布局：

- Quant 独占主要 UB，并改进多 token batch 双缓冲。
- SendMask 复用同一区域并扩大 route batch。
- Reset 再次复用该区域。
- Unpermute 根据实际 UB 扩大 input ring、token batch 和 topk weight batch。

阶段切换前必须回收上一阶段全部 Event 和异步 MTE。

### R13 扩大量化 Combine ring

仅在 profiling 显示 MTE3 回收等待明显时，将当前双 row buffer 扩展为 3 或 4 slot。需要同时满足 Event ID 数量和 R09 布局约束，且不能改变“本 AIV 所需 GMM2 token 全部 ready 后才启动量化 Combine”的同步语义。

### R14 Dispatch/Activation/Combine 跨阶段 MTE 重叠

这是最后评估的激进优化，不进入近期正确性整改：

1. 为三个 stage 分配互不重叠的活跃 buffer 和独立 Event 集合。
2. 把函数末尾 Event drain 提升到 Wave 控制层统一管理。
3. 重新验证 Dispatch 领先水位、SwiGLU 对 GMM2 的直接阻塞和 Combine 完成时机。
4. 只有 profiling 证明收益覆盖额外 UB 占用与同步复杂度时实施。

## 6. Wave 调度整改

### R15 改进 Wave 专家装箱策略

该项已经明确不进入第一版。

**目标不是“尽量靠近两个 tile”**，而是：

- 采用 A8W4 token-grained Wave 主体；
- 当已达到至少两个 GMM tile 的安全水位后，再选择超过目标但最接近目标的专家/token 边界；
- 超长专家允许从中间切断，剩余 token 进入下一 Wave；
- 目标是让每轮核心尽量处理完整的两个 tile，同时维持 Dispatch 至少领先 GMM1 所需的安全水位，避免流水等待。

**待修改**

1. 将 Wave 构造从单纯专家边界推进扩展为专家内 token 游标。
2. GMM1、GMM2、Dispatch、Combine 统一使用 `(expert, token-range)` 描述，不能各自重新推导边界。
3. 空专家、单个超长专家、尾 Wave 和 `maxOutputSize` 截断必须保持单调推进且不重复/遗漏 token。
4. 同步 flag 和 sequence 仍按实际 tile 生产消费，不能只改变 planner 而不改变消费者边界。

## 7. 临时代码清理

### R16 删除 BlockMmad 诊断设施

R01 完成前保留一次性诊断，避免每次 A6 用例运行半小时却只得到单一结果。R01 正确性闭环后必须删除：

- `912101..912106`、`912201..912203` 等条件日志 marker。
- GMM1 Fixpipe 对照、GMM1/GMM2 direct L0C2UB 对照和 partial-K dump。
- 仅为诊断预留的 UB 地址、helper 和编译期开关。
- `mc2/mega_moe/tests/decode_arch92_mmad_debug.py`；如果仍有复用价值，则迁入明确的 debug 工具目录并默认不参与产品构建。
- golden 脚本中只为本次 BlockMmad 定位添加、且不属于长期回归接口的密集 hash/window 输出。

清理前保存最后一次有效定位结论，防止删掉打点后丢失问题边界。

## 8. 推荐执行顺序

1. **R01**：先闭环当前 GMM2 正确性。
2. **R02**：恢复共享专家，补齐 A5 已有功能。
3. **R16**：删除临时诊断，恢复干净业务代码。
4. **R03 + R04 + R06 + R17**：集中清理 A8W4/A4W4、1C2V、URMA 和 Host 不可达分支。
5. **R09**：建立统一的 arch92 UB layout policy。
6. **R08 + R05 + R07**：cumsum 常驻 UB，并同步收敛策略与 Workspace。
7. **R10**：实现 GMM1 到 Activation 的 128 行双缓冲。
8. **R11 + R12**：根据 profiling 继续消费额外 UB。
9. **R13 + R14 + R15**：作为独立性能课题逐项评估，不能和正确性修复混在同一次提交。

## 9. 已完成但不再列为 TODO 的事项

- `mega_moe_def.cpp` 增加 `910_96` 能力注册。
- arch92 Kernel 入口放到 `op_kernel/mega_moe_apt.cpp`，并显式包含 arch92 头。
- arch92 Host Tiling 使用可解析的本地头文件路径。
- FP8 类型去掉错误的 `AscendC::` 命名空间前缀。
- arch92 本地 A8W8 BlockMmad 已建立，当前剩余的是 R01 正确性问题。
- BlockMmad 诊断中的 `GetBlockIdx` 已改为 `AscendC::GetBlockIdx`。
- 第一版沿用 A5 最新 A8W4 token-grained Wave 的专家内切分能力。
- Host 已通过平台接口读取 UB 大小并生成 Dispatch、SendMask、Unpermute 自适应 buffer 配置。

