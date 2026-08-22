# AlltoAllvGroupedMatMulV2 AIV 重写设计

## 1. 目标与范围

本设计重写 `AlltoAllvGroupedMatMulV2` 的通信与计算主链路，用于 MoE decode 小 shape 场景。算子融合：

```text
ataOut     = AlltoAllV(gmmX)
permuteOut = PermuteByLocalExpert(ataOut)
gmmY       = GroupedMatMul(permuteOut, gmmWeight)
```

首版范围：

- 目标平台：Ascend 910C（`Ascend910_93`）。
- 目标 CANN：9.2 及以上；当前开发机仅检测到 CANN 9.1.0，编码可先进行，最终编译与验收需补齐 9.2+ 环境。
- 数据类型：仅 BF16 输入、BF16 权重、BF16 输出，Cube 内部 FP32 累加。
- 数据格式：仅 ND。
- 通信模式：V2 仅支持 `commMode="aiv"`。
- 调度粒度：每轮仅处理一个本地专家，循环 `expertPerRank` 次。
- 不实现 chunk、多专家合并或 M-tile 通信策略。
- 通信不调用 `hccl_.AlltoAllV()` 或其他 AlltoAllV 高级通信接口。
- 主 GMM 与共享专家 MM 均使用 CATLASS，不使用 MatMul/GroupedMatMul 高级 API。
- 保留现有 V2 ACLNN 接口、共享专家可选输入输出以及 `permuteOut` 可选输出。
- V1 保留原有 `ai_cpu` 实现，不因 V2 重写而改变行为。

非目标：

- FP16、INT8、量化或反量化路径。
- 跨超节点通信。
- 算子并发调用。
- 动态 Tensor 形式的 send/recv counts；首版延续现有 V2 仅接受 `aclIntArray` counts 的行为。

## 2. 接口和兼容边界

V2 ACLNN 两段式接口签名保持不变：

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX,
    const aclTensor *gmmWeight,
    const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional,
    const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional,
    const char *group,
    const char *commMode,
    int64_t epWorldSize,
    const aclIntArray *sendCounts,
    const aclIntArray *recvCounts,
    bool transGmmWeight,
    bool transMmWeight,
    bool permuteOutFlag,
    aclTensor *gmmY,
    aclTensor *mmYOptional,
    aclTensor *permuteOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnAlltoAllvGroupedMatMulV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

接口行为调整：

- `commMode` 仅接受 `"aiv"`，默认值改为 `"aiv"`。
- V2 执行前通过 `NnopbaseSetHcclServerType` 选择 MTE server。
- `sendCountsTensorOptional` 和 `recvCountsTensorOptional` 首版必须为空。
- `sendCounts`、`recvCounts` 必须非空，由 host 序列化到 tiling data。
- V1 wrapper、V1 tiling 和旧 kernel 路径继续支持原 `ai_cpu` 行为。

## 3. 数据语义与输出顺序

定义：

- `R = epWorldSize`。
- `E = expertPerRank = gmmWeight.shape[0]`。
- 全局专家数为 `R * E`。
- 当前 rank 为 `r`，本地专家编号为 `e in [0, E)`。
- `sendCounts[dstRank * E + dstLocalExpert]` 表示本 rank 发送给目标全局专家的 token 数。
- 每个 rank 的 `gmmX` 已按目标全局专家编号升序排列。

对于本 rank 的专家 `e`，从源 rank `s` 接收的 token 数由源 rank `s` 的 peer counts 决定：

```text
count(s, r, e) = peerSendCounts_s[r * E + e]
```

源 rank 内的输入偏移：

```text
srcOffset(s, r, e) = sum(peerSendCounts_s[0 : r * E + e])
```

本 rank 当前专家内的目标偏移：

```text
dstOffset(r, e, s) = expertBase(r, e)
                   + sum(count(p, r, e), p in [0, s))
```

当前专家的 GEMM M 轴：

```text
M_e = sum(count(s, r, e), s in [0, R))
```

接收后的排列固定为：

```text
local expert 0: source rank 0, source rank 1, ... source rank R-1
local expert 1: source rank 0, source rank 1, ... source rank R-1
...
```

`permuteOut` 与 CATLASS 输入 A 使用相同排列；`gmmY` 的 token 行顺序与 `permuteOut` 一一对应。

## 4. 总体架构

V2 新增独立 AIV 路径，不复用下列旧组件：

- `HcclA2avOp`
- `A2avGmmScheduler`
- `QuantGroupedMatmul`
- `expertNum` 多专家合并调度

建议新增或调整的核心文件：

```text
mc2/allto_allv_grouped_mat_mul/
├── op_kernel/
│   ├── allto_allv_grouped_mat_mul.cpp
│   ├── allto_allv_grouped_mat_mul_aiv_mode.h
│   ├── allto_allv_grouped_mat_mul_aiv_comm.h
│   ├── allto_allv_grouped_mat_mul_catlass.h
│   ├── allto_allv_grouped_mat_mul_tiling.h
│   └── allto_allv_grouped_mat_mul_tiling_key.h
├── op_host/op_tiling/
│   └── arch22/allto_allv_grouped_mat_mul_aiv_tiling.*
└── op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp
```

入口 kernel 仍使用混合核：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

Host 使用 batch schedule mode，并按 910C 的 AIC/AIV 数量计算混合 block dim。

## 5. HCCL Window 内存布局

每个 rank 的本地 HCCL window：

```text
offset 0
├── inputStaging
│   └── M_local * K * sizeof(bfloat16_t)
├── sendCountsStaging
│   └── R * E * sizeof(int32_t)
└── controlArea
    └── 16 MB，存放跨 rank 同步 flag、epoch 和保留区
```

最低通信缓冲区要求：

```text
M_local * K * sizeof(BF16)
+ R * E * sizeof(int32_t)
+ 16 MB
```

初始化方式参考：

- `mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode.h`
- `mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode.h`

Kernel 从 `GetHcclContext<HCCL_GROUP_ID_0>()` 获取本地 rank、rank size、本地 window 和 peer window。不得调用 AlltoAllV 高级接口。

## 6. 手工通信算法

### 6.1 初始化发布

1. 所有 AIV 清理本次调用使用的本地控制 flag。
2. AIV 将本 rank 完整 `gmmX` 拷入 `inputStaging`。
3. AIV0 将 `sendCounts` 拷入 `sendCountsStaging`。
4. 本 rank AIV 同步。
5. 通过 peer control area 执行全卡同步，确保所有 rank 的 staging 数据可见。
6. AIV0 读取各 peer counts，计算专家 M、专家基址和源/目标偏移表，写入 user workspace metadata。

### 6.2 逐专家收集

```cpp
for (uint32_t expertIdx = 0; expertIdx < expertPerRank; ++expertIdx) {
    // 每个 AIV 负责一个或多个 source rank。
    for (uint32_t srcRank = aivId; srcRank < rankSize; srcRank += aivCoreNum) {
        uint32_t tokenCount = GetPeerCount(srcRank, localRank, expertIdx);
        if (tokenCount == 0) {
            continue;
        }

        uint64_t srcTokenOffset = GetPeerTokenOffset(srcRank, localRank, expertIdx);
        uint64_t dstTokenOffset = GetLocalExpertPeerOffset(expertIdx, srcRank);

        CopyPeerGmToLocalGmPingPong(
            peerWindow[srcRank] + srcTokenOffset * K,
            recvTokenBuffer + dstTokenOffset * K,
            tokenCount * K,
            ubMoveNum);

        if (permuteOutFlag) {
            CopyOrMirrorToPermuteOut(dstTokenOffset, tokenCount);
        }
    }

    SyncAivCores();
    PublishExpertReady(expertIdx, expertM[expertIdx]);
}
```

AIV 核间任务按源 rank 粒度下发；当 `rankSize > aivCoreNum` 时，一个 AIV 以步长 `aivCoreNum` 处理多个源 rank。每段 token 内部使用 GM→UB→GM ping-pong 搬运，尾块采用带长度参数的安全拷贝。

### 6.3 同步协议

- 跨 rank：复用参考算子的 peer flag + epoch 机制，至少包含 publish-ready 和 final-release 两个阶段。
- AIV→AIC：每个本地专家一个 GM ready state，值为本次调用 epoch；AIC 不使用单一复用 flag，避免 ABA。
- AIV 核间：初始化、metadata 完成以及每个专家发布前均进行必要同步。
- 零 token 专家仍发布 ready state，AIC 读到 `M_e=0` 后跳过 GEMM。
- Kernel 结束前执行全卡 final-release，保证任何 rank 不会在 peer 仍读 window 时提前复用缓冲区。

## 7. 通信计算掩盖

workspace 中每个专家的接收区间互不重叠，因此：

```text
time ───────────────────────────────────────────────▶
AIV: publish | gather E0 | gather E1 | gather E2 | ...
AIC:          shared MM | GEMM E0   | GEMM E1   | ...
```

- AIV 发布输入时，AIC 可执行可选共享专家 MM。
- AIV 发布专家 `e` 的 ready state 后立即处理专家 `e+1`。
- AIC 等待专家 `e` ready 后按 CATLASS tile 调度计算。
- AIV 不覆盖已发布区间，因此不需要等待专家 `e` 的 GEMM 完成。
- 首版没有 chunk 循环或多专家合并。

## 8. CATLASS 计算设计

### 8.1 主专家 GEMM

每个本地专家独立构造 CATLASS problem：

```text
A = recvTokenBuffer[expertBase : expertBase + M_e, K]
B = gmmWeight[expertIdx]
C = gmmY[expertBase : expertBase + M_e, N]
```

类型与布局：

- Arch：`Arch::AtlasA2`。
- ElementA/B/C：BF16。
- Accumulator：FP32。
- A/C：ND row-major。
- B：由编译期 `transGmmWeight` 选择 layout。
- `M=M_e` 为每专家运行时值，K/N 固定。
- 不使用 Split-K；保持确定性。

CATLASS 配置参考 `all_gather_matmul_v2` 和 `matmul_reduce_scatter_v2`，使用显式 `GemmCoord`、Layout、BlockMmad 和 BlockScheduler。

### 8.2 AIC 调度

```cpp
for (uint32_t expertIdx = 0; expertIdx < expertPerRank; ++expertIdx) {
    WaitExpertReady(expertIdx);
    uint32_t m = expertM[expertIdx];
    if (m == 0) {
        continue;
    }

    CatlassProblem problem{m, N, K};
    CatlassBlockScheduler scheduler(problem, m0, n0, swizzle...);
    for (auto tile : scheduler.GetCoreLoops(aicId, aicCoreNum)) {
        RunCatlassTile(tile, A, B, C);
    }
}
```

各 AIC 最多相差一个 CATLASS tile；尾 M/N 由 CATLASS actual shape 处理。

### 8.3 权重转置

CATLASS 的 B layout 是编译期属性。需要为：

```text
(transGmmWeight, transMmWeight) = FF / FT / TF / TT
```

编译四个 tiling-key 组合，kernel 内不得用运行时分支改变 B layout。

### 8.4 共享专家 MM

当 `mmX/mmWeight/mmY` 同时存在：

```text
mmY = mmX @ mmWeight
```

共享专家使用独立 CATLASS problem 和 tiling 参数；AIC 在主通信初始化阶段优先执行，使其与 AIV publish/gather E0 重叠。未提供共享专家输入时不初始化或执行该 problem。

## 9. Tiling Key 与 Tiling Data

Tiling key：

```cpp
#define TILINGKEY_TPL_AIV 0

ASCENDC_TPL_ARGS_DECL(
    AlltoAllvGroupedMatMul,
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_GMM_WEIGHT_TRANSPOSE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_MM_WEIGHT_TRANSPOSE, 0, 1),
    ASCENDC_TPL_UINT_DECL(
        TILINGKEY_COMM_MODE,
        ASCENDC_TPL_1_BW,
        ASCENDC_TPL_UI_LIST,
        TILINGKEY_TPL_AIV));
```

核心数据结构：

```cpp
struct AlltoAllvGmmInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t rankSize;
    uint32_t expertPerRank;
    uint32_t isTransposeB;
    uint32_t maxOutputSize;
    uint32_t hasSharedExpert;
    uint32_t hasPermuteOut;
    uint32_t aivCoreNum;
    uint32_t aicCoreNum;
};

struct CoCTiling {
    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t ubMoveNum;
    int32_t swizzleCount;
    int32_t swizzleDirect;
};

struct AlltoAllvGmmAivTilingData {
    HcclInitTiling hcclInitTiling;
    HcclContextTiling hcclContextTiling;
    AlltoAllvGmmInfo gmmInfo;
    AlltoAllvGmmInfo mmInfo;
    CoCTiling gmmCocTiling;
    CoCTiling mmCocTiling;
    int64_t sendCounts[MAX_GLOBAL_EXPERTS];
    int64_t recvCounts[MAX_GLOBAL_EXPERTS];
    uint64_t recvTokenOffset;
    uint64_t countTableOffset;
    uint64_t expertMetaOffset;
    uint64_t expertReadyOffset;
};
```

字段最终类型需以 tiling data 总大小、现有 `MAX_EXPERT_NUM` 常量及 HCCL context ABI 为准；所有结构按 8 字节 pack，并为 GM scalar load 保持 4 字节对齐。

## 10. Host Tiling

Host tiling 步骤：

1. 校验平台、dtype、format、shape、counts、可选输入一致性。
2. 从平台获取 AIC/AIV 数、UB、L1/L0 和 system workspace。
3. 固定 `commMode=TILINGKEY_TPL_AIV`，计算转置组合 tiling key。
4. 设置 batch schedule mode。
5. 计算混合 block dim。
6. 选择 BF16 CATLASS `m0/k0/n0` 与 swizzle；首版沿用参考算子的候选表和合法性校验，不调用 MatMul 高级 tiling。
7. 计算 `ubMoveNum`，满足 32B 对齐与 UB 上限。
8. 计算 workspace 分区偏移并序列化 counts。
9. 通过与参考 AIV 通信算子相同的 `Mc2CcTilingConfig` 初始化 910C full-mesh window/context 资源。
10. 设置 workspace size、block dim 和 tiling key。

## 11. UB 分配

通信搬运阶段使用 BF16 ping-pong：

| Buffer | 单份大小 | 份数 | 用途 |
|---|---:|---:|---|
| `inQueue` | `AlignUp(ubMoveNum * 2, 32)` | 2 | peer GM→UB |
| `outQueue` | 与 `inQueue` 复用或按 API 要求单独分配 | 0/2 | UB→local GM |
| `cumsumBuf` | `AlignUp(E * 4, 32)` | 1 | counts 前缀和 |

首选使用单个 ping-pong queue 完成 GM→UB→GM，以免重复占用 UB；若参考 API 的 queue ownership 要求输入输出分离，则实际约束为：

```text
4 * AlignUp(ubMoveNum * sizeof(BF16), 32)
+ AlignUp(E * sizeof(int32_t), 32)
<= availableUbSize
```

若可复用单 queue，则系数降为 2。Host 按最终 API 形态计算，不写死 192KB。CATLASS 的 L1/L0 空间由模板配置独立管理，BF16 由 Cube 内部 FP32 累加，不在 AIV UB 中分配 FP32 转换 buffer。

## 12. Workspace

User workspace：

```text
recvTokenBytes = AlignUp(maxOutputSize * K * sizeof(BF16), 512)
countTableBytes = AlignUp(R * E * sizeof(int32_t/uint64_t), 512)
expertMetaBytes = AlignUp(E * sizeof(ExpertMeta), 512)
readyBytes = AlignUp(E * sizeof(uint32_t), 512)
catlassAuxBytes = implementation-dependent aligned auxiliary bytes

userWorkspace = recvTokenBytes
              + countTableBytes
              + expertMetaBytes
              + readyBytes
              + catlassAuxBytes
```

总 workspace：

```text
workspaceSize = systemWorkspaceSize + userWorkspace
```

`permuteOutFlag=true` 时仍保留 recv workspace，以保持 CATLASS 输入地址稳定；AIV 在接收时同步镜像到 `permuteOut`。实现阶段若验证可直接将 `permuteOut` 作为 CATLASS A 且不会改变 ABI，可作为后续性能优化，不属于首版正确性路径。

## 13. 参数校验

- 平台必须为 Ascend 910C。
- `commMode == "aiv"`。
- `gmmX/gmmWeight/gmmY` 为 BF16、ND、非空、连续。
- `gmmWeight` 为三维 `[E,K,N]` 或转置物理布局 `[E,N,K]`。
- `gmmY.shape == [sum(recvCounts), N]`。
- `sendCountsTensorOptional == nullptr` 且 `recvCountsTensorOptional == nullptr`。
- `sendCounts/recvCounts` 非空，长度均为 `R * E`。
- `sum(sendCounts) == gmmX.shape[0]`。
- `sum(recvCounts) <= maxOutputSize`，并与 gmmY 第一维一致。
- 全局专家数 `R*E` 在 `[1,1024]` 且满足接口约束。
- `epWorldSize` 属于设计支持集合，且所有 rank 属性一致。
- `mmX/mmWeight/mmY` 同时存在或同时为空；dtype/format/shape 匹配。
- `permuteOutFlag` 与 `permuteOutOptional` 是否存在一致。
- 单专家零 token 或单 peer 零 token合法，但 `gmmX` 空 Tensor 非法。
- 检查 HCCL window 大小满足第 5 节公式。

## 14. 错误处理与安全性

- 所有 token、元素和字节偏移使用 64 位无符号计算，乘法前进行溢出校验。
- 任一 counts 为负立即报错。
- 每段 peer 读区间必须落在该 peer `inputStaging` 内。
- 每段 local 写区间必须落在 `recvTokenBuffer`、`permuteOut` 和 `gmmY` 的合法范围。
- GM/UB 搬运长度按 32B 对齐，但尾块只写有效元素，禁止越界 padding 写。
- 所有 rank 必须执行相同专家轮数和同步阶段；零 token 分支不能跳过 barrier/flag。
- 异常由 host 尽量提前拦截；kernel 不引入会造成部分 rank 提前退出的条件分支。

## 15. 测试设计

### 15.1 Host/API UT

- `aiv` 接受，`ai_cpu`/`ccu`/空字符串拒绝。
- FP16/FP32/INT8 拒绝，BF16 接受。
- 错误 shape、format、counts 长度、counts 和、rank size 拒绝。
- 共享专家三输入输出一致性。
- `permuteOutFlag` 一致性。
- 四种转置 tiling key。
- workspace 和通信缓冲区边界。

### 15.2 mc2_test CPU 标杆

新增 V2 调用入口。标杆步骤：

```python
ata = all_to_all_v(gmm_x, send_counts, recv_counts)
permute = concat_by_local_expert_then_source_rank(ata)
for e in range(expert_per_rank):
    golden_y[e] = permute[e] @ weight[e]
if has_shared_expert:
    golden_mm_y = mm_x @ mm_weight
```

### 15.3 多卡精度用例

- EP=2/4/8。
- 均匀 counts。
- 高度倾斜 counts。
- 某源 rank 对本 rank 全零。
- 某本地专家 `M_e=0`。
- 所有 token 集中到单一专家。
- M/K/N 小 shape。
- K/N 与 CATLASS tile 非整除尾块。
- `transGmmWeight=false/true`。
- `permuteOutFlag=false/true`。
- 无共享专家/有共享专家及 `transMmWeight=false/true`。

精度采用现有 MC2 BF16 双标杆/L1 规则，分别校验 `gmmY`、`permuteOut` 和 `mmY`。

### 15.4 性能与流水验证

- profiler 确认无 AlltoAllV 高级 HCCL kernel。
- profiler 确认主计算为 CATLASS/Cube 路径。
- 打点确认 AIV gather `e+1` 与 AIC GEMM `e` 有重叠。
- 对比当前小算子组合和旧 V2。
- 首版不以 chunk 策略结果作为验收项。

## 16. 实施顺序

1. 新增设计对应的 testcase 文档和 V2 最小 BF16 用例。
2. 调整 V2 op_api，仅接受 AIV，设置 MTE server；保持 V1 不变。
3. 新增 arch22 AIV tiling 和固定 AIV tiling key。
4. 新增 HCCL window 初始化、publish、peer count 读取和同步 helper。
5. 实现逐专家 peer GM→UB→local GM 搬运。
6. 接入主 GMM CATLASS 和四种转置模板。
7. 接入共享专家 CATLASS，并与通信初始化重叠。
8. 完成 UT、编译、安装和 mc2_test 2/4/8 卡验证。
9. 完成精度与 profiler 性能报告。

## 17. 完成定义

- V1 现有路径不回归。
- V2 仅接受 AIV，BF16 规格通过。
- 代码中不调用 AlltoAllV 高级通信接口。
- 代码中不调用 MatMul/GroupedMatMul 高级计算 API。
- 调度循环严格为一次一个本地专家，无 `expertNum` 批量和 chunk。
- `gmmY`、`permuteOut`、`mmY` 在覆盖用例中精度通过。
- 2/4/8 卡无死锁，零 token 专家无死锁。
- 编译、API/host UT 和多卡测试均有可复现命令与结果记录。
- profiler 能证明预期通信计算重叠，并提供与标杆的性能对比。

## 18. 参考

- 内存语义文档：`MC2_doc/ops/alltoallvgroupedmatmul/a2avgmm内存语义文档.md`
- `mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode.h`
- `mc2/all_gather_matmul_v2/op_kernel/gather_matmul.hpp`
- `mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode.h`
- `mc2/matmul_reduce_scatter_v2/op_kernel/matmul.hpp`
- 六月优化提交：`090dfac1e`，本设计明确不沿用其多专家合并调度。
