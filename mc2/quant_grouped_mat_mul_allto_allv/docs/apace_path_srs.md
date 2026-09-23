# QuantGroupedMatMulAlltoAllv apace 通路 SRS 需求详设

| 项 | 内容 |
|---|---|
| 文档版本 | V1.1 |
| 基线代码 | apace 分支（AIC_ONLY 实现提交） |
| 适用芯片 | Ascend 950（dav_c310，`__NPU_ARCH__=3510`） |
| 关联参考 | 镜像算子 PR 11573（allto_allv_quant_grouped_mat_mul，apace AIC-only 形态）；原通路 `GmmA2avScheduler + HcclA2avOp + QuantGroupedMatmul`（mc2_templates） |

> **V1.1 变更记录**：apace 通路任务类型由 MIX_AIC_1_2 切换为 **AIC_ONLY**（第 4 章由规划转为实现，
> 第 3 章 MIX 场景转为历史参考）。切换依据：镜像算子 PR 11573 为 AIC-only 形态（唯一验证先例），
> 其 tiling 配置（SetScheduleMode(1)+SetBlockDim(aicCoreNum)）随之采纳；OB-1 前置风险保留在 4.2 节，
> 启用前建议以 debugStage=7 验证任务层可用性。诊断 stage 语义微调：stage 4 为纯 GMM 隔离
> （不含 SoftBarrier），stage 5 为仅 SoftBarrier。

---

## 1 引言

### 1.1 目的

定义 `QuantGroupedMatMulAlltoAllv` 算子 **apace hcomm 融合通路**的软件需求规格：
- **MIX 场景**（当前实现，`KERNEL_TYPE_MIX_AIC_1_2`）：AIC 计算 + AIV 通信的双核协同形态；
- **AIC_ONLY 场景**（规划实现，`KERNEL_TYPE_AIC_ONLY`）：全 AIC 单侧形态，含已知阻塞项与前置条件。

### 1.2 范围

apace 通路是原通路的**可选加速路径**（默认关闭），复用原算子的 aclnn 接口、输入校验与 infershape，不改变算子的数学语义与输入输出格式。原通路（TT/MX、CCU/AICPU）行为不受任何影响。

### 1.3 术语

| 术语 | 定义 |
|---|---|
| EP | Expert Parallelism，专家并行域，大小 `epWorldSize` |
| e | 单卡部署专家数（gmmWeight.dim0），全局 group 数 = e × epWorldSize ≤ 256 |
| sendCnt / recvCnt | AllToAllV 计数，按 `[expertId + rank × e]` 一维展开 |
| SPARSE groupList | `[专家id, token数]` int64 对，专家主序 `(expertIdx × ep + rankIdx)` |
| rank-first 布局 | 按 rank 分段连续：`[rank0 全部专家行 | rank1 ... | ...]` |
| flag8 / flag9 | 跨核同步 flag：8=mode-0 全 AIC barrier，9=mode-2 AIC→AIV 通知 |
| workingFlag_ | Hccl 客户端内部判定 `GetBlockIdx()==0`，仅工作核写 CCU 参数/敲 doorbell |
| 通算掩盖 | 第 b+1 批 GMM 计算与第 b 批 AllToAllV 通信在时间上重叠 |

---

## 2 总体描述

### 2.1 执行模型（MIX 场景）

```
stream: [MC2 通信域辅助 task] → [apace kernel (MIX 1:2)] → [event]
kernel 内:
  AIC×N: Init(Hccl) → for expert{ Blaze GMM } → for batch{ PipeBarrier
          → flag8 barrier → flag9 通知 } → 共享MM → SyncAll<false>
  AIV×2N: Init(Hccl) → for batch{ WaitFlag(9) → block0: AllToAllV<true> }
          → block0: 逐 handle Wait → SyncAll<false> → 全 AIV Finalize
```

### 2.2 外部接口

与原算子完全一致（aclnn 一段/二段/三段），新增**三层选择开关**（全部满足才走 apace，任一不满足静默回退原通路）：

| 层 | 开关 | 默认 | 说明 |
|---|---|---|---|
| 编译期-构建 | `ENABLE_APACE_QUANT_GMM_A2AV=1`（环境变量，经 op_host/CMakeLists 同步下发 kernel 宏与 tiling 宏） | 关 | 关闭时 kernel apace 分支为兜底空实现、tiling IsCapable 恒 false |
| 编译期-dtype | `APACE_GMM_A2AV_MX_DTYPE_OK()` 宏（随 dtype 二进制实例化） | — | FP8 家族内 X/W ∈ {E4M3FN, E5M2} 任意组合，或均为 E2M1；非法组合实例为空实现 |
| 运行期-tiling | `MC2_QUANT_GMM_A2AV_APACE=1`（各 rank 均须设置） | 关 | tiling 侧 IsCapable 判定 |

### 2.3 设计约束（真机验证结论）

| # | 约束 | 依据 |
|---|---|---|
| C1 | 内核任务类型必须 `MIX_AIC_1_2`，**不可 AIC_ONLY**（MIX 场景下） | MC2 通信域在 stream 上部署的辅助通信 task 依赖 AIV；AIC_ONLY 实测 kernel 仅 Init+Finalize 也卡死（与 kernel 内容无关） |
| C2 | `SetBlockDim(aicCoreNum)` 全核启动 | mode-0 flag8 计数域为调度域内全部 AIC 核，部分核启动时 barrier 永不释放 |
| C3 | 不设置 `SetScheduleMode`（保持默认 0） | 原通路默认模式；独占模式与 MC2 通信 task 的调度交互未知 |
| C4 | Blaze GMM 单 expert 粒度调用、`scaleKAL1 = kAL1 = stepK×baseK`、`l1BufferStage = 2` | 镜像算子验证形态；批粒度+SWAT scaleKL1 组合实测挂起 |
| C5 | `Finalize` 必须有提交（`curHandleId_ > 0`）才可调用 | Hccl::Finalize 零提交时 `handleId = -1` 越界且 Wait 死循环 |
| C6 | 通信参数 rank 维度取 `hccl_.GetRankDim()`（通信域真实值） | 与原通路 HcclA2avOp 对齐；tiling 的 epWorldSize 与通信域不一致时 AllToAllV 参数失配会跨 rank 死等 |
| C7 | tiling `SetDataSize` 取 registeredSize（`QuantGmmA2avTilingData` 大小）并清零 padding | 框架按 REGISTER_TILING_DEFAULT 结构大小处理 tiling 数据 |

---

## 3 MIX 场景需求详设

### 3.1 Tiling 需求

#### 3.1.1 tiling 类注册与选择

- 新增 `ApaceMxQuantGroupedMatmulAllToAllvTiling`，**priority=0**（先于原 MX 类 priority=1、TT 类 priority=2 尝试）；`IsCapable` 不满足返回 `GRAPH_PARAM_INVALID` 自动回退。
- 继承原 MX tiling 类，**复用全量输入校验链**（dtype/shape/attr/sendRecvCounts 校验，计数写入 `taskTilingInfo`）。

#### 3.1.2 IsCapable 判定（全满足才选中 apace）

| # | 条件 | 失败处理 |
|---|---|---|
| 1 | 构建宏 `ENABLE_APACE_QUANT_GMM_A2AV` 已定义 | 编译期恒 false |
| 2 | 环境变量 `MC2_QUANT_GMM_A2AV_APACE=1` | 回退原通路 |
| 3 | 量化模式为 MX（gmmX/gmmWeight 量化模式均为 6） | 回退 |
| 4 | dtype 同位宽：FP8×FP8（E4M3FN/E5M2 任意组合）或 FP4×FP4（均 E2M1） | 回退（Blaze kernel 与原算子均支持 FP8 内混 dtype） |
| 5 | `comm_mode` 属性 == "ccu" | 回退（apace 当前仅支持 CCU 引擎） |
| 6 | gmmX 非空（A > 0）且 `H1 % 64 == 0` | 回退（MX scale 按 64 分组对齐 + SWAT 引擎约束） |

#### 3.1.3 TilingKey 编码

`GET_TPL_TILING_KEY(hasSharedMm, gmmTrans, mmTrans, COMM_MODE=APACE(2), USING_APACE_IMPL=true)`

- TPL DECL 5 参数：`COMPUTE_MATMUL(bool,1b) | GMM_WEIGHT_TRANS(bool,1b) | SHARED_MM_WEIGHT_TRANS(bool,1b) | COMM_MODE(uint,2b,{0,1,2}) | USING_APACE_IMPL(bool,1b)`
- 每个 SEL 组合显式声明 `ASCENDC_TPL_KERNEL_TYPE_SEL(MIX_AIC_1_2)`（16 组全声明；**必须直接内联为第一个参数，自定义宏包裹会导致 opc 不识别**）
- 原 CCU/AICPU 编码值不变（COMM_MODE 扩 2bit 后索引 0/1 不变）

#### 3.1.4 tiling 数据结构 `hcommQuantGmmAlltoAllvTilingData`（apace 自有）

| 字段组 | 内容 |
|---|---|
| HCCL | `Mc2InitTiling mc2InitTiling`（64B）+ `Mc2CcTiling mc2CcTiling`（512B），由 `Mc2CcTilingConfig(group, cmd=8, "AllToAll=level0:fullmesh;level1:pairwise", yDtype 映射, ENGINE_CCU_SCHED)` 生成 |
| Matmul | `QuantMatmulTilingData gmmTile`（SWAT：m=maxM, n=N1, k=H1）+ `sharedMmTile`（m=BS, n=N2, k=H2） |
| 维度 | h1/n1/h2/n2/bs/aTotal/bsk/maxM（uint64×8） |
| ws 偏移 | wsGroupListOffset / wsCTableOffset / wsSharedGroupListOffset |
| 计数 | `int32 sendCnt[256] / recvCnt[256]`（`[expertId + rank×e]` 展开） |
| EP 参数 | e / epWorldSize / hasSharedMm |
| 软 barrier | `volatile uint32 barrierSlots[64]`（host 清零保初值，AIC_ONLY 场景使用） |
| 诊断 | `uint32 debugStage` |

#### 3.1.5 tiling 数据下发要求

- `PostTiling`：`SetDataSize` 取 **max(registeredSize, apaceSize)**（对齐 C7），apace 结构之后 padding 清零；`SetBlockDim(aicCoreNum)`（C2）；不设 ScheduleMode（C3）。
- `GetWorkspaceSize`：`workspaces[0] = libApiWorkSpaceSize + apaceWorkSpaceSize`。

#### 3.1.6 user workspace 布局（tiling 下发偏移，512B 对齐）

```
[0,                       wsGroupListOffset)      sendBuf：A×N1×sizeof(CType)  GMM 输出/通信发送
[wsGroupListOffset,       wsCTableOffset)         groupList：e×ep×2 × int64（SPARSE）
[wsCTableOffset,          wsSharedGroupListOffset) cGroupOffsetTable：e×ep × uint64
[wsSharedGroupListOffset, +8)                     sharedGroupList：int64[BS]
```

#### 3.1.7 GMM tiling 引擎与映射

- 引擎：`QuantMatmulTilingSwat<dtypeA, dtypeB, DT_FLOAT>`（按实际 dtype 对实例化 5 组合），配置 `SetOptimizeEnable(false) / SetMTailAlignEnable(true) / SetAdjustBasicBlockEnable(false) / SetBiasInfo(false)`。
- 映射到 `GroupedMatmulMxKernel::GMMTiling`（**对齐镜像，C4**）：

| GMMTiling 字段 | 取值 |
|---|---|
| baseM / baseN / baseK | SWAT baseM / baseN / baseK |
| kAL1 = kBL1 = scaleKAL1 | `stepK × baseK` |
| dbL0C | SWAT dbL0c |
| l1BufferStage | 固定 `2` |
| groupListType | 主 GMM = SPARSE(2)；共享专家 MM = LENGTH(1) |
| groupNum | 主 GMM 每次 = `ep`（单 expert）；共享 MM = 1 |
| m | 当前 expert 全 rank token 数（共享 MM = BS） |

### 3.2 Kernel 需求

#### 3.2.1 入口结构

```
quant_grouped_mat_mul_allto_allv<..., COMM_MODE, USING_APACE_IMPL>:
  if constexpr (USING_APACE_IMPL) {
      y[0] = 0xA5A5A5A5（部署自检 magic）
      #if 构建开 && dtype OK:
          GET_TILING_DATA_WITH_STRUCT(apace struct)
          impl.Init(...); impl.Run(); return
      #else: return（兜底空实现）
  }
  KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_2)   ← 仅原通路路径设置（apace 由 KERNEL_TYPE_SEL 声明）
  ...原通路...
```

#### 3.2.2 Init（全部核执行）

1. 基址解析与 workspace 偏移绑定（sendBuf / groupList / cTable / sharedGroupList / barrierSlots 取 tilingGM 原址）。
2. `hccl_.InitV2(GetHcclContext<0>(), &mc2InitTiling)`；`hccl_.SetCcTilingV2(offsetof(TilingData, mc2CcTiling))`；`commRankNum_ = hccl_.GetRankDim()`（C6，0 时兜底 ep）。
3. AIC 侧 `BuildGmmTables()`：各 AIC 写相同值后各自消费——
   - groupList（SPARSE）：`groupList[2g]=expertId, groupList[2g+1]=tokenCnt`，g 按 `(expertIdx×ep + rankIdx)` 专家主序；
   - cTable（全局 rank-first）：`cTable[g] = rankBase[r] + 批内前序专家累计`（CType 元素偏移）。

#### 3.2.3 AIC 侧 Run（计算 + 通知）

```
expertTokenOffset = 0
for 批 b（expertNumPerBatch = CeilDiv(e,4)，批数 ≤ 4）:
    for expert eIdx in 批:
        expertTokens = Σ_r sendCnt[eIdx + r*e]
        if expertTokens == 0: continue          // 空 expert 跳过，偏移不推进
        RunGmmKernel<TransGmmB>(gmmTile, groupNum=ep, m=expertTokens,
            A基址 = gmmxGM + 打包偏移(expertTokenOffset×H1，FP4 右移1),
            scaleA基址 = gmmxScaleGM + expertTokenOffset×scaleKLen,
            B/scaleB = 全局基址（sparse 专家 id 寻址）,
            C = sendBufGM（cTable 切片 eIdx×ep）)
        PipeBarrier<PIPE_ALL>()                  // 排空本核 fixpipe
        expertTokenOffset += expertTokens
    flag8: CrossCoreSetFlag<0,PIPE_FIX>(8); CrossCoreWaitFlag(8)   // 全 AIC barrier
    flag9: CrossCoreSetFlag<0x2,PIPE_FIX>(9)                       // 通知本 block AIV
共享专家 MM（可选，groupNum=1，与 AIV 末批通信重叠）
SyncAll<false>()
```

**Blaze GMM 单 expert 调用契约**（C4）：
- groupList 切片 `eIdx×ep×2` 起、长 `ep` 对；cTable 切片 `eIdx×ep` 起、长 `ep`；
- A/scaleA 走 kernel 默认累计 offset（批内基址已推进至该 expert 起点）；B/scaleB 由 SPARSE 专家 id 定位（无表）；C 由 cTable 写 rank-first。

#### 3.2.4 AIV 侧 Run（通信）

```
for 批 b:
    CrossCoreWaitFlag(9)                        // 等本批 sendBuf 落 GM
    if GetBlockIdx()==0: a2avHandles[b] = SubmitAllToAllvBatch(b)
if GetBlockIdx()==0: for h in handles: hccl_.Wait(h)
SyncAll<false>()
if submitted>0: hccl_.Finalize()                // 全 AIV 调用，workingFlag_ 门控生效（C5）
```

#### 3.2.5 同步契约（与原通路逐项一致）

| 原语 | 发起方 | 消费方 | 语义 |
|---|---|---|---|
| flag8 mode-0 `PIPE_FIX` | 全部 AIC（每批 Set+Wait） | 全部 AIC | 全 AIC barrier：闭合本批 sendBuf 多核写依赖（fixpipe 落 GM） |
| flag9 mode-2 `PIPE_FIX` | 每个 AIC（每批 Set 一次） | 本 block 全部 AIV（每批 Wait 一次） | AIC→AIV 通知：本批数据可通信 |
| `SyncAll<false>` | AIC 尾部 1 次 + AIV 尾部 1 次 | 互为配对 | kernel 收尾汇聚 |

#### 3.2.6 通信参数（SubmitAllToAllvBatch）

- rank 维 `rankNum = hccl_.GetRankDim()`（C6）；
- 元素计数（`×N1`）：
  - `sendCnt[r] = Σ_{eIdx∈批} sendCnt[eIdx + r×e] × N1`
  - `recvCnt[r] = Σ_{eIdx∈批} recvCnt[eIdx + r×e] × N1`
- 元素偏移（全局 rank-first sendBuf / rank-major y）：
  - `sendOff[r] = (rankBaseS[r] + Σ_{eIdx<批起点} sendCnt[eIdx + r×e]) × N1`
  - `recvOff[r] = (rankBaseR[r] + Σ_{eIdx<批起点} recvCnt[eIdx + r×e]) × N1`
- 提交：`hccl_.AllToAllV<true>(sendBufGM, ..., yGM, ..., HCCL_DATA_TYPE_FP16/BFP16)`（非阻塞，per-batch handle）。

#### 3.2.7 数据布局需求

| 缓冲 | 布局 | 生产者/消费者 |
|---|---|---|
| gmmX / gmmXScale | 专家主序 `[expert0: rank0..rankN-1 tokens][expert1: ...]` | 输入（dispatch 侧保证） |
| sendBuf | **全局 rank-first** `[rank0 全部专家行][rank1 ...]` | GMM 写（cTable）/ AllToAllV 读 |
| y | **rank-major**（源 rank 聚合顺序，与原通路逐位一致） | AllToAllV 写 |
| mmY | `[BS, N2]` 直写 | 共享专家 MM 写 |

**语义一致性要求**：y 最终布局与原通路完全一致（可互为 golden 对拍）。

#### 3.2.8 dtype 支持

| gmmX | gmmWeight | 支持 |
|---|---|---|
| E4M3FN / E5M2 任意 | E4M3FN / E5M2 任意 | ✅（含交叉混合） |
| E2M1 | E2M1 | ✅ |
| FP16 / BF16 / HIFLOAT8 等非 MX | — | ❌（编译期空实例 + tiling 拦截） |

### 3.3 诊断需求

#### 3.3.1 部署/通路自检

- kernel 入口 apace 分支首核写 `y[0] = 0xA5A5A5A5`（magic）：卡死后 host 读 y 前 4 字节即可判定"新二进制在跑 + apace 分支生效"。
- tiling 侧 ERROR 级日志：选中证据 `Apace tiling选中apace通路(capable)`、key 对账 `Apace tilingKey=N`。

#### 3.3.2 分阶段诊断 `MC2_APACE_GMM_A2AV_DEBUG_STAGE`（各 rank 一致设置）

| stage | 执行内容 | 卡死指向 |
|---|---|---|
| 0（默认） | 完整流程 | — |
| 1 | 仅通信：AIC 只发 flag8/9，AIV 完整提交/等待/Finalize | Hccl/CCU 通信 |
| 2 | 计算+barrier：无任何通信 | GMM+flag 组合 |
| 3 | 计算+提交：无逐批 Wait（尾部统一 Wait） | CCU 执行/Wait 节奏 |
| 4 | 仅 GMM+PipeBarrier+barrier（AIV 直接 SyncAll 返回） | Blaze GMM |
| 5 | 仅 flag 链路（无 GMM 无 Hccl） | flag8/9+SyncAll 契约 |
| 6 | 仅 Init（hccl InitV2/建表后直接 return） | Hccl 初始化 |
| 7 | 空 kernel（不触碰 Hccl/表/GMM/barrier） | 任务启动配置层 |

---

## 4 AIC_ONLY 场景需求详设（规划）

### 4.1 设计目标

1. **消除 AIV 依赖**：全部 AIC 核执行同构流程（计算 + 通信提交），AIV 不参与任何逻辑；
2. **简化同步**：删除跨核 flag8/flag9 与 AIC/AIV 双侧 SyncAll 配对，仅保留 AIC 间同步；
3. **算力提升**：AIV 空闲核让渡（MIX 下 AIV 仅 block0 做通信、其余空转等待）。

### 4.2 已知阻塞项（真机验证，最高优先）

**OB-1：MC2 通信域辅助 task 依赖 AIV。**
真机证据链：AIC_ONLY 任务类型下，kernel 仅执行 `Init + Finalize`（无任何计算/通信）也卡死，日志（`JudgeTaskFinish is_finished=0` + `HrtRaSocketNonBlockRecv` 循环）与 kernel 内容完全无关；同构建下原通路（MIX）正常。
根因：tiling 标记 `NnopbaseSetMc2 + NnopbaseSetHcomGroup` 后，runtime 在 stream 上部署 MC2 通信域辅助 task，该 task 的执行/完成协议依赖 AIV 核存在。

**前置条件（依赖外部演进）**：
- CANN/MC2 runtime 提供通信域 task 的 AIC-only 部署模式，或
- 通信域建域方式改为不向算子 stream 插入 AIV 依赖 task（如独立通信 stream），或
- 确认目标场景（如单卡 ep=1 / 通信域 rank 全在本卡）不触发该 task。

**验收口径**：AIC_ONLY 任务类型下 stage=7（空 kernel）真机可正常完成，方允许启用本场景。

### 4.3 功能需求

#### 4.3.1 选择与实例化

- 与 MIX 场景共用全部三层开关（3.2 节）与 IsCapable 判定；
- TPL SEL 的 apace 组合声明 `ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY)`（内核直接内联，禁用宏包裹）；
- `SetBlockDim(aicCoreNum)`（AIC 核数）。

#### 4.3.2 kernel 执行流（全部 AIC 核同构）

```
Init: hccl_.InitV2/SetCcTilingV2（全核，workingFlag_ 自动门控共享态写入）
      BuildGmmTables（全核写相同值）
Run:
  for expert eIdx:                        // 单 expert 粒度，同 3.2.3
      RunGmmKernel<TransB>(groupNum=ep, m=expertTokens, ...)
      PipeBarrier<PIPE_ALL>()
  for 批 b:
      SoftBarrier(generation=b+1)          // 跨 AIC 软 barrier（4.3.3），替代 flag8
      SubmitAllToAllvBatch(b)              // 全核调用；workingFlag_(block0) 写参数/敲 doorbell
  共享专家 MM（可选）
  PipeBarrier
  for h: hccl_.Wait(h)                    // 全核调用；多核 Wait 协议
  hccl_.Finalize()                        // 全核调用；仅工作核生效
```

**与 MIX 场景差异表**：

| 维度 | MIX（现行） | AIC_ONLY（规划） |
|---|---|---|
| flag8 mode-0 全 AIC barrier | 使用 | **替换为 SoftBarrier**（mode-0 在 AIC-only 内核无先例，见 4.3.3） |
| flag9 mode-2 AIC→AIV 通知 | 使用 | 删除（无 AIV） |
| SyncAll<false> AIC/AIV 配对 | 使用 | 删除（无跨核型配对） |
| Hccl 提交/等待 | 仅 AIV block0 | **全部 AIC 核**（镜像形态；workingFlag_ 门控写侧，多核 Wait 协议：工作核轮询 CKE 推进 finishCntGM_，非工作核跟随） |
| Finalize | 全 AIV | 全 AIC |
| 共享 MM 与通信重叠 | 是 | 是（批循环后执行，与末批通信重叠） |

#### 4.3.3 SoftBarrier 需求（替代 mode-0 硬 barrier）

- **机制**：GM 槽位轮询（`barrierSlots[64]`，host 清零保初值）——
  到达：核 i 写 `slots[i] = generation` + `DataCacheCleanAndInvalid<CACHELINE_OUT>`；
  轮询：Barrier + Clean 后读全部 `slots[j] >= generation`（`rac_server.h` TurnWait 生产同款模式）；
- **generation 语义**：`= 批序号 + 1` 单调递增，免清零；
- **槽位绑定**：`tilingGM + offsetof(TilingData, barrierSlots)`（GM 原址，非栈拷贝）；
- **正确性前提**：`coreNum = GetBlockNum()` = 全部启动核数（C2 全核启动为 SoftBarrier 前提）。

#### 4.3.4 通算掩盖形态

MIX 场景为"批间流水"（批 b 通信与批 b+1 GMM 重叠，由 flag9 门控）；AIC_ONLY 场景为"提交即返回"（AllToAllV<true> 非阻塞后立即进入下一 expert GMM，CCU 引擎异步执行），天然无门控开销；共享 MM 与末批通信重叠。

### 4.4 已实现存量说明

以下组件在历史版本中已实现并保留于当前代码（AIC_ONLY 启用时直接复用）：
- `barrierSlots[64]` 软 barrier 槽位（tiling data 尾部，host 清零）；
- SoftBarrier 实现（曾在 commit 5364e67d 落地，当前 MIX 版未调用）；
- 全核 Hccl 调用路径（曾在 AIC_ONLY 版验证至 stage=1 正常——通信多核协议本身自洽）。

---

## 5 验收标准

### 5.1 功能验收

| # | 项 | 标准 |
|---|---|---|
| A1 | 数值正确性 | 与原通路 golden 对拍：y 逐位一致（各 dtype 组合含 FP8 混合、含/不含共享专家、含空 expert/空批） |
| A2 | mmY 正确性 | 共享专家输出与原通路一致 |
| A3 | 回退无损 | 不设运行时开关时行为与基线（apace 合入前）完全一致；TT/非 ccu/非 MX/H1 非 64 倍数场景自动回退且无副作用 |
| A4 | 通路选择证据 | apace 生效时日志出现 capable/tilingKey 两条 ERROR；y[0]=magic 可检 |

### 5.2 诊断验收

| # | 项 | 标准 |
|---|---|---|
| D1 | stage 二分 | 各 stage 语义与 3.3.2 表一致；stage=7/6 正常完成（部署与 Init 无问题基线） |
| D2 | 卡死定位 | 卡死现场可通过 stage 序列定位到唯一环节 |

### 5.3 AIC_ONLY 场景附加验收

| # | 项 | 标准 |
|---|---|---|
| O1 | 阻塞项解除 | OB-1 前置条件满足（stage=7 空转通过） |
| O2 | SoftBarrier | 全核（≤64 AIC）多次 generation 轮询正确，无提前放行/永久等待 |
| O3 | 多核 Hccl 协议 | 全核 Wait/Finalize 协议收敛（stage=1 通过） |
| O4 | 数值 | 与 MIX 场景 A1 同标准 |

---

## 6 附录

### 6.1 环境变量清单

| 变量 | 作用域 | 用途 |
|---|---|---|
| `ENABLE_APACE_QUANT_GMM_A2AV` | 构建期（环境变量→CMake→kernel/tiling 宏） | apace 编译总开关 |
| `MC2_QUANT_GMM_A2AV_APACE` | 运行时（每个 rank 进程） | tiling 选择 apace 通路 |
| `MC2_APACE_GMM_A2AV_DEBUG_STAGE` | 运行时（每个 rank，须一致） | 分阶段诊断（0-7） |

### 6.2 故障定位快速指南

```
卡死 → y[0] != magic?        → 部署/通路未生效（查三条 ERROR 日志）
     → magic? → stage=7 卡?  → 任务配置层（KERNEL_TYPE_SEL/blockDim）
              → stage=6 卡?  → Hccl Init
              → stage=5 卡?  → flag/SyncAll 契约
              → stage=4 卡?  → Blaze GMM（对照镜像 tiling 映射：scaleKAL1/l1BufferStage/单expert粒度）
              → stage=1 卡?  → 通信参数（rankDim 来源/count-offset 语义/CCU 协议）
```

### 6.3 代码索引

| 组件 | 路径 |
|---|---|
| impl（MIX 双核模型 + 诊断） | `mc2/common/op_kernel/apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/mx_quant_grouped_matmul_all_to_all_v_hcomm_impl.h` |
| tiling 结构 | 同目录 `quant_grouped_matmul_all_to_all_v_tiling_data.h` |
| tiling 类 | `mc2/quant_grouped_mat_mul_allto_allv/op_host/op_tiling/arch35/apace_mx_quant_grouped_mat_mul_allto_allv_tiling.{h,cpp}` |
| kernel 入口 + dtype 宏 | `mc2/quant_grouped_mat_mul_allto_allv/op_kernel/quant_grouped_mat_mul_allto_allv.cpp` |
| tiling key（TPL） | `mc2/quant_grouped_mat_mul_allto_allv/op_kernel/quant_grouped_mat_mul_allto_allv_tiling_key.h` |
| Blaze GMM 引擎 | `mc2/common/op_kernel/apace/kernel/matmul/grouped_matmul/grouped_matmul_mx_kernel.h` |
| SWAT tiling 引擎 | `mc2/common/op_kernel/apace/tiling/quant_matmul_tiling_swat.h` |
| 镜像算子参考 | PR 11573（cann/ops-transformer，allto_allv_quant_grouped_matmul） |
