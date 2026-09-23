# QuantGroupedMatMulAlltoAllv apace 通路代码详细设计

| 项 | 内容 |
|---|---|
| 文档版本 | V1.0 |
| 代码基线 | apace 分支 commit `1b42ba8ba`（MIX 任务类型 + 核型守卫形态） |
| 适用芯片 | Ascend 950（dav_c310，`__NPU_ARCH__=3510`） |
| 关联文档 | `docs/apace_path_srs.md`（需求规格 V2.0） |

---

## 1 总体设计

### 1.1 目标

将 `QuantGroupedMatMulAlltoAllv` 算子 **MX(FP4/FP8) 量化场景**的路由专家 GMM 计算切换为
apace 库 `GroupedMatmulMxKernel`（Blaze 栈），并将 AllToAllV 通信改在 **AIC 核内**经
`Hccl` 提交，二者同核融合执行。TT 量化场景完全保持原通路。

### 1.2 选路机制：tiling key 选路 + AIC_ONLY 任务类型

```
构建（无任何开关，全部通路默认编译）
  └─ TPL 第 5 参数 TILINGKEY_USING_APACE_IMPL（DECL 末位，原 4 参编码值不变）
      ├─ USING=0 组合（key 0~15）  → KERNEL_TYPE_SEL(MIX_AIC_1_2) → 原通路
      └─ USING=1 组合（key 16~31） → KERNEL_TYPE_SEL(AIC_ONLY)    → apace 融合通路
运行时：MX tiling 类编码 USING=1（MX 量化恒走 apace）；TT tiling 类 4 参调用截断为 USING=0
kernel：if constexpr (TILINGKEY_USING_APACE_IMPL) 选路，任务类型全部由 SEL 按组合声明
```

判别宏（`mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/a2av_gmm_utils.h`）：

```cpp
#if defined(ORIG_DTYPE_GMM_X_SCALE) && (ORIG_DTYPE_GMM_X_SCALE == DT_FLOAT8_E8M0)
#define MX_QUANT_MODE true
#else
#define MX_QUANT_MODE false
#endif
```

要点：
- 以 **scale 的 E8M0 dtype** 判别 MX 实例（MX 块量化的 scale 即 E8M0，TT 为 float32），
  语义上无歧义；dtype 与量化模式的对应由 tiling 校验链保证（TT 类强制 float scale，
  MX 类强制 E8M0 scale），实例二进制与运行时 tiling 类选择天然一致；
- **必须带 `defined()` 守卫**：`ORIG_DTYPE_*` 宏缺失时 `#if` 求值为 0。历史方案用
  `ORIG_DTYPE_GMM_X == DT_FLOAT8_xxx` 无守卫判别，宏缺失导致 apace 分支编译为空实现，
  而 tiling 已下发 MC2 通信标记 → stream 通信任务等握手超时 → 全 stage 卡死
  （这是 tiling key 选路方案废弃的直接原因）；
- 编译期分支从结构上消除"空实现 + 通信标记"失败模式：分支未命中的实例编译为原通路。

### 1.3 总体执行流

```
                    ┌───────────────────────────────────────────────┐
                    │ kernel: quant_grouped_mat_mul_allto_allv       │
                    │ 任务类型由 tiling key KERNEL_TYPE_SEL 按组合声明  │
                    └───────────────┬───────────────────────────────┘
      if constexpr (USING=1)        │        USING=0 / TT 实例（原通路）
        ┌──────────────────────────┴───┐   ┌─────┴──────────────────────┐
        │ apace 融合通路（AIC_ONLY）     │   │ 原通路（上游逐字保持）        │
        │  AIC:                        │   │  AIC: QuantGroupedMatmul    │
        │   magic 自检(block0)          │   │  AIV: HcclA2avOp 通信        │
        │   HcommImpl.Init             │   │  调度: GmmA2avScheduler      │
        │     ├ hccl InitV2/SetCc      │   │   (flag8/flag9 AIC↔AIV)     │
        │     └ ComputeOp.Init(四表)    │   └────────────────────────────┘
        │   for 批 b: Process(b)        │
        │     ├ GMM(b) Blaze            │
        │     ├ SyncAll(写落GM)        │
        │     ├ 全核Wait(b-1)+排空     │
        │     └ 全核提交 AllToAllV(b)   │
        │   (批b通信∥批b+1GMM,引擎执行) │
        │   End: Wait末批+Finalize     │
        │   共享专家 MM(可选)            │
        └──────────────────────────────┘
```

---

## 2 代码结构

### 2.1 文件清单

**新增（apace 共享层，算子无关）**

| 文件 | 职责 |
|---|---|
| `mc2/common/op_kernel/apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_impl.h` | 计算适配层 `Apace::QuantGroupedMatmulMx`：组装 Blaze `GroupedMatmulMxKernel`，对外接口与 `MC2KernelTemplate::QuantGroupedMatmul` 一致 |
| `mc2/common/op_kernel/apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_hcomm_impl.h` | 通算融合层 `Apace::QuantGmmA2avHcommImpl`：持有 Hccl 实例，按批驱动"计算→同步→通信" |
| `mc2/common/op_kernel/apace/kernel/matmul/grouped_matmul/grouped_matmul_mx_kernel.h` | Blaze MX 分组矩阵乘引擎（`BlockMmadQgmmMx` + `BlockSchedulerGmmSwatWithTailSplit`） |

**修改（原算子目录，侵入最小化）**

| 文件 | 变更 |
|---|---|
| `op_kernel/quant_grouped_mat_mul_allto_allv.cpp` | kernel 入口：`#if MX_QUANT_MODE` 分支 + apace 调用（AIC 守卫）；`#else` 原通路逐字保持 |
| `op_host/op_tiling/arch35/mx_quant_grouped_mat_mul_allto_allv_tiling.{h,cpp}` | MX tiling 类覆写 `SetGmmA2avWorkspaceInfo()`：workspace 追加 B/scaleB 偏移表；选路证据日志 `ApaceMXPath` |
| `op_host/CMakeLists.txt` | kernel 编译增加 apace 头 include 路径（blaze/tensor_api） |

**零修改**：`quant_grouped_mat_mul_allto_allv_tiling_common.cpp`（与主库字节一致）、
tiling key（`TILINGKEY_USING_APACE_IMPL` 等 apace 选路设施已全部移除，回归上游）、
TT tiling 类、aclnn/op_graph 层。

### 2.2 模块关系与模板参数

```
kernel 入口 (#if MX_QUANT_MODE, if ASCEND_IS_AIC)
 ├─ QuantGmmA2avHcommImpl<TilingData, GmmTilingData, xType, wType, scaleType, yType,
 │                        wFormat, aTrans, bTrans, isLocal, CommMode>
 │    ├─ HcclServerTypeSelector<CommMode> → Hccl<CCU|AICPU>
 │    ├─ ComputeOp = QuantGroupedMatmulMx<..., isLocal, false>
 │    └─ A2avCommParams（send/recv cnt/offset，rank 数组）
 └─ QuantGroupedMatmulMx<TilingData, GmmTilingData, xType, wType, scaleType, yType,
                         wFormat, aTrans, bTrans, isLocal, isA2avGmm>
      └─ Blaze::Gemm::Kernel::GroupedMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>
           ├─ DispatchPolicy = GroupedMatmulWithScaleMx<0, false>
           ├─ BlockMmad = BlockMmad<policy, xType, LayoutA(nd), wType, LayoutB(nd|dn), yType, LayoutC, float, LayoutBias>
           └─ BlockScheduler = BlockSchedulerGmmSwatWithTailSplit
```

模板签名与 `MC2KernelTemplate::QuantGroupedMatmul` 完全一致（11 参数），kernel 入口
可按编译期分支直接替换实例化类型。kernel 入口实例化两处：

- 路由专家（isLocal=false）：`QuantGmmA2avHcommImpl<..., DTYPE_GMM_X_SCALE, ..., TILINGKEY_GROUPED_MATMUL_TRANS, false, TILINGKEY_COMM_MODE>`
- 共享专家（isLocal=true，kernel 入口直接实例化计算适配层）：
  `QuantGroupedMatmulMx<..., DTYPE_MM_X_SCALE, ..., TILINGKEY_MATMUL_TRANS, true, false>`

### 2.3 一次 Process 调用的内部序列（计算适配层）

```
Process(startExpertIdx, expertNum)                     [AIV 守卫提前返回]
  ├─ 空批检查：批内全部 expert token 数为 0 → return（token 偏移不推进）
  ├─ BuildGroupListAndOffsetTables(start, num)
  │    ├─ groupList[LENGTH]：批内 expert-major（槽位 = e*ep + r）存 token 数
  │    ├─ cGroupOffsetTable：批内 rank-major 的 C 输出元素偏移
  │    ├─ bGroupOffsetTable：按 group 的 W 元素偏移（= expert 序号 × perExpertW）
  │    └─ scaleBGroupOffsetTable：按 group 的 scaleB 元素偏移
  ├─ UpdateAddr(start, num)：从固定基址绝对推导本批 A/W/C/scaleA/scaleB 地址
  ├─ InitApaceKernel：GMMTiling/Params 组装 + mxKernel_.Init
  └─ mxKernel_.Run()：Blaze 全核调度执行
```

---

## 3 与原有通路的差异

### 3.1 差异总表

| 维度 | 原通路（TT 实例 / 改造前 MX 实例） | apace 通路（MX 实例） |
|---|---|---|
| 选路 | 单一二进制内运行时调度 | **tiling key 选路**（USING=1 组合；E8M0 判别仅作 TT 实例组合裁剪） |
| 任务类型 | MIX_AIC_1_2（KERNEL_TYPE_SEL 声明） | **AIC_ONLY**（KERNEL_TYPE_SEL 声明，无 AIV） |
| 核型分工 | AIC 计算 + **AIV 通信** | **AIC 计算 + AIC 通信**（AIC_ONLY 任务类型，无 AIV） |
| GMM 引擎 | `QuantGroupedMatmul` → `Mc2GmmASWKernel`（UBMEM） | `QuantGroupedMatmulMx` → Blaze `GroupedMatmulMxKernel` |
| GMM 调用粒度 | 每批一次（groupNum=批内 expert 数） | 每批一次，groupNum=**expertNum×ep**（group 按 (expert,rank) 展开） |
| group 寻址 | groupList(token 数) + **ptrTable**（GetTensorAddr 双重间接） | groupList(LENGTH) + **三张偏移表**（cTable/bTable/scaleBTable），A/scaleA 走公式 |
| 通信提交 | `HcclA2avOp`（AIV block0，mc2_templates 封装） | `Hccl` 模板直调（AIC block0，`AlltoAllV<true>` 非阻塞 + `Wait`） |
| 跨核同步 | flag8(mode-0)/flag9(mode-2) AIC↔AIV 跨核 flag | `SyncAll()`（仅 AIC 域参与；AIV 已提前退出） |
| 调度框架 | `GmmA2avScheduler.Process()` | kernel 入口平铺批循环 |
| tiling 数据 | `QuantGmmA2avTilingData` | **完全复用**（无自有结构） |
| tiling 差异 | — | 仅 MX 类 workspace 扩容（+16·e·ep 字节） |
| workspace 计算区 | groupList + cTable + ptrTable(512B) + TT scale repeat | groupList + cTable + **bTable + scaleBTable** + ptrTable/scale 占位（布局对齐） |
| MX 判别 | 无（原实现即含 MX） | `MX_QUANT_MODE`（E8M0 scale） |
| 部署自检 | 无 | `y[0]=0xA5A5A5A5` magic（block0 首写，通信成功后被覆盖） |
| host 日志 | 无 | `ApaceMXPath selected: tilingKey/e/ep/A/BSK/N1/H1/hasSharedMm`（INFO） |

### 3.2 保持不变的部分

- aclnn 三段接口、infershape、输入校验链（MX tiling 类全部继承）
- tiling 注册体系与 tiling key 编码（`GET_TPL_TILING_KEY` 4 参数原样）
- `taskTilingInfo` 全部字段语义（含 `expertNum = CeilDiv(e,4)` 批调度口径、send/recvCnt 的
  `[expertIdx + rank*e]` 索引约定）
- HCCL tiling 参数生成（基类 `SetHcclTiling` → `Mc2CcTilingConfig`，cmd=8，
  `AlltoAll=level0:fullmesh;level1:pairwise`，engine 按 comm_mode）
- 输出语义：y 的 rank-major 聚合布局、mmY 独立输出、workspace 三段划分

---

## 4 核心设计与实现

### 4.1 AIC 核内通信（为什么通信必须搬进 AIC）

Blaze（cube 模板栈）与 AIV/AICPU 侧 KFC 通信引擎在**同一 kernel 实例**中共存时，
AIV 通信任务（Mc2ServerKernel）参数校验失败（`execute kernel param invalid`，
errorCode 507018，与计算是否执行无关——只要实例化了 Blaze 模板即触发）。
因此 MX 实例的 AllToAllV 不能再走 AIV 的 `HcclA2avOp`，改为 AIC 核内 `Hccl` 直调：

- `hccl_.InitV2(GetHcclContext<0>(), &hcclA2avTiling.hcclInitTiling)` +
  `SetCcTilingV2(offsetof 链)`——与原通路 `HcclA2avOp::Init` 同源同参（复用原 tiling
  的 HCCL 段），保证通信域/CCU 资源分配行为一致；
- 通信提交与等待均为**全核调用**（`AlltoAllV<true>` 非阻塞返回 handle；多核 Wait 协议：
  Hccl 内部 workingFlag_ 门控 block0 写参数/敲 doorbell，工作核轮询推进、非工作核跟随），
  与 AIC_ONLY 生产先例（allto_all_matmul hcomm 全核提交/全核等待）一致；
- AIC_ONLY 任务类型下无 AIV 参与，无需核型守卫（见 4.6）。

### 4.2 批流水与同步协议（通算融合层）

每批（`expertNum = CeilDiv(e,4)`，与原通路同口径，总批数 ≤ 4）：

```cpp
Process(start, num):                          // 全核同构
    computeOp_.Process(start, num);           // 本批 Blaze GMM
    AscendC::SyncAll();                       // ① 本批 sendBuf 多核写落 GM
    if (submitted_ > 0) {                     // ② 全核 Wait 上一批 + PipeBarrier
        hccl_.Wait(a2avHandles_[submitted_-1]);   // （排空参数区 datamove 依赖）
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    a2avHandles_[submitted_] = CommitAlltoAllV(start, num);  // ③ 全核提交（非阻塞）
    submitted_++;
End():
    if (submitted_ > 0) hccl_.Wait(末批);      // ④ 全核等待末批
    AscendC::SyncAll();
    if (submitted_ > 0) hccl_.Finalize();     // 全核调用（workingFlag_ 门控）
```

设计说明：
- **通信不占用 AIC 核**：AllToAllV 提交后由 AICPU/CCU 引擎异步执行，AIC 核即刻进入
  下一批 GMM——批 b 通信与批 b+1 GMM 重叠推进（in-flight ≤ 2），无串行等待；
- ① 是**正确性关键**：本批 C 输出由多个 AIC 核经 fixpipe 写出，必须全核到齐
  （GMM 落 GM）后才能让通信引擎读 sendBuf；
- ② 在提交前等待上一批：通信引擎对参数数组的拷贝经数据搬移管线异步执行，标量侧
  复用参数数组（`commParams_` 成员）前必须 `Wait` + `PipeBarrier` 排空；
- `Finalize` 需有提交才调用——Hccl::Finalize 零提交时内部 handleId=-1 越界死循环
  （历史实测）；本实现每批必有提交（空批也提交零计数通信），恒满足。

### 4.3 通信参数计算（含伸缩和恒等式）

计数与偏移全部为**元素单位**（axis = N1），counts 索引约定 `[expertIdx + rank·e]`：

```
sendCnt[r]   = Σ_{eIdx ∈ 批} sendCnt[eIdx + r·e] · N1
sendOff[r]   = batchStart + 批内前序 rank 段累计
               batchStart = Σ_{批前全部 expert, 全部 rank} sendCnt · N1
recvCnt[r]   = Σ_{eIdx ∈ 批} recvCnt[eIdx + r·e] · N1
recvOff[r]   = Σ_{r'<r} TotalRecv(r') + Σ_{eIdx<start} recvCnt[eIdx + r·e]   （rank-major y）
```

`recvOff` 的实现采用**跨段索引伸缩和**（避免二重前缀循环）：

```cpp
for (i = 1; i < rankDim; i++)
    recvOff[i] = recvOff[i-1] + Σ_{j<e} recvCnt[start + (i-1)·e + j] · N1;
```

索引 `start + (i-1)·e + j` 会越过 rank i-1 段尾进入 rank i 段首，其和恰为
`TotalRecv(i-1) - prefix(i-1) + prefix(i)`，逐级累加后伸缩为正确的
`Σ TotalRecv + prefix(i)`（数学恒等，非近似）。send 侧偏移与 cGroupOffsetTable
的批内 rank-major 布局严格互逆。

### 4.4 计算适配层：四表与基址推进

**为什么需要 B/scaleB 偏移表**：group 按 `(expertIdx, rankIdx)` 展开后 groupIdx =
`e·ep + r` 与 expertIdx 不同序，Blaze 引擎默认公式无法由 groupIdx 推出 W/scaleB
地址，需显式建表；A/scaleA 的 gmmX 物理布局为 expert-major，与批推进公式一致，
置 nullptr 走引擎默认累计 offset。`GroupOffsetTables` 语义：`table[i]` 对应
groupList 第 i 项（按 groupList 位置对齐）。

**基址推进（UpdateAddr）**：从 **Init 时保存的固定基址绝对推导**，禁止相对累加：

```
xGM      = xBaseGM      + ((expertTokenOffset_·H1) >> FP4shift) · sizeof(x)
wGM      = wBaseGM      + ((startExpertIdx·H1·N1) >> FP4shift) · sizeof(w)
yGM      = yBaseGM      + expertTokenOffset_·N1 · sizeof(y)          // 本批 C 基址
xScaleGM = xScaleBaseGM + expertTokenOffset_·scaleK · sizeof(scale)
wScaleGM = wScaleBaseGM + startExpertIdx·N1·scaleK · sizeof(scale)
```

`expertTokenOffset_` 为批前全局 token 累计（空批不推进）。FP4 输入 2 元素/字节，
偏移右移 1 位（`PACKED_SHIFT`）。C 基址经 cGroupOffsetTable 写成批内 rank-major。

**GMMTiling 映射（InitApaceKernel）**：

| 字段 | 取值 | 说明 |
|---|---|---|
| groupNum | isLocal ? 1 : expertNum·ep | 批内 group 数 |
| m | isLocal ? BS : maxM | 调度上界：路由专家取全专家最大 token 数；共享专家实际行数为 BS |
| n / k | N1 / H1 | |
| baseM/baseN/baseK/dbL0C/isBias | 原 `mmTilingData` 同名字段 | **复用原 tiling 引擎产出**（与原通路同源） |
| kAL1/kBL1/scaleKAL1 | **0** | 引擎内兜底为 baseK（正确性优先，规避未经验证的 L1 窗口组合） |
| l1BufferStage | 2（DOUBLE_BUFFER_COUNT） | |
| groupListType | `gmmQuantParams.groupListType`（LENGTH） | 与原 tiling 一致 |

**共享专家（isLocal=true）**：单组 groupList[0]=BS，偏移表全空，`Process(0,1)` 单次
调用；在 `End()`（Finalize）之后执行，输出 mmY 不参与通信。

### 4.5 workspace 布局与 tiling 扩容

MX tiling 类覆写基类虚函数 `SetGmmA2avWorkspaceInfo()`（唯一 tiling 侧差异）：

```
wsGmmComputeWorkspaceSize（MX 模式）:
  [groupList  8·e·ep] [cTable 8·e·ep] [bTable 8·e·ep] [scaleBTable 8·e·ep]
  [ptrTable 512（占位对齐）] [TT scale repeat 8·e（占位对齐）]  → AlignUp 512
其余（wsGmmOutputSize = A·N1·yDtypeSize / wsSharedGmmComputeWorkspaceSize）同基类
```

kernel 侧 `Init` 按相同布局切表（`!isLocal` 分支），共享专家仅用 groupList。
`SetBlockDim(aicCoreNum)`（基类 PostTiling，全核启动）：Blaze 调度器满核分配 block，
闲置核空跑 Run 循环并正常参与 SyncAll。

### 4.6 任务类型：TPL KERNEL_TYPE_SEL 按组合声明（AIC_ONLY）

- `KERNEL_TASK_TYPE_DEFAULT` 为**编译单元级机制**，无法按 tiling key 组合区分任务
  类型（本地构建 14 实例 json 审计实证：`#if` 分支内声明 AIC_ONLY 不生效，全部实例
  仍为 MIX_AIC 1:2）；
- apace 组合的任务类型改由 tiling key SEL 首参 `ASCENDC_TPL_KERNEL_TYPE_SEL
  (ASCENDC_TPL_AIC_ONLY)` 声明（与仓内 allto_all_matmul 的 USING_APACE_IMPL 组合
  同款机制），原通路组合声明 `MIX_AIC_1_2`；kernel 入口不再使用
  KERNEL_TASK_TYPE_DEFAULT；
- AIC_ONLY 下无 AIV 参与：kernel 入口与 fusion 实现均无核型守卫
  （allto_all_matmul 的 apace hcomm 实现同款零守卫形态）；
- USING 为 DECL 末位参数：原 4 参 GET_TPL_TILING_KEY 调用（TT 类/基类）经
  FastEncodeTilingKeyDirect 的 min(decl, args) 截断编码为 USING=0，tiling key 值
  与上游完全一致；MX tiling 类 5 参调用编码 USING=1（新增 bit 位）。

### 4.7 调试与可观测性

| 手段 | 位置 | 用途 |
|---|---|---|
| `ApaceMXPath selected: ...` INFO 日志 | MX tiling 类 `GetTilingKey()` | 运行时选路证据（host plog 过滤）；MX 类命中 ⟹ E8M0 实例 ⟹ apace 分支（编译期保证） |
| `y[0]=0xA5A5A5A5` magic | kernel 入口 AIC block0 首写 | 卡死现场判定"新二进制在跑 + apace 分支生效"（通信成功后被 recv 覆盖） |
| json 产物审计 | `bin/**/QuantGroupedMatMulAlltoAllv_*.json` | 全实例 `MIX_AIC 1:2`；MX 实例（gen param json 中 gmm_x_scale=float8_e8m0）以 .o 尺寸/符号差异确认 apace 已编入 |
| device 侧日志 | 无 | printf 依赖 debug-bus（本地 CANN 为 no-op shim），不可用 |

---

## 5 关键设计决策记录

| # | 决策 | 依据/教训 |
|---|---|---|
| D1 | 编译期实例选路（E8M0 scale + defined 守卫），弃用 tiling key 选路 | tiling key 方案要求全实例编译 apace 分支，dtype 宏无守卫判别在真实实例上误判为否 → 空实现 + MC2 标记 → 通信任务等握手超时（历次全 stage 卡死根因） |
| D2 | 通信在 AIC 核内（弃用 AIV KFC 通路） | Blaze 与 AIV 侧 KFC 同实例共存 → AIV 通信任务参数校验失败（与是否执行计算无关） |
| D3 | apace 组合 AIC_ONLY（TPL KERNEL_TYPE_SEL 按组合声明） | `KERNEL_TASK_TYPE_DEFAULT` 编译单元级无法按组合区分（14 实例 json 实证）；TPL SEL 为仓内已验证机制（allto_all_matmul 同款），无需核型守卫 |
| D4 | 复用原 `QuantGmmA2avTilingData` 与原 MM tiling 字段 | 接口与模板签名对齐 `QuantGroupedMatmul`，原算子目录侵入最小（tiling_common 零修改） |
| D5 | kAL1/kBL1/scaleKAL1 = 0 兜底 baseK | 正确性优先；自定义 L1 窗口组合未经真机验证（历史上 stage=4 Blaze 挂死的嫌疑组合） |
| D6 | 基址从固定基址绝对推导 | 相对累加在多批场景重复计入历史批偏移（实现期自查发现） |
| D7 | 共享专家 m = BS（非 maxM） | 共享专家实际行数与路由专家 token 分布无关，maxM 上界可能小于 BS 导致行截断 |
| D8 | workspace 扩容放在 MX tiling 类覆写（虚函数） | 公共 tiling_common 保持与主库字节一致；TT 类不受影响 |

## 6 已知限制与后续方向

1. **无 device 侧日志**：依赖 magic + host 日志 + json 审计三板斧；若后续 CANN 提供
   可用 debug-bus/printf 通道可补充；
4. **通信引擎**：`HcclServerTypeSelector` 支持 CCU/AICPU 两态，真机验证以 CCU 为主，
   AICPU 模式未充分验证。

## 7 验证状态

- 本地构建：14 实例（MX 10 + TT 4）0 error；全实例 `MIX_AIC 1:2` 与守卫假设一致；
  MX 实例 .o 与 TT 实例符号/尺寸差异显著（apace 分支编入正常）；
- 真机：MX 用例回归待执行（构建部署后无需任何环境变量/开关）。
