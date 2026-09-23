# QuantGroupedMatMulAlltoAllv apace 通路 SRS 需求详设

| 项 | 内容 |
|---|---|
| 文档版本 | V2.0 |
| 适用芯片 | Ascend 950（dav_c310，`__NPU_ARCH__=3510`） |
| 关联实现 | `mc2/common/op_kernel/apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/`（impl + hcomm_impl） |

> **V2.0 变更记录**：通路选择由 tiling key + 运行时环境变量改为**编译期分支**
> （`MX_QUANT_MODE`，按 gmmXScale 的 E8M0 dtype 判别 MX 量化实例）；不再引入自定义
> tiling key / tiling 类 / tiling 数据结构，复用原 `QuantGmmA2avTilingData`；通信与计算
> 均在 AIC 核内完成（AIV 空转）。此前 tiling key 选路方案的 kernel 侧 dtype 宏在真实
> 实例上误判（未定义的 `ORIG_DTYPE_*` 宏经 `#if` 求值为 0），apace 分支退化为空实现，
> 而 tiling 侧已下发通信标记，导致 stream 上通信辅助任务等待握手超时——表现为所有
> 诊断 stage 均卡死。编译期分支从结构上消除该失败模式：分支未命中时实例编译为原通路。

---

## 1 引言

### 1.1 目的

定义 `QuantGroupedMatMulAlltoAllv` 算子 **apace hcomm 融合通路**（MX 量化路径）的软件需求规格：
GMM 计算复用 apace 库 `GroupedMatmulMxKernel`（Blaze block_mmad_qgmm_mx），AllToAllV 通信在
AIC 核内经 `Hccl` 提交，二者同核融合执行。

### 1.2 范围

apace 通路是 MX(FP4/FP8) 量化实例的加速路径（默认编译，无构建开关）。TT 量化实例完全
保持原通路（`GmmA2avScheduler + HcclA2avOp + QuantGroupedMatmul`，AIV 通信）行为。
aclnn 接口、输入校验、infershape、tiling 注册体系均不变。

### 1.3 术语

| 术语 | 定义 |
|---|---|
| EP / e | 专家并行域大小 / 单卡专家数（e ≤ 32，e×ep ≤ 256） |
| expertNum | 每批 GMM 合并的专家数（原通路 `CeilDiv(e, 4)`，批数 ≤ 4） |
| MX 实例 | gmmXScale/gmmWeightScale 为 E8M0 的 kernel 实例（MX 块量化） |
| rank-major | 按 rank 分段连续的物理布局 |
| expert-major | 按 expert 分段连续的物理布局 |

## 2 总体描述

### 2.1 选路机制（tiling key 选路）

| 层 | 开关 | 说明 |
|---|---|---|
| 运行期-tiling key | `TILINGKEY_USING_APACE_IMPL`（TPL 第 5 参数，DECL 末位，原 4 参编码值不变） | MX tiling 类编码 USING=1（MX 量化恒走 apace）；TT tiling 类 4 参调用截断为 USING=0；kernel 以 `if constexpr (TILINGKEY_USING_APACE_IMPL)` 选路 |
| 编译期-实例 | `MX_QUANT_MODE`（E8M0 scale 判别，defined() 守卫） | 仅作 TT 实例的组合裁剪：TT 实例的 USING=1 组合为空实现占位（不会被 tiling 选中） |
| 任务类型 | TPL `ASCENDC_TPL_KERNEL_TYPE_SEL` | USING=0 组合 → `MIX_AIC_1_2`（原通路）；USING=1 组合 → **`AIC_ONLY`**（无 AIV 参与，无需核型守卫）；不再使用 `KERNEL_TASK_TYPE_DEFAULT` |

**设计约束**：
- 实例级分支必须编译期可判：tiling key 方案要求全部 dtype 实例编译 apace 分支，依赖
  `ORIG_DTYPE_*` 宏判别 dtype，宏未定义时 `#if` 求值为 0 导致误判（历史卡死根因）；
  编译期 `MX_QUANT_MODE` 分支使非 MX 实例根本不编译 apace 代码。
- 通信必须在 AIC 核内：Blaze（cube 模板）与 AIV 侧 KFC 通信引擎在同一 kernel 实例中
  共存时，AIV 通信任务参数校验失败（execute kernel param invalid，与是否执行计算无关）。
  apace 通路将 AlltoAllV 改由 AIC 核 `Hccl` 提交，AIV 分支为空。

### 2.2 执行模型

```
kernel（MIX_AIC_1_2，与原通路同任务类型）:
  AIC: hccl InitV2/SetCcTilingV2 → for 批b { GMM(b)（GroupedMatmulMxKernel）
        → SyncAll → block0 提交+等待 AlltoAllV(b) → SyncAll } → End(Finalize)
        → 共享专家 MM（可选，独立输出不走通信）
  AIV: 空转（通信在 AIC 核内完成）
```

### 2.3 数据布局契约

| 缓冲 | 布局 |
|---|---|
| gmmX / gmmXScale | expert-major（A/scaleA 基址按批推进，批内走 kernel 默认公式） |
| gmmOutput（发送缓冲） | 批间串接、批内 rank-major（C 偏移表提供） |
| y | rank-major（recvCounts 按源 rank 聚合顺序，与原通路一致） |
| gmmWeight / gmmWeightScale | expert 粒度（批内经 B/scaleB 偏移表定位，group 序 e×ep+r 与 expert 序不一致） |

## 3 功能需求

### 3.1 计算适配层 `Apace::QuantGroupedMatmulMx`

- 对外接口 `Init/Process/End` 与 `MC2KernelTemplate::QuantGroupedMatmul` 一致，kernel 入口可
  按编译期分支直接替换实例化类型；
- 复用原 tiling 数据（`QuantGmmA2avTilingData` + `GMMQuantTilingData.mmTilingData`）：
  `baseM/baseN/baseK/isBias/dbL0C` 取自原 MM tiling，`groupListType` 取自 `gmmQuantParams`；
  `kAL1/kBL1/scaleKAL1` 填 0 由模板兜底为 baseK（正确性优先），`l1BufferStage=2`；
- 批粒度调用：`groupNum = expertNum×ep`，`m = maxM`（全专家最大 token 数）；空批（批内
  token 全 0）跳过且 token 偏移不推进；
- workspace（gmmComputeWS）内建四表：groupList（LENGTH，expert-major）、cGroupOffsetTable
  （批内 rank-major）、bGroupOffsetTable / scaleBGroupOffsetTable（expert 粒度）；
- FP4 输入按 2 元素/字节右移 1 位折算 A/B 偏移；
- 共享专家（isLocal=true）：单组 groupList，偏移表全空。

### 3.2 通算融合层 `Apace::QuantGmmA2avHcommImpl`

- Init：`hccl_.InitV2(GetHcclContext<0>, hcclInitTiling)` + `SetCcTilingV2(a2avCcTiling 偏移)`
  （复用原 tiling 的 HCCL 参数），随后初始化计算适配层；
- Process（每批，全核同构）：GMM → `SyncAll`（本批写落 GM）→ 全核**直接提交**非阻塞
  `AllToAllV<true>`（无需等待上一批）→ `PipeBarrier`（排空本核管线，保护参数数组复用）。
  通信由 AICPU/CCU 引擎异步执行、不占用 AIC 核：各批通信在引擎侧排队，与本核后续
  批次 GMM 重叠推进；提交/等待均全核调用（多核 Wait 协议：Hccl 内部 workingFlag_
  门控 block0 写参数/敲 doorbell，工作核轮询推进、非工作核跟随）；
- End：全核按提交顺序 `Wait` 全部批次 → `SyncAll` → `hccl_.Finalize()`（须有提交）；
- 通信参数（元素单位，axis=N1）：
  - `sendCnt[r] = Σ_{eIdx∈批} sendCounts[eIdx + r·e] × N1`；
  - `sendOff[r] = 批起点(批前全部 token×N1) + 批内前序 rank 段累计`；
  - `recvCnt[r]` 同构；`recvOff[r]` 为 rank 段基 + 批前前缀（经
    `startExpertIdx + (i-1)·e + j` 跨段索引的伸缩和恒等式计算）；
- HCCL server 类型按 `TILINGKEY_COMM_MODE` 选择（CCU/AICPU）。

### 3.3 Tiling 侧

- 不新增 tiling 类/tiling key，不改公共 tiling_common；`MxQuantGroupedMatmulAllToAllvTiling`
  覆写 `SetGmmA2avWorkspaceInfo()`（基类虚函数），在 apace 构建开启时为
  `wsGmmComputeWorkspaceSize` 扩容 B/scaleB 偏移表（`e×ep×8B×2`，`#ifdef` 门控），
  构建未开启时与公共口径一致；TT 类不受影响。

### 3.4 kernel 入口

- `if constexpr (TILINGKEY_USING_APACE_IMPL)` 分支（内层 `#if MX_QUANT_MODE` 裁剪
  TT 实例占位组合）：block0 写部署自检 magic `y[0]=0xA5A5A5A5`（成功后被 recv
  数据覆盖）→ 按批 `hcommImpl.Process` → `End` → 共享专家 `Process(0,1)`；
- `#else`：原通路逐字保持（`KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_2)`、
  `GmmA2avScheduler.Process()`）。

## 4 验收标准

| # | 项 | 标准 |
|---|---|---|
| A1 | 数值正确性 | MXFP8/MXFP4 与原通路 golden 对拍一致（含共享专家、空 expert、多批 e>4） |
| A2 | TT 无损 | TT 量化实例行为与基线完全一致（编译期隔离） |

| A4 | 通路证据 | 开关开启后 MX 实例走 apace 分支（magic 可检/性能可观测） |

## 5 附录

### 5.1 构建

```
bash build.sh --opkernel --soc=ascend950 --ops=quant_grouped_mat_mul_allto_allv -j8
```

apace 通路默认编译，无构建/运行开关；`MX_QUANT_MODE` 为实例编译期自动判别
（a2av_gmm_utils.h，按 gmmXScale 的 E8M0 dtype，勿手工设置）。

### 5.2 通路命中确认方法

| 时机 | 方法 | 证据 |
|---|---|---|
| 运行时（host plog） | 过滤关键字 `ApaceMXPath`（MX tiling 类 GetTilingKey 打印，INFO 级） | MX 量化实例选中 → 该 dtype 实例二进制固定为 apace 分支（编译期保证），日志同时给出 tilingKey/e/ep/A/N1/H1 等关键参数 |
| 构建产物 | json kernelList 应含两组组合：USING=0（key 0~15，`MIX_AIC` 1:2）与 USING=1（key 16~31，`AIC_ONLY`）；MX tiling 类下发 USING=1 key → 运行时选中 AIC_ONLY kernel |
| 卡死现场 | 读 y 前 4 字节 | `0xA5A5A5A5` = apace 分支已执行（magic 由 block0 首先写入，通信成功后被 recv 数据覆盖；卡死残留即为部署/通路证据） |
| 性能侧证 | msprof | MX 用例 kernel 为 AIC_ONLY 任务类型（无 AIV）；TT 用例行为与基线一致 |

（原 5.2 代码索引顺延为 5.3）

### 5.3 代码索引

| 组件 | 路径 |
|---|---|
| 计算适配层 | `mc2/common/op_kernel/apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_impl.h` |
| 通算融合层 | 同目录 `quant_grouped_matmul_all_to_all_v_hcomm_impl.h` |
| kernel 入口 | `mc2/quant_grouped_mat_mul_allto_allv/op_kernel/quant_grouped_mat_mul_allto_allv.cpp` |
| tiling 扩容（MX 类覆写） | `mc2/quant_grouped_mat_mul_allto_allv/op_host/op_tiling/arch35/mx_quant_grouped_mat_mul_allto_allv_tiling.{h,cpp}`（SetGmmA2avWorkspaceInfo 覆写） |
| Blaze GMM 引擎 | `mc2/common/op_kernel/apace/kernel/matmul/grouped_matmul/grouped_matmul_mx_kernel.h` |

### 5.3 历史方案废弃说明

V1.x 的 tiling key 选路方案（`TILINGKEY_USING_APACE_IMPL` + APACE 通信模式 + 自定义 apace
tiling 类/tiling 数据 + 运行时环境变量 + 7 级 debugStage）已整体废弃移除：该方案要求全部
dtype 实例编译 apace 分支，kernel 侧 dtype 判别宏在真实实例上误判导致分支退化为空实现，
是历次真机卡死的根因；现行编译期分支方案从结构上消除该失败模式。
