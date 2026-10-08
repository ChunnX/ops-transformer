# fit-balance tiling 引擎——详细设计与接入指南

> 通信-计算流水线配平切分引擎（header-only, 950 标卡 4P/8P, AICPU/CCU 通信引擎）。
> 本文档面向接入本引擎的算子开发者：先看接口与示例（5 分钟接入），需要深究行为时再看详细设计。

---

## 一、对外接口

全部符号为 global namespace，include 这一个头文件即可用：

```cpp
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"
```

### 主入口

```cpp
CommCutResult FitBalanceCut(const FitBalanceTilingArgs &args, const FitBalanceCalibration &calib);
```

单入参 + 标定。内部串联 上下文构造（InitCtx）→ 键控求解（Schedule）→ 诊断打印（debug 级）。
返回"1 短 + N 长"的 M 轴切分方案（或退化的等分/单块）。
通信引擎（AICPU/CCU/AIV_URMA）经 `args.commEngine` 传入——op 侧 `ParseCommEngine(comm_mode)` 解析。

```cpp
template <typename LegacyArgs>
FitBalanceTilingArgs MakeFitBalanceArgs(const LegacyArgs &a, TilingCommEngine commEngine);
```

存量算子过渡用：从旧全局 mc2tiling::TilingArgs 挑字段构造引擎入参。
函数是模板（只要求传入类型有同名成员字段，引擎不依赖旧类型定义）；commEngine 由调用方解析后补入。
**新算子直接构造 FitBalanceTilingArgs，不用本函数**。

### 分派判定

```cpp
bool IsCommCalibrated(uint32_t rankDim, TilingCommEngine commEngine, const FitBalanceCalibration &calib);
```

通信标定表是否命中当前形态（通信引擎 + 卡数二维匹配）。算子用它查询本引擎可用性——
**命中即可走 FitBalanceCut；miss 的处理（其他切分引擎/整块不切等）由算子开发者自选**，
本引擎不预设任何兜底策略。

```cpp
TilingCommEngine ParseCommEngine(const char *commMode);   // "aiv_urma"→AIV_URMA, "ccu"→CCU, 其余(含"ai_cpu")→AICPU
```

### 进阶（一般不需要直接用）

```cpp
FitBasedBalanceCtx InitCtx(args, calib);                          // 上下文构造
CommCutResult Schedule(ctx);                                      // 键控求解
void PrintTilingData(args, cut, commTimeActualUs, commTimeNoCutUs); // 诊断打印(通常由 FitBalanceCut 自动调)
const FitBalanceCalibration::CommFitRow *FindCommFitRow(calib, engine, rankDim); // 标定行查找
CommCutResult CutResultFromCof(const CommTilingData &data);        // 核利用率优先切分结果(head/tail)
                                                                   // → 本引擎切分(长/短)形态转换
```

### 核心类型

| 类型 | 说明 |
|---|---|
| `FitBalanceTilingArgs` | 引擎调度入参（10 字段：m/n/k、rankDim、aicCoreNum、dtype 四元组、commEngine）——纯运行时事实 |
| `CommCutResult` | 切分结果（shortTileAtBack / totalTileCnt / numLongTile / numShortTile / longTileLen / shortTileLen） |
| `FitBalanceCalibration` | 客户标定数据总集（dataflow 结构声明 + 通信拟合行 k/b/c + matmul 算力档），算子侧唯一编辑面 |
| `LocalMMPosition` | 流水线结构键控：FRONT（本地 mm 块前置）/ BACK（后置）/ NONE（无本地块） |
| `CommComputeOrder` | 通算顺序：COMM_FIRST（通信在前）/ COMPUTE_FIRST（计算在前） |
| `TilingCommEngine` | 通信引擎：AICPU / CCU / AIV_URMA |

---

## 二、接入示例（新算子视角）

新算子直接使用引擎的 args/cut 类型，无需任何旧类型转换：

```cpp
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"

// ── ① 标定数据: 每算子一份, 客户唯一编辑面 ─────────────────────────────────
// profile 按本算子流水线结构填写; rows 填实测拟合值; mmFit 按 dtype 档填
static const FitBalanceCalibration CALIBRATION = {
    /* dataflow: 数据流形态(接入时定死的结构声明, 引擎据此推导 axis 与全部倍数) */
    {
        CommComputeOrder::COMM_FIRST, // 通信在前: gather 远程输入再算 → 引擎推导 K 轴/A 侧承载
        LocalMMPosition::FRONT,       // 本卡 A 块已在手, 聚合前即可算 → 引擎推导工作集 rankDim−1
        true,                         // 全量形态: 实算 mm 行数与单次通信字节均为切分域×rankDim
    },
    /* commRows: 通信拟合行——{通信引擎, 卡数, k 带宽斜率(us/MB), b 每轮启动开销(us), c 一次性固定开销(us)};
     * 一行 = 一个已标定 (引擎, 卡数) 形态; 未标定形态不写行(IsCommCalibrated 返回 false, 由算子自选策略处理) */
    {
        { {TilingCommEngine::AICPU, 4, 26.5894, 33.17, 20.58}, // 950 标卡 4P 实测(在网行)
          // ↓ 两行新增标定填法演示(数值为演示数——上线前替换为实测值, 未标定则删行留空):
          {TilingCommEngine::AICPU, 8, 22.7301, 33.17, 20.58}, // 新卡档: 同引擎追加一行(8P 实测后照此)
          {TilingCommEngine::AIV_URMA, 4, 3.2, 4.45, 1.8} },   // 新引擎: 引擎维度在行上, 同表加行即可
                                                               // (aiv_urma 是否走本引擎, 标定后视情况决定)
    },
    /* mmFit: matmul 算力档 + 最优基本块 + 平台规格 */
    {
        { {MatmulDtypeKind::QUANT, false, 1, 1, 8192},   // fp8 量化
          {MatmulDtypeKind::QUANT, true, 0, 0, 16384} }, // fp4 打包(键 {0,0} 是现网真实值, 见 4.2)
        {256, 256, 128},                                  // baseBlock
        // 平台规格可不填(默认 950: 频率 1689.6 cycle/us, L2 容量 128MB);
        // 新芯片按规格覆盖, 如 960: cyclePerMicroSec = ..., l2CacheBytes = ... (见 4.4)
    },
};

// ── ② tiling 主流程: 分派 + 一行调用 ─────────────────────────────────────
ge::graphStatus MyOpTiling::DoTiling()
{
    const auto commEngine = ParseCommEngine(commMode_);   // "ai_cpu"/"ccu" → 枚举

    CommCutResult cut;
    if (IsCommCalibrated(rankDim_, commEngine, CALIBRATION)) {
        // 命中: 构造引擎入参并调用(存量算子也可用 MakeFitBalanceArgs 从旧 TilingArgs 转换)
        FitBalanceTilingArgs args = {mValue_, nValue_, kValue_, rankDim_, aicCoreNum_,
                                     aType_, geAType_, inputDtypeSize_, outputDtypeSize_,
                                     ParseCommEngine(commMode_)};
        cut = FitBalanceCut(args, CALIBRATION);
    } else {
        // miss: 未标定形态——切分策略由算子开发者自选(其他切分引擎/整块不切等), 本例从略
        cut = MyOtherTilingStrategy();
    }
    // 直接消费 CommCutResult: kernel 侧按 shortTileAtBack 指示的顺序执行 1 短 + N 长
    UseCutResult(cut);
    return ge::GRAPH_SUCCESS;
}
```

接入要素就一个：**填 CALIBRATION**——结构声明（dataflow 三元组）与实测数据（拟合行/算力档）
都在这一处；调用侧构造 `FitBalanceTilingArgs`（存量算子经 `MakeFitBalanceArgs` 转换）后单入参调用。

**多芯片算子**：标定是普通入参，引擎不含芯片分支——算子侧为每个芯片各备一份 CALIBRATION
（仓内惯例是 tiling 按芯片分目录各持一份），检测平台后选用：

```cpp
static const FitBalanceCalibration CALIB_950 = {...};
static const FitBalanceCalibration CALIB_960 = {...};   // 频率/L2/算力档/通信行全按 960
// DoTiling 内: PlatformAscendC platform(context_->GetPlatformInfo()); 按其结果选芯片
const FitBalanceCalibration &calib = IsNewChip(platform) ? CALIB_960 : CALIB_950;
if (!IsCommCalibrated(rankDim_, commEngine, calib)) { /* miss → 其他切分策略 */ }
cut = FitBalanceCut(args, calib);
```

完整接入清单（新芯片要改哪些数、复验哪些公式）见 4.4。

### 数据流形态（CALIBRATION.dataflow 三元组）

算子接入时定死的**结构声明**，引擎据此推导其余全部结构量：

| 字段 | 回答的问题 | 取值含义 |
|---|---|---|
| `order` | 流水线**哪段先行**？ | `COMM_FIRST` 通信在前（先收输入再算）；`COMPUTE_FIRST` 计算在前（先算再发输出） |
| `localMMPosition` | kernel 侧有没有实现"本地计算块的特殊调度"？ | `FRONT` 通信在前时，不参与通信的本地计算块不依赖通信、提前开始计算（kernel 已实现该调度）；`BACK` 通信在后时，先算被通信依赖的块以求尽早开始通信、本地块后置计算（kernel 已实现该调度）；`NONE` kernel 无此特殊实现 |
| `fullVolume` | 切分域（mValue）是"本卡份"还是"本卡实算全量"？ | `true` 全量形态（实算 mm 行数与单次通信字节均为切分域×rankDim，聚合/分发类）；`false` 单份形态（本卡一份，两者×1） |

**推导规则**（引擎内部，接入方无需再填任何结构量）：

```
通信承载侧   axis   = (order == COMM_FIRST) ? K 轴(传 A 行, 另一维=kValue)
                                          : N 轴(传 C 行, 另一维=nValue)
总量倍数     rowMul = commVolumeMul = fullVolume ? rankDim : 1
工作集份数   shares = (localMMPosition == FRONT) ? rankDim−1      // 本地份剔出稳态、单独按单份建模
                                                : rowMul
```

选错的后果：order/localMM 错 → 配平锚点错位、短块位置错、总时公式形态错（切分整体劣化）；
fullVolume 错 → 通信量与 mm 总量估计整体偏移。

判定方法（只看本算子自己的通信方向与 kernel 实现，不考虑其他因素）：
1. **order**：通信的输入是 matmul 的输入 → `COMM_FIRST`；通信的输入是 matmul 的输出 →
   `COMPUTE_FIRST`
2. **localMMPosition**：与通算顺序和 kernel 侧实现绑定——
   - 通信在前，且存在不参与通信的本地计算块可以不依赖通信直接提前开始计算，且 kernel 侧
     实现了这一调度 → `FRONT`
   - 通信在后，存在不被通信依赖的本地计算块，先算被通信依赖的块以求尽早开始通信、把本地
     块后置计算，且 kernel 侧实现了这一调度 → `BACK`
   - kernel 侧没有对本地计算块的特殊调度实现 → `NONE`（即使数据流上存在本地块）
3. **fullVolume**：mValue 传进引擎的是"本卡份"（实算量 = mValue × rankDim）→ `true`；
   mValue 本身就是本卡实算全量 → `false`

诊断打印内置：`ASCEND_GLOBAL_LOG_LEVEL=0` 时 FitBalanceCut 输出输入 shape / 最终切分 / 启动税三段。

---

## 三、详细设计

### 3.1 文件总览——先认识六个文件

消费方只 include 总接口头文件 `fit_balance_tiling.h`；内部五文件由它间接引入（依赖单向，
`types ← {comm_model, matmul_model, cut_planner} ← solver ← 总接口头`）：

| 文件 | 一句话职责 | 关键内容 |
|---|---|---|
| `fit_balance_tiling.h` | 总接口头文件：对外统一入口 + 流程编排 | FindCommFitRow / IsCommCalibrated / **InitCtx / Schedule / FitBalanceCut** / PrintTilingData / CutResultFromCof / MakeFitBalanceArgs |
| `utils/fit_balance_types.h` | 类型层：引擎共享词汇 | 键控枚举（LocalMMPosition / CommComputeOrder / TilingCommEngine）、数据结构（FitBalanceTilingArgs / CommCutResult / 模型画像）、常量、**FitBalanceCalibration 标定总集**（dataflow 三元组 + commRows 表 + mmFit 算力档与平台规格） |
| `utils/comm_model.h` | 通信模型：HCCL 传输耗时 | HCCLCommModel——k/b/c 三参数 + 三口径（CommTime 单轮 / InverseCommTime 反解 / TotalCommTime 总量） |
| `utils/matmul_model.h` | 计算模型：cube 耗时 | MatmulPerfModel——利用率公式（聚合 0.7 / 本地窗口 0.6 双口径）/ 每行梯度 / MatmulTime 正反解 |
| `utils/cut_planner.h` | 离散切分引擎：整数块整形 | CutPlanner——"1 短 + N 长"的初切 / 离散化七步 / 尾对齐 / NoCut，总长守恒与对齐粒度的守门人 |
| `utils/solver.h` | 求解层：上下文 + 公共片段 | FitBasedBalanceCtx / EstimateTotalTimes / BalanceLongTileLen / EstimateCutTotalTime / ApplySingleRoundCommLimit / L2 族片段 |

> 切分策略的选择完全由算子开发者决定——本目录下还有核利用率优先切分（`core_occupancy_first_tiling.h`）
> 等其他引擎可用，是否与本引擎配合、如何配合（含标定未命中时走什么）均不在本引擎的约定范围内。

### 3.2 核心思想：通信-计算流水线配平

通算融合算子的执行是流水线：通信块与计算块交替重叠。切分要回答两个问题——
**块怎么切**（长短结构）与**切几轮**（轮数经济性）。

本引擎的模型：把 M 轴切成 1 短 + N 长，长块的通信时长恰好掩盖相邻块的计算时长（配平）；
每多切一轮多付一份启动开销 b，用轮数搜索择优。

配平的**锚点 = 流水线的启动块**（最先动起来、打破串行依赖的那块）：
有本地块的算子启动块是本地块（FRONT 在首端 / BACK 在尾端，比短块更好——不等通信或没有发送时限）；
无本地块的算子启动块是短块（通信耗时占比大时放在最前先发、计算耗时占比大时放在最后垫尾）。
长块长度 = 按启动块的耗时反解——这是 FRONT/BACK/NONE 三种形态配平公式的统一原理。

### 3.3 调用全景图

```
FitBalanceCut(args, calib) ───────────────────────────────── [总接口头] 入口, 见 3.4
 ├─ InitCtx(args, calib) ─────────────────────────────────── [总接口头] 上下文构造, 见 3.5
 │   ├─ FindCommFitRow        通信标定选行(miss = 零值行)
 │   ├─ matchesCalcProfile    算力档选行(miss = 默认算力)
 │   ├─ 构造三件套            MatmulPerfModel / HCCLCommModel / CutPlanner
 │   ├─ 数据流倍数推导        volumeMul / commRankCntForMm / mmRankMul
 │   ├─ 通算胶水              SetCommShapeLen / SetCommDTypeSize(+fp4 子字节)
 │   ├─ miss 防御             slope==0 → 抬 minLen, 后续最短可切校验兜底 NoCut
 │   └─ 对齐粒度              SetMinLenByMax(baseM) / SetAlignLength(baseM)
 ├─ Schedule(ctx) ────────────────────────────────────────── [总接口头] 六段求解, 见 3.6
 │   ├─ ① EstimateTotalTimes [solver.h] → 本地窗口(FRONT/BACK) → 短块位置
 │   ├─ ② ExceedsL2Cache → ApplyL2MinLenBoost → 2b 下限 → 最短可切校验(装不下两块则 NoCut)
 │   ├─ ③ 长块反解            InverseCommTime(本地窗口) / BalanceLongTileLen [solver.h]
 │   ├─ ④ 离散化              IsGoodLinearityShape → FitTileLengthDiscrete [cut_planner.h]
 │   │                        七步 → MergeEqualTiles [solver.h]
 │   ├─ ⑤ 轮数搜索+门控       EstimateCutTotalTime [solver.h] 择优; 全败 → NoCut
 │   ├─ ⑥ CCU 整形            ApplySingleRoundCommLimit [solver.h] (256MB 上限)
 │   └─ 尾部精算              commTimeActualUs / commTimeNoCutUs
 └─ PrintTilingData ──────────────────────────────────────── 诊断打印(debug 级), 见 3.7
```

决策层次一句话：**③④ 管块怎么切（配平），⑤ 管切几轮（启动税 vs 重叠收益），⑥ 管硬件上限**。

### 3.4 入口：FitBalanceCut

```cpp
inline CommCutResult FitBalanceCut(const FitBalanceTilingArgs &args, const FitBalanceCalibration &calib)
{
    FitBasedBalanceCtx ctx = InitCtx(args, calib);
    CommCutResult cut = Schedule(ctx);
    PrintTilingData(args, cut, ctx.commTimeActualUs, ctx.commTimeNoCutUs);
    return cut;
}
```

三步串联：构造上下文 → 求解 → 诊断打印。两个入参的分工：**args 是纯运行时事实**（shape /
卡数 / dtype / 通信引擎，每次调用可变），**calib 是纯静态标定**（结构声明 + 实测数据，客户
唯一编辑面）——二者在入口汇合，此后引擎内部不再区分数据来源。
配套的入口族函数（不在主流程内）：`MakeFitBalanceArgs`——存量算子把旧全局 TilingArgs 挑字段
转成引擎 args（模板函数，只要求传入的旧类型有 mValue/nValue 等同名成员字段，引擎不依赖
旧类型定义）；`CutResultFromCof`——把核利用率优先切分的 head/tail 结果转成
本引擎的长短形态（供同时使用两种切分的算子统一输出形态）。

### 3.5 InitCtx——上下文构造（按源码顺序走读）

InitCtx 做一件事：把 args + calib 两组输入"编译"成一个自足的调度上下文
`FitBasedBalanceCtx`——此后 Schedule 只读 ctx，不再接触 calib。ctx 的组成 =
三件套（matmulPerf / commPerf / tilingM）+ 三个键控维度（localMMPosition / order /
commEngine）+ 派生倍数（commRankCntForMm / mmRankMul / commVolumeMul）+ 过程量
（frontMMTime / backMMTime / 诊断量）。按源码顺序九步：

**第 1 步｜读结构声明。** 从 `calib.dataflow` 取 order（通算顺序）与 localMMPosition
（本地块位置）——这两个枚举决定 Schedule 里所有键控分支的走向。

**第 2 步｜通信标定选行（FindCommFitRow，总接口头）。** 单表 commRows 线性扫描：
rankDim==0 的行是未填哨兵（截断）；命中 = (commEngine, rankDim) 二维匹配且斜率非零。
miss 返回 nullptr → 构造零值 CommFitParams（slope=0）——不报错、不回退默认数值，
第 8 步的防御会把它变成 NoCut。正常接入下 op 层已按自选策略处理 miss，此路径只是引擎自防御（兼防 InverseCommTime 除零）。

**第 3 步｜算力档选行。** 先推导匹配键：isQuant（aType 是 HIFLOAT8/E4M3FN/E5M2/INT8，
或 geAType 是 FP4 → QUANT，否则 FP16）+ isFp4；然后逐行匹配四维键
(calcType, isFp4, aBytes==inputDtypeSize, bBytes==inputDtypeSize)。
**未填行判 `computesPerCycle == 0`**——不能判 aBytes==0（fp4 行的 aBytes 就是合法的 0，
这是 CPC 键缺陷的教训）。miss 落 defaultComputesPerCycle（4096）。

**第 4 步｜构造三件套（ctx 的主体）。**

- **MatmulPerfModel（计算模型，matmul_model.h）——回答"算多久"**：
  - 内容：`mmShapeInfo_`（12 字段：m/n/k、baseM/N/K、A/B/C 每元素字节数、coreNum、
    computesPerCycle、频率）+ 过程量 cubeUtil_ / matmulGradient_；
  - 构建：从 args 抄 shape 与字节数（B 字节 = A 字节），baseM/N/K 来自标定 baseBlock，
    SetCalcType 完成 dtype 分类，computesPerCycle 填第 3 步选行结果，频率写死 950 的
    1.65K cycle/us、L2 容量 128MB——均来自标定 mmFit（950 默认值，新芯片按规格改）；
  - 作用：FindCubeUtil（利用率，见 ①）/ GetMatmulGradient（每行耗时系数，见 ①）/
    MatmulTime·InverseMatmulTime（耗时正反解）——①③⑤ 的计算侧估值全部出自它。
- **HCCLCommModel（通信模型，comm_model.h）——回答"传多久"**：
  - 内容：`commEstimatePar_`（k/b/c 三参数）+ `commTypeInfo_`（rankDim、commDtypeSize、
    commMatrixLen、子字节折算 fraction）+ commTimeFactor_（恒 1，历史兼容位）；
  - 构建：rankDim 钳位 ≥2（防 uint64 下溢 / 单卡退化 / 来源防御三层原因），k/b/c 填
    第 2 步选行结果；
  - 作用：三口径公式——单轮 `CommTime(m) = k·x + b`、反解 `InverseCommTime(t) = (t−b)/k`
    （b 每轮全款、整额扣除）、总量 `TotalCommTime(m, 轮数) = k·X + 轮数×b + c`（c 整个算子只付
    一次，只进总量口径）。x 的基准 = m × commMatrixLen × 有效字节数 的 MB，**按单卡数据量
    计——多份数的放大已折进标定斜率 k**（聚合类标定行直接填工具原始值，勿再除 rankDim）；
    fp4 子字节 = dtypeSize 1 / fraction 2 = 0.5 B/元素。
- **CutPlanner（离散切分引擎，cut_planner.h）——回答"整数块怎么落"**：
  - 内容：`cutRes`（切分结果六字段）+ `tileArgs`（mAlignLen 对齐粒度 / minTileLen 单调
    下限 / maxTileCnt 轮次预算 / maxTileLen）+ totalLen（=mValue）；
  - 构建：对齐粒度先取 DEFAULT_BASE_M，**初始状态 = NoCutTiling**（单块整长——引擎任何
    提前退出都落在这个安全形态上）；
  - 作用：④ 的离散化七步（见 3.6）+ NoCut / 超限兜底，总长守恒的守门人。
    SetMinLenByMax 只升不降——② 的三道下限闸门全靠它叠加。

**第 5 步｜轮次预算。** `SetMaxTileCnt(MAX_COMM_ROUNDS=63)`——通信 API 的 FFTS 队列
轮次上限（⑤ 搜索的预算上限即此值）。

**第 6 步｜数据流倍数推导（calib.dataflow → 三个倍数）。**
`volumeMul = fullVolume ? rankDim : 1`（全量形态：切分域是本卡份，实算与单次通信均为
×rankDim；单份形态 ×1）；`commRankCntForMm = (FRONT) ? rankDim−1 : volumeMul`（FRONT
把本地份剔出稳态工作集——它单独按单份建模）；`mmRankMul = commVolumeMul = volumeMul`
（前者是 mm 总行数倍数，后者是 ⑥ 的物理字节倍数——与斜率 k 的口径融合是两个用途）。

**第 7 步｜通信矩阵另一维（D4）。** `commOnKAxis = (order == COMM_FIRST)`——通信在前
传输入 A 侧（另一维 = kValue），在后传输出 C 侧（另一维 = nValue）。SetCommShapeLen 落位。

**第 8 步｜通信 dtype 与 fp4 子字节（D5/D6）+ miss 防御。** K 轴承载：fp4 →
SetCommDTypeSize(1) + fraction=2；否则取 A 字节。N 轴承载：取 C 字节（计算输出永远
不是 fp4）。mm 侧 fp4 口径独立于此分支：A/B 字节抬为 1（L2 估算按 2 倍保守——真实
0.5B 表达不了）。最后 miss 防御：slope==0 → SetMinLenByMax(mValue)——② 的最短可切校验
（总长 < 下限×2）必触发 NoCut。

**第 9 步｜对齐粒度（D7）。** `SetMinLenByMax(baseM)` + `SetAlignLength(baseM)`——
离散对齐粒度与短块种子（baseM 由标定 baseBlock 注入）。

### 3.6 Schedule——六段求解流水（逐段走读）

**① Estimate：总量估计（EstimateTotalTimes，solver.h）。** 干什么：切分前先估"通信总时
vs 计算总时"，产出 bound 判定依据。从头往后五步：
1. `FindCubeUtil(commRankCntForMm)`——聚合稳态利用率，写入 cubeUtil_：
   `util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^0.7)`，
   调和平均 = (m×shares×n)/(m×shares+n)。逐项物理意义：0.75 是 shape 恰达阈值
   （K=1280、调和平均=820）时的平均利用率（锚点）；K 因子 ^0.7——K 越大固定开销摊得越
   薄、指数缓和增长、单因子封顶 1.4；调和平均**惩罚 M/N 偏斜**（m 巨大 n 极小时被短板
   拖住，比几何平均更贴合分块阵列填充率）；shares 乘进 m 因为建模对象是稳态流水的
   工作集形状；
2. `GetMatmulGradient()`——每行耗时系数（物理含义 = 每行输出的 MACs 需求 ÷ 全核每微秒的有效
   MACs 吞吐）：
   `gradient = max(baseN,n) × max(baseK,k) / (computesPerCycle × 频率 × 核数 × util)`
   （computesPerCycle = cube 每 cycle 乘加次数，标定算力档 mmFit.cubeCalc 查出，见 4.2；
   不足基本块的 n/k 按基本块计——硬件按 base 块干活）；
3. `totalMatmulTime = MatmulTime(mValue, mmRankMul) = max(baseM, m) × rowMul × gradient`
   （小于基本块的块也按 baseM 计——阵列粒度下限，防小 shape 耗时趋 0）；
4. `totalCommTimeNoCut = TotalCommTime(mValue, 1)`——**按不切口径（1 轮）估通信总量**：
   此刻轮数未知，不预支切分才会产生的轮次开销。用途：与 totalMatmulTime 比较判定短块
   位置（见下）；⑤ 的不切基准 serialTotalTime 也是同一口径。切分完成后 Schedule 尾部
   再按实际轮数精算（诊断量）。

接着**本地窗口**（仅 FRONT/BACK）：`localMMUtil = FindCubeUtil(m, 1, 0.6)`——单份数据、
流水未满，0.6 指数比稳态 0.7 更保守；`localMMTime = MatmulTime(m, 1) / localMMUtil`
（除以更低的利用率 = 更长的耗时估计）。FRONT → 存 frontMMTime 且**短块恒后置**（本地块
占启动端，短块退尾专职收尾）；BACK → 存 backMMTime 且**短块恒前置**（专职启动通信管道）。

**NONE 的短块位置由 bound 决定，本质是配对方向的选择**：短块是流水线边界配对的锚点，它与
相邻长块构成边界上的"通信 ∥ 计算"配对，长块尺寸按配平等长反解（③）。以计算在前为例：
- 计算重（mm ≥ comm）→ **短块后置**：边界配对 = 倒数第二块（长）的通信 ∥ 最后一块（短）的
  计算，③ 反解 `comm(长) = mm(短)`——通信便宜，要积够"短块计算"的时长需要**大而少**的长块，
  且流水线尾部只暴露短块的通信（极小，收尾快）；
- 通信重（comm > mm）→ **短块前置**：边界配对 = 第一块（短）的通信 ∥ 第二块（长）的计算，
  ③ 反解 `mm(长) = comm(短)`——计算便宜，要积够"短块通信"的时长需要大而少的长块，首部
  只暴露短块的计算（极小，启动快）；
- **放反则反解退化**：如计算重却前置 → `mm(长) = comm(短)`，右边趋零 → 长块被反解成极小 →
  轮数爆炸，轮税吞掉全部重叠收益。
通信在前的配对方向镜像（`mm(bᵢ) ∥ comm(bᵢ₊₁)`，短块在后配 `mm(长)=comm(短)`）。判定式：
COMPUTE_FIRST 比较 (mm ≥ comm)，COMM_FIRST 比较 (comm > mm)。

**② SetShort：短块下限三道闸 + 落位。**
1. `ExceedsL2Cache`：判定 A+B 权重工作集（m×k×aBytes + k×n×bBytes）是否超过芯片 L2 容量
   （标定 mmFit.l2CacheBytes，950 默认 128MB）；
2. `ApplyL2MinLenBoost`：超限 → minLen×2（小块会令 L2 反复换入换出，抬下限让块变大）；
3. **2b 闸门**：minLen ≥ `InverseCommTime(2b)`——短块独立占一轮通信，其通信时长须 ≥
   2b（一轮自身开销 + 至少等量的有效传输）才值得成轮。b 至此贯穿长块反解 / 总量门控 /
   短块下限三处；
4. shortTileLen = minLen 落位（numShortTile = 1）；
5. **最短可切校验（NoCut 兜底）**：切分最少要两块（1 短 + 1 长），每块长度不得低于
   下限——总长 totalLen < 下限×2 时任何合法切分都不存在，输出整块不切（NoCutTiling）直接
   返回。纯长度可行性判定，与耗时模型无关。

**③ SetLong：长块反解（配平核心）。** FRONT：`longTileLen = InverseCommTime(frontMMTime)`
——首长块通信恰好掩盖本地首块计算。BACK：`longTileLen = InverseCommTime(backMMTime)`
——末长块通信掩盖本地尾块。NONE 走 `BalanceLongTileLen`（solver.h）双向配平：按通算顺序确定边界
配对方向（见 ① 的配对说明）后，长块反解为恰好等长覆盖短块的另一侧时长——计算在前 +
短块在后 → `comm(长) = mm(短)`，其余组合镜像。
**afterLong 快照**：③ 后的切分形态整份保存——⑤ 轮数搜索的恢复点（长块反解值与轮数
无关，每档只需重跑 ④，不重算配平）。

**④ 离散化：把 ②③ 的理想配平值整形成合法的整数块方案（FitTileLengthDiscrete 七步，cut_planner.h）。**
②③ 产出的是**配平理想值**：短块长度 = 下限闸门叠加的结果，长块长度 = 时间方程的反解
（"通信时长恰好盖住某段计算"对应的那条行数线）——平衡点概念上可落在任意行数上，这个值
不带任何几何约束。而 kernel 能执行的切分是整数块方案，必须同时满足：**块长对齐** baseM
（matmul 按 base 块干活，不对齐的尾块浪费整 tile）、**块数为整数**、**总长精确守恒**
（各块行长之和恰等于 totalLen）、**总块数 ≤ 轮次预算**。离散化 = 在尽量贴近配平结构的
前提下把理想值重排成满足全部约束的整数方案（CutPlanner 是总长守恒与对齐粒度的守门人）。
入口先定 goodLinearity：有本地块的形态恒 true（行为基线）；否则 `IsGoodLinearityShape`
（k×n ≥ 32MB——线性度好的 shape 通信拟合更可信）。localBlockProtected = 有本地块
（保护端部结构）。七步依次为"对齐取块数 → 残余处理 → 超限兜底 → 尾块收守恒"：
1. `GenerateInitialPartition`：夹取 longTileLen ∈ [short, total−short] → 吸附到 256（mAlignLen）
   的倍数：余数 r = long mod 256，**r > 256×tol 才向上进位，否则向下舍**——tol 为对齐容差，
   计算重形态（短块在后）取 0.5（中性四舍五入），通信重形态（短块在前）取 0.75（消极进位：
   约 3/4 概率向下舍，块长不超出配平反解值——配平两侧时长本已相等，进位使长块超出它要
   掩盖的对侧时长，超出部分无可重叠）；→ 估算长块个数 `numLongTile = (total − short×0.75) /
   long`——短块按 75% **占位折算**（BACK_TILE_RATIO，与对齐容差的 0.75 是两个不同常数），
   使长块个数倾向多算一个，对齐误差由第 7 步的尾块重算吸收（短块是弹性块，长块保持规整）；
2. 残余判定 + balanceCut：残余 ≥ 短块 且 2×短 ≥ 长 且 块数 ≤2 且线性度好 → 索性全等分
   （等分 = 通信轮次最规整的保底形态）；
3. `RedistributeTile`：恰好两块且无本地块保护 → 残余均摊给长短两块（头轻脚重修正）；
   FRONT/BACK 跳过——均摊会破坏端部的本地块结构；
4. moreFrontTile：残余还装得下一个长块 → numLongTile++ + `CalcBackTile`（尾块重算吸收
   余数；若**重算后的尾块** ∈ (0, 半格)（半格 = mAlignLen/2 = baseM/2，默认 128 行）且
   长块数 > 1，末长块并入尾块——碎尾不足半格，独立成轮纯付轮税不划算，合并后总块数少一轮）；
5. 轮次约束：totalTileCnt > maxTileCnt 且**双条件齐备**——总长被轮次预算整除
   （totalLen ÷ maxTileCnt 无余数）且商（等长块的块长）是 mAlignLen 的倍数——才改写为
   纯等长块（maxTileCnt × tileLen，尾块清零，天然守恒且对齐）。任一不满足则落入第 6 步
   兜底（例：4096÷4=1024 整除且对齐 → 4×1024；4000÷4=1000 整除但 1000 不是 256 倍数
   → 走第 6 步）；
6. `MaxTileCntUniform`：任意超限兜底——ceil 对齐均分（少一块给尾块留位），尾块吸收余数；
7. 尾块重算 shortTileLen = 纯余数（总长守恒）。收尾对齐按长块大小分两路：
   - **长块 > 1024 行 → `AlignTileLarge`（对齐短块，长块吸收）**：短块向上对齐到 256
     倍数 → 长块 ceil 均分吃下差值 → 尾块重算回纯余数（守恒闭合）。为什么对齐短块：
     matmul 按 256 行 tile 干活、块长不对齐时末 tile 烧整格只算零头——100 行的碎短块
     浪费 61%，而大长块差零头的浪费比仅约 3%，对齐代价甩给承受得起的一方
     （例：4×512+576 → 4×464+768）；
   - **长块 ≤ 1024 行 → `AlignTileSmall`（不动对齐，只修退化结构）**：① 碎尾吸收——
     尾块 < 256 或低于下限 minTileLen 且长块 > 2 格时，每个长块各让出一格（−256）、
     尾块增长 numLong×256（长块减整格故对齐性不变，转移精确守恒）；② 长短倒挂校正——
     尾块反而大于长块时，按 total/总块数 ceil 对齐重定长块、尾块重算回余数
     （恢复"长为主短为尾"的正常结构）。
最后 `MergeEqualTiles`（solver.h）：长短相等时退化为纯等分（short 清零、numLong+1，
去掉多余的块类型标记）。

**⑤ 轮数搜索 + 切分收益门控。** 动机：③④ 配平产出的轮数 N₀ 是"配平衡最优"而非全局
最优——通信重时 T(N) 随轮数**单调增**（启动税吃收益）、计算重时**单调降**（至首块通信
≈b 触底），最优轮数常在 [1, N₀] 中间，二值判定（N₀ vs 1）会错过。实现四步：
1. `serialTotalTime = TotalCommTime(m, 1) + totalMatmulTime`（不切的全串行基准）；
2. 初次切分（N₀ 轮，配平产出）即第一个候选，`EstimateCutTotalTime` 估值；
3. budget ∈ [2, N₀) 线性枚举：恢复 afterLong → SetMaxTileCnt(budget) 收紧轮次预算 →
   重跑 ④（长短结构随预算重新整形）→ 估值，更小者胜；
4. SetMaxTileCnt 恢复 63；全部候选 ≥ serialTotalTime → NoCutTiling（启动税买不回重叠
   收益）；否则取最优候选。

`EstimateCutTotalTime`（solver.h）的流水线总时近似——不可重叠段在流水线两端，中段取
通信/计算瓶颈侧；firstLen/lastLen 按 shortTileAtBack 取首末块：
```
FRONT:  T = T_comm(首块) + max(T_comm(其余轮), T_mm − T_mm(末块)) + T_mm(末块)
BACK:   T = T_mm(首块) + max(T_mm − T_mm(首块) − 本地窗口, T_comm(除末轮)) + max(T_comm(末轮), 本地窗口)
NONE(计算在前): T = T_mm(首块) + max(T_mm − T_mm(首块), T_comm(除末轮)) + T_comm(末轮)
NONE(通信在前): T = T_comm(首块) + max(T_comm − T_comm(首块), T_mm(除末块)) + T_mm(末块)
单块(含 NoCut 候选) → 返回 serialTotalTime
```
成本：N₀ ≤ 63 档 × 每档纯整数重切 O(块数) = host 侧微秒级。

**⑥ CCU 单轮上限整形（ApplySingleRoundCommLimit，仅 commEngine==CCU）。** HCCL 单次
调用有 256MB 上限：单行通信字节 `rowBytes = commMatrixLen × dtypeSize × volumeMul /
fraction`（物理字节数——须含份数，与 k 的标定口径融合无关）；longTileLen×rowBytes
未超限 → 保持配平结构直接返回；超限 → 候选表 {2048,1024,512,256} 降序取首个 ≤ 上限的
（大块优先 = 少轮次），候选全超限则 floor 对齐兜底；轮次预算 = 63−1（1 轮留给 scale
通道），预算不够则反向取满足预算的最小块；**两约束冲突放弃整形**（保持原切分）；
否则等长重切（主块等长 + 尾块吸收余数，短块位置保持）。

**尾部精算。** `commTimeActualUs = TotalCommTime(m, N_实际)`、`commTimeNoCutUs =
TotalCommTime(m, 1)`——差值 = (N−1)·b 启动税，供 ⑦ 诊断打印。

### 3.7 出口：诊断打印（PrintTilingData）

`ASCEND_GLOBAL_LOG_LEVEL=0` 时输出三段：**输入 shape**（m/n/k/rankDim）→ **最终切分**
（六字段）→ **通信时**（实际 N 轮 vs 不切 1 轮 + 启动税 (N−1)·b）。纯 printf、零日志库
依赖，FitBalanceCut 自动调用，op 侧无需再写日志。

### 3.8 方法-消费方速查

| 方法 | 消费方 |
|---|---|
| FindCommFitRow / IsCommCalibrated | InitCtx 第 2 步选行 / op 分派 |
| CommTime / InverseCommTime | ③ 配平反解 / ② 2b 闸门 / ⑤ 首末块 |
| TotalCommTime | ① Estimate / ⑤ 搜索与门控 / 尾部精算 |
| FindCubeUtil(0.7) / GetMatmulGradient / MatmulTime / InverseMatmulTime | ① Estimate / ③ 配平 / ⑤ 门控 |
| FindCubeUtil(m, 1, 0.6) | ① 本地窗口（FRONT/BACK） |
| ExceedsL2Cache / ApplyL2MinLenBoost | ② 短块下限闸 |
| BalanceLongTileLen | ③（NONE 形态） |
| IsGoodLinearityShape / CutPlanner 离散方法 | ④ / ⑤ 搜索重跑 |
| MergeEqualTiles | ④ 后处理 |
| EstimateCutTotalTime | ⑤ 搜索/门控估值器 |
| ApplySingleRoundCommLimit | ⑥（CCU 键） |
| MakeFitBalanceArgs / CutResultFromCof | 入口族：存量算子转换 / miss 分派（不在主流程） |

---

## 四、标定数据维护指南

### 4.1 通信拟合行（k/b/c）——新形态（如 8P）怎么加

**行结构**：`{通信引擎, 卡数, k, b, c}`，按（通信引擎 + 卡数）二维匹配——引擎维度在行上，
新增引擎（如 AIV_URMA）不改结构体、加行即可。

**加 8P 行的完整步骤**：
1. **实测**（CommBench 工具，AG/aicpu/int8/rank8 配置）：
   - k：turn=1 尺寸扫描（8~96MB/rank）线性拟合，**直接取工具输出的斜率原值**——引擎算的就是
     单卡数据量的 MB，**不要**再做 ÷rankDim 之类的预换算（多份数的放大已折进引擎使用 k 的方式）；
   - b：固定 chunk 扫轮数 T(N)=N(k·chunk+b)+c 拟合，取多组 chunk 交叉一致的中位值
     （16MB 以上大块带宽亚线性，拟合值偏离时应弃用）；
   - c：同拟合截距，多组取中位。
2. **填行**：在 commRows 里追加 `{TilingCommEngine::AICPU, 8, k实测, b实测, c实测}`——
   一行即一个已标定形态，与 4P 行并存，运行时按（引擎, 卡数）自动匹配。
3. **生效路径**：填入后 `IsCommCalibrated(8, ..., AICPU)` 返回 true，8P 调用即可走 fit-balance
   （此前算子在 miss 下自选的策略不再触发）。

**未标定形态的处理**：commRows 里**不写该（引擎, 卡数）组合的行**即可——IsCommCalibrated
返回 false，由算子侧自选策略处理，无需任何开关。**不要用零值行占位**（零斜率行按未填处理同样不命中；
塞占位值上线则是以假数据进 fit-balance，切分决策失真——历史上 CCU 曾以换算默认值占位，已删除）。

**CCU 引擎**：行结构与 aicpu 完全相同但数值独立——CCU 每轮开销与带宽特性与 AICPU 不同，
必须分别标定，不可复用 aicpu 行的数值。

### 4.2 算力档（mmFit.cubeCalc）——新 dtype 怎么加

按 (calcType, isFp4, A/B 字节数) 四维键匹配，加行即加档：
- fp8：`{MatmulDtypeKind::QUANT, false, 1, 1, 算力}`；
- fp4：`{MatmulDtypeKind::QUANT, true, 0, 0, 算力}`——键的 {0,0} 是现网真实值
  （算子侧获取字节数的 GetDataTypeSize 函数没有 fp4 条目，返回 0），不是笔误；
- bf16：`{MatmulDtypeKind::FP16, false, 2, 2, 算力}`；
- miss 落 `defaultComputesPerCycle`（引擎默认 4096，勿用作长期方案）。

### 4.3 何时不需要动标定

- **换 shape**：不需要——标定行按卡数与 dtype 匹配，与 shape 无关；
- **换通信引擎配置**（comm_mode）：不需要——运行时按 commEngine 选表，两表都已标定即可；
- **新算子接入**：dataflow 三元组按算子填，**通信行/算力档可以直接复制同原语已标定算子的
  数值**（标定的是通信与芯片特性，不是算子特性）——这正是标定层与算子解耦的设计收益。

### 4.4 新芯片接入清单（如 960）

引擎不含任何芯片分支——芯片差异全部终结在标定层。新芯片接入按下表逐项处理：

| 项 | 动作 |
|---|---|
| `mmFit.cyclePerMicroSec` / `mmFit.l2CacheBytes` | 按芯片规格填（默认值是 950 的 1689.6 cycle/us / 128MB） |
| `mmFit.cubeCalc` 算力行 | 芯片上实测各 dtype 的 MACs/cycle 后填行 |
| `mmFit.baseBlock` | 芯片上实测最优基本块 |
| `commRows` 通信行 | 芯片平台上重做通信拟合（k/b/c，按引擎×卡数） |
| 利用率公式系数（0.75 锚点 / 820 / 1280 阈值 / 0.7、0.6 指数，见 3.6 ①） | **接入时用实测复验**——这些系数按 950 cube 行为拟合；若 960 实测决策偏差大，将系数外提进标定（独立演进，勿盲搬 950 值） |
| HCCL 单轮 256MB 上限 / 63 轮预算 | 通信 API 层限制，接入时确认新芯片是否变化 |

**红线**：不要在引擎内加 `if (芯片)` 分支——所有芯片差异走标定或（利用率公式这类形态级差异）独立 CR 外提参数。

**一个算子同时支持多芯片**：芯片维度在算子侧，不在引擎，两条路径任选——
- **按仓内惯例分目录**：各芯片 tiling 分目录（`op_tiling/arch35/`、`arch36/`…），各自持有
  本芯片的 CALIBRATION 实例与注册，共用同一引擎；
- **同类内运行时分派**：算子侧准备多份 CALIBRATION，按平台检测结果选用（tiling 时可经
  `PlatformAscendC(platformInfo).GetCurNpuArch()` 拿到芯片）——标定本来就是 FitBalanceCut
  的普通入参，无需任何引擎改动：

  ```cpp
  static const FitBalanceCalibration CALIB_950 = {...};
  static const FitBalanceCalibration CALIB_960 = {...};
  const auto &calib = IsNewChip(npuArch) ? CALIB_960 : CALIB_950;
  cut = FitBalanceCut(args, calib);
  ```
