# fit-balance tiling 引擎——接入指南

> 通信-计算流水线配平切分引擎（header-only, 950 标卡 4P/8P, AICPU/CCU 通信引擎）。
> 本文档面向接入本引擎的算子开发者：接口与示例 5 分钟接入，标定指南覆盖日常维护。

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
          {MatmulDtypeKind::QUANT, true, 0, 0, 16384} }, // fp4 打包(键 {0,0} 是现网真实值, 见 3.2)
        {256, 256, 128},                                  // baseBlock
        // 平台规格可不填(默认 950: 频率 1689.6 cycle/us, L2 容量 128MB);
        // 新芯片按规格覆盖: cyclePerMicroSec = ..., l2CacheBytes = ... (见 3.4)
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
static const FitBalanceCalibration CALIB_NEW = {...};    // 频率/L2/算力档/通信行全按新芯片
// DoTiling 内: PlatformAscendC platform(context_->GetPlatformInfo()); 按其结果选芯片
const FitBalanceCalibration &calib = IsNewChip(platform) ? CALIB_NEW : CALIB_950;
if (!IsCommCalibrated(rankDim_, commEngine, calib)) { /* miss → 其他切分策略 */ }
cut = FitBalanceCut(args, calib);
```

完整接入清单（新芯片要改哪些数、复验哪些公式）见 3.4。

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

## 三、标定数据维护指南

### 3.1 通信拟合行（k/b/c）——新形态（如 8P）怎么加

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

### 3.2 算力档（mmFit.cubeCalc）——新 dtype 怎么加

按 (calcType, isFp4, A/B 字节数) 四维键匹配，加行即加档：
- fp8：`{MatmulDtypeKind::QUANT, false, 1, 1, 算力}`；
- fp4：`{MatmulDtypeKind::QUANT, true, 0, 0, 算力}`——键的 {0,0} 是现网真实值
  （算子侧获取字节数的 GetDataTypeSize 函数没有 fp4 条目，返回 0），不是笔误；
- bf16：`{MatmulDtypeKind::FP16, false, 2, 2, 算力}`；
- miss 落 `defaultComputesPerCycle`（引擎默认 4096，勿用作长期方案）。

### 3.3 何时不需要动标定

- **换 shape**：不需要——标定行按卡数与 dtype 匹配，与 shape 无关；
- **换通信引擎配置**（comm_mode）：不需要——运行时按 commEngine 选表，两表都已标定即可；
- **新算子接入**：dataflow 三元组按算子填，**通信行/算力档可以直接复制同原语已标定算子的
  数值**（标定的是通信与芯片特性，不是算子特性）——这正是标定层与算子解耦的设计收益。

### 3.4 新芯片接入清单

引擎不含任何芯片分支——芯片差异全部终结在标定层。新芯片接入按下表逐项处理：

| 项 | 动作 |
|---|---|
| `mmFit.cyclePerMicroSec` / `mmFit.l2CacheBytes` | 按芯片规格填（默认值是 950 的 1689.6 cycle/us / 128MB） |
| `mmFit.cubeCalc` 算力行 | 芯片上实测各 dtype 的 MACs/cycle 后填行 |
| `mmFit.baseBlock` | 芯片上实测最优基本块 |
| `commRows` 通信行 | 芯片平台上重做通信拟合（k/b/c，按引擎×卡数） |
| 利用率公式系数（0.75 锚点 / 820 / 1280 阈值 / 0.7、0.6 指数） | **接入时用实测复验**——这些系数按 950 cube 行为拟合；若新芯片实测决策偏差大，将系数外提进标定（独立演进，勿盲搬 950 值） |
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
  static const FitBalanceCalibration CALIB_NEW = {...};   // 频率/L2/算力档/通信行全按新芯片
  const auto &calib = IsNewChip(npuArch) ? CALIB_NEW : CALIB_950;
  cut = FitBalanceCut(args, calib);
  ```
