# fit-balance tiling 引擎——接入指南

> 通信-计算流水线配平切分引擎（header-only, 950 标卡 4P/8P, AICPU/CCU 通信引擎）。
> 本文档面向接入本引擎的算子开发者：接口、示例与建模公式。

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

单入参 + 标定。内部串联 上下文构造（InitCtx）→ 键控求解（Schedule）。
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
          {MatmulDtypeKind::QUANT, true, 0, 0, 16384} }, // fp4 打包(键 {0,0} 是现网真实值——字节数查询对 fp4 返回 0)
        {256, 256, 128},                                  // baseBlock
        // 平台规格可不填(默认 950: 频率 1689.6 cycle/us); 新芯片按规格覆盖
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

### 建模公式——标定数据参与的计算

**通信模型**（commRows 的 k/b/c）：

```
通信耗时   y = k·x + 轮数×b + c        x = 通信数据量（MB）= 行数 × 通信轴长 × 每元素字节数
```

k = 带宽斜率（us/MB），b = 每轮启动开销（us，切几轮付几份），c = 一次性固定开销（us，
不随轮数放大）。通信轴长取 K 还是 N 由 dataflow.order 决定（通信在前传 A 侧 → K）。

**计算模型**（mmFit 的算力档 + 平台规格）：

```
每行耗时  gradient = max(baseN,n) × max(baseK,k) / (computesPerCycle × 频率 × 核数 × util)
块耗时    = 块行数 × 行数倍数 × gradient        行数倍数由 dataflow 形态推导（全量=rankDim）
利用率    util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^0.7)
          调和平均 = (m × 工作集份数 × n) / (m × 工作集份数 + n)      本地窗口口径指数取 0.6（更保守）
```

computesPerCycle 即 cubeCalc 算力档（MACs/cycle）；频率为 mmFit 平台规格；核数为运行时
入参 aicCoreNum。引擎用这两个模型做流水线配平（长块的通信时长盖住相邻块的计算时长）与切分收益
判定（轮次启动开销 vs 重叠收益）——**标定数据失真会直接反映为切分决策失真**。

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
