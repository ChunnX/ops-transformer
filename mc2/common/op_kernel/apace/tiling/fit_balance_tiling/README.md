# fit-balance tiling 引擎——详细设计与接入指南

> 通信-计算流水线配平切分引擎（header-only, 950 标卡 4P/8P, AICPU/CCU 通信引擎）。
> 本文档面向接入本引擎的算子开发者：先看接口与示例（5 分钟接入），需要深究行为时再看详细设计。

---

## 一、对外接口

全部符号为 global namespace，include 伞形头即可用：

```cpp
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"
```

### 主入口

```cpp
CommCutResult FitBalanceCut(const FitBalanceTilingArgs &args, LocalMMPosition localMMPosition,
                            CommComputeOrder order, TilingCommEngine commEngine,
                            const FitBalanceCalibration &calib);
```

一行完成切分：内部串联 上下文构造（InitCtx）→ 键控求解（Schedule）→ 诊断打印（debug 级）。
返回"1 短 + N 长"的 M 轴切分方案（或退化的等分/单块）。

### 分派判定

```cpp
bool IsCommCalibrated(uint32_t rankDim, TilingCommEngine commEngine, const FitBalanceCalibration &calib);
```

通信标定表是否命中当前形态（引擎分区 + rankDim 匹配）。算子用它做切分策略分派：
**命中 → FitBalanceCut；miss → 核利用率优先（core_occupancy_first_tiling.h）等其他策略**。

```cpp
TilingCommEngine ParseCommEngine(const char *commMode);   // "ai_cpu"→AICPU, "ccu"→CCU, 其余→AICPU
```

### 进阶（一般不需要直接用）

```cpp
FitBasedBalanceCtx InitCtx(args, localMMPosition, order, commEngine, calib);  // 上下文构造
CommCutResult Schedule(ctx);                                                  // 键控求解
void PrintTilingData(args, cut, commTimeActualUs, commTimeNoCutUs);          // 诊断打印(通常由 FitBalanceCut 自动调)
const FitBalanceCalibration::CommFit::Row *FindCommFitRow(calib, engine, rankDim); // 标定行查找
```

### 核心类型

| 类型 | 说明 |
|---|---|
| `FitBalanceTilingArgs` | 引擎调度入参（9 字段：m/n/k、rankDim、aicCoreNum、dtype 四元组） |
| `CommCutResult` | 切分结果（shortTileAtBack / totalTileCnt / numLongTile / numShortTile / longTileLen / shortTileLen） |
| `FitBalanceCalibration` | 客户标定数据总集（CommProfile 结构参数 + 通信拟合行 k/b/c + matmul 算力档），算子侧唯一编辑面 |
| `LocalMMPosition` | 流水线结构键控：FRONT（本地 mm 块前置）/ BACK（后置）/ NONE（无本地块） |
| `CommComputeOrder` | 通算顺序：COMM_FIRST（通信在前）/ COMPUTE_FIRST（计算在前） |
| `TilingCommEngine` | 通信引擎：AICPU / CCU / AIV_URMA（占位） |

---

## 二、接入示例（新算子视角）

新算子直接使用引擎的 args/cut 类型，无需任何旧类型转换：

```cpp
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"
#include "apace/tiling/core_occupancy_first_tiling.h"   // miss 路径(miss→核利用率优先)

// ── ① 标定数据: 每算子一份, 客户唯一编辑面 ─────────────────────────────────
// profile 按本算子流水线结构填写; rows 填实测拟合值; mmFit 按 dtype 档填
static const FitBalanceCalibration CALIBRATION = {
    /* profile: 通信结构参数 */
    {
        CommAxis::K,                 // 通信承载 A 侧输入(沿 K 轴聚散, 传 A 矩阵行)
        RankMul::RANK_DIM_MINUS_ONE, // mm 工作集倍数(远程 rankDim−1 份参与计算)
        RankMul::RANK_DIM,           // mm 总行数倍数(聚合后算全量 m×rankDim 行)
        RankMul::RANK_DIM,           // 单次通信调用物理字节倍数(256MB 上限判定用)
    },
    /* aicpu 引擎通信拟合行: {卡数, k 带宽斜率(us/MB), b 每轮启动开销(us), c 一次性固定开销(us)} */
    {
        { {4, 26.5894, 33.17, 20.58} },   // 950 标卡 4P 实测; 未标定形态留空 → 分派走 miss 路径
    },
    /* ccu 引擎通信拟合行 */
    {
        { {4, 23.367368, 8.590666, 0.0} },
    },
    /* mmFit: matmul 算力档 + 最优基本块 */
    {
        { {MatmulDtypeKind::QUANT, false, 1, 1, 8192},   // fp8 量化
          {MatmulDtypeKind::QUANT, true, 0, 0, 16384} }, // fp4 打包(键 {0,0} 为 glue 真实口径)
        {256, 256, 128},                                  // baseBlock
    },
};

// ── ② tiling 主流程: 分派 + 一行调用 ─────────────────────────────────────
ge::graphStatus MyOpTiling::DoTiling()
{
    const auto commEngine = ParseCommEngine(commMode_);   // "ai_cpu"/"ccu" → 枚举

    CommCutResult cut;
    if (IsCommCalibrated(rankDim_, commEngine, CALIBRATION)) {
        // 命中: 直接构造引擎入参(9 字段)并调用
        FitBalanceTilingArgs args = {mValue_, nValue_, kValue_, rankDim_, aicCoreNum_,
                                     aType_, geAType_, inputDtypeSize_, outputDtypeSize_};
        cut = FitBalanceCut(args, LocalMMPosition::FRONT,     // 本地 A 块前置(输入侧通信算子)
                            CommComputeOrder::COMM_FIRST, commEngine, CALIBRATION);
    } else {
        // miss: 未标定形态, 走核利用率优先等策略(本例略——见 core_occupancy_first_tiling.h)
        cut = GetCutByCoreOccupancyFirst();
    }
    // 直接消费 CommCutResult: kernel 侧按 shortTileAtBack 指示的顺序执行 1 短 + N 长
    UseCutResult(cut);
    return ge::GRAPH_SUCCESS;
}
```

接入要素就两个：**填 CALIBRATION**（标定数据）+ **选两个枚举**（LocalMMPosition / CommComputeOrder）。

### 为什么引擎不认"通信原语"、而要你传这两个枚举？

引擎被设计为**不区分通信原语**（AllGather/ReduceScatter/AllReduce/AllToAll……）——
不同原语的差异只有两类：**可标定的数值差异**（通信量口径、轴向、算力）全部收进
CALIBRATION 的 CommProfile 与拟合行；而**流水线结构差异**无法用数值表达，必须显式声明，
这就是两个枚举的职责：

| 枚举 | 回答的问题 | 取值含义 | 选错会怎样 |
|---|---|---|---|
| `LocalMMPosition` | 流水线里有没有一块**不经通信的本地 matmul**？在哪端？ | `FRONT` 本地块前置（首端，如输入侧通信的聚合算子——本卡数据已在手，聚合前即可算）；`BACK` 本地块后置（尾端，如输出侧通信的分发算子——本地保留份无需发出）；`NONE` 无本地块 | 配平锚点错位（长块反解目标错）、短块位置错、⑤ 的总时公式形态错——切分结果整体劣化 |
| `CommComputeOrder` | 流水线**哪段先行**？ | `COMM_FIRST` 通信在前（先收数再算）；`COMPUTE_FIRST` 计算在前（先算再发） | 短块放端与不可重叠段位置判断颠倒 |

判定方法（看本算子的数据流即可，不需要理解引擎内部）：
1. 通信的输入是不是 matmul 的输入？是 → 通信在前（`COMM_FIRST`）+ 本地块在首端有机会先行（→
   试想本卡自己的数据要不要经过通信才能参与第一次计算：不用 → `FRONT`；要 → `NONE`）；
2. 通信的输入是 matmul 的输出？→ 计算在前（`COMPUTE_FIRST`）+ 想想本卡算完的份额要不要全部
   发出去：有保留在本地的一份 → `BACK`；全发 → `NONE`。

诊断打印内置：`ASCEND_GLOBAL_LOG_LEVEL=0` 时 FitBalanceCut 输出输入 shape / 最终切分 / 启动税三段。

---

## 三、详细设计

### 3.1 核心思想：通信-计算流水线配平

通算融合算子的执行是流水线：通信块与计算块交替重叠。切分要回答两个问题——
**块怎么切**（长短结构）与**切几轮**（轮数经济性）。

本引擎的模型：把 M 轴切成 1 短 + N 长，长块的通信时长恰好掩盖相邻块的计算时长（配平）；
每多切一轮多付一份启动开销 b，用轮数搜索择优。

### 3.2 通信模型：y = k·x + N·b + c

| 口径 | 公式 | 用途 |
|---|---|---|
| 单轮 | `k·x + b` | 配平反解的语义单位 |
| 反解 | `x = (t − b) / k` | 长块长度 = 反解"要掩盖的时长" |
| 总量 | `k·X + N·b + c` | 轮数搜索/门控的估值器（N 轮 vs 不切之差 = (N−1)·b 启动税） |

k 一律按"引擎 x 口径（m × 通信矩阵另一维 × 有效字节数的 MB）"标定——通信份数口径已融进 k
（AG 行直接填工具原始值，无需 ÷rankDim 预换算）。b=每轮启动开销（单轮全款），c=一次性固定开销。

### 3.3 计算模型

```
util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^mnExp)
gradient = max(baseN,n) × max(baseK,k) / (算力 × 频率 × 核数 × util)
MatmulTime(块长, rowMul) = max(baseM, 块长) × rowMul × gradient
```

mnExp 双口径：**0.7 聚合稳态**（数据齐/流水满，Estimate 用）；**0.6 本地窗口**（单份数据/流水未满，
FRONT/BACK 本地块时长估计用——更保守）。

### 3.4 键控维度（引擎不区分通信原语）

原语的语义差异全部终结在标定层与两个结构枚举：

| 差异 | 承载 |
|---|---|
| 通信份数/字节口径 | 标定斜率 k + CommProfile.commVolumeMul |
| 通信轴向（K/N）与承载侧（A/C dtype） | CommProfile.axis |
| mm 工作集/总行数倍数 | CommProfile.mmWorksetMul / mmTotalMul |
| **流水线结构**（本地块在哪端/有无） | **LocalMMPosition**（引擎唯一结构键控） |
| 通算顺序 | CommComputeOrder |
| 通信引擎 | TilingCommEngine（CCU 触发单轮 256MB 整形） |

`LocalMMPosition` 三态的物理含义：

```
FRONT: 本地 mm 块在流水线首端——启动端被本地块免费占据(零通信依赖),
       短块退到尾部专职快速收尾; 长块 = 反解本地块计算时长(首块通信恰好掩盖它)
BACK:  本地 mm 块在流水线尾端——本地保留份无通信 deadline, 与末块通信重叠,
       短块前置专职启动通信管道; 长块 = 反解本地块计算时长(末块通信恰好掩盖它)
NONE:  无本地块——短块作流水线垫片(按 bound 判定放首/尾),
       长块 = 反解短块的通信/计算时长(双向配平)
```

### 3.5 求解流水（Schedule 六段）

```
① Estimate     总量估计(通算比/bound 判定); FRONT/BACK 算本地窗口时长
② SetShort     短块 = baseM 落位; NoCut 几何门(总长摆不下两块 → 不切)
③ SetLong      长块反解: FRONT/BACK 反解本地窗口时长; NONE 双向配平反解短块
④ 离散化       长短种子 → 合法整数块方案(块数/对齐/尾块/等分判定, CutPlanner 七步)
⑤ 轮数搜索     budget ∈ [2, N₀) 线性枚举: 收紧轮次预算重跑④, 按"流水线总时
   +门控       (含 N·b)"择优; 全部候选不敌全串行 → NoCut
⑥ CCU 整形     单轮通信字节超 256MB → 候选表等长重切(仅 CCU)
```

决策层次：**③④ 管块怎么切（配平），⑤ 管切几轮（启动税 vs 重叠收益），⑥ 管硬件上限**。

⑤ 的流水线总时近似（EstimateCutTotalTime）按 LocalMMPosition/顺序分三形态，
不可重叠段在流水线两端（通信在前的首块通信、计算在前的末块通信、BACK 的本地窗口归末段）。

---

## 四、文件功能介绍

```
fit_balance_tiling/
├── README.md                       本文档
├── fit_balance_tiling.h            对外总接口(伞形头): 引擎文档 + include 全部分文件
│                                   + InitCtx/Schedule/FitBalanceCut/IsCommCalibrated/FindCommFitRow
└── fit_balance_tiling_utils/       内部分文件(消费方经伞形头间接使用, 勿直接 include)
    ├── fit_balance_types.h         类型层: 键控枚举(LocalMMPosition/CommComputeOrder/
    │                               TilingCommEngine/CommAxis/RankMul) + ParseCommEngine +
    │                               数据结构(args/cut/模型画像) + FitBalanceCalibration 标定总集
    ├── comm_model.h                通信模型 HCCLCommModel: y=k·x+N·b+c 三口径
    │                               (CommTime 单轮 / InverseCommTime 反解 / TotalCommTime 总量)
    ├── matmul_model.h              计算模型 MatmulPerfModel: cube 利用率(0.7/0.6 双口径)/
    │                               每行梯度/MatmulTime 正反解
    ├── cut_planner.h               离散切分引擎 CutPlanner: "1 短 + N 长"的初切/离散化七步/
    │                               尾对齐/NoCut——总长守恒与对齐粒度的守门人
    └── solver.h                    求解层: FitBasedBalanceCtx(调度上下文) + 公共逻辑片段
                                    (EstimateTotalTimes/BalanceLongTileLen/EstimateCutTotalTime/
                                    ApplySingleRoundCommLimit 等, 键控分支按需复用)
```

依赖方向单向：`types ← {comm_model, matmul_model, cut_planner} ← solver ← 伞形头`。

### 关联引擎

`apace/tiling/core_occupancy_first_tiling.h`——核利用率优先切分（纯几何，通信跟随 matmul
最优分核）：作为标定 miss 路径与本引擎互补，算子分派二选一。

---

## 五、标定指南（摘要）

- **通信行**：CommBench 工具实测；k 按引擎 x 口径（单卡份 MB）填工具原始值；b 用固定 chunk
  扫轮数 T(N)=N(k·chunk+b)+c 拟合；c 为拟合截距。未标定形态留空（分派自动走 miss 路径）。
- **算力档**：按 (calcType, isFp4, A/B 字节数) 四维键匹配，miss 落 defaultComputesPerCycle。
  fp4 行的键为 {0,0}（GetDataTypeSize 对 fp4 无条目返回 0，glue 真实口径）。
- **CommProfile**：按算子流水线结构填四个参数（见接入示例注释），属结构事实而非实测值。
