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
CommCutResult FitBalanceCut(const FitBalanceTilingArgs &args, const FitBalanceCalibration &calib);
```

单入参 + 标定。内部串联 上下文构造（InitCtx）→ 键控求解（Schedule）→ 诊断打印（debug 级）。
返回"1 短 + N 长"的 M 轴切分方案（或退化的等分/单块）。
通信引擎（AICPU/CCU）经 `args.commEngine` 传入——op 侧 `ParseCommEngine(comm_mode)` 解析。

```cpp
template <typename LegacyArgs>
FitBalanceTilingArgs MakeFitBalanceArgs(const LegacyArgs &a, TilingCommEngine commEngine);
```

存量算子过渡用：从旧全局 mc2tiling::TilingArgs 挑字段构造引擎入参（模板鸭子类型，引擎不依赖
旧类型定义）；commEngine 由调用方解析后补入。**新算子直接构造 FitBalanceTilingArgs，不用本函数**。

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
CommCutResult CutResultFromCof(const CommTilingData &data);                    // COF 结果(head/tail)→
                                                                               // 引擎切分(长/短), miss 分派用
```

### 核心类型

| 类型 | 说明 |
|---|---|
| `FitBalanceTilingArgs` | 引擎调度入参（10 字段：m/n/k、rankDim、aicCoreNum、dtype 四元组、commEngine）——纯运行时事实 |
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
    /* dataflow: 数据流形态(接入时定死的结构声明, 引擎据此推导 axis 与全部倍数) */
    {
        CommComputeOrder::COMM_FIRST, // 通信在前: gather 远程输入再算 → 引擎推导 K 轴/A 侧承载
        LocalMMPosition::FRONT,       // 本卡 A 块已在手, 聚合前即可算 → 引擎推导工作集 rankDim−1
        true,                         // 全量形态: 实算 mm 行数与单次通信字节均为切分域×rankDim
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
        // 命中: 构造引擎入参并调用(存量算子也可用 MakeFitBalanceArgs 从旧 TilingArgs 转换)
        FitBalanceTilingArgs args = {mValue_, nValue_, kValue_, rankDim_, aicCoreNum_,
                                     aType_, geAType_, inputDtypeSize_, outputDtypeSize_,
                                     ParseCommEngine(commMode_)};
        cut = FitBalanceCut(args, CALIBRATION);
    } else {
        // miss: 未标定形态, 走核利用率优先等策略(本例略——见 core_occupancy_first_tiling.h)
        cut = GetCutByCoreOccupancyFirst();
    }
    // 直接消费 CommCutResult: kernel 侧按 shortTileAtBack 指示的顺序执行 1 短 + N 长
    UseCutResult(cut);
    return ge::GRAPH_SUCCESS;
}
```

接入要素就一个：**填 CALIBRATION**——结构声明（dataflow 三元组）与实测数据（拟合行/算力档）
都在这一处；调用侧构造 `FitBalanceTilingArgs`（存量算子经 `MakeFitBalanceArgs` 转换）后单入参调用。

### 数据流形态（CALIBRATION.dataflow 三元组）

算子接入时定死的**结构声明**，引擎据此推导其余全部结构量：

| 字段 | 回答的问题 | 取值含义 |
|---|---|---|
| `order` | 流水线**哪段先行**？ | `COMM_FIRST` 通信在前（先收输入再算）；`COMPUTE_FIRST` 计算在前（先算再发输出） |
| `localMMPosition` | 流水线里有没有一块**不经通信的本地 matmul**？在哪端？ | `FRONT` 本地块前置（首端——本卡数据已在手，聚合前即可算）；`BACK` 本地块后置（尾端——本地保留份无需发出）；`NONE` 无本地块 |
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

判定方法（看本算子的数据流即可）：
1. 通信的输入是不是 matmul 的输入？是 → `COMM_FIRST`；通信的输入是 matmul 的输出 → `COMPUTE_FIRST`
2. 本卡自己的数据要不要经过通信才能参与第一次计算？不用 → `FRONT`；算完的份额有保留在本地的一份
   → `BACK`；都不是 → `NONE`
3. mValue 传进引擎的是"本卡份"（引擎需要 ×rankDim 才是实算量）→ `true`；mValue 本身就是本卡
   实算全量 → `false`

> 口径备注：A2A 系老实现的工作集口径为 rankDim，与单份形态（fullVolume=false）的推导值 1 不同——
> A2A 系当前无引擎消费者，BACK 形态实测校准时定稿。

诊断打印内置：`ASCEND_GLOBAL_LOG_LEVEL=0` 时 FitBalanceCut 输出输入 shape / 最终切分 / 启动税三段。

---

## 三、详细设计

### 3.1 核心思想：通信-计算流水线配平

通算融合算子的执行是流水线：通信块与计算块交替重叠。切分要回答两个问题——
**块怎么切**（长短结构）与**切几轮**（轮数经济性）。

本引擎的模型：把 M 轴切成 1 短 + N 长，长块的通信时长恰好掩盖相邻块的计算时长（配平）；
每多切一轮多付一份启动开销 b，用轮数搜索择优。

配平的**锚点 = 流水线的垫片块**（先动起来打破串行依赖的那块）：
有本地块的算子垫片是本地块（FRONT 首端 / BACK 尾端，比短块更好——零通信依赖或无 deadline）；
无本地块的算子垫片是短块（通信 bound 前置先发、计算 bound 后置垫尾）。长块长度
= 反解垫片块的时长——这是 FRONT/BACK/NONE 三态配平公式的统一原理。

### 3.2 通信模型（comm_model.h）

```
x 的口径: m × commMatrixLen × realDtypeSize 的 MB(单份口径——份数已融进斜率 k)
realDtypeSize = commDtypeSize / expansionFraction   // fp4 子字节: 1/2 = 0.5 B/元素

单轮 CommTime(m)          = k·x + b
反解 InverseCommTime(t)   = (t − b) / k            // b 整额扣除, 换算回 M 行数
总量 TotalCommTime(m, N)  = k·X + N·b + c
```

- **k**（带宽斜率）：按引擎 x 口径拟合，通信份数口径已融进 k——聚合类算子标定行直接填
  工具原始值（如 26.5894），无需 ÷rankDim 预换算；
- **b**（每轮启动开销）：单轮全款——反解时从目标时间整额扣除，配平天然感知它；
- **c**（一次性固定开销）：只进总量口径，不进单轮/反解（整个算子只付一次）；
- **消费链**：单轮→配平反解（③）与 ⑤ 门控的首末块；总量→Estimate 通算比、⑤ 轮数搜索、
  诊断打印（(N−1)·b = 启动税）；
- **坑**：fp4 子字节建模只在 K 轴承载（A 侧输入）时需要——C 侧（计算输出）永远不是 fp4。

### 3.3 计算模型（matmul_model.h）

```
util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^mnExp)
调和平均 = (m × worksetShares × n) / (m × worksetShares + n)
gradient = max(baseN,n) × max(baseK,k) / (computesPerCycle × 频率 × 核数 × util)
MatmulTime(块长, rowMul) = max(baseM, 块长) × rowMul × gradient
InverseMatmulTime(t, rowMul) = t / (rowMul × gradient), 抬到 baseM 下限
```

逐项物理意义：
- **基准 0.75**：shape 恰好达到阈值（K=1280、调和平均=820）时的平均利用率（两因子=1 的锚点）；
- **K 因子 `^0.7`**：K 越大每元素 MAC 越多，装载/流水填充等固定开销摊得越薄；指数缓和增长
  （线性太陡），单因子封顶 1.4；
- **MN 调和平均**：**惩罚偏斜**——M/N 都够大才能铺满 baseM×baseN 的 cube 分块；m 巨大 n 极小
  时被短板拖住。比几何平均更贴合分块阵列的实际填充率；`worksetShares` 乘进 m 是因为利用率
  建模对象是稳态流水的工作集形状（不是单卡裸 m）；
- **mnExp 双口径**：0.7 聚合稳态（数据齐/流水满，Estimate 用）；**0.6 本地窗口**（单份数据/
  流水未满，增长更平缓 → FRONT/BACK 本地块时长估计更保守）；
- **baseM/N/K 下限**（max）：小于基本块的 tile 也按基本块耗时计（阵列粒度）——防小 shape
  耗时趋于 0；算力 `computesPerCycle` 按 (量化类别, isFp4, A/B 字节数) 四维键查标定档，
  miss 落 defaultComputesPerCycle。

### 3.4 数据流形态推导（InitCtx，calibration.h 的 dataflow 栏）

```
commOnKAxis    = (order == COMM_FIRST)          // 通信在前=传输入 A 侧→K 轴; 在后=传输出 C 侧→N 轴
volumeMul      = fullVolume ? rankDim : 1        // 全量形态: 实算 mm 行数与单次通信字节均为切分域×rankDim
mmRowMul       = volumeMul                        // mm 总行数倍数(MatmulTime 的 rowMul)
commVolumeMul  = volumeMul                        // ⑥ 的 256MB 物理字节判定用(与 k 的口径融合无关)
worksetShares  = (localMMPosition == FRONT) ? rankDim−1 : volumeMul
                                                  // FRONT: 本地份剔出稳态工作集(单独按单份建模)
```

`LocalMMPosition` 三态的物理含义：

```
FRONT: 本地 mm 块在流水线首端——启动端被本地块免费占据(零通信依赖),
       短块退到尾部专职快速收尾; 长块 = 反解本地块计算时长(首块通信恰好掩盖它)
BACK:  本地 mm 块在流水线尾端——本地保留份无通信 deadline, 与末块通信重叠,
       短块前置专职启动通信管道; 长块 = 反解本地块计算时长(末块通信恰好掩盖它)
NONE:  无本地块——短块作流水线垫片(按 bound 判定放首/尾),
       长块 = 反解短块的通信/计算时长(双向配平: 短块在后→反解其计算时长,
       在前→反解其通信时长)
```

标定查表：通信行按（引擎分区 aicpu/ccu + rankDim）匹配（`FindCommFitRow`，miss=零值行→
防御 NoCut）；算力行按四维形态键匹配（键推导与模型 dtype 分类同源）。
**算力行哨兵判 `computesPerCycle == 0`**（勿用 aBytes——fp4 的 glue 真实口径 aBytes=0 合法）。

### 3.5 离散切分引擎（cut_planner.h，CutPlanner 七步）

把 ③ 反解出的连续长短种子整形成**合法、对齐、轮次受限**的整数块方案（总长守恒的守门人）：

```
① GenerateInitialPartition  连续长块→离散块数
   夹取 longTileLen ∈ [shortTileLen, totalLen − shortTileLen];
   对齐到 mAlignLen(=baseM), 余数 > 对齐长×tol 才 ceil 进位;
   tol 双值: 短块后置(计算 bound) 0.5——长块规整更重要; 前置(通信 bound) 0.75——倾向向下保块大;
   numLongTile = (totalLen − shortTileLen×0.75) / longTileLen   ← 0.75 权重折算短块占位
② 残余判定 + balanceCut     残余够大且长短接近且块数少(≤2)且线性度好 → 索性全等分
   (等分 = 通信轮次最规整的保底形态)
③ RedistributeTile          恰好两块且无本地块保护时残余均摊(头轻脚重修正);
   FRONT/BACK 跳过——均摊会破坏含本地块的端部结构
④ moreFrontTile             残余还装得下一个长块 → 多切一块比留巨大尾块好
⑤ 轮次约束(整除改写)        totalTileCnt > maxTileCnt 且恰好整除对齐 → 改写纯等长块
⑥ MaxTileCntUniform         任意超限兜底: ceil 对齐均分(少一块给尾块留位), 尾块吸收余数
⑦ 尾块重算 + 对齐收尾       shortTileLen = 纯余数(守恒);
   longTileLen > 1KB → AlignTileLarge(短块 ceil 对齐优先——它独立占一轮通信, 不对齐会粒度破碎);
   ≤ 1KB → AlignTileSmall(碎尾吸收: 尾块 < 半块时从每长块借一格凑大; 大尾校正: 尾>主时重排)
```

### 3.6 求解流水（Schedule 六段）

```
① Estimate     总量估计: FindCubeUtil(聚合 0.7) → gradient → 总 mm/通时间 → 通算比;
               FRONT/BACK 算本地窗口时长(FindCubeUtil(m, 1, 0.6), 除以更保守的本地利用率);
               短块位置: FRONT→恒后置 / BACK→恒前置 / NONE→先行段为瓶颈侧则后置
② SetShort     短块 = minTileLen(种子 = baseM) 落位; NoCut 几何门(总长 < 短块×2 → 不切)
③ SetLong      长块反解: FRONT/BACK = InverseCommTime(本地窗口时长);
               NONE = 双向配平(短块在后→反解其 mm 时长, 在前→反解其通信时长)
④ 离散化       CutPlanner 七步(3.5); 本地块形态恒信线性段 + 保护端部结构
⑤ 轮数搜索     budget ∈ [2, N₀) 线性枚举: 恢复 ③ 后形态(反解与轮数无关, 复用) +
   +门控       收紧轮次预算重跑 ④ → 流水线总时择优; 全部候选不敌全串行 → NoCut
⑥ CCU 整形     单轮通信字节 = 块长 × 另一维 × 字节 × commVolumeMul 超 256MB →
               候选表 {2048,1024,512,256} 等长重切(大块优先=少轮次); 轮次预算 = 63−1;
               两约束冲突放弃整形
```

决策层次：**③④ 管块怎么切（配平），⑤ 管切几轮（启动税 vs 重叠收益），⑥ 管硬件上限**。

⑤ 的动机：配平产出的轮数 N₀ 是"配平衡最优"而非全局最优——通信重时 T(N) 随轮数单调增
（启动税吃收益）、计算重时单调降（至首块通信≈b 触底），最优常在中间，二值判定会错过。
成本：N₀≤63 档 × 纯整数重切 = host 微秒级。

⑤ 的流水线总时近似（EstimateCutTotalTime，三形态，不可重叠段在流水线两端）：
```
FRONT:  T = T_comm(首块) + max(T_comm(其余轮), T_mm − T_mm(末块)) + T_mm(末块)
BACK:   T = T_mm(首块) + max(T_mm − T_mm(首块) − 本地窗口, T_comm(除末轮)) + max(T_comm(末轮), 本地窗口)
NONE:   T = T_mm(首块) + max(T_mm − T_mm(首块), T_comm(除末轮)) + T_comm(末轮)   [计算在前]
```

### 3.7 方法-消费方速查

| 方法 | 消费方 |
|---|---|
| CommTime / InverseCommTime | ③ 配平反解 / ⑤ 门控首末块 |
| TotalCommTime | ① Estimate / ⑤ 搜索与门控 / 诊断打印 |
| FindCubeUtil(0.7) / gradient / MatmulTime / InverseMatmulTime | ① Estimate / ③ 配平 / ⑤ 门控 |
| FindCubeUtil(m, 1, 0.6) | ① 本地窗口（FRONT/BACK） |
| CutPlanner 全部离散方法 | ④ / ⑤ 搜索重跑 / NoCut |
| EstimateCutTotalTime | ⑤ 搜索/门控估值器 |
| ApplySingleRoundCommLimit | ⑥（CCU 键） |
| FindCommFitRow / IsCommCalibrated | 标定查表 / op 分派 |
| MakeFitBalanceArgs | 存量算子旧 args 转换 |
| CutResultFromCof | miss 分派（COF 结果 → 引擎切分形态） |

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
