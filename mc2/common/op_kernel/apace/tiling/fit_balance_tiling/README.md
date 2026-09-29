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
通信引擎（AICPU/CCU）经 `args.commEngine` 传入——op 侧 `ParseCommEngine(comm_mode)` 解析。

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

通信标定表是否命中当前形态（通信引擎 + 卡数二维匹配）。算子用它做切分策略分派：
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
    /* commRows: 通信拟合行——{通信引擎, 卡数, k 带宽斜率(us/MB), b 每轮启动开销(us), c 一次性固定开销(us)};
     * 一行 = 一个已标定 (引擎, 卡数) 形态; 未标定形态不写行(miss → 核利用率优先) */
    {
        { {TilingCommEngine::AICPU, 4, 26.5894, 33.17, 20.58}, // 950 标卡 4P 实测(在网行)
          // ↓ 两行新增标定填法演示(数值为演示数——上线前替换为实测值, 未标定则删行留空):
          {TilingCommEngine::AICPU, 8, 22.7301, 33.17, 20.58}, // 新卡档: 同引擎追加一行(8P 实测后照此)
          {TilingCommEngine::AIV_URMA, 4, 3.2, 4.45, 1.8} },   // 新引擎: 引擎维度在行上, 同表加行即可
                                                               // (AIV_URMA 档当前预留, 生产 URMA 走核利用率优先)
    },
    /* mmFit: matmul 算力档 + 最优基本块 */
    {
        { {MatmulDtypeKind::QUANT, false, 1, 1, 8192},   // fp8 量化
          {MatmulDtypeKind::QUANT, true, 0, 0, 16384} }, // fp4 打包(键 {0,0} 是现网真实值, 见 5.2)
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

### 3.1 核心思想：通信-计算流水线配平

通算融合算子的执行是流水线：通信块与计算块交替重叠。切分要回答两个问题——
**块怎么切**（长短结构）与**切几轮**（轮数经济性）。

本引擎的模型：把 M 轴切成 1 短 + N 长，长块的通信时长恰好掩盖相邻块的计算时长（配平）；
每多切一轮多付一份启动开销 b，用轮数搜索择优。

配平的**锚点 = 流水线的启动块**（最先动起来、打破串行依赖的那块）：
有本地块的算子启动块是本地块（FRONT 在首端 / BACK 在尾端，比短块更好——不等通信或没有发送时限）；
无本地块的算子启动块是短块（通信耗时占比大时放在最前先发、计算耗时占比大时放在最后垫尾）。
长块长度 = 按启动块的耗时反解——这是 FRONT/BACK/NONE 三种形态配平公式的统一原理。

### 3.2 通信模型（comm_model.h）

```
x 的计算基准: m × commMatrixLen × realDtypeSize 的 MB(按单卡数据量计——多份数的放大已折进斜率 k)
realDtypeSize = commDtypeSize / expansionFraction   // fp4 子字节: 1/2 = 0.5 B/元素

单轮 CommTime(m)          = k·x + b
反解 InverseCommTime(t)   = (t − b) / k            // b 整额扣除, 换算回 M 行数
总量 TotalCommTime(m, N)  = k·X + N·b + c
```

- **k**（带宽斜率）：按上面 x 的计算基准拟合，多份数的放大已折进 k——聚合类算子标定行
  直接填工具原始值（如 26.5894），**不要**再除以 rankDim；
- **b**（每轮启动开销）：单轮全款——反解时从目标时间整额扣除，配平天然感知它；
- **c**（一次性固定开销）：只计入总时间公式，不计入单轮时间和反解（整个算子只付一次）；
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
- **mnExp 双值**：0.7 聚合稳态（数据齐/流水满，Estimate 用）；**0.6 本地窗口**（单份数据/
  流水未满，增长更平缓 → FRONT/BACK 本地块时长估计更保守）；
- **baseM/N/K 下限**（max）：小于基本块的 tile 也按基本块耗时计（阵列粒度）——防小 shape
  耗时趋于 0；算力 `computesPerCycle` 按 (量化类别, isFp4, A/B 字节数) 四维键查标定档，
  miss 落 defaultComputesPerCycle。

### 3.4 数据流形态推导（InitCtx，calibration.h 的 dataflow 栏）

```
commOnKAxis    = (order == COMM_FIRST)          // 通信在前=传输入 A 侧→K 轴; 在后=传输出 C 侧→N 轴
volumeMul      = fullVolume ? rankDim : 1        // 全量形态: 实算 mm 行数与单次通信字节均为切分域×rankDim
mmRowMul       = volumeMul                        // mm 总行数倍数(MatmulTime 的 rowMul)
commVolumeMul  = volumeMul                        // ⑥ 的 256MB 物理字节判定用(与斜率 k 无关, 是真实字节数)
worksetShares  = (localMMPosition == FRONT) ? rankDim−1 : volumeMul
                                                  // FRONT: 本地份剔出稳态工作集(单独按单份建模)
```

`LocalMMPosition` 三态的物理含义：

```
FRONT: 本地 mm 块在流水线首端——启动端被本地块免费占据(零通信依赖),
       短块退到尾部专职快速收尾; 长块 = 反解本地块计算时长(首块通信恰好掩盖它)
BACK:  本地 mm 块在流水线尾端——本地保留份无通信 deadline, 与末块通信重叠,
       短块前置专职启动通信管道; 长块 = 反解本地块计算时长(末块通信恰好掩盖它)
NONE:  无本地块——短块兼任启动块(哪侧耗时占比大就放另一端, 由耗时判定),
       长块 = 反解短块的通信/计算时长(双向配平: 短块在后→反解其计算时长,
       在前→反解其通信时长)
```

标定查表：通信行按（通信引擎 + 卡数）二维匹配（`FindCommFitRow`，未命中=空行→
防御性输出不切）；算力行按四维键匹配（键推导与模型 dtype 分类同源）。
**未填行的识别判 `computesPerCycle == 0`**（不能用 aBytes 判空——fp4 行的 aBytes 就是合法值 0）。

### 3.5 离散切分引擎（cut_planner.h，CutPlanner 七步）

把 ③ 反解出的连续长短种子整形成**合法、对齐、轮次受限**的整数块方案（总长守恒的守门人）：

```
① GenerateInitialPartition  连续长块→离散块数
   夹取 longTileLen ∈ [shortTileLen, totalLen − shortTileLen];
   对齐到 mAlignLen(=baseM), 余数 > 对齐长×tol 才 ceil 进位;
   tol 双值: 短块在尾(计算耗时占比大)取 0.5——长块规整更重要; 短块在首(通信耗时占比大)取 0.75——宁可块大不冒进;
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
               短块位置: FRONT→恒后置 / BACK→恒前置 / NONE→先行的一段是瓶颈则短块放另一端
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
NONE:   T = T_mm(首块) + max(T_mm − T_mm(首块), T_comm(除末轮)) + T_comm(末轮)   [计算在前的形态]
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
├── fit_balance_tiling.h            对外总接口头文件: 引擎文档 + include 全部分文件
│                                   + InitCtx/Schedule/FitBalanceCut/IsCommCalibrated/FindCommFitRow
└── fit_balance_tiling_utils/       内部分文件(经总接口头间接引入, 算子勿直接 include)
    ├── fit_balance_types.h         类型层: 键控枚举(LocalMMPosition/CommComputeOrder/
    │                               TilingCommEngine/CommAxis/RankMul) + ParseCommEngine +
    │                               数据结构(args/cut/模型画像) + FitBalanceCalibration 标定总集
    ├── comm_model.h                通信模型 HCCLCommModel: y=k·x+N·b+c(单轮/反解/总时间)
    │                               (CommTime 单轮 / InverseCommTime 反解 / TotalCommTime 总量)
    ├── matmul_model.h              计算模型 MatmulPerfModel: cube 利用率(0.7/0.6 双值)/
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

## 五、标定数据维护指南

### 5.1 通信拟合行（k/b/c）——新形态（如 8P）怎么加

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
3. **生效路径**：填入后 `IsCommCalibrated(8, ..., AICPU)` 返回 true，8P 调用自动从 miss 路径
   （核利用率优先）切到 fit-balance。

**未标定形态的处理**：commRows 里**不写该（引擎, 卡数）组合的行**即可——分派自动走 miss
（核利用率优先），无需任何开关。**不要用零值行占位**（零斜率行按未填处理同样不命中；
塞占位值上线则是以假数据进 fit-balance，切分决策失真——历史上 CCU 曾以换算默认值占位，已删除）。

**CCU 引擎**：行结构与 aicpu 完全相同但数值独立——CCU 每轮开销与带宽特性与 AICPU 不同，
必须分别标定，不可复用 aicpu 行的数值。

### 5.2 算力档（mmFit.cubeCalc）——新 dtype 怎么加

按 (calcType, isFp4, A/B 字节数) 四维键匹配，加行即加档：
- fp8：`{MatmulDtypeKind::QUANT, false, 1, 1, 算力}`；
- fp4：`{MatmulDtypeKind::QUANT, true, 0, 0, 算力}`——键的 {0,0} 是现网真实值
  （算子侧获取字节数的 GetDataTypeSize 函数没有 fp4 条目，返回 0），不是笔误；
- bf16：`{MatmulDtypeKind::FP16, false, 2, 2, 算力}`；
- miss 落 `defaultComputesPerCycle`（引擎默认 4096，勿用作长期方案）。

### 5.3 何时不需要动标定

- **换 shape**：不需要——标定行按卡数与 dtype 匹配，与 shape 无关；
- **换通信引擎配置**（comm_mode）：不需要——运行时按 commEngine 选表，两表都已标定即可；
- **新算子接入**：dataflow 三元组按算子填，**通信行/算力档可以直接复制同原语已标定算子的
  数值**（标定的是通信与芯片特性，不是算子特性）——这正是标定层与算子解耦的设计收益。
