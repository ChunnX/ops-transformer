# GMMSQ MxA8W4 编码方案

> 本文档基于 GMMFR（GroupedMatmulFinalizeRouting）MxA8W4 数据流的完整实现，给出 GMMSQ（GroupedMatmulSwigluQuantV2）MxA8W4 数据流的详细编码方案。按代码层级架构组织，涵盖 tiling data、tiling key、L1/UB buffer 分配、controller/basic_block/cube_compute/vec_compute 各层关键实现逻辑。

---

## 8.5 Tiling Data 详细字段说明

### 8.5.1 结构体定义

GMMSQ 的 Tiling Data 基于 GMMFR 的 `GMMFinalizeRoutingWeightQuantTilingData` 改写，去除 sharedInput 相关字段，新增 yScale 相关字段。

```cpp
namespace GMMSQArch35Tiling {
#pragma pack(push, 8)
struct GMMSQWeightQuantTilingData {
    uint8_t  groupListType;   // 分组列表类型: 0=cumsum模式, 1=count模式
    uint8_t  hasBias;         // 是否有Bias: MxA8W4场景固定为0
    uint16_t reserve1;        // 预留对齐字段

    uint32_t groupNum;        // 专家数E
    uint32_t coreNum;         // AIC核心数
    uint32_t outputBs;        // 输出batch size (总token数M)
    uint32_t groupSize;       // MX分组大小 = 32

    uint64_t kSize;           // K维度大小
    uint64_t nSize;           // N维度大小 (完整N，输出为N/2)
};
#pragma pack(pop)
} // namespace GMMSQArch35Tiling
```

### 8.5.2 字段详细说明

| 字段 | 类型 | 含义 | 取值范围 | 与GMMFR差异 |
|------|------|------|----------|------------|
| `groupListType` | `uint8_t` | 分组列表类型 | 0（cumsum）或 1（count） | **相同** |
| `hasBias` | `uint8_t` | 是否有Bias | MxA8W4场景固定为0 | **相同**（语义不变） |
| `reserve1` | `uint16_t` | 预留对齐 | 0 | **相同** |
| `groupNum` | `uint32_t` | 专家数E | 1 ~ 1024 | **相同** |
| `coreNum` | `uint32_t` | AIC核心数 | 编译期确定 | **相同** |
| `outputBs` | `uint32_t` | 总token数M | ≥ 1 | **相同** |
| `groupSize` | `uint32_t` | MX分组大小 | 固定32 | **相同** |
| `kSize` | `uint64_t` | K维度大小 | ≥ 32, 32对齐 | **相同** |
| `nSize` | `uint64_t` | N维度大小（完整N） | ≥ 128, 128对齐 | **相同**（语义不变，但输出为N/2） |

### 8.5.3 与 GMMFR 的差异点

| GMMFR 字段 | GMMSQ 处理 | 原因 |
|------------|-----------|------|
| `sharedInputOffset` | **去除** | GMMSQ 无 sharedInput |
| `sharedInputLen` | **去除** | GMMSQ 无 sharedInput |
| `sharedInputWeight` | **去除** | GMMSQ 无 sharedInput |
| `initSize` | **去除** | GMMSQ 不需要 GM 清零（无 AtomicAdd） |

**关键差异**：GMMSQ 的 TilingData 比 GMMFR 少 4 个字段（共 12 字节），结构更简洁。`nSize` 字段语义不变（存储完整 N），但实际输出维度为 N/2，在 Controller 层通过 `nSize/2` 计算输出偏移。

---

## 8.6 Tiling Key 详细说明

### 8.6.1 Key 生成逻辑

Tiling Key 用于编译期模板实例化选择，通过 `ASCENDC_TPL_ARGS_DECL` 宏声明支持的模板参数组合。

```cpp
namespace GMMSQArch35WeightQuantTiling {

ASCENDC_TPL_ARGS_DECL(GMMSQWeightQuant,
                      ASCENDC_TPL_UINT_DECL(ATRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0),
                      ASCENDC_TPL_UINT_DECL(BTRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(HASBIAS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ATRANS, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BTRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(HASBIAS, ASCENDC_TPL_UI_LIST, 0)));
} // namespace GMMSQArch35WeightQuantTiling
```

### 8.6.2 支持的模板参数组合

| 参数 | 取值 | 说明 |
|------|------|------|
| `ATRANS` | `{0}` | 左矩阵固定不转置 |
| `BTRANS` | `{0, 1}` | 右矩阵转置（NZ格式为1，ND格式为0） |
| `HASBIAS` | `{0}` | MxA8W4场景固定无Bias |

**总组合数**：`1 × 2 × 1 = 2` 种场景。

### 8.6.3 与 GMMFR 的差异

| 参数 | GMMFR | GMMSQ | 差异说明 |
|------|-------|-------|---------|
| `ATRANS` | `{0}` | `{0}` | **相同** |
| `BTRANS` | `{0, 1}` | `{0, 1}` | **相同** |
| `HASBIAS` | `{0, 1}` | `{0}` | GMMSQ MxA8W4 场景固定无 Bias，减少 1 种组合 |

GMMFR 的 Tiling Key 有 `1 × 2 × 2 = 4` 种组合，GMMSQ 减少为 `1 × 2 × 1 = 2` 种。

---

## 8.7 Kernel 入口

### 8.7.1 场景分发宏

在 `op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp` 中，通过宏 `V310_GMM_SQ_ANTI_QUANT` 识别 MxA8W4 场景：

```cpp
#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_W) && defined(DT_FLOAT8_E4M3FN) && defined(DT_FLOAT4_E2M1)
#if (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && (ORIG_DTYPE_W == DT_FLOAT4_E2M1))
#define V310_GMM_SQ_ANTI_QUANT
#endif
#endif
```

### 8.7.2 Kernel 入口函数签名

```cpp
#if defined(V310_GMM_SQ_ANTI_QUANT)
template <int ATRANS, int BTRANS, int HASBIAS>
#else
// 原有的全量化场景模板参数
#endif
__global__ __aicore__ void
grouped_matmul_swiglu_quant_v2(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                               GM_ADDR bias, GM_ADDR groupList, GM_ADDR xScale,
                               GM_ADDR y, GM_ADDR yScale,
                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
```

### 8.7.3 Controller 实例化伪代码

```cpp
#if defined(V310_GMM_SQ_ANTI_QUANT)
    AscendC::InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // 注册 Tiling Data
    REGISTER_TILING_DEFAULT(GMMSQArch35Tiling::GMMSQWeightQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GMMSQArch35Tiling::GMMSQWeightQuantTilingData, tilingData, tilingGM);
    const GMMSQArch35Tiling::GMMSQWeightQuantTilingData *tiling = &tilingData;

    // 实例化 Controller
    // 模板参数: xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
    //           biasType, yType, yScaleType, hasBias, wqmmConfig, vecConfig
    GMMSQ::GMMSQWeightQuantResplitController<
        DTYPE_X, DTYPE_W, DTYPE_SCALE, DTYPE_SCALE, DTYPE_PERTOKEN_SCALE,
        DTYPE_BIAS, DTYPE_Y, DTYPE_Y_SCALE, HASBIAS,
        WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK,
        WeightQuantBatchMatmulV2::Arch35::VEC_ANTIQUANT_CONFIG_DYNAMIC>
        controller;

    // Init: 传入 x, weight, weightScale, bias(nullptr), groupList, xScale, y, yScale, tiling
    controller.Init(x, weight, weightScale, nullptr, groupList, xScale, y, yScale, tiling);
    controller.Process();
#endif
```

### 8.7.4 与 GMMFR 的差异

| 项目 | GMMFR | GMMSQ |
|------|-------|-------|
| 宏名称 | `V310_GMM_FR_ANTI_QUANT` | `V310_GMM_SQ_ANTI_QUANT` |
| 函数名 | `grouped_matmul_finalize_routing` | `grouped_matmul_swiglu_quant_v2` |
| Init 参数 | 含 logit, rowIndex, shareInput | 去除 logit/rowIndex/shareInput，新增 yScale |
| 模板参数 | 含 sharedInputDType, logitsType, rowIndexType | 去除这 3 个，新增 yScaleType |

---

## 8.8 Controller 层编码方案

### 8.8.1 类定义和模板参数

```cpp
namespace GMMSQ {

#define GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM                                          \
    template <typename xType, typename wType, typename antiQuantScaleType,                 \
              typename scaleType, typename perTokenScaleType, typename biasType,           \
              typename yType, typename yScaleType, bool hasBias,                           \
              const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>

#define GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS                                                   \
    GMMSQWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType,         \
                                      perTokenScaleType, biasType, yType, yScaleType,      \
                                      hasBias, wqmmConfig, vecConfig>

GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
class GMMSQWeightQuantResplitController {
public:
    __aicore__ inline GMMSQWeightQuantResplitController(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale,
                                GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
                                GM_ADDR y, GM_ADDR yScale,
                                const GMMSQWeightQuantTilingData *__restrict baseTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOffsetParam(BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM]);
    __aicore__ inline void SplitNByMultiCore(BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM],
                                             BasicBlockControlParam &ctrlParam, uint64_t basicBlockCount,
                                             uint64_t basicBlockSize);
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize);
    __aicore__ inline uint64_t GetSwitchedProcessId(const BasicBlockControlParam &ctrlParam);

    const GMMSQWeightQuantTilingData *tiling_;

    __gm__ xType *xGm_;
    __gm__ wType *weightGm_;
    __gm__ antiQuantScaleType *antiquantScaleGm_;
    __gm__ biasType *biasGm_;
    __gm__ yType *yGm_;
    __gm__ perTokenScaleType *perTokenScaleGm_;
    __gm__ yScaleType *yScaleGm_;          // 新增：输出 scale GM 地址
    GlobalTensor<int64_t> groupListGm_;

    GMMSQWeightQuantVcvBasicBlock<xType, wType, antiQuantScaleType, scaleType,
                                  perTokenScaleType, biasType, yType, yScaleType,
                                  hasBias, wqmmConfig, vecConfig> basicBlock_;

    uint64_t preOffset_ = 0;

    // 动态K选择阈值
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD = 256;
    static constexpr uint64_t M_L1 = 256;
    static constexpr uint64_t KB_L1_SIZE = 256;
    static constexpr uint64_t N_L1 = 256;  // 256 = 128 Swish + 128 Gate
};
```

### 8.8.2 Init() 方法伪代码

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias,
                            GM_ADDR groupList, GM_ADDR perTokenScale,
                            GM_ADDR y, GM_ADDR yScale,
                            const GMMSQWeightQuantTilingData *__restrict baseTiling)
{
    tiling_ = baseTiling;

    // GM 地址初始化
    xGm_ = reinterpret_cast<__gm__ xType *>(x);
    weightGm_ = reinterpret_cast<__gm__ wType *>(weight);
    biasGm_ = reinterpret_cast<__gm__ biasType *>(bias);
    antiquantScaleGm_ = reinterpret_cast<__gm__ antiQuantScaleType *>(antiquantScale);
    perTokenScaleGm_ = reinterpret_cast<__gm__ perTokenScaleType *>(perTokenScale);
    yGm_ = reinterpret_cast<__gm__ yType *>(y);
    yScaleGm_ = reinterpret_cast<__gm__ yScaleType *>(yScale);  // 新增
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));

    // BasicBlock 初始化（传入 yScale 地址）
    basicBlock_.Init(tiling_->groupSize, yGm_, yScaleGm_);
}
```

### 8.8.3 Process() 方法伪代码

```cpp
__aicore__ inline void Process()
{
    // 1. 获取 BlockIdx（AIC/AIV 配对映射）
    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;  // AIV 核 ID = AIC 核 ID / 2
    }

    // 2. 初始化 offsetParam（K/N 轴基本信息）
    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM];
    InitOffsetParam(offsetParam);

    // 3. 缓存对齐检查（4bit 权重场景：K % 256 != 0 表示不对齐）
    bool isCacheLineUnaligned = offsetParam[0].kSize % 256 != 0;

    // 4. GMMSQ 不需要 InitAtomicGm（无 GM 清零、无 sharedInput）
    // 去除: basicBlock_.InitAtomicGm(...)
    // 去除: SyncAll()

    // 5. Group 遍历主循环
    BasicBlockControlParam ctrlParam;
    ctrlParam.processId = 0;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < tiling_->groupNum; ++groupIdx) {
        // 5.1 从 groupList 获取当前 group 的 M 大小
        ctrlParam.mSize = GetSplitValueFromGroupList(groupIdx);

        if (ctrlParam.mSize > 0 && offsetParam[ctrlParam.processId].nSize > 0) {
            // 5.2 M 轴 L1 分块
            uint64_t mBlkNum = CeilDivide(ctrlParam.mSize, M_L1);
            ctrlParam.mL1Size = CeilDivide(ctrlParam.mSize, mBlkNum);

            // 5.3 更新 GM 地址（含 L2 Cache 优化判断）
            basicBlock_.UpdateGlobalAddr(xGm_, weightGm_, antiquantScaleGm_, nullptr,
                                         perTokenScaleGm_, biasGm_, yGm_, yScaleGm_,
                                         ctrlParam.mL1Size < ctrlParam.mSize || isCacheLineUnaligned);

            // 5.4 多核轮询调度
            ctrlParam.curBasicBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + tiling_->coreNum;
            ctrlParam.basicBlockLimit = startBasicBlockId;

            // 5.5 M 轴遍历
            for (ctrlParam.mOffset = 0; ctrlParam.mOffset < ctrlParam.mSize;
                 ctrlParam.mOffset += ctrlParam.mL1Size) {
                ctrlParam.nOffset = 0;
                // N 轴多核分块（主块）
                SplitNByMultiCore(offsetParam, ctrlParam,
                                  CeilDivide(offsetParam[0].nSize, N_L1), N_L1);
                ctrlParam.basicBlockLimit += CeilDivide(offsetParam[0].nSize, N_L1);
            }
            startBasicBlockId = ctrlParam.basicBlockLimit % tiling_->coreNum;
        }

        // 5.6 跨 group 更新 GM 地址
        UpdateGmAddr(ctrlParam.mSize, offsetParam[0].kSize, offsetParam[0].nSize);
    }

    // 6. 收尾处理
    basicBlock_.End(offsetParam[GetSwitchedProcessId(ctrlParam)]);
}
```

### 8.8.4 SplitNByMultiCore() 方法伪代码

```cpp
__aicore__ inline void SplitNByMultiCore(
    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM],
    BasicBlockControlParam &ctrlParam, uint64_t basicBlockCount, uint64_t basicBlockSize)
{
    // basicBlockSize = N_L1 = 256 (128 Swish + 128 Gate)
    for (; ctrlParam.curBasicBlockId < ctrlParam.basicBlockLimit + basicBlockCount;
         ctrlParam.curBasicBlockId += tiling_->coreNum) {

        // 设置当前 BasicBlock 的 M 参数
        offsetParam[ctrlParam.processId].mSize = ctrlParam.mSize;
        offsetParam[ctrlParam.processId].mOffset = ctrlParam.mOffset;
        offsetParam[ctrlParam.processId].mL1Size =
            ctrlParam.mOffset + ctrlParam.mL1Size > ctrlParam.mSize ?
                ctrlParam.mSize - ctrlParam.mOffset : ctrlParam.mL1Size;

        // N 轴偏移（多核轮询）
        offsetParam[ctrlParam.processId].nOffset =
            ctrlParam.nOffset +
            ((ctrlParam.curBasicBlockId - ctrlParam.basicBlockLimit) % basicBlockCount) * basicBlockSize;

        // N 轴 L1 大小（尾块处理）
        offsetParam[ctrlParam.processId].nL1Size =
            offsetParam[ctrlParam.processId].nOffset + basicBlockSize > tiling_->nSize ?
                tiling_->nSize - offsetParam[ctrlParam.processId].nOffset : basicBlockSize;

        // 输出 GM 地址（FP8 输出）
        offsetParam[ctrlParam.processId].yGmAddr = reinterpret_cast<GM_ADDR>(yGm_);

        // 输出 Scale GM 地址（新增）
        offsetParam[ctrlParam.processId].yScaleGmAddr = reinterpret_cast<GM_ADDR>(yScaleGm_);

        // 动态 K 选择
        offsetParam[ctrlParam.processId].kbL1Size =
            (offsetParam[ctrlParam.processId].mL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD &&
             offsetParam[ctrlParam.processId].nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                MX_A8W4_L1_K_CONFIG_512 : MX_A8W4_L1_K_CONFIG_256;

        // A 矩阵 L1 大小计算（平衡 MTE1 和 MTE2 搬运效率）
        uint64_t mL1Align = CeilAlign(offsetParam[ctrlParam.processId].mL1Size, BLOCK_CUBE);
        uint64_t kaDepth = CeilDivide(offsetParam[ctrlParam.processId].nL1Size, mL1Align * 2);
        uint64_t aL1Size = 128 * GetKBUnit<xType>();  // 无 Bias 场景 128KB
        uint64_t maxKaDepth = aL1Size / (mL1Align * offsetParam[ctrlParam.processId].kbL1Size);
        if (kaDepth > maxKaDepth) { kaDepth = maxKaDepth; }
        offsetParam[ctrlParam.processId].kaL1Size = kaDepth * offsetParam[ctrlParam.processId].kbL1Size;

        // 执行 BasicBlock 计算
        basicBlock_.ComputeBasicBlock(offsetParam[ctrlParam.processId],
                                      offsetParam[GetSwitchedProcessId(ctrlParam)]);
        ctrlParam.processId = GetSwitchedProcessId(ctrlParam);
    }
    ctrlParam.nOffset += basicBlockSize * basicBlockCount;
}
```

### 8.8.5 UpdateGmAddr() 方法伪代码

```cpp
__aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize)
{
    xGm_ += mSize * kSize;
    weightGm_ += (nSize * kSize) >> 1;  // 4bit 权重，字节数减半

    uint64_t kAlignSize = CeilAlign(kSize, K_ALIGNMENT64);
    antiquantScaleGm_ += nSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));
    perTokenScaleGm_ += mSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));

    biasGm_ += nSize;

    // 输出 N/2（关键差异）
    yGm_ += mSize * (nSize / 2);

    // 新增：输出 scale GM 地址更新
    yScaleGm_ += mSize * CeilDivide(CeilAlign(nSize / 2, 64), static_cast<uint64_t>(tiling_->groupSize));

    // 去除: logitsAddr_ += mSize;
    // 去除: rowIndexAddr_ += mSize;
}
```

### 8.8.6 与 GMMFR 的差异点说明

| 差异点 | GMMFR | GMMSQ | 说明 |
|--------|-------|-------|------|
| 成员变量 | 含 `shareInputAddr_`, `logitsAddr_`, `rowIndexAddr_` | 去除这 3 个，新增 `yScaleGm_` | GMMSQ 无 Routing 相关输入 |
| `Init()` | 传入 logits, rowIndex, shareInput | 去除这 3 个参数，新增 yScale | 参数列表简化 |
| `Process()` | 含 `InitAtomicGm()` + `SyncAll()` | **去除** | GMMSQ 不需要 GM 清零 |
| `UpdateGmAddr()` | `yGm_ += mSize * nSize` | `yGm_ += mSize * (nSize / 2)` | 输出维度减半 |
| `UpdateGmAddr()` | 含 `logitsAddr_`, `rowIndexAddr_` 更新 | 去除，新增 `yScaleGm_` 更新 | 后处理逻辑变化 |
| `SplitNByMultiCore()` | `basicBlockSize = 256` | `basicBlockSize = 256`（语义变为 128+128） | N 轴切分大小不变，但输出偏移为 `nOffset/2` |
| `SplitNByMultiCore()` | 无 `yScaleGmAddr` | 新增 `yScaleGmAddr` 设置 | 输出 scale 地址传递 |

---

## 8.9 BasicBlock 层编码方案

### 8.9.1 类定义和模板参数

```cpp
namespace GMMSQ {

#define GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM                                              \
    template <typename xType, typename wType, typename antiQuantScaleType,                  \
              typename scaleType, typename perTokenScaleType, typename biasType,            \
              typename yType, typename yScaleType, bool hasBias,                            \
              const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>

#define GMMSQ_WQ_VCV_BASIC_BLOCK_CLASS                                                       \
    GMMSQWeightQuantVcvBasicBlock<xType, wType, antiQuantScaleType, scaleType,              \
                                  perTokenScaleType, biasType, yType, yScaleType,           \
                                  hasBias, wqmmConfig, vecConfig>

GMMSQ_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
class GMMSQWeightQuantVcvBasicBlock {
public:
    __aicore__ inline GMMSQWeightQuantVcvBasicBlock(){};
    __aicore__ inline void Init(uint64_t antiQuantGroupSize, __gm__ yType *y,
                                __gm__ yScaleType *yScale);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight,
                                            __gm__ antiQuantScaleType *antiquantScale,
                                            __gm__ xType *antiquantOffset,
                                            __gm__ perTokenScaleType *perTokenScale,
                                            __gm__ biasType *bias, __gm__ yType *y,
                                            __gm__ yScaleType *yScale,
                                            const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                             const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void End(const BasicBlockOffsetParam &curOffsetParam);

protected:
    // AIV 路径：三段式结构（Pass1 + PhaseB + Pass2）
    __aicore__ inline void IterateNzNkWithAiv(const BasicBlockOffsetParam &curOffsetParam,
                                              const BasicBlockOffsetParam &lastOffsetParam);
    // AIC 路径：K 循环 MMA 计算
    __aicore__ inline void IterateNzNkWithKAic(const BasicBlockOffsetParam &curOffsetParam,
                                               const BasicBlockOffsetParam &lastOffsetParam);
    // Weight 反量化子循环
    __aicore__ inline void VecComputeNzNkWithStartLimit(uint64_t &kMte2Offset, uint64_t kMte2Limit,
                                                        const BasicBlockOffsetParam &curOffsetParam);

    // 跨核同步原语（与 GMMFR 完全相同）
    template <const pipe_t pipe> __aicore__ inline void SetAivToAic(uint64_t syncFlag);
    template <const pipe_t pipe> __aicore__ inline void WaitAivToAic(uint64_t syncFlag);
    template <const pipe_t pipe> __aicore__ inline void SetAicToAiv(uint64_t syncFlag);
    template <const pipe_t pipe> __aicore__ inline void WaitAicToAiv(uint64_t syncFlag);

    // 计算组件
    GMMSQWeightQuantVecCompute<xType, wType, biasType, yType, yScaleType,
                               hasBias, wqmmConfig, vecConfig> vecCompute_;
    GMMSQWeightQuantCubeCompute<xType, biasType, antiQuantScaleType, perTokenScaleType,
                                float, hasBias, wqmmConfig> cubeCompute_;

    // 循环索引
    uint64_t cvLoopIdx_ = 0;       // K 循环索引（AIC/AIV 共享）
    uint64_t weightL1DbOffset_ = 0; // Weight L1 双缓冲偏移
    uint64_t biasL1DbOffset_ = 0;   // Bias L1 双缓冲偏移

    // L1 Buffer
    LocalTensor<xType> weightL1_;
    LocalTensor<biasType> biasL1_;

    // UB Buffer（复用策略）
    LocalTensor<float> yF32Buffer_;          // MMA Fixpipe 输出（复用 weightHighBit 空间）
};
```

### 8.9.2 Init() 方法伪代码

```cpp
__aicore__ inline void Init(uint64_t antiQuantGroupSize, __gm__ yType *y,
                            __gm__ yScaleType *yScale)
{
    // === L1 Buffer 分配（与 GMMFR 完全相同） ===
    biasL1DbOffset_ = 0;
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256;  // 64KB
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;

    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;

    // Bias L1（无 Bias 场景跳过）
    biasL1_ = LocalTensor<biasType>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(biasType));

    // === UB Buffer 分配（重新布局，复用策略） ===
    // yF32Buffer_ 复用 weightHighBit_ 的空间
    // weightHighBit_ 在 PhaseB 阶段已全部搬运到 L1，UB 空间可安全复用
    constexpr uint64_t ubOffset =
        WeightQuantBatchMatmulV2::Arch35::GetGMMSQMxA8W4BufferInfo<vecConfig>().weightLowbitTotalSize;
    constexpr uint64_t highBitSize =
        WeightQuantBatchMatmulV2::Arch35::GetGMMSQMxA8W4BufferInfo<vecConfig>().weightHighBitTotalSize;
    yF32Buffer_ = LocalTensor<float>(TPosition::LCM, ubOffset, highBitSize);

    // === AIC/AIV 分流初始化 ===
    if ASCEND_IS_AIC {
        // AIC: 初始化 CubeCompute（L1 buffer 分配）
        cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize, biasL1DbOffset_, biasL1_);
        // 通知 AIV：AIC 初始化完成
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    } else {
        // AIV: 初始化 VecCompute（UB buffer 分配 + Event 初始化）
        vecCompute_.Init(y, yScale);  // 传入 yScale 地址（新增）
    }
    cvLoopIdx_ = 0;
}
```

### 8.9.3 ComputeBasicBlock() 方法伪代码

```cpp
__aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                         const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        IterateNzNkWithAiv(curOffsetParam, lastOffsetParam);
    } else {
        IterateNzNkWithKAic(curOffsetParam, lastOffsetParam);
    }
}
```

### 8.9.4 IterateNzNkWithAiv() 方法伪代码（三段式结构）

```cpp
__aicore__ inline void IterateNzNkWithAiv(const BasicBlockOffsetParam &curOffsetParam,
                                          const BasicBlockOffsetParam &lastOffsetParam)
{
    uint64_t kMte2Offset = 0;
    uint64_t curCvLoopIdx = cvLoopIdx_;

    // ===== Pass 1: 预加载前 DOUBLE_BUFFER_NUM 个 K chunk 到 UB 进行反量化 =====
    VecComputeNzNkWithStartLimit(kMte2Offset,
                                 Min(curOffsetParam.kSize, DOUBLE_BUFFER_NUM * curOffsetParam.kbL1Size),
                                 curOffsetParam);

    // ===== Phase B: 处理上一个 BasicBlock 的 SwiGLU + Quant 输出（如果存在）=====
    if (curCvLoopIdx > 0) {
        // 1. 通知 AIC 可以 Fixpipe（AIV 已取走上一个 BB 的 UB 结果）
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);

        // 2. 等待 AIC Fixpipe 完成（结果在 yF32Buffer_ 中）
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

        // 3. SwiGLU 计算
        //    输入: yF32Buffer_[M, 256]，前 128 为 Swish 分支，后 128 为 Gate 分支
        //    输出: yF32Buffer_[M, 128]（原地覆盖前 128 个元素）
        vecCompute_.SwiGLU(yF32Buffer_, lastOffsetParam);

        // 4. MX 量化三步
        //    4.1 ComputeMaxExp: 计算每 32 个元素的 max exponent → maxB16Buffer_
        vecCompute_.ComputeMaxExp(yF32Buffer_, lastOffsetParam);
        //    4.2 ComputeScale: shared_exp → E8M0 scale → scaleE8M0Buffer_
        vecCompute_.ComputeScale(lastOffsetParam);
        //    4.3 ComputeDataForQuantTargetFp8: 量化到 FP8_E4M3FN → yFp8Buffer_
        vecCompute_.ComputeDataForQuantTargetFp8(yF32Buffer_, lastOffsetParam);

        // 5. MTE3 写出到 GM
        //    5.1 写出 yFp8 到 GM
        vecCompute_.CopyOutputFromUb2Gm(lastOffsetParam);
        //    5.2 写出 scaleE8M0 到 GM
        vecCompute_.CopyScaleFromUb2Gm(lastOffsetParam);

        // 6. 同步：通知 MTE3 和 VF 完成
        vecCompute_.SetFlag<MTE3_V>(EVENT_ID_MTE3_TO_V);
    }

    // ===== Pass 2: 继续加载剩余 K chunk =====
    VecComputeNzNkWithStartLimit(kMte2Offset, curOffsetParam.kSize, curOffsetParam);
}
```

### 8.9.5 IterateNzNkWithKAic() 方法伪代码

```cpp
__aicore__ inline void IterateNzNkWithKAic(const BasicBlockOffsetParam &curOffsetParam,
                                           const BasicBlockOffsetParam &lastOffsetParam)
{
    // PhaseB Fixpipe（处理上一个 BasicBlock 的 MMA 结果）
    if (cvLoopIdx_ > 0) {
        // 1. 等待 AIV 通知可以 Fixpipe
        WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        // 2. Fixpipe: L0C → yF32Buffer_（UB）
        cubeCompute_.GetTensorC(yF32Buffer_, lastOffsetParam);
        // 3. 通知 AIV Fixpipe 完成
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }

    // K 循环 MMA 计算
    for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize;
         kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {

        uint64_t kbL1RealSize = (kbL1Offset + curOffsetParam.kbL1Size) >= curOffsetParam.kSize ?
                                    curOffsetParam.kSize - kbL1Offset : curOffsetParam.kbL1Size;

        // 1. 等待 Scale MTE1→MTE2 完成
        cubeCompute_.WaitScaleMTE1ToMTE2(kbL1Offset);
        // 2. MTE1: GM→L1 搬运 MX Scale（ScaleA + ScaleB）
        cubeCompute_.CopyMxScaleGmToL1(curOffsetParam, kbL1Offset);

        // 3. 等待 A 矩阵 MTE1→MTE2 完成
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam);
        // 4. MTE1: GM→L1 搬运 A 矩阵（ND2NZ）
        cubeCompute_.CopyAAndBiasGmToL1(curOffsetParam, kbL1Offset, cvLoopIdx_);

        // 5. Weight L1 尾部填零（K 不对齐时）
        cubeCompute_.PadL1IfNotAlign(weightL1_, weightL1DbOffset_, cvLoopIdx_,
                                     kbL1RealSize, curOffsetParam.nL1Size);

        // 6. 等待 AIV 完成 Weight 反量化（L1 上的 Weight 已就绪）
        WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);

        // 7. 启动 MMA 计算（MTE1→L0→MMAD→Fixpipe 流水）
        cubeCompute_.LaunchMatmul(weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_],
                                  kbL1Offset, kbL1RealSize, cvLoopIdx_, curOffsetParam);

        // 8. 通知 MTE2：A 矩阵搬运完成
        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam);
        // 9. 通知 MTE2：Scale 搬运完成
        cubeCompute_.SetScaleMTE1ToMTE2(kbL1Offset, curOffsetParam);

        // 10. 通知 AIV：MMA 计算完成，可以继续搬运下一块 Weight
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}
```

### 8.9.6 End() 方法伪代码

```cpp
__aicore__ inline void End(const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIC {
        // AIC 侧：最后一次 Fixpipe + 同步收尾
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(yF32Buffer_, lastOffsetParam);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync();  // 等待所有 MTE1→MTE2 和 M→MTE1 Event
    } else {
        // AIV 侧：最后一次 SwiGLU + Quant + 写出
        if (cvLoopIdx_ > 0) {
            SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

            // SwiGLU + MX Quant + 写出（同 PhaseB 逻辑）
            vecCompute_.SwiGLU(yF32Buffer_, lastOffsetParam);
            vecCompute_.ComputeMaxExp(yF32Buffer_, lastOffsetParam);
            vecCompute_.ComputeScale(lastOffsetParam);
            vecCompute_.ComputeDataForQuantTargetFp8(yF32Buffer_, lastOffsetParam);
            vecCompute_.CopyOutputFromUb2Gm(lastOffsetParam);
            vecCompute_.CopyScaleFromUb2Gm(lastOffsetParam);
        }
        // 等待 AIC 完成最后两次搬运
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        vecCompute_.End();  // 等待所有 MTE3→V 和 V→MTE2 Event
    }
}
```

### 8.9.7 与 GMMFR 的差异点说明

| 差异点 | GMMFR | GMMSQ | 说明 |
|--------|-------|-------|------|
| 模板参数 | 含 `sharedInputDType`, `logitsType`, `rowIndexType` | 去除这 3 个，新增 `yScaleType` | 后处理逻辑变化 |
| 成员变量 | `ubOutputF32Buffer_` | `yF32Buffer_`（语义相同，名称变化） | 更清晰的命名 |
| `Init()` | `vecCompute_.Init(sharedInputWeight, y)` | `vecCompute_.Init(y, yScale)` | 去除 sharedInputWeight，新增 yScale |
| `IterateNzNkWithAiv()` PhaseB | `CopyRowIndexLogitsGmToUb → MulLogits → RoutingYToGm` | `SwiGLU → ComputeMaxExp → ComputeScale → ComputeDataForQuantTargetFp8 → CopyOutputFromUb2Gm → CopyScaleFromUb2Gm` | **核心差异**：后处理从 Routing 改为 SwiGLU+Quant |
| `IterateNzNkWithAiv()` Pass1/Pass2 | Weight 反量化 | **完全相同** | 反量化流程不变 |
| `IterateNzNkWithKAic()` | `GetTensorC(ubOutputF32Buffer_, ...)` | `GetTensorC(yF32Buffer_, ...)` | 目标 UB 地址名称变化 |
| `End()` AIV 侧 | `MulLogits + RoutingYToGm` | `SwiGLU + Quant + 写出` | 后处理逻辑变化 |
| 同步信号 | FLAG 8/9/4/3 | **完全相同** | 同步信号数量和语义不变 |

---

## 8.10 CubeCompute 层编码方案

### 8.10.1 类定义和模板参数

```cpp
namespace GMMSQ {

#define GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM                                                 \
    template <typename xType, typename biasType, typename antiQuantScaleType,               \
              typename perTokenScaleType, typename yType, bool hasBias,                     \
              const WqmmConfig &wqmmConfig>

#define GMMSQ_WQ_CUBE_COMPUTE_CLASS                                                          \
    GMMSQWeightQuantCubeCompute<xType, biasType, antiQuantScaleType, perTokenScaleType,     \
                                yType, hasBias, wqmmConfig>

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
class GMMSQWeightQuantCubeCompute {
public:
    __aicore__ inline GMMSQWeightQuantCubeCompute(){};

    // GM 地址更新
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ yType *y,
                                            __gm__ biasType *bias,
                                            __gm__ antiQuantScaleType *antiquantScale,
                                            __gm__ uint64_t *quantScale,
                                            __gm__ perTokenScaleType *perTokenScale);

    // L1 Buffer 分配
    __aicore__ inline void MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize,
                                      uint64_t mxBiasL1DbOffset,
                                      const LocalTensor<biasType> &biasL1);

    // MMA 计算核心
    __aicore__ inline void LaunchMatmul(const LocalTensor<xType> &weightL1,
                                        int64_t kbOffset, uint64_t kbL1RealSize,
                                        uint64_t cvLoopIdx, const BasicBlockOffsetParam &param);

    // MTE1↔MTE2 同步
    __aicore__ inline void WaitMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SetMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WaitScaleMTE1ToMTE2(uint64_t kbGmOffset);
    __aicore__ inline void SetScaleMTE1ToMTE2(uint64_t kbGmOffset, const BasicBlockOffsetParam &offsetParam);

    // GM→L1 搬运
    __aicore__ inline void CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param,
                                              int64_t kaGmOffset, uint64_t cvLoopIdx);
    __aicore__ inline void CopyMxScaleGmToL1(const BasicBlockOffsetParam &param, uint64_t kbL1Offset);

    // Fixpipe 输出
    __aicore__ inline void GetTensorC(LocalTensor<yType> &yUb, const BasicBlockOffsetParam &param);

    // 收尾同步
    __aicore__ inline void EndSync();

    // MMAD 计算
    __aicore__ inline void DoMmadCompute(uint64_t loopId, uint64_t cvLoopIdx,
                                         const L0CopyAndCalcParams &l0CopyAndCalcParams);

    // L1 尾部填零
    __aicore__ inline void PadL1IfNotAlign(const LocalTensor<xType> &l1Buf, uint64_t dbOffset,
                                           uint64_t dbIdx, uint64_t realSize, uint64_t dimSize);

private:
    // L0 Buffer（与 GMMFR 完全相同）
    static constexpr uint64_t L0_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_OFFSET_B8 = 32 * 1024;
    static constexpr uint64_t BIAS_TABLE_OFFSET_B32 = 512;
    static constexpr uint64_t L0C_BUF_SIZE = 256 * 256;

    LocalTensor<L0DataType> l0a_{TPosition::A2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};
    LocalTensor<L0DataType> l0b_{TPosition::B2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};
    LocalTensor<float> l0c_{TPosition::CO1, 0, 256 * GetKBUnit<int8_t>() / sizeof(float)};
    LocalTensor<float> biasTable_{TPosition::C2, 0, 4 * GetKBUnit<int8_t>() / sizeof(float)};

    // L1 Buffer
    LocalTensor<xType> aL1_;
    LocalTensor<fp8_e8m0_t> mxScaleAL1_;
    LocalTensor<fp8_e8m0_t> mxScaleBL1_;
    LocalTensor<biasType> biasL1_;

    // 双缓冲偏移
    uint64_t aL1DbOffset_;
    uint64_t mxScaleAL1DbOffset_;
    uint64_t mxScaleBL1DbOffset_;
    uint64_t biasL1DbOffset_;

    // 索引
    uint64_t l0LoopIdx_ = 0;
    uint64_t l0cOffset_ = 0;
    uint64_t mxScaleBufIdx_ = 0;
    uint64_t aL1BufIdx_ = 0;

    // Event ID
    static constexpr uint32_t EVENT_ID_MTE1_TO_MTE2 = 3;
    static constexpr uint32_t EVENT_ID_SCALE_MTE1_TO_MTE2 = 5;
    static constexpr uint32_t EVENT_ID_M_TO_MTE1 = 3;
    static constexpr uint32_t EVENT_ID_MTE1_TO_M = 3;
    static constexpr uint32_t EVENT_ID_MTE2_TO_MTE1 = 3;
};
```

### 8.10.2 MxA8W4Init() 方法伪代码

```cpp
__aicore__ inline void MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize,
                                  uint64_t mxBiasL1DbOffset, const LocalTensor<biasType> &biasL1)
{
    // L1 空间分配（无 Bias 场景）:
    // ┌──────────────┬───────────────┬───────────────┬──────────┬──────────┬───────────────┬───────────────┬──────────────┐
    // │WeightL1_P0   │ ScaleAL1_P0   │ ScaleBL1_P0   │ AL1_P0   │ AL1_P1   │ ScaleBL1_P1   │ ScaleAL1_P1   │ WeightL1_P1  │
    // │   64KB       │    32KB       │    32KB       │  128KB   │  128KB   │    32KB       │    32KB       │    64KB      │
    // └──────────────┴───────────────┴───────────────┴──────────┴──────────┴───────────────┴───────────────┴──────────────┘

    biasL1_ = biasL1;
    biasL1DbOffset_ = mxBiasL1DbOffset;

    // ScaleA L1 双缓冲分配
    mxScaleAL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize,
                                          l1RemainSize / sizeof(fp8_e8m0_t));
    mxScaleAL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    // ScaleB L1 双缓冲分配
    mxScaleBL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize,
                                          l1RemainSize / sizeof(fp8_e8m0_t));
    mxScaleBL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    // A 矩阵 L1 双缓冲分配（剩余空间全部给 AL1）
    aL1_ = LocalTensor<xType>(TPosition::TSCM, l1StartSize, l1RemainSize);
    aL1DbOffset_ = l1RemainSize >> 1;

    // 初始化 Event（全部置为完成状态）
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
    }
}
```

### 8.10.3 CopyMxScaleGmToL1() 方法伪代码

```cpp
__aicore__ inline void CopyMxScaleGmToL1(const BasicBlockOffsetParam &param, uint64_t kbL1Offset)
{
    // Scale 搬运粒度: MX_SCALE_K_L1_SIZE = 4096
    if (kbL1Offset % MX_SCALE_K_L1_SIZE != 0) { return; }

    uint64_t kSizeAligned = CeilAlign(param.kSize, K_ALIGNMENT64);
    uint64_t scaleKGmSize = kSizeAligned / MX_GROUPSIZE;  // K/32
    uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > kSizeAligned ?
                                    (kSizeAligned - kbL1Offset) / MX_GROUPSIZE :
                                    MX_SCALE_K_L1_SIZE / MX_GROUPSIZE;

    // 搬运 ScaleA（per-token scale）: shape [M, K/32]
    // 使用 Dn2Nz 搬运，将 ND 格式的 scale 转为 NZ 格式
    Dn2NzParams scaleAdn2NzParams;
    ConfigScaleDn2NzParams(param.mL1Size, scaleKGmSize, scaleKL1StandardLen,
                           scaleKL1RealSize, scaleAdn2NzParams);
    int64_t scaleAGmOffset = param.mOffset * scaleKGmSize + kbL1Offset / MX_GROUPSIZE;
    DataCopy(mxScaleAL1_[(mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_].ReinterpretCast<half>(),
             mxScaleAGlobal_[scaleAGmOffset], scaleAdn2NzParams);

    // 搬运 ScaleB（per-channel scale）: shape [N, K/32]
    // GMMSQ 特殊处理：ScaleB 也需要分 Swish 和 Gate 两部分搬运
    //   Swish 分支 scaleB: antiquantScaleGm_[swishNOffset * scaleKGmSize + kOffset]
    //   Gate 分支 scaleB:  antiquantScaleGm_[gateNOffset * scaleKGmSize + kOffset]
    // 在 L1 上拼接为完整的 nL1Size = 256 列

    uint64_t swishNOffset = param.nOffset * scaleKGmSize;
    uint64_t gateNOffset = (param.nOffset + param.nSize / 2) * scaleKGmSize;
    uint64_t kScaleOffset = kbL1Offset / MX_GROUPSIZE;

    // Swish 分支 ScaleB（前 128 列）
    Dn2NzParams scaleBdn2NzParamsSwish;
    ConfigScaleDn2NzParams(param.nL1Size / 2, scaleKGmSize, scaleKL1StandardLen,
                           scaleKL1RealSize, scaleBdn2NzParamsSwish);
    int64_t scaleBGmOffsetSwish = swishNOffset + kScaleOffset;
    DataCopy(mxScaleBL1_[(mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_].ReinterpretCast<half>(),
             mxScaleBGlobal_[scaleBGmOffsetSwish], scaleBdn2NzParamsSwish);

    // Gate 分支 ScaleB（后 128 列）
    Dn2NzParams scaleBdn2NzParamsGate;
    ConfigScaleDn2NzParams(param.nL1Size / 2, scaleKGmSize, scaleKL1StandardLen,
                           scaleKL1RealSize, scaleBdn2NzParamsGate);
    int64_t scaleBGmOffsetGate = gateNOffset + kScaleOffset;
    DataCopy(mxScaleBL1_[(mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_ + MX_SCALE_L1_SIZE / 2].ReinterpretCast<half>(),
             mxScaleBGlobal_[scaleBGmOffsetGate], scaleBdn2NzParamsGate);
}
```

### 8.10.4 CopyAAndBiasGmToL1() 方法伪代码

```cpp
__aicore__ inline void CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param,
                                          int64_t kaGmOffset, uint64_t cvLoopIdx)
{
    if (kaGmOffset % param.kaL1Size != 0) { return; }

    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ?
                               param.kSize - kaGmOffset : param.kaL1Size;

    // ND2NZ 搬运 A 矩阵: shape [M, K] → NZ format [M/16, K/16, 16, 16]
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = param.mL1Size;
    nd2nzParams.dValue = kaL1RealSize;
    nd2nzParams.srcDValue = param.kSize;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(aL1_[(aL1BufIdx_ & 1) * aL1DbOffset_],
             xGlobal_[kaGmOffset + param.mOffset * param.kSize], nd2nzParams);

    // K 不对齐时尾部填零
    PadL1IfNotAlign(aL1_, aL1DbOffset_, aL1BufIdx_, kaL1RealSize, param.mL1Size);
}
```

### 8.10.5 LaunchMatmul() 方法伪代码

```cpp
__aicore__ inline void LaunchMatmul(const LocalTensor<xType> &weightL1,
                                    int64_t kbOffset, uint64_t kbL1RealSize,
                                    uint64_t cvLoopIdx, const BasicBlockOffsetParam &param)
{
    // MTE2→MTE1 同步
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);

    uint64_t aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_;

    // L0 计算参数
    L0CopyAndCalcParams l0CopyAndCalcParams;
    l0CopyAndCalcParams.mL0Size = param.mL1Size;
    l0CopyAndCalcParams.mL1Size = param.mL1Size;
    uint64_t kL0Size = (param.mL1Size <= 128 && param.nL1Size <= 128) ? 256 : 128;
    l0CopyAndCalcParams.kL1Size = param.kaL1Size;
    l0CopyAndCalcParams.nL0Size = param.nL1Size;
    l0CopyAndCalcParams.nL1Size = param.nL1Size;
    l0CopyAndCalcParams.scaleKL1Size = MX_SCALE_K_L1_SIZE;

    // L0C 偏移管理（空间不足时回绕）
    if (kbOffset == 0) {
        uint64_t nextL0cBufSize = CeilAlign(l0CopyAndCalcParams.mL0Size, BLOCK_CUBE) *
                                  CeilAlign(l0CopyAndCalcParams.nL0Size, BLOCK_CUBE);
        if (l0cOffset_ + nextL0cBufSize > L0C_BUF_SIZE) { l0cOffset_ = 0; }
    }

    // K 轴 L0 级循环（L1→L0→MMAD 流水）
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += kL0Size) {
        bool isLastL1K = l1KOffset + kL0Size >= kbL1RealSize;
        l0CopyAndCalcParams.isFirstKLoop = (l1KOffset + kbOffset) == 0;
        l0CopyAndCalcParams.isLastKLoop = l1KOffset + kbOffset + kL0Size >= param.kSize;
        uint64_t realL0k = isLastL1K ? kbL1RealSize - l1KOffset : kL0Size;
        l0CopyAndCalcParams.kL0Size = realL0k;

        uint64_t loopId = l0LoopIdx_ % L0_BUF_NUM;

        // 等待 MMA 释放 L0A/L0B
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId);

        // MTE1: L1→L0A（带 MX ScaleA 融合）
        LoadAAndScaleL1ToL0(l0a_[loopId * L0_BUF_OFFSET_B8],
                            aL1_[aL1Offset + mL1AlignSize * ((l1KOffset + kbOffset) % param.kaL1Size)],
                            mxScaleAL1_[(mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_ + ...],
                            l0CopyAndCalcParams);

        // MTE1: L1→L0B（带 MX ScaleB 融合）
        LoadBAndScaleL1ToL0(l0b_[loopId * L0_BUF_OFFSET_B8],
                            weightL1[nL1AlignSize * l1KOffset],
                            mxScaleBL1_[(mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_ + ...],
                            l0CopyAndCalcParams);

        // MMAD 计算
        DoMmadCompute(loopId, cvLoopIdx, l0CopyAndCalcParams);
    }
}
```

### 8.10.6 DoMmadCompute() 方法伪代码

```cpp
__aicore__ inline void DoMmadCompute(uint64_t loopId, uint64_t cvLoopIdx,
                                     const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    // 首次 K 循环加载 Bias Table
    if constexpr (hasBias) {
        if (l0CopyAndCalcParams.isFirstKLoop) {
            LoadBiasToBt(biasTable_[loopId * BIAS_TABLE_OFFSET_B32],
                         biasL1_[(cvLoopIdx & 1) * biasL1DbOffset_], l0CopyAndCalcParams);
        }
    }

    // MTE1→MMA 同步
    SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);

    // MMAD 矩阵乘累加
    if constexpr (hasBias) {
        if (l0CopyAndCalcParams.isFirstKLoop) {
            MmadCompute(l0c_[l0cOffset_], l0a_[loopId * L0_BUF_OFFSET_B8],
                        l0b_[loopId * L0_BUF_OFFSET_B8],
                        biasTable_[loopId * BIAS_TABLE_OFFSET_B32], l0CopyAndCalcParams);
        } else {
            MmadCompute(l0c_[l0cOffset_], l0a_[loopId * L0_BUF_OFFSET_B8],
                        l0b_[loopId * L0_BUF_OFFSET_B8], l0CopyAndCalcParams);
        }
    } else {
        MmadCompute(l0c_[l0cOffset_], l0a_[loopId * L0_BUF_OFFSET_B8],
                    l0b_[loopId * L0_BUF_OFFSET_B8], l0CopyAndCalcParams);
    }

    // 通知 MTE1：MMA 完成，L0A/L0B 可覆盖
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId);
    l0LoopIdx_++;
}
```

### 8.10.7 GetTensorC() 方法伪代码

```cpp
__aicore__ inline void GetTensorC(LocalTensor<yType> &yUb, const BasicBlockOffsetParam &param)
{
    // Fixpipe: L0C → UB（ROW_MAJOR, F32）
    //
    // 关键说明：
    //   1. MMA 计算时，Swish 和 Gate 分支共享同一个 X 矩阵
    //   2. 但 weight 来自 GM 的不同位置（前 N/2 和后 N/2）
    //   3. 在 UB/L1 上，两个分支的 weight 被拼接为一个完整的 256 列基本块
    //   4. 因此 MMAD 计算时，nL0Size = 256（完整基本块）
    //   5. L0C 输出到 UB 后，再对 N 轴切分：
    //        前 128 列 → Swish 分支
    //        后 128 列 → Gate 分支
    //
    FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = param.mL1Size;   // M 方向实际计算大小
    fixL0CToDstParams.nL0Size = param.nL1Size;   // N 方向实际计算大小（= 256）
    fixL0CToDstParams.outNSize = 256;             // 输出 N 方向总大小（用于 dstStride）
    fixL0CToDstParams.splitNSize = 0;             // 不切分
    fixL0CToDstParams.dstNdStride = 0;

    FixL0CToDst(yUb, l0c_[l0cOffset_], fixL0CToDstParams);

    // 更新 L0C 偏移
    l0cOffset_ += CeilAlign(fixL0CToDstParams.mL0Size, BLOCK_CUBE) *
                  CeilAlign(fixL0CToDstParams.nL0Size, BLOCK_CUBE);
}
```

### 8.10.8 与 GMMFR 的差异点说明

| 差异点 | GMMFR | GMMSQ | 说明 |
|--------|-------|-------|------|
| 类名 | `GMMFRWeightQuantCubeCompute` | `GMMSQWeightQuantCubeCompute` | 命名空间和类名变化 |
| `GetTensorC()` 目标地址 | `ubOutputF32Buffer_` | `yF32Buffer_` | 仅 UB 地址名称变化，语义完全相同 |
| 其他所有函数 | - | **完全不变** | L1/L0 buffer 分配、Scale 搬运、A 矩阵搬运、MMA 计算、Fixpipe 全部复用 |

**结论**：CubeCompute 层几乎不需要改动，只需调整类名和 `GetTensorC()` 的目标 UB 地址。

---

## 8.11 VecCompute 层编码方案

### 8.11.1 类定义和模板参数

```cpp
namespace GMMSQ {

// UB Buffer 信息结构体（GMMSQ 专用）
struct GmmsqUbBufferInfo {
    uint64_t weightHighBitBufferNum;
    uint64_t weightHighBitTotalSize;
    uint64_t weightHighBitSingleBufferSize;
    uint64_t weightLowbitTotalSize;
    uint64_t weightLowBitSingleBufferSize;
    uint64_t biasTotalSize;
    uint64_t biasSingleBufferSize;
    uint64_t biasRescaleTotalSize;
    uint64_t biasRescaleSingleBufferSize;
    // 新增 GMMSQ 专用 Buffer
    uint64_t maxB16TotalSize;        // ComputeMaxExp 中间结果
    uint64_t scaleE8M0TotalSize;     // MX 量化 scale 输出
    uint64_t yFp8TotalSize;          // 量化后 FP8 输出
};

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr GmmsqUbBufferInfo GetGMMSQMxA8W4BufferInfo()
{
    return {
        .weightHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
        .weightHighBitTotalSize = 128 * GetKBUnit<int8_t>(),   // 128KB（从 160KB 减少）
        .weightHighBitSingleBufferSize = 128 * GetKBUnit<int8_t>() / QUADRUPLE_BUFFER_NUM,
        .weightLowbitTotalSize = 64 * GetKBUnit<int8_t>(),     // 64KB（从 80KB 减少）
        .weightLowBitSingleBufferSize = 64 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
        .biasTotalSize = 1 * GetKBUnit<half>(),
        .biasSingleBufferSize = 1 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
        .biasRescaleTotalSize = 1 * GetKBUnit<half>(),
        .biasRescaleSingleBufferSize = 1 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
        .maxB16TotalSize = 2 * GetKBUnit<half>(),              // 2KB
        .scaleE8M0TotalSize = 16 * GetKBUnit<fp8_e8m0_t>(),   // 16KB
        .yFp8TotalSize = 32 * GetKBUnit<fp8_e4m3fn_t>()       // 32KB（复用 weightLowBit 空间）
    };
}

#define GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM                                                     \
    template <typename xType, typename wType, typename biasType, typename yType,               \
              typename yScaleType, bool hasBias, const WqmmConfig &wqmmConfig,                 \
              const VecAntiQuantConfig &vecConfig>

#define GMMSQ_WQ_VEC_COMPUTE_CLASS                                                              \
    GMMSQWeightQuantVecCompute<xType, wType, biasType, yType, yScaleType,                      \
                               hasBias, wqmmConfig, vecConfig>

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
class GMMSQWeightQuantVecCompute {
public:
    __aicore__ inline GMMSQWeightQuantVecCompute(){};

    // === 初始化 ===
    __aicore__ inline void Init(__gm__ yType *yFp32Addr, __gm__ yScaleType *yScaleAddr);

    // === GM 地址更新 ===
    __aicore__ inline void UpdateGlobalAddr(__gm__ wType *weight, __gm__ biasType *bias,
                                            const bool weightL2Cacheable);

    // === 保留的反量化函数（从 GMMFR 直接复用）===
    __aicore__ inline void WaitVToMTE2();
    __aicore__ inline void SetVToMTE2();
    __aicore__ inline void CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset, uint64_t kL1Offset,
                                      uint64_t biasGmOffset, uint64_t biasRealSize,
                                      const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WeightAntiQuantComputeNzNk(uint64_t kRealSize, uint64_t biasRealSize,
                                                      uint64_t kGmOffset,
                                                      const LocalTensor<xType> &weightHighBitL1,
                                                      const LocalTensor<biasType> &biasL1,
                                                      const BasicBlockOffsetParam &offsetParam);

    // === 新增的后处理函数 ===
    // SwiGLU: Swish(yF32[:, 0:N/2]) * yF32[:, N/2:N]
    __aicore__ inline void SwiGLU(const LocalTensor<float> &yF32,
                                  const BasicBlockOffsetParam &param);

    // MX 量化步骤 1: 计算每 32 个元素的 max exponent
    __aicore__ inline void ComputeMaxExp(const LocalTensor<float> &yF32,
                                         const BasicBlockOffsetParam &param);

    // MX 量化步骤 2: 计算 shared_exp → E8M0 scale
    __aicore__ inline void ComputeScale(const BasicBlockOffsetParam &param);

    // MX 量化步骤 3: 量化到 FP8_E4M3FN
    __aicore__ inline void ComputeDataForQuantTargetFp8(const LocalTensor<float> &yF32,
                                                        const BasicBlockOffsetParam &param);

    // MTE3 写出量化结果
    __aicore__ inline void CopyOutputFromUb2Gm(const BasicBlockOffsetParam &param);

    // MTE3 写出 scale
    __aicore__ inline void CopyScaleFromUb2Gm(const BasicBlockOffsetParam &param);

    // === 收尾 ===
    __aicore__ inline void End();

private:
    // 反量化参数准备
    __aicore__ inline void AntiQuantProcessNzMxA8W4(uint64_t biasRealSize, uint64_t ubMte2KSize,
                                                    uint64_t kGmOffset,
                                                    const BasicBlockOffsetParam &offsetParam);
    // UB→L1 搬运反量化后的 weight
    __aicore__ inline void CopyWeightHighBitForAligned(uint64_t antiQuantRealN, uint64_t antiQuantRealK,
                                                       const LocalTensor<xType> &weightHighBitL1);

    // UB Buffer 信息
    constexpr static GmmsqUbBufferInfo UB_BUFFER_INFO = GetGMMSQMxA8W4BufferInfo<vecConfig>();

    // 循环索引
    uint64_t weightMte2LoopIdx_ = 0;   // MTE2 搬运计数
    uint64_t ubComputeLoopIdx_ = 0;    // VF 计算计数

    // Event ID（与 GMMFR 相同）
    static constexpr uint32_t EVENT_ID_V_TO_MTE2 = 0;
    static constexpr uint32_t EVENT_ID_MTE2_TO_V = 0;
    static constexpr uint32_t EVENT_ID_MTE3_TO_V = 0;
    static constexpr uint32_t EVENT_ID_V_TO_MTE3 = 0;

    // GM 地址
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yFp32Global_;
    GlobalTensor<yScaleType> yScaleGlobal_;  // 新增

    // UB Buffer
    LocalTensor<int8_t> weightLowBit_;
    LocalTensor<xType> weightHighBit_;
    LocalTensor<biasType> bias_;
    LocalTensor<biasType> biasRescale_;
    // 新增 GMMSQ 专用 UB Buffer
    LocalTensor<half> maxB16Buffer_;
    LocalTensor<fp8_e8m0_t> scaleE8M0Buffer_;
    LocalTensor<fp8_e4m3fn_t> yFp8Buffer_;

    // 常量
    constexpr static uint32_t C0_SIZE = C0_SIZE_B8;  // 32
    constexpr static uint64_t VEC_REG_ELEM = VECTOR_REG_WIDTH;
    constexpr static uint64_t UB_AVAILABLE_SIZE = 248 * GetKBUnit<int8_t>();
};
```

### 8.11.2 Init() 方法伪代码

```cpp
__aicore__ inline void Init(__gm__ yType *yFp32Addr, __gm__ yScaleType *yScaleAddr)
{
    yFp32Global_.SetGlobalBuffer(yFp32Addr);
    yScaleGlobal_.SetGlobalBuffer(yScaleAddr);  // 新增

    // === UB Buffer 分配（重新布局） ===
    // 布局: [weightLowBit 64KB][weightHighBit 128KB][bias 1KB][biasRescale 1KB]
    //       [maxB16 2KB][scaleE8M0 16KB]
    //       yFp8Buffer_ 复用 weightLowBit 空间（PhaseB 阶段 weightLowBit 已不再使用）
    //       yF32Buffer_ 复用 weightHighBit 空间（在 BasicBlock 层分配）

    weightLowBit_ = LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightLowbitTotalSize);
    uint64_t ubOffset = UB_BUFFER_INFO.weightLowbitTotalSize;

    weightHighBit_ = LocalTensor<xType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.weightHighBitTotalSize);
    ubOffset += UB_BUFFER_INFO.weightHighBitTotalSize;

    bias_ = LocalTensor<biasType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.biasTotalSize);
    ubOffset += UB_BUFFER_INFO.biasTotalSize * sizeof(biasType);

    biasRescale_ = LocalTensor<biasType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.biasRescaleTotalSize);
    ubOffset += UB_BUFFER_INFO.biasRescaleTotalSize * sizeof(biasType);

    // 新增 Buffer
    maxB16Buffer_ = LocalTensor<half>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.maxB16TotalSize);
    ubOffset += UB_BUFFER_INFO.maxB16TotalSize * sizeof(half);

    scaleE8M0Buffer_ = LocalTensor<fp8_e8m0_t>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.scaleE8M0TotalSize);
    ubOffset += UB_BUFFER_INFO.scaleE8M0TotalSize * sizeof(fp8_e8m0_t);

    // yFp8Buffer_ 复用 weightLowBit_ 的前 32KB 空间
    yFp8Buffer_ = weightLowBit_.ReinterpretCast<fp8_e4m3fn_t>();

    // === Event 初始化 ===
    // MTE3→V: 四缓冲
    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }
    // V→MTE2: ubMte2BufferNum 缓冲
    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}
```

### 8.11.3 CopyGmToUb() 方法伪代码

```cpp
__aicore__ inline void CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset, uint64_t kL1Offset,
                                  uint64_t biasGmOffset, uint64_t biasRealSize,
                                  const BasicBlockOffsetParam &offsetParam)
{
    // Bias 搬运（仅首次 K 循环）
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            DataCopyPad2D(bias_[(weightMte2LoopIdx_ & QUADRUPLE_BUFFER_MASK) *
                                UB_BUFFER_INFO.biasSingleBufferSize],
                          biasGlobal_[biasGmOffset], 1, biasRealSize, biasRealSize, biasRealSize);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        }
    }

    if (offsetParam.nL1Size == 0 || ubMte2KSize == 0) { return; }

    // Weight 搬运（4bit，NZ 格式）
    // GMMSQ 特殊处理：Swish 和 Gate 分支的 weight 分别存储在 GM 的不同位置，
    // 但在 UB/L1 上拼接为一个完整的基本块（baseN=256）
    //
    // 全局 N 切分：
    //   Swish 分支: W[0 : N/2, :]
    //   Gate 分支:  W[N/2 : N, :]
    //
    // 当 baseN=256 时：
    //   前 128 列来自 Swish 分支的 nOffset 处
    //   后 128 列来自 Gate 分支的 (nOffset + N/2) 处
    //
    // 分两次搬运到 UB，在 UB 上拼接为连续的 256 列
    //
    // Swish 分支 GM 地址: wGlobal_[(kGmOffset + kL1Offset) * nAlign + nOffset * C0_SIZE]
    // Gate 分支 GM 地址:  wGlobal_[(kGmOffset + kL1Offset) * nAlign + (nOffset + N/2) * C0_SIZE]
    //
    // 第一次搬运：Swish 分支（前 128 列）
    uint64_t swishNOffset = offsetParam.nOffset * static_cast<uint64_t>(C0_SIZE);
    uint64_t gateNOffset = (offsetParam.nOffset + offsetParam.nSize / 2) * static_cast<uint64_t>(C0_SIZE);
    uint64_t kOffset = (kGmOffset + kL1Offset) * offsetParam.nAlign;

    // Swish 分支: 搬运前 128 列到 UB 的前半部分
    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                      UB_BUFFER_INFO.weightLowBitSingleBufferSize].ReinterpretCast<wType>(),
        wGlobal_[kOffset + swishNOffset],
        CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        128 * C0_SIZE,  // Swish 分支 128 列
        256 * C0_SIZE,  // UB 上总宽度（Swish + Gate）
        offsetParam.nAlign * C0_SIZE);

    // Gate 分支: 搬运后 128 列到 UB 的后半部分
    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                      UB_BUFFER_INFO.weightLowBitSingleBufferSize + 128 * C0_SIZE].ReinterpretCast<wType>(),
        wGlobal_[kOffset + gateNOffset],
        CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        128 * C0_SIZE,  // Gate 分支 128 列
        256 * C0_SIZE,  // UB 上总宽度（Swish + Gate）
        offsetParam.nAlign * C0_SIZE);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}
```

### 8.11.4 AntiQuantProcessNzMxA8W4() 方法伪代码

```cpp
__aicore__ inline void AntiQuantProcessNzMxA8W4(uint64_t biasRealSize, uint64_t ubMte2KSize,
                                                uint64_t kGmOffset,
                                                const BasicBlockOffsetParam &offsetParam)
{
    MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams;

    uint64_t ubMte2BufferIdx = weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1);

    // 输入: 4bit weight 在 weightLowBit_ 中
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ wType *)weightLowBit_[ubMte2BufferIdx * UB_BUFFER_INFO.weightLowBitSingleBufferSize].GetPhyAddr();

    // 输出: 8bit weight 写入 weightHighBit_（四缓冲跳写）
    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ xType *)weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) *
                                         VECTOR_REG_WIDTH].GetPhyAddr();

    // N 方向对齐大小
    mxA8W4NzParams.nRealSizeAlign = CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));

    // K 循环次数
    mxA8W4NzParams.loopKNum = CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE));

    // 内循环次数（N 方向）
    mxA8W4NzParams.innerLoopNum =
        CeilDivide(CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                   static_cast<uint64_t>(VECTOR_REG_WIDTH));

    // 跳写 stride（避免 UB Bank 冲突）
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.weightHighBitBufferNum;
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;

    // Bias 反量化参数（如有）
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            mxA8W4NzParams.biasInUbAddr = (__ubuf__ biasType *)bias_.GetPhyAddr(...);
            mxA8W4NzParams.biasOutUbAddr = (__ubuf__ biasType *)biasRescale_.GetPhyAddr(...);
            mxA8W4NzParams.biasLoopNum = CeilDivide(biasRealSize, VEC_MAX_ELEM_B16);
            FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, true>(mxA8W4NzParams);
        } else {
            FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, false>(mxA8W4NzParams);
        }
    } else {
        FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, false>(mxA8W4NzParams);
    }
}
```

### 8.11.5 SwiGLU 计算伪代码（新增）

```cpp
__aicore__ inline void SwiGLU(const LocalTensor<float> &yF32,
                              const BasicBlockOffsetParam &param)
{
    // 输入: yF32[M, N]，其中 N = nL1Size = 256
    //       前 N/2 = 128 为 Swish 分支（对应前 N/2 的 weight）
    //       后 N/2 = 128 为 Gate 分支（对应后 N/2 的 weight）
    // 输出: yF32[M, N/2]（原地覆盖前 128 个元素）
    //
    // 关键说明：
    //   1. MMA 计算时，Swish 和 Gate 分支共享同一个 X 矩阵
    //   2. 但 weight 来自 GM 的不同位置（前 N/2 和后 N/2）
    //   3. 在 UB/L1 上，两个分支的 weight 被拼接为一个完整的 256 列基本块
    //   4. L0C 输出到 UB 后，再对 N 轴切分：
    //        前 128 列 → Swish 分支输入
    //        后 128 列 → Gate 分支输入
    //   5. SwiGLU 计算完成后，输出大小为 M × (N/2)
    //
    // 算法: for m in 0..M-1:
    //         for n in 0..127:
    //           swish_val = yF32[m, n]
    //           gate_val  = yF32[m, n + 128]
    //           sigmoid = 1.0 / (1.0 + exp(-swish_val))
    //           result = swish_val * sigmoid * gate_val
    //           yF32[m, n] = result

    uint64_t mRealSize = param.mL1Size;
    uint64_t nHalfSize = param.nL1Size / 2;  // 128

    // VF 实现
    SwiGLUVf(static_cast<uint16_t>(mRealSize), nHalfSize,
               (__ubuf__ float *)yF32.GetPhyAddr());
}
```

### 8.11.6 MX Quant 计算伪代码（新增）

```cpp
// === ComputeMaxExp: 计算每 32 个元素的 max exponent ===
__aicore__ inline void ComputeMaxExp(const LocalTensor<float> &yF32,
                                     const BasicBlockOffsetParam &param)
{
    // 输入: yF32[M, N/2]，其中 N/2 = 128
    // 输出: maxB16Buffer_[M, ceil(N/2/32)] = maxB16Buffer_[M, 4]
    //
    // 算法: 对每 32 个元素:
    //         取 abs()
    //         提取指数位（float 的 bits[30:23]）
    //         求最大值
    //         存入 maxB16（B16 格式）

    uint64_t mRealSize = param.mL1Size;
    uint64_t nHalfSize = param.nL1Size / 2;  // 128

    ComputeMaxExpVf(static_cast<uint16_t>(mRealSize), nHalfSize,
                      (__ubuf__ float *)yF32.GetPhyAddr(),
                      (__ubuf__ half *)maxB16Buffer_.GetPhyAddr());
}

// === ComputeScale: 计算 shared_exp → E8M0 scale ===
__aicore__ inline void ComputeScale(const BasicBlockOffsetParam &param)
{
    // 输入: maxB16Buffer_[M, ceil(N/2/32)]
    // 输出: scaleE8M0Buffer_[M, ceil(N/2/32)]
    //
    // 算法: shared_exp = round(log2(max_val)) - emax
    //       emax = 8 (for FP8_E4M3FN)
    //       scaleE8M0 = shared_exp（E8M0 格式就是纯指数）

    uint64_t mRealSize = param.mL1Size;
    uint64_t nGroupCount = CeilDivide(param.nL1Size / 2, static_cast<uint64_t>(MX_GROUPSIZE));

    ComputeScaleVf(static_cast<uint16_t>(mRealSize), nGroupCount,
                     (__ubuf__ half *)maxB16Buffer_.GetPhyAddr(),
                     (__ubuf__ fp8_e8m0_t *)scaleE8M0Buffer_.GetPhyAddr());
}

// === ComputeDataForQuantTargetFp8: 量化到 FP8_E4M3FN ===
__aicore__ inline void ComputeDataForQuantTargetFp8(const LocalTensor<float> &yF32,
                                                    const BasicBlockOffsetParam &param)
{
    // 输入: yF32[M, N/2], scaleE8M0Buffer_[M, ceil(N/2/32)]
    // 输出: yFp8Buffer_[M, N/2]
    //
    // 算法: 对每 32 个元素:
    //         scale_f = 2^(scaleE8M0[m, groupIdx])
    //         yF32[m, n] /= scale_f
    //         量化到 FP8_E4M3FN（舍入到最近偶数）

    uint64_t mRealSize = param.mL1Size;
    uint64_t nHalfSize = param.nL1Size / 2;  // 128

    ComputeDataForQuantTargetFp8Vf(
        static_cast<uint16_t>(mRealSize), nHalfSize,
        (__ubuf__ float *)yF32.GetPhyAddr(),
        (__ubuf__ fp8_e8m0_t *)scaleE8M0Buffer_.GetPhyAddr(),
        (__ubuf__ fp8_e4m3fn_t *)yFp8Buffer_.GetPhyAddr());
}
```

### 8.11.7 MTE3 写出伪代码（新增）

```cpp
// === CopyOutputFromUb2Gm: MTE3 写出 yFp8 到 GM ===
__aicore__ inline void CopyOutputFromUb2Gm(const BasicBlockOffsetParam &param)
{
    // 输出地址: yFp32Global_[mOffset * (nSize/2) + nOffset/2]
    // 输出大小: mL1Size × (nL1Size/2) 个 FP8 元素
    uint64_t mRealSize = param.mL1Size;
    uint64_t nHalfSize = param.nL1Size / 2;

    // 计算 GM 偏移（输出 N/2）
    uint64_t yGmOffset = param.mOffset * (param.nSize / 2) + param.nOffset / 2;

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    // 主块搬运
    DataCopyPad2D(yFp32Global_[yGmOffset], yFp8Buffer_, mRealSize,
                  nHalfSize, nHalfSize, param.nSize / 2);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
}

// === CopyScaleFromUb2Gm: MTE3 写出 scaleE8M0 到 GM ===
__aicore__ inline void CopyScaleFromUb2Gm(const BasicBlockOffsetParam &param)
{
    // 输出地址: yScaleGlobal_[mOffset * ceil((nSize/2)/64) + (nOffset/2)/64]
    // 输出大小: mL1Size × ceil((nL1Size/2)/64) 个 E8M0 元素
    uint64_t mRealSize = param.mL1Size;
    uint64_t nGroupCount = CeilDivide(param.nL1Size / 2, static_cast<uint64_t>(MX_GROUPSIZE));
    uint64_t nTotalGroupCount = CeilDivide(param.nSize / 2, static_cast<uint64_t>(MX_GROUPSIZE));

    uint64_t scaleGmOffset = param.mOffset * nTotalGroupCount +
                             (param.nOffset / 2) / MX_GROUPSIZE;

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    DataCopyPad2D(yScaleGlobal_[scaleGmOffset], scaleE8M0Buffer_, mRealSize,
                  nGroupCount, nGroupCount, nTotalGroupCount);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
}
```

### 8.11.8 与 GMMFR 的差异点说明

| 差异点 | GMMFR | GMMSQ | 说明 |
|--------|-------|-------|------|
| 类名 | `GMMFRWeightQuantVecCompute` | `GMMSQWeightQuantVecCompute` | 命名空间和类名变化 |
| 模板参数 | 含 `sharedInputDType`, `logitsType`, `rowIndexType` | 去除这 3 个，新增 `yScaleType` | 后处理逻辑变化 |
| `Init()` 参数 | `(sharedInputWeight, yFp32Addr)` | `(yFp32Addr, yScaleAddr)` | 去除 sharedInputWeight，新增 yScale |
| UB Buffer | weightLowBit 80KB, weightHighBit 160KB, rowIndex 2KB, logits 1KB | weightLowBit 64KB, weightHighBit 128KB, maxB16 2KB, scaleE8M0 16KB, yFp8 32KB（复用） | 重新布局，新增 4 个 Buffer |
| 去除的函数 | `MulLogits`, `RoutingYToGm`, `CopyRowIndexLogitsGmToUb`, `CopyShareInputGmToUb`, `CopyShareInputUbToGm`, `CastAndMulWithSharedWeightVf`, `InitGmToZero` | - | GMMSQ 不需要 Routing 和 sharedInput |
| 新增的函数 | - | `SwiGLU`, `ComputeMaxExp`, `ComputeScale`, `ComputeDataForQuantTargetFp8`, `CopyOutputFromUb2Gm`, `CopyScaleFromUb2Gm` | SwiGLU + MX Quant 后处理 |
| 保留的函数 | `CopyGmToUb`, `WeightAntiQuantComputeNzNk`, `AntiQuantProcessNzMxA8W4`, `CopyWeightHighBitForAligned`, `WaitVToMTE2`, `SetVToMTE2`, `End` | **完全相同** | 反量化流程不变 |

---

## 8.12 Buffer 分配详细方案

### 8.12.1 L1 Buffer 分配示意图

无 Bias 场景（MxA8W4 默认场景）：

```
L1 Address Space (0 ~ 512 KB):
┌────────────────────────────────────────────────────────────────────────────┐
│ Offset    │ Size    │ Buffer          │ Purpose                             │
├───────────┼─────────┼─────────────────┼─────────────────────────────────────┤
│ 0x00000   │ 64 KB   │ WeightL1_P0     │ AIV MTE3 写入反量化后的 Weight       │
│           │         │                 │ AIC MMA 读取（双缓冲 P0）             │
├───────────┼─────────┼─────────────────┼─────────────────────────────────────┤
│ 0x10000   │ 32 KB   │ ScaleAL1_P0     │ AIC MTE1 写入 MX ScaleA（per-token） │
│           │         │                 │ AIC L0 加载（双缓冲 P0）              │
├───────────┼─────────┼─────────────────┼─────────────────────────────────────┤
│ 0x18000   │ 32 KB   │ ScaleBL1_P0     │ AIC MTE1 写入 MX ScaleB（per-channel）│
│           │         │                 │ AIC L0 加载（双缓冲 P0）              │
├───────────┼─────────┼─────────────────┼─────────────────────────────────────┤
│ 0x20000   │ 128 KB  │ AL1_P0          │ AIC MTE1 写入 A 矩阵（ND2NZ）        │
│           │         │                 │ AIC L0 加载（双缓冲 P0）              │
├───────────┼─────────┼─────────────────┼─────────────────────────────────────┤
│ 0x40000   │ 128 KB  │ AL1_P1          │ AIC MTE1 写入 A 矩阵（双缓冲 P1）    │
├───────────┼─────────┼─────────────────┼─────────────────────────────────────┤
│ 0x60000   │ 32 KB   │ ScaleBL1_P1     │ MX ScaleB 双缓冲 P1                  │
├───────────┼─────────┼─────────────────┼─────────────────────────────────────┤
│ 0x68000   │ 32 KB   │ ScaleAL1_P1     │ MX ScaleA 双缓冲 P1                  │
├───────────┼─────────┼─────────────────┼─────────────────────────────────────┤
│ 0x70000   │ 64 KB   │ WeightL1_P1     │ Weight 双缓冲 P1                     │
└───────────┴─────────┴─────────────────┴─────────────────────────────────────┘

双缓冲切换逻辑:
  cvLoopIdx_ & 1 == 0 → 使用 P0（AIC 读取 WeightL1_P0，AIV 写入 WeightL1_P1）
  cvLoopIdx_ & 1 == 1 → 使用 P1（AIC 读取 WeightL1_P1，AIV 写入 WeightL1_P0）
```

### 8.12.2 UB Buffer 分配示意图

```
UB Address Space (0 ~ 248 KB):
┌────────────────────────────────────────────────────────────────────────────┐
│ Offset    │ Size    │ Buffer            │ Purpose                           │
├───────────┼─────────┼───────────────────┼───────────────────────────────────┤
│ 0x00000   │ 64 KB   │ weightLowBit_     │ MTE2 搬运 4bit Weight（GM→UB）     │
│           │         │ [buf0][buf1]      │ 双缓冲/四缓冲（ubMte2BufferNum）    │
│           │         │                   │ ↓ PhaseB 阶段复用为 yFp8Buffer_    │
│           │         │                   │   （32KB，前一半空间）              │
├───────────┼─────────┼───────────────────┼───────────────────────────────────┤
│ 0x10000   │ 128 KB  │ weightHighBit_    │ VF 反量化输出（4bit→8bit）          │
│           │         │ [buf0][buf1]      │ 四缓冲跳写（避免 Bank 冲突）        │
│           │         │ [buf2][buf3]      │ MTE3 搬运 UB→L1                    │
│           │         │                   │ ↓ PhaseB 阶段复用为 yF32Buffer_    │
│           │         │                   │   （MMA Fixpipe 输出，128KB）       │
├───────────┼─────────┼───────────────────┼───────────────────────────────────┤
│ 0x30000   │ 1 KB    │ bias_             │ MTE2 搬运 Bias（GM→UB）             │
│           │         │                   │ 双缓冲/四缓冲                      │
├───────────┼─────────┼───────────────────┼───────────────────────────────────┤
│ 0x30400   │ 1 KB    │ biasRescale_      │ VF Bias 反量化输出                 │
│           │         │                   │ MTE3 搬运 UB→L1                    │
├───────────┼─────────┼───────────────────┼───────────────────────────────────┤
│ 0x30800   │ 2 KB    │ maxB16Buffer_     │ ComputeMaxExp 中间结果              │
│           │         │                   │ M × ceil(N/2/32) 个 B16 元素       │
├───────────┼─────────┼───────────────────┼───────────────────────────────────┤
│ 0x31000   │ 16 KB   │ scaleE8M0Buffer_  │ MX 量化 scale 输出                 │
│           │         │                   │ M × ceil(N/2/32) 个 E8M0 元素      │
├───────────┼─────────┼───────────────────┼───────────────────────────────────┤
│ 0x35000   │ (空闲)  │                   │ 剩余约 36KB                        │
└───────────┴─────────┴───────────────────┴───────────────────────────────────┘

总使用量: 64 + 128 + 1 + 1 + 2 + 16 = 212 KB < 248 KB ✅

复用策略:
  1. yF32Buffer_ 复用 weightHighBit_ 空间（128KB）
     - 时机: PhaseB 阶段，weightHighBit 已全部搬运到 L1
     - 安全条件: 确保 MTE3 搬运 weightHighBit→L1 全部完成（WaitFlag<MTE3_V>）

  2. yFp8Buffer_ 复用 weightLowBit_ 前 32KB 空间
     - 时机: PhaseB 阶段，weightLowBit 已不再使用
     - 安全条件: PhaseB 阶段无 MTE2 搬运 weightLowBit 的操作
```

### 8.12.3 L0 Buffer 分配示意图

```
L0 Address Space:
┌────────────────────────────────────────────────────────────────────────────┐
│ Buffer       │ Position │ Size    │ Purpose                                 │
├──────────────┼──────────┼─────────┼─────────────────────────────────────────┤
│ L0A buf0     │ A2[0]    │ 32 KB   │ A 矩阵 tile（MTE1 L1→L0A 加载）          │
│ L0A buf1     │ A2[32K]  │ 32 KB   │ 双缓冲 P1                               │
├──────────────┼──────────┼─────────┼─────────────────────────────────────────┤
│ L0B buf0     │ B2[0]    │ 32 KB   │ B 矩阵 tile（MTE1 L1→L0B 加载）          │
│ L0B buf1     │ B2[32K]  │ 32 KB   │ 双缓冲 P1                               │
├──────────────┼──────────┼─────────┼─────────────────────────────────────────┤
│ L0C          │ CO1      │ 256 KB  │ MMA 累加结果（256×256 float）            │
│              │          │         │ 支持多个 BasicBlock 结果暂存              │
├──────────────┼──────────┼─────────┼─────────────────────────────────────────┤
│ BiasTable    │ C2[0]    │ 2 KB    │ Bias 查表（双缓冲）                       │
│ BiasTable    │ C2[512]  │ 2 KB    │ 双缓冲 P1                               │
└──────────────┴──────────┴─────────┴─────────────────────────────────────────┘
```

### 8.12.4 各 Buffer 详细说明

| Buffer | 位置 | 大小 | 缓冲数 | 切换索引 | 用途 |
|--------|------|------|--------|----------|------|
| WeightL1 | L1 | 64KB × 2 | 2 | `cvLoopIdx_ & 1` | AIV MTE3 写入反量化 Weight → AIC MMA 读取 |
| ScaleAL1 | L1 | 32KB × 2 | 2 | `mxScaleBufIdx_ & 1` | AIC MTE1 写入 MX ScaleA → L0 加载 |
| ScaleBL1 | L1 | 32KB × 2 | 2 | `mxScaleBufIdx_ & 1` | AIC MTE1 写入 MX ScaleB → L0 加载 |
| AL1 | L1 | 128KB × 2 | 2 | `aL1BufIdx_ & 1` | AIC MTE1 写入 A 矩阵（ND2NZ）→ L0 加载 |
| weightLowBit | UB | 64KB | 2/4 | `weightMte2LoopIdx_ % N` | MTE2 搬运 4bit Weight → VF 消费 |
| weightHighBit | UB | 128KB | 4 | `ubComputeLoopIdx_ & 3` | VF 反量化输出 → MTE3 搬运到 L1 |
| bias | UB | 1KB | 2/4 | `weightMte2LoopIdx_ & 3` | MTE2 搬运 Bias → VF 消费 |
| biasRescale | UB | 1KB | 2/4 | `ubComputeLoopIdx_ & 3` | VF Bias 反量化输出 → MTE3 搬运到 L1 |
| maxB16Buffer | UB | 2KB | 1 | - | ComputeMaxExp 中间结果 |
| scaleE8M0Buffer | UB | 16KB | 1 | - | MX 量化 scale 输出 |
| yFp8Buffer | UB | 32KB（复用） | 1 | - | 量化后 FP8 输出（复用 weightLowBit 空间） |
| yF32Buffer | UB | 128KB（复用） | 1 | - | MMA Fixpipe 输出（复用 weightHighBit 空间） |
| L0A | L0 | 32KB × 2 | 2 | `l0LoopIdx_ % 2` | MTE1 加载 A tile → MMA 计算 |
| L0B | L0 | 32KB × 2 | 2 | `l0LoopIdx_ % 2` | MTE1 加载 B tile → MMA 计算 |
| L0C | L0 | 256KB | 1 | `l0cOffset_` 递增 | MMA 累加结果 |
| BiasTable | L0 | 2KB × 2 | 2 | `loopId` | Bias 查表 |

---

## 8.13 同步机制

### 8.13.1 跨核同步信号定义

跨核同步信号定义在 `tool.h` 中，GMMSQ 与 GMMFR **完全相同**：

```cpp
// 同步信号常量
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;          // AIV → AIC
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;          // AIC → AIV
static constexpr uint64_t SYNC_AIC_FIX_AIV_VF_FLAG = 3;   // AIC → AIV (Fixpipe)
static constexpr uint64_t SYNC_AIV_MTE3_AIC_FIX_FLAG = 4; // AIV → AIC (Fixpipe)
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t FLAG_ID_MAX = 16;
```

| 信号 | 值 | 方向 | 关联 Pipe | 语义 |
|------|-----|------|-----------|------|
| `SYNC_AIV_AIC_FLAG` | 8 | AIV → AIC | PIPE_MTE1 / PIPE_MTE3 | AIV 完成 Weight 反量化并写入 L1，AIC 可以开始 MMA |
| `SYNC_AIC_AIV_FLAG` | 9 | AIC → AIV | PIPE_MTE1 / PIPE_MTE3 | AIC 完成 MMA 计算，AIV 可以继续搬运下一块 Weight |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG` | 4 | AIV → AIC | PIPE_MTE3 / PIPE_FIX | AIV 已取走上一个 BasicBlock 的 UB 结果，AIC 可以 Fixpipe |
| `SYNC_AIC_FIX_AIV_VF_FLAG` | 3 | AIC → AIV | PIPE_FIX / PIPE_V | AIC Fixpipe 完成，UB 上的 MMA 结果已就绪，AIV 可以做 SwiGLU+Quant |

### 8.13.2 Pipe Event 定义

#### AIC 侧 Event

| Event 名称 | Event ID | 方向 | 缓冲数 | 用途 |
|-----------|----------|------|--------|------|
| `EVENT_ID_MTE1_TO_MTE2` | 3 | MTE1 → MTE2 | 2 | A 矩阵 MTE1 搬运完成后通知 MTE2 |
| `EVENT_ID_SCALE_MTE1_TO_MTE2` | 5 | MTE1 → MTE2 | 2 | MX Scale MTE1 搬运完成后通知 MTE2 |
| `EVENT_ID_M_TO_MTE1` | 3 | MMA → MTE1 | 2 | MMA 完成后通知 MTE1 可覆盖 L0A/L0B |
| `EVENT_ID_MTE1_TO_M` | 3 | MTE1 → MMA | 1 | MTE1 加载 L0A/L0B 完成后通知 MMA |
| `EVENT_ID_MTE2_TO_MTE1` | 3 | MTE2 → MTE1 | 1 | MTE2（AIV 侧 Weight 反量化）完成后通知 MTE1 |

#### AIV 侧 Event

| Event 名称 | Event ID | 方向 | 缓冲数 | 用途 |
|-----------|----------|------|--------|------|
| `EVENT_ID_MTE3_TO_V` | 0 | MTE3 → VF | 4 | weightHighBit MTE3 搬运完成后通知 VF 可覆盖 |
| `EVENT_ID_V_TO_MTE2` | 0 | VF → MTE2 | 2/4 | weightLowBit VF 消费完后通知 MTE2 可覆盖 |
| `EVENT_ID_MTE2_TO_V` | 0 | MTE2 → VF | 1 | weightLowBit MTE2 搬运完成后通知 VF 可读 |
| `EVENT_ID_V_TO_MTE3` | 0 | VF → MTE3 | 1 | VF 反量化完成后通知 MTE3 可搬运到 L1 |

### 8.13.3 同步时序说明

#### K 循环内同步时序（Weight 反量化 ↔ MMA 计算）

```
时间 →

AIC (Cube Core)                              AIV (Vector Core)
│                                            │
│  WaitScaleMTE1ToMTE2()                     │  WaitVToMTE2()
│  CopyMxScaleGmToL1()  ← MTE1: Scale→L1    │  CopyGmToUb()  ← MTE2: Weight→UB
│  WaitMTE1ToMTE2()                          │    SetFlag<MTE2_V> / WaitFlag<MTE2_V>
│  CopyAAndBiasGmToL1() ← MTE1: A→L1        │
│  PadL1IfNotAlign()                         │  WaitAicToAiv<PIPE_MTE3>(FLAG9)
│                                            │      等待 AIC 完成上一轮 MMA
│  ★ WaitAivToAic<PIPE_MTE1>(FLAG8) ◄───────┤
│      等待 AIV Weight 反量化完成             │  WeightAntiQuantComputeNzNk()
│                                            │    WaitFlag<MTE3_V> → VF反量化
│  LaunchMatmul()                            │    SetFlag<V_MTE3> / WaitFlag<V_MTE3>
│    WaitFlag<M_MTE1>                        │    MTE3: UB→L1
│    LoadA L1→L0A + ScaleA                   │    SetFlag<MTE3_V>
│    LoadB L1→L0B + ScaleB                   │
│    SetFlag<MTE1_M> / WaitFlag<MTE1_M>      │  ★ SetAivToAic<PIPE_MTE3>(FLAG8) ──→
│    MmadCompute                              │      通知 AIC Weight 就绪
│    SetFlag<M_MTE1>                         │
│                                            │  SetVToMTE2()
│  SetMTE1ToMTE2()                           │    SetFlag<V_MTE2>
│  SetScaleMTE1ToMTE2()                      │
│                                            │
│  ★ SetAicToAiv<PIPE_MTE1>(FLAG9) ────────→│
│      通知 AIV 可继续搬运                    │
```

#### BasicBlock 边界同步时序（Fixpipe ↔ SwiGLU+Quant）

```
时间 → (BasicBlock 边界)

AIC (Cube Core)                              AIV (Vector Core)
│                                            │
│  (进入新 BB 的 IterateNzNkWithKAic)         │  (进入新 BB 的 IterateNzNkWithAiv)
│                                            │
│  if (cvLoopIdx_ > 0):                      │  Pass1: 预加载前 2 块 Weight
│                                            │    VecComputeNzNkWithStartLimit(0, 2*kbL1Size)
│                                            │
│    ★ WaitAivToAic<PIPE_FIX>(FLAG4) ◄───────┤  PhaseB:
│        等待 AIV 取走上一个 BB 的 UB 结果      │
│                                            │    ★ SetAivToAic<PIPE_MTE3>(FLAG4) ──→
│    GetTensorC(yF32Buffer_, lastParam)      │        通知 AIC 可以 Fixpipe
│      Fixpipe: L0C → UB (F32)              │
│                                            │    ★ WaitAicToAiv<PIPE_V>(FLAG3) ◄──
│    ★ SetAicToAiv<PIPE_FIX>(FLAG3) ────────→│        等待 AIC Fixpipe 完成
│        通知 AIV UB 结果已就绪                │
│                                            │    SwiGLU(yF32Buffer_, lastParam)
│  (继续当前 BB 的 K 循环)                     │    ComputeMaxExp(yF32Buffer_, lastParam)
│    for kbL1Offset:                         │    ComputeScale(lastParam)
│      ... (含 FLAG8/FLAG9 跨核同步)           │    ComputeDataForQuantTargetFp8(yF32Buffer_, lastParam)
│                                            │    CopyOutputFromUb2Gm(lastParam)
│                                            │    CopyScaleFromUb2Gm(lastParam)
│                                            │
│                                            │  Pass2: 继续加载剩余 Weight
│                                            │    VecComputeNzNkWithStartLimit(2*kbL1Size, kSize)
```

#### 多 BasicBlock 流水线全景

```
时间 → (3 个 BasicBlock 的完整流水线)

BB0:                              BB1:                              BB2:
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ AIC:                 │ │ AIC:                 │ │ AIC:                 │
│  K循环0: CopyA+MMA   │ │  ★ Fixpipe BB0→UB   │ │  ★ Fixpipe BB1→UB   │
│  K循环1: CopyA+MMA   │ │  K循环0: CopyA+MMA   │ │  K循环0: CopyA+MMA   │
│  K循环2: CopyA+MMA   │ │  K循环1: CopyA+MMA   │ │  K循环1: CopyA+MMA   │
│  K循环3: CopyA+MMA   │ │  K循环2: CopyA+MMA   │ │  K循环2: CopyA+MMA   │
│                      │ │  K循环3: CopyA+MMA   │ │  K循环3: CopyA+MMA   │
├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤
│ AIV:                 │ │ AIV:                 │ │ AIV:                 │
│  Pass1: WQ_K0, WQ_K1 │ │  Pass1: WQ_K0, WQ_K1 │ │  Pass1: WQ_K0, WQ_K1 │
│  (无 PhaseB)          │ │  PhaseB:             │ │  PhaseB:             │
│  Pass2: WQ_K2, WQ_K3 │ │    ★ SwiGLU+Quant    │ │    ★ SwiGLU+Quant    │
│                      │ │    (处理 BB0 结果)     │ │    (处理 BB1 结果)     │
│                      │ │  Pass2: WQ_K2, WQ_K3 │ │  Pass2: WQ_K2, WQ_K3 │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
        │                          │                          │
        └── BB0 无 PhaseB ─────────┘── BB1 PhaseB 处理 BB0 ──┘── BB2 PhaseB 处理 BB1

End 阶段: AIC Fixpipe BB2→UB, AIV SwiGLU+Quant 处理 BB2 结果
```

### 8.13.4 与 GMMFR 的同步差异

| 差异点 | GMMFR | GMMSQ | 说明 |
|--------|-------|-------|------|
| 跨核同步信号 | FLAG 8/9/4/3 | FLAG 8/9/4/3 | **完全相同** |
| 双 Flag 机制 | AIC→AIV 双设单等，AIV→AIC 单设双等 | **完全相同** | 同步原语不变 |
| K 循环同步 | FLAG 8/9 控制 Weight↔MMA | **完全相同** | 反量化流程不变 |
| BB 边界同步 | FLAG 4/3 控制 Fixpipe↔Routing | FLAG 4/3 控制 Fixpipe↔SwiGLU+Quant | 同步信号不变，后处理逻辑变化 |
| AIV 侧 Event | 含 Routing 相关 Event（S_MTE2, MTE2_S, RL_MTE2_V, FR_V_MTE2） | **去除** Routing 相关 Event | GMMSQ 无 Routing 流水线 |
| AIC 侧 Event | MTE1_MTE2, SCALE_MTE1_TO_MTE2, M_MTE1, MTE1_M, MTE2_MTE1 | **完全相同** | MMA 计算流水线不变 |

---

## 附录：关键常量汇总

| 常量 | 值 | 说明 |
|------|-----|------|
| `L1_SIZE` | 512 KB | L1 总大小 |
| `M_L1` | 256 | M 轴 L1 分块大小 |
| `N_L1` | 256 | N 轴 L1 分块大小（128 Swish + 128 Gate） |
| `KB_L1_SIZE` | 256 | K 轴 L1 分块基础大小 |
| `MX_A8W4_L1_K_CONFIG_256` | 256 | 动态 K 选择：小 M/N 时使用 256 |
| `MX_A8W4_L1_K_CONFIG_512` | 512 | 动态 K 选择：大 M/N 时使用 512 |
| `MX_GROUPSIZE` | 32 | MX 量化分组大小 |
| `MX_SCALE_K_L1_SIZE` | 4096 | Scale L1 搬运粒度 |
| `K_ALIGNMENT64` | 64 | K 轴对齐粒度 |
| `C0_SIZE_B8` | 32 | 8bit 数据类型的 C0 大小 |
| `DOUBLE_BUFFER_NUM` | 2 | 双缓冲数量 |
| `QUADRUPLE_BUFFER_NUM` | 4 | 四缓冲数量 |
| `BASIC_BLOCK_PROCESS_NUM` | 2 | 乒乓 BasicBlock 数量 |
| `UB_AVAILABLE_SIZE` | 248 KB | UB 可用空间 |
| `VECTOR_REG_WIDTH` | 256 B | 向量寄存器宽度 |
| `BLOCK_CUBE` | 16 | Cube 计算基本块大小 |