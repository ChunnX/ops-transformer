# MOE Distribute Dispatch Tiling 重构架构方案

> 版本：v0.3
> 日期：2026-04-17
> 范围：dispatch_v2（含 hostkfc 共用）、dispatch_v3（v2 薄包装）、dispatch_setup、dispatch_teardown、arch35(CCU)
> 排除：v1、A2 专有逻辑
> 层次：仅 tiling（op_host/op_tiling），不涉及 kernel/tilingData/tilingKey 修改

---

## 一、现状问题总结

### 1.1 代码规模

| 算子 | 文件数 | 行数 | 架构风格 |
|------|--------|------|---------|
| dispatch_v2 | 5 | 3159 | 匿名命名空间静态自由函数，无类 |
| dispatch_v2/arch35(CCU) | 2 | 995 | 类继承 MoeTilingBase（同时服务 V1/V2） |
| dispatch_setup | 7 | 1151 | 类继承 MoeTilingBase → Base → arch 子类 |
| dispatch_teardown | 7 | 1059 | 类继承 MoeTilingBase → Base → arch 子类 |
| dispatch_v3 | 1 | 124 | v2 薄包装 |
| **合计** | **22** | **6488** | |

### 1.2 主要问题

1. **架构风格分裂**：v2 用匿名命名空间静态自由函数（无类），setup/teardown/arch35 用框架继承体系（MoeTilingBase → Base → arch 子类）。

2. **大面积代码重复**：
   - ~30 组常量在 4+ 个文件的匿名命名空间各自定义（量化枚举、对齐值、通信常量、范围上界）
   - 属性校验（epWorldSize/epRankId/moeExpertNum/quantMode 范围检查）在 v2/setup/teardown/arch35 各有一套
   - tensor 维度/dtype/format 校验在 v2/helper/arch35 各有一套
   - tokenMsgSize 按 quantMode 的 5 路计算在 3+ 处重复
   - sharedExpert 关系校验在 v2/setup 重复，teardown 缺失
   - DispatchV2Config 20+ 行赋值块在 3 处复制粘贴

3. **框架继承无实际价值**：
   - setup/teardown 的 arch32 子类 IsCapable 永远返回 false，且未注册——实际死代码
   - arch 子类唯一差异是 SetHcommCfg（3-15 行代码）
   - 继承增加了 7 个头文件/源文件的开销

4. **arch35(CCU) 自含重复**：959 行中大部分是 v2 校验逻辑的独立重写，同时通过 OpVersionManager 服务 V1 和 V2。

5. **参数膨胀**：v2 多个函数 8-11 个参数（CheckTensorShape 达 10 参数，CheckTensorDim 达 9 参数）。

6. **已发现 bug**（仅记录，本次不修改）：
   - teardown: `CheckOptionalAttrValue` 中 quantMode 校验使用 `&&` 应为 `||`（永远为 false）
   - teardown: `CheckRelationTensorDataType` 已声明但未在主流程中调用
   - teardown: 缺失 sharedExpert 组合校验
   - teardown: `SYSTEM_NEED_WORKSAPCE` 拼写错误（未使用的重复常量）
   - v2/helper: 函数名 typo `CheckCommom...`（应为 Common）
   - setup/teardown arch35: `protocal` typo（应为 protocol，受限于框架 API）
   - teardown: dynamicScales dtype 对 MX_QUANT 统一校验 float，与 v2-A5 的 e8m0 不一致

---

## 二、设计原则

1. **去框架继承**：移除对 `MoeTilingBase` 的继承关系，各算子注册普通函数入口
2. **公共校验用 C 风格参数化函数**：无状态、inline、参数 <= 5、单函数 <= 50 行
3. **独立特性用 checker 函数组按需组合**：elastic/activeMask/performance/fullmesh 各自封装
4. **量化特性用子类继承**：每个 quantMode 一个 QuantChecker 子类，arch/算子差异通过构造参数传入
5. **架构与业务解耦**：不同 arch 的差异（通信配置、范围上界）通过参数化/配置传入，不用继承
6. **阶段间传递用算子局部 TilingCtx**：避免参数膨胀
7. **不修改 tilingData 结构体**：MoeDistributeDispatchV2Info / SetupInfo / TeardownInfo 保持原样
8. **不修改 tilingKey 计算逻辑**：保持现有编码方式
9. **不劣化性能**：公共函数 inline，常量编译期确定
10. **不省略原有校验项**
11. **已发现 bug 仅记录不修改**
12. **错误处理**：公共层校验函数失败时使用 `OPS_LOG_E` 宏输出错误日志，返回 `graphStatus`（`ge::GRAPH_FAILED`）
13. **命名空间策略**：各公共层文件使用各自命名空间（如 `moe_dispatch_defs`、`elastic_info_check` 等），如有需要再统一调整
14. **头文件依赖**：允许向上依赖公共部分 — quant_rules.h 可依赖 defs.h，各 checker 可依赖 utils.h 和 defs.h；禁止横向依赖（checker 之间不互相 include）和反向依赖（公共层不依赖算子层）

---

## 三、目录结构

```
mc2/common/op_host/op_tiling/moe_dispatch/     [新建公共层]
  |
  +-- defs.h                         常量 + 枚举
  +-- utils.h                        C 风格工具函数 + Rule 批量校验
  |
  +-- quant_checker_base.h             量化校验基类 QuantCheckerBase
  +-- quant_checker_unquant.h          UnquantChecker (mode=0)
  +-- quant_checker_static.h           StaticQuantChecker (mode=1)
  +-- quant_checker_dynamic_base.h     DynamicQuantCheckerBase (动态量化共有骨架)
  +-- quant_checker_pertoken.h         PerTokenChecker (mode=2)
  +-- quant_checker_pergroup.h         PerGroupChecker (mode=3)
  +-- quant_checker_mx.h              MxQuantChecker (mode=4, arch 参数化)
  +-- quant_checker_factory.h          CreateQuantChecker 工厂函数
  +-- active_mask_checker.h          ActiveMask 可选特性校验（函数组）
  +-- elastic_info_checker.h         ElasticInfo 可选特性校验（函数组）
  +-- performance_info_checker.h     PerformanceInfo 可选特性校验（函数组）
  +-- fullmesh_checker.h             FullMeshV2 可选特性校验（函数组）
  |
  +-- comm_cfg.h/cpp                 通信配置统一实现
  +-- hccl_buffer_checker.h          HCCL buffer 大小校验

mc2/moe_distribute_dispatch_v2/op_host/op_tiling/   [精简后]
  +-- dispatch_v2_tiling.cpp         A3/A5 MTE 主流程（引用公共层）
  +-- dispatch_v2_layout.h           v2 的 MoeDispatchTilingLayout 定义 + v2 TilingCtx
  +-- dispatch_v2_win_size.h         v2 win size 校验（仅 v2 特有公式）
  +-- arch35/
        +-- dispatch_ccu_tiling.cpp  CCU 主流程（引用公共层，仅 CCU 特有逻辑）

mc2/moe_distribute_dispatch_v3/op_host/op_tiling/   [基本不变]
  +-- moe_distribute_dispatch_v3_tiling.cpp   仍为 v2 薄包装（构造 v2 Layout 调 v2 入口）

mc2/moe_distribute_dispatch_setup/op_host/op_tiling/   [精简后]
  +-- dispatch_setup_tiling.cpp      单文件，无继承，引用公共层

mc2/moe_distribute_dispatch_teardown/op_host/op_tiling/ [精简后]
  +-- dispatch_teardown_tiling.cpp   单文件，无继承，引用公共层
```

---

## 四、核心数据结构

### 4.1 MoeDispatchTilingLayout -- 算子原型索引映射

各算子定义自己的 Layout，存储 tensor 和 attr 在 context 中的索引位置。
解决"不同算子输入输出顺序不同"的问题，使公共层校验函数与具体算子原型解耦。

```cpp
// 可选索引缺省值
constexpr int32_t LAYOUT_INDEX_ABSENT = -1;

struct MoeDispatchTilingLayout {
    // ========== 输入 tensor 索引 ==========
    // 必选索引用 uint32_t；可选索引用 int32_t（LAYOUT_INDEX_ABSENT 表示不存在）
    int32_t  contextIndex;              // 仅 v3 使用（isMc2Context=true），v2 为 ABSENT
    uint32_t xIndex;
    uint32_t expertIdsIndex;
    int32_t  scalesIndex;               // 可选
    int32_t  activeMaskIndex;           // 可选
    int32_t  expertScalesIndex;         // 可选
    int32_t  elasticInfoIndex;          // 可选
    int32_t  performanceInfoIndex;      // 可选

    // ========== 输出 tensor 索引 ==========
    // 各算子输出不同，由各算子 Layout 实例填充
    // v2: expandX + expandExpertIds + expertTokenNums + expandXDynamicScales
    //     + epRecvCounters + assistInfo [+ moeRecvCount(A5)]
    // setup: y + expandIdx + commCmdInfo
    // teardown: expandXOut + expertTokenNums + assistInfo + expandXDynamicScales
    uint32_t outputStartIndex;          // 第一个输出 tensor 的索引
    uint32_t outputCount;               // 输出 tensor 个数

    // ========== 属性索引（共 17 个，对应当前 DispatchV2Config） ==========
    uint32_t attrGroupEpIndex;
    uint32_t attrEpWorldSizeIndex;
    uint32_t attrEpRankIdIndex;
    uint32_t attrMoeExpertNumIndex;
    int32_t  attrCclBufferSizeIndex;    // v3 专用（与 moeExpertNum 共享默认位置），v2 为 ABSENT
    uint32_t attrGroupTpIndex;
    uint32_t attrTpWorldSizeIndex;
    uint32_t attrTpRankIdIndex;
    uint32_t attrExpertSharedTypeIndex;
    uint32_t attrSharedExpertNumIndex;
    uint32_t attrSharedExpertRankNumIndex;
    uint32_t attrQuantModeIndex;
    uint32_t attrGlobalBsIndex;
    uint32_t attrExpertTokenNumsTypeIndex;
    int32_t  attrCommAlgIndex;          // 可选（setup/teardown 无此属性）
    int32_t  attrZeroExpertNumIndex;    // 可选
    int32_t  attrCopyExpertNumIndex;    // 可选
    int32_t  attrConstExpertNumIndex;   // 可选

    // ========== 标志位 ==========
    bool     isMc2Context;              // v3 设为 true，v2 为 false
};
```

> **与现有 DispatchV2Config 的对应关系**：Layout 是 DispatchV2Config 的重构替代。
> 当前 DispatchV2Config 有 8 个输入索引 + 17 个属性索引 + 1 个 isMc2Context 标志（共 26 字段），
> 本 Layout 完整覆盖全部字段，并：
> - 可选字段改为 `int32_t` + `LAYOUT_INDEX_ABSENT` 语义
> - 增加 `outputStartIndex` + `outputCount` 支持输出 tensor 校验
> - 各算子（v2/v3/setup/teardown）构造各自的 Layout 实例
>
> v3 复用 v2 逻辑时，通过构造不同 Layout 实例做索引桥接（输入索引全部偏移 +1，isMc2Context=true）。

### 4.2 TensorCheckRule -- Tensor 校验规则声明

单个 tensor 的校验规则，从 Layout 中读取索引。
校验流程**按 Tensor 拆分**：每个 tensor 一次完成 dim+dtype+format 校验。

```cpp
constexpr uint32_t MAX_VALID_DTYPES = 4;

struct TensorCheckRule {
    uint32_t    index;                          // tensor 在 context 中的索引（从 Layout 获取）
    TensorType  type;                           // INPUT / OUTPUT
    uint32_t    expectedDim;                    // 期望维度（0 = 不检查）
    DataType    validDtypes[MAX_VALID_DTYPES];  // 合法 dtype 集合（空 = 不检查）
    uint32_t    dtypeCount;                     // validDtypes 有效个数
    const char* name;                           // 日志用名称
};
```

- `CheckTensorRules(ctx, rules, count)` -- 批量校验必选 tensor
- `CheckOneTensorRule(ctx, rule)` -- 单个校验（用于条件性 tensor：scales/activeMask/elasticInfo 等）

> **条件性 tensor** 在主流程中根据可选特性标志判断是否存在，存在时构建 rule 并调用 CheckOneTensorRule。
> 校验工具与必选 tensor 完全统一。

### 4.3 AttrCheckRule -- 属性校验规则声明

单个属性的范围校验规则，与 TensorCheckRule 对称设计。

```cpp
struct AttrCheckRule {
    uint32_t    attrIndex;    // 属性在 context 中的索引（从 Layout 获取）
    const char* attrName;     // 日志用名称
    uint32_t    minVal;       // 下界（含）
    uint32_t    maxVal;       // 上界（含）
    bool        required;     // 是否必选（必选时 null 即报错）
};
```

- `CheckAttrRules(ctx, rules, count)` -- 批量范围校验
- **跨属性关联校验**（epRankId < epWorldSize、sharedExpert 整除关系等）由单独的 C 风格函数处理
- 各算子的特化上界（如 v2 的 MAX_EP_WORLD_SIZE=768 vs setup/teardown 的 384）写在各自的 AttrCheckRule 中

### 4.4 量化校验类体系 -- QuantChecker 继承结构

每个 quantMode 一个子类，arch/算子差异通过构造参数传入。

#### ASCII 类图

```
                        QuantCheckerBase (抽象基类)
                        ├─ shouldHaveScales(xDtype) → bool          [virtual]
                        ├─ checkScales(ctx, desc, dim0, h) → status [virtual]
                        ├─ checkDynamicScales(ctx, desc, dim0, h)   [virtual]
                        ├─ calcTokenMsgSize(h) → uint32_t           [virtual]
                        ├─ getDynamicScalesDimNum() → uint32_t      [virtual]
                        ├─ getDynamicScalesDtype() → DataType       [virtual]
                        └─ calcDynamicScalesDim1(h) → uint32_t      [virtual]
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   UnquantChecker      StaticQuantChecker    DynamicQuantCheckerBase
   (mode=0)            (mode=1)                     │
   scales: 不需要      scales: 必需(1D)       ┌─────┼──────────┐
   dynScales: 不需要    dynScales: 不需要      │     │          │
   tokenMsg: ALIGN_256                  PerToken  PerGroup   MxQuant
                                        Checker   Checker    Checker
                                        (mode=2)  (mode=3)   (mode=4)
                                        dyn: 2D   dyn: 2D    dyn: 1D
                                        dtype:    dtype:     dtype: 构造参数
                                        float     float      (float/e8m0)
                                        dim1:     dim1:      scales: 禁止
                                        ceil128   ceil32Even
```

> `DynamicQuantCheckerBase` 抽取 PerToken/PerGroup/MX 共有的 dynamicScales 校验骨架，
> 子类仅 override dim/dtype/计算路径差异。

#### 基类定义

```cpp
class QuantCheckerBase {
public:
    explicit QuantCheckerBase(uint32_t mode) : mode_(mode) {}
    virtual ~QuantCheckerBase() = default;

    uint32_t mode() const { return mode_; }

    // ---- 各子类必须实现 ----
    // scales 是否必须存在（受 xDtype 影响）
    virtual bool ShouldHaveScales(DataType xDtype) const = 0;
    // scales 完整校验（dim + dtype + shape）
    virtual graphStatus CheckScales(TilingContext* ctx, const TensorDesc* desc,
                                    uint32_t expectedDim0, uint32_t h) const = 0;
    // dynamicScales 完整校验
    virtual graphStatus CheckDynamicScales(TilingContext* ctx, const TensorDesc* desc,
                                          uint32_t expectedDim0, uint32_t h) const = 0;
    // tokenMsgSize 计算
    virtual uint32_t CalcTokenMsgSize(uint32_t h) const = 0;

    // ---- 默认实现可 override ----
    virtual uint32_t GetDynamicScalesDimNum() const { return 0; }
    virtual DataType GetDynamicScalesDtype() const { return DT_FLOAT; }
    virtual uint32_t CalcDynamicScalesDim1(uint32_t h) const { return 0; }

protected:
    uint32_t mode_;
};
```

#### 构造参数化处理 arch 差异

各算子主流程构造 checker 实例时传入 arch 相关参数，而非在 checker 内部判断 arch：

```cpp
// v2-A5 主流程：MX_QUANT 的 dynamicScales dtype 为 e8m0
auto mxChecker = MxQuantChecker(DT_E8M0);

// teardown 主流程：MX_QUANT 的 dynamicScales dtype 为 float
auto mxChecker = MxQuantChecker(DT_FLOAT);
```

#### 工厂函数

```cpp
// 根据 quantMode 创建对应 checker（arch 差异参数由调用方提供）
// 返回 unique_ptr<QuantCheckerBase>
std::unique_ptr<QuantCheckerBase> CreateQuantChecker(uint32_t quantMode,
                                                     DataType mxDynScalesDtype = DT_FLOAT);
```

#### 各算子使用方式

```cpp
// 阶段 F 量化校验（以 v2 为例）
auto checker = CreateQuantChecker(tc.quantMode, archMxDtype);  // 构造
tc.hasScales = checker->ShouldHaveScales(xDtype);               // 判定 scales 必填性
if (tc.hasScales) {
    CHECK_SUCCESS(checker->CheckScales(ctx, scalesDesc, dim0, h));
}
CHECK_SUCCESS(checker->CheckDynamicScales(ctx, dynScalesDesc, dim0, h));
tc.tokenMsgSize = checker->CalcTokenMsgSize(h);
```

#### 扩展方式

| 场景 | 操作 |
|------|------|
| 新增 quantMode=5 | 新增子类 + CreateQuantChecker 加一个 case |
| 算子 X 新增支持某 mode | 在算子 X 的 supportedModes 中加入该 mode（CreateQuantChecker 已覆盖） |
| 某 mode 在不同 arch 下 dtype 不同 | 通过构造参数传入，无需新增子类 |
| 某 mode 在不同算子下校验逻辑完全不同 | 可继承已有子类 override 对应方法 |

### 4.5 算子 TilingCtx -- 阶段间中间值载体

各算子定义自己的局部 TilingCtx 结构体，承载阶段间传递的全部中间值。
各阶段子函数统一接收 `TilingCtx*` 参数，避免参数膨胀。

```cpp
// 示例：v2 的 TilingCtx（最大最复杂）
struct DispatchV2TilingCtx {
    // 阶段 B 产出：属性值
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t tpWorldSize;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    bool     isLayered;
    bool     isFullMeshV2;
    // ...

    // 阶段 C 产出：可选特性标志
    bool hasElastic;
    bool hasActiveMask;
    bool hasPerformance;

    // 阶段 D 产出：中间计算值
    uint32_t localExpertNum;
    bool     isSharedRank;
    uint32_t A;

    // 阶段 F 产出：量化计算值
    bool     hasScales;
    uint32_t tokenMsgSize;
    uint32_t dynamicScalesDim1;
};
```

各算子各自定义自己的 TilingCtx（字段不同）：
- **DispatchV2TilingCtx**：最大，含 elastic/tp/layered/fullmesh 等全部标志
- **SetupTilingCtx**：较小，无 elastic/tp/layered/fullmesh
- **TeardownTilingCtx**：最小

> TilingCtx 仅在主流程函数内作为局部变量存活，不是公共层的数据结构。

### 4.6 公共函数返回的小结果结构

各公共计算函数返回自己的小结果结构，由调用方选择性消费：

```cpp
// shared expert 计算结果
struct SharedExpertResult {
    uint32_t localExpertNum;
    bool     isSharedRank;
    uint32_t moeExpertPerRankNum;
};

// elastic 计算结果（仅 v2 使用）
struct ElasticResult {
    bool     hasElastic;
    uint32_t adjustedA;
    uint32_t adjustedExpertTokenDim0;
    uint32_t adjustedEpRecvDim0;
};
```

各模板按需使用：v2 消费 SharedExpertResult + ElasticResult，setup 仅消费 SharedExpertResult，teardown 仅消费 SharedExpertResult。结果值由各模板主流程自行写入 tilingData。

---

## 五、公共层组件设计

### 5.1 defs.h -- 常量与枚举

**量化模式枚举**：
```
UNQUANT=0, STATIC_QUANT=1, PERTOKEN_DYNAMIC_QUANT=2, PERGROUP_DYNAMIC_QUANT=3, MX_QUANT=4
```

**对齐常量**：
```
ALIGN_32=32, ALIGN_128=128, ALIGN_256=256, ALIGN_512=512, EVEN_ALIGN=2
```

**通信常量**：
```
OP_TYPE_ALL_TO_ALL=8, OP_TYPE_ALL_GATHER=6, OP_TYPE_BATCH_WRITE=18
URMA_COMM=2, SDMA_COMM=0
```

**维度常量**：
```
ONE_DIM=1, TWO_DIMS=2
```

**公共范围基础值**：
```
MIN_H_DEFAULT=1024, MAX_H_DEFAULT=8192
MAX_BS_DEFAULT=512, MAX_K_DEFAULT=16
MAX_SHARED_EXPERT_NUM=4, MIN_EP_WORLD_SIZE=2
MAX_GROUP_NAME_LENGTH=128
```

**其他公共常量**：
```
ASSIST_INFO_FACTOR=128, COMM_CMD_INFO_FACTOR=16
MX_BLOCK_SIZE=32, PERGROUP_BLOCK_SIZE=128
SYSTEM_NEED_WORKSPACE=16*1024*1024
```

> 各算子的特化上界写在各自的 AttrCheckRule 中，不放在 defs.h。

### 5.2 utils.h -- C 风格工具函数

**Tensor/Attr 批量校验**（基于 Rule 结构）：
- `CheckTensorRules(ctx, rules, count)` -- 按 Tensor 拆分，逐个完成 dim+dtype+format 校验
- `CheckOneTensorRule(ctx, rule)` -- 单个 tensor 校验（条件性 tensor 统一入口）
- `CheckAttrRules(ctx, rules, count)` -- 批量属性范围校验

**原子校验工具**（供 Rule 校验内部和特性 checker 使用）：
- `CheckDimNum(shape, expected)` -- 维度数校验
- `CheckDtypeIn(dtype, validSet, count)` -- dtype 在合法集合中
- `CheckNotFractalNZ(format)` -- format 非 FRACTAL_NZ
- `CheckRangeU32(val, min, max)` -- uint32 范围校验

**公共中间值计算**：
- `CalcLocalExpertNum(epRankId, sharedRankNum, expertNum, epWorldSize)` → SharedExpertResult
- `CalcA(bs, k, localExpertNum, isSharedRank, hasElastic, ...)` -- A 值独立计算
- `CeilDiv(a, b)` -- 向上整除
- `AlignUp(val, align)` -- 向上对齐

**平台信息获取**：
- `GetPlatformInfo(context)` → `{aivNum, ubSize, blockDim}`

> 所有函数均为 `inline`，无状态，参数 <= 5。

### 5.3 quant_checker -- 量化特性校验（类继承体系）

采用 quantMode 子类继承 + arch 参数化，替代原有的全局静态规则 + 函数组。

**文件结构**：
```
moe_dispatch/
  +-- quant_checker_base.h          基类 QuantCheckerBase 定义
  +-- quant_checker_unquant.h       UnquantChecker (mode=0)
  +-- quant_checker_static.h        StaticQuantChecker (mode=1)
  +-- quant_checker_dynamic_base.h  DynamicQuantCheckerBase (PerToken/PerGroup/MX 共有骨架)
  +-- quant_checker_pertoken.h      PerTokenChecker (mode=2)
  +-- quant_checker_pergroup.h      PerGroupChecker (mode=3)
  +-- quant_checker_mx.h            MxQuantChecker (mode=4, 构造参数: dynScalesDtype)
  +-- quant_checker_factory.h       CreateQuantChecker 工厂函数
```

**基类接口**（见 4.4 节详细定义）：
- `ShouldHaveScales(xDtype)` -- scales 是否必须存在
- `CheckScales(ctx, desc, dim0, h)` -- scales 完整校验
- `CheckDynamicScales(ctx, desc, dim0, h)` -- dynamicScales 完整校验
- `CalcTokenMsgSize(h)` -- tokenMsgSize 计算
- `GetDynamicScalesDimNum()` / `GetDynamicScalesDtype()` / `CalcDynamicScalesDim1(h)` -- 规则查询

**公共辅助函数**（仍为 C 风格，供各 checker 子类内部使用）：
- `GetExpectedDynamicScalesDim0(A, tpWorldSize)` -- dim0 期望值计算（跨 quantMode 通用）
- `CheckQuantModeMap(quantMode, expandXDtype)` -- QUANT_MODE_MAP 查表校验（放公共层）

**各子类核心差异**：

| 子类 | scales 必填性 | dynamicScales dim | dynamicScales dtype | tokenMsgSize 公式 |
|------|-------------|-------------------|--------------------|--------------------|
| UnquantChecker | 不需要 | 不需要 | - | AlignUp(h, 256) |
| StaticQuantChecker | 必需(1D) | 不需要 | - | AlignUp(h, 256) |
| PerTokenChecker | 按 xDtype | 2D | float | AlignUp(AlignUp(h,32)+AlignUp(h,128), 512) |
| PerGroupChecker | 按 xDtype | 2D | float | AlignUp(AlignUp(h,128)+CeilDiv(h,128), 512) |
| MxQuantChecker | 禁止 | 1D | **构造参数**(float/e8m0) | AlignUp(AlignUp(h,256)+CeilDiv(h,32), 512) |

### 5.4 可选特性校验函数组

4 个有独立 tensor 校验 + 互斥规则的可选特性，各为命名空间内 2-3 个 C 风格函数。

> **设计原则**：可选特性 checker 仅做**互斥规则校验**和**产出标志/结果结构**。
> 可选 tensor 的基本校验（dim+dtype+format）统一在阶段 E 中通过 TensorCheckRule 处理。
> 可选 tensor 的 shape 精确值校验在阶段 G 中执行。

**elastic_info_checker** (namespace elastic_info_check)：
- `CheckCompatibility(nodeName, isLayered, hasTp)` -- 互斥规则（与 layered/tp 互斥）
- `CheckShape(ctx, inputIndex, epWorldSize)` -- dim0 = 4 + 2*epWorldSize
- `CalcAdjustments(localExpertNum, sharedA, moeA, epWorldSize)` → ElasticResult

**active_mask_checker** (namespace active_mask_check)：
- `CheckCompatibility(nodeName, isLayered, isExpertMask, globalBs)` -- 互斥/约束规则
- `CheckShape(ctx, inputIndex, bs, k)` -- shape 值校验（1D: dim0=bs; 2D: dim0=bs, dim1=k）

**performance_info_checker** (namespace performance_info_check)：
- `CheckCompatibility(nodeName, isLayered)` -- 互斥规则
- `CheckShape(ctx, inputIndex, epWorldSize)` -- dim0 = epWorldSize

**fullmesh_checker** (namespace fullmesh_check)：
- `CheckConstraints(nodeName, bs, k, tpWorldSize)` -- BS<=256, K<=12, tp!=2

### 5.5 归入属性校验/中间值计算的特性

以下特性**不单独作为 checker 函数组**，归入公共属性关联校验或中间值计算：

| 特性 | 归属 | 处理方式 |
|------|------|---------|
| shared_expert | 阶段 B 关联校验 + 阶段 D 中间值 | `CheckSharedExpertRelation(num, rankNum)` + `CalcLocalExpertNum(...)` → SharedExpertResult |
| tp | 阶段 B 属性范围 + 阶段 C 互斥 | AttrCheckRule 校验范围; 互斥规则（与 localExpertNum>1、elastic）在阶段 C 中内联处理 |
| layered | 阶段 B 确定标志 + 影响各阶段 | `isLayered` 标志传入各校验函数参数；范围收窄写在 AttrCheckRule 中 |
| zeroExpert | v2 独有逻辑 | 当前留在 v2 内部；Layout 预留可选字段；后续提取时抽为 `zero_expert_check::CheckAndCalc()` |
| isMc2Context | v2 config 标志 | 通过 Layout/TilingCtx 传递，影响 hcommCfg 和 winSize 路径 |
| scales | 量化关联 | 存在性由 `QuantCheckerBase::ShouldHaveScales(xDtype)` 判定；校验由 `QuantCheckerBase::CheckScales()` 执行 |

### 5.6 comm_cfg -- 通信配置

统一封装当前散布在各算子中的 SetHcommCfg 逻辑：
- `SetEpAllToAllCfg(context, algConfig, isUrma)` -- EP AllToAll（覆盖 setup/teardown/v2 的 EP 通信）
- `SetTpAllGatherCfg(context)` -- TP AllGather（v2 TP 场景）
- `SetCcuCommCfg(context, ...)` -- CCU 专用通信配置

### 5.7 hccl_buffer_checker -- HCCL buffer 校验

- `CalcMinBuffSize(params)` -- 按参数计算最小 buffer 需求
- `CheckBuffSizeFromContext(context, groupIndex, minRequired)` -- 从 HCCL group 获取实际大小并比较
- `CheckBuffSizeFromAttr(cclBufferSize, minRequired)` -- 从属性获取（V3/Mc2Context 场景）

---

## 六、可选特性矩阵

共 10 个可选特性，完整支持情况：

| # | 特性 | 类型 | v2-A3 MTE | v2-A5 MTE | arch35 CCU | setup | teardown |
|---|------|------|-----------|-----------|------------|-------|----------|
| 1 | elastic_info | 可选输入 | 支持 | 支持 | 不支持 | 不支持 | 不支持 |
| 2 | xActiveMask | 可选输入 | 支持(1D+2D) | 支持(1D+2D) | 支持(仅1D) | 支持(仅1D) | 不支持 |
| 3 | performance_info | 可选输入 | 支持 | 支持 | 不支持 | 不支持 | 不支持 |
| 4 | fullmesh_v2 | commAlg 属性 | 支持 | 支持 | 不支持 | 不支持 | 不支持 |
| 5 | tp (tpWorldSize>1) | 属性 | 支持 | 支持 | 不支持 | 不支持 | 不支持 |
| 6 | layered/hierarchy | commAlg 属性 | 支持 | 支持 | 不支持 | 不支持 | 不支持 |
| 7 | shared_expert (>0) | 属性 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 8 | zeroExpert (>0) | 属性 | 支持 | 支持 | 不支持 | 不支持 | 不支持 |
| 9 | scales | 可选输入 | 支持 | 支持 | 支持 | 支持 | 不支持 |
| 10 | isMc2Context | config 标志 | 不支持 | 支持 | 不支持 | 不支持 | 不支持 |

分类处理：
- **独立 checker 函数组**（有 tensor 校验 + 互斥规则）：elastic_info、xActiveMask、performance_info、fullmesh_v2
- **归入属性校验/中间值计算**（影响公共计算但无独立 tensor）：shared_expert、tp、layered、zeroExpert
- **归入量化 checker 类体系**：scales
- **通过 config 标志处理**：isMc2Context

---

## 七、各算子重构后的主流程

所有算子遵循统一阶段模式（不做流程模板化，各算子自行编排调用公共组件）：

```
阶段 A: 获取 tilingData 指针
阶段 B: 属性读取 + 校验 + 写入 tilingData (AttrCheckRule 批量 + 关联校验)
阶段 C: 可选特性互斥校验 (elastic/activeMask/performance/fullmesh — 仅互斥规则，产出 bool 标志)
阶段 D: 中间值计算 (SharedExpertResult + ElasticResult + A 值)
阶段 E: tensor 基本校验 (TensorCheckRule 批量 — 必选 + 根据标志的条件性 tensor)
阶段 F: 量化校验 (QuantChecker 子类多态 — scales + dynamicScales，此时 A 已就绪)
阶段 G: shape 精确值校验 (算子独有逻辑，消费 A + 量化结果 + 特性标志)
阶段 H: HCCL buffer 校验
阶段 I: 设置 (platform / workspace / tilingKey / hcommCfg)
阶段 J: 日志输出
```

> **阶段顺序设计依据**：
> - A 值依赖 SharedExpertResult(阶段D) + ElasticResult(阶段C/D)，必须在阶段 D 计算
> - 量化校验(阶段F)中 dynamicScales dim0 依赖 A 值，必须在阶段 D 之后
> - tensor 基本校验(阶段E)中条件性 tensor 需要阶段 C 的标志
> - shape 精确值(阶段G)消费阶段 D 的 A 值 + 阶段 F 的量化计算结果

### 主流程代码模式

```cpp
graphStatus DispatchV2TilingFunc(TilingContext* ctx) {
    DispatchV2TilingCtx tc{};

    /* A */ auto* tilingData = GetTilingData<...>(ctx);
    /* B */ CHECK_SUCCESS(CheckAndGetAttrs(ctx, layout, &tc));
    /* C */ CHECK_SUCCESS(CheckOptionalFeatures(ctx, layout, &tc));
    /* D */ CalcIntermediateValues(&tc);
    /* E */ CHECK_SUCCESS(CheckTensors(ctx, layout, tc));
    /* F */ CHECK_SUCCESS(CheckQuant(ctx, layout, &tc));
    /* G */ CHECK_SUCCESS(CheckShapeValues(ctx, layout, tc));
    /* H */ CHECK_SUCCESS(CheckHcclBuffSize(ctx, tc));
    /* I */ SetPlatformAndComm(ctx, tilingData, tc);
    /* J */ PrintTilingDataInfo(tilingData);
    return ge::GRAPH_SUCCESS;
}
```

### 7.1 各算子的阶段差异

| 阶段 | v2 (MTE) | v2 (CCU/arch35) | setup | teardown |
|------|----------|-----------------|-------|----------|
| B | 17 个属性（含 TP/layered/commAlg/zeroExpert） | 13 个属性（含 V1/V2 版本分支） | 11 个属性 | 12 个属性 |
| C | elastic+performance+activeMask+fullmesh 互斥 | 仅 activeMask | activeMask 互斥 | 无 |
| D | SharedExpertResult + ElasticResult + A | SharedExpertResult + A | SharedExpertResult | SharedExpertResult |
| E | 必选(x+expertIds+expandX+6输出) + 条件(elastic/mask/perf/scales) | 同左(A5 dtype) | 必选(x+expertIds+y+expandIdx+commCmdInfo) + 条件(scales/mask) | 必选(x+y+expertIds+commCmdInfo+expandXOut+3输出) |
| F | QuantChecker 全套(scales+dynamicScales+expandX dtype) | 同左 | QuantChecker 部分(yOut dtype+scales+tokenMsgSize) | QuantChecker 部分(expandXOut dtype+dynamicScales+tokenMsgSize) |
| G | A 值相关 + layered 公式 + elastic 影响 | A 值相关（简化） | yOut/expandIdx/commCmdInfo dim0 | expandXOut/assistInfo/expertTokenNums dim0 |
| H | CheckWinSize（复杂：layered/fullmesh/TP 多路） | CCU 专用公式 | CalcMinBuffSize + 比较 | CalcMinBuffSize + 比较 |
| I.wspc | 16MB + 512*aivNum^2 | 16MB + aivNum 相关 | 32MB | 16MB |
| I.key | GET_TPL_TILING_KEY 宏 | TILINGKEY_TPL_CCU | 1000 + quantMode + scales 标记 | 10000 + quantMode + shared 标记 |
| I.comm | EP AllToAll [+ TP AllGather] | CCU 引擎 + AllToAll | arch: A3=BatchWrite, A5=AllToAll+URMA | AllToAll + URMA |

### 7.2 v3 不变

v3 仍为 v2 薄包装：构造 v2 Layout（索引偏移 + isMc2Context=true）→ 调用 v2 公共入口。

---

## 八、架构差异处理方式

| 差异点 | 处理方式 |
|--------|---------|
| 范围上界（v2: ep<=768, setup/td: <=384 等） | 各算子在 AttrCheckRule 中写入特化上界 |
| v2 A3 vs A5 MTE | TilingCtx 中 archType 字段区分，仅影响量化 dtype 校验分支 |
| arch35 CCU | 保留独立文件，引用公共层；CCU 仅支持 V2 调用，不再走 V1 分支。V1 指 moe_distribute_dispatch 算子，不在本次重构范围内 |
| setup/teardown arch | 去继承后，注册单个函数入口，内部按 arch 判断调用不同 comm_cfg 函数 |
| layered/fullmesh/TP | TilingCtx 中标志位，通过参数传给公共校验函数 |

---

## 九、待细化事项

1. 各算子主流程的详细伪代码
2. dispatch 系列内部是否有适合抽取为基类的公共部分（待评估）
3. 现有 UT 的适配策略
4. v2 helper 文件中 V1 接口的处理方式
5. 迁移阶段规划
6. 各 QuantChecker 子类的具体实现细节（CalcTokenMsgSize 公式、ShouldHaveScales 逻辑等）
7. 各算子 TilingCtx 的完整字段定义
8. 各算子 MoeDispatchTilingLayout 的完整字段定义
