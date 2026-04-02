# AttentionPioneerMetadata AICPU算子 开发过程记录

## 修订记录

| 日期 | 修订版本 | 修改描述 | 作者 |
| :---: | :---: | :---: | :---: |
| 2026-04-02 | 1.0 | 初稿 | - |

---

## 1. 项目背景

将A3平台上已有的 `AiInfraFusedInferAttentionSinkMetadata` AICPU算子适配到A5平台（Ascend 950DT/950PR），在 `attention_pioneer` 算子目录下新增 `AttentionPioneerMetadata` AICPU算子。

**参考代码仓**: `omni-ops/inference/ascendc/src/ops-transformer/attention/ai_infra_fused_infer_attention_sink_metadata`

**目标代码仓**: `ops-transformer/attention/attention_pioneer`

---

## 2. 整体改动清单

### 2.1 新增文件列表

| 序号 | 文件路径 | 说明 |
|------|----------|------|
| 1 | `op_host/common/attention_pioneer_metadata.h` | MetaData常量定义与内存布局 |
| 2 | `op_kernel_aicpu/cpu_context_util.h` | 属性读取工具函数 |
| 3 | `op_kernel_aicpu/attention_pioneer_metadata_aicpu.h` | AICPU kernel头文件 |
| 4 | `op_kernel_aicpu/attention_pioneer_metadata_aicpu.cpp` | AICPU kernel完整实现（~1000行） |
| 5 | `op_kernel_aicpu/CMakeLists.txt` | AICPU kernel构建配置 |
| 6 | `op_graph/attention_pioneer_metadata_proto.h` | GE图层算子原型注册（REG_OP） |
| 7 | `op_host/attention_pioneer_metadata/attention_pioneer_metadata_def.cpp` | OpDef注册 |
| 8 | `op_host/attention_pioneer_metadata/attention_pioneer_metadata_infershape.cpp` | InferShape/InferDtype |
| 9 | `op_host/attention_pioneer_metadata/CMakeLists.txt` | OpDef构建配置 |
| 10 | `op_host/op_api/l0_attention_pioneer_metadata.h` | L0 API头文件 |
| 11 | `op_host/op_api/l0_attention_pioneer_metadata.cpp` | L0 API实现（AICPU task注册） |
| 12 | `op_host/op_api/aclnn_attention_pioneer_metadata.h` | aclnn对外头文件 |
| 13 | `op_host/op_api/aclnn_attention_pioneer_metadata.cpp` | aclnn实现 |
| 14 | `attention_pioneer_metadata_detailed_design.md` | 详细设计文档 |

### 2.2 未修改的已有文件

已有的 `split_core.h/cpp`、`attention_pioneer_def.cpp` 等文件均未修改，新算子代码完全独立。

---

## 3. 关键改动详解与踩坑记录

### 3.1 核数常量适配 (A3→A5)

**改动内容**：
- `AP_AIC_CORE_NUM`: 32（A3中 AIC_CORE_NUM=36 是数组预留上限，实际硬件24核）
- `AP_AIV_CORE_NUM`: 64（A3中 AIV_CORE_NUM=72 是数组预留上限，实际硬件48核）
- `AP_METADATA_SIZE`: 1024（与A3保持一致，空间充足）
- AICPU kernel默认核数: `aicCoreNum_=32, aivCoreNum_=64`（A3中是24/48）

**为什么改动**：A5 V100平台的硬件核数为AIC=32、AIV=64，与A3(AIC=24, AIV=48)不同。

**计算过程**：
```
AIC部分: 32 * 10 = 320
AIV部分: 64 * 3 = 192
BASE部分: 10
总计: 522 个 uint32_t
```
1024个int32_t完全足够（与A3的FIASINK_META_SIZE=1024保持一致）。

**踩坑**:
1. A3的MetaData头文件中 `AIC_CORE_NUM=36, AIV_CORE_NUM=72` 并非A3的实际硬件核数，而是MetaData数组维度的预留上限。A3的AICPU kernel中 `aicCoreNum_=24, aivCoreNum_=48` 才是实际默认值。最初误将A3的数组预留值(36/72)当作A3实际核数，然后按比例放大到A5，得到了错误的48/96。
2. 核数错误会导致MetaData内存布局完全错位——`GetAPAIVMetaAbsIndex` 和 `GetAPBaseMetaAbsIndex` 函数计算偏移时依赖 `AP_AIC_CORE_NUM` 和 `AP_AIV_CORE_NUM`，主算子读metadata时会读到错误的位置。

### 3.2 类型命名前缀 (AP_)

**改动内容**：所有枚举、结构体、常量均增加 `AP` 前缀。

| A3 | A5 |
|----|-----|
| `BlockType` | `APBlockType` |
| `SparseMode` | `APSparseMode` |
| `FlashDecodeResult` | `APFlashDecodeResult` |
| `SplitResult` | `APSplitResult` |
| `BaseInfo` | `APBaseInfo` |
| `SplitParam` | `APSplitParam` |
| `SplitInfo` | `APSplitInfo` |
| `CostInfo` | `APCostInfo` |
| `SplitContext` | `APSplitContext` |
| `BatchCache` | `APBatchCache` |
| `S1GCache` | `APS1GCache` |
| `CoreCache` | `APCoreCache` |
| `AssignContext` | `APAssignContext` |

**为什么改动**：
- A5仓库 `attention_pioneer` 已有 `split_core.h`，其中定义了同名但不同的 `SplitResult`、`BaseInfo` 等结构体
- AICPU kernel的结构体包含额外的 sink 相关字段，与Host侧的不兼容
- 加前缀避免命名冲突，保持两套代码独立

**踩坑**: 如果不加前缀，编译时会出现重定义错误，尤其是在头文件被间接包含的情况下。

### 3.3 不复用Host侧split_core的决策

**背景分析**：A5已有 `op_host/common/split_core.h/cpp`（namespace `ap_optiling`），分核逻辑与A3 AICPU kernel相似，但存在关键差异：

| 差异点 | A5 split_core (Host) | A3 AICPU kernel |
|--------|---------------------|-----------------|
| BaseInfo.sinkNumber | 无 | 有 |
| BaseInfo.keySinkNumber | 无 | 有 |
| BaseInfo.supportFD | 无 | 有 |
| CalcS1GCacheWithSinkNumber | 无 | 有 |
| UpdateCurS2IdxWithSinkNumber | 无 | 有 |
| MLA支持 | 无（Host单独处理） | 有 |
| 编译环境 | Host侧(x86/ARM) | AICPU |

**决策**：AICPU kernel内完整实现分核逻辑，不复用Host侧split_core。

**理由**：
1. AICPU需要支持Sink相关的额外分核逻辑（CalcS1GCacheWithSinkNumber、UpdateCurS2IdxWithSinkNumber）
2. AICPU和Host的编译环境不同，头文件依赖不一致
3. 修改Host侧split_core会影响已有的Host Tiling流程，风险太大
4. 分核代码需要作为类的成员函数，访问类成员变量

### 3.4 GenMetaData中AIV循环的核数处理

**A3原始代码**：
```cpp
for (uint32_t i = 0; i < AIV_CORE_NUM; i++) {
    // AIV_CORE_NUM = 72，硬编码
}
```

**A5改动**：
```cpp
uint32_t aivCoreNum = aicCoreNum_ * cvRatio_;
for (uint32_t i = 0; i < aivCoreNum; i++) {
    // 使用动态计算的AIV核数
}
```

**为什么改动**：
- A3的AIV核数是固定的72（36*2），直接用常量
- A5的AIC:AIV比例通过 `cvRatio_` 属性传入，不一定都是1:2
- 使用动态计算更灵活，适配不同配置的A5芯片

**踩坑**: 最初直接用 `AP_AIV_CORE_NUM`(96) 作为循环上限，但这样在48核AIC配置下就是正确的(48*2=96)。如果未来有不同核数配置，硬编码会出错。使用 `aicCoreNum_ * cvRatio_` 动态计算更安全。

### 3.5 GenMetaData中BASE Metadata写入

**A3原始代码**：
```cpp
for (uint32_t i = 0; i < BASE_METADATA_SIZE; i++) {
    metaDataPtr[aicTotalSize + aivTotalSize + i] = 0;
}
// 然后逐个赋值
metaDataPtr[aicTotalSize + aivTotalSize + M_BASE_SIZE_INDEX] = mBaseSize_;
...
```

**A5改动**：
```cpp
uint32_t baseOffset = aicTotalSize + aivTotalSize;
// 先清零
for (uint32_t i = 0; i < AP_BASE_METADATA_SIZE; i++) {
    metaDataPtr[baseOffset + i] = 0;
}
// 再赋值
metaDataPtr[baseOffset + AP_M_BASE_SIZE_INDEX] = mBaseSize_;
...
```

**为什么改动**：引入 `baseOffset` 局部变量减少重复计算，提高可读性。逻辑完全等价。

### 3.6 算子注册名称

**A3**: `REGISTER_CPU_KERNEL(kAiInfraFusedInferAttentionSink, AiInfraFusedInferAttentionSinkMetadataKernel)`

**A5**: `REGISTER_CPU_KERNEL(kAttentionPioneerMetadata, AttentionPioneerMetadataKernel)`

**说明**：算子注册名需要与proto.h中的REG_OP名称对应，用于GE框架查找kernel实现。

### 3.7 属性名称 K_sink_num → k_sink_num

**A3 proto.h**: `.ATTR(K_sink_num, Int, 0)`
**A5 proto.h**: `.ATTR(k_sink_num, Int, 0)`

**为什么改动**：A5统一使用小写下划线命名风格。AICPU kernel中通过 `GetAttrValueOpt(ctx, "k_sink_num", kSinkNum_)` 获取，需要名称一致。

**踩坑**: A3中属性名是 `K_sink_num`（大写K开头），但L0层传入时用的key是 `"k_sink_num"`（全小写）。在A3中这两处名称不完全一致，是A3的一个小bug/特殊处理。A5统一改为 `k_sink_num`，避免混淆。

### 3.8 MetaData内存布局结构体

**A3**: `FiaSinkMetaData` 结构体
```cpp
struct FiaSinkMetaData {
    uint32_t aicMetadata[AIC_CORE_NUM][AIC_METADATA_SIZE];
    uint32_t aivMetadata[AIV_CORE_NUM][AIV_METADATA_SIZE];
    uint32_t baseMetadata[BASE_METADATA_SIZE];
};
```

**A5**: `APMetaData` 结构体（在 `detail` namespace内）
```cpp
namespace detail {
struct APMetaData {
    uint32_t aicMetadata[AP_AIC_CORE_NUM][AP_AIC_METADATA_SIZE]; // [48][10]
    uint32_t aivMetadata[AP_AIV_CORE_NUM][AP_AIV_METADATA_SIZE]; // [96][3]
    uint32_t baseMetadata[AP_BASE_METADATA_SIZE];                // [10]
};
} // namespace detail
```

**为什么改动**：
- 数组维度跟随核数增大
- 放入 `detail` namespace，表示这是内部实现细节
- 增加 `static_assert` 确保MetaData tensor能容纳结构体

### 3.9 Kernel侧MetaData访问函数

**A5新增**（在 `attention_pioneer_metadata.h` 中，`__CCE_AICORE__` guard内）：
```cpp
__aicore__ inline uint32_t GetAPAICMetaAbsIndex(uint32_t coreIdx, uint32_t metaIdx);
__aicore__ inline uint32_t GetAPAIVMetaAbsIndex(uint32_t coreIdx, uint32_t metaIdx);
__aicore__ inline uint32_t GetAPBaseMetaAbsIndex(uint32_t metaIdx);
```

**为什么新增**：这些函数供主算子 `AttentionPioneer` 在kernel侧读取metaData tensor中的分核信息。A3中也有对应的函数。通过 `__CCE_AICORE__` 宏控制，仅在kernel编译时可见。

### 3.10 L0 API的AICPU Task注册

**关键代码**：
```cpp
static internal::AicpuTaskSpace space("AttentionPioneerMetadata");

auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
    AttentionPioneerMetadata,
    OP_ATTR_NAMES({...}),
    OP_INPUT(actualSeqLengthsOptional, actualSeqLengthsKvOptional),
    OP_OUTPUT(metaData),
    OP_ATTR(...));
```

**说明**：
- `AicpuTaskSpace` 的名称必须与算子注册名一致
- `OP_ATTR_NAMES` 中的属性顺序必须与 `OP_ATTR` 中的参数顺序严格一致
- 属性名字符串必须与proto.h中的定义完全匹配

**踩坑**: OP_ATTR_NAMES和OP_ATTR的顺序如果不一致，运行时属性值会错位，导致参数校验失败或计算结果错误。这类bug非常难排查，因为编译不会报错。

### 3.11 aclnn层的socVersion获取

**代码**：
```cpp
const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
const char *socVersion = npuInfo.GetSocLongVersion().c_str();
```

**说明**：socVersion不由用户传入，而是在aclnn层从平台信息中自动获取。这样上层调用者不需要关心底层芯片版本。aclnn头文件的接口参数中不包含socVersion，它是在GetWorkspaceSize内部获取后传给L0层的。

---

## 4. 编译构建说明

### 4.1 构建依赖

AICPU kernel的CMakeLists使用 `add_modules_sources_aicpu(DEPENDENCIES attention_pioneer)`，表示：
- AICPU kernel模块依赖主算子 `attention_pioneer`（用于获取common头文件路径）
- 构建系统会自动将 `op_kernel_aicpu/` 下的 `.cpp` 文件编译为AICPU kernel

### 4.2 文件包含关系

```
aclnn_attention_pioneer_metadata.cpp
  └── l0_attention_pioneer_metadata.h/cpp
        └── ADD_TO_LAUNCHER_LIST_AICPU → 注册AICPU task

attention_pioneer_metadata_aicpu.cpp (AICPU侧独立编译)
  ├── attention_pioneer_metadata_aicpu.h (数据结构定义)
  ├── cpu_context_util.h (属性读取工具)
  └── REGISTER_CPU_KERNEL → 注册kernel实现

attention_pioneer_metadata.h (被两侧共享)
  ├── AICPU kernel (GenMetaData时使用常量)
  ├── InferShape (使用AP_METADATA_SIZE)
  └── 主算子kernel (使用GetAPxxxMetaAbsIndex读取metadata)
```

---

## 5. 测试建议

### 5.1 单元测试要点

1. **metadata正确性**：构造不同的actual_seq_lens，对比AICPU输出的metadata与Host侧SplitCore的输出
2. **边界值测试**：
   - batch_size=1, s1=1, s2=1（最小场景）
   - 大batch（256+），确保不越界
   - actual_seq_lens全为0
   - actual_seq_lens包含负值（应报错）
3. **MLA/GQA模式覆盖**：
   - MLA: numHeadsQ=1, numHeadsKv=1, headDimQK != headDimV
   - GQA: numHeadsQ=32, numHeadsKv=4
4. **Sink特性**：
   - sink_num=0（无sink）
   - sink_num>0, k_sink_num>0
5. **sparse_mode覆盖**：0~4每种模式

### 5.2 集成测试要点

1. ACL Graph场景：验证双流执行、Event同步、多层metadata复用
2. 端到端正确性：对比新旧方案的主算子最终输出
3. 性能测试：验证AICPU执行时间 < 0.5ms，端到端无Host bound

---

## 6. 总结

本次开发的主要工作量集中在：

1. **理解A3参考实现**（~1770行核心代码），梳理分核算法流程
2. **A3→A5适配**：核数常量、命名规范、MetaData布局扩展
3. **解决命名冲突**：引入AP前缀避免与已有split_core冲突
4. **分层实现**：aclnn → L0 → AICPU kernel 三层完整实现
5. **构建集成**：CMakeLists适配A5构建体系

核心风险点在于分核算法的正确性验证，建议通过对比测试（新旧方案输出比对）确保无误。
