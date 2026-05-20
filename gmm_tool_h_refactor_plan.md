# GMM 算子 tool.h 头文件重构方案

## 一、现状与问题

### 1.1 同名文件冲突

三个不同路径下的头文件均命名为 `tool.h`：

| # | 文件路径 | 当前命名空间 |
|---|---------|------------|
| 1 | `gmm/grouped_matmul/op_kernel/a16w4_msd/tool.h` | `GROUPED_MATMUL::A16W4Msd` |
| 2 | `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/tool.h` | `WeightQuantBatchMatmulV2::Arch35` |
| 3 | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/tool.h` | `WeightQuantBatchMatmulV2::Arch35` |

### 1.2 重复宏保护（严重编译隐患）

三个文件使用**完全相同的** include guard：

```cpp
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H
#define GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H
```

这意味着先被编译单元包含的任一文件会屏蔽后两个文件的全部内容，导致符号缺失或行为异常。

### 1.3 代码重复

| 内容 | 文件1 (a16w4_msd) | 文件2 (finalize_routing) | 文件3 (weight_quant_basic_block) |
|------|:-----------------:|:------------------------:|:--------------------------------:|
| `CeilAlign` 模板 | ✓ | ✓ | ✓ |
| `CeilAlign(uint32_t)` | ✗ | ✓ | ✓ |
| `CeilDiv` / `CeilDivide` | ✓ | ✓ | ✓ |
| `Min` | ✗ | ✓ | ✓ |
| `DataCopyPad2D` (GM→Local) | ✓ | ✓ | ✓ |
| `DataCopyPad2D` (Local→GM) | ✓ | ✓ | ✓ |
| `GetKBUnit` | ✓ | ✓ | ✓ |
| `QuantType` 枚举 | ✗ | ✓ | ✓ |
| 常量定义 (L1_SIZE 等) | ✗ | ✓ | ✓ |
| `IsMxA8W4` | ✗ | ✓ | ✓ |
| `MatmulL1GmType` | ✗ | ✓ | ✓ |

**文件2 与 文件3 的代码相似度超过 95%**（仅文件2多了 `VEC_MAX_ELEM_B32` 常量），属于典型的复制粘贴冗余。

### 1.4 引用关系

各 `tool.h` 当前被以下文件引用：

**文件1 (`a16w4_msd/tool.h`) 被引用：**
- `gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_basic_block.h`
- `gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_controller.h`
- `gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_cube_service.h`
- `gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_vector_service.h`

**文件2 (`finalize_routing/arch35/common/tool.h`) 被引用：**
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/common/basic_block_config.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_cube_compute.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vcv_basic_block.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vec_compute.h`

**文件3 (`grouped_matmul/arch35/weight_quant_basic_block/tool.h`) 被引用：**
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/anti_quant_y_vf.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_api/weight_quant_basic_api_v1.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nz.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_cube_compute.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h`

---

## 二、重构目标

1. **消除同名头文件**：每个头文件具有唯一、表意的文件名。
2. **消除重复宏保护**：每个头文件使用唯一的 include guard。
3. **消除重复代码**：公共工具函数和 arch35 公共定义只保留一份。
4. **向后兼容**：各算子的编译行为和功能逻辑保持不变。

---

## 三、具体方案

### 3.1 新建公共工具头文件

**新建**：`gmm/op_kernel/common/gmm_tool_common.h`

负责存放三个算子共用的基础工具函数。

```cpp
#ifndef GMM_OP_KERNEL_COMMON_TOOL_H
#define GMM_OP_KERNEL_COMMON_TOOL_H

#include "kernel_operator.h"

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::fp4x2_e1m2_t;
using AscendC::fp4x2_e2m1_t;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;

namespace GmmOpKernelCommon {

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

__aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b)
{
    ASCENDC_ASSERT(a <= (std::numeric_limits<uint32_t>::max() - b),
                   { KERNEL_LOG(KERNEL_ERROR, "CeilAlign uint32 over limit."); });
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T CeilDivide(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                     uint32_t blockCount, uint32_t blockLen,
                                     uint32_t dstInnerLength, uint32_t srcInnerLength)
{
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }

    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                     uint32_t dim1, uint32_t dim0,
                                     uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = CeilDivide((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ constexpr uint32_t GetKBUnit()
{
    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        return 2048; // 2048个4bit是1KB
    }
    return 1024 / sizeof(T); // 1KB总共大小为1024
}

} // namespace GmmOpKernelCommon
#endif
```

**关键统一决策：**
- `DataCopyPad2D`：统一支持 `int4b_t`、`fp4x2_e2m1_t`、`fp4x2_e1m2_t`（采用文件2/3的完整版本）。
- `GetKBUnit`：统一使用 `1024 / sizeof(T)` 通用实现（文件2/3版本），对 `int4b_t` 等半字节类型同样正确。
- `CeilAlign(uint32_t)`：统一加入，增加溢出安全检查。
- `CeilDivide`：统一命名，替换文件1中的 `CeilDiv`。

### 3.2 新建 Arch35 公共头文件

**新建**：`gmm/op_kernel/common/arch35_weight_quant_common.h`

负责合并文件2和文件3中 `WeightQuantBatchMatmulV2::Arch35` 命名空间下的所有公共定义。

```cpp
#ifndef GMM_OP_KERNEL_COMMON_ARCH35_WEIGHT_QUANT_COMMON_H
#define GMM_OP_KERNEL_COMMON_ARCH35_WEIGHT_QUANT_COMMON_H

#include "gmm/op_kernel/common/gmm_tool_common.h"
#include "kernel_operator.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::fp8_e8m0_t;
using AscendC::GetUserWorkspace;
using AscendC::TPosition;
using AscendC::VECTOR_REG_WIDTH;
using matmul::MatmulCallBackFunc;
using matmul::MatmulImpl;
using matmul::MatmulType;
using matmul::MatmulTypeWithScale;

#define SHORT_MIX_LOG(format, ...)

namespace WeightQuantBatchMatmulV2::Arch35 {

// 引入公共工具函数
using GmmOpKernelCommon::CeilAlign;
using GmmOpKernelCommon::CeilDivide;
using GmmOpKernelCommon::DataCopyPad2D;
using GmmOpKernelCommon::GetKBUnit;
using GmmOpKernelCommon::Min;

enum class QuantType {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4,
};

// buffer相关定义
static constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;
static constexpr int32_t DOUBLE_BUFFER_NUM = 2;
static constexpr int32_t SINGLE_BUFFER_NUM = 1;
static constexpr int64_t L1_SIZE = 512;
static constexpr int64_t L1_SIZE_BYTE = L1_SIZE * 1024;
static constexpr int64_t L1_HALF_SIZE = L1_SIZE / 2;
static constexpr int64_t L1_SIZE_WITH_QUANTSCALE = 504;
static constexpr int64_t L1_SIZE_WITH_QUANTSCALE_BYTE = L1_SIZE_WITH_QUANTSCALE * 1024;
static constexpr int64_t BIAS_L1_SIZE = 4;
static constexpr uint64_t A_L1_MAX_SIZE_WITH_BIAS_QUANT = 240UL * 1024UL;
static constexpr uint64_t MX_BIAS_SINGLE_VECTOR_SIZE = 128;
static constexpr uint64_t MX_SCALE_K_L1_SIZE = 4096;

// 控制参数定义
static constexpr int32_t BASIC_BLOCK_PROCESS_NUM = 2;
static constexpr uint64_t SCALE_COPY_GROUP_SIZE = 2;
static constexpr int32_t SCALE_COPY_DEFAULT_STRIDE = 0;
static constexpr int32_t SCALE_COPY_DEFAULT_N_STRIDE = 1;

// 参数约束定义
static constexpr uint64_t MX_GROUPSIZE = 32;
static constexpr uint64_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(half);
static constexpr uint64_t VEC_MAX_ELEM_B32 = VECTOR_REG_WIDTH / sizeof(float);
static constexpr uint32_t FP32_BLOCK_SIZE = 8;
static constexpr uint32_t FP16_BLOCK_SIZE = 16;
static constexpr int32_t C0_SIZE_B8 = 32;
static constexpr uint32_t SCALE_FACTOR_B_BIT = 8;

// 同步定义
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
static constexpr uint64_t SYNC_AIC_FIX_AIV_VF_FLAG = 3;
static constexpr uint64_t SYNC_AIV_MTE3_AIC_FIX_FLAG = 4;
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t FLAG_ID_MAX = 16;

template <typename xType, QuantType antiQuantType>
__aicore__ constexpr bool IsMxA8W4()
{
    return antiQuantType == QuantType::MX && IsSameType<xType, fp8_e4m3fn_t>::value;
}

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulL1GmType : MatmulType<POSITION, FORMAT, TYPE, ISTRANS, LAYOUT, IBSHARE> {
    constexpr static TPosition srcPos = TPosition::GM;
};

} // namespace WeightQuantBatchMatmulV2::Arch35
#endif
```

### 3.3 改造各算子目录头文件

#### 3.3.1 文件1：`a16w4_msd/tool.h` → `a16w4_msd/a16w4_msd_tool.h`

```cpp
#ifndef GROUPED_MATMUL_A16W4_MSD_TOOL_H
#define GROUPED_MATMUL_A16W4_MSD_TOOL_H

#include "gmm/op_kernel/common/gmm_tool_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GetUserWorkspace;

namespace GROUPED_MATMUL::A16W4Msd {
using GmmOpKernelCommon::CeilAlign;
using GmmOpKernelCommon::CeilDivide;
using GmmOpKernelCommon::DataCopyPad2D;
using GmmOpKernelCommon::GetKBUnit;

// 如A16W4Msd算子有其他特有工具函数，在此处定义

} // namespace GROUPED_MATMUL::A16W4Msd
#endif
```

#### 3.3.2 文件2：`finalize_routing/arch35/common/tool.h` → `finalize_routing/arch35/common/finalize_routing_arch35_common.h`

```cpp
#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_ARCH35_COMMON_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_ARCH35_COMMON_H

#include "gmm/op_kernel/common/arch35_weight_quant_common.h"

// 如finalize_routing算子有其他特有定义，在此处添加

#endif
```

#### 3.3.3 文件3：`grouped_matmul/arch35/weight_quant_basic_block/tool.h` → `grouped_matmul/arch35/weight_quant_basic_block/weight_quant_basic_block_tool.h`

```cpp
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_TOOL_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_TOOL_H

#include "gmm/op_kernel/common/arch35_weight_quant_common.h"

// 如weight_quant_basic_block算子有其他特有定义，在此处添加

#endif
```

---

## 四、引用路径修改清单

需要修改引用这三个头文件的所有源文件的 `#include` 路径：

| 原引用文件 | 原 include 路径 | 新 include 路径 |
|-----------|----------------|----------------|
| `grouped_matmul_weight_quant_a16w4_msd_basic_block.h` | `#include "tool.h"` | `#include "a16w4_msd_tool.h"` |
| `grouped_matmul_weight_quant_a16w4_msd_controller.h` | `#include "tool.h"` | `#include "a16w4_msd_tool.h"` |
| `grouped_matmul_weight_quant_a16w4_msd_cube_service.h` | `#include "tool.h"` | `#include "a16w4_msd_tool.h"` |
| `grouped_matmul_weight_quant_a16w4_msd_vector_service.h` | `#include "tool.h"` | `#include "a16w4_msd_tool.h"` |
| `basic_block_config.h` (finalize_routing) | `#include "tool.h"` | `#include "finalize_routing_arch35_common.h"` |
| `gmm_fr_weight_quant_cube_compute.h` | `#include "../common/tool.h"` | `#include "../common/finalize_routing_arch35_common.h"` |
| `gmm_fr_weight_quant_resplit_controller.h` | `#include "../common/tool.h"` | `#include "../common/finalize_routing_arch35_common.h"` |
| `gmm_fr_weight_quant_vcv_basic_block.h` | `#include "../common/tool.h"` | `#include "../common/finalize_routing_arch35_common.h"` |
| `gmm_fr_weight_quant_vec_compute.h` | `#include "../common/tool.h"` | `#include "../common/finalize_routing_arch35_common.h"` |
| `anti_quant_y_vf.h` | `#include "tool.h"` | `#include "weight_quant_basic_block_tool.h"` |
| `weight_quant_basic_api_v1.h` | `#include "../tool.h"` | `#include "../weight_quant_basic_block_tool.h"` |
| `basic_block_config.h` (grouped_matmul) | `#include "tool.h"` | `#include "weight_quant_basic_block_tool.h"` |
| `basic_block_vf_nd.h` | `#include "tool.h"` | `#include "weight_quant_basic_block_tool.h"` |
| `basic_block_vf_nz.h` | `#include "tool.h"` | `#include "weight_quant_basic_block_tool.h"` |
| `weight_quant_basic_block.h` | `#include "tool.h"` | `#include "weight_quant_basic_block_tool.h"` |
| `weight_quant_cube_compute.h` | `#include "tool.h"` | `#include "weight_quant_basic_block_tool.h"` |
| `weight_quant_vcv_basic_block.h` | `#include "tool.h"` | `#include "weight_quant_basic_block_tool.h"` |
| `weight_quant_vec_compute.h` | `#include "tool.h"` | `#include "weight_quant_basic_block_tool.h"` |

---

## 五、实施步骤

1. **创建公共目录**：新建 `gmm/op_kernel/common/` 目录。
2. **新建公共头文件**：
   - 创建 `gmm_tool_common.h`，实现统一工具函数。
   - 创建 `arch35_weight_quant_common.h`，合并文件2和文件3的公共内容。
3. **改造并重命名各算子头文件**：
   - 将 `a16w4_msd/tool.h` 重命名为 `a16w4_msd/a16w4_msd_tool.h`，改为引入 `gmm_tool_common.h`。
   - 将 `finalize_routing/arch35/common/tool.h` 重命名为 `finalize_routing/arch35/common/finalize_routing_arch35_common.h`，改为引入 `arch35_weight_quant_common.h`。
   - 将 `grouped_matmul/arch35/weight_quant_basic_block/tool.h` 重命名为 `grouped_matmul/arch35/weight_quant_basic_block/weight_quant_basic_block_tool.h`，改为引入 `arch35_weight_quant_common.h`。
4. **更新所有引用路径**：按"四、引用路径修改清单"更新 18 个引用文件的 `#include`。
5. **编译验证**：依次编译三个算子，确认无编译错误和功能退化。
6. **清理旧文件**：编译通过后，删除三个旧的 `tool.h` 文件。

---

## 六、收益

- **消除同名文件和重复宏保护**，彻底避免潜在的编译屏蔽问题。
- **工具函数只维护一份**，减少重复代码约 70%（文件2和文件3几乎完全合并到公共头文件）。
- **后续扩展只需修改一处**：例如新增 4bit 类型支持、调整 `GetKBUnit` 逻辑、修改同步常量等，无需在多个文件中同步修改。
- **新增算子可直接复用**：后续开发新的 GMM 算子时，可直接引入 `gmm_tool_common.h` 或 `arch35_weight_quant_common.h`，避免再次复制粘贴。
